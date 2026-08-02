/*
 * Copyright 2012-2025 CodeLibs Project and the Others.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 */
package org.codelibs.fess.embedding.ollama;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.codelibs.core.lang.StringUtil;
import org.codelibs.fess.embedding.AbstractEmbeddingClient;
import org.codelibs.fess.embedding.EmbeddingException;
import org.codelibs.fess.ollama.OllamaUrlUtil;
import org.codelibs.fess.util.CredentialUrlUtil;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Embedding client implementation for Ollama.
 * Calls Ollama's {@code POST /api/embed} endpoint.
 *
 * @see <a href="https://ollama.ai/">Ollama</a>
 */
public class OllamaEmbeddingClient extends AbstractEmbeddingClient {

    private static final Logger logger = LogManager.getLogger(OllamaEmbeddingClient.class);

    /** Shared ObjectMapper instance for JSON processing. */
    protected static final ObjectMapper objectMapper = new ObjectMapper();

    /** The name identifier for the Ollama embedding client. */
    protected static final String NAME = "ollama";

    /** Hard cap on a single backoff sleep, regardless of computed delay. */
    private static final long MAX_BACKOFF_MS = 60_000L;

    /**
     * Maximum number of texts sent in a single {@code POST /api/embed} request.
     * Ollama documents no hard limit on the {@code input} array (unlike the
     * OpenAI sibling's 2048 or Gemini's 100), so this is a protective soft cap
     * rather than an API requirement. fess core's {@code ChunkVectorHelper} can
     * flatten every chunk across a whole {@code bulk_size} of documents into one
     * list, and sending that as a single request would hold all inputs in memory
     * and block a (typically local) Ollama server, which embeds sequentially,
     * until every vector is computed - inviting out-of-memory or response-timeout
     * failures. 128 keeps each request bounded while still amortizing HTTP
     * overhead across a reasonable batch; larger lists are split into contiguous
     * sub-batches of at most this size and their vectors concatenated in order.
     */
    private static final int MAX_BATCH_ITEMS = 128;

    private static final String CONFIG_RETRY_MAX = "retry.max";
    private static final String CONFIG_RETRY_BASE_DELAY_MS = "retry.base.delay.ms";
    private static final String CONFIG_DOCUMENT_PREFIX = "document.prefix";
    private static final String CONFIG_QUERY_PREFIX = "query.prefix";
    private static final String CONFIG_TRUNCATE = "truncate";

    /**
     * Key suffix for the Ollama endpoint, appended to {@link #getConfigPrefix()} via {@link #getConfigString}.
     */
    private static final String CONFIG_API_URL_SUFFIX = "api.url";

    /**
     * Set once the userinfo refusal has been reported. The availability check runs on a
     * timer, so an unguarded ERROR would repeat for as long as the misconfiguration stands.
     */
    private final AtomicBoolean userinfoRejectionReported = new AtomicBoolean();

    /**
     * Default embedding model.
     *
     * <p>{@code embeddinggemma} is multilingual, where {@code nomic-embed-text} is trained on
     * English. On a Japanese corpus the latter barely separates documents at all - measured on a
     * 14-document set, every document scored between 0.76 and 0.83 for a paraphrase query and the
     * correct document ranked last - which reads as a broken semantic search rather than as a
     * model mismatch. {@code embeddinggemma} ranked the same document first on the same corpus.
     * Both emit 768-dimensional vectors, so {@code content_chunker.embedding.dimension} is
     * unchanged either way and switching between them costs only a re-index.
     */
    protected static final String DEFAULT_MODEL = "embeddinggemma";

    /** Default prefix prepended to document/chunk texts, per the {@code embeddinggemma} convention. */
    protected static final String DEFAULT_DOCUMENT_PREFIX = "title: none | text: ";

    /** Default prefix prepended to query texts, per the {@code embeddinggemma} convention. */
    protected static final String DEFAULT_QUERY_PREFIX = "task: search result | query: ";

    /** Document prefix used by the {@code nomic-embed} family. */
    protected static final String NOMIC_DOCUMENT_PREFIX = "search_document: ";

    /** Query prefix used by the {@code nomic-embed} family. */
    protected static final String NOMIC_QUERY_PREFIX = "search_query: ";

    /**
     * Default constructor.
     */
    public OllamaEmbeddingClient() {
        // Default constructor
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected boolean checkAvailabilityNow() {
        final String apiUrl = getApiUrl();
        if (StringUtil.isBlank(apiUrl)) {
            return false;
        }
        if (isUserinfoRefused(apiUrl)) {
            // Fail closed: this method is reached synchronously from init(), so a throw here
            // would escape the container's eager init-method assembler. See isUserinfoRefused.
            return false;
        }
        try {
            final HttpGet request = OllamaUrlUtil.createHttpGet(OllamaUrlUtil.appendPath(apiUrl, "/api/tags"), apiUrlConfigKey());
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                if (statusCode < 200 || statusCode >= 300) {
                    return false;
                }
                final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                return isModelAvailable(responseBody);
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                logger.debug("[Embedding:OLLAMA] Ollama is not available. url={}, error={}", CredentialUrlUtil.maskCredentialInUrl(apiUrl),
                        e.getMessage());
            }
            return false;
        }
    }

    /**
     * Reports whether {@code apiUrl} carries a userinfo subcomponent, logging the remedy at
     * ERROR the first time it does. See
     * {@link OllamaUrlUtil#userinfoRejectionMessage(String)} for why such an endpoint can
     * never issue a request and what the operator should configure instead.
     *
     * <p><b>Callers must fail closed on the availability path.</b>
     * {@link #checkAvailabilityNow()} is reached synchronously from {@link #init()}
     * ({@code init -> startAvailabilityCheck -> updateAvailability -> checkAvailabilityNow}),
     * and {@code init()} is the DI container's eager init method; an exception thrown there
     * aborts container assembly and stops the application from starting. Reporting the client
     * unavailable leaves a misconfigured endpoint disabled and diagnosed instead of fatal.
     * The embed path is not on that path and does throw, so the caller gets the remedy rather
     * than an opaque protocol failure.
     *
     * <p>The message names only the configuration key and the proxy settings, never any part
     * of the configured value, so the credential reaches neither the log nor the exception.
     *
     * @param apiUrl the configured endpoint
     * @return {@code true} when the endpoint must be refused
     */
    protected boolean isUserinfoRefused(final String apiUrl) {
        if (!CredentialUrlUtil.hasUserInfo(apiUrl)) {
            return false;
        }
        if (userinfoRejectionReported.compareAndSet(false, true)) {
            logger.error("[Embedding:OLLAMA] {}", OllamaUrlUtil.userinfoRejectionMessage(apiUrlConfigKey()));
        }
        return true;
    }

    /**
     * Checks if the configured model is available in Ollama.
     *
     * @param responseBody the response body from /api/tags endpoint
     * @return true if the configured model is available
     */
    protected boolean isModelAvailable(final String responseBody) {
        final String configuredModel = getModel();
        if (StringUtil.isBlank(configuredModel)) {
            return true;
        }
        try {
            final JsonNode jsonNode = objectMapper.readTree(responseBody);
            if (jsonNode.has("models")) {
                final JsonNode models = jsonNode.get("models");
                for (final JsonNode model : models) {
                    if (model.has("name")) {
                        final String modelName = model.get("name").asText();
                        if (normalizeModelName(configuredModel).equals(normalizeModelName(modelName))) {
                            return true;
                        }
                    }
                }
            }
            logger.warn("[Embedding:OLLAMA] Configured model not found. model={}", configuredModel);
            return false;
        } catch (final Exception e) {
            logger.warn("[Embedding:OLLAMA] Failed to parse Ollama models response. error={}", e.getMessage());
            return false;
        }
    }

    private String normalizeModelName(final String name) {
        return name.endsWith(":latest") ? name.substring(0, name.length() - 7) : name;
    }

    @Override
    public List<float[]> embedDocuments(final List<String> texts) {
        return embedInSubBatches("embedDocuments", applyPrefix(texts, getDocumentPrefix()));
    }

    @Override
    public List<float[]> embedQuery(final List<String> texts) {
        return embedInSubBatches("embedQuery", applyPrefix(texts, getQueryPrefix()));
    }

    /**
     * Splits the (already prefixed, if applicable) {@code texts} into contiguous
     * sub-batches of at most {@link #MAX_BATCH_ITEMS} items, calls
     * {@link #callEmbedApi(String, List)} once per sub-batch, and concatenates
     * the resulting vectors in input order. Lists at or below the cap are sent as
     * a single request. See {@link #MAX_BATCH_ITEMS} for why the cap exists.
     *
     * <p>Invariants preserved across the concatenation: output order equals input
     * order; the result count equals the input count; empty/{@code null} input
     * returns an empty list with no API call; and any sub-batch failure (after its
     * own retries) propagates, failing the whole call.
     *
     * @param operation log label, e.g. {@code "embedDocuments"} or {@code "embedQuery"}
     * @param texts the texts to embed, in the form to send as-is to the API
     * @return the parsed vectors, one per input text, in the same order
     * @throws EmbeddingException if any sub-batch call fails or returns an unusable response
     */
    private List<float[]> embedInSubBatches(final String operation, final List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Collections.emptyList();
        }
        final int total = texts.size();
        if (total <= MAX_BATCH_ITEMS) {
            return callEmbedApi(operation, texts);
        }
        final List<float[]> vectors = new ArrayList<>(total);
        for (int start = 0; start < total; start += MAX_BATCH_ITEMS) {
            final int end = Math.min(start + MAX_BATCH_ITEMS, total);
            // subList is a read-only view; callEmbedApi only serializes and sizes it.
            vectors.addAll(callEmbedApi(operation, texts.subList(start, end)));
        }
        return vectors;
    }

    /**
     * Prepends {@code prefix} to every element of {@code texts}. A blank
     * ({@code null} or empty) prefix is treated as a no-op, leaving {@code texts}
     * unchanged rather than concatenating an empty string.
     *
     * @param texts the input texts
     * @param prefix the prefix to prepend, or blank for none
     * @return the prefixed texts, or {@code texts} unchanged when {@code prefix} is blank
     */
    static List<String> applyPrefix(final List<String> texts, final String prefix) {
        if (texts == null || texts.isEmpty() || StringUtil.isBlank(prefix)) {
            return texts;
        }
        final List<String> prefixed = new ArrayList<>(texts.size());
        for (final String text : texts) {
            prefixed.add(prefix + text);
        }
        return prefixed;
    }

    /**
     * Calls Ollama's {@code POST /api/embed} endpoint with the given
     * (already prefixed, if applicable) texts as a single request. Callers must
     * bound the sub-batch size; {@link #embedInSubBatches(String, List)} splits
     * larger inputs into sub-batches of at most {@link #MAX_BATCH_ITEMS} and
     * invokes this method once per sub-batch.
     *
     * @param operation log label, e.g. {@code "embedDocuments"} or {@code "embedQuery"}
     * @param texts the texts to embed, in the form to send as-is to the API
     * @return the parsed vectors, one per input text, in the same order
     * @throws EmbeddingException if the provider call fails or returns an unusable response
     */
    private List<float[]> callEmbedApi(final String operation, final List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Collections.emptyList();
        }
        final String apiUrl = getApiUrl();
        if (isUserinfoRefused(apiUrl)) {
            throw new EmbeddingException(OllamaUrlUtil.userinfoRejectionMessage(apiUrlConfigKey()));
        }
        final String url = OllamaUrlUtil.appendPath(apiUrl, "/api/embed");
        final Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", getModel());
        requestBody.put("input", texts);
        // Always sent explicitly rather than relying on Ollama's server-side default,
        // which is truncate=true. See isTruncateEnabled() for the trade-off.
        requestBody.put(CONFIG_TRUNCATE, isTruncateEnabled());
        final long startTime = System.currentTimeMillis();
        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            return executeWithRetry(operation, () -> {
                final HttpPost httpRequest = OllamaUrlUtil.createHttpPost(url, apiUrlConfigKey());
                httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (statusCode < 200 || statusCode >= 300) {
                        if (isRetryableStatus(statusCode)) {
                            throw new RetryableHttpException(statusCode, response.getReasonPhrase());
                        }
                        logger.warn("[Embedding:OLLAMA] API error. url={}, statusCode={}, message={}",
                                CredentialUrlUtil.maskCredentialInUrl(url), statusCode, response.getReasonPhrase());
                        throw new EmbeddingException("Ollama API error: " + statusCode + " " + response.getReasonPhrase());
                    }
                    String responseBody;
                    try {
                        responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                    } catch (final org.apache.hc.core5.http.ParseException pe) {
                        throw new IOException("Failed to parse Ollama response body", pe);
                    }
                    final List<float[]> vectors = parseEmbedResponse(responseBody, texts.size());
                    logger.info("[Embedding:OLLAMA] {} response received. count={}, elapsedTime={}ms", operation, vectors.size(),
                            System.currentTimeMillis() - startTime);
                    return vectors;
                }
            });
        } catch (final EmbeddingException e) {
            throw e;
        } catch (final Exception e) {
            logger.warn("[Embedding:OLLAMA] Failed to call Ollama embed API. url={}, error={}", CredentialUrlUtil.maskCredentialInUrl(url),
                    e.getMessage(), e);
            throw new EmbeddingException("Failed to call Ollama embed API", e);
        }
    }

    /**
     * Parses the {@code /api/embed} response body into a list of vectors,
     * validating that the returned vector count matches {@code expectedCount}
     * and that every vector's length matches {@link #getDimension()}.
     *
     * @param responseBody the raw JSON response body
     * @param expectedCount the expected number of vectors (= number of input texts)
     * @return the parsed vectors, in response order
     * @throws EmbeddingException if the response is malformed or a count/dimension mismatch is detected
     */
    protected List<float[]> parseEmbedResponse(final String responseBody, final int expectedCount) {
        final JsonNode jsonNode;
        try {
            jsonNode = objectMapper.readTree(responseBody);
        } catch (final IOException e) {
            throw new EmbeddingException("Failed to parse Ollama embed response", e);
        }
        final JsonNode embeddingsNode = jsonNode.path("embeddings");
        if (!embeddingsNode.isArray()) {
            throw new EmbeddingException("Ollama embed response missing 'embeddings' array");
        }
        if (embeddingsNode.size() != expectedCount) {
            throw new EmbeddingException(
                    "Ollama embed response count mismatch: expected=" + expectedCount + ", actual=" + embeddingsNode.size());
        }
        final int dimension = getDimension();
        // Ollama's /api/embed returns embeddings positionally with no per-item id/index
        // field (unlike the OpenAI sibling plugin), so reassembling by array order is correct.
        final List<float[]> vectors = new ArrayList<>(embeddingsNode.size());
        int vectorIndex = 0;
        for (final JsonNode vectorNode : embeddingsNode) {
            if (!vectorNode.isArray() || vectorNode.size() != dimension) {
                throw new EmbeddingException("Ollama embed vector dimension mismatch: expected=" + dimension + ", actual="
                        + (vectorNode.isArray() ? vectorNode.size() : -1));
            }
            final float[] vector = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                final JsonNode componentNode = vectorNode.get(i);
                if (componentNode == null || !componentNode.isNumber()) {
                    throw new EmbeddingException("Ollama embed vector component is not numeric: index=" + vectorIndex + ", position=" + i);
                }
                // A JSON magnitude that overflows the double range (e.g. 1e999) parses as a
                // NUMBER node whose value is +/-Infinity and passes isNumber(); reject any
                // non-finite component so an unusable vector is never stored silently.
                final float component = (float) componentNode.asDouble();
                if (!Float.isFinite(component)) {
                    throw new EmbeddingException("Ollama embed vector component is not finite: index=" + vectorIndex + ", position=" + i);
                }
                vector[i] = component;
            }
            vectors.add(vector);
            vectorIndex++;
        }
        return vectors;
    }

    /**
     * Gets the Ollama API URL. Normalized so callers can append fixed paths
     * like {@code /api/embed} or {@code /api/tags} without producing
     * duplicates (trailing {@code /} and a trailing {@code /api} segment
     * are stripped).
     *
     * @return the normalized API base URL
     */
    protected String getApiUrl() {
        return OllamaUrlUtil.normalizeBaseUrl(getConfigString(CONFIG_API_URL_SUFFIX, "http://localhost:11434"));
    }

    /**
     * Builds the full {@code api.url} configuration key (e.g.
     * {@code content_chunker.embedding.ollama.api.url}) from {@link #getConfigPrefix()} and
     * {@link #CONFIG_API_URL_SUFFIX}, so the name reported in the userinfo-refusal message can
     * never drift from the key {@link #getApiUrl()} actually reads.
     *
     * @return the fully-qualified {@code api.url} configuration key
     */
    private String apiUrlConfigKey() {
        return getConfigPrefix() + "." + CONFIG_API_URL_SUFFIX;
    }

    /**
     * Starts the model/prefix consistency diagnostic after the shared initialization.
     *
     * <p>The HTTP client, its two-tier timeouts and the availability check all come from
     * {@link AbstractEmbeddingClient#init()}; only the startup warning is Ollama-specific, because
     * only Ollama drives the document/query distinction through a text prefix whose correct value
     * depends on the model family.</p>
     */
    @Override
    public void init() {
        super.init();
        // Same condition super.init() early-returns on: this client's model/prefix pairing is not
        // worth warning about while a different provider is the configured one.
        if (getName().equals(getEmbeddingType())) {
            warnIfPrefixModelMismatch();
        }
    }

    /**
     * Gets the configured Ollama embedding model name.
     *
     * @return the model name (default {@link #DEFAULT_MODEL})
     */
    protected String getModel() {
        return getConfigString("model", DEFAULT_MODEL);
    }

    @Override
    protected int getTimeout() {
        return getConfigInt("timeout", 60000);
    }

    @Override
    protected String getConfigPrefix() {
        return "content_chunker.embedding.ollama";
    }

    /**
     * Gets the prefix prepended to document/chunk texts before embedding
     * (see {@link #embedDocuments(List)}). Defaults to the {@code embeddinggemma}
     * convention {@link #DEFAULT_DOCUMENT_PREFIX}; set to an empty string to
     * disable prefixing for models that don't use it, or to
     * {@link #NOMIC_DOCUMENT_PREFIX} for a {@code nomic-embed} model.
     *
     * @return the configured document prefix
     */
    protected String getDocumentPrefix() {
        return getConfigString(CONFIG_DOCUMENT_PREFIX, DEFAULT_DOCUMENT_PREFIX);
    }

    /**
     * Gets the prefix prepended to query texts before embedding (see
     * {@link #embedQuery(List)}). Defaults to the {@code embeddinggemma}
     * convention {@link #DEFAULT_QUERY_PREFIX}; set to an empty string to
     * disable prefixing for models that don't use it, or to
     * {@link #NOMIC_QUERY_PREFIX} for a {@code nomic-embed} model.
     *
     * @return the configured query prefix
     */
    protected String getQueryPrefix() {
        return getConfigString(CONFIG_QUERY_PREFIX, DEFAULT_QUERY_PREFIX);
    }

    /**
     * Whether Ollama may truncate an input that exceeds the embedding model's context
     * window, sent explicitly as the {@code truncate} field of every
     * {@code POST /api/embed} request rather than left to the server-side default.
     *
     * <p>Defaults to {@code true}, matching Ollama's own documented default, so behavior
     * is unchanged. The trade-off is deliberate and worth understanding:
     *
     * <ul>
     * <li>{@code truncate=true} (default): an over-context chunk is silently cut down to
     * fit and a well-formed vector of the correct dimension is returned. Every check in
     * {@link #parseEmbedResponse(String, int)} passes, so the discarded tail - and the
     * relevance loss it causes - leaves no trace in the logs. Sending the flag explicitly
     * at least makes the choice visible in the request and configurable.</li>
     * <li>{@code truncate=false}: Ollama rejects the over-context input instead, so the
     * chunk fails loudly and the document surfaces as an error rather than being indexed
     * with a quietly degraded vector. This is the safer setting for relevance, but a
     * failed document is currently not recoverable without a re-crawl, which is why it is
     * not the default.</li>
     * </ul>
     *
     * <p>Chunk size - and therefore how often the limit is reached at all - is governed by
     * {@code content_chunker.length.chunk_size}; sizing chunks to the model's context
     * window is the real fix, and this flag only decides how the overflow is reported.
     *
     * <p>An unparseable value keeps the {@code true} default and emits a WARN, so a typo
     * cannot silently flip indexing into hard per-document failures.
     *
     * @return the value of {@code content_chunker.embedding.ollama.truncate} (default {@code true})
     */
    protected boolean isTruncateEnabled() {
        final String value = getConfigString(CONFIG_TRUNCATE, Boolean.TRUE.toString());
        if (StringUtil.isBlank(value)) {
            return true;
        }
        final String normalized = value.trim();
        if (Boolean.FALSE.toString().equalsIgnoreCase(normalized)) {
            return false;
        }
        if (!Boolean.TRUE.toString().equalsIgnoreCase(normalized)) {
            logger.warn("[Embedding:OLLAMA] Invalid {}.{}='{}', using default true", getConfigPrefix(), CONFIG_TRUNCATE, value);
        }
        return true;
    }

    /**
     * Emits a startup WARN when the configured task prefixes follow a different model family's
     * convention than the configured model.
     *
     * <p>The prefixes are plain configurable strings, and a mismatched pair still returns
     * well-formed vectors of the correct dimension - every response check passes, so nothing
     * fails and only relevance degrades. Two mistakes are worth surfacing:
     *
     * <ul>
     * <li>a recognized model paired with the <em>other</em> family's prefixes, which is what
     * switching {@code model} to {@code nomic-embed-text} and leaving the
     * {@code embeddinggemma} defaults in place produces;</li>
     * <li>an unrecognized model with any non-blank prefix, where the convention is unknown and
     * blanking both is the safe default.</li>
     * </ul>
     *
     * <p>The prefixes are left untouched - behavior is unchanged - and blanking both (the
     * documented escape hatch) always suppresses the warning. Invoked once from {@link #init()}.
     */
    protected void warnIfPrefixModelMismatch() {
        final String model = getModel();
        final String documentPrefix = getDocumentPrefix();
        final String queryPrefix = getQueryPrefix();
        if (StringUtil.isBlank(documentPrefix) && StringUtil.isBlank(queryPrefix)) {
            return;
        }
        if (looksLikeNomicModel(model)) {
            if (!usesNomicPrefixes(documentPrefix, queryPrefix)) {
                logger.warn(
                        "[Embedding:OLLAMA] Model '{}' is a nomic-embed model, but the configured prefixes are not the "
                                + "nomic convention (document.prefix='{}', query.prefix='{}'). Set {}.{}='{}' and {}.{}='{}', or set both "
                                + "to an empty value, to avoid degraded relevance.",
                        model, documentPrefix, queryPrefix, getConfigPrefix(), CONFIG_DOCUMENT_PREFIX, NOMIC_DOCUMENT_PREFIX,
                        getConfigPrefix(), CONFIG_QUERY_PREFIX, NOMIC_QUERY_PREFIX);
            }
            return;
        }
        if (looksLikeGemmaModel(model)) {
            if (!usesGemmaPrefixes(documentPrefix, queryPrefix)) {
                logger.warn("[Embedding:OLLAMA] Model '{}' is an embeddinggemma model, but the configured prefixes are not the "
                        + "embeddinggemma convention (document.prefix='{}', query.prefix='{}'). Set {}.{}='{}' and {}.{}='{}', or set "
                        + "both to an empty value, to avoid degraded relevance.", model, documentPrefix, queryPrefix, getConfigPrefix(),
                        CONFIG_DOCUMENT_PREFIX, DEFAULT_DOCUMENT_PREFIX, getConfigPrefix(), CONFIG_QUERY_PREFIX, DEFAULT_QUERY_PREFIX);
            }
            return;
        }
        logger.warn(
                "[Embedding:OLLAMA] Model '{}' is not a model family whose task-prefix convention this plugin knows, but prefixes "
                        + "are applied (document.prefix='{}', query.prefix='{}'). If this model does not use these prefixes, set {}.{} "
                        + "and {}.{} to an empty value to avoid degraded relevance.",
                model, documentPrefix, queryPrefix, getConfigPrefix(), CONFIG_DOCUMENT_PREFIX, getConfigPrefix(), CONFIG_QUERY_PREFIX);
    }

    /**
     * Returns whether {@code model} looks like a {@code nomic-embed} model, for which the
     * {@link #NOMIC_DOCUMENT_PREFIX}/{@link #NOMIC_QUERY_PREFIX} prefixes are appropriate. The
     * check is a case-insensitive {@code "nomic"} substring match, covering tags such as
     * {@code nomic-embed-text}, {@code nomic-embed-text:latest}, and {@code nomic-embed-text-v1.5}.
     *
     * @param model the configured model name (may be {@code null})
     * @return true when the name contains {@code "nomic"}
     */
    static boolean looksLikeNomicModel(final String model) {
        return model != null && model.toLowerCase(Locale.ROOT).contains("nomic");
    }

    /**
     * Returns whether {@code model} looks like an {@code embeddinggemma} model, for which the
     * {@link #DEFAULT_DOCUMENT_PREFIX}/{@link #DEFAULT_QUERY_PREFIX} prefixes are appropriate.
     * The check is a case-insensitive {@code "embeddinggemma"} substring match, covering tags
     * such as {@code embeddinggemma}, {@code embeddinggemma:latest} and
     * {@code embeddinggemma:300m-bf16}.
     *
     * @param model the configured model name (may be {@code null})
     * @return true when the name contains {@code "embeddinggemma"}
     */
    static boolean looksLikeGemmaModel(final String model) {
        return model != null && model.toLowerCase(Locale.ROOT).contains("embeddinggemma");
    }

    /**
     * Returns whether the configured prefixes follow the {@code nomic-embed} convention. A blank
     * prefix counts as matching, so clearing only one side is not reported as a mismatch.
     *
     * @param documentPrefix the configured document prefix
     * @param queryPrefix the configured query prefix
     * @return true when neither prefix contradicts the nomic convention
     */
    static boolean usesNomicPrefixes(final String documentPrefix, final String queryPrefix) {
        return matchesOrBlank(documentPrefix, NOMIC_DOCUMENT_PREFIX) && matchesOrBlank(queryPrefix, NOMIC_QUERY_PREFIX);
    }

    /**
     * Returns whether the configured prefixes follow the {@code embeddinggemma} convention. The
     * document side matches on the {@code "title:"}/{@code "| text:"} shape rather than on the
     * exact default, because the convention interpolates the document's own title and an
     * operator may legitimately write {@code "title: manual | text: "}. A blank prefix counts as
     * matching.
     *
     * @param documentPrefix the configured document prefix
     * @param queryPrefix the configured query prefix
     * @return true when neither prefix contradicts the embeddinggemma convention
     */
    static boolean usesGemmaPrefixes(final String documentPrefix, final String queryPrefix) {
        final boolean documentOk;
        if (StringUtil.isBlank(documentPrefix)) {
            documentOk = true;
        } else {
            final String lower = documentPrefix.toLowerCase(Locale.ROOT).trim();
            documentOk = lower.startsWith("title:") && lower.contains("text:");
        }
        return documentOk && matchesOrBlank(queryPrefix, DEFAULT_QUERY_PREFIX);
    }

    /**
     * Returns whether {@code value} is blank or equals {@code expected}, ignoring case and
     * surrounding whitespace. The trim matters because the conventions carry a trailing space
     * that a properties file round-trip can drop.
     *
     * @param value the configured value
     * @param expected the convention's value
     * @return true when {@code value} is blank or matches {@code expected}
     */
    private static boolean matchesOrBlank(final String value, final String expected) {
        return StringUtil.isBlank(value) || value.trim().equalsIgnoreCase(expected.trim());
    }

    /**
     * Functional interface for the retryable HTTP call body executed by
     * {@link #executeWithRetry(String, HttpCall)}.
     *
     * @param <T> the call result type
     */
    @FunctionalInterface
    interface HttpCall<T> {
        T call() throws IOException;
    }

    /**
     * Internal signaling exception thrown by the HTTP call body when the
     * response status code is retryable. Caught by
     * {@link #executeWithRetry(String, HttpCall)}; never escapes the client.
     */
    static final class RetryableHttpException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        final int statusCode;
        final String reason;

        RetryableHttpException(final int statusCode, final String reason) {
            super("retryable http error: " + statusCode + " " + reason);
            this.statusCode = statusCode;
            this.reason = reason;
        }
    }

    /**
     * Returns whether the given HTTP status code should be retried:
     * {@code 429}, {@code 500}, {@code 502}, {@code 503}, {@code 504}.
     *
     * @param statusCode the HTTP status code
     * @return true when the status is retryable
     */
    static boolean isRetryableStatus(final int statusCode) {
        return statusCode == 429 || statusCode == 500 || statusCode == 502 || statusCode == 503 || statusCode == 504;
    }

    /**
     * Maximum total attempts (including the first) for a retryable call.
     *
     * @return the value of {@code content_chunker.embedding.ollama.retry.max} (default 3)
     */
    protected int getRetryMaxAttempts() {
        return getConfigInt(CONFIG_RETRY_MAX, 3);
    }

    /**
     * Base delay in milliseconds for exponential backoff between retries.
     *
     * @return the value of {@code content_chunker.embedding.ollama.retry.base.delay.ms} (default 2000)
     */
    protected long getRetryBaseDelayMs() {
        return getConfigLong(CONFIG_RETRY_BASE_DELAY_MS, 2000L);
    }

    /**
     * Executes {@code call} with retry on {@link RetryableHttpException} and
     * on transient connect-time {@link IOException}s. {@link EmbeddingException}
     * (RuntimeException) is not caught here and propagates immediately.
     * Backoff is exponential ({@code base * 2^(attempt-1)}) with +/-20% jitter.
     *
     * @param operation log label, e.g. {@code "embed"}
     * @param call the HTTP call body
     * @param <T> the call result type
     * @return the call result on success
     * @throws IOException if the call throws a non-retryable {@link IOException} or the retry budget is exhausted
     */
    <T> T executeWithRetry(final String operation, final HttpCall<T> call) throws IOException {
        final int maxAttempts = Math.max(1, getRetryMaxAttempts());
        final long baseDelay = Math.max(0L, getRetryBaseDelayMs());
        IOException lastIo = null;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return call.call();
            } catch (final RetryableHttpException e) {
                if (attempt == maxAttempts) {
                    logger.warn("[Embedding:OLLAMA] {} retry exhausted. attempts={}, lastStatus={}", operation, attempt, e.statusCode);
                    throw new IOException("Ollama API retryable error: " + e.statusCode + " " + e.reason, e);
                }
                sleepBackoff(operation, attempt, maxAttempts, baseDelay, "status", e.statusCode);
            } catch (final IOException e) {
                if (attempt == maxAttempts) {
                    lastIo = e;
                    break;
                }
                sleepBackoff(operation, attempt, maxAttempts, baseDelay, "exception", e.getClass().getSimpleName());
            }
        }
        if (lastIo == null) {
            throw new IllegalStateException("executeWithRetry exited without exception or success");
        }
        throw lastIo;
    }

    /**
     * Sleeps an exponential-backoff interval with +/-20% jitter and a hard cap.
     *
     * @param operation log label
     * @param attempt 1-based current attempt index
     * @param maxAttempts total attempts including the first
     * @param baseDelay base delay in milliseconds (already clamped to >=0)
     * @param logFieldKey log field name carrying the cause ("status" or "exception")
     * @param logFieldValue log field value for the cause
     * @throws IOException if the sleep is interrupted
     */
    private void sleepBackoff(final String operation, final int attempt, final int maxAttempts, final long baseDelay,
            final String logFieldKey, final Object logFieldValue) throws IOException {
        final long jitter = (long) (baseDelay * 0.2 * ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
        final long delay = Math.min(MAX_BACKOFF_MS, (long) (baseDelay * Math.pow(2, attempt - 1)) + jitter);
        final long sleepMs = Math.max(0, delay);
        logger.info("[Embedding:OLLAMA] {} retrying. attempt={}/{}, {}={}, sleepMs={}", operation, attempt, maxAttempts, logFieldKey,
                logFieldValue, sleepMs);
        try {
            Thread.sleep(sleepMs);
        } catch (final InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IOException("Retry interrupted", ie);
        }
    }
}
