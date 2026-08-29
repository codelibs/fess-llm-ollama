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
package org.codelibs.fess.llm.ollama;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.config.ConnectionConfig;
import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClientBuilder;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManagerBuilder;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.util.Timeout;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.codelibs.core.lang.StringUtil;
import org.codelibs.fess.llm.AbstractLlmClient;
import org.codelibs.fess.llm.LlmChatRequest;
import org.codelibs.fess.llm.LlmChatResponse;
import org.codelibs.fess.llm.LlmException;
import org.codelibs.fess.llm.LlmMessage;
import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.ollama.OllamaUrlUtil;
import org.codelibs.fess.util.CredentialUrlUtil;
import org.codelibs.fess.util.ComponentUtil;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;

/**
 * LLM client implementation for Ollama.
 *
 * Ollama provides a local LLM server that can run various models
 * like Llama, Mistral, etc. on your own hardware.
 *
 * @see <a href="https://ollama.ai/">Ollama</a>
 */
public class OllamaLlmClient extends AbstractLlmClient {

    private static final Logger logger = LogManager.getLogger(OllamaLlmClient.class);
    /** The name identifier for the Ollama LLM client. */
    protected static final String NAME = "ollama";

    /** Hard cap on a single backoff sleep, regardless of computed delay. */
    private static final long MAX_BACKOFF_MS = 60_000L;

    /** done_reason values that indicate a normal stream termination. */
    private static final Set<String> NORMAL_DONE_REASONS = Set.of("stop", "load", "unload");

    private static final String CONFIG_RETRY_MAX = "retry.max";
    private static final String CONFIG_RETRY_BASE_DELAY_MS = "retry.base.delay.ms";
    private static final String CONFIG_CONNECT_TIMEOUT = "connect.timeout";

    /** Configuration key holding the Ollama endpoint, named in the userinfo refusal message. */
    private static final String CONFIG_API_URL = "rag.llm.ollama.api.url";

    /**
     * Set once the userinfo refusal has been reported. The availability check runs on a
     * timer, so an unguarded ERROR would repeat for as long as the misconfiguration stands.
     */
    private final AtomicBoolean userinfoRejectionReported = new AtomicBoolean();

    /**
     * Default constructor.
     */
    public OllamaLlmClient() {
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
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] Ollama is not available. apiUrl is blank");
            }
            return false;
        }
        if (isUserinfoRefused(apiUrl)) {
            // Fail closed: this method is reached synchronously from init(), so a throw here
            // would escape the container's eager init-method assembler. See isUserinfoRefused.
            return false;
        }
        try {
            final HttpGet request = OllamaUrlUtil.createHttpGet(OllamaUrlUtil.appendPath(apiUrl, "/api/tags"), CONFIG_API_URL);
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                if (statusCode < 200 || statusCode >= 300) {
                    if (logger.isDebugEnabled()) {
                        logger.debug("[LLM:OLLAMA] Ollama availability check failed. url={}, statusCode={}",
                                CredentialUrlUtil.maskCredentialInUrl(apiUrl), statusCode);
                    }
                    return false;
                }

                final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                return isModelAvailable(responseBody);
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] Ollama is not available. url={}, error={}", CredentialUrlUtil.maskCredentialInUrl(apiUrl),
                        e.getMessage());
            }
            return false;
        }
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
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] Model name is not configured, skipping model availability check");
            }
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
                            if (logger.isDebugEnabled()) {
                                logger.debug("[LLM:OLLAMA] Model found. configured={}, found={}", configuredModel, modelName);
                            }
                            return true;
                        }
                    }
                }
            }
            logger.warn("[LLM:OLLAMA] Configured model not found. model={}", configuredModel);
            return false;
        } catch (final Exception e) {
            logger.warn("[LLM:OLLAMA] Failed to parse Ollama models response. error={}", e.getMessage());
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
     * Request-time callers ({@link #chat(LlmChatRequest)},
     * {@link #streamChat(LlmChatRequest, LlmStreamCallback)}) are not on that path and do
     * throw, so the caller gets the remedy rather than an opaque protocol failure.
     *
     * <p>The message names only the configuration key and the proxy settings, never any part
     * of the configured value, so the credential reaches neither the log nor the exception.
     *
     * @param apiUrl the configured endpoint.
     * @return {@code true} when the endpoint must be refused.
     */
    protected boolean isUserinfoRefused(final String apiUrl) {
        if (!CredentialUrlUtil.hasUserInfo(apiUrl)) {
            return false;
        }
        if (userinfoRejectionReported.compareAndSet(false, true)) {
            logger.error("[LLM:OLLAMA] {}", OllamaUrlUtil.userinfoRejectionMessage(CONFIG_API_URL));
        }
        return true;
    }

    @Override
    public LlmChatResponse chat(final LlmChatRequest request) {
        final String apiUrl = getApiUrl();
        if (isUserinfoRefused(apiUrl)) {
            throw new LlmException(OllamaUrlUtil.userinfoRejectionMessage(CONFIG_API_URL), LlmException.ERROR_CONNECTION);
        }
        final String url = OllamaUrlUtil.appendPath(apiUrl, "/api/chat");
        final Map<String, Object> requestBody = buildRequestBody(request, false);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OLLAMA] Sending chat request to Ollama. url={}, model={}, messageCount={}",
                    CredentialUrlUtil.maskCredentialInUrl(url), requestBody.get("model"), request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] requestBody={}", json);
            }
            return executeWithRetry("chat", () -> {
                final HttpPost httpRequest = OllamaUrlUtil.createHttpPost(url, CONFIG_API_URL);
                httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (statusCode < 200 || statusCode >= 300) {
                        if (isRetryableStatus(statusCode)) {
                            throw new RetryableHttpException(statusCode, response.getReasonPhrase());
                        }
                        logger.warn("[LLM:OLLAMA] API error. url={}, statusCode={}, message={}", CredentialUrlUtil.maskCredentialInUrl(url),
                                statusCode, response.getReasonPhrase());
                        throw new LlmException("Ollama API error: " + statusCode + " " + response.getReasonPhrase(),
                                resolveErrorCode(statusCode));
                    }

                    String responseBody;
                    try {
                        responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                    } catch (final org.apache.hc.core5.http.ParseException pe) {
                        throw new IOException("Failed to parse Ollama response body", pe);
                    }
                    if (logger.isDebugEnabled()) {
                        logger.debug("[LLM:OLLAMA] responseBody={}", responseBody);
                    }
                    final JsonNode jsonNode = objectMapper.readTree(responseBody);

                    final LlmChatResponse chatResponse = new LlmChatResponse();
                    final JsonNode messageNode = jsonNode.path("message");
                    final String content = messageNode.path("content").asText(null);
                    if (content != null) {
                        chatResponse.setContent(content);
                    }
                    final String finishReason = jsonNode.path("done_reason").asText(null);
                    if (finishReason != null) {
                        chatResponse.setFinishReason(finishReason);
                    }
                    final String responseModel = jsonNode.path("model").asText(null);
                    if (responseModel != null) {
                        chatResponse.setModel(responseModel);
                    }
                    if (jsonNode.has("prompt_eval_count")) {
                        chatResponse.setPromptTokens(jsonNode.get("prompt_eval_count").asInt());
                    }
                    if (jsonNode.has("eval_count")) {
                        chatResponse.setCompletionTokens(jsonNode.get("eval_count").asInt());
                    }
                    if (logger.isDebugEnabled()) {
                        final JsonNode thinkingNode = messageNode.path("thinking");
                        if (!thinkingNode.isMissingNode()) {
                            logger.debug("[LLM:OLLAMA] Thinking response received. thinkingLength={}", thinkingNode.asText().length());
                        }
                    }
                    logger.info(
                            "[LLM:OLLAMA] Chat response received. model={}, promptTokens={}, completionTokens={}, contentLength={}, elapsedTime={}ms",
                            chatResponse.getModel(), chatResponse.getPromptTokens(), chatResponse.getCompletionTokens(),
                            chatResponse.getContent() != null ? chatResponse.getContent().length() : 0,
                            System.currentTimeMillis() - startTime);
                    return chatResponse;
                }
            });
        } catch (final LlmException e) {
            throw e;
        } catch (final Exception e) {
            logger.warn("[LLM:OLLAMA] Failed to call Ollama API. url={}, error={}", CredentialUrlUtil.maskCredentialInUrl(url),
                    e.getMessage(), e);
            throw new LlmException("Failed to call Ollama API", LlmException.ERROR_CONNECTION, e);
        }
    }

    @Override
    public void streamChat(final LlmChatRequest request, final LlmStreamCallback callback) {
        final String apiUrl = getApiUrl();
        if (isUserinfoRefused(apiUrl)) {
            final LlmException refusal =
                    new LlmException(OllamaUrlUtil.userinfoRejectionMessage(CONFIG_API_URL), LlmException.ERROR_CONNECTION);
            callback.onError(refusal);
            throw refusal;
        }
        final String url = OllamaUrlUtil.appendPath(apiUrl, "/api/chat");
        final Map<String, Object> requestBody = buildRequestBody(request, true);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OLLAMA] Starting streaming chat request to Ollama. url={}, model={}, messageCount={}",
                    CredentialUrlUtil.maskCredentialInUrl(url), requestBody.get("model"), request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] requestBody={}", json);
            }
            executeWithRetry("streamChat", () -> {
                final HttpPost httpRequest = OllamaUrlUtil.createHttpPost(url, CONFIG_API_URL);
                httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (statusCode < 200 || statusCode >= 300) {
                        if (isRetryableStatus(statusCode)) {
                            throw new RetryableHttpException(statusCode, response.getReasonPhrase());
                        }
                        logger.warn("[LLM:OLLAMA] Streaming API error. url={}, statusCode={}, message={}",
                                CredentialUrlUtil.maskCredentialInUrl(url), statusCode, response.getReasonPhrase());
                        throw new LlmException("Ollama API error: " + statusCode + " " + response.getReasonPhrase(),
                                resolveErrorCode(statusCode));
                    }

                    final var contentTypeHeader = response.getFirstHeader("Content-Type");
                    final String contentType = contentTypeHeader == null ? "" : contentTypeHeader.getValue();
                    if (logger.isDebugEnabled()) {
                        logger.debug("[LLM:OLLAMA] Stream response received. status={}, contentType={}", statusCode,
                                contentType.isEmpty() ? "<absent>" : contentType);
                    }
                    if (!contentType.toLowerCase(Locale.ROOT).startsWith("application/x-ndjson")) {
                        logger.warn(
                                "[LLM:OLLAMA] Unexpected Content-Type for streaming response. "
                                        + "expected=application/x-ndjson, actual='{}'. Likely a misconfigured proxy or version mismatch.",
                                contentType.isEmpty() ? "<absent>" : contentType);
                    }

                    if (response.getEntity() == null) {
                        logger.warn("[LLM:OLLAMA] Empty response from Ollama streaming API. url={}",
                                CredentialUrlUtil.maskCredentialInUrl(url));
                        throw new LlmException("Empty response from Ollama");
                    }

                    consumeStream((String) requestBody.get("model"), response, callback, startTime);
                    return null;
                }
            }, callback);
        } catch (final LlmException e) {
            callback.onError(e);
            throw e;
        } catch (final IOException e) {
            logger.warn("[LLM:OLLAMA] Failed to stream from Ollama API. url={}, error={}", CredentialUrlUtil.maskCredentialInUrl(url),
                    e.getMessage(), e);
            final LlmException llmException = new LlmException("Failed to stream from Ollama API", LlmException.ERROR_CONNECTION, e);
            callback.onError(llmException);
            throw llmException;
        }
    }

    /**
     * Consumes the NDJSON streaming body and emits chunks via {@code callback}.
     * Caller is responsible for closing {@code response}.
     *
     * @param model the model name (used for log context).
     * @param response the HTTP response holding the NDJSON entity.
     * @param callback the stream callback to invoke for each chunk.
     * @param startTime the millisecond timestamp captured before the request, for elapsed-time logs.
     * @throws IOException if reading the stream fails.
     */
    private void consumeStream(final String model, final org.apache.hc.client5.http.impl.classic.CloseableHttpResponse response,
            final LlmStreamCallback callback, final long startTime) throws IOException {
        int chunkCount = 0;
        int objectCount = 0;
        int parseErrorCount = 0;
        long firstChunkTime = 0;
        String doneReason = null;
        long totalDurationNs = 0L;
        long loadDurationNs = 0L;
        long promptEvalDurationNs = 0L;
        long evalDurationNs = 0L;
        int promptEvalCount = 0;
        int evalCount = 0;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.getEntity().getContent(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (StringUtil.isBlank(line)) {
                    continue;
                }
                try {
                    final JsonNode jsonNode = objectMapper.readTree(line);
                    objectCount++;

                    final JsonNode errorNode = jsonNode.path("error");
                    if (!errorNode.isMissingNode() && !errorNode.isNull()) {
                        final String errorMessage = errorNode.asText();
                        logger.warn("[LLM:OLLAMA] Stream error received from Ollama. model={}, error={}", model, errorMessage);
                        throw new LlmException("Ollama stream error: " + errorMessage, LlmException.ERROR_INVALID_RESPONSE);
                    }

                    final boolean done = jsonNode.has("done") && jsonNode.get("done").asBoolean();

                    final JsonNode messageNode = jsonNode.path("message");
                    final JsonNode contentNode = messageNode.path("content");
                    if (!contentNode.isMissingNode()) {
                        final String content = contentNode.asText();
                        if (content.isEmpty() && !done && !messageNode.path("thinking").isMissingNode()) {
                            // Skip thinking-only chunk
                            continue;
                        }
                        callback.onChunk(content, done);
                        if (chunkCount == 0) {
                            firstChunkTime = System.currentTimeMillis() - startTime;
                        }
                        chunkCount++;
                    } else if (done) {
                        callback.onChunk("", true);
                    }

                    if (done) {
                        doneReason = jsonNode.path("done_reason").asText(null);
                        totalDurationNs = jsonNode.path("total_duration").asLong(0L);
                        loadDurationNs = jsonNode.path("load_duration").asLong(0L);
                        promptEvalDurationNs = jsonNode.path("prompt_eval_duration").asLong(0L);
                        evalDurationNs = jsonNode.path("eval_duration").asLong(0L);
                        promptEvalCount = jsonNode.path("prompt_eval_count").asInt(0);
                        evalCount = jsonNode.path("eval_count").asInt(0);
                        break;
                    }
                } catch (final JacksonException e) {
                    parseErrorCount++;
                    logger.warn("[LLM:OLLAMA] Failed to parse streaming response. line={}", line, e);
                }
            }
        }

        final long evalDurationMs = evalDurationNs / 1_000_000L;
        final String tokensPerSecond = evalDurationMs > 0 ? String.format(Locale.ROOT, "%.2f", evalCount * 1000.0 / evalDurationMs) : "n/a";
        logger.info(
                "[LLM:OLLAMA] Stream completed. chunkCount={}, objectCount={}, firstChunkMs={}, elapsedTime={}ms, "
                        + "doneReason={}, totalDurationMs={}, loadDurationMs={}, promptEvalDurationMs={}, "
                        + "evalDurationMs={}, promptEvalCount={}, evalCount={}, tokensPerSecond={}, parseErrorCount={}",
                chunkCount, objectCount, firstChunkTime, System.currentTimeMillis() - startTime, doneReason, totalDurationNs / 1_000_000L,
                loadDurationNs / 1_000_000L, promptEvalDurationNs / 1_000_000L, evalDurationMs, promptEvalCount, evalCount, tokensPerSecond,
                parseErrorCount);

        if (doneReason != null && !NORMAL_DONE_REASONS.contains(doneReason)) {
            logger.warn("[LLM:OLLAMA] Stream finished abnormally. doneReason={}, evalCount={}, " + "promptEvalCount={}, model={}",
                    doneReason, evalCount, promptEvalCount, model);
        }
    }

    /**
     * Builds the request body for the Ollama API.
     *
     * @param request the chat request
     * @param stream whether to enable streaming
     * @return the request body as a map
     */
    protected Map<String, Object> buildRequestBody(final LlmChatRequest request, final boolean stream) {
        final Map<String, Object> body = new HashMap<>();

        String model = request.getModel();
        if (StringUtil.isBlank(model)) {
            model = getModel();
        }
        body.put("model", model);

        final List<Map<String, String>> messages = request.getMessages().stream().map(this::convertMessage).collect(Collectors.toList());
        body.put("messages", messages);

        body.put("stream", stream);

        final Map<String, Object> options = new HashMap<>();

        applyGlobalOptions(options);

        if (request.getTemperature() != null) {
            options.put("temperature", request.getTemperature());
        }
        if (request.getMaxTokens() != null) {
            options.put("num_predict", request.getMaxTokens());
        }
        if (request.getExtraParams() != null) {
            final String topP = request.getExtraParam("top_p");
            if (topP != null) {
                try {
                    options.put("top_p", Double.parseDouble(topP));
                } catch (final NumberFormatException e) {
                    logger.warn("[LLM:OLLAMA] Invalid top_p value, skipping. value={}", topP);
                }
            }
            final String topK = request.getExtraParam("top_k");
            if (topK != null) {
                try {
                    options.put("top_k", Integer.parseInt(topK));
                } catch (final NumberFormatException e) {
                    logger.warn("[LLM:OLLAMA] Invalid top_k value, skipping. value={}", topK);
                }
            }
            final String numCtx = request.getExtraParam("num_ctx");
            if (numCtx != null) {
                try {
                    options.put("num_ctx", Integer.parseInt(numCtx));
                } catch (final NumberFormatException e) {
                    logger.warn("[LLM:OLLAMA] Invalid num_ctx value, skipping. value={}", numCtx);
                }
            }
        }
        if (!options.isEmpty()) {
            body.put("options", options);
        }

        final String thinkingLevel = request.getExtraParam("thinking_level");
        if (thinkingLevel != null && isValidThinkingLevel(thinkingLevel)) {
            body.put("think", thinkingLevel.toLowerCase(Locale.ROOT));
        } else {
            final Integer thinkingBudget = request.getThinkingBudget();
            if (thinkingBudget != null) {
                body.put("think", thinkingBudget > 0);
            }
        }

        return body;
    }

    /**
     * Returns whether the given value is one of the string thinking levels recognized by
     * Ollama's Chat API ({@code "high"}, {@code "medium"}, {@code "low"}). Required for
     * GPT-OSS family models which ignore the boolean form of {@code think}.
     *
     * @param value the candidate level string (case-insensitive); {@code null} is rejected.
     * @return {@code true} when the value is a recognized level.
     * @see <a href="https://docs.ollama.com/capabilities/thinking">Ollama thinking docs</a>
     */
    static boolean isValidThinkingLevel(final String value) {
        if (value == null) {
            return false;
        }
        final String normalized = value.toLowerCase(Locale.ROOT);
        return "high".equals(normalized) || "medium".equals(normalized) || "low".equals(normalized);
    }

    /**
     * Applies global options from {@code rag.llm.ollama.options.*} system properties to the options map.
     *
     * @param options the options map to populate
     */
    protected void applyGlobalOptions(final Map<String, Object> options) {
        if (!ComponentUtil.hasComponent("systemProperties")) {
            return;
        }
        final String optionsPrefix = getConfigPrefix() + ".options.";
        final var systemProperties = ComponentUtil.getSystemProperties();
        for (final String key : systemProperties.stringPropertyNames()) {
            if (key.startsWith(optionsPrefix)) {
                final String optionName = key.substring(optionsPrefix.length());
                final String value = systemProperties.getProperty(key);
                if (value != null && !value.isEmpty()) {
                    options.put(optionName, parseOptionValue(value));
                }
            }
        }
    }

    /**
     * Parses a string value into an appropriate type (Integer, Double, Boolean, or String).
     *
     * @param value the string value to parse
     * @return the parsed value
     */
    protected Object parseOptionValue(final String value) {
        try {
            return Integer.parseInt(value);
        } catch (final NumberFormatException e) {
            // not an integer
        }
        try {
            return Double.parseDouble(value);
        } catch (final NumberFormatException e) {
            // not a double
        }
        if ("true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) {
            return Boolean.parseBoolean(value);
        }
        return value;
    }

    /**
     * Converts an LlmMessage to a map for the API request.
     *
     * @param message the message to convert
     * @return the message as a map
     */
    protected Map<String, String> convertMessage(final LlmMessage message) {
        final Map<String, String> map = new HashMap<>();
        map.put("role", message.getRole());
        map.put("content", message.getContent());
        return map;
    }

    /**
     * Normalizes a model name by stripping the {@code :latest} suffix.
     *
     * @param name the model name
     * @return the normalized model name
     */
    private String normalizeModelName(final String name) {
        return name.endsWith(":latest") ? name.substring(0, name.length() - 7) : name;
    }

    /**
     * Gets the Ollama API URL.
     *
     * <p>Normalizes the configured value so that callers can append fixed paths like
     * {@code /api/chat} or {@code /api/tags} without producing duplicates. Trailing
     * {@code /} and a trailing {@code /api} segment (as documented in
     * <a href="https://docs.ollama.com/api/introduction">Ollama API introduction</a>:
     * {@code http://localhost:11434/api}, {@code https://ollama.com/api}) are stripped.
     *
     * @return the normalized API base URL (without trailing slash or {@code /api}).
     */
    protected String getApiUrl() {
        final String raw = ComponentUtil.getFessConfig().getOrDefault(CONFIG_API_URL, "http://localhost:11434");
        return normalizeApiUrl(raw);
    }

    /**
     * Strips a trailing {@code /} and a trailing {@code /api} segment from an Ollama base
     * URL, leaving the host root that the client can suffix with {@code /api/chat} or
     * {@code /api/tags}, and leaving any query string or fragment in place. Idempotent.
     * Delegates to {@link OllamaUrlUtil#normalizeBaseUrl(String)} so the two clients cannot
     * drift apart.
     *
     * @param url the raw configured URL.
     * @return the normalized URL, or the input unchanged when blank.
     */
    static String normalizeApiUrl(final String url) {
        return OllamaUrlUtil.normalizeBaseUrl(url);
    }

    @Override
    protected String getModel() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.model", "gemma4:e4b");
    }

    @Override
    protected int getTimeout() {
        return getConfigInt("timeout", 60000);
    }

    /**
     * Gets the TCP connect timeout in milliseconds. Separate from
     * {@link #getTimeout()} (response/read timeout) so that local Ollama
     * deployments can fail fast on connection issues while still allowing
     * minutes for token generation and first-call model load.
     *
     * @return the connect timeout in milliseconds.
     */
    protected int getConnectTimeout() {
        return getConfigInt(CONFIG_CONNECT_TIMEOUT, 5000);
    }

    /**
     * Overrides {@link AbstractLlmClient#init()} to apply distinct connect and response
     * timeouts. The base implementation uses a single {@link #getTimeout()} value for
     * all three of: connection-request, response, and connect.
     *
     * <p><b>Drift warning:</b> If {@code AbstractLlmClient.init()} adds new HTTP-client
     * configuration (e.g. an interceptor or new connection-pool setting), this override
     * must be updated to match. Source of truth: {@code repos/fess/.../AbstractLlmClient.java}.
     */
    @Override
    public void init() {
        if (!getName().equals(getLlmType())) {
            if (logger.isDebugEnabled()) {
                logger.debug("Skipping availability check. llmType={}, name={}", getLlmType(), getName());
            }
            return;
        }

        if (httpClient != null) {
            // Defensive: re-init scenarios should release the prior pool before swapping.
            try {
                httpClient.close();
            } catch (final IOException e) {
                logger.warn("[LLM:OLLAMA] Failed to close prior HTTP client during re-init", e);
            }
        }
        httpClient = buildHttpClient();
        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OLLAMA] {} initialized. model={}, connectTimeout={}ms, responseTimeout={}ms, maxConcurrent={}", getName(),
                    getModel(), getConnectTimeout(), getTimeout(), getMaxConcurrentRequests());
        }

        concurrencyLimiter = new Semaphore(getMaxConcurrentRequests());
        startAvailabilityCheck();
    }

    /**
     * Builds the {@link CloseableHttpClient} with two-tier timeouts (connect vs response/read)
     * and the shared proxy configuration. Used by {@link #init()} and mirrored by tests.
     *
     * @return a configured {@link CloseableHttpClient}.
     */
    protected CloseableHttpClient buildHttpClient() {
        final int connectTimeout = getConnectTimeout();
        final int responseTimeout = getTimeout();
        final RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(Timeout.ofMilliseconds(connectTimeout))
                .setResponseTimeout(Timeout.ofMilliseconds(responseTimeout))
                .build();
        final HttpClientBuilder builder = HttpClients.custom()
                .setConnectionManager(PoolingHttpClientConnectionManagerBuilder.create()
                        .setDefaultConnectionConfig(
                                ConnectionConfig.custom().setConnectTimeout(Timeout.ofMilliseconds(connectTimeout)).build())
                        .build())
                .setDefaultRequestConfig(requestConfig)
                .disableAutomaticRetries();
        configureProxy(builder);
        return builder.build();
    }

    @Override
    protected String getConfigPrefix() {
        return "rag.llm.ollama";
    }

    @Override
    protected void applyPromptTypeParams(final LlmChatRequest request, final String promptType) {
        super.applyPromptTypeParams(request, promptType);
        final String prefix = getConfigPrefix() + "." + promptType;
        final String defaultPrefix = getConfigPrefix() + ".default";

        final String topP = getConfigWithFallback(prefix + ".top.p", defaultPrefix + ".top.p");
        if (topP != null) {
            request.putExtraParam("top_p", topP);
        }
        final String topK = getConfigWithFallback(prefix + ".top.k", defaultPrefix + ".top.k");
        if (topK != null) {
            request.putExtraParam("top_k", topK);
        }
        final String numCtx = getConfigWithFallback(prefix + ".num.ctx", defaultPrefix + ".num.ctx");
        if (numCtx != null) {
            request.putExtraParam("num_ctx", numCtx);
        }

        if (request.getTemperature() == null) {
            final String temperature = getConfigWithFallback(prefix + ".temperature", defaultPrefix + ".temperature");
            if (temperature != null) {
                request.setTemperature(Double.parseDouble(temperature));
            }
        }
        if (request.getMaxTokens() == null) {
            final String maxTokens = getConfigWithFallback(prefix + ".max.tokens", defaultPrefix + ".max.tokens");
            if (maxTokens != null) {
                request.setMaxTokens(Integer.parseInt(maxTokens));
            }
        }
        if (request.getThinkingBudget() == null) {
            final String thinkingBudget = getConfigWithFallback(prefix + ".thinking.budget", defaultPrefix + ".thinking.budget");
            if (thinkingBudget != null) {
                request.setThinkingBudget(Integer.parseInt(thinkingBudget));
            }
        }
        if (request.getExtraParam("thinking_level") == null) {
            final String thinkingLevel = getConfigWithFallback(prefix + ".thinking.level", defaultPrefix + ".thinking.level");
            if (thinkingLevel != null) {
                if (isValidThinkingLevel(thinkingLevel)) {
                    request.putExtraParam("thinking_level", thinkingLevel);
                } else {
                    logger.warn("[LLM:OLLAMA] Invalid thinking.level value, ignoring. value={}, allowed=[high,medium,low]", thinkingLevel);
                }
            }
        }
        applyDefaultParams(request, promptType);
    }

    /**
     * Applies default generation parameters based on prompt type.
     * Only sets defaults when user has not configured the parameter.
     *
     * @param request the LLM chat request
     * @param promptType the prompt type (e.g. "intent", "evaluation", "answer")
     */
    protected void applyDefaultParams(final LlmChatRequest request, final String promptType) {
        switch (promptType) {
        case "intent":
            if (request.getTemperature() == null) {
                request.setTemperature(0.1);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "evaluation":
            if (request.getTemperature() == null) {
                request.setTemperature(0.1);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(512);
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "unclear":
        case "noresults":
        case "docnotfound":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(512);
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "direct":
        case "faq":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(4096);
            }
            break;
        case "answer":
            if (request.getTemperature() == null) {
                request.setTemperature(0.5);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(8192);
            }
            break;
        case "summary":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(8192);
            }
            break;
        case "queryregeneration":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        default:
            break;
        }
    }

    /**
     * Gets a config value with fallback. Returns the primary key's value if present, otherwise the fallback key's value.
     *
     * @param primaryKey the primary config key
     * @param fallbackKey the fallback config key
     * @return the config value, or null if neither key is set
     */
    protected String getConfigWithFallback(final String primaryKey, final String fallbackKey) {
        final var config = ComponentUtil.getFessConfig();
        final String value = config.getOrDefault(primaryKey, null);
        if (value != null) {
            return value;
        }
        return config.getOrDefault(fallbackKey, null);
    }

    @Override
    protected int getAvailabilityCheckInterval() {
        return getConfigInt("availability.check.interval", 60);
    }

    @Override
    protected boolean isRagChatEnabled() {
        return Boolean.parseBoolean(ComponentUtil.getFessConfig().getOrDefault("rag.chat.enabled", "false"));
    }

    @Override
    protected String getLlmType() {
        return ComponentUtil.getFessConfig().getSystemProperty("rag.llm.name", "ollama");
    }

    @Override
    protected int getContextMaxChars(final String promptType) {
        final String key = "rag.llm.ollama." + promptType + ".context.max.chars";
        final String configValue = ComponentUtil.getFessConfig().getOrDefault(key, null);
        if (configValue != null) {
            final int value = Integer.parseInt(configValue);
            if (value > 0) {
                return value;
            }
            logger.warn("Invalid context max chars for promptType={}: {}. Using default.", promptType, value);
        }
        switch (promptType) {
        case "answer":
            return 10000;
        case "summary":
            return 10000;
        case "faq":
            return 6000;
        default:
            return 6000;
        }
    }

    @Override
    protected int getEvaluationMaxRelevantDocs() {
        return getConfigInt("chat.evaluation.max.relevant.docs", 3);
    }

    @Override
    protected int getEvaluationDescriptionMaxChars() {
        return getConfigInt("chat.evaluation.description.max.chars", 500);
    }

    @Override
    protected int getHistoryMaxChars() {
        return getConfigInt("history.max.chars", 4000);
    }

    @Override
    protected int getIntentHistoryMaxMessages() {
        return getConfigInt("intent.history.max.messages", 6);
    }

    @Override
    protected int getIntentHistoryMaxChars() {
        return getConfigInt("intent.history.max.chars", 3000);
    }

    @Override
    public int getHistoryAssistantMaxChars() {
        return getConfigInt("history.assistant.max.chars", 500);
    }

    @Override
    public int getHistoryAssistantSummaryMaxChars() {
        return getConfigInt("history.assistant.summary.max.chars", 500);
    }

    /**
     * Functional interface for the retryable HTTP call body executed by
     * {@link #executeWithRetry(String, HttpCall)}.
     *
     * @param <T> the call result type.
     */
    @FunctionalInterface
    interface HttpCall<T> {
        T call() throws IOException;
    }

    /**
     * Internal signaling exception thrown by the HTTP call body when the response status
     * code is retryable (per {@link #isRetryableStatus(int)}). Caught by
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
     * Returns whether the given HTTP status code should be retried. Retryable statuses
     * cover Ollama's documented common errors: {@code 429} (Too Many Requests, returned
     * by Ollama Cloud and rate-limited proxies), {@code 500} (transient internal error),
     * {@code 502} (Bad Gateway, also documented as a common error), {@code 503} (queue
     * overload, the primary target for self-hosted), and {@code 504} (gateway timeout
     * when behind a reverse proxy).
     *
     * @param statusCode the HTTP status code.
     * @return {@code true} when the status is retryable.
     * @see <a href="https://docs.ollama.com/api/errors">Ollama errors</a>
     */
    static boolean isRetryableStatus(final int statusCode) {
        return statusCode == 429 || statusCode == 500 || statusCode == 502 || statusCode == 503 || statusCode == 504;
    }

    /**
     * Maximum total attempts (including the first) for a retryable call.
     *
     * @return the value of {@code rag.llm.ollama.retry.max} (default {@code 3}).
     */
    protected int getRetryMaxAttempts() {
        return getConfigInt(CONFIG_RETRY_MAX, 3);
    }

    /**
     * Base delay in milliseconds for exponential backoff between retries.
     *
     * @return the value of {@code rag.llm.ollama.retry.base.delay.ms} (default {@code 2000}).
     */
    protected long getRetryBaseDelayMs() {
        final String raw = ComponentUtil.getFessConfig().getOrDefault(getConfigPrefix() + "." + CONFIG_RETRY_BASE_DELAY_MS, "2000");
        try {
            return Long.parseLong(raw);
        } catch (final NumberFormatException e) {
            logger.warn("[LLM:OLLAMA] Invalid {}.{}='{}', using default 2000ms", getConfigPrefix(), CONFIG_RETRY_BASE_DELAY_MS, raw);
            return 2000L;
        }
    }

    /**
     * Executes {@code call} with retry on {@link RetryableHttpException} and on transient
     * connect-time {@link IOException}s. {@link LlmException} (RuntimeException) is NOT
     * caught here and propagates immediately. Backoff is exponential
     * ({@code base * 2^(attempt-1)}) with +/-20% jitter via {@link ThreadLocalRandom}.
     *
     * <p>Streaming callers wrap only the HTTP {@code execute} + status check; once the
     * NDJSON body starts flowing, partial-stream errors propagate without retry.
     *
     * @param operation log label, e.g. {@code "chat"} or {@code "streamChat"}.
     * @param call the HTTP call body.
     * @param <T> the call result type.
     * @return the call result on success.
     * @throws IOException if the call throws a non-retryable {@link IOException} or the retry
     *             budget is exhausted.
     */
    <T> T executeWithRetry(final String operation, final HttpCall<T> call) throws IOException {
        return executeWithRetry(operation, call, null);
    }

    /**
     * Same as {@link #executeWithRetry(String, HttpCall)} but additionally notifies the
     * given {@link LlmStreamCallback} (when non-{@code null}) between attempts via
     * {@link LlmStreamCallback#onRetry(String, int, int, long, Throwable)}.
     *
     * @param operation log label, e.g. {@code "chat"} or {@code "streamChat"}.
     * @param call the HTTP call body.
     * @param callback optional callback to notify on retry; may be {@code null}.
     * @param <T> the call result type.
     * @return the call result on success.
     * @throws IOException if the call throws a non-retryable {@link IOException} or the retry
     *             budget is exhausted.
     */
    <T> T executeWithRetry(final String operation, final HttpCall<T> call, final LlmStreamCallback callback) throws IOException {
        final int maxAttempts = Math.max(1, getRetryMaxAttempts());
        final long baseDelay = Math.max(0L, getRetryBaseDelayMs());
        IOException lastIo = null;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return call.call();
            } catch (final RetryableHttpException e) {
                if (attempt == maxAttempts) {
                    logger.warn("[LLM:OLLAMA] {} retry exhausted. attempts={}, lastStatus={}", operation, attempt, e.statusCode);
                    throw new IOException("Ollama API retryable error: " + e.statusCode + " " + e.reason, e);
                }
                sleepBackoff(operation, attempt, maxAttempts, baseDelay, "status", e.statusCode, callback, e);
            } catch (final IOException e) {
                if (attempt == maxAttempts) {
                    lastIo = e;
                    break;
                }
                sleepBackoff(operation, attempt, maxAttempts, baseDelay, "exception", e.getClass().getSimpleName(), callback, e);
            }
        }
        if (lastIo == null) {
            throw new IllegalStateException("executeWithRetry exited without exception or success");
        }
        throw lastIo;
    }

    /**
     * Sleeps an exponential-backoff interval with +/-20% jitter and a hard cap.
     * Logs the retry decision at INFO and, when a callback is provided, invokes
     * {@link LlmStreamCallback#onRetry(String, int, int, long, Throwable)} immediately
     * after the log line and before the actual sleep. Restores interrupt status if
     * interrupted. Exceptions thrown by the callback are swallowed (logged at DEBUG)
     * so retry behavior is never affected by callback bugs.
     *
     * @param operation log label.
     * @param attempt 1-based current attempt index.
     * @param maxAttempts total attempts including the first.
     * @param baseDelay base delay in milliseconds (already clamped to >=0).
     * @param logFieldKey log field name carrying the cause ("status" or "exception").
     * @param logFieldValue log field value for the cause.
     * @param callback optional callback to notify; may be {@code null}.
     * @param cause the cause of the retry passed to the callback.
     * @throws IOException if the sleep is interrupted.
     */
    private void sleepBackoff(final String operation, final int attempt, final int maxAttempts, final long baseDelay,
            final String logFieldKey, final Object logFieldValue, final LlmStreamCallback callback, final Throwable cause)
            throws IOException {
        final long jitter = (long) (baseDelay * 0.2 * ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
        final long delay = Math.min(MAX_BACKOFF_MS, (long) (baseDelay * Math.pow(2, attempt - 1)) + jitter);
        final long sleepMs = Math.max(0, delay);
        logger.info("[LLM:OLLAMA] {} retrying. attempt={}/{}, {}={}, sleepMs={}", operation, attempt, maxAttempts, logFieldKey,
                logFieldValue, sleepMs);
        if (callback != null) {
            try {
                callback.onRetry(operation, attempt, maxAttempts, sleepMs, cause);
            } catch (final Exception cbEx) {
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OLLAMA] onRetry callback threw. error={}", cbEx.getMessage());
                }
            }
        }
        try {
            Thread.sleep(sleepMs);
        } catch (final InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IOException("Retry interrupted", ie);
        }
    }

}
