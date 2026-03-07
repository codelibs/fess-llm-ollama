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
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.core5.http.ContentType;
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
import org.codelibs.fess.util.ComponentUtil;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;

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

    /** The system prompt for LLM interactions. */
    protected String systemPrompt;
    /** The prompt for detecting user intent. */
    protected String intentDetectionPrompt;
    /** The system prompt for handling unclear intents. */
    protected String unclearIntentSystemPrompt;
    /** The system prompt for handling no results. */
    protected String noResultsSystemPrompt;
    /** The system prompt for handling document not found. */
    protected String documentNotFoundSystemPrompt;
    /** The prompt for evaluating responses. */
    protected String evaluationPrompt;
    /** The system prompt for answer generation. */
    protected String answerGenerationSystemPrompt;
    /** The system prompt for summary generation. */
    protected String summarySystemPrompt;
    /** The system prompt for FAQ answer generation. */
    protected String faqAnswerSystemPrompt;
    /** The system prompt for direct answer generation. */
    protected String directAnswerSystemPrompt;

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
        try {
            final HttpGet request = new HttpGet(apiUrl + "/api/tags");
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                if (statusCode < 200 || statusCode >= 300) {
                    if (logger.isDebugEnabled()) {
                        logger.debug("[LLM:OLLAMA] Ollama availability check failed. url={}, statusCode={}", apiUrl, statusCode);
                    }
                    return false;
                }

                final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                return isModelAvailable(responseBody);
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] Ollama is not available. url={}, error={}", apiUrl, e.getMessage());
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
            logger.warn("Configured model not found in Ollama. model={}", configuredModel);
            return false;
        } catch (final Exception e) {
            logger.warn("[LLM:OLLAMA] Failed to parse Ollama models response. error={}", e.getMessage());
            return false;
        }
    }

    @Override
    public LlmChatResponse chat(final LlmChatRequest request) {
        final String url = getApiUrl() + "/api/chat";
        final Map<String, Object> requestBody = buildRequestBody(request, false);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OLLAMA] Sending chat request to Ollama. url={}, model={}, messageCount={}", url, requestBody.get("model"),
                    request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] requestBody={}", json);
            }
            final HttpPost httpRequest = new HttpPost(url);
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));

            try (var response = getHttpClient().execute(httpRequest)) {
                final int statusCode = response.getCode();
                if (statusCode < 200 || statusCode >= 300) {
                    logger.warn("Ollama API error. url={}, statusCode={}, message={}", url, statusCode, response.getReasonPhrase());
                    throw new LlmException("Ollama API error: " + statusCode + " " + response.getReasonPhrase(),
                            resolveErrorCode(statusCode));
                }

                final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OLLAMA] responseBody={}", responseBody);
                }
                final JsonNode jsonNode = objectMapper.readTree(responseBody);

                final LlmChatResponse chatResponse = new LlmChatResponse();
                if (jsonNode.has("message") && jsonNode.get("message").has("content")) {
                    chatResponse.setContent(jsonNode.get("message").get("content").asText());
                }
                if (jsonNode.has("done_reason")) {
                    chatResponse.setFinishReason(jsonNode.get("done_reason").asText());
                }
                if (jsonNode.has("model")) {
                    chatResponse.setModel(jsonNode.get("model").asText());
                }
                if (jsonNode.has("prompt_eval_count")) {
                    chatResponse.setPromptTokens(jsonNode.get("prompt_eval_count").asInt());
                }
                if (jsonNode.has("eval_count")) {
                    chatResponse.setCompletionTokens(jsonNode.get("eval_count").asInt());
                }

                if (logger.isDebugEnabled()) {
                    logger.debug(
                            "Received chat response from Ollama. model={}, promptTokens={}, completionTokens={}, contentLength={}, elapsedTime={}ms",
                            chatResponse.getModel(), chatResponse.getPromptTokens(), chatResponse.getCompletionTokens(),
                            chatResponse.getContent() != null ? chatResponse.getContent().length() : 0,
                            System.currentTimeMillis() - startTime);
                }

                return chatResponse;
            }
        } catch (final LlmException e) {
            throw e;
        } catch (final Exception e) {
            logger.warn("Failed to call Ollama API. url={}, error={}", url, e.getMessage(), e);
            throw new LlmException("Failed to call Ollama API", LlmException.ERROR_CONNECTION, e);
        }
    }

    @Override
    public void streamChat(final LlmChatRequest request, final LlmStreamCallback callback) {
        final String url = getApiUrl() + "/api/chat";
        final Map<String, Object> requestBody = buildRequestBody(request, true);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:OLLAMA] Starting streaming chat request to Ollama. url={}, model={}, messageCount={}", url,
                    requestBody.get("model"), request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:OLLAMA] requestBody={}", json);
            }
            final HttpPost httpRequest = new HttpPost(url);
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));

            try (var response = getHttpClient().execute(httpRequest)) {
                final int statusCode = response.getCode();
                if (statusCode < 200 || statusCode >= 300) {
                    logger.warn("Ollama streaming API error. url={}, statusCode={}, message={}", url, statusCode,
                            response.getReasonPhrase());
                    throw new LlmException("Ollama API error: " + statusCode + " " + response.getReasonPhrase(),
                            resolveErrorCode(statusCode));
                }

                if (response.getEntity() == null) {
                    logger.warn("Empty response from Ollama streaming API. url={}", url);
                    throw new LlmException("Empty response from Ollama");
                }

                int chunkCount = 0;
                try (BufferedReader reader =
                        new BufferedReader(new InputStreamReader(response.getEntity().getContent(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (StringUtil.isBlank(line)) {
                            continue;
                        }
                        try {
                            final JsonNode jsonNode = objectMapper.readTree(line);
                            final boolean done = jsonNode.has("done") && jsonNode.get("done").asBoolean();

                            if (jsonNode.has("message") && jsonNode.get("message").has("content")) {
                                final String content = jsonNode.get("message").get("content").asText();
                                callback.onChunk(content, done);
                                chunkCount++;
                            } else if (done) {
                                callback.onChunk("", true);
                            }

                            if (done) {
                                break;
                            }
                        } catch (final JsonProcessingException e) {
                            logger.warn("Failed to parse Ollama streaming response. line={}", line, e);
                        }
                    }
                }

                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:OLLAMA] Completed streaming chat from Ollama. url={}, chunkCount={}, elapsedTime={}ms", url,
                            chunkCount, System.currentTimeMillis() - startTime);
                }
            }
        } catch (final LlmException e) {
            callback.onError(e);
            throw e;
        } catch (final IOException e) {
            logger.warn("Failed to stream from Ollama API. url={}, error={}", url, e.getMessage(), e);
            final LlmException llmException = new LlmException("Failed to stream from Ollama API", LlmException.ERROR_CONNECTION, e);
            callback.onError(llmException);
            throw llmException;
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

        return body;
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

    /** Sets the system prompt for LLM interactions.
     * @param systemPrompt the system prompt */
    public void setSystemPrompt(final String systemPrompt) {
        this.systemPrompt = systemPrompt;
    }

    /** Sets the prompt for detecting user intent.
     * @param intentDetectionPrompt the intent detection prompt */
    public void setIntentDetectionPrompt(final String intentDetectionPrompt) {
        this.intentDetectionPrompt = intentDetectionPrompt;
    }

    /** Sets the system prompt for handling unclear intents.
     * @param unclearIntentSystemPrompt the unclear intent system prompt */
    public void setUnclearIntentSystemPrompt(final String unclearIntentSystemPrompt) {
        this.unclearIntentSystemPrompt = unclearIntentSystemPrompt;
    }

    /** Sets the system prompt for handling no results.
     * @param noResultsSystemPrompt the no results system prompt */
    public void setNoResultsSystemPrompt(final String noResultsSystemPrompt) {
        this.noResultsSystemPrompt = noResultsSystemPrompt;
    }

    /** Sets the system prompt for handling document not found.
     * @param documentNotFoundSystemPrompt the document not found system prompt */
    public void setDocumentNotFoundSystemPrompt(final String documentNotFoundSystemPrompt) {
        this.documentNotFoundSystemPrompt = documentNotFoundSystemPrompt;
    }

    /** Sets the prompt for evaluating responses.
     * @param evaluationPrompt the evaluation prompt */
    public void setEvaluationPrompt(final String evaluationPrompt) {
        this.evaluationPrompt = evaluationPrompt;
    }

    /** Sets the system prompt for answer generation.
     * @param answerGenerationSystemPrompt the answer generation system prompt */
    public void setAnswerGenerationSystemPrompt(final String answerGenerationSystemPrompt) {
        this.answerGenerationSystemPrompt = answerGenerationSystemPrompt;
    }

    /** Sets the system prompt for summary generation.
     * @param summarySystemPrompt the summary system prompt */
    public void setSummarySystemPrompt(final String summarySystemPrompt) {
        this.summarySystemPrompt = summarySystemPrompt;
    }

    /** Sets the system prompt for FAQ answer generation.
     * @param faqAnswerSystemPrompt the FAQ answer system prompt */
    public void setFaqAnswerSystemPrompt(final String faqAnswerSystemPrompt) {
        this.faqAnswerSystemPrompt = faqAnswerSystemPrompt;
    }

    /** Sets the system prompt for direct answer generation.
     * @param directAnswerSystemPrompt the direct answer system prompt */
    public void setDirectAnswerSystemPrompt(final String directAnswerSystemPrompt) {
        this.directAnswerSystemPrompt = directAnswerSystemPrompt;
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
     * @return the API URL
     */
    protected String getApiUrl() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.api.url", "http://localhost:11434");
    }

    @Override
    protected String getModel() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.model", "gemma3:4b");
    }

    @Override
    protected int getTimeout() {
        return Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.timeout", "60000"));
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
        final var config = ComponentUtil.getFessConfig();

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
            final String defaultTemp = config.getOrDefault(defaultPrefix + ".temperature", null);
            if (defaultTemp != null) {
                request.setTemperature(Double.parseDouble(defaultTemp));
            }
        }
        if (request.getMaxTokens() == null) {
            final String defaultMaxTokens = config.getOrDefault(defaultPrefix + ".max.tokens", null);
            if (defaultMaxTokens != null) {
                request.setMaxTokens(Integer.parseInt(defaultMaxTokens));
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
            break;
        case "evaluation":
            if (request.getTemperature() == null) {
                request.setTemperature(0.1);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(512);
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
            break;
        case "direct":
        case "faq":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(1024);
            }
            break;
        case "answer":
        case "summary":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(2048);
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
        return Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.availability.check.interval", "60"));
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
    protected int getContextMaxChars() {
        final int value = Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.chat.context.max.chars", "4000"));
        if (value <= 0) {
            logger.warn("Invalid context max chars: {}. Using default: 4000", value);
            return 4000;
        }
        return value;
    }

    @Override
    protected int getEvaluationMaxRelevantDocs() {
        final int value =
                Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.chat.evaluation.max.relevant.docs", "3"));
        if (value <= 0) {
            logger.warn("Invalid evaluation max relevant docs: {}. Using default: 3", value);
            return 3;
        }
        return value;
    }

    @Override
    protected int getEvaluationDescriptionMaxChars() {
        final int value =
                Integer.parseInt(ComponentUtil.getFessConfig().getOrDefault("rag.llm.ollama.chat.evaluation.description.max.chars", "500"));
        if (value <= 0) {
            logger.warn("Invalid evaluation description max chars: {}. Using default: 500", value);
            return 500;
        }
        return value;
    }

    @Override
    protected String getSystemPrompt() {
        if (systemPrompt == null) {
            throw new LlmException("systemPrompt is not configured for " + getName());
        }
        return systemPrompt;
    }

    @Override
    protected String getIntentDetectionPrompt() {
        if (intentDetectionPrompt == null) {
            throw new LlmException("intentDetectionPrompt is not configured for " + getName());
        }
        return intentDetectionPrompt;
    }

    @Override
    protected String getUnclearIntentSystemPrompt() {
        if (unclearIntentSystemPrompt == null) {
            throw new LlmException("unclearIntentSystemPrompt is not configured for " + getName());
        }
        return unclearIntentSystemPrompt;
    }

    @Override
    protected String getNoResultsSystemPrompt() {
        if (noResultsSystemPrompt == null) {
            throw new LlmException("noResultsSystemPrompt is not configured for " + getName());
        }
        return noResultsSystemPrompt;
    }

    @Override
    protected String getDocumentNotFoundSystemPrompt() {
        if (documentNotFoundSystemPrompt == null) {
            throw new LlmException("documentNotFoundSystemPrompt is not configured for " + getName());
        }
        return documentNotFoundSystemPrompt;
    }

    @Override
    protected String getEvaluationPrompt() {
        if (evaluationPrompt == null) {
            throw new LlmException("evaluationPrompt is not configured for " + getName());
        }
        return evaluationPrompt;
    }

    @Override
    protected String getAnswerGenerationSystemPrompt() {
        if (answerGenerationSystemPrompt == null) {
            throw new LlmException("answerGenerationSystemPrompt is not configured for " + getName());
        }
        return answerGenerationSystemPrompt;
    }

    @Override
    protected String getSummarySystemPrompt() {
        if (summarySystemPrompt == null) {
            throw new LlmException("summarySystemPrompt is not configured for " + getName());
        }
        return summarySystemPrompt;
    }

    @Override
    protected String getFaqAnswerSystemPrompt() {
        if (faqAnswerSystemPrompt == null) {
            throw new LlmException("faqAnswerSystemPrompt is not configured for " + getName());
        }
        return faqAnswerSystemPrompt;
    }

    @Override
    protected String getDirectAnswerSystemPrompt() {
        if (directAnswerSystemPrompt == null) {
            throw new LlmException("directAnswerSystemPrompt is not configured for " + getName());
        }
        return directAnswerSystemPrompt;
    }
}
