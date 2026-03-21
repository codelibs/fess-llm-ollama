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
            logger.warn("[LLM:OLLAMA] Configured model not found. model={}", configuredModel);
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
                    logger.warn("[LLM:OLLAMA] API error. url={}, statusCode={}, message={}", url, statusCode, response.getReasonPhrase());
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

                if (logger.isDebugEnabled() && jsonNode.has("message") && jsonNode.get("message").has("thinking")) {
                    final String thinking = jsonNode.get("message").get("thinking").asText();
                    logger.debug("[LLM:OLLAMA] Thinking response received. thinkingLength={}", thinking.length());
                }

                logger.info(
                        "[LLM:OLLAMA] Chat response received. model={}, promptTokens={}, completionTokens={}, contentLength={}, elapsedTime={}ms",
                        chatResponse.getModel(), chatResponse.getPromptTokens(), chatResponse.getCompletionTokens(),
                        chatResponse.getContent() != null ? chatResponse.getContent().length() : 0, System.currentTimeMillis() - startTime);

                return chatResponse;
            }
        } catch (final LlmException e) {
            throw e;
        } catch (final Exception e) {
            logger.warn("[LLM:OLLAMA] Failed to call Ollama API. url={}, error={}", url, e.getMessage(), e);
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
                    logger.warn("[LLM:OLLAMA] Streaming API error. url={}, statusCode={}, message={}", url, statusCode,
                            response.getReasonPhrase());
                    throw new LlmException("Ollama API error: " + statusCode + " " + response.getReasonPhrase(),
                            resolveErrorCode(statusCode));
                }

                if (response.getEntity() == null) {
                    logger.warn("[LLM:OLLAMA] Empty response from Ollama streaming API. url={}", url);
                    throw new LlmException("Empty response from Ollama");
                }

                int chunkCount = 0;
                long firstChunkTime = 0;
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
                                if (content.isEmpty() && !done && jsonNode.get("message").has("thinking")) {
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
                                break;
                            }
                        } catch (final JsonProcessingException e) {
                            logger.warn("[LLM:OLLAMA] Failed to parse streaming response. line={}", line, e);
                        }
                    }
                }

                logger.info("[LLM:OLLAMA] Stream completed. chunkCount={}, firstChunkMs={}, elapsedTime={}ms", chunkCount, firstChunkTime,
                        System.currentTimeMillis() - startTime);
            }
        } catch (final LlmException e) {
            callback.onError(e);
            throw e;
        } catch (final IOException e) {
            logger.warn("[LLM:OLLAMA] Failed to stream from Ollama API. url={}, error={}", url, e.getMessage(), e);
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

        final Integer thinkingBudget = request.getThinkingBudget();
        if (thinkingBudget != null) {
            body.put("think", thinkingBudget > 0);
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
        return getConfigInt("timeout", 60000);
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
            if (request.getTemperature() == null) {
                request.setTemperature(0.5);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(2048);
            }
            break;
        case "summary":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(2048);
            }
            break;
        case "queryregeneration":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
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

}
