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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.hc.client5.http.config.ConnectionConfig;
import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManagerBuilder;
import org.apache.hc.core5.util.Timeout;
import org.codelibs.fess.llm.LlmChatRequest;
import org.codelibs.fess.llm.LlmChatResponse;
import org.codelibs.fess.llm.LlmException;
import org.codelibs.fess.llm.LlmMessage;
import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

public class OllamaLlmClientTest extends UnitFessTestCase {

    private TestableOllamaLlmClient client;

    @Override
    public void setUp(final TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableOllamaLlmClient();
    }

    @Override
    public void tearDown(final TestInfo testInfo) throws Exception {
        if (client != null) {
            client.destroy();
        }
        super.tearDown(testInfo);
    }

    @Test
    public void test_getName() {
        assertEquals("ollama", client.getName());
    }

    @Test
    public void test_isModelAvailable_withMatchingModel() {
        client.setTestModel("llama3:latest");
        final String responseBody = "{\"models\":[{\"name\":\"llama3:latest\"},{\"name\":\"mistral:latest\"}]}";
        assertTrue(client.isModelAvailable(responseBody));
    }

    @Test
    public void test_isModelAvailable_withNonMatchingModel() {
        client.setTestModel("gpt-4");
        final String responseBody = "{\"models\":[{\"name\":\"llama3:latest\"},{\"name\":\"mistral:latest\"}]}";
        assertFalse(client.isModelAvailable(responseBody));
    }

    @Test
    public void test_isModelAvailable_withBlankModel() {
        client.setTestModel("");
        final String responseBody = "{\"models\":[{\"name\":\"llama3:latest\"}]}";
        assertTrue(client.isModelAvailable(responseBody));
    }

    @Test
    public void test_isModelAvailable_withNullModel() {
        client.setTestModel(null);
        final String responseBody = "{\"models\":[{\"name\":\"llama3:latest\"}]}";
        assertTrue(client.isModelAvailable(responseBody));
    }

    @Test
    public void test_isModelAvailable_withEmptyModels() {
        client.setTestModel("llama3:latest");
        final String responseBody = "{\"models\":[]}";
        assertFalse(client.isModelAvailable(responseBody));
    }

    @Test
    public void test_isModelAvailable_withNoModelsField() {
        client.setTestModel("llama3:latest");
        final String responseBody = "{}";
        assertFalse(client.isModelAvailable(responseBody));
    }

    @Test
    public void test_isModelAvailable_withInvalidJson() {
        client.setTestModel("llama3:latest");
        final String responseBody = "invalid json";
        assertFalse(client.isModelAvailable(responseBody));
    }

    @Test
    public void test_convertMessage() {
        final LlmMessage message = new LlmMessage("user", "Hello");
        final Map<String, String> result = client.convertMessage(message);
        assertEquals("user", result.get("role"));
        assertEquals("Hello", result.get("content"));
    }

    @Test
    public void test_convertMessage_systemRole() {
        final LlmMessage message = new LlmMessage("system", "You are a helpful assistant.");
        final Map<String, String> result = client.convertMessage(message);
        assertEquals("system", result.get("role"));
        assertEquals("You are a helpful assistant.", result.get("content"));
    }

    @Test
    public void test_convertMessage_assistantRole() {
        final LlmMessage message = new LlmMessage("assistant", "I can help you.");
        final Map<String, String> result = client.convertMessage(message);
        assertEquals("assistant", result.get("role"));
        assertEquals("I can help you.", result.get("content"));
    }

    @Test
    public void test_buildRequestBody_withDefaults() {
        client.setTestModel("llama3:latest");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(1000);

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setTemperature(0.7);
        request.setMaxTokens(1000);

        final Map<String, Object> body = client.buildRequestBody(request, false);
        assertEquals("llama3:latest", body.get("model"));
        assertEquals(Boolean.FALSE, body.get("stream"));
        assertNotNull(body.get("messages"));

        @SuppressWarnings("unchecked")
        final List<Map<String, String>> messages = (List<Map<String, String>>) body.get("messages");
        assertEquals(1, messages.size());
        assertEquals("user", messages.get(0).get("role"));
        assertEquals("Hello", messages.get(0).get("content"));

        @SuppressWarnings("unchecked")
        final Map<String, Object> options = (Map<String, Object>) body.get("options");
        assertNotNull(options);
        assertEquals(0.7, options.get("temperature"));
        assertEquals(1000, options.get("num_predict"));
    }

    @Test
    public void test_buildRequestBody_withRequestOverrides() {
        client.setTestModel("llama3:latest");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(1000);

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setModel("mistral:latest");
        request.setTemperature(0.5);
        request.setMaxTokens(500);

        final Map<String, Object> body = client.buildRequestBody(request, true);
        assertEquals("mistral:latest", body.get("model"));
        assertEquals(Boolean.TRUE, body.get("stream"));

        @SuppressWarnings("unchecked")
        final Map<String, Object> options = (Map<String, Object>) body.get("options");
        assertNotNull(options);
        assertEquals(0.5, options.get("temperature"));
        assertEquals(500, options.get("num_predict"));
    }

    @Test
    public void test_buildRequestBody_withMultipleMessages() {
        client.setTestModel("llama3:latest");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(1000);

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("system", "You are a helpful assistant."));
        request.addMessage(new LlmMessage("user", "Hello"));
        request.addMessage(new LlmMessage("assistant", "Hi! How can I help?"));
        request.addMessage(new LlmMessage("user", "Tell me about Fess."));

        final Map<String, Object> body = client.buildRequestBody(request, false);

        @SuppressWarnings("unchecked")
        final List<Map<String, String>> messages = (List<Map<String, String>>) body.get("messages");
        assertEquals(4, messages.size());
    }

    @Test
    public void test_buildRequestBody_withNumCtx() {
        client.setTestModel("llama3:latest");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setTemperature(0.7);
        request.putExtraParam("num_ctx", "4096");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        @SuppressWarnings("unchecked")
        final Map<String, Object> options = (Map<String, Object>) body.get("options");
        assertNotNull(options);
        assertEquals(4096, options.get("num_ctx"));
    }

    @Test
    public void test_buildRequestBody_withTopPAndTopK() {
        client.setTestModel("llama3:latest");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.putExtraParam("top_p", "0.9");
        request.putExtraParam("top_k", "40");
        request.putExtraParam("num_ctx", "8192");

        final Map<String, Object> body = client.buildRequestBody(request, false);

        @SuppressWarnings("unchecked")
        final Map<String, Object> options = (Map<String, Object>) body.get("options");
        assertNotNull(options);
        assertEquals(0.9, options.get("top_p"));
        assertEquals(40, options.get("top_k"));
        assertEquals(8192, options.get("num_ctx"));
    }

    @Test
    public void test_parseOptionValue_integer() {
        assertEquals(42, client.parseOptionValue("42"));
        assertEquals(0, client.parseOptionValue("0"));
        assertEquals(-1, client.parseOptionValue("-1"));
    }

    @Test
    public void test_parseOptionValue_double() {
        assertEquals(1.1, client.parseOptionValue("1.1"));
        assertEquals(0.5, client.parseOptionValue("0.5"));
    }

    @Test
    public void test_parseOptionValue_boolean() {
        assertEquals(true, client.parseOptionValue("true"));
        assertEquals(false, client.parseOptionValue("false"));
        assertEquals(true, client.parseOptionValue("TRUE"));
    }

    @Test
    public void test_parseOptionValue_string() {
        assertEquals("hello", client.parseOptionValue("hello"));
    }

    @Test
    public void test_init_and_destroy() {
        final TestableOllamaLlmClient testClient = new TestableOllamaLlmClient();
        testClient.setTestApiUrl("http://localhost:11434");
        testClient.setTestModel("llama3:latest");
        testClient.setTestTimeout(30000);
        // init() requires ComponentUtil, so we test the HTTP client setup directly
        assertNull(testClient.getTestHttpClient());
        testClient.initHttpClient();
        assertNotNull(testClient.getTestHttpClient());
        testClient.destroy();
        assertNull(testClient.getTestHttpClient());
    }

    @Test
    public void test_destroy_withNullHttpClient() {
        final TestableOllamaLlmClient testClient = new TestableOllamaLlmClient();
        assertNull(testClient.getTestHttpClient());
        // Should not throw
        testClient.destroy();
        assertNull(testClient.getTestHttpClient());
    }

    @Test
    public void test_chat_success() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"message\":{\"content\":\"Hello! How can I help?\"},\"done_reason\":\"stop\","
                    + "\"model\":\"llama3:latest\",\"prompt_eval_count\":10,\"eval_count\":20,\"done\":true}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.setTestTemperature(0.7);
            client.setTestMaxTokens(1000);
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final LlmChatResponse response = client.chat(request);

            assertNotNull(response);
            assertEquals("Hello! How can I help?", response.getContent());
            assertEquals("stop", response.getFinishReason());
            assertEquals("llama3:latest", response.getModel());
            assertEquals(10, response.getPromptTokens());
            assertEquals(20, response.getCompletionTokens());

            final RecordedRequest recordedRequest = server.takeRequest();
            assertEquals("/api/chat", recordedRequest.getPath());
            assertEquals("POST", recordedRequest.getMethod());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_chat_apiError() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            server.enqueue(new MockResponse().setResponseCode(500).setBody("Internal Server Error"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.setTestTemperature(0.7);
            client.setTestMaxTokens(1000);
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            try {
                client.chat(request);
                fail("Expected LlmException");
            } catch (final LlmException e) {
                assertTrue(e.getMessage().contains("Ollama API error"));
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_success() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String streamResponse = "{\"message\":{\"content\":\"Hello\"},\"done\":false}\n"
                    + "{\"message\":{\"content\":\" world\"},\"done\":false}\n" + "{\"message\":{\"content\":\"!\"},\"done\":true}\n";
            server.enqueue(new MockResponse().setBody(streamResponse).setHeader("Content-Type", "application/x-ndjson"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.setTestTemperature(0.7);
            client.setTestMaxTokens(1000);
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final List<String> chunks = new ArrayList<>();
            final List<Boolean> doneFlags = new ArrayList<>();

            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    chunks.add(content);
                    doneFlags.add(done);
                }

                @Override
                public void onError(final Throwable e) {
                    fail("Unexpected error: " + e.getMessage());
                }
            });

            assertEquals(3, chunks.size());
            assertEquals("Hello", chunks.get(0));
            assertEquals(" world", chunks.get(1));
            assertEquals("!", chunks.get(2));
            assertFalse(doneFlags.get(0));
            assertFalse(doneFlags.get(1));
            assertTrue(doneFlags.get(2));
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_apiError() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            server.enqueue(new MockResponse().setResponseCode(503).setBody("Service Unavailable"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.setTestTemperature(0.7);
            client.setTestMaxTokens(1000);
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final List<Throwable> errors = new ArrayList<>();

            try {
                client.streamChat(request, new LlmStreamCallback() {
                    @Override
                    public void onChunk(final String content, final boolean done) {
                        fail("Should not receive chunks on error");
                    }

                    @Override
                    public void onError(final Throwable e) {
                        errors.add(e);
                    }
                });
                fail("Expected LlmException");
            } catch (final LlmException e) {
                assertTrue(e.getMessage().contains("Ollama API error"));
            }

            assertEquals(1, errors.size());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_checkAvailabilityNow_success() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String tagsResponse = "{\"models\":[{\"name\":\"llama3:latest\"}]}";
            server.enqueue(new MockResponse().setBody(tagsResponse).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.initHttpClient();

            assertTrue(client.checkAvailabilityNow());

            final RecordedRequest recordedRequest = server.takeRequest();
            assertEquals("/api/tags", recordedRequest.getPath());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_checkAvailabilityNow_blankApiUrl() {
        client.setTestApiUrl("");
        assertFalse(client.checkAvailabilityNow());
    }

    @Test
    public void test_checkAvailabilityNow_nullApiUrl() {
        client.setTestApiUrl(null);
        assertFalse(client.checkAvailabilityNow());
    }

    @Test
    public void test_checkAvailabilityNow_serverError() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            server.enqueue(new MockResponse().setResponseCode(500).setBody("Internal Server Error"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.initHttpClient();

            assertFalse(client.checkAvailabilityNow());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_checkAvailabilityNow_modelNotFound() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String tagsResponse = "{\"models\":[{\"name\":\"mistral:latest\"}]}";
            server.enqueue(new MockResponse().setBody(tagsResponse).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.initHttpClient();

            assertFalse(client.checkAvailabilityNow());
        } finally {
            server.shutdown();
        }
    }

    // --- Thinking (reasoning model) tests ---

    @Test
    public void test_buildRequestBody_thinkingBudgetZero() {
        client.setTestModel("qwen3.5:35b");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setThinkingBudget(0);

        final Map<String, Object> body = client.buildRequestBody(request, false);
        assertEquals(Boolean.FALSE, body.get("think"));
    }

    @Test
    public void test_buildRequestBody_thinkingBudgetPositive() {
        client.setTestModel("qwen3.5:35b");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setThinkingBudget(1024);

        final Map<String, Object> body = client.buildRequestBody(request, false);
        assertEquals(Boolean.TRUE, body.get("think"));
    }

    @Test
    public void test_buildRequestBody_thinkingBudgetNull() {
        client.setTestModel("qwen3.5:35b");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));

        final Map<String, Object> body = client.buildRequestBody(request, false);
        assertFalse(body.containsKey("think"));
    }

    @Test
    public void test_chat_withThinkingResponse() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"message\":{\"content\":\"The answer is 42.\","
                    + "\"thinking\":\"Let me think about this carefully...\"},\"done_reason\":\"stop\","
                    + "\"model\":\"qwen3.5:35b\",\"prompt_eval_count\":15,\"eval_count\":30,\"done\":true}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("qwen3.5:35b");
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "What is the meaning of life?"));

            final LlmChatResponse response = client.chat(request);

            assertNotNull(response);
            assertEquals("The answer is 42.", response.getContent());
            assertEquals("stop", response.getFinishReason());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_withThinkingChunks() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String streamResponse = "{\"message\":{\"content\":\"\",\"thinking\":\"Let me think...\"},\"done\":false}\n"
                    + "{\"message\":{\"content\":\"\",\"thinking\":\"Still thinking...\"},\"done\":false}\n"
                    + "{\"message\":{\"content\":\"Hello\"},\"done\":false}\n" + "{\"message\":{\"content\":\" world\"},\"done\":false}\n"
                    + "{\"message\":{\"content\":\"!\"},\"done\":true}\n";
            server.enqueue(new MockResponse().setBody(streamResponse).setHeader("Content-Type", "application/x-ndjson"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("qwen3.5:35b");
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final List<String> chunks = new ArrayList<>();
            final List<Boolean> doneFlags = new ArrayList<>();

            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    chunks.add(content);
                    doneFlags.add(done);
                }

                @Override
                public void onError(final Throwable e) {
                    fail("Unexpected error: " + e.getMessage());
                }
            });

            assertEquals(3, chunks.size());
            assertEquals("Hello", chunks.get(0));
            assertEquals(" world", chunks.get(1));
            assertEquals("!", chunks.get(2));
            assertFalse(doneFlags.get(0));
            assertFalse(doneFlags.get(1));
            assertTrue(doneFlags.get(2));
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_thinkingDisabled() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String streamResponse = "{\"message\":{\"content\":\"Hello\"},\"done\":false}\n"
                    + "{\"message\":{\"content\":\" world\"},\"done\":false}\n" + "{\"message\":{\"content\":\"!\"},\"done\":true}\n";
            server.enqueue(new MockResponse().setBody(streamResponse).setHeader("Content-Type", "application/x-ndjson"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("qwen3.5:35b");
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));
            request.setThinkingBudget(0);

            final List<String> chunks = new ArrayList<>();

            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    chunks.add(content);
                }

                @Override
                public void onError(final Throwable e) {
                    fail("Unexpected error: " + e.getMessage());
                }
            });

            assertEquals(3, chunks.size());
            assertEquals("Hello", chunks.get(0));
            assertEquals(" world", chunks.get(1));
            assertEquals("!", chunks.get(2));
        } finally {
            server.shutdown();
        }
    }

    // --- applyDefaultParams thinking tests ---

    @Test
    public void test_applyDefaultParams_intent_noThinkingDefault() {
        final LlmChatRequest request = new LlmChatRequest();
        assertNull(request.getThinkingBudget());

        client.applyDefaultParams(request, "intent");

        assertNull(request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_evaluation_noThinkingDefault() {
        final LlmChatRequest request = new LlmChatRequest();
        assertNull(request.getThinkingBudget());

        client.applyDefaultParams(request, "evaluation");

        assertNull(request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_answer_noThinkingDefault() {
        final LlmChatRequest request = new LlmChatRequest();
        assertNull(request.getThinkingBudget());

        client.applyDefaultParams(request, "answer");

        assertNull(request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_summary_noThinkingDefault() {
        final LlmChatRequest request = new LlmChatRequest();
        assertNull(request.getThinkingBudget());

        client.applyDefaultParams(request, "summary");

        assertNull(request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_direct_noThinkingDefault() {
        final LlmChatRequest request = new LlmChatRequest();
        assertNull(request.getThinkingBudget());

        client.applyDefaultParams(request, "direct");

        assertNull(request.getThinkingBudget());
    }

    // --- gemma3 compatibility tests (non-reasoning model) ---

    @Test
    public void test_chat_gemma3_noThinkingField() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"message\":{\"content\":\"Hello! How can I help?\"},\"done_reason\":\"stop\","
                    + "\"model\":\"gemma3:4b\",\"prompt_eval_count\":10,\"eval_count\":20,\"done\":true}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("gemma3:4b");
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final LlmChatResponse response = client.chat(request);

            assertNotNull(response);
            assertEquals("Hello! How can I help?", response.getContent());
            assertEquals("stop", response.getFinishReason());
            assertEquals("gemma3:4b", response.getModel());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_gemma3_noThinkingField() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String streamResponse = "{\"message\":{\"content\":\"Hello\"},\"done\":false}\n"
                    + "{\"message\":{\"content\":\" from\"},\"done\":false}\n" + "{\"message\":{\"content\":\" gemma3\"},\"done\":false}\n"
                    + "{\"message\":{\"content\":\"!\"},\"done\":true}\n";
            server.enqueue(new MockResponse().setBody(streamResponse).setHeader("Content-Type", "application/x-ndjson"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("gemma3:4b");
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final List<String> chunks = new ArrayList<>();
            final List<Boolean> doneFlags = new ArrayList<>();

            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    chunks.add(content);
                    doneFlags.add(done);
                }

                @Override
                public void onError(final Throwable e) {
                    fail("Unexpected error: " + e.getMessage());
                }
            });

            assertEquals(4, chunks.size());
            assertEquals("Hello", chunks.get(0));
            assertEquals(" from", chunks.get(1));
            assertEquals(" gemma3", chunks.get(2));
            assertEquals("!", chunks.get(3));
            assertFalse(doneFlags.get(0));
            assertFalse(doneFlags.get(1));
            assertFalse(doneFlags.get(2));
            assertTrue(doneFlags.get(3));
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_buildRequestBody_gemma3_thinkingBudgetZero() {
        client.setTestModel("gemma3:4b");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setTemperature(0.7);
        request.setMaxTokens(1000);
        request.setThinkingBudget(0);

        final Map<String, Object> body = client.buildRequestBody(request, false);
        assertEquals("gemma3:4b", body.get("model"));
        assertEquals(Boolean.FALSE, body.get("think"));

        @SuppressWarnings("unchecked")
        final Map<String, Object> options = (Map<String, Object>) body.get("options");
        assertNotNull(options);
        assertEquals(0.7, options.get("temperature"));
        assertEquals(1000, options.get("num_predict"));
    }

    @Test
    public void test_getHistoryMaxChars_default() {
        assertEquals(4000, client.testGetHistoryMaxChars());
    }

    @Test
    public void test_getIntentHistoryMaxMessages_default() {
        assertEquals(6, client.testGetIntentHistoryMaxMessages());
    }

    @Test
    public void test_getHistoryAssistantMaxChars_default() {
        assertEquals(500, client.testGetHistoryAssistantMaxChars());
    }

    // --- Testable subclass ---

    static class TestableOllamaLlmClient extends OllamaLlmClient {

        private String testApiUrl = "http://localhost:11434";
        private String testModel = "llama3:latest";
        private int testTimeout = 30000;
        private double testTemperature = 0.7;
        private int testMaxTokens = 1000;

        void setTestApiUrl(final String apiUrl) {
            this.testApiUrl = apiUrl;
        }

        void setTestModel(final String model) {
            this.testModel = model;
        }

        void setTestTimeout(final int timeout) {
            this.testTimeout = timeout;
        }

        void setTestTemperature(final double temperature) {
            this.testTemperature = temperature;
        }

        void setTestMaxTokens(final int maxTokens) {
            this.testMaxTokens = maxTokens;
        }

        @Override
        protected String getApiUrl() {
            return testApiUrl;
        }

        @Override
        protected String getModel() {
            return testModel;
        }

        @Override
        protected int getTimeout() {
            return testTimeout;
        }

        protected double getTemperature() {
            return testTemperature;
        }

        protected int getMaxTokens() {
            return testMaxTokens;
        }

        @Override
        protected int getHistoryMaxChars() {
            return 4000;
        }

        @Override
        protected int getIntentHistoryMaxMessages() {
            return 6;
        }

        @Override
        protected int getIntentHistoryMaxChars() {
            return 3000;
        }

        @Override
        public int getHistoryAssistantMaxChars() {
            return 500;
        }

        @Override
        public int getHistoryAssistantSummaryMaxChars() {
            return 500;
        }

        int testGetHistoryMaxChars() {
            return getHistoryMaxChars();
        }

        int testGetIntentHistoryMaxMessages() {
            return getIntentHistoryMaxMessages();
        }

        int testGetHistoryAssistantMaxChars() {
            return getHistoryAssistantMaxChars();
        }

        public CloseableHttpClient getTestHttpClient() {
            return httpClient;
        }

        void initHttpClient() {
            final int timeout = getTimeout();
            final RequestConfig requestConfig = RequestConfig.custom()
                    .setConnectionRequestTimeout(Timeout.ofMilliseconds(timeout))
                    .setResponseTimeout(Timeout.ofMilliseconds(timeout))
                    .build();
            httpClient = HttpClients.custom()
                    .setConnectionManager(PoolingHttpClientConnectionManagerBuilder.create()
                            .setDefaultConnectionConfig(
                                    ConnectionConfig.custom().setConnectTimeout(Timeout.ofMilliseconds(timeout)).build())
                            .build())
                    .setDefaultRequestConfig(requestConfig)
                    .disableAutomaticRetries()
                    .build();
        }
    }
}
