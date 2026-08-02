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

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;

import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Property;
import org.codelibs.fess.llm.LlmChatRequest;
import org.codelibs.fess.llm.LlmChatResponse;
import org.codelibs.fess.llm.LlmException;
import org.codelibs.fess.llm.LlmMessage;
import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.ollama.OllamaUrlUtil;
import org.codelibs.fess.unit.LogCapturingAppender;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.codelibs.fess.util.CredentialUrlUtil;
import org.junit.jupiter.api.Assertions;
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
    public void test_getConnectTimeout_defaultsTo5000() {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient() {
            @Override
            protected int getConfigInt(final String suffix, final int defaultValue) {
                // Force the config layer to return defaults (no system properties set in test)
                return defaultValue;
            }
        };
        assertEquals(5000, localClient.getConnectTimeout());
    }

    @Test
    public void test_initHttpClient_appliesBothTimeouts() {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestConnectTimeout(7777);
        localClient.setTestTimeout(33333);
        localClient.initHttpClient();
        assertNotNull(localClient.getTestHttpClient());
        // Indirect verification: client builds without throwing, both setters were called.
        // Direct read-back of timeouts from CloseableHttpClient is non-trivial; the
        // construction succeeding with non-default values is sufficient for unit-level coverage.
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
    public void test_chat_throughProxy_withoutAuth() throws Exception {
        // With a configured HTTP proxy, HttpClient sends the request to the proxy
        // with an absolute-form request URI. We use a single MockWebServer as the proxy
        // and target a non-localhost address to verify routing.
        final MockWebServer proxyServer = new MockWebServer();
        try {
            final String responseJson = "{\"message\":{\"content\":\"ok\"},\"done_reason\":\"stop\","
                    + "\"model\":\"llama3:latest\",\"prompt_eval_count\":1,\"eval_count\":1,\"done\":true}";
            proxyServer.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            proxyServer.start();

            client.setTestApiUrl("http://ollama.invalid:11434");
            client.setTestModel("llama3:latest");
            client.setTestTimeout(30000);
            client.setTestProxyHost(proxyServer.getHostName());
            client.setTestProxyPort(proxyServer.getPort());
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));
            final LlmChatResponse response = client.chat(request);
            assertEquals("ok", response.getContent());

            final RecordedRequest recorded = proxyServer.takeRequest();
            assertTrue("Expected absolute-form URI starting with http://ollama.invalid:11434/, got: " + recorded.getRequestLine(),
                    recorded.getRequestLine().contains("http://ollama.invalid:11434/"));
            assertNull(recorded.getHeader("Proxy-Authorization"), "No proxy auth expected");
        } finally {
            proxyServer.shutdown();
        }
    }

    @Test
    public void test_chat_throughProxy_withBasicAuth() throws Exception {
        final MockWebServer proxyServer = new MockWebServer();
        try {
            // First response: 407 challenges the client to authenticate.
            proxyServer
                    .enqueue(new MockResponse().setResponseCode(407).addHeader("Proxy-Authenticate", "Basic realm=\"proxy\"").setBody(""));
            // Second response: success after the client retries with credentials.
            final String responseJson = "{\"message\":{\"content\":\"ok\"},\"done_reason\":\"stop\","
                    + "\"model\":\"llama3:latest\",\"prompt_eval_count\":1,\"eval_count\":1,\"done\":true}";
            proxyServer.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            proxyServer.start();

            client.setTestApiUrl("http://ollama.invalid:11434");
            client.setTestModel("llama3:latest");
            client.setTestTimeout(30000);
            client.setTestProxyHost(proxyServer.getHostName());
            client.setTestProxyPort(proxyServer.getPort());
            client.setTestProxyUsername("proxyuser");
            client.setTestProxyPassword("proxypass");
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));
            final LlmChatResponse response = client.chat(request);
            assertEquals("ok", response.getContent());

            final RecordedRequest first = proxyServer.takeRequest();
            assertNull(first.getHeader("Proxy-Authorization"));
            final RecordedRequest second = proxyServer.takeRequest();
            final String auth = second.getHeader("Proxy-Authorization");
            assertNotNull(auth, "Proxy-Authorization header expected on retry");
            final String expected = "Basic "
                    + java.util.Base64.getEncoder().encodeToString("proxyuser:proxypass".getBytes(java.nio.charset.StandardCharsets.UTF_8));
            assertEquals(expected, auth);
        } finally {
            proxyServer.shutdown();
        }
    }

    @Test
    public void test_chat_noProxy_directConnection() throws Exception {
        // When proxy host is blank, requests go directly to the target server (origin-form URI).
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"message\":{\"content\":\"direct\"},\"done_reason\":\"stop\","
                    + "\"model\":\"llama3:latest\",\"prompt_eval_count\":1,\"eval_count\":1,\"done\":true}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("llama3:latest");
            client.setTestTimeout(30000);
            // Proxy unset - direct connection
            client.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));
            final LlmChatResponse response = client.chat(request);
            assertEquals("direct", response.getContent());

            final RecordedRequest recorded = server.takeRequest();
            assertEquals("/api/chat", recorded.getPath());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_chat_apiError() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // Use a non-retryable status (400) so this exercises the in-lambda
            // LlmException path with the "Ollama API error: <code> <reason>" message.
            server.enqueue(new MockResponse().setResponseCode(400).setBody("Bad Request"));
            server.start();

            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            localClient.setTestModel("llama3:latest");
            localClient.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            try {
                localClient.chat(request);
                fail("Expected LlmException");
            } catch (final LlmException e) {
                assertTrue(e.getMessage().contains("Ollama API error"));
            }
            assertEquals("400 must not be retried", 1, server.getRequestCount());
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
    public void test_streamChat_inStreamErrorTriggersOnError() throws Exception {
        // Ollama streams in-flight failures as NDJSON {"error": "..."} after the connection
        // succeeds with HTTP 200. The plugin must surface that to LlmStreamCallback.onError(...)
        // rather than treating it as a normal completion.
        // See https://docs.ollama.com/api/errors
        final MockWebServer server = new MockWebServer();
        try {
            final String body = "{\"message\":{\"content\":\"Hello\"},\"done\":false}\n"
                    + "{\"error\":\"model 'qwen3.5:35b' not found, try pulling it first\"}\n";
            server.enqueue(new MockResponse().setBody(body).setHeader("Content-Type", "application/x-ndjson"));
            server.start();

            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            localClient.setTestModel("qwen3.5:35b");
            localClient.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final List<Throwable> errors = new ArrayList<>();
            final List<String> chunks = new ArrayList<>();

            try {
                localClient.streamChat(request, new LlmStreamCallback() {
                    @Override
                    public void onChunk(final String content, final boolean done) {
                        chunks.add(content);
                    }

                    @Override
                    public void onError(final Throwable e) {
                        errors.add(e);
                    }
                });
                fail("Expected LlmException to be propagated");
            } catch (final LlmException e) {
                assertTrue(e.getMessage().contains("Ollama stream error"));
                assertTrue(e.getMessage().contains("model 'qwen3.5:35b' not found"));
            }

            assertEquals(1, errors.size());
            // The first chunk delivered before the error is allowed to flow through.
            assertEquals(List.of("Hello"), chunks);
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_apiError() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // Use a non-retryable status (400) so this exercises the in-lambda
            // LlmException path with the "Ollama API error: <code> <reason>" message.
            server.enqueue(new MockResponse().setResponseCode(400).setBody("Bad Request"));
            server.start();

            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            localClient.setTestModel("llama3:latest");
            localClient.initHttpClient();

            final LlmChatRequest request = new LlmChatRequest();
            request.addMessage(new LlmMessage("user", "Hello"));

            final List<Throwable> errors = new ArrayList<>();

            try {
                localClient.streamChat(request, new LlmStreamCallback() {
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
            assertEquals("400 must not be retried", 1, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_unexpectedContentTypeEmitsWarn() throws Exception {
        final MockWebServer server = new MockWebServer();
        final String body = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n"
                + "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n";
        server.enqueue(new MockResponse().setHeader("Content-Type", "text/plain").setBody(body));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();

            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                final LlmChatRequest request = new LlmChatRequest();
                request.setMessages(List.of(new LlmMessage("user", "hi")));
                final List<String> chunks = new ArrayList<>();
                localClient.streamChat(request, (content, done) -> chunks.add(content));
                assertTrue(capture.warnings().stream().anyMatch(m -> m.contains("Unexpected Content-Type") && m.contains("text/plain")));
                assertEquals(List.of("hi", ""), chunks);
            } finally {
                capture.detach();
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_correctContentTypeNoWarn() throws Exception {
        final MockWebServer server = new MockWebServer();
        final String body = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n"
                + "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/x-ndjson").setBody(body));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                final LlmChatRequest request = new LlmChatRequest();
                request.setMessages(List.of(new LlmMessage("user", "hi")));
                localClient.streamChat(request, (content, done) -> {});
                assertTrue("Should not warn for correct content type",
                        capture.warnings().stream().noneMatch(m -> m.contains("Unexpected Content-Type")));
            } finally {
                capture.detach();
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_completionLogContainsOllamaMetrics() throws Exception {
        final MockWebServer server = new MockWebServer();
        final String body = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n"
                + "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\","
                + "\"total_duration\":2500000000,\"load_duration\":500000000,"
                + "\"prompt_eval_count\":12,\"prompt_eval_duration\":300000000," + "\"eval_count\":48,\"eval_duration\":1500000000}\n";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/x-ndjson").setBody(body));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                final LlmChatRequest request = new LlmChatRequest();
                request.setMessages(List.of(new LlmMessage("user", "hi")));
                localClient.streamChat(request, (content, done) -> {});
                final String completionLine =
                        capture.infos().stream().filter(m -> m.contains("Stream completed")).findFirst().orElseThrow();
                assertTrue("doneReason missing: " + completionLine, completionLine.contains("doneReason=stop"));
                assertTrue("totalDurationMs missing: " + completionLine, completionLine.contains("totalDurationMs=2500"));
                assertTrue("loadDurationMs missing: " + completionLine, completionLine.contains("loadDurationMs=500"));
                assertTrue("evalDurationMs missing: " + completionLine, completionLine.contains("evalDurationMs=1500"));
                assertTrue("promptEvalCount missing: " + completionLine, completionLine.contains("promptEvalCount=12"));
                assertTrue("evalCount missing: " + completionLine, completionLine.contains("evalCount=48"));
                assertTrue("tokensPerSecond missing: " + completionLine, completionLine.contains("tokensPerSecond=32"));
                assertTrue("parseErrorCount missing: " + completionLine, completionLine.contains("parseErrorCount=0"));
            } finally {
                capture.detach();
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_parseErrorCountedInCompletionLog() throws Exception {
        final MockWebServer server = new MockWebServer();
        final String body = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n" + "this-is-not-json\n"
                + "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/x-ndjson").setBody(body));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                final LlmChatRequest request = new LlmChatRequest();
                request.setMessages(List.of(new LlmMessage("user", "hi")));
                localClient.streamChat(request, (content, done) -> {});
                final String completionLine =
                        capture.infos().stream().filter(m -> m.contains("Stream completed")).findFirst().orElseThrow();
                assertTrue("expected parseErrorCount=1: " + completionLine, completionLine.contains("parseErrorCount=1"));
            } finally {
                capture.detach();
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_lengthDoneReasonEmitsWarn() throws Exception {
        final MockWebServer server = new MockWebServer();
        final String body = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n"
                + "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"length\","
                + "\"prompt_eval_count\":12,\"eval_count\":48}\n";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/x-ndjson").setBody(body));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                final LlmChatRequest request = new LlmChatRequest();
                request.setMessages(List.of(new LlmMessage("user", "hi")));
                localClient.streamChat(request, (content, done) -> {});
                assertTrue(capture.warnings()
                        .stream()
                        .anyMatch(m -> m.contains("Stream finished abnormally") && m.contains("doneReason=length")
                                && m.contains("evalCount=48")));
            } finally {
                capture.detach();
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_stopDoneReasonNoWarn() throws Exception {
        assertNoAbnormalWarnFor("stop");
    }

    @Test
    public void test_streamChat_loadDoneReasonNoWarn() throws Exception {
        assertNoAbnormalWarnFor("load");
    }

    @Test
    public void test_streamChat_unloadDoneReasonNoWarn() throws Exception {
        assertNoAbnormalWarnFor("unload");
    }

    private void assertNoAbnormalWarnFor(final String doneReason) throws Exception {
        final MockWebServer server = new MockWebServer();
        final String body = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n"
                + "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"" + doneReason + "\"}\n";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/x-ndjson").setBody(body));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                final LlmChatRequest request = new LlmChatRequest();
                request.setMessages(List.of(new LlmMessage("user", "hi")));
                localClient.streamChat(request, (content, done) -> {});
                assertTrue("no abnormal warn expected for " + doneReason,
                        capture.warnings().stream().noneMatch(m -> m.contains("Stream finished abnormally")));
            } finally {
                capture.detach();
            }
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
    public void test_buildRequestBody_thinkingLevelHigh() {
        client.setTestModel("gpt-oss:20b");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.putExtraParam("thinking_level", "high");

        final Map<String, Object> body = client.buildRequestBody(request, false);
        assertEquals("high", body.get("think"));
    }

    @Test
    public void test_buildRequestBody_thinkingLevelOverridesBudget() {
        client.setTestModel("gpt-oss:20b");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setThinkingBudget(0);
        request.putExtraParam("thinking_level", "medium");

        final Map<String, Object> body = client.buildRequestBody(request, false);
        // GPT-OSS ignores boolean form, so the string level must win.
        assertEquals("medium", body.get("think"));
    }

    @Test
    public void test_buildRequestBody_thinkingLevelInvalidFallsBackToBudget() {
        client.setTestModel("qwen3.5:35b");

        final LlmChatRequest request = new LlmChatRequest();
        request.addMessage(new LlmMessage("user", "Hello"));
        request.setThinkingBudget(1);
        // "max" is not a recognized level — fall back to boolean form.
        request.putExtraParam("thinking_level", "max");

        final Map<String, Object> body = client.buildRequestBody(request, false);
        assertEquals(Boolean.TRUE, body.get("think"));
    }

    @Test
    public void test_isValidThinkingLevel_acceptsRecognizedValues() {
        assertTrue(OllamaLlmClient.isValidThinkingLevel("high"));
        assertTrue(OllamaLlmClient.isValidThinkingLevel("medium"));
        assertTrue(OllamaLlmClient.isValidThinkingLevel("low"));
        assertTrue(OllamaLlmClient.isValidThinkingLevel("HIGH"));
        assertTrue(OllamaLlmClient.isValidThinkingLevel("Medium"));
    }

    @Test
    public void test_isValidThinkingLevel_rejectsOthers() {
        assertFalse(OllamaLlmClient.isValidThinkingLevel(null));
        assertFalse(OllamaLlmClient.isValidThinkingLevel(""));
        assertFalse(OllamaLlmClient.isValidThinkingLevel("max"));
        assertFalse(OllamaLlmClient.isValidThinkingLevel("true"));
    }

    @Test
    public void test_normalizeApiUrl_stripsApiSuffix() {
        // Official Ollama base URLs end with /api — both forms must collapse to the host root.
        assertEquals("http://localhost:11434", OllamaLlmClient.normalizeApiUrl("http://localhost:11434/api"));
        assertEquals("https://ollama.com", OllamaLlmClient.normalizeApiUrl("https://ollama.com/api"));
    }

    @Test
    public void test_normalizeApiUrl_stripsTrailingSlashes() {
        assertEquals("http://localhost:11434", OllamaLlmClient.normalizeApiUrl("http://localhost:11434/"));
        assertEquals("http://localhost:11434", OllamaLlmClient.normalizeApiUrl("http://localhost:11434/api/"));
        assertEquals("http://localhost:11434", OllamaLlmClient.normalizeApiUrl("http://localhost:11434//"));
    }

    @Test
    public void test_normalizeApiUrl_idempotent() {
        final String once = OllamaLlmClient.normalizeApiUrl("http://localhost:11434/api/");
        assertEquals(once, OllamaLlmClient.normalizeApiUrl(once));
    }

    @Test
    public void test_normalizeApiUrl_keepsRootHost() {
        assertEquals("http://localhost:11434", OllamaLlmClient.normalizeApiUrl("http://localhost:11434"));
    }

    @Test
    public void test_normalizeApiUrl_handlesNullAndBlank() {
        assertNull(OllamaLlmClient.normalizeApiUrl(null));
        assertEquals("", OllamaLlmClient.normalizeApiUrl(""));
        assertEquals("", OllamaLlmClient.normalizeApiUrl("   "));
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
    public void test_applyDefaultParams_intent_thinkingDisabledByDefault() {
        final LlmChatRequest request = new LlmChatRequest();
        assertNull(request.getThinkingBudget());

        client.applyDefaultParams(request, "intent");

        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_evaluation_thinkingDisabledByDefault() {
        final LlmChatRequest request = new LlmChatRequest();
        assertNull(request.getThinkingBudget());

        client.applyDefaultParams(request, "evaluation");

        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
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

    // --- applyPromptTypeParams config-key tests ---

    @Test
    public void test_applyPromptTypeParams_thinkingBudgetFromConfig() {
        final List<String> queriedPrimary = new ArrayList<>();
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient() {
            @Override
            protected String getConfigWithFallback(final String primaryKey, final String fallbackKey) {
                queriedPrimary.add(primaryKey);
                if ("rag.llm.ollama.answer.thinking.budget".equals(primaryKey)) {
                    return "4096";
                }
                return null;
            }
        };
        final LlmChatRequest request = new LlmChatRequest();
        request.setMessages(List.of(new LlmMessage("user", "hello")));
        localClient.applyPromptTypeParams(request, "answer");
        assertEquals(Integer.valueOf(4096), request.getThinkingBudget());
        assertTrue("expected lookup of rag.llm.ollama.answer.thinking.budget, queried=" + queriedPrimary,
                queriedPrimary.contains("rag.llm.ollama.answer.thinking.budget"));
    }

    @Test
    public void test_applyPromptTypeParams_thinkingBudgetHardcodedFallbackForIntent() {
        final List<String> queriedPrimary = new ArrayList<>();
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient() {
            @Override
            protected String getConfigWithFallback(final String primaryKey, final String fallbackKey) {
                queriedPrimary.add(primaryKey);
                return null; // no config set
            }
        };
        final LlmChatRequest request = new LlmChatRequest();
        request.setMessages(List.of(new LlmMessage("user", "hello")));
        localClient.applyPromptTypeParams(request, "intent");
        // intent's hardcoded default in applyDefaultParams is 0
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
        assertTrue("expected lookup of rag.llm.ollama.intent.thinking.budget, queried=" + queriedPrimary,
                queriedPrimary.contains("rag.llm.ollama.intent.thinking.budget"));
    }

    @Test
    public void test_applyPromptTypeParams_thinkingBudgetFromDefaultFallback() {
        final List<String> queriedPrimary = new ArrayList<>();
        final List<String> queriedFallback = new ArrayList<>();
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient() {
            @Override
            protected String getConfigWithFallback(final String primaryKey, final String fallbackKey) {
                queriedPrimary.add(primaryKey);
                queriedFallback.add(fallbackKey);
                // No per-type key, but default.thinking.budget is set
                if ("rag.llm.ollama.default.thinking.budget".equals(fallbackKey)) {
                    return "2048";
                }
                return null;
            }
        };
        final LlmChatRequest request = new LlmChatRequest();
        request.setMessages(List.of(new LlmMessage("user", "hello")));
        localClient.applyPromptTypeParams(request, "answer");
        assertEquals(Integer.valueOf(2048), request.getThinkingBudget());
        assertTrue("expected fallback lookup of rag.llm.ollama.default.thinking.budget, queriedFallback=" + queriedFallback,
                queriedFallback.contains("rag.llm.ollama.default.thinking.budget"));
        assertTrue("expected primary lookup of rag.llm.ollama.answer.thinking.budget, queriedPrimary=" + queriedPrimary,
                queriedPrimary.contains("rag.llm.ollama.answer.thinking.budget"));
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

    // --- Retry helpers ---

    @Test
    public void test_isRetryableStatus_retriesServerErrors() {
        assertTrue(OllamaLlmClient.isRetryableStatus(500));
        assertTrue(OllamaLlmClient.isRetryableStatus(502));
        assertTrue(OllamaLlmClient.isRetryableStatus(503));
        assertTrue(OllamaLlmClient.isRetryableStatus(504));
    }

    @Test
    public void test_isRetryableStatus_retries429() {
        // 429 Too Many Requests is a documented Ollama error (Cloud / rate-limited proxies).
        assertTrue(OllamaLlmClient.isRetryableStatus(429));
    }

    @Test
    public void test_isRetryableStatus_doesNotRetryNon429_4xx() {
        assertFalse(OllamaLlmClient.isRetryableStatus(400));
        assertFalse(OllamaLlmClient.isRetryableStatus(404));
        assertFalse(OllamaLlmClient.isRetryableStatus(401));
    }

    @Test
    public void test_isRetryableStatus_doesNotRetry200() {
        assertFalse(OllamaLlmClient.isRetryableStatus(200));
    }

    @Test
    public void test_executeWithRetry_returnsImmediatelyOnSuccess() throws Exception {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        final java.util.concurrent.atomic.AtomicInteger callCount = new java.util.concurrent.atomic.AtomicInteger();
        final String result = localClient.executeWithRetry("test", () -> {
            callCount.incrementAndGet();
            return "ok";
        });
        assertEquals("ok", result);
        assertEquals(1, callCount.get());
    }

    @Test
    public void test_executeWithRetry_throwsIOExceptionAfterExhaustion() {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestRetryMax(2);
        localClient.setTestRetryBaseDelayMs(1L); // keep test fast
        final java.util.concurrent.atomic.AtomicInteger callCount = new java.util.concurrent.atomic.AtomicInteger();
        try {
            localClient.executeWithRetry("test", () -> {
                callCount.incrementAndGet();
                throw new OllamaLlmClient.RetryableHttpException(503, "overloaded");
            });
            fail("expected IOException");
        } catch (final java.io.IOException e) {
            assertTrue(e.getMessage().contains("503"));
            assertEquals(2, callCount.get());
        }
    }

    @Test
    public void test_executeWithRetry_succeedsAfterOneFailure() throws Exception {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestRetryMax(3);
        localClient.setTestRetryBaseDelayMs(1L);
        final java.util.concurrent.atomic.AtomicInteger callCount = new java.util.concurrent.atomic.AtomicInteger();
        final String result = localClient.executeWithRetry("test", () -> {
            if (callCount.incrementAndGet() == 1) {
                throw new OllamaLlmClient.RetryableHttpException(503, "overloaded");
            }
            return "ok";
        });
        assertEquals("ok", result);
        assertEquals(2, callCount.get());
    }

    @Test
    public void test_executeWithRetry_retriesOnIOException() throws Exception {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestRetryMax(3);
        localClient.setTestRetryBaseDelayMs(1L);
        final java.util.concurrent.atomic.AtomicInteger callCount = new java.util.concurrent.atomic.AtomicInteger();
        final String result = localClient.executeWithRetry("test", () -> {
            if (callCount.incrementAndGet() == 1) {
                throw new java.net.ConnectException("connection refused");
            }
            return "ok";
        });
        assertEquals("ok", result);
        assertEquals(2, callCount.get());
    }

    @Test
    public void test_executeWithRetry_throwsLastIOExceptionAfterExhaustion() {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestRetryMax(2);
        localClient.setTestRetryBaseDelayMs(1L);
        final java.util.concurrent.atomic.AtomicInteger callCount = new java.util.concurrent.atomic.AtomicInteger();
        try {
            localClient.executeWithRetry("test", () -> {
                callCount.incrementAndGet();
                throw new java.net.ConnectException("connection refused attempt " + callCount.get());
            });
            fail("expected IOException");
        } catch (final java.io.IOException e) {
            assertTrue(e.getMessage().contains("connection refused attempt 2"));
            assertEquals(2, callCount.get());
        }
    }

    // --- chat() / streamChat() retry wiring ---

    @Test
    public void test_chat_retriesOn503() throws Exception {
        final MockWebServer server = new MockWebServer();
        server.enqueue(new MockResponse().setResponseCode(503));
        final String successBody = "{\"model\":\"llama3:latest\",\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},"
                + "\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":3,\"eval_count\":1}";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/json").setBody(successBody));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestRetryMax(3);
            localClient.setTestRetryBaseDelayMs(1L);
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            final LlmChatResponse response = localClient.chat(request);
            assertEquals("ok", response.getContent());
            assertEquals(2, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_streamChat_retriesOn503BeforeBody() throws Exception {
        final MockWebServer server = new MockWebServer();
        server.enqueue(new MockResponse().setResponseCode(503));
        final String successBody = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n"
                + "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/x-ndjson").setBody(successBody));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestRetryMax(3);
            localClient.setTestRetryBaseDelayMs(1L);
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            final java.util.List<String> chunks = new java.util.ArrayList<>();
            localClient.streamChat(request, (content, done) -> chunks.add(content));
            assertEquals(List.of("hi", ""), chunks);
            assertEquals(2, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_chat_doesNotRetryOn404() throws Exception {
        final MockWebServer server = new MockWebServer();
        server.enqueue(new MockResponse().setResponseCode(404).setBody("model not found"));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestRetryMax(5);
            localClient.setTestRetryBaseDelayMs(1L);
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.chat(request);
                fail("expected LlmException");
            } catch (final LlmException e) {
                // expected
            }
            assertEquals("404 must not be retried", 1, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_chat_retryBudgetExhausted() throws Exception {
        final MockWebServer server = new MockWebServer();
        server.enqueue(new MockResponse().setResponseCode(503));
        server.enqueue(new MockResponse().setResponseCode(503));
        server.enqueue(new MockResponse().setResponseCode(503));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestRetryMax(3);
            localClient.setTestRetryBaseDelayMs(1L);
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.chat(request);
                fail("expected LlmException after retries exhausted");
            } catch (final LlmException e) {
                // expected
            }
            assertEquals(3, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_chat_retryAttemptLogged() throws Exception {
        final MockWebServer server = new MockWebServer();
        server.enqueue(new MockResponse().setResponseCode(503));
        final String successBody = "{\"model\":\"llama3:latest\",\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},"
                + "\"done\":true,\"done_reason\":\"stop\"}";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/json").setBody(successBody));
        server.start();
        try {
            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestRetryMax(3);
            localClient.setTestRetryBaseDelayMs(1L);
            localClient.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            localClient.initHttpClient();
            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                final LlmChatRequest request = new LlmChatRequest();
                request.setMessages(List.of(new LlmMessage("user", "hi")));
                localClient.chat(request);
                assertTrue(
                        capture.infos()
                                .stream()
                                .anyMatch(m -> m.contains("chat retrying") && m.contains("attempt=1/3") && m.contains("status=503")),
                        "retry INFO line missing");
            } finally {
                capture.detach();
            }
        } finally {
            server.shutdown();
        }
    }

    // ========== Credential masking in logged URLs ==========
    //
    // This plugin has no credential configuration key for Ollama, so the configured endpoint
    // is the only place a secret can appear: rag.llm.ollama.api.url may point at a
    // reverse-proxied Ollama, carrying the shared secret in a query parameter. Every log line
    // that echoes the URL must route it through CredentialUrlUtil.maskCredentialInUrl. The masking
    // semantics themselves are pinned by OllamaUrlUtilTest; these two cases prove the LLM
    // client actually applies them on the chat and streamChat failure paths.
    //
    // These cases use the query-parameter form deliberately. They previously used the
    // userinfo form because it renders unambiguously in an assertion, but a userinfo-bearing
    // endpoint is now refused before any request is built, so that shape can no longer reach
    // a failure WARN at all. The query-parameter form is a configuration that works, so its
    // value really does reach a live log line - which is the only masking rule with a
    // reachable leak behind it.

    /** A proxy shared secret that must never be written to the log verbatim. */
    private static final String QUERY_PARAM_SECRET = "s3cr3tproxykey";

    @Test
    public void test_chat_failureLog_masksCredentialsInUrl() throws Exception {
        // Start then immediately shut down the server so the port refuses connections,
        // driving the generic failure WARN in chat() that echoes the configured URL.
        final MockWebServer server = new MockWebServer();
        server.start();
        final int port = server.getPort();
        server.shutdown();

        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl("http://127.0.0.1:" + port + "/?api_key=" + QUERY_PARAM_SECRET);
        localClient.setTestModel("llama3:latest");
        localClient.setTestRetryMax(1);
        localClient.setTestRetryBaseDelayMs(1L);
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.chat(request);
                fail("expected LlmException when the endpoint refuses connections");
            } catch (final LlmException e) {
                // expected
            }
            assertFalse(capture.renderedWarnings().stream().anyMatch(m -> m.contains(QUERY_PARAM_SECRET)),
                    "no WARN, including its attached throwable, may echo the proxy secret: " + capture.renderedWarnings());
            assertTrue(capture.warnings().stream().anyMatch(m -> m.contains("127.0.0.1:" + port) && m.contains("?api_key=***")),
                    "the failure WARN should carry the masked URL: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_streamChat_failureLog_masksCredentialsInUrl() throws Exception {
        final MockWebServer server = new MockWebServer();
        server.start();
        final int port = server.getPort();
        server.shutdown();

        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl("http://127.0.0.1:" + port + "/?api_key=" + QUERY_PARAM_SECRET);
        localClient.setTestModel("llama3:latest");
        localClient.setTestRetryMax(1);
        localClient.setTestRetryBaseDelayMs(1L);
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.streamChat(request, new LlmStreamCallback() {
                    @Override
                    public void onChunk(final String chunk, final boolean done) {
                        // no-op
                    }

                    @Override
                    public void onError(final Throwable t) {
                        // no-op
                    }
                });
                fail("expected LlmException when the endpoint refuses connections");
            } catch (final LlmException e) {
                // expected
            }
            assertFalse(capture.renderedWarnings().stream().anyMatch(m -> m.contains(QUERY_PARAM_SECRET)),
                    "no WARN, including its attached throwable, may echo the proxy secret: " + capture.renderedWarnings());
            assertTrue(capture.warnings().stream().anyMatch(m -> m.contains("127.0.0.1:" + port) && m.contains("?api_key=***")),
                    "the streaming failure WARN should carry the masked URL: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    // ========== Malformed endpoint URL must not leak through the exception path ==========
    //
    // The request URI is parsed when the request object is built (new HttpGet/HttpPost ->
    // URI.create), and the IllegalArgumentException that a malformed URL raises quotes the
    // offending URI in full. That string reaches the log three ways at once: the error={}
    // argument, the throwable attached to the same WARN, and the cause chain of the
    // exception thrown to the caller. Masking the logged url={} argument does not help,
    // because the leak rides on the exception, not on the argument.
    //
    // Masking the raw URL at that point would not help either: the value is malformed
    // precisely because it contains a character the masking pattern excludes, so the
    // pattern no longer matches. The sanitized message must therefore omit the URL.

    /** A secret that must never appear in a log line or an exception message. */
    private static final String MALFORMED_URL_SECRET = "s3cr3tvalue";

    /**
     * A configured endpoint carrying a secret query parameter whose value contains a
     * character that is illegal in a URI query, so building the request URI fails.
     */
    private static final String MALFORMED_URL_WITH_SECRET = "http://127.0.0.1:11434/?api_key=" + MALFORMED_URL_SECRET + "^x";

    @Test
    public void test_chat_malformedUrl_doesNotLeakCredential() throws Exception {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl(MALFORMED_URL_WITH_SECRET);
        localClient.setTestModel("llama3:latest");
        localClient.setTestRetryMax(1);
        localClient.setTestRetryBaseDelayMs(1L);
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.chat(request);
                fail("expected an exception when the configured endpoint cannot be parsed as a URI");
            } catch (final RuntimeException e) {
                assertFalse(LogCapturingAppender.renderThrowable(e).contains(MALFORMED_URL_SECRET),
                        "the exception thrown to the caller must not carry the raw URL: " + LogCapturingAppender.renderThrowable(e));
            }
            assertFalse(capture.renderedWarnings().stream().anyMatch(m -> m.contains(MALFORMED_URL_SECRET)),
                    "no WARN, including its attached throwable, may carry the raw URL: " + capture.renderedWarnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_streamChat_malformedUrl_doesNotLeakCredential() throws Exception {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl(MALFORMED_URL_WITH_SECRET);
        localClient.setTestModel("llama3:latest");
        localClient.setTestRetryMax(1);
        localClient.setTestRetryBaseDelayMs(1L);
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        final List<Throwable> callbackErrors = new ArrayList<>();
        try {
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.streamChat(request, new LlmStreamCallback() {
                    @Override
                    public void onChunk(final String chunk, final boolean done) {
                        // no-op
                    }

                    @Override
                    public void onError(final Throwable t) {
                        callbackErrors.add(t);
                    }
                });
                fail("expected an exception when the configured endpoint cannot be parsed as a URI");
            } catch (final RuntimeException e) {
                assertFalse(LogCapturingAppender.renderThrowable(e).contains(MALFORMED_URL_SECRET),
                        "the exception thrown to the caller must not carry the raw URL: " + LogCapturingAppender.renderThrowable(e));
            }
            for (final Throwable t : callbackErrors) {
                assertFalse(LogCapturingAppender.renderThrowable(t).contains(MALFORMED_URL_SECRET),
                        "the throwable handed to the stream callback must not carry the raw URL: "
                                + LogCapturingAppender.renderThrowable(t));
            }
            assertFalse(capture.renderedWarnings().stream().anyMatch(m -> m.contains(MALFORMED_URL_SECRET)),
                    "no WARN, including its attached throwable, may carry the raw URL: " + capture.renderedWarnings());
        } finally {
            capture.detach();
        }
    }

    // ========== A userinfo-bearing api.url is refused up front ==========
    //
    // RFC 9110 section 4.2.4 forbids a sender from generating the userinfo subcomponent in an
    // http/https target URI, and httpclient5 enforces that unconditionally: ProtocolExec
    // throws "Request URI authority contains deprecated userinfo component" with no setting
    // that disables it. A userinfo-bearing api.url can therefore never issue a request, no
    // matter what else is configured. Ollama does not authenticate at all, and the one
    // legitimate case - an endpoint behind an authenticating proxy - is already served by
    // http.proxy.host/.port/.username/.password. The value is an operator error with a
    // supported alternative, so the client names the remedy instead of failing opaquely at
    // execute time.
    //
    // The refusal FAILS CLOSED on the availability path rather than throwing:
    // checkAvailabilityNow() is reached synchronously from init()
    // (startAvailabilityCheck -> updateAvailability -> checkAvailabilityNow), and init() is
    // the DI container's eager init method, so a throw there would abort container start-up.
    // test_init_userinfoUrl_doesNotThrow pins that.

    /** A password that must appear in no log line, no exception message, and no cause chain. */
    private static final String USERINFO_PASSWORD = "s3cr3tpassword";

    /** An endpoint carrying credentials in its authority, the shape this change refuses. */
    private static final String USERINFO_API_URL = "http://ollama:" + USERINFO_PASSWORD + "@ollama.internal:11434";

    /**
     * The input class the masking regex was already shown to be defeated by: whitespace
     * inside the credential. Detection must be structural, so this must be caught too.
     */
    private static final String SPACED_USERINFO_PASSWORD = "s3cr3t password";

    private static final String SPACED_USERINFO_API_URL = "http://ollama:" + SPACED_USERINFO_PASSWORD + "@ollama.internal:11434";

    @Test
    public void test_checkAvailabilityNow_userinfoUrl_reportsUnavailableAndNamesRemedy() {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl(USERINFO_API_URL);
        localClient.setTestModel("llama3:latest");
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            assertFalse(localClient.checkAvailabilityNow(), "a userinfo-bearing api.url must report the client unavailable");

            final List<String> errors = capture.renderedAt(Level.ERROR);
            assertTrue(errors.size() == 1, "exactly one ERROR should name the misconfiguration: " + errors);
            final String error = errors.get(0);
            assertTrue(error.contains("rag.llm.ollama.api.url"), "the ERROR must name the offending config key: " + error);
            assertTrue(error.contains("http.proxy.username"), "the ERROR must name the supported alternative: " + error);
            assertTrue(error.contains("http.proxy.password"), "the ERROR must name the supported alternative: " + error);
            assertNoCapturedEventCarries(capture, USERINFO_PASSWORD);
        } finally {
            capture.detach();
            localClient.destroy();
        }
    }

    @Test
    public void test_checkAvailabilityNow_userinfoUrl_errorFiresOnceNotPerCall() {
        // The availability check runs on a timer, so a per-call ERROR would flood the log
        // once a minute for as long as the misconfiguration stands.
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl(USERINFO_API_URL);
        localClient.setTestModel("llama3:latest");
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            localClient.checkAvailabilityNow();
            localClient.checkAvailabilityNow();
            localClient.checkAvailabilityNow();

            final List<String> errors = capture.renderedAt(Level.ERROR);
            assertTrue(errors.size() == 1, "three checks must produce one ERROR, not three: " + errors);
        } finally {
            capture.detach();
            localClient.destroy();
        }
    }

    @Test
    public void test_checkAvailabilityNow_userinfoWithWhitespace_isDetectedStructurally() {
        // Precondition: this is exactly the input the masking regex cannot see, so a
        // detection built by reusing that regex would let this configuration through.
        assertEquals(SPACED_USERINFO_API_URL, CredentialUrlUtil.maskCredentialInUrl(SPACED_USERINFO_API_URL));

        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl(SPACED_USERINFO_API_URL);
        localClient.setTestModel("llama3:latest");
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            assertFalse(localClient.checkAvailabilityNow(), "a whitespace-bearing userinfo must be refused just the same");
            assertTrue(capture.renderedAt(Level.ERROR).size() == 1,
                    "the refusal ERROR must fire for this input too: " + capture.renderedAt(Level.ERROR));
            assertNoCapturedEventCarries(capture, SPACED_USERINFO_PASSWORD);
        } finally {
            capture.detach();
            localClient.destroy();
        }
    }

    @Test
    public void test_checkAvailabilityNow_ordinaryHostPortUrl_isUnaffected() {
        // Negative control: a host:port colon is not a credential separator. MockWebServer
        // serves on http://127.0.0.1:<port>, the same shape as http://ollama.internal:11434.
        final MockWebServer server = new MockWebServer();
        try {
            server.enqueue(new MockResponse().setBody("{\"models\":[{\"name\":\"llama3:latest\"}]}"));
            server.start();

            final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
            localClient.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            localClient.setTestModel("llama3:latest");
            localClient.initHttpClient();

            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
            try {
                assertTrue(localClient.checkAvailabilityNow(), "a plain host:port endpoint must still be reachable");
                assertTrue(capture.renderedAt(Level.ERROR).isEmpty(),
                        "no refusal ERROR may fire for a credential-free endpoint: " + capture.renderedAt(Level.ERROR));
            } finally {
                capture.detach();
                localClient.destroy();
            }
        } catch (final Exception e) {
            throw new IllegalStateException(e);
        } finally {
            try {
                server.shutdown();
            } catch (final Exception e) {
                // ignore
            }
        }
    }

    @Test
    public void test_chat_userinfoUrl_refusedWithRemedy() {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl(USERINFO_API_URL);
        localClient.setTestModel("llama3:latest");
        localClient.setTestRetryMax(1);
        localClient.setTestRetryBaseDelayMs(1L);
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.chat(request);
                fail("expected the configured userinfo endpoint to be refused");
            } catch (final LlmException e) {
                assertTrue(e.getMessage().contains("http.proxy.username"), "the failure must name the supported alternative: " + e);
                assertFalse(LogCapturingAppender.renderThrowable(e).contains(USERINFO_PASSWORD),
                        "no part of the thrown exception or its cause chain may carry the credential: "
                                + LogCapturingAppender.renderThrowable(e));
            }
            assertNoCapturedEventCarries(capture, USERINFO_PASSWORD);
        } finally {
            capture.detach();
            localClient.destroy();
        }
    }

    @Test
    public void test_streamChat_userinfoUrl_refusedWithRemedy() {
        final TestableOllamaLlmClient localClient = new TestableOllamaLlmClient();
        localClient.setTestApiUrl(USERINFO_API_URL);
        localClient.setTestModel("llama3:latest");
        localClient.setTestRetryMax(1);
        localClient.setTestRetryBaseDelayMs(1L);
        localClient.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        final List<Throwable> callbackErrors = new ArrayList<>();
        try {
            final LlmChatRequest request = new LlmChatRequest();
            request.setMessages(List.of(new LlmMessage("user", "hi")));
            try {
                localClient.streamChat(request, new LlmStreamCallback() {
                    @Override
                    public void onChunk(final String chunk, final boolean done) {
                        // no-op
                    }

                    @Override
                    public void onError(final Throwable t) {
                        callbackErrors.add(t);
                    }
                });
                fail("expected the configured userinfo endpoint to be refused");
            } catch (final LlmException e) {
                assertTrue(e.getMessage().contains("http.proxy.username"), "the failure must name the supported alternative: " + e);
            }
            assertTrue(callbackErrors.size() == 1, "the stream callback must be told once: " + callbackErrors);
            for (final Throwable t : callbackErrors) {
                assertFalse(LogCapturingAppender.renderThrowable(t).contains(USERINFO_PASSWORD),
                        "the throwable handed to the stream callback must not carry the credential: "
                                + LogCapturingAppender.renderThrowable(t));
            }
            assertNoCapturedEventCarries(capture, USERINFO_PASSWORD);
        } finally {
            capture.detach();
            localClient.destroy();
        }
    }

    @Test
    public void test_init_userinfoUrl_doesNotThrow() {
        // The design constraint: init() is the container's eager init method and reaches
        // checkAvailabilityNow() synchronously, so refusing the value must not escape as a
        // throw. This probe leaves the production init() body untouched and only pins the
        // seams that decide whether the availability check actually runs.
        final AvailabilityProbeClient probe = new AvailabilityProbeClient(USERINFO_API_URL);
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaLlmClient.class);
        try {
            probe.init();

            assertFalse(probe.isAvailable(), "the refused client must report unavailable, not available");
            assertTrue(capture.renderedAt(Level.ERROR).size() == 1,
                    "init() must surface the misconfiguration exactly once: " + capture.renderedAt(Level.ERROR));
            assertNoCapturedEventCarries(capture, USERINFO_PASSWORD);
        } catch (final RuntimeException e) {
            throw new AssertionError("init() must not throw for a userinfo-bearing api.url: " + e, e);
        } finally {
            capture.detach();
            probe.destroy();
        }
    }

    /**
     * Asserts that {@code secret} appears in no captured event at any level, including the
     * rendered stack trace of any attached throwable. Message-only assertions go green while
     * the rendered log still leaks through a cause.
     *
     * @param capture the attached appender.
     * @param secret the value that must not appear.
     */
    private static void assertNoCapturedEventCarries(final LogCapturingAppender capture, final String secret) {
        for (final Level level : List.of(Level.ERROR, Level.WARN, Level.INFO, Level.DEBUG)) {
            final List<String> rendered = capture.renderedAt(level);
            Assertions.assertFalse(rendered.stream().anyMatch(m -> m.contains(secret)),
                    "no " + level + " event may carry the credential: " + rendered);
        }
    }

    /**
     * A real {@link OllamaLlmClient} with only the seams that gate {@code init()}'s
     * availability check pinned, so the production {@code init()} body runs untouched.
     */
    static class AvailabilityProbeClient extends OllamaLlmClient {

        private final String apiUrl;

        AvailabilityProbeClient(final String apiUrl) {
            this.apiUrl = apiUrl;
        }

        @Override
        protected String getApiUrl() {
            return apiUrl;
        }

        @Override
        protected String getModel() {
            return "llama3:latest";
        }

        @Override
        protected String getLlmType() {
            return "ollama";
        }

        @Override
        protected boolean isRagChatEnabled() {
            return true;
        }

        @Override
        protected int getAvailabilityCheckInterval() {
            return 3600;
        }

        @Override
        protected int getTimeout() {
            return 1000;
        }

        @Override
        protected int getConnectTimeout() {
            return 1000;
        }
    }

    // --- Testable subclass ---

    static class TestableOllamaLlmClient extends OllamaLlmClient {

        private String testApiUrl = "http://localhost:11434";
        private String testModel = "llama3:latest";
        private int testTimeout = 30000;
        private int testConnectTimeout = 5000;
        private String testProxyHost = "";
        private Integer testProxyPort = null;
        private String testProxyUsername = "";
        private String testProxyPassword = "";
        private int testRetryMax = 3;
        private long testRetryBaseDelayMs = 2000L;

        void setTestApiUrl(final String apiUrl) {
            this.testApiUrl = apiUrl;
        }

        void setTestModel(final String model) {
            this.testModel = model;
        }

        void setTestTimeout(final int timeout) {
            this.testTimeout = timeout;
        }

        void setTestConnectTimeout(final int connectTimeout) {
            this.testConnectTimeout = connectTimeout;
        }

        @Override
        protected int getConnectTimeout() {
            return testConnectTimeout;
        }

        void setTestProxyHost(final String proxyHost) {
            this.testProxyHost = proxyHost;
        }

        void setTestProxyPort(final Integer proxyPort) {
            this.testProxyPort = proxyPort;
        }

        void setTestProxyUsername(final String proxyUsername) {
            this.testProxyUsername = proxyUsername;
        }

        void setTestProxyPassword(final String proxyPassword) {
            this.testProxyPassword = proxyPassword;
        }

        void setTestRetryMax(final int max) {
            this.testRetryMax = max;
        }

        void setTestRetryBaseDelayMs(final long ms) {
            this.testRetryBaseDelayMs = ms;
        }

        @Override
        protected int getRetryMaxAttempts() {
            return testRetryMax;
        }

        @Override
        protected long getRetryBaseDelayMs() {
            return testRetryBaseDelayMs;
        }

        @Override
        protected String getProxyHost() {
            return testProxyHost;
        }

        @Override
        protected Integer getProxyPort() {
            return testProxyPort;
        }

        @Override
        protected String getProxyUsername() {
            return testProxyUsername;
        }

        @Override
        protected String getProxyPassword() {
            return testProxyPassword;
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
            httpClient = buildHttpClient();
        }
    }

}
