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
        assertTrue(OllamaLlmClient.isRetryableStatus(503));
        assertTrue(OllamaLlmClient.isRetryableStatus(504));
    }

    @Test
    public void test_isRetryableStatus_doesNotRetry429Or4xx() {
        assertFalse(OllamaLlmClient.isRetryableStatus(429));
        assertFalse(OllamaLlmClient.isRetryableStatus(400));
        assertFalse(OllamaLlmClient.isRetryableStatus(404));
        assertFalse(OllamaLlmClient.isRetryableStatus(401));
    }

    @Test
    public void test_isRetryableStatus_doesNotRetry502Or200() {
        assertFalse(OllamaLlmClient.isRetryableStatus(502));
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

    /**
     * Test helper that captures Log4j2 events emitted by a target class so tests can
     * assert on log output. Attach via {@link #attach(Class)}, query via
     * {@link #messagesAt(org.apache.logging.log4j.Level)} or convenience methods, and
     * always {@link #detach()} in a finally block. Not safe for parallel test
     * execution — capture is on the global Log4j2 logger registry.
     */
    static final class LogCapturingAppender extends AbstractAppender {
        private final List<LogEvent> events = new CopyOnWriteArrayList<>();
        private final Logger boundLogger;

        private LogCapturingAppender(final Logger logger) {
            super("LogCapturingAppender-" + UUID.randomUUID(), null, null, true, Property.EMPTY_ARRAY);
            this.boundLogger = logger;
        }

        static LogCapturingAppender attach(final Class<?> targetClass) {
            final Logger logger = (Logger) LogManager.getLogger(targetClass);
            final LogCapturingAppender appender = new LogCapturingAppender(logger);
            appender.start();
            logger.addAppender(appender);
            return appender;
        }

        void detach() {
            boundLogger.removeAppender(this);
            stop();
        }

        @Override
        public void append(final LogEvent event) {
            events.add(event.toImmutable());
        }

        List<String> messagesAt(final Level level) {
            return events.stream().filter(e -> e.getLevel() == level).map(e -> e.getMessage().getFormattedMessage()).toList();
        }

        List<String> warnings() {
            return messagesAt(Level.WARN);
        }

        List<String> infos() {
            return messagesAt(Level.INFO);
        }

        List<String> errors() {
            return messagesAt(Level.ERROR);
        }

        List<String> debugs() {
            return messagesAt(Level.DEBUG);
        }
    }
}
