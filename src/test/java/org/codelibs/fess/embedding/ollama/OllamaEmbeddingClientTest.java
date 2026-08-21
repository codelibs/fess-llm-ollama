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

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;

import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Property;
import org.codelibs.core.timer.TimeoutTask;
import org.codelibs.fess.embedding.EmbeddingException;
import org.codelibs.fess.mylasta.direction.FessConfig;
import org.codelibs.fess.ollama.OllamaUrlUtil;
import org.codelibs.fess.unit.LogCapturingAppender;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.codelibs.fess.util.CredentialUrlUtil;
import org.codelibs.fess.util.ComponentUtil;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

public class OllamaEmbeddingClientTest extends UnitFessTestCase {

    /** The real config key read by the production (non-overridden) {@link OllamaEmbeddingClient#getDimension()}. */
    private static final String DIMENSION_CONFIG_KEY = "content_chunker.embedding.dimension";

    private TestableOllamaEmbeddingClient client;

    // The "systemProperties" component (org.codelibs.fess.unit.TestSystemProperties,
    // registered in test_app.xml) is a JVM-lifetime singleton under UTFlute's default
    // container-reuse behavior: without isUseOneTimeContainer(), a value set here would
    // leak into every other test class that shares the same container instance across
    // the whole suite, corrupting them order-dependently while the suite stays green.
    // This class sets/removes real content_chunker.embedding.ollama.* keys on that
    // component (see the *_real_* tests below), so it forces a fresh container per test.
    @Override
    protected boolean isUseOneTimeContainer() {
        return true;
    }

    @Override
    public void setUp(final TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableOllamaEmbeddingClient();
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
    public void test_embedDocuments_success() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3],[0.4,0.5,0.6]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            final List<float[]> result = client.embedDocuments(List.of("chunk one", "chunk two"));

            assertEquals(2, result.size());
            assertEquals(3, result.get(0).length);
            assertEquals(0.1f, result.get(0)[0]);
            assertEquals(0.2f, result.get(0)[1]);
            assertEquals(0.6f, result.get(1)[2]);

            final RecordedRequest recordedRequest = server.takeRequest();
            assertEquals("/api/embed", recordedRequest.getPath());
            assertEquals("POST", recordedRequest.getMethod());
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("\"model\":\"nomic-embed-text\""), "request body should carry the model name: " + body);
            assertTrue(body.contains("chunk one") && body.contains("chunk two"), "request body should carry both inputs: " + body);
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_appliesDefaultDocumentPrefix() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            client.embedDocuments(List.of("chunk one"));

            final RecordedRequest recordedRequest = server.takeRequest();
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("title: none | text: chunk one"), "request body should carry the document-prefixed input: " + body);
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedQuery_appliesDefaultQueryPrefix() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            final List<float[]> result = client.embedQuery(List.of("what is fess"));

            assertEquals(1, result.size());
            assertEquals(3, result.get(0).length);

            final RecordedRequest recordedRequest = server.takeRequest();
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("task: search result | query: what is fess"),
                    "request body should carry the query-prefixed input: " + body);
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_blankPrefixAddsNoPrefix() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.setTestDocumentPrefix("");
            client.initHttpClient();

            client.embedDocuments(List.of("chunk one"));

            final RecordedRequest recordedRequest = server.takeRequest();
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("\"input\":[\"chunk one\"]"), "request body should carry the unprefixed input: " + body);
            assertFalse(body.contains("search_document:"), "request body should not carry the document prefix: " + body);
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedQuery_blankPrefixAddsNoPrefix() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.setTestQueryPrefix("");
            client.initHttpClient();

            client.embedQuery(List.of("what is fess"));

            final RecordedRequest recordedRequest = server.takeRequest();
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("\"input\":[\"what is fess\"]"), "request body should carry the unprefixed input: " + body);
            assertFalse(body.contains("search_query:"), "request body should not carry the query prefix: " + body);
        } finally {
            server.shutdown();
        }
    }

    // ========== Explicit truncate flag (FINDING F4.4) ==========
    //
    // Ollama's documented default for /api/embed is truncate=true: an input longer than the
    // model's context window is silently cut down to fit, and the server still returns a
    // well-formed vector of the right dimension, so every check in parseEmbedResponse passes
    // and the relevance loss leaves no trace. The flag is therefore always sent explicitly,
    // sourced from content_chunker.embedding.ollama.truncate (default true, preserving the
    // current behavior) so operators can opt into hard failures instead.

    /** The real config key read by the production {@link OllamaEmbeddingClient#isTruncateEnabled()}. */
    private static final String TRUNCATE_CONFIG_KEY = "content_chunker.embedding.ollama.truncate";

    @Test
    public void test_embedDocuments_sendsTruncateTrueByDefault() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            client.embedDocuments(List.of("chunk one"));

            final RecordedRequest recordedRequest = server.takeRequest();
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("\"truncate\":true"), "request body should send truncate explicitly as true by default: " + body);
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedQuery_sendsTruncateTrueByDefault() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            client.embedQuery(List.of("what is fess"));

            final RecordedRequest recordedRequest = server.takeRequest();
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("\"truncate\":true"), "query requests should send truncate explicitly as true by default: " + body);
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_sendsTruncateFalseWhenConfigured() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            // isTruncateEnabled() is the sole config read left on the call path (every other
            // seam is overridden by TestableOllamaEmbeddingClient), and it reads through the
            // "systemProperties" channel (content_chunker.embedding.ollama.* is unified onto
            // FessProp#getSystemProperty), not fess_config.properties/getOrDefault.
            ComponentUtil.getSystemProperties().setProperty(TRUNCATE_CONFIG_KEY, "false");

            client.embedDocuments(List.of("chunk one"));

            final RecordedRequest recordedRequest = server.takeRequest();
            final String body = recordedRequest.getBody().readUtf8();
            assertTrue(body.contains("\"truncate\":false"), "configuring truncate=false must be sent through to Ollama: " + body);
        } finally {
            ComponentUtil.getSystemProperties().remove(TRUNCATE_CONFIG_KEY);
            server.shutdown();
        }
    }

    @Test
    public void test_isTruncateEnabled_real_defaultsToTrue() {
        final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
        assertTrue(realClient.isTruncateEnabled(), "an unset truncate key must preserve Ollama's documented default of true");
    }

    @Test
    public void test_isTruncateEnabled_real_parsesFalse() {
        // isTruncateEnabled() reads content_chunker.embedding.ollama.truncate through the
        // inherited getConfigString(), which resolves via FessProp#getSystemProperty (the
        // "systemProperties" component), not getOrDefault/fess_config.properties.
        ComponentUtil.getSystemProperties().setProperty(TRUNCATE_CONFIG_KEY, " FALSE ");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertFalse(realClient.isTruncateEnabled(), "'FALSE' (padded, upper case) must be honored as false");
        } finally {
            ComponentUtil.getSystemProperties().remove(TRUNCATE_CONFIG_KEY);
        }
    }

    @Test
    public void test_isTruncateEnabled_real_invalidValue_warnsAndDefaultsTrue() {
        // Boolean.parseBoolean("ture") would silently return false and flip a true-default
        // switch into hard per-document failures, so an unparseable value must warn and keep
        // the documented default instead.
        ComponentUtil.getSystemProperties().setProperty(TRUNCATE_CONFIG_KEY, "ture");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertTrue(realClient.isTruncateEnabled(), "an unparseable truncate value must fall back to true");
            assertTrue(capture.warnings().stream().anyMatch(m -> m.contains("truncate") && m.contains("ture")),
                    "an invalid truncate value must emit a WARN naming the key and value: " + capture.warnings());
        } finally {
            capture.detach();
            ComponentUtil.getSystemProperties().remove(TRUNCATE_CONFIG_KEY);
        }
    }

    @Test
    public void test_embedDocuments_dimensionMismatch_throws() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // Server returns 3-dim vectors but the configured dimension is 4.
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(4);
            client.initHttpClient();

            try {
                client.embedDocuments(List.of("chunk one"));
                fail("expected EmbeddingException on dimension mismatch");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("dimension"), "message should mention dimension: " + e.getMessage());
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_nonNumericVectorComponent_throws() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // The vector's second element is a JSON null instead of a number. A naive
            // Jackson asDouble() call would silently coerce this to 0.0 and corrupt the
            // stored vector instead of surfacing a clear error.
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,null,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            try {
                client.embedDocuments(List.of("chunk one"));
                fail("expected EmbeddingException on non-numeric vector component");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("not numeric"), "message should mention non-numeric component: " + e.getMessage());
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_nonFiniteVectorComponent_throws() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // 1e999 overflows the double range and parses as a NUMBER node whose value is
            // Infinity: isNumber() returns true, so the non-numeric guard alone lets it
            // through and casting to float would store a non-finite (Infinity) component,
            // silently corrupting the vector. It must be rejected with a clear error.
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,1e999,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            try {
                client.embedDocuments(List.of("chunk one"));
                fail("expected EmbeddingException on non-finite vector component");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("not finite"), "message should mention non-finite component: " + e.getMessage());
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_doesNotRetryOn404() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            server.enqueue(new MockResponse().setResponseCode(404).setBody("model not found"));
            server.start();

            client.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.setTestRetryMax(5);
            client.setTestRetryBaseDelayMs(1L);
            client.initHttpClient();

            try {
                client.embedDocuments(List.of("chunk"));
                fail("expected EmbeddingException");
            } catch (final EmbeddingException e) {
                // expected
            }
            assertEquals("404 must not be retried", 1, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_retriesOn503() throws Exception {
        final MockWebServer server = new MockWebServer();
        server.enqueue(new MockResponse().setResponseCode(503));
        final String successBody = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/json").setBody(successBody));
        server.start();
        try {
            client.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.setTestRetryMax(3);
            client.setTestRetryBaseDelayMs(1L);
            client.initHttpClient();

            final List<float[]> result = client.embedDocuments(List.of("chunk"));

            assertEquals(1, result.size());
            assertEquals(2, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_getDimension_throwsWhenUnconfigured() {
        client.setTestDimension(null);
        try {
            client.getDimension();
            fail("expected EmbeddingException when dimension is unconfigured");
        } catch (final EmbeddingException e) {
            // expected
        }
    }

    // ========== Real (non-overridden) getDimension() coverage ==========
    //
    // The tests above exercise TestableOllamaEmbeddingClient's own hand-written
    // getDimension() override, never the production method. These tests use a
    // plain `new OllamaEmbeddingClient()` (no subclass) to drive the real
    // ComponentUtil.getFessConfig().getSystemProperty("content_chunker.embedding.dimension", ...)
    // config-read seam directly, via the "systemProperties" test component
    // registered in test_app.xml (org.codelibs.fess.unit.TestSystemProperties).
    // That component instance is not guaranteed to be recreated per test method
    // (verified empirically), so each test explicitly sets/removes the key it
    // needs and restores it in a finally block to stay order-independent.

    @Test
    public void test_getDimension_real_throwsWhenUnconfigured() {
        ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException when dimension is unconfigured");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("not configured"), "message should mention not configured: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_real_throwsOnNonNumericValue() {
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "not-a-number");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException on non-numeric dimension value");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("Invalid"), "message should mention the invalid value: " + e.getMessage());
                assertTrue(e.getCause() instanceof NumberFormatException, "cause should be NumberFormatException: " + e.getCause());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_real_returnsConfiguredValue() {
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "1536");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals(1536, realClient.getDimension());
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_real_throwsOnZero() {
        // A parseable but non-positive value must be rejected up front with a clear
        // EmbeddingException, not returned as 0 (which later surfaces as a misleading
        // "dimension mismatch" or a NegativeArraySizeException in parseEmbedResponse).
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "0");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException on zero dimension value");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("positive"), "message should mention it must be positive: " + e.getMessage());
                assertTrue(e.getMessage().contains("0"), "message should echo the offending value: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_real_throwsOnNegative() {
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "-5");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException on negative dimension value");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("positive"), "message should mention it must be positive: " + e.getMessage());
                assertTrue(e.getMessage().contains("-5"), "message should echo the offending value: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    // ========== Real (non-overridden) getRetryBaseDelayMs() coverage (FINDING F2) ==========
    //
    // The TestableOllamaEmbeddingClient overrides getRetryBaseDelayMs(), so these
    // tests drive the production method directly via a plain OllamaEmbeddingClient
    // whose config read is redirected through the real "systemProperties" component
    // (content_chunker.embedding.ollama.* reads via getConfigString()/getSystemProperty,
    // not getOrDefault/fess_config.properties). A typo'd (non-numeric) value must fall
    // back to the 2000ms default AND emit a WARN so the misconfiguration is visible to
    // operators (matching OllamaLlmClient.getRetryBaseDelayMs()).

    /** The real config key read by the production {@link OllamaEmbeddingClient#getRetryBaseDelayMs()}. */
    private static final String RETRY_BASE_DELAY_CONFIG_KEY = "content_chunker.embedding.ollama.retry.base.delay.ms";

    @Test
    public void test_getRetryBaseDelayMs_real_invalidValue_warnsAndReturnsDefault() {
        // getRetryBaseDelayMs() reads content_chunker.embedding.ollama.retry.base.delay.ms
        // through the inherited getConfigString(), which resolves via
        // FessProp#getSystemProperty (the "systemProperties" component), not
        // getOrDefault/fess_config.properties.
        ComponentUtil.getSystemProperties().setProperty(RETRY_BASE_DELAY_CONFIG_KEY, "not-a-number");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals(2000L, realClient.getRetryBaseDelayMs());
            assertTrue(capture.warnings().stream().anyMatch(m -> m.contains("retry.base.delay.ms") && m.contains("not-a-number")),
                    "an invalid retry.base.delay.ms must emit a WARN naming the key and value: " + capture.warnings());
        } finally {
            capture.detach();
            ComponentUtil.getSystemProperties().remove(RETRY_BASE_DELAY_CONFIG_KEY);
        }
    }

    @Test
    public void test_getRetryBaseDelayMs_real_validValue_isParsed() {
        ComponentUtil.getSystemProperties().setProperty(RETRY_BASE_DELAY_CONFIG_KEY, "500");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals(500L, realClient.getRetryBaseDelayMs());
        } finally {
            ComponentUtil.getSystemProperties().remove(RETRY_BASE_DELAY_CONFIG_KEY);
        }
    }

    // ========== Real (non-overridden) config-channel coverage ==========
    //
    // content_chunker.embedding.ollama.* is unified onto FessProp#getSystemProperty (the
    // "systemProperties" component, backed by conf/system.properties or a
    // -Dfess.system.<key> JVM argument), never onto FessConfig#getOrDefault
    // (fess_config.properties, loaded once at container boot). TestableOllamaEmbeddingClient
    // overrides every one of these getters for the tests above, so none of them exercise the
    // real config-read seam. Six of the seven tests below drive a plain (non-overridden)
    // OllamaEmbeddingClient against that seam and would catch a regression back onto
    // getOrDefault; the last one covers a different contract, never touches the seam, and
    // would not catch such a regression - see its own comment for what it does guard.
    //
    // The three *_readsFromSystemProperty tests (api.url, model, document.prefix) set a value
    // on the systemProperties component and are directly falsifiable against a channel
    // regression: if the corresponding getter were changed back to read via getOrDefault, that
    // value would never be observed there and the assertion would fail against the hardcoded
    // default instead.
    //
    // The two *_defaultsTo* tests (api.url, model) additionally stub FessConfig#getOrDefault to
    // answer a value distinct from both the documented default and an unset system property,
    // which makes them equally falsifiable against a channel regression: reverting the getter
    // to getOrDefault would surface that stubbed fess_config.properties value instead of the
    // documented default, reddening the assertion.
    //
    // test_getDocumentPrefix_real_emptyStringDisablesPrefix is also channel-falsifiable (see
    // its own comment) but pins a different contract on top: that an explicitly empty
    // document.prefix is honored rather than treated as absent and replaced by the default.

    /** The real config key read by the production {@link OllamaEmbeddingClient#getApiUrl()}. */
    private static final String API_URL_CONFIG_KEY = "content_chunker.embedding.ollama.api.url";

    /** The real config key read by the production {@link OllamaEmbeddingClient#getModel()}. */
    private static final String MODEL_CONFIG_KEY = "content_chunker.embedding.ollama.model";

    /** The real config key read by the production {@link OllamaEmbeddingClient#getDocumentPrefix()}. */
    private static final String DOCUMENT_PREFIX_CONFIG_KEY = "content_chunker.embedding.ollama.document.prefix";

    @Test
    public void test_getApiUrl_real_readsFromSystemProperty() {
        ComponentUtil.getSystemProperties().setProperty(API_URL_CONFIG_KEY, "http://ollama-config-test.example:12345");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals("a configured api.url must be read from the systemProperties channel, not fess_config.properties",
                    "http://ollama-config-test.example:12345", realClient.getApiUrl());
        } finally {
            ComponentUtil.getSystemProperties().remove(API_URL_CONFIG_KEY);
        }
    }

    @Test
    public void test_getApiUrl_real_defaultsToLocalhost() {
        // Stub fess_config.properties to answer a value distinct from both the documented
        // default and an unset system property, so this test is falsifiable against the
        // channel: if getApiUrl() were ever changed back to read via getOrDefault, it would
        // surface this stubbed value instead of the documented default and the assertion
        // below would fail.
        final FessConfig original = ComponentUtil.getFessConfig();
        ComponentUtil.setFessConfig(new FessConfig.SimpleImpl() {
            private static final long serialVersionUID = 1L;

            @Override
            public String getOrDefault(final String key, final String defaultValue) {
                if (API_URL_CONFIG_KEY.equals(key)) {
                    return "http://fess-config-channel-trap.example:1";
                }
                return defaultValue;
            }
        });
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals("an unset api.url must fall back to the documented default, not a fess_config.properties value",
                    "http://localhost:11434", realClient.getApiUrl());
        } finally {
            ComponentUtil.setFessConfig(original);
        }
    }

    @Test
    public void test_getModel_real_readsFromSystemProperty() {
        ComponentUtil.getSystemProperties().setProperty(MODEL_CONFIG_KEY, "mxbai-embed-large");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals("a configured model must be read from the systemProperties channel, not fess_config.properties",
                    "mxbai-embed-large", realClient.getModel());
        } finally {
            ComponentUtil.getSystemProperties().remove(MODEL_CONFIG_KEY);
        }
    }

    @Test
    public void test_getModel_real_defaultsToEmbeddingGemma() {
        // Stub fess_config.properties to answer a value distinct from both the documented
        // default and an unset system property, so this test is falsifiable against the
        // channel: if getModel() were ever changed back to read via getOrDefault, it would
        // surface this stubbed value instead of the documented default and the assertion
        // below would fail.
        final FessConfig original = ComponentUtil.getFessConfig();
        ComponentUtil.setFessConfig(new FessConfig.SimpleImpl() {
            private static final long serialVersionUID = 1L;

            @Override
            public String getOrDefault(final String key, final String defaultValue) {
                if (MODEL_CONFIG_KEY.equals(key)) {
                    return "fess-config-channel-trap";
                }
                return defaultValue;
            }
        });
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals("an unset model must fall back to the documented default, not a fess_config.properties value", "embeddinggemma",
                    realClient.getModel());
        } finally {
            ComponentUtil.setFessConfig(original);
        }
    }

    @Test
    public void test_getDocumentPrefix_real_readsFromSystemProperty() {
        // document.prefix is routed through the inherited getConfigString(), the same
        // helper api.url/model now use, so this is a positive control for the shared seam
        // rather than a per-getter duplicate of the two tests above.
        ComponentUtil.getSystemProperties().setProperty(DOCUMENT_PREFIX_CONFIG_KEY, "custom_document: ");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals("a configured document.prefix must be read from the systemProperties channel, not fess_config.properties",
                    "custom_document: ", realClient.getDocumentPrefix());
        } finally {
            ComponentUtil.getSystemProperties().remove(DOCUMENT_PREFIX_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDocumentPrefix_real_emptyStringDisablesPrefix() {
        // README.md documents "set to an empty string to disable" for document.prefix; this
        // pins that contract through the real systemProperties config seam (as opposed to
        // TestableOllamaEmbeddingClient.setTestDocumentPrefix(""), which only overrides the
        // getter and never touches config - see test_embedDocuments_blankPrefixAddsNoPrefix).
        ComponentUtil.getSystemProperties().setProperty(DOCUMENT_PREFIX_CONFIG_KEY, "");
        try {
            final OllamaEmbeddingClient realClient = new OllamaEmbeddingClient();
            assertEquals("an explicitly empty document.prefix must disable prefixing, not fall back to the default", "",
                    realClient.getDocumentPrefix());
        } finally {
            ComponentUtil.getSystemProperties().remove(DOCUMENT_PREFIX_CONFIG_KEY);
        }
    }

    // ========== checkAvailabilityNow() / isModelAvailable() coverage (FINDING 1) ==========

    @Test
    public void test_checkAvailabilityNow_modelAvailable_exactMatch() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String tagsJson = "{\"models\":[{\"name\":\"llama3\"},{\"name\":\"nomic-embed-text\"}]}";
            server.enqueue(new MockResponse().setBody(tagsJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.initHttpClient();

            assertTrue(client.checkAvailabilityNow(), "configured model present in /api/tags should be available");

            final RecordedRequest recordedRequest = server.takeRequest();
            assertEquals("/api/tags", recordedRequest.getPath());
            assertEquals("GET", recordedRequest.getMethod());
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_checkAvailabilityNow_modelAvailable_latestTagStripped() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // Ollama reports the model with an explicit ":latest" tag while the configured
            // name carries none; normalizeModelName() strips ":latest" so they still match.
            final String tagsJson = "{\"models\":[{\"name\":\"nomic-embed-text:latest\"}]}";
            server.enqueue(new MockResponse().setBody(tagsJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.initHttpClient();

            assertTrue(client.checkAvailabilityNow(), "':latest'-tagged model should match the untagged configured name");
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_checkAvailabilityNow_modelNotFound_returnsFalse() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            final String tagsJson = "{\"models\":[{\"name\":\"some-other-model\"}]}";
            server.enqueue(new MockResponse().setBody(tagsJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.initHttpClient();

            assertFalse(client.checkAvailabilityNow(), "configured model absent from /api/tags should be unavailable");
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_checkAvailabilityNow_non2xxResponse_returnsFalse() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            server.enqueue(new MockResponse().setResponseCode(500).setBody("server error"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.initHttpClient();

            assertFalse(client.checkAvailabilityNow(), "a non-2xx /api/tags response should be treated as unavailable");
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_checkAvailabilityNow_connectionFailure_returnsFalse() throws Exception {
        // Start then immediately shut down the server so the port refuses connections,
        // driving checkAvailabilityNow()'s catch block deterministically.
        final MockWebServer server = new MockWebServer();
        server.start();
        final String url = server.url("").toString().replaceAll("/$", "");
        server.shutdown();

        client.setTestApiUrl(url);
        client.setTestModel("nomic-embed-text");
        client.initHttpClient();

        assertFalse(client.checkAvailabilityNow(), "a connection failure during the check should return false, not throw");
    }

    // ========== parseEmbedResponse() error paths (FINDINGS 2 and 3) ==========

    @Test
    public void test_embedDocuments_malformedJsonResponse_throws() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // A 2xx response whose body is not valid JSON must surface as a clear
            // EmbeddingException from parseEmbedResponse's readTree failure.
            server.enqueue(new MockResponse().setBody("not json{").setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            try {
                client.embedDocuments(List.of("chunk one"));
                fail("expected EmbeddingException on malformed JSON response");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("Failed to parse Ollama embed response"),
                        "message should indicate a parse failure: " + e.getMessage());
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_countMismatch_throws() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // Two inputs but only one embedding vector returned: a count mismatch,
            // distinct from a per-vector dimension mismatch.
            final String responseJson = "{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}";
            server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            try {
                client.embedDocuments(List.of("chunk one", "chunk two"));
                fail("expected EmbeddingException on embedding count mismatch");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("count mismatch"),
                        "message should mention count mismatch, not dimension: " + e.getMessage());
            }
        } finally {
            server.shutdown();
        }
    }

    // ========== Retry exhaustion (FINDING 4) ==========

    @Test
    public void test_embedDocuments_retryExhaustion_throwsEmbeddingException() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // Every attempt returns a retryable 503; once the retry budget is exhausted
            // the failure must surface as EmbeddingException (wrapping the IOException),
            // never a raw IOException leaking out of executeWithRetry.
            server.enqueue(new MockResponse().setResponseCode(503));
            server.enqueue(new MockResponse().setResponseCode(503));
            server.enqueue(new MockResponse().setResponseCode(503));
            server.start();

            client.setTestApiUrl(server.url("/").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.setTestRetryMax(3);
            client.setTestRetryBaseDelayMs(1L);
            client.initHttpClient();

            try {
                client.embedDocuments(List.of("chunk"));
                fail("expected EmbeddingException after retry exhaustion");
            } catch (final EmbeddingException e) {
                assertTrue(e.getCause() instanceof java.io.IOException,
                        "cause should be the IOException from exhausted retries: " + e.getCause());
            }
            assertEquals("all retry attempts should have been made", 3, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    // ========== Credential masking in logged URLs (FINDING F4.12) ==========
    //
    // The configured Ollama base URL is echoed into WARN/DEBUG log lines on every failure
    // path. This plugin has no credential configuration key for Ollama, so that URL is the
    // only place a secret can appear: a reverse proxy in front of Ollama carries a shared
    // secret as a query parameter (?api_key=...), which would otherwise be written verbatim
    // into fess.log.
    //
    // The masking itself now lives in the shared OllamaUrlUtil, which the sibling
    // OllamaLlmClient uses too; its semantics are pinned once by OllamaUrlUtilTest rather
    // than restated per client. The behavioural case below proves this client applies it.
    //
    // The case uses the query-parameter form deliberately. It previously used the userinfo
    // form because it renders unambiguously in an assertion, but a userinfo-bearing endpoint
    // is now refused before any request is built, so that shape can no longer reach a failure
    // WARN at all. The query-parameter form is a configuration that works, so its value
    // really does reach a live log line.

    /** A proxy shared secret that must never be written to the log verbatim. */
    private static final String QUERY_PARAM_SECRET = "s3cr3tproxykey";

    @Test
    public void test_embedDocuments_failureLog_masksCredentialsInUrl() throws Exception {
        // Start then immediately shut down the server so the port refuses connections, driving
        // the generic failure WARN in callEmbedApi that echoes the configured URL.
        final MockWebServer server = new MockWebServer();
        server.start();
        final int port = server.getPort();
        server.shutdown();

        client.setTestApiUrl("http://127.0.0.1:" + port + "/?api_key=" + QUERY_PARAM_SECRET);
        client.setTestModel("nomic-embed-text");
        client.setTestDimension(3);
        client.setTestRetryMax(1);
        client.setTestRetryBaseDelayMs(1L);
        client.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            try {
                client.embedDocuments(List.of("chunk"));
                fail("expected EmbeddingException when the endpoint refuses connections");
            } catch (final EmbeddingException e) {
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

    // ========== Malformed endpoint URL must not leak through the exception path ==========
    //
    // The request URI is parsed when the request object is built (new HttpGet/HttpPost ->
    // URI.create), and the IllegalArgumentException that a malformed URL raises quotes the
    // offending URI in full. That string reaches the log three ways at once: the error={}
    // argument, the throwable attached to the same WARN, and the cause chain of the
    // EmbeddingException thrown to the caller. Masking the logged url={} argument does not
    // help, because the leak rides on the exception rather than on that argument.

    /** A secret that must never appear in a log line or an exception message. */
    private static final String MALFORMED_URL_SECRET = "s3cr3tvalue";

    /**
     * A configured endpoint carrying a secret query parameter whose value contains a
     * character that is illegal in a URI query, so building the request URI fails.
     */
    private static final String MALFORMED_URL_WITH_SECRET = "http://127.0.0.1:11434/?api_key=" + MALFORMED_URL_SECRET + "^x";

    @Test
    public void test_embedDocuments_malformedUrl_doesNotLeakCredential() throws Exception {
        client.setTestApiUrl(MALFORMED_URL_WITH_SECRET);
        client.setTestModel("nomic-embed-text");
        client.setTestDimension(3);
        client.setTestRetryMax(1);
        client.setTestRetryBaseDelayMs(1L);
        client.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            try {
                client.embedDocuments(List.of("chunk"));
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

    // ========== A userinfo-bearing api.url is refused up front ==========
    //
    // RFC 9110 section 4.2.4 forbids a sender from generating the userinfo subcomponent in an
    // http/https target URI, and httpclient5 enforces that unconditionally, so a
    // userinfo-bearing api.url can never issue a request. Ollama does not authenticate, and
    // an endpoint behind an authenticating proxy is already served by
    // http.proxy.host/.port/.username/.password. Mirrors the sibling
    // OllamaLlmClientTest section; see it for the fail-closed rationale.

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
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestModel("nomic-embed-text");
        client.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            assertFalse(client.checkAvailabilityNow(), "a userinfo-bearing api.url must report the client unavailable");

            final List<String> errors = capture.renderedAt(Level.ERROR);
            assertTrue(errors.size() == 1, "exactly one ERROR should name the misconfiguration: " + errors);
            final String error = errors.get(0);
            assertTrue(error.contains("content_chunker.embedding.ollama.api.url"),
                    "the ERROR must name the offending config key: " + error);
            assertTrue(error.contains("http.proxy.username"), "the ERROR must name the supported alternative: " + error);
            assertTrue(error.contains("http.proxy.password"), "the ERROR must name the supported alternative: " + error);
            assertNoCapturedEventCarries(capture, USERINFO_PASSWORD);
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_checkAvailabilityNow_userinfoUrl_errorFiresOnceNotPerCall() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestModel("nomic-embed-text");
        client.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.checkAvailabilityNow();
            client.checkAvailabilityNow();
            client.checkAvailabilityNow();

            assertTrue(capture.renderedAt(Level.ERROR).size() == 1,
                    "three checks must produce one ERROR, not three: " + capture.renderedAt(Level.ERROR));
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_checkAvailabilityNow_userinfoWithWhitespace_isDetectedStructurally() {
        // Precondition: this is exactly the input the masking regex cannot see, so a
        // detection built by reusing that regex would let this configuration through.
        assertEquals(SPACED_USERINFO_API_URL, CredentialUrlUtil.maskCredentialInUrl(SPACED_USERINFO_API_URL));

        client.setTestApiUrl(SPACED_USERINFO_API_URL);
        client.setTestModel("nomic-embed-text");
        client.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            assertFalse(client.checkAvailabilityNow(), "a whitespace-bearing userinfo must be refused just the same");
            assertTrue(capture.renderedAt(Level.ERROR).size() == 1,
                    "the refusal ERROR must fire for this input too: " + capture.renderedAt(Level.ERROR));
            assertNoCapturedEventCarries(capture, SPACED_USERINFO_PASSWORD);
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_checkAvailabilityNow_ordinaryHostPortUrl_isUnaffected() throws Exception {
        // Negative control: a host:port colon is not a credential separator. MockWebServer
        // serves on http://127.0.0.1:<port>, the same shape as http://ollama.internal:11434.
        final MockWebServer server = new MockWebServer();
        try {
            server.enqueue(new MockResponse().setBody("{\"models\":[{\"name\":\"nomic-embed-text\"}]}"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.initHttpClient();

            final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
            try {
                assertTrue(client.checkAvailabilityNow(), "a plain host:port endpoint must still be reachable");
                assertTrue(capture.renderedAt(Level.ERROR).isEmpty(),
                        "no refusal ERROR may fire for a credential-free endpoint: " + capture.renderedAt(Level.ERROR));
            } finally {
                capture.detach();
            }
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_userinfoUrl_refusedWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestModel("nomic-embed-text");
        client.setTestDimension(3);
        client.setTestRetryMax(1);
        client.setTestRetryBaseDelayMs(1L);
        client.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            try {
                client.embedDocuments(List.of("chunk"));
                fail("expected the configured userinfo endpoint to be refused");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("http.proxy.username"), "the failure must name the supported alternative: " + e);
                assertFalse(LogCapturingAppender.renderThrowable(e).contains(USERINFO_PASSWORD),
                        "no part of the thrown exception or its cause chain may carry the credential: "
                                + LogCapturingAppender.renderThrowable(e));
            }
            assertNoCapturedEventCarries(capture, USERINFO_PASSWORD);
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_embedQuery_userinfoUrl_refusedWithRemedy() {
        client.setTestApiUrl(USERINFO_API_URL);
        client.setTestModel("nomic-embed-text");
        client.setTestDimension(3);
        client.setTestRetryMax(1);
        client.setTestRetryBaseDelayMs(1L);
        client.initHttpClient();

        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            try {
                client.embedQuery(List.of("q"));
                fail("expected the configured userinfo endpoint to be refused");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("http.proxy.password"), "the failure must name the supported alternative: " + e);
                assertFalse(LogCapturingAppender.renderThrowable(e).contains(USERINFO_PASSWORD),
                        "no part of the thrown exception or its cause chain may carry the credential: "
                                + LogCapturingAppender.renderThrowable(e));
            }
            assertNoCapturedEventCarries(capture, USERINFO_PASSWORD);
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_userinfoUrl_doesNotThrow() {
        // The design constraint: init() is the container's eager init method and reaches
        // checkAvailabilityNow() synchronously, so refusing the value must not escape as a
        // throw. This probe leaves the production init() body untouched.
        final AvailabilityProbeClient probe = new AvailabilityProbeClient(USERINFO_API_URL);
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
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
     * @param capture the attached appender
     * @param secret the value that must not appear
     */
    private static void assertNoCapturedEventCarries(final LogCapturingAppender capture, final String secret) {
        for (final Level level : List.of(Level.ERROR, Level.WARN, Level.INFO, Level.DEBUG)) {
            final List<String> rendered = capture.renderedAt(level);
            Assertions.assertFalse(rendered.stream().anyMatch(m -> m.contains(secret)),
                    "no " + level + " event may carry the credential: " + rendered);
        }
    }

    /**
     * A real {@link OllamaEmbeddingClient} with only the seams that gate {@code init()}'s
     * availability check pinned, so the production {@code init()} body runs untouched.
     */
    static class AvailabilityProbeClient extends OllamaEmbeddingClient {

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
            return "nomic-embed-text";
        }

        @Override
        protected String getEmbeddingType() {
            return "ollama";
        }

        @Override
        protected boolean isContentChunkerEnabled() {
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

    // ========== Prefix/model mismatch WARN (MED-5) ==========
    //
    // The task prefixes are plain configurable strings, and a pair that belongs to a
    // different model family than the configured model still yields valid embeddings of the
    // correct dimension - nothing fails, only relevance degrades. init() therefore emits a
    // one-time WARN when the two disagree. The cases below cover both recognized families in
    // each direction plus an unrecognized model, because the mistake the defaults now invite
    // is the opposite of the one they invited before: switching model to nomic-embed-text and
    // leaving the embeddinggemma defaults in place. Blanking both prefixes (the documented
    // escape hatch) always suppresses the warning. The WARN is a diagnostic only.

    @Test
    public void test_init_gemmaModelWithGemmaPrefixes_doesNotWarn() {
        client.setTestModel("embeddinggemma");
        // Keep the shipped defaults, which are the embeddinggemma convention.
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertFalse(capture.warnings().stream().anyMatch(m -> m.contains("degraded relevance")),
                    "the shipped default model and prefixes must not warn about each other: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_gemmaModelTaggedLatest_doesNotWarn() {
        // The tag form must be recognized too; conf/system.properties commonly carries it.
        client.setTestModel("embeddinggemma:latest");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertFalse(capture.warnings().stream().anyMatch(m -> m.contains("degraded relevance")),
                    "a tagged embeddinggemma model must not warn with the default prefixes: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_nomicModelWithGemmaPrefixes_warns() {
        // The regression the new defaults invite: model switched, prefixes left alone.
        client.setTestModel("nomic-embed-text");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertTrue(
                    capture.warnings()
                            .stream()
                            .anyMatch(m -> m.contains("nomic-embed-text") && m.contains("search_document: ")
                                    && m.contains("degraded relevance")),
                    "a nomic model left on the embeddinggemma defaults must warn and name the nomic prefixes: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_nomicModelWithNomicPrefixes_doesNotWarn() {
        client.setTestModel("nomic-embed-text");
        client.setTestDocumentPrefix("search_document: ");
        client.setTestQueryPrefix("search_query: ");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertFalse(capture.warnings().stream().anyMatch(m -> m.contains("degraded relevance")),
                    "a nomic model with the nomic convention must not warn: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_gemmaModelWithNomicPrefixes_warns() {
        client.setTestModel("embeddinggemma");
        client.setTestDocumentPrefix("search_document: ");
        client.setTestQueryPrefix("search_query: ");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertTrue(
                    capture.warnings()
                            .stream()
                            .anyMatch(m -> m.contains("embeddinggemma") && m.contains("task: search result | query: ")
                                    && m.contains("degraded relevance")),
                    "an embeddinggemma model with nomic prefixes must warn and name the gemma prefixes: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_gemmaModelWithInterpolatedTitle_doesNotWarn() {
        // The embeddinggemma convention interpolates the document's own title, so a document
        // prefix matching the "title: ... | text: " shape must be accepted, not just the
        // "title: none" default.
        client.setTestModel("embeddinggemma");
        client.setTestDocumentPrefix("title: manual | text: ");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertFalse(capture.warnings().stream().anyMatch(m -> m.contains("degraded relevance")),
                    "a title-interpolated gemma document prefix must not warn: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_unknownModelWithPrefix_warns() {
        client.setTestModel("mxbai-embed-large");
        // Keep the shipped (non-blank) prefixes on a model of neither known family.
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertTrue(capture.warnings()
                    .stream()
                    .anyMatch(m -> m.contains("mxbai-embed-large") && m.contains("empty value") && m.contains("degraded relevance")),
                    "an unrecognized model with prefixes must warn and point at blanking them: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    @Test
    public void test_init_unknownModelBlankPrefixes_doesNotWarn() {
        client.setTestModel("mxbai-embed-large");
        client.setTestDocumentPrefix("");
        client.setTestQueryPrefix("");
        final LogCapturingAppender capture = LogCapturingAppender.attach(OllamaEmbeddingClient.class);
        try {
            client.init();
            assertFalse(capture.warnings().stream().anyMatch(m -> m.contains("degraded relevance")),
                    "blanking both prefixes must suppress the prefix-mismatch WARN: " + capture.warnings());
        } finally {
            capture.detach();
        }
    }

    // ========== init() provider-selection guard (FINDING F4.1) ==========
    //
    // Every embedding plugin on the classpath is instantiated and init()'d by the DI
    // container, so each one must stay completely inert unless it is the provider named by
    // content_chunker.embedding.name. Without the guard an unselected Ollama client would
    // still build an HTTP client and schedule a recurring availability check that polls a
    // host the operator never asked it to contact.
    //
    // The probe below overrides only the two seams that decide whether init() proceeds; the
    // production init() body runs untouched, and both effects it would have (the httpClient
    // field and the scheduled availabilityCheckTask) are read straight off the inherited
    // protected fields, so no production accessor is needed.

    @Test
    public void test_init_skipsWhenAnotherProviderIsSelected() {
        // Content chunking is enabled here on purpose: it is the only condition under which
        // startAvailabilityCheck() would actually schedule a task, so asserting the task is
        // null is a real assertion rather than a vacuous one.
        final ProviderSelectionProbeClient probe = new ProviderSelectionProbeClient("opensearch", true);
        try {
            probe.init();

            assertNull(probe.httpClientRef(), "a client whose provider is not selected must not build an HTTP client");
            assertNull(probe.availabilityCheckTaskRef(), "a client whose provider is not selected must not schedule availability checks");
        } finally {
            probe.destroy();
        }
    }

    @Test
    public void test_init_proceedsWhenThisProviderIsSelected() {
        // Positive control for the test above: the same probe, differing only in the selected
        // provider name, does initialise - so a null httpClient there is caused by the guard
        // and not by the probe being inert for some unrelated reason.
        final ProviderSelectionProbeClient probe = new ProviderSelectionProbeClient("ollama", false);
        try {
            probe.init();

            assertNotNull(probe.httpClientRef(), "the selected provider's client must build an HTTP client");
        } finally {
            probe.destroy();
        }
    }

    /**
     * A real {@link OllamaEmbeddingClient} with only the two provider-selection seams pinned,
     * exposing the inherited protected state that {@code init()} would populate.
     */
    static class ProviderSelectionProbeClient extends OllamaEmbeddingClient {

        private final String embeddingType;
        private final boolean contentChunkerEnabled;

        ProviderSelectionProbeClient(final String embeddingType, final boolean contentChunkerEnabled) {
            this.embeddingType = embeddingType;
            this.contentChunkerEnabled = contentChunkerEnabled;
        }

        @Override
        protected String getEmbeddingType() {
            return embeddingType;
        }

        @Override
        protected boolean isContentChunkerEnabled() {
            return contentChunkerEnabled;
        }

        CloseableHttpClient httpClientRef() {
            return httpClient;
        }

        TimeoutTask availabilityCheckTaskRef() {
            return availabilityCheckTask;
        }
    }

    // ========== Sub-batch item cap (FIX 4) ==========
    //
    // OllamaEmbeddingClient.MAX_BATCH_ITEMS is a private constant (128). These
    // tests mirror that value locally: an input larger than the cap must be split
    // into contiguous sub-batches, each sent as its own /api/embed request, with
    // the per-sub-batch vectors concatenated back in input order.
    private static final int MAX_BATCH_ITEMS = 128;

    @Test
    public void test_embedDocuments_splitsIntoSubBatchesPreservingOrder() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            // 130 inputs with a cap of 128 -> two sub-batches sized 128 and 2.
            // Each sub-batch response tags its vectors with a distinct first
            // component (1.0 vs 2.0) so the concatenation order is observable.
            final int total = MAX_BATCH_ITEMS + 2;
            server.enqueue(
                    new MockResponse().setBody(embeddingsResponse(MAX_BATCH_ITEMS, 3, 1.0f)).setHeader("Content-Type", "application/json"));
            server.enqueue(new MockResponse().setBody(embeddingsResponse(2, 3, 2.0f)).setHeader("Content-Type", "application/json"));
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            final List<String> inputs = new ArrayList<>(total);
            for (int i = 0; i < total; i++) {
                inputs.add("text-" + i);
            }

            final List<float[]> result = client.embedDocuments(inputs);

            // Count and order preserved across the concatenation.
            assertEquals(total, result.size());
            assertEquals(1.0f, result.get(0)[0]);
            assertEquals(1.0f, result.get(MAX_BATCH_ITEMS - 1)[0]);
            assertEquals(2.0f, result.get(MAX_BATCH_ITEMS)[0]);
            assertEquals(2.0f, result.get(total - 1)[0]);

            // Exactly one request per sub-batch, carrying the expected slice sizes.
            assertEquals("input exceeding the cap must be split into two sub-batch requests", 2, server.getRequestCount());
            final String firstBody = server.takeRequest().getBody().readUtf8();
            final String secondBody = server.takeRequest().getBody().readUtf8();
            assertEquals("first sub-batch should carry MAX_BATCH_ITEMS inputs", MAX_BATCH_ITEMS,
                    countOccurrences(firstBody, "title: none | text: "));
            assertEquals("second sub-batch should carry the remaining inputs", 2, countOccurrences(secondBody, "title: none | text: "));
        } finally {
            server.shutdown();
        }
    }

    @Test
    public void test_embedDocuments_emptyInput_returnsEmptyWithNoApiCall() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            server.start();

            client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
            client.setTestModel("nomic-embed-text");
            client.setTestDimension(3);
            client.initHttpClient();

            final List<float[]> result = client.embedDocuments(List.of());

            assertTrue(result.isEmpty(), "empty input should return an empty list");
            assertEquals("empty input must not hit the API", 0, server.getRequestCount());
        } finally {
            server.shutdown();
        }
    }

    /** Builds an {@code /api/embed} response of {@code count} vectors of length {@code dim}, each vector's first component set to {@code firstComponent}. */
    private static String embeddingsResponse(final int count, final int dim, final float firstComponent) {
        final StringBuilder sb = new StringBuilder("{\"model\":\"nomic-embed-text\",\"embeddings\":[");
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append('[');
            for (int j = 0; j < dim; j++) {
                if (j > 0) {
                    sb.append(',');
                }
                sb.append(j == 0 ? firstComponent : 0.5f);
            }
            sb.append(']');
        }
        return sb.append("]}").toString();
    }

    /** Counts non-overlapping occurrences of {@code needle} in {@code haystack}. */
    private static int countOccurrences(final String haystack, final String needle) {
        int count = 0;
        int idx = 0;
        while ((idx = haystack.indexOf(needle, idx)) >= 0) {
            count++;
            idx += needle.length();
        }
        return count;
    }

    static class TestableOllamaEmbeddingClient extends OllamaEmbeddingClient {

        private String testApiUrl = "http://localhost:11434";
        private String testModel = "nomic-embed-text";
        private int testTimeout = 30000;
        private int testConnectTimeout = 5000;
        private int testRetryMax = 3;
        private long testRetryBaseDelayMs = 2000L;
        private Integer testDimension = 768;
        private String testDocumentPrefix = DEFAULT_DOCUMENT_PREFIX;
        private String testQueryPrefix = DEFAULT_QUERY_PREFIX;

        void setTestApiUrl(final String apiUrl) {
            this.testApiUrl = apiUrl;
        }

        void setTestModel(final String model) {
            this.testModel = model;
        }

        void setTestDimension(final Integer dimension) {
            this.testDimension = dimension;
        }

        void setTestRetryMax(final int max) {
            this.testRetryMax = max;
        }

        void setTestRetryBaseDelayMs(final long ms) {
            this.testRetryBaseDelayMs = ms;
        }

        void setTestDocumentPrefix(final String prefix) {
            this.testDocumentPrefix = prefix;
        }

        void setTestQueryPrefix(final String prefix) {
            this.testQueryPrefix = prefix;
        }

        @Override
        protected String getApiUrl() {
            return testApiUrl;
        }

        // The base-class default for content_chunker.embedding.name is "opensearch"
        // (AbstractEmbeddingClient.EMBEDDING_NAME_DEFAULT), which would make init()
        // skip this client. Tests exercising init() select ollama explicitly, matching
        // an operator setting content_chunker.embedding.name=ollama.
        @Override
        protected String getEmbeddingType() {
            return "ollama";
        }

        @Override
        protected String getModel() {
            return testModel;
        }

        @Override
        public int getDimension() {
            if (testDimension == null) {
                throw new EmbeddingException("content_chunker.embedding.dimension is not configured");
            }
            return testDimension;
        }

        @Override
        protected int getTimeout() {
            return testTimeout;
        }

        @Override
        protected int getConnectTimeout() {
            return testConnectTimeout;
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
        protected String getDocumentPrefix() {
            return testDocumentPrefix;
        }

        @Override
        protected String getQueryPrefix() {
            return testQueryPrefix;
        }

        void initHttpClient() {
            httpClient = buildHttpClient();
        }
    }

}
