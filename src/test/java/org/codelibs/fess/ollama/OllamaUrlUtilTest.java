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
package org.codelibs.fess.ollama;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.PrintWriter;
import java.io.StringWriter;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.codelibs.fess.util.CredentialUrlUtil;
import org.junit.jupiter.api.Test;

/**
 * Tests the credential masking and the guarded request factories shared by the Ollama LLM
 * and embedding clients.
 *
 * <p>Both {@code OllamaLlmClient} and {@code OllamaEmbeddingClient} delegate to
 * {@link OllamaUrlUtil}, so this one set of cases fixes the semantics for both clients at
 * once.
 */
public class OllamaUrlUtilTest {

    @Test
    public void test_userinfoRejectionMessage_namesTheKeyAndTheSupportedAlternative() {
        final String message = OllamaUrlUtil.userinfoRejectionMessage("rag.llm.ollama.api.url");
        assertAll(() -> assertTrue(message.contains("rag.llm.ollama.api.url"), message),
                () -> assertTrue(message.contains("http.proxy.host"), message),
                () -> assertTrue(message.contains("http.proxy.username"), message),
                () -> assertTrue(message.contains("http.proxy.password"), message),
                // Naming the standard keeps the refusal diagnosable: it is not a local policy
                // choice but a rule the HTTP client enforces unconditionally.
                () -> assertTrue(message.contains("RFC 9110"), message));
        // The message is logged and carried in an exception, so it is built from the key and
        // the remedy alone: the configured value is not an input and cannot appear in it.
        assertEquals(message.replace("rag.llm.ollama.api.url", "content_chunker.embedding.ollama.api.url"),
                OllamaUrlUtil.userinfoRejectionMessage("content_chunker.embedding.ollama.api.url"),
                "the key is the only part of the message that varies");
    }

    // ========== Guarded request factories ==========
    //
    // A malformed endpoint fails while the request object is built, and the raw
    // IllegalArgumentException from URI.create quotes the offending URI in full. The
    // factories replace it with a message that names no part of the input.

    /** A secret that must survive nowhere in the replacement exception. */
    private static final String SECRET = "s3cr3tvalue";

    /** The configuration key the factories name in place of the offending URL. */
    private static final String CONFIG_KEY = "rag.llm.ollama.api.url";

    /** Malformed because {@code ^} is illegal in a URI query. */
    private static final String MALFORMED_URL = "http://127.0.0.1:11434/api/chat?api_key=" + SECRET + "^x";

    @Test
    public void test_createHttpPost_malformedUrlFailureNamesNoPartOfTheUrl() {
        final IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OllamaUrlUtil.createHttpPost(MALFORMED_URL, CONFIG_KEY));
        assertAll(() -> assertFalse(render(e).contains(SECRET), "the rendered failure must not carry the secret: " + render(e)),
                () -> assertFalse(render(e).contains("127.0.0.1"), "the rendered failure must not carry the URL at all: " + render(e)),
                () -> assertNull(e.getCause(), "no cause may be attached: URISyntaxException renders the URI verbatim"),
                () -> assertTrue(e.getMessage().startsWith("Invalid URL configured in " + CONFIG_KEY),
                        "unexpected message: " + e.getMessage()),
                // The URI-syntax reason and index are constants of the parser, not slices of
                // the input, so keeping them stays diagnosable without leaking.
                () -> assertTrue(e.getMessage().contains("Illegal character in query"), "unexpected message: " + e.getMessage()),
                () -> assertTrue(e.getMessage().contains("at index "), "unexpected message: " + e.getMessage()));
    }

    @Test
    public void test_createHttpGet_malformedUrlFailureNamesNoPartOfTheUrl() {
        final IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OllamaUrlUtil.createHttpGet(MALFORMED_URL, CONFIG_KEY));
        assertAll(() -> assertFalse(render(e).contains(SECRET), "the rendered failure must not carry the secret: " + render(e)),
                () -> assertFalse(render(e).contains("127.0.0.1"), "the rendered failure must not carry the URL at all: " + render(e)),
                () -> assertNull(e.getCause(), "no cause may be attached: URISyntaxException renders the URI verbatim"));
    }

    @Test
    public void test_createHttp_wellFormedUrlIsUnaffected() throws Exception {
        assertEquals("http://localhost:11434/api/chat",
                OllamaUrlUtil.createHttpPost("http://localhost:11434/api/chat", CONFIG_KEY).getUri().toString());
        assertEquals("http://localhost:11434/api/tags",
                OllamaUrlUtil.createHttpGet("http://localhost:11434/api/tags", CONFIG_KEY).getUri().toString());
    }

    @Test
    public void test_createHttpPost_maskingAloneWouldNotHaveSufficed() {
        // The character that makes the URL unparseable also defeats the masking patterns, so
        // the sanitized message has to omit the URL rather than mask it. This pins the reason
        // sanitizeUriFailure does not simply call maskCredentialInUrl.
        final String malformedUserinfo = "http://user:pa ss@127.0.0.1:11434/api/chat";
        assertEquals(malformedUserinfo, CredentialUrlUtil.maskCredentialInUrl(malformedUserinfo),
                "a whitespace-bearing userinfo does not match the masking pattern");
        final IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> OllamaUrlUtil.createHttpPost(malformedUserinfo, CONFIG_KEY));
        assertFalse(render(e).contains("pa ss"), "the rendered failure must not carry the secret: " + render(e));
    }

    // ========== normalizeBaseUrl / appendPath ==========
    //
    // The endpoint used to be assembled by plain concatenation (apiUrl + "/api/tags"), which
    // is silently wrong whenever the configured endpoint carries a query string: the result
    // is a well-formed URL whose request target is the wrong path and whose credential value
    // has the API path glued onto it. Nothing rejects it, so the only way to catch the
    // regression is to pin the assembled string.

    @Test
    public void test_normalizeBaseUrl_stripsTrailingSlashAndApiSegment() {
        assertAll(() -> assertEquals("http://localhost:11434", OllamaUrlUtil.normalizeBaseUrl("http://localhost:11434")),
                () -> assertEquals("http://localhost:11434", OllamaUrlUtil.normalizeBaseUrl("http://localhost:11434/")),
                () -> assertEquals("http://localhost:11434", OllamaUrlUtil.normalizeBaseUrl("http://localhost:11434/api")),
                () -> assertEquals("http://localhost:11434", OllamaUrlUtil.normalizeBaseUrl("http://localhost:11434/api/")),
                () -> assertEquals("http://localhost:11434", OllamaUrlUtil.normalizeBaseUrl("  http://localhost:11434/api/  ")),
                () -> assertEquals("https://ollama.com", OllamaUrlUtil.normalizeBaseUrl("https://ollama.com/api")),
                () -> assertEquals("http://gw/ollama", OllamaUrlUtil.normalizeBaseUrl("http://gw/ollama/api")));
    }

    @Test
    public void test_normalizeBaseUrl_isIdempotent() {
        final String once = OllamaUrlUtil.normalizeBaseUrl("http://localhost:11434/api/");
        assertEquals(once, OllamaUrlUtil.normalizeBaseUrl(once));
    }

    @Test
    public void test_normalizeBaseUrl_appliesToThePathNotTheWholeString() {
        // Without the query-aware split these two would be returned unchanged, because the
        // literal string does not end in "/" or "/api".
        assertAll(
                () -> assertEquals("http://gw/ollama?api_key=s3cr3t",
                        OllamaUrlUtil.normalizeBaseUrl("http://gw/ollama/api?api_key=s3cr3t")),
                () -> assertEquals("http://gw?api_key=s3cr3t", OllamaUrlUtil.normalizeBaseUrl("http://gw/?api_key=s3cr3t")),
                () -> assertEquals("http://gw#frag", OllamaUrlUtil.normalizeBaseUrl("http://gw/api#frag")));
    }

    @Test
    public void test_normalizeBaseUrl_passesThroughNullAndBlank() {
        assertAll(() -> assertNull(OllamaUrlUtil.normalizeBaseUrl(null)), () -> assertEquals("", OllamaUrlUtil.normalizeBaseUrl("")),
                () -> assertEquals("", OllamaUrlUtil.normalizeBaseUrl("   ")));
    }

    @Test
    public void test_appendPath_withoutQueryIsPlainConcatenation() {
        assertAll(() -> assertEquals("http://localhost:11434/api/tags", OllamaUrlUtil.appendPath("http://localhost:11434", "/api/tags")),
                () -> assertEquals("http://gw/ollama/api/embed", OllamaUrlUtil.appendPath("http://gw/ollama", "/api/embed")));
    }

    @Test
    public void test_appendPath_keepsQueryBehindTheAppendedPath() {
        // The regression this guards: "http://gw?api_key=s3cr3t" + "/api/tags" produced
        // "http://gw?api_key=s3cr3t/api/tags", i.e. a request for "/" carrying the secret
        // value "s3cr3t/api/tags".
        assertAll(
                () -> assertEquals("http://gw/api/tags?api_key=s3cr3t", OllamaUrlUtil.appendPath("http://gw?api_key=s3cr3t", "/api/tags")),
                () -> assertEquals("http://gw/ollama/api/embed?api_key=s3cr3t&v=2",
                        OllamaUrlUtil.appendPath("http://gw/ollama?api_key=s3cr3t&v=2", "/api/embed")));
    }

    @Test
    public void test_appendPath_keepsFragmentLast() {
        assertAll(() -> assertEquals("http://gw/api/tags#frag", OllamaUrlUtil.appendPath("http://gw#frag", "/api/tags")),
                () -> assertEquals("http://gw/api/tags?k=1#frag", OllamaUrlUtil.appendPath("http://gw?k=1#frag", "/api/tags")));
    }

    @Test
    public void test_appendPath_resultStillMasks() {
        // The assembled URL is what reaches every url={} log argument, so masking has to hold
        // after the path is spliced in front of the query.
        final String assembled = OllamaUrlUtil.appendPath(OllamaUrlUtil.normalizeBaseUrl("http://gw/?api_key=s3cr3t"), "/api/embed");
        assertEquals("http://gw/api/embed?api_key=s3cr3t", assembled);
        assertEquals("http://gw/api/embed?api_key=***", CredentialUrlUtil.maskCredentialInUrl(assembled));
        assertFalse(CredentialUrlUtil.maskCredentialInUrl(assembled).contains("s3cr3t"));
    }

    @Test
    public void test_appendPath_assembledUrlIsAcceptedByTheRequestFactories() throws Exception {
        // A query-bearing endpoint has to survive URI parsing too, otherwise the fix would
        // only move the failure.
        final String assembled = OllamaUrlUtil.appendPath("http://gw?api_key=s3cr3t", "/api/tags");
        final HttpGet request = OllamaUrlUtil.createHttpGet(assembled, CONFIG_KEY);
        assertEquals("/api/tags", request.getUri().getPath(), "the API path must be the request path, not part of the query");
        assertEquals("api_key=s3cr3t", request.getUri().getQuery(), "the credential must survive as the query, intact");
        assertNotNull(OllamaUrlUtil.createHttpPost(assembled, CONFIG_KEY));
    }

    @Test
    public void test_appendPath_nullBaseReturnsThePath() {
        assertEquals("/api/tags", OllamaUrlUtil.appendPath(null, "/api/tags"));
    }

    /**
     * Renders {@code t} the way a log appender would, including the stack traces of the whole
     * cause chain. Asserting only on {@link Throwable#getMessage()} would pass while the
     * rendered output still leaks through a cause.
     *
     * @param t the throwable to render
     * @return the full rendered stack trace
     */
    private static String render(final Throwable t) {
        assertNotNull(t);
        final StringWriter writer = new StringWriter();
        t.printStackTrace(new PrintWriter(writer));
        return writer.toString();
    }
}
