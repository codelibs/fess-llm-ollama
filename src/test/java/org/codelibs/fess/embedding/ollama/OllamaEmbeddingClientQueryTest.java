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

import java.util.List;

import org.codelibs.fess.embedding.ollama.OllamaEmbeddingClientTest.TestableOllamaEmbeddingClient;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

/**
 * Query normalisation: {@code embedQuery} strips Fess/Lucene query syntax before embedding,
 * {@code embedDocuments} never does, and the stripping happens <em>before</em> the query prefix
 * is applied.
 *
 * <p>The two entry points carry different kinds of text. A document chunk is prose that
 * legitimately contains parentheses, quotation marks, colons and the word "AND"; a query, on the
 * RAG path, is a Fess query string assembled by the intent step and its operators are markup, not
 * words. This is a separate axis from the document/query prefix, which continues to tell the
 * model which side of a retrieval pair the text belongs to.</p>
 */
public class OllamaEmbeddingClientQueryTest extends UnitFessTestCase {

    private TestableOllamaEmbeddingClient client;

    @Override
    public void setUp(final TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableOllamaEmbeddingClient();
    }

    // -----------------------------------------------------------------------
    // ordering: strip before prefixing
    // -----------------------------------------------------------------------

    /**
     * The invariant specific to this client, and the reason the order is spelled out in
     * {@code embedQuery}'s Javadoc rather than left to the reader.
     *
     * <p>{@code QUERY_FIELD_PREFIX} matches {@code \b\w+:}, and both shipped query prefixes are
     * built out of exactly that: the default {@code "task: search result | query: "} and the
     * {@code nomic-embed} convention {@code "search_query: "}. Normalising after prefixing would
     * eat the prefix - {@code "search_query: "} disappears outright - and the only symptom would
     * be quietly worse recall, because the request still succeeds and still returns vectors.</p>
     */
    @Test
    public void test_embedQuery_stripsBeforeApplyingTheQueryPrefix() throws Exception {
        assertPrefixSurvives("search_query: ", "search_query: 陶芸 釉薬 種類 使い方");
        assertPrefixSurvives("task: search result | query: ", "task: search result | query: 陶芸 釉薬 種類 使い方");
    }

    private void assertPrefixSurvives(final String prefix, final String expected) throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            enqueueOneVector(server);
            server.start();
            setupClient(server);
            client.setTestQueryPrefix(prefix);

            client.embedQuery(List.of("+\"陶芸\" +\"釉薬\" (種類 OR 使い方)"));

            assertEquals(expected, firstInputOf(server.takeRequest()));
        } finally {
            server.shutdown();
        }
    }

    // -----------------------------------------------------------------------
    // toPlainQuery
    // -----------------------------------------------------------------------

    /**
     * The invariant that bounds this change's blast radius.
     *
     * <p>In fess 15.8.0 exactly two call sites reach {@code embedQuery}:
     * {@code SemanticChunkSearcher#search}, which calls it only after its own
     * {@code isPlainQuery(query)} returned true, and
     * {@code DefaultChatContentFetcher#resolveQueryVector}, which calls it with whatever the
     * intent step produced. Everything this method removes is something
     * {@code SemanticChunkSearcher.QUERY_SYNTAX_PATTERN} already rejects, so on that first call
     * site the transform is the identity and the semantic branch keeps embedding byte-for-byte
     * what it embedded before.</p>
     */
    @Test
    public void test_toPlainQuery_isIdentityForQueriesTheSemanticSearcherAccepts() {
        // Every string here passes SemanticChunkSearcher#isPlainQuery, so it is exactly the
        // population that reaches embedQuery from the semantic branch.
        final List<String> plain = List.of("自転車 変速 調整 方法", "珈琲 焙煎 温度 コーヒー豆", "bicycle derailleur adjustment", "天体観測 必要なもの 初心者 準備",
                "焙煎の温度はどのくらいですか", "nomic-embed-text", "mxbai-embed-large", "machine-learning 入門", "Fess", "検索エンジン");
        for (final String q : plain) {
            assertEquals(q, client.toPlainQuery(q));
        }
    }

    @Test
    public void test_toPlainQuery_removesRequiredTermPrefixes() {
        assertEquals("陶芸 釉薬", client.toPlainQuery("+陶芸 +釉薬"));
        assertEquals("Fess Docker", client.toPlainQuery("+Fess +Docker"));
        assertEquals("Fess Docker", client.toPlainQuery("+Fess -Docker"));
    }

    @Test
    public void test_toPlainQuery_removesQuotesAndGrouping() {
        assertEquals("養蜂 巣箱 管理 コツ 方法", client.toPlainQuery("+\"養蜂\" +\"巣箱\" (管理 OR コツ OR 方法)"));
        assertEquals("tutorial guide howto", client.toPlainQuery("(tutorial OR guide OR howto)"));
    }

    @Test
    public void test_toPlainQuery_removesFieldPrefixAndBoost() {
        // The field name is a schema name, not content: keeping "title" would add a term the
        // user never asked about.
        assertEquals("Fess", client.toPlainQuery("title:\"Fess\"^2"));
        assertEquals("大容量トークン検証用ドキュメント structure outline 節 セクション",
                client.toPlainQuery("title:\"大容量トークン検証用ドキュメント\" (structure OR outline OR 節 OR セクション)"));
    }

    @Test
    public void test_toPlainQuery_removesBooleanOperatorsAndRangeKeyword() {
        assertEquals("Fess Docker", client.toPlainQuery("Fess AND Docker"));
        assertEquals("Fess Docker", client.toPlainQuery("Fess NOT Docker"));
        assertEquals("Fess Docker", client.toPlainQuery("Fess && Docker"));
        assertEquals("Fess Docker", client.toPlainQuery("Fess || Docker"));
        assertEquals("2020 2024", client.toPlainQuery("[2020 TO 2024]"));
    }

    @Test
    public void test_toPlainQuery_keepsHyphenAndPlusInsideATerm() {
        // Only a leading +/- is an operator. Stripping mid-token would corrupt real terms.
        assertEquals("nomic-embed-text", client.toPlainQuery("nomic-embed-text"));
        assertEquals("C++ 入門", client.toPlainQuery("C++ 入門"));
        assertEquals("e-mail アドレス", client.toPlainQuery("+e-mail アドレス"));
    }

    /**
     * A query made only of operators must not normalize away to nothing: the prefix would then be
     * embedded on its own, which carries no information about what was asked. The original string
     * is embedded instead - degrading to the previous behaviour beats that.
     */
    @Test
    public void test_toPlainQuery_fallsBackToTheOriginalWhenNothingSurvives() {
        assertEquals("()", client.toPlainQuery("()"));
        assertEquals("AND OR", client.toPlainQuery("AND OR"));
        assertEquals("() AND OR", client.toPlainQuery("() AND OR"));
    }

    @Test
    public void test_toPlainQuery_passesNullAndBlankThrough() {
        assertNull(client.toPlainQuery(null));
        assertEquals("", client.toPlainQuery(""));
        assertEquals("   ", client.toPlainQuery("   "));
    }

    /**
     * {@code ?} and {@code *} are Lucene wildcards, so they go with the rest of the markup even
     * when they read as ordinary punctuation.
     *
     * <p>This is safe for both callers rather than a judgement call about English punctuation.
     * {@code SemanticChunkSearcher.QUERY_SYNTAX_PATTERN} contains {@code ?} too, so a query
     * carrying one never reaches {@code embedQuery} from the semantic branch at all; and the
     * string the chat branch passes is a Fess query built by the intent step, not a typed
     * sentence. A trailing {@code ?} also contributes nothing to an embedding.</p>
     */
    @Test
    public void test_toPlainQuery_removesWildcards() {
        assertEquals("what is fess", client.toPlainQuery("what is fess?"));
        assertEquals("検索エンジン", client.toPlainQuery("検索エンジン*"));
        assertEquals("fess search", client.toPlainQuery("fess? search*"));
    }

    @Test
    public void test_toPlainQuery_collapsesTheWhitespaceItLeavesBehind() {
        // Removing an operator leaves a gap; a run of spaces would otherwise be embedded.
        assertEquals("陶芸 釉薬", client.toPlainQuery("+陶芸    +釉薬"));
    }

    // -----------------------------------------------------------------------
    // wire behaviour
    // -----------------------------------------------------------------------

    @Test
    public void test_embedQuery_sendsTheNormalisedText() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            enqueueOneVector(server);
            server.start();
            setupClient(server);
            client.setTestQueryPrefix("");

            client.embedQuery(List.of("+\"養蜂\" +\"巣箱\" (管理 OR コツ)"));

            assertEquals("養蜂 巣箱 管理 コツ", firstInputOf(server.takeRequest()));
        } finally {
            server.shutdown();
        }
    }

    /**
     * Document text is prose. Removing its punctuation would change what is indexed, and would do
     * so asymmetrically from the query side, so {@code embedDocuments} must send the text through
     * untouched apart from its own prefix.
     */
    @Test
    public void test_embedDocuments_sendsTheTextUntouched() throws Exception {
        final MockWebServer server = new MockWebServer();
        try {
            enqueueOneVector(server);
            server.start();
            setupClient(server);
            client.setTestDocumentPrefix("");

            final String prose = "The AND gate (see figure 2) outputs \"1\" only when both inputs are 1.";
            client.embedDocuments(List.of(prose));

            assertEquals(prose, firstInputOf(server.takeRequest()));
        } finally {
            server.shutdown();
        }
    }

    // -----------------------------------------------------------------------
    // helpers
    // -----------------------------------------------------------------------

    private void enqueueOneVector(final MockWebServer server) {
        server.enqueue(new MockResponse().setBody("{\"model\":\"nomic-embed-text\",\"embeddings\":[[0.1,0.2,0.3]]}")
                .setHeader("Content-Type", "application/json"));
    }

    private void setupClient(final MockWebServer server) {
        client.setTestApiUrl(server.url("").toString().replaceAll("/$", ""));
        client.setTestModel("nomic-embed-text");
        client.setTestDimension(3);
        client.initHttpClient();
    }

    private static String firstInputOf(final RecordedRequest request) throws Exception {
        final JsonNode body = new ObjectMapper().readTree(request.getBody().readUtf8());
        return body.get("input").get(0).asText();
    }
}
