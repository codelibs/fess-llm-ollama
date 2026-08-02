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

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.codelibs.fess.util.CredentialUrlUtil;

/**
 * URL helpers shared by the Ollama LLM and embedding clients.
 *
 * <p>The two clients live in different packages ({@code org.codelibs.fess.llm.ollama}
 * and {@code org.codelibs.fess.embedding.ollama}), so a package-private helper on
 * either one cannot reach the other. This class holds the single copy so the two cannot drift.
 *
 * <p>These helpers exist because the configured Ollama endpoint is the only place in this
 * plugin where a secret can appear: there is no credential configuration key for Ollama
 * here at all, and the clients never authenticate. A secret reaches these code paths only
 * when an operator points {@code rag.llm.ollama.api.url} (or
 * {@code content_chunker.embedding.ollama.api.url}) at a reverse proxy that guards Ollama,
 * and carries the shared secret in the URL itself.
 *
 * <p>What stays here is Ollama-specific: the {@code /api}-segment normalization, the
 * query/fragment-preserving path append, and the wording of the refusal. Detecting and masking a
 * credential inside a URL is provider-agnostic and lives in {@link CredentialUrlUtil}.
 */
public final class OllamaUrlUtil {

    private OllamaUrlUtil() {
        // no instances
    }

    /**
     * Normalizes a configured Ollama endpoint into a base a fixed API path can be appended to.
     *
     * <p>Strips a trailing {@code /} and a trailing {@code /api} segment, so the forms shown in
     * the <a href="https://docs.ollama.com/api/introduction">Ollama API introduction</a>
     * ({@code http://localhost:11434}, {@code http://localhost:11434/},
     * {@code http://localhost:11434/api}) all reduce to the same host root. Idempotent.
     *
     * <p>The stripping applies to the <em>path</em> only. A query string or fragment is split
     * off first and re-attached unchanged, so an endpoint such as
     * {@code http://gateway/ollama/api?api_key=s3cr3t} normalizes to
     * {@code http://gateway/ollama?api_key=s3cr3t} rather than being left untouched because the
     * literal string does not end in {@code /api}.
     *
     * @param url the raw configured URL, possibly {@code null}
     * @return the normalized URL, or the input unchanged when {@code null} or blank
     */
    public static String normalizeBaseUrl(final String url) {
        if (url == null) {
            return null;
        }
        final String trimmed = url.trim();
        if (trimmed.isEmpty()) {
            return trimmed;
        }
        final int cut = pathEnd(trimmed);
        String path = trimmed.substring(0, cut);
        final String suffix = trimmed.substring(cut);
        while (path.endsWith("/")) {
            path = path.substring(0, path.length() - 1);
        }
        if (path.endsWith("/api")) {
            path = path.substring(0, path.length() - 4);
        }
        while (path.endsWith("/")) {
            path = path.substring(0, path.length() - 1);
        }
        return path + suffix;
    }

    /**
     * Appends a fixed API path such as {@code /api/embed} to a configured endpoint, keeping any
     * query string or fragment behind the appended path.
     *
     * <p>Plain string concatenation is wrong whenever the endpoint carries a query, and it fails
     * silently: {@code http://gateway/?api_key=s3cr3t} + {@code /api/tags} yields
     * {@code http://gateway/?api_key=s3cr3t/api/tags}, whose request target is the path
     * {@code /} with the secret's value corrupted into {@code s3cr3t/api/tags}. Nothing rejects
     * that URL - it is well-formed - so the request simply goes to the wrong place with a
     * mangled credential. A guarded Ollama endpoint carrying its shared secret as a query
     * parameter is exactly the configuration {@link CredentialUrlUtil#maskCredentialInUrl(String)} exists to
     * protect, so it has to actually work.
     *
     * @param baseUrl the configured endpoint, already {@link #normalizeBaseUrl(String) normalized}
     * @param path the API path to append, starting with {@code /}
     * @return the request URL
     */
    public static String appendPath(final String baseUrl, final String path) {
        if (baseUrl == null) {
            return path;
        }
        final int cut = pathEnd(baseUrl);
        return baseUrl.substring(0, cut) + path + baseUrl.substring(cut);
    }

    /**
     * Returns the offset at which the path component of {@code url} ends, i.e. the index of the
     * first {@code ?} or {@code #}, or the length when neither is present.
     *
     * @param url the URL to scan
     * @return the end offset of the path component
     */
    private static int pathEnd(final String url) {
        final int query = url.indexOf('?');
        final int fragment = url.indexOf('#');
        if (query < 0) {
            return fragment < 0 ? url.length() : fragment;
        }
        return fragment < 0 ? query : Math.min(query, fragment);
    }

    /**
     * Builds the operator-facing explanation for a refused userinfo-bearing endpoint. The
     * text names the offending configuration key and the supported alternative, and contains
     * no part of the configured value, so it is safe to log and to carry in an exception
     * message.
     *
     * <p>The refusal is not a policy choice this plugin is free to relax. RFC 9110 section
     * 4.2.4 states that a sender MUST NOT generate the userinfo subcomponent when an
     * {@code http} or {@code https} URI reference is generated as a target URI, and
     * httpclient5 enforces it: {@code ProtocolExec} (and its async counterpart) throw
     * {@code ProtocolException("Request URI authority contains deprecated userinfo component")}
     * unconditionally, with no setting that disables it. A userinfo-bearing endpoint can
     * therefore never issue a request; refusing it merely turns an opaque runtime failure
     * into an actionable configuration error.
     *
     * <p>Ollama itself does not authenticate, and neither do the sibling provider plugins
     * carry credentials this way. The legitimate case - an Ollama endpoint sitting behind an
     * authenticating forward proxy - is already served by Fess's
     * {@code http.proxy.host}/{@code .port}/{@code .username}/{@code .password}, which
     * fess-core wires into a {@code BasicCredentialsProvider}.
     *
     * @param configKey the configuration key holding the offending value
     * @return the message to log and to carry in the refusal exception
     */
    public static String userinfoRejectionMessage(final String configKey) {
        return configKey + " must not embed credentials in the URL authority (the 'name:secret@' before the host). "
                + "RFC 9110 section 4.2.4 forbids that form in an http/https target URI and the HTTP client rejects such a "
                + "request outright, so this endpoint can never be contacted. Ollama does not authenticate; if it sits behind "
                + "an authenticating proxy, remove the credentials from the URL and set http.proxy.host, http.proxy.port, "
                + "http.proxy.username and http.proxy.password instead.";
    }

    /**
     * Builds a {@code GET} request for {@code url}, replacing the URI-parse failure with one that
     * does not quote the URL. See
     * {@link CredentialUrlUtil#invalidUrlException(String, IllegalArgumentException)}.
     *
     * @param url the request URL
     * @param configKey the configuration key {@code url} was read from, named in the replacement
     *            exception
     * @return the request
     * @throws IllegalArgumentException if {@code url} cannot be parsed as a URI
     */
    public static HttpGet createHttpGet(final String url, final String configKey) {
        try {
            return new HttpGet(url);
        } catch (final IllegalArgumentException e) {
            throw CredentialUrlUtil.invalidUrlException(configKey, e);
        }
    }

    /**
     * Builds a {@code POST} request for {@code url}, replacing the URI-parse failure with one that
     * does not quote the URL. See
     * {@link CredentialUrlUtil#invalidUrlException(String, IllegalArgumentException)}.
     *
     * @param url the request URL
     * @param configKey the configuration key {@code url} was read from, named in the replacement
     *            exception
     * @return the request
     * @throws IllegalArgumentException if {@code url} cannot be parsed as a URI
     */
    public static HttpPost createHttpPost(final String url, final String configKey) {
        try {
            return new HttpPost(url);
        } catch (final IllegalArgumentException e) {
            throw CredentialUrlUtil.invalidUrlException(configKey, e);
        }
    }

}
