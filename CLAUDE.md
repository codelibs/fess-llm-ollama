# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama LLM plugin for Fess. Provides Ollama integration for Fess's RAG features (intent detection, answer generation, document summarization, FAQ handling, relevance evaluation) via the Ollama `/api/chat` and `/api/tags` endpoints.

Single-class plugin: `OllamaLlmClient` extends `AbstractLlmClient` from the Fess core project.

## Build & Test

Requires Java 21 and `fess-parent` POM installed locally (CI checks out and installs it from `codelibs/fess-parent` main branch).

```bash
# Build (includes tests)
mvn clean package

# Run tests only
mvn test

# Run a single test
mvn test -Dtest=OllamaLlmClientTest#test_chat_success

# Install fess-parent locally (needed if not already in ~/.m2)
cd /path/to/fess-parent && mvn install -Dgpg.skip=true
```

Code formatting is enforced by `formatter-maven-plugin` and license headers by `license-maven-plugin` (both configured in fess-parent).

## Architecture

- `OllamaLlmClient` — the only production class. Extends `AbstractLlmClient` (from `fess` core, provided scope). Implements `chat()`, `streamChat()`, and `checkAvailabilityNow()`. Configuration is read from `fess_config.properties` via `ComponentUtil.getFessConfig()` with prefix `rag.llm.ollama.*`. `getApiUrl()` normalizes the configured URL by stripping a trailing `/api` segment and trailing slash, so that `http://localhost:11434`, `http://localhost:11434/`, and `http://localhost:11434/api` (the form shown in the official Ollama docs) all resolve to the same base.
- `OllamaEmbeddingClient` — implements `AbstractEmbeddingClient` for the content-chunk embedding
  SPI, prefix `content_chunker.embedding.ollama.*`. `embedQuery()` runs its texts through
  `toPlainQuery()` (removes Fess/Lucene query syntax) **before** `applyPrefix` prepends
  `query.prefix`; `embedDocuments()` only prefixes. The order is load-bearing — both shipped
  query prefixes (`task: search result | query: `, `search_query: `) match the `\b\w+:` pattern
  `toPlainQuery` removes, so normalizing after prefixing would delete the prefix and degrade
  recall silently. `OllamaEmbeddingClientQueryTest#test_embedQuery_stripsBeforeApplyingTheQueryPrefix`
  pins it; the sibling `fess-llm-openai` (PR #25) and `fess-llm-gemini` (PR #26) carry the same
  `toPlainQuery` contract, so keep the three in step.
- Ollama-specific parameter mapping: `temperature` → `temperature`, `maxTokens` → `num_predict`, `top_p`/`top_k`/`num_ctx` via extra params. Global options from `rag.llm.ollama.options.*` system properties.
- Per-prompt-type config supports fallback to `rag.llm.ollama.default.*` keys.
- HTTP via Apache HttpClient 5. Streaming uses NDJSON line-by-line parsing.

### Logging keys

`streamChat` emits one `[LLM:OLLAMA] Stream completed.` INFO line per call carrying:
`chunkCount`, `objectCount`, `firstChunkMs`, `elapsedTime`, `doneReason`,
`totalDurationMs`, `loadDurationMs`, `promptEvalDurationMs`, `evalDurationMs`,
`promptEvalCount`, `evalCount`, `tokensPerSecond`, `parseErrorCount`.

When `done_reason` is anything other than `stop`/`load`/`unload`, both `chat()`
and `streamChat()` emit an extra WARN line so context truncation (`length`) and
future abnormal reasons can be alerted on without enabling DEBUG.

Enable `org.codelibs.fess.llm.ollama` at DEBUG to additionally log:
- the JSON request body (`requestBody=`),
- HTTP status + `Content-Type` of the streaming response,
- the `thinking` field length when reasoning models emit one.

### Retries and timeouts

Retries: HTTP `429`, `500`, `502`, `503`, `504` and connect-time `IOException`
are retried up to `rag.llm.ollama.retry.max` times (default `3`) with
exponential backoff starting at `rag.llm.ollama.retry.base.delay.ms` (default
`2000`) and ±20% jitter. The retryable set tracks the documented Ollama errors
(<https://docs.ollama.com/api/errors>) and covers Ollama Cloud rate limits.
Other `4xx` is treated as a configuration error. Streaming retries only the
initial connect — once the NDJSON body starts flowing, partial-stream errors
(transport failures **or** in-stream `{"error": "..."}` payloads) propagate
immediately to `LlmStreamCallback.onError(...)`.

Timeouts are two-tier:
- `rag.llm.ollama.connect.timeout` (default `5000`) — TCP connect, connection-request acquisition.
- `rag.llm.ollama.timeout` (default `60000`) — response/read timeout.

The override of `init()` mirrors `AbstractLlmClient.init()` from `repos/fess`.
If the base class adds new HTTP-client configuration (interceptors, pool
settings), update the override to match.

### Per-prompt-type config keys

Each prompt type (`intent`, `evaluation`, `unclear`, `noresults`, `docnotfound`,
`direct`, `faq`, `answer`, `summary`, `queryregeneration`) supports the
following per-type overrides via `fess_config.properties`:

- `rag.llm.ollama.<type>.thinking.budget` — boolean form (`0` ⇒ `think: false`, `>0` ⇒ `think: true`)
- `rag.llm.ollama.<type>.thinking.level` — string form (`high` / `medium` / `low`); required for GPT-OSS family models which ignore the boolean form. When set, overrides the boolean derived from `thinking.budget` for that prompt type.
- `rag.llm.ollama.<type>.max.tokens`
- `rag.llm.ollama.<type>.temperature`
- `rag.llm.ollama.<type>.top.p`, `.top.k`, `.num.ctx`
- `rag.llm.ollama.<type>.context.max.chars`

Resolution order: `<type>.<param>` → `default.<param>` → hardcoded type default
(in `applyDefaultParams`) → unset. Explicit user values on the request always win
over computed defaults.

## Testing

Tests use `UnitFessTestCase` (extends utflute's `WebContainerTestCase`) with `test_app.xml` container config. HTTP interactions are tested with OkHttp `MockWebServer`. The `TestableOllamaLlmClient` inner class overrides config methods to avoid `ComponentUtil` dependency in tests.

## Coding Conventions

- Follow Fess coding style: `final` on all local variables and parameters, Log4j2 logging, Apache License 2.0 headers on all files.
- Config keys use dot-separated notation: `rag.llm.ollama.<section>.<param>`.
- Debug logs use `[LLM:OLLAMA]` prefix pattern.
