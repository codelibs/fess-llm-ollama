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

- `OllamaLlmClient` — the only production class. Extends `AbstractLlmClient` (from `fess` core, provided scope). Implements `chat()`, `streamChat()`, and `checkAvailabilityNow()`. Configuration is read from `fess_config.properties` via `ComponentUtil.getFessConfig()` with prefix `rag.llm.ollama.*`.
- Ollama-specific parameter mapping: `temperature` → `temperature`, `maxTokens` → `num_predict`, `top_p`/`top_k`/`num_ctx` via extra params. Global options from `rag.llm.ollama.options.*` system properties.
- Per-prompt-type config supports fallback to `rag.llm.ollama.default.*` keys.
- HTTP via Apache HttpClient 5. Streaming uses NDJSON line-by-line parsing.

## Testing

Tests use `UnitFessTestCase` (extends utflute's `WebContainerTestCase`) with `test_app.xml` container config. HTTP interactions are tested with OkHttp `MockWebServer`. The `TestableOllamaLlmClient` inner class overrides config methods to avoid `ComponentUtil` dependency in tests.

## Coding Conventions

- Follow Fess coding style: `final` on all local variables and parameters, Log4j2 logging, Apache License 2.0 headers on all files.
- Config keys use dot-separated notation: `rag.llm.ollama.<section>.<param>`.
- Debug logs use `[LLM:OLLAMA]` prefix pattern.
