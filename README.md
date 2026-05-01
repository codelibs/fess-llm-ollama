Ollama LLM Plugin for Fess
==========================

## Overview

This plugin provides Ollama integration for Fess's RAG (Retrieval-Augmented Generation) features. It enables Fess to use locally hosted Ollama models for AI-powered search capabilities including intent detection, answer generation, document summarization, and FAQ handling.

## Download

See [Maven Repository](https://repo1.maven.org/maven2/org/codelibs/fess/fess-llm-ollama/).

## Requirements

- Fess 15.x or later
- Java 21 or later
- Ollama server running locally or accessible via network

## Installation

1. Download the plugin JAR from the Maven Repository
2. Place it in your Fess plugin directory
3. Restart Fess

For detailed instructions, see the [Plugin Administration Guide](https://fess.codelibs.org/14.19/admin/plugin-guide.html).

## Configuration

Configure the following properties in `fess_config.properties`:

| Property | Default | Description |
|----------|---------|-------------|
| `rag.llm.name` | - | Set to `ollama` to use this plugin |
| `rag.chat.enabled` | `false` | Enable RAG chat feature |
| `rag.llm.ollama.api.url` | `http://localhost:11434` | Ollama API endpoint URL |
| `rag.llm.ollama.answer.context.max.chars` | `10000` | Maximum characters for document context in answer generation |
| `rag.llm.ollama.availability.check.interval` | `60` | Interval (seconds) for checking Ollama server availability |
| `rag.llm.ollama.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |
| `rag.llm.ollama.connect.timeout` | `5000` | TCP connect timeout (ms). Separate from `timeout` (read/response). |
| `rag.llm.ollama.default.max.tokens` | (unset) | Fallback when `<type>.max.tokens` is not set. |
| `rag.llm.ollama.default.temperature` | (unset) | Fallback when `<type>.temperature` is not set. |
| `rag.llm.ollama.default.thinking.budget` | (unset) | Fallback when `<type>.thinking.budget` is not set. |
| `rag.llm.ollama.faq.context.max.chars` | `6000` | Maximum characters for document context in FAQ generation |
| `rag.llm.ollama.model` | `gemma4:e4b` | Model name (e.g., `llama3:latest`, `mistral`) |
| `rag.llm.ollama.retry.base.delay.ms` | `2000` | Base delay (ms) for exponential backoff with ±20% jitter. |
| `rag.llm.ollama.retry.max` | `3` | Maximum total attempts on retryable HTTP errors (500/503/504) and connect-time IOExceptions. |
| `rag.llm.ollama.summary.context.max.chars` | `10000` | Maximum characters for document context in summary generation |
| `rag.llm.ollama.timeout` | `60000` | Response/read timeout (ms). For TCP connect timeout see `rag.llm.ollama.connect.timeout`. |

### Recommended num_ctx Setting

For `gemma4:e4b` with 16GB GPU, set:

```properties
rag.llm.ollama.default.num.ctx=8192
```

### Per-Prompt-Type Parameters

You can configure `top_p` and `top_k` sampling parameters for each prompt type:

| Property | Description |
|----------|-------------|
| `rag.llm.ollama.<promptType>.top.p` | Top-p (nucleus) sampling parameter |
| `rag.llm.ollama.<promptType>.top.k` | Top-k sampling parameter |

## Retry behavior

Both `chat()` and `streamChat()` retry on:

- HTTP `500`, `503` (Ollama queue overload via `OLLAMA_MAX_QUEUE`), `504`
- `IOException` raised before a response is received (DNS, TCP, TLS, idle-socket failures)

`429` is **not** retried — Ollama does not return 429. `4xx` errors are surfaced as
`LlmException` immediately.

Streaming retries only the initial HTTP request. Once NDJSON bytes start flowing,
in-stream errors propagate immediately (no replay).

Defaults can be overridden via `rag.llm.ollama.retry.max` and
`rag.llm.ollama.retry.base.delay.ms`.

## Stream completion log

A single INFO line is emitted per `streamChat()` call:

```
[LLM:OLLAMA] Stream completed. chunkCount=N, objectCount=N, firstChunkMs=N,
  elapsedTime=Nms, doneReason=stop, totalDurationMs=N, loadDurationMs=N,
  promptEvalDurationMs=N, evalDurationMs=N, promptEvalCount=N, evalCount=N,
  tokensPerSecond=N.NN, parseErrorCount=0
```

A sibling WARN line is emitted when `done_reason` is anything other than `stop`,
`load`, or `unload` — most commonly `length` (context window truncation):

```
[LLM:OLLAMA] Stream finished abnormally. doneReason=length, evalCount=N, ...
```

## Reasoning Model Configuration (e.g., qwen3.5)

Reasoning models like `qwen3.5` use internal thinking tokens that improve answer quality
but consume output tokens. Configure thinking per prompt type for optimal results.

```properties
rag.llm.ollama.model=qwen3.5:35b
rag.llm.ollama.timeout=120000

# Structured output / short responses - disable thinking
rag.llm.ollama.intent.thinking.budget=0
rag.llm.ollama.evaluation.thinking.budget=0
rag.llm.ollama.unclear.thinking.budget=0
rag.llm.ollama.noresults.thinking.budget=0
rag.llm.ollama.docnotfound.thinking.budget=0

# Answer generation - enable thinking with increased token limit
rag.llm.ollama.answer.thinking.budget=1
rag.llm.ollama.answer.max.tokens=16384
rag.llm.ollama.summary.thinking.budget=1
rag.llm.ollama.summary.max.tokens=16384
rag.llm.ollama.direct.thinking.budget=1
rag.llm.ollama.direct.max.tokens=8192
rag.llm.ollama.faq.thinking.budget=1
rag.llm.ollama.faq.max.tokens=8192
```

The `thinking.budget` parameter controls the Ollama `think` flag:
- `0` — disable thinking (`think: false`)
- Any positive value — enable thinking (`think: true`)
- Not set — use model default (reasoning models default to thinking enabled)

When thinking is enabled, increase `max.tokens` to accommodate both thinking and content tokens.

## Features

- **Intent Detection** - Determines user intent (search, summary, FAQ, unclear) and generates Lucene queries
- **Answer Generation** - Generates answers based on search results with citation support
- **Document Summarization** - Summarizes specific documents
- **FAQ Handling** - Provides direct, concise answers to FAQ-type questions
- **Relevance Evaluation** - Identifies the most relevant documents for answer generation
- **Streaming Support** - Real-time response streaming via NDJSON format
- **Availability Checking** - Validates Ollama server and model availability at configurable intervals

## Ollama API Endpoints Used

- `GET /api/tags` - Lists available models for availability checking
- `POST /api/chat` - Performs chat completion (supports both standard and streaming modes)

## Development

### Building from Source

```bash
mvn clean package
```

### Running Tests

```bash
mvn test
```

## License

Apache License 2.0
