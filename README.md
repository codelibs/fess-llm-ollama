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

This plugin's properties are split across Fess's two independent configuration channels;
each subsection below states which one applies. Read the channel note before editing a
property — setting it in the wrong file is a silent no-op.

`rag.llm.name` selects which registered LLM client Fess's RAG feature uses, and belongs in
**`conf/system.properties`** (or a `-Dfess.system.rag.llm.name` JVM argument) — not
`fess_config.properties`. Set it to `ollama` to activate this plugin:

| Property | Default | Description |
|----------|---------|-------------|
| `rag.llm.name` | - | Set to `ollama` to use this plugin. Read from `conf/system.properties`, not `fess_config.properties`. |

Configure the following properties in `fess_config.properties` (the LastaFlute config store,
loaded once at container boot; changes require a Fess restart):

| Property | Default | Description |
|----------|---------|-------------|
| `rag.chat.enabled` | `false` | Enable RAG chat feature |
| `rag.llm.ollama.api.url` | `http://localhost:11434` | Ollama server root URL. The plugin appends `/api/chat` and `/api/tags`, so a trailing `/` or `/api` (the form shown in the Ollama docs, e.g. `http://localhost:11434/api` or `https://ollama.com/api`) is stripped automatically. A query string is preserved and stays behind the appended path, so an endpoint such as `http://gateway/ollama?api_key=...` becomes `http://gateway/ollama/api/chat?api_key=...`. |
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
| `rag.llm.ollama.retry.max` | `3` | Maximum total attempts on retryable HTTP errors (429/500/502/503/504) and connect-time IOExceptions. |
| `rag.llm.ollama.summary.context.max.chars` | `10000` | Maximum characters for document context in summary generation |
| `rag.llm.ollama.timeout` | `60000` | Response/read timeout (ms). For TCP connect timeout see `rag.llm.ollama.connect.timeout`. |

### Content Chunk Embedding

When Fess's content-chunking RAG feature (`content_chunker.enabled=true`) is configured to use
this plugin as its embedding provider (`content_chunker.embedding.name=ollama`), the following
properties configure `OllamaEmbeddingClient`, which calls Ollama's `POST /api/embed` endpoint.

**Every `content_chunker.embedding.ollama.*` property below belongs in
`conf/system.properties`** (or as a `-Dfess.system.<key>` JVM argument) — the same channel
every other `content_chunker.*` setting uses, admin-visible read-only under System Info >
Config Info > App Properties. Setting one of these in `fess_config.properties` instead has no
effect.

Most of these properties are read on every call, so an edit takes effect without a Fess
restart. The exceptions are `timeout`, `connect.timeout`, and `availability.check.interval`,
which are read once when the embedding client initializes and require a restart to pick up a
change.

| Property | Default | Description |
|----------|---------|-------------|
| `content_chunker.embedding.ollama.api.url` | `http://localhost:11434` | Ollama server root URL. Same handling as `rag.llm.ollama.api.url` (trailing `/` or `/api` stripped, query string preserved behind the appended path). |
| `content_chunker.embedding.ollama.model` | `embeddinggemma` | Embedding model name. The default is multilingual; `nomic-embed-text` is English-only and separates non-English documents poorly. Change `document.prefix` and `query.prefix` together with this key -- see the note below the table. |
| `content_chunker.embedding.ollama.document.prefix` | `title: none \| text: ` | Task prefix prepended to document/chunk texts before embedding, per the `embeddinggemma` convention. Replace `none` with the document's own title if you have one. Set to an empty string to disable for models that don't use task prefixes. |
| `content_chunker.embedding.ollama.query.prefix` | `task: search result \| query: ` | Task prefix prepended to query texts before embedding, per the `embeddinggemma` convention. Set to an empty string to disable for models that don't use task prefixes. |
| `content_chunker.embedding.ollama.truncate` | `true` | Sent explicitly as the `truncate` field of every `/api/embed` request. `true` (Ollama's own default) silently cuts an over-context chunk down to fit and still returns a valid vector, so the relevance loss is invisible; `false` makes Ollama reject the input instead, so the chunk fails loudly rather than being indexed with a degraded vector. Chunk size is governed by `content_chunker.length.chunk_size`. An unparseable value keeps `true` and logs a WARN. |
| `content_chunker.embedding.ollama.timeout` | `60000` | Response/read timeout (ms). |
| `content_chunker.embedding.ollama.connect.timeout` | `5000` | TCP connect timeout (ms). Separate from `timeout` (read/response). |
| `content_chunker.embedding.ollama.availability.check.interval` | `60` | Interval (seconds) for checking Ollama server availability. |
| `content_chunker.embedding.ollama.retry.max` | `3` | Maximum total attempts on retryable HTTP errors (429/500/502/503/504) and connect-time IOExceptions. |
| `content_chunker.embedding.ollama.retry.base.delay.ms` | `2000` | Base delay (ms) for exponential backoff with ±20% jitter. |

Also requires the shared `content_chunker.embedding.dimension` property (embedding vector
dimension, also read from `conf/system.properties`) to be set, independent of this plugin.
The default model emits 768-dimensional vectors, so `content_chunker.embedding.dimension=768`.

#### Choosing a model, and the task prefixes that go with it

Embedding models are trained with their own task prefixes, and a prefix belonging to a
different model family still produces well-formed vectors of the correct dimension — nothing
fails, only relevance degrades. So `model`, `document.prefix` and `query.prefix` must be
changed together. The plugin logs a WARN at startup when they disagree.

| Model | `document.prefix` | `query.prefix` |
|-------|-------------------|----------------|
| `embeddinggemma` (default, 768, multilingual) | `title: none \| text: ` | `task: search result \| query: ` |
| `nomic-embed-text` (768, English) | `search_document: ` | `search_query: ` |
| anything else | empty, unless the model documents its own | empty |

`nomic-embed-text` is trained on English. Measured on a 14-document Japanese corpus, every
document scored between 0.76 and 0.83 for a paraphrase query and the expected document ranked
last; `embeddinggemma` ranked the same document first on the same corpus. Both emit
768-dimensional vectors, so switching between them costs a re-index and no mapping change.

Changing the model requires re-running the chunk-vector job over the whole index: vectors
already stored were produced by the previous model and are not comparable with the new one.

#### Query normalization

`embedQuery()` strips Fess/Lucene query syntax — `+required` terms, `(a OR b)` groups,
`title:"x"^2` field boosts, quoted phrases, `?`/`*` wildcards — before embedding, because on the
RAG path the string it receives is a Fess query built by the LLM's intent step and those operators
are markup rather than words. `embedDocuments()` strips nothing: document text is prose whose
punctuation is content. A query left empty by the removals is embedded unchanged.

The stripping happens **before** `query.prefix` is prepended, and the order is not
interchangeable: both shipped prefixes end in a `word:` sequence, which is itself one of the
patterns removed. Normalizing afterwards would eat the prefix — `search_query: ` would vanish
outright — and the model would lose the task hint with no error to show for it.

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

- HTTP `429` (Too Many Requests; Ollama Cloud and rate-limited proxies)
- HTTP `500`, `502`, `503` (Ollama queue overload via `OLLAMA_MAX_QUEUE`), `504`
- `IOException` raised before a response is received (DNS, TCP, TLS, idle-socket failures)

Other `4xx` errors are surfaced as `LlmException` immediately.

Streaming retries only the initial HTTP request. Once NDJSON bytes start flowing,
in-stream errors (HTTP transport failures **or** NDJSON `{"error": "..."}` payloads)
propagate immediately to `LlmStreamCallback.onError(...)` — no replay.

The retry status set tracks the documented [Ollama errors](https://docs.ollama.com/api/errors).

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

The `thinking.budget` parameter controls the Ollama `think` flag as a boolean:
- `0` — disable thinking (`think: false`)
- Any positive value — enable thinking (`think: true`)
- Not set — use model default (reasoning models default to thinking enabled)

When thinking is enabled, increase `max.tokens` to accommodate both thinking and content tokens.

### thinking.level (GPT-OSS and other models that ignore the boolean form)

Per [Ollama's thinking docs](https://docs.ollama.com/capabilities/thinking), the `think`
field also accepts the string values `high`, `medium`, and `low`. GPT-OSS models in
particular ignore the boolean form. Use `rag.llm.ollama.<promptType>.thinking.level`
(or `rag.llm.ollama.default.thinking.level`) to send a string instead of a boolean:

```properties
rag.llm.ollama.model=gpt-oss:20b
rag.llm.ollama.answer.thinking.level=high
rag.llm.ollama.intent.thinking.level=low
```

When `thinking.level` is set, it overrides the boolean derived from `thinking.budget`
for that prompt type. Allowed values: `high`, `medium`, `low` (case-insensitive).
Invalid values are ignored with a WARN log and fall back to `thinking.budget`.

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
