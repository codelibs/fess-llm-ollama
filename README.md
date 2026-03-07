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
| `rag.llm.ollama.model` | `gemma3:4b` | Model name (e.g., `llama3:latest`, `mistral`) |
| `rag.llm.ollama.timeout` | `60000` | HTTP request timeout in milliseconds |
| `rag.llm.ollama.availability.check.interval` | `60` | Interval (seconds) for checking Ollama server availability |
| `rag.llm.ollama.answer.context.max.chars` | `10000` | Maximum characters for document context in answer generation |
| `rag.llm.ollama.summary.context.max.chars` | `10000` | Maximum characters for document context in summary generation |
| `rag.llm.ollama.faq.context.max.chars` | `6000` | Maximum characters for document context in FAQ generation |
| `rag.llm.ollama.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |

### Recommended num_ctx Setting

For `gemma3:4b` with 16GB GPU, set:

```properties
rag.llm.ollama.default.num.ctx=8192
```

### Per-Prompt-Type Parameters

You can configure `top_p` and `top_k` sampling parameters for each prompt type:

| Property | Description |
|----------|-------------|
| `rag.llm.ollama.<promptType>.top.p` | Top-p (nucleus) sampling parameter |
| `rag.llm.ollama.<promptType>.top.k` | Top-k sampling parameter |

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
