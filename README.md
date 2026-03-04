Ollama LLM Plugin for Fess
==========================

## Overview

This plugin provides Ollama integration for Fess's LLM (Large Language Model) features. It enables Fess to use locally hosted Ollama models for AI-powered search capabilities such as intent detection, answer generation, and document summarization.

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
| `llm.api.url` | `http://localhost:11434` | Ollama API endpoint URL |
| `llm.model` | - | Model name (e.g., `llama3:latest`) |
| `llm.temperature` | `0.7` | Temperature for response generation |
| `llm.max.tokens` | `1000` | Maximum tokens for response |
| `llm.timeout` | `30000` | HTTP request timeout in milliseconds |

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
