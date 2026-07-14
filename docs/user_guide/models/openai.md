# OpenAI Model Configuration

This section provides details on configuring OpenAI models for use in the MADA Orchestrator.

## Fields

All fields, besides `extra`, can be either a literal value or an environment variable.

| Field Name        | Description                                           | Required? |
| ----------------- | ----------------------------------------------------- | --------- |
| `provider`        | The provider name, must be `openai`.                 | Yes       |
| `model`           | The name of the OpenAI model to use (e.g., "gpt-4"). | Yes       |
| `api_key`         | The API key for authentication. In addition to being either a literal value or an environment variable, this can also be a file path. | Yes       |
| `base_url`        | The base URL of the OpenAI API endpoint.             | Yes       |
| `extra`           | Additional settings (e.g., temperature, max_tokens). | No        |

MADA uses the system trust store for TLS verification when connecting to
OpenAI-compatible endpoints. If your environment requires a custom CA bundle,
set `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE` before starting MADA.

## Example

```json
"model": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${API_KEY}",
    "base_url": "https://api.openai.com/v1",
    "extra": {
        "temperature": 0.7,
        "max_tokens": 2048
    }
}
```
