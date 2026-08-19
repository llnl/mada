# LivAI Model Configuration

This section provides details on configuring LivAI models for use in the MADA Orchestrator. LivAI serves every model that it has available through OpenAI's API, so configuration will use the exact same fields as [OpenAI Model Configuration](./openai.md), even for non-OpenAI models.

## Fields

All fields, besides `extra`, can be either a literal value or an environment variable.

| Field Name        | Description                                           | Required? |
| ----------------- | ----------------------------------------------------- | --------- |
| `provider`        | The provider name, must be `livai`.                   | Yes       |
| `model`           | The name of the OpenAI model to use (e.g., "gpt-4").  | Yes       |
| `api_key`         | The API key for authentication. In addition to being either a literal value or an environment variable, this can also be a file path. | Yes       |
| `base_url`        | The base URL of the OpenAI API endpoint.              | Yes       |
| `verify`          | TLS verification setting. Use `true` for env/system trust resolution, `false` to disable verification, or a CA bundle path. | No |
| `extra`           | Additional settings (e.g., temperature, max_tokens).  | No        |

## Example

```json
"model": {
    "provider": "livai",
    "model": "claude-sonnet-4.5",
    "api_key": "${API_KEY}",
    "base_url": "https://livai-api.llnl.gov/v1",
    "verify": true,
    "extra": {
        "temperature": 0.7,
        "max_tokens": 2048
    }
}
```
