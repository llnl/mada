# OpenAI API Mode

## What Is OpenAI API Mode?

OpenAI API mode exposes the configured MADA agent team through an OpenAI-compatible HTTP API. This is intended for tools that already know how to speak to `/v1/models` and `/v1/chat/completions`, such as Open WebUI.

## Starting OpenAI API Mode

Launch the API server with:

```bash
mada-openai-api configs/mada_config.json
```

You can also launch it through the main entrypoint:

```bash
mada openai-api configs/mada_config.json
```

By default, the server listens on `http://0.0.0.0:8000` and exposes the model name `mada`.

## Useful Options

```bash
mada-openai-api --host 127.0.0.1 --port 8000 --model-name mada-team configs/mada_config.json
```

If you want the API to require a client key:

```bash
mada-openai-api --api-key local-dev-key configs/mada_config.json
```

If your streamable HTTP MCP servers require a bearer token:

```bash
mada-openai-api --bearer-token "$MCP_BEARER_TOKEN" configs/mada_config.json
```

## Connecting Open WebUI

In Open WebUI, add a new OpenAI connection with:

- Base URL: `http://localhost:8000/v1`
- API Key: any string if you did not configure `--api-key`, otherwise the value you configured
- Model: the value from `--model-name`, such as `mada-team`

Open WebUI sends the full chat history in each request. The MADA API mode rebuilds the conversation for each HTTP request so callers do not share state with the CLI or Gradio sessions.

When `orchestration.mode` is `magentic`, MADA still keeps the same OpenAI-compatible request and response shapes. The server runs a fresh Magentic workflow for each request, rebuilds state from the supplied transcript, and returns only the final assistant text rather than internal planning or progress events.
