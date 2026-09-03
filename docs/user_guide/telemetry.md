# Telemetry

MADA can record a timeline of every agent run — which agent was invoked, which
tools it called, how long each step took, how many tokens each LLM call used,
and (optionally) the exact prompts and responses. Useful for debugging odd
agent behavior, watching token cost, and finding slow steps.

Under the hood MADA emits OpenTelemetry (OTLP) via
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework)'s
built-in instrumentation, so any OTLP-compatible dashboard works. The
recommended local option is Microsoft's own Aspire Dashboard.

## Install the optional dependency

The OpenTelemetry SDK is not part of MADA's default install. Add the
`telemetry` extra:

```bash
pip install --pre -e '.[telemetry]'
```

Then enable telemetry in your config:

```json
{
  "telemetry": {
    "enabled": true
  }
}
```

## Run Aspire Dashboard

```bash
docker run --rm -it -p 18888:18888 -p 4317:18889 \
  -e DASHBOARD__OTLP__AUTHMODE=Unsecured \
  -e DASHBOARD__FRONTEND__AUTHMODE=Unsecured \
  mcr.microsoft.com/dotnet/aspire-dashboard:9.0
```

The two `AUTHMODE=Unsecured` flags are only appropriate for localhost dev —
they turn off the login screen and the OTLP bearer-token requirement.

## Point MADA at it

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
mada-cli configs/example_helpful_critic.json
```

Open [http://localhost:18888](http://localhost:18888) and click **Traces** to
see each run. Expand a trace to see the delegation chain
(orchestrator → sub-agents → LLM calls → tool invocations) with per-span
timing and attributes.

## Environment variables

| Variable | Purpose |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Where to send traces/metrics/logs. Unset = no export. |
| `ENABLE_SENSITIVE_DATA` | Set to `true` to include prompt/response text in spans. Off by default. |

Both are read by Microsoft Agent Framework directly; MADA does not shadow
them. Any other standard OTel env var (`OTEL_SERVICE_NAME`,
`OTEL_RESOURCE_ATTRIBUTES`, etc.) is also honored.

## Disabling telemetry

Remove the `telemetry` block from your config, or set `"enabled": false`.
With telemetry disabled, `setup_telemetry()` returns before touching MSAF,
so `OTEL_EXPORTER_OTLP_ENDPOINT` is ignored even if set.

## Other backends

Aspire is one option; any OTLP receiver works. Point
`OTEL_EXPORTER_OTLP_ENDPOINT` at whatever you already run.
