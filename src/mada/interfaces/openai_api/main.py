# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
OpenAI-compatible HTTP interface for MADA Orchestrator.

This module exposes the configured MADA agent team through a small FastAPI app
that implements the OpenAI-compatible `/v1/models` and `/v1/chat/completions`
endpoints expected by tools such as Open WebUI.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
import secrets
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import click

try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
except (
    ImportError
) as exc:  # pragma: no cover - exercised only in missing dependency environments
    FastAPI = None
    Header = None
    HTTPException = None
    JSONResponse = None
    StreamingResponse = None
    FASTAPI_IMPORT_ERROR = exc
else:
    FASTAPI_IMPORT_ERROR = None

try:
    import uvicorn
except (
    ImportError
) as exc:  # pragma: no cover - exercised only in missing dependency environments
    uvicorn = None
    UVICORN_IMPORT_ERROR = exc
else:
    UVICORN_IMPORT_ERROR = None

from mada.core.config import AppConfig, OrchestrationConfig
from mada.core import load_config_from_json
from mada.core.telemetry import setup_telemetry

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator


DEFAULT_MODEL_NAME = "mada"


def _get_orchestration_config(config: AppConfig) -> OrchestrationConfig:
    return getattr(config, "orchestration", None) or OrchestrationConfig()


class OrchestratorStartupError(RuntimeError):
    """
    Raised when the orchestrator cannot be initialized for API requests.

    This is used to convert lower-level startup failures, most commonly MCP
    connection problems, into a stable error type that the FastAPI layer can map
    to a `503 Service Unavailable` response.
    """


def _format_startup_error_message(exc: BaseException) -> str:
    """
    Convert orchestrator startup failures into a concise user-facing message.

    The most common failure mode here is unreachable MCP infrastructure. Keep the
    message short and actionable so API clients get something they can surface.

    Args:
        exc: The original exception raised while starting the orchestrator.

    Returns:
        A user-facing error message suitable for inclusion in an HTTP error
        response body.
    """
    details = str(exc).strip() or exc.__class__.__name__
    lowered = details.lower()

    if "connect" in lowered or "connection" in lowered or "cancellederror" in lowered:
        return (
            "MADA could not connect to one or more MCP servers. "
            "Check the MCP server processes and the URLs/commands in your config. "
            f"Details: {details}"
        )

    return f"MADA failed to initialize the configured agent team. Details: {details}"


def _require_fastapi() -> None:
    """
    Raise a clear runtime error if FastAPI dependencies are missing.

    Raises:
        RuntimeError: If `fastapi` or `uvicorn` could not be imported in the
            current Python environment.
    """
    if FASTAPI_IMPORT_ERROR is not None or uvicorn is None:
        missing = []
        if FASTAPI_IMPORT_ERROR is not None:
            missing.append("fastapi")
        if UVICORN_IMPORT_ERROR is not None:
            missing.append("uvicorn")
        packages = ", ".join(missing) or "fastapi, uvicorn"
        raise RuntimeError(
            f"OpenAI API mode requires {packages}. Install the project dependencies again."
        )


class MADAOpenAIAPIService:
    """
    Manage the shared orchestrator instance used by the HTTP API.

    The API server keeps one lazily-initialized orchestrator for the process and
    reuses it across requests. Model listing and health checks can succeed before
    the orchestrator has been started; chat requests trigger startup on demand.

    Attributes:
        config: Full application configuration used to construct the
            orchestrator and its agents.
        model_name: Model identifier exposed through `/models` and
            `/v1/models`.
        api_key: Optional API key required for incoming HTTP requests.
        bearer_token: Optional bearer token forwarded to streamable HTTP MCP
            servers as `X-Token`.
        orchestrator: Lazily-created orchestrator instance, or `None` before
            startup completes successfully.
        _startup_lock: Async lock used to ensure only one request performs the
            lazy startup sequence.
    """

    def __init__(
        self,
        config: AppConfig,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpenAI API service wrapper.

        Args:
            config: Parsed application configuration.
            model_name: Externally visible model identifier to publish via the
                OpenAI-compatible model listing endpoints.
            api_key: Optional API key that callers must provide through
                `Authorization: Bearer ...` or `x-api-key`.
            bearer_token: Optional token forwarded to configured streamable HTTP
                MCP servers during orchestrator startup.
        """
        self.config = config
        self.model_name = model_name
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.orchestrator: Optional[MADAOrchestrator] = None
        self._startup_lock = asyncio.Lock()

    async def startup(self) -> None:
        """
        Initialize the orchestrator and all configured agents.

        Returns:
            `None`. The initialized orchestrator is stored on `self.orchestrator`.

        Raises:
            OrchestratorStartupError: If the orchestrator or one of its MCP
                connections fails to initialize.
            KeyboardInterrupt: Propagated unchanged.
            SystemExit: Propagated unchanged.
        """
        if self.orchestrator is not None:
            return

        from mada.core.orchestrator import MADAOrchestrator

        orchestrator = None
        try:
            orchestrator = MADAOrchestrator(
                model_config=self.config.model,
                database_config=self.config.database,
                orchestration_config=_get_orchestration_config(self.config),
                bearer_token=self.bearer_token,
            )
            await orchestrator.__aenter__()
            await orchestrator.initialize_orchestrator(
                self.config.agents,
                self.config.mcp_servers,
                getattr(self.config, "a2a_agents", {}),
            )
            self.orchestrator = orchestrator
        except BaseException as exc:
            if orchestrator is not None:
                await orchestrator.__aexit__(None, None, None)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise OrchestratorStartupError(_format_startup_error_message(exc)) from exc

    async def ensure_started(self) -> None:
        """
        Initialize the orchestrator once, on demand.

        Concurrent requests are synchronized so that only one request performs
        startup and the remaining requests observe the initialized orchestrator.

        Returns:
            `None`.

        Raises:
            OrchestratorStartupError: If startup fails.
        """
        if self.orchestrator is not None:
            return

        async with self._startup_lock:
            if self.orchestrator is None:
                await self.startup()

    async def shutdown(self) -> None:
        """
        Release all orchestrator resources.

        Returns:
            `None`.
        """
        if self.orchestrator is None:
            return
        await self.orchestrator.__aexit__(None, None, None)
        self.orchestrator = None

    def validate_api_key(
        self, authorization: Optional[str], x_api_key: Optional[str]
    ) -> None:
        """
        Validate the caller-provided API key when one is configured.

        Open WebUI can send either a Bearer token or `x-api-key`.

        Args:
            authorization: Optional `Authorization` header value from the
                incoming request.
            x_api_key: Optional `x-api-key` header value from the incoming
                request.

        Returns:
            `None`.

        Raises:
            HTTPException: If an API key is configured and the request does not
                provide a matching key.
        """
        if not self.api_key:
            return

        provided_key = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()

        if not secrets.compare_digest(provided_key or "", self.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def collect_response(self, messages: list[dict[str, Any]]) -> str:
        """
        Collect the full assistant response for non-streaming requests.

        Args:
            messages: OpenAI-style chat messages from the HTTP request body.

        Returns:
            The full assistant response text assembled from streamed chunks.

        Raises:
            RuntimeError: If the orchestrator has not been initialized yet.
        """
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")

        chunks = []
        async for chunk in self.orchestrator.process_openai_messages(messages):
            chunks.append(chunk)
        return "".join(chunks)

    async def stream_response(
        self, messages: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Yield OpenAI-compatible SSE events for a streaming response.

        Args:
            messages: OpenAI-style chat messages from the HTTP request body.

        Yields:
            Server-sent event payload strings formatted for the OpenAI streaming
            chat completions protocol.

        Raises:
            RuntimeError: If the orchestrator has not been initialized yet.
        """
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        initial = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(initial)}\n\n"

        async for chunk in self.orchestrator.process_openai_messages(messages):
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"


def create_openai_api_app(
    config: AppConfig,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> FastAPI:
    """
    Build and return a FastAPI app backed by the configured MADA orchestrator.

    Args:
        config: Parsed MADA application configuration.
        model_name: Model identifier to expose through the OpenAI-compatible
            model listing endpoints.
        api_key: Optional API key required for incoming requests.
        bearer_token: Optional bearer token forwarded to streamable HTTP MCP
            servers during orchestrator startup.

    Returns:
        A configured FastAPI application exposing health, model listing, and
        chat completion routes.

    Raises:
        RuntimeError: If FastAPI or Uvicorn dependencies are unavailable.
    """
    _require_fastapi()
    service = MADAOpenAIAPIService(
        config=config,
        model_name=model_name,
        api_key=api_key,
        bearer_token=bearer_token,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mada_service = service
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(title="MADA OpenAI API", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {
            "status": "ok",
            "orchestrator_initialized": "true"
            if service.orchestrator is not None
            else "false",
        }

    async def get_models(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        service.validate_api_key(authorization, x_api_key)
        return {
            "object": "list",
            "data": [{"id": service.model_name, "object": "model", "owned_by": "mada"}],
        }

    app.get("/models")(get_models)
    app.get("/v1/models")(get_models)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
    ):
        service.validate_api_key(authorization, x_api_key)
        try:
            await service.ensure_started()
        except OrchestratorStartupError as exc:
            configured_servers = (
                ", ".join((service.config.mcp_servers or {}).keys()) or "none"
            )
            print(
                "No MCP servers connected; returning 503 for /v1/chat/completions. "
                f"Configured MCP servers: {configured_servers}",
                file=sys.stderr,
                flush=True,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        requested_model = body.get("model")
        if requested_model and requested_model != service.model_name:
            raise HTTPException(
                status_code=404, detail=f"Unknown model '{requested_model}'"
            )

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400, detail="'messages' must be a non-empty list"
            )

        if body.get("stream", False):
            return StreamingResponse(
                service.stream_response(messages), media_type="text/event-stream"
            )

        content = await service.collect_response(messages)
        created = int(time.time())
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": service.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(response)

    return app


def run_openai_api(
    config: AppConfig,
    host: str,
    port: int,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> None:
    """
    Launch the OpenAI-compatible FastAPI server.

    Args:
        config: Parsed MADA application configuration.
        host: Host interface to bind the HTTP server to.
        port: TCP port to bind the HTTP server to.
        model_name: Model identifier to expose through model listing routes.
        api_key: Optional API key required for incoming requests.
        bearer_token: Optional bearer token forwarded to streamable HTTP MCP
            servers during orchestrator startup.

    Returns:
        `None`.
    """
    _require_fastapi()
    app = create_openai_api_app(
        config=config,
        model_name=model_name,
        api_key=api_key,
        bearer_token=bearer_token,
    )
    uvicorn.run(app, host=host, port=port)


def openai_api_entrypoint(
    host: str,
    port: int,
    model_name: str,
    api_key: Optional[str],
    bearer_token: Optional[str],
    config_file: str,
) -> None:
    """
    Load config and start the OpenAI-compatible API server.

    Args:
        host: Host interface to bind the HTTP server to.
        port: TCP port to bind the HTTP server to.
        model_name: Model identifier to expose through model listing routes.
        api_key: Optional API key required for incoming requests.
        bearer_token: Optional bearer token forwarded to streamable HTTP MCP
            servers during orchestrator startup.
        config_file: Path to the JSON configuration file to load.

    Returns:
        `None`.
    """
    try:
        print(f"Loading configuration from {config_file}")
        config = load_config_from_json(config_file)
        setup_telemetry(disabled=config.telemetry.disabled)
        print(f"Serving OpenAI-compatible API on http://{host}:{port}/v1")
        run_openai_api(
            config=config,
            host=host,
            port=port,
            model_name=model_name,
            api_key=api_key,
            bearer_token=bearer_token,
        )
    except Exception as e:
        print(f"Error launching OpenAI API interface: {e}")
        sys.exit(1)


@click.command(
    name="mada-openai-api",
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    show_default=True,
    help="Host interface to bind.",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="Port for the OpenAI-compatible API.",
)
@click.option(
    "--model-name",
    type=str,
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Model identifier exposed by /v1/models.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="Optional API key that incoming requests must provide.",
)
@click.option(
    "--bearer-token",
    type=str,
    default=None,
    help="Optional bearer token forwarded to streamable HTTP MCP servers as X-Token.",
)
@click.argument("config_file", type=str)
def main(
    host: str,
    port: int,
    model_name: str,
    api_key: Optional[str],
    bearer_token: Optional[str],
    config_file: str,
) -> None:
    """
    Run MADA Orchestrator as an OpenAI-compatible API.

    Args:
        host: Host interface to bind the HTTP server to.
        port: TCP port to bind the HTTP server to.
        model_name: Model identifier to expose through the OpenAI-compatible
            model listing endpoints.
        api_key: Optional API key required for incoming requests.
        bearer_token: Optional bearer token forwarded to streamable HTTP MCP
            servers during orchestrator startup.
        config_file: Path to the MADA Orchestrator configuration file.

    Returns:
        `None`.
    """
    openai_api_entrypoint(host, port, model_name, api_key, bearer_token, config_file)


if __name__ == "__main__":
    main()
