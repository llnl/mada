# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Agent-to-Agent HTTP interface for MADA Orchestrator.

This module exposes the configured MADA planning agent as an A2A-compatible
service. The MADA agent card is available under the standard
`/.well-known/agent-card.json` path.

This is the server-side A2A entry point: use it when another A2A client or
agent needs to discover MADA and send work to MADA. The client-side support
for MADA calling other A2A agents lives in `mada.core.a2a_client` and is wired
through the `a2a.agents` configuration block.
"""

from __future__ import annotations

import asyncio
import ipaddress
import inspect
import json
import re
import secrets
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import click
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.helpers import new_text_message
from a2a.types import AgentCard
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from google.protobuf.json_format import ParseDict
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mada.core import load_config_from_json
from mada.core.config import A2AConfig, AppConfig, OrchestrationConfig
from mada.core.orchestration.stream_events import apply_text_control
from mada.interfaces.startup_errors import format_startup_error_message
from mada.core.telemetry import setup_telemetry

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator


class A2AStartupError(RuntimeError):
    """Raised when the orchestrator cannot be initialized for A2A requests."""


class A2AAuthMiddleware:
    """
    Validate API keys for state-changing A2A requests.
    """

    def __init__(self, app: Any, service: "MADAA2AService") -> None:
        self.app = app
        self.service = service

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("method", "").upper() != "GET"
            and scope.get("path") in {"/", "/a2a"}
        ):
            headers = Headers(scope=scope)
            try:
                self.service.validate_api_key(
                    headers.get("authorization"),
                    headers.get("x-api-key"),
                )
            except HTTPException as exc:
                response = JSONResponse(
                    {"detail": exc.detail},
                    status_code=exc.status_code,
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


class MADAA2AService:
    """
    Manage the shared orchestrator instance used by the A2A API.
    """

    def __init__(
        self,
        config: AppConfig,
        public_url: str,
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ) -> None:
        """
        Initialize the service wrapper for one shared orchestrator instance.
        """
        self.config = config
        self.a2a_config = getattr(config, "a2a", None) or A2AConfig()
        self.public_url = self.a2a_config.url or public_url
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.orchestrator: Optional[MADAOrchestrator] = None
        self._startup_lock = asyncio.Lock()

    async def startup(self) -> None:
        """
        Start and initialize the orchestrator used to serve A2A requests.
        """
        if self.orchestrator is not None:
            return

        from mada.core.orchestrator import MADAOrchestrator

        orchestrator = None
        try:
            orchestrator = MADAOrchestrator(
                model_config=self.config.model,
                database_config=self.config.database,
                orchestration_config=getattr(self.config, "orchestration", None)
                or OrchestrationConfig(),
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
            raise A2AStartupError(format_startup_error_message(exc)) from exc

    async def ensure_started(self) -> None:
        """
        Lazily start the orchestrator once across concurrent requests.
        """
        if self.orchestrator is not None:
            return

        async with self._startup_lock:
            if self.orchestrator is None:
                await self.startup()

    async def shutdown(self) -> None:
        """
        Shut down the shared orchestrator if it has been initialized.
        """
        if self.orchestrator is None:
            return
        await self.orchestrator.__aexit__(None, None, None)
        self.orchestrator = None

    def validate_api_key(
        self, authorization: Optional[str], x_api_key: Optional[str]
    ) -> None:
        """
        Validate the configured API key against request headers.
        """
        if not self.api_key:
            return

        provided_key = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()

        if not secrets.compare_digest(provided_key or "", self.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def build_agent_card(self) -> Dict[str, Any]:
        """
        Build the public A2A agent card for this MADA service.
        """
        if self.a2a_config.card_path:
            card = self._load_agent_card_file()
            card["supportedInterfaces"] = [
                {
                    "url": self.public_url,
                    "protocolBinding": TransportProtocol.JSONRPC.value,
                    "protocolVersion": PROTOCOL_VERSION_1_0,
                }
            ]
            card.setdefault("capabilities", {"streaming": True})
            card.setdefault("defaultInputModes", ["text/plain"])
            card.setdefault("defaultOutputModes", ["text/plain"])
            return card

        return {
            "name": self.a2a_config.name,
            "description": self.a2a_config.description,
            "version": self.a2a_config.version,
            "supportedInterfaces": [
                {
                    "url": self.public_url,
                    "protocolBinding": TransportProtocol.JSONRPC.value,
                    "protocolVersion": PROTOCOL_VERSION_1_0,
                }
            ],
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": self._build_skills(),
        }

    def _load_agent_card_file(self) -> Dict[str, Any]:
        """
        Load and validate a standalone A2A agent card JSON file.
        """
        card_path = Path(self.a2a_config.card_path)
        try:
            with card_path.open("r", encoding="utf-8") as card_file:
                card = json.load(card_file)
        except OSError as exc:
            raise RuntimeError(f"Could not read A2A agent card: {card_path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"A2A agent card is not valid JSON: {card_path}"
            ) from exc

        if not isinstance(card, dict):
            raise RuntimeError(f"A2A agent card must be a JSON object: {card_path}")
        return card

    def _build_skills(self) -> list[dict[str, Any]]:
        """
        Build A2A skill entries from configuration or configured MADA agents.
        """
        if self.a2a_config.skills:
            return self.a2a_config.skills

        skills = []
        for agent in self.config.agents:
            if getattr(agent, "agent_name", "") == "PlanningAgent":
                continue
            name = getattr(agent, "agent_name", "") or "MADA Agent"
            description = getattr(agent, "description", "") or name
            skill_id = (
                re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
                or "mada-agent"
            )
            skills.append(
                {
                    "id": skill_id,
                    "name": name,
                    "description": description,
                    "tags": [getattr(agent, "domain", "") or "mada"],
                }
            )

        if skills:
            return skills

        return [
            {
                "id": "mada-orchestration",
                "name": "MADA orchestration",
                "description": self.a2a_config.description,
                "tags": ["mada"],
            }
        ]

    async def collect_response(self, message: str) -> str:
        """
        Collect a complete orchestrator response for a single A2A message.

        Uses isolated sessions to avoid interference with the main orchestrator
        session, but does not persist conversation history across A2A requests
        by design (A2A protocol is stateless per-request).
        """
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")
        return await self.orchestrator.collect_message_response(
            message,
            isolated_session=True,
            stateless_session=True,
        )

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        """
        Yield the authoritative orchestrator response for a single A2A message.

        Magentic can emit replacement/error control chunks after provisional
        text. A2A string streams have no retract operation, so Magentic output
        is buffered and yielded only after its authoritative content is known.
        Other orchestration modes stream chunks immediately.

        Uses isolated sessions to avoid interference with the main orchestrator
        session, but does not persist conversation history across A2A requests
        by design (A2A protocol is stateless per-request).
        """
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")

        orchestration_config = (
            getattr(self.config, "orchestration", None) or OrchestrationConfig()
        )
        is_magentic = orchestration_config.mode == "magentic"
        chunks: list[str] = []
        async for chunk in self.orchestrator.process_message(
            message,
            isolated_session=True,
            stateless_session=True,
        ):
            if not is_magentic:
                content = str(chunk)
                if content:
                    yield content
                continue

            handled, terminal = apply_text_control(chunks, chunk)
            if handled:
                if terminal:
                    break
                continue

            content = str(chunk)
            if content:
                chunks.append(content)

        content = "".join(chunks)
        if content:
            yield content


def _extract_message_text(value: Any) -> str:
    """
    Extract text content from supported A2A message-like shapes.
    """
    if isinstance(value, str):
        return value

    get_user_input = getattr(value, "get_user_input", None)
    if callable(get_user_input):
        text = get_user_input()
        return str(text or "")

    if not isinstance(value, dict):
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            value = model_dump()
        else:
            return str(getattr(value, "text", "") or "")

    message = value.get("message", value)
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""

    parts = message.get("parts")
    if not isinstance(parts, list):
        return str(message.get("text", "") or "")

    text_parts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "text" or part.get("type") == "text":
            text = part.get("text")
            if text:
                text_parts.append(str(text))

    return "\n".join(text_parts)


async def _enqueue_event(event_queue: Any, event: Any) -> None:
    """
    Enqueue an SDK event across minor EventQueue API differences.
    """
    enqueue = getattr(event_queue, "enqueue_event", None)
    if callable(enqueue):
        result = enqueue(event)
        if inspect.isawaitable(result):
            await result
        return

    put = getattr(event_queue, "put", None)
    if callable(put):
        result = put(event)
        if inspect.isawaitable(result):
            await result
        return

    raise RuntimeError("Unsupported A2A event queue implementation")


class MADAA2AExecutor(AgentExecutor):
    """
    A2A SDK executor adapter for the shared MADA orchestrator service.
    """

    def __init__(self, service: MADAA2AService) -> None:
        self.service = service

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute one A2A request through the MADA orchestrator.
        """
        try:
            await self.service.ensure_started()
        except A2AStartupError as exc:
            configured_servers = (
                ", ".join((self.service.config.mcp_servers or {}).keys()) or "none"
            )
            print(
                "No MCP servers connected; returning A2A startup error. "
                f"Configured MCP servers: {configured_servers}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(str(exc)) from exc

        message_text = _extract_message_text(context).strip()
        if not message_text:
            raise ValueError("A2A request must include a text message part")

        content = await self.service.collect_response(message_text)
        await _enqueue_event(event_queue, new_text_message(content))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Report that MADA does not support cancelling in-flight orchestrator work.
        """
        raise RuntimeError("A2A task cancellation is not supported")


def create_a2a_app(
    config: AppConfig,
    public_url: str,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> Starlette:
    """
    Build and return an A2A SDK app backed by the configured MADA orchestrator.
    """
    service = MADAA2AService(
        config=config,
        public_url=public_url,
        api_key=api_key,
        bearer_token=bearer_token,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette):
        """
        Attach the A2A service to app state and clean it up on shutdown.
        """
        app.state.mada_a2a_service = service
        try:
            yield
        finally:
            await service.shutdown()

    async def health(request: Request) -> JSONResponse:
        """
        Report whether the A2A process is running and initialized.
        """
        return JSONResponse(
            {
                "status": "ok",
                "orchestrator_initialized": "true"
                if service.orchestrator is not None
                else "false",
            }
        )

    public_agent_card = ParseDict(service.build_agent_card(), AgentCard())
    request_handler = DefaultRequestHandler(
        agent_executor=MADAA2AExecutor(service),
        task_store=InMemoryTaskStore(),
        agent_card=public_agent_card,
    )

    routes = [
        Route("/health", health, methods=["GET"]),
        *create_agent_card_routes(public_agent_card),
        *create_jsonrpc_routes(request_handler, "/"),
        *create_jsonrpc_routes(request_handler, "/a2a"),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(A2AAuthMiddleware, service=service)

    return app


def run_a2a(
    config: AppConfig,
    host: str,
    port: int,
    public_url: Optional[str] = None,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> None:
    """
    Launch the A2A server.
    """
    card_url = _resolve_public_a2a_url(host, port, public_url)
    app = create_a2a_app(
        config=config,
        public_url=card_url,
        api_key=api_key,
        bearer_token=bearer_token,
    )
    uvicorn.run(app, host=host, port=port)


def _resolve_public_a2a_url(
    host: str,
    port: int,
    public_url: Optional[str] = None,
) -> str:
    """
    Return the URL to advertise in the A2A agent card.
    """
    if public_url:
        return public_url
    advertised_host = _resolve_advertised_host(host)
    if ":" in advertised_host and not advertised_host.startswith("["):
        advertised_host = f"[{advertised_host}]"
    return f"http://{advertised_host}:{port}"


def _resolve_advertised_host(host: str) -> str:
    """
    Resolve wildcard bind hosts to a non-loopback local address for agent cards.

    Deployments behind NAT or a reverse proxy should use ``--public-url`` to
    advertise their externally routed URL.
    """
    if host not in {"0.0.0.0", "::"}:
        return host

    family = socket.AF_INET if host == "0.0.0.0" else socket.AF_INET6
    try:
        addresses = socket.getaddrinfo(
            socket.gethostname(),
            None,
            family=family,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        addresses = []

    for _, _, _, _, sockaddr in addresses:
        address = sockaddr[0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if not (
            parsed.is_loopback
            or parsed.is_unspecified
            or parsed.is_link_local
            or parsed.is_multicast
        ):
            return address

    return socket.getfqdn()


def a2a_entrypoint(
    host: str,
    port: int,
    public_url: Optional[str],
    api_key: Optional[str],
    bearer_token: Optional[str],
    config_file: str,
) -> None:
    """
    Load config and start the A2A API server.
    """
    try:
        print(f"Loading configuration from {config_file}")
        config = load_config_from_json(config_file)
        setup_telemetry(disabled=config.telemetry.disabled)
        card_url = _resolve_public_a2a_url(host, port, public_url)
        print(f"Serving A2A API on {card_url}")
        run_a2a(
            config=config,
            host=host,
            port=port,
            public_url=public_url,
            api_key=api_key,
            bearer_token=bearer_token,
        )
    except Exception as e:
        print(f"Error launching A2A interface: {e}")
        sys.exit(1)


@click.command(
    name="mada-a2a",
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
    help="Port for the A2A API.",
)
@click.option(
    "--public-url",
    type=str,
    default=None,
    help="Externally reachable URL to publish in the A2A agent card.",
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
    public_url: Optional[str],
    api_key: Optional[str],
    bearer_token: Optional[str],
    config_file: str,
) -> None:
    """
    Run MADA Orchestrator as an A2A agent.
    """
    a2a_entrypoint(host, port, public_url, api_key, bearer_token, config_file)


if __name__ == "__main__":
    main()
