# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared configuration helpers for the example A2A agents."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from google.protobuf.json_format import ParseDict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "example_a2a_agents.json"
)


def expand_env_vars(value: str | None) -> str | None:
    """
    Expand `${VAR}` and `${VAR:-default}` placeholders in config values.
    """
    if value is None:
        return value

    def replace_env_var(match):
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default_value = var_expr.split(":-", 1)
            return os.getenv(var_name, default_value)
        return os.getenv(var_expr, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace_env_var, value)


def load_model_settings(config_path: str | None) -> dict[str, str]:
    """
    Load model settings from a MADA config JSON file.
    """
    if not config_path:
        return {}

    path = Path(config_path).expanduser()
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        return {}

    settings = {}
    for key in ("provider", "model", "api_key", "base_url"):
        value = model_config.get(key)
        if isinstance(value, str):
            settings[key] = expand_env_vars(value) or ""

    api_key = settings.get("api_key")
    if api_key and Path(api_key).expanduser().exists():
        settings["api_key"] = Path(api_key).expanduser().read_text().strip()

    return settings


def create_a2a_example_app(
    agent_executor: AgentExecutor,
    agent_card_path: Path,
    public_url: str,
) -> Starlette:
    """
    Build the common Starlette A2A app used by the example agents.
    """

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    card = json.loads(agent_card_path.read_text(encoding="utf-8"))
    card["supportedInterfaces"] = [
        {
            "url": public_url,
            "protocolBinding": TransportProtocol.JSONRPC.value,
            "protocolVersion": PROTOCOL_VERSION_1_0,
        }
    ]
    public_agent_card = ParseDict(card, AgentCard())
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
        agent_card=public_agent_card,
    )

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            *create_agent_card_routes(public_agent_card),
            *create_jsonrpc_routes(request_handler, "/"),
            *create_jsonrpc_routes(request_handler, "/a2a"),
        ]
    )
