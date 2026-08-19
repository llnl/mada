# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simple Google ADK-backed A2A agent for MADA."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from a2a_example_utils import (
    DEFAULT_CONFIG_PATH,
    create_a2a_example_app,
    load_model_settings,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers import new_text_message
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.genai import types
from starlette.applications import Starlette


DEFAULT_MCP_URL = "http://localhost:9102/mcp"
DEFAULT_AGENT_CARD_PATH = (
    Path(__file__).parent / "agent_cards" / "google_adk_agent_card.json"
)
APP_NAME = "mada_google_adk_a2a_agent"


class GoogleADKA2AAgent:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        mcp_url: str = DEFAULT_MCP_URL,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.mcp_url = mcp_url
        self._session_service = None
        self._runner = None

    @property
    def runner(self):
        if self._runner is None:
            agent = Agent(
                name="GoogleADKAgent",
                model=self._build_adk_model(),
                description="Simple Google ADK remote agent callable from MADA.",
                instruction=(
                    "You are a concise remote specialist called by MADA. "
                    "Complete the delegated task and return only the useful result. "
                    "Use your available MCP tools when they are relevant."
                ),
                tools=[
                    McpToolset(
                        connection_params=StreamableHTTPConnectionParams(
                            url=self.mcp_url,
                        )
                    )
                ],
            )
            self._session_service = InMemorySessionService()
            self._runner = Runner(
                agent=agent,
                app_name=APP_NAME,
                session_service=self._session_service,
            )
        return self._runner

    def _build_adk_model(self):
        provider = self.provider.lower()
        if provider in {"openai", "livai"}:
            if self.api_key:
                os.environ["OPENAI_API_KEY"] = self.api_key
            if self.base_url:
                os.environ["OPENAI_API_BASE"] = self.base_url
                os.environ["OPENAI_BASE_URL"] = self.base_url
            return LiteLlm(model=f"openai/{self.model}")

        return self.model

    async def run(self, task: str) -> str:
        runner = self.runner
        user_id = f"mada-user-{uuid.uuid4().hex}"
        session_id = f"mada-session-{uuid.uuid4().hex}"
        await self._session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=task)],
        )

        final_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if not event.is_final_response():
                continue
            if event.content and event.content.parts:
                final_text = "\n".join(
                    part.text
                    for part in event.content.parts
                    if getattr(part, "text", None)
                )

        return final_text


class GoogleADKA2AExecutor(AgentExecutor):
    """
    A2A SDK executor that delegates requests to the Google ADK example agent.
    """

    def __init__(self, agent: GoogleADKA2AAgent) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.get_user_input().strip()
        if not task:
            raise ValueError("A2A request must include a text message part")
        text = await self.agent.run(task)
        await event_queue.enqueue_event(new_text_message(text))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise RuntimeError("A2A task cancellation is not supported")


def create_app(agent: GoogleADKA2AAgent, public_url: str) -> Starlette:
    return create_a2a_example_app(
        GoogleADKA2AExecutor(agent),
        DEFAULT_AGENT_CARD_PATH,
        public_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple Google ADK A2A agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind")
    parser.add_argument(
        "--config",
        default=os.getenv("MADA_CONFIG") or str(DEFAULT_CONFIG_PATH),
        help="MADA config JSON to read default model settings from.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider override. Defaults to the MADA config provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override. Defaults to the MADA config model.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. Defaults to the MADA config api_key.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL override for OpenAI-compatible ADK models.",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("A2A_AVERAGE_MCP_URL") or DEFAULT_MCP_URL,
        help="Column-average MCP server URL.",
    )
    parser.add_argument("--public-url", default=None)
    args = parser.parse_args()

    import uvicorn

    model_settings = load_model_settings(args.config)
    provider = args.provider or model_settings.get("provider")
    model = args.model or model_settings.get("model")
    api_key = args.api_key or model_settings.get("api_key")
    base_url = args.base_url or model_settings.get("base_url")
    if not provider or not model:
        raise RuntimeError(
            "Google ADK A2A example requires provider and model from the MADA "
            "config or explicit --provider/--model overrides."
        )
    if provider.lower() in {"openai", "livai"} and (not api_key or not base_url):
        raise RuntimeError(
            "OpenAI-compatible ADK model providers require api_key and base_url "
            "from the MADA config or explicit --api-key/--base-url overrides."
        )

    public_url = args.public_url or f"http://localhost:{args.port}"
    app = create_app(
        GoogleADKA2AAgent(provider, model, api_key, base_url, args.mcp_url),
        public_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
