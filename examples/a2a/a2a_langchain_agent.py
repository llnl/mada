# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simple LangChain-backed A2A agent for MADA."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any

from a2a_example_utils import (
    DEFAULT_CONFIG_PATH,
    create_a2a_example_app,
    load_model_settings,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers import new_text_message
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from starlette.applications import Starlette


DEFAULT_MCP_URL = "http://localhost:9101/mcp"
DEFAULT_AGENT_CARD_PATH = (
    Path(__file__).parent / "agent_cards" / "langchain_agent_card.json"
)


class LangChainA2AAgent:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        mcp_url: str = DEFAULT_MCP_URL,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.mcp_url = mcp_url
        self._llm = None
        self._tools = None

    @property
    def llm(self):
        if self._llm is None:
            kwargs = {"model": self.model}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._llm = ChatOpenAI(**kwargs)
        return self._llm

    async def run(self, task: str) -> str:
        tools = await self._get_tools()
        tools_by_name = {tool.name: tool for tool in tools}
        messages = [
            (
                "system",
                "You are a concise remote specialist called by MADA. "
                "Complete the delegated task and return only the useful result. "
                "Use your available MCP tools when they are relevant.",
            ),
            ("human", task),
        ]

        response = await self.llm.bind_tools(tools).ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", []) or []
        if not tool_calls:
            return str(getattr(response, "content", response))

        messages.append(response)
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.get("name"))
            if tool is None:
                continue
            result = await tool.ainvoke(tool_call.get("args") or {})
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call.get("id") or f"tool-{uuid.uuid4().hex}",
                )
            )

        response = await self.llm.ainvoke(messages)
        return str(getattr(response, "content", response))

    async def _get_tools(self) -> list[Any]:
        if self._tools is None:
            client = MultiServerMCPClient(
                {
                    "example": {
                        "transport": "streamable_http",
                        "url": self.mcp_url,
                    }
                }
            )
            self._tools = await client.get_tools()
        return self._tools


class LangChainA2AExecutor(AgentExecutor):
    """
    A2A SDK executor that delegates requests to the LangChain example agent.
    """

    def __init__(self, agent: LangChainA2AAgent) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.get_user_input().strip()
        if not task:
            raise ValueError("A2A request must include a text message part")
        text = await self.agent.run(task)
        await event_queue.enqueue_event(new_text_message(text))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise RuntimeError("A2A task cancellation is not supported")


def create_app(agent: LangChainA2AAgent, public_url: str) -> Starlette:
    return create_a2a_example_app(
        LangChainA2AExecutor(agent),
        DEFAULT_AGENT_CARD_PATH,
        public_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple LangChain A2A agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind")
    parser.add_argument(
        "--config",
        default=os.getenv("MADA_CONFIG") or str(DEFAULT_CONFIG_PATH),
        help="MADA config JSON to read default model settings from.",
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
        help=(
            "OpenAI-compatible base URL override. Defaults to the MADA config base_url."
        ),
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("A2A_TABLE_MCP_URL") or DEFAULT_MCP_URL,
        help="Table-reader MCP server URL.",
    )
    parser.add_argument("--public-url", default=None)
    args = parser.parse_args()

    import uvicorn

    model_settings = load_model_settings(args.config)
    model = args.model or model_settings.get("model")
    api_key = args.api_key or model_settings.get("api_key")
    base_url = args.base_url or model_settings.get("base_url")
    if not model or not api_key or not base_url:
        raise RuntimeError(
            "LangChain A2A example requires model, api_key, and base_url from "
            "the MADA config or explicit --model/--api-key/--base-url overrides."
        )

    public_url = args.public_url or f"http://localhost:{args.port}"
    app = create_app(
        LangChainA2AAgent(model, api_key, base_url, args.mcp_url),
        public_url,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
