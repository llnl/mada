# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mada.core.config import (
    AgentConfig,
    MCPServerConfig,
    OpenAIModelConfig,
    RemoteA2AAgentConfig,
)
from mada.core.orchestration.stream_events import InternalError
from mada.core.orchestrator import MADAOrchestrator


@pytest.mark.asyncio
async def test_connect_agent_passes_verify_to_httpx(monkeypatch):
    captured = {}

    class DummyAsyncClient:
        def __init__(self, *, headers, timeout, verify):
            captured["async_client_verify"] = verify

        async def aclose(self):
            return None

    class DummyMCPTool:
        def __init__(self, *, name, url, http_client):
            self.name = name
            self.url = url
            self.http_client = http_client

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def close(self):
            return None

    class DummyExitStack:
        async def enter_async_context(self, tool):
            return tool

    def fake_resolve_httpx_verify_value(*, verify=True):
        captured["resolve_verify_arg"] = verify
        return verify

    async def fake_create_chat_agent(agent_config, tools=None, **kwargs):
        return object()

    monkeypatch.setattr(
        "mada.core.coordinator.chat_client_factory.create",
        lambda _: object(),
    )
    monkeypatch.setattr("mada.core.orchestrator.httpx.AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        "mada.core.orchestrator.MCPStreamableHTTPTool",
        DummyMCPTool,
    )
    monkeypatch.setattr(
        "mada.core.orchestrator.resolve_httpx_verify_value",
        fake_resolve_httpx_verify_value,
    )

    orchestrator = MADAOrchestrator(
        model_config=OpenAIModelConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        session_manager=object(),
    )
    orchestrator.exit_stack = DummyExitStack()
    monkeypatch.setattr(orchestrator, "create_chat_agent", fake_create_chat_agent)

    await orchestrator.connect_agent(
        AgentConfig(
            agent_name="TestAgent",
            description="Test agent",
            instructions="You are a test agent.",
            mcp_servers=["test_server"],
        ),
        {
            "test_server": MCPServerConfig(
                transport="streamable-http",
                url="https://mcp.example.invalid/mcp",
                verify=False,
            )
        },
    )

    assert captured["resolve_verify_arg"] is False
    assert captured["async_client_verify"] is False


@pytest.mark.asyncio
async def test_load_remote_a2a_agent_cards_skips_unavailable_agents(monkeypatch):
    closed_clients = []

    class DummyRemoteA2AClient:
        def __init__(self, name, config):
            self.name = name
            self.config = config

        async def get_agent_card(self):
            if self.name == "bad":
                raise RuntimeError("offline")
            return {"description": "Ready"}

        async def aclose(self):
            closed_clients.append(self.name)

    monkeypatch.setattr(
        "mada.core.orchestrator.RemoteA2AClient",
        DummyRemoteA2AClient,
    )
    monkeypatch.setattr(
        "mada.core.coordinator.chat_client_factory.create",
        lambda _: object(),
    )

    orchestrator = MADAOrchestrator(
        model_config=OpenAIModelConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        session_manager=object(),
    )
    orchestrator.a2a_agents = {
        "good": RemoteA2AAgentConfig(url="https://good.example/a2a"),
        "bad": RemoteA2AAgentConfig(url="https://bad.example/a2a"),
    }

    failed_agents = await orchestrator._load_remote_a2a_agent_cards()

    assert orchestrator.a2a_agents == {
        "good": RemoteA2AAgentConfig(url="https://good.example/a2a")
    }
    assert orchestrator._a2a_agent_cards == {"good": {"description": "Ready"}}
    assert set(orchestrator._a2a_clients_by_agent) == {"good"}
    assert closed_clients == ["bad"]
    assert failed_agents == [
        {
            "agent": "bad",
            "url": "https://bad.example/a2a",
            "error": "offline",
        }
    ]


@pytest.mark.asyncio
async def test_collect_message_response_surfaces_internal_error(monkeypatch):
    orchestrator = MADAOrchestrator(
        model_config=OpenAIModelConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        session_manager=object(),
    )

    async def process_message(*args, **kwargs):
        yield "partial"
        yield InternalError("Error processing message: boom")

    monkeypatch.setattr(orchestrator, "process_message", process_message)

    response = await orchestrator.collect_message_response("hello")

    assert response == "Error processing message: boom"
