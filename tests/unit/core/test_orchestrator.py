# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mada.core.config import AgentConfig, MCPServerConfig, OpenAIModelConfig
from mada.core.coordinator import MCPAgentManager
from mada.core.orchestrator import MADAOrchestrator


@pytest.mark.asyncio
async def test_create_chat_agent_passes_agent_extra_to_as_agent(monkeypatch):
    captured = {}
    created_agent = object()

    class DummyClient:
        def as_agent(self, **kwargs):
            captured.update(kwargs)
            return created_agent

    monkeypatch.setattr(
        "mada.core.coordinator.chat_client_factory.create",
        lambda _: DummyClient(),
    )

    manager = MCPAgentManager(
        model_config=OpenAIModelConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        )
    )

    agent = await manager.create_chat_agent(
        AgentConfig(
            agent_name="TestAgent",
            description="Test agent",
            instructions="You are a test agent.",
            mcp_servers=[],
            extra={"default_options": {"store": False}},
        ),
        tools=["test-tool"],
    )

    assert agent is created_agent
    assert captured["name"] == "TestAgent"
    assert captured["instructions"] == "You are a test agent."
    assert captured["tools"] == ["test-tool"]
    assert captured["default_options"] == {"store": False}


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
