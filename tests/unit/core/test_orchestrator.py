# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from unittest.mock import MagicMock, call

import pytest

from mada.core.config import (
    AgentConfig,
    MCPServerConfig,
    OpenAIModelConfig,
    RemoteA2AAgentConfig,
)
from mada.core.coordinator import MCPAgentManager
from mada.core.media import ImageAttachment
from mada.core.orchestration.stream_events import InternalError
from mada.core.orchestrator import MADAOrchestrator


PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


def test_image_routing_uses_configured_specialist_name():
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    orchestrator.mcp_servers = {
        "local_files": MCPServerConfig(
            transport="stdio",
            command="mada-read-image-mcp",
            description="Reads local images",
        )
    }
    orchestrator._mcp_tools_by_server = {}

    renderer = AgentConfig(
        agent_name="Renderer",
        description="Displays local images",
        mcp_servers=["local_files"],
    )

    guidance = orchestrator._image_routing_guidance([renderer])

    assert "Renderer" in guidance
    assert "SimulationAgent" not in guidance


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
async def test_connect_agent_passes_verify_to_httpx2(monkeypatch):
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
    monkeypatch.setattr("mada.core.orchestrator.httpx2.AsyncClient", DummyAsyncClient)
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


def test_normalize_transcript_preserves_image_only_message():
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png", filename="plot.png")

    transcript = orchestrator._normalize_transcript_messages(
        [{"role": "assistant", "content": "", "attachments": [attachment]}]
    )

    assert transcript == [
        {"role": "assistant", "content": "[Image attachment: plot.png]"}
    ]


def test_normalize_transcript_appends_image_context_to_text_message():
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png", filename="plot.png")

    transcript = orchestrator._normalize_transcript_messages(
        [
            {
                "role": "assistant",
                "content": "Here is the plot.",
                "attachments": [attachment],
            }
        ]
    )

    assert transcript == [
        {
            "role": "assistant",
            "content": "Here is the plot.\n[Image attachment: plot.png]",
        }
    ]


def test_persist_completed_turn_stores_placeholder_for_image_only_response():
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    orchestrator.session_manager = MagicMock()
    orchestrator.background_tasks = MagicMock()
    orchestrator.background_tasks.user_message_already_started_background_task.return_value = False
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png", filename="plot.png")

    orchestrator._persist_completed_turn(
        {
            "message": "Create a plot",
            "assistant_reply": "",
            "image_attachments": [attachment],
        }
    )

    assert orchestrator.session_manager.add_message.call_args_list == [
        call("user", "Create a plot"),
        call(
            "assistant",
            "[Image attachment: plot.png]",
            attachments=[attachment],
        ),
    ]


@pytest.mark.asyncio
async def test_persist_isolated_response_stores_placeholder_for_image_only_response():
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    orchestrator.session_manager = MagicMock()
    orchestrator.background_tasks = MagicMock()
    orchestrator.background_tasks.user_message_already_started_background_task.return_value = False
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png", filename="plot.png")

    await orchestrator._persist_isolated_response(
        "Create a plot", "", image_attachments=[attachment]
    )

    assert orchestrator.session_manager.add_message.call_args_list == [
        call("user", "Create a plot"),
        call(
            "assistant",
            "[Image attachment: plot.png]",
            attachments=[attachment],
        ),
    ]
