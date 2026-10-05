# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from mada.core.config import AgentConfig
from mada.core.orchestration.agent_as_tool_strategy import (
    AgentAsToolOrchestrationStrategy,
)


@pytest.mark.asyncio
async def test_missing_named_mcp_definitions_fail_participant_initialization():
    strategy = AgentAsToolOrchestrationStrategy()
    orchestrator = SimpleNamespace(
        mcp_servers={},
        specialist_agents=[],
    )

    all_tools, failed_servers, failed_agents = await strategy._initialize_participants(
        orchestrator,
        [
            AgentConfig(
                agent_name="ToolAgent",
                description="Uses a named MCP server",
                instructions="Use tools.",
                mcp_servers=["missing_server"],
            )
        ],
    )

    assert all_tools == []
    assert failed_servers == []
    assert failed_agents == ["ToolAgent"]
    assert orchestrator.specialist_agents == []


@pytest.mark.asyncio
async def test_missing_named_mcp_definitions_preserve_legacy_server_path_fallback():
    strategy = AgentAsToolOrchestrationStrategy()
    orchestrator = SimpleNamespace(
        mcp_servers={},
        specialist_agents=[],
    )
    legacy_calls = []

    async def connect_legacy_agent(orchestrator, config, all_tools, failed_agents):
        legacy_calls.append(config.agent_name)
        all_tools.append(f"{config.agent_name}: {config.server_path}")

    strategy._connect_legacy_agent = connect_legacy_agent

    all_tools, failed_servers, failed_agents = await strategy._initialize_participants(
        orchestrator,
        [
            AgentConfig(
                agent_name="LegacyToolAgent",
                description="Uses a legacy MCP server path",
                instructions="Use tools.",
                mcp_servers=["missing_server"],
                server_path="/tmp/legacy_server.py",
            )
        ],
    )

    assert legacy_calls == ["LegacyToolAgent"]
    assert all_tools == ["LegacyToolAgent: /tmp/legacy_server.py"]
    assert failed_servers == []
    assert failed_agents == []


@pytest.mark.asyncio
async def test_isolated_agent_as_tool_turn_uses_persistence_session_history():
    strategy = AgentAsToolOrchestrationStrategy()
    calls = {}

    class PlanningAgent:
        def run(self, prompt, *, session, stream):
            calls["prompt"] = prompt

            async def responses():
                yield SimpleNamespace(text="done")

            return responses()

    async def create_run_session(
        isolated_session, primary_session_id=None, context_session_ids=None
    ):
        assert isolated_session is True
        calls["primary_session_id"] = primary_session_id
        calls["context_session_ids"] = context_session_ids
        return None, object(), {}, False

    async def build_persisted_context_transcript(
        latest_user_message=None,
        primary_session_id=None,
        context_session_ids=None,
    ):
        calls["latest_user_message"] = latest_user_message
        calls["history_session_id"] = primary_session_id
        calls["context_session_ids"] = context_session_ids
        return [
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": latest_user_message},
        ]

    async def persist_isolated_response(*args, **kwargs):
        calls["persist"] = (args, kwargs)

    def build_prompt_from_transcript(messages):
        calls["transcript"] = messages
        return "rebuilt prompt"

    orchestrator = SimpleNamespace(
        planning_agent=PlanningAgent(),
        _create_run_session=create_run_session,
        build_persisted_context_transcript=build_persisted_context_transcript,
        build_prompt_from_transcript=build_prompt_from_transcript,
        _persist_isolated_response=persist_isolated_response,
    )

    chunks = [
        chunk
        async for chunk in strategy.process_message(
            orchestrator,
            "follow up",
            isolated_session=True,
            persistence_session_id="chat-1",
        )
    ]

    assert chunks == ["done"]
    assert calls["history_session_id"] == "chat-1"
    assert calls["latest_user_message"] == "follow up"
    assert calls["transcript"] == [
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "follow up"},
    ]
    assert calls["prompt"] == "rebuilt prompt"


@pytest.mark.asyncio
async def test_shared_agent_as_tool_turn_rebuilds_prompt_when_context_changes():
    strategy = AgentAsToolOrchestrationStrategy()
    calls = {}

    class PlanningAgent:
        def run(self, prompt, *, session, stream):
            calls["prompt"] = prompt

            async def responses():
                yield SimpleNamespace(text="done")

            return responses()

    async def create_run_session(
        isolated_session, primary_session_id=None, context_session_ids=None
    ):
        assert isolated_session is False
        calls["primary_session_id"] = primary_session_id
        calls["context_session_ids"] = context_session_ids
        return 1, object(), {}, True

    async def build_persisted_context_transcript(
        latest_user_message=None,
        primary_session_id=None,
        context_session_ids=None,
    ):
        calls["latest_user_message"] = latest_user_message
        return [
            {"role": "assistant", "content": "from context"},
            {"role": "user", "content": latest_user_message},
        ]

    async def commit_completed_turn(*args, **kwargs):
        calls["commit"] = (args, kwargs)

    orchestrator = SimpleNamespace(
        planning_agent=PlanningAgent(),
        _create_run_session=create_run_session,
        build_persisted_context_transcript=build_persisted_context_transcript,
        build_prompt_from_transcript=lambda messages: f"rebuilt::{messages[-1]['content']}",
        _commit_completed_turn=commit_completed_turn,
        _process_message_error=lambda error: f"error: {error}",
    )

    chunks = [
        chunk
        async for chunk in strategy.process_message(
            orchestrator,
            "follow up",
            context_session_ids=["context-a", "context-b"],
        )
    ]

    assert chunks == ["done"]
    assert calls["primary_session_id"] is None
    assert calls["context_session_ids"] == ["context-a", "context-b"]
    assert calls["latest_user_message"] == "follow up"
    assert calls["prompt"] == "rebuilt::follow up"
