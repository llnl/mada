# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from unittest.mock import MagicMock, patch

import pytest

from mada.core.config import AgentConfig, OpenAIModelConfig
from mada.core.coordinator import MCPAgentManager


def _agent_config() -> AgentConfig:
    """
    Build a minimal agent config for coordinator tests.

    Returns:
        An agent config with no MCP servers.
    """
    return AgentConfig.from_dict(
        {
            "agent_name": "WorkerAgent",
            "description": "Does work",
            "mcp_servers": [],
            "instructions": "Help with tests.",
        }
    )


def _manager(skill_tools=None) -> MCPAgentManager:
    """
    Build a manager with a mocked model client.

    Args:
        skill_tools: Optional runtime skill tools to register.

    Returns:
        A manager whose `model_client` is a MagicMock.
    """
    model_config = OpenAIModelConfig(
        provider="openai",
        model="gpt-test",
        api_key="test-key",
        base_url="https://llm.example/v1",
    )
    with patch.object(MCPAgentManager, "_setup_model_client", return_value=MagicMock()):
        return MCPAgentManager(model_config=model_config, skill_tools=skill_tools)


@pytest.mark.unit
class TestCreateChatAgent:
    @pytest.mark.asyncio
    async def test_skill_tools_are_passed_to_the_agent(self):
        """Test that configured skill tools reach the created agent."""
        skill_tool = MagicMock(name="load_skill")
        manager = _manager(skill_tools=[skill_tool])

        await manager.create_chat_agent(_agent_config())

        _, kwargs = manager.model_client.as_agent.call_args
        assert kwargs["tools"] == [skill_tool]

    @pytest.mark.asyncio
    async def test_mcp_tools_and_skill_tools_are_combined(self):
        """Test that MCP tools and skihll tools are both passed through."""
        mcp_tool = MagicMock(name="mcp_tool")
        skill_tool = MagicMock(name="load_skill")
        manager = _manager(skill_tools=[skill_tool])

        await manager.create_chat_agent(_agent_config(), tools=[mcp_tool])

        _, kwargs = manager.model_client.as_agent.call_args
        assert kwargs["tools"] == [mcp_tool, skill_tool]

    @pytest.mark.asyncio
    async def test_no_skill_tools_passes_only_mcp_tools(self):
        """Test that an empty skill tool list leaves MCP tools untouched."""
        mcp_tool = MagicMock(name="mcp_tool")
        manager = _manager()

        await manager.create_chat_agent(_agent_config(), tools=[mcp_tool])

        _, kwargs = manager.model_client.as_agent.call_args
        assert kwargs["tools"] == [mcp_tool]

    @pytest.mark.asyncio
    async def test_caller_tool_list_is_not_mutated(self):
        """Test that the caller's tool list is copied rather than extended."""
        mcp_tools = [MagicMock(name="mcp_tool")]
        manager = _manager(skill_tools=[MagicMock(name="load_skill")])

        await manager.create_chat_agent(_agent_config(), tools=mcp_tools)

        assert len(mcp_tools) == 1
