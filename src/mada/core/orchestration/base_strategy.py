# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base interface for orchestrator initialization strategies.
"""

import logging
import sys
import traceback
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Tuple

from agent_framework import MCPStdioTool

from mada.core.config import AgentConfig, MCPServerConfig, RemoteA2AAgentConfig

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator

try:
    BaseExceptionGroup
except NameError:
    BaseExceptionGroup = Exception  # fallback for type checkers/runtime


LOG = logging.getLogger(__name__)


class BaseOrchestrationStrategy(ABC):
    """
    Internal strategy boundary for orchestration modes.

    Concrete strategies encapsulate the setup and request-processing flow for a
    specific orchestration mode, including which agents participate, how tools
    are connected, which coordinator agent is created, and how requests stream
    responses.
    """

    mode: str = ""

    async def _initialize_participants(
        self,
        orchestrator: "MADAOrchestrator",
        participant_configs: List[AgentConfig],
    ) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
        """
        Initialize configured specialist agents for an orchestration run.
        """
        all_tools = []
        failed_servers = []
        failed_agents = []

        for config in participant_configs:
            if not config.agent_name:
                continue

            if config.mcp_servers and orchestrator.mcp_servers:
                await self._connect_configured_agent(
                    orchestrator,
                    config,
                    all_tools,
                    failed_servers,
                    failed_agents,
                )
                continue

            if config.server_path:
                await self._connect_legacy_agent(
                    orchestrator, config, all_tools, failed_agents
                )
                continue

            if config.mcp_servers:
                if not orchestrator.mcp_servers:
                    LOG.error(
                        "Agent %s references MCP servers but no named MCP server "
                        "definitions were loaded: %s",
                        config.agent_name,
                        ", ".join(config.mcp_servers),
                    )
                    failed_agents.append(config.agent_name)
                    continue

            if not config.mcp_servers:
                await self._connect_agent_without_tools(
                    orchestrator, config, failed_agents
                )
                continue

        return all_tools, failed_servers, failed_agents

    async def _connect_configured_agent(
        self,
        orchestrator: "MADAOrchestrator",
        config: AgentConfig,
        all_tools: List[str],
        failed_servers: List[Dict[str, str]],
        failed_agents: List[str],
    ) -> None:
        """
        Connect one agent through named MCP server definitions.
        """
        try:
            (
                agent,
                mcp_tools,
                tool_names,
                agent_failed_servers,
            ) = await orchestrator.connect_agent(config, orchestrator.mcp_servers)
            orchestrator.specialist_agents.append(agent)
            orchestrator._mcp_tool_count += len(mcp_tools)
            all_tools.extend([f"{config.agent_name}: {tool}" for tool in tool_names])

            for failed_server in agent_failed_servers or []:
                failed_servers.append(
                    {
                        "agent": config.agent_name,
                        "server": failed_server["name"],
                        "url": failed_server["url"],
                        "error": failed_server["error"],
                    }
                )

            if tool_names:
                LOG.info(
                    "Connected agent %s with %d MCP tools",
                    config.agent_name,
                    len(tool_names),
                )
                return

            LOG.warning(
                "Agent %s connected but no MCP servers available",
                config.agent_name,
            )
            if config.mcp_servers:
                failed_agents.append(config.agent_name)
        except BaseExceptionGroup as eg:
            LOG.error(
                "Multiple errors connecting agent %s (%d errors)",
                config.agent_name,
                len(eg.exceptions),
            )
            failed_agents.append(config.agent_name)
        except Exception as e:
            LOG.error("Failed to connect agent %s: %s", config.agent_name, e)
            traceback.print_exc()
            failed_agents.append(config.agent_name)

    async def _connect_legacy_agent(
        self,
        orchestrator: "MADAOrchestrator",
        config: AgentConfig,
        all_tools: List[str],
        failed_agents: List[str],
    ) -> None:
        """
        Connect one legacy agent directly from its configured `server_path`.
        """
        try:
            is_python = config.server_path.endswith(".py")
            command = sys.executable if is_python else "node"
            args = ["-u", config.server_path] if is_python else [config.server_path]
            mcp_tool = MCPStdioTool(
                name=f"{config.agent_name}_mcp",
                command=command,
                args=args,
            )

            mcp_tool = await orchestrator.exit_stack.enter_async_context(mcp_tool)
            orchestrator._agent_descriptions[config.agent_name] = config.description

            agent = await orchestrator.create_chat_agent(
                config,
                tools=[mcp_tool],
            )

            orchestrator.specialist_agents.append(agent)
            orchestrator._mcp_tool_count += 1
            all_tools.append(f"{config.agent_name}: {config.server_path}")
            LOG.info("Connected legacy agent %s with 1 MCP tool", config.agent_name)
        except Exception as e:
            LOG.error("Failed to connect legacy agent %s: %s", config.agent_name, e)
            failed_agents.append(config.agent_name)

    async def _connect_agent_without_tools(
        self,
        orchestrator: "MADAOrchestrator",
        config: AgentConfig,
        failed_agents: List[str],
    ) -> None:
        """
        Create an agent that relies only on the model client and no MCP tools.
        """
        LOG.info("Creating agent %s without MCP tools", config.agent_name)
        try:
            orchestrator._agent_descriptions[config.agent_name] = config.description
            agent = await orchestrator.create_chat_agent(config, tools=[])
            orchestrator.specialist_agents.append(agent)
        except Exception as e:
            LOG.error("Failed to create agent %s: %s", config.agent_name, e)
            failed_agents.append(config.agent_name)

    def _resolve_active_participant_configs(
        self,
        orchestrator: "MADAOrchestrator",
        participant_configs: List[AgentConfig],
    ) -> List[AgentConfig]:
        """
        Return configs for specialists that were successfully initialized.
        """
        active_specialist_names = {
            agent.name for agent in orchestrator.specialist_agents
        }
        return [
            config
            for config in participant_configs
            if config.agent_name in active_specialist_names
        ]

    def _build_status(
        self,
        orchestrator: "MADAOrchestrator",
        failed_servers: List[Dict[str, str]],
        failed_agents: List[str],
        failed_a2a_agents: List[Dict[str, str]] | None = None,
    ) -> str:
        """
        Build a user-facing initialization summary for the current run.
        """
        failed_a2a_agents = failed_a2a_agents or []
        status_parts = [
            (
                "Connection Successful: Orchestrator initialized with "
                f"{orchestrator._mcp_tool_count} MCP Servers and "
                f"{len(orchestrator.a2a_agents)} remote A2A agents and "
                f"{len(orchestrator.specialist_agents) + 1} agents"
            )
        ]

        if failed_servers:
            status_parts.append(
                f"\nWARNING: {len(failed_servers)} MCP server(s) failed to connect:"
            )
            for failed_server in failed_servers:
                status_parts.append(
                    f"  • {failed_server['agent']}/{failed_server['server']} at {failed_server['url']}"
                )
                status_parts.append(f"    Error: {failed_server['error']}")

        if failed_a2a_agents:
            status_parts.append(
                f"\nWARNING: {len(failed_a2a_agents)} remote A2A agent(s) unavailable:"
            )
            for failed_agent in failed_a2a_agents:
                status_parts.append(
                    f"  • {failed_agent['agent']} at {failed_agent['url']}"
                )
                status_parts.append(f"    Error: {failed_agent['error']}")

        if failed_agents:
            status_parts.append(
                f"\nERROR: {len(failed_agents)} agent(s) failed to initialize: {', '.join(failed_agents)}"
            )

        return "\n".join(status_parts)

    @abstractmethod
    async def initialize(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] | None = None,
        a2a_agents: Dict[str, RemoteA2AAgentConfig] | None = None,
    ) -> Tuple[str, List[str]]:
        """
        Initialize the orchestrator for the strategy's orchestration mode.

        Args:
            orchestrator: Orchestrator instance being configured.
            agent_configs: Agent definitions available to the strategy.
            mcp_servers: Named MCP server definitions available to the strategy.
            a2a_agents: Named remote A2A agents available to the strategy.

        Returns:
            A user-facing status message and a flat list of connected tool names.
        """
        pass

    @abstractmethod
    async def process_openai_messages(
        self,
        orchestrator: "MADAOrchestrator",
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Process OpenAI-style chat messages for this orchestration mode.
        """
        pass

    @abstractmethod
    async def process_message(
        self,
        orchestrator: "MADAOrchestrator",
        message: str,
        isolated_session: bool = False,
        persistence_session_id: str | None = None,
        stateless_session: bool = False,
        record_to_db: bool = True,
        background_poll_session_id: str | None = None,
        persistence_message: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process one interactive user message for this orchestration mode.
        """
        pass
