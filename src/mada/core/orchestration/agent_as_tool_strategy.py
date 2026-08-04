# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Agent-as-tool orchestration strategy implementation.
"""

import logging
import sys
import traceback
from typing import TYPE_CHECKING, Dict, List, Tuple

from agent_framework import MCPStdioTool

from mada.core.config import AgentConfig, MCPServerConfig
from mada.core.orchestration.base_strategy import BaseOrchestrationStrategy

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator

try:
    BaseExceptionGroup
except NameError:
    BaseExceptionGroup = Exception  # fallback for type checkers/runtime


LOG = logging.getLogger(__name__)


class AgentAsToolOrchestrationStrategy(BaseOrchestrationStrategy):
    """
    Planning-agent-plus-`as_tool()` orchestration.

    This strategy initializes specialist agents first, exposes them to the
    planning agent as callable tools, and then creates the session from the
    planning agent.
    """

    mode = "agent-as-tool"

    async def _initialize_participants(
        self,
        orchestrator: "MADAOrchestrator",
        participant_configs: List[AgentConfig],
    ) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
        """
        Initialize each configured specialist agent for this orchestration run.

        Agents are connected through named MCP servers when available, fall back
        to the legacy `server_path` mode when configured, or are created as
        model-only agents when no tools are defined.
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

            if not config.mcp_servers:
                await self._connect_agent_without_tools(
                    orchestrator, config, failed_agents
                )
                continue

            LOG.warning(f"Agent {config.agent_name} has no MCP servers configured")

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
        Connect one agent through its named MCP server definitions.

        Successful tool names are appended to `all_tools`, while partial or full
        connection failures are recorded in `failed_servers` and `failed_agents`.
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
                    f"Connected agent {config.agent_name} with {len(tool_names)} MCP tools"
                )
                return

            LOG.warning(
                f"Agent {config.agent_name} connected but no MCP servers available"
            )
            if config.mcp_servers:
                failed_agents.append(config.agent_name)
        except BaseExceptionGroup as eg:
            LOG.error(
                f"Multiple errors connecting agent {config.agent_name} ({len(eg.exceptions)} errors)"
            )
            failed_agents.append(config.agent_name)
        except Exception as e:
            LOG.error(f"Failed to connect agent {config.agent_name}: {e}")
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

        This preserves backward compatibility for older single-script MCP server
        definitions that predate the shared `mcp_servers` configuration block.
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
            LOG.info(f"Connected legacy agent {config.agent_name} with 1 MCP tool")
        except Exception as e:
            LOG.error(f"Failed to connect legacy agent {config.agent_name}: {e}")
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
        LOG.info(f"Creating agent {config.agent_name} without MCP tools")
        try:
            orchestrator._agent_descriptions[config.agent_name] = config.description
            agent = await orchestrator.create_chat_agent(config, tools=[])
            orchestrator.specialist_agents.append(agent)
        except Exception as e:
            LOG.error(f"Failed to create agent {config.agent_name}: {e}")
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

    def _initialize_planning_agent(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        active_participant_configs: List[AgentConfig],
    ) -> None:
        """
        Create the planning agent and open the orchestration session.
        """
        orchestrator.planning_agent = orchestrator._create_planning_agent(
            agent_configs=agent_configs,
            participant_configs=active_participant_configs,
        )
        orchestrator.session = orchestrator.planning_agent.create_session()

    def _build_status(
        self,
        orchestrator: "MADAOrchestrator",
        failed_servers: List[Dict[str, str]],
        failed_agents: List[str],
    ) -> str:
        """
        Build a user-facing initialization summary for the current run.
        """
        status_parts = [
            (
                "Connection Successful: Orchestrator initialized with "
                f"{orchestrator._mcp_tool_count} MCP Servers and "
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

        if failed_agents:
            status_parts.append(
                f"\nERROR: {len(failed_agents)} agent(s) failed to initialize: {', '.join(failed_agents)}"
            )

        return "\n".join(status_parts)

    async def initialize(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] | None = None,
    ) -> Tuple[str, List[str]]:
        """
        Initialize the agent-as-tool orchestration flow end to end.

        The strategy resets orchestrator state, initializes participating
        specialists, creates the planning agent around the successfully active
        specialists, and returns a connection summary plus discovered tools.
        """
        orchestrator.specialist_agents = []
        orchestrator._mcp_tool_count = 0
        participant_configs = orchestrator.resolve_participant_configs(agent_configs)
        orchestrator.mcp_servers = mcp_servers or {}
        all_tools, failed_servers, failed_agents = await self._initialize_participants(
            orchestrator, participant_configs
        )
        active_participant_configs = self._resolve_active_participant_configs(
            orchestrator, participant_configs
        )
        self._initialize_planning_agent(
            orchestrator, agent_configs, active_participant_configs
        )
        status = self._build_status(orchestrator, failed_servers, failed_agents)
        LOG.info(status)

        return status, all_tools
