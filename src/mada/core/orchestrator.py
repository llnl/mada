# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Core orchestration logic for MADA multi-agent system.

This module contains the main orchestration logic extracted from the interface layers.
It manages MCP server connections, agent creation, and Agent Framework coordination
using the Planning Agent + Agent-as-Tool pattern.
"""

import asyncio
import copy
import logging
import sys
import traceback
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, AsyncGenerator
from contextlib import AsyncExitStack
import httpx
import httpcore

from agent_framework import Agent, AgentSession, MCPStdioTool, MCPStreamableHTTPTool
from agent_framework.exceptions import ToolException

from mada.core.config import AgentConfig, DatabaseConfig, ModelConfig, MCPServerConfig
from mada.core.coordinator import MCPAgentManager
from mada.core.database import ChatSessionManager

try:
    BaseExceptionGroup
except NameError:
    BaseExceptionGroup = Exception  # fallback for type checkers/runtime


LOG = logging.getLogger(__name__)


class MADAOrchestrator(MCPAgentManager):
    """
    Core orchestrator for MADA multi-agent system.
    Manages MCP server connections, agent creation, and planning agent coordination
    using the Agent Framework's Planning Agent + Agent-as-Tool pattern.
    This class contains the core logic extracted from interface-specific implementations.

    Attributes:
        exit_stack (AsyncExitStack): Context manager for server connections
        specialist_agents (List[Agent]): Specialist agents in the session
        planning_agent (Agent): The planning/coordination agent that routes to specialists
        session: AgentSession for maintaining conversation state
        mcp_servers (Dict[str, MCPServerConfig]): A dictionary store of MCP server configurations
        session_manager (ChatSessionManager): The high-level API for interacting
            with the database.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        database_config: Optional[DatabaseConfig] = None,
        session_manager: ChatSessionManager = None,
        bearer_token: Optional[str] = None,
        timeout: int = 86400,
    ):
        """
        Initialize the MADA orchestrator.

        Args:
            model_config: Model configuration for LLM communication.
            database_config: Database configuration for the chat history store.
            session_manager: Optional preconstructed session manager. If not
                provided, one is created from `database_config`.
            bearer_token: Optional token forwarded to streamable HTTP MCP
                servers as `X-Token`.
            timeout: Timeout in seconds for server operations.
        """
        super().__init__(model_config, timeout)
        self.exit_stack = AsyncExitStack()
        self.specialist_agents = []
        self.planning_agent = None
        self.mcp_servers = {}
        self.session = None
        self._session_lock = asyncio.Lock()
        self._next_turn_id = 1
        self._next_turn_commit_id = 1
        self._completed_turns: Dict[int, Dict[str, Any]] = {}
        self._agent_descriptions = {}
        self._mcp_tool_count = 0
        # Initialize the database
        self.session_manager = session_manager or ChatSessionManager(database_config)
        # Authentication bearer token
        self.bearer_token = bearer_token

    def _get_planning_agent_config(
        self, agent_configs: List[AgentConfig]
    ) -> Optional[AgentConfig]:
        """
        Return the AgentConfig entry for the PlanningAgent if one exists.

        The user can define a PlanningAgent in the config, for example:

            {
              "agent_name": "PlanningAgent",
              "description": "Custom planner behavior",
              "instructions": "You are a planning agent that..."
            }

        This entry is used to customize the planning agent created by the orchestrator.
        """
        for cfg in agent_configs:
            if cfg.agent_name == "PlanningAgent":
                return cfg
        return None

    async def _cleanup_http_client(self, http_client, context: str = ""):
        """
        Safely close an HTTP client, suppressing any errors.

        Args:
            http_client: The HTTP client to close
            context: Optional context string for debug logging (e.g., server name)
        """
        if http_client:
            try:
                await http_client.aclose()
            except (Exception, BaseExceptionGroup) as e:
                context_str = f" for {context}" if context else ""
                LOG.debug(f"Error closing HTTP client{context_str} (suppressed): {e}")

    async def _cleanup_failed_tool(self, mcp_tool, http_client):
        """
        Clean up a failed MCP tool's resources properly.

        When an MCP tool fails during __aenter__(), its async generators may be left
        in an incomplete state. This method properly closes them to avoid cleanup warnings.

        Args:
            mcp_tool: The MCP tool that failed to initialize
            http_client: The HTTP client used by the tool (for streamable-http tools)
        """
        # Try to close the MCP tool's async generators
        if mcp_tool:
            try:
                # Temporarily suppress agent_framework logger warnings about cancel scope
                # errors during cleanup - these are expected when cleaning up failed tools
                af_logger = logging.getLogger("agent_framework")
                original_level = af_logger.level
                af_logger.setLevel(logging.ERROR)

                try:
                    # For MCPStreamableHTTPTool and MCPStdioTool, call __aexit__ to properly
                    # clean up async generators even though __aenter__ failed
                    if hasattr(mcp_tool, "__aexit__"):
                        await mcp_tool.__aexit__(None, None, None)
                finally:
                    # Restore original logging level
                    af_logger.setLevel(original_level)

            except (Exception, BaseExceptionGroup) as e:
                # Suppress all errors during cleanup of a failed initialization
                # Note: RuntimeErrors with "cancel scope" are already handled by agent_framework
                # and don't propagate here
                LOG.debug(f"Error during failed tool cleanup (suppressed): {e}")

        # Close the HTTP client if it was created
        await self._cleanup_http_client(http_client)

    async def _handle_mcp_connection_error(
        self,
        server_name: str,
        server_url: str,
        error_msg: str,
        mcp_tool=None,
        http_client=None,
    ):
        """
        Handle MCP connection errors with consistent cleanup and logging.

        Args:
            server_name: Name of the MCP server
            server_url: URL of the MCP server (or command for stdio)
            error_msg: Human-readable error message
            mcp_tool: MCP tool to cleanup (optional) - only pass if fully initialized
            http_client: HTTP client to cleanup (optional)

        Returns:
            Dict with server failure information
        """
        # Cleanup MCP tool (only if it was successfully initialized)
        # Note: This should only be called with mcp_tool if it was successfully
        # entered into a context. If enter_async_context() failed, don't pass mcp_tool.
        if mcp_tool:
            try:
                await mcp_tool.close()
            except (Exception, BaseExceptionGroup) as e:
                LOG.debug(f"Error closing MCP tool {server_name}: {e}")

        # Cleanup HTTP client
        await self._cleanup_http_client(http_client, context=server_name)

        # Log the error
        LOG.error(
            f"  Cannot connect to MCP server '{server_name}' at {server_url}: {error_msg}"
        )

        # Return failure info
        return {"name": server_name, "url": server_url, "error": error_msg}

    async def connect_agent(
        self, agent_config: AgentConfig, mcp_servers: Dict[str, MCPServerConfig]
    ) -> Tuple[Agent, List, List[str], List[Dict[str, str]]]:
        """
        Connect to multiple MCP servers and create an associated chat agent.

        Args:
            agent_config: Configuration for the agent to create.
            mcp_servers: Dictionary of available MCP server configurations.

        Returns:
            Tuple containing:
            1. The created agent instance.
            2. The list of MCP tool objects successfully connected for that agent.
            3. The list of MCP server names corresponding to those tools.
            4. A list of failed_servers each of which is a dicts with 'name', 'url', 'error'
        """
        all_tool_names = []
        all_tools = []
        failed_servers = []

        # Connect to each MCP server that this agent should use
        for server_name in agent_config.mcp_servers:
            if server_name not in mcp_servers:
                LOG.warning(
                    f"MCP server '{server_name}' not found in configuration for agent '{agent_config.agent_name}'"
                )
                continue

            server_config = mcp_servers[server_name]

            # Create MCP tool
            http_client_to_cleanup = None  # Track HTTP client for cleanup on failure

            if server_config.transport == "stdio":
                server_path = server_config.command or getattr(
                    agent_config, "server_path", None
                )
                if not server_path:
                    LOG.error(f"No command/server_path for stdio server {server_name}")
                    continue
                is_python = server_path.endswith(".py")
                command = server_config.python_executable if is_python else "node"
                # -u for unbuffered output
                args = ["-u", server_path] if is_python else [server_path]
                mcp_tool = MCPStdioTool(
                    name=server_name,
                    command=command,
                    args=args,
                )
            elif server_config.transport == "streamable-http":
                if not server_config.url:
                    LOG.error(f"No URL for streamable-http server {server_name}")
                    continue

                # Set up headers for MCP streamable HTTP - include content negotiation headers
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream, application/json",
                    "Cache-Control": "no-cache",
                }

                # Add bearer token if provided
                if self.bearer_token:
                    headers["X-Token"] = self.bearer_token

                # MCPStreamableHTTPTool requires an http_client with custom headers, not a headers parameter
                http_client = httpx.AsyncClient(headers=headers, timeout=180.0)
                http_client_to_cleanup = (
                    http_client  # Store for cleanup if connection fails
                )

                mcp_tool = MCPStreamableHTTPTool(
                    name=server_name,
                    url=server_config.url,
                    http_client=http_client,
                )
            else:
                LOG.error(f"Unsupported transport: {server_config.transport}")
                continue

            # Start the MCP server connection
            LOG.info(
                f"Attempting to connect to MCP server '{server_name}' at {server_config.url}..."
            )

            try:
                mcp_tool = await self.exit_stack.enter_async_context(mcp_tool)
                all_tools.append(mcp_tool)
                all_tool_names.append(server_name)
                LOG.info(f"✓ Successfully connected to MCP server: {server_name}")
            except asyncio.CancelledError:
                # Connection was cancelled - properly close the tool to cleanup async generators
                await self._cleanup_failed_tool(mcp_tool, http_client_to_cleanup)
                failed_servers.append(
                    await self._handle_mcp_connection_error(
                        server_name,
                        server_config.url,
                        "Connection cancelled (server unavailable)",
                        mcp_tool=None,
                        http_client=None,
                    )
                )
                continue
            except (httpx.ConnectError, httpcore.ConnectError):
                # Connection failed - properly close the tool to cleanup async generators
                await self._cleanup_failed_tool(mcp_tool, http_client_to_cleanup)
                failed_servers.append(
                    await self._handle_mcp_connection_error(
                        server_name,
                        server_config.url,
                        "Connection refused or server not available",
                        mcp_tool=None,
                        http_client=None,
                    )
                )
                continue
            except (
                httpx.TimeoutException,
                httpcore.ReadTimeout,
                httpcore.WriteTimeout,
                httpcore.PoolTimeout,
            ):
                # Connection timed out - properly close the tool to cleanup async generators
                await self._cleanup_failed_tool(mcp_tool, http_client_to_cleanup)
                failed_servers.append(
                    await self._handle_mcp_connection_error(
                        server_name,
                        server_config.url,
                        "Server timeout",
                        mcp_tool=None,
                        http_client=None,
                    )
                )
                continue
            except BaseExceptionGroup as e:
                # Multiple errors - properly close the tool to cleanup async generators
                await self._cleanup_failed_tool(mcp_tool, http_client_to_cleanup)
                failed_servers.append(
                    await self._handle_mcp_connection_error(
                        server_name,
                        server_config.url,
                        f"Multiple connection errors ({len(e.exceptions)} errors)",
                        mcp_tool=None,
                        http_client=None,
                    )
                )
                continue
            except ToolException as e:
                # Check for specific MCP protocol errors
                # Properly close the tool to cleanup async generators
                await self._cleanup_failed_tool(mcp_tool, http_client_to_cleanup)
                error_detail = str(e)
                if "Session terminated" in error_detail:
                    error_msg = "MCP session rejected (check authentication/token)"
                else:
                    error_msg = f"MCP initialization failed: {error_detail[:100]}"
                failed_servers.append(
                    await self._handle_mcp_connection_error(
                        server_name,
                        server_config.url,
                        error_msg,
                        mcp_tool=None,
                        http_client=None,
                    )
                )
                continue
            except Exception as e:
                # Generic error - properly close the tool to cleanup async generators
                await self._cleanup_failed_tool(mcp_tool, http_client_to_cleanup)
                failed_servers.append(
                    await self._handle_mcp_connection_error(
                        server_name,
                        server_config.url,
                        f"{type(e).__name__}: {str(e)[:100]}",
                        mcp_tool=None,
                        http_client=None,
                    )
                )
                continue

        # Store agent description for later use in as_tool()
        self._agent_descriptions[agent_config.agent_name] = agent_config.description

        # Create agent with MCP tools
        agent = await self.create_chat_agent(
            agent_config,
            tools=all_tools,
        )
        LOG.info(
            f"Agent '{agent_config.agent_name}' created with {len(all_tools)} MCP tools"
        )

        return agent, all_tools, all_tool_names, failed_servers

    def _create_planning_agent(self, agent_configs: List[AgentConfig]) -> Agent:
        """
        Create the planning agent that coordinates other agents.

        If the user provides an AgentConfig with agent_name == "PlanningAgent"
        in the top-level "agents" list, its settings (especially instructions)
        are used as the base for the planning agent configuration.

        The planning agent uses the as_tool() pattern to call specialist agents.

        Args:
            agent_configs: List of agent configurations used to describe the
                team and customize the planning agent.

        Returns:
            Planning agent instance with specialist agents exposed as tools.
        """
        team_description = self._generate_team_description(agent_configs)

        # Convert each specialist agent to a tool using as_tool()
        agent_tools = []
        for agent in self.specialist_agents:
            description = self._agent_descriptions.get(
                agent.name, f"Specialist agent: {agent.name}"
            )
            agent_tool = agent.as_tool(
                name=agent.name,
                description=description,
                arg_name="task",
                arg_description="The task to delegate to this agent",
            )
            agent_tools.append(agent_tool)

        # Try to get a user defined planning agent config
        planning_cfg = self._get_planning_agent_config(agent_configs)

        # Check for MCP servers in PlanningAgent config
        if planning_cfg and planning_cfg.mcp_servers:
            LOG.warning(
                "PlanningAgent MCP server support is not implemented. "
                "MCP servers listed in PlanningAgent config will be ignored."
            )

        if planning_cfg and planning_cfg.instructions:
            # Use user provided instructions and append the team description / guidelines
            base_instructions = planning_cfg.instructions.strip()
        else:
            # Default behavior if no PlanningAgent config is provided
            base_instructions = """You are a planning agent for the MADA multi-agent system.

Your specialist agents (available as tools) can be delegated tasks.
"""

        # Always append up to date team description and guidelines so the planning
        # agent knows how to use the tools.
        instructions = f"""{base_instructions}

Your specialist agents (available as tools):
{team_description}

Guidelines:
- Delegate to specialist agents when the request matches their expertise
- Answer directly only for questions about the system itself
- Avoid infinite loops between agents
- After receiving results, synthesize and respond to the user
"""

        # Name and any other settings can also come from planning_cfg, but we
        # hard code the name "PlanningAgent" for now to keep downstream logic simple.
        agent_name = planning_cfg.agent_name if planning_cfg else "PlanningAgent"

        agent_kwargs = {}
        if planning_cfg:
            agent_kwargs.update(planning_cfg.extra)

        planning_agent = self.model_client.as_agent(
            name=agent_name,
            instructions=instructions,
            tools=agent_tools,
            **agent_kwargs,
        )

        return planning_agent

    def _generate_team_description(self, agent_configs: List[AgentConfig]) -> str:
        """
        Generate formatted team member descriptions.

        Args:
            agent_configs: Agent configurations to summarize.

        Returns:
            Multiline text describing each configured agent and its role.
        """
        lines = [
            f"    {agent.agent_name}: {agent.description}" for agent in agent_configs
        ]
        return "\n".join(lines)

    async def initialize_orchestrator(
        self,
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] = None,
    ) -> Tuple[str, List[str]]:
        """
        Initialize the orchestrator with the given agent configurations.

        Args:
            agent_configs: List of agent configurations to set up.
            mcp_servers: Dictionary of MCP server configurations.

        Returns:
            Tuple containing:
            1. A human-readable status message describing the initialized team.
            2. A list of available tool labels in `agent_name: tool_name` format.
        """
        self.specialist_agents = []
        self._mcp_tool_count = 0
        all_tools = []
        failed_servers = []  # Track failed server connections
        failed_agents = []  # Track agents that failed to connect

        self.mcp_servers = mcp_servers or {}

        for config in agent_configs:
            if not config.agent_name:
                continue

            # skip the PlanningAgent here, it will be created separately
            if config.agent_name == "PlanningAgent":
                continue

            if config.mcp_servers and self.mcp_servers:
                try:
                    (
                        agent,
                        mcp_tools,
                        tool_names,
                        agent_failed_servers,
                    ) = await self.connect_agent(config, self.mcp_servers)
                    self.specialist_agents.append(agent)
                    self._mcp_tool_count += len(mcp_tools)
                    all_tools.extend(
                        [f"{config.agent_name}: {tool}" for tool in tool_names]
                    )

                    # Track any failed servers for this agent
                    if agent_failed_servers:
                        for fs in agent_failed_servers:
                            failed_servers.append(
                                {
                                    "agent": config.agent_name,
                                    "server": fs["name"],
                                    "url": fs["url"],
                                    "error": fs["error"],
                                }
                            )

                    if len(tool_names) > 0:
                        LOG.info(
                            f"Connected agent {config.agent_name} with {len(tool_names)} MCP tools"
                        )
                    else:
                        LOG.warning(
                            f"Agent {config.agent_name} connected but no MCP servers available"
                        )
                        if len(config.mcp_servers) > 0:
                            failed_agents.append(config.agent_name)
                except BaseExceptionGroup as eg:
                    LOG.error(
                        f"Multiple errors connecting agent {config.agent_name} ({len(eg.exceptions)} errors)"
                    )
                    failed_agents.append(config.agent_name)
                    continue
                except Exception as e:
                    LOG.error(f"Failed to connect agent {config.agent_name}: {e}")
                    traceback.print_exc()
                    failed_agents.append(config.agent_name)
                    continue
            elif config.server_path:
                # Legacy support for single server_path
                try:
                    is_python = config.server_path.endswith(".py")
                    command = sys.executable if is_python else "node"
                    # Use -u for unbuffered Python output
                    args = (
                        ["-u", config.server_path]
                        if is_python
                        else [config.server_path]
                    )
                    mcp_tool = MCPStdioTool(
                        name=f"{config.agent_name}_mcp",
                        command=command,
                        args=args,
                    )

                    # Start the MCP server connection
                    mcp_tool = await self.exit_stack.enter_async_context(mcp_tool)

                    self._agent_descriptions[config.agent_name] = config.description

                    agent = await self.create_chat_agent(
                        config,
                        tools=[mcp_tool],
                    )

                    self.specialist_agents.append(agent)
                    self._mcp_tool_count += 1
                    all_tools.append(f"{config.agent_name}: {config.server_path}")
                    LOG.info(
                        f"Connected legacy agent {config.agent_name} with 1 MCP tool"
                    )
                except Exception as e:
                    LOG.error(
                        f"Failed to connect legacy agent {config.agent_name}: {e}"
                    )
                    continue
            elif not config.mcp_servers:
                LOG.info(f"Creating agent {config.agent_name} without MCP tools")
                try:
                    self._agent_descriptions[config.agent_name] = config.description
                    agent = await self.create_chat_agent(config, tools=[])
                    self.specialist_agents.append(agent)
                except Exception as e:
                    LOG.error(f"Failed to create agent {config.agent_name}: {e}")
                    continue
            else:
                LOG.warning(f"Agent {config.agent_name} has no MCP servers configured")
                continue

        self.planning_agent = self._create_planning_agent(agent_configs)
        self.session = self.planning_agent.create_session()

        # Build status message
        status_parts = []
        status_parts.append(
            f"Connection Successful: Orchestrator initialized with {self._mcp_tool_count} MCP Servers and {len(self.specialist_agents) + 1} agents"
        )

        # Report any failed connections
        if failed_servers:
            status_parts.append(
                f"\nWARNING: {len(failed_servers)} MCP server(s) failed to connect:"
            )
            for fs in failed_servers:
                status_parts.append(f"  • {fs['agent']}/{fs['server']} at {fs['url']}")
                status_parts.append(f"    Error: {fs['error']}")

        if failed_agents:
            status_parts.append(
                f"\nERROR: {len(failed_agents)} agent(s) failed to initialize: {', '.join(failed_agents)}"
            )

        status = "\n".join(status_parts)
        LOG.info(status)

        return status, all_tools

    def _stringify_openai_content(self, content: Any) -> str:
        """
        Convert OpenAI-style message content into plain text.

        Content may be a plain string or a list of structured parts. Text parts are
        preserved and image parts are rendered as markdown URLs so downstream models
        still receive a useful reference.

        Args:
            content: Message content from an OpenAI-style request. This may be a
                plain string, a list of structured content parts, or another
                JSON-serializable value.

        Returns:
            Plain text suitable for inclusion in a prompt to the planning
            agent.
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url", "")
                    if image_url:
                        parts.append(f"[Image: {image_url}]")

            return "\n".join(part for part in parts if part)

        return str(content)

    def build_prompt_from_openai_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Flatten OpenAI-style chat messages into a single prompt.

        Open WebUI sends the full conversation history on each request. The agent
        framework session used by the CLI is not shared here; instead we rebuild the
        transcript for each HTTP request to keep requests isolated.

        Args:
            messages: OpenAI-style chat messages. Each message is expected to
                include at least a `role` and `content`.

        Returns:
            A single prompt string that contains the conversation transcript and
            instructions to continue as the assistant.
        """
        transcript = []
        for message in messages:
            role = message.get("role") or "user"
            role = str(role).strip().lower() or "user"
            content = self._stringify_openai_content(message.get("content")).strip()
            if not content:
                continue
            transcript.append(f"{role.upper()}:\n{content}")

        if not transcript:
            return "USER:\nPlease introduce yourself."

        conversation = "\n\n".join(transcript)
        return (
            "Continue the conversation below and respond as the assistant to the "
            "latest user request.\n\n"
            f"{conversation}"
        )

    def _capture_history_lengths(self, session: AgentSession) -> Dict[str, int]:
        """
        Record the number of stored history messages per provider in a session.

        Args:
            session: Session snapshot to inspect.

        Returns:
            Mapping of provider state key to stored message count.
        """
        history_lengths: Dict[str, int] = {}
        for provider_name, provider_state in session.state.items():
            if not isinstance(provider_state, dict):
                continue
            messages = provider_state.get("messages")
            if isinstance(messages, list):
                history_lengths[provider_name] = len(messages)
        return history_lengths

    async def _reserve_turn(self) -> int:
        """
        Reserve the next conversational turn identifier.

        Returns:
            Monotonic turn identifier in user-submission order.
        """
        async with self._session_lock:
            turn_id = self._next_turn_id
            self._next_turn_id += 1
        return turn_id

    async def _clone_shared_session(self) -> Tuple[AgentSession, Dict[str, int]]:
        """
        Snapshot the active planning-agent session for isolated async execution.

        Returns:
            The cloned session and baseline history lengths.
        """
        async with self._session_lock:
            if self.session is None:
                raise RuntimeError("Orchestrator session not initialized.")
            session_snapshot = AgentSession.from_dict(self.session.to_dict())
            history_lengths = self._capture_history_lengths(session_snapshot)
        return session_snapshot, history_lengths

    def _merge_session_delta(
        self,
        target_session: AgentSession,
        source_session: AgentSession,
        history_lengths: Dict[str, int],
    ) -> None:
        """
        Merge newly stored history from an isolated session back into the shared session.

        Args:
            target_session: Shared session to update.
            source_session: Completed isolated session.
            history_lengths: Baseline history lengths captured before the run.
        """
        for provider_name, source_state in source_session.state.items():
            if not isinstance(source_state, dict):
                if provider_name not in target_session.state:
                    target_session.state[provider_name] = copy.deepcopy(source_state)
                continue

            target_state = target_session.state.setdefault(provider_name, {})
            if not isinstance(target_state, dict):
                target_session.state[provider_name] = copy.deepcopy(source_state)
                continue

            source_messages = source_state.get("messages")
            if isinstance(source_messages, list):
                start_index = history_lengths.get(provider_name, 0)
                delta_messages = source_messages[start_index:]
                if delta_messages:
                    target_messages = target_state.setdefault("messages", [])
                    if not isinstance(target_messages, list):
                        target_messages = []
                        target_state["messages"] = target_messages
                    target_messages.extend(copy.deepcopy(delta_messages))

            for key, value in source_state.items():
                if key == "messages":
                    continue
                target_state.setdefault(key, copy.deepcopy(value))

        if source_session.service_session_id and not target_session.service_session_id:
            target_session.service_session_id = source_session.service_session_id

    async def _flush_completed_turns(self) -> None:
        """
        Commit any completed turns whose predecessors have already been applied.
        """
        while True:
            completed_turn = self._completed_turns.pop(self._next_turn_commit_id, None)
            if completed_turn is None:
                return

            if self.session is None:
                raise RuntimeError("Orchestrator session not initialized.")

            self._merge_session_delta(
                self.session,
                completed_turn["completed_session"],
                completed_turn["history_lengths"],
            )
            self.session_manager.add_message("user", completed_turn["message"])

            assistant_reply = completed_turn["assistant_reply"]
            if assistant_reply.strip():
                self.session_manager.add_message("assistant", assistant_reply)

            self._next_turn_commit_id += 1

    async def _commit_isolated_turn(
        self,
        turn_id: int,
        message: str,
        assistant_reply: str,
        completed_session: AgentSession,
        history_lengths: Dict[str, int],
    ) -> None:
        """
        Persist an isolated turn back into the shared session and chat database.

        Args:
            turn_id: Reserved turn identifier for commit ordering.
            message: User message for the turn.
            assistant_reply: Plain-text assistant reply aggregated during streaming.
            completed_session: The isolated session after the run completed.
            history_lengths: Baseline history lengths captured before the run.
        """
        async with self._session_lock:
            self._completed_turns[turn_id] = {
                "message": message,
                "assistant_reply": assistant_reply,
                "completed_session": completed_session,
                "history_lengths": history_lengths,
            }
            await self._flush_completed_turns()

    async def process_openai_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Process OpenAI-style chat messages without reusing the interactive session.

        Each request gets a fresh Agent Framework session so HTTP callers do not
        share state with the CLI, Gradio, or each other.

        Args:
            messages: OpenAI-style chat messages representing the full request
                history.

        Yields:
            Response text chunks from the planning agent, plus bracketed tool
            call notices when the underlying stream reports them.
        """
        if not self.planning_agent:
            yield "Error: Orchestrator not initialized."
            return

        prompt = self.build_prompt_from_openai_messages(messages)
        request_session = self.planning_agent.create_session()

        try:
            response_started = False
            stream = self.planning_agent.run(
                prompt, session=request_session, stream=True
            )
            async for chunk in stream:
                if chunk.text:
                    response_started = True
                    yield chunk.text
                elif hasattr(chunk, "contents") and chunk.contents:
                    for content in chunk.contents:
                        if hasattr(content, "to_dict"):
                            item = content.to_dict()
                            if item.get("type") in (
                                "function_call",
                                "tool_call",
                            ) and item.get("name"):
                                yield f"\n[Calling: {item['name']}]\n"

            if not response_started:
                LOG.warning(
                    "No text chunks received from planning agent for OpenAI API request"
                )

        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield error_msg

    async def process_message(self, message: str) -> AsyncGenerator[str, None]:
        """
        Process a user message through the planning agent.

        The planning agent routes to specialist agents as needed using the
        as_tool() pattern. Each call runs against an isolated snapshot of the
        active session so overlapping async requests do not mutate the shared
        conversation state mid-stream. Successful turns are merged back into the
        shared session after streaming completes.

        Args:
            message: User input message.

        Yields:
            Response text chunks from the planning agent, plus bracketed tool
            call notices when the underlying stream reports them.
        """
        if not self.planning_agent:
            yield "Error: Orchestrator not initialized. Call initialize_orchestrator() first."
            return

        aggregated_assistant_reply = ""
        tool_calls = []
        turn_id: int
        run_session: AgentSession
        history_lengths: Dict[str, int]

        try:
            turn_id = await self._reserve_turn()
            run_session, history_lengths = await self._clone_shared_session()

            response_started = False
            stream = self.planning_agent.run(message, session=run_session, stream=True)
            async for chunk in stream:
                if chunk.text:
                    response_started = True
                    aggregated_assistant_reply += chunk.text
                    yield chunk.text
                elif hasattr(chunk, "contents") and chunk.contents:
                    for content in chunk.contents:
                        if hasattr(content, "to_dict"):
                            d = content.to_dict()
                            if d.get("type") in (
                                "function_call",
                                "tool_call",
                            ) and d.get("name"):
                                call_id = d.get("call_id")
                                name = d.get("name")

                                if call_id and call_id not in tool_calls:
                                    tool_calls.append(call_id)
                                    yield f"\n[Calling: {name}]\n"

            if not response_started:
                LOG.warning("No text chunks received from planning agent")

            await self._commit_isolated_turn(
                turn_id=turn_id,
                message=message,
                assistant_reply=aggregated_assistant_reply,
                completed_session=run_session,
                history_lengths=history_lengths,
            )

        except Exception as e:
            # Check for authentication errors
            error_str = str(e)
            if (
                " 401" in error_str
                or error_str.startswith("401")
                or "Authentication" in error_str
                or "auth_error" in error_str
            ):
                # Check if it's an unexpanded environment variable
                if "${" in error_str and "}" in error_str:
                    error_msg = (
                        "\n  AUTHENTICATION ERROR: API key not set correctly\n\n"
                        "Your API key appears to be an unexpanded environment variable.\n\n"
                        "Problem: The configuration contains '${VARIABLE_NAME}' but the environment variable is not set.\n\n"
                        "Solutions:\n"
                        "  1. Set the environment variable before running:\n"
                        "     export ANTHROPIC_API_KEY='your-api-key-here'\n"
                        "     (or whichever variable name is in your config)\n\n"
                        "  2. Or update your config file to use the actual API key directly\n\n"
                        f"Details: {error_str}\n"
                    )
                else:
                    error_msg = (
                        "\n  AUTHENTICATION ERROR: Invalid API key\n\n"
                        "The API key provided is not valid or is in the wrong format.\n\n"
                        "Solutions:\n"
                        "  1. Check that your API key is correct\n"
                        "  2. Verify the key format matches what the service expects\n"
                        "  3. Ensure the API key has not expired\n\n"
                        f"Details: {error_str}\n"
                    )
                LOG.error(error_msg)
                yield error_msg
            else:
                # Generic error handling
                error_msg = f"Error processing message: {e}"
                LOG.error(error_msg)
                traceback.print_exc()
                yield error_msg

    async def cleanup(self) -> None:
        """
        Clean up all resources and connections.

        Returns:
            `None`.
        """
        try:
            await self.exit_stack.aclose()
            self.specialist_agents.clear()
            self.planning_agent = None
            self.session = None
            self._completed_turns.clear()
            self._next_turn_id = 1
            self._next_turn_commit_id = 1
            self._agent_descriptions.clear()
            LOG.info("Orchestrator cleanup completed")
        except BaseExceptionGroup as eg:
            # Handle exception groups from anyio task groups during cleanup
            # This is expected when MCP servers failed to connect - their async generators
            # will raise errors during cleanup, which we can safely suppress
            LOG.debug(
                f"Async cleanup errors suppressed ({len(eg.exceptions)} errors) - this is expected for failed MCP connections"
            )
        except RuntimeError as e:
            # Suppress "Attempted to exit cancel scope in different task" errors
            # These occur when async generators from failed MCP connections are cleaned up
            if "cancel scope" in str(e):
                LOG.debug(
                    f"Async generator cleanup error suppressed (expected for failed MCP connections): {e}"
                )
            else:
                LOG.error(f"Runtime error during cleanup: {e}")
        except Exception as e:
            LOG.error(f"Error during cleanup: {e}")

    async def __aenter__(self) -> "MADAOrchestrator":
        """
        Async context manager entry.

        Returns:
            The orchestrator instance itself.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        """
        Async context manager exit.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc_val: Exception instance raised inside the context, if any.
            exc_tb: Traceback associated with the exception, if any.
        """
        await self.cleanup()
