# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Core orchestration logic for MADA multi-agent system.

This module contains the shared orchestration runtime extracted from the
interface layers. It owns MCP server connections, agent creation, session
persistence, and strategy selection. Mode-specific request handling lives in
`mada.core.orchestration`.
"""

import asyncio
import copy
import logging
import traceback
from types import TracebackType
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type
from contextlib import AsyncExitStack
import httpx
import httpcore

from agent_framework import (
    Agent,
    AgentSession,
    MCPStdioTool,
    MCPStreamableHTTPTool,
)
from agent_framework.exceptions import ToolException

from mada.core.background_tasks import BackgroundTaskManager
from mada.core.config import (
    AgentConfig,
    DatabaseConfig,
    ModelConfig,
    MCPServerConfig,
    OrchestrationConfig,
)
from mada.core.coordinator import MCPAgentManager
from mada.core.database import ChatSessionManager
from mada.core.orchestration import (
    AgentAsToolOrchestrationStrategy,
    BaseOrchestrationStrategy,
    MagenticOrchestrationStrategy,
)


LOG = logging.getLogger(__name__)


class MADAOrchestrator(MCPAgentManager):
    """
    Core orchestrator for MADA multi-agent system.

    Manages MCP server connections, agent creation, shared session state, and
    persistence. Concrete orchestration behavior is delegated to the configured
    strategy so mode-specific logic stays out of this shared runtime.

    Attributes:
        exit_stack (AsyncExitStack): Context manager for server connections
        specialist_agents (List[Agent]): Specialist agents in the session
        planning_agent (Agent): Agent-as-tool planner, when that mode is active.
        manager_agent (Agent): Hidden Magentic manager, when that mode is active.
        session: AgentSession for maintaining conversation state
        mcp_servers (Dict[str, MCPServerConfig]): A dictionary store of MCP server configurations
        session_manager (ChatSessionManager): The high-level API for interacting
            with the database.
        background_tasks (BackgroundTaskManager): Manager for non-blocking user
            queries and MCP server-side background task polling.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        database_config: Optional[DatabaseConfig] = None,
        session_manager: ChatSessionManager = None,
        orchestration_config: Optional[OrchestrationConfig] = None,
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

        Raises:
            Exception: Propagates session manager initialization failures.
        """
        super().__init__(model_config, timeout)
        self.exit_stack = AsyncExitStack()
        self.specialist_agents = []
        self.planning_agent = None
        self.manager_agent = None
        self.mcp_servers = {}
        self.session = None
        self._session_lock = asyncio.Lock()
        self._next_turn_id = 1
        self._next_turn_commit_id = 1
        self._completed_turns: Dict[int, Dict[str, Any]] = {}
        self._mcp_tools_by_server: Dict[str, Any] = {}
        self._agent_descriptions = {}
        self._mcp_tool_count = 0
        self._control_agent: Optional[Agent] = None
        self.orchestration = orchestration_config or OrchestrationConfig()
        self.orchestration_strategy = self._build_orchestration_strategy(
            self.orchestration.mode
        )
        # Initialize the database
        self.session_manager = session_manager or ChatSessionManager(database_config)
        self.background_tasks = BackgroundTaskManager(
            self.session_manager,
            self._mcp_tools_by_server,
            self.collect_message_response,
        )
        # Authentication bearer token
        self.bearer_token = bearer_token

    def _build_orchestration_strategy(self, mode: str) -> BaseOrchestrationStrategy:
        """
        Select the internal orchestration strategy for the configured mode.

        Args:
            mode: Normalized orchestration mode.

        Returns:
            A concrete orchestration strategy.

        Raises:
            ValueError: If the mode is not supported.
        """
        if mode == AgentAsToolOrchestrationStrategy.mode:
            return AgentAsToolOrchestrationStrategy()
        if mode == MagenticOrchestrationStrategy.mode:
            return MagenticOrchestrationStrategy()

        raise ValueError(f"unsupported orchestration mode: {mode}")

    def resolve_participant_configs(
        self, agent_configs: List[AgentConfig]
    ) -> List[AgentConfig]:
        """
        Resolve the specialist agents participating in orchestration.

        Args:
            agent_configs: All configured agents, including an optional
                `PlanningAgent`.

        Returns:
            Ordered list of specialist agent configs to include.
        """
        specialist_configs = [
            config
            for config in agent_configs
            if config.agent_name and config.agent_name != "PlanningAgent"
        ]
        specialist_by_name = {
            config.agent_name: config
            for config in specialist_configs
            if config.agent_name
        }

        self.orchestration.validate_participants(
            [config.agent_name for config in agent_configs if config.agent_name]
        )

        if self.orchestration.participants is None:
            return specialist_configs

        resolved_configs = []
        seen_names = set()
        for participant_name in self.orchestration.participants:
            if participant_name in seen_names:
                continue
            resolved_configs.append(specialist_by_name[participant_name])
            seen_names.add(participant_name)

        return resolved_configs

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

        This entry customizes the visible planner in `agent-as-tool` mode and
        the hidden manager in `magentic` mode.
        """
        for cfg in agent_configs:
            if cfg.agent_name == "PlanningAgent":
                return cfg
        return None

    def _get_control_agent(self) -> Agent:
        """
        Return a tool-free agent for internal control turns.

        Autonomy control prompts must never trigger tool calls, so they run
        through a dedicated agent with no tools and no shared chat session.
        """
        if self._control_agent is not None:
            return self._control_agent

        instructions = (
            "You are an internal control agent for the MADA orchestrator.\n"
            "You MUST NOT call any tools. You will be given a control prompt "
            "that requires outputting a strict key=value format. Follow it "
            "exactly.\n"
            "Do not include explanations or extra text."
        )
        self._control_agent = self.model_client.as_agent(
            name="AutonomyControl",
            instructions=instructions,
            tools=[],
        )
        return self._control_agent

    async def run_control_prompt(self, prompt: str) -> str:
        """
        Run a tool-free control prompt and return the aggregated text.

        This is used for autonomy gating decisions and must not alter the
        shared interactive planning session or the chat database.
        """
        agent = self._get_control_agent()
        session = agent.create_session()
        aggregated = ""
        stream = agent.run(prompt, session=session, stream=True)
        async for chunk in stream:
            if getattr(chunk, "text", None):
                aggregated += chunk.text
        return aggregated

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
                        await asyncio.wait_for(
                            mcp_tool.__aexit__(None, None, None),
                            timeout=max(float(self.timeout), 0.1),
                        )
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

                connection_timeout = max(float(self.timeout), 0.1)
                # MCPStreamableHTTPTool requires an http_client with custom headers, not a headers parameter
                http_client = httpx.AsyncClient(
                    headers=headers, timeout=connection_timeout
                )
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
                mcp_tool = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(mcp_tool),
                    timeout=max(float(self.timeout), 0.1),
                )
                all_tools.append(mcp_tool)
                all_tool_names.append(server_name)
                self._mcp_tools_by_server[server_name] = mcp_tool
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
                asyncio.TimeoutError,
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

    def _create_planning_agent(
        self,
        agent_configs: List[AgentConfig],
        participant_configs: List[AgentConfig],
    ) -> Agent:
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
        team_description = self._generate_team_description(participant_configs)

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
            f"    {agent.agent_name}: {agent.description}"
            for agent in agent_configs
            if agent.agent_name != "PlanningAgent"
        ]
        if not lines:
            return "    (no specialist agents configured)"
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
        return await self.orchestration_strategy.initialize(
            orchestrator=self,
            agent_configs=agent_configs,
            mcp_servers=mcp_servers,
        )

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

    def _normalize_transcript_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Normalize OpenAI-style messages or persisted chat history into a shared
        role/content transcript format.
        """
        transcript = []
        for message in messages:
            role = message.get("role") or "user"
            role = str(role).strip().lower() or "user"
            content = self._stringify_openai_content(message.get("content")).strip()
            if not content:
                continue
            transcript.append({"role": role, "content": content})

        return transcript

    def build_prompt_from_transcript(self, messages: List[Dict[str, Any]]) -> str:
        """
        Flatten a normalized transcript into a single prompt.

        This is used both for OpenAI-style HTTP request messages and for
        persisted interactive session history when rebuilding a fresh workflow.

        Args:
            messages: Transcript messages with `role` and `content` keys.

        Returns:
            A single prompt string that contains the conversation transcript and
            instructions to continue as the assistant.
        """
        transcript = [
            f"{message['role'].upper()}:\n{message['content']}"
            for message in self._normalize_transcript_messages(messages)
        ]

        if not transcript:
            return "USER:\nPlease introduce yourself."

        conversation = "\n\n".join(transcript)
        return (
            "Continue the conversation below and respond as the assistant to the "
            "latest user request.\n\n"
            f"{conversation}"
        )

    def build_prompt_from_openai_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Flatten OpenAI-style chat messages into a single prompt.
        """
        return self.build_prompt_from_transcript(messages)

    async def process_openai_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Process OpenAI-style chat messages using the configured strategy.
        """
        async for chunk in self.orchestration_strategy.process_openai_messages(
            self, messages
        ):
            yield chunk

    async def process_message(
        self,
        message: str,
        isolated_session: bool = False,
        record_to_db: bool = True,
        background_poll_session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message using the configured strategy.
        """
        async for chunk in self.orchestration_strategy.process_message(
            self, message, isolated_session=isolated_session
        ):
            yield chunk

    @staticmethod
    def _process_message_error(error: Exception) -> str:
        """
        Format and log a message-processing error.

        Args:
            error: Exception raised while processing a user message.

        Returns:
            User-facing error message.
        """
        error_str = str(error)
        is_auth_error = (
            " 401" in error_str
            or error_str.startswith("401")
            or "Authentication" in error_str
            or "auth_error" in error_str
        )

        if not is_auth_error:
            error_msg = f"Error processing message: {error}"
            LOG.error(error_msg)
            traceback.print_exc()
            return error_msg

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
        return error_msg

    async def _create_run_session(
        self,
        isolated_session: bool,
    ) -> Tuple[Optional[int], AgentSession, Dict[str, int]]:
        """
        Create the agent session used for one message turn.

        Args:
            isolated_session: If True, create a fresh session. Otherwise, copy
                the shared orchestrator session and reserve a turn ID for later
                ordered commit.

        Returns:
            Tuple containing the turn ID, the run session, and provider message
            history lengths captured before streaming.

        Raises:
            RuntimeError: If the shared orchestrator session is not initialized.
        """
        if isolated_session:
            return None, self.planning_agent.create_session(), {}

        async with self._session_lock:
            if self.session is None:
                raise RuntimeError("Orchestrator session not initialized.")

            turn_id = self._next_turn_id
            self._next_turn_id += 1
            run_session = AgentSession.from_dict(self.session.to_dict())
            history_lengths = self._provider_message_lengths(run_session)

        return turn_id, run_session, history_lengths

    @staticmethod
    def _provider_message_lengths(session: AgentSession) -> Dict[str, int]:
        """
        Capture provider message counts before a turn streams.

        Args:
            session: Agent session whose provider state should be inspected.

        Returns:
            Mapping of provider name to the initial message count.
        """
        history_lengths = {}
        for provider_name, provider_state in session.state.items():
            if isinstance(provider_state, dict) and isinstance(
                provider_state.get("messages"), list
            ):
                history_lengths[provider_name] = len(provider_state["messages"])
        return history_lengths

    def _persist_isolated_response(
        self,
        message: str,
        assistant_reply: str,
        record_to_db: bool = True,
        background_poll_session_id: Optional[str] = None,
    ) -> None:
        """
        Persist a response produced from an isolated agent session.

        Args:
            message: User message for the isolated turn.
            assistant_reply: Aggregated assistant response text.

        Returns:
            None.

        Raises:
            Exception: Propagates database persistence failures.
        """
        self.background_tasks.start_background_tool_poll_from_reply_if_needed(
            assistant_reply,
            session_id=background_poll_session_id,
        )

        if not record_to_db:
            return

        if not self.background_tasks.user_message_already_started_background_task(
            message
        ):
            self.session_manager.add_message("user", message)
        if assistant_reply.strip():
            self.session_manager.add_message("assistant", assistant_reply)

    async def _commit_completed_turn(
        self,
        turn_id: Optional[int],
        message: str,
        assistant_reply: str,
        run_session: AgentSession,
        history_lengths: Dict[str, int],
        record_to_db: bool = True,
        background_poll_session_id: Optional[str] = None,
    ) -> None:
        """
        Queue and commit a completed shared-session turn in turn order.

        Args:
            turn_id: Reserved turn ID for this completed turn.
            message: User message for the completed turn.
            assistant_reply: Aggregated assistant response text.
            run_session: Agent session used to process the turn.
            history_lengths: Provider message counts captured before streaming.

        Returns:
            None.

        Raises:
            RuntimeError: If the shared orchestrator session is not initialized.
            Exception: Propagates database persistence failures.
        """
        if turn_id is None:
            raise RuntimeError("Cannot commit an isolated turn to shared session.")

        async with self._session_lock:
            if self.session is None:
                raise RuntimeError("Orchestrator session not initialized.")

            self._completed_turns[turn_id] = {
                "message": message,
                "assistant_reply": assistant_reply,
                "run_session": run_session,
                "history_lengths": history_lengths,
                "record_to_db": record_to_db,
                "background_poll_session_id": background_poll_session_id,
            }

            while self._next_turn_commit_id in self._completed_turns:
                completed = self._completed_turns.pop(self._next_turn_commit_id)
                self._merge_completed_session(
                    completed["run_session"], completed["history_lengths"]
                )
                self._persist_completed_turn(completed)
                self._next_turn_commit_id += 1

    def _merge_completed_session(
        self,
        completed_session: AgentSession,
        history_lengths: Dict[str, int],
    ) -> None:
        """
        Merge one completed run session into the shared orchestrator session.

        Args:
            completed_session: Agent session used by the completed turn.
            history_lengths: Provider message counts captured before streaming.

        Returns:
            None.

        Raises:
            RuntimeError: If the shared orchestrator session is not initialized.
        """
        if self.session is None:
            raise RuntimeError("Orchestrator session not initialized.")

        for provider_name, source_state in completed_session.state.items():
            if not isinstance(source_state, dict):
                self.session.state[provider_name] = copy.deepcopy(source_state)
                continue

            target_state = self.session.state.setdefault(provider_name, {})
            if not isinstance(target_state, dict):
                target_state = {}
                self.session.state[provider_name] = target_state

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
                if key != "messages":
                    target_state[key] = copy.deepcopy(value)

        if completed_session.service_session_id:
            self.session.service_session_id = completed_session.service_session_id

    def _persist_completed_turn(self, completed: Dict[str, Any]) -> None:
        """
        Persist one completed shared-session turn and start MCP polling if needed.

        Args:
            completed: Completed turn metadata from `_completed_turns`.

        Returns:
            None.

        Raises:
            Exception: Propagates database persistence failures.
        """
        message = completed["message"]
        assistant_reply = completed["assistant_reply"]

        self.background_tasks.start_background_tool_poll_from_reply_if_needed(
            assistant_reply,
            session_id=completed.get("background_poll_session_id"),
        )

        if not completed.get("record_to_db", True):
            return

        if not self.background_tasks.user_message_already_started_background_task(
            message
        ):
            self.session_manager.add_message("user", message)
        if assistant_reply.strip():
            self.session_manager.add_message("assistant", assistant_reply)

    async def collect_message_response(
        self,
        message: str,
        isolated_session: bool = False,
        first_tool_call: Optional[asyncio.Event] = None,
        first_tool_state: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Collect a streamed assistant response into a single string.

        Args:
            message: User message to process through the configured strategy.
            isolated_session: If True, process the message with a fresh agent
                session instead of the shared orchestrator session. This is used
                for overlapping background work so concurrent queries do not
                share or mutate the same agent conversation state while another
                turn is still running. This is more conservative than
                append-only transcript merging because `AgentSession` can carry
                provider-specific state beyond messages, and an overlapping turn
                cannot include another unfinished turn's eventual result in its
                context.
            first_tool_call: Optional event set when the streamed response first
                reports a tool call.
            first_tool_state: Optional mutable mapping populated with the first
                tool call name under the `name` key.

        Returns:
            Concatenated assistant response text, including any tool-call notices
            yielded by `process_message`.

        Raises:
            Exception: Propagates unexpected failures from `process_message`.
        """
        response_chunks = []
        async for response_chunk in self.process_message(
            message,
            isolated_session=isolated_session,
        ):
            response_chunks.append(response_chunk)
            if (
                first_tool_call
                and first_tool_state is not None
                and response_chunk.startswith("\n[Calling:")
            ):
                first_tool_state["name"] = (
                    response_chunk.strip()[len("[Calling:") :].rstrip("]").strip()
                )
                first_tool_call.set()
        return "".join(response_chunks)

    async def cleanup(self) -> None:
        """
        Clean up all resources and connections.

        Returns:
            `None`.
        """
        try:
            await self.background_tasks.cleanup()
            await asyncio.wait_for(
                self.exit_stack.aclose(),
                timeout=max(float(self.timeout), 0.1),
            )
            self.specialist_agents.clear()
            self.planning_agent = None
            self.manager_agent = None
            self.session = None
            self._completed_turns.clear()
            self._next_turn_id = 1
            self._next_turn_commit_id = 1
            self._mcp_tools_by_server.clear()
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
