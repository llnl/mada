# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MCP client session for Gradio interface.

Provides a client session that connects to MCP servers and processes messages
for the Gradio interface, adapted to work with MADA's architecture.
"""

import logging
import traceback
from typing import Any, AsyncGenerator, Dict, List, Tuple

import gradio as gr

from mada.core.config import (
    AgentConfig,
    DatabaseConfig,
    MCPServerConfig,
    ModelConfig,
    OrchestrationConfig,
    RemoteA2AAgentConfig,
)
from mada.core.background_tasks import is_background_task_start_ack
from mada.core.database import ChatSessionManager
from mada.core.orchestrator import MADAOrchestrator
from mada.interfaces.gradio.utils import create_agent_table, cycle_through_tools
from mada.core.skills.skill_registry import SkillRegistry

LOG = logging.getLogger("mada-gradio")


class MCPGradioClientSession:
    """
    MCP client session for Gradio interface.

    This class handles connecting to MCP servers and processing messages
    for the Gradio interface using the existing MADAOrchestrator.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        agents: List[AgentConfig],
        database_config: DatabaseConfig,
        mcp_servers: MCPServerConfig = None,
        skill_registry: SkillRegistry = None,
        skill_tools: List[Any] = None,
        a2a_agents: Dict[str, RemoteA2AAgentConfig] = None,
        orchestration_config: OrchestrationConfig = None,
        blocking: bool = False,
    ):
        """
        Initialize the MCP client session.

        Args:
            model_config: Model configuration for MADA.
            agents: List of agent configurations.
            database_config: Configuration for the chat history database.
            mcp_servers: MCP server configurations available to the agents.
            skill_registry: Registry of manifest-based skills to advertise to
                the planning agent. An empty registry is used when omitted.
            skill_tools: Runtime tools for loading skills and running skill
                scripts.
            a2a_agents: Remote A2A agent configurations available to the
                planning agent.
            orchestration_config: Orchestration mode and participant settings.
            blocking: If True, wait for each response inline. If False,
                submit queries in the background and return immediately.
        """
        self.model_config = model_config
        self.agents = agents
        self.database_config = database_config
        self.blocking = blocking
        self.orchestrator = None
        self.initialized = False
        self.mcp_servers = mcp_servers or {}
        self.a2a_agents = a2a_agents or {}
        self.orchestration_config = orchestration_config or OrchestrationConfig()
        self.session_manager = ChatSessionManager(database_config)
        self.session_bearer_token = None  # Store session bearer token
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_tools = list(skill_tools or [])

    async def connect_servers(
        self, agent_table: gr.Dataframe, request: gr.Request
    ) -> Tuple[str, str]:
        """
        Connect to MCP servers based on the agent table configuration.

        Args:
            agent_table: Gradio DataFrame containing agent configurations
            request: Gradio Request object containing HTTP request information.
                Note: This parameter is automatically injected by Gradio when the
                function signature includes `request: gr.Request`. It does not need
                to be added to the additional_inputs list.

        Returns:
            Tuple of (status_message, tools_description)
        """
        # Log request information for debugging/auditing
        if request:
            LOG.info(f"Request headers: {dict(request.headers)}")
            if request.client:
                LOG.info(f"Client IP: {request.client.host}")

            # Store request wormhole community subtoken for use in session
            if hasattr(request, "headers") and "x-subtoken" in request.headers:
                token = request.headers["x-subtoken"]
                self.session_bearer_token = token

        try:
            LOG.info("Starting MCP server connection...")
            if self.orchestrator is None:
                LOG.info("Creating orchestrator instance...")
                self.orchestrator = MADAOrchestrator(
                    self.model_config,
                    self.database_config,
                    session_manager=self.session_manager,
                    skill_registry=self.skill_registry,
                    skill_tools=self.skill_tools,
                    orchestration_config=self.orchestration_config,
                    bearer_token=self.session_bearer_token,
                )
                # Enter the async context manager (required for proper setup)
                LOG.info("Entering orchestrator async context...")
                await self.orchestrator.__aenter__()
                LOG.info("Orchestrator context entered successfully")

            # Initialize orchestrator with agents
            LOG.info(
                f"Initializing orchestrator with {len(self.agents)} agents and {len(self.mcp_servers)} MCP servers..."
            )
            status_msg, tools = await self.orchestrator.initialize_orchestrator(
                agent_configs=self.agents,  # Use provided agents
                mcp_servers=self.mcp_servers,
                a2a_agents=self.a2a_agents,
            )
            LOG.info("Orchestrator initialization complete!")
            status_msg = (
                f"Orchestration mode: {self.orchestration_config.mode} | {status_msg}"
            )

            agent_dict = cycle_through_tools(self.orchestrator.specialist_agents)
            configured_participants = self.orchestration_config.participants
            if configured_participants is None:
                active_agents = [
                    agent for agent in self.agents if agent.agent_name in agent_dict
                ]
                table_agents = active_agents or self.agents
                table_agent_dict = agent_dict if active_agents else None
            else:
                table_agents = [
                    agent
                    for agent in self.agents
                    if agent.agent_name in configured_participants
                    and agent.agent_name in agent_dict
                ]
                table_agent_dict = agent_dict
            self.initialized = True
            return gr.Button(status_msg, elem_id="green_btn"), create_agent_table(
                table_agents,
                table_agent_dict,
                self.a2a_agents,
            )

        except BaseExceptionGroup as eg:
            error_msg = f"Failed to connect to MCP servers: {len(eg.exceptions)} error(s) occurred"
            LOG.error(error_msg, exc_info=True)
            return error_msg, f"Connection failed: {len(eg.exceptions)} errors"
        except Exception as e:
            error_msg = f"Failed to connect to MCP servers: {e}"
            LOG.error(error_msg)
            LOG.error("Full traceback:", exc_info=True)
            return gr.Button(error_msg, variant="stop"), create_agent_table(
                self.agents,
                a2a_agents=self.a2a_agents,
            )

    def list_sessions(self) -> List[str]:
        """
        List all sessions available and create a list of labels for each session.

        Returns:
            A list of labels of the form "timestamp | ID" for each session.
        """
        sessions = self.session_manager.list_sessions()
        return [
            f"{ts.isoformat(sep=' ', timespec='seconds')} | {sid}"
            for sid, ts in sessions
        ]

    def get_session_choices(self) -> List[str]:
        """
        List all chat sessions.

        This is used on startup of the Gradio app.

        Returns:
            A list of chat session labels of the form "timestamp | session ID".
        """
        return self.list_sessions()

    def update_session_choices(self) -> gr.update:
        """
        Update the list of chat sessions.

        Returns:
            A gradio update object with new chat session choices.
        """
        return gr.update(choices=self.list_sessions(), value=None)

    def create_new_session(self) -> Tuple[gr.update, List]:
        """
        Create a new chat session.

        Returns:
            A tuple containing a gradio update object for updating the
                sessions list and an empty chat history.
        """
        new_id = self.session_manager.create_session_id()
        self.session_manager.create_new_session(new_id)
        self.session_manager.select_session(new_id)
        updated_sessions = self.list_sessions()
        return gr.update(
            choices=updated_sessions, value=None
        ), []  # update sessions list, empty chat history

    def _extract_id_from_label(self, session_label: str) -> str:
        """
        Extract session id from label.

        Args:
            session_label (str): The label of the session of the form "timestamp | ID"

        Returns:
            The session ID.
        """
        try:
            session_id = session_label.split("|", 1)[1].strip()
        except Exception:
            session_id = session_label.strip()

        return session_id

    def select_session(self, session_label: str) -> List[Dict[str, str]]:
        """
        Given a label like 'timestamp | session_id', select that session and return its history.

        Args:
            session_label (str): The label of the chat session that the user wants to select.

        Returns:
            A list of the chat history.
        """
        if not session_label:
            return []  # nothing selected

        # Extract session id from label
        session_id = self._extract_id_from_label(session_label)

        history = self.session_manager.select_session(session_id)

        return history

    def delete_session(self, session_label: str) -> Tuple[gr.update, List]:
        """
        Delete a chat session.

        Args:
            session_label (str): The label of the chat session that the user wants to delete.

        Returns:
            A tuple containing a gradio update object for updating the
                sessions list and an empty chat history.
        """
        # Handle case where no session is selected
        if not session_label:
            # Return current state unchanged
            updated_sessions = self.list_sessions()
            return gr.update(choices=updated_sessions, value=None), []

        session_id = self._extract_id_from_label(session_label)
        self.session_manager.delete_session(session_id)
        updated_sessions = self.list_sessions()
        return gr.update(
            choices=updated_sessions, value=None
        ), []  # update sessions list, clear chat history

    def delete_all_sessions(self) -> Tuple[gr.update, List]:
        """
        Delete all chat sessions with proper error handling.

        Returns:
            A tuple containing:
            - gradio update object for updating the sessions list
            - chat history (empty if successful, unchanged if error)
        """
        try:
            LOG.info("Attempting to delete all sessions")
            self.session_manager.delete_all_sessions(confirm=False)
            LOG.info("Successfully deleted all sessions")
            return gr.update(choices=[], value=None), []
        except Exception as e:
            LOG.error(f"Failed to delete all sessions: {e}")
            # Return current state unchanged if deletion fails
            current_sessions = self.list_sessions()
            return gr.update(choices=current_sessions), []

    async def process_message(
        self,
        message: str,
        history: List,
        agent_table: gr.Dataframe,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message through the orchestrator's background task manager.

        Args:
            message: User input message
            history: Chat history (not used in current implementation)
            agent_table: Agent configuration table (not used in current implementation)

        Returns:
            Async generator that emits response strings for the Gradio chat UI.

        Yields:
            The assistant response, or a background-task start acknowledgement.
        """
        if not self.initialized:
            yield "Error: MCP servers not connected. Please connect first."
            return

        if not self.orchestrator:
            yield "Error: Orchestrator not initialized."
            return

        try:
            response = await self.orchestrator.background_tasks.run_query(
                message, blocking=self.blocking
            )
            if is_background_task_start_ack(response):
                self.session_manager.add_message("user", message)
                self.session_manager.add_message("assistant", response)
            yield response

        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield error_msg

    async def get_task_status_markdown(self) -> str:
        """
        Render background task state for the Gradio task panel.

        Returns:
            Markdown text describing current background task status.
        """
        if not self.orchestrator:
            return "### Task Status\nNo orchestrator connected."

        task_snapshot = await self.orchestrator.background_tasks.get_task_snapshot()
        if not task_snapshot:
            return "### Task Status\nNo background tasks yet."

        lines = ["### Task Status"]
        for task_id, task_state in reversed(list(task_snapshot.items())):
            status = task_state.get("status", "unknown")
            tool_name = task_state.get("tool_name")
            if tool_name:
                lines.append(f"- `{task_id}` [{status}] {tool_name}")
            else:
                lines.append(f"- `{task_id}` [{status}]")

        return "\n".join(lines)

    async def refresh_chat_and_task_status(self, history: List[Any]) -> Tuple[Any, str]:
        """
        Refresh chat history and background task status for the Gradio UI.

        Args:
            history: Current Gradio chat history.

        Returns:
            Tuple containing the refreshed chat history or `gr.skip()`, and the
            task status markdown.

        Raises:
            Exception: Propagates unexpected chat-history load failures.
        """
        task_status = await self.get_task_status_markdown()
        if (
            not self.orchestrator
            or await self.orchestrator.background_tasks.count_pending_tasks() > 0
        ):
            return gr.skip(), task_status

        persisted_history = self.session_manager.load_history()
        if persisted_history == list(history):
            return gr.skip(), task_status

        return persisted_history, task_status

    async def cleanup(self) -> None:
        """
        Clean up resources.

        Returns:
            None. Cleanup errors are logged and suppressed.
        """
        if self.orchestrator:
            try:
                await self.orchestrator.__aexit__(None, None, None)
            except BaseExceptionGroup as eg:
                LOG.warning(
                    f"Multiple errors during cleanup ({len(eg.exceptions)} errors, suppressed)"
                )
            except Exception as e:
                LOG.error(f"Error during cleanup: {e}")
        self.initialized = False
