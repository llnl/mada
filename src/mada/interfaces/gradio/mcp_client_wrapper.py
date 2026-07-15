# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MCP client session for Gradio interface.

Provides a client session that connects to MCP servers and processes messages
for the Gradio interface, adapted to work with MADA's architecture.
"""

import logging
import traceback
from typing import AsyncGenerator, Dict, List, Set, Tuple

import gradio as gr

from mada.core.config import AgentConfig, DatabaseConfig, MCPServerConfig, ModelConfig
from mada.core.database import ChatSessionManager
from mada.core.orchestrator import MADAOrchestrator
from mada.interfaces.gradio.utils import create_agent_table, cycle_through_tools

try:
    BaseExceptionGroup
except NameError:
    BaseExceptionGroup = Exception  # fallback for type checkers/runtime


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
    ):
        """
        Initialize the MCP client session.

        Args:
            model_config: Model configuration for MADA
            agents: List of agent configurations
        """
        self.model_config = model_config
        self.agents = agents
        self.database_config = database_config
        self.orchestrator = None
        self.initialized = False
        self.mcp_servers = mcp_servers or {}
        self.session_manager = ChatSessionManager(database_config)
        self.session_bearer_token = None  # Store session bearer token
        self._last_task_status_markdown = ""
        self._display_history: List[Dict[str, str]] = []
        self._displayed_task_results: Set[str] = set()

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
                mcp_servers=self.mcp_servers,  # Placeholder for MCP server config, replace with real config when available
            )
            LOG.info("Orchestrator initialization complete!")

            agent_dict = cycle_through_tools(self.orchestrator.specialist_agents)
            self.initialized = True
            return gr.Button(status_msg, elem_id="green_btn"), create_agent_table(
                self.agents, agent_dict
            )

        except BaseExceptionGroup as eg:
            error_msg = f"Failed to connect to MCP servers: {len(eg.exceptions)} error(s) occurred"
            LOG.error(error_msg, exc_info=True)
            return error_msg, f"Connection failed: {len(eg.exceptions)} errors"
        except Exception as e:
            error_msg = f"Failed to connect to MCP servers: {e}"
            LOG.error(error_msg)
            LOG.error("Full traceback:", exc_info=True)
            return gr.Button(error_msg, variant="stop"), create_agent_table(self.agents)

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
        self._display_history = list(history)

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
            self._display_history = []
            self._displayed_task_results.clear()
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
        Process a user message through the orchestrator.

        Args:
            message: User input message
            history: Chat history (not used in current implementation)
            agent_table: Agent configuration table (not used in current implementation)

        Yields:
            Response chunks from the orchestrator
        """
        if not self.initialized:
            yield "Error: MCP servers not connected. Please connect first."
            return

        if not self.orchestrator:
            yield "Error: Orchestrator not initialized."
            return

        try:
            await self.orchestrator.run_query(message, self.blocking)

        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield error_msg

    async def get_task_status_markdown(self) -> str:
        """
        Render background task state for the Gradio task panel.

        Returns:
            Markdown summary of background task status.
        """
        if not self.orchestrator:
            return "### Task Status\nNo orchestrator connected."

        task_snapshot = await self.orchestrator.get_task_snapshot()
        if not task_snapshot:
            return "### Task Status\nNo background tasks yet."

        lines = ["### Task Status"]
        for task_id, task_state in reversed(list(task_snapshot.items())):
            status = task_state.get("status", "unknown")
            message = task_state.get("message", "").strip() or "(empty)"
            lines.append(f"- `{task_id}` [{status}] `{message[:80]}`")

        return "\n".join(lines)

    async def cleanup(self):
        """Clean up resources."""
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
