# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MCP client session for Gradio interface.

Provides a client session that connects to MCP servers and processes messages
for the Gradio interface, adapted to work with MADA's architecture.
"""

import asyncio
import logging
import traceback
from typing import Any, AsyncGenerator, Dict, List, Tuple

import gradio as gr

from mada.core.autonomy import (
    MAX_AUTONOMY_WAIT_SECONDS,
    build_autonomy_enabled_prompt,
    build_autonomy_followup_prompt,
    default_wait_seconds_from_user_message,
    max_autonomy_followups,
    parse_autonomy_control,
    tail_text,
)
from mada.core.config import (
    AgentConfig,
    DatabaseConfig,
    MCPServerConfig,
    ModelConfig,
    OrchestrationConfig,
    RemoteA2AAgentConfig,
)
from mada.core.database import ChatSessionManager
from mada.core.orchestrator import MADAOrchestrator
from mada.interfaces.gradio.utils import create_agent_table, cycle_through_tools

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
        a2a_agents: Dict[str, RemoteA2AAgentConfig] = None,
        orchestration_config: OrchestrationConfig = None,
        blocking: bool = False,
    ):
        """
        Initialize the MCP client session.

        Args:
            model_config: Model configuration for MADA
            agents: List of agent configurations
            database_config: Database configuration for chat history
            mcp_servers: Dictionary of MCP server configurations
            a2a_agents: Dictionary of remote A2A agent configurations
            orchestration_config: Orchestration mode configuration
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
        self._pending_clarifications: Dict[str, str] = {}
        self._autonomy_cancel_events: Dict[str, asyncio.Event] = {}
        self._active_response_sessions: set[str] = set()

    def _get_autonomy_cancel_event(self, session_id: str) -> asyncio.Event:
        event = self._autonomy_cancel_events.get(session_id)
        if event is None:
            event = asyncio.Event()
            self._autonomy_cancel_events[session_id] = event
        return event

    def _persist_turn_to_session(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        """
        Persist a complete user/assistant turn to the session that started it.
        """
        if not session_id or not assistant_message.strip():
            return

        self.session_manager.add_message_to_session(session_id, "user", user_message)
        self.session_manager.add_message_to_session(
            session_id, "assistant", assistant_message
        )

    def request_stop(self, session_label: str | None) -> str:
        """
        Request that any in-flight autonomy loop stop for the given session.
        """
        if session_label:
            session_id = self._extract_id_from_label(session_label)
        else:
            session_id = self.session_manager.current_session_id

        if not session_id:
            return "No active session selected."

        self._get_autonomy_cancel_event(session_id).set()
        return "Stop requested."

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
                mcp_servers=self.mcp_servers,  # Placeholder for MCP server config, replace with real config when available
                a2a_agents=self.a2a_agents,  # Remote A2A agent configurations
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
                table_agents, table_agent_dict
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
        self._pending_clarifications.pop(new_id, None)
        self._autonomy_cancel_events.pop(new_id, None)
        self._active_response_sessions.discard(new_id)
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
        self._pending_clarifications.pop(session_id, None)
        self._autonomy_cancel_events.pop(session_id, None)
        self._active_response_sessions.discard(session_id)
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
            self._pending_clarifications.clear()
            self._autonomy_cancel_events.clear()
            self._active_response_sessions.clear()
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
        autonomy_level: int | float = 0,
        show_autonomy_debug: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message for the Gradio chat UI.

        Args:
            message: User input message
            history: Chat history (not used in current implementation)
            agent_table: Agent configuration table (not used in current implementation)
            autonomy_level: 0 disables autonomous follow-ups; 1-9 enables a
                bounded internal self-followup loop.
            show_autonomy_debug: If True, show autonomy parse and decision
                details inline.

        Returns:
            Async generator that emits response strings for the Gradio chat UI.

        Yields:
            The assistant response buffer for the current chat turn.
        """
        if not self.initialized:
            yield "Error: MCP servers not connected. Please connect first."
            return

        if not self.orchestrator:
            yield "Error: Orchestrator not initialized."
            return

        assistant_buffer = ""
        original_user_message = message
        session_id = self.session_manager.current_session_id
        if not session_id:
            session_id = self.session_manager.create_session_id()
            self.session_manager.create_new_session(session_id)
            self.session_manager.select_session(session_id)
        self._active_response_sessions.add(session_id)

        try:
            level = int(autonomy_level or 0)
            if level < 0:
                level = 0
            if level > 9:
                level = 9

            if level <= 0:
                response = await self.orchestrator.background_tasks.run_query(
                    message,
                    blocking=self.blocking,
                )
                if (
                    response.startswith("[task-")
                    and "Started in background." in response
                ):
                    self._persist_turn_to_session(session_id, message, response)
                yield response
                return

            user_message = original_user_message
            cancel_event = self._get_autonomy_cancel_event(session_id)
            cancel_event.clear()

            pending_clarification = self._pending_clarifications.pop(session_id, None)
            if pending_clarification:
                message = (
                    "Previous question you asked the user:\n"
                    f"{pending_clarification}\n\n"
                    "User answer:\n"
                    f"{user_message}"
                )

            max_followups = max_autonomy_followups(level)

            last_reply = ""
            prompt_for_model = build_autonomy_enabled_prompt(
                message,
                level=level,
                followups_used=0,
                followups_max=max_followups,
            )
            async for chunk in self.orchestrator.process_message(
                prompt_for_model,
                record_to_db=False,
                background_poll_session_id=session_id,
            ):
                last_reply += chunk
                assistant_buffer += chunk
                yield assistant_buffer

            followup_count = 0
            while followup_count < max_followups:
                if cancel_event.is_set():
                    assistant_buffer += "\n\n---\n\n[Autonomy stopped]\n"
                    yield assistant_buffer
                    break

                decision_prompt = (
                    "[INTERNAL CONTROL MESSAGE]\n"
                    "Choose exactly one:\n"
                    "- WAIT: sleep briefly and then issue a follow-up query\n"
                    "- CONTINUE: immediately issue a follow-up query\n"
                    "- ASK: ask the user ONE question (only if truly blocked)\n"
                    "- STOP: stop\n\n"
                    f"Autonomy level: {level} (0=off, 9=most autonomous)\n"
                    f"Follow-up safety cap used: {followup_count}/{max_followups}\n\n"
                    "User request:\n"
                    f"{message}\n\n"
                    "Assistant reply:\n"
                    f"{(last_reply or '').strip()}\n\n"
                    "Assistant transcript so far (tail):\n"
                    f"{tail_text(assistant_buffer, 2000).strip()}\n\n"
                    "Important:\n"
                    "- The system CAN wait in real time. Do NOT claim you cannot wait.\n"
                    "- If the user requested timed repetition (e.g., 'every 20 seconds'), choose WAIT and set AUTONOMY_WAIT_SECONDS accordingly.\n"
                    "- If the user requested 'until sufficient', choose a reasonable stopping point and STOP once you reach it.\n"
                    "- Do not include explanations, code fences, or any other text.\n\n"
                    "Output EXACTLY this format, with no extra text:\n"
                    "AUTONOMY_DECISION=<CONTINUE|WAIT|ASK|STOP>\n"
                    "AUTONOMY_QUERY=<single-line query to run next>\n"
                    "AUTONOMY_WAIT_SECONDS=<integer seconds to wait (WAIT only)>\n"
                    "AUTONOMY_QUESTION=<single-line question for the user>\n"
                    "Rules:\n"
                    "- If decision is CONTINUE, set AUTONOMY_QUERY and leave AUTONOMY_QUESTION empty.\n"
                    "- If decision is WAIT, set AUTONOMY_WAIT_SECONDS and AUTONOMY_QUERY, and leave AUTONOMY_QUESTION empty.\n"
                    "- If decision is ASK, set AUTONOMY_QUESTION and leave AUTONOMY_QUERY empty.\n"
                    "- If decision is STOP, leave all other fields empty.\n"
                )

                decision_text = await self.orchestrator.run_control_prompt(
                    decision_prompt
                )
                (
                    decision,
                    next_query,
                    question,
                    wait_seconds,
                    parse_ok,
                ) = parse_autonomy_control(decision_text)

                decision_preview = " ".join(
                    [
                        f"AUTONOMY_DECISION={decision}",
                        f"AUTONOMY_QUERY={'<set>' if bool(next_query) else ''}".strip(),
                        (
                            f"AUTONOMY_WAIT_SECONDS={wait_seconds if wait_seconds else ''}"
                        ).strip(),
                        (
                            f"AUTONOMY_QUESTION={'<set>' if bool(question) else ''}"
                        ).strip(),
                    ]
                ).strip()
                LOG.info(
                    "Autonomy decision level=%s parse_ok=%s decision=%s next_query=%s question=%s preview=%s",
                    level,
                    parse_ok,
                    decision,
                    "yes" if bool(next_query) else "no",
                    "yes" if bool(question) else "no",
                    decision_preview or "<empty>",
                )

                if show_autonomy_debug:
                    assistant_buffer += (
                        "\n\n---\n\n"
                        f"[Autonomy debug] level={level} parse_ok={parse_ok} decision={decision} "
                        f"next_query={'yes' if bool(next_query) else 'no'} wait_seconds={wait_seconds or 0} "
                        f"question={'yes' if bool(question) else 'no'}\n"
                        f"[Autonomy debug] decision_preview={decision_preview or '<empty>'}\n"
                    )
                    yield assistant_buffer

                if decision == "ASK" and question:
                    assistant_buffer += f"\n\n{question}"
                    yield assistant_buffer
                    self._pending_clarifications[session_id] = question
                    break

                if decision == "STOP":
                    break

                if decision not in ("CONTINUE", "WAIT") or not next_query:
                    break

                if decision == "WAIT":
                    if wait_seconds <= 0:
                        wait_seconds = default_wait_seconds_from_user_message(
                            user_message
                        )
                    if wait_seconds > MAX_AUTONOMY_WAIT_SECONDS:
                        wait_seconds = MAX_AUTONOMY_WAIT_SECONDS

                    assistant_buffer += "\n\n---\n\n[Autonomous wait]\n\n"
                    assistant_buffer += f"[Wait seconds] {wait_seconds}\n"
                    assistant_buffer += f"[Autonomous query] {next_query}\n"
                    yield assistant_buffer

                    try:
                        await asyncio.wait_for(
                            cancel_event.wait(), timeout=wait_seconds
                        )
                        assistant_buffer += "\n\n---\n\n[Autonomy stopped]\n"
                        yield assistant_buffer
                        break
                    except asyncio.TimeoutError:
                        pass
                else:
                    assistant_buffer += "\n\n---\n\n[Autonomous follow-up]\n\n"
                    assistant_buffer += f"[Autonomous query] {next_query}\n\n"
                    yield assistant_buffer

                previous_reply = last_reply
                last_reply = ""
                followup_prompt = build_autonomy_followup_prompt(
                    next_query,
                    original_request=message,
                    last_reply=previous_reply,
                    assistant_buffer=assistant_buffer,
                    level=level,
                    followups_used=followup_count,
                    followups_max=max_followups,
                )
                async for chunk in self.orchestrator.process_message(
                    followup_prompt,
                    record_to_db=False,
                    background_poll_session_id=session_id,
                ):
                    last_reply += chunk
                    assistant_buffer += chunk
                    yield assistant_buffer

                followup_count += 1

            if assistant_buffer.strip():
                self._persist_turn_to_session(
                    session_id, user_message, assistant_buffer
                )

        except asyncio.CancelledError:
            if session_id:
                self._get_autonomy_cancel_event(session_id).set()
            marker = "\n\n---\n\n[Generation cancelled]\n"
            assistant_buffer = (assistant_buffer or "") + marker
            if assistant_buffer.strip() and int(autonomy_level or 0) > 0:
                self._persist_turn_to_session(
                    session_id, original_user_message, assistant_buffer
                )
            yield assistant_buffer
            return
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            if assistant_buffer.strip():
                assistant_buffer += f"\n\n{error_msg}"
                assistant_reply = assistant_buffer
            else:
                assistant_reply = error_msg
            if int(autonomy_level or 0) > 0:
                self._persist_turn_to_session(
                    session_id, original_user_message, assistant_reply
                )
            yield assistant_reply
        finally:
            if session_id:
                self._active_response_sessions.discard(session_id)

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
        session_id = self.session_manager.current_session_id
        if session_id and session_id in self._active_response_sessions:
            return gr.skip(), task_status
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
