# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Agent-as-tool orchestration strategy implementation.
"""

import logging
import traceback
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Tuple

from mada.core.config import AgentConfig, MCPServerConfig, RemoteA2AAgentConfig
from mada.core.orchestration.base_strategy import BaseOrchestrationStrategy

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator


LOG = logging.getLogger(__name__)


class AgentAsToolOrchestrationStrategy(BaseOrchestrationStrategy):
    """
    Planning-agent-plus-`as_tool()` orchestration.

    This strategy initializes specialist agents first, exposes them to the
    planning agent as callable tools, and then creates the session from the
    planning agent.
    """

    mode = "agent-as-tool"

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
        orchestrator.manager_agent = None

    def _remote_a2a_tool_labels(self, orchestrator: "MADAOrchestrator") -> List[str]:
        """
        Build user-facing labels for remote A2A agents.
        """
        labels = []
        for agent_name in orchestrator.a2a_agents:
            card = orchestrator._a2a_agent_cards.get(agent_name, {})
            labels.append(f"A2A: {agent_name} - {card['description']}")
        return labels

    @staticmethod
    def _tool_call_notices_from_chunk(chunk: Any, tool_calls: List[Any]) -> List[str]:
        """
        Return tool-call notices for first appearances of tool calls in a chunk.
        """
        if not hasattr(chunk, "contents") or not chunk.contents:
            return []

        notices = []
        for content in chunk.contents:
            if not hasattr(content, "to_dict"):
                continue

            item = content.to_dict()
            if item.get("type") not in ("function_call", "tool_call"):
                continue

            name = item.get("name")
            if not name:
                continue

            call_id = item.get("call_id")
            call_key = call_id or name
            if call_key in tool_calls:
                continue

            tool_calls.append(call_key)
            notices.append(f"\n[Calling: {name}]\n")

        return notices

    async def _stream_response(
        self,
        orchestrator: "MADAOrchestrator",
        prompt: str,
        *,
        session,
        include_tool_notices: bool,
    ) -> AsyncGenerator[str, None]:
        """
        Stream output from the reusable planning-agent runtime.
        """
        response_started = False
        tool_calls = []
        stream = orchestrator.planning_agent.run(prompt, session=session, stream=True)
        async for chunk in stream:
            if chunk.text:
                response_started = True
                yield chunk.text
                continue

            if not include_tool_notices:
                continue

            for notice in self._tool_call_notices_from_chunk(chunk, tool_calls):
                yield notice

        if not response_started:
            LOG.warning("No text chunks received from planning agent")

    async def initialize(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] | None = None,
        a2a_agents: Dict[str, RemoteA2AAgentConfig] | None = None,
    ) -> Tuple[str, List[str]]:
        """
        Initialize the agent-as-tool orchestration flow end to end.

        The strategy resets orchestrator state, initializes participating
        specialists, creates the planning agent around the successfully active
        specialists, and returns a connection summary plus discovered tools.
        """
        orchestrator.specialist_agents = []
        orchestrator._mcp_tool_count = 0
        orchestrator._agent_descriptions = {}
        participant_configs = orchestrator.resolve_participant_configs(agent_configs)
        orchestrator.mcp_servers = mcp_servers or {}
        orchestrator.a2a_agents = a2a_agents or {}
        failed_a2a_agents = await orchestrator._load_remote_a2a_agent_cards()
        all_tools, failed_servers, failed_agents = await self._initialize_participants(
            orchestrator, participant_configs
        )
        all_tools.extend(self._remote_a2a_tool_labels(orchestrator))
        active_participant_configs = self._resolve_active_participant_configs(
            orchestrator, participant_configs
        )
        self._initialize_planning_agent(
            orchestrator, agent_configs, active_participant_configs
        )
        status = self._build_status(
            orchestrator,
            failed_servers,
            failed_agents,
            failed_a2a_agents,
        )
        LOG.info(status)

        return status, all_tools

    async def process_openai_messages(
        self,
        orchestrator: "MADAOrchestrator",
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Process OpenAI-style chat messages without shared session reuse.
        """
        if not orchestrator.planning_agent:
            yield "Error: Orchestrator not initialized."
            return

        transcript_messages = orchestrator._normalize_transcript_messages(messages)
        prompt = orchestrator.build_prompt_from_transcript(transcript_messages)
        request_session = orchestrator.planning_agent.create_session()

        try:
            async for chunk in self._stream_response(
                orchestrator,
                prompt,
                session=request_session,
                include_tool_notices=True,
            ):
                yield chunk
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield error_msg

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
        Process a user message through the planning agent.
        """
        if not orchestrator.planning_agent:
            yield "Error: Orchestrator not initialized. Call initialize_orchestrator() first."
            return

        aggregated_assistant_reply = ""
        tool_calls = []
        background_task_descriptors = []
        response_started = False

        try:
            (
                turn_id,
                run_session,
                history_lengths,
            ) = await orchestrator._create_run_session(isolated_session)

            prompt = message
            if (
                isolated_session
                and persistence_session_id is not None
                and not stateless_session
            ):
                history = await orchestrator._load_history_for_session(
                    persistence_session_id
                )
                transcript_messages = orchestrator._normalize_transcript_messages(
                    [*history, {"role": "user", "content": message}]
                )
                prompt = orchestrator.build_prompt_from_transcript(transcript_messages)

            stream = orchestrator.planning_agent.run(
                prompt, session=run_session, stream=True
            )
            async for chunk in stream:
                if chunk.text:
                    response_started = True
                    aggregated_assistant_reply += chunk.text
                    yield chunk.text
                    continue

                for notice in self._tool_call_notices_from_chunk(chunk, tool_calls):
                    yield notice

            if not response_started:
                LOG.warning("No text chunks received from planning agent")

            if isolated_session:
                if stateless_session:
                    orchestrator.background_tasks.start_background_tool_poll_from_reply_if_needed(
                        aggregated_assistant_reply,
                        persist_result=False,
                    )
                    for descriptor in background_task_descriptors:
                        orchestrator.background_tasks.start_background_tool_poll_from_reply_if_needed(
                            descriptor,
                            persist_result=False,
                        )
                    return
                await orchestrator._persist_isolated_response(
                    persistence_message or message,
                    aggregated_assistant_reply,
                    background_task_descriptors=background_task_descriptors,
                    session_id=persistence_session_id,
                    record_to_db=record_to_db,
                    background_poll_session_id=background_poll_session_id,
                )
                return

            await orchestrator._commit_completed_turn(
                turn_id,
                persistence_message or message,
                aggregated_assistant_reply,
                run_session,
                history_lengths,
                background_task_descriptors=background_task_descriptors,
                record_to_db=record_to_db,
                background_poll_session_id=background_poll_session_id,
            )
        except Exception as e:
            yield orchestrator._process_message_error(e)
