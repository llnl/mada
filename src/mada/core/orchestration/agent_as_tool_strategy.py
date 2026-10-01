# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Agent-as-tool orchestration strategy implementation.
"""

import json
import logging
import re
import traceback
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Tuple

from mada.core.config import AgentConfig, MCPServerConfig, RemoteA2AAgentConfig
from mada.core.media import (
    ImageAttachment,
    extract_image_attachments,
    strip_image_payload_text,
)
from mada.core.orchestration.base_strategy import BaseOrchestrationStrategy
from mada.core.orchestration.stream_events import (
    InternalImageSignal,
    InternalResponseReplacement,
)

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

    _IMAGE_STREAM_START_RE = re.compile(
        r"!\[[^\]]*\]\(\s*data:image/(?:gif|jpeg|png|webp);base64,|"
        r"data:image/(?:gif|jpeg|png|webp);base64,|"
        r"(?:\[\s*)?\{\s*[\"'](?:type|mimeType|media_type)[\"']\s*:\s*[\"']"
        r"(?:image|data|uri)|"
        r"(?:\[\s*)?\{(?=[^{}]*[\"'](?:type|mimeType|media_type)[\"']\s*:\s*"
        r"[\"'](?:image|data|uri))",
        re.IGNORECASE,
    )
    _IMAGE_STREAM_PREFIXES = (
        "data:image/",
        '{"type":"image',
        '{"mimeType":"image/',
        '{"media_type":"image/',
        '[{"type":"image',
        '[{"mimeType":"image/',
        '[{"media_type":"image/',
    )

    @classmethod
    def _serialized_image_start(cls, text: str) -> int | None:
        """Return the start of a possibly split serialized image payload."""
        matches = [match.start() for match in cls._IMAGE_STREAM_START_RE.finditer(text)]
        for prefix in cls._IMAGE_STREAM_PREFIXES:
            minimum_length = 5 if prefix == "data:image/" else 4
            for length in range(minimum_length, min(len(prefix), len(text)) + 1):
                start = len(text) - length
                if text.endswith(prefix[:length]) and not (
                    prefix == "data:image/" and start > 0 and text[start - 1].isalnum()
                ):
                    matches.append(start)

        if not matches:
            return None

        start = min(matches)
        data_start = text.find("data:image/", start)
        if data_start >= 0:
            markdown_start = text.rfind("![", start, data_start + 1)
            if markdown_start >= 0:
                markdown_prefix = text[markdown_start:data_start]
                if re.fullmatch(r"!\[[^\]]*\]\(\s*", markdown_prefix):
                    start = markdown_start
        return start

    @classmethod
    def _safe_text_chunk(cls, pending_text: str, chunk_text: str) -> tuple[str, str]:
        """Return safe text to emit and a possibly incomplete image suffix."""
        combined = pending_text + chunk_text
        start = cls._serialized_image_start(combined)
        if start is None:
            return combined, ""

        payload = combined[start:]
        attachments = extract_image_attachments(payload)
        if not attachments and not cls._serialized_image_is_complete(payload):
            return combined[:start], payload

        return strip_image_payload_text(combined, strip_outer_whitespace=False), ""

    @staticmethod
    def _serialized_image_is_complete(payload: str) -> bool:
        """Return whether a possible serialized image has a closing boundary."""
        stripped = payload.lstrip()
        if stripped.startswith("!["):
            return ")" in stripped
        if stripped.startswith(("{", "[")):
            try:
                json.JSONDecoder().raw_decode(stripped)
            except json.JSONDecodeError:
                return False
            return True
        return bool(
            re.search(
                r"data:image/(?:gif|jpeg|png|webp);base64,[^\s)]+",
                stripped,
                re.IGNORECASE,
            )
        )

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

    def _skill_tool_labels(self, orchestrator: "MADAOrchestrator") -> List[str]:
        """
        Build user-facing labels for manifest-based skill tools.
        """
        return [
            f"Skill Tool: {getattr(tool, 'name', getattr(tool, '__name__', 'unknown'))}"
            for tool in orchestrator.skill_tools
        ]

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
        text_chunks = []
        pending_text = ""
        emitted_text = ""
        image_digests: set[str] = set()
        stream = orchestrator.planning_agent.run(prompt, session=session, stream=True)
        async for chunk in stream:
            for attachment in extract_image_attachments(chunk):
                if attachment.digest in image_digests:
                    continue
                image_digests.add(attachment.digest)
                yield InternalImageSignal(attachment)

            if chunk.text:
                text_chunks.append(chunk.text)
                for attachment in extract_image_attachments(pending_text + chunk.text):
                    if attachment.digest in image_digests:
                        continue
                    image_digests.add(attachment.digest)
                    yield InternalImageSignal(attachment)
                safe_text, pending_text = self._safe_text_chunk(
                    pending_text, chunk.text
                )
                if safe_text:
                    response_started = True
                    emitted_text += safe_text
                    yield safe_text
                continue

            if not include_tool_notices:
                continue

            for notice in self._tool_call_notices_from_chunk(chunk, tool_calls):
                yield notice

        assembled_text = "".join(text_chunks)
        for attachment in extract_image_attachments(assembled_text):
            if attachment.digest in image_digests:
                continue
            image_digests.add(attachment.digest)
            yield InternalImageSignal(attachment)

        text = strip_image_payload_text(assembled_text)
        if text.strip() != emitted_text.strip():
            if text.startswith(emitted_text):
                remainder = text[len(emitted_text) :]
                if remainder:
                    response_started = True
                    emitted_text += remainder
                    yield remainder
            elif text:
                response_started = True
                yield InternalResponseReplacement(text)
        if text and not response_started:
            response_started = True

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
        all_tools.extend(self._skill_tool_labels(orchestrator))
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
        image_attachments: list[ImageAttachment] = []
        image_digests: set[str] = set()
        text_chunks: list[str] = []
        pending_text = ""
        emitted_text = ""
        capture_token = None
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

            if hasattr(orchestrator, "_begin_image_capture"):
                capture_token, captured_images = orchestrator._begin_image_capture()
            else:
                captured_images = []
            captured_image_index = 0
            stream = orchestrator.planning_agent.run(
                prompt, session=run_session, stream=True
            )
            async for chunk in stream:
                available_images = [
                    *captured_images[captured_image_index:],
                    *extract_image_attachments(chunk),
                ]
                captured_image_index = len(captured_images)
                for attachment in available_images:
                    if attachment.digest in image_digests:
                        continue
                    image_digests.add(attachment.digest)
                    image_attachments.append(attachment)
                    yield InternalImageSignal(attachment)

                if chunk.text:
                    text_chunks.append(chunk.text)
                    for attachment in extract_image_attachments(
                        pending_text + chunk.text
                    ):
                        if attachment.digest in image_digests:
                            continue
                        image_digests.add(attachment.digest)
                        image_attachments.append(attachment)
                        yield InternalImageSignal(attachment)
                    safe_text, pending_text = self._safe_text_chunk(
                        pending_text, chunk.text
                    )
                    if safe_text:
                        response_started = True
                        emitted_text += safe_text
                        yield safe_text
                    continue

                for notice in self._tool_call_notices_from_chunk(chunk, tool_calls):
                    yield notice

            for attachment in captured_images[captured_image_index:]:
                if attachment.digest in image_digests:
                    continue
                image_digests.add(attachment.digest)
                image_attachments.append(attachment)
                yield InternalImageSignal(attachment)

            assembled_text = "".join(text_chunks)
            for attachment in extract_image_attachments(assembled_text):
                if attachment.digest in image_digests:
                    continue
                image_digests.add(attachment.digest)
                image_attachments.append(attachment)
                yield InternalImageSignal(attachment)

            aggregated_assistant_reply = strip_image_payload_text(assembled_text)
            if aggregated_assistant_reply.strip() != emitted_text.strip():
                if aggregated_assistant_reply.startswith(emitted_text):
                    remainder = aggregated_assistant_reply[len(emitted_text) :]
                    if remainder:
                        response_started = True
                        emitted_text += remainder
                        yield remainder
                elif aggregated_assistant_reply:
                    response_started = True
                    yield InternalResponseReplacement(aggregated_assistant_reply)

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
                    message,
                    aggregated_assistant_reply,
                    image_attachments=image_attachments,
                    background_task_descriptors=background_task_descriptors,
                    session_id=persistence_session_id,
                )
                return

            await orchestrator._commit_completed_turn(
                turn_id,
                message,
                aggregated_assistant_reply,
                run_session,
                history_lengths,
                image_attachments=image_attachments,
                background_task_descriptors=background_task_descriptors,
            )
        except Exception as e:
            yield orchestrator._process_message_error(e)
        finally:
            if capture_token is not None and hasattr(
                orchestrator, "_end_image_capture"
            ):
                orchestrator._end_image_capture(capture_token)
