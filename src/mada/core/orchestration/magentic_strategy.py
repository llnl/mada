# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Magentic orchestration strategy implementation.
"""

import asyncio
import inspect
import json
import logging
import time
import traceback
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Tuple

from agent_framework import Agent, Message

from mada.core.config import AgentConfig, MCPServerConfig, RemoteA2AAgentConfig
from mada.core.orchestration.base_strategy import BaseOrchestrationStrategy
from mada.core.orchestration.stream_events import (
    InternalError,
    InternalResponseReplacement,
    InternalToolCallSignal,
    response_replacement,
)

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator

try:
    from agent_framework import MagenticBuilder
except ImportError:  # pragma: no cover - depends on installed agent framework version
    try:
        from agent_framework.orchestrations import (  # type: ignore[attr-defined]
            MagenticBuilder,
        )
    except ImportError:  # pragma: no cover
        MagenticBuilder = None


LOG = logging.getLogger(__name__)


class MagenticOrchestrationStrategy(BaseOrchestrationStrategy):
    """
    Peer specialist group chat coordinated by a hidden manager agent.
    """

    mode = "magentic"
    _MAX_SYNTHESIS_TOOL_WORK_CHARS = 32_000
    _MAX_SYNTHESIS_TOOL_RESULT_CHARS = 8_000
    _MAX_SYNTHESIS_TIMEOUT_SECONDS = 5.0
    _SYNTHESIS_TIMEOUT_FRACTION = 0.2
    _MIN_SYNTHESIS_TIMEOUT_SECONDS = 0.1

    def _create_manager_agent(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        participant_configs: List[AgentConfig],
    ) -> Agent:
        """
        Create the hidden manager agent used by Magentic orchestration.
        """
        team_description = orchestrator._generate_team_description(participant_configs)
        planning_cfg = orchestrator._get_planning_agent_config(agent_configs)

        if planning_cfg and planning_cfg.mcp_servers:
            LOG.warning(
                "PlanningAgent MCP server support is not implemented. "
                "MCP servers listed in PlanningAgent config will be ignored."
            )

        if planning_cfg and planning_cfg.instructions:
            base_instructions = planning_cfg.instructions.strip()
        else:
            base_instructions = """You are the hidden manager for MADA's Magentic orchestration mode.

Coordinate the specialist agents as peers, track plan and progress internally,
and produce the final response for the user."""

        instructions = f"""{base_instructions}

Specialist participants:
{team_description}

Guidelines:
- Coordinate the specialists as a peer conversation
- Select only the specialists needed for the current request
- Re-plan at most once when the current approach stalls or conflicts
- Do not repeat a completed tool call or ask a specialist for the same result
- Stop deliberating as soon as the request is answerable
- Keep internal planning and progress chatter out of the final user-facing answer
- Produce the final synthesized assistant response for the user
"""

        agent_kwargs = {}
        if planning_cfg:
            agent_kwargs.update(planning_cfg.extra)

        agent_name = planning_cfg.agent_name if planning_cfg else "PlanningAgent"
        return orchestrator.model_client.as_agent(
            name=agent_name,
            instructions=instructions,
            **agent_kwargs,
        )

    @staticmethod
    def _clone_agent(agent: Agent) -> Agent:
        """
        Create a runtime-local Agent while reusing its client and tool handles.

        MagenticBuilder creates fresh executors for each workflow, but those
        executors still hold the Agent instance supplied to the builder.  Agent
        instances lazily acquire history providers and retain other mutable
        invocation state, so sharing them across overlapping background
        workflows can cross-wire a tool result.  The model client and connected
        tool handles are safe to reuse; the Agent wrapper and its option/tool
        containers are not.
        """
        missing = object()
        default_options = dict(getattr(agent, "default_options", {}) or {})
        # This is a runtime option,
        # keeping it scoped to the cloned agents rather than changing the shared
        # client or source agent configuration.
        default_options["store"] = False
        default_options.pop("conversation_id", None)

        # Keep this compatible with Agent Framework releases that have added or
        # removed optional Agent constructor fields.  Inspect the signature
        # before moving values out of default_options so older layouts retain
        # their tools and instructions.
        try:
            parameters = inspect.signature(Agent).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_arbitrary_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        def accepts_constructor_option(name: str) -> bool:
            return accepts_arbitrary_keywords or not parameters or name in parameters

        def normalize_tools(value: Any) -> list[Any]:
            if value is None:
                return []
            if isinstance(value, (list, tuple, set, frozenset)):
                return list(value)
            return [value]

        # Agent Framework has used both layouts for these two values.  In some
        # releases they are constructor fields (`agent.tools` and
        # `agent.instructions`); in others they are kept in default_options.
        # Prefer an explicitly present top-level value, including an empty tool
        # list, and only use default_options as a compatibility fallback.
        top_level_tools = getattr(agent, "tools", missing)
        if top_level_tools is missing or top_level_tools is None:
            tools = normalize_tools(default_options.get("tools"))
        else:
            tools = normalize_tools(top_level_tools)

        top_level_instructions = getattr(agent, "instructions", missing)
        if top_level_instructions is missing or top_level_instructions is None:
            instructions = default_options.get("instructions")
        else:
            instructions = top_level_instructions

        # MCP tools can be exposed separately on some framework versions and
        # together with `tools` on others.  Reuse the handles, but never give a
        # runtime duplicate entries for the same tool object.
        seen_tool_ids = {id(tool) for tool in tools}
        for tool in list(getattr(agent, "mcp_tools", []) or []):
            if id(tool) not in seen_tool_ids:
                tools.append(tool)
                seen_tool_ids.add(id(tool))

        if accepts_constructor_option("tools"):
            default_options.pop("tools", None)
        elif top_level_tools is not missing or "tools" in default_options:
            default_options["tools"] = tools

        if accepts_constructor_option("instructions"):
            default_options.pop("instructions", None)
        elif top_level_instructions is not missing or "instructions" in default_options:
            default_options["instructions"] = instructions

        constructor_options = {
            "client": agent.client,
            "instructions": instructions,
            "id": getattr(agent, "id", None),
            "name": getattr(agent, "name", None),
            "description": getattr(agent, "description", None),
            "tools": tools,
            "default_options": default_options,
            "context_providers": list(getattr(agent, "context_providers", []) or []),
            "middleware": (
                list(agent.middleware)
                if getattr(agent, "middleware", None) is not None
                else None
            ),
            "require_per_service_call_history_persistence": getattr(
                agent, "require_per_service_call_history_persistence", False
            ),
            "compaction_strategy": getattr(agent, "compaction_strategy", None),
            "tokenizer": getattr(agent, "tokenizer", None),
            "additional_properties": dict(
                getattr(agent, "additional_properties", {}) or {}
            ),
        }

        # Passing only accepted fields avoids falling back to the shared Agent
        # (which would defeat per-workflow isolation) merely because an
        # optional field is unknown.
        if not accepts_arbitrary_keywords and parameters:
            constructor_options = {
                key: value
                for key, value in constructor_options.items()
                if key in parameters
            }

        clone = Agent(**constructor_options)
        for attribute in ("id", "name", "description"):
            value = getattr(agent, attribute, None)
            if value is not None and getattr(clone, attribute, None) != value:
                MagenticOrchestrationStrategy._set_agent_metadata(
                    clone, attribute, value
                )
        return clone

    def _runtime_agent(self, agent: Agent) -> Agent:
        """Return a per-workflow copy for framework Agent instances."""
        # Keep compatibility with custom SupportsAgentRun implementations, but
        # never silently share a real Agent after a cloning failure: that would
        # bring back the cross-workflow history race this wrapper prevents.
        if not isinstance(agent, Agent):
            LOG.debug("Using custom Magentic agent %r", agent)
            return agent
        return self._clone_agent(agent)

    def _create_builder(self, orchestrator: "MADAOrchestrator"):
        """
        Create a fresh Magentic builder for a request.
        """
        if MagenticBuilder is None:
            raise RuntimeError(
                "Magentic orchestration requires agent_framework MagenticBuilder support"
            )

        builder_kwargs = {
            "participants": [
                self._runtime_agent(agent) for agent in orchestrator.specialist_agents
            ],
            "manager_agent": self._runtime_agent(orchestrator.manager_agent),
        }

        # Keep native replanning behind the outer convergence guard. Its stall
        # counter has different boundary semantics, so it must not supersede
        # MADA's synthesis decision.
        configured = getattr(orchestrator, "orchestration", None)
        max_rounds = max(1, int(getattr(configured, "max_rounds", 4)))
        max_stalls = max(1, int(getattr(configured, "max_stalls", 1)))
        try:
            builder_parameters = inspect.signature(MagenticBuilder).parameters
        except (TypeError, ValueError):
            builder_parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in builder_parameters.values()
        )
        if "max_stall_count" in builder_parameters or accepts_kwargs:
            builder_kwargs["max_stall_count"] = max(max_rounds, max_stalls) + 1

        return MagenticBuilder(**builder_kwargs)

    @staticmethod
    def _set_agent_metadata(agent: Agent, attribute: str, value: str) -> None:
        """
        Best-effort assignment for Agent Framework metadata attributes.
        """
        try:
            setattr(agent, attribute, value)
        except (AttributeError, TypeError):
            try:
                object.__setattr__(agent, attribute, value)
            except (AttributeError, TypeError):
                LOG.warning(
                    "Unable to set Magentic participant %s on agent %s",
                    attribute,
                    getattr(agent, "name", "<unknown>"),
                )

    def _preserve_participant_metadata(
        self,
        orchestrator: "MADAOrchestrator",
        participant_configs: List[AgentConfig],
    ) -> None:
        """
        Preserve configured participant IDs and descriptions for Magentic routing.
        """
        config_by_name = {config.agent_name: config for config in participant_configs}
        for agent in orchestrator.specialist_agents:
            config = config_by_name.get(getattr(agent, "name", ""))
            if not config:
                continue
            self._set_agent_metadata(agent, "id", config.agent_name)
            self._set_agent_metadata(agent, "description", config.description)

    def _build_runtime(self, orchestrator: "MADAOrchestrator"):
        """
        Build a runnable Magentic workflow instance.
        """
        builder = self._create_builder(orchestrator)

        for method_name in ("build", "create_workflow", "create"):
            method = getattr(builder, method_name, None)
            if callable(method):
                return method()

        return builder

    _IGNORED_EVENT_TYPES = frozenset(
        {
            "plan",
            "progress",
            "replan",
            "checkpoint",
            "function_call",
            "tool_call",
        }
    )

    _TEXT_KEYS = (
        "final_output",
        "final_response",
        "assistant_response",
        "response",
        "output",
        "content",
        "text",
        "contents",
        "items",
        "messages",
        "data",
    )

    _TOOL_CALL_KEYS = (
        "function_call",
        "tool_call",
        "function_calls",
        "tool_calls",
    )

    _TOOL_COLLECTION_KEYS = ("tool_calls", "tools", "function_calls", "functions")
    _TOOL_RECURSION_KEYS = ("contents", "items", "messages", "data")

    _FINAL_EVENT_TYPES = frozenset(
        {"final", "final_output", "final_response", "result"}
    )

    @staticmethod
    def _payload_value(payload: Any, key: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key, None)

    def _extract_text(self, payload: Any) -> str:
        """
        Best-effort extraction of a final assistant reply from Magentic results.
        """
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (list, tuple)):
            return "".join(
                text for item in payload if (text := self._extract_text(item))
            )

        event_type = self._event_type(payload)
        if event_type in self._IGNORED_EVENT_TYPES or event_type in (
            "tool_result",
            "function_result",
        ):
            return ""

        # Extract text from known keys/attributes
        for key in self._TEXT_KEYS:
            value = self._payload_value(payload, key)
            if isinstance(value, str) and value.strip():
                return value
            if value is not None:
                text = self._extract_text(value)
                if text.strip():
                    return text

        if self._contains_tool_call(payload):
            # This is a tool invocation, not user-facing text
            return ""

        # Fallback: try to_dict() for objects (but not if it contains tool calls)
        if hasattr(payload, "to_dict"):
            try:
                return self._extract_text(payload.to_dict())
            except (TypeError, ValueError):
                pass

        # No text found - don't fall back to arbitrary attribute scanning
        # as that returns metadata strings like "agent_response_update"
        return ""

    @staticmethod
    def _event_type(payload: Any) -> str:
        """
        Return a normalized Magentic event type when one is available.
        """
        if isinstance(payload, dict):
            return str(payload.get("type") or payload.get("event") or "").lower()
        return str(
            getattr(payload, "type", "") or getattr(payload, "event", "")
        ).lower()

    def _is_terminal_output_event(self, event: Any) -> bool:
        """
        Return whether an event should be exposed as output or contains data to preserve.
        """
        event_type = self._convergence_event_type(event)
        if event_type in {
            "final",
            "final_output",
            "final_response",
            "output",
            "result",
            "assistant_response",
            "response",
        }:
            return True

        if event_type:
            return False

        if isinstance(event, str):
            return bool(event.strip())

        for key in (
            "final_output",
            "final_response",
            "assistant_response",
            "role",
            "content",
            "contents",
            "messages",
            "text",
        ):
            if isinstance(event, dict) and key in event:
                return True
            if hasattr(event, key):
                return True

        if hasattr(event, "to_dict"):
            try:
                return self._is_terminal_output_event(event.to_dict())
            except (TypeError, ValueError):
                return False

        return False

    @classmethod
    def _conversation_history_messages(
        cls,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Return conversation history for Magentic workflow context.

        Background task acknowledgments (e.g., "[task-123] Started in background.") are
        kept so follow-up requests see that work is already in progress. Without these,
        Magentic can re-run the same long-running tool or respond to the wrong turn.
        """
        return history

    @classmethod
    def _contains_tool_call(cls, payload: Any) -> bool:
        """
        Return whether payload or its contents describe a tool/function call.
        """
        event_type = cls._event_type(payload)
        if event_type in ("function_call", "tool_call"):
            return True

        if any(
            cls._has_tool_call_payload(cls._payload_value(payload, key))
            for key in cls._TOOL_CALL_KEYS
        ):
            return True

        for key in cls._TOOL_RECURSION_KEYS:
            value = cls._payload_value(payload, key)
            if isinstance(value, (list, tuple)):
                if any(cls._contains_tool_call(item) for item in value):
                    return True
            elif isinstance(value, dict) and cls._contains_tool_call(value):
                return True

        return False

    @classmethod
    def _has_tool_call_payload(cls, value: Any) -> bool:
        """
        Return whether a tool-call field contains an actual call payload.
        """
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return bool(value)
        if isinstance(value, (list, tuple, set)):
            return any(cls._has_tool_call_payload(item) for item in value)
        return True

    @classmethod
    def _tool_call_name_from_payload(cls, payload: Any) -> str | None:
        """
        Return the MCP function/tool name from a nested Magentic call payload.
        """
        if payload is None:
            return None

        if isinstance(payload, str):
            return payload.strip() or None

        if isinstance(payload, (list, tuple, set)):
            for item in payload:
                if name := cls._tool_call_name_from_payload(item):
                    return name
            return None

        if not any(
            cls._payload_value(payload, key) is not None
            for key in ("executor_id", "agent_id")
        ):
            for key in ("name", "function_name", "tool_name"):
                value = cls._payload_value(payload, key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        for key in ("function", "tool"):
            if name := cls._tool_call_name_from_payload(
                cls._payload_value(payload, key)
            ):
                return name

        event_type = cls._event_type(payload)
        if event_type in {"function_call", "tool_call"} or any(
            cls._payload_value(payload, key) is not None for key in cls._TOOL_CALL_KEYS
        ):
            for key in ("name", "function_name", "tool_name"):
                value = cls._payload_value(payload, key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        for key in (*cls._TOOL_CALL_KEYS, *cls._TOOL_COLLECTION_KEYS):
            if name := cls._tool_call_name_from_payload(
                cls._payload_value(payload, key)
            ):
                return name

        return None

    @staticmethod
    def _tool_call_signals(
        participant_name: Any,
    ) -> List[str]:
        if not participant_name:
            return []

        return [InternalToolCallSignal(str(participant_name))]

    @classmethod
    def _call_notices_from_event(
        cls,
        event: Any,
        seen_executor_ids: set,
    ) -> List[str]:
        """
        Return invisible handoff signals when real MCP tool calls occur.

        In blocking=False mode, BackgroundTaskManager should detach only after
        a real tool execution. Actual Magentic tool invocations are surfaced as
        streamed output events carrying AgentResponseUpdate with function_call,
        not just executor_invoked events.

        Uses seen_executor_ids to deduplicate signals from the same tool execution
        (executor_invoked + output + tool_result all carry the same executor_id).
        """
        event_type = cls._event_type(event)
        if event_type == "magentic_orchestrator":
            event_type = cls._convergence_event_type(event)

        if event_type == "output":
            data = cls._payload_value(event, "data")
            # Executor ID is on event for Agent Framework, or in data for other formats
            executor_id = (
                cls._payload_value(event, "executor_id")
                or cls._payload_value(data, "executor_id")
                or cls._payload_value(data, "agent_id")
            )
            if (
                executor_id
                and executor_id not in seen_executor_ids
                and cls._contains_tool_call(data)
            ):
                seen_executor_ids.add(executor_id)
                return cls._tool_call_signals(
                    cls._tool_call_name_from_payload(data) or executor_id
                )
            return []

        if event_type not in {"executor_invoked", "tool_result", "function_result"}:
            return []

        data = cls._payload_value(event, "data")
        # Executor ID is on event for Agent Framework, or in data for other formats
        executor_id = cls._payload_value(event, "executor_id") or cls._payload_value(
            data, "executor_id"
        )

        if event_type == "executor_invoked":
            has_tools = any(
                cls._has_tool_call_payload(cls._payload_value(data, key))
                for key in cls._TOOL_COLLECTION_KEYS
            )
            if not has_tools or not executor_id:
                return []

        if executor_id and executor_id not in seen_executor_ids:
            seen_executor_ids.add(executor_id)
            return cls._tool_call_signals(
                cls._tool_call_name_from_payload(data) or executor_id
            )
        return []

    @classmethod
    def _background_task_descriptors_from_event(cls, event: Any) -> List[str]:
        """
        Return JSON descriptors for server-side background MCP tasks.

        When a Magentic specialist starts a server-side background MCP task,
        Agent Framework surfaces the descriptor inside executor_completed.data
        (AgentExecutorResponse / AgentResponseUpdate with function_result contents),
        not just in top-level tool_result events.
        """
        event_type = cls._convergence_event_type(event)
        if event_type not in {"tool_result", "function_result", "executor_completed"}:
            return []

        candidates = [
            value
            for key in ("data", "result", "content", "output")
            if (value := cls._payload_value(event, key)) is not None
        ]

        descriptors = []
        for candidate in candidates or [event]:
            descriptors.extend(cls._background_task_descriptors_from_value(candidate))
        return list(dict.fromkeys(descriptors))

    @classmethod
    def _background_task_descriptors_from_value(cls, value: Any) -> List[str]:
        """
        Extract parseable running background-task descriptors from a nested payload.

        In real Magentic worker responses, background-task JSON lives under
        structured messages/contents/items paths. The function_result wrapper
        itself usually has empty .text, so we must traverse the structured
        response (AgentResponse/AgentResponseUpdate) to find the task_id.
        """
        if value is None:
            return []

        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")

        if isinstance(value, str):
            parsed_values = cls._parse_json_candidates(value)
            descriptors = []
            for parsed in parsed_values:
                descriptors.extend(cls._background_task_descriptors_from_value(parsed))
            return descriptors

        if isinstance(value, (list, tuple)):
            descriptors = []
            for item in value:
                descriptors.extend(cls._background_task_descriptors_from_value(item))
            return descriptors

        if isinstance(value, dict):
            descriptor = cls._background_task_descriptor_json(value)
            if descriptor:
                return [descriptor]

            descriptors = []
            # Traverse structured response paths that contain background task descriptors
            for key in (
                "data",
                "result",
                "content",
                "output",
                "text",
                "message",
                "messages",
                "contents",
                "items",
                "event_data",
                "payload",
                "function_result",
                "tool_result",
            ):
                if key in value:
                    descriptors.extend(
                        cls._background_task_descriptors_from_value(value[key])
                    )
            return descriptors

        # Try to extract from object attributes
        if hasattr(value, "to_dict"):
            try:
                return cls._background_task_descriptors_from_value(value.to_dict())
            except (TypeError, ValueError):
                pass

        # Check common attribute names on structured objects
        for attr in (
            "text",
            "content",
            "contents",
            "messages",
            "items",
            "data",
            "event_data",
            "payload",
        ):
            if hasattr(value, attr):
                attr_value = getattr(value, attr, None)
                if attr_value is not None:
                    descriptors = cls._background_task_descriptors_from_value(
                        attr_value
                    )
                    if descriptors:
                        return descriptors

        return []

    @staticmethod
    def _parse_json_candidates(value: str) -> List[Any]:
        """
        Parse a JSON value from a full string or from the widest JSON object within it.
        """
        value = value.strip()
        if not value:
            return []

        # Try full string first
        try:
            return [json.loads(value)]
        except json.JSONDecodeError:
            pass

        # Try extracting widest {...} substring
        start = value.find("{")
        end = value.rfind("}")
        if start != -1 and end > start:
            try:
                return [json.loads(value[start : end + 1])]
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _background_task_descriptor_json(value: Dict[str, Any]) -> str:
        """
        Return a canonical JSON descriptor if the payload starts a background task.
        """
        task_id = value.get("task_id")
        if not task_id:
            return ""

        status = str(value.get("status") or "running").strip().lower()
        if status != "running":
            return ""

        descriptor = {
            "task_id": task_id,
            "status": status,
            "tool_name": value.get("tool_name", "background_tool"),
        }
        return json.dumps(descriptor, default=str)

    @staticmethod
    def _background_task_ack(descriptors: List[str]) -> str:
        """
        Build a concise user-facing acknowledgement when no final text is available.

        Format matches what BackgroundTasks and Gradio persistence checks expect.
        Kept in conversation history so follow-up requests see work is in progress.
        """
        if not descriptors:
            return ""

        try:
            descriptor = json.loads(descriptors[0])
        except json.JSONDecodeError:
            return "Started in background."

        task_id = descriptor.get("task_id")
        if task_id:
            return f"[{task_id}] Started in background."
        return "Started in background."

    @classmethod
    def _reply_for_persistence(
        cls, assistant_reply: str, background_task_descriptors: List[str]
    ) -> str:
        """Persist an answer, or an acknowledgement when no answer exists."""
        ack = cls._background_task_ack(background_task_descriptors)
        if assistant_reply.strip():
            return assistant_reply
        return ack

    async def _iter_result_events(
        self,
        result: Any,
        *,
        close_deadline: float | None = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Iterate over Magentic workflow events or result payloads.
        """
        if not isinstance(result, AsyncIterable) and inspect.isawaitable(result):
            result = await result

        final_response_called = False
        try:
            if isinstance(result, AsyncIterable):
                async for event in result:
                    yield event
                get_final_response = getattr(result, "get_final_response", None)
                if callable(get_final_response):
                    final_response_called = True
                    final = get_final_response()
                    if inspect.isawaitable(final):
                        final = await final
                    if final is not None:
                        yield final
                return

            if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
                for event in result:
                    yield event
                return

            yield result
        finally:
            if close_deadline is None:
                await self._close_async_iterator(
                    result, finalize_response=not final_response_called
                )
            else:
                await self._close_async_iterator_until(
                    result,
                    close_deadline,
                    finalize_response=not final_response_called,
                    initiate_after_deadline=True,
                )

    @staticmethod
    async def _close_async_iterator(
        value: Any, *, finalize_response: bool = True
    ) -> None:
        """Stop an async stream and its underlying iterator when supported."""
        close = getattr(value, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return

        close = getattr(value, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return

        # Agent Framework ResponseStream exposes the active async generator as
        # a private iterator but does not itself expose aclose(). Close that
        # generator, then use the stream's finalization path so cleanup hooks
        # still run for an aborted workflow.
        iterator = getattr(value, "_iterator", None)
        if iterator is None:
            source = getattr(value, "_stream_source", None)
            if source is not None and source is not value:
                if asyncio.isfuture(source):
                    source.cancel()
                else:
                    await MagenticOrchestrationStrategy._close_async_iterator(
                        source, finalize_response=False
                    )

        close = getattr(iterator, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

        get_final_response = getattr(value, "get_final_response", None)
        if finalize_response and callable(get_final_response) and iterator is not None:
            try:
                result = get_final_response()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                LOG.exception("Magentic workflow stream finalization failed")

        cleanup = getattr(value, "_run_cleanup_hooks", None)
        if callable(cleanup):
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                LOG.exception("Magentic workflow stream cleanup failed")

    async def _close_async_iterator_until(
        self,
        value: Any,
        deadline: float,
        *,
        finalize_response: bool = True,
        initiate_after_deadline: bool = False,
    ) -> bool:
        """Close a stream without allowing teardown to exceed the deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0 and not initiate_after_deadline:
            return False

        if finalize_response:
            close_task = asyncio.create_task(self._close_async_iterator(value))
        else:
            close_task = asyncio.create_task(
                self._close_async_iterator(value, finalize_response=False)
            )

        def consume_task_result(task: asyncio.Task[Any]) -> None:
            if not task.cancelled():
                task.exception()

        if remaining <= 0:
            # Let nested async-generator finalizers start, but never retain a
            # cleanup task after the request deadline has elapsed.
            await asyncio.sleep(0)
            if close_task.done():
                try:
                    await close_task
                    return True
                except asyncio.CancelledError:
                    return False
                except Exception:
                    LOG.exception("Magentic workflow stream teardown failed")
                    return False

            close_task.cancel()
            try:
                await close_task
            except asyncio.CancelledError:
                pass
            except Exception:
                LOG.exception("Magentic workflow stream teardown failed")
            return False

        try:
            async with asyncio.timeout(remaining):
                await asyncio.shield(close_task)
            return True
        except asyncio.TimeoutError:
            if not close_task.done():
                close_task.cancel()
            close_task.add_done_callback(consume_task_result)
            return False
        except asyncio.CancelledError:
            # Cleanup is best effort, but preserve cancellation of the request.
            if not close_task.done():
                close_task.cancel()
            close_task.add_done_callback(consume_task_result)
            raise
        except Exception:
            LOG.exception("Magentic workflow stream teardown failed")
            close_task.add_done_callback(consume_task_result)
            return False

    def _start_runtime(
        self,
        runtime: Any,
        message_payload: Any,
    ) -> Any:
        """
        Start a Magentic runtime with the best supported streaming API.
        """
        run = getattr(runtime, "run", None)
        if callable(run):
            try:
                return run(message_payload, stream=True)
            except TypeError as e:
                if "unexpected keyword argument 'stream'" not in str(e):
                    raise
                return run(message_payload)

        for method_name in ("run_stream", "stream", "invoke"):
            method = getattr(runtime, method_name, None)
            if callable(method):
                return method(message_payload)

        raise RuntimeError(
            "Unable to execute Magentic workflow with the installed builder."
        )

    async def _iter_workflow_events(
        self,
        orchestrator: "MADAOrchestrator",
        transcript_messages: List[Dict[str, Any]],
        *,
        close_deadline: float | None = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Run a Magentic workflow and yield its events.

        Magentic uses messages[0] as the task to plan against, so we must pass
        a single user message containing the latest request, not the full transcript.
        """
        if not orchestrator.manager_agent:
            raise RuntimeError("Magentic manager is not initialized.")

        # Build a single prompt from the full transcript for Magentic planning
        # Magentic uses messages[0] as the task, so passing multi-message history
        # would cause it to plan around the oldest message instead of latest request
        if not transcript_messages:
            task_message = Message(role="user", contents=["Please introduce yourself."])
        else:
            # Flatten transcript into single prompt that Magentic can plan against
            prompt = orchestrator.build_prompt_from_transcript(transcript_messages)
            task_message = Message(role="user", contents=[prompt])

        runtime = self._build_runtime(orchestrator)
        result = self._start_runtime(runtime, [task_message])
        event_stream = self._iter_result_events(result, close_deadline=close_deadline)
        try:
            async for event in event_stream:
                yield event
        finally:
            if close_deadline is None:
                await self._close_async_iterator(event_stream)
            else:
                await self._close_async_iterator_until(
                    event_stream,
                    close_deadline,
                    finalize_response=False,
                    initiate_after_deadline=True,
                )

    @staticmethod
    def _normalize_convergence_event_name(value: Any) -> str:
        """Normalize framework enum/string event names to stable internal names."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        value = getattr(value, "value", value)
        value = getattr(value, "name", value)
        name = str(value).rsplit(".", 1)[-1].strip().lower()
        name = name.replace("-", "_").replace(" ", "_")
        return {
            "plan_created": "plan",
            "replanned": "replan",
            "progress_ledger_updated": "progress",
        }.get(name, name)

    @classmethod
    def _convergence_payload(cls, event: Any) -> Any:
        """Return the payload carried by either wrapped or native events."""
        data = cls._payload_value(event, "data")
        if data is not None:
            return data
        if cls._payload_value(event, "event_type") is not None:
            return event
        return event

    @classmethod
    def _convergence_event_type(cls, event: Any) -> str:
        """Return the Magentic event type, including nested WorkflowEvent data."""
        event_type = cls._normalize_convergence_event_name(cls._event_type(event))
        direct_type = cls._payload_value(event, "event_type")
        if direct_type is not None:
            normalized_direct_type = cls._normalize_convergence_event_name(direct_type)
            if normalized_direct_type:
                return normalized_direct_type

        data = cls._payload_value(event, "data")
        nested_type = cls._payload_value(data, "event_type")
        if nested_type is not None:
            normalized_nested_type = cls._normalize_convergence_event_name(nested_type)
            if normalized_nested_type:
                return normalized_nested_type
        return event_type

    @classmethod
    def _convergence_signature(cls, event: Any) -> str | None:
        """Return a stable signature for an event representing planning progress."""
        event_type = cls._convergence_event_type(event)
        if event_type not in {
            "plan",
            "progress",
            "replan",
            "checkpoint",
            "executor_invoked",
            "executor_completed",
            "tool_result",
            "function_result",
        }:
            return None

        data = cls._convergence_payload(event)
        if event_type in {"executor_invoked", "executor_completed"}:
            identity = cls._payload_value(event, "executor_id") or cls._payload_value(
                data, "executor_id"
            )
            if identity:
                return f"{event_type}:{identity}"

        if event_type in {"tool_result", "function_result"}:
            identity = cls._payload_value(event, "executor_id") or cls._payload_value(
                data, "executor_id"
            )
            if identity:
                return f"{event_type}:{identity}"

        # _extract_text intentionally ignores plan/progress/control event types
        # because they are not user-facing output. They still need their own
        # payloads here: otherwise every status update becomes just
        # ``progress`` and looks like a repeated stall.
        text = cls._progress_ledger_signature(data)
        if not text:
            text = cls._progress_text(data) or cls._progress_text(event)
        if text:
            return f"{event_type}:{text}"
        return None

    @classmethod
    def _progress_text(cls, value: Any) -> str:
        """Extract descriptive text from non-user-facing progress payloads."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            return " ".join(
                text for item in value if (text := cls._progress_text(item))
            ).strip()
        if isinstance(value, dict):
            for key in (
                "text",
                "message",
                "answer",
                "progress",
                "plan",
                "progress_ledger",
                "ledger",
                "description",
                "content",
            ):
                text = cls._progress_text(value.get(key))
                if text:
                    return text
            for key in (
                "data",
                "details",
                "result",
                "progress_ledger",
                "ledger",
                "contents",
                "items",
                "messages",
            ):
                text = cls._progress_text(value.get(key))
                if text:
                    return text
            return cls._progress_text(value.get("status"))
        if hasattr(value, "to_dict"):
            try:
                return cls._progress_text(value.to_dict())
            except (TypeError, ValueError):
                pass
        for attribute in (
            "text",
            "message",
            "progress",
            "plan",
            "progress_ledger",
            "ledger",
            "description",
            "content",
            "data",
            "contents",
            "items",
            "messages",
        ):
            if hasattr(value, attribute):
                text = cls._progress_text(getattr(value, attribute, None))
                if text:
                    return text
        if hasattr(value, "status"):
            return cls._progress_text(getattr(value, "status", None))
        return ""

    @classmethod
    def _progress_ledger_signature(cls, value: Any) -> str:
        """Return the structured progress state from a Magentic ledger."""
        candidates = [value]
        content = cls._payload_value(value, "content")
        if content is not None:
            candidates.insert(0, content)

        states = []
        for candidate in candidates:
            for field in (
                "is_progress_being_made",
                "is_in_loop",
                "next_speaker",
                "instruction_or_question",
            ):
                state = cls._payload_value(candidate, field)
                if state is None:
                    continue
                answer = cls._payload_value(state, "answer")
                if answer is None:
                    answer = state
                answer_text = cls._progress_text(answer)
                if answer_text:
                    states.append(f"{field}={answer_text}")
        return ";".join(dict.fromkeys(states))

    @classmethod
    def _progress_ledger_is_stalled(cls, event: Any) -> bool | None:
        """Return the native Magentic stall state when a ledger is present."""
        if cls._convergence_event_type(event) != "progress":
            return None

        data = cls._convergence_payload(event)
        candidates = [data]
        content = cls._payload_value(data, "content")
        if content is not None:
            candidates.insert(0, content)

        progress_answer = None
        loop_answer = None
        for candidate in candidates:
            for field in ("is_progress_being_made", "is_in_loop"):
                state = cls._payload_value(candidate, field)
                if state is None:
                    continue
                answer = cls._payload_value(state, "answer")
                if answer is None:
                    answer = state
                if isinstance(answer, bool):
                    value = answer
                elif isinstance(answer, str):
                    normalized = answer.strip().lower()
                    if normalized not in {"true", "false"}:
                        continue
                    value = normalized == "true"
                else:
                    continue
                if field == "is_progress_being_made":
                    progress_answer = value
                else:
                    loop_answer = value

        if progress_answer is None and loop_answer is None:
            return None
        return progress_answer is False or loop_answer is True

    @classmethod
    def _progress_ledger_is_satisfied(cls, event: Any) -> bool | None:
        """Return whether a progress ledger already satisfies the request."""
        if cls._convergence_event_type(event) != "progress":
            return None

        data = cls._convergence_payload(event)
        candidates = [data]
        content = cls._payload_value(data, "content")
        if content is not None:
            candidates.insert(0, content)

        for candidate in candidates:
            state = cls._payload_value(candidate, "is_request_satisfied")
            if state is None:
                continue
            answer = cls._payload_value(state, "answer")
            if answer is None:
                answer = state
            if isinstance(answer, bool):
                return answer
            if isinstance(answer, str) and answer.strip().lower() in {"true", "false"}:
                return answer.strip().lower() == "true"
        return None

    @classmethod
    def _is_text_progress_event(cls, event: Any) -> bool:
        """Return whether an event contains useful plain-text participant work."""
        if cls._convergence_event_type(event) not in {
            "output",
            "intermediate",
            "assistant_response",
            "response",
        }:
            return False
        if cls._contains_tool_call(event):
            return False
        return bool(cls._progress_text(event))

    @staticmethod
    def _signature_is_stalled(signature: str | None) -> bool:
        """Return whether a prior progress signature contained a stalled ledger."""
        if not signature or not signature.startswith("progress:"):
            return False
        signature = signature.split(
            MagenticOrchestrationStrategy._CONVERGENCE_OUTPUT_MARKER, 1
        )[0].lower()
        return (
            "is_progress_being_made=false" in signature
            or "is_in_loop=true" in signature
        )

    _CONVERGENCE_OUTPUT_MARKER = "\x1eoutput="

    @classmethod
    def _signature_base(cls, signature: str | None) -> str | None:
        if signature is None:
            return None
        return signature.split(cls._CONVERGENCE_OUTPUT_MARKER, 1)[0]

    @classmethod
    def _signature_output(cls, signature: str | None) -> str | None:
        if signature is None:
            return None
        base, marker, encoded = signature.partition(cls._CONVERGENCE_OUTPUT_MARKER)
        if not marker:
            return base[5:] if base.startswith("text:") else None
        try:
            output = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return output if isinstance(output, str) else None

    @classmethod
    def _signature_with_output(
        cls, signature: str | None, output: str | None
    ) -> str | None:
        if signature is None or output is None:
            return signature
        return (
            f"{cls._signature_base(signature)}{cls._CONVERGENCE_OUTPUT_MARKER}"
            f"{json.dumps(output)}"
        )

    @classmethod
    def _signature_is_progress(cls, signature: str | None) -> bool:
        """Return whether a signature represents a progress update."""
        return bool(
            signature and cls._signature_base(signature).startswith("progress:")
        )

    @classmethod
    def _convergence_round_increment(cls, event: Any, seen_round_ids: set[str]) -> int:
        """Return the number of new manager coordination rounds represented."""
        event_type = cls._convergence_event_type(event)
        if event_type == "progress":
            # Magentic emits one progress-ledger update at the start of every
            # coordination round. Plan/replan events only describe setup or a
            # reset and must not be counted in addition to that update.
            return 1
        return 0

    @classmethod
    def _update_convergence(
        cls,
        event: Any,
        *,
        rounds: int,
        stalls: int,
        max_rounds: int,
        max_stalls: int,
        last_signature: str | None,
        seen_round_ids: set[str],
    ) -> tuple[int, int, str | None, str | None]:
        """Update convergence state and return a stop reason when one is reached."""
        rounds += cls._convergence_round_increment(event, seen_round_ids)
        signature = cls._convergence_signature(event)
        ledger_stalled = cls._progress_ledger_is_stalled(event)
        ledger_satisfied = cls._progress_ledger_is_satisfied(event)
        event_type = cls._convergence_event_type(event)
        last_signature_base = cls._signature_base(last_signature)
        last_output = cls._signature_output(last_signature)
        preserves_stall_state = (
            cls._signature_is_stalled(last_signature)
            and event_type == "executor_invoked"
        )
        resets_stall_state = event_type in {
            "executor_completed",
            "tool_result",
            "function_result",
        }
        if cls._is_text_progress_event(event):
            # Compatibility streams may repeat cumulative output verbatim. A
            # repeated transport update is not a new convergence cycle, while
            # genuinely new output resets a prior no-progress run.
            text = cls._progress_text(event)
            text_signature = f"text:{text}"
            if text != last_output:
                stalls = 0
                last_signature = text_signature
            elif cls._signature_is_progress(last_signature):
                last_signature = cls._signature_with_output(last_signature, text)
        elif resets_stall_state:
            stalls = 0
            last_signature = signature
        elif preserves_stall_state:
            pass
        elif ledger_stalled is not None:
            stalls = (
                stalls + 1
                if ledger_stalled and cls._signature_is_stalled(last_signature)
                else 0
            )
            last_signature = cls._signature_with_output(signature, last_output)
        elif signature is not None:
            stalls = stalls + 1 if signature == last_signature_base else 0
            last_signature = cls._signature_with_output(signature, last_output)

        reason = None
        # Progress is reported at the start of a round. Stop on the first
        # progress event after the configured final round so that final round
        # can finish and produce useful work for synthesis.
        if ledger_satisfied is True:
            reason = None
        elif rounds > max_rounds:
            reason = f"round limit ({max_rounds})"
        elif stalls >= max_stalls:
            reason = f"no-progress limit ({max_stalls})"
        return rounds, stalls, last_signature, reason

    @classmethod
    def _completed_tool_work(cls, event: Any) -> str:
        """Serialize completed tool output for bounded-run synthesis."""
        event_type = cls._convergence_event_type(event)
        if event_type not in {
            "tool_result",
            "function_result",
            "executor_completed",
        }:
            return ""

        data = cls._convergence_payload(event)
        result_keys = (
            "result",
            "content",
            "output",
            "tool_result",
            "function_result",
        )

        def nested_values(payload: Any) -> list[Any]:
            return [
                value
                for key in result_keys
                if (value := cls._payload_value(payload, key)) is not None
            ]

        def nested_tool_results(payload: Any) -> list[Any]:
            """Find function/tool results inside executor completion payloads."""
            if payload is None:
                return []
            payload_type = cls._convergence_event_type(payload)
            if payload_type in {"function_result", "tool_result"}:
                values = nested_values(payload)
                return values or [payload]
            if isinstance(payload, dict):
                results = [
                    value
                    for key in ("function_result", "tool_result")
                    if (value := payload.get(key)) is not None
                ]
                if results:
                    return results
                nested = []
                for key in (
                    "data",
                    "result",
                    "content",
                    "output",
                    "text",
                    "message",
                    "messages",
                    "contents",
                    "items",
                    "event_data",
                    "payload",
                ):
                    if key in payload:
                        nested.extend(nested_tool_results(payload[key]))
                return nested
            if isinstance(payload, (list, tuple)):
                return [
                    result for item in payload for result in nested_tool_results(item)
                ]
            if hasattr(payload, "to_dict"):
                try:
                    return nested_tool_results(payload.to_dict())
                except (TypeError, ValueError):
                    pass
            nested = []
            for attribute in (
                "data",
                "result",
                "content",
                "output",
                "text",
                "message",
                "messages",
                "contents",
                "items",
                "event_data",
                "payload",
                "function_result",
                "tool_result",
            ):
                if hasattr(payload, attribute):
                    nested.extend(
                        nested_tool_results(getattr(payload, attribute, None))
                    )
            return nested

        if event_type == "executor_completed":
            # A completed executor can also contain an ordinary specialist
            # response. Only serialize it when the payload carries an actual
            # function/tool result, which is commonly nested under .data.
            candidates = nested_tool_results(data if data is not None else event)
            if not candidates:
                return ""
        else:
            candidates = nested_values(event)
            if data is not None:
                candidates.extend(nested_values(data) or ([] if candidates else [data]))
            if not candidates:
                candidates = [event]

        serialized = []
        for candidate in candidates or [event]:
            if isinstance(candidate, bytes):
                text = candidate.decode("utf-8", errors="replace")
            elif isinstance(candidate, str):
                text = candidate
            else:
                if hasattr(candidate, "to_dict"):
                    try:
                        candidate = candidate.to_dict()
                    except (TypeError, ValueError):
                        pass
                try:
                    text = json.dumps(candidate, default=str)
                except (TypeError, ValueError):
                    text = str(candidate)
            if text.strip():
                serialized.append(text.strip())

        if not serialized:
            return ""
        tool_name = cls._tool_call_name_from_payload(event)
        label = (
            tool_name
            or cls._payload_value(event, "executor_id")
            or cls._payload_value(data, "executor_id")
            or event_type
        )
        return f"{label}: {''.join(serialized)}"

    async def _synthesize_response(
        self,
        orchestrator: "MADAOrchestrator",
        transcript_messages: List[Dict[str, Any]],
        candidate_text: str,
        *,
        completed_tool_work: List[str] | None = None,
        timeout_seconds: float,
    ) -> str:
        """Ask the manager to finalize bounded Magentic work into one answer."""
        if timeout_seconds <= 0:
            return ""

        task = orchestrator.build_prompt_from_transcript(transcript_messages)
        completed_work = "\n\n".join(completed_tool_work or [])
        prompt = (
            "The Magentic team reached a convergence safety limit while working "
            "on this request.\n\n"
            f"{task}\n\n"
            "Completed specialist/tool results:\n"
            f"{completed_work or '(No completed tool result was captured.)'}\n\n"
            "Work completed so far:\n"
            f"{candidate_text or '(No user-facing text was produced.)'}\n\n"
            "Provide the best useful final answer now. Do not continue planning, "
            "repeat tool calls, mention this safety limit, or describe internal "
            "agent coordination. If the work is incomplete, clearly state what is "
            "known and the next concrete step."
        )

        manager = self._runtime_agent(orchestrator.manager_agent)
        run = getattr(manager, "run", None)
        if not callable(run):
            return ""

        deadline = time.monotonic() + timeout_seconds
        try:
            async with asyncio.timeout(timeout_seconds):
                result = self._invoke_synthesis_run(run, prompt)
                streamed_text = ""
                final_text = ""
                async for event in self._iter_result_events(
                    result, close_deadline=deadline
                ):
                    event_text = self._extract_text(event)
                    if not event_text:
                        continue
                    if self._convergence_event_type(event) in self._FINAL_EVENT_TYPES:
                        final_text = event_text
                        continue
                    _, streamed_text = self._stream_text_update(
                        streamed_text, event_text
                    )
                return (final_text or streamed_text).strip()
        except asyncio.TimeoutError:
            LOG.warning("Magentic synthesis timed out")
            return ""
        except Exception:
            LOG.exception("Magentic synthesis failed")
            return ""

    @staticmethod
    def _invoke_synthesis_run(run: Any, prompt: str) -> Any:
        """Invoke the regular Agent Framework run wrapper."""
        try:
            return run(prompt, stream=False)
        except TypeError as error:
            if "unexpected keyword argument 'stream'" not in str(error):
                raise
            return run(prompt)

    @classmethod
    def _synthesis_timeout_budget(cls, timeout_seconds: float) -> float:
        """Reserve part of the request budget for bounded-run synthesis."""
        return min(
            cls._MAX_SYNTHESIS_TIMEOUT_SECONDS,
            max(
                cls._MIN_SYNTHESIS_TIMEOUT_SECONDS,
                timeout_seconds * cls._SYNTHESIS_TIMEOUT_FRACTION,
            ),
        )

    async def _stream_workflow_response(
        self,
        orchestrator: "MADAOrchestrator",
        transcript_messages: List[Dict[str, Any]],
        *,
        include_tool_notices: bool,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Stream Magentic notices and return the final assistant reply as an event.
        """
        streamed_text = ""
        final_text = ""
        background_task_descriptors = []
        completed_tool_work = []
        seen_executor_ids = set()
        convergence_round_ids = set()
        configured = getattr(orchestrator, "orchestration", None)
        max_rounds = max(1, int(getattr(configured, "max_rounds", 4)))
        max_stalls = max(1, int(getattr(configured, "max_stalls", 1)))
        timeout_seconds = max(1.0, float(getattr(configured, "timeout_seconds", 120)))
        synthesis_budget = self._synthesis_timeout_budget(timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        # Keep the synthesis budget outside the workflow deadline. A workflow
        # that runs until its safety timeout must still leave time for the
        # manager to turn partial work into a useful answer.
        workflow_timeout = max(0.0, timeout_seconds - synthesis_budget)
        workflow_deadline = deadline - synthesis_budget
        rounds = 0
        stalls = 0
        last_signature = None
        stop_reason = None
        workflow_event_factory = self._iter_workflow_events
        try:
            workflow_event_parameters = inspect.signature(
                workflow_event_factory
            ).parameters
        except (TypeError, ValueError):
            workflow_event_parameters = {}
        supports_close_deadline = "close_deadline" in workflow_event_parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in workflow_event_parameters.values()
        )
        if supports_close_deadline:
            workflow_events = workflow_event_factory(
                orchestrator,
                transcript_messages,
                close_deadline=workflow_deadline,
            )
        else:
            workflow_events = workflow_event_factory(orchestrator, transcript_messages)

        try:
            async with asyncio.timeout(workflow_timeout):
                async for event in workflow_events:
                    (
                        rounds,
                        stalls,
                        last_signature,
                        convergence_reason,
                    ) = self._update_convergence(
                        event,
                        rounds=rounds,
                        stalls=stalls,
                        max_rounds=max_rounds,
                        max_stalls=max_stalls,
                        last_signature=last_signature,
                        seen_round_ids=convergence_round_ids,
                    )

                    if include_tool_notices:
                        for notice in self._call_notices_from_event(
                            event, seen_executor_ids
                        ):
                            yield "notice", notice

                    convergence_event_type = self._convergence_event_type(event)

                    if convergence_event_type in (
                        "tool_result",
                        "function_result",
                        "executor_completed",
                    ):
                        if work := self._completed_tool_work(event):
                            completed_tool_work.append(
                                work[: self._MAX_SYNTHESIS_TOOL_RESULT_CHARS]
                            )
                            while (
                                sum(map(len, completed_tool_work))
                                > self._MAX_SYNTHESIS_TOOL_WORK_CHARS
                            ):
                                completed_tool_work.pop(0)
                        for descriptor in self._background_task_descriptors_from_event(
                            event
                        ):
                            background_task_descriptors.append(descriptor)
                            yield "background_task", descriptor
                        if convergence_reason:
                            stop_reason = convergence_reason
                            break
                        continue

                    if convergence_reason:
                        stop_reason = convergence_reason
                        break

                    if not self._is_terminal_output_event(event):
                        continue

                    event_text = self._extract_text(event)
                    if not event_text:
                        continue

                    if convergence_event_type in self._FINAL_EVENT_TYPES:
                        # The last final event is authoritative.
                        final_text = event_text
                        continue

                    chunk, streamed_text = self._stream_text_update(
                        streamed_text, event_text
                    )
                    if chunk:
                        yield "chunk", chunk
        except asyncio.TimeoutError:
            stop_reason = (
                None if final_text else f"wall-clock limit ({timeout_seconds:g}s)"
            )
        finally:
            # Teardown must not consume the time reserved for synthesis.
            closed = await self._close_async_iterator_until(
                workflow_events,
                workflow_deadline,
                initiate_after_deadline=True,
            )
            if not closed and not stop_reason and not final_text:
                stop_reason = f"wall-clock limit ({timeout_seconds:g}s)"

        synthesized = ""
        if stop_reason and not final_text:
            LOG.warning("Magentic workflow stopped at %s", stop_reason)
            synthesis_timeout = min(
                synthesis_budget,
                max(0.0, deadline - time.monotonic()),
            )
            if synthesis_timeout > 0:
                try:
                    async with asyncio.timeout(synthesis_timeout):
                        synthesized = await self._synthesize_response(
                            orchestrator,
                            transcript_messages,
                            final_text or streamed_text,
                            completed_tool_work=completed_tool_work,
                            timeout_seconds=synthesis_timeout,
                        )
                except asyncio.TimeoutError:
                    LOG.warning("Magentic synthesis exceeded the request deadline")
            if synthesized:
                final_text = synthesized

        bg_ack = self._background_task_ack(background_task_descriptors)
        main_output = final_text or streamed_text or bg_ack

        # Stream delta/replacement if main_output differs from streamed
        if main_output and main_output != streamed_text:
            if bg_ack and main_output == bg_ack:
                # Background task ack is a fallback when no final text is available.
                yield "chunk", InternalResponseReplacement(main_output)
            elif final_text and final_text.startswith(streamed_text):
                # Final text extends streamed - yield delta only
                delta = final_text[len(streamed_text) :]
                if delta:
                    yield "chunk", delta
            elif final_text:
                # Final text replaces streamed
                yield "chunk", InternalResponseReplacement(main_output)
        yield "final", main_output

    @staticmethod
    def _stream_text_update(streamed_text: str, event_text: str) -> Tuple[str, str]:
        """
        Return the next display chunk for delta or cumulative text updates.
        """
        if event_text.startswith(streamed_text):
            return event_text[len(streamed_text) :], event_text
        return event_text, streamed_text + event_text

    async def initialize(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] | None = None,
        a2a_agents: Dict[str, RemoteA2AAgentConfig] | None = None,
    ) -> Tuple[str, List[str]]:
        """
        Initialize the Magentic orchestration flow end to end.
        """
        if MagenticBuilder is None:
            raise RuntimeError(
                "Magentic orchestration requires agent_framework MagenticBuilder support"
            )

        orchestrator.specialist_agents = []
        orchestrator._mcp_tool_count = 0
        orchestrator._agent_descriptions = {}
        participant_configs = orchestrator.resolve_participant_configs(agent_configs)
        orchestrator.mcp_servers = mcp_servers or {}
        orchestrator.a2a_agents = {}
        orchestrator._a2a_agent_cards.clear()
        if a2a_agents:
            LOG.warning(
                "Remote A2A agents are configured but are not used in "
                "magentic orchestration mode."
            )

        all_tools, failed_servers, failed_agents = await self._initialize_participants(
            orchestrator, participant_configs
        )
        active_participant_configs = self._resolve_active_participant_configs(
            orchestrator, participant_configs
        )
        self._preserve_participant_metadata(
            orchestrator,
            active_participant_configs,
        )

        if not active_participant_configs:
            raise RuntimeError(
                "Magentic orchestration requires at least one active specialist agent."
            )

        orchestrator.planning_agent = None
        orchestrator.session = None
        orchestrator.manager_agent = self._create_manager_agent(
            orchestrator,
            agent_configs=agent_configs,
            participant_configs=active_participant_configs,
        )

        status = self._build_status(orchestrator, failed_servers, failed_agents)
        LOG.info(status)

        return status, all_tools

    async def process_openai_messages(
        self,
        orchestrator: "MADAOrchestrator",
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Process OpenAI-style chat messages through a fresh Magentic workflow.

        Streams chunks incrementally as they arrive from Magentic. Uses structured
        markers (InternalResponseReplacement, InternalError) for replacements
        and errors rather than string-prefix detection, allowing normal model
        output to contain phrases like "Error processing message:" without being
        misinterpreted as internal signals.
        """
        if not orchestrator.manager_agent:
            yield "Error: Orchestrator not initialized."
            return

        transcript_messages = orchestrator._normalize_transcript_messages(messages)
        try:
            streamed_text = ""
            final_text = ""
            async for kind, value in self._stream_workflow_response(
                orchestrator,
                transcript_messages,
                include_tool_notices=False,
            ):
                if kind == "chunk":
                    replacement_text = response_replacement(value)
                    if replacement_text is not None:
                        streamed_text = str(replacement_text)
                    else:
                        streamed_text += value
                    yield value
                elif kind == "final":
                    final_text = value
                elif kind == "background_task":
                    continue

            final_replacement = response_replacement(final_text)
            final_output = (
                str(final_replacement)
                if final_replacement is not None
                else (final_text or streamed_text)
            )
            if final_output and final_output != streamed_text:
                if final_output.startswith(streamed_text):
                    delta = final_output[len(streamed_text) :]
                    if delta:
                        yield delta
                else:
                    yield InternalResponseReplacement(final_output)
            elif not final_output:
                LOG.warning("No final assistant text received from Magentic workflow")
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield InternalError(error_msg)

    async def process_message(
        self,
        orchestrator: "MADAOrchestrator",
        message: str,
        isolated_session: bool = False,
        persistence_session_id: str | None = None,
        stateless_session: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message through a fresh Magentic workflow.
        """
        if not orchestrator.manager_agent:
            yield "Error: Orchestrator not initialized. Call initialize_orchestrator() first."
            return

        try:
            background_task_descriptors = []
            turn_id = None
            streamed_text = ""

            # Load history inside lock for non-isolated sessions to ensure atomicity
            # between turn_id reservation and history snapshot
            if isolated_session:
                if stateless_session:
                    history = []
                elif persistence_session_id is None:
                    # Isolated without explicit session: load current history for context
                    # (used by CLI/UI background follow-ups) but don't persist
                    history = orchestrator.session_manager.load_history()
                else:
                    history = await orchestrator._load_history_for_session(
                        persistence_session_id
                    )
            else:
                async with orchestrator._session_lock:
                    turn_id = orchestrator._next_turn_id
                    orchestrator._next_turn_id += 1
                    history = orchestrator.session_manager.load_history()

            history = self._conversation_history_messages(history)
            transcript_messages = orchestrator._normalize_transcript_messages(
                [*history, {"role": "user", "content": message}]
            )

            aggregated_assistant_reply = ""
            async for kind, value in self._stream_workflow_response(
                orchestrator,
                transcript_messages,
                include_tool_notices=True,
            ):
                if kind == "notice":
                    yield value
                elif kind == "chunk":
                    replacement_text = response_replacement(value)
                    if replacement_text is not None:
                        streamed_text = str(replacement_text)
                        yield value
                    else:
                        streamed_text += value
                        yield value
                elif kind == "final":
                    aggregated_assistant_reply = value
                elif kind == "background_task":
                    background_task_descriptors.append(value)

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
                elif persistence_session_id is not None:
                    persisted_reply = self._reply_for_persistence(
                        aggregated_assistant_reply,
                        background_task_descriptors,
                    )
                    await orchestrator._persist_isolated_response(
                        message,
                        persisted_reply,
                        background_task_descriptors=background_task_descriptors,
                        session_id=persistence_session_id,
                    )
            else:
                persisted_reply = self._reply_for_persistence(
                    aggregated_assistant_reply,
                    background_task_descriptors,
                )
                await orchestrator._commit_completed_turn(
                    turn_id,
                    message,
                    persisted_reply,
                    run_session=None,
                    history_lengths={},
                    background_task_descriptors=background_task_descriptors,
                )

            output = aggregated_assistant_reply or streamed_text
            if output.strip() and output != streamed_text:
                if output.startswith(streamed_text):
                    delta = output[len(streamed_text) :]
                    if delta:
                        yield delta
                else:
                    yield InternalResponseReplacement(output)
            elif not output.strip():
                LOG.warning("No final assistant text received from Magentic workflow")
        except Exception as e:
            if turn_id is not None:
                try:
                    await orchestrator._retire_failed_turn(turn_id)
                except Exception:
                    LOG.exception("Failed to retire Magentic turn %s", turn_id)
            error_msg = f"Error processing message: {e}"
            LOG.error(error_msg)
            traceback.print_exc()
            yield InternalError(error_msg)
