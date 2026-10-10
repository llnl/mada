# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Background task management for MADA orchestration.

This module contains the async task bookkeeping used by interfaces and
server-side MCP background tools.
"""

import asyncio
import copy
import json
from typing import Any, Awaitable, Callable, Dict, Set

from mada.core.database import ChatSessionManager
from mada.core.media import (
    ImageAttachment,
    extract_image_attachments,
    strip_image_payload_text,
)


CollectMessageResponse = Callable[..., Awaitable[str]]


def _parse_background_task_descriptor_payload(value: str) -> Any | None:
    """
    Parse JSON from a string, trying both the full string and the widest {...} substring.

    Returns:
        Parsed JSON object/array, or None if parsing fails.
    """
    value = value.strip()
    if not value:
        return None

    # Try full string first
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # Try extracting widest {...} substring
    start = value.find("{")
    end = value.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _background_task_result_from_payload(
    value: Any, task_id: str
) -> Dict[str, Any] | None:
    """Find a task-result descriptor inside common MCP response containers."""
    candidates: list[Dict[str, Any]] = []
    visited: set[int] = set()

    def visit(item: Any) -> None:
        if item is None or isinstance(item, (int, float, bool)):
            return
        if isinstance(item, bytes):
            visit(item.decode("utf-8", errors="replace"))
            return
        if isinstance(item, str):
            parsed = _parse_background_task_descriptor_payload(item)
            if parsed is not None and parsed != item:
                visit(parsed)
            return

        item_id = id(item)
        if item_id in visited:
            return
        visited.add(item_id)

        if isinstance(item, (list, tuple, set)):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            if "status" in item and (
                not item.get("task_id") or str(item.get("task_id")) == str(task_id)
            ):
                candidates.append(item)
            for key in (
                "content",
                "contents",
                "data",
                "items",
                "output",
                "outputs",
                "result",
                "tool_result",
                "function_result",
            ):
                if key in item:
                    visit(item[key])
            return

        text = getattr(item, "text", None)
        if text is not None:
            visit(text)
        for attribute in ("content", "contents", "data", "items", "result"):
            if hasattr(item, attribute):
                visit(getattr(item, attribute, None))

    visit(value)
    return next(
        (
            candidate
            for candidate in candidates
            if str(candidate.get("status")).lower() != "not_found"
        ),
        candidates[0] if candidates else None,
    )


def _attachment_digests(attachments: Any) -> set[str]:
    """Return stable identities for persisted image attachments."""
    return {
        digest
        for attachment in attachments or []
        if isinstance((digest := getattr(attachment, "digest", None)), str)
    }


def _format_background_task_message(value: Any) -> str:
    """Render structured tool output as concise assistant-facing Markdown."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            parsed = json.loads(value.strip())
        except (TypeError, json.JSONDecodeError):
            return strip_image_payload_text(value)
        return _format_background_task_message(parsed)
    if isinstance(value, (list, tuple, set)):
        messages = [
            formatted
            for item in value
            if (formatted := _format_background_task_message(item))
        ]
        return "\n\n".join(dict.fromkeys(messages))
    if not isinstance(value, dict):
        return str(value)

    item_type = str(value.get("type") or "").lower()
    if item_type in {"image", "data", "uri"} and extract_image_attachments(value):
        return ""

    # Prefer prose supplied by the tool over rendering its transport envelope.
    for key in ("message", "summary", "text", "result", "output", "error"):
        if key in value:
            formatted = _format_background_task_message(value[key])
            if formatted:
                return formatted

    metadata_keys = {"task_id", "status", "tool_name", "type"}
    details = []
    for key, item in value.items():
        if key in metadata_keys:
            continue
        formatted = _format_background_task_message(item)
        if formatted:
            label = str(key).replace("_", " ").strip().capitalize()
            details.append(f"- **{label}:** {formatted}")
    return "\n".join(details)


def is_background_task_start_ack(content: str) -> bool:
    """
    Return whether text is a background-task start acknowledgement.

    Recognizes formats:
    - "Started in background."
    - "[task-123] Started in background."
    - "[uuid-or-any-id] Started in background."
    - "[task-123] Started background tool `tool_name`"
    """
    if content.strip() == "Started in background.":
        return True
    if not content.startswith("["):
        return False

    # Find closing bracket for task ID
    bracket_end = content.find("]")
    if bracket_end == -1:
        return False

    # Check for recognized ACK phrases after the ID
    remainder = content[bracket_end + 1 :].strip()
    return remainder.startswith("Started in background.") or remainder.startswith(
        "Started background tool"
    )


class BackgroundTaskManager:
    """
    Manage interface background queries and MCP server-side background tools.

    Attributes:
        session_manager: Chat session manager used to read and persist chat
            history.
        mcp_tools_by_server: Mutable mapping of connected MCP tools by server
            name.
            The orchestrator owns the mapping and updates it as servers connect.
        collect_message_response: Callable that runs a user message and returns
            the full assistant response.
    """

    def __init__(
        self,
        session_manager: ChatSessionManager,
        mcp_tools_by_server: Dict[str, Any],
        collect_message_response: CollectMessageResponse,
    ) -> None:
        """
        Initialize the background task manager.

        Args:
            session_manager: Chat session manager used to read and persist chat
                history.
            mcp_tools_by_server: Mutable mapping of connected MCP tools by
                server name.
            collect_message_response: Callable that runs a user message and
                returns the full assistant response.
        """
        self.session_manager = session_manager
        self.mcp_tools_by_server = mcp_tools_by_server
        self.collect_message_response = collect_message_response

        self._task_lock = asyncio.Lock()
        self._next_background_task_id = 1
        self._pending_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._task_results: Dict[str, Dict[str, str]] = {}
        self._hidden_task_ids: Set[str] = set()
        self._background_tool_poll_tasks: Dict[str, asyncio.Task[None]] = {}
        self._active_agent_queries = 0

    def start_background_tool_poll_from_reply_if_needed(
        self,
        reply_text: str,
        *,
        persist_result: bool = True,
    ) -> None:
        """
        Start polling when an assistant reply contains a running MCP task descriptor.

        MCP tools can return a JSON object containing `task_id`, `status`, and
        `tool_name`. When the status is `running`, this method registers a poller
        task that waits for the server-side task result. By default, the final
        assistant message is persisted to the active chat session.

        Args:
            reply_text: Assistant reply text that may contain a background task
                descriptor as JSON.
            persist_result: Whether to persist the final result to the active
                chat session. Stateless interfaces should set this to False.

        Returns:
            None.

        Raises:
            RuntimeError: If called without a running event loop while a poller
                needs to be created.
        """
        descriptor = _parse_background_task_descriptor_payload(reply_text)
        if not isinstance(descriptor, dict) or not descriptor.get("task_id"):
            return

        task_id = descriptor["task_id"]
        status = descriptor.get("status", "running")
        tool_name = descriptor.get("tool_name", "background_tool")
        if status != "running" or task_id in self._background_tool_poll_tasks:
            return

        session_id = self.session_manager.current_session_id if persist_result else None
        poll_task = asyncio.create_task(
            self._poll_background_tool(task_id, tool_name, session_id)
        )
        self._background_tool_poll_tasks[task_id] = poll_task
        self._pending_tasks[task_id] = poll_task
        self._task_results[task_id] = {
            "status": "running",
            "type": "mcp_tool",
            "tool_name": tool_name,
        }

    def user_message_already_started_background_task(self, message: str) -> bool:
        """
        Return whether chat history already contains a background-task ack.

        This prevents isolated follow-up processing from writing a duplicate user
        turn when an interface has already persisted the background-task start
        message.

        Args:
            message: User message text to look for in chat history.

        Returns:
            True if the message is followed by a background-task acknowledgement,
            otherwise False. History loading errors are suppressed and treated
            as no match.
        """
        try:
            history = self.session_manager.load_history()
        except Exception:
            return False

        if not isinstance(history, list):
            return False

        for index, entry in enumerate(history[:-1]):
            next_entry = history[index + 1]
            if not isinstance(entry, dict) or not isinstance(next_entry, dict):
                continue
            if entry.get("role") != "user" or entry.get("content") != message:
                continue
            if next_entry.get("role") != "assistant":
                continue
            content = next_entry.get("content", "")
            if isinstance(content, str) and is_background_task_start_ack(content):
                return True

        return False

    async def _poll_background_tool(
        self,
        task_id: str,
        tool_name: str,
        session_id: str | None,
    ) -> None:
        """
        Poll a server-side MCP background tool until it finishes.

        The final result and any image attachments are written to the original
        chat session as an assistant message unless that result is already the
        latest assistant turn. The task is then removed from the pending-task
        registry.

        Args:
            task_id: Server-side MCP background task identifier.
            tool_name: Name of the MCP tool that started the task.
            session_id: Optional chat session ID where the final result should
                be persisted.

        Returns:
            None.

        Raises:
            Exception: Propagates unexpected MCP invocation or database write
                failures.
        """
        while True:
            attachments: list[ImageAttachment] = []
            result = {
                "task_id": task_id,
                "status": "not_found",
                "message": "Background task not found on connected MCP servers.",
            }
            for mcp_tool in self.mcp_tools_by_server.values():
                for fn in getattr(mcp_tool, "_functions", []):
                    if getattr(fn, "name", None) != "get_background_task_result":
                        continue

                    try:
                        payload = await fn.invoke(task_id=task_id)
                    except TypeError:
                        payload = await fn.invoke({"task_id": task_id})

                    parsed_result = _background_task_result_from_payload(
                        payload, task_id
                    )
                    if parsed_result is None:
                        parsed_result = {
                            "task_id": task_id,
                            "status": "failed",
                            "error": str(payload),
                        }

                    if str(parsed_result.get("status")).lower() != "not_found":
                        result = parsed_result
                        attachments = extract_image_attachments(payload)
                        break
                if str(result.get("status")).lower() != "not_found":
                    break

            status = str(result.get("status", "unknown")).lower()

            async with self._task_lock:
                self._task_results[task_id] = {
                    "status": status,
                    "type": "mcp_tool",
                    "tool_name": tool_name,
                }

            if status == "running":
                await asyncio.sleep(5)
                continue

            tool_message = _format_background_task_message(result.get("message"))
            if tool_message:
                message = tool_message
            elif status == "completed":
                message = _format_background_task_message(result.get("result"))
                if not message and attachments:
                    message = "\n".join(
                        f"[Image attachment: {attachment.filename}]"
                        for attachment in attachments
                    )
                if not message:
                    message = "Background job completed."
            else:
                message = _format_background_task_message(
                    result.get("error")
                    or result.get("message")
                    or result
                )
                if not message:
                    message = "Background job failed."

            if session_id is not None and not self._assistant_result_already_persisted(
                session_id, message, attachments
            ):
                if attachments:
                    self.session_manager.chat_db.add_message(
                        session_id,
                        "assistant",
                        message,
                        attachments=attachments,
                    )
                else:
                    self.session_manager.chat_db.add_message(
                        session_id, "assistant", message
                    )
            async with self._task_lock:
                self._pending_tasks.pop(task_id, None)
                self._background_tool_poll_tasks.pop(task_id, None)
            return

    def _assistant_result_already_persisted(
        self,
        session_id: str,
        message: str,
        attachments: list[ImageAttachment],
    ) -> bool:
        """Check the latest assistant turn before persisting a polled result."""
        try:
            history = self.session_manager.load_history(session_id)
        except Exception:
            return False

        latest_assistant = next(
            (
                entry
                for entry in reversed(history or [])
                if isinstance(entry, dict) and entry.get("role") == "assistant"
            ),
            None,
        )
        if not latest_assistant:
            return False

        latest_content = latest_assistant.get("content", "")
        same_message = latest_content == message or (
            isinstance(latest_content, str)
            and _format_background_task_message(latest_content) == message
        )
        if not attachments:
            return same_message

        current_digests = {attachment.digest for attachment in attachments}
        latest_digests = _attachment_digests(latest_assistant.get("attachments"))
        # The same image can arrive through the normal assistant turn and the
        # background poller with slightly different text. Treat the attachment
        # as the stable identity so the rich result is persisted only once.
        return bool(current_digests) and current_digests.issubset(latest_digests)

    async def run_query(self, user_input: str, blocking: bool = True) -> str:
        """
        Run a user query inline or as an interface background task.

        In blocking mode, the full response is collected and returned. In
        non-blocking mode, the query is detached only after the agent starts a
        tool call; simple text-only responses are still returned inline.
        Additional non-blocking queries use isolated agent sessions while a
        background agent task is already running, which avoids unsafe append-only
        merges of provider session state.

        Args:
            user_input: User query text to process.
            blocking: If True, wait for the full response inline. If False,
                return a task acknowledgement after the first tool call starts.

        Returns:
            The full assistant response, or a background-task start
            acknowledgement.

        Raises:
            Exception: Propagates failures from the response collection callable
                before the query is detached.
        """
        if blocking:
            response = await self.collect_message_response(user_input)
            print(response)
            print("")
            return response

        # Capture the chat before scheduling the task. An isolated follow-up
        # can start after the interface has switched to another chat session.
        originating_session_id = self.session_manager.current_session_id
        originating_context_session_ids = self.session_manager.get_context_session_ids()

        async with self._task_lock:
            use_isolated_session = self._active_agent_queries > 0 or any(
                task.get("type") == "agent"
                and task.get("status") in ("pending", "running")
                for task in self._task_results.values()
            )
            self._active_agent_queries += 1

        first_tool_call = asyncio.Event()
        first_tool_state = {}

        task = asyncio.create_task(
            self.collect_message_response(
                user_input,
                isolated_session=use_isolated_session,
                persistence_session_id=(
                    originating_session_id if use_isolated_session else None
                ),
                context_session_ids=(
                    originating_context_session_ids if use_isolated_session else None
                ),
                first_tool_call=first_tool_call,
                first_tool_state=first_tool_state,
            )
        )

        def _release_active_query(_: asyncio.Task[str]) -> None:
            async def _release() -> None:
                async with self._task_lock:
                    self._active_agent_queries -= 1

            asyncio.create_task(_release())

        task.add_done_callback(_release_active_query)
        tool_waiter = asyncio.create_task(first_tool_call.wait())
        done, _ = await asyncio.wait(
            {task, tool_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if task in done:
            tool_waiter.cancel()
            response = await task
            print(response)
            print("")
            return response

        async with self._task_lock:
            task_id = f"task-{self._next_background_task_id}"
            self._next_background_task_id += 1
            self._pending_tasks[task_id] = task
            self._task_results[task_id] = {
                "status": "pending",
                "type": "agent",
                "tool_name": first_tool_state.get("name", "tool call"),
            }

        def _done_callback(done_task: asyncio.Task[str]) -> None:
            async def _finalize() -> None:
                async with self._task_lock:
                    if task_id in self._hidden_task_ids:
                        self._pending_tasks.pop(task_id, None)
                        self._task_results.pop(task_id, None)
                        return

                    task_state = self._task_results.setdefault(
                        task_id, {"status": "pending"}
                    )
                    try:
                        result = done_task.result()
                        task_state["status"] = "completed"
                        print("\nBackground job completed:")
                        print(result)
                        print("")
                    except asyncio.CancelledError:
                        task_state["status"] = "cancelled"
                        print("\nBackground job cancelled.\n")
                    except Exception as e:
                        task_state["status"] = "failed"
                        print(f"\nBackground job failed: {e}\n")
                    finally:
                        self._pending_tasks.pop(task_id, None)

            asyncio.create_task(_finalize())

        task.add_done_callback(_done_callback)
        message = "Started in background."
        print(f"{message}\n")
        return message

    async def get_task_snapshot(self) -> Dict[str, Dict[str, str]]:
        """
        Return a copy of interface and MCP background task state.

        Each key is a task ID, and each value contains status metadata suitable
        for CLI or Gradio task-status displays.

        Returns:
            Deep copy of task state keyed by task ID.
        """
        async with self._task_lock:
            return copy.deepcopy(self._task_results)

    async def count_pending_tasks(self) -> int:
        """
        Return the current number of in-flight background tasks.

        Returns:
            Number of tasks currently tracked as pending.
        """
        async with self._task_lock:
            return len(self._pending_tasks)

    async def cleanup(self) -> None:
        """
        Cancel active background tasks and reset task state.

        Returns:
            None.
        """
        async with self._task_lock:
            tasks = list(self._pending_tasks.values()) + list(
                self._background_tool_poll_tasks.values()
            )

            for task in tasks:
                if not task.done():
                    task.cancel()

            self._pending_tasks.clear()
            self._task_results.clear()
            self._hidden_task_ids.clear()
            self._background_tool_poll_tasks.clear()
            self._next_background_task_id = 1
