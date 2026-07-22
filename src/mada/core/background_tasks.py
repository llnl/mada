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


CollectMessageResponse = Callable[..., Awaitable[str]]


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

    def start_background_tool_poll_from_reply_if_needed(self, reply_text: str) -> None:
        """
        Start polling when an assistant reply contains a running MCP task descriptor.

        MCP tools can return a JSON object containing `task_id`, `status`, and
        `tool_name`. When the status is `running`, this method registers a poller
        task that waits for the server-side task result and persists the final
        assistant message.

        Args:
            reply_text: Assistant reply text that may contain a background task
                descriptor as JSON.

        Returns:
            None.

        Raises:
            RuntimeError: If called without a running event loop while a poller
                needs to be created.
        """
        reply_text = reply_text.strip()
        try:
            descriptor = json.loads(reply_text)
        except json.JSONDecodeError:
            descriptor = None
            start = reply_text.find("{")
            end = reply_text.rfind("}")
            if start != -1 and end > start:
                try:
                    descriptor = json.loads(reply_text[start : end + 1])
                except json.JSONDecodeError:
                    descriptor = None

        if not isinstance(descriptor, dict) or not descriptor.get("task_id"):
            return

        task_id = descriptor["task_id"]
        status = descriptor.get("status", "running")
        tool_name = descriptor.get("tool_name", "background_tool")
        if status != "running" or task_id in self._background_tool_poll_tasks:
            return

        session_id = self.session_manager.current_session_id
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
            if isinstance(content, str) and content.startswith("[task-"):
                return "Started in background." in content

        return False

    async def _poll_background_tool(
        self,
        task_id: str,
        tool_name: str,
        session_id: str,
    ) -> None:
        """
        Poll a server-side MCP background tool until it finishes.

        The final result is written to the original chat session as an assistant
        message, and the task is removed from the pending-task registry.

        Args:
            task_id: Server-side MCP background task identifier.
            tool_name: Name of the MCP tool that started the task.
            session_id: Chat session ID where the final result should be
                persisted.

        Returns:
            None.

        Raises:
            Exception: Propagates unexpected MCP invocation or database write
                failures.
        """
        while True:
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

                    if isinstance(payload, list) and payload:
                        payload = payload[0]
                    if hasattr(payload, "text"):
                        payload = payload.text
                    if isinstance(payload, bytes):
                        payload = payload.decode("utf-8", errors="replace")

                    try:
                        payload = (
                            json.loads(payload) if isinstance(payload, str) else payload
                        )
                    except (TypeError, json.JSONDecodeError):
                        payload = {
                            "task_id": task_id,
                            "status": "failed",
                            "error": str(payload),
                        }

                    if (
                        isinstance(payload, dict)
                        and payload.get("status") != "not_found"
                    ):
                        result = payload
                        break
                if result.get("status") != "not_found":
                    break

            status = result.get("status", "unknown")

            async with self._task_lock:
                self._task_results[task_id] = {
                    "status": status,
                    "type": "mcp_tool",
                    "tool_name": tool_name,
                }

            if status == "running":
                await asyncio.sleep(5)
                continue

            if status == "completed":
                output = result.get("result", "")
                if not isinstance(output, str):
                    output = json.dumps(output, indent=2, default=str)
                message = (
                    f"[{task_id}] Background tool `{tool_name}` completed:\n{output}"
                )
            else:
                output = (
                    result.get("error")
                    or result.get("message")
                    or json.dumps(result, indent=2, default=str)
                )
                message = (
                    f"[{task_id}] Background tool `{tool_name}` {status}:\n{output}"
                )

            self.session_manager.chat_db.add_message(session_id, "assistant", message)
            async with self._task_lock:
                self._pending_tasks.pop(task_id, None)
            return

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

        async with self._task_lock:
            use_isolated_session = any(
                task.get("type") == "agent"
                and task.get("status") in ("pending", "running")
                for task in self._task_results.values()
            )

        first_tool_call = asyncio.Event()
        first_tool_state = {}

        task = asyncio.create_task(
            self.collect_message_response(
                user_input,
                isolated_session=use_isolated_session,
                first_tool_call=first_tool_call,
                first_tool_state=first_tool_state,
            )
        )
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
                        print(f"\n[{task_id}] Completed:")
                        print(result)
                        print("")
                    except asyncio.CancelledError:
                        task_state["status"] = "cancelled"
                        print(f"\n[{task_id}] Cancelled.\n")
                    except Exception as e:
                        task_state["status"] = "failed"
                        print(f"\n[{task_id}] Failed: {e}\n")
                    finally:
                        self._pending_tasks.pop(task_id, None)

            asyncio.create_task(_finalize())

        task.add_done_callback(_done_callback)
        message = f"[{task_id}] Started in background."
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
