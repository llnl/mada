# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Per-request tool accounting and live activity for Magentic workflows."""

from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from agent_framework import FunctionMiddleware, tool

from mada.core.media import strip_image_payload_text
from mada.core.orchestration.campaign_state import CampaignLedger, run_identity


def payload_objects(value: Any):
    """Read structured MCP results without interpreting prose as executable data."""
    if isinstance(value, str):
        try:
            yield from payload_objects(json.loads(value))
        except (ValueError, TypeError):
            return
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from payload_objects(item)
    elif isinstance(value, dict):
        yield value
        for key in ("text", "content", "contents", "items", "result", "data"):
            if key in value:
                yield from payload_objects(value[key])
    elif hasattr(value, "to_dict"):
        yield from payload_objects(value.to_dict())


def submission_batch(arguments: Any, argument: str | None = None) -> list:
    """Count explicit run entries; ambiguous/file-based batches fail closed."""
    if hasattr(arguments, "model_dump"):
        arguments = arguments.model_dump()

    def decode(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    if argument:
        value = arguments
        for part in argument.split("."):
            value = decode(value)
            if not isinstance(value, dict) or part not in value:
                raise ValueError(
                    f"Submission requires an explicit batch at {argument!r}."
                )
            value = value[part]
        candidates = [decode(value)]
    else:
        # Inspect only batch containers, not nested parameter vectors inside jobs.
        def batches(value, key=None):
            value = decode(value)
            if isinstance(value, list):
                return (
                    [value]
                    if key in {"jobs", "runs", "parameter_runs", "job_specs"}
                    else []
                )
            if isinstance(value, dict):
                return [
                    batch
                    for name, item in value.items()
                    for batch in batches(item, name)
                ]
            return []

        candidates = batches(arguments)
    if len(candidates) != 1 or not isinstance(candidates[0], list) or not candidates[0]:
        raise ValueError(
            "Submission must contain exactly one nonempty explicit jobs/runs/parameter_runs/job_specs list, or use the configured batch_argument path; file paths and ambiguous batches are not accepted."
        )
    # String-encoded specs and object specs must have the same budget identity.
    batch = [decode(run) for run in candidates[0]]
    if any(not isinstance(run, dict) or not run for run in batch):
        raise ValueError(
            "Each submitted run must resolve to a nonempty job object; "
            "raw strings and file paths are not accepted."
        )
    return batch


@dataclass
class RuntimeActivity:
    budget: dict[str, Any] | None = None
    read_only: bool = False
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    submitted: int = 0
    submissions: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started: float = field(default_factory=time.monotonic)
    sequence: int = 0
    closed: bool = False
    ledger: CampaignLedger | None = field(default=None, init=False)

    def __post_init__(self):
        if self.budget and self.budget.get("campaign_id"):
            self.ledger = CampaignLedger(
                self.budget["ledger_path"], self.budget["campaign_id"], self.budget
            )
            self.submitted = len(self.ledger.snapshot()["runs"])

    def critique_tool(self, agent: str):
        @tool
        async def record_campaign_critique(
            summary: str, recommend_followup: bool
        ) -> str:
            """Record your own public critique after reading completed initial results.

            The follow-up remains locked until both designated specialists have
            recorded critiques recommending it and initial results are comparable.
            """
            if self.closed or self.ledger is None:
                raise ValueError("Campaign is not active")
            if self.read_only:
                raise ValueError(
                    "Analysis-only request: campaign critiques cannot be recorded"
                )
            return json.dumps(self.ledger.critique(agent, summary, recommend_followup))

        return record_campaign_critique

    def emit(self, event_type: str, agent: str, **data):
        self.queue.put_nowait({"type": event_type, "executor_id": agent, "data": data})

    async def reserve(self, name: str, arguments: Any):
        if self.closed:
            raise ValueError(
                "The workflow has stopped; no further tool calls are permitted."
            )
        if not self.budget:
            return
        if self.read_only and name not in self.budget.get("read_only_tools", []):
            raise ValueError(f"Analysis-only request: tool {name!r} is disabled")
        if name == "record_campaign_critique" and self.ledger:
            return
        allowed = self.budget["allowed_tools"]
        if name not in allowed:
            raise ValueError(
                f"Tool {name!r} is not enabled for this constrained campaign."
            )
        if name != self.budget["submission_tool"]:
            return
        batch = submission_batch(arguments, self.budget.get("batch_argument"))
        if self.ledger:
            # SQLite reserves atomically across requests, processes and restarts.
            self.submitted = self.ledger.reserve(batch)
            return
        fingerprints = [run_identity(run) for run in batch]
        async with self.lock:
            if self.closed:
                raise ValueError(
                    "The workflow has stopped; no further submissions are permitted."
                )
            if not self.submitted and len(batch) > self.budget.get(
                "initial_limit", self.budget["limit"]
            ):
                raise ValueError(
                    "Initial batch is too large; reserve capacity for the follow-up decision."
                )
            if len(set(fingerprints)) != len(
                fingerprints
            ) or self.submissions.intersection(fingerprints):
                raise ValueError(
                    "Duplicate submission blocked. Inspect the original jobs; do not resubmit an uncertain or failed run."
                )
            if self.submitted + len(batch) > self.budget["limit"]:
                raise ValueError(
                    f"Simulation budget exceeded: {self.submitted}/{self.budget['limit']} already reserved; requested {len(batch)} more."
                )
            # Reserve before I/O and never refund on cancellation/network ambiguity.
            self.submitted += len(batch)
            self.submissions.update(fingerprints)


CURRENT_ACTIVITY: ContextVar[RuntimeActivity | None] = ContextVar(
    "magentic_activity", default=None
)


class ToolActivityMiddleware(FunctionMiddleware):
    def __init__(self, activity: RuntimeActivity, agent: str, tools: list[Any]):
        self.activity = activity
        self.agent = agent
        self.tools = tools

    async def process(self, context, call_next):
        name = context.function.name
        self.activity.sequence += 1
        call_id = f"tool-{self.activity.sequence}"
        started = time.monotonic()
        try:
            await self.activity.reserve(name, context.arguments)
        except ValueError as error:
            context.result = json.dumps({"status": "blocked", "error": str(error)})
            self.activity.emit(
                "activity",
                self.agent,
                name=name,
                call_id=call_id,
                status="blocked",
                detail=str(error),
            )
            self.activity.emit(
                "tool_result", self.agent, name=name, result=context.result
            )
            return
        self.activity.emit(
            "activity",
            self.agent,
            name=name,
            call_id=call_id,
            status="started",
            submitted=self.activity.submitted,
        )
        if self.activity.budget and name == self.activity.budget["submission_tool"]:
            arguments = context.arguments
            if hasattr(arguments, "model_dump"):
                arguments = arguments.model_dump()
            self.activity.emit(
                "submission_reserved",
                self.agent,
                name=name,
                result={
                    "status": "submission_attempted",
                    "reserved_simulations": self.activity.submitted,
                    "limit": self.activity.budget["limit"],
                    "arguments": arguments,
                },
            )
        try:
            await call_next()
            if self.activity.budget and not await self._await_background_result(
                context, name, call_id
            ):
                self.activity.emit(
                    "tool_result", self.agent, name=name, result=context.result
                )
                return
        except BaseException as error:
            status = (
                "interrupted" if isinstance(error, asyncio.CancelledError) else "failed"
            )
            self.activity.emit(
                "activity",
                self.agent,
                name=name,
                call_id=call_id,
                status=status,
                detail=str(error),
                elapsed=time.monotonic() - started,
            )
            raise
        result = context.result
        if self.activity.ledger and name == self.activity.budget.get("analysis_tool"):
            for item in payload_objects(result):
                if "manifest_runs" in item:
                    self.activity.ledger.observe(item)
        failed = any(
            item.get("isError")
            or item.get("is_error")
            or item.get("status") in {"failed", "error"}
            for item in payload_objects(result)
        )
        self.activity.emit(
            "activity",
            self.agent,
            name=name,
            call_id=call_id,
            status="failed" if failed else "completed",
            elapsed=time.monotonic() - started,
        )
        self.activity.emit("tool_result", self.agent, name=name, result=result)

    def _background_poller(self, function):
        groups = [
            getattr(tool, "functions", getattr(tool, "_functions", [])) or []
            for tool in self.tools
        ]
        owners = [group for group in groups if any(fn is function for fn in group)]
        if not owners:
            owners = [
                group
                for group in groups
                if any(getattr(fn, "name", None) == function.name for fn in group)
            ]
        if len(owners) != 1:
            return None
        # Task IDs are server-local: never borrow a poller without an owner.
        candidates = owners[0]
        pollers = {
            id(fn): fn
            for fn in candidates
            if getattr(fn, "name", None) == "get_background_task_result"
        }
        return next(iter(pollers.values())) if len(pollers) == 1 else None

    async def _await_background_result(self, context, name: str, call_id: str) -> bool:
        """Return False if background work remains unresolved without a poller."""
        descriptors = {
            str(item["task_id"]): item
            for item in payload_objects(context.result)
            if item.get("task_id")
            and item.get("status") in {"running", "pending", "queued"}
        }
        if not descriptors:
            return True
        # Poll only the MCP server owning this function. Task IDs need not be
        # unique across servers. Keep results in the same invocation so the
        # next specialist receives completed work without another user turn.
        poll = self._background_poller(context.function)
        if poll is None:
            self.activity.emit(
                "activity",
                self.agent,
                name=name,
                call_id=call_id,
                status="pending",
                detail="No background-result polling tool is available; execution is not complete.",
            )
            return False
        self.activity.emit("tool_result", self.agent, name=name, result=context.result)
        results = []
        for task_id in descriptors:
            while True:
                arguments = {"task_id": task_id}
                try:
                    await self.activity.reserve(poll.name, arguments)
                except ValueError as error:
                    self.activity.emit(
                        "activity",
                        self.agent,
                        name=name,
                        call_id=call_id,
                        status="pending",
                        detail=f"Background polling blocked: {error}",
                    )
                    return False
                result = await poll.invoke(arguments=arguments)
                payload = next(
                    (item for item in payload_objects(result) if "status" in item), None
                )
                if payload is None:
                    raise ValueError(
                        f"Unrecognized background status for task {task_id}; do not resubmit."
                    )
                status = payload["status"]
                self.activity.emit(
                    "activity",
                    self.agent,
                    name=name,
                    call_id=call_id,
                    status=status,
                    detail=f"Task {task_id}",
                )
                if status not in {"running", "pending", "queued"}:
                    results.append(payload)
                    self.activity.emit("task_resolved", self.agent, task_id=task_id)
                    break
                await asyncio.sleep(self.activity.budget.get("poll_seconds", 2))
        context.result = json.dumps(
            {"result": results},
            default=lambda value: (
                value.to_dict() if hasattr(value, "to_dict") else str(value)
            ),
        )
        return True


def activity_text(event: dict) -> str:
    data = event["data"]
    elapsed = f" ({data['elapsed']:.1f}s)" if "elapsed" in data else ""
    detail = strip_image_payload_text(str(data.get("detail", "")))[:500]
    return f"\n{event['executor_id']}: `{data.get('name', 'workflow')}` {data['status']}{elapsed}. {detail}\n"
