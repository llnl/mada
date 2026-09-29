# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import time
from types import SimpleNamespace

import pytest
from agent_framework import Agent, ResponseStream

from mada.core.orchestration import magentic_strategy
from mada.core.orchestration.magentic_strategy import MagenticOrchestrationStrategy


def test_each_magentic_runtime_gets_fresh_agent_wrappers(monkeypatch):
    runtime_agents = []

    class Builder:
        def __init__(self, *, participants, manager_agent):
            runtime_agents.append((participants, manager_agent))

        def build(self):
            return object()

    monkeypatch.setattr(magentic_strategy, "MagenticBuilder", Builder)

    client = object()
    participant = Agent(
        client=client,
        name="specialist",
        description="Specialist",
        default_options={"store": True},
    )
    manager = Agent(
        client=client,
        name="manager",
        description="Manager",
        default_options={"store": True},
    )
    orchestrator = SimpleNamespace(
        specialist_agents=[participant],
        manager_agent=manager,
    )
    strategy = MagenticOrchestrationStrategy()

    strategy._build_runtime(orchestrator)
    strategy._build_runtime(orchestrator)

    first_participants, first_manager = runtime_agents[0]
    second_participants, second_manager = runtime_agents[1]
    assert first_participants[0] is not participant
    assert first_manager is not manager
    assert second_participants[0] is not first_participants[0]
    assert second_manager is not first_manager
    assert first_participants[0].client is client
    assert first_manager.client is client
    assert first_participants[0].default_options["store"] is False
    assert first_manager.default_options["store"] is False
    assert second_participants[0].default_options["store"] is False
    assert second_manager.default_options["store"] is False


def test_builder_keeps_native_stall_recovery_behind_round_limit(monkeypatch):
    captured = {}

    class Builder:
        def __init__(self, *, participants, manager_agent, **kwargs):
            captured.update(
                participants=participants,
                manager_agent=manager_agent,
                kwargs=kwargs,
            )

        def build(self):
            return object()

    monkeypatch.setattr(magentic_strategy, "MagenticBuilder", Builder)

    client = object()
    participant = Agent(client=client, name="specialist")
    manager = Agent(client=client, name="manager")
    orchestrator = SimpleNamespace(
        specialist_agents=[participant],
        manager_agent=manager,
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=10),
    )

    MagenticOrchestrationStrategy()._build_runtime(orchestrator)

    assert captured["kwargs"] == {"max_stall_count": 11}


def test_repeated_progress_reaches_stall_limit():
    strategy = MagenticOrchestrationStrategy()
    seen_round_ids = set()

    state = strategy._update_convergence(
        {"type": "progress", "text": "waiting for the same result"},
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=seen_round_ids,
    )
    assert state[3] is None

    state = strategy._update_convergence(
        {"type": "progress", "text": "waiting for the same result"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=seen_round_ids,
    )
    assert state[3] == "no-progress limit (1)"


def test_distinct_progress_updates_do_not_count_as_stalls():
    strategy = MagenticOrchestrationStrategy()
    seen_round_ids = set()
    state = strategy._update_convergence(
        {"type": "progress", "text": "specialist A completed"},
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=seen_round_ids,
    )

    state = strategy._update_convergence(
        {"type": "progress", "text": "specialist B completed"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=seen_round_ids,
    )

    assert state[1] == 0
    assert state[3] is None


def test_nested_workflow_event_types_drive_convergence():
    strategy = MagenticOrchestrationStrategy()
    seen_round_ids = set()
    event = {
        "type": "magentic_orchestrator",
        "data": {
            "event_type": "PLAN_CREATED",
            "plan": "delegate the request",
        },
    }

    state = strategy._update_convergence(
        event,
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=seen_round_ids,
    )

    assert state[0] == 0
    assert state[2] == "plan:delegate the request"


def test_native_magentic_progress_events_drive_convergence():
    strategy = MagenticOrchestrationStrategy()
    event = SimpleNamespace(
        event_type="PROGRESS_LEDGER_UPDATED",
        content={
            "is_progress_being_made": {"answer": False},
            "is_in_loop": {"answer": True},
            "next_speaker": {"answer": "researcher"},
            "instruction_or_question": {"answer": "Try again."},
        },
    )

    state = strategy._update_convergence(
        event,
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    assert state[0] == 1
    assert state[1] == 0
    assert state[3] is None

    state = strategy._update_convergence(
        event,
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[1] == 1
    assert state[3] == "no-progress limit (1)"


@pytest.mark.asyncio
async def test_wrapped_output_event_is_returned_to_user(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    async def events(_orchestrator, _messages):
        yield {
            "type": "magentic_orchestrator",
            "data": {
                "event_type": "OUTPUT",
                "text": "wrapped answer",
            },
        }

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=10),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert results[-1] == ("final", "wrapped answer")


@pytest.mark.asyncio
async def test_mixed_awaitable_async_stream_preserves_events():
    strategy = MagenticOrchestrationStrategy()

    class MixedResult:
        def __init__(self):
            self.index = 0
            self.awaited = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            values = [
                {"type": "output", "text": "first"},
                {"type": "output", "text": "second"},
            ]
            if self.index >= len(values):
                raise StopAsyncIteration
            value = values[self.index]
            self.index += 1
            return value

        def __await__(self):
            async def resolve():
                self.awaited = True
                return {"type": "final", "text": "collapsed"}

            return resolve().__await__()

        async def aclose(self):
            return None

    result = MixedResult()
    events = [event async for event in strategy._iter_result_events(result)]

    assert [event["text"] for event in events] == ["first", "second"]
    assert not result.awaited


@pytest.mark.asyncio
async def test_closing_result_event_iterator_closes_plain_generator():
    strategy = MagenticOrchestrationStrategy()
    generator_closed = False

    def source():
        nonlocal generator_closed
        try:
            yield {"type": "output", "text": "partial"}
            yield {"type": "output", "text": "unreachable"}
        finally:
            generator_closed = True

    result = source()
    events = strategy._iter_result_events(result)
    await anext(events)
    await events.aclose()

    assert generator_closed


def test_progress_ledger_state_drives_stall_detection():
    strategy = MagenticOrchestrationStrategy()
    event = {
        "type": "magentic_orchestrator",
        "data": {
            "event_type": "PROGRESS_LEDGER_UPDATED",
            "content": {
                "is_progress_being_made": {"answer": False},
                "is_in_loop": {"answer": True},
            },
        },
    }

    state = strategy._update_convergence(
        event,
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    assert state[2] == "progress:is_progress_being_made=false;is_in_loop=true"

    state = strategy._update_convergence(
        event,
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )
    assert state[3] == "no-progress limit (1)"


def test_satisfied_progress_ledger_can_finish_past_round_limit():
    strategy = MagenticOrchestrationStrategy()
    event = {
        "type": "magentic_orchestrator",
        "data": {
            "event_type": "PROGRESS_LEDGER_UPDATED",
            "content": {
                "is_request_satisfied": {"answer": True},
                "is_progress_being_made": {"answer": False},
                "is_in_loop": {"answer": True},
            },
        },
    }

    state = strategy._update_convergence(
        event,
        rounds=4,
        stalls=1,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )

    assert state[0] == 5
    assert state[3] is None


def test_progress_ledger_stall_state_counts_even_when_signature_changes():
    strategy = MagenticOrchestrationStrategy()

    def event(instruction):
        return {
            "type": "magentic_orchestrator",
            "data": {
                "event_type": "PROGRESS_LEDGER_UPDATED",
                "content": {
                    "is_progress_being_made": {"answer": False},
                    "is_in_loop": {"answer": False},
                    "next_speaker": {"answer": "researcher"},
                    "instruction_or_question": {"answer": instruction},
                },
            },
        }

    state = strategy._update_convergence(
        event("Try a different source."),
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        event("Try a broader source."),
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[1] == 1
    assert state[3] == "no-progress limit (1)"


def test_completed_executor_work_resets_stalled_ledger_state():
    strategy = MagenticOrchestrationStrategy()

    stalled_ledger = {
        "type": "magentic_orchestrator",
        "data": {
            "event_type": "PROGRESS_LEDGER_UPDATED",
            "content": {
                "is_progress_being_made": {"answer": False},
                "is_in_loop": {"answer": False},
                "next_speaker": {"answer": "researcher"},
                "instruction_or_question": {"answer": "Retry the search."},
            },
        },
    }
    state = strategy._update_convergence(
        stalled_ledger,
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    for event_type in ("executor_invoked", "executor_completed", "tool_result"):
        state = strategy._update_convergence(
            {"type": event_type, "executor_id": "same-call"},
            rounds=state[0],
            stalls=state[1],
            max_rounds=4,
            max_stalls=1,
            last_signature=state[2],
            seen_round_ids=set(),
        )
    state = strategy._update_convergence(
        stalled_ledger,
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[1] == 0
    assert state[3] is None


@pytest.mark.parametrize("result_type", ["tool_result", "function_result"])
def test_empty_tool_result_clears_stalled_ledger_state(result_type):
    strategy = MagenticOrchestrationStrategy()
    stalled_ledger = {
        "type": "magentic_orchestrator",
        "data": {
            "event_type": "PROGRESS_LEDGER_UPDATED",
            "content": {
                "is_progress_being_made": {"answer": False},
                "is_in_loop": {"answer": False},
            },
        },
    }
    state = strategy._update_convergence(
        stalled_ledger,
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": result_type, "call_id": "call-1"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        stalled_ledger,
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[1] == 0
    assert state[2] is not None
    assert state[3] is None


def test_fresh_output_resets_stalled_ledger_state():
    strategy = MagenticOrchestrationStrategy()
    state = strategy._update_convergence(
        {"type": "progress", "text": "waiting"},
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": "output", "text": "new specialist work"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": "progress", "text": "waiting"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[1] == 0
    assert state[3] is None


def test_text_progress_resets_stall_detection():
    strategy = MagenticOrchestrationStrategy()
    ledger = {
        "type": "magentic_orchestrator",
        "data": {
            "event_type": "PROGRESS_LEDGER_UPDATED",
            "content": {
                "is_progress_being_made": {"answer": True},
                "is_in_loop": {"answer": False},
                "next_speaker": {"answer": "writer"},
                "instruction_or_question": {"answer": "Continue."},
            },
        },
    }
    state = strategy._update_convergence(
        ledger,
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": "output", "text": "The specialist found a useful result."},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        ledger,
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[1] == 0
    assert state[3] is None


def test_progress_ledger_handoffs_do_not_count_as_stalls():
    strategy = MagenticOrchestrationStrategy()

    def event(next_speaker, instruction):
        return {
            "type": "magentic_orchestrator",
            "data": {
                "event_type": "PROGRESS_LEDGER_UPDATED",
                "content": {
                    "is_progress_being_made": {"answer": True},
                    "is_in_loop": {"answer": False},
                    "next_speaker": {"answer": next_speaker},
                    "instruction_or_question": {"answer": instruction},
                },
            },
        }

    state = strategy._update_convergence(
        event("researcher", "Find the relevant facts."),
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        event("writer", "Draft the answer from those facts."),
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[0] == 2
    assert state[1] == 0
    assert state[3] is None


def test_workflow_supersteps_do_not_count_as_magentic_rounds():
    strategy = MagenticOrchestrationStrategy()
    seen_round_ids = set()
    state = (0, 0, None, None)

    for superstep in range(1, 5):
        state = strategy._update_convergence(
            {
                "type": "superstep_started",
                "data": {"superstep": superstep},
            },
            rounds=state[0],
            stalls=state[1],
            max_rounds=4,
            max_stalls=10,
            last_signature=state[2],
            seen_round_ids=seen_round_ids,
        )

    assert state[0] == 0
    assert state[3] is None


def test_executor_events_do_not_count_as_magentic_rounds():
    strategy = MagenticOrchestrationStrategy()
    seen_round_ids = set()
    state = (0, 0, None, None)

    for event_type in ("executor_invoked", "executor_completed", "progress"):
        state = strategy._update_convergence(
            {
                "type": event_type,
                "executor_id": "agent-a",
                "text": "still working",
            },
            rounds=state[0],
            stalls=state[1],
            max_rounds=4,
            max_stalls=10,
            last_signature=state[2],
            seen_round_ids=seen_round_ids,
        )

    assert state[0] == 1


def test_magentic_plan_events_do_not_count_rounds():
    strategy = MagenticOrchestrationStrategy()
    seen_round_ids = set()
    state = strategy._update_convergence(
        {"type": "plan", "round_id": "r1"},
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=10,
        last_signature=None,
        seen_round_ids=seen_round_ids,
    )
    state = strategy._update_convergence(
        {"type": "replan", "round_id": "r2"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=10,
        last_signature=state[2],
        seen_round_ids=seen_round_ids,
    )

    assert state[0] == 0
    assert state[3] is None

    state = strategy._update_convergence(
        {"type": "replan", "round_id": "r3"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=2,
        max_stalls=10,
        last_signature=state[2],
        seen_round_ids=seen_round_ids,
    )

    assert state[3] is None


def test_progress_events_enforce_round_limit():
    strategy = MagenticOrchestrationStrategy()
    state = (0, 0, None, None)

    for round_number in range(1, 5):
        state = strategy._update_convergence(
            {"type": "progress", "text": f"round {round_number}"},
            rounds=state[0],
            stalls=state[1],
            max_rounds=4,
            max_stalls=10,
            last_signature=state[2],
            seen_round_ids=set(),
        )
        assert state[3] is None

    state = strategy._update_convergence(
        {"type": "progress", "text": "round 5"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=10,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[0] == 5
    assert state[3] == "round limit (4)"


def test_completed_tool_work_is_serialized_for_synthesis():
    work = MagenticOrchestrationStrategy._completed_tool_work(
        {
            "type": "tool_result",
            "executor_id": "call-1",
            "data": {"value": 42},
        }
    )

    assert "call-1" in work
    assert "42" in work


def test_executor_completion_is_not_serialized_as_tool_work():
    work = MagenticOrchestrationStrategy._completed_tool_work(
        {
            "type": "executor_completed",
            "executor_id": "participant-1",
            "output": "ordinary specialist response",
        }
    )

    assert work == ""


def test_executor_completion_preserves_nested_tool_result():
    work = MagenticOrchestrationStrategy._completed_tool_work(
        {
            "type": "executor_completed",
            "executor_id": "call-3",
            "data": {"function_result": {"content": "tool answer"}},
        }
    )

    assert "call-3" in work
    assert "tool answer" in work


def test_executor_completion_preserves_nested_function_result_event():
    work = MagenticOrchestrationStrategy._completed_tool_work(
        {
            "type": "executor_completed",
            "executor_id": "call-4",
            "data": {"event_type": "FUNCTION_RESULT", "content": "42"},
        }
    )

    assert "call-4" in work
    assert "42" in work


def test_wrapped_completed_tool_work_preserves_nested_content():
    work = MagenticOrchestrationStrategy._completed_tool_work(
        {
            "type": "magentic_orchestrator",
            "data": {
                "event_type": "TOOL_RESULT",
                "executor_id": "call-2",
                "content": {"value": "specialist result"},
            },
        }
    )

    assert work == 'call-2: {"value": "specialist result"}'


@pytest.mark.asyncio
async def test_stalled_workflow_synthesizes_and_replaces_provisional_text(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    async def events(_orchestrator, _messages):
        yield {
            "type": "magentic_orchestrator",
            "data": {
                "event_type": "TOOL_RESULT",
                "executor_id": "call-1",
                "result": {"value": 42},
            },
        }
        yield {"type": "output", "text": "draft answer"}
        yield {"type": "progress", "text": "waiting"}
        yield {"type": "progress", "text": "waiting"}

    async def synthesize(
        _orchestrator,
        _messages,
        candidate_text,
        *,
        completed_tool_work,
        timeout_seconds,
    ):
        assert candidate_text == "draft answer"
        assert completed_tool_work and "42" in completed_tool_work[0]
        assert 0 < timeout_seconds <= strategy._synthesis_timeout_budget(120)
        return "final answer"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=120),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert results[0] == ("chunk", "draft answer")
    assert results[-2][0] == "chunk"
    assert getattr(results[-2][1], "_mada_response_replacement") == "final answer"
    assert results[-1] == ("final", "final answer")


@pytest.mark.asyncio
async def test_background_task_preserves_synthesized_fallback(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    async def events(_orchestrator, _messages):
        yield {
            "type": "tool_result",
            "executor_id": "call-1",
            "result": {
                "task_id": "task-42",
                "status": "running",
                "tool_name": "long_query",
            },
        }
        yield {"type": "progress", "text": "waiting"}
        yield {"type": "progress", "text": "waiting"}

    async def synthesize(
        _orchestrator,
        _messages,
        _candidate_text,
        *,
        completed_tool_work,
        timeout_seconds,
    ):
        assert completed_tool_work
        assert timeout_seconds > 0
        return "synthesized fallback"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=120),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert any(kind == "background_task" for kind, _value in results)
    assert results[-1] == ("final", "synthesized fallback")


@pytest.mark.asyncio
async def test_process_message_persists_final_answer_without_background_ack_prefix(
    monkeypatch,
):
    strategy = MagenticOrchestrationStrategy()
    persisted = {}

    async def events(_orchestrator, _messages, *, include_tool_notices):
        assert include_tool_notices
        yield "background_task", '{"task_id":"task-42","status":"running"}'
        yield "final", "synthesized fallback"

    async def commit(_turn_id, _message, assistant_reply, **_kwargs):
        persisted["reply"] = assistant_reply

    monkeypatch.setattr(strategy, "_stream_workflow_response", events)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        _session_lock=asyncio.Lock(),
        _next_turn_id=1,
        session_manager=SimpleNamespace(load_history=lambda: []),
        _normalize_transcript_messages=lambda messages: messages,
        _commit_completed_turn=commit,
    )

    [chunk async for chunk in strategy.process_message(orchestrator, "hello")]

    assert persisted["reply"] == "synthesized fallback"


@pytest.mark.asyncio
async def test_early_convergence_closes_workflow_stream_before_synthesis(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    stream_closed = False

    async def events(_orchestrator, _messages):
        nonlocal stream_closed
        try:
            yield {"type": "progress", "text": "waiting"}
            yield {"type": "progress", "text": "waiting"}
            await asyncio.sleep(10)
        finally:
            stream_closed = True

    async def synthesize(
        _orchestrator,
        _messages,
        _candidate_text,
        *,
        completed_tool_work,
        timeout_seconds,
    ):
        assert stream_closed
        assert completed_tool_work == []
        assert timeout_seconds > 0
        return "final answer"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=120),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert stream_closed
    assert results[-1] == ("final", "final answer")


@pytest.mark.asyncio
async def test_closing_result_event_iterator_closes_underlying_stream():
    strategy = MagenticOrchestrationStrategy()
    stream_closed = False

    async def source():
        nonlocal stream_closed
        try:
            yield {"type": "output", "text": "partial"}
            await asyncio.sleep(10)
        finally:
            stream_closed = True

    events = strategy._iter_result_events(source())
    await anext(events)
    await events.aclose()

    assert stream_closed


@pytest.mark.asyncio
async def test_aborting_response_stream_runs_cleanup_hooks():
    strategy = MagenticOrchestrationStrategy()
    stream_closed = False
    cleanup_called = False

    async def source():
        nonlocal stream_closed
        try:
            yield {"type": "output", "text": "partial"}
            await asyncio.sleep(10)
        finally:
            stream_closed = True

    async def cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    stream = ResponseStream(source()).with_cleanup_hook(cleanup)
    events = strategy._iter_result_events(stream)
    await anext(events)
    await events.aclose()

    assert stream_closed
    assert cleanup_called


@pytest.mark.asyncio
async def test_aborting_unopened_response_stream_closes_source():
    strategy = MagenticOrchestrationStrategy()

    class Source:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(10)

        async def aclose(self):
            self.closed = True

    source = Source()
    stream = ResponseStream(source)

    closed = await strategy._close_async_iterator_until(stream, time.monotonic() + 1)

    assert closed
    assert source.closed


@pytest.mark.asyncio
async def test_workflow_event_teardown_is_bounded(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_finished = asyncio.Event()

    class Source:
        def __init__(self):
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index:
                await asyncio.sleep(10)
            self.index += 1
            return {"type": "output", "text": "partial"}

        async def aclose(self):
            close_started.set()
            try:
                await close_release.wait()
            finally:
                close_finished.set()

    source = Source()

    monkeypatch.setattr(strategy, "_build_runtime", lambda _orchestrator: object())
    monkeypatch.setattr(strategy, "_start_runtime", lambda _runtime, _messages: source)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        build_prompt_from_transcript=lambda _messages: "task",
    )
    events = strategy._iter_workflow_events(
        orchestrator,
        [{"role": "user", "content": "hello"}],
        close_deadline=time.monotonic() - 1,
    )
    await anext(events)

    started = time.monotonic()
    await events.aclose()

    assert time.monotonic() - started < 1
    await close_started.wait()
    close_release.set()
    await close_finished.wait()


@pytest.mark.asyncio
async def test_completed_result_final_response_is_called_once():
    strategy = MagenticOrchestrationStrategy()

    class FinalizingResult:
        _iterator = object()

        def __init__(self):
            self.index = 0
            self.final_response_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index:
                raise StopAsyncIteration
            self.index += 1
            return {"type": "output", "text": "answer"}

        async def get_final_response(self):
            self.final_response_calls += 1
            return {"type": "final", "text": "answer"}

    result = FinalizingResult()
    events = [event async for event in strategy._iter_result_events(result)]

    assert result.final_response_calls == 1
    assert events[-1] == {"type": "final", "text": "answer"}


@pytest.mark.asyncio
async def test_synthesis_handles_cumulative_compatibility_updates(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    class CompatibilityManager:
        def run(self, _prompt):
            return [
                {"type": "output", "text": "Hi"},
                {"type": "output", "text": "Hi there"},
                {"type": "output", "text": "Hi there"},
            ]

    manager = CompatibilityManager()
    monkeypatch.setattr(strategy, "_runtime_agent", lambda _agent: manager)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        build_prompt_from_transcript=lambda _messages: "task",
    )

    result = await strategy._synthesize_response(
        orchestrator,
        [{"role": "user", "content": "hello"}],
        "draft",
        completed_tool_work=[],
        timeout_seconds=1,
    )

    assert result == "Hi there"


@pytest.mark.asyncio
async def test_synthesis_replaces_streamed_text_with_wrapped_final_event(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    class CompatibilityManager:
        def run(self, _prompt):
            return [
                {"type": "output", "text": "draft"},
                {
                    "type": "magentic_orchestrator",
                    "data": {"event_type": "FINAL_OUTPUT", "text": "final"},
                },
            ]

    monkeypatch.setattr(
        strategy, "_runtime_agent", lambda _agent: CompatibilityManager()
    )
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        build_prompt_from_transcript=lambda _messages: "task",
    )

    result = await strategy._synthesize_response(
        orchestrator,
        [{"role": "user", "content": "hello"}],
        "draft",
        completed_tool_work=[],
        timeout_seconds=1,
    )

    assert result == "final"


@pytest.mark.asyncio
async def test_synthesis_preserves_mixed_awaitable_async_stream(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    class MixedResult:
        def __init__(self):
            self.awaited = False
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.closed:
                raise StopAsyncIteration
            self.closed = True
            return {"type": "output", "text": "streamed synthesis"}

        def __await__(self):
            async def resolve():
                self.awaited = True
                return {"type": "final", "text": "collapsed synthesis"}

            return resolve().__await__()

        async def aclose(self):
            self.closed = True

    mixed_result = MixedResult()

    class Manager:
        def run(self, _prompt):
            return mixed_result

    monkeypatch.setattr(strategy, "_runtime_agent", lambda _agent: Manager())
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        build_prompt_from_transcript=lambda _messages: "task",
    )

    result = await strategy._synthesize_response(
        orchestrator,
        [{"role": "user", "content": "hello"}],
        "draft",
        completed_tool_work=[],
        timeout_seconds=1,
    )

    assert result == "streamed synthesis"
    assert not mixed_result.awaited


@pytest.mark.asyncio
async def test_synthesis_stream_teardown_is_bounded_by_timeout(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    close_release = asyncio.Event()
    close_started = asyncio.Event()
    close_finished = asyncio.Event()

    class SlowStream:
        async def __anext__(self):
            await asyncio.sleep(10)

        def __aiter__(self):
            return self

        async def aclose(self):
            close_started.set()
            try:
                await close_release.wait()
            finally:
                close_finished.set()

    stream = SlowStream()

    class Manager:
        def run(self, _prompt):
            return stream

    monkeypatch.setattr(strategy, "_runtime_agent", lambda _agent: Manager())
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        build_prompt_from_transcript=lambda _messages: "task",
    )

    started = time.monotonic()
    result = await strategy._synthesize_response(
        orchestrator,
        [{"role": "user", "content": "hello"}],
        "draft",
        completed_tool_work=[],
        timeout_seconds=0.01,
    )

    assert result == ""
    assert time.monotonic() - started < 1
    close_release.set()
    if close_started.is_set():
        await close_finished.wait()


def test_duplicate_text_updates_do_not_reach_stall_limit():
    strategy = MagenticOrchestrationStrategy()
    state = strategy._update_convergence(
        {"type": "output", "text": "same cumulative answer"},
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": "output", "text": "same cumulative answer"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[1] == 0
    assert state[3] is None


def test_progress_stall_survives_cumulative_output_updates():
    strategy = MagenticOrchestrationStrategy()
    state = strategy._update_convergence(
        {"type": "output", "text": "draft"},
        rounds=0,
        stalls=0,
        max_rounds=4,
        max_stalls=1,
        last_signature=None,
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": "progress", "text": "waiting"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": "output", "text": "draft"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )
    state = strategy._update_convergence(
        {"type": "progress", "text": "waiting"},
        rounds=state[0],
        stalls=state[1],
        max_rounds=4,
        max_stalls=1,
        last_signature=state[2],
        seen_round_ids=set(),
    )

    assert state[3] == "no-progress limit (1)"


@pytest.mark.asyncio
async def test_duplicate_cumulative_output_does_not_preempt_final_event(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    async def events(_orchestrator, _messages):
        yield {"type": "output", "text": "Hi"}
        yield {"type": "output", "text": "Hi there"}
        yield {"type": "output", "text": "Hi there"}
        yield {"type": "final", "text": "authoritative answer"}

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=10),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert results[-1] == ("final", "authoritative answer")


@pytest.mark.asyncio
async def test_workflow_reserves_synthesis_timeout(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    timeout_values = []
    original_timeout = asyncio.timeout

    def tracked_timeout(value):
        timeout_values.append(value)
        return original_timeout(value)

    async def events(_orchestrator, _messages):
        yield {"type": "final", "text": "complete"}

    monkeypatch.setattr(magentic_strategy.asyncio, "timeout", tracked_timeout)
    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=10),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert timeout_values[0] == 8.0
    assert len(timeout_values) == 2
    assert 0 < timeout_values[1] <= 8.0
    assert results[-1] == ("final", "complete")


@pytest.mark.asyncio
async def test_teardown_timeout_preserves_emitted_final_answer(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    synthesis_called = False

    async def events(_orchestrator, _messages):
        yield {"type": "final", "text": "authoritative answer"}

    async def close(_value):
        raise asyncio.TimeoutError

    async def synthesize(*_args, **_kwargs):
        nonlocal synthesis_called
        synthesis_called = True
        return "fallback answer"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_close_async_iterator", close)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=10),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert not synthesis_called
    assert results[-1] == ("final", "authoritative answer")


@pytest.mark.asyncio
async def test_teardown_timeout_preserves_convergence_synthesis_reason(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    async def events(_orchestrator, _messages):
        yield {"type": "output", "text": "draft answer"}
        yield {"type": "progress", "text": "waiting"}
        yield {"type": "progress", "text": "waiting"}

    async def close(_value):
        await asyncio.sleep(10)

    async def synthesize(
        _orchestrator,
        _messages,
        candidate_text,
        *,
        completed_tool_work,
        timeout_seconds,
    ):
        assert candidate_text == "draft answer"
        assert not completed_tool_work
        assert timeout_seconds > 0
        return "synthesized answer"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_close_async_iterator", close)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=1),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert results[-1] == ("final", "synthesized answer")


@pytest.mark.asyncio
async def test_teardown_timeout_synthesizes_provisional_output(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    async def events(_orchestrator, _messages):
        yield {"type": "output", "text": "provisional answer"}

    async def close(_value):
        raise asyncio.TimeoutError

    async def synthesize(
        _orchestrator,
        _messages,
        candidate_text,
        *,
        completed_tool_work,
        timeout_seconds,
    ):
        assert candidate_text == "provisional answer"
        assert not completed_tool_work
        assert timeout_seconds > 0
        return "synthesized answer"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_close_async_iterator", close)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=10),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert results[-1] == ("final", "synthesized answer")


@pytest.mark.asyncio
async def test_workflow_timeout_synthesizes_with_reserved_budget(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    synthesis_called = False

    async def events(_orchestrator, _messages):
        await asyncio.sleep(2)
        yield {"type": "output", "text": "late draft"}

    async def synthesize(*_args, **_kwargs):
        nonlocal synthesis_called
        synthesis_called = True
        return "best effort answer"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=1),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert synthesis_called
    assert results[-1] == ("final", "best effort answer")


@pytest.mark.asyncio
async def test_timeout_synthesizes_instead_of_using_partial_stream(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    synthesis_called = False

    async def events(_orchestrator, _messages):
        yield {"type": "output", "text": "completed streamed answer"}
        await asyncio.sleep(2)

    async def synthesize(*_args, **_kwargs):
        nonlocal synthesis_called
        synthesis_called = True
        return "fallback answer"

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)
    monkeypatch.setattr(strategy, "_synthesize_response", synthesize)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        orchestration=SimpleNamespace(max_rounds=4, max_stalls=1, timeout_seconds=1),
    )

    results = []
    async for kind, value in strategy._stream_workflow_response(
        orchestrator, [{"role": "user", "content": "hello"}], include_tool_notices=False
    ):
        results.append((kind, value))

    assert synthesis_called
    assert results[-1] == ("final", "fallback answer")


@pytest.mark.asyncio
async def test_stream_cleanup_is_bounded_by_request_deadline(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    close_release = asyncio.Event()
    close_started = asyncio.Event()
    close_finished = asyncio.Event()
    close_cancelled = False

    async def slow_close(_value):
        nonlocal close_cancelled
        close_started.set()
        try:
            await close_release.wait()
        except asyncio.CancelledError:
            close_cancelled = True
            raise
        finally:
            close_finished.set()

    monkeypatch.setattr(strategy, "_close_async_iterator", slow_close)
    started = time.monotonic()
    close_task = asyncio.create_task(
        strategy._close_async_iterator_until(object(), started + 0.01)
    )
    await close_started.wait()
    closed = await close_task

    assert not closed
    assert time.monotonic() - started < 1
    assert close_cancelled
    close_release.set()
    await close_finished.wait()


@pytest.mark.asyncio
async def test_completed_post_deadline_cleanup_reports_success(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    async def close(_value):
        return None

    monkeypatch.setattr(strategy, "_close_async_iterator", close)
    closed = await strategy._close_async_iterator_until(
        object(),
        time.monotonic() - 1,
        initiate_after_deadline=True,
    )

    assert closed


@pytest.mark.asyncio
async def test_stream_cleanup_skips_after_deadline(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    close_started = asyncio.Event()

    async def slow_close(_value):
        close_started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr(strategy, "_close_async_iterator", slow_close)
    closed = await strategy._close_async_iterator_until(object(), time.monotonic() - 1)

    assert not closed
    await asyncio.sleep(0)
    assert not close_started.is_set()


@pytest.mark.asyncio
async def test_stream_cleanup_propagates_caller_cancellation(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    close_started = asyncio.Event()

    async def slow_close(_value):
        close_started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr(strategy, "_close_async_iterator", slow_close)
    task = asyncio.create_task(
        strategy._close_async_iterator_until(object(), time.monotonic() + 10)
    )
    await close_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_teardown_failure_is_best_effort():
    strategy = MagenticOrchestrationStrategy()

    async def close(_value):
        raise RuntimeError("cleanup failed")

    strategy._close_async_iterator = close
    closed = await strategy._close_async_iterator_until(object(), time.monotonic() + 1)

    assert not closed


@pytest.mark.asyncio
async def test_synchronous_synthesis_result_is_supported(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    class Manager:
        calls = 0

        def run(self, _prompt, *, stream=False):
            self.calls += 1
            return "final answer"

    manager = Manager()
    monkeypatch.setattr(strategy, "_runtime_agent", lambda _agent: manager)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        build_prompt_from_transcript=lambda _messages: "task",
    )

    result = await strategy._synthesize_response(
        orchestrator,
        [{"role": "user", "content": "hello"}],
        "draft",
        completed_tool_work=[],
        timeout_seconds=1,
    )

    assert result == "final answer"
    assert manager.calls == 1


@pytest.mark.asyncio
async def test_awaitable_synthesis_run_is_cancelled_by_timeout(monkeypatch):
    strategy = MagenticOrchestrationStrategy()

    class AsyncManager:
        calls = 0

        def run(self, _prompt, *, stream=False):
            self.calls += 1

            async def response():
                await asyncio.sleep(10)
                return "late answer"

            return response()

    manager = AsyncManager()
    monkeypatch.setattr(strategy, "_runtime_agent", lambda _agent: manager)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        build_prompt_from_transcript=lambda _messages: "task",
    )

    result = await strategy._synthesize_response(
        orchestrator,
        [{"role": "user", "content": "hello"}],
        "draft",
        completed_tool_work=[],
        timeout_seconds=0.01,
    )

    assert result == ""
    assert manager.calls == 1


def test_clone_agent_preserves_top_level_configuration(monkeypatch):
    constructed = {}

    class VariantAgent:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(magentic_strategy, "Agent", VariantAgent)

    explicit_tool = object()
    mcp_tool = object()
    source = SimpleNamespace(
        client=object(),
        instructions="Use the specialist instructions.",
        tools=[explicit_tool],
        mcp_tools=[explicit_tool, mcp_tool],
        default_options={
            "instructions": "stale instructions",
            "tools": [object()],
            "temperature": 0.2,
        },
    )

    clone = MagenticOrchestrationStrategy._clone_agent(source)

    assert clone is not source
    assert constructed["instructions"] == source.instructions
    assert constructed["tools"] == [explicit_tool, mcp_tool]
    assert constructed["default_options"] == {"temperature": 0.2, "store": False}


def test_clone_agent_normalizes_single_tool_configurations(monkeypatch):
    constructed = {}

    class VariantAgent:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(magentic_strategy, "Agent", VariantAgent)

    def tool():
        return None

    top_level_source = SimpleNamespace(
        client=object(),
        tools=tool,
        default_options={},
    )
    MagenticOrchestrationStrategy._clone_agent(top_level_source)
    assert constructed["tools"] == [tool]

    constructed.clear()
    legacy_source = SimpleNamespace(
        client=object(),
        default_options={"tools": tool},
    )
    MagenticOrchestrationStrategy._clone_agent(legacy_source)
    assert constructed["tools"] == [tool]


def test_clone_agent_overrides_source_store_option(monkeypatch):
    constructed = {}

    class VariantAgent:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(magentic_strategy, "Agent", VariantAgent)

    source = SimpleNamespace(
        client=object(),
        default_options={
            "store": True,
            "temperature": 0.2,
            "conversation_id": "shared-conversation",
        },
    )

    MagenticOrchestrationStrategy._clone_agent(source)

    assert constructed["default_options"] == {
        "store": False,
        "temperature": 0.2,
    }
    assert source.default_options["store"] is True


def test_clone_agent_preserves_legacy_default_option_configuration(monkeypatch):
    constructed = {}

    class LegacyAgent:
        def __init__(self, *, client, default_options):
            constructed.update(client=client, default_options=default_options)
            self.client = client
            self.default_options = default_options

    monkeypatch.setattr(magentic_strategy, "Agent", LegacyAgent)

    tools = [object()]
    mcp_tool = object()
    source = SimpleNamespace(
        client=object(),
        id="specialist-id",
        name="specialist",
        description="Specialist description",
        mcp_tools=[mcp_tool],
        default_options={
            "instructions": "Use the specialist instructions.",
            "tools": tools,
            "temperature": 0.2,
        },
    )

    clone = MagenticOrchestrationStrategy._clone_agent(source)

    assert clone is not source
    assert constructed["client"] is source.client
    assert constructed["default_options"] == {
        "instructions": "Use the specialist instructions.",
        "tools": [tools[0], mcp_tool],
        "temperature": 0.2,
        "store": False,
    }
    assert clone.id == source.id
    assert clone.name == source.name
    assert clone.description == source.description


def test_clone_agent_preserves_tools_instructions_and_unrelated_options(
    monkeypatch,
):
    constructed = {}

    class LegacyAgent:
        def __init__(self, *, client, default_options):
            constructed.update(client=client, default_options=default_options)

    monkeypatch.setattr(magentic_strategy, "Agent", LegacyAgent)

    explicit_tool = object()
    source = SimpleNamespace(
        client=object(),
        default_options={
            "instructions": "Keep these instructions.",
            "tools": [explicit_tool],
            "temperature": 0.4,
            "max_tokens": 200,
        },
    )

    MagenticOrchestrationStrategy._clone_agent(source)

    assert constructed["default_options"] == {
        "instructions": "Keep these instructions.",
        "tools": [explicit_tool],
        "temperature": 0.4,
        "max_tokens": 200,
        "store": False,
    }
