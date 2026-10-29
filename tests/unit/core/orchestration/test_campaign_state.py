# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from agent_framework import FunctionInvocationContext, tool

from mada.core.config import OrchestrationConfig
from mada.core.orchestration import magentic_strategy
from mada.core.orchestration.campaign_state import CampaignLedger, execution_requested
from mada.core.orchestration.magentic_runtime import (
    CURRENT_ACTIVITY,
    RuntimeActivity,
    ToolActivityMiddleware,
)
from mada.core.orchestration.magentic_strategy import MagenticOrchestrationStrategy


def policy(tmp_path):
    return {
        "campaign_id": "case",
        "ledger_path": str(tmp_path / "ledger.db"),
        "limit": 3,
        "initial_limit": 2,
        "submission_tool": "submit_jobs",
        "batch_argument": "run_info_json",
        "critique_agents": ["SimulationAgent", "JobManagementAgent"],
        "analysis_tool": "read_vertex_campaign",
        "allowed_tools": [
            "submit_jobs",
            "generate_parameter_runs",
            "read_vertex_campaign",
            "read_image",
            "check_job_status",
        ],
        "read_only_tools": ["read_vertex_campaign", "read_image", "check_job_status"],
    }


def runs():
    return [{"id": str(i), "run_location": f"/case/run{i}"} for i in range(4)]


def evidence(comparable=True, count=2):
    return {
        "manifest_runs": runs()[:count],
        "runs": [
            {
                "id": str(i),
                "parameters": {"velocity_0": i + 0.5},
                "velocity_probe": {"speed": i + 0.1, "evaluation_time": 10},
                "time_integration_complete_in_log": True,
            }
            for i in range(count)
        ],
        "comparison": {
            "same_final_saved_time": comparable,
            "note": "Compare the saved times",
        },
    }


def test_persistent_split_initial_gate_restart_and_negative_critique(tmp_path):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    ledger.reserve(runs()[:1])
    ledger.reserve(runs()[1:2])
    restarted = CampaignLedger(p["ledger_path"], "case", p)
    assert len(restarted.snapshot()["runs"]) == 2
    with pytest.raises(ValueError, match="Follow-up locked"):
        restarted.reserve(runs()[2:3])
    with pytest.raises(ValueError, match="completed"):
        restarted.critique("SimulationAgent", "Too early", True)
    wrong = evidence()
    wrong["manifest_runs"][0]["run_location"] = "/another/run0"
    restarted.observe(wrong)
    assert not restarted.snapshot()["results_available"]
    restarted.observe(evidence())
    restarted.critique("SimulationAgent", "Explore the upper range", True)
    with pytest.raises(ValueError, match="each required specialist"):
        restarted.reserve(runs()[2:3])
    restarted.critique("JobManagementAgent", "Cost not justified", False)
    with pytest.raises(ValueError, match="each required specialist"):
        restarted.reserve(runs()[2:3])
    assert restarted.critique("JobManagementAgent", "Cost is justified", True)[
        "followup_released"
    ]
    restarted.reserve(runs()[2:3])
    with pytest.raises(ValueError, match="budget exceeded"):
        CampaignLedger(p["ledger_path"], "case", p).reserve(runs()[3:])
    with pytest.raises(ValueError, match="policy changed"):
        CampaignLedger(p["ledger_path"], "case", {**p, "limit": 4})


def test_parallel_requests_share_atomic_reservations(tmp_path):
    p = policy(tmp_path)

    def attempt(index):
        ledger = CampaignLedger(p["ledger_path"], "case", p)
        try:
            ledger.reserve([runs()[index]])
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        accepted = list(executor.map(attempt, range(4)))
    assert sum(accepted) == 2


def test_existing_submissions_import_is_idempotent_and_cannot_reset(tmp_path):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    assert ledger.import_submitted(runs()[:2]) == 2
    assert ledger.import_submitted(runs()[:2]) == 2
    with pytest.raises(ValueError, match="Cannot replace"):
        ledger.import_submitted(runs()[1:3])
    with pytest.raises(ValueError, match="Follow-up locked"):
        ledger.reserve(runs()[2:3])
    assert len(ledger.snapshot()["runs"]) == 2


def test_concurrent_followups_cannot_spend_the_same_slot(tmp_path):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    ledger.reserve(runs()[:2])
    ledger.observe(evidence())
    for agent in p["critique_agents"]:
        ledger.critique(agent, "A follow-up is warranted", True)

    def reserve(index):
        try:
            CampaignLedger(p["ledger_path"], "case", p).reserve([runs()[index]])
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sum(executor.map(reserve, (2, 3))) == 1
    assert len(ledger.snapshot()["runs"]) == 3


def test_different_times_and_changed_evidence_keep_followup_locked(tmp_path):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    ledger.reserve(runs()[:2])
    ledger.observe(evidence(False))
    for agent in p["critique_agents"]:
        assert not ledger.critique(agent, "Recommend another run", True)[
            "followup_released"
        ]
    with pytest.raises(ValueError, match="comparable"):
        ledger.reserve(runs()[2:3])
    ledger.observe(evidence(True))
    assert ledger.snapshot()["critiques"] == {}
    for agent in p["critique_agents"]:
        ledger.critique(agent, "Now comparable", True)
    updated = evidence()
    updated["runs"][0]["velocity_probe"]["speed"] = 0.9
    ledger.observe(updated)
    assert ledger.snapshot()["critiques"] == {}


def test_completed_followup_evidence_and_critiques_survive_restart(tmp_path):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    ledger.reserve(runs()[:2])
    ledger.observe(evidence())
    for agent in p["critique_agents"]:
        ledger.critique(agent, "Run the follow-up", True)
    ledger.reserve(runs()[2:3])

    # Old or unrelated reports cannot mark the newly reserved run complete.
    ledger.observe(evidence())
    wrong = evidence(count=3)
    wrong["manifest_runs"][2]["run_location"] = "/another/run2"
    ledger.observe(wrong)
    assert not ledger.snapshot()["results_available"]

    ledger.observe(evidence(count=3))
    assert ledger.snapshot()["results_available"]
    assert ledger.snapshot()["comparable"]
    for agent in p["critique_agents"]:
        result = ledger.critique(agent, "Final results reviewed", True)
        assert not result["followup_released"]  # The budget is exhausted.
    restarted = CampaignLedger(p["ledger_path"], "case", p)
    assert restarted.snapshot() == ledger.snapshot()
    restarted.observe(evidence(count=3))
    assert set(restarted.snapshot()["critiques"]) == set(p["critique_agents"])
    with pytest.raises(ValueError, match="budget exceeded"):
        restarted.reserve(runs()[3:])


@pytest.mark.parametrize("change", ["order", "comparison_metadata"])
def test_equivalent_evidence_preserves_critiques_and_followup_gate(tmp_path, change):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    ledger.reserve(runs()[:2])
    report = evidence()
    report["runs"].reverse()
    ledger.observe(report)
    for agent in p["critique_agents"]:
        ledger.critique(agent, "Follow up", True)
    before = ledger.snapshot()

    reordered = evidence()
    reordered["manifest_runs"].reverse()
    if change == "comparison_metadata":
        reordered["comparison"]["note"] = "Updated explanatory text"
        reordered["comparison"]["display_title"] = "Completed runs"
    ledger.observe(reordered)
    assert ledger.snapshot() == before
    assert ledger.reserve(runs()[2:3]) == 3


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize(
    "invalid", ["swapped_ids", "missing_ids", "missing_result_id", "duplicate_ids"]
)
def test_observations_preserve_ids_and_run_mapping(tmp_path, count, invalid):
    p = {**policy(tmp_path), "initial_limit": count}
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    reserved = runs()[:count]
    report = evidence(count=count)
    if invalid == "missing_ids":
        for row in report["manifest_runs"] + report["runs"]:
            row.pop("id")
    elif invalid == "swapped_ids":
        if count == 2:
            report["manifest_runs"][0]["id"], report["manifest_runs"][1]["id"] = (
                "1",
                "0",
            )
        else:
            report["manifest_runs"][0]["id"] = report["runs"][0]["id"] = "other"
    elif invalid == "missing_result_id":
        report["runs"][0].pop("id")
    else:
        report["manifest_runs"].append(dict(report["manifest_runs"][0]))
        report["runs"].append(dict(report["runs"][0]))
    ledger.reserve(reserved)
    before = ledger.snapshot()
    ledger.observe(report)
    assert ledger.snapshot() == before
    with pytest.raises(ValueError, match="completed"):
        ledger.critique("SimulationAgent", "Follow up", True)


@pytest.mark.parametrize("method", ["reserve", "import_submitted"])
@pytest.mark.parametrize("invalid_id", [None, "", " \t", [], {}, True])
def test_persistent_runs_require_ids_before_writing(tmp_path, method, invalid_id):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    before = ledger.snapshot()
    invalid = {"run_location": "/case/run0"}
    if invalid_id is not None:
        invalid["id"] = invalid_id
    with pytest.raises(ValueError, match="nonempty.*id"):
        getattr(ledger, method)([invalid])
    assert ledger.snapshot() == before


def test_persistent_ids_are_unique_within_and_across_submissions(tmp_path):
    p = policy(tmp_path)
    ledger = CampaignLedger(p["ledger_path"], "case", p)
    first = {"id": 0, "run_location": "/case/run0"}
    reused_id = {"id": "0", "run_location": "/case/run1"}
    with pytest.raises(ValueError, match="IDs must be unique"):
        ledger.reserve([first, reused_id])
    assert ledger.snapshot()["runs"] == []
    ledger.reserve([first])
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="IDs must be unique"):
        ledger.reserve([reused_id])
    assert ledger.snapshot() == before


@pytest.mark.parametrize(
    "text",
    [
        "Review the two simulations already completed. Don't run any new simulations.",
        "Analyze existing results",
        "Run a campaign. Do not submit any simulations.",
        "What happened to the campaign?",
        "analysis-only: run the campaign",
        "Please submit one follow-up simulation. Do not submit any new jobs.",
        "Generate two new jobs. No new simulations.",
        "Generate a summary of the simulation jobs.",
        "Please submit the campaign analysis.",
        "Please run the campaign analysis",
        "Continue the simulations analysis",
        "Run an analysis of the campaign",
        "Run the campaign results review",
        "No new simulations, please run the campaign analysis.",
        "Please run the campaign. No more jobs.",
        "No need to run the campaign, please run the simulations analysis.",
        "Do not continue the simulations, please run the campaign analysis.",
    ],
)
def test_analysis_or_ambiguous_requests_are_read_only(text):
    assert not execution_requested(text)


@pytest.mark.parametrize(
    "text",
    [
        "Run a constrained Vertex-CFD campaign using at most 3 simulations.",
        "Please continue the campaign.",
        "Can you execute one follow-up simulation?",
        "No rush, please run the campaign",
        "No worries, continue the simulations",
        "Run a campaign with no more than 3 simulations.",
        "Please submit one follow-up simulation",
        "Generate two new jobs",
        "Can you submit one job?",
        "Please generate two simulations.",
    ],
)
def test_explicit_execution_request(text):
    assert execution_requested(text)
    assert not execution_requested(text, "analysis")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, name",
    [
        ("Please submit one follow-up simulation", "submit_jobs"),
        ("Generate two new jobs", "generate_parameter_runs"),
    ],
)
async def test_submit_generate_requests_enable_tools_without_bypassing_budget(
    tmp_path, prompt, name
):
    p = policy(tmp_path)
    activity = RuntimeActivity(budget=p, read_only=not execution_requested(prompt))
    activity.ledger.reserve(runs()[:2])
    arguments = {"run_info_json": json.dumps(runs()[2:3])}
    if name == "submit_jobs":
        with pytest.raises(ValueError, match="Follow-up locked"):
            await activity.reserve(name, arguments)
        activity.ledger.observe(evidence())
        for agent in p["critique_agents"]:
            activity.ledger.critique(agent, "Follow up", True)
    await activity.reserve(name, arguments)
    assert len(activity.ledger.snapshot()["runs"]) == (
        3 if name == "submit_jobs" else 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [
        "Analyze existing results",
        "Please run the campaign analysis",
        "Continue the simulations analysis",
    ],
)
async def test_analysis_middleware_blocks_execution_before_io(tmp_path, prompt):
    p = policy(tmp_path)
    activity = RuntimeActivity(budget=p, read_only=not execution_requested(prompt))

    @tool
    async def submit_jobs(run_info_json: str):
        return "unexpected"

    context = FunctionInvocationContext(
        function=submit_jobs, arguments={"run_info_json": json.dumps(runs()[:2])}
    )

    async def forbidden():
        pytest.fail("Execution must be blocked before invoking the server")

    await ToolActivityMiddleware(activity, "JobManagementAgent", []).process(
        context, forbidden
    )
    assert "Analysis-only" in context.result
    with pytest.raises(ValueError, match="Analysis-only"):
        await activity.reserve("generate_parameter_runs", {})
    await activity.reserve("read_vertex_campaign", {})
    assert activity.ledger.snapshot()["runs"] == []


@pytest.mark.asyncio
async def test_analysis_blocks_critique_writes_directly_and_through_middleware(
    tmp_path,
):
    p = policy(tmp_path)
    activity = RuntimeActivity(budget=p, read_only=True)
    activity.ledger.reserve(runs()[:2])
    activity.ledger.observe(evidence())
    before = activity.ledger.snapshot()
    critic = activity.critique_tool("SimulationAgent")
    arguments = {"summary": "Run a follow-up", "recommend_followup": True}

    with pytest.raises(ValueError, match="Analysis-only"):
        await critic.invoke(arguments=arguments)

    context = FunctionInvocationContext(function=critic, arguments=arguments)

    async def forbidden():
        pytest.fail("Analysis must not invoke a ledger-writing tool")

    await ToolActivityMiddleware(activity, "SimulationAgent", [critic]).process(
        context, forbidden
    )
    assert json.loads(context.result)["status"] == "blocked"
    assert activity.ledger.snapshot() == before
    with pytest.raises(ValueError, match="each required specialist"):
        activity.ledger.reserve(runs()[2:3])


def test_compact_synthesis_preserves_second_run_and_comparison_before_logs():
    report = evidence(False)
    for row in report["runs"]:
        row["stdout_tail"] = "lots of solver output" * 5000
    strategy = MagenticOrchestrationStrategy()
    captured = strategy._completed_tool_work(
        {"type": "tool_result", "data": {"result": json.dumps(report)}}
    )[:8000]
    assert '"same_final_saved_time": false' in captured
    assert '"speed": 1.1' in captured
    assert "Compare the saved times" in captured
    assert "lots of solver output" not in captured


@pytest.mark.asyncio
async def test_critique_tool_is_bound_to_its_specialist(tmp_path):
    p = policy(tmp_path)
    activity = RuntimeActivity(budget=p)
    activity.ledger.reserve(runs()[:2])
    activity.ledger.observe(evidence())
    critic = activity.critique_tool("SimulationAgent")
    await critic.invoke(
        arguments={"summary": "Diversity needs improvement", "recommend_followup": True}
    )
    assert set(activity.ledger.snapshot()["critiques"]) == {"SimulationAgent"}
    with pytest.raises(ValueError, match="required campaign critic"):
        activity.ledger.critique("PlanningAgent", "Bypass", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["top_level", "default_options"])
@pytest.mark.parametrize("with_existing_tools", [False, True])
async def test_builder_exposes_critique_in_active_tool_layout(
    monkeypatch, tmp_path, layout, with_existing_tools
):
    class VariantAgent:
        def __init__(self, *, tools=None, default_options=None, **kwargs):
            self.__dict__.update(kwargs)
            self.default_options = dict(default_options or {})
            if layout == "top_level":
                self.tools = list(tools or [])
            else:
                self.default_options["tools"] = list(tools or [])

    class Builder:
        def __init__(self, *, participants, manager_agent):
            self.participants = participants

    monkeypatch.setattr(magentic_strategy, "Agent", VariantAgent)
    monkeypatch.setattr(magentic_strategy, "MagenticBuilder", Builder)

    @tool
    async def existing_tool() -> str:
        """An existing specialist capability."""
        return "Available"

    original_tools = [existing_tool] if with_existing_tools else []
    specialists = [
        VariantAgent(client=object(), name=name, tools=original_tools)
        for name in policy(tmp_path)["critique_agents"]
    ]
    activity = RuntimeActivity(budget=policy(tmp_path))
    activity.ledger.reserve(runs()[:2])
    activity.ledger.observe(evidence())
    orchestrator = SimpleNamespace(
        specialist_agents=specialists,
        manager_agent=VariantAgent(client=object(), name="manager"),
    )

    def active_tools(agent):
        return agent.tools if layout == "top_level" else agent.default_options["tools"]

    token = CURRENT_ACTIVITY.set(activity)
    try:
        strategy = MagenticOrchestrationStrategy()
        builder = strategy._create_builder(orchestrator)
        second_builder = strategy._create_builder(orchestrator)
    finally:
        CURRENT_ACTIVITY.reset(token)

    for source, participant, second in zip(
        specialists, builder.participants, second_builder.participants
    ):
        assert active_tools(source) == original_tools
        assert active_tools(participant)[:-1] == original_tools
        critic = active_tools(participant)[-1]
        assert critic.name == "record_campaign_critique"
        assert critic is not active_tools(second)[-1]
        await critic.invoke(
            arguments={"summary": "A follow-up is useful", "recommend_followup": True}
        )

    assert set(activity.ledger.snapshot()["critiques"]) == set(
        policy(tmp_path)["critique_agents"]
    )
    assert activity.ledger.reserve(runs()[2:3]) == 3


def test_campaign_path_resolution_and_invalid_configuration(tmp_path):
    p = policy(tmp_path)
    p["ledger_path"] = "budgets/state.db"
    cfg = OrchestrationConfig(mode="magentic", simulation_budget=p)
    cfg.resolve_campaign_paths(tmp_path)
    assert p["ledger_path"] == str(tmp_path / "budgets/state.db")
    for change in (
        {"critique_agents": ["SimulationAgent"]},
        {"read_only_tools": ["submit_jobs", "read_vertex_campaign"]},
        {"execution_mode": "unrestricted"},
    ):
        with pytest.raises(ValueError):
            OrchestrationConfig(
                mode="magentic", simulation_budget={**policy(tmp_path), **change}
            )
