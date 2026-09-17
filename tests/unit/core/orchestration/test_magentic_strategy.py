# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

from agent_framework import Agent

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


def test_clone_agent_overrides_source_store_option(monkeypatch):
    constructed = {}

    class VariantAgent:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(magentic_strategy, "Agent", VariantAgent)

    source = SimpleNamespace(
        client=object(),
        default_options={"store": True, "temperature": 0.2},
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
