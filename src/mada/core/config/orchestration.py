# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Orchestration configuration definitions and validation utilities.

This module defines the top-level orchestration settings used to choose the
internal coordination pattern for MADA and, optionally, constrain which
specialist agents participate in that pattern.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mada.core.config.utils import expand_env_vars


DEFAULT_ORCHESTRATION_MODE = "agent-as-tool"
SUPPORTED_ORCHESTRATION_MODES = frozenset(
    {
        DEFAULT_ORCHESTRATION_MODE,
        "magentic",
    }
)


@dataclass
class OrchestrationConfig:
    """
    Configuration for MADA orchestration behavior.

    Attributes:
        mode:
            Internal orchestration pattern to use.
        participants:
            Optional ordered list of specialist agent names to include in the
            orchestration pattern. When omitted, all non-`PlanningAgent`
            agents participate.
        max_rounds:
            Maximum number of Magentic coordination rounds for one request.
        timeout_seconds:
            Maximum wall-clock time for one Magentic request.
        max_stalls:
            Maximum number of consecutive Magentic cycles without meaningful
            progress before synthesis is requested.
        simulation_budget:
            Optional submission policy: limit, initial_limit,
            submission_tool, allowed_tools, optional batch_argument and
            poll_seconds. A campaign_id and ledger_path enable durable accounting,
            critique_agents gate follow-ups, and read_only_tools restrict analysis
            requests. Requires explicit one-simulation-per-job batches.
    """

    mode: str = DEFAULT_ORCHESTRATION_MODE
    participants: list[str] | None = None
    max_rounds: int = 4
    timeout_seconds: int = 120
    max_stalls: int = 1
    simulation_budget: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized_mode = expand_env_vars(self.mode or "").strip().lower()
        self.mode = normalized_mode or DEFAULT_ORCHESTRATION_MODE

        for field_name in ("max_rounds", "timeout_seconds", "max_stalls"):
            value = getattr(self, field_name)
            if isinstance(value, str):
                expanded_value = expand_env_vars(value)
                if expanded_value != value:
                    try:
                        value = int(expanded_value.strip())
                    except (AttributeError, ValueError):
                        # Let the common validation below report the field-specific
                        # positive-integer error for malformed env values.
                        pass
                    setattr(self, field_name, value)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"'orchestration.{field_name}' must be a positive integer"
                )

        if self.mode not in SUPPORTED_ORCHESTRATION_MODES:
            raise ValueError(f"unsupported orchestration mode: {self.mode}")

        if self.simulation_budget is not None:
            budget = self.simulation_budget
            if not isinstance(budget, dict):
                raise ValueError("simulation_budget must be an object")
            if self.mode != "magentic":
                raise ValueError("simulation_budget requires magentic mode")
            limit = budget.get("limit")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("simulation_budget.limit must be a positive integer")
            initial = budget.get("initial_limit", limit)
            if (
                isinstance(initial, bool)
                or not isinstance(initial, int)
                or not 1 <= initial <= limit
            ):
                raise ValueError(
                    "simulation_budget.initial_limit must be between 1 and limit"
                )
            allowed = budget.get("allowed_tools")
            if (
                not isinstance(allowed, list)
                or not allowed
                or any(
                    not isinstance(name, str) or not name.strip() for name in allowed
                )
            ):
                raise ValueError(
                    "simulation_budget.allowed_tools must list permitted tool names"
                )
            if budget.get("submission_tool") not in allowed:
                raise ValueError(
                    "simulation_budget.submission_tool must be in allowed_tools"
                )
            if "campaign_id" in budget:
                for key in ("campaign_id", "ledger_path", "analysis_tool"):
                    value = budget.get(key)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"simulation_budget.{key} is required for persistent campaigns"
                        )
                    budget[key] = expand_env_vars(value).strip()
                    if "${" in budget[key]:
                        raise ValueError(
                            f"simulation_budget.{key} has unresolved environment variables"
                        )
                critics = budget.get("critique_agents")
                if (
                    not isinstance(critics, list)
                    or len(critics) < 2
                    or any(
                        not isinstance(name, str) or not name.strip()
                        for name in critics
                    )
                    or len(set(critics)) != len(critics)
                ):
                    raise ValueError(
                        "Persistent campaigns require at least two distinct critique_agents"
                    )
                read_only = budget.get("read_only_tools")
                if (
                    not isinstance(read_only, list)
                    or not read_only
                    or any(
                        not isinstance(name, str) or name not in allowed
                        for name in read_only
                    )
                    or budget["submission_tool"] in read_only
                    or budget["analysis_tool"] not in read_only
                ):
                    raise ValueError(
                        "read_only_tools must include analysis_tool, be allowed, and exclude submission_tool"
                    )
                if budget.get("execution_mode", "auto") not in {"auto", "analysis"}:
                    raise ValueError("execution_mode must be auto or analysis")
            argument = budget.get("batch_argument")
            if argument is not None and (
                not isinstance(argument, str) or not argument.strip()
            ):
                raise ValueError(
                    "simulation_budget.batch_argument must be a nonempty dotted argument path"
                )
            poll = budget.get("poll_seconds", 2)
            if (
                isinstance(poll, bool)
                or not isinstance(poll, (int, float))
                or not 0 < poll <= 60
            ):
                raise ValueError(
                    "simulation_budget.poll_seconds must be between 0 and 60"
                )

        if self.participants is None:
            return

        if not isinstance(self.participants, list):
            raise ValueError(
                "'orchestration.participants' must be a list of agent names"
            )

        normalized_participants = []
        for participant in self.participants:
            if not isinstance(participant, str):
                raise ValueError(
                    "'orchestration.participants' must contain only agent names"
                )

            participant_name = participant.strip()
            if not participant_name:
                raise ValueError(
                    "'orchestration.participants' must not contain empty agent names"
                )

            normalized_participants.append(participant_name)

        self.participants = normalized_participants

    def resolve_campaign_paths(self, config_dir: str | Path | None) -> None:
        if self.simulation_budget and self.simulation_budget.get("campaign_id"):
            path = Path(self.simulation_budget["ledger_path"]).expanduser()
            if not path.is_absolute():
                path = Path(config_dir or Path.cwd()) / path
            self.simulation_budget["ledger_path"] = str(path.resolve())

    def validate_participants(self, agent_names: Iterable[str]) -> None:
        """
        Validate configured participants against the configured agents.

        Args:
            agent_names: Names of configured agents.

        Raises:
            ValueError: If `participants` names do not match configured
                specialist agents.
        """
        agent_names = list(agent_names)
        if self.simulation_budget and self.simulation_budget.get("campaign_id"):
            unknown = set(self.simulation_budget["critique_agents"]) - set(
                self.participants or agent_names
            )
            if unknown or "PlanningAgent" in self.simulation_budget["critique_agents"]:
                raise ValueError("critique_agents must be specialist participants")
        if self.participants is None:
            return

        if "PlanningAgent" in self.participants:
            raise ValueError(
                "PlanningAgent cannot be selected as an orchestration participant"
            )

        available_names = {name for name in agent_names if name != "PlanningAgent"}
        unknown_names = [
            participant
            for participant in self.participants
            if participant not in available_names
        ]
        if unknown_names:
            missing = ", ".join(unknown_names)
            raise ValueError(f"unknown orchestration participants: {missing}")


def load_orchestration_config(
    config_dict: dict[str, Any] | None,
) -> OrchestrationConfig:
    """
    Load orchestration configuration from a dictionary.

    Args:
        config_dict: Serialized orchestration settings, or `None`.

    Returns:
        A validated orchestration configuration object.
    """
    if config_dict is None:
        return OrchestrationConfig()

    if not isinstance(config_dict, dict):
        raise ValueError("'orchestration' must be an object")

    return OrchestrationConfig(**config_dict)
