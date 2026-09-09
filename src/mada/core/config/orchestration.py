# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Orchestration configuration definitions and validation utilities.

This module defines the top-level orchestration settings used to choose the
internal coordination pattern for MADA and, optionally, constrain which
specialist agents participate in that pattern.
"""

from dataclasses import dataclass
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
    """

    mode: str = DEFAULT_ORCHESTRATION_MODE
    participants: list[str] | None = None

    def __post_init__(self) -> None:
        normalized_mode = expand_env_vars(self.mode or "").strip().lower()
        self.mode = normalized_mode or DEFAULT_ORCHESTRATION_MODE

        if self.mode not in SUPPORTED_ORCHESTRATION_MODES:
            raise ValueError(f"unsupported orchestration mode: {self.mode}")

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

    def validate_participants(self, agent_names: Iterable[str]) -> None:
        """
        Validate configured participants against the configured agents.

        Args:
            agent_names: Names of configured agents.

        Raises:
            ValueError: If `participants` names do not match configured
                specialist agents.
        """
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
