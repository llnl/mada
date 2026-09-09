# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
A2A interface and remote agent configuration definitions.

`A2AConfig` models MADA's own A2A identity for `a2a.self` when MADA is run as
an A2A server. `RemoteA2AAgentConfig` models remote agents under `a2a.agents`
that the orchestrator can call as tools.
"""

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

from mada.core.config.utils import expand_env_vars


@dataclass
class A2AConfig:
    """
    Configuration for the Agent-to-Agent HTTP interface.

    Attributes:
        name: Public agent name advertised in the A2A agent card.
        description: Public agent description advertised in the A2A agent card.
        version: Public agent version advertised in the A2A agent card.
        url: Externally reachable A2A endpoint URL. When omitted, the runtime
            host and port are used to build a local URL for the agent card.
        card_path: Optional path to a standalone A2A agent card JSON file.
            When provided, the A2A interface serves this card and overrides its
            `url` field with the runtime public URL.
        skills: Optional skill entries to expose in the A2A agent card. When
            omitted, skills are derived from configured MADA agents.
    """

    name: str = "MADA"
    description: str = "MADA multi-agent orchestration service"
    version: str = "0.2.0"
    url: str = ""
    card_path: str = ""
    skills: list[dict[str, Any]] = field(default_factory=list)
    card_path_base: InitVar[str | Path | None] = None

    def __post_init__(self, card_path_base: str | Path | None) -> None:
        """
        Normalize fields and validate generated-card skill entries.
        """
        self.name = expand_env_vars(self.name or "").strip() or "MADA"
        self.description = (
            expand_env_vars(self.description or "").strip()
            or "MADA multi-agent orchestration service"
        )
        self.version = expand_env_vars(self.version or "").strip() or "0.2.0"
        self.url = expand_env_vars(self.url or "").strip()
        self.card_path = expand_env_vars(self.card_path or "").strip()
        if self.card_path and card_path_base:
            resolved_card_path = Path(self.card_path)
            if not resolved_card_path.is_absolute():
                self.card_path = str(
                    (Path(card_path_base) / resolved_card_path).resolve()
                )

        if self.skills is None:
            self.skills = []
        if not isinstance(self.skills, list):
            raise ValueError("'a2a.self.skills' must be a list")
        for skill in self.skills:
            if not isinstance(skill, dict):
                raise ValueError("'a2a.self.skills' must contain only objects")


def load_a2a_config(
    config_dict: dict[str, Any] | None,
    card_path_base: str | Path | None = None,
) -> A2AConfig:
    """
    Load self A2A configuration from a dictionary.

    Args:
        config_dict: Serialized A2A settings, or `None`.

    Returns:
        A validated A2A configuration object.
    """
    if config_dict is None:
        return A2AConfig(card_path_base=card_path_base)

    if not isinstance(config_dict, dict):
        raise ValueError("'a2a.self' must be an object")

    return A2AConfig(**config_dict, card_path_base=card_path_base)


@dataclass
class RemoteA2AAgentConfig:
    """
    Configuration for a remote A2A agent that MADA can delegate work to.

    Attributes:
        url: JSON-RPC endpoint for the remote A2A agent.
        card_url: Optional explicit URL for the remote A2A agent card. When
            omitted, MADA discovers the card from standard paths derived from
            `url`.
        timeout: HTTP timeout in seconds for calls to this remote agent.
        api_key: Optional API key sent as `x-api-key`.
        headers: Optional additional HTTP headers.
    """

    url: str
    card_url: str = ""
    timeout: float = 180.0
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize fields and validate required remote-agent settings.
        """
        self.url = expand_env_vars(self.url or "").strip()
        if not self.url:
            raise ValueError("'a2a.agents.<name>.url' must not be empty")

        self.card_url = expand_env_vars(self.card_url or "").strip()
        self.api_key = expand_env_vars(self.api_key or "").strip()

        if not isinstance(self.headers, dict):
            raise ValueError("'a2a.agents.<name>.headers' must be an object")

        expanded_headers = {}
        for key, value in self.headers.items():
            expanded_headers[str(key)] = expand_env_vars(str(value))
        self.headers = expanded_headers


def load_a2a_agents_config(
    config_dict: dict[str, Any] | None,
) -> dict[str, RemoteA2AAgentConfig]:
    """
    Load remote A2A agent definitions from a dictionary.
    """
    if config_dict is None:
        return {}

    if not isinstance(config_dict, dict):
        raise ValueError("'a2a.agents' must be an object")

    agents = {}
    for name, agent_config in config_dict.items():
        if not isinstance(agent_config, dict):
            raise ValueError("'a2a.agents' values must be objects")
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("'a2a.agents' must not contain empty names")
        agents[clean_name] = RemoteA2AAgentConfig(**agent_config)

    return agents
