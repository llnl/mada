# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Agent configuration definitions and serialization utilities.

This module defines the `AgentConfig` dataclass used to represent individual
agent settings in the MADA system. It includes fields for agent identity,
descriptive metadata, MCP server connections, instructions, and additional
arbitrary configuration, along with helper methods for table display and
dictionary-based serialization and deserialization.
"""

import json
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


LOG = logging.getLogger("mada-interface")


@dataclass
class AgentConfig:
    """
    Configuration for an individual agent in the MADA system.

    This class defines the identifying information, MCP server connections,
    and optional metadata for an autonomous agent.

    Attributes:
        agent_name (str): The name or identifier of the agent.
        description (str): A short human-readable description of the agent.
        mcp_servers (List[str]): List of MCP server names this agent should connect to.
        domain (Optional[str]): The domain or specialization of the agent.
        instructions (Optional[str]): A system prompt used to initialize the agent's behavior.
        server_path (Optional[str]): Legacy field for backward compatibility.
    """

    agent_name: str
    description: str
    mcp_servers: List[str]
    domain: Optional[str] = ""
    instructions: Optional[str] = ""
    server_path: Optional[str] = ""  # Legacy field for backward compatibility
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_headers(cls) -> List[str]:
        """
        Return the field names formatted as human-readable table headers.

        Returns:
            List[str]: List of field names with underscores replaced by spaces and capitalized.
        """
        return [attr.name.replace("_", " ").title() for attr in fields(cls)]

    def to_dict(self) -> Dict[str, str]:
        """
        Convert this AgentConfig instance into a dictionary with raw field names as keys.

        Returns:
            Dict[str, str]: A dictionary representation of the agent configuration.
        """
        result = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, list):
                # Convert list to comma-separated string for display
                result[name] = ", ".join(value) if value else ""
            elif isinstance(value, dict):
                result[name] = json.dumps(value)
            else:
                result[name] = str(value or "")
        return result

    @classmethod
    def from_dict(cls, agent_dict: Dict[str, Any]) -> "AgentConfig":
        """
        Create an AgentConfig instance from a dictionary using raw field names as keys.

        Args:
            agent_dict (Dict[str, str]): Dictionary with keys matching AgentConfig field names.

        Returns:
            AgentConfig: A new AgentConfig instance initialized from the dictionary.
        """
        kwargs = {}

        # Backward compatibility: map system_message -> instructions if needed
        if (
            "instructions" not in agent_dict or not agent_dict.get("instructions")
        ) and "system_message" in agent_dict:
            agent_dict = dict(agent_dict)  # shallow copy to avoid mutating caller
            agent_dict["instructions"] = agent_dict.get("system_message", "")

        for name, dataclass_field in cls.__dataclass_fields__.items():
            value = agent_dict.get(name, "")

            # Handle list fields (like mcp_servers)
            if dataclass_field.type == List[str] or str(
                dataclass_field.type
            ).startswith("typing.List"):
                if isinstance(value, str):
                    # Convert comma-separated string back to list
                    kwargs[name] = [s.strip() for s in value.split(",") if s.strip()]
                elif isinstance(value, list):
                    kwargs[name] = value
                else:
                    kwargs[name] = []
            # Handle dict fields (like extra)
            elif dataclass_field.type == Dict[str, Any] or str(
                dataclass_field.type
            ).startswith("typing.Dict"):
                if isinstance(value, str):
                    try:
                        kwargs[name] = json.loads(value)
                    except Exception:
                        kwargs[name] = {}
                elif isinstance(value, dict):
                    kwargs[name] = value
                else:
                    kwargs[name] = {}
            else:
                if isinstance(value, str):
                    kwargs[name] = str(value)
                elif isinstance(value, list):
                    kwargs[name] = " ".join(value)
                else:
                    kwargs[name] = ""

        return cls(**kwargs)
