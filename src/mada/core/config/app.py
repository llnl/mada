# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Application configuration definitions and loading utilities.

This module defines the top-level configuration model for the MADA
multi-agent application. It provides `AppConfig`, which aggregates model,
agent, database, MCP server, and interface configuration objects, along with
`load_config_from_json` for loading a complete application configuration from
a JSON file.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

from mada.core.config.agents import AgentConfig
from mada.core.config.database import DatabaseConfig, load_database_config
from mada.core.config.interface import InterfaceConfig
from mada.core.config.mcp_servers import MCPServerConfig
from mada.core.config.models import ModelConfig, load_model_config
from mada.core.config.orchestration import (
    OrchestrationConfig,
    load_orchestration_config,
)


LOG = logging.getLogger("mada-interface")


@dataclass
class AppConfig:
    """
    Top-level configuration for the MADA multi-agent application.

    This configuration aggregates model settings, interface layout
    settings, and agent definitions. It can be loaded from a JSON file
    using the `from_dict` method.

    Attributes:
        model (ModelConfig): Provider-specific configuration for the model backend.
        agents (List[AgentConfig]): List of agent definitions including server paths and descriptions.
        database (DatabaseConfig): Configuration for the database connection.
        interface (InterfaceConfig): Configuration for the Gradio interface layout and options.
    """

    model: ModelConfig
    agents: List[AgentConfig]
    database: DatabaseConfig
    mcp_servers: Dict[str, MCPServerConfig] = None  # MCP server configurations
    interface: InterfaceConfig = None  # Optional, used only by the Gradio app
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """
        Create an AppConfig instance from a dictionary.

        This method extracts and constructs the model, interface, and
        agent configurations from a flat configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing keys 'model', 'interface', and 'agents'.

        Returns:
            AppConfig: The fully populated application configuration.

        Raises:
            ValueError: If the 'agents' list is missing or empty.
        """
        app_conf = {}

        # Load model configuration
        model_settings = config_dict.get("model")
        if not model_settings:
            raise ValueError(
                "Please provide 'model' settings in the configuration for your app."
            )
        model_cfg = load_model_config(model_settings)
        app_conf["model"] = model_cfg

        # Load agents
        agent_entries = config_dict.get("agents")
        if not agent_entries:
            raise ValueError(
                "No agents were provided. Please define at least one agent."
            )
        agent_cfgs = [AgentConfig.from_dict(agent) for agent in agent_entries]
        app_conf["agents"] = agent_cfgs

        # Load database
        database_config = config_dict.get("database", {})
        app_conf["database"] = load_database_config(database_config)

        orchestration_cfg = load_orchestration_config(config_dict.get("orchestration"))
        orchestration_cfg.validate_participants(
            [agent.agent_name for agent in agent_cfgs]
        )
        app_conf["orchestration"] = orchestration_cfg

        # Load MCP servers configuration (optional)
        python_exe = config_dict.get("python_executable", sys.executable)
        mcp_servers_entry = config_dict.get("mcp_servers")
        if mcp_servers_entry:
            mcp_servers_cfg = {}
            for name, server_config in mcp_servers_entry.items():
                if "python_executable" not in server_config:
                    server_config = {**server_config, "python_executable": python_exe}
                mcp_servers_cfg[name] = MCPServerConfig(**server_config)
            app_conf["mcp_servers"] = mcp_servers_cfg

        # Load interface configuration (optional for multiagent app)
        interface_entry = config_dict.get("interface")
        if interface_entry:
            interface_cfg = InterfaceConfig(**config_dict["interface"])
            app_conf["interface"] = interface_cfg

        return cls(**app_conf)


def load_config_from_json(path: str) -> AppConfig:
    """
    Load application configuration from a JSON file.

    Args:
        path (str): Path to the JSON configuration file.

    Returns:
        AppConfig: The parsed application configuration object.
    """
    with open(path, "r") as f:
        config_dict = json.load(f)

    return AppConfig.from_dict(config_dict)
