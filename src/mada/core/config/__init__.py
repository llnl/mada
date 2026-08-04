# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Configuration package for the MADA orchestrator.

This package provides configuration models and helper utilities used to define
application behavior, model backends, agents, database settings, MCP server
connections, interface settings, and environment variable expansion.

Modules:
    agents:
        Defines [`AgentConfig`][core.config.agents.AgentConfig] for individual
        agent configuration and serialization helpers.
    app:
        Defines [`AppConfig`][core.config.app.AppConfig] and utilities for loading
        full application configuration from JSON.
    database:
        Defines database configuration models for SQLite and PostgreSQL, along
        with database config loading helpers.
    interface:
        Defines [`InterfaceConfig`][core.config.interface.InterfaceConfig] for
        multi-agent interface layout and UI customization settings.
    mcp_servers:
        Defines [`MCPServerConfig`][core.config.mcp_servers.MCPServerConfig] for
        individual MCP server connection and launch settings.
    models:
        Defines provider model configuration classes and model config loading
        helpers.
    orchestration:
        Defines [`OrchestrationConfig`][core.config.orchestration.OrchestrationConfig]
        and helpers for loading supported orchestration mode settings.
    utils:
        Defines shared utility helpers, including environment variable
        expansion.
"""

from mada.core.config.agents import AgentConfig
from mada.core.config.app import AppConfig, load_config_from_json
from mada.core.config.database import (
    DatabaseConfig,
    PostgreSQLConfig,
    SQLiteConfig,
    load_database_config,
)
from mada.core.config.interface import InterfaceConfig
from mada.core.config.mcp_servers import MCPServerConfig
from mada.core.config.models import (
    ModelConfig,
    BaseModelConfig,
    BedrockModelConfig,
    OpenAIModelConfig,
    load_model_config,
)
from mada.core.config.orchestration import (
    DEFAULT_ORCHESTRATION_MODE,
    OrchestrationConfig,
    SUPPORTED_ORCHESTRATION_MODES,
    load_orchestration_config,
)
from mada.core.config.utils import expand_env_vars

__all__ = [
    "AgentConfig",
    "AppConfig",
    "BaseModelConfig",
    "BedrockModelConfig",
    "DatabaseConfig",
    "InterfaceConfig",
    "MCPServerConfig",
    "ModelConfig",
    "OpenAIModelConfig",
    "OrchestrationConfig",
    "PostgreSQLConfig",
    "DEFAULT_ORCHESTRATION_MODE",
    "SUPPORTED_ORCHESTRATION_MODES",
    "SQLiteConfig",
    "expand_env_vars",
    "load_config_from_json",
    "load_database_config",
    "load_model_config",
    "load_orchestration_config",
]
