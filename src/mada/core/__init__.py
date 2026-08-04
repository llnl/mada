# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Core orchestration functionality.

This module contains the fundamental components for orchestrating multi-agent workflows.
"""

from mada.core.chat_clients import (
    BedrockAdapter,
    LivAIAdapter,
    MADAChatClientFactory,
    OpenAIAdapter,
    ProviderAdapter,
    chat_client_factory,
)
from mada.core.config import (
    AgentConfig,
    AppConfig,
    BaseModelConfig,
    BedrockModelConfig,
    DatabaseConfig,
    InterfaceConfig,
    MCPServerConfig,
    ModelConfig,
    OpenAIModelConfig,
    OrchestrationConfig,
    PostgreSQLConfig,
    SQLiteConfig,
    DEFAULT_ORCHESTRATION_MODE,
    SUPPORTED_ORCHESTRATION_MODES,
    expand_env_vars,
    load_config_from_json,
    load_database_config,
    load_model_config,
    load_orchestration_config,
)
from mada.core.background_tasks import BackgroundTaskManager
from mada.core.coordinator import MCPAgentManager
from mada.core.orchestrator import MADAOrchestrator


__all__ = [
    "AgentConfig",
    "AppConfig",
    "BaseModelConfig",
    "BedrockAdapter",
    "BedrockModelConfig",
    "BackgroundTaskManager",
    "DatabaseConfig",
    "InterfaceConfig",
    "LivAIAdapter",
    "MADAChatClientFactory",
    "MADAOrchestrator",
    "MCPAgentManager",
    "MCPServerConfig",
    "ModelConfig",
    "OpenAIAdapter",
    "OpenAIModelConfig",
    "OrchestrationConfig",
    "PostgreSQLConfig",
    "ProviderAdapter",
    "SQLiteConfig",
    "DEFAULT_ORCHESTRATION_MODE",
    "SUPPORTED_ORCHESTRATION_MODES",
    "chat_client_factory",
    "expand_env_vars",
    "load_config_from_json",
    "load_database_config",
    "load_model_config",
    "load_orchestration_config",
]
