# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base interface for orchestrator initialization strategies.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple

from mada.core.config import AgentConfig, MCPServerConfig

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator


class BaseOrchestrationStrategy(ABC):
    """
    Internal strategy boundary for orchestrator initialization patterns.

    Concrete strategies encapsulate the setup flow for a specific orchestration
    mode, including which agents participate, how tools are connected, and how
    the orchestrator session is initialized.
    """

    mode: str = ""

    @abstractmethod
    async def initialize(
        self,
        orchestrator: "MADAOrchestrator",
        agent_configs: List[AgentConfig],
        mcp_servers: Dict[str, MCPServerConfig] | None = None,
    ) -> Tuple[str, List[str]]:
        """
        Initialize the orchestrator for the strategy's orchestration mode.

        Args:
            orchestrator: Orchestrator instance being configured.
            agent_configs: Agent definitions available to the strategy.
            mcp_servers: Named MCP server definitions available to the strategy.

        Returns:
            A user-facing status message and a flat list of connected tool names.
        """
        pass
