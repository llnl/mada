# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base interface for orchestrator initialization strategies.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Tuple

from mada.core.config import AgentConfig, MCPServerConfig

if TYPE_CHECKING:
    from mada.core.orchestrator import MADAOrchestrator


class BaseOrchestrationStrategy(ABC):
    """
    Internal strategy boundary for orchestration modes.

    Concrete strategies encapsulate the setup and request-processing flow for a
    specific orchestration mode, including which agents participate, how tools
    are connected, which coordinator agent is created, and how requests stream
    responses.
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

    @abstractmethod
    async def process_openai_messages(
        self,
        orchestrator: "MADAOrchestrator",
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Process OpenAI-style chat messages for this orchestration mode.
        """
        pass

    @abstractmethod
    async def process_message(
        self,
        orchestrator: "MADAOrchestrator",
        message: str,
        isolated_session: bool = False,
        record_to_db: bool = True,
        background_poll_session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process one interactive user message for this orchestration mode.
        """
        pass
