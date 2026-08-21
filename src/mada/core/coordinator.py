# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MCP agent management utilities for the MADA multi-agent system.

This module defines the `MCPAgentManager` class, a reusable base for managing
agents that interact with MCP (Model Context Protocol) tool servers.

Responsibilities of this module include:
- Creating and configuring chat agents with model and instruction settings
- Instantiating an OpenAI-compatible model client for use by agents

Typical usage:
    Subclass `MCPAgentManager` to implement specific session logic (e.g., console or web UI),
    then call `create_chat_agent()` with an `AgentConfig` to instantiate tool-using agents.
"""

from typing import Any, Dict, List, Optional

from agent_framework import Agent, BaseChatClient

from mada.core.config import AgentConfig, ModelConfig
from mada.core.chat_clients import chat_client_factory


class MCPAgentManager:
    """
    Base class for managing agent creation with MCP (Model Context Protocol) tools.

    This class provides shared functionality for creating agents and managing
    model client configuration. It can be extended by different implementations
    for console or web UI integrations.

    Attributes:
        model_config (ModelConfig): Configuration for the model client.
        timeout (int): Timeout in seconds for server operations.
        model_client (BaseChatClient): Client for generating chat responses.

    Methods:
        create_chat_agent: Given an agent configuration, create an Agent.
    """

    def __init__(
        self, model_config: Optional[ModelConfig] = None, timeout: int = 86400
    ):
        """
        Initialize the MCP agent manager.

        Args:
            model_config: Configuration for the model client. If None, uses environment variables.
            timeout: Timeout in seconds for server operations.
        """
        self.model_config = model_config or ModelConfig()
        self.timeout = timeout
        self.model_client = self._setup_model_client()

    def _setup_model_client(self) -> BaseChatClient:
        """
        Instantiate and return the chat client.

        Returns:
            A configured model client ready to be used by agents.
        """
        return chat_client_factory.create(self.model_config)

    async def create_chat_agent(
        self,
        agent_config: AgentConfig,
        tools: List[Any] = None,
        **kwargs: Dict[str, Any],
    ) -> Agent:
        """
        Create an Agent from configuration.

        Args:
            agent_config: Configuration for the agent.
            tools: List of tools to provide to the agent.
            **kwargs: Additional arguments to pass to Agent constructor.

        Returns:
            The created Agent.
        """
        agent_kwargs = dict(agent_config.extra or {})
        agent_kwargs.update(kwargs)

        return self.model_client.as_agent(
            name=agent_config.agent_name,
            instructions=agent_config.instructions,
            tools=tools or [],
            **agent_kwargs,
        )
