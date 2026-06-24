# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Provider adapter abstractions.

This module defines the shared abstract base class used to implement
provider-specific chat client adapters.

A provider adapter is responsible for:

- validating that a model configuration object is compatible with the provider
- performing any provider-specific setup before client creation
- constructing a Microsoft Agent Framework chat client instance directly from
  the user-supplied model configuration
"""

from abc import ABC, abstractmethod
from typing import Type

from agent_framework import BaseChatClient

from mada.core.config import BaseModelConfig


class ProviderAdapter(ABC):
    """
    Abstract base class for provider-specific chat client adapters.

    Attributes:
        provider_name:
            Name of the provider handled by the adapter.
        chat_client:
            Chat client class used to instantiate provider-specific clients.

    Methods:
        validate_model_config:
            Validate a model configuration before client creation.
        pre_create:
            Perform provider-specific setup before client creation.
        create_client:
            Validate configuration and create a chat client instance.
    """

    provider_name: str
    chat_client: Type[BaseChatClient]

    @abstractmethod
    def validate_model_config(self, model_config: BaseModelConfig) -> None:
        """
        Validate a model configuration for this provider.

        Args:
            model_config:
                Model configuration to validate.

        Raises:
            ValueError:
                If the configuration is invalid for the provider.
        """
        pass

    def pre_create(self, model_config: BaseModelConfig) -> None:
        """
        Run provider-specific logic before constructing the chat client.

        This hook allows subclasses to inspect configuration or perform setup
        before the client instance is created.

        Args:
            model_config:
                Model configuration being used to create the client.
        """
        pass

    def create_client(self, model_config: BaseModelConfig) -> BaseChatClient:
        """
        Create a provider-specific chat client from a model configuration.

        This method validates the configuration, executes any provider-specific
        pre-creation hook, and instantiates the configured chat client using
        the exact user-supplied model name.

        Args:
            model_config:
                Model configuration used to build the client.

        Returns:
            An initialized provider-specific chat client.
        """
        self.validate_model_config(model_config)
        self.pre_create(model_config)

        kwargs = model_config.to_client_kwargs()
        return self.chat_client(**kwargs)
