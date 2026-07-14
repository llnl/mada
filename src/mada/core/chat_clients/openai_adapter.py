# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
OpenAI provider adapter implementation.

This module defines [`OpenAIAdapter`][core.chat_clients.openai_adapter.OpenAIAdapter],
a [`ProviderAdapter`][core.chat_clients.provider_adapter.ProviderAdapter] implementation
for OpenAI chat models. It validates that incoming model configuration objects are
instances of [`OpenAIModelConfig`][core.config.models.OpenAIModelConfig] before client
creation.
"""

from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from mada.core.config import BaseModelConfig, OpenAIModelConfig
from mada.core.chat_clients.provider_adapter import ProviderAdapter
from mada.core.tls import resolve_httpx_verify_value


class OpenAIAdapter(ProviderAdapter):
    """
    Provider adapter for OpenAI chat models.

    This adapter ensures that client creation uses `OpenAIModelConfig`
    instances.

    Attributes:
        provider_name:
            Name of the provider handled by this adapter.
        chat_client:
            Chat client class used to create OpenAI chat clients.

    Methods:
        validate_model_config:
            Validate that `model_config` is an `OpenAIModelConfig` instance.
    """

    provider_name = "openai"
    chat_client = OpenAIChatClient

    def pre_create(self, model_config: BaseModelConfig) -> None:
        """
        Create an OpenAI SDK client that uses MADA's shared TLS configuration.

        OpenAI's default client relies on ``certifi``, which can reject
        institution-managed certificate chains on HPC systems. MADA already uses
        the system trust store for MCP HTTP clients, so mirror that behavior for
        OpenAI-compatible providers as well.
        """
        self.validate_model_config(model_config)

        if model_config.extra.get("async_client") is not None:
            return

        client_kwargs = {
            "api_key": model_config.api_key,
            "base_url": model_config.base_url,
            "http_client": DefaultAsyncHttpxClient(
                verify=resolve_httpx_verify_value(verify=model_config.verify),
            ),
        }

        if org_id := model_config.extra.get("org_id"):
            client_kwargs["organization"] = org_id

        if default_headers := model_config.extra.get("default_headers"):
            client_kwargs["default_headers"] = default_headers

        model_config.extra["async_client"] = AsyncOpenAI(**client_kwargs)

    def validate_model_config(self, model_config: BaseModelConfig) -> None:
        """
        Validate that `model_config` is compatible with the OpenAI adapter.

        Args:
            model_config:
                Model configuration to validate.

        Raises:
            TypeError:
                Raised if `model_config` is not an `OpenAIModelConfig`
                instance.
        """
        if not isinstance(model_config, OpenAIModelConfig):
            raise TypeError(
                f"{self.provider_name} adapter requires OpenAIModelConfig, "
                f"got {type(model_config).__name__}"
            )
