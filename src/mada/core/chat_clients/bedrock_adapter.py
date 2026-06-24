# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
AWS Bedrock provider adapter implementation.

This module defines [`BedrockAdapter`][core.chat_clients.bedrock_adapter.BedrockAdapter],
a [`ProviderAdapter`][core.chat_clients.provider_adapter.ProviderAdapter] implementation
for AWS Bedrock chat models. It validates that incoming model configuration objects are
instances of [`BedrockModelConfig`][core.config.models.BedrockModelConfig] and prepares
AWS authentication environment variables before chat client creation.
"""

import os

from agent_framework.amazon import BedrockChatClient

from mada.core.config import BaseModelConfig, BedrockModelConfig
from mada.core.chat_clients.provider_adapter import ProviderAdapter


class BedrockAdapter(ProviderAdapter):
    """
    Provider adapter for AWS Bedrock chat models.

    This adapter validates Bedrock-specific model configuration objects and sets
    AWS credential-related environment variables before constructing a chat
    client.

    Attributes:
        provider_name:
            Name of the provider handled by this adapter.
        chat_client:
            Chat client class used to create Bedrock chat clients.

    Methods:
        validate_model_config:
            Validate that `model_config` is a `BedrockModelConfig` instance.
        _set_env_if_allowed:
            Set an environment variable if permitted by the current
            configuration.
        pre_create:
            Apply environment-based credential configuration before client
            creation.
    """

    provider_name = "aws-bedrock"
    chat_client = BedrockChatClient

    def validate_model_config(self, model_config: BaseModelConfig) -> None:
        """
        Validate that `model_config` is compatible with the Bedrock adapter.

        Args:
            model_config:
                Model configuration to validate.

        Raises:
            TypeError:
                Raised if `model_config` is not a `BedrockModelConfig`
                instance.
        """
        if not isinstance(model_config, BedrockModelConfig):
            raise TypeError(
                f"{self.provider_name} adapter requires BedrockModelConfig, "
                f"got {type(model_config).__name__}"
            )

    def _set_env_if_allowed(
        self, name: str, value: str, override: bool = False
    ) -> None:
        """
        Set an environment variable if a value is provided and overwriting is
        allowed.

        Args:
            name:
                Name of the environment variable to set.
            value:
                Value to assign to `name`.
            override:
                Whether an existing conflicting environment variable may be
                overwritten.

        Raises:
            RuntimeError:
                Raised if `name` is already set to a different value and
                `override` is `False`.
        """
        if not value:
            return

        existing = os.environ.get(name)
        if existing and existing != value and not override:
            raise RuntimeError(
                f"Refusing to overwrite existing environment variable '{name}'. "
                f"Set override_env_credentials=True to allow this."
            )

        os.environ[name] = value

    def pre_create(self, model_config: BedrockModelConfig) -> None:
        """
        Prepare AWS credential environment variables before client creation.

        Args:
            model_config:
                Bedrock model configuration containing credential values and
                override settings.

        Raises:
            RuntimeError:
                Raised if an existing environment variable would be overwritten
                without permission.
        """
        override = getattr(model_config, "override_env_credentials", False)

        self._set_env_if_allowed(
            "AWS_BEARER_TOKEN_BEDROCK", model_config.bearer_token, override
        )
        self._set_env_if_allowed("AWS_PROFILE", model_config.aws_profile, override)
        self._set_env_if_allowed(
            "AWS_ACCESS_KEY_ID", model_config.aws_access_key_id, override
        )
        self._set_env_if_allowed(
            "AWS_SECRET_ACCESS_KEY", model_config.aws_secret_access_key, override
        )
        self._set_env_if_allowed(
            "AWS_SESSION_TOKEN", model_config.aws_session_token, override
        )
