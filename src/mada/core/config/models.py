# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Model configuration classes and loading utilities.

This module defines configuration dataclasses used to describe provider-specific
model settings for chat client creation. It includes base configuration
behavior, OpenAI-compatible configuration, AWS Bedrock configuration, and a
helper function for loading the appropriate configuration type from a
dictionary.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from mada.core.config.utils import expand_env_vars


LOG = logging.getLogger("mada-interface")


@dataclass(kw_only=True)
class BaseModelConfig:
    """
    Base configuration for provider model settings.

    This class stores common configuration fields shared across provider
    implementations and defines the interface for validation and conversion
    into chat client keyword arguments.

    Attributes:
        provider:
            Name of the provider associated with the model configuration.
        model:
            Requested model name.
        extra:
            Additional provider-specific keyword arguments to pass to the chat
            client.

    Methods:
        __post_init__:
            Expand environment variables in common configuration fields.
        validate:
            Validate the configuration.
        to_client_kwargs:
            Convert the configuration into keyword arguments for chat client
            construction.
    """

    provider: str
    model: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Expand environment variables in shared configuration fields.
        """
        self.provider = expand_env_vars(self.provider)
        self.model = expand_env_vars(self.model)

    def validate(self) -> None:
        """
        Validate the configuration.

        Subclasses should override this method to implement provider-specific
        validation rules.
        """
        pass

    def to_client_kwargs(self) -> Dict[str, Any]:
        """
        Convert the configuration into chat client keyword arguments.

        Returns:
            A dictionary of keyword arguments for chat client construction.

        Raises:
            NotImplementedError:
                Raised if the subclass does not implement this method.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class OpenAIModelConfig(BaseModelConfig):
    """
    Configuration for OpenAI-compatible chat clients.

    This configuration supports literal API keys, environment-variable-expanded
    values, or file paths containing API keys, along with the base URL for the
    target service.

    Attributes:
        provider:
            Name of the provider associated with the model configuration.
        model:
            Requested model name.
        extra:
            Additional provider-specific keyword arguments to pass to the chat
            client.
        api_key:
            API key value, environment-expanded string, or path to a file
            containing the API key.
        base_url:
            Base URL for the OpenAI-compatible endpoint.

    Methods:
        __post_init__:
            Expand environment variables, resolve file-based API keys, and
            validate the configuration.
        validate:
            Validate that required OpenAI-compatible settings are present.
        to_client_kwargs:
            Convert the configuration into keyword arguments for an
            OpenAI-compatible chat client.
    """

    api_key: str  # Can be literal string, env var, or path to file containing API key
    base_url: str

    def __post_init__(self):
        """
        Expand environment variables, resolve file-based API keys, and validate
        the configuration.
        """
        super().__post_init__()
        self.api_key = expand_env_vars(self.api_key)
        self.base_url = expand_env_vars(self.base_url)

        if os.path.exists(self.api_key):
            with open(self.api_key, "r") as api_key_file:
                self.api_key = api_key_file.read().strip()

        self.validate()

    def validate(self) -> None:
        """
        Validate required OpenAI-compatible configuration values.

        Raises:
            RuntimeError:
                Raised if `api_key` or `base_url` is missing.
        """
        if not self.api_key:
            raise RuntimeError(
                "API key not found for OpenAI-compatible model config. "
                "Please specify 'api_key'."
            )
        if not self.base_url:
            raise RuntimeError(
                "Base URL not found for OpenAI-compatible model config. "
                "Please specify 'base_url'."
            )

    def to_client_kwargs(self) -> Dict[str, Any]:
        """
        Convert the configuration into OpenAI-compatible chat client keyword
        arguments.

        Returns:
            A dictionary of keyword arguments for OpenAI-compatible chat client
            construction.
        """
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            **self.extra,
        }


@dataclass(kw_only=True)
class BedrockModelConfig(BaseModelConfig):
    """
    Configuration for AWS Bedrock chat clients.

    This configuration supports multiple authentication styles, including
    bearer-token authentication, AWS profile-based authentication, and direct
    AWS credential values.

    Attributes:
        provider:
            Name of the provider associated with the model configuration.
        model:
            Requested model name.
        extra:
            Additional provider-specific keyword arguments to pass to the chat
            client.
        region:
            AWS region for the Bedrock service.
        bearer_token:
            Optional bearer token value, or path to a file containing the
            token.
        aws_profile:
            Optional AWS profile name.
        aws_access_key_id:
            Optional AWS access key ID.
        aws_secret_access_key:
            Optional AWS secret access key.
        aws_session_token:
            Optional AWS session token.
        override_env_credentials:
            Whether adapter setup may overwrite existing AWS credential-related
            environment variables.

    Methods:
        __post_init__:
            Expand environment variables, resolve file-based bearer tokens, and
            validate the configuration.
        validate:
            Validate required Bedrock region and authentication settings.
        to_client_kwargs:
            Convert the configuration into keyword arguments for a Bedrock chat
            client.
    """

    region: str

    # Pick whichever auth style your runtime actually uses
    bearer_token: Optional[str] = None  # can use API key from Bedrock
    aws_profile: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    # Choose whether we're allowed to override AWS_* env variables from these settings
    override_env_credentials: bool = False

    def __post_init__(self):
        """
        Expand environment variables, resolve file-based bearer tokens, and
        validate the configuration.
        """
        super().__post_init__()
        self.region = expand_env_vars(self.region)
        self.bearer_token = expand_env_vars(self.bearer_token)
        self.aws_profile = expand_env_vars(self.aws_profile)
        self.aws_access_key_id = expand_env_vars(self.aws_access_key_id)
        self.aws_secret_access_key = expand_env_vars(self.aws_secret_access_key)
        self.aws_session_token = expand_env_vars(self.aws_session_token)

        if self.bearer_token and os.path.exists(self.bearer_token):
            with open(self.bearer_token, "r") as token_file:
                self.bearer_token = token_file.read().strip()

        self.validate()

    def validate(self) -> None:
        """
        Validate required Bedrock configuration values and authentication
        settings.

        Raises:
            RuntimeError:
                Raised if `region` is missing, if no supported authentication
                method is provided, or if more than one authentication method
                is provided.
        """
        if not self.region:
            raise RuntimeError("BedrockModelConfig requires 'region'")

        if self.aws_access_key_id and not self.aws_secret_access_key:
            raise RuntimeError(
                "BedrockModelConfig received 'aws_access_key_id' without "
                "'aws_secret_access_key'"
            )

        if self.aws_secret_access_key and not self.aws_access_key_id:
            raise RuntimeError(
                "BedrockModelConfig received 'aws_secret_access_key' without "
                "'aws_access_key_id'"
            )

        has_bearer = bool(self.bearer_token)
        has_profile = bool(self.aws_profile)
        has_full_aws_creds = bool(self.aws_access_key_id and self.aws_secret_access_key)

        if self.aws_session_token and not has_full_aws_creds:
            raise RuntimeError(
                "BedrockModelConfig received 'aws_session_token' without "
                "both 'aws_access_key_id' and 'aws_secret_access_key'"
            )

        auth_method_count = sum(
            [
                has_bearer,
                has_profile,
                has_full_aws_creds,
            ]
        )

        if auth_method_count == 0:
            raise RuntimeError(
                "BedrockModelConfig requires exactly one authentication method: "
                "'bearer_token', 'aws_profile', or both "
                "'aws_access_key_id' and 'aws_secret_access_key'"
            )

        if auth_method_count > 1:
            raise RuntimeError(
                "BedrockModelConfig received multiple authentication methods. "
                "Specify exactly one of: 'bearer_token', 'aws_profile', or "
                "both 'aws_access_key_id' and 'aws_secret_access_key'"
            )

    def to_client_kwargs(self) -> Dict[str, Any]:
        """
        Convert the configuration into Bedrock chat client keyword arguments.

        Returns:
            A dictionary of keyword arguments for Bedrock chat client
            construction.
        """
        return {
            "model": self.model,
            "region": self.region,
            **self.extra,
        }


ModelConfig = Union[OpenAIModelConfig, BedrockModelConfig]


def load_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
    """
    Load a provider-specific model configuration object from a dictionary.

    Args:
        config_dict:
            Dictionary containing serialized model configuration values.

    Returns:
        An initialized provider-specific model configuration instance.

    Raises:
        ValueError:
            Raised if `provider` is missing or unsupported.
    """
    provider = config_dict.get("provider")
    if not provider:
        raise ValueError("Model config must include 'provider'")

    if provider in {"openai", "livai"}:
        return OpenAIModelConfig(**config_dict)

    if provider == "aws-bedrock":
        return BedrockModelConfig(**config_dict)

    raise ValueError(f"Unsupported model provider: {provider}")
