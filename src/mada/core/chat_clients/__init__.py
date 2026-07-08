# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Chat client adapter package.

This package provides provider-specific chat client adapters and factory
utilities for creating chat clients from model configuration objects. It
exposes adapter implementations for supported providers, along with the shared
[`ProviderAdapter`][core.chat_clients.provider_adapter.ProviderAdapter] base
class and the
[`MADAChatClientFactory`][core.chat_clients.chat_client_factory.MADAChatClientFactory]
used to create chat clients.

Modules:
    bedrock_adapter:
        Provides the [`BedrockAdapter`][core.chat_clients.bedrock_adapter.BedrockAdapter]
        implementation for AWS Bedrock models.
    chat_client_factory:
        Provides [`MADAChatClientFactory`][core.chat_clients.chat_client_factory.MADAChatClientFactory]
        and the shared `chat_client_factory` instance for constructing chat clients.
    livai_adapter:
        Provides the [`LivAIAdapter`][core.chat_clients.livai_adapter.LivAIAdapter]
        implementation for LivAI-hosted models.
    openai_adapter:
        Provides the [`OpenAIAdapter`][core.chat_clients.openai_adapter.OpenAIAdapter]
        implementation for OpenAI models.
    provider_adapter:
        Provides the shared [`ProviderAdapter`][core.chat_clients.provider_adapter.ProviderAdapter]
        base class and related model metadata utilities.
"""

from mada.core.chat_clients.bedrock_adapter import BedrockAdapter
from mada.core.chat_clients.chat_client_factory import (
    MADAChatClientFactory,
    chat_client_factory,
)
from mada.core.chat_clients.livai_adapter import LivAIAdapter
from mada.core.chat_clients.openai_adapter import OpenAIAdapter
from mada.core.chat_clients.provider_adapter import ProviderAdapter


__all__ = [
    "BedrockAdapter",
    "LivAIAdapter",
    "MADAChatClientFactory",
    "OpenAIAdapter",
    "ProviderAdapter",
    "chat_client_factory",
]
