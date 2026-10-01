# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LivAI provider adapter implementation.

This module defines [`LivAIAdapter`][core.chat_clients.livai_adapter.LivAIAdapter],
an [`OpenAIAdapter`][core.chat_clients.openai_adapter.OpenAIAdapter] specialization
for LivAI-hosted models.
"""

from agent_framework.openai import OpenAIChatCompletionClient

from mada.core.chat_clients.openai_adapter import OpenAIAdapter


class LivAIAdapter(OpenAIAdapter):
    """
    Provider adapter for LivAI-hosted chat models.

    LivAI's LiteLLM proxy fronts Azure OpenAI, which does not yet expose the
    newer Responses API (requires api-version >= 2025-03-01-preview). So we
    use OpenAIChatCompletionClient (hits /v1/chat/completions) instead of
    OpenAIChatClient (hits /v1/responses).

    Attributes:
        provider_name:
            Name of the provider handled by this adapter.
        chat_client:
            Chat client class used to create LivAI chat clients.
    """

    provider_name = "livai"
    chat_client = OpenAIChatCompletionClient
