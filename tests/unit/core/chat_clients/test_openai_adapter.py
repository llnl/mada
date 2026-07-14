# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mada.core.chat_clients.openai_adapter import OpenAIAdapter
from mada.core.config import OpenAIModelConfig


@pytest.mark.unit
def test_openai_adapter_injects_async_client(monkeypatch):
    captured = {}

    class DummyHttpClient:
        def __init__(self, *, verify):
            captured["verify"] = verify

    class DummyAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

    monkeypatch.setattr(
        "mada.core.chat_clients.openai_adapter.resolve_httpx_verify_value",
        lambda: "/tmp/test-ca.pem",
    )
    monkeypatch.setattr(
        "mada.core.chat_clients.openai_adapter.DefaultAsyncHttpxClient",
        DummyHttpClient,
    )
    monkeypatch.setattr(
        "mada.core.chat_clients.openai_adapter.AsyncOpenAI",
        DummyAsyncOpenAI,
    )

    model_config = OpenAIModelConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        extra={
            "org_id": "test-org",
            "default_headers": {"X-Test": "1"},
        },
    )

    OpenAIAdapter().pre_create(model_config)

    assert isinstance(model_config.extra["async_client"], DummyAsyncOpenAI)
    assert captured["verify"] == "/tmp/test-ca.pem"
    assert captured["client_kwargs"]["api_key"] == "sk-test"
    assert captured["client_kwargs"]["base_url"] == "https://example.invalid/v1"
    assert captured["client_kwargs"]["organization"] == "test-org"
    assert captured["client_kwargs"]["default_headers"] == {"X-Test": "1"}


@pytest.mark.unit
def test_openai_adapter_preserves_explicit_async_client():
    existing_client = object()
    model_config = OpenAIModelConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        extra={"async_client": existing_client},
    )

    OpenAIAdapter().pre_create(model_config)

    assert model_config.extra["async_client"] is existing_client
