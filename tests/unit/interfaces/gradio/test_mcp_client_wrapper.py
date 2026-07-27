# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unit tests for the Gradio MCP client wrapper.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mada.interfaces.gradio.mcp_client_wrapper import MCPGradioClientSession


class FakeSessionManager:
    def __init__(self, current_session_id: str = "origin"):
        self.current_session_id = current_session_id
        self.messages = []

    def create_session_id(self) -> str:
        return "created"

    def create_new_session(self, session_id: str):
        self.current_session_id = session_id

    def select_session(self, session_id: str):
        self.current_session_id = session_id
        return []

    def add_message_to_session(self, session_id: str, role: str, message: str):
        self.messages.append((session_id, role, message))


def make_client(orchestrator: MagicMock) -> MCPGradioClientSession:
    client = MCPGradioClientSession.__new__(MCPGradioClientSession)
    client.initialized = True
    client.orchestrator = orchestrator
    client.blocking = False
    client.session_manager = FakeSessionManager()
    client._pending_clarifications = {}
    client._autonomy_cancel_events = {}
    client._active_response_sessions = set()
    return client


def async_chunks(*chunks: str):
    async def _generator():
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    return _generator()


@pytest.mark.asyncio
async def test_autonomy_reply_persists_to_originating_session_after_switch():
    orchestrator = MagicMock()
    orchestrator.process_message.return_value = async_chunks("assistant reply")
    orchestrator.run_control_prompt = AsyncMock(
        return_value=(
            "AUTONOMY_DECISION=STOP\n"
            "AUTONOMY_QUERY=\n"
            "AUTONOMY_WAIT_SECONDS=\n"
            "AUTONOMY_QUESTION=\n"
        )
    )
    client = make_client(orchestrator)

    stream = client.process_message(
        "hello",
        history=[],
        agent_table=None,
        autonomy_level=1,
    )
    first_chunk = await stream.__anext__()
    client.session_manager.current_session_id = "other"
    remaining_chunks = [chunk async for chunk in stream]

    assert first_chunk == "assistant reply"
    assert remaining_chunks == []
    assert client.session_manager.messages == [
        ("origin", "user", "hello"),
        ("origin", "assistant", "assistant reply"),
    ]
    assert (
        orchestrator.process_message.call_args.kwargs["background_poll_session_id"]
        == "origin"
    )


@pytest.mark.asyncio
async def test_immediate_autonomy_error_persists_complete_turn():
    async def failing_stream():
        raise RuntimeError("network down")
        yield ""

    orchestrator = MagicMock()
    orchestrator.process_message.return_value = failing_stream()
    client = make_client(orchestrator)

    chunks = [
        chunk
        async for chunk in client.process_message(
            "hello",
            history=[],
            agent_table=None,
            autonomy_level=1,
        )
    ]

    assert chunks == ["Error processing message: network down"]
    assert client.session_manager.messages == [
        ("origin", "user", "hello"),
        ("origin", "assistant", "Error processing message: network down"),
    ]
    assert (
        orchestrator.process_message.call_args.kwargs["background_poll_session_id"]
        == "origin"
    )
