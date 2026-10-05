from types import SimpleNamespace

import pytest

from mada.core.orchestration.agent_as_tool_strategy import (
    AgentAsToolOrchestrationStrategy,
)


class _FakeContent:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _FakeChunk:
    def __init__(self, text, contents):
        self.text = text
        self.contents = contents


@pytest.mark.asyncio
async def test_stream_response_emits_tool_notice_when_chunk_also_has_text():
    strategy = AgentAsToolOrchestrationStrategy()
    tool_chunk = _FakeChunk(
        "partial text",
        [
            _FakeContent(
                {
                    "type": "response_update",
                    "contents": [{"type": "function_call", "name": "slow_tool"}],
                }
            )
        ],
    )

    async def _stream():
        yield tool_chunk

    orchestrator = SimpleNamespace(
        planning_agent=SimpleNamespace(run=lambda prompt, session, stream: _stream())
    )

    outputs = [
        chunk
        async for chunk in strategy._stream_response(
            orchestrator,
            "hello",
            session=object(),
            include_tool_notices=True,
        )
    ]

    assert outputs[0] == ""
    assert outputs[1] == "\n[Calling: slow_tool]\n"
    assert "partial text" in outputs
