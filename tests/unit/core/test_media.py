import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agent_framework import AgentResponseUpdate, Content
from mcp.types import ImageContent

from mada.core.media import ImageAttachment, RichResponse, extract_image_attachments
from mada.core.orchestration.stream_events import InternalImageSignal
from mada.core.orchestration.magentic_strategy import MagenticOrchestrationStrategy
from mada.core.orchestrator import MADAOrchestrator


PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


def test_extracts_image_from_agent_framework_function_result():
    image = Content.from_data(PNG_DATA, "image/png")
    result = Content.from_function_result("call-1", result=[image])
    update = AgentResponseUpdate(contents=[result])

    attachments = extract_image_attachments(update)

    assert len(attachments) == 1
    assert attachments[0].data == PNG_DATA
    assert attachments[0].media_type == "image/png"


def test_extracts_standard_mcp_image_content():
    image = ImageContent(
        data=base64.b64encode(PNG_DATA).decode("ascii"),
        mimeType="image/png",
    )

    assert extract_image_attachments(image)[0].data == PNG_DATA


def test_nested_duplicate_images_are_returned_once():
    image = Content.from_data(PNG_DATA, "image/png")

    attachments = extract_image_attachments(
        {"type": "tool_result", "data": {"items": [image, image]}}
    )

    assert len(attachments) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "image", "data": "not-base64", "mimeType": "image/png"},
        {
            "type": "image",
            "data": base64.b64encode(b"not-a-png").decode("ascii"),
            "mimeType": "image/png",
        },
        {
            "type": "image",
            "data": base64.b64encode(PNG_DATA).decode("ascii"),
            "mimeType": "image/svg+xml",
        },
    ],
)
def test_invalid_images_are_ignored(payload):
    assert extract_image_attachments(payload) == []


def test_oversized_images_are_ignored(monkeypatch):
    monkeypatch.setattr("mada.core.media.MAX_IMAGE_BYTES", 4)
    payload = {
        "type": "image",
        "data": base64.b64encode(PNG_DATA).decode("ascii"),
        "mimeType": "image/png",
    }

    assert extract_image_attachments(payload) == []


@pytest.mark.asyncio
async def test_collect_message_response_preserves_images_and_text(monkeypatch):
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png")
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)

    async def process_message(*args, **kwargs):
        yield InternalImageSignal(attachment)
        yield "description"
        yield InternalImageSignal(attachment)

    monkeypatch.setattr(orchestrator, "process_message", process_message)

    response = await orchestrator.collect_message_response("show me")

    assert isinstance(response, RichResponse)
    assert response == "description"
    assert response.images == (attachment,)


def test_task_local_capture_deduplicates_images():
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    from contextvars import ContextVar

    orchestrator._image_capture = ContextVar("test_image_capture", default=None)
    token, captured = orchestrator._begin_image_capture()
    image = Content.from_data(PNG_DATA, "image/png")
    update = SimpleNamespace(contents=[image, image])
    try:
        orchestrator._capture_image_attachments(update)
    finally:
        orchestrator._end_image_capture(token)

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_specialist_agent_tool_callback_captures_images():
    from contextvars import ContextVar

    specialist = MagicMock()
    specialist.name = "Renderer"
    specialist.as_tool.return_value = object()
    model_client = MagicMock()
    model_client.as_agent.return_value = object()

    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    orchestrator.specialist_agents = [specialist]
    orchestrator._agent_descriptions = {"Renderer": "Renders images"}
    orchestrator.a2a_agents = {}
    orchestrator.skill_tools = []
    orchestrator.skill_registry = MagicMock()
    orchestrator.skill_registry.has_skills.return_value = False
    orchestrator.model_client = model_client
    orchestrator._image_capture = ContextVar("callback_image_capture", default=None)

    orchestrator._create_planning_agent([], [])
    callback = specialist.as_tool.call_args.kwargs["stream_callback"]
    token, captured = orchestrator._begin_image_capture()
    try:
        await callback(
            AgentResponseUpdate(contents=[Content.from_data(PNG_DATA, "image/png")])
        )
    finally:
        orchestrator._end_image_capture(token)

    assert len(captured) == 1
    assert captured[0].data == PNG_DATA


@pytest.mark.asyncio
async def test_magentic_stream_surfaces_images_before_final_text(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    image = Content.from_data(PNG_DATA, "image/png")

    async def events(*args, **kwargs):
        yield {"type": "tool_result", "data": {"items": [image]}}
        yield {"type": "final", "text": "Rendered"}

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)

    output = [
        item
        async for item in strategy._stream_workflow_response(
            SimpleNamespace(), [], include_tool_notices=False
        )
    ]

    assert output[0][0] == "image"
    assert output[0][1].data == PNG_DATA
    assert output[-1] == ("final", "Rendered")
