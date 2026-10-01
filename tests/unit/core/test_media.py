import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agent_framework import AgentResponseUpdate, Content
from mcp.types import ImageContent

from mada.core.media import (
    ImageAttachment,
    RichResponse,
    extract_image_attachments,
    strip_image_payload_text,
)
from mada.core.orchestration.agent_as_tool_strategy import (
    AgentAsToolOrchestrationStrategy,
)
from mada.core.orchestration.stream_events import (
    InternalImageSignal,
    InternalResponseReplacement,
)
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


def test_image_attachment_filename_is_safe_for_transcript_markers():
    attachment = ImageAttachment.from_data(
        PNG_DATA,
        "image/png",
        filename="plot]\n[synthetic].png",
    )

    assert attachment.filename.isprintable()
    assert "\n" not in attachment.filename
    assert "]" not in attachment.filename


def test_magentic_does_not_expose_image_data_as_text():
    strategy = MagenticOrchestrationStrategy()
    payload = {
        "type": "image",
        "data": base64.b64encode(PNG_DATA).decode("ascii"),
        "mimeType": "image/png",
    }

    assert strategy._extract_text(payload) == ""


def test_extracts_and_strips_markdown_data_uri():
    encoded = base64.b64encode(PNG_DATA).decode("ascii")
    text = f"Here is the plot.\n![plot](data:image/png;base64,{encoded})"

    attachments = extract_image_attachments(text)

    assert len(attachments) == 1
    assert strip_image_payload_text(text) == "Here is the plot."


def test_strips_invalid_serialized_image_payload_without_displaying_base64():
    text = (
        'Here is the image. {"type":"image","mimeType":"image/png",'
        '"data":"not-valid-base64"}'
    )

    assert strip_image_payload_text(text) == "Here is the image."


@pytest.mark.asyncio
async def test_collect_message_response_converts_serialized_image_text(monkeypatch):
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)
    encoded = base64.b64encode(PNG_DATA).decode("ascii")

    async def process_message(*args, **kwargs):
        yield f"Here is the plot. ![plot](data:image/png;base64,{encoded})"

    monkeypatch.setattr(orchestrator, "process_message", process_message)

    response = await orchestrator.collect_message_response("show me")

    assert isinstance(response, RichResponse)
    assert response == "Here is the plot."
    assert len(response.images) == 1


@pytest.mark.asyncio
async def test_agent_as_tool_reassembles_split_serialized_image_text():
    encoded = base64.b64encode(PNG_DATA).decode("ascii")
    chunks = [
        SimpleNamespace(text="Here ![plot](data:image/png;base64,"),
        SimpleNamespace(text=f"{encoded})"),
    ]

    class PlanningAgent:
        async def stream(self):
            for chunk in chunks:
                yield chunk

        def run(self, prompt, session, stream):
            return self.stream()

    strategy = AgentAsToolOrchestrationStrategy()
    output = [
        item
        async for item in strategy._stream_response(
            SimpleNamespace(planning_agent=PlanningAgent()),
            "show me",
            session=None,
            include_tool_notices=False,
        )
    ]

    assert len(output) == 2
    assert str(output[0]).startswith("[Image attachment: ")
    assert output[1] == "Here"


@pytest.mark.asyncio
async def test_magentic_openai_stream_forwards_image_signal(monkeypatch):
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png")
    strategy = MagenticOrchestrationStrategy()

    async def workflow(*args, **kwargs):
        yield "image", attachment
        yield "final", ""

    monkeypatch.setattr(strategy, "_stream_workflow_response", workflow)
    orchestrator = SimpleNamespace(
        manager_agent=object(),
        _normalize_transcript_messages=lambda messages: messages,
    )

    output = [item async for item in strategy.process_openai_messages(orchestrator, [])]

    assert len(output) == 1
    assert str(output[0]).startswith("[Image attachment: ")


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


@pytest.mark.asyncio
async def test_collect_message_response_uses_text_fallback_for_image_only(
    monkeypatch,
):
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png", filename="plot.png")
    orchestrator = MADAOrchestrator.__new__(MADAOrchestrator)

    async def process_message(*args, **kwargs):
        yield InternalImageSignal(attachment)

    monkeypatch.setattr(orchestrator, "process_message", process_message)

    response = await orchestrator.collect_message_response("show me")

    assert response == "[Image attachment: plot.png]"
    assert response.images == (attachment,)


def test_internal_image_signal_has_text_fallback():
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png", filename="plot.png")

    signal = InternalImageSignal(attachment)

    assert str(signal) == "[Image attachment: plot.png]"


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


@pytest.mark.asyncio
async def test_magentic_stream_replaces_cumulative_image_payload_text(monkeypatch):
    strategy = MagenticOrchestrationStrategy()
    encoded = base64.b64encode(PNG_DATA).decode("ascii")
    partial = "Result: data:image/png;base64,"
    complete = f"{partial}{encoded}"

    async def events(*args, **kwargs):
        yield {"type": "output", "text": partial}
        yield {"type": "output", "text": complete}

    monkeypatch.setattr(strategy, "_iter_workflow_events", events)

    output = [
        item
        async for item in strategy._stream_workflow_response(
            SimpleNamespace(), [], include_tool_notices=False
        )
    ]

    assert output[0] == ("chunk", partial)
    assert isinstance(output[1][1], InternalResponseReplacement)
    assert output[2] == ("final", "Result:")
