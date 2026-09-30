import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import gradio as gr
import pytest

from mada.core.media import ImageAttachment, RichResponse
from mada.interfaces.gradio.mcp_client_wrapper import MCPGradioClientSession


PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


def _client() -> MCPGradioClientSession:
    client = MCPGradioClientSession.__new__(MCPGradioClientSession)
    client._attachment_cache = tempfile.TemporaryDirectory(prefix="mada-gradio-test-")
    return client


def test_gradio_chatbot_accepts_mixed_text_and_image_content():
    client = _client()
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png")
    try:
        content = client._format_message_content("Rendered image", [attachment])
        result = gr.Chatbot().postprocess([{"role": "assistant", "content": content}])
    finally:
        client._attachment_cache.cleanup()

    assert result.root[0].content[0].type == "text"
    assert result.root[0].content[1].type == "file"
    assert result.root[0].content[1].file.mime_type == "image/png"


def test_persisted_attachment_is_formatted_for_gradio():
    client = _client()
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png")
    try:
        history = client._format_history_for_gradio(
            [
                {
                    "role": "assistant",
                    "content": "Done",
                    "attachments": [attachment],
                }
            ]
        )
    finally:
        client._attachment_cache.cleanup()

    assert history[0]["content"][0] == "Done"
    assert history[0]["content"][1]["file"]["mime_type"] == "image/png"


def test_persisted_image_placeholder_is_hidden_from_gradio():
    client = _client()
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png")
    try:
        history = client._format_history_for_gradio(
            [
                {
                    "role": "assistant",
                    "content": f"[Image attachment: {attachment.filename}]",
                    "attachments": [attachment],
                }
            ]
        )
    finally:
        client._attachment_cache.cleanup()

    assert len(history[0]["content"]) == 1
    assert history[0]["content"][0]["file"]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_gradio_client_yields_rich_response_as_mixed_content():
    client = _client()
    attachment = ImageAttachment.from_data(PNG_DATA, "image/png")

    class BackgroundTasks:
        async def run_query(self, message, blocking):
            return RichResponse("Done", [attachment])

    client.initialized = True
    client.blocking = True
    client.orchestrator = SimpleNamespace(background_tasks=BackgroundTasks())
    client.session_manager = MagicMock()
    try:
        responses = [
            response
            async for response in client.process_message("render", [], MagicMock())
        ]
    finally:
        client._attachment_cache.cleanup()

    assert responses[0][0] == "Done"
    assert responses[0][1]["type"] == "file"
