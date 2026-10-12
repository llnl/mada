import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mada.core.background_tasks import BackgroundTaskManager

PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


@pytest.mark.asyncio
async def test_background_completion_persists_image_attachments():
    image = {
        "type": "image",
        "mimeType": "image/png",
        "data": base64.b64encode(PNG_DATA).decode("ascii"),
    }
    tool = SimpleNamespace(
        name="get_background_task_result",
        invoke=AsyncMock(
            return_value={
                "status": "completed",
                "result": {
                    "message": "Done",
                    "path": "/tmp/plot.png",
                    "count": 3,
                    "items": [{"type": "text", "text": "Rendered"}, image],
                },
            }
        ),
    )
    session_manager = MagicMock()
    manager = BackgroundTaskManager(
        session_manager,
        {"server": SimpleNamespace(_functions=[tool])},
        AsyncMock(),
    )

    await manager._poll_background_tool("task-1", "plot", "session-1")

    call = session_manager.chat_db.add_message.call_args
    assert call.args[:2] == ("session-1", "assistant")
    assert "Done" in call.args[2]
    assert "/tmp/plot.png" in call.args[2]
    assert '"count": 3' in call.args[2]
    assert image["data"] not in call.args[2]


@pytest.mark.asyncio
async def test_image_only_background_completion_persists_empty_text():
    image = {
        "type": "image",
        "mimeType": "image/png",
        "data": base64.b64encode(PNG_DATA).decode("ascii"),
    }
    tool = SimpleNamespace(
        name="get_background_task_result",
        invoke=AsyncMock(return_value={"status": "completed", "result": image}),
    )
    session_manager = MagicMock()
    manager = BackgroundTaskManager(
        session_manager,
        {"server": SimpleNamespace(_functions=[tool])},
        AsyncMock(),
    )

    await manager._poll_background_tool("task-2", "plot", "session-1")

    call = session_manager.chat_db.add_message.call_args
    assert call.args[2].endswith("completed:\n")
    assert call.args[2] != "null"
    assert len(call.kwargs["attachments"]) == 1
    assert call.kwargs["attachments"][0].data == PNG_DATA


@pytest.mark.asyncio
async def test_blocking_query_snapshots_session_and_context():
    session_manager = SimpleNamespace(
        current_session_id="primary",
        get_context_session_ids=lambda: ["reference"],
        session_exists=lambda session_id: True,
    )
    collect_message_response = AsyncMock(return_value="Done")
    manager = BackgroundTaskManager(session_manager, {}, collect_message_response)

    response = await manager.run_query("Hello", blocking=True)

    assert response == "Done"
    collect_message_response.assert_awaited_once_with(
        "Hello",
        persistence_session_id="primary",
        context_session_ids=["reference"],
    )
