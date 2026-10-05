from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import gradio as gr
import pytest

from mada.core.config import MCPServerConfig, OpenAIModelConfig, OrchestrationConfig, SQLiteConfig
from mada.core.media import ImageAttachment
from mada.interfaces.gradio.mcp_client_wrapper import MCPGradioClientSession


def _create_client(tmp_path: Path) -> MCPGradioClientSession:
    database_config = SQLiteConfig(path=tmp_path / "test_histories.db")
    return MCPGradioClientSession(
        model_config=OpenAIModelConfig(
            provider="openai",
            model="dummy-model",
            api_key="api-key",
            base_url="base-url",
        ),
        agents=["a1", "a2"],
        database_config=database_config,
        mcp_servers={"s1": MCPServerConfig(transport="stdio")},
        a2a_agents={},
        orchestration_config=OrchestrationConfig(),
    )


class TestMCPGradioClientSession:
    def test_history_is_prefix_detects_optimistic_chat_state(self):
        """Timer refresh should not overwrite newer local chat state with stale DB rows."""
        persisted_history = [{"role": "user", "content": "hello"}]
        current_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]

        assert MCPGradioClientSession._history_is_prefix(
            persisted_history, current_history
        )

    def test_format_history_for_gradio_strips_non_chat_fields(self, tmp_path: Path):
        """Persisted history is normalized to role/content pairs for Gradio chat."""
        client = _create_client(tmp_path)

        history = client._format_history_for_gradio(
            [
                {
                    "role": "user",
                    "content": "hello",
                    "timestamp": "2026-01-01T00:00:00",
                },
                {
                    "role": "assistant",
                    "content": "world",
                    "timestamp": "2026-01-01T00:00:01",
                },
            ]
        )

        assert history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_select_session_formats_attachments_for_gradio(self, tmp_path: Path):
        """Reloading a stored session rehydrates image attachments for the chatbot."""
        client = _create_client(tmp_path)
        attachment = ImageAttachment.from_data(
            b"\x89PNG\r\n\x1a\nattachment", "image/png"
        )

        session_id = client.session_manager.create_session_id()
        client.session_manager.create_new_session(session_id)
        client.session_manager.add_message("assistant", "Rendered image", [attachment])

        session_update = client.update_session_choices()
        session_label = session_update["value"]

        history, _, _ = client.select_session(session_label)

        assert history[0]["role"] == "assistant"
        assert history[0]["content"][0] == "Rendered image"
        assert history[0]["content"][1]["file"]["mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_refresh_chat_preserves_optimistic_history_before_task_registration(
        self, tmp_path: Path
    ):
        """The refresh timer should not blank the chat while the send path is still persisting."""
        client = _create_client(tmp_path)
        client.session_manager.create_new_session("session-1")
        client.session_manager.select_session("session-1")
        client.orchestrator = SimpleNamespace(
            background_tasks=SimpleNamespace(
                get_task_snapshot=AsyncMock(return_value={}),
                count_pending_tasks=AsyncMock(return_value=0),
            )
        )

        optimistic_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]

        refreshed_history, task_status, session_update = (
            await client.refresh_chat_and_task_status(
                optimistic_history,
                client._current_session_label(),
            )
        )

        assert refreshed_history == gr.skip()
        assert task_status == "### Task Status\nNo background tasks yet."
        assert session_update == gr.skip()

    def test_update_session_choices_selects_current_session(self, tmp_path: Path):
        """The session list update keeps the current primary chat selected."""
        client = _create_client(tmp_path)

        client.create_new_session()

        session_update = client.update_session_choices()

        assert session_update["value"] is not None
        assert session_update["value"] in session_update["choices"]

    def test_ensure_active_session_visible_creates_missing_primary_session(
        self, tmp_path: Path
    ):
        """The first prompt can materialize the implicit primary chat in the sidebar."""
        client = _create_client(tmp_path)

        client.ensure_active_session_visible()
        session_update = client.update_session_choices()

        assert session_update["value"] is not None
        assert session_update["value"] in session_update["choices"]
