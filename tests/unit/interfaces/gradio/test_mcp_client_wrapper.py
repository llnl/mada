import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mada.core.background_tasks import BackgroundTaskManager
from mada.core.config import (
    MCPServerConfig,
    OpenAIModelConfig,
    OrchestrationConfig,
    SQLiteConfig,
)
from mada.core.media import ImageAttachment, RichResponse
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
    @pytest.mark.asyncio
    async def test_activity_survives_final_answer_replacement(self, tmp_path):
        from mada.core.orchestration.stream_events import (
            InternalActivitySignal,
            InternalResponseReplacement,
        )

        client = _create_client(tmp_path)
        client.initialized = True
        client.blocking = True

        async def process_message(_message):
            yield InternalActivitySignal("SimulationAgent: generated two candidates.\n")
            yield "Draft answer"
            yield InternalResponseReplacement("Final answer")

        client.orchestrator = SimpleNamespace(process_message=process_message)
        output = [item async for item in client.process_message("run", [], None)]
        assert output[-1] == "SimulationAgent: generated two candidates.\nFinal answer"

    @pytest.mark.asyncio
    async def test_process_message_streams_cumulative_progress(self, tmp_path: Path):
        client = _create_client(tmp_path)
        client.initialized = True
        client.blocking = True

        async def process_message(_message):
            yield "\n**Magentic round 1/3**\n"
            yield "SimulationAgent is generating candidates."
            yield "\nFinal recommendation."

        client.orchestrator = SimpleNamespace(process_message=process_message)

        output = [
            item async for item in client.process_message("run the workflow", [], None)
        ]

        assert output == [
            "\n**Magentic round 1/3**\n",
            "\n**Magentic round 1/3**\nSimulationAgent is generating candidates.",
            "\n**Magentic round 1/3**\nSimulationAgent is generating candidates.\nFinal recommendation.",
        ]

    @pytest.mark.asyncio
    async def test_magentic_stream_shows_start_status(self, tmp_path: Path):
        client = _create_client(tmp_path)
        client.initialized = True
        client.blocking = True
        client.orchestration_config.mode = "magentic"

        async def process_message(_message):
            yield "final answer"

        client.orchestrator = SimpleNamespace(process_message=process_message)

        output = [item async for item in client.process_message("run", [], None)]

        assert output[0] == "\n**Magentic workflow started.**\n"
        assert output[-1].endswith("final answer")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["agent_as_tool", "magentic"])
    async def test_nonblocking_query_detaches_and_populates_task_panel(
        self, tmp_path, mode
    ):
        client = _create_client(tmp_path)
        client.initialized = True
        client.orchestration_config.mode = mode
        client.create_new_session()
        finish = asyncio.Event()

        async def collect_response(message, **kwargs):
            kwargs["first_tool_state"]["name"] = "check_job_status"
            kwargs["first_tool_call"].set()
            await finish.wait()
            return "Jobs completed"

        manager = BackgroundTaskManager(client.session_manager, {}, collect_response)
        client.orchestrator = SimpleNamespace(background_tasks=manager)

        async def collect_chat():
            return [
                item async for item in client.process_message("check jobs", [], None)
            ]

        try:
            output = await asyncio.wait_for(collect_chat(), timeout=2)
            assert output == ["[task-1] Started in background."]
            assert not finish.is_set()
            panel = await client.get_task_status_markdown()
            assert "`task-1` [pending] check_job_status" in panel
            history = client.session_manager.load_history()
            assert [(item["role"], item["content"]) for item in history] == [
                ("user", "check jobs"),
                ("assistant", output[0]),
            ]
        finally:
            finish.set()
            await asyncio.gather(*manager._pending_tasks.values())
            # Completion bookkeeping is scheduled by the query's done callback.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        assert "`task-1` [completed]" in await client.get_task_status_markdown()

    @pytest.mark.asyncio
    async def test_nonblocking_text_response_stays_inline(self, tmp_path):
        client = _create_client(tmp_path)
        client.initialized = True
        manager = BackgroundTaskManager(
            client.session_manager, {}, AsyncMock(return_value="Hello")
        )
        client.orchestrator = SimpleNamespace(background_tasks=manager)

        output = [item async for item in client.process_message("hello", [], None)]

        assert output == ["Hello"]
        assert await manager.get_task_snapshot() == {}
        assert client.session_manager.load_history() == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["Velocity plot", "[Image attachment: plot.png]"])
    async def test_nonblocking_inline_response_preserves_images(self, tmp_path, text):
        client = _create_client(tmp_path)
        client.initialized = True
        attachment = ImageAttachment.from_data(
            b"\x89PNG\r\n\x1a\nimage-data", "image/png", "plot.png"
        )
        manager = BackgroundTaskManager(
            client.session_manager,
            {},
            AsyncMock(return_value=RichResponse(text, [attachment])),
        )
        client.orchestrator = SimpleNamespace(background_tasks=manager)

        output = [item async for item in client.process_message("show plot", [], None)]

        content = output[0]
        if text == "Velocity plot":
            assert content[:-1] == [text]
        else:
            assert len(content) == 1
        assert content[-1]["type"] == "file"
        assert Path(content[-1]["file"]["path"]).read_bytes() == attachment.data

    def test_history_for_display_strips_non_chat_fields(self):
        """Persisted history is normalized to role/content pairs for Gradio chat."""
        history = MCPGradioClientSession._history_for_display(
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

    def test_update_session_choices_selects_current_session(self, tmp_path: Path):
        """The session list update keeps the current primary chat selected."""
        client = _create_client(tmp_path)

        client.create_new_session()

        session_update = client.update_session_choices()

        assert session_update["value"] is not None
        assert session_update["value"] in session_update["choices"]

    def test_select_context_sessions_excludes_selected_primary_session(
        self, tmp_path: Path
    ):
        client = _create_client(tmp_path)
        client.create_new_session()
        first_label = client._current_session_label()
        client.create_new_session()
        second_label = client._current_session_label()

        client.select_context_sessions([first_label], primary_session_label=first_label)

        assert client.session_manager.get_context_session_ids() == []
        assert second_label != first_label

    def test_ensure_active_session_visible_creates_missing_primary_session(
        self, tmp_path: Path
    ):
        """The first prompt can materialize the implicit primary chat in the sidebar."""
        client = _create_client(tmp_path)

        client.ensure_active_session_visible()
        session_update = client.update_session_choices()

        assert session_update["value"] is not None
        assert session_update["value"] in session_update["choices"]
