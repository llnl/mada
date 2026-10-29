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
from mada.core.orchestrator import MADAOrchestrator
from mada.core.orchestration.stream_events import (
    InternalActivitySignal,
    InternalResponseReplacement,
    InternalToolCallSignal,
)
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

        async def process_message(_message, **kwargs):
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

        async def process_message(_message, **kwargs):
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

        async def process_message(_message, **kwargs):
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
    @pytest.mark.parametrize("deleted_sessions", [False, True])
    async def test_background_progress_refreshes_before_completion(
        self, tmp_path, deleted_sessions
    ):
        client = _create_client(tmp_path)
        client.initialized = True
        client.orchestration_config.mode = "magentic"
        client.create_new_session()
        if deleted_sessions:
            client.session_manager.delete_all_sessions(confirm=False)
            assert client.session_manager.current_session_id is None
        advance, progressed, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        captured = {}

        async def process_message(message, **kwargs):
            captured.update(kwargs)
            yield InternalActivitySignal("SimulationAgent: starting\n")
            yield InternalToolCallSignal("submit_jobs")
            await advance.wait()
            yield InternalActivitySignal("JobManagementAgent: simulation completed\n")
            for chunk in ["hello", " ", "world"]:
                yield chunk
            progressed.set()
            await finish.wait()
            yield InternalResponseReplacement("Final answer")
            client.session_manager.chat_db.add_message(
                kwargs["persistence_session_id"], "assistant", "Final answer"
            )

        orchestrator = object.__new__(MADAOrchestrator)
        orchestrator.process_message = process_message
        manager = BackgroundTaskManager(
            client.session_manager, {}, orchestrator.collect_message_response
        )
        orchestrator.background_tasks = manager
        client.orchestrator = orchestrator

        async def refresh(history):
            return await client.refresh_chat_and_task_status(
                history, client._current_session_label()
            )

        try:
            output = [item async for item in client.process_message("run", [], None)]
            assert output == ["[task-1] Started in background."]
            original_session = client.session_manager.current_session_id
            assert original_session is not None
            saved = client.session_manager.load_history()
            assert saved[-1]["content"] == output[0]
            preview = (await refresh([]))[0]
            assert "SimulationAgent: starting" in preview[-1]["content"]
            assert await manager.count_pending_tasks() == 1
            assert not finish.is_set()
            assert client.session_manager.load_history() == saved

            advance.set()
            await asyncio.wait_for(progressed.wait(), 2)
            updated = (await refresh(preview))[0]
            assert "simulation completed" in updated[-1]["content"]
            assert "hello world" in updated[-1]["content"]
            assert client._formatted_history_cache[-1]["content"] == output[0]

            client.create_new_session()
            other_chat = (await refresh(updated))[0]
            assert other_chat == []
            assert captured["persistence_session_id"] == original_session
            finish.set()
            await asyncio.gather(*manager._pending_tasks.values())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert client.session_manager.load_history() == []
            client.session_manager.select_session(original_session)
            final = (await refresh(updated))[0]
            assert final[-1]["content"] == "Final answer"
            assert all(
                "hello world" not in str(message["content"]) for message in final
            )
        finally:
            advance.set()
            finish.set()
            await manager.cleanup()

    @pytest.mark.asyncio
    async def test_ack_and_context_stay_with_chat_selected_before_first_tool(
        self, tmp_path
    ):
        client = _create_client(tmp_path)
        client.initialized = True
        client.create_new_session()
        original_session = client.session_manager.current_session_id
        started, first_tool, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        captured = {}

        async def collect(message, **kwargs):
            captured.update(kwargs)
            started.set()
            await first_tool.wait()
            kwargs["first_tool_state"]["name"] = "submit_jobs"
            kwargs["first_tool_call"].set()
            await finish.wait()
            return "Done"

        manager = BackgroundTaskManager(client.session_manager, {}, collect)
        client.orchestrator = SimpleNamespace(background_tasks=manager)

        async def ask():
            return [item async for item in client.process_message("run", [], None)]

        request = asyncio.create_task(ask())
        try:
            await asyncio.wait_for(started.wait(), 2)
            client.create_new_session()
            client.session_manager.set_context_sessions([original_session])
            first_tool.set()
            await asyncio.wait_for(request, 2)
            assert client.session_manager.load_history() == []
            assert captured["persistence_session_id"] == original_session
            assert captured["context_session_ids"] == []
            saved = client.session_manager.load_history(original_session)
            assert saved[-1]["content"] == "[task-1] Started in background."
        finally:
            first_tool.set()
            finish.set()
            await request
            await manager.cleanup()

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
