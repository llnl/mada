from pathlib import Path

from mada.core.config import MCPServerConfig, OpenAIModelConfig, OrchestrationConfig, SQLiteConfig
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

    def test_ensure_active_session_visible_creates_missing_primary_session(
        self, tmp_path: Path
    ):
        """The first prompt can materialize the implicit primary chat in the sidebar."""
        client = _create_client(tmp_path)

        client.ensure_active_session_visible()
        session_update = client.update_session_choices()

        assert session_update["value"] is not None
        assert session_update["value"] in session_update["choices"]
