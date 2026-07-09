# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the following entry point modules:

- mada/main.py -> The `mada-orchestrator` command.
- mada/interface/cli/main.py -> The `mada-cli` command.
- mada/interface/gradio/main.py -> The `mada-gradio` command.
"""

from pathlib import Path
from typing import Callable
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from mada.core.config import (
    MCPServerConfig,
    OpenAIModelConfig,
    SQLiteConfig,
)
from mada.interfaces.cli.main import MADACLIInterface, async_main
from mada.interfaces.cli.main import main as cli_main
from mada.interfaces.gradio.main import (
    create_gradio_app,
    gradio_entrypoint,
    run_gradio,
)
from mada.interfaces.gradio.main import main as gradio_main
from mada.interfaces.openai_api.main import (
    MADAOpenAIAPIService,
    create_openai_api_app,
    openai_api_entrypoint,
)
from mada.interfaces.openai_api.main import (
    main as openai_api_main,
)
from mada.main import (
    _run_cli_from_args,
    _run_gradio_from_args,
    _run_openai_api_from_args,
    main,
)

try:
    from fastapi.testclient import TestClient
except (
    ImportError
):  # pragma: no cover - exercised only in missing dependency environments
    TestClient = None


class DummyInterfaceConfig:
    def __init__(self, port=7860, share=False):
        self.port = port
        self.share = share


class DummyConfig:
    def __init__(self, interface=None, database=None):
        self.interface = interface
        self.model = OpenAIModelConfig(
            provider="openai",
            model="dummy-model",
            api_key="api-key",
            base_url="base-url",
        )
        self.agents = ["a1", "a2"]
        self.mcp_servers = {"s1": MCPServerConfig(transport="stdio")}
        self.database = database


@pytest.fixture
def db_config(tmp_path: Path) -> SQLiteConfig:
    """
    Sets up a database configuration for testing

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        A database config for testing
    """
    return SQLiteConfig(path=tmp_path / "test_histories.db")


@pytest.fixture
def create_dummy_config(db_config: SQLiteConfig) -> Callable:
    """
    A function for setting up DummyConfig instances with temporary
    SQLite databases.

    Args:
        db_config: A database config for testing

    Returns:
        Callable:
            A function for setting up DummyConfig instances with temporary
            SQLite databases.
    """

    def _create_dummy_config(interface=None):
        return DummyConfig(interface=interface, database=db_config)

    return _create_dummy_config


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.unit
class TestMADAOrchestratorCmd:
    class TestMADAOrchestratorMain:
        def test_main_dispatches_to_gradio(self, runner):
            """
            Test that the main entry point correctly dispatches to the Gradio interface
            when the 'gradio' mode is specified.
            """
            # We patch the helper so we do not run Gradio.
            with patch("mada.main._run_gradio_from_args") as mock_run_gradio:
                result = runner.invoke(
                    main, ["gradio", "-p", "7860", "-s", "config.json"]
                )
                assert result.exit_code == 0
                mock_run_gradio.assert_called_once_with(
                    ["-p", "7860", "-s", "config.json"]
                )

        def test_main_dispatches_to_cli(self, runner):
            """
            Test that the main entry point correctly dispatches to the CLI interface
            when the 'cli' mode is specified.
            """
            with patch("mada.main._run_cli_from_args") as mock_run_cli:
                result = runner.invoke(main, ["cli", "config.json"])
                assert result.exit_code == 0
                mock_run_cli.assert_called_once_with(["config.json"])

        def test_main_dispatches_to_openai_api(self, runner):
            """
            Test that the main entry point correctly dispatches to the OpenAI API
            interface when the 'openai-api' mode is specified.
            """
            with patch("mada.main._run_openai_api_from_args") as mock_run_openai_api:
                result = runner.invoke(
                    main, ["openai-api", "--port", "8000", "config.json"]
                )
                assert result.exit_code == 0
                mock_run_openai_api.assert_called_once_with(
                    ["--port", "8000", "config.json"]
                )

    class TestRunGradioFromArgs:
        def test_run_gradio_from_args_calls_entrypoint(self):
            """
            Test that the helper function `_run_gradio_from_args` calls the Gradio
            entry point with the correct arguments.
            """
            with patch("mada.main.gradio_entrypoint") as mock_entry:
                _run_gradio_from_args(["-p", "7860", "-s", "config.json"])
                mock_entry.assert_called_once_with(7860, True, "config.json")

        def test_run_gradio_from_args_without_optional_flags(self):
            """
            Test `_run_gradio_from_args` when optional flags are not provided.
            Ensures default values are used.
            """
            with patch("mada.main.gradio_entrypoint") as mock_entry:
                _run_gradio_from_args(["config.json"])
                # port is None, share is False by default
                mock_entry.assert_called_once_with(None, False, "config.json")

    class TestRunCLIFromArgs:
        def test_run_cli_from_args_calls_async_main(self):
            """
            Test that `_run_cli_from_args` calls the asynchronous main function
            for the CLI interface with the correct arguments.
            """
            with (
                patch("mada.main.cli_async_main") as mock_async_main,
                patch("mada.main.asyncio.run") as mock_asyncio_run,
            ):
                _run_cli_from_args(["config.json"])

                # Ensure we call asyncio.run with the coroutine returned by cli_async_main("config.json")
                mock_async_main.assert_called_once_with("config.json")
                # We do not care about exact object, just that asyncio.run was used
                mock_asyncio_run.assert_called_once()

    class TestRunOpenAIApiFromArgs:
        def test_run_openai_api_from_args_calls_entrypoint(self):
            """
            Test that the helper function `_run_openai_api_from_args` calls the
            OpenAI API entry point with the correct arguments.
            """
            with patch(
                "mada.interfaces.openai_api.main.openai_api_entrypoint"
            ) as mock_entry:
                _run_openai_api_from_args(
                    [
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8000",
                        "--model-name",
                        "mada-api",
                        "config.json",
                    ]
                )
                mock_entry.assert_called_once_with(
                    "127.0.0.1", 8000, "mada-api", None, None, "config.json"
                )

        def test_run_openai_api_from_args_uses_defaults(self):
            """
            Test `_run_openai_api_from_args` when optional flags are not provided.
            """
            with patch(
                "mada.interfaces.openai_api.main.openai_api_entrypoint"
            ) as mock_entry:
                _run_openai_api_from_args(["config.json"])
                mock_entry.assert_called_once_with(
                    "0.0.0.0", 8000, "mada-team", None, None, "config.json"
                )


@pytest.mark.unit
class TestMADAGradioCmd:
    class TestGradioMain:
        def test_main_calls_gradio_entrypoint_with_args(self, runner):
            """
            Test that the Gradio main function calls the Gradio entry point
            with the correct arguments passed via the CLI.
            """
            with patch(
                "mada.interfaces.gradio.main.gradio_entrypoint"
            ) as mock_entrypoint:
                result = runner.invoke(gradio_main, ["-p", "9999", "-s", "config.json"])

                assert result.exit_code == 0
                mock_entrypoint.assert_called_once_with(9999, True, "config.json")

        def test_main_works_with_only_config_file(self, runner):
            """
            Test that the Gradio main function works correctly when only
            the configuration file is provided.
            """
            with patch(
                "mada.interfaces.gradio.main.gradio_entrypoint"
            ) as mock_entrypoint:
                result = runner.invoke(gradio_main, ["config.json"])

                assert result.exit_code == 0
                # port should be None, share False by default
                mock_entrypoint.assert_called_once_with(None, False, "config.json")

    class TestGradioEntrypoint:
        def test_gradio_entrypoint_happy_path_uses_config_and_run_gradio(
            self, create_dummy_config: Callable
        ):
            """
            Test that the Gradio entry point correctly uses the configuration
            file and launches the Gradio interface without errors.
            """
            dummy_interface = DummyInterfaceConfig(port=1234, share=False)
            config = create_dummy_config(interface=dummy_interface)

            with (
                patch(
                    "mada.interfaces.gradio.main.load_config_from_json",
                    return_value=config,
                ) as mock_load,
                patch("mada.interfaces.gradio.main.run_gradio") as mock_run,
                patch("mada.interfaces.gradio.main.sys.exit") as mock_exit,
            ):
                gradio_entrypoint(port=None, share=False, config_file="config.json")

                mock_load.assert_called_once_with("config.json")
                mock_run.assert_called_once_with(config)
                # Should not call sys.exit on success
                mock_exit.assert_not_called()
                # Port / share unchanged when no overrides
                assert config.interface.port == 1234
                assert config.interface.share is False

        def test_gradio_entrypoint_applies_port_and_share_overrides(
            self, create_dummy_config: Callable
        ):
            """
            Test that the Gradio entry point applies port and share overrides
            provided via the CLI.
            """
            dummy_interface = DummyInterfaceConfig(port=1234, share=False)
            config = create_dummy_config(interface=dummy_interface)

            with (
                patch(
                    "mada.interfaces.gradio.main.load_config_from_json",
                    return_value=config,
                ),
                patch("mada.interfaces.gradio.main.run_gradio") as mock_run,
                patch("mada.interfaces.gradio.main.sys.exit") as mock_exit,
            ):
                gradio_entrypoint(port=9999, share=True, config_file="config.json")

                # Overrides applied
                assert config.interface.port == 9999
                assert config.interface.share is True

                mock_run.assert_called_once_with(config)
                mock_exit.assert_not_called()

        def test_gradio_entrypoint_exits_if_no_interface_config(
            self, create_dummy_config: Callable
        ):
            """
            Test that the Gradio entry point exits with an error if the
            configuration file does not contain interface settings.
            """
            # config.interface is None
            config = create_dummy_config(interface=None)

            with (
                patch(
                    "mada.interfaces.gradio.main.load_config_from_json",
                    return_value=config,
                ),
                patch("mada.interfaces.gradio.main.run_gradio") as mock_run,
                patch("mada.interfaces.gradio.main.sys.exit") as mock_exit,
            ):
                gradio_entrypoint(port=None, share=False, config_file="config.json")

                # Should not call run_gradio because interface is missing
                mock_run.assert_not_called()

                # Should exit
                mock_exit.assert_called()

        def test_gradio_entrypoint_exits_with_code_1_on_exception(self):
            """
            Test that the Gradio entry point exits with code 1 when an
            unexpected exception occurs.
            """
            with (
                patch("mada.interfaces.gradio.main.load_config_from_json") as mock_load,
                patch("mada.interfaces.gradio.main.sys.exit") as mock_exit,
            ):
                mock_load.side_effect = RuntimeError("Bad config")

                gradio_entrypoint(port=None, share=False, config_file="config.json")

                # Should exit with code 1 on unexpected error
                mock_exit.assert_called_once_with(1)

    class TestRunGradio:
        def test_run_gradio_launches_interface_with_defaults(
            self, create_dummy_config: Callable, db_config: SQLiteConfig
        ):
            """
            Test that the `run_gradio` function launches the Gradio interface
            with default settings when no interface configuration is provided.
            """
            config = create_dummy_config(interface=None)

            mock_blocks = MagicMock()
            mock_blocks.queue.return_value = None

            mock_loop = MagicMock()

            with (
                patch(
                    "mada.interfaces.gradio.main.MCPGradioClientSession"
                ) as mock_client_cls,
                patch(
                    "mada.interfaces.gradio.main.MADAMultiAgentGradioInterface"
                ) as mock_iface_cls,
                patch(
                    "mada.interfaces.gradio.main.asyncio.new_event_loop",
                    return_value=mock_loop,
                ) as mock_new_loop,
                patch("mada.interfaces.gradio.main.asyncio.set_event_loop"),
            ):
                mock_iface_instance = mock_iface_cls.return_value
                mock_iface_instance.create_interface.return_value = mock_blocks

                run_gradio(config)

                mock_new_loop.assert_called_once()
                mock_client_cls.assert_called_once_with(
                    model_config=OpenAIModelConfig(
                        provider="openai",
                        model="dummy-model",
                        api_key="api-key",
                        base_url="base-url",
                    ),
                    agents=["a1", "a2"],
                    database_config=db_config,
                    mcp_servers={"s1": MCPServerConfig(transport="stdio")},
                )
                mock_iface_cls.assert_called_once()
                mock_iface_instance.create_interface.assert_called_once()
                mock_blocks.queue.assert_called_once_with(max_size=20)

                mock_blocks.launch.assert_called_once_with(
                    server_name="0.0.0.0",
                    server_port=7860,
                    share=False,
                    debug=True,
                    css=ANY,
                    js=ANY,
                )

        def test_run_gradio_launches_interface_with_configured_port_and_share(
            self, create_dummy_config: Callable
        ):
            """
            Test that the `run_gradio` function launches the Gradio interface
            with the port and share settings specified in the configuration.
            """
            interface_cfg = DummyInterfaceConfig(port=9000, share=True)
            config = create_dummy_config(interface=interface_cfg)

            mock_blocks = MagicMock()
            with (
                patch("mada.interfaces.gradio.main.MCPGradioClientSession"),
                patch(
                    "mada.interfaces.gradio.main.MADAMultiAgentGradioInterface"
                ) as mock_iface_cls,
                patch(
                    "mada.interfaces.gradio.main.asyncio.new_event_loop"
                ) as mock_new_loop,
                patch("mada.interfaces.gradio.main.asyncio.set_event_loop"),
            ):
                mock_new_loop.return_value = MagicMock()

                mock_iface_instance = mock_iface_cls.return_value
                mock_iface_instance.create_interface.return_value = mock_blocks

                run_gradio(config)

                mock_blocks.launch.assert_called_once_with(
                    server_name="0.0.0.0",
                    server_port=9000,
                    share=True,
                    debug=True,
                    css=ANY,
                    js=ANY,
                )

    class TestCreateGradioApp:
        def test_create_gradio_app_builds_blocks_from_config(
            self, create_dummy_config: Callable
        ):
            """
            Test that the `create_gradio_app` function builds a Gradio Blocks
            interface using the provided configuration file.
            """
            dummy_config = create_dummy_config()
            dummy_blocks = MagicMock()

            with (
                patch(
                    "mada.interfaces.gradio.main.load_config_from_json",
                    return_value=dummy_config,
                ) as mock_load,
                patch(
                    "mada.interfaces.gradio.main.MCPGradioClientSession"
                ) as mock_client_cls,
                patch(
                    "mada.interfaces.gradio.main.MADAMultiAgentGradioInterface"
                ) as mock_iface_cls,
            ):
                mock_iface_instance = mock_iface_cls.return_value
                mock_iface_instance.create_interface.return_value = dummy_blocks

                result = create_gradio_app("config.json")

                mock_load.assert_called_once_with("config.json")
                mock_client_cls.assert_called_once()
                mock_iface_cls.assert_called_once()
                mock_iface_instance.create_interface.assert_called_once()
                assert result is dummy_blocks


@pytest.mark.unit
class TestMADAOpenAIApiCmd:
    class TestOpenAIApiMain:
        def test_main_calls_openai_api_entrypoint_with_args(self, runner):
            """
            Test that the OpenAI API main function calls the entry point with
            the correct CLI arguments.
            """
            with patch(
                "mada.interfaces.openai_api.main.openai_api_entrypoint"
            ) as mock_entrypoint:
                result = runner.invoke(
                    openai_api_main,
                    [
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "9000",
                        "--model-name",
                        "mada-api",
                        "config.json",
                    ],
                )

                assert result.exit_code == 0
                mock_entrypoint.assert_called_once_with(
                    "127.0.0.1", 9000, "mada-api", None, None, "config.json"
                )

        def test_main_works_with_only_config_file(self, runner):
            """
            Test that the OpenAI API main function uses default values when only
            the configuration file is provided.
            """
            with patch(
                "mada.interfaces.openai_api.main.openai_api_entrypoint"
            ) as mock_entrypoint:
                result = runner.invoke(openai_api_main, ["config.json"])

                assert result.exit_code == 0
                mock_entrypoint.assert_called_once_with(
                    "0.0.0.0", 8000, "mada", None, None, "config.json"
                )

    class TestOpenAIApiEntrypoint:
        def test_openai_api_entrypoint_happy_path_uses_config_and_runs_server(
            self, create_dummy_config: Callable
        ):
            """
            Test that the OpenAI API entry point correctly loads the config and
            launches the API server.
            """
            config = create_dummy_config()

            with (
                patch(
                    "mada.interfaces.openai_api.main.load_config_from_json",
                    return_value=config,
                ) as mock_load,
                patch("mada.interfaces.openai_api.main.run_openai_api") as mock_run,
                patch("mada.interfaces.openai_api.main.sys.exit") as mock_exit,
            ):
                openai_api_entrypoint(
                    host="127.0.0.1",
                    port=8000,
                    model_name="mada-api",
                    api_key="secret",
                    bearer_token="token",
                    config_file="config.json",
                )

                mock_load.assert_called_once_with("config.json")
                mock_run.assert_called_once_with(
                    config=config,
                    host="127.0.0.1",
                    port=8000,
                    model_name="mada-api",
                    api_key="secret",
                    bearer_token="token",
                )
                mock_exit.assert_not_called()

        def test_openai_api_entrypoint_exits_with_code_1_on_exception(self):
            """
            Test that the OpenAI API entry point exits with code 1 when an
            unexpected exception occurs.
            """
            with (
                patch(
                    "mada.interfaces.openai_api.main.load_config_from_json"
                ) as mock_load,
                patch("mada.interfaces.openai_api.main.sys.exit") as mock_exit,
            ):
                mock_load.side_effect = RuntimeError("Bad config")

                openai_api_entrypoint(
                    host="127.0.0.1",
                    port=8000,
                    model_name="mada-api",
                    api_key=None,
                    bearer_token=None,
                    config_file="config.json",
                )

                mock_exit.assert_called_once_with(1)

    @pytest.mark.skipif(
        TestClient is None, reason="fastapi test client is not installed"
    )
    class TestCreateOpenAIApiApp:
        def test_models_endpoint_returns_exposed_model(
            self, create_dummy_config: Callable
        ):
            """
            Test that `/v1/models` returns the configured exposed model name.
            """
            config = create_dummy_config()

            with patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()):
                app = create_openai_api_app(config, model_name="mada-api")
                with TestClient(app) as client:
                    response = client.get("/v1/models")

                assert response.status_code == 200
                assert response.json()["data"][0]["id"] == "mada-api"

        def test_models_endpoint_without_v1_returns_exposed_model(
            self, create_dummy_config: Callable
        ):
            """
            Test that `/models` returns the configured exposed model name.
            """
            config = create_dummy_config()

            with patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()):
                app = create_openai_api_app(config, model_name="mada-api")
                with TestClient(app) as client:
                    response = client.get("/models")

                assert response.status_code == 200
                assert response.json()["data"][0]["id"] == "mada-api"

        def test_health_endpoint_reports_not_initialized_before_first_chat(
            self, create_dummy_config: Callable
        ):
            """
            Test that `/health` reports the orchestrator as uninitialized until a
            chat request triggers startup.
            """
            config = create_dummy_config()

            with patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()):
                app = create_openai_api_app(config, model_name="mada-api")
                with TestClient(app) as client:
                    response = client.get("/health")

                assert response.status_code == 200
                assert response.json()["orchestrator_initialized"] == "false"

        def test_chat_completions_returns_openai_shape(
            self, create_dummy_config: Callable
        ):
            """
            Test that the non-streaming chat completion endpoint returns an
            OpenAI-compatible completion payload.
            """
            config = create_dummy_config()

            with (
                patch.object(MADAOpenAIAPIService, "ensure_started", new=AsyncMock()),
                patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()),
                patch.object(
                    MADAOpenAIAPIService,
                    "collect_response",
                    new=AsyncMock(return_value="hello from mada"),
                ),
            ):
                app = create_openai_api_app(config, model_name="mada-api")
                with TestClient(app) as client:
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "mada-api",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )

                assert response.status_code == 200
                payload = response.json()
                assert payload["object"] == "chat.completion"
                assert payload["choices"][0]["message"]["content"] == "hello from mada"

        def test_chat_completions_returns_503_when_orchestrator_startup_fails(
            self, create_dummy_config: Callable
        ):
            """
            Test that chat completion requests return a clean 503 when MCP-backed
            orchestrator startup fails.
            """
            config = create_dummy_config()

            with (
                patch("mada.core.orchestrator.MADAOrchestrator") as mock_orchestrator,
                patch.object(
                    MADAOpenAIAPIService,
                    "shutdown",
                    new=AsyncMock(),
                ),
            ):
                mock_instance = mock_orchestrator.return_value
                mock_instance.__aenter__ = AsyncMock(return_value=None)
                mock_instance.__aexit__ = AsyncMock(return_value=None)
                mock_instance.initialize_orchestrator = AsyncMock(
                    side_effect=RuntimeError("random startup failure")
                )

                app = create_openai_api_app(config, model_name="mada-api")
                with TestClient(app) as client:
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "mada-api",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )

            assert response.status_code == 503
            detail = response.json()["detail"]
            assert (
                "MADA failed to initialize the configured agent team" in detail
                or "MADA could not connect to one or more MCP servers" in detail
            )
            assert "random startup failure" in detail

        def test_chat_completions_streams_sse_chunks(
            self, create_dummy_config: Callable
        ):
            """
            Test that the streaming chat completion endpoint emits OpenAI-style SSE
            events and terminates with `[DONE]`.
            """
            config = create_dummy_config()

            async def fake_stream_response(_messages):
                yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                yield "data: [DONE]\n\n"

            with (
                patch.object(MADAOpenAIAPIService, "ensure_started", new=AsyncMock()),
                patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()),
                patch.object(
                    MADAOpenAIAPIService,
                    "stream_response",
                    side_effect=fake_stream_response,
                ),
            ):
                app = create_openai_api_app(config, model_name="mada-api")
                with TestClient(app) as client:
                    with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json={
                            "model": "mada-api",
                            "stream": True,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    ) as response:
                        body = "".join(response.iter_text())

                assert response.status_code == 200
                assert "data: [DONE]" in body


@pytest.mark.unit
class TestMADACLICmd:
    class TestCLIMain:
        def test_main_calls_asyncio_run_with_async_main(self, runner):
            """
            Test that the CLI main function calls `asyncio.run` with the
            asynchronous main function.
            """
            with patch("mada.interfaces.cli.main.asyncio.run") as mock_run:
                result = runner.invoke(cli_main, ["config.json"])

                assert result.exit_code == 0
                # Confirm asyncio.run was called once
                mock_run.assert_called_once()

        def test_main_calls_asyncio_run_with_async_main_strict(self, runner):
            """
            Test that the CLI main function calls `asyncio.run` with the
            asynchronous main function and verifies the coroutine object.
            """
            with patch("mada.interfaces.cli.main.asyncio.run") as mock_run:
                result = runner.invoke(cli_main, ["config.json"])

                assert result.exit_code == 0
                (call_arg,), _ = mock_run.call_args
                # call_arg should be a coroutine object from async_main("config.json")
                assert call_arg.cr_code is async_main.__code__
                assert call_arg.cr_await is None or hasattr(call_arg, "cr_frame")

    class TestAsyncMain:
        @pytest.mark.asyncio
        async def test_async_main_happy_path(self, create_dummy_config: Callable):
            """
            Test that the asynchronous main function for the CLI interface
            runs successfully with a valid configuration file.
            """
            dummy_config = create_dummy_config()

            with (
                patch(
                    "mada.interfaces.cli.main.load_config_from_json",
                    return_value=dummy_config,
                ) as mock_load,
                patch("mada.interfaces.cli.main.MADACLIInterface") as mock_cli_class,
                patch("mada.interfaces.cli.main.sys.exit") as mock_exit,
            ):
                mock_cli_instance = AsyncMock()
                mock_cli_class.return_value = mock_cli_instance

                await async_main("config.json")

                mock_load.assert_called_once_with("config.json")
                mock_cli_class.assert_called_once_with(dummy_config, blocking=False)
                mock_cli_instance.run.assert_awaited_once()
                # Should not call sys.exit on success
                mock_exit.assert_not_called()

        @pytest.mark.asyncio
        async def test_async_main_file_not_found_exits_with_code_1(self):
            """
            Test that the asynchronous main function exits with code 1
            when the configuration file is not found.
            """
            with (
                patch("mada.interfaces.cli.main.load_config_from_json") as mock_load,
                patch("mada.interfaces.cli.main.sys.exit") as mock_exit,
            ):
                mock_load.side_effect = FileNotFoundError("no file")

                await async_main("missing.json")

                mock_load.assert_called_once_with("missing.json")
                mock_exit.assert_called_once_with(1)

        @pytest.mark.asyncio
        async def test_async_main_generic_error_exits_with_code_1(self):
            """
            Test that the asynchronous main function exits with code 1
            when a generic runtime error occurs.
            """
            with (
                patch("mada.interfaces.cli.main.load_config_from_json") as mock_load,
                patch("mada.interfaces.cli.main.sys.exit") as mock_exit,
            ):
                mock_load.side_effect = RuntimeError("boom")

                await async_main("config.json")

                mock_exit.assert_called_once_with(1)

    class TestMADACLIInterface:
        @pytest.mark.asyncio
        async def test_cli_interface_run_quit_immediately(
            self, create_dummy_config: Callable
        ):
            """
            Test that the CLI interface exits immediately when the user
            types 'quit' as the first input.
            """
            config = create_dummy_config()

            orchestrator_mock = AsyncMock()
            orchestrator_mock.__aenter__.return_value = orchestrator_mock
            orchestrator_mock.__aexit__.return_value = False
            orchestrator_mock.initialize_orchestrator.return_value = (
                "ok",
                ["tool1", "tool2"],
            )
            orchestrator_mock.process_message = AsyncMock()

            with (
                patch(
                    "mada.interfaces.cli.main.MADAOrchestrator",
                    return_value=orchestrator_mock,
                ),
                patch.object(
                    MADACLIInterface, "startup_session_menu", return_value=True
                ),
                patch(
                    "mada.interfaces.cli.main.asyncio.to_thread",
                    new=AsyncMock(side_effect=["quit"]),
                ),
                patch("builtins.print") as mock_print,
            ):
                cli = MADACLIInterface(config)
                await cli.run()

                orchestrator_mock.initialize_orchestrator.assert_awaited_once_with(
                    config.agents, config.mcp_servers
                )
                orchestrator_mock.process_message.assert_not_called()

                printed_texts = "".join(
                    str(call.args[0]) for call in mock_print.call_args_list
                )
                assert "Goodbye!" in printed_texts

        @pytest.mark.asyncio
        async def test_cli_interface_run_processes_one_message_in_blocking_mode(
            self, create_dummy_config: Callable
        ):
            """
            Test that the CLI interface processes one user message and
            correctly handles the response from the orchestrator.
            """
            config = create_dummy_config()

            orchestrator_mock = MagicMock()
            orchestrator_mock.__aenter__ = AsyncMock(return_value=orchestrator_mock)
            orchestrator_mock.__aexit__ = AsyncMock(return_value=False)
            orchestrator_mock.initialize_orchestrator = AsyncMock(
                return_value=("ok", [])
            )

            async def fake_process_message(_msg):
                yield "chunk1"
                yield "chunk2"

            orchestrator_mock.process_message = MagicMock(
                side_effect=fake_process_message
            )

            with (
                patch(
                    "mada.interfaces.cli.main.MADAOrchestrator",
                    return_value=orchestrator_mock,
                ),
                patch.object(
                    MADACLIInterface, "startup_session_menu", return_value=True
                ),
                patch(
                    "mada.interfaces.cli.main.asyncio.to_thread",
                    new=AsyncMock(side_effect=["hello", "quit"]),
                ),
                patch("builtins.print") as mock_print,
            ):
                cli = MADACLIInterface(config, blocking=True)
                await cli.run()

                orchestrator_mock.process_message.assert_called_once_with("hello")

                printed_texts = "".join(
                    str(call.args[0]) for call in mock_print.call_args_list
                )
                assert "chunk1" in printed_texts
                assert "chunk2" in printed_texts

        @pytest.mark.asyncio
        async def test_cli_interface_run_uses_background_query_mode_by_default(
            self, create_dummy_config: Callable
        ):
            """
            Test that the CLI routes queries through background mode by default.
            """
            config = create_dummy_config()

            orchestrator_mock = AsyncMock()
            orchestrator_mock.__aenter__.return_value = orchestrator_mock
            orchestrator_mock.__aexit__.return_value = False
            orchestrator_mock.initialize_orchestrator.return_value = ("ok", [])

            with (
                patch(
                    "mada.interfaces.cli.main.MADAOrchestrator",
                    return_value=orchestrator_mock,
                ),
                patch.object(
                    MADACLIInterface, "startup_session_menu", return_value=True
                ),
                patch(
                    "mada.interfaces.cli.main.asyncio.to_thread",
                    new=AsyncMock(side_effect=["hello", "quit"]),
                ),
                patch.object(
                    MADACLIInterface, "run_query", new=AsyncMock()
                ) as mock_run_query,
                patch("builtins.print") as mock_print,
            ):
                cli = MADACLIInterface(config)
                await cli.run()

                mock_run_query.assert_awaited_once_with("hello")
                printed_texts = "".join(
                    str(call.args[0]) for call in mock_print.call_args_list
                )
                assert "Query mode: background" in printed_texts
