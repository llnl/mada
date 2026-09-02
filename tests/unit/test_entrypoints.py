# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the following entry point modules:

- mada/main.py -> The `mada-orchestrator` command.
- mada/interface/cli/main.py -> The `mada-cli` command.
- mada/interface/gradio/main.py -> The `mada-gradio` command.
"""

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import httpx
from click.testing import CliRunner

from mada.core.config import (
    A2AConfig,
    MCPServerConfig,
    OpenAIModelConfig,
    OrchestrationConfig,
    SQLiteConfig,
    TelemetryConfig,
)
from mada.core.orchestration.stream_events import (
    InternalError,
    InternalResponseReplacement,
)
from mada.interfaces.a2a.main import (
    MADAA2AService,
    _resolve_public_a2a_url,
    a2a_entrypoint,
    create_a2a_app,
)
from mada.interfaces.a2a.main import main as a2a_main
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
    _run_a2a_from_args,
    _run_cli_from_args,
    _run_gradio_from_args,
    _run_openai_api_from_args,
    main,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, VERSION_HEADER


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
        self.orchestration = OrchestrationConfig()
        self.a2a = A2AConfig()
        self.a2a_agents = {}
        self.telemetry = TelemetryConfig()


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


def _asgi_test_client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


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

        def test_main_dispatches_to_a2a(self, runner):
            """
            Test that the main entry point correctly dispatches to the A2A API
            interface when the 'a2a' mode is specified.
            """
            with patch("mada.main._run_a2a_from_args") as mock_run_a2a:
                result = runner.invoke(main, ["a2a", "--port", "8000", "config.json"])
                assert result.exit_code == 0
                mock_run_a2a.assert_called_once_with(["--port", "8000", "config.json"])

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
                (call_arg,), _ = mock_asyncio_run.call_args
                call_arg.close()

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

    class TestRunA2AFromArgs:
        def test_run_a2a_from_args_calls_entrypoint(self):
            """
            Test that the helper function `_run_a2a_from_args` calls the A2A
            entry point with the correct arguments.
            """
            with patch("mada.interfaces.a2a.main.a2a_entrypoint") as mock_entry:
                _run_a2a_from_args(
                    [
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8000",
                        "--public-url",
                        "https://mada.example/a2a",
                        "config.json",
                    ]
                )
                mock_entry.assert_called_once_with(
                    "127.0.0.1",
                    8000,
                    "https://mada.example/a2a",
                    None,
                    None,
                    "config.json",
                )

        def test_run_a2a_from_args_uses_defaults(self):
            """
            Test `_run_a2a_from_args` when optional flags are not provided.
            """
            with patch("mada.interfaces.a2a.main.a2a_entrypoint") as mock_entry:
                _run_a2a_from_args(["config.json"])
                mock_entry.assert_called_once_with(
                    "0.0.0.0", 8000, None, None, None, "config.json"
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

        def test_gradio_entrypoint_reports_unsupported_orchestration_mode(self):
            with (
                patch("mada.interfaces.gradio.main.load_config_from_json") as mock_load,
                patch("mada.interfaces.gradio.main.sys.exit") as mock_exit,
                patch("builtins.print") as mock_print,
            ):
                mock_load.side_effect = ValueError(
                    "unsupported orchestration mode: magentic"
                )

                gradio_entrypoint(port=None, share=False, config_file="config.json")

                mock_exit.assert_called_once_with(1)
                printed = " ".join(
                    " ".join(str(arg) for arg in call.args)
                    for call in mock_print.call_args_list
                )
                assert "unsupported orchestration mode: magentic" in printed

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
                    a2a_agents={},
                    orchestration_config=OrchestrationConfig(),
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

        def test_openai_api_entrypoint_reports_unsupported_orchestration_mode(self):
            with (
                patch(
                    "mada.interfaces.openai_api.main.load_config_from_json"
                ) as mock_load,
                patch("mada.interfaces.openai_api.main.sys.exit") as mock_exit,
                patch("builtins.print") as mock_print,
            ):
                mock_load.side_effect = ValueError(
                    "unsupported orchestration mode: magentic"
                )

                openai_api_entrypoint(
                    host="127.0.0.1",
                    port=8000,
                    model_name="mada-api",
                    api_key=None,
                    bearer_token=None,
                    config_file="config.json",
                )

                mock_exit.assert_called_once_with(1)
                printed = " ".join(
                    " ".join(str(arg) for arg in call.args)
                    for call in mock_print.call_args_list
                )
                assert "unsupported orchestration mode: magentic" in printed

    class TestOpenAIApiService:
        @pytest.mark.asyncio
        async def test_collect_response_replaces_stale_magentic_chunks(
            self, create_dummy_config: Callable
        ):
            service = MADAOpenAIAPIService(config=create_dummy_config())

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "stale partial"
                yield InternalResponseReplacement("authoritative final")

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            response = await service.collect_response(
                [{"role": "user", "content": "hello"}]
            )

            assert response == "authoritative final"

        @pytest.mark.asyncio
        async def test_stream_response_emits_content_incrementally(
            self, create_dummy_config: Callable
        ):
            service = MADAOpenAIAPIService(config=create_dummy_config())
            never_finish = asyncio.Event()

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "first chunk"
                await never_finish.wait()

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            stream = service.stream_response([{"role": "user", "content": "hello"}])
            await anext(stream)
            content_event = await asyncio.wait_for(anext(stream), timeout=1)
            await stream.aclose()

            payload = json.loads(content_event.removeprefix("data: "))
            assert payload["choices"][0]["delta"]["content"] == "first chunk"

        @pytest.mark.asyncio
        async def test_stream_response_emits_replacement_before_content(
            self, create_dummy_config: Callable
        ):
            service = MADAOpenAIAPIService(config=create_dummy_config())

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield InternalResponseReplacement("authoritative final")

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            events = [
                event
                async for event in service.stream_response(
                    [{"role": "user", "content": "hello"}]
                )
            ]
            payloads = [
                json.loads(event.removeprefix("data: "))
                for event in events
                if event.startswith("data: {")
            ]
            content = "".join(
                [
                    payload["choices"][0]["delta"].get("content", "")
                    for payload in payloads
                ]
            )

            assert content == "authoritative final"

        @pytest.mark.asyncio
        async def test_stream_response_emits_authoritative_magentic_content(
            self, create_dummy_config: Callable
        ):
            service = MADAOpenAIAPIService(config=create_dummy_config())

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "authoritative final"

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            events = [
                event
                async for event in service.stream_response(
                    [{"role": "user", "content": "hello"}]
                )
            ]
            payloads = [
                json.loads(event.removeprefix("data: "))
                for event in events
                if event.startswith("data: {")
            ]
            content = "".join(
                payload["choices"][0]["delta"].get("content", "")
                for payload in payloads
                if "choices" in payload
            )
            errors = [payload["error"] for payload in payloads if "error" in payload]

            assert content == "authoritative final"
            assert errors == []

        @pytest.mark.asyncio
        async def test_stream_response_emits_clean_error_content(
            self, create_dummy_config: Callable
        ):
            service = MADAOpenAIAPIService(config=create_dummy_config())

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "Error processing message: boom"

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            events = [
                event
                async for event in service.stream_response(
                    [{"role": "user", "content": "hello"}]
                )
            ]
            payloads = [
                json.loads(event.removeprefix("data: "))
                for event in events
                if event.startswith("data: {")
            ]
            content = "".join(
                payload["choices"][0]["delta"].get("content", "")
                for payload in payloads
                if "choices" in payload
            )

            assert content == "Error processing message: boom"

        @pytest.mark.asyncio
        async def test_stream_response_reports_error_after_partial_content(
            self, create_dummy_config: Callable
        ):
            service = MADAOpenAIAPIService(config=create_dummy_config())

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "partial"
                yield InternalError("Error processing message: boom")

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            events = [
                event
                async for event in service.stream_response(
                    [{"role": "user", "content": "hello"}]
                )
            ]
            payloads = [
                json.loads(event.removeprefix("data: "))
                for event in events
                if event.startswith("data: {")
            ]
            content = "".join(
                payload["choices"][0]["delta"].get("content", "")
                for payload in payloads
                if "choices" in payload
            )
            assert all("choices" in payload for payload in payloads)
            finish_reasons = [
                payload["choices"][0]["finish_reason"]
                for payload in payloads
                if "choices" in payload
            ]

            assert content == "partial\n\n[Error: Error processing message: boom]"
            assert finish_reasons[-1] == "stop"
            assert events[-1] == "data: [DONE]\n\n"

        @pytest.mark.asyncio
        async def test_stream_response_buffers_and_sends_replacement_in_magentic(
            self, create_dummy_config: Callable
        ):
            config = create_dummy_config()
            config.orchestration = SimpleNamespace(mode="magentic")
            service = MADAOpenAIAPIService(config=config)

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "partial"
                yield InternalResponseReplacement("authoritative final")

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            events = [
                event
                async for event in service.stream_response(
                    [{"role": "user", "content": "hello"}]
                )
            ]
            payloads = [
                json.loads(event.removeprefix("data: "))
                for event in events
                if event.startswith("data: {")
            ]
            content = "".join(
                payload["choices"][0]["delta"].get("content", "")
                for payload in payloads
                if "choices" in payload
            )
            assert all("choices" in payload for payload in payloads)
            finish_reasons = [
                payload["choices"][0]["finish_reason"]
                for payload in payloads
                if "choices" in payload
            ]

            # Magentic mode buffers - replacement wins, not error message
            assert content == "authoritative final"
            assert finish_reasons[-1] == "stop"
            assert events[-1] == "data: [DONE]\n\n"

        @pytest.mark.asyncio
        async def test_stream_response_incremental_in_agent_as_tool(
            self, create_dummy_config: Callable
        ):
            config = create_dummy_config()
            config.orchestration = SimpleNamespace(mode="agent-as-tool")
            service = MADAOpenAIAPIService(config=config)

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "chunk1"
                yield "chunk2"
                yield "chunk3"

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            events = [
                event
                async for event in service.stream_response(
                    [{"role": "user", "content": "hello"}]
                )
            ]
            payloads = [
                json.loads(event.removeprefix("data: "))
                for event in events
                if event.startswith("data: {")
            ]
            content_chunks = [
                payload["choices"][0]["delta"].get("content", "")
                for payload in payloads
                if "choices" in payload
                and payload["choices"][0]["delta"].get("content")
            ]

            # agent-as-tool streams immediately - each chunk arrives separately
            assert content_chunks == ["chunk1", "chunk2", "chunk3"]
            assert events[-1] == "data: [DONE]\n\n"

        @pytest.mark.asyncio
        async def test_stream_response_flushes_buffer_before_error(
            self, create_dummy_config: Callable
        ):
            config = create_dummy_config()
            config.orchestration = SimpleNamespace(mode="magentic")
            service = MADAOpenAIAPIService(config=config)

            async def process_openai_messages(messages):
                assert messages == [{"role": "user", "content": "hello"}]
                yield "partial answer"
                yield InternalError("something went wrong")

            service.orchestrator = SimpleNamespace(
                process_openai_messages=process_openai_messages
            )

            events = [
                event
                async for event in service.stream_response(
                    [{"role": "user", "content": "hello"}]
                )
            ]
            payloads = [
                json.loads(event.removeprefix("data: "))
                for event in events
                if event.startswith("data: {")
            ]
            content = "".join(
                payload["choices"][0]["delta"].get("content", "")
                for payload in payloads
                if "choices" in payload
            )

            # Buffered content flushed, then error appended
            assert content.startswith("partial answer")
            assert "[Error: something went wrong]" in content
            assert events[-1] == "data: [DONE]\n\n"

    @pytest.mark.skipif(
        not hasattr(httpx, "ASGITransport"),
        reason="httpx ASGI transport is not installed",
    )
    class TestCreateOpenAIApiApp:
        @pytest.mark.asyncio
        async def test_models_endpoint_returns_exposed_model(
            self, create_dummy_config: Callable
        ):
            """
            Test that `/v1/models` returns the configured exposed model name.
            """
            config = create_dummy_config()

            with patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()):
                app = create_openai_api_app(config, model_name="mada-api")
                async with _asgi_test_client(app) as client:
                    response = await client.get("/v1/models")

                assert response.status_code == 200
                assert response.json()["data"][0]["id"] == "mada-api"

        @pytest.mark.asyncio
        async def test_models_endpoint_without_v1_returns_exposed_model(
            self, create_dummy_config: Callable
        ):
            """
            Test that `/models` returns the configured exposed model name.
            """
            config = create_dummy_config()

            with patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()):
                app = create_openai_api_app(config, model_name="mada-api")
                async with _asgi_test_client(app) as client:
                    response = await client.get("/models")

                assert response.status_code == 200
                assert response.json()["data"][0]["id"] == "mada-api"

        @pytest.mark.asyncio
        async def test_health_endpoint_reports_not_initialized_before_first_chat(
            self, create_dummy_config: Callable
        ):
            """
            Test that `/health` reports the orchestrator as uninitialized until a
            chat request triggers startup.
            """
            config = create_dummy_config()

            with patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()):
                app = create_openai_api_app(config, model_name="mada-api")
                async with _asgi_test_client(app) as client:
                    response = await client.get("/health")

                assert response.status_code == 200
                assert response.json()["orchestrator_initialized"] == "false"

        @pytest.mark.asyncio
        async def test_chat_completions_returns_openai_shape(
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
                async with _asgi_test_client(app) as client:
                    response = await client.post(
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

        @pytest.mark.asyncio
        async def test_chat_completions_returns_503_when_orchestrator_startup_fails(
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
                async with _asgi_test_client(app) as client:
                    response = await client.post(
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

        @pytest.mark.asyncio
        async def test_chat_completions_streams_sse_chunks(
            self, create_dummy_config: Callable
        ):
            """
            Test that the streaming chat completion endpoint emits OpenAI-style SSE
            events and terminates with `[DONE]`.
            """
            config = create_dummy_config()

            stream_chunks = [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
                "data: [DONE]\n\n",
            ]

            async def stream_response(self, messages):
                assert messages == [{"role": "user", "content": "hello"}]
                for chunk in stream_chunks:
                    yield chunk

            with (
                patch.object(MADAOpenAIAPIService, "ensure_started", new=AsyncMock()),
                patch.object(MADAOpenAIAPIService, "shutdown", new=AsyncMock()),
                patch.object(
                    MADAOpenAIAPIService,
                    "stream_response",
                    stream_response,
                ),
            ):
                app = create_openai_api_app(config, model_name="mada-api")
                async with _asgi_test_client(app) as client:
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "mada-api",
                            "stream": True,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                    body = response.text

            assert response.status_code == 200
            assert "data: [DONE]" in body


@pytest.mark.unit
class TestMADAA2ACmd:
    class TestA2AMain:
        def test_main_calls_a2a_entrypoint_with_args(self, runner):
            """
            Test that the A2A main function calls the entry point with the
            correct CLI arguments.
            """
            with patch("mada.interfaces.a2a.main.a2a_entrypoint") as mock_entrypoint:
                result = runner.invoke(
                    a2a_main,
                    [
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "9000",
                        "--public-url",
                        "https://mada.example/a2a",
                        "config.json",
                    ],
                )

                assert result.exit_code == 0
                mock_entrypoint.assert_called_once_with(
                    "127.0.0.1",
                    9000,
                    "https://mada.example/a2a",
                    None,
                    None,
                    "config.json",
                )

        def test_main_works_with_only_config_file(self, runner):
            """
            Test that the A2A main function uses default values when only the
            configuration file is provided.
            """
            with patch("mada.interfaces.a2a.main.a2a_entrypoint") as mock_entrypoint:
                result = runner.invoke(a2a_main, ["config.json"])

                assert result.exit_code == 0
                mock_entrypoint.assert_called_once_with(
                    "0.0.0.0", 8000, None, None, None, "config.json"
                )

    class TestA2AEntrypoint:
        def test_a2a_entrypoint_happy_path_uses_config_and_runs_server(
            self, create_dummy_config: Callable
        ):
            """
            Test that the A2A entry point loads the config and launches the API
            server.
            """
            config = create_dummy_config()

            with (
                patch(
                    "mada.interfaces.a2a.main.load_config_from_json",
                    return_value=config,
                ) as mock_load,
                patch("mada.interfaces.a2a.main.run_a2a") as mock_run,
                patch("mada.interfaces.a2a.main.sys.exit") as mock_exit,
            ):
                a2a_entrypoint(
                    host="127.0.0.1",
                    port=8000,
                    public_url="https://mada.example/a2a",
                    api_key="secret",
                    bearer_token="token",
                    config_file="config.json",
                )

                mock_load.assert_called_once_with("config.json")
                mock_run.assert_called_once_with(
                    config=config,
                    host="127.0.0.1",
                    port=8000,
                    public_url="https://mada.example/a2a",
                    api_key="secret",
                    bearer_token="token",
                )
                mock_exit.assert_not_called()

        def test_a2a_entrypoint_exits_with_code_1_on_exception(self):
            """
            Test that the A2A entry point exits with code 1 when an unexpected
            exception occurs.
            """
            with (
                patch("mada.interfaces.a2a.main.load_config_from_json") as mock_load,
                patch("mada.interfaces.a2a.main.sys.exit") as mock_exit,
            ):
                mock_load.side_effect = RuntimeError("Bad config")

                a2a_entrypoint(
                    host="127.0.0.1",
                    port=8000,
                    public_url=None,
                    api_key=None,
                    bearer_token=None,
                    config_file="config.json",
                )

                mock_exit.assert_called_once_with(1)

    class TestA2AUrl:
        def test_default_public_url_uses_non_loopback_ipv4_for_wildcard_host(self):
            with patch(
                "mada.interfaces.a2a.main.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("192.0.2.10", 0))],
            ):
                assert (
                    _resolve_public_a2a_url("0.0.0.0", 8000) == "http://192.0.2.10:8000"
                )

        def test_default_public_url_uses_non_loopback_ipv6_for_ipv6_wildcard_host(self):
            with patch(
                "mada.interfaces.a2a.main.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("2001:db8::10", 0, 0, 0))],
            ):
                assert (
                    _resolve_public_a2a_url("::", 8000) == "http://[2001:db8::10]:8000"
                )

        def test_default_public_url_preserves_explicit_public_url(self):
            assert (
                _resolve_public_a2a_url(
                    "0.0.0.0",
                    8000,
                    "https://mada.example/a2a",
                )
                == "https://mada.example/a2a"
            )

    class TestA2AService:
        @pytest.mark.asyncio
        async def test_collect_response_uses_isolated_session(
            self, create_dummy_config: Callable
        ):
            config = create_dummy_config()
            service = MADAA2AService(
                config=config,
                public_url="https://mada.example/a2a",
            )

            async def collect_message_response(
                message,
                isolated_session=False,
                persistence_session_id=None,
                stateless_session=False,
            ):
                assert message == "hello"
                assert isolated_session is True
                assert stateless_session is True
                # A2A uses stateless isolated sessions, no persistence_session_id
                assert persistence_session_id is None
                return "world"

            service.orchestrator = SimpleNamespace(
                collect_message_response=collect_message_response,
            )

            response = await service.collect_response("hello")

            assert response == "world"

        @pytest.mark.asyncio
        async def test_stream_response_yields_late_magentic_replacement(
            self, create_dummy_config: Callable
        ):
            config = create_dummy_config()
            config.orchestration = SimpleNamespace(mode="magentic")
            service = MADAA2AService(
                config=config,
                public_url="https://mada.example/a2a",
            )

            async def process_message(
                message,
                isolated_session=False,
                persistence_session_id=None,
                stateless_session=False,
            ):
                assert message == "hello"
                assert isolated_session is True
                assert stateless_session is True
                # A2A uses stateless isolated sessions, no persistence_session_id
                assert persistence_session_id is None
                yield "stale partial"
                yield InternalResponseReplacement("authoritative final")

            service.orchestrator = SimpleNamespace(
                process_message=process_message,
            )

            chunks = [chunk async for chunk in service.stream_response("hello")]

            assert chunks == ["authoritative final"]

        @pytest.mark.asyncio
        async def test_stream_response_yields_magentic_error(
            self, create_dummy_config: Callable
        ):
            config = create_dummy_config()
            config.orchestration = SimpleNamespace(mode="magentic")
            service = MADAA2AService(
                config=config,
                public_url="https://mada.example/a2a",
            )

            async def process_message(
                message,
                isolated_session=False,
                persistence_session_id=None,
                stateless_session=False,
            ):
                assert message == "hello"
                assert isolated_session is True
                assert stateless_session is True
                assert persistence_session_id is None
                yield "partial"
                yield InternalError("Error processing message: boom")

            service.orchestrator = SimpleNamespace(
                process_message=process_message,
            )

            chunks = [chunk async for chunk in service.stream_response("hello")]

            assert chunks == ["Error processing message: boom"]

        @pytest.mark.asyncio
        async def test_stream_response_streams_incrementally_in_agent_as_tool_mode(
            self, create_dummy_config: Callable
        ):
            service = MADAA2AService(
                config=create_dummy_config(),
                public_url="https://mada.example/a2a",
            )

            async def process_message(
                message,
                isolated_session=False,
                persistence_session_id=None,
                stateless_session=False,
            ):
                assert message == "hello"
                assert isolated_session is True
                assert stateless_session is True
                assert persistence_session_id is None
                yield "first"
                yield " second"

            service.orchestrator = SimpleNamespace(
                process_message=process_message,
            )

            chunks = [chunk async for chunk in service.stream_response("hello")]

            assert chunks == ["first", " second"]

    @pytest.mark.skipif(
        not hasattr(httpx, "ASGITransport"),
        reason="httpx ASGI transport is not installed",
    )
    class TestCreateA2AApp:
        @pytest.mark.asyncio
        async def test_agent_card_endpoint_returns_configured_metadata(
            self, create_dummy_config: Callable
        ):
            """
            Test that the standard agent card endpoint returns A2A metadata.
            """
            config = create_dummy_config()
            config.a2a = A2AConfig(
                name="MADA Test",
                description="Test A2A agent",
                version="9.9.9",
            )

            with patch.object(MADAA2AService, "shutdown", new=AsyncMock()):
                app = create_a2a_app(config, public_url="https://mada.example/a2a")
                async with _asgi_test_client(app) as client:
                    response = await client.get("/.well-known/agent-card.json")

            assert response.status_code == 200
            payload = response.json()
            assert payload["name"] == "MADA Test"
            assert payload["description"] == "Test A2A agent"
            assert payload["supportedInterfaces"][0]["url"] == (
                "https://mada.example/a2a"
            )
            assert payload["supportedInterfaces"][0]["protocolVersion"] == "1.0"
            assert payload["capabilities"]["streaming"] is True

        def test_agent_card_endpoint_can_serve_card_file(
            self, create_dummy_config: Callable, tmp_path: Path
        ):
            """
            Test that the standard agent card endpoint can load a standalone card.
            """
            card_path = tmp_path / "agent-card.json"
            card_path.write_text(
                json.dumps(
                    {
                        "name": "FileBackedMADA",
                        "description": "Loaded from a card file",
                        "version": "1.0.0",
                        "supportedInterfaces": [
                            {
                                "url": "http://placeholder",
                                "protocolBinding": "JSONRPC",
                                "protocolVersion": "1.0",
                            }
                        ],
                        "skills": [
                            {
                                "id": "file-backed",
                                "name": "File backed card",
                                "description": "Served from JSON",
                                "tags": ["a2a"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = create_dummy_config()
            config.a2a = A2AConfig(card_path=str(card_path))

            service = MADAA2AService(
                config=config, public_url="https://mada.example/a2a"
            )
            payload = service.build_agent_card()

            assert payload["name"] == "FileBackedMADA"
            assert payload["description"] == "Loaded from a card file"
            assert payload["supportedInterfaces"][0]["url"] == (
                "https://mada.example/a2a"
            )
            assert payload["supportedInterfaces"][0]["protocolVersion"] == "1.0"

        @pytest.mark.asyncio
        async def test_message_send_returns_a2a_task(
            self, create_dummy_config: Callable
        ):
            """
            Test that JSON-RPC `SendMessage` returns a completed A2A task.
            """
            config = create_dummy_config()

            with (
                patch.object(MADAA2AService, "ensure_started", new=AsyncMock()),
                patch.object(MADAA2AService, "shutdown", new=AsyncMock()),
                patch.object(
                    MADAA2AService,
                    "collect_response",
                    new=AsyncMock(return_value="hello from mada"),
                ),
            ):
                app = create_a2a_app(config, public_url="https://mada.example/a2a")
                async with _asgi_test_client(app) as client:
                    response = await client.post(
                        "/",
                        headers={VERSION_HEADER: PROTOCOL_VERSION_1_0},
                        json={
                            "jsonrpc": "2.0",
                            "id": "req-1",
                            "method": "SendMessage",
                            "params": {
                                "message": {
                                    "messageId": "msg-1",
                                    "role": "ROLE_USER",
                                    "parts": [{"text": "hello"}],
                                }
                            },
                        },
                    )

            assert response.status_code == 200
            payload = response.json()
            assert payload["id"] == "req-1"
            assert payload["result"]["message"]["parts"][0]["text"] == "hello from mada"


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
                (call_arg,), _ = mock_run.call_args
                call_arg.close()

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
                call_arg.close()

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

        @pytest.mark.asyncio
        async def test_async_main_reports_unsupported_orchestration_mode(self):
            with (
                patch("mada.interfaces.cli.main.load_config_from_json") as mock_load,
                patch("mada.interfaces.cli.main.sys.exit") as mock_exit,
                patch("builtins.print") as mock_print,
            ):
                mock_load.side_effect = ValueError(
                    "unsupported orchestration mode: magentic"
                )

                await async_main("config.json")

                mock_exit.assert_called_once_with(1)
                printed = " ".join(
                    " ".join(str(arg) for arg in call.args)
                    for call in mock_print.call_args_list
                )
                assert "unsupported orchestration mode: magentic" in printed

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
            orchestrator_mock.background_tasks = AsyncMock()
            orchestrator_mock.initialize_orchestrator.return_value = (
                "ok",
                ["tool1", "tool2"],
            )
            orchestrator_mock.background_tasks.count_pending_tasks.return_value = 0
            prompt_session = MagicMock()
            prompt_session.prompt_async = AsyncMock(return_value="quit")

            with (
                patch(
                    "mada.interfaces.cli.main.MADAOrchestrator",
                    return_value=orchestrator_mock,
                ),
                patch.object(
                    MADACLIInterface, "startup_session_menu", return_value=True
                ),
                patch(
                    "mada.interfaces.cli.main.PromptSession",
                    return_value=prompt_session,
                ),
                patch(
                    "mada.interfaces.cli.main.patch_stdout",
                    return_value=nullcontext(),
                ),
                patch("builtins.print") as mock_print,
            ):
                cli = MADACLIInterface(config)
                await cli.run()

                orchestrator_mock.initialize_orchestrator.assert_awaited_once_with(
                    config.agents, config.mcp_servers, config.a2a_agents
                )
                orchestrator_mock.background_tasks.run_query.assert_not_called()

                printed_texts = "".join(
                    str(call.args[0]) for call in mock_print.call_args_list
                )
                assert "Goodbye!" in printed_texts

        @pytest.mark.asyncio
        async def test_cli_interface_run_processes_one_message(
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
            orchestrator_mock.background_tasks = MagicMock()
            orchestrator_mock.background_tasks.count_pending_tasks = AsyncMock(
                return_value=0
            )
            orchestrator_mock.background_tasks.run_query = AsyncMock()
            prompt_session = MagicMock()
            prompt_session.prompt_async = AsyncMock(side_effect=["hello", "quit"])

            with (
                patch(
                    "mada.interfaces.cli.main.MADAOrchestrator",
                    return_value=orchestrator_mock,
                ),
                patch.object(
                    MADACLIInterface, "startup_session_menu", return_value=True
                ),
                patch(
                    "mada.interfaces.cli.main.PromptSession",
                    return_value=prompt_session,
                ),
                patch(
                    "mada.interfaces.cli.main.patch_stdout",
                    return_value=nullcontext(),
                ),
                patch("builtins.print") as mock_print,
            ):
                cli = MADACLIInterface(config)
                await cli.run()

                orchestrator_mock.background_tasks.run_query.assert_awaited_once_with(
                    "hello", blocking=False
                )

                printed_texts = "".join(
                    str(call.args[0]) for call in mock_print.call_args_list
                )
                assert "Goodbye!" in printed_texts
