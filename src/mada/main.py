# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Main entry point for MADA.

Provides the main launcher that dispatches to different interface modes.
"""

import click

from mada.interfaces.cli.main import main as cli_entrypoint
from mada.interfaces.gradio.main import gradio_entrypoint


def _run_gradio_from_args(args: list[str]):
    """
    Run the Gradio mode for MADA.

    Args:
        args: command line arguments.
    """

    @click.command(
        context_settings={
            "help_option_names": ["-h", "--help"],
        },
    )
    @click.option(
        "-p",
        "--port",
        type=int,
        help="Port for Gradio server. Overrides 'interface.port' setting in the configuration file.",
    )
    @click.option(
        "-s",
        "--share",
        is_flag=True,
        help="Enable Gradio sharing. Overrides 'interface.share' setting in the configuration file.",
    )
    @click.argument(
        "config_file",
        type=str,
    )
    def gradio_cmd(port: int | None, share: bool, config_file: str) -> None:
        """
        Run MADA in Gradio mode.

        CONFIG_FILE is the path to the MADA configuration file.
        """
        gradio_entrypoint(port, share, config_file)

    # Invoke the inner command with the given args list
    gradio_cmd.main(args=args, standalone_mode=False)


def _run_cli_from_args(args: list[str]):
    """
    Run the CLI mode for MADA.

    Args:
        args: command line arguments.
    """
    cli_entrypoint.main(args=args, standalone_mode=False)


def _run_openai_api_from_args(args: list[str]):
    """
    Run the OpenAI-compatible API mode for MADA.

    Args:
        args: command line arguments.
    """
    from mada.interfaces.openai_api.main import openai_api_entrypoint

    @click.command(
        context_settings={
            "help_option_names": ["-h", "--help"],
        },
    )
    @click.option(
        "--host",
        type=str,
        default="0.0.0.0",
        show_default=True,
        help="Host interface to bind.",
    )
    @click.option(
        "-p",
        "--port",
        type=int,
        default=8000,
        show_default=True,
        help="Port for the OpenAI-compatible API.",
    )
    @click.option(
        "--model-name",
        type=str,
        default="mada-team",
        show_default=True,
        help="Model identifier exposed by /v1/models.",
    )
    @click.option(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key that incoming requests must provide.",
    )
    @click.option(
        "--bearer-token",
        type=str,
        default=None,
        help="Optional bearer token forwarded to streamable HTTP MCP servers as X-Token.",
    )
    @click.argument(
        "config_file",
        type=str,
    )
    def openai_api_cmd(
        host: str,
        port: int,
        model_name: str,
        api_key: str | None,
        bearer_token: str | None,
        config_file: str,
    ) -> None:
        """
        Run MADA in OpenAI-compatible API mode.

        CONFIG_FILE is the path to the MADA configuration file.
        """
        openai_api_entrypoint(
            host, port, model_name, api_key, bearer_token, config_file
        )

    openai_api_cmd.main(args=args, standalone_mode=False)


@click.command(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.argument(
    "mode",
    type=click.Choice(["gradio", "cli", "openai-api"], case_sensitive=False),
)
@click.pass_context
def main(ctx: click.Context, mode: str) -> None:
    """
    Run MADA.

    MODE is one of 'gradio', 'cli', or 'openai-api' and will determine the interface.

    Examples:

      mada gradio -p 7860 -s config.json

      mada cli config.json

      mada openai-api --port 8000 config.json
    """
    mode = mode.lower()

    # Remaining args after MODE
    remaining = list(ctx.args)

    if mode == "gradio":
        _run_gradio_from_args(remaining)
    elif mode == "cli":
        _run_cli_from_args(remaining)
    elif mode == "openai-api":
        _run_openai_api_from_args(remaining)
    else:
        # Protected by click.Choice, here just in case
        raise click.ClickException(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
