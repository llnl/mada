"""
Basic usage example for MADA.

This example demonstrates the simplest way to get started with MADA
using the provided configuration file.

Usage:
    python examples/basic_usage.py
"""

import asyncio
from pathlib import Path

from mada.core import load_config_from_json
from mada.interfaces.cli.main import MADACLIInterface


def main():
    """
    Basic example using the main configuration file.

    This is the simplest way to get started - just load the main config
    and launch an interface.
    """
    # Get path to main config file
    config_path = Path(__file__).parent.parent / "configs" / "mada_config.json"

    print("Loading main MADA configuration...")
    print(f"Config file: {config_path}")

    try:
        # Load the configuration
        config = load_config_from_json(str(config_path))

        print("Configuration loaded successfully")
        print(f"Agents: {[agent.agent_name for agent in config.agents]}")
        print(
            f"MCP Servers: {list(config.mcp_servers.keys()) if config.mcp_servers else 'None'}"
        )

        print("\nStarting CLI interface...")
        print("Try asking: 'Help me design a shaped charge optimization study'")
        print("Or: 'Set up a parameter sweep for hydrodynamics simulation'")
        print("Type 'quit' to exit\n")

        # Launch the CLI interface
        asyncio.run(run_cli(config))

    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Make sure you're running from the top level of the MADA repository")
    except Exception as e:
        print(f"Error loading configuration: {e}")


async def run_cli(config):
    """Run the CLI interface with the given configuration."""
    cli = MADACLIInterface(config)
    await cli.run()


if __name__ == "__main__":
    main()
