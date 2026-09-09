# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MADA - Multi-agent orchestration system for MADA workflows.

This package provides orchestration capabilities for coordinating multiple
autonomous agents that interact with MCP servers to execute complex workflows.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import inspect
import tomllib


PACKAGE_NAME = "mada"


# TODO: Once we can just use `agent-framework` as a single dependency again,
# figure out which version supports fastmcp>=4.0 and mcp>=2.0 and remove this
# compatibility layer.
def _install_mcp_compatibility_aliases() -> None:
    """
    Add compatibility aliases expected by older Agent Framework releases.

    Agent Framework 1.9 still reads ``InitializeResult.protocolVersion`` while
    MCP 2.x exposes ``protocol_version``. Install legacy camelCase aliases on
    MCP model types at import time so MCP server interactions work until the
    framework dependency is updated.
    """
    try:
        import mcp.types as mcp_types
        from mcp.shared import exceptions as mcp_exceptions
    except Exception:
        return

    for _, model_type in inspect.getmembers(mcp_types, inspect.isclass):
        model_fields = getattr(model_type, "model_fields", None)
        if not model_fields:
            continue

        for field_name in model_fields:
            if "_" not in field_name:
                continue

            parts = field_name.split("_")
            alias = parts[0] + "".join(part.capitalize() for part in parts[1:])
            if hasattr(model_type, alias):
                continue

            setattr(
                model_type,
                alias,
                property(lambda self, field_name=field_name: getattr(self, field_name)),
            )

    if not hasattr(mcp_exceptions, "McpError") and hasattr(mcp_exceptions, "MCPError"):
        mcp_exceptions.McpError = mcp_exceptions.MCPError


def _version_from_pyproject() -> str:
    """
    Retrieve the version from the pyproject.toml file at the top of the repo.

    Returns:
        A string denoting the version from pyproject.toml.

    Raises:
        PackageNotFoundError: If the package name in the
            pyproject.toml is not 'mada'.
    """
    for parent in Path(__file__).resolve().parents:
        pyproject_path = parent / "pyproject.toml"
        if not pyproject_path.is_file():
            continue

        with pyproject_path.open("rb") as pyproject_file:
            project = tomllib.load(pyproject_file).get("project", {})

        if project.get("name") == PACKAGE_NAME:
            return project["version"]

    raise PackageNotFoundError(PACKAGE_NAME)


try:
    __version__ = _version_from_pyproject()
except PackageNotFoundError:
    __version__ = version(PACKAGE_NAME)


_install_mcp_compatibility_aliases()
