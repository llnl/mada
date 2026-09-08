# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MADA - Multi-agent orchestration system for MADA workflows.

This package provides orchestration capabilities for coordinating multiple
autonomous agents that interact with MCP servers to execute complex workflows.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


PACKAGE_NAME = "mada"


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
