# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Shared formatting for orchestrator startup failures surfaced by interfaces.
"""


def format_startup_error_message(exc: BaseException) -> str:
    """
    Convert orchestrator startup failures into concise user-facing text.
    """
    details = str(exc).strip() or exc.__class__.__name__
    lowered = details.lower()

    if "connect" in lowered or "connection" in lowered or "cancellederror" in lowered:
        return (
            "MADA could not connect to one or more MCP servers. "
            "Check the MCP server processes and the URLs/commands in your config. "
            f"Details: {details}"
        )

    return f"MADA failed to initialize the configured agent team. Details: {details}"
