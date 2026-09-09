# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Shared helpers for manifest-based skills.
"""

from typing import List


def is_tool_allowed(allowed_tools: List[str], tool_name: str) -> bool:
    """
    Return True when a manifest allowlist permits the named runtime tool.

    An empty or absent allowlist permits every tool.

    Args:
        allowed_tools: Tool names permitted by a skill manifest.
        tool_name: Runtime tool name to check.

    Returns:
        True when the tool is permitted.
    """
    if not allowed_tools:
        return True
    return tool_name in allowed_tools
