# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MCP server configuration definitions.

This module defines configuration objects used to describe individual MCP
servers. It currently provides `MCPServerConfig`, a dataclass for specifying
transport type, connection details, launch command information, descriptive
metadata, and the Python executable used for stdio-based server startup.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional


LOG = logging.getLogger("mada-interface")


@dataclass
class MCPServerConfig:
    """
    Configuration for an individual MCP server.

    Attributes:
        transport (str): Transport method ('streamable-http' or 'stdio')
        url (Optional[str]): URL for streamable-http transport
        command (Optional[str]): Command to launch server for stdio transport
        description (Optional[str]): Human-readable description of the server
        python_executable (str): Path to Python executable (used for stdio transport).
    """

    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    description: Optional[str] = ""
    python_executable: str = sys.executable
