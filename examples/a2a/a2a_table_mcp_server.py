# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
CSV table-reader MCP server used by the LangChain A2A example agent.
This is separate from MADA as this only used for A2A examples.
"""

from __future__ import annotations

import argparse
import csv
from io import StringIO

from fastmcp import FastMCP


SAMPLE_CSV = """experiment,temperature_c,pressure_kpa
alpha,21.2,101.3
beta,24.8,99.8
gamma,19.6,103.1
delta,22.4,100.6
"""


def create_server() -> FastMCP:
    mcp = FastMCP(name="A2A Table Reader MCP Server")

    @mcp.tool()
    def read_sample_table(row_limit: int = 4) -> str:
        """
        Read a small built-in CSV table and return it as text.
        """
        rows = list(csv.DictReader(StringIO(SAMPLE_CSV)))
        limit = max(1, min(row_limit, len(rows)))
        selected_rows = rows[:limit]
        headers = list(rows[0].keys())

        widths = {
            header: max(len(header), *(len(row[header]) for row in selected_rows))
            for header in headers
        }
        lines = ["Sample experiment table loaded from built-in CSV:"]
        lines.append("  ".join(header.ljust(widths[header]) for header in headers))
        for row in selected_rows:
            lines.append(
                "  ".join(row[header].ljust(widths[header]) for header in headers)
            )
        return "\n".join(lines)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the A2A table-reader MCP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=9101, help="Port to bind")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="MCP transport",
    )
    args = parser.parse_args()

    server = create_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
        return

    server.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        stateless_http=True,
        uvicorn_config={"access_log": False},
    )


if __name__ == "__main__":
    main()
