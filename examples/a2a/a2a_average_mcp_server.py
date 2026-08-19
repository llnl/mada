# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
CSV column-average MCP server used by the Google ADK A2A example agent.
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
    mcp = FastMCP(name="A2A Column Average MCP Server")

    @mcp.tool()
    def calculate_column_averages(columns: str = "all") -> str:
        """
        Calculate averages for numeric columns in a built-in CSV table.
        """
        rows = list(csv.DictReader(StringIO(SAMPLE_CSV)))
        numeric_columns = []
        for column in rows[0]:
            try:
                for row in rows:
                    float(row[column])
            except ValueError:
                continue
            numeric_columns.append(column)

        if columns.strip().lower() != "all":
            requested = [
                column.strip()
                for column in columns.split(",")
                if column.strip() in numeric_columns
            ]
            if not requested:
                return (
                    "No requested numeric columns were found. "
                    f"Available numeric columns: {', '.join(numeric_columns)}."
                )
            numeric_columns = requested

        lines = ["Column averages from the sample experiment table:"]
        for column in numeric_columns:
            values = [float(row[column]) for row in rows]
            value = sum(values) / len(values)
            lines.append(f"- {column}: {value:.2f}")
        return "\n".join(lines)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the A2A column-average MCP server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=9102, help="Port to bind")
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
