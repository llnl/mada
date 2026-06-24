# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from typing import Any, List

import gradio as gr

from mada.core.config import AgentConfig


def create_agent_table(
    agents: List[AgentConfig],
    agent_dict: dict | None = None,
) -> gr.Dataframe:
    """
    Create a Gradio Dataframe component for agent configuration.

    Args:
        agents: List of agent configurations
        agent_dict: An optional mapping from agent name to MCP server tool data
            produced by `cycle_through_tools` and is used to populate the MCP
            Server Tools column.

    Returns:
        The configured Gradio Dataframe component representing agents
    """

    # Headers for the agent table
    headers = [
        "Agent Name",
        "Description",
        "Domain",
        "MCP Servers",
        "MCP Server Tools",
        "Instructions",
    ]

    # Convert agents to table rows
    table_rows = []
    for agent in agents:
        row = [
            agent.agent_name,
            agent.description,
            getattr(agent, "domain", ""),
            ", ".join(agent.mcp_servers) if agent.mcp_servers else "",
            "Need to connect first..."
            if agent_dict is None
            else str(agent_dict[agent.agent_name]),
            agent.instructions,
        ]
        table_rows.append(row)

    return gr.Dataframe(
        headers=headers,
        datatype=["str"] * len(headers),
        row_count=len(table_rows),
        column_count=len(headers),
        label="Agents",
        value=table_rows,
        interactive=False,
        elem_id="my_df",
        column_widths=[f"{math.floor(100 / len(row))}%"] * len(row),
    )


def cycle_through_tools(
    agents: List[AgentConfig],
    sim_server: str | None = None,
    return_tool: bool = False,
) -> Any:
    """
    Build a mapping of agents to MCP servers and their available tool names.

    Iterates through each agent, inspects its configured MCP tools, and
    collects the function names exposed by each MCP server. Optionally,
    the function can search for and return a specific tool function named
    `in_situ_viz` from the specified simulation server.

    Args:
        agents: List of agent configurations
        sim_server: The name of the MCP server to inspect when `return_tool`
            is `True`. MCP servers with different names are skipped in that case.
        return_tool: If `False`, return a nested dictionary mapping agent
            names to MCP server names and tool names. If `True`, search for
            a tool function named `in_situ_viz` and return that function object
            if found.

    Returns:
        If `return_tool` is `False`, returns a dictionary of the form:

            {
                agent_name: {
                    mcp_server_name: [tool_name, ...]
                }
            }

        If `return_tool` is `True`, returns the function object whose name is
        `in_situ_viz` from the specified `sim_server`, or `None` if no such
        function is found.
    """
    agent_dict = {}
    for agent in agents:
        agent_name = getattr(agent, "name", None)
        agent_dict[agent_name] = {}

        for mcp_tool in getattr(agent, "mcp_tools", []):
            mcp_server_name = getattr(mcp_tool, "name", None)
            agent_dict[agent_name][mcp_server_name] = []
            if return_tool and mcp_server_name != sim_server:
                continue

            for fn in getattr(mcp_tool, "_functions", []):
                mcp_tool_name = getattr(fn, "name", None)
                agent_dict[agent_name][mcp_server_name].append(mcp_tool_name)
                if return_tool and mcp_tool_name == "in_situ_viz":
                    return fn

    if return_tool:
        return None
    else:
        return agent_dict
