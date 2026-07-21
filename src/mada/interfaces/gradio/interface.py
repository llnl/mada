# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Multi-agent Gradio interface adapted from mada_multiagent.

This provides a configurable Gradio interface for multi-agent interactions
with MCP servers, adapted to work with MADA's configuration system.
"""

import base64
import json
import logging
from typing import Any, List

import gradio as gr

from mada.core.config import AgentConfig, InterfaceConfig
from mada.interfaces.gradio.mcp_client_wrapper import MCPGradioClientSession
from mada.interfaces.gradio.utils import create_agent_table, cycle_through_tools

LOG = logging.getLogger("mada-gradio")


class MADAMultiAgentGradioInterface:
    """
    Configurable multi-agent Gradio interface.

    This class provides a customizable UI for interacting with multiple agents via
    a Gradio front end. It supports configurable agent tables, connection panels,
    and chat interfaces.

    Attributes:
        interface_config: Interface configuration settings
        agents: List of agent configurations
        client: Client object for handling server connections and message processing
    """

    def __init__(
        self,
        interface_config: InterfaceConfig,
        agents: List[AgentConfig],
        client: MCPGradioClientSession,
    ):
        """
        Initialize the multi-agent Gradio interface.

        Args:
            interface_config: Interface configuration object
            agents: List of agent configurations
            client: Client responsible for managing server connections and message processing
        """
        self.interface_config = interface_config
        self.agents = agents
        self.client = client

        # In-situ Visualization
        self.in_situ_viz_config = getattr(self.interface_config, "in_situ_viz", None)
        self.in_situ_viz_tool = None
        self.in_situ_viz_task_id = None
        self.in_situ_viz_in_progress = False

    def create_custom_components(self, demo: gr.Blocks) -> None:
        """
        Hook for adding custom Gradio components to the interface.

        This method is called after the standard interface components
        have been created but before the chat interface is added.

        Args:
            demo: The Gradio Blocks container to add components into
        """
        pass

    def get_additional_chat_inputs(
        self, agent_table: gr.Dataframe
    ) -> List[gr.Component]:
        """
        Provide additional input components for the chat interface.

        Args:
            agent_table: The agent configuration table component

        Returns:
            List of Gradio components to include as additional inputs
        """
        return [agent_table]

    def create_accordion(self) -> gr.Accordion:
        """
        Create a collapsible accordion component for MCP server connection settings.

        Returns:
            The configured Gradio Accordion component
        """
        return gr.Accordion("MCP Server Connection", open=True)

    def create_chat_interface(self, agent_table: gr.Dataframe) -> gr.Chatbot:
        """
        Create the chat message and input components.

        Args:
            agent_table: The agent configuration table to include as input

        Returns:
            The configured Gradio chatbot component
        """
        placeholder = "Describe your workflow or ask for help..."
        if self.interface_config and hasattr(self.interface_config, "chat_placeholder"):
            placeholder = self.interface_config.chat_placeholder

        chatbot = gr.Chatbot(
            label="Chat",
            height=480,
            elem_id="mada-chatbot",
        )

        message_box = gr.Textbox(
            label="",
            placeholder=placeholder,
            elem_id="mada-message-box",
        )
        message_box.submit(
            fn=self._send_message,
            inputs=[
                message_box,
                chatbot,
                *self.get_additional_chat_inputs(agent_table),
            ],
            outputs=[message_box, chatbot],
            show_progress="hidden",
        )

        return chatbot

    async def _send_message(
        self,
        message: str,
        history: List[Any],
        *additional_inputs: Any,
    ):
        """
        Add the user message and stream the assistant response.
        """
        if not message or not message.strip():
            yield gr.skip(), history
            return

        updated_history = list(history or [])
        updated_history.append({"role": "user", "content": message})
        yield "", updated_history

        updated_history.append({"role": "assistant", "content": ""})
        async for response in self.client.process_message(
            message, updated_history, *additional_inputs
        ):
            updated_history[-1] = {
                "role": "assistant",
                "content": response,
            }
            yield gr.skip(), updated_history

    def create_interface(self) -> gr.Blocks:
        """
        Construct and return the complete Gradio interface for the multi-agent system.

        Returns:
            The assembled Gradio Blocks container representing the full interface
        """
        title = "MADA Multi-Agent Orchestrator"
        description = "Coordinate multiple agents for complex engineering workflows"

        if self.interface_config:
            title = getattr(self.interface_config, "title", title)
            description = getattr(self.interface_config, "description", description)

        with gr.Blocks(fill_height=True) as demo:
            # Title and description
            gr.Markdown(f"# {title}")
            gr.Markdown(description)

            with gr.Row():
                # Left sidebar for sessions
                with gr.Column(scale=1, min_width=220):
                    gr.Markdown("### Chats")

                    new_chat_btn = gr.Button("➕ New chat", variant="secondary")
                    session_list = gr.Radio(
                        choices=self.client.get_session_choices(),
                        label="Sessions",
                        interactive=True,
                    )
                    delete_chat_btn = gr.Button(
                        "🗑️ Delete selected chat", variant="stop"
                    )
                    delete_all_btn = gr.Button("🗑️ Delete ALL chats", variant="stop")

                    # Confirmation panel for delete all (initially hidden)
                    with gr.Group(visible=False) as delete_all_confirm:
                        gr.Markdown("⚠️ **WARNING**")
                        gr.Markdown(
                            "This will permanently delete ALL chat sessions and cannot be undone!"
                        )
                        with gr.Row():
                            confirm_delete_all_btn = gr.Button(
                                "Yes, delete all", variant="stop", size="sm"
                            )
                            cancel_delete_all_btn = gr.Button(
                                "Cancel", variant="secondary", size="sm"
                            )

                # Right main column for tools + chat
                with gr.Column(scale=4):
                    # MCP Server connection section
                    with self.create_accordion():
                        agent_table = create_agent_table(self.agents)

                        connect_button = gr.Button(
                            "Connect to MCP Servers", variant="primary"
                        )

                        connect_button.click(
                            fn=self.client.connect_servers,
                            inputs=[agent_table],
                            outputs=[connect_button, agent_table],
                            show_progress="full",
                        )

                    if self.in_situ_viz_config is not None:
                        sim_server = self.in_situ_viz_config["sim_server"]
                        update_time = self.in_situ_viz_config[
                            "update_time"
                        ]  # in seconds

                        gr.Markdown(
                            f"### Simulation Visualization: Updates every {update_time} seconds."
                        )

                        self.sim_status = gr.Markdown(
                            value=f"Using {sim_server} for in-situ visualization..."
                        )

                        self.sim_image = gr.HTML(label="Latest Simulation Frame")

                        self.refresh_timer = gr.Timer(value=update_time)

                        self.refresh_timer.tick(
                            fn=self.in_situ_viz,
                            inputs=[],
                            outputs=[self.sim_image, self.sim_status],
                            show_progress="hidden",
                            concurrency_limit=1,
                        )

                    # Custom components (subclasses can add their own)
                    self.create_custom_components(demo)

                    task_status = gr.Markdown(
                        "### Task Status\nNo background tasks yet."
                    )
                    task_refresh = gr.Timer(value=0.5)

                    # Chat interface
                    chatbot = self.create_chat_interface(agent_table)

            task_refresh.tick(
                fn=self.client.refresh_chat_and_task_status,
                inputs=[chatbot],
                outputs=[chatbot, task_status],
                show_progress="hidden",
                concurrency_limit=1,
            )

            # Update session list after connect, to be sure DB exists
            connect_button.click(
                fn=self.client.update_session_choices,
                inputs=None,
                outputs=[session_list],
            )

            # New chat button -> create new session, update session list, select it, clear history
            new_chat_btn.click(
                fn=self.client.create_new_session,
                inputs=None,
                outputs=[session_list, chatbot],  # set choices, value, history
            )

            # Selecting a session -> load history into chatbot
            session_list.change(
                fn=self.client.select_session, inputs=session_list, outputs=chatbot
            )

            # Deleting a session
            delete_chat_btn.click(
                fn=self.client.delete_session,
                inputs=session_list,
                outputs=[session_list, chatbot],
            )

            # Show confirmation panel when "Delete ALL chats" is clicked
            delete_all_btn.click(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=delete_all_confirm,
            )

            # Actually delete all sessions when confirmed
            confirm_delete_all_btn.click(
                fn=self.client.delete_all_sessions,
                inputs=None,
                outputs=[session_list, chatbot],
            ).then(
                # Hide confirmation panel after deletion
                fn=lambda: gr.update(visible=False),
                inputs=None,
                outputs=delete_all_confirm,
            )

            # Hide confirmation panel when cancelled
            cancel_delete_all_btn.click(
                fn=lambda: gr.update(visible=False),
                inputs=None,
                outputs=delete_all_confirm,
            )

        return demo

    async def in_situ_viz(self) -> tuple[Any, str]:
        """
        Fetch and render the latest in situ visualization from the configured MCP server.

        The `config.json` `interface` section must define an `in_situ_viz` entry in
        the following format:

            "in_situ_viz": {
                "sim_server": "my_sim_server",
                "update_time": 60
            }

        The configured `sim_server` must expose an MCP tool named `in_situ_viz` that
        returns a dictionary with the following structure:

            {
                "gif_path": gif_out,  # Path to the GIF file
                "status": "Status string.",  # Simulation status
                "step": step,  # Simulation time step
                "message": f"Rendered t={times[-1]:.6f}",  # Simulation time message
                "updated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),  # Last update timestamp
            }

        Returns:
            A 2-tuple containing the visualization output and a status message.

            - The first value is either an HTML string for display or `gr.skip()`
            if no image update should occur.
            - The second value is a status string describing the latest tool
            response, or an error message if the request fails.
        """
        # Skip stale queued updates if a previous visualization request is still running
        if self.in_situ_viz_in_progress:
            return gr.skip(), gr.skip()

        self.in_situ_viz_in_progress = True
        sim_server = self.in_situ_viz_config["sim_server"]

        try:
            # Check if the mcp server has in_situ_viz tool
            if self.in_situ_viz_tool is None:
                self.in_situ_viz_tool = cycle_through_tools(
                    self.client.orchestrator.specialist_agents,
                    sim_server,
                    return_tool=True,
                )

                if self.in_situ_viz_tool is None:
                    return (
                        gr.skip(),
                        f"Could not find tool 'in_situ_viz' for server '{sim_server}'.",
                    )

            # If the last render started a background task, poll that task.
            if self.in_situ_viz_task_id:
                result_tool = self._find_sim_server_tool(
                    sim_server, "get_background_task_result"
                )

                if result_tool is None:
                    return (
                        gr.skip(),
                        f"{self.in_situ_viz_task_id}: Waiting for render result.",
                    )

                try:
                    payload = await result_tool.invoke(task_id=self.in_situ_viz_task_id)
                except TypeError:
                    payload = await result_tool.invoke(
                        {"task_id": self.in_situ_viz_task_id}
                    )

                result = self._tool_payload_to_dict(payload)

                if result.get("status") == "completed" and "result" in result:
                    result = result["result"]
                    if isinstance(result, str):
                        result = json.loads(result)

            # Otherwise, start a new render.
            else:
                try:
                    payload = await self.in_situ_viz_tool.invoke()
                    result = self._tool_payload_to_dict(payload)
                except Exception as exc:
                    task_id = self._task_id_from_error(exc)
                    if task_id:
                        self.in_situ_viz_task_id = task_id
                        return gr.skip(), f"{task_id}: Visualization render is running."
                    raise

            if result.get("status") == "running" and result.get("task_id"):
                self.in_situ_viz_task_id = result["task_id"]
                message = result.get("message", "Visualization render is running.")
                return gr.skip(), f"{self.in_situ_viz_task_id}: {message}"

            self.in_situ_viz_task_id = None

            gif_path = result.get("gif_path")
            status = result.get("status", "unknown")
            step = result.get("step", "n/a")
            message = result.get("message", "")
            updated_at = result.get("updated_at", "")

            status_text = f"GIF Path: {gif_path} | Updated: {updated_at} | Status: {status} | Step: {step} | {message}"

            if not gif_path:
                return gr.skip(), status_text

            # Convert gif to base64 since gradio doesn't allow paths from outside app directory
            with open(gif_path, "rb") as f:
                gif_b64 = base64.b64encode(f.read()).decode("utf-8")

            html = f"""
                <div style="display:flex; justify-content:center;">
                    <img src="data:image/gif;base64,{gif_b64}" style="max-height:300px; max-width:100%;" />
                </div>
                """

            return html, status_text

        except Exception as exc:
            LOG.exception("Failed to fetch latest simulation frame")
            return gr.skip(), f"Error fetching frame: {exc}"
        finally:
            self.in_situ_viz_in_progress = False

    def _find_sim_server_tool(self, sim_server: str, tool_name: str) -> Any:
        for agent in self.client.orchestrator.specialist_agents:
            for mcp_tool in getattr(agent, "mcp_tools", []):
                if getattr(mcp_tool, "name", None) != sim_server:
                    continue
                for fn in getattr(mcp_tool, "_functions", []):
                    if getattr(fn, "name", None) == tool_name:
                        return fn
        return None

    @staticmethod
    def _tool_payload_to_dict(payload: Any) -> dict:
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if getattr(payload, "structured_content", None) is not None:
            payload = payload.structured_content
        elif hasattr(payload, "text"):
            payload = payload.text
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected tool payload to be a dict, got {type(payload)}")
        return payload

    @staticmethod
    def _task_id_from_error(exc: Exception) -> str | None:
        message = str(exc)
        marker = '"task_id": "'
        if marker not in message:
            return None

        return message.split(marker, 1)[1].split('"', 1)[0]
