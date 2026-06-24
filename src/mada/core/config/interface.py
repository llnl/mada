# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
User interface configuration definitions.

This module defines configuration objects for customizing the multi-agent
interface layout and behavior. It currently provides `InterfaceConfig`, which
stores display text, server settings, accordion defaults, and additional
component-specific keyword arguments for interface customization.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict


LOG = logging.getLogger("mada-interface")


@dataclass
class InterfaceConfig:
    """
    Configuration settings for the multi-agent interface layout.

    This class defines how the Gradio interface is visually presented,
    including titles, descriptions, layout settings, and optional UI
    customization parameters.

    Attributes:
        title (str): The title displayed at the top of the interface.
        description (str): A brief description shown below the title.
        chat_placeholder (str): Placeholder text in the chat input field.
        port (int): Port to run the Gradio interface on.
        share (bool): Whether to create a public sharing link.
        connection_accordion_open (bool): Whether the connection accordion is open by default.
        connection_accordion_label (str): The label for the connection accordion section.
        accordion_kwargs (Dict[str, Any]): Additional keyword arguments for customizing the accordion component.
        dataframe_kwargs (Dict[str, Any]): Additional keyword arguments for customizing the agents DataFrame.
        chat_interface_kwargs (Dict[str, Any]): Additional keyword arguments for customizing the chat interface.
    """

    title: str
    description: str
    chat_placeholder: str = "Type your message here"
    port: int = 7860
    share: bool = False
    connection_accordion_open: bool = False
    connection_accordion_label: str = "Connect to MCP Servers"
    in_situ_viz: dict = None

    # Component-specific kwargs for advanced customization
    accordion_kwargs: Dict[str, Any] = field(default_factory=dict)
    dataframe_kwargs: Dict[str, Any] = field(default_factory=dict)
    chat_interface_kwargs: Dict[str, Any] = field(default_factory=dict)
