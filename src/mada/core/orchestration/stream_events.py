# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Internal stream-control events shared by orchestration strategies and interfaces.

Control sentinels are empty strings so existing text streams can pass them
without emitting content. Activity and image signals include visible fallback
text. Metadata attributes carry instructions to protocol-aware consumers.
"""

from typing import Any

from mada.core.media import ImageAttachment


TOOL_CALL_NAME_ATTR = "_mada_tool_call_name"
RESPONSE_REPLACEMENT_ATTR = "_mada_response_replacement"
ERROR_MESSAGE_ATTR = "_mada_error_message"
IMAGE_ATTACHMENT_ATTR = "_mada_image_attachment"
ACTIVITY_ATTR = "_mada_activity"


class InternalActivitySignal(str):
    """Visible activity text kept separately from the replaceable final answer."""

    def __new__(cls, text: str):
        value = str.__new__(cls, text)
        setattr(value, ACTIVITY_ATTR, True)
        return value


def is_activity(chunk: Any) -> bool:
    return bool(getattr(chunk, ACTIVITY_ATTR, False))


class InternalToolCallSignal(str):
    """
    Empty stream chunk that carries background-detach metadata.
    """

    def __new__(cls, tool_call_name: str):
        value = str.__new__(cls, "")
        setattr(value, TOOL_CALL_NAME_ATTR, tool_call_name)
        return value


class InternalResponseReplacement(str):
    """
    Empty stream chunk that replaces previously collected response text.
    """

    def __new__(cls, replacement: str):
        value = str.__new__(cls, "")
        setattr(value, RESPONSE_REPLACEMENT_ATTR, replacement)
        return value


class InternalError(str):
    """
    Empty stream chunk that signals an internal terminal error condition.
    """

    def __new__(cls, error_message: str):
        value = str.__new__(cls, "")
        setattr(value, ERROR_MESSAGE_ATTR, error_message)
        return value


class InternalImageSignal(str):
    """Stream chunk carrying an MCP image and a safe text fallback."""

    def __new__(cls, attachment: ImageAttachment):
        value = str.__new__(cls, f"[Image attachment: {attachment.filename}]")
        setattr(value, IMAGE_ATTACHMENT_ATTR, attachment)
        return value


def tool_call_name(chunk: Any) -> str | None:
    """
    Return the internal tool-call name carried by a stream chunk, if present.
    """
    value = getattr(chunk, TOOL_CALL_NAME_ATTR, None)
    return None if value is None else str(value)


def response_replacement(chunk: Any) -> str | None:
    """
    Return replacement text carried by a stream chunk, if present.
    """
    value = getattr(chunk, RESPONSE_REPLACEMENT_ATTR, None)
    return None if value is None else str(value)


def error_message(chunk: Any) -> str | None:
    """
    Return terminal error text carried by a stream chunk, if present.
    """
    value = getattr(chunk, ERROR_MESSAGE_ATTR, None)
    return None if value is None else str(value)


def image_attachment(chunk: Any) -> ImageAttachment | None:
    """Return the image attachment carried by a stream chunk, if present."""
    value = getattr(chunk, IMAGE_ATTACHMENT_ATTR, None)
    return value if isinstance(value, ImageAttachment) else None


def apply_text_control(chunks: list[str], chunk: Any) -> tuple[bool, bool]:
    """
    Apply replacement/error controls to a text accumulator.

    Returns:
        A pair of booleans: whether the chunk was a control chunk, and whether
        collection should stop because the control chunk was terminal.
    """
    replacement = response_replacement(chunk)
    if replacement is not None:
        chunks[:] = [replacement]
        return True, False

    message = error_message(chunk)
    if message is not None:
        chunks[:] = [message]
        return True, True

    return False, False
