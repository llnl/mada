# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rich media values passed between orchestration, persistence, and interfaces."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LOG = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_IMAGE_DATA_URI_RE = re.compile(
    r"data:(image/(?:gif|jpeg|png|webp));base64,([^\s)]+)", re.IGNORECASE
)
_IMAGE_MARKDOWN_RE = re.compile(
    r"!\[[^\]]*\]\(\s*(data:image/(?:gif|jpeg|png|webp);base64,[^)]*)\s*\)",
    re.IGNORECASE,
)


def _matches_image_signature(data: bytes, media_type: str) -> bool:
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if media_type == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if media_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


@dataclass(frozen=True)
class ImageAttachment:
    """A validated image returned by an MCP tool."""

    data: bytes
    media_type: str
    filename: str

    @property
    def digest(self) -> str:
        """Return a stable digest used to deduplicate streamed tool results."""
        return hashlib.sha256(self.data).hexdigest()

    @classmethod
    def from_data(
        cls,
        data: bytes,
        media_type: str,
        filename: str | None = None,
    ) -> "ImageAttachment":
        """Validate raw image data and create an attachment."""
        normalized_type = media_type.lower().split(";", 1)[0].strip()
        extension = SUPPORTED_IMAGE_MEDIA_TYPES.get(normalized_type)
        if extension is None:
            raise ValueError(f"Unsupported image media type: {media_type}")
        if not data:
            raise ValueError("Image data is empty")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            )
        if not _matches_image_signature(data, normalized_type):
            raise ValueError(f"Image data does not match {normalized_type}")

        digest = hashlib.sha256(data).hexdigest()
        safe_filename = Path(filename).name if filename else None
        if safe_filename:
            safe_filename = "".join(
                "_" if character in "[]" or not character.isprintable() else character
                for character in safe_filename
            ).strip()
        return cls(
            data=bytes(data),
            media_type=normalized_type,
            filename=safe_filename or f"mcp-image-{digest[:12]}{extension}",
        )


class RichResponse(str):
    """A text response carrying image attachments for rich interfaces.

    This remains a ``str`` subclass so existing CLI, HTTP, and A2A consumers
    continue to behave as before.
    """

    images: tuple[ImageAttachment, ...]

    def __new__(
        cls, text: str, images: Iterable[ImageAttachment] = ()
    ) -> "RichResponse":
        value = str.__new__(cls, text)
        value.images = tuple(images)
        return value


def _decode_image_data(data: Any, media_type: Any) -> ImageAttachment | None:
    if not isinstance(media_type, str):
        return None
    media_type = media_type.lower().split(";", 1)[0].strip()
    if not media_type.startswith("image/"):
        return None

    try:
        if isinstance(data, bytes):
            decoded = data
        elif isinstance(data, str):
            encoded = data
            if encoded.startswith("data:"):
                header, separator, encoded = encoded.partition(",")
                if not separator or ";base64" not in header:
                    return None
                uri_media_type = header[5:].split(";", 1)[0]
                if uri_media_type:
                    media_type = uri_media_type
            decoded = base64.b64decode(encoded, validate=True)
        else:
            return None
        return ImageAttachment.from_data(decoded, media_type)
    except (binascii.Error, ValueError) as exc:
        LOG.warning("Ignoring invalid MCP image content: %s", exc)
        return None


def _json_candidates(value: str) -> list[str]:
    """Return possible JSON objects embedded in assistant text."""
    candidates = [value.strip()]
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        candidates.append(value[start : end + 1])
    return candidates


def _is_serialized_image_payload(value: Any) -> bool:
    """Return whether a decoded JSON value describes an image payload."""
    if not isinstance(value, dict):
        return False
    item_type = str(value.get("type") or "").lower()
    media_type = value.get("media_type") or value.get("mimeType")
    return item_type == "image" or (
        item_type in {"data", "uri"}
        and isinstance(media_type, str)
        and media_type.lower().startswith("image/")
    )


def extract_image_attachments(value: Any) -> list[ImageAttachment]:
    """Extract image attachments from nested MCP/Agent Framework content.

    Agent Framework represents an MCP ``ImageContent`` as a data ``Content``
    nested under a function-result item's ``items`` collection. Magentic wraps
    the same values in workflow event containers, so this traversal accepts the
    common mapping and object shapes used by both orchestration modes.
    """
    found: list[ImageAttachment] = []
    digests: set[str] = set()
    visited: set[int] = set()

    def add(attachment: ImageAttachment | None) -> None:
        if attachment is None or attachment.digest in digests:
            return
        digests.add(attachment.digest)
        found.append(attachment)

    def visit(item: Any) -> None:
        if item is None or isinstance(item, (int, float, bool, bytes)):
            return

        if isinstance(item, str):
            for match in _IMAGE_MARKDOWN_RE.finditer(item):
                media_type, encoded = match.group(1).split(";", 1)
                add(_decode_image_data(f"data:{media_type};{encoded}", media_type))
            for match in _IMAGE_DATA_URI_RE.finditer(item):
                add(_decode_image_data(match.group(2), match.group(1)))
            for candidate in _json_candidates(item):
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if parsed != item:
                    visit(parsed)
            return

        item_id = id(item)
        if item_id in visited:
            return
        visited.add(item_id)

        if isinstance(item, ImageAttachment):
            add(item)
            return

        if isinstance(item, (list, tuple, set)):
            for child in item:
                visit(child)
            return

        if isinstance(item, dict):
            item_type = str(item.get("type") or "").lower()
            media_type = item.get("media_type") or item.get("mimeType")
            if item_type == "image":
                add(_decode_image_data(item.get("data"), media_type))
            elif item_type in {"data", "uri"}:
                add(_decode_image_data(item.get("uri") or item.get("data"), media_type))

            for key in (
                "content",
                "contents",
                "data",
                "items",
                "messages",
                "output",
                "outputs",
                "result",
                "tool_result",
                "function_result",
            ):
                if key in item:
                    visit(item[key])
            return

        item_type = str(getattr(item, "type", "") or "").lower()
        media_type = getattr(item, "media_type", None) or getattr(
            item, "mimeType", None
        )
        if item_type == "image":
            add(_decode_image_data(getattr(item, "data", None), media_type))
        elif item_type in {"data", "uri"}:
            add(
                _decode_image_data(
                    getattr(item, "uri", None) or getattr(item, "data", None),
                    media_type,
                )
            )

        for attribute in (
            "content",
            "contents",
            "data",
            "items",
            "messages",
            "output",
            "outputs",
            "result",
            "tool_result",
            "function_result",
        ):
            if hasattr(item, attribute):
                visit(getattr(item, attribute, None))

    visit(value)
    return found


def strip_image_payload_text(value: str) -> str:
    """Remove serialized image payloads from assistant-visible text."""
    cleaned = _IMAGE_MARKDOWN_RE.sub("", value)
    cleaned = _IMAGE_DATA_URI_RE.sub("", cleaned)
    cleaned = re.sub(r"!\[[^\]]*\]\(\s*\)", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if _is_serialized_image_payload(parsed) or extract_image_attachments(candidate):
            cleaned = cleaned[:start] + cleaned[end + 1 :]
    return cleaned.strip()
