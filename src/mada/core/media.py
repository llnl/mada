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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

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
    ) -> ImageAttachment:
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

    def __new__(cls, text: str, images: Iterable[ImageAttachment] = ()) -> Self:
        value = str.__new__(cls, text)
        value.images = tuple(images)
        return value


def _decode_image_data(data: Any, media_type: Any) -> ImageAttachment | None:
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
                if isinstance(uri_media_type, str) and uri_media_type:
                    media_type = uri_media_type
            decoded = base64.b64decode(encoded, validate=True)
        else:
            return None
        if not isinstance(media_type, str):
            return None
        media_type = media_type.lower().split(";", 1)[0].strip()
        if not media_type.startswith("image/"):
            return None
        return ImageAttachment.from_data(decoded, media_type)
    except (binascii.Error, ValueError) as exc:
        LOG.warning("Ignoring invalid MCP image content: %s", exc)
        return None


def _json_candidates(value: str) -> list[str]:
    """Return possible JSON values embedded in assistant text."""
    candidates = [value.strip()]
    decoder = json.JSONDecoder()
    for start, character in enumerate(value):
        if character not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(value, start)
        except json.JSONDecodeError:
            continue
        candidate = value[start:end]
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _is_serialized_image_payload(value: Any) -> bool:
    """Return whether a decoded JSON value describes an image payload."""
    if not isinstance(value, dict):
        return False
    item_type = str(value.get("type") or "").lower()
    media_type = value.get("media_type") or value.get("mimeType")
    uri = value.get("uri") or value.get("data")
    has_image_data_uri = (
        isinstance(uri, str)
        and uri.lower().startswith("data:image/")
        and ";base64," in uri.lower()
    )
    return item_type == "image" or (
        item_type in {"data", "uri"}
        and (
            has_image_data_uri
            or (isinstance(media_type, str) and media_type.lower().startswith("image/"))
        )
    )


def _contains_serialized_image_payload(value: Any) -> bool:
    """Return whether decoded JSON contains an image payload descriptor."""
    if _is_serialized_image_payload(value):
        return True
    if isinstance(value, dict):
        return any(_contains_serialized_image_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_serialized_image_payload(item) for item in value)
    if isinstance(value, str):
        for candidate in _json_candidates(value):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if parsed != value and _contains_serialized_image_payload(parsed):
                return True
    return False


def _serialized_non_image_text(value: Any) -> str:
    """Extract sanitized user-visible text from serialized mixed content."""
    if _is_serialized_image_payload(value):
        return ""
    if isinstance(value, str):
        return strip_image_payload_text(value, strip_outer_whitespace=False)
    if isinstance(value, list):
        text_parts = []
        for item in value:
            text = _serialized_non_image_text(item)
            if text:
                text_parts.append(text)
        return "".join(text_parts)
    if not isinstance(value, dict):
        return ""

    item_type = str(value.get("type") or "").lower()
    if item_type == "text":
        text = value.get("text", value.get("content", ""))
        return (
            strip_image_payload_text(str(text), strip_outer_whitespace=False)
            if text is not None
            else ""
        )
    if isinstance(value.get("text"), str) and value["text"].strip():
        return strip_image_payload_text(value["text"], strip_outer_whitespace=False)

    for key in (
        "content",
        "contents",
        "items",
        "messages",
        "output",
        "outputs",
        "result",
        "tool_result",
        "function_result",
    ):
        if key in value:
            text = _serialized_non_image_text(value[key])
            if text:
                return text

    cleaned = _remove_serialized_image_payloads(value)
    if cleaned in (None, {}, []):
        return ""
    return json.dumps(cleaned, indent=2, default=str)


def _remove_serialized_image_payloads(value: Any) -> Any:
    """Remove image descriptors while preserving arbitrary JSON fields."""
    if _is_serialized_image_payload(value):
        return None
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _remove_serialized_image_payloads(item)) is not None
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _remove_serialized_image_payloads(item)) is not None
        ]
    if isinstance(value, str):
        return strip_image_payload_text(value, strip_outer_whitespace=False)
    return value


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
                data_uri = match.group(1)
                media_type = data_uri[5:].split(";", 1)[0]
                add(_decode_image_data(data_uri, media_type))
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
            # Workflow output events contain cumulative user-visible text.
            # Do not interpret image-looking text there as a tool attachment;
            # the response sanitizer handles it when the text is finalized.
            if item_type not in {"output", "output_text"} and "text" in item:
                visit(item["text"])
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
        if item_type not in {"output", "output_text"} and hasattr(item, "text"):
            visit(getattr(item, "text", None))

    visit(value)
    return found


def strip_image_payload_text(value: str, *, strip_outer_whitespace: bool = True) -> str:
    """Remove serialized image payloads from assistant-visible text."""
    cleaned = _IMAGE_MARKDOWN_RE.sub("", value)
    cleaned = _IMAGE_DATA_URI_RE.sub("", cleaned)
    cleaned = re.sub(r"!\[[^\]]*\]\(\s*\)", "", cleaned)

    decoder = json.JSONDecoder()
    spans: list[tuple[int, int]] = []
    for start, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            parsed, end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            continue
        if _contains_serialized_image_payload(parsed) or extract_image_attachments(
            parsed
        ):
            spans.append((start, end))

    outermost_spans = [
        (start, end)
        for start, end in spans
        if not any(
            other_start <= start
            and end <= other_end
            and (other_start, other_end) != (start, end)
            for other_start, other_end in spans
        )
    ]
    for start, end in sorted(outermost_spans, reverse=True):
        replacement = ""
        try:
            parsed, _ = decoder.raw_decode(cleaned, start)
            replacement = _serialized_non_image_text(parsed)
        except json.JSONDecodeError:
            pass
        cleaned = cleaned[:start] + replacement + cleaned[end:]
    return cleaned.strip() if strip_outer_whitespace else cleaned
