# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stdio MCP server for returning approved image files as ImageContent."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from fastmcp import FastMCP
from mcp.types import ImageContent

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_TYPES = {
    ".gif": ("image/gif", (b"GIF87a", b"GIF89a")),
    ".jpeg": ("image/jpeg", (b"\xff\xd8\xff",)),
    ".jpg": ("image/jpeg", (b"\xff\xd8\xff",)),
    ".png": ("image/png", (b"\x89PNG\r\n\x1a\n",)),
    ".webp": ("image/webp", (b"RIFF",)),
}

mcp = FastMCP("local-images")


def _allowed_roots() -> list[Path]:
    configured = os.environ.get("MADA_IMAGE_ROOTS", "")
    roots = [
        Path(item).expanduser().resolve()
        for item in configured.split(os.pathsep)
        if item
    ]
    return roots or [Path.cwd().resolve()]


def _load_image(path: str) -> tuple[bytes, str, str]:
    image_path = Path(path).expanduser().resolve()
    if not any(
        root == image_path or root in image_path.parents for root in _allowed_roots()
    ):
        raise ValueError("Image path is outside the configured MADA_IMAGE_ROOTS")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
        )

    media_type, signatures = SUPPORTED_TYPES.get(image_path.suffix.lower(), (None, ()))
    if media_type is None:
        media_type = mimetypes.guess_type(image_path.name)[0] or ""
    data = image_path.read_bytes()
    if media_type not in {item[0] for item in SUPPORTED_TYPES.values()}:
        raise ValueError(f"Unsupported image type: {image_path.suffix}")
    if signatures and not any(data.startswith(signature) for signature in signatures):
        raise ValueError("Image data does not match its file type")
    return data, media_type, image_path.name


@mcp.tool()
def read_image(path: str) -> ImageContent:
    """Read an approved local image and return native MCP ImageContent."""
    data, media_type, _filename = _load_image(path)
    return ImageContent(
        data=base64.b64encode(data).decode("ascii"), mimeType=media_type
    )


def main() -> None:
    """Run the server over stdio for use as an MCP child process."""
    mcp.run(transport="stdio", show_banner=False, stateless=True)


if __name__ == "__main__":
    main()
