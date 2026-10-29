# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stdio MCP server for approved images and read-only Vertex-CFD artifacts."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from fastmcp import FastMCP
from mcp.types import ImageContent

from mada.core.media import ImageAttachment

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

mcp = FastMCP("local-images")


def _allowed_roots() -> list[Path]:
    configured = os.environ.get("MADA_IMAGE_ROOTS", "")
    config_dir = (
        Path(os.environ.get("MADA_CONFIG_DIR", str(Path.cwd()))).expanduser().resolve()
    )
    roots = [
        (
            Path(item).expanduser()
            if Path(item).expanduser().is_absolute()
            else config_dir / Path(item).expanduser()
        ).resolve()
        for item in configured.split(os.pathsep)
        if item
    ]
    return roots


def _approved_path(path: str, *, directory: bool = False) -> Path:
    requested_path = Path(path).expanduser()
    roots = _allowed_roots()
    if requested_path.is_absolute():
        candidates = [requested_path.resolve()]
    else:
        # Relative artifact paths are relative to an approved root, rather
        # than to the MCP server's process CWD.
        candidates = [(root / requested_path).resolve() for root in roots]

    allowed_candidates = [
        candidate
        for candidate in candidates
        if any(root == candidate or root in candidate.parents for root in roots)
    ]
    if not allowed_candidates:
        raise ValueError("Image path is outside the configured MADA_IMAGE_ROOTS")
    selected = next(
        (
            candidate
            for candidate in allowed_candidates
            if (candidate.is_dir() if directory else candidate.is_file())
        ),
        allowed_candidates[0],
    )
    if not (selected.is_dir() if directory else selected.is_file()):
        raise FileNotFoundError(selected)
    return selected


def _load_image(path: str) -> tuple[bytes, str, str]:
    image_path = _approved_path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
        )

    media_type = SUPPORTED_TYPES.get(image_path.suffix.lower())
    if media_type is None:
        media_type = mimetypes.guess_type(image_path.name)[0] or ""
    data = image_path.read_bytes()
    if media_type not in set(SUPPORTED_TYPES.values()):
        raise ValueError(f"Unsupported image type: {image_path.suffix}")
    ImageAttachment.from_data(data, media_type, filename=image_path.name)
    return data, media_type, image_path.name


@mcp.tool()
def check_vertex_artifact_support() -> dict:
    """Read-only preflight for the artifact reader, its roots and dependencies."""
    try:
        # Exercise the decoder, not just module discovery or a console script.
        import io

        import numpy
        from scipy.io import netcdf_file

        buffer = io.BytesIO()
        writer = netcdf_file(buffer, "w", version=2)
        writer.createDimension("n", 1)
        writer.createVariable("probe", "d", ("n",))[:] = [3.0]
        writer.flush()
        payload = buffer.getvalue()
        writer.close()
        with netcdf_file(io.BytesIO(payload), "r", mmap=False) as reader:
            usable = float(reader.variables["probe"][0]) == 3.0
        return {
            "velocity_reader_available": usable,
            "approved_roots_available": bool(_allowed_roots())
            and all(root.is_dir() for root in _allowed_roots()),
            "numpy_version": numpy.__version__,
        }
    except (ImportError, OSError, ValueError, TypeError) as error:
        return {
            "velocity_reader_available": False,
            "error": str(error),
            "remedy": "Install mada[vertex-tools] in the MCP server environment",
        }


@mcp.tool()
def read_vertex_campaign(output_dir: str) -> dict:
    """Read an approved Vertex-CFD campaign without executing simulations.

    After generation, returns actual sampled parameters and manifest_runs for
    the Flux handoff. After completion/post-processing, also returns plot paths,
    log tails, and measured final center-probe velocity from classic NetCDF
    Exodus files. Values include evaluation times; different final times are not
    automatically comparable steady states. Images can be displayed with
    read_image using the returned paths. Uses the same MADA_IMAGE_ROOTS as images.
    """
    from mada.examples.vertex_campaign_artifacts import read_campaign

    return read_campaign(_approved_path(output_dir, directory=True))


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
