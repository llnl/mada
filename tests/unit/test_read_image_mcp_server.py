import pytest

pytest.importorskip("fastmcp")

from mada.examples import read_image_mcp_server as image_server


PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


def test_read_image_returns_mcp_image_content(tmp_path, monkeypatch):
    image_path = tmp_path / "plot.png"
    image_path.write_bytes(PNG_DATA)
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path))

    content = image_server.read_image(str(image_path))

    assert content.type == "image"
    assert content.data


def test_read_image_rejects_paths_outside_allowed_roots(tmp_path, monkeypatch):
    image_path = tmp_path / "plot.png"
    image_path.write_bytes(PNG_DATA)
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path / "allowed"))

    with pytest.raises(ValueError, match="outside"):
        image_server.read_image(str(image_path))


def test_read_image_rejects_invalid_webp_signature(tmp_path, monkeypatch):
    image_path = tmp_path / "plot.webp"
    image_path.write_bytes(b"RIFF" + b"\x00" * 4 + b"NOPE")
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path))

    with pytest.raises(ValueError, match="does not match"):
        image_server.read_image(str(image_path))
