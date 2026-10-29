import pytest

pytest.importorskip("fastmcp")

from mada.examples import read_image_mcp_server as image_server


PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


def test_artifact_health_check_exercises_netcdf_reader(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path))
    result = image_server.check_vertex_artifact_support()
    assert result["velocity_reader_available"] is True
    assert result["approved_roots_available"] is True


def test_artifact_health_check_reports_missing_scipy(monkeypatch):
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "scipy.io":
            raise ImportError("scipy unavailable in server environment")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    result = image_server.check_vertex_artifact_support()
    assert result["velocity_reader_available"] is False
    assert "vertex-tools" in result["remedy"]


def test_read_image_returns_mcp_image_content(tmp_path, monkeypatch):
    image_path = tmp_path / "plot.png"
    image_path.write_bytes(PNG_DATA)
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path))

    content = image_server.read_image(str(image_path))

    assert content.type == "image"
    assert content.data


def test_read_image_resolves_relative_paths_against_allowed_roots(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "plots" / "frame.png"
    image_path.parent.mkdir()
    image_path.write_bytes(PNG_DATA)
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)

    content = image_server.read_image("plots/frame.png")

    assert content.type == "image"


def test_read_image_rejects_paths_outside_allowed_roots(tmp_path, monkeypatch):
    image_path = tmp_path / "plot.png"
    image_path.write_bytes(PNG_DATA)
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path / "allowed"))

    with pytest.raises(ValueError, match="outside"):
        image_server.read_image(str(image_path))


def test_read_image_rejects_requests_when_roots_are_unset(tmp_path, monkeypatch):
    image_path = tmp_path / "plot.png"
    image_path.write_bytes(PNG_DATA)
    monkeypatch.delenv("MADA_IMAGE_ROOTS", raising=False)

    with pytest.raises(ValueError, match="outside"):
        image_server.read_image(str(image_path))


def test_read_image_rejects_invalid_webp_signature(tmp_path, monkeypatch):
    image_path = tmp_path / "plot.webp"
    image_path.write_bytes(b"RIFF" + b"\x00" * 4 + b"NOPE")
    monkeypatch.setenv("MADA_IMAGE_ROOTS", str(tmp_path))

    with pytest.raises(ValueError, match="does not match"):
        image_server.read_image(str(image_path))
