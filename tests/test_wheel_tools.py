"""Inspect real ZIP metadata without executing any wheel contents."""

from zipfile import ZipFile

import pytest

from loopqwen.wheel_tools import tools


def make(path, payload, member="sample-1.0.dist-info/METADATA"):
    with ZipFile(path, "w") as archive:
        archive.writestr(member, payload)
        archive.writestr(
            "sample/__init__.py",
            "raise RuntimeError('Never import an inspected wheel')",
        )


def test_wheel_snapshot_preserves_repeated_and_absent_fields(tmp_path):
    path = tmp_path / "sample-1.0-py3-none-any.whl"
    raw = (
        "Metadata-Version: 2.4\r\nName: sample\r\nVersion: 1.0\r\n"
        "Summary: Décembre\r\nRequires-Dist: first>=1\r\nRequires-Dist: second\r\n"
        "\r\nRequires-Python: >=9\r\n"
    )
    make(path, raw)
    (tool,) = tools([path, path])
    get = lambda field: tool.callback(path.name, field)["values"]
    assert get("requires-dist") == ["first>=1", "second"]
    assert get("Requires-Python") == [] and get("Summary") == ["Décembre"]
    path.unlink()
    assert get("Name") == ["sample"]
    with pytest.raises(KeyError):
        tool.callback("different-1.0-py3-none-any.whl", "Name")
    with pytest.raises(ValueError):
        get("Name\nVersion")


def test_wheel_rejects_ambiguous_or_mislabeled_metadata(tmp_path):
    path = tmp_path / "sample-1.0-py3-none-any.whl"
    make(path, "Metadata-Version: 2.4\nName: another\nVersion: 1.0\n\n")
    with pytest.raises(ValueError, match="identity"):
        tools([path])
    with ZipFile(path, "a") as z:
        z.writestr("other-1.0.dist-info/METADATA", "Name: sample\nVersion: 1.0\n")
    with pytest.raises(ValueError, match="one top-level"):
        tools([path])
