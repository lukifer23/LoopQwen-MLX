import os
import subprocess
import sys
from pathlib import Path

import pytest

from loopqwen.ordinary_model import ordinary


def test_installed_comparison_import_is_independent_of_checkout(tmp_path):
    project = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from loopqwen.tool_compare import DenseAssistant; from loopqwen.generate_cli import run",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(project)},
        check=True,
    )


def test_registered_inventory_with_missing_text_files_fails_closed(tmp_path):
    project = Path(__file__).resolve().parents[1]
    inventory = (
        project / "research/results/packed-release-retention-04/export-files.json"
    )
    (tmp_path / "tool-export-files.json").write_bytes(inventory.read_bytes())
    with pytest.raises(ValueError, match="certified Base file inventory"):
        ordinary(tmp_path)


def test_explicit_offline_control_rejects_an_unknown_directory(tmp_path, monkeypatch):
    def unexpected_download():
        pytest.fail("An explicitly selected offline control fell through to the Hub")

    monkeypatch.setattr("loopqwen.ordinary_model.load_base", unexpected_download)
    with pytest.raises(ValueError, match="explicit Base checkpoint"):
        ordinary(tmp_path, require_release=True)
