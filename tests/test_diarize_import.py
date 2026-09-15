"""Ensure the pyannote worker resolves under launchd's tools-only import path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_worker_target_is_locatable_with_only_tools_on_import_path(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    script = r"""
import sys
from pathlib import Path

tools_dir = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path[:] = [
    str(tools_dir),
    *[
        item for item in sys.path
        if item and Path(item).resolve() != repo and Path(item).resolve() != tools_dir
    ],
]
import diarize
print(diarize.pyannote_worker_path())
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tools_dir), str(repo)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == repo / "pyannote_mp_worker.py"
