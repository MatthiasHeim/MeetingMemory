"""The watcher is launched from tools/, so the repo-root worker must be found."""

from __future__ import annotations

import inspect
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import diarize  # noqa: E402


def test_diarization_timeout_default_is_fifteen_minutes():
    import inspect

    import transcribe_watcher as tw

    assert diarize.DIARIZATION_TIMEOUT_SECONDS == 900
    assert (
        inspect.signature(diarize.run_pyannote_diarization).parameters["timeout_seconds"].default
        == 900
    )
    assert "DIARIZATION_TIMEOUT_SECONDS" in inspect.getsource(
        tw.TranscribeWatcher._run_diarization_safe
    )
    body = inspect.getsource(tw.TranscribeWatcher)
    assert "Using pyannote diarization prior" in body
    assert "diarization.enabled: false" in body
    assert "single-source" in body


def test_diarization_process_target_lives_in_tools_and_fixes_the_path():
    source = inspect.getsource(diarize.run_pyannote_diarization)
    assert "_pyannote_child" in source
    assert "pyannote_proc_entrypoint" not in source
    child = inspect.getsource(diarize._pyannote_child)
    assert "_ensure_worker_importable" in child
    assert "pyannote_proc_entrypoint" in child


def test_spawned_child_finds_the_repo_root_worker():
    """Reproduce launchd: sys.path[0] is tools/, and the parent cannot import the worker."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "ok" in result.stdout


def _probe() -> None:
    tools = ROOT / "tools"
    sys.path = [str(tools)] + [
        entry for entry in sys.path if Path(entry).resolve() != ROOT.resolve()
    ]
    import importlib

    importlib.invalidate_caches()
    try:
        import pyannote_mp_worker  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit("parent imported pyannote_mp_worker; the tools-only path was not reproduced")
    import diarize as watcher_diarize

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=watcher_diarize._pyannote_child,
        args=({"import_probe": True}, queue),
    )
    proc.start()
    message = queue.get(timeout=30)
    proc.join(timeout=10)
    if not message.get("ok"):
        raise SystemExit(f"probe failed: {message}")
    origin = Path(message["probe"]).resolve()
    if origin != (ROOT / "pyannote_mp_worker.py").resolve():
        raise SystemExit(f"worker origin {origin}")
    print("ok")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        _probe()
