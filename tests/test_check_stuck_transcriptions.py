"""Two-sided tests for tools/check_stuck_transcriptions.py.

Positive: the 2026-11-09 Livio hang shape (last line "Sending to Gemini API...",
no transcript, >2h) must be flagged with exit 1. Negative: completed runs,
runs whose transcript exists, runs younger than 2h, and recordings re-queued
after a watcher restart must NOT be flagged (exit 0). A missing log is exit 2.
"""
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import check_stuck_transcriptions as cst  # noqa: E402

COMPLETED = """\
2026-11-09 16:35:50 - INFO - Starting Gemini processing: 2026-11-09_15-59-47.wav
2026-11-09 16:35:50 - INFO - Sending to Gemini API...
2026-11-09 16:44:12 - INFO - Gemini processing complete in 535.5s
2026-11-09 16:59:51 - INFO - Saved result to: 2026-11-09_15-59-47.json
"""
HUNG = """\
2026-11-09 22:37:00 - INFO - Starting Gemini processing: 2026-11-09_21-29-23.wav
2026-11-09 22:37:13 - INFO - Converting WAV to MP3...
2026-11-09 22:38:39 - INFO - Audio duration: 67.5 minutes
2026-11-09 22:38:39 - INFO - Sending to Gemini API...
"""
REQUEUED = """\
2026-11-10 10:53:40 - INFO - Starting Gemini processing: 2026-11-09_21-29-23.wav
2026-11-10 10:55:10 - INFO - Sending to Gemini API...
"""


@pytest.fixture
def env(tmp_path):
    log = tmp_path / "watcher.log"
    transcripts = tmp_path / "Transcripts"
    transcripts.mkdir()
    recordings = tmp_path / "Recordings"
    recordings.mkdir()
    for stem in ("2026-11-09_21-29-23", "2026-11-09_15-59-47"):
        (recordings / f"{stem}.wav").write_bytes(b"RIFF")
    return log, transcripts


def run(log, transcripts, now, *extra):
    return cst.main(["--log", str(log), "--transcripts", str(transcripts),
                     "--recordings", str(transcripts.parent / "Recordings"),
                     "--now", now, *extra])


# ---- positive: must flag -------------------------------------------------

def test_flags_the_livio_hang(env, capsys):
    log, tr = env
    log.write_text(COMPLETED + HUNG)
    (tr / "2026-11-09_15-59-47.json").write_text("{}")
    assert run(log, tr, "2026-11-10 08:00:00") == 1
    out = capsys.readouterr().out
    assert "2026-11-09_21-29-23" in out and "2026-11-09_15-59-47" not in out


def test_flags_just_past_threshold(env):
    log, tr = env
    log.write_text(HUNG)
    assert run(log, tr, "2026-11-10 00:38:40") == 1  # 2h00m01s


def test_flags_stuck_even_if_watcher_moved_on_and_died(env):
    """Hung call, watcher restarted, processed another file, never retried this one."""
    log, tr = env
    other = COMPLETED.replace("2026-11-09 16:", "2026-11-10 09:")
    log.write_text(HUNG + other)
    (tr / "2026-11-09_15-59-47.json").write_text("{}")
    assert run(log, tr, "2026-11-10 10:00:00") == 1


def test_cli_exit_code_and_json(env):
    log, tr = env
    log.write_text(HUNG)
    proc = subprocess.run(
        [sys.executable, str(TOOLS / "check_stuck_transcriptions.py"), "--log", str(log),
         "--transcripts", str(tr), "--recordings", str(tr.parent / "Recordings"), "--now", "2026-11-10 08:00:00", "--json"],
        capture_output=True, text=True)
    assert proc.returncode == 1
    assert '"stem": "2026-11-09_21-29-23"' in proc.stdout


def test_notify_failure_is_exit_2_not_silent(env, monkeypatch):
    log, tr = env
    log.write_text(HUNG)

    def boom(msg):
        raise RuntimeError("telegram down")
    monkeypatch.setattr(cst, "send_notification", boom)
    assert run(log, tr, "2026-11-10 08:00:00", "--notify") == 2


def test_notify_sends_stable_message(env, monkeypatch):
    log, tr = env
    log.write_text(HUNG)
    sent = []
    monkeypatch.setattr(cst, "send_notification", sent.append)
    assert run(log, tr, "2026-11-10 08:00:00", "--notify") == 1
    assert run(log, tr, "2026-11-10 09:30:00", "--notify") == 1
    assert len(sent) == 2 and sent[0] == sent[1]  # identical text -> notify dedupes


# ---- negative: must NOT flag ---------------------------------------------

def test_completed_run_is_ok(env):
    log, tr = env
    log.write_text(COMPLETED)
    assert run(log, tr, "2026-11-10 08:00:00") == 0


def test_transcript_present_is_ok(env):
    log, tr = env
    log.write_text(HUNG)
    (tr / "2026-11-09_21-29-23.json").write_text("{}")
    assert run(log, tr, "2026-11-10 08:00:00") == 0


def test_under_threshold_is_ok(env):
    log, tr = env
    log.write_text(HUNG)
    assert run(log, tr, "2026-11-10 00:38:00") == 0  # 1h59m21s


def test_requeued_after_restart_resets_clock(env):
    log, tr = env
    log.write_text(HUNG + REQUEUED)
    assert run(log, tr, "2026-11-10 11:30:00") == 0


def test_send_without_any_start_line_is_ignored(env):
    log, tr = env
    log.write_text("2026-11-09 22:38:39 - INFO - Sending to Gemini API...\n")
    assert run(log, tr, "2026-11-10 08:00:00") == 0


def test_wav_gone_is_not_stuck(env):
    """2026-07-02_14-03-11 shape: hung, MP3 only, WAV removed -> restart can't help."""
    log, tr = env
    log.write_text(HUNG)
    (tr.parent / "Recordings" / "2026-11-09_21-29-23.wav").unlink()
    assert run(log, tr, "2026-11-10 08:00:00") == 0


def test_missing_log_is_exit_2(env):
    log, tr = env
    assert run(log, tr, "2026-11-10 08:00:00") == 2
