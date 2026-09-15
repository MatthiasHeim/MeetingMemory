"""Process boundary for the signed ScreenCaptureKit recorder.

The native app owns audio callbacks, durable segments, and the atomic manifest.
This bridge deliberately owns only one session directory and one launch token.
It uses LaunchServices by default because direct execution of a bundled binary
does not have the same macOS TCC attribution guarantees.

``status()`` is intentionally cheap: it only reads the already-atomic manifest,
which makes it safe to call from a menu-bar UI timer.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


MANIFEST_NAME = "capture-manifest.json"
_ACTIVE_STATUSES = {"starting", "recording", "degraded"}
_FINAL_STATUSES = {"complete", "degraded", "failed"}


class NativeCaptureError(RuntimeError):
    """Base error for an owned native capture session."""


class NativeCaptureStartError(NativeCaptureError):
    """The native recorder reported a terminal startup failure."""


class NativeCaptureReadinessTimeout(NativeCaptureError):
    """Readiness was not reached; capture is intentionally left running."""


class NativeCaptureStopError(NativeCaptureError):
    """A process identity check or graceful finalization failed."""


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read an atomic JSON file without mutating process or file state."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _process_start_signature(pid: int) -> Optional[str]:
    """Return an OS start-time signature used to reject PID reuse."""

    if pid <= 0:
        return None
    result = subprocess.run(
        ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    signature = result.stdout.strip()
    return signature or None


def _process_command(pid: int) -> Optional[str]:
    if pid <= 0:
        return None
    result = subprocess.run(
        ["/bin/ps", "-ww", "-o", "command=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    command = result.stdout.strip()
    return command or None


class NativeCaptureSession:
    """Launch and observe one native capture app session.

    Args:
        bundle: ``MeetingNativeCapture.app`` built by build_native_capture.sh.
        session_dir: Fresh directory where the native recorder writes its
            manifest and immutable relative-path segments.
        duration: Optional recorder-side auto-stop duration in seconds.
        launch_mode: ``launchservices`` (default) preserves app-bundle TCC
            attribution. ``direct`` exists only for local non-capture tests.

    ``start()`` and ``stop()`` return the full manifest/status dictionary. A
    readiness timeout raises ``NativeCaptureReadinessTimeout`` but deliberately
    does *not* stop the process, so an already-running one-source capture can
    become a visible degraded session rather than losing its segments.
    """

    def __init__(
        self,
        bundle: os.PathLike[str] | str,
        session_dir: os.PathLike[str] | str,
        *,
        duration: Optional[float] = None,
        launch_mode: str = "launchservices",
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.bundle = Path(bundle).expanduser().resolve()
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.manifest_path = self.session_dir / MANIFEST_NAME
        self.duration = duration
        self.launch_mode = launch_mode
        self.poll_interval_seconds = max(0.01, poll_interval_seconds)
        # `argparse` treats a bare next argument beginning with '-' as another
        # option. Prefix the URL-safe random component so a rare leading '-'
        # cannot make a direct test launch fail before it publishes ownership.
        self._session_token = f"session-{secrets.token_urlsafe(32)}"
        # The native app registers this exact live process with kqueue before
        # capture starts. Unlike a polling PID check, the kernel registration
        # remains bound to this process generation and notices an owner crash
        # even when LaunchServices keeps the app process alive.
        self._owner_pid = os.getpid()
        self._direct_process: Optional[subprocess.Popen[bytes]] = None
        self._observed_pid: Optional[int] = None
        self._observed_start_signature: Optional[str] = None
        self._observed_session_id: Optional[str] = None
        self._started = False

        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive when provided")
        if launch_mode not in {"launchservices", "direct"}:
            raise ValueError("launch_mode must be 'launchservices' or 'direct'")

    @property
    def executable_path(self) -> Path:
        return self.bundle / "Contents" / "MacOS" / "native_capture"

    def status(self) -> Dict[str, Any]:
        """Return the last manifest snapshot, using only a file read.

        It intentionally neither polls the process nor acquires ownership of a
        PID so repeated menu-bar status updates remain side-effect free.
        """

        manifest = _read_json(self.manifest_path)
        if manifest is None:
            return {
                "status": "not_started" if not self._started else "starting",
                "finalized": False,
                "manifest_path": str(self.manifest_path),
            }
        result = dict(manifest)
        result["manifest_path"] = str(self.manifest_path)
        return result

    def start(self, timeout_seconds: float = 20.0) -> Dict[str, Any]:
        """Launch the recorder and wait for a real source-buffer readiness state."""

        if self._started:
            raise NativeCaptureError("this NativeCaptureSession has already been started")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.executable_path.is_file():
            raise NativeCaptureError(
                f"native capture executable is missing: {self.executable_path}; "
                "build MeetingNativeCapture.app first"
            )
        if self.manifest_path.exists():
            raise NativeCaptureError(
                f"refusing to reuse a session directory containing {MANIFEST_NAME}: {self.session_dir}"
            )

        # A session object can be created before a worker forks. Bind the
        # native monitor to the process that actually launches the app.
        self._owner_pid = os.getpid()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        arguments = [
            str(self.executable_path),
            "--output-dir",
            str(self.session_dir),
            "--session-token",
            self._session_token,
            "--owner-pid",
            str(self._owner_pid),
        ]
        if self.duration is not None:
            arguments.extend(["--duration", str(self.duration)])

        if self.launch_mode == "launchservices":
            self._launch_via_launchservices(arguments)
        else:
            self._launch_directly(arguments)
        self._started = True

        deadline = time.monotonic() + timeout_seconds
        last_status: Dict[str, Any] = self.status()
        while time.monotonic() < deadline:
            manifest = _read_json(self.manifest_path)
            if manifest is not None:
                self._validate_owned_manifest(manifest)
                self._observe_process(manifest)
                last_status = self._with_manifest_path(manifest)
                status = str(manifest.get("status", "starting"))
                if status == "failed":
                    raise NativeCaptureStartError(self._error_message("native capture failed to start", manifest))
                if self._is_ready_enough(manifest):
                    return last_status
            self._raise_if_direct_process_died()
            time.sleep(self.poll_interval_seconds)

        # Never kill here. In particular, a system source that starts late may
        # already have valid microphone segments that must remain recoverable.
        detail = self._with_manifest_path(last_status)
        raise NativeCaptureReadinessTimeout(
            f"native capture did not reach a usable ready/degraded state within {timeout_seconds:.1f}s; "
            f"it remains running and can be inspected at {self.manifest_path}. Last status: {detail.get('status')}"
        )

    def stop(self, timeout_seconds: float = 20.0) -> Dict[str, Any]:
        """Signal only this session's recorder and wait for a finalized manifest and reap."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self._started:
            raise NativeCaptureStopError("cannot stop a session that this bridge did not start")

        manifest = _read_json(self.manifest_path)
        if manifest is None:
            raise NativeCaptureStopError(f"native recorder never published {self.manifest_path}")
        self._validate_owned_manifest(manifest)
        self._observe_process(manifest)

        if not bool(manifest.get("finalized", False)):
            self._signal_owned_recorder(manifest)

        deadline = time.monotonic() + timeout_seconds
        last_manifest = manifest
        while time.monotonic() < deadline:
            current = _read_json(self.manifest_path)
            if current is not None:
                self._validate_owned_manifest(current)
                last_manifest = current
                if bool(current.get("finalized", False)):
                    self._wait_for_owned_process_exit(deadline)
                    return self._with_manifest_path(current)
            self._raise_if_direct_process_died(allow_after_finalization=True)
            time.sleep(self.poll_interval_seconds)
        raise NativeCaptureStopError(
            f"native capture did not finalize within {timeout_seconds:.1f}s; "
            f"last status: {last_manifest.get('status')} ({self.manifest_path})"
        )

    def _launch_via_launchservices(self, arguments: list[str]) -> None:
        # LaunchServices starts the executable *as the signed app bundle*, which
        # is necessary for the same TCC identity that owns screen/microphone
        # prompts. `open` itself is not treated as the recorder process.
        command = ["/usr/bin/open", "-n", str(self.bundle), "--args", *arguments[1:]]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise NativeCaptureError(
                "LaunchServices could not open MeetingNativeCapture.app: "
                f"{result.stderr.strip() or result.stdout.strip() or result.returncode}"
            )

    def _launch_directly(self, arguments: list[str]) -> None:
        # Test-only mode: it deliberately avoids claiming that direct execution
        # receives production TCC attribution.
        self._direct_process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    def _validate_owned_manifest(self, manifest: Dict[str, Any]) -> None:
        token = manifest.get("session_token")
        if token != self._session_token:
            raise NativeCaptureError(
                f"manifest at {self.manifest_path} does not belong to this bridge session"
            )
        session_id = manifest.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise NativeCaptureError("native manifest has no session_id")
        if self._observed_session_id is None:
            self._observed_session_id = session_id
        elif self._observed_session_id != session_id:
            raise NativeCaptureError("native manifest session_id changed during this bridge session")

    def _observe_process(self, manifest: Dict[str, Any]) -> None:
        process = manifest.get("process")
        if not isinstance(process, dict):
            raise NativeCaptureError("native manifest has no process identity")
        pid = process.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            raise NativeCaptureError("native manifest process.pid is invalid")

        if self._direct_process is not None and pid != self._direct_process.pid:
            raise NativeCaptureError("manifest PID does not match the bridge-owned direct child")

        signature = _process_start_signature(pid)
        if signature is None and not bool(manifest.get("finalized", False)):
            raise NativeCaptureError(f"native manifest PID {pid} is not running")
        if self._observed_pid is None:
            self._observed_pid = pid
            self._observed_start_signature = signature
        elif self._observed_pid != pid:
            raise NativeCaptureError("native manifest PID changed during this bridge session")
        elif signature is not None and signature != self._observed_start_signature:
            raise NativeCaptureError("native manifest PID was reused by another process")

    def _is_ready_enough(self, manifest: Dict[str, Any]) -> bool:
        status = str(manifest.get("status", "starting"))
        if status not in {"recording", "degraded", "complete"}:
            return False
        readiness = manifest.get("readiness")
        if isinstance(readiness, dict):
            return bool(readiness.get("mic")) or bool(readiness.get("system"))
        tracks = manifest.get("tracks")
        if isinstance(tracks, list):
            return any(isinstance(track, dict) and track.get("first_buffer_ready") for track in tracks)
        return False

    def _signal_owned_recorder(self, manifest: Dict[str, Any]) -> None:
        process = manifest.get("process")
        assert isinstance(process, dict)  # validated by _observe_process
        pid = process["pid"]

        if self._direct_process is not None:
            if self._direct_process.poll() is None:
                self._direct_process.send_signal(signal.SIGINT)
            return

        if self._observed_pid != pid or self._observed_start_signature is None:
            raise NativeCaptureStopError("bridge did not observe an owned native recorder PID")
        current_signature = _process_start_signature(pid)
        if current_signature != self._observed_start_signature:
            raise NativeCaptureStopError("refusing to signal a PID whose start identity changed")
        command = _process_command(pid)
        expected = str(self.executable_path)
        if command is None or (expected not in command and "native_capture" not in command):
            raise NativeCaptureStopError("refusing to signal a process that is not the native capture executable")
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            # The final manifest may have been written immediately before the
            # process exited; the finalization wait below makes that observable.
            return

    def _wait_for_owned_process_exit(self, deadline: float) -> None:
        if self._direct_process is not None:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                self._direct_process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise NativeCaptureStopError("direct native recorder finalized but was not reaped") from exc
            return

        if self._observed_pid is None:
            raise NativeCaptureStopError("bridge has no observed recorder PID to reap")
        while time.monotonic() < deadline:
            signature = _process_start_signature(self._observed_pid)
            if signature is None:
                return
            if signature != self._observed_start_signature:
                raise NativeCaptureStopError("recorder PID was reused before it could be reaped")
            time.sleep(self.poll_interval_seconds)
        raise NativeCaptureStopError("native recorder finalized but its owned process did not exit")

    def _raise_if_direct_process_died(self, *, allow_after_finalization: bool = False) -> None:
        if self._direct_process is None:
            return
        code = self._direct_process.poll()
        if code is None:
            return
        manifest = _read_json(self.manifest_path)
        if allow_after_finalization and manifest is not None and bool(manifest.get("finalized", False)):
            return
        raise NativeCaptureError(
            f"direct native recorder exited with code {code} before publishing a finalized manifest"
        )

    def _with_manifest_path(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(manifest)
        result["manifest_path"] = str(self.manifest_path)
        return result

    @staticmethod
    def _error_message(prefix: str, manifest: Dict[str, Any]) -> str:
        errors = manifest.get("errors")
        if isinstance(errors, list) and errors:
            last = errors[-1]
            if isinstance(last, dict):
                detail = last.get("message") or last.get("code")
                if detail:
                    return f"{prefix}: {detail}"
        return prefix


__all__ = [
    "NativeCaptureError",
    "NativeCaptureReadinessTimeout",
    "NativeCaptureSession",
    "NativeCaptureStartError",
    "NativeCaptureStopError",
]
