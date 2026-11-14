"""Regression test for the silent Gemini hang (2026-07-28, 2026-09-17, 2026-11-09).

The watcher froze for hours in SSL_read after "Sending to Gemini API..." because
google-genai sends every request with timeout=None unless http_options.timeout
is set, so a socket that dies without a RST blocks forever.

These tests point the processor's real google-genai client at a local server
that accepts the connection and then goes silent, and assert both sides:
  * a client WITHOUT a timeout is still blocked after several seconds
    (the bug reproduces — the test is not vacuous), and
  * GeminiAudioProcessor's client raises within its timeout and
    _generate_with_retry fails loudly with RuntimeError.
"""
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

genai = pytest.importorskip("google.genai")
from google.genai import types  # noqa: E402

import gemini_processor  # noqa: E402
from gemini_processor import GeminiAudioProcessor  # noqa: E402


class StallingServer:
    """TCP server that accepts, optionally sends a partial streaming reply, then stalls."""

    def __init__(self, mode: str):
        self.mode = mode  # "before_headers" | "mid_stream"
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.conns = []
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            self.conns.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        conn.settimeout(2)
        try:
            conn.recv(65536)  # read (part of) the request, never answer fully
        except OSError:
            pass
        if self.mode == "mid_stream":
            body = b'data: {"candidates":[{"content":{"parts":[{"text":"{\\"transcript\\": \\"[00:00] A: hal"}]}}]}\r\n\r\n'
            head = (
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            try:
                conn.sendall(head + f"{len(body):x}\r\n".encode() + body + b"\r\n")
            except OSError:
                pass
        # then: silence, connection held open (like a dead peer with no RST)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self._stop.set()
        for c in self.conns:
            try:
                c.close()
            except OSError:
                pass
        self.sock.close()


@pytest.fixture(params=["before_headers", "mid_stream"])
def server(request):
    s = StallingServer(request.param)
    yield s
    s.close()


def _stream_call(client):
    stream = client.models.generate_content_stream(
        model="gemini-3.8-flash", contents=["hi"],
    )
    for _ in stream:
        pass


def test_client_without_timeout_hangs(server):
    """Negative control: the pre-fix client construction blocks indefinitely."""
    client = genai.Client(
        api_key="test-key", http_options=types.HttpOptions(base_url=server.base_url)
    )
    t = threading.Thread(target=lambda: _call_quietly(client), daemon=True)
    t.start()
    t.join(timeout=5)
    assert t.is_alive(), "client without timeout returned — control is not reproducing the hang"


def _call_quietly(client):
    try:
        _stream_call(client)
    except Exception:
        pass


def test_processor_client_times_out_and_fails_loudly(server, monkeypatch):
    """The fixed processor raises instead of hanging, and retries then gives up."""
    real_client = genai.Client

    def client_at_local_server(*args, http_options=None, **kwargs):
        assert http_options is not None and http_options.timeout, (
            "GeminiAudioProcessor must construct its client with a timeout"
        )
        http_options = http_options.model_copy(update={"base_url": server.base_url})
        return real_client(*args, http_options=http_options, **kwargs)

    monkeypatch.setattr(genai, "Client", client_at_local_server)
    monkeypatch.setattr(gemini_processor.time, "sleep", lambda s: None)

    proc = GeminiAudioProcessor(api_key="test-key", model="gemini-3.8-flash",
                                timeout_seconds=1)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="All 3 attempts failed"):
        proc._generate_with_retry(prompt="hi", audio_content=None, max_attempts=3)
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"3 attempts at a 1s timeout took {elapsed:.1f}s"


def test_processor_timeout_wired_from_config_value():
    proc = GeminiAudioProcessor(api_key="test-key", timeout_seconds=600)
    assert proc.client._api_client._http_options.timeout == 600_000


@pytest.mark.parametrize("bad", [0, None, -5])
def test_processor_rejects_missing_timeout(bad):
    with pytest.raises(ValueError):
        GeminiAudioProcessor(api_key="test-key", timeout_seconds=bad)
