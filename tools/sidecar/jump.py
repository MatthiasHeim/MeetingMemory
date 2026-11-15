"""Hand a slice of the live transcript to a new Claude session through the local jump server.

The HTTP call is injectable, so tests never touch the network. Nothing here raises:
every failure becomes a short German message for the status line of the window.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

JUMP_URL = "http://localhost:7842/jump"
TOKEN_PATH = Path.home() / ".claude" / "jump-server-token"
JUMP_REPO = "Brain"
JUMP_TOPIC = "Kontext aus dem Meeting-Recorder"
JUMP_TIMEOUT_SECONDS = 25.0
MAX_CONTEXT_CHARS = 60_000

# post(url, payload, timeout) -> parsed JSON reply of the server. May raise.
PostFn = Callable[[str, dict, float], dict]


@dataclass(frozen=True)
class JumpResult:
    ok: bool
    message: str


def build_prompt(context_text: str, range_label: str) -> str:
    context = context_text.strip()
    if len(context) > MAX_CONTEXT_CHARS:
        context = "[…]\n" + context[-MAX_CONTEXT_CHARS:]
    return (
        "Hintergrund: Ich arbeite gerade und der Meeting-Recorder nimmt mit. "
        f"Das ist das Transkript ({range_label}); es dient nur als Kontext.\n\n"
        f"{context}\n\n"
        'Antworte zuerst nur mit "What do you want me to do?" und benutze keine Tools, '
        "bevor ich dir gesagt habe, was du tun sollst."
    )


def build_payload(token: str, context_text: str, range_label: str) -> dict:
    prompt = build_prompt(context_text, range_label)
    return {"token": token, "prompt": prompt, "default": prompt, "topic": JUMP_TOPIC, "repo": JUMP_REPO}


def _read_token(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed local URL
        return json.loads(response.read().decode("utf-8") or "{}")


def start_claude(context_text: str, *, range_label: str, token_path: Path | None = None,
                 post: PostFn | None = None, url: str = JUMP_URL) -> JumpResult:
    """POST the transcript range to the jump server. Never raises."""
    if not context_text.strip():
        return JumpResult(False, "Noch kein Transkript für Claude.")
    token = _read_token(token_path or TOKEN_PATH)
    if not token:
        return JumpResult(False, "Claude nicht gestartet: kein Jump-Token gefunden.")
    try:
        reply = (post or _http_post)(url, build_payload(token, context_text, range_label), JUMP_TIMEOUT_SECONDS)
    except Exception as exc:  # server down, timeout, 403, bad JSON
        return JumpResult(False, f"Claude nicht gestartet: Jump-Server nicht erreichbar ({type(exc).__name__}).")
    if not isinstance(reply, dict) or not reply.get("ok"):
        detail = str(reply.get("error") or "") if isinstance(reply, dict) else ""
        return JumpResult(False, ("Claude nicht gestartet: " + detail[:80]).strip().rstrip(":"))
    return JumpResult(True, "Claude gestartet.")
