"""Live Hochdeutsch transcript, off the audio callback.

Every tick takes the last minute of microphone audio and, when present, the
tail of the system-audio WAV captured at that same moment. Gemini returns
speaker-labelled lines. Only lines newer than the ones already committed are
kept. A failure sets a status string and does not touch recording.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .calibration_policy import OWNER_APPROVED_MODEL, OWNER_APPROVED_PROMPT_THRESHOLD
from .clean_prompt import clean_prompt_text
from .judges import (
    APPROVED_GEMINI_BASE_URL,
    Judge,
    JudgeError,
    _load_json_response,
    _verify_pinned_gemini_destination,
    pinned_gemini_client,
)
from .live_audio import (
    copy_chunk_references,
    read_wav_pcm_tail,
    stereo_tail_wav,
    tail_from_chunks,
)
from .prompts import PROMPT_CONSECUTIVE_GAP_SECONDS, _prompt_runs
from .questions import DICTATING_PROMPT_QUESTION
from .transcript import TranscriptLine, format_timestamp


LIVE_TICK_SECONDS = 20.0
LIVE_TAIL_SECONDS = 60.0
LIVE_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class TailSnapshot:
    """Mic and system tails captured together, before any network call."""

    mic: np.ndarray
    mic_rate: int
    system: np.ndarray | None
    system_rate: int
    window_start: float
    window_end: float
    # A closed tail is the last one (stop, or the end of a replay). Open tails
    # withhold a line that is still running into the window edge.
    closed: bool = False


@dataclass(frozen=True)
class LiveLine:
    """One committed Hochdeutsch line on the microphone clock."""

    start: float
    end: float
    speaker: str
    text: str

    def as_transcript_line(self, index: int) -> TranscriptLine:
        return TranscriptLine(
            index=index,
            seconds=self.start,
            timestamp=format_timestamp(self.start),
            speaker=self.speaker,
            text=self.text,
        )


@dataclass(frozen=True)
class LiveCard:
    """A prompt card: clean text for the clipboard, verbatim lines underneath."""

    key: str
    lines: tuple[TranscriptLine, ...]
    clean_text: str

    @property
    def verbatim_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def copy_text(self) -> str:
        return self.clean_text.strip() or self.verbatim_text

    @property
    def start_seconds(self) -> float:
        return self.lines[0].seconds if self.lines else 0.0

    @property
    def end_seconds(self) -> float:
        return self.lines[-1].seconds if self.lines else 0.0


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _stem(text: str) -> str:
    return _norm(text).replace("…", " ").replace("...", " ").strip(" .")


def _same_utterance(left: str, right: str) -> bool:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 20 and len(b) >= 20 and (a in b or b in a):
        return True
    return False


def _extends(existing: str, incoming: str) -> bool:
    """True when ``incoming`` is a longer hearing of the same utterance."""
    old, new = _stem(existing), _stem(incoming)
    if len(new) < len(old) + 8:
        return False
    if old and old in new:
        return True
    head = old[:48]
    return len(head) >= 24 and head in new


def _looks_unfinished(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(("...", "…", ",", ":", ";")):
        return True
    return stripped[-1] not in ".?!"


def _without_open_tail(
    lines: Sequence[LiveLine], window_end: float, *, closed: bool
) -> list[LiveLine]:
    """Hold only a line that is still cut off at the end of an open tail.

    A finished sentence that happens to end near the window edge is committed
    immediately. Holding every trailing line delayed it by another tick and
    pushed transcript lag past the live budget. A later tail still replaces a
    committed fragment when it hears the same utterance more completely.
    """
    if closed or not lines:
        return list(lines)
    last = max(lines, key=lambda line: (line.end, line.start))
    if window_end - last.end > 2.0 or not _looks_unfinished(last.text):
        return list(lines)
    return [line for line in lines if line is not last]


def commit_new_lines(
    committed: Sequence[LiveLine], incoming: Sequence[LiveLine]
) -> tuple[list[LiveLine], list[LiveLine], int | None]:
    """Keep lines newer than the committed frontier and drop overlap repeats.

    A later tail may hear the same utterance more completely (the first window
    ended mid-sentence). That line is replaced in place. ``revised_from`` is
    the earliest replaced index, so the caller can rescore from there.
    """
    all_lines = list(committed)
    added: list[LiveLine] = []
    revised_from: int | None = None
    frontier = all_lines[-1].end if all_lines else -1.0

    def _note_revision(index: int) -> None:
        nonlocal revised_from
        if revised_from is None or index < revised_from:
            revised_from = index

    for line in sorted(incoming, key=lambda item: (item.start, item.end)):
        text = line.text.strip()
        if not text:
            continue
        stored = LiveLine(line.start, max(line.end, line.start), line.speaker.strip() or "Ich", text)
        extension_at = None
        for index in range(len(all_lines) - 1, max(-1, len(all_lines) - 13), -1):
            other = all_lines[index]
            if abs(other.start - stored.start) > 15:
                continue
            if _norm(other.speaker) != _norm(stored.speaker):
                continue
            if _extends(other.text, stored.text):
                extension_at = index
                break
        if extension_at is not None:
            previous = all_lines[extension_at]
            all_lines[extension_at] = LiveLine(
                previous.start,
                max(previous.end, stored.end),
                previous.speaker,
                stored.text,
            )
            _note_revision(extension_at)
            frontier = max(frontier, all_lines[extension_at].end)
            continue
        if all_lines and stored.start < frontier - 2.0:
            continue
        if any(
            _same_utterance(stored.text, other.text) and abs(other.start - stored.start) <= 15
            for other in all_lines[-12:]
        ):
            continue
        if all_lines and stored.end <= frontier + 0.05 and stored.start <= frontier:
            continue
        all_lines.append(stored)
        added.append(stored)
        frontier = max(frontier, stored.end)
    return all_lines, added, revised_from


def snapshot_recording_tails(recorder: Any, tail_seconds: float = LIVE_TAIL_SECONDS) -> TailSnapshot | None:
    """Snapshot mic references and the system-file tail at one moment.

    Called only from the live worker. The audio callback is not involved.
    """
    chunks = copy_chunk_references(getattr(recorder, "audio_data", []) or [])
    rate = int(getattr(recorder, "sample_rate", 0) or 0)
    if rate <= 0:
        return None
    mic, window_start, window_end = tail_from_chunks(chunks, rate, tail_seconds)
    system = None
    system_rate = rate
    path = getattr(recorder, "_sys_wav", None)
    if path is not None:
        wav_path = Path(path)
        try:
            if wav_path.is_file() and wav_path.stat().st_size > 44:
                system, system_rate = read_wav_pcm_tail(wav_path, tail_seconds)
        except (OSError, ValueError):
            system = None
    if mic.size == 0 and (system is None or getattr(system, "size", 0) == 0):
        return None
    return TailSnapshot(mic, rate, system, system_rate, window_start, window_end)


class GeminiLiveTranscriber:
    """Hochdeutsch transcript of one mic/system tail via pinned ``gemini-3.8-flash``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        types_module: Any | None = None,
        model: str = OWNER_APPROVED_MODEL,
        timeout_seconds: float = 22,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        if client is not None:
            _verify_pinned_gemini_destination(client)
            if _visible_base_url(client) is None:
                raise JudgeError("Gemini live client has no verifiable Google API destination")
            self.client = client
            self.types = types_module
            return
        self.client, self.types = pinned_gemini_client(api_key)

    def transcribe_tails(
        self, mic: np.ndarray, mic_rate: int, system: np.ndarray | None, system_rate: int
    ) -> list[dict[str, Any]]:
        """Return line dicts with offsets relative to the start of the tail.

        One retry, then the error propagates to the worker, which records a
        status line and leaves recording running.
        """
        last: Exception | None = None
        for _attempt in range(2):
            try:
                return self._once(mic, mic_rate, system, system_rate)
            except (JudgeError, OSError, ValueError) as exc:
                last = exc
        assert last is not None
        raise last

    def _once(
        self, mic: np.ndarray, mic_rate: int, system: np.ndarray | None, system_rate: int
    ) -> list[dict[str, Any]]:
        _verify_pinned_gemini_destination(self.client)
        audio = stereo_tail_wav(mic, mic_rate, system, system_rate, LIVE_SAMPLE_RATE)
        if len(audio) < 64:
            return []
        has_system = system is not None and getattr(system, "size", 0) > 0
        prompt = _transcription_prompt(has_system=has_system)
        parts: list[Any] = [prompt]
        if self.types is None:
            parts.append({"mime_type": "audio/wav", "data": audio})
            config: Any = {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": _transcript_schema(),
                "http_options": {
                    "base_url": APPROVED_GEMINI_BASE_URL,
                    "api_version": "v1beta",
                    "timeout": int(self.timeout_seconds * 1000),
                },
            }
        else:
            parts.append(self.types.Part.from_bytes(data=audio, mime_type="audio/wav"))
            config = self.types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=4096,
                media_resolution=self.types.MediaResolution.MEDIA_RESOLUTION_LOW,
                thinking_config=self.types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=_transcript_schema(),
                http_options=self.types.HttpOptions(
                    base_url=APPROVED_GEMINI_BASE_URL,
                    api_version="v1beta",
                    timeout=int(self.timeout_seconds * 1000),
                ),
            )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=parts, config=config
            )
        except Exception as exc:
            raise JudgeError(f"live transcript request failed: {type(exc).__name__}") from exc
        return _lines_from_response(response)


def _visible_base_url(client: Any) -> str | None:
    from .judges import _visible_client_base_url

    return _visible_client_base_url(client)


def _transcription_prompt(has_system: bool) -> str:
    remote = (
        "The WAV is stereo and both channels end at the same moment. "
        "The left channel is the local microphone; label that speaker 'Ich'. "
        "The right channel is remote system audio; label those speakers 'Andere' unless a name is unmistakable."
        if has_system
        else "The WAV left channel is the local microphone. Label that speaker 'Ich'."
    )
    return (
        "Transcribe this meeting tail into Hochdeutsch, word for word. Do not summarise and do not shorten. "
        f"{remote} "
        "Normalise Swiss German into standard German. Keep English words and sentences in English. "
        "Keep names of AI tools as names. In particular, keep 'Claude' when it is spoken; do not rewrite it as an ordinary word. "
        "The local speaker sometimes dictates an instruction to Claude or to an agent in Swiss German "
        "(for example 'ich würde Claude sagen, er soll die Änderung umsetzen und nachfragen, was noch unklar ist'). "
        "Transcribe that instruction through to the last word you hear. "
        "Merge consecutive words from the same speaker into one line when the pause is under two seconds. "
        "A short remark by someone else does not end the local speaker's sentence: write the remark on its own line "
        "and continue the local speaker on the next line. "
        "Never write an ellipsis or three dots. Write the words that were spoken. Do not invent speech. Omit silence. "
        "start_offset and end_offset are seconds from the beginning of this audio, not from the start of the meeting. "
        'Return JSON {"lines":[{"speaker":"...","text":"...","start_offset":0,"end_offset":0}]}.'
    )


def _transcript_schema() -> dict[str, Any]:
    line = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "text": {"type": "string"},
            "start_offset": {"type": "number"},
            "end_offset": {"type": "number"},
        },
        "required": ["speaker", "text", "start_offset", "end_offset"],
    }
    return {
        "type": "object",
        "properties": {"lines": {"type": "array", "items": line}},
        "required": ["lines"],
    }


def _lines_from_response(response: Any) -> list[dict[str, Any]]:
    payload = _load_json_response(response)
    rows = payload.get("lines")
    if not isinstance(rows, list):
        raise JudgeError("live transcript JSON has no lines array")
    lines: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            start = float(row.get("start_offset"))
            end = float(row.get("end_offset"))
        except (TypeError, ValueError):
            continue
        if start != start or end != end:
            continue
        speaker = row.get("speaker")
        lines.append(
            {
                "speaker": speaker.strip() if isinstance(speaker, str) and speaker.strip() else "Ich",
                "text": text.strip(),
                "start_offset": max(0.0, start),
                "end_offset": max(0.0, end),
            }
        )
    return lines


def _absolute_lines(raw: Sequence[dict[str, Any]], window_start: float) -> list[LiveLine]:
    lines: list[LiveLine] = []
    for row in raw:
        start = window_start + float(row["start_offset"])
        end = window_start + float(row["end_offset"])
        if end < start:
            end = start
        lines.append(LiveLine(start, end, str(row["speaker"]), str(row["text"])))
    return lines


UpdateCallback = Callable[[tuple[LiveLine, ...], tuple[LiveCard, ...], str], None]
Cleaner = Callable[[str], str]


class LiveEngine:
    """Worker that ticks the live pipeline without joining the audio callback."""

    def __init__(
        self,
        *,
        snapshot: Callable[[float], TailSnapshot | None],
        transcriber: GeminiLiveTranscriber,
        judge: Judge | None = None,
        clean_prompt: Cleaner | None = None,
        on_update: UpdateCallback | None = None,
        interval: float = LIVE_TICK_SECONDS,
        tail: float = LIVE_TAIL_SECONDS,
        prompt_threshold: float = OWNER_APPROVED_PROMPT_THRESHOLD,
    ):
        self._snapshot = snapshot
        self._transcriber = transcriber
        self._judge = judge
        self._clean_prompt = clean_prompt
        self._on_update = on_update
        self.interval = interval
        self.tail = tail
        self.prompt_threshold = prompt_threshold
        self.lines: list[LiveLine] = []
        self.cards: list[LiveCard] = []
        self.status = ""
        self.last_transcribe_seconds = 0.0
        self._scores: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="meeting-sidecar-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the worker to finish. Does not wait, so Stop stays responsive."""
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self.interval):
                break
            self._safe_tick()
        self._safe_tick(closed=True)

    def _safe_tick(self, *, closed: bool = False) -> None:
        try:
            snapshot = self._snapshot(self.tail)
            if snapshot is None:
                return
            if closed and not snapshot.closed:
                snapshot = replace(snapshot, closed=True)
            self.process_snapshot(snapshot)
        except Exception as exc:
            self._fail(exc)

    def process_snapshot(self, snapshot: TailSnapshot) -> list[LiveLine]:
        """Transcribe one tail, commit new lines, and refresh prompt cards."""
        started = time.perf_counter()
        raw = self._transcriber.transcribe_tails(
            snapshot.mic, snapshot.mic_rate, snapshot.system, snapshot.system_rate
        )
        self.last_transcribe_seconds = time.perf_counter() - started
        incoming = _absolute_lines(raw, snapshot.window_start)
        incoming = _without_open_tail(incoming, snapshot.window_end, closed=snapshot.closed)
        self.lines, added, revised_from = commit_new_lines(self.lines, incoming)
        self.status = ""
        if added or revised_from is not None:
            try:
                if revised_from is not None:
                    self._scores = self._scores[:revised_from]
                    previous = revised_from
                else:
                    previous = len(self.lines) - len(added)
                self._extend_scores(previous)
                self._refresh_cards()
            except Exception as exc:
                self.status = (
                    f"Prompt-Erkennung unterbrochen ({type(exc).__name__}). "
                    "Die Aufnahme läuft weiter."
                )
        self._publish()
        return added

    def _extend_scores(self, previous_count: int) -> None:
        if self._judge is None:
            self._scores = [0.0] * len(self.lines)
            return
        transcript = [line.as_transcript_line(index) for index, line in enumerate(self.lines)]
        context = 6
        start = max(0, previous_count - context)
        subset = transcript[start:]
        scores = self._judge.judge(
            subset, {"dictating_prompt": DICTATING_PROMPT_QUESTION}, context=context
        )["dictating_prompt"]
        if len(scores) != len(subset):
            raise JudgeError("prompt judge did not score every new line")
        self._scores = self._scores[:start] + [float(score) for score in scores]

    def _refresh_cards(self) -> None:
        if self._judge is None or not self.lines or len(self._scores) != len(self.lines):
            return
        transcript = [line.as_transcript_line(index) for index, line in enumerate(self.lines)]
        runs = _prompt_runs(self._scores, transcript, threshold=self.prompt_threshold)
        seen: set[str] = set()
        cards: list[LiveCard] = []
        for run in runs:
            if not run:
                continue
            key = f"{transcript[run[0]].seconds:.1f}"
            seen.add(key)
            verbatim_lines = tuple(transcript[index] for index in run)
            verbatim = "\n".join(line.text for line in verbatim_lines)
            previous = next((card for card in self.cards if card.key == key), None)
            if previous is not None and previous.verbatim_text == verbatim:
                cards.append(previous)
                continue
            clean = ""
            if self._clean_prompt is not None:
                try:
                    clean = self._clean_prompt(verbatim).strip()
                except Exception as exc:
                    self.status = (
                        f"Sauberer Prompt fehlgeschlagen ({type(exc).__name__}). "
                        "Die Aufnahme läuft weiter."
                    )
                    clean = ""
            cards.append(LiveCard(key, verbatim_lines, clean))
        # Keep a card that was already shown if a later rescore drops it for one tick.
        for card in self.cards:
            if card.key not in seen:
                cards.append(card)
        cards.sort(key=lambda card: card.start_seconds)
        self.cards = cards

    def _fail(self, exc: Exception) -> None:
        self.status = f"Live-Transkription unterbrochen ({type(exc).__name__}). Die Aufnahme läuft weiter."
        self._publish()

    def _publish(self) -> None:
        if self._on_update is None:
            return
        self._on_update(tuple(self.lines), tuple(self.cards), self.status)


def replay_wav(
    path: Path,
    *,
    transcriber: GeminiLiveTranscriber,
    judge: Judge | None = None,
    clean_prompt: Cleaner | None = None,
    tick: float = LIVE_TICK_SECONDS,
    tail: float = LIVE_TAIL_SECONDS,
    prompt_threshold: float = OWNER_APPROVED_PROMPT_THRESHOLD,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Feed a finished WAV through the live pipeline on a simulated clock.

    Audio is revealed only up to the current tick. Lag is tick time plus the
    real transcribe duration, minus the line's audio time. No window is opened.
    """
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="int16", always_2d=True)
    duration = len(audio) / float(rate) if rate else 0.0
    begin = max(0.0, start_seconds)
    finish = duration if end_seconds is None else min(duration, end_seconds)
    engine = LiveEngine(
        snapshot=lambda _tail: None,
        transcriber=transcriber,
        judge=judge,
        clean_prompt=clean_prompt,
        interval=tick,
        tail=tail,
        prompt_threshold=prompt_threshold,
    )
    lags: list[float] = []
    printed_cards: dict[str, tuple[str, str]] = {}
    write = emit or print
    cursor = begin
    while cursor < finish - 1e-6:
        cursor = min(finish, cursor + tick)
        window_start = max(0.0, cursor - tail)
        start_index = int(window_start * rate)
        end_index = max(start_index, int(cursor * rate))
        frame = audio[start_index:end_index]
        mic = frame[:, 0] if frame.size else np.zeros(0, dtype=np.int16)
        system = frame[:, 1:] if frame.ndim == 2 and frame.shape[1] > 1 else None
        closed = cursor >= finish - 1e-6
        snapshot = TailSnapshot(
            mic, int(rate), system, int(rate), window_start, cursor, closed=closed
        )
        try:
            added = engine.process_snapshot(snapshot)
        except Exception as exc:
            write(f"STATUS {engine.status or type(exc).__name__}")
            continue
        # Lag is audio time to the transcript line. Prompt judging runs after
        # the line exists and does not count toward that lag.
        available = cursor + engine.last_transcribe_seconds
        for line in added:
            lag = available - line.start
            lags.append(lag)
            write(
                f"LINE {format_timestamp(line.start)} {line.speaker}: {line.text} lag={lag:.1f}s"
            )
        for card in engine.cards:
            signature = (card.verbatim_text, card.clean_text)
            if printed_cards.get(card.key) == signature:
                continue
            printed_cards[card.key] = signature
            write(
                f"CARD {format_timestamp(card.start_seconds)}-{format_timestamp(card.end_seconds)}"
            )
            write(f"CLEAN: {card.clean_text}")
            write(f"VERBATIM: {card.verbatim_text}")
        if engine.status:
            write(f"STATUS {engine.status}")
    median = _median(lags)
    write(
        f"median_lag_seconds={median:.2f}" if median is not None else "median_lag_seconds="
    )
    write(f"lines={len(engine.lines)} cards={len(engine.cards)}")
    return {
        "lines": tuple(engine.lines),
        "cards": tuple(engine.cards),
        "lags": tuple(lags),
        "median_lag_seconds": median,
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# Re-exported so a caller can see the merge gap the live cards use.
LIVE_PROMPT_GAP_SECONDS = PROMPT_CONSECUTIVE_GAP_SECONDS


def default_clean_prompt(api_key: str | None = None) -> Cleaner:
    def clean(verbatim: str) -> str:
        return clean_prompt_text(verbatim, api_key=api_key)

    return clean
