#!/usr/bin/env python3
"""
Gemini Audio Processor - Transcription and audio-only analysis.

Processes audio files through Gemini API to get signals that only the audio
reveals: verbatim transcript, diarization (speaking_pct), overall sentiment,
per-speaker emotional arcs, pacing (wpm/hesitations/pauses), interruption
events, and energy levels.

Text-reasoning fields (title, summary, key points, tags, meeting type, action
items, decisions, coaching feedback) used to live here too, but were moved out
because they're better produced by Claude with full Brain context. The prior
prompt's coaching scaffolding also competed for the model's attention with
transcription itself.
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# Audio-only analysis prompt. Single-shot path uses this on raw audio. Chunked
# path runs each chunk through this, then runs a separate text-only reduce-pass
# (REDUCE_PASS_PROMPT) over the merged transcript.
AUDIO_ANALYSIS_PROMPT = """You are transcribing and analyzing a meeting recording. Capture signals that only the audio reveals — these cannot be recovered later from transcript text alone.

## TRANSCRIPTION (REQUIRED)

Transcribe the ENTIRE audio word-for-word. Do not summarize or skip any parts.

- Speaker labels: "Matthias:" for the host (Lailix consultant), otherwise speaker names if identifiable from context, else "Speaker 1:", "Speaker 2:", etc.
- Timestamps at EVERY speaker change: start each speaker turn with `[MM:SS]` (or `[HH:MM:SS]` for recordings over an hour), e.g. `[14:23] Matthias: ...`. Every change of speaker MUST carry its own timestamp — downstream attribution verification depends on it.
- Transcribe in the original spoken language AND dialect, exactly as spoken. If a speaker talks Swiss German (Schweizerdeutsch / Mundart), write the Swiss German words verbatim — do NOT translate or normalize them into standard High German (Hochdeutsch). For example, keep "Nei, also ich ha eigentlich Ferie" — never rewrite it as "Nein, also ich habe eigentlich Ferien". Preserve mixed-language (e.g. Swiss German + English) segments verbatim.
- Mark unclear sections `[inaudible]` or `[unclear]`.

## OUTPUT FORMAT

Return a single JSON object (raw — no markdown code fences) with these fields:

{
  "transcript": "Complete verbatim transcript with speaker labels and timestamps",
  "language": "Primary language (e.g., 'German', 'Swiss German', 'English', 'Mixed German/English')",
  "participants": [
    {"name": "Matthias", "role": "host", "speaking_pct": 60, "total_seconds": 1230},
    {"name": "Speaker B", "role": "participant", "speaking_pct": 40, "total_seconds": 820}
  ],
  "overall_sentiment": "positive | neutral | negative | mixed",
  "sentiment_intensity": "mild | moderate | strong",
  "speaker_emotions": [
    {
      "speaker": "Speaker B",
      "arc": [
        {"time": "[14:23]", "tone": "skeptical", "energy": "medium", "trigger": "scope question raised", "quote": "<verbatim quote in original language>"},
        {"time": "[18:01]", "tone": "agitated", "energy": "high", "trigger": "pricing pushback", "quote": "<verbatim quote in original language>"}
      ]
    }
  ],
  "speaker_pacing": {
    "Matthias": {"wpm_avg": 145, "hesitation_count": 8, "longest_pause_sec": 3.2},
    "Speaker B": {"wpm_avg": 110, "hesitation_count": 22, "longest_pause_sec": 5.8}
  },
  "interruptions": [
    {"time": "[12:45]", "interrupter": "Matthias", "interruptee": "Speaker B", "context_quote": "<boundary moment showing both speakers>"}
  ],
  "energy_levels": {
    "Matthias": {"avg": "high", "arc": [{"time": "[00:00]", "level": "high"}, {"time": "[15:00]", "level": "medium"}]},
    "Speaker B": {"avg": "medium", "arc": [{"time": "[00:00]", "level": "medium"}, {"time": "[18:00]", "level": "high"}]}
  }
}

The names "Speaker B"/"Speaker C" above are PLACEHOLDERS for the schema. Replace them with the actual speaker's name when known (from the KNOWN ATTENDEES block below, if present), or with "Speaker B", "Speaker C", etc. if you cannot identify the speaker. Never carry "Speaker B" through as a literal name when a real name is available.

## TONE VOCABULARY

Use specific tone labels from this set:
- Positive: enthusiastic, warm, confident, engaged, curious, supportive, amused
- Neutral: neutral, professional, measured, focused, attentive
- Negative: skeptical, agitated, frustrated, hesitant, anxious, defensive, dismissive, tired
- Complex: thoughtful, conflicted, evasive, polite-but-distant

## ENERGY LEVELS

`high` (animated, fast, leaning in) | `medium` (steady, conversational) | `low` (slow, withdrawn, tired)

## INTERRUPTIONS

Only flag genuine interruptions (cutting someone off mid-thought), not collaborative cross-talk or short acknowledgments ("ja", "mhm", "right"). The `context_quote` should capture the boundary moment showing both speakers.

## SPEAKER_EMOTIONS GUIDANCE

Capture meaningful shifts: energy changes, topic transitions, points of friction, moments of agreement or breakthrough. Aim for 3-8 arc events per active speaker per 15-minute segment. Apply the same observational standard to the host as to anyone else — Matthias's emotional arc matters too.

## PACING MEASUREMENTS

Derive from actual speech rate, not transcript text length. `hesitation_count` is filler words and audible self-corrections ("ähm", "äh", "I mean", "sorry"). `longest_pause_sec` is the longest mid-utterance silence by that speaker (excluding turn-taking gaps).

## NOTES

- The transcript field must contain the COMPLETE recording, not a summary.
- For mixed-language meetings, transcribe each segment in its original language.
- If audio quality is poor in a section, indicate `[inaudible]` and continue.
- Do NOT include title, summary, key_points, tags, meeting_type, action_items, decisions, or coaching feedback — these are produced downstream by an LLM with full repository context.
"""


# Reduce-pass prompt. Used for chunked recordings where each chunk produced its
# own per-speaker analysis. This call sees the merged transcript + all chunk
# analyses concatenated and produces a single coherent meeting-wide view.
REDUCE_PASS_PROMPT = """You receive (a) the merged transcript of a meeting that was transcribed in chunks, and (b) the per-chunk audio analyses concatenated together. Produce a single coherent meeting-wide analysis.

Return raw JSON (no fences) with exactly these fields:

{
  "overall_sentiment": "positive | neutral | negative | mixed",
  "sentiment_intensity": "mild | moderate | strong",
  "speaker_emotions": [
    {"speaker": "<name>", "arc": [{"time": "[MM:SS]", "tone": "...", "energy": "...", "trigger": "...", "quote": "..."}]}
  ],
  "speaker_pacing": {"<speaker>": {"wpm_avg": <int>, "hesitation_count": <int>, "longest_pause_sec": <float>}},
  "interruptions": [{"time": "[MM:SS]", "interrupter": "<name>", "interruptee": "<name>", "context_quote": "..."}],
  "energy_levels": {"<speaker>": {"avg": "high|medium|low", "arc": [{"time": "[MM:SS]", "level": "..."}]}}
}

## RULES

- Merge per-chunk arcs into one timeline per speaker. Drop near-duplicates introduced by chunk overlap (events within 30 seconds of each other with the same tone).
- For pacing, average `wpm_avg` weighted by chunk duration; sum `hesitation_count`; take the max of `longest_pause_sec`.
- For energy `avg`, use the dominant level across the chunks weighted by speaking time.
- `overall_sentiment` and `sentiment_intensity` describe the whole meeting, not the dominant chunk. A meeting that goes positive → tense → resolved is `mixed` with `moderate` intensity.
- Do NOT add fields beyond the schema above. Do NOT produce a transcript (it's already merged).
"""


SELF_NAME = "Matthias Heim"


def _build_attendees_prefix(known_attendees: Optional[list[dict]]) -> str:
    """Render a "KNOWN ATTENDEES" block to prepend to the audio prompt.

    Anchors Gemini to the real attendee list from Google Calendar so it
    doesn't invent names per chunk (the cross-chunk-drift bug behind the
    Antonella mislabel). Returns "" if no usable non-self attendees were
    provided — in that case the prompt is byte-identical to the legacy one.
    """
    if not known_attendees:
        return ""
    non_self = [
        p for p in known_attendees
        if isinstance(p, dict)
        and (p.get("name") or "").strip()
        and (p.get("role") or "").lower() != "self"
        and (p.get("name") or "").strip().lower() != SELF_NAME.lower()
    ]
    if not non_self:
        return ""

    lines = [f"- {SELF_NAME} (host, Lailix)"]
    for p in non_self:
        name = (p.get("name") or "").strip()
        company = (p.get("company") or "").strip()
        role = (p.get("role") or "").strip()
        descriptor = company or role or "participant"
        lines.append(f"- {name} ({descriptor})")

    return (
        "## KNOWN ATTENDEES (USE THESE EXACT NAMES)\n\n"
        "This recording is from a meeting with these attendees "
        "(from Google Calendar):\n"
        + "\n".join(lines) + "\n\n"
        "Use these EXACT names for speaker labels (first name only is fine, "
        "e.g. \"Antonella:\"). Do NOT invent other names. If you genuinely "
        "cannot tell which attendee is speaking, use \"Speaker A:\", "
        "\"Speaker B:\" — but never invent a name not in the list above.\n\n"
    )


def _build_channel_map_prefix(channel_segments: Optional[list]) -> str:
    """Render an "AUDIO CHANNEL MAP" block to prepend to the audio prompt.

    `channel_segments` is the [(t0, t1, 'host'|'remote'|'both')] list from
    `channel_vad.compute_channel_vad` — physical-layer ground truth from the
    hybrid recording's separate mic channel. Returns "" when no segments are
    provided (mono recordings, VAD unavailable) — in that case the prompt is
    byte-identical to the no-map one.
    """
    if not channel_segments:
        return ""
    # Channel attribution is best-effort: ANY failure here must degrade to
    # the no-map prompt, never block transcription.
    try:
        from channel_vad import render_map_text
        map_text = render_map_text(channel_segments)
    except Exception as e:
        logger.warning(f"channel map rendering failed ({e}); no-map prompt")
        return ""
    if not map_text:
        return ""
    return (
        "## AUDIO CHANNEL MAP (GROUND TRUTH)\n\n"
        f"The host's ({SELF_NAME.split()[0]}'s) microphone was recorded on a "
        "separate physical audio channel from the remote participants' audio. "
        "The speaking map below was computed directly from those channel "
        "signals, so it is GROUND TRUTH for who is speaking when:\n\n"
        "- `host` spans: ONLY the host is speaking. Words spoken here are "
        f"{SELF_NAME.split()[0]}'s.\n"
        f"- `remote` spans: the host is NOT speaking. Never attribute words "
        f"in these spans to {SELF_NAME.split()[0]}.\n"
        "- `both` spans: overlapping speech (e.g. backchannel like \"ja\", "
        "\"mhm\", or genuine crosstalk) — attribute by voice as usual.\n\n"
        "Span boundaries are accurate to about ±1 second; very short "
        "interjections near a boundary may belong to the adjacent span.\n\n"
        + map_text + "\n\n"
        "If your voice-based speaker judgement conflicts with a `host` or "
        "`remote` span, the map wins — re-attribute the words accordingly.\n\n"
    )


def _build_diarization_map_prefix(diarization_segments: Optional[list]) -> str:
    """Render an acoustic speaker-map prior for Gemini.

    Pyannote runs on the same mono audio Gemini receives, so these timestamps
    align with Gemini's audio. It is a prior, not physical ground truth:
    pyannote is good at turn boundaries and anonymous speaker changes, while
    Gemini still decides which real person each anonymous speaker is from
    names, intros, language, and the attendee list.
    """
    if not diarization_segments:
        return ""
    try:
        from diarize import render_map_text
        map_text = render_map_text(diarization_segments)
    except Exception as e:
        logger.warning(f"diarization map rendering failed ({e}); no prior")
        return ""
    if not map_text:
        return ""
    return (
        "## ACOUSTIC SPEAKER MAP (PRIOR)\n\n"
        "A local diarization model analyzed the same mono audio you are "
        "transcribing and estimated anonymous speaker turns. Use this as a "
        "confidence-annotated prior for turn boundaries and speaker changes, "
        "not as a final naming authority.\n\n"
        "- High-confidence spans: strongly prefer the shown turn boundary and "
        "anonymous speaker change unless the audio content clearly contradicts it.\n"
        "- Medium-confidence spans: use the map as helpful evidence together "
        "with voice, content, language, and conversational flow.\n"
        "- Low-confidence or overlapped spans: treat the map as uncertain; "
        "decide by content, language, and the actual audio.\n\n"
        "Bind anonymous labels such as SPEAKER_00/SPEAKER_01 to real names "
        "using the KNOWN ATTENDEES block, self-introductions, direct address, "
        "and context. If a label cannot be identified, keep a generic speaker "
        "label consistently instead of inventing a name.\n\n"
        + map_text + "\n\n"
    )


@dataclass
class GeminiResult:
    """Result from Gemini audio processing — audio-derived signals only.

    Text-reasoning fields (title, summary, action items, coaching, etc.) are
    NOT here by design. They are produced downstream by Claude in the Brain
    repo, which has access to ClientContext, SALES_COACHING.md, recent meeting
    patterns, and the wiki — context Gemini does not have.
    """
    # Transcript
    transcript: str
    language: str

    # Diarization (audio-derived)
    participants: list[dict] = field(default_factory=list)  # [{name, role, speaking_pct, total_seconds}]

    # Whole-meeting affect (post reduce-pass for chunked recordings)
    overall_sentiment: str = "neutral"      # positive | neutral | negative | mixed
    sentiment_intensity: str = "moderate"   # mild | moderate | strong

    # Per-speaker signals (irreplaceable from text)
    speaker_emotions: list[dict] = field(default_factory=list)   # [{speaker, arc:[{time,tone,energy,trigger,quote}]}]
    speaker_pacing: dict = field(default_factory=dict)           # {speaker: {wpm_avg, hesitation_count, longest_pause_sec}}
    interruptions: list[dict] = field(default_factory=list)      # [{time, interrupter, interruptee, context_quote}]
    energy_levels: dict = field(default_factory=dict)            # {speaker: {avg, arc:[{time,level}]}}

    # Processing metadata
    input_tokens: int = 0
    output_tokens: int = 0
    audio_duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    model: str = "gemini-2.5-flash"
    chunked: bool = False
    chunk_count: int = 1
    reduce_pass_used: bool = False

    # Raw response for debugging
    raw_response: Optional[dict] = None

    # Error (if processing failed)
    error: Optional[str] = None

    # Forensic log of the channel-based attribution verification pass
    # (speaker_verify). Set by the watcher AFTER verification so the flip
    # decisions persist in the on-disk JSON even when no calendar match
    # exists to carry them into participant_resolution_log.
    speaker_verification_log: Optional[dict] = None

    @property
    def parsed_response(self) -> dict:
        """Dict form for JSON serialization on disk and DB writes."""
        out = {
            "transcript": self.transcript,
            "language": self.language,
            "participants": self.participants,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_intensity": self.sentiment_intensity,
            "speaker_emotions": self.speaker_emotions,
            "speaker_pacing": self.speaker_pacing,
            "interruptions": self.interruptions,
            "energy_levels": self.energy_levels,
            "_meta": {
                "model": self.model,
                "chunked": self.chunked,
                "chunk_count": self.chunk_count,
                "reduce_pass_used": self.reduce_pass_used,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "processing_time_seconds": self.processing_time_seconds,
                "audio_duration_seconds": self.audio_duration_seconds,
            },
        }
        if self.speaker_verification_log is not None:
            out["speaker_verification"] = self.speaker_verification_log
        return out


class GeminiAudioProcessor:
    """Processor for audio files using Gemini 2.5 Flash."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_output_tokens: int = 65536,
        temperature: float = 0.1,
        timeout_seconds: int = 600
    ):
        """Initialize the Gemini processor.

        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env var.
            model: Gemini model to use (default: gemini-2.5-flash)
            max_output_tokens: Maximum output tokens (default: 65536 for full transcripts)
            temperature: Generation temperature (default: 0.1 for accuracy)
            timeout_seconds: Request timeout in seconds (default: 600 for long audio)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it as an environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

        # Import here to allow the module to load without the dependency
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            # Use standard client - custom httpx clients cause issues
            # The Files API handles large uploads reliably
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

    # Audio duration (seconds) above which chunked processing is used.
    # History: 15 min was the old sweet spot for gemini-2.5-flash, which
    # hallucinated (repeat-loops) on long single-shot audio. The pipeline now
    # runs gemini-3-flash-preview, which transcribes a full 60-min meeting in
    # one call cleanly — no repeat-loop, and crucially NO cross-chunk speaker
    # drift/swap (verified 2026-06-04 on a real 60-min Swiss-German 1:1; see
    # docs/transcription-single-call-investigation.md). So the threshold is
    # raised to 60 min: meetings up to an hour go single-call and skip the
    # chunk-boundary speaker bugs entirely. Recordings beyond 60 min still
    # chunk (output-token + request-reliability headroom), and process_audio
    # falls back to chunked if a single-call attempt fails, so a long
    # single-call disconnect never loses the whole meeting.
    CHUNK_THRESHOLD_SEC = 60 * 60
    CHUNK_DURATION_SEC = 15 * 60
    CHUNK_OVERLAP_SEC = 30

    def _get_duration(self, audio_path: Path) -> float:
        import subprocess
        r = subprocess.run(
            ['/opt/homebrew/bin/ffprobe', '-v', 'error',
             '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', str(audio_path)],
            capture_output=True, text=True)
        return float(r.stdout.strip() or 0)

    def _chunk_audio(self, audio_path: Path) -> list:
        """Split audio into overlapping chunks for reliable long-audio transcription.

        Returns list of (chunk_path, offset_seconds) tuples.
        Single-element list if audio doesn't need chunking.
        """
        import subprocess, tempfile
        duration = self._get_duration(audio_path)
        if duration <= self.CHUNK_THRESHOLD_SEC:
            return [(audio_path, 0.0)]

        temp_dir = Path(tempfile.gettempdir()) / f"gemini_chunks_{audio_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        chunks = []
        start = 0.0
        idx = 0
        while start < duration:
            end = min(start + self.CHUNK_DURATION_SEC, duration)
            chunk_path = temp_dir / f"chunk_{idx:02d}.mp3"
            r = subprocess.run(
                ['/opt/homebrew/bin/ffmpeg', '-y', '-i', str(audio_path),
                 '-ss', f'{start:.2f}', '-t', f'{end-start:.2f}',
                 '-c', 'copy', str(chunk_path)],
                capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg chunk failed: {r.stderr.decode()[:200]}")
            chunks.append((chunk_path, start))
            if end >= duration:
                break
            start = end - self.CHUNK_OVERLAP_SEC
            idx += 1
        logger.info(f"Split {duration/60:.1f}min audio into {len(chunks)} chunks")
        return chunks

    def _shift_timestamps(self, text: str, offset_sec: float) -> str:
        """Add offset to [MM:SS] or [HH:MM:SS] timestamps in transcript text."""
        import re
        def shift(m):
            parts = [int(x) for x in m.group(1).split(':')]
            total = sum(p * 60**(len(parts)-1-i) for i, p in enumerate(parts)) + int(offset_sec)
            h, rem = divmod(total, 3600)
            mm, ss = divmod(rem, 60)
            return f"[{h:02d}:{mm:02d}:{ss:02d}]" if h else f"[{mm:02d}:{ss:02d}]"
        return re.sub(r'\[(\d{1,2}(?::\d{2}){1,2})\]', shift, text)

    def process_audio(
        self,
        audio_path: Path,
        custom_prompt: Optional[str] = None,
        known_attendees: Optional[list[dict]] = None,
        channel_segments: Optional[list] = None,
        diarization_segments: Optional[list] = None,
    ) -> GeminiResult:
        """Process an audio file with Gemini, returning transcript and analysis.

        Args:
            audio_path: Path to the audio file (MP3, WAV, etc.)
            custom_prompt: Optional custom prompt (default: AUDIO_ANALYSIS_PROMPT)
            known_attendees: Optional list of calendar attendee records
                (`{name, role, company, ...}`). When provided AND containing
                at least one non-self attendee, a "KNOWN ATTENDEES" block is
                prepended to the audio prompt so Gemini uses real names
                instead of guessing. Still forwarded to chunked audio (>60min)
                to prevent cross-chunk name drift.
            channel_segments: Optional [(t0, t1, 'host'|'remote'|'both')]
                channel-VAD segments (see channel_vad.compute_channel_vad).
                When provided, an "AUDIO CHANNEL MAP (GROUND TRUTH)" block is
                prepended so Gemini anchors host/remote attribution to the
                physical mic channel instead of voice similarity. Timestamps
                are meeting-global; the chunked path slices them per chunk.
            diarization_segments: Optional pyannote prior segments
                (`{start,end,label,confidence,level,overlapped}`). Timestamps
                are meeting-global; the chunked path slices them per chunk.

        Returns:
            GeminiResult with transcript and audio-derived signals.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        total_duration = self._get_duration(audio_path)

        # Long audio: chunk-and-merge (each chunk is a single-shot call,
        # stitched + reduce-pass).
        if total_duration > self.CHUNK_THRESHOLD_SEC:
            return self._process_chunked(
                audio_path, custom_prompt, total_duration,
                known_attendees=known_attendees,
                channel_segments=channel_segments,
                diarization_segments=diarization_segments,
            )

        # Short-enough audio: one single-shot call (no chunk boundaries → no
        # cross-chunk speaker drift/swap). If that fails on a recording long
        # enough to chunk, fall back to chunked rather than lose the whole
        # meeting to one disconnect.
        try:
            return self._process_single_shot(
                audio_path, custom_prompt, total_duration,
                known_attendees=known_attendees,
                channel_segments=channel_segments,
                diarization_segments=diarization_segments,
            )
        except Exception as e:
            if total_duration > self.CHUNK_DURATION_SEC:
                logger.warning(
                    f"Single-shot failed ({e}); falling back to chunked processing."
                )
                return self._process_chunked(
                    audio_path, custom_prompt, total_duration,
                    known_attendees=known_attendees,
                    channel_segments=channel_segments,
                    diarization_segments=diarization_segments,
                )
            raise

    def _process_single_shot(
        self,
        audio_path: Path,
        custom_prompt: Optional[str],
        total_duration: float,
        known_attendees: Optional[list[dict]] = None,
        channel_segments: Optional[list] = None,
        diarization_segments: Optional[list] = None,
    ) -> GeminiResult:
        """Transcribe an audio file in a single Gemini call (no chunking).

        Raises on upload/generation failure; the caller decides whether to
        fall back to chunked processing.
        """
        start_time = time.time()
        base_prompt = custom_prompt or AUDIO_ANALYSIS_PROMPT
        prompt = (
            _build_attendees_prefix(known_attendees)
            + _build_diarization_map_prefix(diarization_segments)
            + _build_channel_map_prefix(channel_segments)
            + base_prompt
        )

        file_size = audio_path.stat().st_size
        file_size_mb = file_size / 1024 / 1024
        logger.info(f"Processing audio: {audio_path.name}")
        logger.info(f"File size: {file_size_mb:.1f} MB")

        # Determine MIME type based on extension
        suffix = audio_path.suffix.lower()
        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
        }
        mime_type = mime_types.get(suffix, 'audio/mpeg')

        # Always use Files API for audio files (more reliable for uploads)
        # The inline approach can timeout on slower connections
        logger.info(f"Uploading audio to Gemini Files API ({file_size_mb:.1f} MB)...")

        try:
            uploaded_file = self.client.files.upload(
                file=str(audio_path),
                config={"mime_type": mime_type}
            )
            logger.debug(f"Uploaded file: {uploaded_file.name}")

            # Wait for file to be processed
            wait_count = 0
            while uploaded_file.state.name == "PROCESSING":
                wait_count += 1
                if wait_count > 60:  # Max 2 minutes waiting
                    raise RuntimeError("File processing timeout")
                logger.debug(f"Waiting for file processing... ({wait_count})")
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name != "ACTIVE":
                raise RuntimeError(f"File upload failed: {uploaded_file.state.name}")

            audio_content = uploaded_file
            logger.info("Audio uploaded successfully")

        except Exception as e:
            logger.error(f"Files API upload failed: {e}")
            # Fallback to inline data for small files
            if file_size_mb < 5:
                logger.info("Falling back to inline data...")
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                audio_content = self.types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                )
                uploaded_file = None
            else:
                raise

        # Generate content with audio, using streaming + retry to handle
        # server-side disconnects that occur on long audio requests.
        # Note: Do NOT use response_mime_type="application/json" — it causes
        # RemoteProtocolError on audio >15min. The prompt already requests JSON.
        logger.info(f"Generating transcript and analysis with {self.model}...")
        response_text, response = self._generate_with_retry(
            prompt=prompt,
            audio_content=audio_content,
            max_attempts=3,
        )

        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.1f}s")

        # Parse response
        result = self._parse_response(response, processing_time, response_text)
        result.audio_duration_seconds = total_duration

        # Clean up uploaded file if we used Files API
        if uploaded_file is not None:
            try:
                self.client.files.delete(name=uploaded_file.name)
                logger.debug("Cleaned up uploaded file")
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file: {e}")

        return result

    def _generate_with_retry(self, prompt: str, audio_content: Optional[Any] = None,
                              max_attempts: int = 3, validate_json: bool = True) -> tuple:
        """Stream generate_content with retry on transient disconnects.

        Gemini's server disconnects with RemoteProtocolError on ~30% of
        long-audio requests. Streaming + retry makes the pipeline reliable.

        Pass `audio_content=None` for a text-only call (used by the reduce-pass
        merge over chunked transcripts).

        With `validate_json=True` (default) a completed-but-unparseable JSON
        response is also retried, not just transport-level disconnects — a
        malformed/truncated Gemini reply used to silently drop the recording.

        Returns: (text, response) where response has usage_metadata and text.
        """
        contents = [prompt] if audio_content is None else [prompt, audio_content]
        last_error = None
        for attempt in range(1, max_attempts + 1):
            chunks = []
            try:
                stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                final_response = None
                for chunk in stream:
                    if chunk.text:
                        chunks.append(chunk.text)
                    final_response = chunk  # last chunk carries usage_metadata
                text = ''.join(chunks)
                if not text:
                    raise RuntimeError("Empty response from Gemini")
                if validate_json:
                    # A stream can complete yet still be unusable: truncated
                    # mid-JSON ("Unterminated string") or wrapped in prose
                    # ("Extra data"). Treat that like a disconnect — retry it —
                    # instead of returning unparseable text that silently
                    # dropped the recording downstream.
                    self._strip_and_parse_json(text)
                return text, final_response
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini attempt {attempt}/{max_attempts} failed after "
                    f"{len(''.join(chunks))} chars: {e}"
                )
                if attempt < max_attempts:
                    wait = 10 * attempt
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
        raise RuntimeError(f"All {max_attempts} attempts failed: {last_error}")

    def _shift_event_times(self, events: list[dict], offset_sec: float,
                           time_key: str = "time") -> list[dict]:
        """Shift `[MM:SS]` timestamps inside a list of dict events by offset.

        Used to convert per-chunk timestamps in `speaker_emotions[].arc[]`,
        `interruptions`, and `energy_levels[].arc[]` to global meeting time.
        """
        shifted = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            new_ev = dict(ev)
            t = new_ev.get(time_key)
            if isinstance(t, str) and t.startswith('['):
                new_ev[time_key] = self._shift_timestamps(t, offset_sec)
            shifted.append(new_ev)
        return shifted

    def _shift_chunk_times(self, chunk_result: 'GeminiResult', offset_sec: float) -> None:
        """In-place shift of all timestamps in a chunk result to global time."""
        # speaker_emotions: [{speaker, arc:[{time,...}]}]
        for entry in chunk_result.speaker_emotions:
            if isinstance(entry, dict) and 'arc' in entry:
                entry['arc'] = self._shift_event_times(entry['arc'], offset_sec)
        # interruptions: [{time, ...}]
        chunk_result.interruptions = self._shift_event_times(
            chunk_result.interruptions, offset_sec
        )
        # energy_levels: {speaker: {avg, arc:[{time, level}]}}
        for spk, data in chunk_result.energy_levels.items():
            if isinstance(data, dict) and 'arc' in data:
                data['arc'] = self._shift_event_times(data['arc'], offset_sec)

    def _process_chunked(self, audio_path: Path, custom_prompt: Optional[str],
                          total_duration: float,
                          known_attendees: Optional[list[dict]] = None,
                          channel_segments: Optional[list] = None,
                          diarization_segments: Optional[list] = None,
                          ) -> GeminiResult:
        """Chunk long audio, transcribe each chunk, merge with reduce-pass.

        Per-chunk analyses (sentiment, speaker_emotions, pacing, etc.) are NOT
        kept naively — that's what the prior implementation did and it was
        wrong on long meetings (chunk 1 wins). Instead, after all chunks come
        back, a single text-only Gemini call (REDUCE_PASS_PROMPT) sees the
        merged transcript + concatenated per-chunk analyses and produces one
        coherent meeting-wide view.

        `known_attendees` is forwarded to EVERY chunk's Gemini call so all
        chunks anchor to the same calendar attendee list — that's the fix
        for cross-chunk speaker-name drift. `channel_segments` and
        `diarization_segments` (meeting-global times) are sliced per chunk and
        re-based to chunk-relative time so their maps match each chunk's local
        [MM:SS] timestamps.
        """
        start_time = time.time()
        chunks = self._chunk_audio(audio_path)
        logger.info(f"Processing {len(chunks)} chunks of {audio_path.name}")

        slice_fn = None
        if channel_segments:
            try:
                from channel_vad import slice_segments as slice_fn
            except ImportError:
                logger.warning("channel_vad not importable; chunks get no map")
        diarize_slice_fn = None
        if diarization_segments:
            try:
                from diarize import slice_segments as diarize_slice_fn
            except ImportError:
                logger.warning("diarize not importable; chunks get no prior")

        merged_transcript_parts = []
        input_tokens_total = 0
        output_tokens_total = 0
        chunk_results: list['GeminiResult'] = []
        chunk_offsets: list[float] = []
        chunk_language = "unknown"
        chunk_participants: list[dict] = []

        for i, (chunk_path, offset) in enumerate(chunks):
            logger.info(f"  Chunk {i+1}/{len(chunks)} at offset {offset/60:.1f}min")
            chunk_segments = None
            chunk_diarization_segments = None
            if slice_fn is not None:
                # Single-chunk case: the "chunk" IS the whole file (single-
                # shot fallback for a 15-60 min recording) — slice to the
                # full duration, not CHUNK_DURATION_SEC, or the map would
                # silently truncate at 15:00 while claiming ground truth.
                chunk_end = (
                    total_duration if len(chunks) == 1
                    else offset + self.CHUNK_DURATION_SEC
                )
                try:
                    chunk_segments = slice_fn(
                        channel_segments, offset, chunk_end
                    ) or None
                except Exception as e:
                    logger.warning(f"channel map slice failed ({e}); no map")
            if diarize_slice_fn is not None:
                chunk_end = (
                    total_duration if len(chunks) == 1
                    else offset + self.CHUNK_DURATION_SEC
                )
                try:
                    chunk_diarization_segments = diarize_slice_fn(
                        diarization_segments, offset, chunk_end
                    ) or None
                except Exception as e:
                    logger.warning(
                        f"diarization prior slice failed ({e}); no prior"
                    )
            try:
                # Call the single-shot path directly (chunks are ≤15min;
                # the degenerate single chunk is the original file). NOT
                # process_audio: for the single-chunk fallback that would
                # recurse single-shot→chunked→single-shot indefinitely on
                # a deterministic failure. Forward known_attendees so every
                # chunk sees the same list.
                chunk_result = self._process_single_shot(
                    chunk_path, custom_prompt,
                    self._get_duration(chunk_path),
                    known_attendees=known_attendees,
                    channel_segments=chunk_segments,
                    diarization_segments=chunk_diarization_segments,
                )
            except Exception as e:
                logger.error(f"  Chunk {i+1} failed: {e}, continuing with others")
                merged_transcript_parts.append(f"\n\n[CHUNK {i+1} FAILED: {e}]\n\n")
                continue

            input_tokens_total += chunk_result.input_tokens
            output_tokens_total += chunk_result.output_tokens

            # Pick first non-empty language and participants list (these don't
            # benefit from reduce-pass; they're stable across chunks).
            if chunk_language == "unknown" and chunk_result.language not in (None, "", "unknown"):
                chunk_language = chunk_result.language
            if not chunk_participants and chunk_result.participants:
                chunk_participants = chunk_result.participants

            # Shift all timestamps to global time
            chunk_result.transcript = self._shift_timestamps(chunk_result.transcript, offset)
            self._shift_chunk_times(chunk_result, offset)

            merged_transcript_parts.append(chunk_result.transcript)
            chunk_results.append(chunk_result)
            chunk_offsets.append(offset)

        merged_transcript = '\n\n'.join(t for t in merged_transcript_parts if t.strip())

        # Cleanup chunk files
        import shutil
        if chunks and chunks[0][0] != audio_path:
            try:
                shutil.rmtree(chunks[0][0].parent)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        # All chunks failed — return a transcript-only error result
        if not chunk_results:
            processing_time = time.time() - start_time
            return GeminiResult(
                transcript=merged_transcript,
                language="unknown",
                processing_time_seconds=processing_time,
                model=self.model,
                chunked=True,
                chunk_count=len(chunks),
                reduce_pass_used=False,
                error="All chunks failed",
            )

        # Reduce-pass: text-only call to merge per-chunk analyses
        reduce_input = self._build_reduce_input(merged_transcript, chunk_results, chunk_offsets)
        reduce_data = None
        reduce_pass_ok = False
        try:
            logger.info(f"Running reduce-pass over {len(chunk_results)} chunk analyses...")
            reduce_text, reduce_response = self._generate_with_retry(
                prompt=REDUCE_PASS_PROMPT + "\n\n" + reduce_input,
                audio_content=None,
                max_attempts=2,
            )
            reduce_data = self._strip_and_parse_json(reduce_text)
            if reduce_response and reduce_response.usage_metadata:
                input_tokens_total += reduce_response.usage_metadata.prompt_token_count or 0
                output_tokens_total += reduce_response.usage_metadata.candidates_token_count or 0
            reduce_pass_ok = True
        except Exception as e:
            logger.warning(f"Reduce-pass failed ({e}); falling back to chunk-1 analysis")

        processing_time = time.time() - start_time
        logger.info(f"Chunked processing complete in {processing_time:.1f}s")

        # Build merged result. Prefer reduce-pass output; fall back to chunk-1
        # for whichever fields the reduce-pass didn't deliver.
        first = chunk_results[0]
        if reduce_data is None:
            reduce_data = {}

        return GeminiResult(
            transcript=merged_transcript,
            language=chunk_language or first.language or "unknown",
            participants=chunk_participants or first.participants,
            overall_sentiment=self._validate_sentiment(reduce_data.get("overall_sentiment", first.overall_sentiment)),
            sentiment_intensity=self._validate_intensity(reduce_data.get("sentiment_intensity", first.sentiment_intensity)),
            speaker_emotions=reduce_data.get("speaker_emotions") or self._merge_speaker_emotions(chunk_results),
            speaker_pacing=reduce_data.get("speaker_pacing") or self._merge_speaker_pacing(chunk_results),
            interruptions=reduce_data.get("interruptions") or self._merge_interruptions(chunk_results),
            energy_levels=reduce_data.get("energy_levels") or self._merge_energy_levels(chunk_results),
            input_tokens=input_tokens_total,
            output_tokens=output_tokens_total,
            audio_duration_seconds=total_duration,
            processing_time_seconds=processing_time,
            model=self.model,
            chunked=True,
            chunk_count=len(chunks),
            reduce_pass_used=reduce_pass_ok,
            raw_response={"chunks": [c.parsed_response for c in chunk_results], "reduce": reduce_data},
        )

    @staticmethod
    def _strip_and_parse_json(text: str) -> dict:
        """Strip markdown fences and parse JSON, salvaging two common Gemini
        formatting faults before giving up:
          - ```json fences around the object
          - prose before/after the object ("Extra data" / leading commentary),
            by extracting the outermost {...} span.
        Genuine truncation (no closing brace — "Unterminated string") still
        raises JSONDecodeError, which `_generate_with_retry` treats as a
        retryable generation failure rather than dropping the recording.
        """
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # Salvage a JSON object wrapped in prose / trailing commentary.
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end > start:
                return json.loads(s[start:end + 1])
            raise

    @staticmethod
    def _build_reduce_input(merged_transcript: str, chunk_results: list,
                             chunk_offsets: list[float]) -> str:
        """Render the input for the reduce-pass call.

        Caps the merged transcript at ~120k chars to stay well under the
        text-only context budget; the per-chunk analyses are usually small.
        """
        TRANSCRIPT_CAP = 120_000
        clipped = merged_transcript[:TRANSCRIPT_CAP]
        if len(merged_transcript) > TRANSCRIPT_CAP:
            clipped += f"\n\n[...transcript truncated at {TRANSCRIPT_CAP} chars for reduce-pass...]"
        chunk_blocks = []
        for cr, off in zip(chunk_results, chunk_offsets):
            block = {
                "chunk_offset_sec": off,
                "overall_sentiment": cr.overall_sentiment,
                "sentiment_intensity": cr.sentiment_intensity,
                "speaker_emotions": cr.speaker_emotions,
                "speaker_pacing": cr.speaker_pacing,
                "interruptions": cr.interruptions,
                "energy_levels": cr.energy_levels,
            }
            chunk_blocks.append(block)
        return (
            "## MERGED TRANSCRIPT\n\n" + clipped +
            "\n\n## PER-CHUNK ANALYSES\n\n" + json.dumps(chunk_blocks, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _merge_speaker_emotions(chunk_results: list) -> list[dict]:
        """Naive fallback: union arcs by speaker name (used if reduce-pass fails)."""
        by_speaker: dict[str, list] = {}
        for cr in chunk_results:
            for entry in cr.speaker_emotions:
                if not isinstance(entry, dict):
                    continue
                spk = entry.get("speaker")
                if not spk:
                    continue
                by_speaker.setdefault(spk, []).extend(entry.get("arc", []) or [])
        return [{"speaker": s, "arc": arc} for s, arc in by_speaker.items()]

    @staticmethod
    def _merge_speaker_pacing(chunk_results: list) -> dict:
        """Average wpm, sum hesitations, max longest_pause across chunks."""
        agg: dict[str, dict] = {}
        for cr in chunk_results:
            for spk, vals in (cr.speaker_pacing or {}).items():
                if not isinstance(vals, dict):
                    continue
                a = agg.setdefault(spk, {"wpm_sum": 0.0, "n": 0, "hesitation_count": 0, "longest_pause_sec": 0.0})
                a["wpm_sum"] += float(vals.get("wpm_avg") or 0)
                a["n"] += 1
                a["hesitation_count"] += int(vals.get("hesitation_count") or 0)
                a["longest_pause_sec"] = max(a["longest_pause_sec"], float(vals.get("longest_pause_sec") or 0))
        out = {}
        for spk, a in agg.items():
            out[spk] = {
                "wpm_avg": int(a["wpm_sum"] / a["n"]) if a["n"] else 0,
                "hesitation_count": a["hesitation_count"],
                "longest_pause_sec": a["longest_pause_sec"],
            }
        return out

    @staticmethod
    def _merge_interruptions(chunk_results: list) -> list[dict]:
        out = []
        for cr in chunk_results:
            for ev in (cr.interruptions or []):
                if isinstance(ev, dict):
                    out.append(ev)
        return out

    @staticmethod
    def _merge_energy_levels(chunk_results: list) -> dict:
        by_speaker: dict[str, dict] = {}
        for cr in chunk_results:
            for spk, data in (cr.energy_levels or {}).items():
                if not isinstance(data, dict):
                    continue
                slot = by_speaker.setdefault(spk, {"avg_counts": {"high": 0, "medium": 0, "low": 0}, "arc": []})
                avg = data.get("avg")
                if avg in slot["avg_counts"]:
                    slot["avg_counts"][avg] += 1
                slot["arc"].extend(data.get("arc", []) or [])
        out = {}
        for spk, slot in by_speaker.items():
            counts = slot["avg_counts"]
            avg = max(counts.items(), key=lambda kv: kv[1])[0] if any(counts.values()) else "medium"
            out[spk] = {"avg": avg, "arc": slot["arc"]}
        return out

    @staticmethod
    def _validate_sentiment(s: Any) -> str:
        return s if s in ("positive", "neutral", "negative", "mixed") else "neutral"

    @staticmethod
    def _validate_intensity(s: Any) -> str:
        return s if s in ("mild", "moderate", "strong") else "moderate"

    def _parse_response(self, response: Any, processing_time: float,
                         text: Optional[str] = None) -> GeminiResult:
        """Parse Gemini single-shot response into GeminiResult.

        Schema is the audio-only one (see AUDIO_ANALYSIS_PROMPT). Fields that
        used to live here (title, summary, key_points, tags, meeting_type,
        action_items, decisions_made, lailix_feedback) are no longer expected.
        """
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        if text is None:
            text = response.text
        logger.debug(f"Response length: {len(text)} chars")

        try:
            data = self._strip_and_parse_json(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Raw response: {text[:500]}...")
            return GeminiResult(
                transcript=text,
                language="unknown",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                processing_time_seconds=processing_time,
                model=self.model,
                error=f"JSONDecodeError: {e}",
                raw_response={"raw_text": text, "parse_error": str(e)},
            )

        return GeminiResult(
            transcript=data.get("transcript", ""),
            language=data.get("language", "unknown"),
            participants=data.get("participants", []),
            overall_sentiment=self._validate_sentiment(data.get("overall_sentiment", "neutral")),
            sentiment_intensity=self._validate_intensity(data.get("sentiment_intensity", "moderate")),
            speaker_emotions=data.get("speaker_emotions", []) or [],
            speaker_pacing=data.get("speaker_pacing", {}) or {},
            interruptions=data.get("interruptions", []) or [],
            energy_levels=data.get("energy_levels", {}) or {},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            processing_time_seconds=processing_time,
            model=self.model,
            raw_response=data,
        )

    def estimate_cost(self, audio_duration_seconds: float) -> dict:
        """Estimate the cost of processing audio.

        Args:
            audio_duration_seconds: Duration of audio in seconds

        Returns:
            Dict with estimated tokens and cost
        """
        # Gemini: 32 tokens per second of audio
        audio_tokens = int(audio_duration_seconds * 32)

        # Estimate output tokens (varies by meeting length)
        # Rough estimate: ~500 tokens per minute of transcript + 1000 for analysis
        estimated_output = int((audio_duration_seconds / 60) * 500 + 1000)

        # Gemini 2.5 Flash pricing (as of Jan 2025)
        # Input: $0.15 per 1M tokens (text), audio may vary
        # Output: $0.60 per 1M tokens
        input_cost = audio_tokens / 1_000_000 * 0.15
        output_cost = estimated_output / 1_000_000 * 0.60

        return {
            "audio_tokens": audio_tokens,
            "estimated_output_tokens": estimated_output,
            "estimated_input_cost_usd": input_cost,
            "estimated_output_cost_usd": output_cost,
            "estimated_total_cost_usd": input_cost + output_cost
        }


def process_audio_file(
    audio_path: Path,
    api_key: Optional[str] = None,
    known_attendees: Optional[list[dict]] = None,
    diarization_segments: Optional[list] = None,
) -> GeminiResult:
    """Convenience function to process an audio file.

    Args:
        audio_path: Path to the audio file
        api_key: Optional API key (default: from environment)
        known_attendees: Optional calendar attendee records — see
            GeminiAudioProcessor.process_audio for semantics.
        diarization_segments: Optional pyannote prior segments — see
            GeminiAudioProcessor.process_audio for semantics.

    Returns:
        GeminiResult with transcript and analysis
    """
    processor = GeminiAudioProcessor(api_key=api_key)
    return processor.process_audio(
        audio_path,
        known_attendees=known_attendees,
        diarization_segments=diarization_segments,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python gemini_processor.py <audio_file.mp3>")
        print("\nEnvironment: Set GEMINI_API_KEY")
        sys.exit(1)

    audio_file = Path(sys.argv[1])

    # Estimate cost first
    from audio_converter import get_audio_duration
    duration = get_audio_duration(audio_file)
    processor = GeminiAudioProcessor()
    cost = processor.estimate_cost(duration)

    print(f"\nAudio duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Estimated audio tokens: {cost['audio_tokens']:,}")
    print(f"Estimated cost: ${cost['estimated_total_cost_usd']:.4f}")

    # Ask for confirmation
    response = input("\nProceed with transcription? [y/N] ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Process
    result = processor.process_audio(audio_file)

    print(f"\n{'='*60}")
    print(f"Language: {result.language}")
    print(f"Overall sentiment: {result.overall_sentiment} ({result.sentiment_intensity})")
    print(f"Chunked: {result.chunked} (count={result.chunk_count}, reduce_pass={result.reduce_pass_used})")

    print(f"\nParticipants:")
    for p in result.participants:
        print(f"  - {p.get('name')} ({p.get('role')}): {p.get('speaking_pct')}% / {p.get('total_seconds')}s")

    print(f"\nSpeaker emotion arcs:")
    for entry in result.speaker_emotions:
        spk = entry.get("speaker", "?")
        arc = entry.get("arc", [])
        print(f"  {spk}: {len(arc)} events")
        for ev in arc[:3]:
            print(f"    {ev.get('time')} {ev.get('tone')}/{ev.get('energy')} — {ev.get('trigger')}")
        if len(arc) > 3:
            print(f"    ... +{len(arc)-3} more")

    print(f"\nPacing:")
    for spk, vals in (result.speaker_pacing or {}).items():
        print(f"  {spk}: {vals.get('wpm_avg')} wpm, {vals.get('hesitation_count')} hesitations, longest pause {vals.get('longest_pause_sec')}s")

    print(f"\nInterruptions: {len(result.interruptions or [])}")
    for ev in (result.interruptions or [])[:5]:
        print(f"  {ev.get('time')}: {ev.get('interrupter')} → {ev.get('interruptee')}")

    print(f"\n{'='*60}")
    print(f"Processing time: {result.processing_time_seconds:.1f}s")
    print(f"Input tokens: {result.input_tokens:,}")
    print(f"Output tokens: {result.output_tokens:,}")

    # Save transcript to file
    output_file = audio_file.with_suffix('.transcript.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.raw_response, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full result to: {output_file}")
