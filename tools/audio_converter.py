#!/usr/bin/env python3
"""
Audio Converter - WAV to MP3 conversion for Gemini audio processing.

Extracts the microphone channel (channel 3) from multi-channel recordings
and converts to high-quality MP3 for efficient upload to Gemini API.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Absolute paths for launchd compatibility
FFMPEG_PATH = '/opt/homebrew/bin/ffmpeg'
FFPROBE_PATH = '/opt/homebrew/bin/ffprobe'
SILENCE_THRESHOLD_DB = -60.0

TOPOLOGY_SINGLE_SOURCE = "single_source"
TOPOLOGY_MULTI_SOURCE_GENUINE = "multi_source_genuine"
TOPOLOGY_UNKNOWN = "unknown"

# Ducked pre-mix defaults (2026-08-11 gating spike, see
# convert_for_gemini_ducked docstring). Sample rate matches MeetingRecorder's
# native capture rate; the other four were the winning values in the
# controlled A/B on the 2026-08-10 recording.
DUCK_SAMPLE_RATE = 48000
DUCK_DB_DEFAULT = 18.0
DUCK_ATTACK_MS_DEFAULT = 50.0
DUCK_RELEASE_MS_DEFAULT = 120.0
DUCK_HANGOVER_MS_DEFAULT = 250.0
DUCK_MIN_DWELL_MS_DEFAULT = 400.0


@dataclass
class AudioInfo:
    """Information about an audio file."""
    channels: int
    sample_rate: int
    duration_seconds: float
    file_size_bytes: int


@dataclass
class SourceTopologyInfo:
    """Per-channel activity summary used to classify capture topology."""

    total_channels: int
    active_channels: list[int]
    channel_mean_db: dict[int, float]
    topology: str


def get_audio_info(audio_path: Path) -> AudioInfo:
    """Get detailed information about an audio file using ffprobe.

    Args:
        audio_path: Path to the audio file

    Returns:
        AudioInfo with channels, sample rate, duration, and file size

    Raises:
        RuntimeError: If ffprobe fails
    """
    cmd = [
        FFPROBE_PATH, '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=channels,sample_rate:format=duration',
        '-of', 'csv=p=0:s=,',
        str(audio_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    # Parse output: "sample_rate,channels\nduration"
    lines = result.stdout.strip().split('\n')
    stream_info = lines[0].split(',')

    sample_rate = int(stream_info[0])
    channels = int(stream_info[1])
    duration = float(lines[1]) if len(lines) > 1 else 0.0

    return AudioInfo(
        channels=channels,
        sample_rate=sample_rate,
        duration_seconds=duration,
        file_size_bytes=audio_path.stat().st_size
    )


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in seconds
    """
    cmd = [
        FFPROBE_PATH, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def get_channel_count(audio_path: Path) -> int:
    """Get the number of audio channels.

    Args:
        audio_path: Path to the audio file

    Returns:
        Number of channels
    """
    cmd = [
        FFPROBE_PATH, '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=channels',
        '-of', 'csv=p=0',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return int(result.stdout.strip())


def probe_channel_activity(audio_path: Path, probe_seconds: int = 60,
                           silence_db: float = SILENCE_THRESHOLD_DB
                           ) -> SourceTopologyInfo:
    """Probe per-channel mean volume and classify source topology.

    Some macOS capture pipelines (e.g. BlackHole with unusual routing) put
    audio on only one of the 3 channels, with the other two at digital silence.
    Equal-weight pre-mixing attenuates the real signal; extracting only the
    active channel(s) preserves Swiss German ASR accuracy.

    Topology is intentionally conservative:
      - single_source: zero/one active channel, including 3-channel files with
        only ch0 active (in-room single mic) or only one routed channel active.
      - multi_source_genuine: MeetingRecorder hybrid layout where ch0 (host
        mic) and at least one system channel (ch1+) are both active.
      - unknown: multiple active channels, but not the host+system shape.
    """
    import re
    channels = get_channel_count(audio_path)
    active = []
    channel_mean_db: dict[int, float] = {}
    for i in range(channels):
        p = subprocess.run(
            [FFMPEG_PATH, '-i', str(audio_path), '-t', str(probe_seconds),
             '-af', f'pan=mono|c0=c{i},volumedetect', '-f', 'null', '-'],
            capture_output=True, text=True,
        )
        m = re.search(r'mean_volume:\s*(-?[0-9.]+)\s*dB', p.stderr)
        mean_db = float(m.group(1)) if m else float('-inf')
        channel_mean_db[i] = mean_db
        logger.debug(f"  channel {i}: mean_volume={mean_db} dB")
        if mean_db > silence_db:
            active.append(i)

    if len(active) <= 1:
        topology = TOPOLOGY_SINGLE_SOURCE
    elif channels >= 3 and 0 in active and any(i >= 1 for i in active):
        topology = TOPOLOGY_MULTI_SOURCE_GENUINE
    else:
        topology = TOPOLOGY_UNKNOWN

    return SourceTopologyInfo(
        total_channels=channels,
        active_channels=active,
        channel_mean_db=channel_mean_db,
        topology=topology,
    )


def classify_source_topology(audio_path: Path, probe_seconds: int = 60,
                             silence_db: float = SILENCE_THRESHOLD_DB
                             ) -> SourceTopologyInfo:
    """Return active-channel info and source topology for a recording."""
    return probe_channel_activity(audio_path, probe_seconds, silence_db)


def detect_active_channels(audio_path: Path, probe_seconds: int = 60,
                            silence_db: float = SILENCE_THRESHOLD_DB) -> list:
    """Detect which channels contain audio above the silence threshold."""
    info = probe_channel_activity(audio_path, probe_seconds, silence_db)
    active = info.active_channels
    return active


def _run_loudnorm_mp3(
    input_path: Path,
    output_path: Path,
    quality: int = 2,
    prefilters: Optional[list] = None,
) -> Path:
    """Shared MP3-encode tail: EBU R128 loudness normalization + libmp3lame.

    Both conversion routes (`convert_to_mp3`'s channel-extraction mix and
    `convert_for_gemini_ducked`'s numpy-mixed intermediate WAV) funnel
    through this one function so they cannot drift apart on the loudness
    target or encoder settings — `voice_refs/matthias.mp3`, the reference
    clip used for voice fingerprinting, was normalized to the same -16 LUFS
    target and a mismatch here would silently reintroduce the ASR-accuracy
    gap the original loudnorm addition fixed.

    `prefilters` are audio filters (e.g. the `pan=mono|...` channel
    extraction) applied BEFORE loudnorm, in the same `-af` chain — pass
    `None`/`[]` when the input is already a mixed-down mono WAV.
    """
    audio_filters = list(prefilters or [])
    audio_filters.append('loudnorm=I=-16:LRA=11:TP=-1.5')

    cmd = [FFMPEG_PATH, '-y', '-i', str(input_path)]
    cmd.extend(['-af', ','.join(audio_filters)])
    cmd.extend([
        '-codec:a', 'libmp3lame',
        '-qscale:a', str(quality),
        str(output_path)
    ])

    logger.debug(f"FFmpeg command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    return output_path


def convert_to_mp3(
    input_path: Path,
    output_path: Optional[Path] = None,
    extract_channel: Optional[int] = None,
    quality: int = 2
) -> Path:
    """Convert audio file to high-quality MP3 optimized for speech recognition.

    For MeetingRecorder 3-channel audio:
    - Channel 0-1: System audio (stereo)
    - Channel 2: Microphone (mono) <- This is what we want

    Args:
        input_path: Path to input audio file (WAV, etc.)
        output_path: Path for output MP3 (default: same name with .mp3)
        extract_channel: Channel index to extract (0-based).
                        If None, auto-detects: extracts channel 2 for 3-channel,
                        or mixes all channels for stereo.
        quality: LAME quality setting (0=best, 9=worst). Default 2 (~190kbps VBR)

    Returns:
        Path to the converted MP3 file

    Raises:
        RuntimeError: If ffmpeg conversion fails
        FileNotFoundError: If input file doesn't exist
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix('.mp3')
    else:
        output_path = Path(output_path)

    # Get channel count to determine extraction strategy
    num_channels = get_channel_count(input_path)
    logger.info(f"Input file has {num_channels} channels")

    # Determine channel extraction
    if extract_channel is not None:
        # User specified a channel
        active_channels = [extract_channel]
        logger.info(f"Using user-specified channel {extract_channel}")
    elif num_channels == 1:
        active_channels = None
        logger.info("Input is mono, no channel extraction needed")
    elif num_channels >= 3:
        # Detect active channels — BlackHole routing sometimes puts audio on
        # only one of 3 channels; averaging silent channels attenuates signal.
        active_channels = detect_active_channels(input_path)
        if not active_channels:
            logger.warning(f"No active channels detected, falling back to equal-weight mix")
            active_channels = list(range(num_channels))
        else:
            logger.info(f"Detected active channels: {active_channels}")
    else:
        # Stereo — standard case, downmix with -ac 1
        active_channels = None
        logger.info(f"Input has {num_channels} channels, will mix to mono")

    # Compose the pre-loudnorm filter chain. EBU R128 loudness normalization
    # is appended unconditionally (inside _run_loudnorm_mp3) because mic gain
    # in MeetingRecorder captures varies by ~14 dB between sessions, and
    # low-gain takes push Swiss German ASR below threshold. Target -16 LUFS
    # matches the reference clip used for voice fingerprinting
    # (voice_refs/matthias.mp3).
    audio_filters: list[str] = []
    if active_channels and len(active_channels) == 1:
        audio_filters.append(f'pan=mono|c0=c{active_channels[0]}')
    elif active_channels and len(active_channels) > 1:
        weight = 1.0 / len(active_channels)
        audio_filters.append(
            'pan=mono|c0=' + '+'.join(f'{weight}*c{i}' for i in active_channels)
        )
    elif num_channels > 1:
        # No active-channel detection ran (stereo path) — do the mono
        # downmix inside the filter chain so loudnorm sees mono input.
        weight = 1.0 / num_channels
        audio_filters.append(
            'pan=mono|c0=' + '+'.join(f'{weight}*c{i}' for i in range(num_channels))
        )
    # else: mono input — no channel filter needed

    _run_loudnorm_mp3(input_path, output_path, quality=quality, prefilters=audio_filters)

    output_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Converted {input_path.name} -> {output_path.name} ({output_size_mb:.1f} MB)")

    return output_path


def convert_for_gemini(
    input_path: Path,
    output_dir: Optional[Path] = None
) -> Path:
    """Convert audio file optimized for Gemini API upload.

    This is the main entry point for the Gemini audio pipeline.
    It extracts the mic channel from MeetingRecorder recordings
    and converts to high-quality MP3.

    Args:
        input_path: Path to input WAV file
        output_dir: Directory for output MP3 (default: same directory as input)

    Returns:
        Path to the converted MP3 file
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_path = input_path.with_suffix('.mp3')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.mp3"

    return convert_to_mp3(
        input_path=input_path,
        output_path=output_path,
        extract_channel=None,  # Auto-detect
        quality=2  # High quality for clear transcription
    )


def _decode_raw_pcm(input_path: Path, channels: int, sr: int = DUCK_SAMPLE_RATE):
    """Decode to raw, UNfiltered float32 frames (n_samples, channels) at `sr`.

    Unlike `channel_vad._decode_bandpassed`, this applies no band-pass: the
    duck gate multiplies the same raw ch0 samples that get mixed and shipped
    to Gemini, so what we duck must be the exact audio Gemini hears, not a
    speech-band-filtered analysis copy.
    """
    import numpy as np
    cmd = [
        FFMPEG_PATH, '-v', 'error', '-i', str(input_path),
        '-ar', str(sr), '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1',
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg raw decode failed: {result.stderr.decode(errors='replace')[:400]}"
        )
    data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n_frames = len(data) // channels
    return data[: n_frames * channels].reshape(n_frames, channels)


def _dilate_forward(active, n_windows: int):
    """OR in 'was active up to n_windows ago' — extends True runs forward in
    time. Used as the duck's hangover: keep ch0 attenuated briefly after
    remote speech ends so a trailing host word doesn't get clipped by an
    early release.
    """
    out = active.copy()
    for shift in range(1, n_windows + 1):
        out[shift:] |= active[:-shift]
    return out


def _debounce_min_dwell(binary, min_dwell_windows: int):
    """Absorb a reopen (True -> False) shorter than `min_dwell_windows`
    back into the preceding closed (True) run.

    Gating spike caveat (unproven as causal, cheap insurance): a runaway
    Gemini repetition loop appeared near a rapidly-toggling host/remote
    zone on the 32-min test file. Sub-second open/close flutter modulates
    ch0's gain rapidly enough to look like a glitch; this debounce enforces
    a minimum dwell time in the closed state before the gate is allowed to
    reopen, so a burst of sub-threshold toggles collapses into one longer
    closed run instead of chattering. Mirrors the same forward-merge shape
    as `channel_vad._segments_from_labels`'s silence-gap bridging.
    """
    import numpy as np
    if min_dwell_windows <= 0 or len(binary) == 0:
        return binary
    change_idx = np.flatnonzero(np.diff(binary.astype(np.int8)) != 0) + 1
    bounds = [0] + change_idx.tolist() + [len(binary)]
    out = np.zeros(len(binary), dtype=bool)
    cur_start = 0
    cur_val = bool(binary[0])
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        v = bool(binary[s])
        if v == cur_val:
            continue  # contiguous run of the same value — just extends
        if (not v) and cur_val and (e - s) < min_dwell_windows:
            continue  # short reopen — absorb into the ongoing closed run
        out[cur_start:s] = cur_val
        cur_start, cur_val = s, v
    out[cur_start:len(binary)] = cur_val
    return out


def build_duck_gain_envelope(
    remote_active,
    window_sec: float,
    n_samples: int,
    sr: int = DUCK_SAMPLE_RATE,
    duck_db: float = DUCK_DB_DEFAULT,
    attack_ms: float = DUCK_ATTACK_MS_DEFAULT,
    release_ms: float = DUCK_RELEASE_MS_DEFAULT,
    hangover_ms: float = DUCK_HANGOVER_MS_DEFAULT,
    min_dwell_ms: float = DUCK_MIN_DWELL_MS_DEFAULT,
):
    """Sample-level gain envelope for channel 0 (host mic).

    1.0 while the remote channel (ch1/ch2) is silent, attenuated by
    `duck_db` while it's active (+hangover, +min-dwell debounce), with
    linear attack/release ramps at every transition. Ported from the
    2026-08-11 gating-spike prototype (Brain scratchpad
    gating-spike/build_variants.py + build_full.py), which A/B-tested this
    exact envelope shape against a hard gate (mute) and a system-audio-only
    mix on a real bleeding-mic recording — see convert_for_gemini_ducked's
    docstring for the results.

    Args:
        remote_active: bool array, one entry per `window_sec`-wide window
            (channel_vad.ChannelVAD.remote_active).
        window_sec: window width in seconds (channel_vad.ChannelVAD.window_sec).
        n_samples: length of the output gain array, in samples at `sr`.
        sr: sample rate the gain envelope is expressed at.
        duck_db: attenuation applied to ch0 while remote is active.
        attack_ms / release_ms: ramp duration entering / leaving the duck.
        hangover_ms: extend the duck this long past the end of remote speech.
        min_dwell_ms: minimum time the duck must stay engaged before it's
            allowed to reopen (see `_debounce_min_dwell`).

    Returns:
        float32 ndarray of length `n_samples`, values in [floor, 1.0].
    """
    import numpy as np
    floor = 10 ** (-abs(duck_db) / 20.0)

    remote_active = np.asarray(remote_active, dtype=bool)
    hang_windows = int(round(hangover_ms / 1000.0 / window_sec))
    dilated = _dilate_forward(remote_active, hang_windows)
    min_dwell_windows = int(round(min_dwell_ms / 1000.0 / window_sec))
    debounced = _debounce_min_dwell(dilated, min_dwell_windows)

    win_samples = int(round(window_sec * sr))
    binary = np.repeat(debounced.astype(np.int8), win_samples)
    if len(binary) < n_samples:
        binary = np.pad(binary, (0, n_samples - len(binary)), mode="edge")
    else:
        binary = binary[:n_samples]

    gain = np.where(binary.astype(bool), floor, 1.0).astype(np.float32)
    edges = np.flatnonzero(np.diff(binary) != 0) + 1
    attack_n = max(1, int(round(attack_ms / 1000.0 * sr)))
    release_n = max(1, int(round(release_ms / 1000.0 * sr)))
    n = len(gain)
    for e in edges:
        entering_duck = bool(binary[e])
        if entering_duck:
            n_ramp = min(attack_n, n - e)
            start_val, end_val = 1.0, floor
        else:
            n_ramp = min(release_n, n - e)
            start_val, end_val = floor, 1.0
        if n_ramp <= 0:
            continue
        seg = np.linspace(start_val, end_val, n_ramp, endpoint=True, dtype=np.float32)
        gain[e:e + n_ramp] = seg
    return gain


def convert_for_gemini_ducked(
    input_path: Path,
    channel_vad,
    output_dir: Optional[Path] = None,
    duck_db: float = DUCK_DB_DEFAULT,
    attack_ms: float = DUCK_ATTACK_MS_DEFAULT,
    release_ms: float = DUCK_RELEASE_MS_DEFAULT,
    hangover_ms: float = DUCK_HANGOVER_MS_DEFAULT,
    min_dwell_ms: float = DUCK_MIN_DWELL_MS_DEFAULT,
    quality: int = 2,
) -> Path:
    """Convert a bleeding-mic 3-channel recording for Gemini, ducking ch0
    (host mic) instead of mixing it in at full level.

    Why this exists: `convert_for_gemini`'s equal-weight pre-mix
    (`pan=mono|c0=0.333*c0+0.333*c1+0.333*c2`) assumes ch0 (host mic) and
    ch1/ch2 (system audio) carry independent content. When the host records
    without headphones, the mic also picks up the loudspeakers — so that
    mix sends every remote utterance to Gemini TWICE: once clean from the
    system tap, once smeared through room acoustics via the mic. A
    controlled A/B on a real bleeding-mic recording (2026-08-11 gating
    spike, Brain scratchpad `gating-spike/`) showed this doubling produces
    duplicated lines AND phantom speakers — the echo gets transcribed as a
    second person, e.g.
    `[02:46] Matthias: "...wie du gseit häsch."` /
    `[02:48] Speaker 2: "Wie du gseit häsch."`. Ducking ch0 whenever ch1/ch2
    is active collapsed near-duplicate pairs from 3 to 0 on the test
    excerpt, and host content was NOT lost — host word count in the
    heavy-bleed zone rose to 105-117% of baseline (removing the muddying
    doubled audio helps Gemini parse overlap, it doesn't starve it).

    Echo *cancellation* was separately proven impossible on this hardware
    (mic/system sample-level coherence ~0.001, from non-linear macOS mic
    AGC) — gating works instead because it only needs to know *when* the
    remote is speaking, which channel_vad already measures independently.

    Only call this for `channel_admissible=False` recordings (see
    `transcribe_watcher._process_with_gemini`); clean/headphone recordings
    keep using `convert_for_gemini` completely unchanged.

    Args:
        input_path: original 3-channel WAV (ch0=host mic, ch1/ch2=system).
        channel_vad: a `channel_vad.ChannelVAD` computed from THIS wav
            (needs `.remote_active` and `.window_sec`).
        output_dir: directory for the output MP3 (default: alongside input).
        duck_db, attack_ms, release_ms, hangover_ms, min_dwell_ms: see
            `build_duck_gain_envelope`.
        quality: LAME quality (0=best, 9=worst); default matches `convert_to_mp3`.

    Returns:
        Path to the converted MP3 file.

    Raises:
        FileNotFoundError: if input_path doesn't exist.
        ValueError: if input_path has fewer than 3 channels.
        RuntimeError: if ffmpeg decode/encode fails.
    """
    import numpy as np
    import soundfile as sf

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_dir is None:
        output_path = input_path.with_suffix('.mp3')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.mp3"

    channels = get_channel_count(input_path)
    if channels < 3:
        raise ValueError(
            f"{input_path.name} has {channels} channel(s); ducked conversion "
            "needs the 3-channel host-mic/system-audio hybrid layout"
        )

    frames = _decode_raw_pcm(input_path, channels=channels, sr=DUCK_SAMPLE_RATE)
    n = len(frames)
    c0, c1, c2 = frames[:, 0], frames[:, 1], frames[:, 2]

    gain = build_duck_gain_envelope(
        remote_active=channel_vad.remote_active,
        window_sec=channel_vad.window_sec,
        n_samples=n,
        sr=DUCK_SAMPLE_RATE,
        duck_db=duck_db,
        attack_ms=attack_ms,
        release_ms=release_ms,
        hangover_ms=hangover_ms,
        min_dwell_ms=min_dwell_ms,
    )
    mix = (gain * c0 + c1 + c2) / 3.0

    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.999:
        mix = mix * (0.999 / peak)
        logger.debug(f"convert_for_gemini_ducked: normalized peak {peak:.3f} -> 0.999")

    tmp_wav = output_path.with_name(f"{output_path.stem}.ducked_tmp.wav")
    try:
        sf.write(str(tmp_wav), mix.astype(np.float32), DUCK_SAMPLE_RATE, subtype='PCM_16')
        _run_loudnorm_mp3(tmp_wav, output_path, quality=quality)
    finally:
        tmp_wav.unlink(missing_ok=True)

    output_size_mb = output_path.stat().st_size / 1024 / 1024
    bleed = channel_vad.host_bleed_rate() if hasattr(channel_vad, "host_bleed_rate") else None
    bleed_str = f"{bleed:.2f}" if bleed is not None else "n/a"
    logger.info(
        f"Converted (ducked -{abs(duck_db):.0f}dB, host_bleed_rate={bleed_str}): "
        f"{input_path.name} -> {output_path.name} ({output_size_mb:.1f} MB)"
    )
    return output_path


if __name__ == "__main__":
    # Simple test
    import sys
    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python audio_converter.py <input.wav> [output.mp3]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Show audio info
    info = get_audio_info(input_file)
    print(f"Audio info: {info.channels} channels, {info.sample_rate}Hz, {info.duration_seconds:.1f}s")

    # Convert
    result = convert_to_mp3(input_file, output_file)
    print(f"Output: {result}")
