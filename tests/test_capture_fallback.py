import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
pytest.importorskip("rumps")
from meeting_recorder import AudioRecorder


def test_unfinished_system_header_preserves_mic_and_both_raw_files(tmp_path):
    recorder = AudioRecorder({"audio": {"archive_dir": str(tmp_path / "archive"), "compress_archives": False}})
    recorder.tmp_dir = tmp_path
    recorder._mic_wav = tmp_path / "mic.wav"
    recorder._sys_wav = tmp_path / "sys.wav"
    recorder._sys_wav.write_bytes(b"unfinished wav header"*100)
    recorder.output_file = tmp_path / "final.wav"
    recorder.audio_data = [np.full((8000,1), 1000, dtype=np.int16)]
    recorder.recording = True
    output = recorder.stop()
    data, _ = sf.read(output, dtype="int16")
    assert np.all(data == 1000) and len(data) == 8000
    assert len(list((tmp_path / "archive/final").glob("*.wav"))) == 2
