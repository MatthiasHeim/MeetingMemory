import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compress_capture


def test_corrupt_compression_copy_never_replaces_source(tmp_path, monkeypatch):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"original audio"*1000)
    original = source.read_bytes()
    def corrupt(args, **kwargs):
        if args[0] == "/usr/bin/ditto":
            Path(args[-1]).write_bytes(b"corrupt")
    monkeypatch.setattr(compress_capture.subprocess, "run", corrupt)
    with pytest.raises(RuntimeError, match="byte verification"):
        compress_capture.compress(source)
    assert source.read_bytes() == original
    assert not source.with_name("audio.wav.compression.tmp").exists()
