"""The configured Gemini deadline must reach the SDK client."""

from __future__ import annotations

import sys
import types as pytypes
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from gemini_processor import GeminiAudioProcessor  # noqa: E402


def test_configured_timeout_is_passed_to_sdk_http_options_in_milliseconds(monkeypatch):
    calls = {}

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            calls["http_options"] = kwargs
            self.timeout = kwargs["timeout"]

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs

    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    genai.Client = FakeClient
    sdk_types = pytypes.ModuleType("google.genai.types")
    sdk_types.HttpOptions = FakeHttpOptions
    genai.types = sdk_types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", sdk_types)

    processor = GeminiAudioProcessor(api_key="test-key", timeout_seconds=12.345)

    assert calls["http_options"]["timeout"] == 12_345
    assert calls["client"]["api_key"] == "test-key"
    assert calls["client"]["http_options"].timeout == 12_345
    assert processor.timeout_seconds == 12.345


def test_sdk_timeout_never_receives_zero_milliseconds(monkeypatch):
    calls = {}

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            calls["timeout"] = kwargs["timeout"]

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    genai.Client = FakeClient
    sdk_types = pytypes.ModuleType("google.genai.types")
    sdk_types.HttpOptions = FakeHttpOptions
    genai.types = sdk_types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", sdk_types)

    GeminiAudioProcessor(api_key="test-key", timeout_seconds=0.0001)

    assert calls["timeout"] == 1
