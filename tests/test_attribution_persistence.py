import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import neon_insert
from attribution_gate import gate


def test_source_metadata_keeps_attribution_visible_to_db_only_consumers(monkeypatch):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    monkeypatch.setattr(neon_insert, "_get_conn", lambda: conn)
    report = {"status": "needs_review", "speaker_dependent_actions": "hold"}
    neon_insert.update_source_with_gemini(123, {"transcript": "Hi", "_meta": {"speaker_attribution": report}})
    sql, params = cursor.execute.call_args_list[0].args
    assert "COALESCE(metadata" in sql and "|| %s::jsonb" in sql
    assert json.loads(params[-2])["speaker_attribution"] == report
    assert params[-1] == 123


def test_missing_or_partial_attribution_does_not_authorize_named_commitments():
    for data in ({}, {"_meta": {"speaker_attribution": {"speaker_dependent_actions": "hold"}}}):
        assert not gate(data)["allow_named_commitments"]
        assert gate(data)["allow_content_only_extraction"]
