import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import neon_insert
from attribution_gate import gate
from attribution_gate import safe_enrichment


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


def test_held_source_does_not_publish_inferred_people_or_launch_actions(tmp_path):
    import transcribe_watcher as tw
    data = {"participants": [{"name": "Inferred Person"}], "speaker_pacing": {"Inferred Person": {}},
            "_meta": {"speaker_attribution": {"speaker_dependent_actions": "hold"}}}
    safe = safe_enrichment(data)
    assert safe["participants"] == [] and safe["speaker_pacing"] == {}
    assert data["participants"]  # immutable raw revision retained
    path = tmp_path / "held.json"; path.write_text(json.dumps(data))
    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = MagicMock(); watcher._trigger_claude = MagicMock()
    assert watcher._trigger_claude_if_attributed(path, 123) is False
    watcher._trigger_claude.assert_not_called()
    data["_meta"]["speaker_attribution"]["speaker_dependent_actions"] = "require_turn_evidence"
    path.write_text(json.dumps(data))
    assert watcher._trigger_claude_if_attributed(path, 123) is True
    watcher._trigger_claude.assert_called_once()


def test_hold_is_seeded_atomically_even_if_later_enrichment_fails(tmp_path, monkeypatch):
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps({"transcript": "Words", "_meta": {"speaker_attribution": {"speaker_dependent_actions": "hold"}}}))
    conn = MagicMock(); cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (123,)
    monkeypatch.setattr(neon_insert, "_get_conn", lambda: conn)
    neon_insert.insert_source(transcript_path=str(path), title="Meeting")
    metadata = json.loads(cursor.execute.call_args.args[1][-1])
    assert metadata["speaker_attribution"]["speaker_dependent_actions"] == "hold"
