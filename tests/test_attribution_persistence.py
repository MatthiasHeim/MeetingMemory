import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import neon_insert
from attribution_gate import apply_calendar_bind_to_attribution
from attribution_gate import calendar_bind_ambiguous
from attribution_gate import gate
from attribution_gate import safe_calendar_publication
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


def test_malformed_json_retains_raw_text_insert_fallback(tmp_path, monkeypatch):
    path = tmp_path / "meeting.json"; path.write_text('{"transcript": "incomplete')
    conn = MagicMock(); cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (123,)
    monkeypatch.setattr(neon_insert, "_get_conn", lambda: conn)
    assert neon_insert.insert_source(transcript_path=str(path), title="Meeting") == 123
    params = cursor.execute.call_args.args[1]
    assert path.read_text() in params
    assert "speaker_attribution" not in json.loads(params[-1])


def test_held_counterpart_stays_out_of_calendar_and_notification(monkeypatch):
    import copy
    import transcribe_watcher as tw
    from gemini_processor import GeminiResult
    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = MagicMock()
    result = GeminiResult("[00:00] Speaker B: Hello", "en")
    result.speaker_attribution = {"speaker_dependent_actions": "hold"}
    verified = {"participant_details": [{"name": "Matthias Heim", "role": "self"}], "company": None}
    monkeypatch.setattr(tw, "SPEAKER_HINTS_AVAILABLE", True)
    monkeypatch.setattr(tw, "_detect_counterpart", lambda _: {
        "name": "Unverified Person", "company": "Unverified Company", "method": "text", "evidence": "guess"})
    inferred = watcher._infer_counterpart_if_unknown(result, copy.deepcopy(verified))
    assert inferred["company"] == "Unverified Company"
    published = watcher._calendar_for_publication(result, inferred, verified)
    assert published == verified
    monkeypatch.setattr(watcher, "_telegram_notify_script", lambda: "stub")
    run = MagicMock(); monkeypatch.setattr(tw.subprocess, "run", run)
    watcher._notify_telegram_meeting_captured(123, result, published, 600)
    msg = run.call_args.args[0][-1]
    assert "held" in msg and "running" not in msg and "Unverified" not in msg
    result.speaker_attribution = {"speaker_dependent_actions": "require_turn_evidence"}
    assert watcher._calendar_for_publication(result, inferred, verified) == inferred


def _calendar_match(match_count, counterpart="Tanja Example", company="itesys"):
    return {
        "participant_details": [
            {"name": "Matthias Heim", "role": "self", "company": "Lailix"},
            {"name": counterpart, "role": "participant", "company": company},
        ],
        "company": company,
        "calendar_event_id": "evt-chosen",
        "participant_resolution_log": {
            "calendar_search": {
                "match_count": match_count,
                "identity_authoritative": match_count == 1,
                "ambiguous": match_count > 1,
                "chosen_event_title": f"{counterpart} / Matthias",
            }
        },
    }


def test_ambiguous_calendar_bind_holds_named_extraction_and_hides_counterpart(tmp_path):
    """match_count > 1 is a candidate roster, not identity (2026-09-04 Tanja↔Sarah)."""
    collision = _calendar_match(2)
    data = {
        "participants": [{"name": "Tanja Example"}],
        "speaker_pacing": {"Tanja Example": {}},
        "_meta": {"speaker_attribution": {"speaker_dependent_actions": "require_turn_evidence"}},
        "participant_resolution_log": collision["participant_resolution_log"],
    }
    decision = gate(data)
    assert decision["speaker_dependent_actions"] == "hold"
    assert "calendar_identity_review" in decision["missing_stages"]
    assert not decision["allow_named_commitments"]

    report = apply_calendar_bind_to_attribution(
        {"speaker_dependent_actions": "require_turn_evidence", "status": "partially_checked"},
        collision["participant_resolution_log"]["calendar_search"],
    )
    assert report["speaker_dependent_actions"] == "hold"
    assert report["status"] == "needs_review"
    assert report["calendar_search"]["identity_authoritative"] is False

    path = tmp_path / "collision.json"
    path.write_text(json.dumps({
        **data,
        "_meta": {"speaker_attribution": report},
    }))
    import transcribe_watcher as tw
    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = MagicMock()
    watcher._trigger_claude = MagicMock()
    assert watcher._trigger_claude_if_attributed(path, 940) is False
    watcher._trigger_claude.assert_not_called()

    from gemini_processor import GeminiResult
    result = GeminiResult("[00:00] Speaker B: Hello", "de")
    result.speaker_attribution = report
    published = watcher._calendar_for_publication(result, collision, collision)
    assert published["company"] is None
    assert published["calendar_event_id"] is None
    assert [p["name"] for p in published["participant_details"]] == ["Matthias Heim"]
    search = published["participant_resolution_log"]["calendar_search"]
    assert search["match_count"] == 2 and search["ambiguous"] is True


def test_unique_calendar_bind_does_not_hold_on_calendar_alone(tmp_path):
    unique = _calendar_match(1)
    data = {
        "_meta": {"speaker_attribution": {"speaker_dependent_actions": "require_turn_evidence"}},
        "participant_resolution_log": unique["participant_resolution_log"],
    }
    assert gate(data)["speaker_dependent_actions"] == "require_turn_evidence"
    assert "calendar_identity_review" not in gate(data)["missing_stages"]
    assert calendar_bind_ambiguous(unique) is False
    assert safe_calendar_publication(unique) == unique

    path = tmp_path / "unique.json"
    path.write_text(json.dumps(data))
    import transcribe_watcher as tw
    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = MagicMock()
    watcher._trigger_claude = MagicMock()
    assert watcher._trigger_claude_if_attributed(path, 939) is True
    watcher._trigger_claude.assert_called_once()

    from gemini_processor import GeminiResult
    result = GeminiResult("[00:00] Speaker B: Hello", "de")
    result.speaker_attribution = {"speaker_dependent_actions": "require_turn_evidence"}
    assert watcher._calendar_for_publication(result, unique, unique) == unique


def test_empty_calendar_bind_does_not_hold_on_calendar_alone(tmp_path):
    empty = {
        "participant_details": [{"name": "Matthias Heim", "role": "self", "company": "Lailix"}],
        "company": None,
        "calendar_event_id": None,
        "participant_resolution_log": {
            "calendar_search": {
                "match_count": 0,
                "identity_authoritative": False,
                "ambiguous": False,
            }
        },
    }
    data = {
        "_meta": {"speaker_attribution": {"speaker_dependent_actions": "require_turn_evidence"}},
        "participant_resolution_log": empty["participant_resolution_log"],
    }
    assert gate(data)["speaker_dependent_actions"] == "require_turn_evidence"
    assert calendar_bind_ambiguous(empty) is False
    assert safe_calendar_publication(empty) == empty

    path = tmp_path / "empty.json"
    path.write_text(json.dumps(data))
    import transcribe_watcher as tw
    watcher = tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    watcher.logger = MagicMock()
    watcher._trigger_claude = MagicMock()
    assert watcher._trigger_claude_if_attributed(path, 100) is True

    from gemini_processor import GeminiResult
    result = GeminiResult("[00:00] Speaker B: Hello", "de")
    result.speaker_attribution = {"speaker_dependent_actions": "require_turn_evidence"}
    assert watcher._calendar_for_publication(result, empty, empty) == empty
