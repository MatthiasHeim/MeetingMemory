"""voice_library: the local, biometric, never-leaves-the-machine store."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import voice_library as vl  # noqa: E402
from voice_library import (  # noqa: E402
    BASIS_HUMAN, BASIS_SELF, LibraryError, PersonConflict, VoiceLibrary, normalise_name,
)
from voice_synth import base_vectors, noisy  # noqa: E402

B = base_vectors(["Person A", "Person B", "Person C", "host"])


def _stem(i):
    return f"2026-10-{i:02d}_10-00-00"


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def test_round_trip_survives_a_new_instance(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    res = lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=9, basis=BASIS_HUMAN,
                         crm_contact_id=42)
    assert res["created"] and res["samples"] == 1
    prof = VoiceLibrary(tmp_path / "lib").profiles()
    assert [(p.name, p.role, p.crm_contact_id) for p in prof] == [("Person A", "other", 42)]
    assert prof[0].centroids.shape == (1, B["Person A"].size)
    assert np.isclose(np.linalg.norm(prof[0].centroids[0]), 1.0)
    assert np.allclose(prof[0].centroids[0], B["Person A"], atol=1e-6)
    row = VoiceLibrary(tmp_path / "lib").people()[0]["samples"][0]
    assert (row["stem"], row["segments"], row["basis"]) == (_stem(1), 9, BASIS_HUMAN)


def test_permissions_are_private(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    lib.add_sample("Matthias Heim", B["host"], stem=_stem(1), segments=5, basis=BASIS_SELF, role="self")
    assert _mode(lib.root) == 0o700 and _mode(lib.vectors_dir) == 0o700
    files = [p for p in lib.root.rglob("*") if p.is_file()]
    assert len(files) >= 4                    # index, lock, two vectors
    assert all(_mode(p) == 0o600 for p in files), {str(p): oct(_mode(p)) for p in files}


def test_reading_never_creates_the_directory(tmp_path):
    lib = VoiceLibrary(tmp_path / "nope")
    assert lib.profiles() == [] and lib.people() == [] and not lib.has_others()
    assert lib.stats()["people"] == 0
    assert not (tmp_path / "nope").exists()


def test_forget_one_person_deletes_vectors_and_entry(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    lib.add_sample("Person B", B["Person B"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    assert len(list(lib.vectors_dir.glob("*.npy"))) == 2
    assert lib.forget("  person   a ") == 1          # normalised
    assert [p["name"] for p in lib.people()] == ["Person B"]
    assert len(list(lib.vectors_dir.glob("*.npy"))) == 1
    assert lib.forget("Nobody") == 0


def test_forget_all_and_cli_requires_yes(tmp_path, capsys):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    assert vl.main(["--library-dir", str(lib.root), "forget", "--all"]) == 2
    assert lib.people()                       # refused, nothing deleted
    assert vl.main(["--library-dir", str(lib.root), "forget", "--all", "--yes"]) == 0
    assert lib.people() == [] and not list(lib.vectors_dir.glob("*.npy"))


def test_keeps_only_the_most_recent_n(tmp_path, monkeypatch):
    monkeypatch.setattr(vl, "MAX_SAMPLES_PER_PERSON", 3)
    lib = VoiceLibrary(tmp_path / "lib")
    for i in (5, 1, 4, 2, 3):                 # inserted out of order
        lib.add_sample("Person A", noisy(B["Person A"], 1, seed=i)[0], stem=_stem(i), segments=5,
                       basis=BASIS_HUMAN)
    stems = [s["stem"] for s in lib.people()[0]["samples"]]
    assert stems == [_stem(3), _stem(4), _stem(5)]
    assert len(list(lib.vectors_dir.glob("*.npy"))) == 3      # dropped vectors are deleted


def test_same_recording_replaces_instead_of_duplicating(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    res = lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=8, basis=BASIS_HUMAN)
    assert res["replaced"] and res["samples"] == 1
    assert len(list(lib.vectors_dir.glob("*.npy"))) == 1


def test_exclude_stem_leaves_the_recordings_own_enrolment_out(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    assert lib.profiles(exclude_stem=_stem(1)) == []
    assert not lib.has_others(exclude_stem=_stem(1)) and lib.has_others()


def test_write_is_atomic(tmp_path, monkeypatch):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    before = lib.index_path.read_bytes()
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        lib.add_sample("Person B", B["Person B"], stem=_stem(2), segments=5, basis=BASIS_HUMAN)
    monkeypatch.setattr(os, "replace", real)
    assert lib.index_path.read_bytes() == before                       # old index intact
    assert not [p for p in lib.root.rglob("*.tmp")]                    # no partial file left
    assert [p["name"] for p in lib.people()] == ["Person A"]


def test_never_merges_people(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN, crm_contact_id=1)
    # Near-identical names are different people (no fuzzy matching).
    lib.add_sample("Person Ab", B["Person B"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    assert sorted(p["name"] for p in lib.people()) == ["Person A", "Person Ab"]
    # Same name, a different CRM contact: refuse rather than merge.
    with pytest.raises(PersonConflict):
        lib.add_sample("Person A", B["Person C"], stem=_stem(2), segments=5, basis=BASIS_HUMAN,
                       crm_contact_id=2)
    assert len(lib.people()) == 2 and len(lib.people()[0]["samples"]) == 1


def test_crm_id_is_preferred_over_the_name(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN, crm_contact_id=7)
    lib.add_sample("Person A Renamed", B["Person A"], stem=_stem(2), segments=5, basis=BASIS_HUMAN,
                   crm_contact_id=7)
    people = lib.people()
    assert len(people) == 1 and people[0]["name"] == "Person A Renamed" and len(people[0]["samples"]) == 2


def test_name_matching_is_exact_after_normalisation(tmp_path):
    assert normalise_name("  Zoë   MÜLLER ") == "zoë müller"
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Zoë Müller", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    lib.add_sample("zoë  müller", B["Person A"], stem=_stem(2), segments=5, basis=BASIS_HUMAN)
    lib.add_sample("Zoe Muller", B["Person B"], stem=_stem(3), segments=5, basis=BASIS_HUMAN)  # no accent folding
    assert sorted(len(p["samples"]) for p in lib.people()) == [1, 2]


def test_host_is_never_stored_as_another_person(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    with pytest.raises(PersonConflict):
        lib.add_sample("Matthias Heim", B["host"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    lib.add_sample("Matthias Heim", B["host"], stem=_stem(1), segments=5, basis=BASIS_SELF, role="self")
    lib.add_sample("ignored name", B["host"], stem=_stem(2), segments=5, basis=BASIS_SELF, role="self")
    people = lib.people()
    assert len(people) == 1 and people[0]["role"] == "self" and len(people[0]["samples"]) == 2


def test_bad_input_is_refused_before_anything_is_written(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    for kw in ({"vector": np.zeros(8)}, {"vector": np.array([np.nan] * 8)}, {"basis": "guess"},
               {"name": "  "}, {"stem": "not-a-stem"}):
        args = dict(name="Person A", vector=B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
        args.update(kw)
        with pytest.raises(ValueError):
            lib.add_sample(args.pop("name"), args.pop("vector"), **args)
    assert not lib.index_path.exists()


def test_dimension_mismatch_and_corrupt_index(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    with pytest.raises(LibraryError):
        lib.add_sample("Person B", np.ones(8), stem=_stem(2), segments=5, basis=BASIS_HUMAN)
    lib.index_path.write_text("{not json")
    with pytest.raises(LibraryError):
        lib.profiles()


def test_missing_vector_file_costs_one_sample_not_the_library(tmp_path):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    lib.add_sample("Person B", B["Person B"], stem=_stem(1), segments=5, basis=BASIS_HUMAN)
    next(lib.vectors_dir.glob("*.npy")).unlink()
    assert len(lib.profiles()) == 1


def test_cli_never_prints_a_vector(tmp_path, capsys):
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_sample("Person A", B["Person A"], stem=_stem(1), segments=5, basis=BASIS_HUMAN, crm_contact_id=3)
    lib.add_sample("Matthias Heim", B["host"], stem=_stem(1), segments=5, basis=BASIS_SELF, role="self")
    assert vl.main(["--library-dir", str(lib.root), "list"]) == 0
    assert vl.main(["--library-dir", str(lib.root), "stats"]) == 0
    out = capsys.readouterr().out
    assert "Person A" in out and "crm#3" in out and "human_confirmed" in out
    stats = json.loads(out[out.index("{"):])
    assert stats["people"] == 2 and stats["others"] == 1 and stats["self_enrolled"] is True
    assert stats["samples_by_basis"] == {"human_confirmed": 1, "self_channel": 1}
    for v in B["Person A"][:8]:                       # none of the vector's digits appear
        assert f"{abs(float(v)):.4f}" not in out
