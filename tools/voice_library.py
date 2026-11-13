#!/usr/bin/env python3
"""voice_library — local store of voice fingerprints (speaker embeddings).

Voice fingerprints are BIOMETRIC data. This module is the only place they are
persisted, and it keeps them to one directory on this machine:

  * never committed (the default location is outside every repo), never
    uploaded, never written into transcript JSON, `_meta` or the database;
  * directory mode 0700, every file 0600, every write atomic;
  * the CLI never prints a vector, only names, counts and stems.

Layout (`voice_matching.library_dir`, default
``~/Documents/MeetingRecorder/voice_library/``)::

    index.json            people + sample metadata, no vectors
    vectors/<id>.npy      one L2-normalised float32 centroid per sample
    .lock                 flock target for writers

A person's profile is several per-meeting centroids (one sample per recording,
the most recent ``MAX_SAMPLES_PER_PERSON`` kept). A sample records the
recording stem, how many segments it was built from and its basis:

  * ``human_confirmed`` - someone confirmed the name in the Confirm speakers
    panel;
  * ``self_channel``   - the host, taken from mic-only turns of a recording
    whose host label the system-track stage verified.

People are never merged: a name is matched exactly after normalisation (NFKC,
case-folded, whitespace collapsed), a CRM contact id is preferred when both
sides have one, and an id/name clash raises instead of guessing.

CLI::

    python tools/voice_library.py list
    python tools/voice_library.py stats
    python tools/voice_library.py forget --name "Person A"
    python tools/voice_library.py forget --all --yes
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import sys
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

SCHEMA_VERSION = 1

# Per-meeting centroids kept per person (most recent by recording stem).
# Voices drift with microphones and rooms; ten meetings are enough to cover
# that and keep the best-centroid match robust without growing forever.
MAX_SAMPLES_PER_PERSON = 10

BASIS_HUMAN = "human_confirmed"
BASIS_SELF = "self_channel"
BASES = (BASIS_HUMAN, BASIS_SELF)

ROLE_SELF = "self"
ROLE_OTHER = "other"

SELF_ID = "self"
SELF_NAME = "Matthias Heim"

DEFAULT_LIBRARY_DIR = Path.home() / "Documents" / "MeetingRecorder" / "voice_library"
CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"

_STEM_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


class LibraryError(RuntimeError):
    """The library is unreadable or a write was refused."""


class PersonConflict(LibraryError):
    """Adding this sample would merge or confuse two people."""


def normalise_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name or "")
    return " ".join(text.casefold().split())


def default_library_dir(config: Optional[dict] = None) -> Path:
    raw = ((config or {}).get("voice_matching") or {}).get("library_dir")
    return Path(os.path.expanduser(str(raw))) if raw else DEFAULT_LIBRARY_DIR


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read-only config load; a missing or broken file is an empty config."""
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class Profile:
    """One person as the matcher sees them. Holds vectors; never log it."""
    person_id: str
    name: str
    role: str
    crm_contact_id: Optional[int]
    centroids: np.ndarray                       # (n_samples, dim), unit rows
    stems: list[str] = field(default_factory=list)


class VoiceLibrary:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()

    # -- paths ------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    # -- reading (never creates anything) ---------------------------------

    def _read_index(self) -> dict:
        if not self.index_path.exists():
            return {"schema_version": SCHEMA_VERSION, "dim": None, "people": []}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LibraryError(f"voice library index unreadable: {type(exc).__name__}") from exc
        if (not isinstance(data, dict) or not isinstance(data.get("people"), list)
                or data.get("schema_version") != SCHEMA_VERSION):
            raise LibraryError("voice library index has an unexpected shape or version")
        return data

    def people(self) -> list[dict]:
        """Index rows (no vectors)."""
        return json.loads(json.dumps(self._read_index()["people"]))

    def has_others(self, exclude_stem: Optional[str] = None) -> bool:
        for p in self._read_index()["people"]:
            if p.get("role") == ROLE_SELF:
                continue
            if any(s.get("stem") != exclude_stem for s in p.get("samples") or []):
                return True
        return False

    def profiles(self, exclude_stem: Optional[str] = None) -> list[Profile]:
        """Profiles with vectors loaded. Samples built from `exclude_stem` are
        left out so a recording is never matched against its own enrolment."""
        out: list[Profile] = []
        for p in self._read_index()["people"]:
            rows, stems = [], []
            for s in p.get("samples") or []:
                if exclude_stem is not None and s.get("stem") == exclude_stem:
                    continue
                path = self.vectors_dir / f"{s['id']}.npy"
                try:
                    vec = np.load(path, allow_pickle=False)
                except (OSError, ValueError):
                    continue   # a missing vector costs one sample, not the library
                if vec.ndim != 1:
                    continue
                rows.append(vec.astype(np.float32))
                stems.append(s.get("stem"))
            if rows:
                out.append(Profile(p["id"], p["name"], p.get("role", ROLE_OTHER),
                                   p.get("crm_contact_id"), np.vstack(rows), stems))
        return out

    # -- writing ----------------------------------------------------------

    def _ensure_dirs(self) -> None:
        for d in (self.root, self.vectors_dir):
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self._ensure_dirs()
        lock = self.root / ".lock"
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o600)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _write_index(self, index: dict) -> None:
        self._atomic_write(self.index_path,
                           json.dumps(index, indent=2, ensure_ascii=False).encode("utf-8"))

    def _write_vector(self, sample_id: str, vec: np.ndarray) -> None:
        import io
        buf = io.BytesIO()
        np.save(buf, vec.astype(np.float32), allow_pickle=False)
        self._atomic_write(self.vectors_dir / f"{sample_id}.npy", buf.getvalue())

    def _unlink_vectors(self, sample_ids: list[str]) -> None:
        for sid in sample_ids:
            with contextlib.suppress(OSError):
                (self.vectors_dir / f"{sid}.npy").unlink()

    @staticmethod
    def _find(index: dict, name: str, crm_contact_id: Optional[int]) -> Optional[dict]:
        """CRM id first; else the exact normalised name. A clash raises."""
        norm = normalise_name(name)
        by_name = next((p for p in index["people"]
                        if p.get("role") != ROLE_SELF and normalise_name(p["name"]) == norm), None)
        if crm_contact_id is not None:
            by_id = next((p for p in index["people"]
                          if p.get("crm_contact_id") == crm_contact_id), None)
            if by_id is not None:
                return by_id          # renamed in the CRM: the id wins
            if by_name is not None and by_name.get("crm_contact_id") not in (None, crm_contact_id):
                raise PersonConflict(
                    f"name {name!r} already belongs to a different CRM contact; not merging")
        return by_name

    def add_sample(self, name: str, vector: np.ndarray, *, stem: str, segments: int,
                   basis: str, role: str = ROLE_OTHER,
                   crm_contact_id: Optional[int] = None) -> dict:
        """Add (or replace, for the same stem) one per-meeting centroid.

        Returns {"person_id", "created", "replaced", "samples"}. Raises
        PersonConflict / ValueError / LibraryError; callers decide what a
        refusal costs (the watcher and speaker_review both log and go on).
        """
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}")
        if role not in (ROLE_SELF, ROLE_OTHER):
            raise ValueError("role must be 'self' or 'other'")
        if not normalise_name(name):
            raise ValueError("a person needs a name")
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if vec.size == 0 or not np.isfinite(vec).all() or norm < 1e-9:
            raise ValueError("vector is empty, non-finite or zero")
        vec = vec / norm
        if not _STEM_RE.match(stem or ""):
            raise ValueError("stem must look like YYYY-MM-DD_HH-MM-SS")

        with self._locked():
            index = self._read_index()
            if index.get("dim") not in (None, vec.size):
                raise LibraryError("vector dimension differs from the library's")
            index["dim"] = vec.size
            if role == ROLE_SELF:
                person = next((p for p in index["people"] if p.get("role") == ROLE_SELF), None)
            else:
                if normalise_name(name) == normalise_name(SELF_NAME):
                    raise PersonConflict("refusing to store the host under another person's entry")
                person = self._find(index, name, crm_contact_id)
            created = person is None
            if created:
                person = {"id": SELF_ID if role == ROLE_SELF else uuid.uuid4().hex[:12],
                          "name": name.strip(), "role": role,
                          "crm_contact_id": crm_contact_id, "samples": []}
                index["people"].append(person)
            elif role == ROLE_OTHER:
                person["name"] = name.strip()          # latest confirmed spelling
                if crm_contact_id is not None:
                    person["crm_contact_id"] = crm_contact_id

            replaced = [s for s in person["samples"] if s.get("stem") == stem]
            person["samples"] = [s for s in person["samples"] if s.get("stem") != stem]
            sample = {"id": uuid.uuid4().hex, "stem": stem, "segments": int(segments),
                      "basis": basis, "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            person["samples"].append(sample)
            person["samples"].sort(key=lambda s: s["stem"])
            dropped = person["samples"][:-MAX_SAMPLES_PER_PERSON]
            person["samples"] = person["samples"][-MAX_SAMPLES_PER_PERSON:]

            self._write_vector(sample["id"], vec)      # vector first, index second
            self._write_index(index)
            self._unlink_vectors([s["id"] for s in replaced + dropped])
            return {"person_id": person["id"], "created": created,
                    "replaced": bool(replaced), "samples": len(person["samples"])}

    def forget(self, name: Optional[str] = None, *, all_people: bool = False) -> int:
        """Delete a person's vectors and index entry (or everyone's). Returns
        the number of people removed."""
        if (name is None) == (not all_people):
            raise ValueError("give a name or all_people=True, not both")
        if not self.index_path.exists() and not self.vectors_dir.exists():
            return 0
        with self._locked():
            index = self._read_index() if self.index_path.exists() else \
                {"schema_version": SCHEMA_VERSION, "dim": None, "people": []}
            if all_people:
                gone, keep = index["people"], []
            else:
                norm = normalise_name(name)
                gone = [p for p in index["people"] if normalise_name(p["name"]) == norm]
                keep = [p for p in index["people"] if p not in gone]
            index["people"] = keep
            if not keep:
                index["dim"] = None
            self._write_index(index)
            self._unlink_vectors([s["id"] for p in gone for s in p.get("samples") or []])
            if all_people:       # also sweep orphans a crash may have left behind
                for f in self.vectors_dir.glob("*.npy"):
                    with contextlib.suppress(OSError):
                        f.unlink()
            return len(gone)

    def stats(self) -> dict:
        index = self._read_index()
        people = index["people"]
        samples = [s for p in people for s in p.get("samples") or []]
        size = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file()) if self.root.exists() else 0
        return {
            "dir": str(self.root), "people": len(people),
            "self_enrolled": any(p.get("role") == ROLE_SELF for p in people),
            "others": sum(1 for p in people if p.get("role") != ROLE_SELF),
            "samples": len(samples),
            "samples_by_basis": {b: sum(1 for s in samples if s.get("basis") == b) for b in BASES},
            "bytes": size, "dim": index.get("dim"),
            "newest_stem": max((s["stem"] for s in samples), default=None),
            "oldest_stem": min((s["stem"] for s in samples), default=None),
            "max_samples_per_person": MAX_SAMPLES_PER_PERSON,
        }


# -- CLI -------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library-dir", type=Path,
                    help="default: voice_matching.library_dir from config.yaml, else "
                         "~/Documents/MeetingRecorder/voice_library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="people and their samples (no vectors)")
    sub.add_parser("stats", help="counts and size (no vectors)")
    f = sub.add_parser("forget", help="delete a person's vectors and entry")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--name")
    g.add_argument("--all", action="store_true")
    f.add_argument("--yes", action="store_true", help="required with --all")
    args = ap.parse_args(argv)

    lib = VoiceLibrary(args.library_dir or default_library_dir(load_config()))
    try:
        if args.cmd == "list":
            people = lib.people()
            if not people:
                print(f"(empty) {lib.root}")
            for p in people:
                stems = [s["stem"] for s in p.get("samples") or []]
                bases = sorted({s["basis"] for s in p.get("samples") or []})
                crm = f" crm#{p['crm_contact_id']}" if p.get("crm_contact_id") is not None else ""
                print(f"{p['name']}  [{p.get('role')}]{crm}  {len(stems)} sample(s)  "
                      f"{','.join(bases)}  {stems[0] if stems else '-'} .. {stems[-1] if stems else '-'}")
        elif args.cmd == "stats":
            print(json.dumps(lib.stats(), indent=2))
        else:
            if args.all and not args.yes:
                print("refusing: --all deletes every voice profile; add --yes", file=sys.stderr)
                return 2
            n = lib.forget(args.name, all_people=args.all)
            print(f"removed {n} person(s)")
            return 0 if n or args.all else 1
    except LibraryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
