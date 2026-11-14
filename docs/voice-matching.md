# Voice matching

Which person speaks, from the voice itself. Local only: the embedding model
(`pyannote/embedding`, `pyannote/embedding/pytorch_model.bin`) runs on this
machine, the fingerprints live in one private directory, nothing is sent
anywhere.

The system-track stage (`tools/system_track_attribution.py`) decides host vs
remote. This stage (`tools/voice_match.py`) adds *which* remote person, using a
library (`tools/voice_library.py`) of per-meeting voice centroids.

## Config (`~/Documents/MeetingRecorder/config.yaml`)

```yaml
voice_matching:
  enabled: true        # default true when the key is absent
  store_others: true   # enrol confirmed non-host people from archived tracks
  auto_apply: false    # true: rename GENERIC labels with a clear, conflict-free match
  library_dir: ~/Documents/MeetingRecorder/voice_library
  # device: mps | cpu  (default: mps if available)
  # timeout_seconds: 600
```

## Privacy

* Fingerprints are biometric. They are never committed (`.gitignore`), never
  written into transcript JSON or the database, never printed, never sent out.
* The library directory is mode 0700, every file 0600, writes are atomic.
* Logs and `_meta.voice_match` carry names and similarity scores only.
* `python tools/voice_library.py forget --name "X"` / `forget --all --yes`
  delete vectors and index entries; `list` and `stats` show no vectors.

## Flow

1. Watcher (two-track recordings only), after the system-track stage and
   before the coherence gate: cut up to 14 segments per label (turns >= 7 s;
   remote speakers from the system track where it is active >= 70 % of the
   turn, the host from the mic where it is active <= 12 %), embed them in a
   worker process (timeout 600 s, never raises), match against the library.
2. Result in `_meta.voice_match`: per label best / runner-up / verdict
   (`clear` = score >= 0.70 and margin >= 0.20 and >= 3 segments), a host
   check, conflicts. With `auto_apply: false` the transcript is never touched.
3. The host's mic-only centroid is added to the self profile
   (`basis: self_channel`) when the system-track stage verified the host label.
4. `tools/speaker_review.py propose` shows "voice matches X (0.91)" as
   evidence; a clear voice match becomes the proposal unless a high-confidence
   transcript-based proposal names someone else (then both are shown and the
   proposal is flagged `voice_conflict`).
5. `speaker_review.py confirm` (the human confirmation) enrols each confirmed
   non-host person from the archived system track (`basis: human_confirmed`)
   when `store_others` is true. If `CaptureArchive/<stem>/` has already been
   pruned (`tools/prune_recordings.py`) nothing is enrolled; the confirmation
   itself is unaffected.

## CLI

```
python tools/voice_match.py evaluate --all --library-dir /tmp/vm-eval      # leave-one-meeting-out dry run
python tools/voice_match.py enrol-confirmed --library-dir DIR [--use-db]   # backfill from confirmed meetings
python tools/voice_library.py list | stats | forget --name X | forget --all --yes
```
