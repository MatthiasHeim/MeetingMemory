# Source preservation and retention

Raw capture files are evidence. A mono MP3, a transcript, and a completed
InsightBase record are useful derivatives, but none proves that the original
multichannel audio or separate microphone/system tracks can be reconstructed.
`tools/prune_recordings.py` consequently preserves raw files by default.

## Legacy source exports

`audio_converter.extract_source_tracks(input_path, output_dir)` creates
full-duration FLAC source artifacts without mixing or resampling:

| Input layout | Outputs | Meaning |
| --- | --- | --- |
| Known legacy 3-channel WAV | `mic` (channel 0 mono), `system` (channels 1-2 stereo) | The system track remains one stereo source; its channels are not remote participants. |
| Mono or stereo | one `audio` artifact, same channel count | Opaque source: stereo does not establish a host/remote split. |
| Other 3+ channel layout | one opaque multichannel `audio` artifact | Channel identities are unknown and must not be invented. |

Every legacy result records `start_seconds: 0` and
`timing_basis: "unverified_sample_zero"`. That is a file-local origin, not
evidence that mic and system capture began together. Native capture manifests
with a verified host-clock timeline use their own path and are not certificates
for legacy source replacement.

## Certified replacement manifest

A raw candidate can only be deleted when
`Transcripts/<stem>.source-replacement.json` contains a reviewed v1 manifest
with a separate entry for **each** raw file being retired (the merged recording
and every archive mic/system track are independent candidates):

```json
{
  "schema_version": 1,
  "replacements": [
    {
      "original_path": "/absolute/path/to/Recordings/2026-09-15_10-00-00.wav",
      "original_sha256": "exact 64-character SHA-256",
      "replacement_path": "/absolute/path/to/SourceTracks/2026-09-15_10-00-00.audio.flac",
      "replacement_sha256": "exact 64-character SHA-256",
      "lossless_decode": {
        "verified": true,
        "original": {
          "sample_count": 0,
          "sample_rate": 48000,
          "channels": 3,
          "pcm_sha256": "canonical decoded PCM SHA-256"
        },
        "replacement": {
          "sample_count": 0,
          "sample_rate": 48000,
          "channels": 3,
          "pcm_sha256": "same canonical decoded PCM SHA-256"
        }
      }
    }
  ]
}
```

Generate an entry read-only, after the FLAC is in its durable destination:

```bash
/Users/Matthias/Repos/MeetingMemory/venv/bin/python tools/prune_recordings.py \
  --verify-lossless-replacement ORIGINAL.wav REPLACEMENT.flac
```

The command neither writes a manifest nor removes data. Before deletion, the
pruner computes both file hashes again and streams both decode paths to check
the PCM hash, sample count, sample rate, and channel count again. The replacement
must be an independent FLAC outside `Recordings/` and `CaptureArchive/`.

Malformed or partial transcripts/manifests, a changed source or replacement,
an invalid replacement path, and every `.hold` marker fail closed. The pruner
never recursively removes an archive directory, so `capture.json`, notes, and
unmapped companion tracks remain intact.

## Migration and storage

Existing recordings have no certified FLAC replacement merely because APFS
compression, an MP3, or a transcript exists. They remain retained until a
deliberate migration produces and verifies a FLAC replacement per original
track. Plan storage for those source tracks before enabling deletion: a
lossless source archive is expected to be larger than the current MP3 library.
Until that migration is reviewed, retention is preservation, not cleanup.
