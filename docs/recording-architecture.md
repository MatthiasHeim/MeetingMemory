# Recording Architecture Investigation

_Date: 2026-05-29 · Context: speaker-attribution bugs traced to mono capture · macOS 26.2_

## TL;DR

The attribution problem is fundamentally a **recording** problem, not a model problem.
Right now every recording is a single acoustic mic channel with both speakers blended
together — no model can reliably un-blend that. The fix has **two independent halves**,
and you need **both**:

1. **Clean remote channel** — capture the call's system audio *digitally* (not through the
   air). Today this never happens: channels 0/1 are digital silence in every file.
2. **Clean mic channel** — wear **headphones**. If remote audio plays through the speakers,
   the mic records Matthias **plus** an acoustic echo of the remote voices, so the mic
   channel is never clean Matthias no matter how perfectly the routing is fixed.

Fixing only #1 gives a clean remote channel and a still-contaminated mic channel — and it
will look like "the fix didn't work." Headphones are the highest-leverage, zero-cost change.

---

## 1. Root cause (empirical)

Probed the 3 most recent recordings (incl. 2026-05-28). Every file:

| Channel | Source | Mean volume |
|---|---|---|
| ch0 | BlackHole L (system audio) | **−91 dB (digital silence)** |
| ch1 | BlackHole R (system audio) | **−91 dB (digital silence)** |
| ch2 | MacBook mic | −23 to −45 dB (live, gain varies wildly) |

So the remote participant's voice has **never** been captured through BlackHole. It only
reaches the recording acoustically — remote voice → laptop speakers → air → mic — mixed
into the same channel as Matthias. That blend is why diarization keeps swapping speakers.

**Why ch0/1 are silent:** the macOS **Default Output Device is "MacBook Pro Speakers"**, so
call audio plays straight out the speakers and never enters BlackHole. The Multi-Output
Device is currently only the **Default *System* Output** (alert beeps), which is a different,
near-useless setting. For BlackHole to receive audio, the **Default Output Device** itself
must route through it.

The gain variance on ch2 (−23 vs −45 dB) is also why the loudnorm fix (committed) matters.

---

## 2. Two orthogonal problems

| Problem | Fixed by | Independent of |
|---|---|---|
| Remote channel cleanliness | digital system-audio capture (routing / Audio Hijack / CATap) | mic |
| Mic channel cleanliness | **headphones** (kills acoustic bleed) | capture method |

A dedicated headset/USB mic also raises Matthias's channel SNR and removes the gain swings —
solves echo *and* mic quality in one move.

---

## 3. The current audio stack is cluttered and fragile

Four virtual audio drivers are installed simultaneously:

- **BlackHole 2ch** (48 kHz) — the intended loopback
- **Background Music Device** (8 kHz) — open-source system-audio router; **app not currently
  running**, only the driver is loaded (latent clutter, but hijacks the default output if the
  app is ever launched)
- **ParrotAudioPlugin** — another recording tool's virtual device
- **Microsoft Teams Audio** — Teams' own virtual device

Plus an **Aggregate Device whose nominal rate is currently 8000 Hz** (the recorder forces
48 kHz at capture, so files are 48 kHz — but an 8 kHz aggregate clock is a latent footgun: if
a member can't hit 48 kHz it silently resamples). Stacking four virtual drivers + aggregates
is exactly the kind of setup that produces "I set Multi-Output and lost all sound."

Note: **Wispr Flow** is running with `MacCatapSystemAudioLoopbackCapture` — proof that the
modern Core Audio tap APIs work on this machine (relevant to Option C).

---

## 4. Why "set Multi-Output → no sound" — known causes (walk this checklist)

Do **not** just retry; one of these is the cause:

1. **Device ordering (most likely).** In Audio MIDI Setup → Multi-Output Device, the physical
   output (**Built-in Output / MacBook Pro Speakers**) must be **checked AND listed first
   (top)**. If BlackHole is first, macOS outputs nothing. Fix: uncheck/recheck to reorder.
2. **Sample-rate / master-clock mismatch.** Every member (Built-in, BlackHole) must be the
   same rate — set both to **48000 Hz**. Enable **Drift Correction** on the non-master members
   (BlackHole), with Built-in as the clock master.
3. **Background Music app hijack** — only if you launch that app; otherwise harmless. Consider
   uninstalling it and Parrot to de-clutter.
4. **Volume keys stop working — this is EXPECTED**, not a fault. macOS disables volume keys for
   any Multi-Output Device. Adjust volume per-app or on the individual device in Audio MIDI
   Setup. Not a reason to abandon the approach.

---

## 5. Options

### A. Fix the free BlackHole + Multi-Output + Aggregate setup
- **Effort:** config only + ~small pipeline change (split the 3 channels in post: ch2 → Matthias,
  ch0/1 → remote). The channel-split half is already sketched as Phase 5.
- **Pros:** $0; keeps the existing Python recorder; minimal code.
- **Cons:** fragile — breaks whenever the output device changes (plug in headphones/monitor and
  BlackHole stops being fed again); depends on the cluttered 4-driver stack; needs discipline.

### B. Audio Hijack (Rogue Amoeba, ~$80) — pragmatic robust
- Purpose-built. Auto-detects Teams/Zoom/Meet and records **both halves** with one click:
  local mic on the left channel, remote on the right (or as separate files). Built-in
  monitoring so you always hear the call. Survives device changes.
- **Pros:** robust, near-zero fiddling, separate tracks out of the box, handles VoIP apps natively.
- **Cons:** ~$80; replaces the menu-bar recorder, so the watcher needs to ingest Audio Hijack's
  output files (its output folder → existing pipeline). Modest integration.

### C. Native Core Audio tap / ScreenCaptureKit (Swift helper) — most robust, most modern
- macOS 14.2+ Core Audio process taps (what Wispr Flow uses here) and ScreenCaptureKit capture
  **system audio + mic as separate streams with no virtual devices and no routing**, immune to
  default-device changes.
- **Pros:** the "right" 2026 architecture; no driver clutter; per-app capture (record only the
  call, not notifications/music).
- **Cons:** requires a small **Swift** helper binary. Do **not** drive it from pyobjc — the
  Python bindings throw `SCStreamErrorDomain −3805` / silent no-callback bugs; a compiled Swift
  CLI that writes two WAVs is the reliable shape.

**Architecture rule for all paths:** prefer **one clocked capture session** (a single aggregate,
or Audio Hijack, or one Swift helper writing both streams) over **two independent capture
processes** (e.g. sounddevice mic + a separate tap). Two processes drift out of sync and wreck
the timestamp-based merge.

---

## 6. Recommendation

There is no "free = best" here — time has already been lost to the fragile stack, and this is
business-critical capture. Weigh by appetite:

- **Want it solid fastest, ~$80 ok → Option B (Audio Hijack).** Best robustness-per-effort;
  auto-records VoIP calls; separate tracks immediately. Integration = point the watcher at its
  output folder.
- **Want $0 and minimal change, accept fragility → Option A**, but commit to headphones and to
  re-checking the routing whenever you change output devices.
- **Want the durable long-term architecture → Option C** (Swift helper), schedule as real dev work.

**Immediate, testable step (you run it — I won't flip your audio devices mid-day before the
21:00 BlueCare call):**

1. Audio MIDI Setup → Multi-Output Device: Built-in Output checked + listed first; BlackHole
   checked with Drift Correction on; both at 48 kHz.
2. Set **Default Output Device** (the real one, ⌥-click the menu-bar speaker icon or System
   Settings → Sound → Output) to **Multi-Output Device**.
3. Play any audio and confirm you still hear it.
4. Record ~30 s with the MeetingRecorder while audio plays, then check:
   `ffmpeg -i <file>.wav -t 20 -af "pan=mono|c0=c0,volumedetect" -f null -` → **ch0 should now
   read > −60 dB.** That is the green light that unblocks Phase 5 (channel-aware diarization).

---

## 7. Scope — what channel separation does and doesn't fix

- ✅ **1:1 calls (host vs one remote):** perfect separation → fixes all four botched meetings
  (183/184/189/199 were all 1:1s).
- ⚠️ **Calls with multiple remote people:** the remote channel still has several voices →
  still needs model diarization, but only on that channel, and Matthias is always isolated.
- ❌ **In-person meetings:** one room, one mic → channel separation gives nothing; only model
  diarization applies.

---

## 7b. Option C build status (2026-05-29)

Built a Swift Core Audio process-tap recorder (`tools/audio_tap_recorder.swift`). Progress:

- ✅ **Tap + aggregate + IO proc + WAV writing all work** (compiles clean, runs, writes 48 kHz).
- ✅ **TCC permission SOLVED.** An unsigned/unbundled CLI never triggers the macOS
  `kTCCServiceAudioCapture` prompt — macOS silently feeds it −91 dB. Wrapping the binary in a
  **signed .app bundle** with `NSAudioCaptureUsageDescription` registered and granted the
  permission (TCC now lists `com.lailix.meetingmemory.audiotap` = allowed). So system-audio
  capture is unblocked, contingent on shipping inside a signed bundle.
- ✅ **Multi-stream capture FIXED & validated.** The mic (sub-device) and tap are *separate*
  input streams in the aggregate (mic stream 0 = 1 ch, system tap stream 1 = 2 ch). The IO proc
  now enumerates both streams, interleaves them, and writes a 3-channel WAV (ch0=mic,
  ch1=sysL, ch2=sysR). Verified live:
  - SYSTEM channels captured at **−18.9 dB** when launched via LaunchServices (`open …app`).
  - MIC channel captured at **−24 dB** when the process holds the microphone permission.

  So the whole architecture is proven end-to-end — single clocked aggregate, no drift, no
  virtual-device clutter, no output-device juggling.

- 🔑 **The one remaining step is permissions on the deployed bundle.** It needs BOTH grants:
  - `kTCCServiceAudioCapture` (system tap) — already granted to the test bundle.
  - `kTCCServiceMicrophone` (mic sub-device) — a background `LSUIElement` app can't show the
    prompt, so grant it from a FOREGROUND app or via System Settings → Privacy & Security →
    Microphone. This is why it belongs in the MeetingRecorder menu-bar app (foreground, can
    prompt for both on first run).

**Deployment shape:** run the recorder inside a signed .app bundle (the MeetingRecorder
menu-bar app itself, packaged via py2app + codesign, embedding/shelling out to this helper) so
it keeps both permissions across launches. Ad-hoc (self-signed) works for one machine; an Apple
Developer ID signature avoids re-grant friction on updates. **Crucial:** the process must be
launched AS the bundle (LaunchServices) — a directly-exec'd binary isn't attributed to the
bundle id and the system tap silently returns −91 dB.

## 8. Interim software mitigations (already landed / validated)

While the recording fix is the main lever, two software changes already help on mono audio:

- **Loudnorm** (committed `bdc39af`) — normalizes mic gain; fixed meeting 184's mislabel.
- **Dialect prompt fix** (committed `106615f`) — stops Swiss German → High German normalization.
- **Model:** with the dialect fix, `gemini-3.5-flash` got the meeting-199 slice **3/3 correct on
  attribution AND preserved Swiss German** (vs production `gemini-3-flash-preview` 0/3 on the
  same slice). Promising as a model bump, but still probabilistic on full meetings — a mitigation,
  not the cure. The cure is channel separation.
