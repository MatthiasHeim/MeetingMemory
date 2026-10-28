# Contextual recorder cards and native presentation

Prepared in `codex/recorder-presentation-panel`. The running recorder remains on cf43968e until a separately approved restart with capture stopped. No provider, model, preference, completed recording or active capture was changed by this work.

## Behaviour

- Live ASR remains pinned Gemini 3.8 Flash. It now targets six-second start intervals with one request at a time, a 24-second rolling tail, and extra overlap after a slow request (up to 120 seconds). Unfinished speech appears immediately as explicitly provisional text; card extraction uses committed source. This is short batch transcription, not streaming. More frequent calls increase request volume.
- Contextual cards use local Decider when the recording opted in; otherwise the existing pinned Gemini route. The categorical taxonomy distinguishes actual requests, settled decisions, explicit action commitments, ordinary discussion and uncertain evidence. It does not reuse or lower the old 0.92 AI-dictation cutoff.
- Exact sentence spans retain line identity and character offsets; German ordinal dates and common abbreviations stay intact. Context helps interpretation; only selected evidence reaches the card formatter. A go-ahead can adopt a multi-sentence proposal. Existing-card relations extend/correct one card; repeated exact evidence does not create duplicate cards. Unrelated work creates another. Revisions rewind from the changed source and invalidate dependent responses; an unrelated later revision does not suppress an earlier completed card.
- Gemini formats selected evidence by type. Later corrections supersede older details; source passages remain available. Formatting failures leave verbatim evidence visible. Card-model payloads normalize speaker labels to Ich/Andere, preventing a personal ASR label from being treated as a stated owner; exact source labels remain available in the transcript.
- AI questions/ideas are explicitly differentiated from extracted statements, including in copied text. A separate audience switch defaults to client-facing at each new session. Private mode can produce internal coaching. Client mode immediately excludes internal cards, details, highlighting and copy targets; it also performs a separate bounded content check through the selected classifier before publishing generated advice. The switch does not redact the actual conversation transcript or extracted statements.
- Suggestions are coalesced and limited to at most two per generation, at most once per 45 seconds of changed source (explicit audience changes can request the new audience sooner). No outbound actions, sales actions or external context retrieval occur.
- Transcript, extraction, wording and suggestion work remain separate. The contextual worker owns its bounded retry timer, including after Stop; it then marks exhausted source unknown and continues. Backlog and concurrent failures are visible. Retiring a session wakes and cancels pending retries; a retired session performs no further queued inference and cannot repaint a reused panel. Suggestions validate their cited evidence and log accepted/discarded counts; changed evidence requests a fresh result without the normal 45-second delay.
- One native AppKit panel supports compact and presentation layouts, explicit size toggles and native full-screen. Close hides it; the menu restores the same window and session. New recording resets content but preserves window geometry. Minimize remains ordinary macOS minimize. Opening the idle panel does not start capture.
- Menu labels have native NSMenuItem tooltips with an always-accessible help view. Hover presentation depends on macOS; tooltip setter/getter were checked on actual rumps/AppKit.

## Evidence and limits

Before this change, measured live ASR requests took roughly 3.3–7.4 seconds, followed by a fixed 20-second wait. First committed text appeared on the second cycle around 48 seconds after Start. The old classifier also scored a real example 0.913 below its 0.92 cutoff. The screenshot showed two contextless or misleading cards, so the fix addresses the task and grouping rather than lowering a cutoff.

The synthetic categorical fixture now has seven development and ten holdout cases, including the complete abstract instruction exposed by end-to-end testing. The final categorical run passed Decider 16/17 (7/7 development, 9/10 holdout) and Gemini 17/17. The remaining Decider disagreement is the deliberately quoted instruction followed by “I disagree and we are not doing that”: Decider calls the present rejection a decision while the original label expects none. This is recorded as a strict failure, not silently relabelled or used to tune the holdout. These small synthetic sets do not establish production recall or precision.

Development iterations initially failed polite requests and adopted multi-part plans. Replacing mutually exclusive individual antecedent choices with a contextual relation followed by independent exact-span selection fixed both. Raw reports retain the earlier failures. Normal Decider fixture calls took about 4–9 seconds, with ~16 seconds for a two-stage adopted plan; under concurrent recorder load earlier runs took longer. Long continuous speech can still create visible classification backlog; no unbounded concurrent request burst is used.

A synthetic speech check uses macOS Samantha TTS, no microphone, and the real pinned transcription provider. The final incremental run displayed the first six-second fragment provisionally after a 3.94-second request, then committed text on the next cycle (2.85-second request). Audio availability is simulated in this check; it does not establish wall-clock latency for a real meeting or full speaker accuracy. One synthetic transcript rendered “report” as “reports”, illustrating that ASR remains fallible.

All 12 native preview checks passed for source highlighting, close/hide/reopen, session preservation, audience suppression, presentation mode, actual native full-screen entry/exit, and disabled capture in preview. Screenshots were inspected. Multiline source sizing now uses the actual NSTextFieldCell renderer. UTF-16 ranges preserve highlights around emoji. The complete synthetic audio → Gemini ASR → Decider → Gemini wording → native panel run completed in 34.225 seconds with two requests, one decision, one action and two labelled AI questions, with no errors. This final panel was also visually inspected.

One independent Opus review identified a blocking German sentence-boundary bug and additional retry, source-role and stale-result issues. These were fixed and regression-tested; real Decider/formatter calls preserve “15. März”, “3. Oktober”, and “z. B. Dr. Meier … 2. Entwurf” without invented years. The review's suggested clearing of the previous session on Stop/reopen was not adopted: preserving the same session and mode is an explicit user decision. New recording still resets to client-facing, and switching to client immediately hides internal suggestions. No second review was requested. Raw findings and an adjudication are retained locally.

Final focused regression run: **139 passed**, covering contextual semantics, asynchronous workers, provider gates, recorder integration, source selection and calibration. One existing optional torchcodec/FFmpeg warning remains in the audio-import test; it did not fail any test.

Operational logs contain timings/counts, attempt counts, classification kind and error class, never source text. UI publication timings measure dispatch/render callback latency, not physical display refresh.

## Reproducibility

- `tools/evaluate_contextual_cards.py --backend decider|gemini --report PATH [--split development|holdout]` uses only repository synthetic fixtures and a temporary policy sidecar.
- `tools/evaluate_live_incremental.py --out PATH` generates synthetic speech locally and sends only that synthetic audio to the existing Gemini model.
- `tools/preview_recorder_panel.py --out PATH [--background PATH]` opens a labelled native preview with synthetic text and no audio capture, saves screenshots and check results.
- Local evidence: `~/.local/share/agent-skill-state/meeting-recorder/` (semantic reports, synthetic wording/audio checks and native previews). No private recording was replayed or sent for new inference.

## Assets

The user approved relocating the wall logo and correcting its first L against `/Users/Matthias/Lailix/assets/lailix_logo.png`: first L/final LIX navy; italic ai and triangular accent gold. The generated, clean-reencoded raised-logo image is the default. The separately approved pointing-hand variant remains optional. Original Desktop asset is untouched. See `tools/sidecar/assets/README.md`.
