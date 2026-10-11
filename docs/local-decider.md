# Local Perplexity Decider (2026-10-10)

MeetingRecorder's **Decider lokal verwenden** preference selects local `pplx-decider-v1.1-27b` for live prompt detection and topic clips, and for opted-in finished transcripts. The existing preference was enabled on this laptop. Older `use_jev` preferences and per-recording `jev` flags remain readable; new preference writes use `use_decider`. Historical recordings keep their saved choice. The CLI accepts `--judge decider`, with `--judge jev` retained as an alias.

The adapter uses the shared `decider` skill's helper, which pins loopback `127.0.0.1:8092`, bypasses proxies, refuses redirects and checks returned model/probabilities. No Jev or OpenRouter request is made. Local failures leave audio recording running and surface a decision-detection error. Transcription and clean-prompt rewriting continue to use the existing separate Gemini integration.

The custom MLX runtime is necessary: these are classification weights with noncausal full attention and a dedicated 255-row readout, not a normal chat model. The pinned 4-bit revision is `66383f28b37637658b02b8e82b1029598b138642`. The shared skill documents installation and the `com.lailix.decider` LaunchAgent; runtime files and reports live under `~/.local/share/agent-skill-state/decider/`. Installed LM Studio 0.4.16 does not expose the needed decision endpoint, so this service runs separately.

Cache keys now identify the `decider` backend and actual model; Jev probabilities are not reused. Attempts are recorded as metadata in `~/.local/share/meeting-sidecar/decider-audit.jsonl`, with no transcript/request text. The adapter handles the actual `noul` response field and batches up to four concurrent line requests while preserving transcript order.

## Validation and operating points

- 15/15 synthetic cases passed, including English, Hochdeutsch and Swiss German, all three output schemas, repeatability, concurrency and invalid/oversized input rejection.
- The real recorder adapter correctly separated a dictated task from ordinary chatter; a repeat request was served from the score cache in under 1 ms.
- The separate demonstration harness's existing 10 development questions passed through the same real local service.
- The local reference evaluation sampled 60 English relevance lines and 60 Swiss German relevance lines with a fixed seed, then evaluated all 161 Swiss German prompt lines. It used the exact runtime questions and six preceding lines. No meeting text was sent externally or copied into this repository.

The reference labels are model-generated, with known ambiguity. Operating thresholds are a pilot choice based on these samples, not proof of calibrated confidence or an independent production accuracy estimate. Decider's relevance probabilities were more conservative than the previous model: its clip thresholds are 0.10 to seed and 0.025 to grow (wide: 0.05/0.01), retaining the existing deterministic clip selection and no-match candidate behavior. Other providers' thresholds are unchanged. Prompt threshold and final aggregate measurements are recorded below.

Short synthetic requests initially took a median 1.33 seconds. With four queued context-rich callers, relevance requests had a median 12–14 seconds; bulk processing and live suggestions can lag under load. Audio capture runs independently. Model/runtime files occupy about 15–16 GB; the idle LM Studio Qwen model was unloaded from RAM to make room, preserving its files.

The [performance assessment](decider-performance.md) separates these queued measurements from interactive latency and proposes comparisons that preserve the model and inputs. Disk size is not runtime memory: a later shared-host experiment recorded a 35.9 GiB Decider OS process footprint (a snapshot, not a measured peak).

Regression suite:

```sh
venv/bin/python -m pytest tests/test_sidecar_decider.py tests/test_sidecar_gate_judges.py tests/test_sidecar_live.py tests/test_sidecar_adversarial_regressions.py tests/test_sidecar_recorder_support.py tests/test_sidecar_prompts_calibration.py tests/test_sidecar_transcript_selection.py -q
```

| Reference check | Lines | Decider AUC | Saved Jev AUC |
| --- | ---: | ---: | ---: |
| English relevance sample | 60 | 0.911 | 0.944 |
| Swiss German relevance sample | 60 | 0.923 | 0.925 |
| Swiss German prompt detection | 161 | 0.995 | 0.999 |

The initial Decider prompt threshold is **0.92**: all 6 labelled prompt lines detected, with 2 false positives among 155 other lines (precision 75%, recall 100% on this set). The threshold was selected on this same reference set; these are calibration results, not held-out accuracy. The saved Jev baseline at 0.45 detected 6/6 with 1 false positive.

All 91 recorder regression tests passed. The shared helper's transport/schema tests passed, including proxy bypass, redirect refusal, wrong-model rejection and malformed probabilities. The demonstration harness passed 205 unit tests across the full run plus the added report-identity regression.
