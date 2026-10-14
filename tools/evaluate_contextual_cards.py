"""Explicit opt-in semantic evaluation, using synthetic fixtures only."""
from __future__ import annotations
import argparse
import json
import tempfile
import time
from pathlib import Path
from sidecar.card_semantics import CardEvidence, classify_event, split_sources, TAXONOMY
from sidecar.gate import initialise_recording_sidecar
from sidecar.judges import DeciderJudge, GeminiJudge
from sidecar.live import LiveLine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['decider', 'gemini'])
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--split', choices=['development', 'holdout'])
    args = parser.parse_args()
    cases = json.loads((Path(__file__).resolve().parents[1] / 'tests/fixtures/contextual_cards.json').read_text())
    if args.split:
        cases = [case for case in cases if case['split'] == args.split]
    results = []
    with tempfile.TemporaryDirectory(prefix='synthetic-card-eval-') as directory:
        root = Path(directory)
        stem = '2026-10-14_00-00-00'
        initialise_recording_sidecar(stem, jev=True, external_attendees=False, root=root)
        judge = DeciderJudge(stem=stem, recordings_root=root, audit_root=root / 'audit') if args.backend == 'decider' else GeminiJudge()
        for case in cases:
            decisions = []
            original_choose = judge.choose
            def traced(state, questions):
                result = original_choose(state, questions)
                decisions.append(result)
                return result
            judge.choose = traced
            context = split_sources([LiveLine(i * 5, i * 5 + 4, 'Andere', text) for i, text in enumerate(case['context'])])
            current = split_sources([LiveLine(100, 104, 'Ich', case['current'])])[0]
            cards = []
            if case.get('card'):
                cards = [CardEvidence('existing', 'request', tuple(split_sources([LiveLine(0, 4, 'Ich', case['card'])])))]
            started = time.monotonic()
            try:
                event = classify_event(current, context, cards, judge)
                selected = [i for i, span in enumerate(context) if span.id in event.source_ids]
                ok = event.kind == case['kind']
                if case.get('relation') == 'existing': ok &= event.target == 'existing'
                if case.get('relation') == 'new': ok &= event.target is None and event.source_ids == (current.id,)
                ok &= all(index in selected for index in case.get('include', []))
                ok &= all(index not in selected for index in case.get('exclude', []))
                row = dict(id=case['id'], split=case['split'], passed=ok, actual=event.kind, target=event.target, selected=selected)
            except Exception as exc:
                row = dict(id=case['id'], split=case['split'], passed=False, error=type(exc).__name__)
            judge.choose = original_choose
            row['decisions'] = decisions
            row['seconds'] = round(time.monotonic() - started, 3)
            results.append(row)
            print(json.dumps(row), flush=True)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(dict(taxonomy=TAXONOMY, backend=args.backend, synthetic_only=True, results=results), indent=2) + '\n')
    print(f"Passed {sum(row['passed'] for row in results)}/{len(results)}")

if __name__ == '__main__': main()
