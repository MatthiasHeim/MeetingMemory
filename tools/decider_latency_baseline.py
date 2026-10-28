"""Bounded loopback-only timing replay; private fixture bodies stay outside Git.

Run --prepare first, inspect metadata, then explicitly run --run after checking
the recorder is idle and coordinating with other Decider users. This never loads
weights, changes runtime settings, restarts services, or invokes a cloud model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

ROOT = Path.home() / '.local/share/agent-skill-state/decider'
OUT = ROOT / 'latency-2026-10-11'
CORPUS = Path.home() / '.local/share/meeting-sidecar/eval-2026-09-26'
RECORDINGS = Path.home() / 'Documents/MeetingRecorder'
MODEL = 'pplx-decider-v1.1-27b'
REVISION = '66383f28b37637658b02b8e82b1029598b138642'
BASE = 'http://127.0.0.1:8092'


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save(name: str, value) -> None:
    OUT.mkdir(mode=0o700, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o600)


def prepare() -> None:
    from sidecar.calibration import _version_lines
    from sidecar.judges import _line_payload, normalise_questions
    from sidecar.questions import DICTATING_PROMPT_QUESTION
    from sidecar.service import relevance_question

    if (OUT / 'manifest.private.json').exists():
        raise RuntimeError('Frozen manifest already exists; do not silently replace it')
    spec = importlib.util.spec_from_file_location('saved_acceptance', ROOT / 'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # definitions only; never calls main/inference
    short = []
    for index in (0, 2, 6, 9):
        name, text, _ = module.CASES[index]
        short.append({'id': name, 'category': 'short', 'source': 'acceptance',
                      'source_index': index, 'body': {'model': MODEL,
                      'state': {'previous_lines': [], 'current_line': {'text': text, 'speaker': 'Speaker'}},
                      'questions': {'prompt': module.prompt_q}}})
    relevance, prompt = [], []
    for dataset, version, qid in [('A', 'en', 'relevant'), ('B', 'gsw', 'relevant'), ('C', 'gsw', 'dictating_prompt')]:
        data = json.loads((CORPUS / f'ds_{dataset}.json').read_text())
        saved = json.loads((ROOT / f'scores-{dataset}.json').read_text())
        probabilities = dict(zip(saved['indexes'], saved['probabilities']))
        lines = _version_lines(data, version)
        supplied = DICTATING_PROMPT_QUESTION if dataset == 'C' else relevance_question(str(data['topic']))
        q = normalise_questions({qid: supplied})[qid]
        question = {'type': 'noul', 'instructions': q.instructions,
                    'criteria': {'true': q.true_criteria, 'false': q.false_criteria}}
        candidates = {}
        for index in saved['indexes']:
            if index < 6:
                continue
            body = {'model': MODEL, 'state': {
                'previous_lines': [_line_payload(x) for x in lines[index - 6:index]],
                'current_line': _line_payload(lines[index])}, 'questions': {qid: question}}
            candidates[index] = body
        ordered = sorted(candidates, key=lambda i: (len(json.dumps(candidates[i])), i))
        quantiles = (.25, .5, .75) if dataset == 'C' else (.25, .75)
        selected = [ordered[round((len(ordered) - 1) * q)] for q in quantiles]
        if dataset == 'C':
            selected.append(min((i for i in ordered if i not in selected),
                                key=lambda i: (abs(probabilities[i] - .92), i)))
        target = prompt if dataset == 'C' else relevance
        for i in selected:
            target.append({'id': f'{dataset}-{i}', 'category': 'prompt' if dataset == 'C' else 'relevance',
                           'source': dataset, 'source_index': i, 'body': candidates[i]})
    # Fixed interleaving, identical in every condition; not latency-based selection.
    rows = [row for group in zip(short, relevance, prompt) for row in group]
    assert len(rows) == 12
    for row in rows:
        row['body_json'] = json.dumps(row.pop('body'))  # matches original evaluator serialization
        row['request_sha256'] = digest(row['body_json'].encode())
    save('manifest.private.json', rows)
    metadata = [{k: v for k, v in r.items() if k != 'body_json'} for r in rows]
    hashes = {}
    for name in ['server.py', 'decider_mlx.py', 'config.json', 'decision_config.json',
                 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
        path = ROOT / 'model' / name
        if path.exists():
            hashes[name] = digest(path.read_bytes())
    packages = sorted(p.name for p in (ROOT / 'venv/lib/python3.12/site-packages').glob('*.dist-info'))
    save('manifest-metadata.json', {'model': MODEL, 'revision': REVISION, 'fixtures': metadata,
         'source_hashes': hashes, 'installed_packages': packages,
         'selection': 'Four saved short cases; A/B recorded context at request-byte quartiles 25/75; C quartiles 25/50/75 plus closest saved probability to .92; all real cases retain six prior lines.',
         'pass_order': ['sequential-1', 'concurrent-1', 'concurrent-2', 'sequential-2'],
         'warmup': 'First short, relevance and prompt fixture, once each, separately reported',
         'thresholds': {'prompt': .92, 'clip_seed': .1, 'clip_grow': .025, 'wide_seed': .05, 'wide_grow': .01}})
    print(json.dumps({'prepared': metadata, 'private_directory': str(OUT)}), flush=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(path: str, data: bytes | None = None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=180) as response:
        if response.geturl() != BASE + path:
            raise RuntimeError('Unexpected redirect')
        return json.load(response)


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, timeout=20, check=True)
    # LM Studio prints its empty-model inventory to stderr even on success.
    return (result.stdout or result.stderr).strip()


def recording_snapshot():
    processes = command('/bin/ps', '-axo', 'comm=')
    if 'AudioTapRecorder.app/Contents/MacOS/audio_tap_recorder' in processes:
        raise RuntimeError('Active audio tap: stop benchmark submissions')
    files = {}
    for directory in [RECORDINGS / 'Recordings', RECORDINGS / '.tmp']:
        for p in directory.iterdir():
            if p.is_file():
                s = p.stat()
                files[str(p)] = (s.st_mtime_ns, s.st_size)
    return files


def resources(label: str):
    pids = command('/usr/sbin/lsof', '-t', '-iTCP:8092', '-sTCP:LISTEN').splitlines()
    if len(set(pids)) != 1:
        raise RuntimeError('Ambiguous service process')
    return {'label': label, 'utc': datetime.now(timezone.utc).isoformat(),
            'stats': request('/stats'), 'pid': int(pids[0]),
            'process_rss_kib_cpu': command('/bin/ps', '-p', pids[0], '-o', 'rss=,pcpu=,time='),
            'vm_stat': command('/usr/bin/vm_stat'),
            'swapusage': command('/usr/sbin/sysctl', 'vm.swapusage'),
            'memory_pressure': command('/usr/bin/memory_pressure', '-Q'),
            'lmstudio_models': command(str(Path.home() / '.lmstudio/bin/lms'), 'ps')}


def run() -> None:
    if (OUT / 'results.json').exists():
        raise RuntimeError('Existing results: refuse accidental benchmark rerun')
    assert (ROOT / 'revision.txt').read_text().strip() == REVISION
    rows = json.loads((OUT / 'manifest.private.json').read_text())
    metadata = json.loads((OUT / 'manifest-metadata.json').read_text())
    for name, sha in metadata['source_hashes'].items():
        assert digest((ROOT / 'model' / name).read_bytes()) == sha
    for row in rows:
        assert digest(row['body_json'].encode()) == row['request_sha256']
    assert len(rows) == 12
    # The menu app has no status API. Require latest initialized capture to be completed,
    # no running system tap, and stop new submissions on any recording/temp-file change.
    latest = max((RECORDINGS / 'Recordings').glob('*.sidecar.json'), key=lambda p: p.stat().st_mtime)
    stem = latest.name.removesuffix('.sidecar.json')
    if not (RECORDINGS / 'Recordings' / f'{stem}.wav').exists():
        raise RuntimeError('Latest initialized capture has no saved audio; recorder may be active')
    archive = RECORDINGS / 'CaptureArchive' / stem
    if not any(json.loads(p.read_text()).get('stopped_wall_time') for p in archive.glob('*.json')):
        raise RuntimeError('Latest initialized capture has no stop record')
    initial_files = recording_snapshot()
    output = {'model': MODEL, 'revision': REVISION, 'manifest_sha256': digest((OUT / 'manifest.private.json').read_bytes()),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'warmup': [], 'passes': [], 'resources': [],
              'preflight': 'Latest sidecar has saved WAV and archive stop record; no audio tap; file-change guard active.'}
    save('results.json', output)  # checkpoint also prevents an unnoticed rerun after a failure

    def infer(row):
        if recording_snapshot() != initial_files:
            raise RuntimeError('Recording activity changed: stop benchmark submissions')
        start = time.perf_counter()
        response = request('/v1/systemone', row['body_json'].encode())
        elapsed = time.perf_counter() - start
        expected = json.loads(row['body_json'])['questions']
        assert response['model'] == MODEL and set(response['answers']) == set(expected)
        for key, answer in response['answers'].items():
            assert answer['type'] == expected[key]['type'] == 'noul'
            assert math.isfinite(answer['noul']) and 0 <= answer['noul'] <= 1
        return {'id': row['id'], 'category': row['category'], 'request_sha256': row['request_sha256'],
                'seconds': elapsed, 'answers': response['answers'], 'input_tokens': response['usage']['input_tokens']}

    output['resources'].append(resources('before-warmup'))
    for row in rows[:3]:
        result = infer(row)
        output['warmup'].append(result)
        save('results.json', output)
        print(json.dumps({'warmup': row['id'], 'seconds': result['seconds']}), flush=True)
    for label, concurrency in [('sequential-1', 1), ('concurrent-1', 4), ('concurrent-2', 4), ('sequential-2', 1)]:
        output['resources'].append(resources('before-' + label))
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(infer, rows))
        entry = {'condition': label, 'concurrency': concurrency, 'wall_seconds': time.perf_counter() - started, 'requests': results}
        output['passes'].append(entry)
        output['resources'].append(resources('after-' + label))
        save('results.json', output)
        print(json.dumps({'pass': label, 'n': len(results), 'wall_seconds': entry['wall_seconds'],
                          'median_seconds': statistics.median(r['seconds'] for r in results)}), flush=True)
    output['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save('results.json', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    prepare() if args.prepare else run()
