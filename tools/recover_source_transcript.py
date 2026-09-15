#!/usr/bin/env python3
"""Recover a draft from original sources; never overwrite canonical transcripts.

Use --prepare-only to inspect source provenance before spending API budget.
Offsets are explicit diagnostic estimates, never speaker identity evidence.
"""
import argparse
import json
import os
from pathlib import Path
from source_inputs import prepare_recording_sources
from speaker_integrity import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--offsets-json',type=Path)
    parser.add_argument('--model',default='gemini-2.5-pro')
    parser.add_argument('--chunk-seconds',type=float,default=180)
    parser.add_argument('--request-timeout-seconds',type=float,default=120)
    parser.add_argument('--job-timeout-seconds',type=float,default=600)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--recover-stopped-native', action='store_true',
                        help='Recover closed native checkpoints only after its recorded PID has exited; leave original manifest unchanged')
    args=parser.parse_args()
    output=args.output_dir.expanduser().resolve()
    canonical=Path.home()/'Documents/MeetingRecorder/Transcripts'
    if output==canonical.resolve() or output.is_relative_to(canonical.resolve()):
        parser.error('Use a separate recovery directory, not canonical Transcripts')
    output.mkdir(parents=True,exist_ok=True)
    offsets=json.loads(args.offsets_json.read_text()) if args.offsets_json else None
    sources,capture=prepare_recording_sources(args.audio,output/'sources',manifest_path=args.manifest,offsets=offsets,
                                            recover_stopped=args.recover_stopped_native)
    atomic_json(output/'inputs.json',{'sources':sources,'_meta':capture})
    if args.prepare_only:
        print(json.dumps({'prepared_sources':len(sources),'manifest':str(output/'inputs.json'),'timeline_verified':capture['timeline_verified']}));return
    from transcription_jobs import SourceTranscriptionPipeline
    pipeline=SourceTranscriptionPipeline(state_dir=output/'jobs',model=args.model,
        chunk_seconds=args.chunk_seconds,request_timeout_seconds=args.request_timeout_seconds,
        job_timeout_seconds=args.job_timeout_seconds)
    payload=pipeline.run(sources,session_id=args.audio.stem,retry_failed=True,capture_gaps=capture.get('capture_gaps'))
    payload.setdefault('_meta',{}).update(capture_provenance=capture,transcription_pipeline='durable_sources',recovery_draft=True)
    payload['_meta']['partial'] = bool(payload['_meta'].get('missing_ranges') or capture.get('capture_errors'))
    if capture.get('capture_errors') and payload['_meta'].get('durable_job_status') == 'complete':
        payload['_meta']['durable_job_status'] = 'needs_review'
    atomic_json(output/'transcript.draft.json',payload)
    (output/'transcript.draft.txt').write_text(payload.get('transcript','')+'\n')
    print(json.dumps({'draft':str(output/'transcript.draft.json'),'status':payload['_meta'].get('durable_job_status'),'retryable_ranges':len(payload['_meta'].get('retryable_ranges',[]))}))

if __name__=='__main__':main()
