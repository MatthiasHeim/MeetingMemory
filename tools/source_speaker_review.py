#!/usr/bin/env python3
"""Apply explicit human identity confirmations to individual transcript segments.

A confirmed sentence never labels an entire source, chunk or speaker cluster.
This tool writes a new local draft; it does not publish or trigger extraction.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from speaker_integrity import atomic_json


def segment_id(segment, payload):
    basis={k:segment.get(k) for k in ('source_id','start_seconds','end_seconds','text','speaker')}
    meta = payload.get('_meta') or {}
    identity = meta.get('job_key') or meta.get('durable_session_id')
    if not identity:
        raise ValueError('Transcript has no durable source identity for confirmation')
    basis['transcript_identity'] = identity
    return hashlib.sha256(json.dumps(basis,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def review_examples(payload, candidates=(), limit=6):
    examples=[];groups={}
    for seg in payload.get('segments',[]):
        if not isinstance(seg,dict) or not str(seg.get('text','')).strip():continue
        group=(seg.get('source_id'),seg.get('speaker'))
        # Prefer an identifiable sentence over an isolated acknowledgement.
        score = (len(seg['text'].split()) >= 4, -abs(len(seg['text']) - 90))
        if group not in groups or score > groups[group][0]:
            groups[group] = (score, seg)
    queues = {}
    for _, seg in groups.values():
        queues.setdefault(seg.get('source_id'), []).append(seg)
    selected = []
    while any(queues.values()) and len(selected) < limit:
        for queue in queues.values():
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    for seg in selected:
        examples.append({'segment_id':segment_id(seg, payload),'sentence':seg['text'],
                         'source_id':seg.get('source_id'),'start_seconds':seg.get('start_seconds'),
                         'end_seconds':seg.get('end_seconds'),
                         'options':list(dict.fromkeys([*candidates,'Someone else','Not sure'])),
                         'selected':None})
    return {'schema_version':1,'question':'Who said this sentence?','examples':examples,
            'instruction':'Confirm only the quoted segment. No names are preselected.'}


def apply_confirmations(payload, confirmations):
    out=copy.deepcopy(payload);segments=out.get('segments') or []
    index={segment_id(seg, payload):seg for seg in segments};seen=set();applied=[]
    for item in confirmations:
        key=item.get('segment_id');name=item.get('name');text=item.get('sentence')
        if key in seen:raise ValueError('Duplicate segment confirmation')
        seen.add(key)
        if key not in index or index[key].get('text')!=text:
            raise ValueError('Stale or mismatched sentence; review the current audio again')
        if item.get('basis')!='human_confirmed':
            raise ValueError('Identity must be explicitly human confirmed')
        if not isinstance(name,str) or not name.strip() or len(name)>100 or any(c in name for c in '\r\n:'):
            raise ValueError('Invalid confirmed name')
        if name.strip().lower() in ('not sure','someone else','unknown'):
            continue
        seg=index[key]
        if seg.get('confirmed_name') and seg['confirmed_name']!=name.strip():
            raise ValueError('Existing confirmation conflicts; preserve both for review')
        seg['confirmed_name']=name.strip();seg['identity_evidence']={'basis':'human_confirmed','segment_id':key,'sentence':text}
        applied.append(key)
    meta=out.setdefault('_meta',{});attribution=meta.setdefault('speaker_attribution',{})
    attribution.update(speaker_dependent_actions='hold',status='needs_review',accuracy_measured=False)
    attribution['confirmed_segment_ids']=sorted(set(attribution.get('confirmed_segment_ids',[])+applied))
    attribution['identity_basis']='human_confirmed_segments_only'
    # Display confirmed names only on matching segments; retain the source-local
    # label in each structured segment so the confirmation remains auditable.
    lines=[]
    for seg in segments:
        seconds=float(seg.get('start_seconds',0))
        if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid segment time')
        h,rest=divmod(int(seconds),3600);m,s=divmod(rest,60)
        stamp=f'{h:02d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'
        label=seg.get('confirmed_name') or seg.get('speaker') or seg.get('source_id','Unknown')
        lines.append(f'[{stamp}] {label}: {seg.get("text","")}')
    out['transcript']='\n'.join(lines)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('transcript',type=Path)
    p.add_argument('--confirmations',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--candidates',nargs='*',default=[]);args=p.parse_args()
    if args.transcript.resolve()==args.output.resolve():p.error('Write a new draft to preserve the input')
    payload=json.loads(args.transcript.read_text())
    result=apply_confirmations(payload,json.loads(args.confirmations.read_text())) if args.confirmations else review_examples(payload,args.candidates)
    atomic_json(args.output,result);print(str(args.output))

if __name__=='__main__':main()
