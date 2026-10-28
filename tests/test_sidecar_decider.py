from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sidecar.gate import JudgeGateError, initialise_recording_sidecar
from sidecar.judges import DeciderJudge, JevJudge, JudgeError
from sidecar.live_session import RecordingLiveSession
from sidecar.recorder_support import jev_preference_enabled, write_jev_preference, preference_path
from sidecar.service import create_judge
from sidecar.transcript import TranscriptLine

STEM='2026-10-10_12-00-00'
def enabled(root):
    initialise_recording_sidecar(STEM,jev=True,external_attendees=None,root=root)

def test_decider_and_legacy_alias_use_actual_model_and_cache_identity(tmp_path):
    enabled(tmp_path)
    helper=tmp_path/'helper.py';helper.touch()
    judge=DeciderJudge(stem=STEM,recordings_root=tmp_path,helper_path=helper)
    assert JevJudge is DeciderJudge
    assert judge.backend=='decider'
    assert judge.model=='pplx-decider-v1.1-27b'
    cached=create_judge(STEM,requested='decider',recordings_root=tmp_path,cache_root=tmp_path/'cache',jev_factory=lambda **kw:judge)
    assert cached.backend=='decider'
    assert cached.model==judge.model

def test_parallel_decisions_preserve_order_and_context(tmp_path,monkeypatch):
    enabled(tmp_path);helper=tmp_path/'helper.py';helper.touch()
    judge=DeciderJudge(stem=STEM,recordings_root=tmp_path,helper_path=helper,audit_root=tmp_path/'audit')
    observed={}
    def call(body):
        current=body['state']['current_line']['line_index']
        observed[current]=[x['line_index'] for x in body['state']['previous_lines']]
        time.sleep((4-current)*.005)
        return {'q':{'type':'noul','noul':current/10}}
    monkeypatch.setattr(judge,'_call',call)
    lines=[TranscriptLine(i,float(i),f'00:0{i}','Speaker',f'Synthetic {i}') for i in range(5)]
    assert judge.judge(lines,{'q':'Is the line relevant?'},context=2)['q']==[0,.1,.2,.3,.4]
    assert observed=={0:[],1:[0],2:[0,1],3:[1,2],4:[2,3]}

def test_revoked_preference_stops_before_helper_and_outage_has_no_fallback(tmp_path,monkeypatch):
    enabled(tmp_path);helper=tmp_path/'helper.py';helper.touch()
    judge=DeciderJudge(stem=STEM,recordings_root=tmp_path,helper_path=helper)
    calls=[]
    def run(*args,**kwargs):
        calls.append(args);return SimpleNamespace(returncode=1,stdout='',stderr='offline')
    monkeypatch.setattr('sidecar.judges.subprocess.run',run)
    with pytest.raises(JudgeError,match='helper rejected'):judge._call({'state':'synthetic','questions':{}})
    assert len(calls)==1
    initialise_recording_sidecar(STEM,jev=False,external_attendees=None,root=tmp_path)
    with pytest.raises(JudgeGateError):judge._call({})
    assert len(calls)==1

def test_live_session_uses_local_factory_for_decisions_and_clips(tmp_path,monkeypatch):
    selected=[]
    local=SimpleNamespace(backend='decider')
    monkeypatch.setattr('sidecar.live_session.DeciderJudge',lambda **kw:selected.append(kw) or local)
    monkeypatch.setattr('sidecar.live_session.GeminiJudge',lambda **kw:pytest.fail('cloud decision fallback'))
    session=RecordingLiveSession(SimpleNamespace(),api_key=None,copy_text=lambda _:None,open_window=False,decision_stem=STEM,recordings_root=tmp_path)
    assert session.engine._judge_factory() is local
    assert session._judge_factory() is local
    assert all(x=={'stem':STEM,'recordings_root':tmp_path} for x in selected)

def test_new_preference_preserves_legacy_choice_and_explicit_false(tmp_path):
    cfg=tmp_path/'config.yaml';path=preference_path(cfg)
    path.write_text(json.dumps({'use_jev':True}))
    assert jev_preference_enabled(cfg)
    write_jev_preference(cfg,False)
    assert json.loads(path.read_text())=={'use_decider':False}
    assert not jev_preference_enabled(cfg)

def test_decider_clip_thresholds_do_not_change_other_models():
    from sidecar.decider_policy import clip_thresholds
    from sidecar.selection import select_clip
    lines=[TranscriptLine(i,float(i),f'00:0{i}','Speaker',f'Substantive example {i}') for i in range(3)]
    assert select_clip(lines,[.15,.04,.2]).has_fallback
    clip=select_clip(lines,[.15,.04,.2],**clip_thresholds('decider'))
    assert clip.indices==(0,1,2)
    assert clip.seed_threshold==.1 and clip.grow_threshold==.025
    assert clip_thresholds('gemini')=={}
