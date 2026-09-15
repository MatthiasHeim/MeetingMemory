import json
import logging
import sys
import types
import pytest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import transcribe_watcher as tw


def watcher(tmp_path):
    w=tw.TranscribeWatcher.__new__(tw.TranscribeWatcher)
    w.config={'processing':{'transcription_pipeline':'durable_sources'}}
    w.processing_mode='gemini';w.transcripts_dir=tmp_path;w.logger=logging.getLogger('test')
    return w


def test_resume_failed_only_not_uncertain_or_historical(tmp_path):
    w=watcher(tmp_path);wav=tmp_path/'recording.wav';target=wav.with_suffix('.json')
    target.write_text(json.dumps({'_meta':{'partial':True}}));assert not w._wav_needs_processing(wav)
    target.write_text(json.dumps({'_meta':{'transcription_pipeline':'durable_sources','retryable_ranges':[{'start':0,'end':5}]}}));assert w._wav_needs_processing(wav)
    target.write_text(json.dumps({'_meta':{'transcription_pipeline':'durable_sources','partial':True,'retryable_ranges':[]}}));assert not w._wav_needs_processing(wav)


def test_publish_before_seed_failure_and_preserve_previous(tmp_path,monkeypatch):
    import source_inputs
    monkeypatch.setattr(source_inputs,'prepare_recording_sources',lambda *a,**k:([{'source_id':'mic','path':'test.wav','start_seconds':0,'timing_basis':'unverified_sample_zero'}],{'capture_gaps':[]}))
    class Pipeline:
        def __init__(self,**kwargs):pass
        def run(self,*a,**k):return {'transcript':'[00:00] mic:chunk0:A: Hello','segments':[],'_meta':{'retryable_ranges':[]}}
    monkeypatch.setitem(sys.modules,'transcription_jobs',types.SimpleNamespace(SourceTranscriptionPipeline=Pipeline))
    w=watcher(tmp_path);wav=tmp_path/'recording.wav';target=wav.with_suffix('.json');target.write_text('{"transcript":"old"}')
    w._seed_insightbase_source=lambda p:(_ for _ in ()).throw(RuntimeError('database offline'))
    try:w._process_with_durable_sources(wav)
    except RuntimeError:pass
    saved=json.loads(target.read_text());assert saved['transcript'].endswith('Hello')
    assert saved['_meta']['speaker_attribution']['speaker_dependent_actions']=='hold'
    revisions=list((tmp_path/'.source-jobs/recording/published-revisions').glob('*.json'))
    assert len(revisions)==1 and json.loads(revisions[0].read_text())['transcript']=='old'


def test_durable_dispatch_does_not_require_legacy_gemini_processor(tmp_path,monkeypatch):
    w=watcher(tmp_path);w.gemini_processor=None;w._process_with_durable_sources=lambda p:'used'
    assert w._process_with_gemini(tmp_path/'x.wav')=='used'


def test_durable_startup_does_not_fall_back_to_noscribe(tmp_path, monkeypatch):
    config = {'paths': {'recordings': str(tmp_path / 'recordings'),
                        'transcripts': str(tmp_path / 'transcripts')},
              'processing': {'mode': 'gemini', 'transcription_pipeline': 'durable_sources'}}
    monkeypatch.setattr(tw, 'GEMINI_AVAILABLE', False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    w = tw.TranscribeWatcher(config, logging.getLogger('test'))
    assert w.processing_mode == 'gemini' and w.gemini_processor is None
    config['processing']['transcription_pipeline'] = 'misspelled_pipeline'
    with pytest.raises(ValueError, match='Unknown transcription'):
        tw.TranscribeWatcher(config, logging.getLogger('test'))
