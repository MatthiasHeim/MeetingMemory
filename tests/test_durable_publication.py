import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import neon_insert as ni


class Cursor:
    def __init__(self, existing):self.existing=existing;self.calls=[]
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def execute(self,sql,args):self.calls.append((sql,args))
    def fetchall(self):return self.existing
    def fetchone(self):return (11,)
class Connection:
    def __init__(self,cursor):self.c=cursor
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def cursor(self):return self.c
    def close(self):pass


def payload(tmp_path):
    p=tmp_path/'2026-09-15_10-00-00.json';p.write_text(json.dumps({'transcript':'recovered words','_meta':{'transcription_pipeline':'durable_sources','durable_session_id':'a'*64,'partial':True,'speaker_attribution':{'speaker_dependent_actions':'hold'}}}))
    return p


def test_retry_updates_same_source_under_transaction_lock(tmp_path,monkeypatch):
    p=payload(tmp_path);c=Cursor([(7,)]);monkeypatch.setattr(ni,'_get_conn',lambda:Connection(c))
    assert ni.insert_source(str(p),p.stem)==7
    assert 'pg_advisory_xact_lock' in c.calls[0][0]
    assert any('UPDATE sources' in s for s,a in c.calls)
    assert not any('INSERT INTO sources' in s for s,a in c.calls)
    assert c.calls[-1][1][0]=='recovered words'


def test_new_source_metadata_contains_idempotency_key(tmp_path,monkeypatch):
    p=payload(tmp_path);c=Cursor([]);monkeypatch.setattr(ni,'_get_conn',lambda:Connection(c))
    assert ni.insert_source(str(p),p.stem)==11
    assert json.loads(c.calls[-1][1][-1])['durable_session_id']=='a'*64


def test_duplicate_existing_rows_fail_closed(tmp_path,monkeypatch):
    p=payload(tmp_path);c=Cursor([(7,),(8,)]);monkeypatch.setattr(ni,'_get_conn',lambda:Connection(c))
    with pytest.raises(RuntimeError,match='Duplicate'):ni.insert_source(str(p),p.stem)
