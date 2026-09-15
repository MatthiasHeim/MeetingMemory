import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from source_speaker_review import segment_id,apply_confirmations,review_examples


def payload():
    return {'_meta': {'job_key': 'meeting-one-input-hash'}, 'segments':[{'source_id':'system','start_seconds':1,'end_seconds':2,'speaker':'system:chunk0:A','text':'Hello'}, {'source_id':'system','start_seconds':3,'end_seconds':4,'speaker':'system:chunk0:A','text':'Another sentence'}]}


def test_sentence_confirmation_does_not_expand_to_cluster():
    data=payload();s=data['segments'][0];c={'segment_id':segment_id(s,data),'name':'Person','sentence':'Hello','basis':'human_confirmed'}
    out=apply_confirmations(data,[c]);assert out['segments'][0]['confirmed_name']=='Person'
    assert 'confirmed_name' not in out['segments'][1] and 'confirmed_name' not in data['segments'][0]
    assert out['_meta']['speaker_attribution']['speaker_dependent_actions']=='hold'


def test_stale_or_model_confirmation_rejected():
    data=payload();s=data['segments'][0];c={'segment_id':segment_id(s,data),'name':'Person','sentence':'Wrong','basis':'human_confirmed'}
    with pytest.raises(ValueError,match='mismatched'):apply_confirmations(data,[c])
    c.update(sentence='Hello',basis='model')
    with pytest.raises(ValueError,match='human'):apply_confirmations(data,[c])


def test_examples_have_sentence_choices_and_no_default():
    review=review_examples(payload(),['Matthias','Person'])
    assert review['examples'][0]['sentence']=='Another sentence'
    assert review['examples'][0]['selected'] is None
    assert review['examples'][0]['options']==['Matthias','Person','Someone else','Not sure']


def test_same_sentence_from_another_recording_cannot_reuse_confirmation():
    data = payload()
    confirmation = {'segment_id': segment_id(data['segments'][0], data),
                    'sentence': 'Hello', 'name': 'Person', 'basis': 'human_confirmed'}
    data['_meta']['job_key'] = 'different-meeting-input-hash'
    with pytest.raises(ValueError, match='mismatched'):
        apply_confirmations(data, [confirmation])


def test_review_balances_sources_and_prefers_whole_sentences():
    data = payload()
    data['segments'] = [
        {'source_id': 'mic', 'speaker': 'mic:1', 'text': 'Mhm.'},
        {'source_id': 'mic', 'speaker': 'mic:1', 'text': 'We should review this together tomorrow.'},
        {'source_id': 'mic', 'speaker': 'mic:2', 'text': 'Another sentence from the microphone.'},
        {'source_id': 'system', 'speaker': 'system:1', 'text': 'I will send the updated proposal tomorrow.'},
    ]
    review = review_examples(data, ['Person'], limit=2)
    assert [e['source_id'] for e in review['examples']] == ['mic', 'system']
    assert review['examples'][0]['sentence'] != 'Mhm.'
