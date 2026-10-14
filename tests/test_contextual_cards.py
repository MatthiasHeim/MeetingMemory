"""Source, relation, failure and audience regressions with synthetic evidence."""
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import numpy as np
import pytest
from sidecar.card_semantics import CardEvent, ContextualCardExtractor, split_sources
from sidecar.judges import JudgeError, _validated_choices
from sidecar.live import LiveCard, LiveEngine, LiveLine, TailSnapshot
from sidecar.panel_layout import visible_cards, source_ranges, utf16_length, transcript_content


def line(text, time=0, speaker='Ich'):
    return LiveLine(time, time + 3, speaker, text)


class ScriptedJudge:
    backend = 'synthetic'
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []
    def choose(self, state, questions):
        self.calls.append((state, questions))
        return next(self.answers)


def test_complete_thought_extends_corrects_and_keeps_unrelated_task_separate():
    lines = [line('Prepare the launch report.', 0), line('Include pricing.', 5),
             line('Actually, exclude confidential prices.', 10), line('Check the backup.', 15)]
    judge = ScriptedJudge([{'kind': 'request', 'relation': 'new'},
                           {'kind': 'request', 'relation': 'card_0'},
                           {'kind': 'request', 'relation': 'card_0'},
                           {'kind': 'request', 'relation': 'new'}])
    extractor = ContextualCardExtractor()
    for _ in lines:
        extractor.advance(lines, judge)
    assert len(extractor.cards) == 2
    assert [span.text for span in extractor.cards[0].spans] == [entry.text for entry in lines[:3]]
    assert extractor.cards[1].spans[0].text == 'Check the backup.'
    # Nothing is inferred about speaker identity, owners or dates by code.
    assert all(span.text == lines[span.line_index].text[span.start_char:span.end_char]
               for card in extractor.cards for span in card.spans)


def test_go_ahead_selects_minimal_relevant_context_instead_of_contextless_card():
    lines = [line('The weather is sunny. Group related requests into one card.', 0, 'Andere'),
             line('Yes, implement that.', 5)]
    judge = ScriptedJudge([{'kind': 'none', 'relation': 'new'},
                           {'kind': 'none', 'relation': 'new'},
                           {'kind': 'request', 'relation': 'context'},
                           {'include_0': 'exclude', 'include_1': 'include'}])
    extractor = ContextualCardExtractor()
    for _ in range(3): extractor.advance(lines, judge)
    assert len(extractor.cards) == 1
    assert [span.text for span in extractor.cards[0].spans] == ['Group related requests into one card.', 'Yes, implement that.']
    assert extractor.cards[0].spans[0].start_char == len('The weather is sunny. ')


def test_source_revision_rewinds_decisions_but_index_shift_maps_exact_spans():
    lines = [line('Ordinary talk.', 0), line('Choose green.', 5)]
    extractor = ContextualCardExtractor()
    judge = ScriptedJudge([{'kind':'none','relation':'new'}, {'kind':'decision','relation':'new'}])
    extractor.advance(lines, judge); extractor.advance(lines, judge)
    assert extractor.cards[0].kind == 'decision'
    extractor.reconcile([line('Ordinary talk.', 0), line('Choose blue.', 5)])
    assert not extractor.cards and len(extractor.events) == 1
    assert extractor.pending == 1


def test_native_source_ranges_use_utf16_and_reject_stale_evidence():
    lines = [line('Hello 👋. Please keep the robot visible.', 0)]
    spans = split_sources(lines)
    text, _ = transcript_content(lines)
    start, length = source_ranges(lines, [spans[1]])[0]
    assert text.encode('utf-16-le')[start*2:(start+length)*2].decode('utf-16-le') == spans[1].text
    assert source_ranges([line('Changed.', 0)], spans) == []
    assert utf16_length('👋') == 2


def test_private_suggestions_are_hidden_in_every_client_layout_and_unknown_mode():
    public = LiveCard('public', (), 'Ask about scope.', kind='suggestion_question', audience='client')
    private = LiveCard('private', (), 'Discuss an internal opportunity.', kind='suggestion_idea', audience='internal')
    extracted = LiveCard('grounded', (), 'Agreed on green.', kind='decision')
    for mode in ('client', 'presentation', 'compact', '', None):
        assert visible_cards((public, private, extracted), mode) == (public, extracted)
    assert visible_cards((public, private, extracted), 'internal') == (public, private, extracted)


def test_unknown_provider_choice_is_never_treated_as_permission():
    with pytest.raises(JudgeError):
        _validated_choices({'kind': {'choice':'invented'}}, {'kind': {'criteria': {'none':'No card'}}})


def test_contextual_worker_never_blocks_transcript_and_rejects_stale_result():
    entered, release = threading.Event(), threading.Event()
    class Judge:
        backend = 'synthetic'
        def choose(self, state, questions):
            if state['CURRENT']['text'] == 'Prepare the red report.':
                entered.set(); assert release.wait(3)
            return {'kind':'request','relation':'new'}
    class Transcriber:
        text = 'Prepare the red report.'
        def transcribe_tails(self, *args):
            return [{'speaker':'Ich','text':self.text,'start_offset':0,'end_offset':3}]
    transcriber = Transcriber()
    updates = []
    engine = LiveEngine(snapshot=lambda _: None, transcriber=transcriber, judge=Judge(),
                        contextual_cards=True, on_update=lambda lines,cards,status: updates.append((lines,cards)))
    snapshot = TailSnapshot(np.zeros(16),16000,None,16000,0,10,closed=True)
    engine.process_snapshot(snapshot)
    assert entered.wait(1)
    transcriber.text = 'Prepare the blue report.'
    engine.process_snapshot(snapshot)
    assert engine.lines[0].text == transcriber.text
    assert not engine.cards
    release.set()
    deadline = time.monotonic() + 3
    while engine._prompt_thread is not None and time.monotonic() < deadline:
        time.sleep(.01)
    assert len(engine.cards) == 1
    assert engine.cards[0].verbatim_text == transcriber.text
    assert all('red' not in card.verbatim_text for _, cards in updates for card in cards)


def test_failed_contextual_source_is_visible_and_later_task_still_processed(monkeypatch):
    monkeypatch.setattr('sidecar.live.PROMPT_RETRY_SECONDS', 0)
    class Judge:
        backend='synthetic'
        def choose(self,state,questions):
            if state['CURRENT']['text'] == 'Broken request.': raise RuntimeError('synthetic')
            return {'kind':'request','relation':'new'}
    engine=LiveEngine(snapshot=lambda _:None, judge=Judge(), contextual_cards=True, synchronous_prompts=True)
    engine.lines=[line('Broken request.',0),line('Check the backup.',5)]
    for _ in range(3): engine._request_prompt_work()
    assert engine._contextual.failed == 1
    assert engine.cards[0].verbatim_text == 'Check the backup.'
    assert '1 Stellen nicht ausgewertet' in engine.status


def test_retired_session_finishes_no_more_queued_provider_work():
    entered, release = threading.Event(), threading.Event()
    class Judge:
        backend='synthetic'
        calls=0
        def choose(self,state,questions):
            self.calls += 1
            entered.set(); assert release.wait(3)
            return {'kind':'request','relation':'new'}
    judge=Judge(); updates=[]
    engine=LiveEngine(snapshot=lambda _:None, judge=judge, contextual_cards=True,
                      on_update=lambda *args:updates.append(args))
    engine.lines=[line('Prepare the report.',0),line('Check the backup.',5)]
    engine._request_prompt_work(); assert entered.wait(1)
    engine.retire(); release.set()
    deadline=time.monotonic()+3
    while engine._prompt_thread is not None and time.monotonic()<deadline: time.sleep(.01)
    assert judge.calls == 1
    assert updates == []


def test_short_transcription_schedule_does_not_add_network_time_to_interval():
    starts=[]
    class Transcriber:
        def transcribe_tails(self,*args):
            starts.append(time.monotonic())
            time.sleep(.08)
            return []
    engine=LiveEngine(snapshot=lambda _:TailSnapshot(np.zeros(16),16000,None,16000,0,5),
                      transcriber=Transcriber(), interval=.14)
    engine.start()
    deadline=time.monotonic()+2
    while len(starts)<3 and time.monotonic()<deadline: time.sleep(.01)
    engine.retire(); engine._thread.join(1)
    assert len(starts)>=3
    assert max(b-a for a,b in zip(starts,starts[1:])) < .20


def test_suggestions_require_valid_source_ids_and_copy_as_suggestions():
    from types import SimpleNamespace
    from sidecar.card_assistant import CardAssistant
    class Fake:
        def _generate(self,prompt,schema):
            return SimpleNamespace(text='{"suggestions":[{"kind":"question","text":"What next?","source_ids":["invented"]}]}')
    assistant=CardAssistant(judge=Fake())
    with pytest.raises(JudgeError,match='unknown evidence'):
        assistant.suggest(split_sources([line('Prepare the report.')]), [], 'client')
    assert LiveCard('x',(),'Ask about scope.','suggestion_question',audience='client').copy_text.startswith('KI-Vorschlag:')


def test_client_publication_review_fails_closed_on_internal_or_uncertain_advice():
    from sidecar.card_assistant import client_safe_suggestions
    suggestions=[('question','Clarify the scope.',()),('idea','Private commercial coaching.',()),('idea','Ambiguous.',())]
    judge=ScriptedJudge([{'s0':'appropriate','s1':'private','s2':'unclear'}])
    assert client_safe_suggestions(suggestions,judge)==suggestions[:1]
    with pytest.raises(JudgeError,match='review unavailable'):
        client_safe_suggestions(suggestions,None)


def test_first_unfinished_caption_is_visible_but_not_classified_yet():
    class Transcriber:
        def transcribe_tails(self,*args):
            return [{'speaker':'Ich','text':'Please prepare the','start_offset':1,'end_offset':6}]
    judge=ScriptedJudge([]);updates=[]
    engine=LiveEngine(snapshot=lambda _:None,transcriber=Transcriber(),judge=judge,
                      contextual_cards=True,synchronous_prompts=True,
                      on_update=lambda lines,cards,status:updates.append(lines))
    engine.process_snapshot(TailSnapshot(np.zeros(16),16000,None,16000,0,6))
    assert not engine.lines and not engine.cards and not judge.calls
    assert updates[-1][0].text=='Please prepare the'
    assert updates[-1][0].provisional


def test_reusing_exact_evidence_extends_instead_of_duplicating_existing_card():
    extractor=ContextualCardExtractor()
    extractor.sources=split_sources([LiveLine(0,4,'Ich','Prepare the report.'),LiveLine(5,9,'Ich','Yes, do that.')])
    first,second=extractor.sources
    extractor.events=[CardEvent('request',None,(first.id,)),CardEvent('request',None,(first.id,second.id))]
    extractor._rebuild()
    assert len(extractor.cards)==1
    assert [span.text for span in extractor.cards[0].spans]==['Prepare the report.','Yes, do that.']


def test_suggestions_wait_for_published_contextual_result():
    calls=[]
    class Judge:
        backend='synthetic'
        def choose(self,state,questions):return {'kind':'none','relation':'new'}
    engine=LiveEngine(snapshot=lambda _:None,judge=Judge(),contextual_cards=True,synchronous_prompts=True,
                      suggest_cards=lambda *args:calls.append(args) or [])
    engine.lines=[LiveLine(0,10,'Ich','This is ordinary talk. The coffee is nice. The train arrived.')]
    engine._request_suggestions()
    assert calls==[]
    engine._request_prompt_work()
    deadline=time.monotonic()+2
    while engine._suggest_thread is not None and time.monotonic()<deadline:time.sleep(.01)
    assert len(calls)==1
    engine.lines.append(LiveLine(11,14,'Ich','A new source has not been classified.'))
    engine._suggest_last_at=0
    engine._request_suggestions()
    assert len(calls)==1


@pytest.mark.parametrize('text', [
    'Ich schicke dir den Bericht bis zum 15. März.',
    'Wir haben uns auf den 3. Oktober geeinigt.',
    'Bitte ruf z. B. Dr. Meier an und frag nach dem 2. Entwurf.',
    'Send this to Mr. Smith by 3. November.',
])
def test_dates_ordinals_and_abbreviations_remain_exact_complete_evidence(text):
    spans=split_sources([line(text)])
    assert len(spans)==1
    assert spans[0].text==text
    assert (spans[0].start_char,spans[0].end_char)==(0,len(text))


def test_sentence_selection_still_separates_unrelated_completed_statements():
    text='Lieferung am 15. März. Das Wetter ist schön. Bitte prüfe die Sicherung.'
    assert [span.text for span in split_sources([line(text)])]==[
        'Lieferung am 15. März.','Das Wetter ist schön.','Bitte prüfe die Sicherung.']


def test_retry_continues_after_stop_without_any_transcript_tick(monkeypatch):
    monkeypatch.setattr('sidecar.live.PROMPT_RETRY_SECONDS', .02)
    class Judge:
        backend='synthetic'
        calls=0
        def choose(self,state,questions):
            self.calls+=1
            if self.calls==1:raise RuntimeError('transient')
            return {'kind':'request','relation':'new'}
    judge=Judge()
    engine=LiveEngine(snapshot=lambda _:None,judge=judge,contextual_cards=True)
    engine.lines=[line('Prepare the report.')]
    engine.stop();engine._request_prompt_work()
    deadline=time.monotonic()+2
    while not engine.cards and time.monotonic()<deadline:time.sleep(.01)
    assert engine.cards and judge.calls==2
    assert not engine.status
    engine.retire()


def test_changed_unprocessed_tail_does_not_suppress_earlier_card():
    entered,release,second=threading.Event(),threading.Event(),threading.Event()
    class Judge:
        backend='synthetic'
        def choose(self,state,questions):
            if state['CURRENT']['text']=='Prepare the report.':
                entered.set(); assert release.wait(2)
                return {'kind':'request','relation':'new'}
            second.set();raise RuntimeError('keep tail pending')
    updates=[]
    engine=LiveEngine(snapshot=lambda _:None,judge=Judge(),contextual_cards=True,
                      on_update=lambda lines,cards,status:updates.append(cards))
    engine.lines=[line('Prepare the report.',0),line('Initial tail.',5)]
    engine._request_prompt_work();assert entered.wait(1)
    with engine._lock:engine.lines[1]=line('Revised tail.',5)
    release.set();assert second.wait(1)
    engine.retire()
    assert any(cards and cards[0].verbatim_text=='Prepare the report.' for cards in updates)


def test_personal_speaker_labels_stay_in_source_but_do_not_become_model_owners():
    from sidecar.card_assistant import CardAssistant
    from types import SimpleNamespace
    span=split_sources([line('I will send the report.',0,'Alice')])[0]
    assert span.speaker=='Alice' and span.payload()['speaker']=='Andere'
    seen=[]
    class Fake:
        def _generate(self,prompt,schema):
            seen.append(prompt);return SimpleNamespace(text='{"text":"I will send the report."}')
    card=LiveCard('x',(line('I will send the report.',0,'Alice').as_transcript_line(0),),'','action',(span,))
    assert CardAssistant(judge=Fake()).clean(card)=='I will send the report.'
    assert 'Alice' not in seen[0]


def test_suggestions_accept_unchanged_cited_evidence_and_report_stale_discards():
    original=[line('Prepare the report.',0),line('The coffee is warm.',5),line('The train arrived.',10)]
    sources=tuple(split_sources(original))
    diagnostics=[]
    engine=LiveEngine(snapshot=lambda _:None,contextual_cards=True,
                      suggest_cards=lambda *_:[('question','What scope?',(sources[0],)),
                                                ('idea','Check the arrival.',(sources[2],))],
                      on_diagnostics=diagnostics.append)
    # An unrelated line insertion rebases the still-valid evidence. Only the
    # changed cited tail is discarded, and a fresh attempt is requested.
    engine.lines=[line('Opening remark.',-5),*original[:2],line('The train was cancelled.',10)]
    retried=[]
    engine._request_suggestions=lambda:retried.append(True)
    engine._generate_suggestions(sources,(),'internal')
    cards=engine._suggestions['internal']
    assert len(cards)==1 and cards[0].clean_text=='What scope?'
    assert cards[0].spans[0].line_index==1
    assert diagnostics[-1]['card_count']==1
    assert diagnostics[-1]['discarded_count']==1
    assert retried==[True]


def test_retire_wakes_contextual_retry_wait_without_another_provider_call(monkeypatch):
    monkeypatch.setattr('sidecar.live.PROMPT_RETRY_SECONDS',30)
    attempted=threading.Event()
    calls=[]
    class Judge:
        def choose(self,*_):
            calls.append(True);attempted.set();raise RuntimeError('temporary')
    engine=LiveEngine(snapshot=lambda _:None,judge=Judge(),contextual_cards=True)
    engine.lines=[line('Prepare the report.')]
    engine._request_prompt_work();assert attempted.wait(1)
    worker=engine._prompt_thread
    engine.retire();worker.join(1)
    assert not worker.is_alive() and len(calls)==1
