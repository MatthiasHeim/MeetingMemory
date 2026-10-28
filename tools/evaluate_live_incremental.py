"""Bounded live-ASR check using newly generated synthetic speech only (no microphone)."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
import soundfile as sf
from sidecar.live import GeminiLiveTranscriber, LiveEngine, TailSnapshot

TEXT = ('Please group related requests into one card. [[slnc 800]] '
        'Also highlight the exact source passages. [[slnc 1000]] '
        'We have decided to meet on Thursday. [[slnc 500]] '
        'I will bring the test report.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    aiff=args.out/'synthetic.aiff';wav=args.out/'synthetic.wav'
    subprocess.run(['say','-v','Samantha','-r','150','-o',str(aiff),TEXT],check=True)
    subprocess.run(['afconvert','-f','WAVE','-d','LEI16@16000',str(aiff),str(wav)],check=True)
    audio,rate=sf.read(wav,dtype='int16',always_2d=True);audio=audio[:,0]
    duration=len(audio)/rate
    diagnostics=[]; displayed=[]
    engine=LiveEngine(snapshot=lambda _:None,transcriber=GeminiLiveTranscriber(),on_diagnostics=diagnostics.append,on_update=lambda lines,cards,status:displayed.append(lines))
    ticks=[];end=0
    while end<duration:
        end=min(duration,end+6);begin=max(0,end-24)
        started=time.monotonic()
        engine.process_snapshot(TailSnapshot(audio[int(begin*rate):int(end*rate)],rate,None,rate,begin,end,closed=end==duration))
        row={'audio_end':round(end,3),'request_seconds':round(time.monotonic()-started,3),
             'committed_lines':len(engine.lines),'held':engine._held is not None,
             'displayed_lines':len(displayed[-1]) if displayed else 0,
             'text':[line.text for line in engine.lines]}
        ticks.append(row);print(json.dumps(row),flush=True)
        (args.out/'result.json').write_text(json.dumps({'synthetic_only':True,'duration':duration,'ticks':ticks,'diagnostics':diagnostics},indent=2)+'\n')

if __name__=='__main__':main()
