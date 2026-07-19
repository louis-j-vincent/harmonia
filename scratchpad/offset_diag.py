import json, sys
from pathlib import Path
import numpy as np, librosa, mirdata

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
AUDIO = REPO/"docs/audio"
vid = json.loads((REPO/"docs/plots/.yt_video_ids.json").read_text())
meta = json.loads((REPO/"docs/plots/.yt_audio_meta.json").read_text())

# billboard corpus files
import re
srv = (REPO/"scripts/harmonia_server.py").read_text()
m = re.search(r"_BILLBOARD_CORPUS_FILES = \[(.*?)\]", srv, re.S)
files=[]
for line in m.group(1).splitlines():
    mm=re.search(r'"([^"]+)"|scratchpad', line)
    if 'scratchpad' in line or 'SCRATCH' in line or '/' in line:
        pass
# simpler: just load the two known corpus jsons
corpus={}
for name in ["billboard_search_results.json","billboard_search_results_60.json"]:
    p=REPO/"scratchpad"/name
    if p.exists():
        try: corpus.update(json.loads(p.read_text()))
        except Exception as e: print("skip",name,e)
vid2track={}
for tid,v in corpus.items():
    best=v.get("best") or []
    if best: vid2track[best[0]]=tid

ds=mirdata.initialize("billboard")

# map audio files -> filename key -> video_id -> track
audiofiles=sorted(AUDIO.glob("*.m4a"))
fname_by_vid={v:k for k,v in vid.items()}

def gt_for_track(tid):
    cf=ds.track(tid).chords_full
    return [(float(a),float(b),str(l)) for (a,b),l in zip(cf.intervals,cf.labels)]

results=[]
for af in audiofiles:
    stem=af.stem
    key=f"inferred_{stem}.html"
    v=vid.get(key)
    tid=vid2track.get(v) if v else None
    if not tid:
        print(f"NO GT: {stem}  vid={v}")
        continue
    gt=gt_for_track(tid)
    # first musical GT chord change (skip leading N/X silence marker at t0)
    changes=[t0 for (t0,t1,l) in gt]
    # onset envelope
    y,sr=librosa.load(str(af),sr=22050,mono=True,duration=40)
    oenv=librosa.onset.onset_strength(y=y,sr=sr)
    times=librosa.times_like(oenv,sr=sr)
    onsets=librosa.onset.onset_detect(onset_envelope=oenv,sr=sr,units='time',backtrack=True)
    # audio start: first significant sound (RMS threshold)
    rms=librosa.feature.rms(y=y)[0]
    rt=librosa.times_like(rms,sr=sr)
    thr=0.02*np.max(rms)
    idx=np.argmax(rms>thr)
    audio_start=rt[idx]
    first_onset=onsets[0] if len(onsets) else float('nan')
    # GT first non-silence label start
    gt_first_chord=None
    for (t0,t1,l) in gt:
        if l not in ("N","X",""):
            gt_first_chord=t0; break
    gt_first_boundary=changes[1] if len(changes)>1 else changes[0]
    dur_audio=librosa.get_duration(path=str(af))
    gt_dur=gt[-1][1]
    results.append(dict(stem=stem,tid=tid,vid=v,
        gt0=gt[0][0], gt_first_chord=gt_first_chord,
        audio_start=round(float(audio_start),3),
        first_onset=round(float(first_onset),3),
        dur_audio=round(dur_audio,2), gt_dur=round(gt_dur,2),
        onsets_head=[round(float(x),3) for x in onsets[:6]],
        gt_head=[(round(t0,3),l) for (t0,t1,l) in gt[:6]]))
    print(f"\n=== {stem}  tid={tid} ===")
    print(f" audio_dur={dur_audio:.2f}  gt_dur={gt_dur:.2f}  diff={dur_audio-gt_dur:+.2f}")
    print(f" audio_start(RMS)={audio_start:.3f}  first_onset={first_onset:.3f}")
    print(f" GT first chord (non-N) start={gt_first_chord}")
    print(f" GT head: {[(round(t0,2),l) for (t0,t1,l) in gt[:6]]}")
    print(f" onsets head: {[round(float(x),2) for x in onsets[:8]]}")

json.dump(results, open(REPO/"scratchpad/offset_diag_results.json","w"), indent=2)
print("\nSaved.")
