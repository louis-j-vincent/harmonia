"""nnls_batch.py — extend the NNLS-vs-BP48 screen to a broader corpus sample.

For each requested corpus song: download WAV, run from-scratch NNLS chroma,
aggregate per GT block, compute peak/mean + bass/full-argmax->root vs BP48 from
the existing corpus. One WAV at a time, deleted immediately (disk discipline).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scratchpad"))
from harmonia.data.yt_chord_corpus import download_audio
import nnls_chroma as N

AUDIO = REPO/"data/cache/billboard_60/audio"; AUDIO.mkdir(parents=True, exist_ok=True)
merged = {**json.load(open(REPO/"scratchpad/billboard_search_results.json")),
          **json.load(open(REPO/"scratchpad/billboard_search_results_60.json"))}
d = np.load(REPO/"data/cache/billboard_bp48_60_fixed_beatgrid.npz", allow_pickle=True)
SID=d["song_id"]; T0=d["t0"]; T1=d["t1"]; ROOT=d["root"].astype(int); F48=d["feat48_abs"]

def pm(v): v=np.asarray(v,float); return v.max()/(v.mean()+1e-12)
def l2(v): n=np.linalg.norm(v); return v/n if n>1e-9 else v

def run(tid):
    if tid not in merged: return None
    vid=merged[tid]["best"][0]; sid=f"bb_{tid}"
    m=SID==sid
    if m.sum()==0: return None
    t0=T0[m]; t1=T1[m]; root=ROOT[m]; f48=F48[m]
    o=np.argsort(t0); t0,t1,root,f48=t0[o],t1[o],root[o],f48[o]
    try:
        wav=download_audio(vid, AUDIO)
    except Exception as e:
        print(f"{sid} DOWNLOAD FAIL {e}", flush=True); return None
    try:
        ex=N.extract(wav)
    except Exception as e:
        print(f"{sid} EXTRACT FAIL {e}", flush=True); return None
    finally:
        wav.unlink(missing_ok=True)
    A,notes,times=ex["act"],ex["notes"],ex["times"]
    nn_pm=[]; nn_bass=[]; nn_full=[]
    for a,b in zip(t0,t1):
        full,bass,treb=N.block_chroma(A,notes,times,a,b)
        nn_pm.append(pm(np.hstack([l2(bass),l2(treb)])))
        nn_bass.append(int(bass.argmax())); nn_full.append(int(full.argmax()))
    nn_bass=np.array(nn_bass); nn_full=np.array(nn_full)
    r={"sid":sid,"n":int(m.sum()),
       "nnls_pm":float(np.mean(nn_pm)),
       "bp48_pm":float(np.mean([pm(v) for v in f48])),
       "nnls_bass_root":float((nn_bass==root).mean()),
       "nnls_full_root":float((nn_full==root).mean()),
       "bp48_bass_root":float((f48[:,24:36].argmax(1)==root).mean())}
    print(json.dumps(r), flush=True)
    return r

if __name__=="__main__":
    tids=sys.argv[1:]
    out=[run(t) for t in tids]; out=[r for r in out if r]
    p=REPO/"scratchpad/nnls_batch_results.json"
    prev=json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(prev+out, indent=2))
    print(f"\nwrote {len(out)} (total {len(prev)+len(out)})")
