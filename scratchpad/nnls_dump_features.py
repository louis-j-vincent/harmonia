"""Dump per-block NNLS 24-dim features (+ matched BP48 feat48 + GT root) for a
list of corpus songs, into one npz, for a head-to-head trained root-head CV.
One WAV at a time, deleted immediately."""
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
SID=d["song_id"]; T0=d["t0"]; T1=d["t1"]; ROOT=d["root"].astype(int); F48=d["feat48"]  # root-relative? use feat48_abs
F48A=d["feat48_abs"]
def l2(v): n=np.linalg.norm(v); return v/n if n>1e-9 else v

def run(tid):
    if tid not in merged: return None
    vid=merged[tid]["best"][0]; sid=f"bb_{tid}"; m=SID==sid
    if m.sum()==0: return None
    t0=T0[m]; t1=T1[m]; root=ROOT[m]; f48a=F48A[m]
    o=np.argsort(t0); t0,t1,root,f48a=t0[o],t1[o],root[o],f48a[o]
    try: wav=download_audio(vid, AUDIO)
    except Exception as e: print(sid,'dlfail',e,flush=True); return None
    try: ex=N.extract(wav)
    except Exception as e: print(sid,'exfail',e,flush=True); return None
    finally: wav.unlink(missing_ok=True)
    A,notes,times=ex["act"],ex["notes"],ex["times"]
    nn=[]
    for a,b in zip(t0,t1):
        full,bass,treb=N.block_chroma(A,notes,times,a,b)
        nn.append(np.hstack([l2(bass),l2(treb)]))
    print(sid,'ok',m.sum(),flush=True)
    return dict(sid=[sid]*int(m.sum()), nnls24=np.array(nn,np.float32),
               feat48a=f48a.astype(np.float32), root=root.astype(np.int64))

if __name__=="__main__":
    tids=sys.argv[1:]; parts=[r for r in (run(t) for t in tids) if r]
    out=dict(sid=np.array(sum((p['sid'] for p in parts),[])),
             nnls24=np.vstack([p['nnls24'] for p in parts]),
             feat48a=np.vstack([p['feat48a'] for p in parts]),
             root=np.concatenate([p['root'] for p in parts]))
    p=REPO/"scratchpad/nnls_feats.npz"
    np.savez(p, **out); print('saved',out['root'].shape, out['nnls24'].shape)
