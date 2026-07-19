"""nnls_real_extract.py — REAL Mauch NNLS-Chroma VAMP plugin head-to-head.

Mirrors scratchpad/nnls_dump_features.py exactly, but replaces the from-scratch
`nnls_chroma.extract`+`block_chroma` with the real `nnls-chroma:nnls-chroma`
VAMP plugin (bothchroma output, 24-dim bass|treble, index 0 = A).

Per block [t0,t1): mean bothchroma over frames, roll to C-first pc frame
(out_pc = roll(v12, 9)), L2-norm each 12-half, stack -> nnls24 (matches prior
representation). BP48 feat48_abs comes free from the corpus (same blocks).

Outputs scratchpad/nnls_real_feats.npz (nnls24, feat48a, root, sid) +
scratchpad/nnls_real_muddiness.json (per-song peak/mean, entropy, argmax proxy).
Downloads one WAV at a time, deletes immediately (disk discipline).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import librosa, vamp

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.data.yt_chord_corpus import download_audio

AUDIO = REPO / "data/cache/billboard_60/audio"; AUDIO.mkdir(parents=True, exist_ok=True)
merged = {**json.load(open(REPO / "scratchpad/billboard_search_results.json")),
          **json.load(open(REPO / "scratchpad/billboard_search_results_60.json"))}
d = np.load(REPO / "data/cache/billboard_bp48_60_fixed_beatgrid.npz", allow_pickle=True)
SID = d["song_id"]; T0 = d["t0"]; T1 = d["t1"]; ROOT = d["root"].astype(int)
F48A = d["feat48_abs"]
SR = 44100


def l2(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v

def peak_mean(v):
    v = np.asarray(v, float); return v.max() / (v.mean() + 1e-12)

def norm_entropy(v):
    v = np.asarray(v, float); s = v.sum()
    if s <= 1e-12: return 1.0
    p = v / s; p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(v)))


def extract_bothchroma(wav):
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    out = vamp.collect(y.astype(np.float32), SR, "nnls-chroma:nnls-chroma", output="bothchroma")
    step, arr = out["matrix"]           # arr (T,24), index 0 = A
    step = float(step)
    times = np.arange(arr.shape[0]) * step
    return arr, times


def block_both(arr, times, a, b):
    m = (times >= a) & (times < b)
    if m.sum() == 0:
        j = int(np.argmin(np.abs(times - 0.5 * (a + b)))); m = np.zeros(len(times), bool); m[j] = True
    seg = arr[m].mean(0)                 # (24,) index 0=A
    bass = np.roll(seg[:12], 9)          # -> C-first pc frame
    treb = np.roll(seg[12:], 9)
    return bass, treb


def run(tid):
    if tid not in merged: return None
    vid = merged[tid]["best"][0]; sid = f"bb_{tid}"; m = SID == sid
    if m.sum() == 0: return None
    t0 = T0[m]; t1 = T1[m]; root = ROOT[m]; f48a = F48A[m]
    o = np.argsort(t0); t0, t1, root, f48a = t0[o], t1[o], root[o], f48a[o]
    try:
        wav = download_audio(vid, AUDIO)
    except Exception as e:
        print(sid, "dlfail", e, flush=True); return None
    try:
        arr, times = extract_bothchroma(wav)
    except Exception as e:
        print(sid, "exfail", repr(e), flush=True); return None
    finally:
        wav.unlink(missing_ok=True)

    nn24, treb12 = [], []
    bass_arg, treb_arg = [], []
    for a, b in zip(t0, t1):
        bass, treb = block_both(arr, times, a, b)
        nn24.append(np.hstack([l2(bass), l2(treb)])); treb12.append(treb)
        bass_arg.append(int(bass.argmax())); treb_arg.append(int(treb.argmax()))
    nn24 = np.array(nn24, np.float32); treb12 = np.array(treb12)
    bass_arg = np.array(bass_arg); treb_arg = np.array(treb_arg)
    bp_bass_arg = f48a[:, 24:36].argmax(1); bp_note_arg = f48a[:, 12:24].argmax(1)

    mud = dict(sid=sid, n=int(m.sum()),
               nnls_pm24=float(np.mean([peak_mean(v) for v in nn24])),
               nnls_ent24=float(np.mean([norm_entropy(v) for v in nn24])),
               bp48_pm48=float(np.mean([peak_mean(v) for v in f48a])),
               bp48_ent48=float(np.mean([norm_entropy(v) for v in f48a])),
               nnls_pm_treb12=float(np.mean([peak_mean(v) for v in treb12])),
               bp48_pm_note12=float(np.mean([peak_mean(l2(v[12:24])) for v in f48a])),
               nnls_bass_root=float((bass_arg == root).mean()),
               nnls_treb_root=float((treb_arg == root).mean()),
               bp48_bass_root=float((bp_bass_arg == root).mean()),
               bp48_note_root=float((bp_note_arg == root).mean()))
    print(sid, "ok n=%d NNpm=%.2f BPpm=%.2f NNbassRt=%.2f BPbassRt=%.2f" %
          (mud["n"], mud["nnls_pm24"], mud["bp48_pm48"], mud["nnls_bass_root"], mud["bp48_bass_root"]), flush=True)
    return dict(sid=[sid] * int(m.sum()), nnls24=nn24, feat48a=f48a.astype(np.float32),
                root=root.astype(np.int64)), mud


if __name__ == "__main__":
    tids = sys.argv[1:] or json.load(open(REPO / "scratchpad/nnls_real_tids.json"))
    parts = []; muds = []
    for t in tids:
        r = run(t)
        if r: parts.append(r[0]); muds.append(r[1])
    if parts:
        out = dict(sid=np.array(sum((p["sid"] for p in parts), [])),
                   nnls24=np.vstack([p["nnls24"] for p in parts]),
                   feat48a=np.vstack([p["feat48a"] for p in parts]),
                   root=np.concatenate([p["root"] for p in parts]))
        np.savez(REPO / "scratchpad/nnls_real_feats.npz", **out)
        json.dump(muds, open(REPO / "scratchpad/nnls_real_muddiness.json", "w"), indent=2)
        print("saved", out["root"].shape, out["nnls24"].shape, "songs", len(parts))
