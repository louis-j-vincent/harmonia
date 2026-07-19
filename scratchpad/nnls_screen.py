"""nnls_screen.py — Part 2 screen: NNLS-on-our-audio vs BP48-on-our-audio.

Same GT chord-segment blocks (t0/t1/root from the corpus). Downloads WAV,
runs from-scratch NNLS chroma, aggregates per block, computes the session's
muddiness diagnostic (peak/mean, norm-entropy) + bass/treble-argmax->root
proxy. BP48 numbers come free from the existing corpus feat48_abs. WAV deleted
after use (disk discipline).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scratchpad"))
from harmonia.data.yt_chord_corpus import download_audio
import nnls_chroma as N

SONGS = {  # tid -> (video_id, label)
    "362":  ("JRiAMe1zsQ0", "Last Kiss (HARD, BP48 root .05)"),
    "1111": ("3joI5VtuNV0", "Land of 1000 Dances (CLEAN, BP48 root .97)"),
    "1027": ("qNHWpi7CKgU", "Lucky (inv-heavy)"),
    "887":  ("NerII_Zgd5I", "Me Myself and I (inv story)"),
}
AUDIO = REPO/"data/cache/billboard_60/audio"; AUDIO.mkdir(parents=True, exist_ok=True)

d = np.load(REPO/"data/cache/billboard_bp48_60_fixed_beatgrid.npz", allow_pickle=True)
SID = d["song_id"]; T0 = d["t0"]; T1 = d["t1"]; ROOT = d["root"].astype(int)
F48 = d["feat48_abs"]


def peak_mean(v):
    v = np.asarray(v, float); s = v.sum()
    return v.max() / (v.mean() + 1e-12)

def norm_entropy(v):
    v = np.asarray(v, float); s = v.sum()
    if s <= 1e-12: return 1.0
    p = v / s; p = p[p > 0]
    return float(-(p*np.log(p)).sum() / np.log(len(v)))

def l2(v):
    n = np.linalg.norm(v); return v/n if n > 1e-9 else v


def run(tid):
    vid, label = SONGS[tid]; sid = f"bb_{tid}"
    m = SID == sid
    t0 = T0[m]; t1 = T1[m]; root = ROOT[m]; f48 = F48[m]
    order = np.argsort(t0); t0,t1,root,f48 = t0[order],t1[order],root[order],f48[order]

    wav = download_audio(vid, AUDIO)
    try:
        ex = N.extract(wav)
    finally:
        wav.unlink(missing_ok=True)
    A, notes, times = ex["act"], ex["notes"], ex["times"]

    nn_full24, nn_treb12 = [], []
    nn_bass_arg, nn_treb_arg, nn_full_arg = [], [], []
    for a, b in zip(t0, t1):
        full, bass, treb = N.block_chroma(A, notes, times, a, b)
        v24 = np.hstack([l2(bass), l2(treb)])
        nn_full24.append(v24); nn_treb12.append(treb)
        nn_bass_arg.append(int(bass.argmax())); nn_treb_arg.append(int(treb.argmax()))
        nn_full_arg.append(int(full.argmax()))
    nn_full24 = np.array(nn_full24); nn_treb12 = np.array(nn_treb12)
    nn_bass_arg = np.array(nn_bass_arg); nn_treb_arg = np.array(nn_treb_arg)
    nn_full_arg = np.array(nn_full_arg)

    # BP48 blocks (feat48_abs = [on,nt,bass,treb], each 12 L2-normed)
    bp_bass_arg = f48[:,24:36].argmax(1); bp_treb_arg = f48[:,36:48].argmax(1)
    bp_note_arg = f48[:,12:24].argmax(1)

    res = {
        "sid": sid, "label": label, "n": int(m.sum()),
        # muddiness — full vector
        "nnls_pm_full24": float(np.mean([peak_mean(v) for v in nn_full24])),
        "nnls_ent_full24": float(np.mean([norm_entropy(v) for v in nn_full24])),
        "bp48_pm_full48": float(np.mean([peak_mean(v) for v in f48])),
        "bp48_ent_full48": float(np.mean([norm_entropy(v) for v in f48])),
        # muddiness — single treble/note 12-chroma (cleanest apples-to-apples)
        "nnls_pm_treb12": float(np.mean([peak_mean(v) for v in nn_treb12])),
        "nnls_ent_treb12": float(np.mean([norm_entropy(v) for v in nn_treb12])),
        "bp48_pm_note12": float(np.mean([peak_mean(l2(v[12:24])) for v in f48])),
        "bp48_ent_note12": float(np.mean([norm_entropy(v[12:24]) for v in f48])),
        # root proxy
        "nnls_bass_root_acc": float((nn_bass_arg == root).mean()),
        "nnls_treb_root_acc": float((nn_treb_arg == root).mean()),
        "nnls_full_root_acc": float((nn_full_arg == root).mean()),
        "bp48_bass_root_acc": float((bp_bass_arg == root).mean()),
        "bp48_note_root_acc": float((bp_note_arg == root).mean()),
    }
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    tids = sys.argv[1:] or list(SONGS)
    out = [run(t) for t in tids]
    (REPO/"scratchpad/nnls_screen_results.json").write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY (root acc: bass | treble/note) ===")
    print(f"{'song':32s} {'NNLS pm24':>9} {'BP48 pm48':>9} {'NN bassRt':>9} {'BP bassRt':>9} {'NN trebRt':>9} {'BP noteRt':>9}")
    for r in out:
        print(f"{r['sid']+' '+r['label'][:26]:32s} {r['nnls_pm_full24']:9.2f} {r['bp48_pm_full48']:9.2f} "
              f"{r['nnls_bass_root_acc']:9.2f} {r['bp48_bass_root_acc']:9.2f} "
              f"{r['nnls_treb_root_acc']:9.2f} {r['bp48_note_root_acc']:9.2f}")
