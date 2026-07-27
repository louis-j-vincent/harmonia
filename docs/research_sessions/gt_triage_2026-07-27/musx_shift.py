"""Timing-offset curve measured with the THIRD-PARTY model's posteriors.

The chroma matched filter is weak when a song's chords share tones (Stand By Me:
A / F#m / D / E all overlap).  music-x-lab's frame posteriors give an independent
per-frame distribution over root+triad AND over the sounding bass note, which
separates those four chords cleanly.  For each window we shift the frozen GT by
tau and take the mean log-likelihood the third-party model assigns to the GT
chord at the shifted position; argmax tau is the local offset.

This never touches our own decode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
               "22c6b747-cea4-410f-85ae-f6ed58a9d2eb/scratchpad")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

from harmonia.models.musx_redecode import frame_posteriors, FRAME_DT   # noqa: E402
from triage_analyze import SONGS, triad_of                              # noqa: E402

MUSX_TRIADS = ["maj", "min", "sus4", "sus2", "dim", "aug"]
W, HOP = 16.0, 4.0
TAUS = np.round(np.arange(-3.0, 3.001, 0.05), 3)


def load(song_id):
    gt = json.loads((REPO / "golden" / "brick0" / f"{song_id}.gt.json").read_text())
    probs = frame_posteriors(REPO / gt["audio_path"])
    tri, bas = probs[0], probs[1]
    n = tri.shape[0]
    # log P(root+triad) per frame for each of the 72 chord classes, and log P(bass)
    lt = np.log(np.clip(tri, 1e-6, None))
    lb = np.log(np.clip(bas, 1e-6, None))
    return gt, lt, lb, n


def chord_col(root_pc: int, quality: str) -> int:
    t = triad_of(quality)
    ti = MUSX_TRIADS.index(t) if t in MUSX_TRIADS else 0
    return 1 + ti * 12 + root_pc


def curve(song_id: str, use_bass: bool = True):
    gt, lt, lb, n = load(song_id)
    cs = [c for c in gt["gt_chords"] if c["root_pc"] is not None]
    t0s, t1s = gt["gt_chords"][0]["t0"], gt["gt_chords"][-1]["t1"]
    out = []
    w0 = t0s
    while w0 + W <= t1s + HOP:
        w1 = min(w0 + W, t1s)
        sel = [c for c in cs if c["t1"] > w0 and c["t0"] < w1]
        nuniq = len({(c["root_pc"], c["quality"]) for c in sel})
        if len(sel) < 2 or nuniq < 2:
            out.append({"t": (w0 + w1) / 2, "tau": None, "nuniq": nuniq})
            w0 += HOP
            continue
        sc = []
        for tau in TAUS:
            tot, ws = 0.0, 0.0
            for c in sel:
                a, b = max(c["t0"], w0) + tau, min(c["t1"], w1) + tau
                lo, hi = int(round(a / FRAME_DT)), int(round(b / FRAME_DT))
                lo, hi = max(lo, 0), min(hi, n)
                if hi <= lo:
                    continue
                v = lt[lo:hi, chord_col(c["root_pc"], c["quality"])].mean()
                if use_bass and c.get("bass_pc") is not None:
                    v = v + lb[lo:hi, 1 + c["bass_pc"]].mean()
                tot += v * (hi - lo)
                ws += (hi - lo)
            sc.append(tot / ws if ws else np.nan)
        sc = np.array(sc)
        if np.all(np.isnan(sc)):
            out.append({"t": (w0 + w1) / 2, "tau": None, "nuniq": nuniq})
        else:
            i = int(np.nanargmax(sc))
            out.append({"t": (w0 + w1) / 2, "t0": w0, "t1": w1,
                        "tau": float(TAUS[i]), "peak": float(sc[i]),
                        "at0": float(sc[np.argmin(np.abs(TAUS))]),
                        "sharp": float(sc[i] - np.nanmedian(sc)), "nuniq": nuniq})
        w0 += HOP
    return out


if __name__ == "__main__":
    res = {}
    for s in sys.argv[1:] or SONGS:
        c = curve(s)
        res[s] = c
        ok = [x for x in c if x.get("tau") is not None]
        tau = np.array([x["tau"] for x in ok])
        tt = np.array([x["t"] for x in ok])
        sl, ic = np.polyfit(tt, tau, 1)
        print(f"{s:<24} n={len(ok):3d} median={np.median(tau):+.3f}s "
              f"|tau|>0.3: {np.mean(np.abs(tau) > 0.3):.0%}  "
              f"slope={sl*60:+.3f}s/min start{ic+sl*tt[0]:+.2f} end{ic+sl*tt[-1]:+.2f}")
        print("   tau:", " ".join(f"{x['tau']:+.2f}" for x in ok))
        print("   shp:", " ".join(f"{x['sharp']:.2f}" for x in ok))
    (SCRATCH / "musx_shift.json").write_text(json.dumps(res))
