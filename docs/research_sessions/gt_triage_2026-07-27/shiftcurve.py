"""Sliding-window timing-offset curve: how far, and in which direction, does the
frozen GT sit away from the audio at each point in the song?

For each 16 s window we shift the GT chords bodily by tau and score them against
raw CQT chroma (chord-tone templates only - never our decode).  argmax tau is
the local offset; the peak sharpness says whether tau is identifiable at all.
A flat line at 0 = aligned; a constant offset = anchor error; a sloping line =
progressive DRIFT.
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

from harmonia.core.chroma import chroma_cqt_ltas          # noqa: E402
from triage_analyze import template, centre_norm, SONGS   # noqa: E402

W = 16.0
HOP = 4.0
TAUS = np.round(np.arange(-2.5, 2.501, 0.05), 3)


def curve(song_id: str):
    gt = json.loads((REPO / "golden" / "brick0" / f"{song_id}.gt.json").read_text())
    import librosa
    y, sr = librosa.load(str(REPO / gt["audio_path"]), sr=22050, mono=True)
    ch, ft = chroma_cqt_ltas(y, sr, hop_length=512)
    cs = [c for c in gt["gt_chords"] if c["root_pc"] is not None]
    tmpl = {}
    for c in cs:
        k = (c["root_pc"], c["quality"])
        if k not in tmpl:
            tmpl[k] = centre_norm(template(*k)[None])[0]

    def mfit(k, t0, t1):
        lo, hi = int(np.searchsorted(ft, t0)), int(np.searchsorted(ft, t1))
        if hi <= lo:
            return None
        v = centre_norm(ch[:, lo:hi].mean(axis=1)[None])[0]
        return float(tmpl[k] @ v)

    t0s, t1s = gt["gt_chords"][0]["t0"], gt["gt_chords"][-1]["t1"]
    out = []
    w0 = t0s
    while w0 + W <= t1s + HOP:
        w1 = min(w0 + W, t1s)
        sel = [c for c in cs if c["t1"] > w0 and c["t0"] < w1]
        # need real harmonic variety inside the window for tau to be identifiable
        nuniq = len({(c["root_pc"], c["quality"]) for c in sel})
        if len(sel) >= 2 and nuniq >= 2:
            sc = []
            for tau in TAUS:
                tot, ws = 0.0, 0.0
                for c in sel:
                    a, b = max(c["t0"], w0), min(c["t1"], w1)
                    f = mfit((c["root_pc"], c["quality"]), a + tau, b + tau)
                    if f is None:
                        continue
                    tot += f * (b - a)
                    ws += (b - a)
                sc.append(tot / ws if ws else np.nan)
            sc = np.array(sc)
            i = int(np.nanargmax(sc))
            out.append({"t": (w0 + w1) / 2, "t0": w0, "t1": w1,
                        "tau": float(TAUS[i]), "peak": float(sc[i]),
                        "at0": float(sc[np.argmin(np.abs(TAUS))]),
                        "sharp": float(sc[i] - np.nanmedian(sc)),
                        "nuniq": nuniq})
        else:
            out.append({"t": (w0 + w1) / 2, "t0": w0, "t1": w1, "tau": None,
                        "nuniq": nuniq})
        w0 += HOP
    return out


if __name__ == "__main__":
    res = {}
    for s in sys.argv[1:] or SONGS:
        c = curve(s)
        res[s] = c
        ok = [x for x in c if x.get("tau") is not None and x.get("sharp", 0) > 0.03]
        if ok:
            tt = np.array([x["t"] for x in ok])
            tau = np.array([x["tau"] for x in ok])
            sl, ic = np.polyfit(tt, tau, 1)
            print(f"{s:<24} n={len(ok):3d} median={np.median(tau):+.3f}s "
                  f"range=[{tau.min():+.2f},{tau.max():+.2f}] "
                  f"slope={sl*60:+.3f}s/min  start{ic+sl*tt[0]:+.2f} end{ic+sl*tt[-1]:+.2f}")
            print("   tau:", " ".join(f"{x['tau']:+.2f}" for x in c
                                      if x.get("tau") is not None))
            print("   shp:", " ".join(f"{x.get('sharp',0):.2f}" for x in c
                                      if x.get("tau") is not None))
        else:
            print(f"{s:<24} not identifiable")
    (SCRATCH / "shiftcurve.json").write_text(json.dumps(res))
