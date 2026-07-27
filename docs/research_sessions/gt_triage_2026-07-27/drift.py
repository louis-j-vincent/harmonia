"""Per-boundary timing offset: where does the audio actually change chord,
relative to where the frozen GT says it changes?

Method (independent of our decode): at every GT boundary where the chord symbol
really changes, slide a short analysis window across +/-2.5 s and score the raw
CQT chroma against the OUTGOING and INCOMING chord-tone templates.  The audio's
own change instant is the crossover of the two scores.  offset = crossover - GT
boundary.  A song with a progressive timing slip shows offsets that grow with
time; a naming error shows offsets scattered around zero.
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
from triage_analyze import (template, centre_norm, SONGS)  # noqa: E402

HALF = 0.60          # half-width of the local analysis window (s)
SEARCH = 2.5         # +/- search range around the GT boundary (s)
STEP = 0.05


def boundary_offsets(song_id: str):
    gt = json.loads((REPO / "golden" / "brick0" / f"{song_id}.gt.json").read_text())
    import librosa
    y, sr = librosa.load(str(REPO / gt["audio_path"]), sr=22050, mono=True)
    ch, ft = chroma_cqt_ltas(y, sr, hop_length=512)
    cs = gt["gt_chords"]

    def win(t0, t1):
        lo, hi = int(np.searchsorted(ft, t0)), int(np.searchsorted(ft, t1))
        if hi <= lo:
            return None
        return centre_norm(ch[:, lo:hi].mean(axis=1)[None])[0]

    out = []
    for i in range(len(cs) - 1):
        a, b = cs[i], cs[i + 1]
        if a["root_pc"] is None or b["root_pc"] is None:
            continue
        if (a["root_pc"], a["quality"]) == (b["root_pc"], b["quality"]):
            continue
        tb = float(a["t1"])
        # do not search past the neighbouring boundaries
        lo_lim = max(float(a["t0"]) + HALF, tb - SEARCH)
        hi_lim = min(float(b["t1"]) - HALF, tb + SEARCH)
        if hi_lim - lo_lim < 0.4:
            continue
        ta = centre_norm(template(a["root_pc"], a["quality"])[None])[0]
        tb_ = centre_norm(template(b["root_pc"], b["quality"])[None])[0]
        # symmetric matched filter: outgoing chord explains [t-HALF, t] AND
        # incoming chord explains [t, t+HALF].  argmax is the audio's own change
        # instant; the estimator is symmetric so it carries no built-in lag.
        ts, sc = [], []
        t = lo_lim
        while t <= hi_lim:
            wl, wr = win(t - HALF, t), win(t, t + HALF)
            if wl is not None and wr is not None:
                ts.append(t)
                sc.append(float(ta @ wl + tb_ @ wr))
            t += STEP
        if len(ts) < 5:
            continue
        ts, sc = np.array(ts), np.array(sc)
        cross = float(ts[int(sc.argmax())])
        conf = float(sc.max() - np.median(sc))
        out.append({"i": i, "t_gt": tb, "t_audio": cross,
                    "offset": None if cross is None else float(cross - tb),
                    "conf": conf,
                    "from": a.get("label"), "to": b.get("label")})
    return out


if __name__ == "__main__":
    res = {}
    for s in sys.argv[1:] or SONGS:
        o = boundary_offsets(s)
        res[s] = o
        ok = [x for x in o if x["offset"] is not None and x["conf"] > 0.10]
        if ok:
            offs = np.array([x["offset"] for x in ok])
            ts = np.array([x["t_gt"] for x in ok])
            sl, ic = np.polyfit(ts, offs, 1)
            print(f"{s:<24} n={len(ok):3d}/{len(o):3d} median={np.median(offs):+.3f}s "
                  f"IQR={np.percentile(offs,75)-np.percentile(offs,25):.3f} "
                  f"slope={sl*60:+.3f} s/min  (fit at t0 {ic+sl*ts[0]:+.2f} -> "
                  f"tEnd {ic+sl*ts[-1]:+.2f})")
        else:
            print(f"{s:<24} no usable boundaries")
    (SCRATCH / "offsets.json").write_text(json.dumps(res))
