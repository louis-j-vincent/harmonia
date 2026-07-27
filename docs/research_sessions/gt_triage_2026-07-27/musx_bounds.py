"""Per-boundary timing offset measured on the third-party model's log-posteriors.

Symmetric matched filter: at candidate time t, score = mean log P(outgoing chord)
over [t-HALF, t] + mean log P(incoming chord) over [t, t+HALF].  argmax t is where
the third-party model thinks the change really is.  offset = t - GT boundary.

music-x-lab's raw output is known to run ~+113 ms LATE against this benchmark
(docs/research_sessions/musx_frame_posteriors_2026-07-27.md); that constant is
reported alongside so a per-song median near +0.11 s reads as "aligned".
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
from triage_analyze import SONGS                                        # noqa: E402
from musx_shift import chord_col                                        # noqa: E402

MUSX_LATENCY = 0.113
HALF, SEARCH, STEP = 0.70, 2.5, 0.05


def offsets(song_id: str):
    gt = json.loads((REPO / "golden" / "brick0" / f"{song_id}.gt.json").read_text())
    tri, bas = frame_posteriors(REPO / gt["audio_path"])[:2]
    n = tri.shape[0]
    lt = np.log(np.clip(tri, 1e-6, None))
    lb = np.log(np.clip(bas, 1e-6, None))
    cs = gt["gt_chords"]

    def ll(c, a, b):
        lo, hi = max(int(round(a / FRAME_DT)), 0), min(int(round(b / FRAME_DT)), n)
        if hi <= lo:
            return None
        v = lt[lo:hi, chord_col(c["root_pc"], c["quality"])].mean()
        if c.get("bass_pc") is not None:
            v = 0.5 * v + 0.5 * lb[lo:hi, 1 + c["bass_pc"]].mean()
        return float(v)

    out = []
    for i in range(len(cs) - 1):
        a, b = cs[i], cs[i + 1]
        if a["root_pc"] is None or b["root_pc"] is None:
            continue
        if (a["root_pc"], a["quality"], a.get("bass_pc")) == \
           (b["root_pc"], b["quality"], b.get("bass_pc")):
            continue
        tb = float(a["t1"])
        lo_lim = max(float(a["t0"]) + HALF, tb - SEARCH)
        hi_lim = min(float(b["t1"]) - HALF, tb + SEARCH)
        if hi_lim - lo_lim < 0.4:
            continue
        ts, sc = [], []
        t = lo_lim
        while t <= hi_lim:
            la, lbv = ll(a, t - HALF, t), ll(b, t, t + HALF)
            if la is not None and lbv is not None:
                ts.append(t)
                sc.append(la + lbv)
            t += STEP
        if len(ts) < 5:
            continue
        ts, sc = np.array(ts), np.array(sc)
        i_ = int(sc.argmax())
        out.append({"i": i, "t_gt": tb, "t_audio": float(ts[i_]),
                    "offset": float(ts[i_] - tb),
                    "sharp": float(sc[i_] - np.median(sc)),
                    "from": a.get("label"), "to": b.get("label")})
    return out


if __name__ == "__main__":
    res = {}
    for s in sys.argv[1:] or SONGS:
        o = offsets(s)
        res[s] = o
        ok = [x for x in o if x["sharp"] > 0.30]
        if not ok:
            print(f"{s:<24} nothing identifiable")
            continue
        off = np.array([x["offset"] for x in ok])
        tt = np.array([x["t_gt"] for x in ok])
        sl, ic = np.polyfit(tt, off, 1)
        print(f"{s:<24} n={len(ok):3d}/{len(o):3d} median={np.median(off):+.3f}s "
              f"(latency-corrected {np.median(off)-MUSX_LATENCY:+.3f}) "
              f"MAD={np.median(np.abs(off-np.median(off))):.3f} "
              f"slope={sl*60:+.3f}s/min start{ic+sl*tt[0]:+.2f} end{ic+sl*tt[-1]:+.2f} "
              f"|off|>0.5: {np.mean(np.abs(off)>0.5):.0%}")
    (SCRATCH / "musx_bounds.json").write_text(json.dumps(res))
