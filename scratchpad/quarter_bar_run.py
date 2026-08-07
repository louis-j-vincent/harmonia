"""Runner: harmonia_min redecode over the frozen 7, one dir per arm.

Arms (each = a `quarter_beats` value for harmonia_min.musx.redecode):
  * none      — half-bar only, the shipped behaviour (CONTROL — harmonia_min
                has never been scored on Brick 0, so this number comes first);
  * all       — quarter-bar transitions everywhere at beat_trans_penalty[2];
  * all_fixlat— same, but latency grid pinned to the control arm's chosen
                latency (the legal-transition set changes the latency pick;
                this arm isolates the decode effect from the latency shift).

Zero audio decode: cached beats (harmonia_min/state/beats) + cached musx
posteriors (data/cache/musx_probs). Dumps
scratchpad/quarter_bar_pred/<arm>/<song_id>.json:
  {"song_id","arm","latency_s","n_segments","chords":[{"label","start_s","end_s"}]}

Run: .venv/bin/python scratchpad/quarter_bar_run.py [arm ...]
     (default: none all all_fixlat, in that order)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia_min import musx as _musx  # noqa: E402

BRICK0 = REPO / "golden" / "brick0"
BEATS = REPO / "harmonia_min" / "state" / "beats"
PROBS = REPO / "data" / "cache" / "musx_probs"
OUT = REPO / "scratchpad" / "quarter_bar_pred"

FROZEN7 = ["bein_green", "blue_bossa", "blue_bossa_backing", "close_to_you",
           "every_breath_you_take", "georgia_on_my_mind", "stand_by_me"]


def _load(song_id: str):
    gt = json.loads((BRICK0 / f"{song_id}.gt.json").read_text())
    stem = Path(gt["audio_path"]).stem
    bd = json.loads((BEATS / f"{stem}.json").read_text())
    z = np.load(PROBS / f"{stem}.npz")
    probs = [z[n] for n in ("triad", "bass", "s7", "s9", "s11", "s13")]
    bt = np.asarray(bd["beats"], float)
    db = np.asarray(bd["downbeats"], float)
    step = float(np.median(np.diff(bt)))
    bpb = int(round(np.median(np.diff(db)) / step)) if len(db) >= 3 else 4
    bpb = bpb if 2 <= bpb <= 7 else 4
    return bt, db, bpb, probs


def run_arm(arm: str):
    (OUT / arm).mkdir(parents=True, exist_ok=True)
    for song in FROZEN7:
        bt, db, bpb, probs = _load(song)
        kw = {}
        if arm == "none":
            qb = None
        elif arm in ("all", "all_fixlat"):
            qb = "all"
        elif arm.startswith("all_p"):
            # all_p60 -> quarter everywhere with beat_trans_penalty[2] = 60
            qb = "all"
            kw["beat_trans_penalty"] = (15.0, 45.0, float(arm[5:]))
        else:
            raise SystemExit(f"unknown arm {arm}")
        if arm == "all_fixlat":
            ctrl = json.loads((OUT / "none" / f"{song}.json").read_text())
            kw["latency_grid"] = (float(ctrl["latency_s"]),)
        t = time.time()
        segments, latency = _musx.redecode(bt, probs, downbeat_times=db,
                                           beats_per_bar=bpb,
                                           quarter_beats=qb, **kw)
        rec = {"song_id": song, "arm": arm, "latency_s": latency,
               "n_segments": len(segments),
               "chords": [{"label": lab,
                           "start_s": round(float(t0), 3),
                           "end_s": round(float(t1), 3)}
                          for (t0, t1, lab) in segments]}
        (OUT / arm / f"{song}.json").write_text(json.dumps(rec))
        print(f"{arm:<11}{song:<22} {len(segments):>4} segs  "
              f"lat={latency * 1000:>5.0f}ms  {time.time() - t:.1f}s")


if __name__ == "__main__":
    arms = sys.argv[1:] or ["none", "all", "all_fixlat"]
    for a in arms:
        run_arm(a)
