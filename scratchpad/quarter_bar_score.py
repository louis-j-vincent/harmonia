"""Scorer for the quarter-bar arms (see quarter_bar_run.py).

Per arm, against the frozen-7 Brick-0 GT (golden/brick0):
  * Brick-0 duration-weighted metrics (score_song(pred_chords=...)):
    mirex_root / partial_credit / strict — pooled over the 7;
  * boundary P/R/F1 at ±0.25 s (greedy 1:1 within-tol matching of chord-change
    instants, first onset excluded) — the metric a granularity change moves;
  * over-segmentation ratio  #pred / #GT chords;
  * residue histogram of predicted onsets on the detected beat grid
    (bar / half / quarter / offgrid) — did the decoder actually USE the level?

Run: .venv/bin/python scratchpad/quarter_bar_score.py [arm ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.eval.accuracy_score import chord_from_label, score_song  # noqa: E402

BRICK0 = REPO / "golden" / "brick0"
BEATS = REPO / "harmonia_min" / "state" / "beats"
PRED = REPO / "scratchpad" / "quarter_bar_pred"

FROZEN7 = ["bein_green", "blue_bossa", "blue_bossa_backing", "close_to_you",
           "every_breath_you_take", "georgia_on_my_mind", "stand_by_me"]

TOL = 0.25


def _grid(song_id: str):
    gt = json.loads((BRICK0 / f"{song_id}.gt.json").read_text())
    stem = Path(gt["audio_path"]).stem
    bd = json.loads((BEATS / f"{stem}.json").read_text())
    bt = np.asarray(bd["beats"], float)
    db = np.asarray(bd["downbeats"], float)
    step = float(np.median(np.diff(bt)))
    bpb = int(round(np.median(np.diff(db)) / step)) if len(db) >= 3 else 4
    bpb = bpb if 2 <= bpb <= 7 else 4
    db_idx = np.unique([int(np.argmin(np.abs(bt - t))) for t in db])
    phase = int(np.round(np.median(db_idx % bpb)))
    return gt, bt, bpb, phase, step


def _residues(onsets, bt, bpb, phase, step):
    hist = {"bar": 0, "half": 0, "quarter": 0, "off": 0}
    for t0 in onsets:
        i = int(np.argmin(np.abs(bt - t0)))
        if abs(bt[i] - t0) > 0.35 * step:
            hist["off"] += 1
            continue
        r = (i - phase) % bpb
        hist["bar" if r == 0 else "half" if r == bpb // 2 else "quarter"] += 1
    return hist


def _boundary_prf(pred_on, gt_on, tol=TOL):
    gt_left = sorted(gt_on)
    tp = 0
    used = [False] * len(gt_left)
    for p in sorted(pred_on):
        best, bi = tol + 1, -1
        for k, g in enumerate(gt_left):
            if used[k]:
                continue
            d = abs(p - g)
            if d < best:
                best, bi = d, k
        if bi >= 0 and best <= tol:
            used[bi] = True
            tp += 1
    fp, fn = len(pred_on) - tp, len(gt_left) - tp
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f = 2 * p * r / max(1e-9, p + r)
    return p, r, f


def score_arm(arm: str):
    num = tot = 0.0
    pooled = {"mirex_root": 0.0, "partial_credit": 0.0, "strict": 0.0}
    n_pred_all = n_gt_all = 0
    hist_all = {"bar": 0, "half": 0, "quarter": 0, "off": 0}
    bp = br = 0.0
    rows = []
    for song in FROZEN7:
        rec = json.loads((PRED / arm / f"{song}.json").read_text())
        chords = [chord_from_label(c["start_s"], c["end_s"], c["label"])
                  for c in rec["chords"]]
        s = score_song(BRICK0 / f"{song}.gt.json", pred_chords=chords)
        gt, bt, bpb, phase, step = _grid(song)
        gt_on = [float(c["t0"]) for c in gt["gt_chords"][1:]]
        pred_on = [float(c["start_s"]) for c in rec["chords"][1:]
                   if c["label"] != "N"]
        p, r, f = _boundary_prf(pred_on, gt_on)
        hist = _residues(pred_on, bt, bpb, phase, step)
        for k in hist_all:
            hist_all[k] += hist[k]
        w = s.duration_s
        num += 1
        tot += w
        for k in pooled:
            pooled[k] += getattr(s, k) * w
        n_pred_all += len(rec["chords"])
        n_gt_all += len(gt["gt_chords"])
        bp += p * w
        br += r * w
        rows.append((song, s.mirex_root, s.partial_credit, s.strict,
                     p, r, len(rec["chords"]), len(gt["gt_chords"]), hist))
    out = {k: v / tot for k, v in pooled.items()}
    out.update(bnd_p=bp / tot, bnd_r=br / tot,
               overseg=n_pred_all / n_gt_all, hist=hist_all)
    return out, rows


def main():
    arms = sys.argv[1:] or sorted(p.name for p in PRED.iterdir() if p.is_dir())
    print(f"{'arm':<12}{'root':>7}{'partial':>9}{'strict':>8}"
          f"{'bndP':>7}{'bndR':>7}{'ovseg':>7}   onset residues")
    per_song = {}
    for arm in arms:
        o, rows = score_arm(arm)
        per_song[arm] = rows
        h = o["hist"]
        print(f"{arm:<12}{o['mirex_root']:>7.4f}{o['partial_credit']:>9.4f}"
              f"{o['strict']:>8.4f}{o['bnd_p']:>7.3f}{o['bnd_r']:>7.3f}"
              f"{o['overseg']:>7.2f}   bar={h['bar']} half={h['half']} "
              f"q={h['quarter']} off={h['off']}")
    if len(arms) > 1:
        base = {r[0]: r for r in per_song[arms[0]]}
        print(f"\nper-song mirex_root deltas vs {arms[0]}:")
        for arm in arms[1:]:
            ds = [f"{r[0]}:{r[1] - base[r[0]][1]:+.4f}"
                  for r in per_song[arm] if abs(r[1] - base[r[0]][1]) > 5e-4]
            print(f"  {arm:<12} {'  '.join(ds) if ds else '(no change)'}")


if __name__ == "__main__":
    main()
