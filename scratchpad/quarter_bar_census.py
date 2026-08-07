"""Census: how much GT chord mass needs QUARTER-BAR boundaries? (feat/quarter-bar)

Premise screen (CLAUDE.md rule #2) for opening the quarter-bar transition level
in harmonia_min's redecode. For every frozen-7 GT chord onset:

  * snap it to the detected beat grid (harmonia_min/state/beats, the SAME grid
    the decoder transitions on), classify its beat-in-bar residue:
    bar / half-bar / quarter-bar / off-grid;
  * measure the time from the onset to the nearest LEGAL half-bar boundary —
    summed, that is the UPPER BOUND on the overlap mass quarter-bar can recover.

Uses the retimed overlay GT (gt_repair_2026-07-27, beat-snapped) when present,
raw brick0 otherwise. Also prints the chords-per-bar histogram (bars holding
>= 3 chords are unwritable at half-bar granularity).

Run: .venv/bin/python scratchpad/quarter_bar_census.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BRICK0 = REPO / "golden" / "brick0"
OVERLAY = REPO / "golden" / "frozen_parity" / "gt_repair_2026-07-27"
BEATS = REPO / "harmonia_min" / "state" / "beats"

FROZEN7 = ["bein_green", "blue_bossa", "blue_bossa_backing", "close_to_you",
           "every_breath_you_take", "georgia_on_my_mind", "stand_by_me"]


def load_song(song_id: str):
    gt = json.loads((BRICK0 / f"{song_id}.gt.json").read_text())
    ov_path = OVERLAY / f"{song_id}.overlay.json"
    source = "brick0"
    chords = gt["gt_chords"]
    if ov_path.exists():
        ov = json.loads(ov_path.read_text())
        if ov.get("retimed"):
            chords = ov["gt_chords"]
            source = "overlay(retimed)"
    stem = Path(gt["audio_path"]).stem
    bd = json.loads((BEATS / f"{stem}.json").read_text())
    return chords, np.asarray(bd["beats"], float), \
        np.asarray(bd["downbeats"], float), source


def classify(song_id: str):
    chords, bt, db, source = load_song(song_id)
    step = float(np.median(np.diff(bt)))
    bpb = int(round(np.median(np.diff(db)) / step)) if len(db) >= 3 else 4
    bpb = bpb if 2 <= bpb <= 7 else 4
    # beat index of each downbeat -> residue class of beat i is (i - i_db) mod bpb
    db_idx = np.unique([int(np.argmin(np.abs(bt - t))) for t in db])
    phase = int(np.round(np.median(db_idx % bpb)))  # downbeat residue in beat idx

    half = bpb // 2
    counts = {"bar": 0, "half": 0, "quarter": 0, "offgrid": 0}
    stake_s = 0.0          # sum of onset -> nearest legal boundary distances
    quarter_onsets = []
    n_bars_ge3 = 0

    # chords per bar
    per_bar: dict[int, int] = {}
    for c in chords:
        t0 = float(c["t0"])
        b = int(np.searchsorted(db, t0 + 1e-6) - 1)
        per_bar[b] = per_bar.get(b, 0) + 1
    n_bars_ge3 = sum(1 for v in per_bar.values() if v >= 3)

    for c in chords[1:]:   # first onset: no boundary decision upstream of it
        t0 = float(c["t0"])
        i = int(np.argmin(np.abs(bt - t0)))
        if abs(bt[i] - t0) > 0.35 * step:
            counts["offgrid"] += 1
            continue
        r = (i - phase) % bpb
        if r == 0:
            counts["bar"] += 1
        elif r == half:
            counts["half"] += 1
        else:
            counts["quarter"] += 1
            quarter_onsets.append((t0, c["label"]))
            # nearest legal boundary: the closest beat whose residue is 0 or half
            legal = [j for j in range(max(0, i - bpb), min(len(bt), i + bpb + 1))
                     if (j - phase) % bpb in (0, half)]
            if legal:
                stake_s += float(min(abs(bt[j] - t0) for j in legal))
    dur = sum(float(c["t1"]) - float(c["t0"]) for c in chords)
    return {"song": song_id, "source": source, "bpb": bpb, "n": len(chords) - 1,
            **counts, "stake_s": round(stake_s, 2),
            "gt_dur_s": round(dur, 1), "bars_ge3": n_bars_ge3,
            "quarter_examples": quarter_onsets[:6]}


def main():
    rows = [classify(s) for s in FROZEN7]
    hdr = f"{'song':<22}{'src':<18}{'n':>4}{'bar':>5}{'half':>5}{'1/4':>5}" \
          f"{'off':>5}{'stake_s':>9}{'dur_s':>8}{'bars>=3':>8}"
    print(hdr)
    print("-" * len(hdr))
    tot = {"n": 0, "bar": 0, "half": 0, "quarter": 0, "offgrid": 0,
           "stake_s": 0.0, "gt_dur_s": 0.0, "bars_ge3": 0}
    for r in rows:
        print(f"{r['song']:<22}{r['source']:<18}{r['n']:>4}{r['bar']:>5}"
              f"{r['half']:>5}{r['quarter']:>5}{r['offgrid']:>5}"
              f"{r['stake_s']:>9.2f}{r['gt_dur_s']:>8.1f}{r['bars_ge3']:>8}")
        for k in tot:
            tot[k] += r[k]
    print("-" * len(hdr))
    print(f"{'TOTAL':<22}{'':<18}{tot['n']:>4}{tot['bar']:>5}{tot['half']:>5}"
          f"{tot['quarter']:>5}{tot['offgrid']:>5}{tot['stake_s']:>9.2f}"
          f"{tot['gt_dur_s']:>8.1f}{tot['bars_ge3']:>8}")
    q = tot["quarter"]
    print(f"\nquarter-bar onsets: {q}/{tot['n']} "
          f"({100.0 * q / max(1, tot['n']):.1f}%)  |  "
          f"upper-bound overlap at stake: {tot['stake_s']:.1f}s "
          f"of {tot['gt_dur_s']:.0f}s GT ({100 * tot['stake_s'] / tot['gt_dur_s']:.2f}%)")
    for r in rows:
        if r["quarter_examples"]:
            ex = ", ".join(f"{t:.1f}s {lab}" for t, lab in r["quarter_examples"])
            print(f"  {r['song']}: {ex}")


if __name__ == "__main__":
    main()
