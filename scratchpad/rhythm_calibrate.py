#!/usr/bin/env python3
"""rhythm_calibrate.py — same-section vs different-section calibration for the
rhythm SSM, mirroring the calibration Louis ran for the chord SSM in
pattern_slide.py (documented there as same-section 0.76-1.00 / different-
section 0.17-0.70 on This Love).

Protocol: use ``pattern_slide.segment()``'s output (a chord-SSM-derived
labelling of every slot, A/B/C/...) as the KNOWN reference. For every learned
block pattern (row0, d, label), slide it across the WHOLE song on a candidate
S matrix (chord or rhythm) using ``pattern_slide.slide_across_x``, sample the
diagonal score at every d-spaced grid position, and bucket the score as
"same-section" or "different-section" by comparing the label AT that position
(from a per-slot label array built from the merged blocks) against the
pattern's own label. Reported for chord_ssm first (sanity check: must
reproduce the documented range) then for each rhythm variant.

Run:  .venv/bin/python scratchpad/rhythm_calibrate.py [song-substring]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from pattern_slide import rigid_slots, chord_ssm, segment, slide_across_x, SLOTS_PER_BAR
from rhythm_ssm import rhythm_ssm
from section_merge_declined import _load_payload
from harmonia.models.rigid_grid import rigid_grid_for
from harmonia.serving.config import AUDIO_DIR


def per_slot_labels(blocks: list[dict], n: int) -> list[str]:
    lab = [None] * n
    for b in blocks:
        for t in range(b["start"], min(b["end"], n)):
            lab[t] = b["label"]
    return lab


def calibrate(S: np.ndarray, blocks: list[dict], labels: list[str]):
    """Returns dict(same=[scores], diff=[scores]) sampling every learned
    pattern's slide-diagonal curve at every d-spaced grid position."""
    n = S.shape[0]
    same, diff = [], []
    pairs_same, pairs_diff = [], []
    seen = set()
    for b in blocks:
        row0, d, lbl = b["row0"], b["d"], b["label"]
        key = (row0, d)
        if key in seen:
            continue
        seen.add(key)
        diag, _sq = slide_across_x(S, row0, d)
        for t in range(0, n - d + 1, d):
            if labels[t] is None:
                continue
            v = float(diag[t])
            if np.isnan(v):
                continue
            if labels[t] == lbl:
                same.append(v)
                pairs_same.append((row0, t, d, lbl))
            else:
                diff.append(v)
                pairs_diff.append((row0, t, d, lbl, labels[t]))
    return {"same": same, "diff": diff, "pairs_same": pairs_same, "pairs_diff": pairs_diff}


def report(name: str, cal: dict):
    s, d = cal["same"], cal["diff"]
    def rng(v):
        if not v:
            return "n=0"
        p25, p50, p75 = np.percentile(v, [25, 50, 75])
        return (f"{min(v):.3f}-{max(v):.3f}  n={len(v)}  mean={np.mean(v):.3f}  "
                f"median={p50:.3f}  IQR=[{p25:.3f},{p75:.3f}]")
    print(f"  [{name}] same-section:      {rng(s)}")
    print(f"  [{name}] different-section: {rng(d)}")
    if s and d:
        gap_lo = min(s) - max(d)
        gap_median = float(np.median(s) - np.median(d))
        print(f"  [{name}] range-gap (min-same - max-diff) = {gap_lo:+.3f}"
              + ("   CLEAN" if gap_lo >= 0 else "   overlap")
              + f"   |   median-gap (same - diff) = {gap_median:+.3f}")


def run_song(sub: str):
    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    tonic = int((P.get("home") or {}).get("tonic", 0)) % 12
    grid = rigid_grid_for(P.get("chords", []), tonic_pc=tonic)
    seq, names, n_bars, bar_sec = rigid_slots(P)
    S_chord = chord_ssm(names)
    n = S_chord.shape[0]

    print(f"\n{'=' * 70}\n{slug}  n_bars={n_bars} bar_sec={bar_sec:.3f}  n_slots={n}\n{'=' * 70}")
    blocks = segment(S_chord, seq, names, log=lambda *a, **k: None)
    labels = per_slot_labels(blocks, n)
    form = " ".join(f"{b['label']}x{b['mult']}" if b["mult"] > 1 else b["label"] for b in blocks)
    print(f"chord-SSM-derived form (the reference labelling): {form}")
    for b in blocks:
        print(f"    {b['label']} bars {b['start']//SLOTS_PER_BAR}-{(b['end']-1)//SLOTS_PER_BAR}"
              f"  ({b['d']//SLOTS_PER_BAR}-bar loop x{b['reps']})")

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    variants, source = rhythm_ssm(audio_path, grid, return_debug=True)
    print(f"drum source: {source}")

    print("\n-- calibration (slide-diagonal score, same vs different labelled section) --")
    cal_chord = calibrate(S_chord, blocks, labels)
    report("chord_ssm (sanity check)", cal_chord)
    cals = {}
    for vname, S in variants.items():
        cal = calibrate(S, blocks, labels)
        cals[vname] = cal
        report(f"rhythm:{vname}", cal)

    return dict(slug=slug, S_chord=S_chord, variants=variants, blocks=blocks,
                labels=labels, cal_chord=cal_chord, cals=cals, n=n, n_bars=n_bars,
                names=names, source=source)


if __name__ == "__main__":
    songs = sys.argv[1:] if len(sys.argv) > 1 else ["this_love"]
    for s in songs:
        run_song(s)
