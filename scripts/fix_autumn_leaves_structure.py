#!/usr/bin/env python3
"""Fix the Autumn Leaves chart-structure bug: the missing second-A final G-6.

Bug (found by the user, 2026-07-14): Autumn Leaves is a 32-bar AABC form where
each A section ends on a DOUBLE G-6 (Gm6 Gm6). The iReal chart correctly has the
double G-6 at the end of the FIRST A (bars 6-7) but only a SINGLE G-6 at the end
of the SECOND A. As a result every bar from the start of the bridge (B section)
onward is shifted LEFT by exactly one bar, and the 32-bar chorus is stored as 31
bars. The same omission repeats in the second chorus.

Fix: insert one G-6 (section A, downbeat) immediately before each A->B boundary,
then renumber every bar so indices are contiguous again. Multi-chord bars (the
ii-V "G-7 F#7" / "F-7 E7" bars in the C section, stored as two entries sharing a
bar index) are kept grouped.

This is a metric-position (bar-index) fix only. It does NOT change the trusted
head bars 0-7, so refitting the perfect grid (scripts/fit_beat_grid.py) yields
the SAME tempo but re-lays every downstream chord onto the correct grid slot.

Idempotent: detects the A->B boundary by "a lone G-6 in section A directly
followed by section B" and only inserts when the preceding bar is not already a
double G-6, so re-running on already-fixed data is a no-op.

Usage:
    python scripts/fix_autumn_leaves_structure.py \
        --annot docs/plots/annotations/irealb_autumn_leaves.html.json
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


def a_to_b_boundaries(chords):
    """Indices i where chords[i] is an A-section G-6 immediately followed by a
    B-section chord AND the bar before it is not already a G-6 (i.e. the A
    section currently ends on a *single* G-6 and needs a second one inserted)."""
    idxs = []
    for i in range(len(chords) - 1):
        c, nxt = chords[i], chords[i + 1]
        if c.get("label") != "G-6" or c.get("section") != "A":
            continue
        if nxt.get("section") != "B":
            continue
        prev = chords[i - 1] if i > 0 else None
        already_double = prev is not None and prev.get("label") == "G-6" \
            and prev.get("bar") != c.get("bar")
        if not already_double:
            idxs.append(i)
    return idxs


def insert_missing_g6(chords):
    chords = [dict(c) for c in chords]
    for i in reversed(a_to_b_boundaries(chords)):
        g6, nxt = chords[i], chords[i + 1]
        new = copy.deepcopy(g6)
        new["beat"] = 0
        new["section"] = "A"
        new["label"] = "G-6"
        # cosmetic times only (grid recomputes from bar index); keep monotone
        new["t0"] = g6.get("t1", g6.get("t0"))
        new["t1"] = nxt.get("t0", new["t0"])
        new["_inserted"] = True
        chords.insert(i + 1, new)
    return chords


def renumber_bars(chords):
    """Reassign contiguous bar indices. Two entries share a bar iff they had the
    same original bar and neither is a freshly-inserted downbeat."""
    for c in chords:
        c.setdefault("_orig_bar", c["bar"])
    new_bar = 0
    prev = None
    for c in chords:
        if prev is not None:
            same = (
                c["_orig_bar"] == prev["_orig_bar"]
                and not c.get("_inserted")
                and not prev.get("_inserted")
            )
            if not same:
                new_bar += 1
        c["bar"] = new_bar
        prev = c
    for c in chords:
        c.pop("_orig_bar", None)
        c.pop("_inserted", None)
    return chords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    args = ap.parse_args()

    p = Path(args.annot)
    d = json.loads(p.read_text(encoding="utf-8"))
    orig = d.get("chords", [])

    boundaries = a_to_b_boundaries([dict(c) for c in orig])
    if not boundaries:
        print("no A->B single-G-6 boundaries found; chart already correct — no-op")
        return

    fixed = renumber_bars(insert_missing_g6(orig))
    d["chords"] = fixed
    d["modified"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")

    print(f"inserted {len(boundaries)} G-6 bar(s) at A->B boundaries "
          f"(list idx {boundaries})")
    print(f"chords: {len(orig)} -> {len(fixed)}")
    print(f"max bar: {max(c['bar'] for c in orig)} -> {max(c['bar'] for c in fixed)}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
