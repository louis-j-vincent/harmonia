"""Cheap premise-screen for extending the chord tree beyond 7ths (see
docs/extensions_beyond_7ths_2026-07-07.md).

Two questions, both answered without training anything:
  1. Label supply  — how often do 9/11/13/alt extensions occur in the corpus?
  2. Audio recoverability — in chords with NO extension, how much energy already
     sits on the tension pitch classes? A high incidental floor means a genuine
     extension is hard to separate from audio.

Usage: .venv/bin/python scripts/probe_extensions.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "accomp_db" / "db.jsonl"
FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"

EXT_PATS = [("9", re.compile(r"(?<![b#])9")), ("11", re.compile(r"(?<![b#])11")),
            ("13", re.compile(r"(?<![b#])13")), ("b9", re.compile(r"b9")),
            ("#9", re.compile(r"#9")), ("#11", re.compile(r"#11")),
            ("b13", re.compile(r"b13")), ("alt", re.compile(r"alt|^at$")),
            ("add", re.compile(r"add"))]


def label_supply():
    recs = [json.loads(l) for l in open(DB)]
    tot = with_ext = 0
    tally = Counter()
    for r in recs:
        for e in r["chord_timeline"]:
            q = re.sub(r"^[A-G][b#]?", "", e["ireal"].split("/")[0])
            tot += 1
            hit = any(pat.search(q) for _, pat in EXT_PATS)
            with_ext += hit
            for name, pat in EXT_PATS:
                if pat.search(q):
                    tally[name] += 1
    print(f"total chord instances: {tot}")
    print(f"with any extension: {with_ext} ({with_ext / tot:.1%})")
    for k, v in tally.most_common():
        print(f"  {k:5s} {v:6d}  {v / tot:.2%}")


def audio_floor():
    d = np.load(FEAT, allow_pickle=True)
    exact, exlab = d["exact"], list(d["exact_labels"])
    tens = {"9th": 2, "11th": 5, "#11": 6, "13th": 9, "b9": 1}
    chord_tones = {"dom7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10]}
    for feat in ("onset", "treble", "bass"):
        print(f"=== {feat} (tension pc energy as % of chord-tone level) ===")
        F = d[feat]
        for lab, pcs in chord_tones.items():
            m = F[exact == exlab.index(lab)]
            m = m / (m.sum(1, keepdims=True) + 1e-9)
            mean = m.mean(0)
            ct = mean[pcs].mean()
            s = "  ".join(f"{n}={mean[pc] / ct:.0%}" for n, pc in tens.items())
            print(f"  {lab}: chord-tone={ct:.3f} (uniform 0.083) | {s}")


if __name__ == "__main__":
    label_supply()
    print()
    audio_floor()
