"""Functional pattern abstraction: are tritone subs the same pattern? (user's idea)

A V7→I and a bII7→I are functionally the same (both dominants resolving to I), so
ii-V-I and ii-bII-I should be ONE pattern, not two. Encoding chords by function
(collapsing tritone-substitute dominants) should shrink the pattern vocabulary and
let a progression model learn faster from less data.

Collapse rule (first level of a "pattern tree"): a dominant-7th on scale degree d
and the one a tritone away (d+6) share the same tritone and resolution, so map both
to degree d mod 6. Everything else unchanged. This makes V7 (deg 7) ≡ bII7 (deg 1),
V7/V ≡ bII7/V, etc.

Measures, on the full jazz corpus (symbolic):
  1. how much the collapse concentrates 3-chord patterns (fewer distinct trigrams,
     more mass on the top ones);
  2. whether it merges ii-V-I with ii-bII-I (sanity);
  3. next-chord prediction as a function of TRAINING DATA SIZE — the abstraction
     should win most when data is scarce (its whole point).

Usage: .venv/bin/python scripts/experiment_pattern_abstraction.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord  # noqa: E402
from analyze_accomp_priors import merged_events, parse_key  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
BUCKET_BASE7 = {
    "maj": "majT", "6": "majT", "maj7": "maj7", "dom7": "dom7", "dom7alt": "dom7",
    "min": "minT", "m6": "minT", "min7": "min7", "minmaj7": "minmaj7",
    "dim": "dimT", "dim7": "dim7", "m7b5": "m7b5",
    "aug": "augT", "aug7": "aug7", "augmaj7": "augmaj7",
    "sus2": "susT", "sus4": "susT", "7sus4": "7sus4",
}
DEG = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]


def literal(tok):
    return tok  # (deg, b7)


def collapsed(tok):
    """Tritone-equivalence for dominants: V7 ≡ bII7 (degree mod 6)."""
    deg, b7 = tok
    if b7 in ("dom7", "aug7"):
        return (deg % 6, "DOM")     # dominant-function class, tritone-folded
    return tok


B7_FAMILY = {
    "majT": "M", "maj7": "M", "dom7": "M",
    "minT": "m", "min7": "m", "minmaj7": "m",
    "dimT": "d", "dim7": "d", "m7b5": "d",
    "augT": "A", "aug7": "A", "augmaj7": "A",
    "susT": "s", "7sus4": "s",
}


def enc_family(tok):
    """Degree + coarse family (drop the 7th) — a coarser pattern alphabet."""
    deg, b7 = tok
    return (deg, B7_FAMILY.get(b7, "?"))


def enc_tt_family(tok):
    """Tritone-fold dominants AND drop to family — the most abstract level."""
    deg, b7 = tok
    if b7 in ("dom7", "aug7"):
        return (deg % 6, "DOM")
    return (deg, B7_FAMILY.get(b7, "?"))


ENCODINGS = [
    ("literal (deg, 7th)", literal),
    ("tritone-folded dominants", collapsed),
    ("degree + family (no 7th)", enc_family),
    ("tritone-fold + family", enc_tt_family),
]


def build_sequences():
    seqs = []
    for rec in map(json.loads, open(DB)):
        if rec["corpus"] != "jazz1460":
            continue
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic = k[0]
        s = []
        for chord, _, _ in merged_events(rec):
            p = parse_chord(chord)
            if p is None or p[1] not in BUCKET_BASE7:
                continue
            s.append(((p[0] - tonic) % 12, BUCKET_BASE7[p[1]]))
        if len(s) >= 4:
            seqs.append((rec["song_id"], s))
    return seqs


def fmt(tok):
    d, q = tok
    return f"{DEG[d] if d < 12 else d}:{q}" if q != "DOM" else f"{DEG[d]}/{DEG[(d+5) % 12]}:DOM"


def main():
    seqs = build_sequences()
    n_chords = sum(len(s) for _, s in seqs)
    print(f"{len(seqs)} jazz songs, {n_chords} chords\n")

    # ── 1. pattern concentration: distinct trigrams + top-k coverage ──────────
    print("Pattern representation levels — how much each abstraction concentrates 3-chord patterns:")
    print(f"    {'encoding':<28}{'types':>7}{'trigrams':>10}{'top50 cover':>13}")
    for name, enc in ENCODINGS:
        tri = Counter()
        uni = Counter()
        for _, s in seqs:
            e = [enc(t) for t in s]
            for a, b, c in zip(e, e[1:], e[2:]):
                tri[(a, b, c)] += 1
            uni.update(e)
        tri_tot = sum(tri.values())
        top50 = sum(n for _, n in tri.most_common(50))
        print(f"    {name:<28}{len(uni):>7}{len(tri):>10}{top50/tri_tot:>12.0%}")
    print()

    # ── 2. sanity: does ii-V-I merge with ii-bII-I? ───────────────────────────
    iiVI = ((2, "min7"), (7, "dom7"), (0, "maj7"))
    iibIII = ((2, "min7"), (1, "dom7"), (0, "maj7"))
    lit = Counter()
    col = Counter()
    for _, s in seqs:
        for a, b, c in zip(s, s[1:], s[2:]):
            lit[(a, b, c)] += 1
            col[(collapsed(a), collapsed(b), collapsed(c))] += 1
    merged_key = (collapsed(iiVI[0]), collapsed(iiVI[1]), collapsed(iiVI[2]))
    print("Sanity — ii-V-I vs its tritone-sub ii-bII-I:")
    print(f"    literal   ii-V-I count   {lit[iiVI]:>5},  ii-bII-I count {lit[iibIII]:>5} (separate)")
    print(f"    collapsed merged pattern {col[merged_key]:>5} (= {lit[iiVI]}+{lit[iibIII]}"
          f"+other tritone variants)\n")

    # ── 3. next-chord prediction vs TRAINING DATA SIZE ────────────────────────
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    test = [seqs[i] for i in order[: len(seqs) // 5]]
    pool = [seqs[i] for i in order[len(seqs) // 5:]]

    def bigram_acc(train, enc):
        bg = defaultdict(Counter)
        ug = Counter()
        for _, s in train:
            e = [enc(t) for t in s]
            for a, b in zip(e, e[1:]):
                bg[a][b] += 1  # predict the ENCODED next token
            ug.update(e)
        ug_top = ug.most_common(1)[0][0] if ug else None
        best = {a: c.most_common(1)[0][0] for a, c in bg.items()}
        n = ok = 0
        for _, s in test:
            e = [enc(t) for t in s]
            for a, b in zip(e, e[1:]):
                n += 1
                ok += best.get(a, ug_top) == b
        return ok / n

    print("Next-chord prediction vs training-set size (each predicts its OWN encoded token):")
    print(f"    {'train songs':<12}" + "".join(f"{n.split(' (')[0][:16]:>18}" for n, _ in ENCODINGS))
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        ntr = max(int(len(pool) * frac), 1)
        train = pool[:ntr]
        cells = "".join(f"{bigram_acc(train, enc):>18.1%}" for _, enc in ENCODINGS)
        print(f"    {ntr:<12}{cells}")
    print("\nNote: coarser encodings predict a SMALLER target space, so higher accuracy")
    print("partly reflects an easier target — the load-bearing signal is pattern")
    print("concentration (1) and whether coarser levels degrade less with little data.")


if __name__ == "__main__":
    main()
