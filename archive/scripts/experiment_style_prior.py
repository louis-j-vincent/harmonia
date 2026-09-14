"""Does knowing the music STYLE help predict chords? Hierarchical style prior.

User's idea: a prior/one-hot for the style (rhythm, cadence, structure) — better
hierarchical (broad genre → fine feel) — should help the chord/progression models,
since a blues, a jazz ballad and a dixieland tune use very different vocabularies.

Tested symbolically across 5 iReal corpora (jazz, pop, blues, country, dixieland).
Two things:
  1. how different the chord-quality mix is across styles (why a style prior can help);
  2. next-chord prediction: style-agnostic vs conditioned on broad style (corpus)
     vs fine style (the chart's own style tag, e.g. "Medium Swing" / "Bossa"),
     with a hierarchical back-off. Split by tune.

CSVs → docs/results/ for the blog.

Usage: .venv/bin/python scripts/experiment_style_prior.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords, to_mma_chord  # noqa: E402
from analyze_accomp_emission import parse_chord  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402
from build_audio_chord_features import BUCKET_BASE7, BUCKET_FAMILY  # noqa: E402

IREAL = REPO / "data" / "ireal"
RESULTS = REPO / "docs" / "results"
CORPORA = ["jazz1460", "pop400", "blues50", "country", "dixieland1"]


def fine_style(s):
    s = (s or "").lower()
    for key in ["up tempo swing", "medium up swing", "medium swing", "slow swing", "ballad",
                "bossa", "samba", "latin", "even 8", "rock", "rnb", "funk", "shuffle",
                "slow blues", "blues", "waltz", "swing"]:
        if key in s:
            return key
    return "other"


def build():
    tunes = []
    for c in CORPORA:
        p = IREAL / f"{c}.txt"
        if not p.exists():
            continue
        for t in load_playlist(p):
            k = parse_key(t.key)
            if k is None:
                continue
            tonic = k[0]
            seq = []
            try:
                measures = sectionized_measures(t)
            except Exception:
                continue
            for _, meas in measures:
                for tok in split_chords(meas):
                    mma = to_mma_chord(tok)
                    if mma is None or mma == "z":
                        continue
                    pc = parse_chord(mma)
                    if pc is None or pc[1] not in BUCKET_BASE7:
                        continue
                    seq.append(((pc[0] - tonic) % 12, BUCKET_BASE7[pc[1]]))
            if len(seq) >= 6:
                tunes.append({"broad": c, "fine": fine_style(t.style), "seq": seq})
    return tunes


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    tunes = build()
    print(f"{len(tunes)} tunes across {len(CORPORA)} corpora\n")

    # ── 1. chord-quality (family) mix per style ───────────────────────────────
    fam_by_style = defaultdict(Counter)
    for t in tunes:
        for _, b7 in t["seq"]:
            fam_by_style[t["broad"]][BUCKET_FAMILY_of(b7)] += 1
    fams = ["major", "minor", "diminished", "augmented", "suspended", "dominant"]
    print("Chord-family mix by style (share of chords):")
    print(f"    {'style':<12}" + "".join(f"{f[:5]:>8}" for f in fams))
    rows_mix = []
    for c in CORPORA:
        cc = fam_by_style.get(c)
        if not cc:
            continue
        tot = sum(cc.values())
        print(f"    {c:<12}" + "".join(f"{cc[f]/tot:>8.0%}" for f in fams))
        rows_mix.append({"style": c, **{f: round(cc[f] / tot, 3) for f in fams}})
    with open(RESULTS / "style_chord_mix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["style"] + fams); w.writeheader(); w.writerows(rows_mix)

    # ── 2. next-chord prediction: agnostic vs broad vs fine style ─────────────
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tunes))
    test = [tunes[i] for i in idx[: len(tunes) // 5]]
    train = [tunes[i] for i in idx[len(tunes) // 5:]]

    def fit(train, key):
        tbl = defaultdict(lambda: defaultdict(Counter))  # ctxkey -> prev -> Counter(next)
        glob = defaultdict(Counter)
        for t in train:
            ck = key(t)
            for a, b in zip(t["seq"], t["seq"][1:]):
                tbl[ck][a][b] += 1
                glob[a][b] += 1
        gbest = {a: c.most_common(1)[0][0] for a, c in glob.items()}
        return tbl, gbest

    def evaluate(key):
        tbl, gbest = fit(train, key)
        # precompute best-per-context with backoff to global
        n = ok = 0
        for t in test:
            ck = key(t)
            ctx = tbl.get(ck, {})
            for a, b in zip(t["seq"], t["seq"][1:]):
                n += 1
                cand = ctx.get(a)
                pred = cand.most_common(1)[0][0] if cand and sum(cand.values()) >= 3 else gbest.get(a)
                ok += pred == b
        return ok / n

    agn = evaluate(lambda t: "ALL")
    broad = evaluate(lambda t: t["broad"])
    fine = evaluate(lambda t: (t["broad"], t["fine"]))
    print("\nNext-chord prediction (bigram + backoff), by style conditioning:")
    print(f"    style-agnostic            : {agn:.1%}")
    print(f"    + broad style (corpus)    : {broad:.1%}   (Δ {broad-agn:+.1%})")
    print(f"    + fine style (broad,feel) : {fine:.1%}   (Δ {fine-agn:+.1%})")
    with open(RESULTS / "style_prediction.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conditioning", "next_chord_acc", "gain_vs_agnostic"])
        w.writerow(["agnostic", round(agn, 3), 0.0])
        w.writerow(["broad_style", round(broad, 3), round(broad - agn, 3)])
        w.writerow(["fine_style_hierarchical", round(fine, 3), round(fine - agn, 3)])
    print(f"\nCSVs → {RESULTS}/style_chord_mix.csv, style_prediction.csv")


# base7 -> family (+ 'dominant' broken out, since it's the style-discriminating one)
_B7FAM = {}
for _bk, _fam in BUCKET_FAMILY.items():
    _B7FAM[BUCKET_BASE7[_bk]] = _fam
_DOM = {BUCKET_BASE7[b] for b in ("dom7", "dom7alt", "aug7", "7sus4")}


def BUCKET_FAMILY_of(b7):
    if b7 in _DOM:
        return "dominant"
    return _B7FAM.get(b7, "other")


if __name__ == "__main__":
    main()
