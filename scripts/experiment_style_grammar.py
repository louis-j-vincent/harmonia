"""b) What DIFFERS per style — the distinctive progressions (user: blues I-IV, jazz vi-ii-V-I).

experiment_style_prior showed styles share the ii-V-I grammar (conditioning helps
only +0.7%). But the DISTINCTIVE bit is real: which progressions is each style
unusually FOND of, relative to the corpus average? This finds them by lift =
P(bigram | style) / P(bigram | all), for bigrams with enough support — the
progressions that make a style sound like itself.

Symbolic, all 5 iReal corpora. Prints + CSV + a plot. No audio, disk-safe.

Usage: .venv/bin/python scripts/experiment_style_grammar.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords, to_mma_chord  # noqa: E402
from analyze_accomp_emission import parse_chord  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402

IREAL = REPO / "data" / "ireal"
RESULTS = REPO / "docs" / "results"
PLOTS = REPO / "docs" / "plots"
CORPORA = ["jazz1460", "pop400", "blues50", "country", "dixieland1"]
DEG = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]
NICE = {"jazz1460": "jazz", "pop400": "pop", "blues50": "blues",
        "country": "country", "dixieland1": "dixieland"}


def fam_short(bucket):
    f = BUCKET_FAMILY.get(bucket, "?")
    return {"major": "", "minor": "m", "diminished": "°", "augmented": "+", "suspended": "s"}[f]


def tok(root, tonic, bucket):
    return DEG[(root - tonic) % 12] + fam_short(bucket)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    style_bg = {c: Counter() for c in CORPORA}
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
                for ch in split_chords(meas):
                    mma = to_mma_chord(ch)
                    if not mma or mma == "z":
                        continue
                    pc = parse_chord(mma)
                    if pc and pc[1] in BUCKET_FAMILY:
                        seq.append(tok(pc[0], tonic, pc[1]))
            for a, b in zip(seq, seq[1:]):
                if a != b:
                    style_bg[c][(a, b)] += 1

    # global bigram distribution
    glob = Counter()
    for c in CORPORA:
        glob.update(style_bg[c])
    gtot = sum(glob.values())

    rows = []
    print("Distinctive progressions per style (lift = how much more this style uses it "
          "than average):\n")
    top_by_style = {}
    for c in CORPORA:
        st = style_bg[c]; stot = sum(st.values())
        if stot < 200:
            continue
        cand = []
        for bg, n in st.items():
            if n >= 15:                                   # enough support
                lift = (n / stot) / (glob[bg] / gtot)
                cand.append((lift, n / stot, bg))
        cand.sort(reverse=True)
        top = cand[:8]
        top_by_style[NICE[c]] = top
        print(f"  {NICE[c].upper():<10} " + "   ".join(
            f"{a}→{b} ({lift:.1f}×)" for lift, _, (a, b) in top[:6]))
        for lift, share, (a, b) in top:
            rows.append({"style": NICE[c], "progression": f"{a}->{b}",
                         "lift_vs_avg": round(lift, 2), "share_in_style": round(share, 4)})

    with open(RESULTS / "style_grammar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # plot: top distinctive progression per style
    fig, axes = plt.subplots(1, len(top_by_style), figsize=(3.4 * len(top_by_style), 4.2))
    for ax, (style, top) in zip(np.atleast_1d(axes), top_by_style.items()):
        labs = [f"{a}→{b}" for _, _, (a, b) in top[:6]][::-1]
        lifts = [lift for lift, _, _ in top[:6]][::-1]
        ax.barh(range(len(labs)), lifts, color="#4C72B0")
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=9)
        ax.set_title(style, fontweight="bold"); ax.axvline(1, color="#999", ls="--", lw=1)
        ax.set_xlabel("lift vs corpus avg")
    fig.suptitle("What makes each style sound like itself — its most over-used progressions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "style_grammar.png", dpi=140, bbox_inches="tight")
    print(f"\nCSV → {RESULTS/'style_grammar.csv'}   plot → {PLOTS/'style_grammar.png'}")


if __name__ == "__main__":
    main()
