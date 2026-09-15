"""Tree level 2: the SEVENTH (base7), given root. Family (triad) is ~94%; can we
add the 7th (maj7/dom7/min7/min7b5/dim7/6/…) reliably, and does confidence-gating
let us report it 'only when confident' (the project's hierarchical rule)?

Runs on the extracted oracle-segment table (root-relative), no rendering. Reports:
  - base7 accuracy (14-way) by feature set, 5-fold by song, vs the 'perfect' ceiling;
  - whether the NOTE/sustain channel carries the 7th better than onset (7ths are held);
  - accuracy-vs-coverage under a confidence gate (max softmax prob): the deeper we
    only go when confident, the higher accuracy on the covered fraction.

Usage: .venv/bin/python scripts/seventh_model_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"


def cv_proba(X, y, grp):
    """5-fold-by-song out-of-fold predictions + max-prob confidence."""
    n = len(y); pred = np.zeros(n, int); conf = np.zeros(n)
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        P = clf.predict_proba(sc.transform(X[te]))
        pred[te] = clf.classes_[P.argmax(1)]; conf[te] = P.max(1)
    return pred, conf


def main():
    d = np.load(FEAT, allow_pickle=True)
    on, nt, ba, tr, perf, kp = (d["onset"], d["note"], d["bass"], d["treble"],
                                d["perfect"], d["key_prior"])
    b7 = d["base7"].astype(int); grp = d["song"]; labels = list(d["base7_labels"])
    n = len(b7)
    counts = np.bincount(b7, minlength=len(labels))
    print(f"{n} segments, {len(set(grp))} songs, {len(labels)} base-seventh classes")
    print("class mix:", {labels[i]: int(counts[i]) for i in np.argsort(-counts)[:8]}, "…")
    print(f"majority base rate: {counts.max() / n:.1%}\n")

    sets = {
        "onset only":            on,
        "note only":             nt,
        "onset+note+bass+treble": np.hstack([on, nt, ba, tr]),
        "  + key_prior":         np.hstack([on, nt, ba, tr, kp]),
        "perfect (GT-MIDI ceiling)": perf,
    }
    print(f"{'feature set':<28} {'base7 acc':>9}")
    best_pred = best_conf = None
    for name, X in sets.items():
        pred, conf = cv_proba(X, b7, grp)
        acc = (pred == b7).mean()
        print(f"{name:<28} {acc:>9.1%}")
        if name == "onset+note+bass+treble":
            best_pred, best_conf = pred, conf

    # confidence-gated accuracy vs coverage (only-go-deeper-when-confident)
    def gate(pred, conf, y, title):
        print(f"\n{title} — confidence gate (report only when max-prob >= t):")
        print(f"  {'thresh':>6} {'coverage':>9} {'acc@covered':>12}")
        for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
            mask = conf >= t
            acc = (pred[mask] == y[mask]).mean() if mask.any() else float("nan")
            print(f"  {t:>6.1f} {mask.mean():>9.1%} {acc:>12.1%}")

    gate(best_pred, best_conf, b7, "SEVENTH (base7)")

    # tree level 3: EXACT (extensions 9/11/13), same features
    ex = d["exact"].astype(int); ex_labels = list(d["exact_labels"])
    print(f"\nEXACT level: {len(ex_labels)} classes, majority base rate "
          f"{np.bincount(ex).max() / n:.1%}")
    ep, ec = cv_proba(np.hstack([on, nt, ba, tr]), ex, grp)
    print(f"  exact acc (onset+note+bass+treble): {(ep == ex).mean():.1%}   "
          f"perfect ceiling {(cv_proba(perf, ex, grp)[0] == ex).mean():.1%}")
    gate(ep, ec, ex, "EXACT")


if __name__ == "__main__":
    main()
