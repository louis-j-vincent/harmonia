"""Labeling #3b: quality/family emission (given root). Root is fixed (93% model);
the remaining majmin gap is the THIRD/quality. Runs entirely on the already-extracted
oracle-segment feature table (root-relative), so NO rendering — fast iteration.

Questions:
  - which chroma channel carries the third/seventh (onset vs note/sustain)?
  - does the KEY prior help pick the third (the user's longstanding lever)?
  - family (5-way) and third (maj/min/none, what majmin actually scores) accuracy,
    5-fold by song, vs the 'perfect' (GT-MIDI) ceiling.

Usage: .venv/bin/python scripts/quality_model_experiment.py
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
# family index → third class for majmin: maj-third / min-third / no-third
# family_labels = [major, minor, diminished, augmented, suspended]
THIRD = np.array([0, 1, 1, 0, 2])   # 0=maj third, 1=min third, 2=none(sus)


def cv_acc(X, y, grp, y2=None):
    """5-fold-by-song accuracy; if y2 given, also accuracy of THIRD[pred]==THIRD[true]."""
    n = len(y); pred = np.zeros(n, int)
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    fam = (pred == y).mean()
    third = (THIRD[pred] == THIRD[y]).mean() if y2 is None else None
    return fam, third, pred


def main():
    d = np.load(FEAT, allow_pickle=True)
    on, nt, ba, tr = d["onset"], d["note"], d["bass"], d["treble"]
    perf, kp = d["perfect"], d["key_prior"]
    fam = d["family"].astype(int); grp = d["song"]
    fam_labels = list(d["family_labels"])
    print(f"{len(fam)} oracle segments, {len(set(grp))} songs, families {fam_labels}")
    print(f"family base rate (majority): {np.bincount(fam).max() / len(fam):.1%}   "
          f"third base rate: {np.bincount(THIRD[fam]).max() / len(fam):.1%}\n")

    sets = {
        "onset only":            on,
        "note only":             nt,
        "onset+note":            np.hstack([on, nt]),
        "onset+note+bass+treble (current)": np.hstack([on, nt, ba, tr]),
        "  + key_prior":         np.hstack([on, nt, ba, tr, kp]),
        "perfect (GT-MIDI ceiling)": perf,
        "perfect + key_prior":   np.hstack([perf, kp]),
    }
    print(f"{'feature set':<36} {'family':>7} {'third':>7}")
    for name, X in sets.items():
        f, t, _ = cv_acc(X, fam, grp)
        print(f"{name:<36} {f:>7.1%} {t:>7.1%}")

    # confusion of the current model (where does the third go wrong?)
    _, _, pred = cv_acc(np.hstack([on, nt, ba, tr]), fam, grp)
    print("\ncurrent-model family confusion (row=true, col=pred):")
    C = np.zeros((5, 5), int)
    for t, p in zip(fam, pred):
        C[t, p] += 1
    print("            " + " ".join(f"{l[:5]:>6}" for l in fam_labels))
    for i, l in enumerate(fam_labels):
        print(f"  {l[:10]:<10} " + " ".join(f"{C[i, j]:>6}" for j in range(5)))


if __name__ == "__main__":
    main()
