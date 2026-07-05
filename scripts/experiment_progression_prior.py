"""Progression as a weighted PRIOR (not raw features) — does it help with the right weight?

Earlier, dumping previous-chord/root-motion one-hots into the classifier HURT
(overfit song-specific patterns under grouped CV). The user's hypothesis: it
overfit because the data was small, and the right way is a regularized prior with
a tuned weight. This tests exactly that:

  combined_score(quality) = log P_audio(quality) + w · log P_prog(quality | context)

where P_prog is a smoothed table P(quality | prev base-7th, root motion) fit on
train songs, and w is swept. If the best w is > 0 and beats audio-alone under
grouped-by-song CV, progression helps; if best w = 0, it doesn't (on this data).

Usage: .venv/bin/python scripts/experiment_progression_prior.py
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def fit_prog_table(prev_b7, root_interval, y, n_classes, idx):
    """P(quality | prev_b7, root_interval) with Laplace smoothing, from `idx` rows."""
    table = defaultdict(lambda: np.ones(n_classes) * 0.5)
    for i in idx:
        table[(prev_b7[i], root_interval[i])][y[i]] += 1
    return {k: v / v.sum() for k, v in table.items()}


def main():
    d = np.load(FEAT, allow_pickle=True)
    audio = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    prev_b7, root_interval = d["prev_b7"], d["root_interval"]
    groups = d["song"]
    print(f"{len(audio)} instances, {len(set(groups.tolist()))} songs, 5-fold grouped CV\n")

    weights = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
    for name, key in [("FAMILY", "family"), ("SEVENTH", "base7"), ("EXACT", "exact")]:
        y = d[key].astype(int)
        nc = int(y.max() + 1)
        gkf = GroupKFold(n_splits=5)
        acc_by_w = {w: [] for w in weights}
        for tr, te in gkf.split(audio, y, groups):
            sc = StandardScaler().fit(audio[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(audio[tr]), y[tr])
            # map predict_proba (over clf.classes_) into the full nc-wide space
            proba = clf.predict_proba(sc.transform(audio[te]))
            full = np.full((len(te), nc), 1e-9)
            full[:, clf.classes_] = proba
            log_audio = np.log(full + 1e-9)
            prog = fit_prog_table(prev_b7, root_interval, y, nc, tr)
            log_prog = np.zeros((len(te), nc))
            uniform = np.log(np.ones(nc) / nc)
            for r, i in enumerate(te):
                p = prog.get((prev_b7[i], root_interval[i]))
                log_prog[r] = np.log(p) if p is not None else uniform
            for w in weights:
                pred = np.argmax(log_audio + w * log_prog, axis=1)
                acc_by_w[w].append((pred == y[te]).mean())
        means = {w: float(np.mean(a)) for w, a in acc_by_w.items()}
        best_w = max(means, key=means.get)
        base = means[0.0]
        cells = "  ".join(f"w{w}:{means[w]:.1%}" for w in weights)
        print(f"{name:<9} {cells}")
        print(f"{'':<9} audio-alone {base:.1%} → best w={best_w} {means[best_w]:.1%} "
              f"(Δ {means[best_w]-base:+.1%})\n")


if __name__ == "__main__":
    main()
