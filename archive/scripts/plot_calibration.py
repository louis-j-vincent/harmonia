"""Reliability diagram — is the model's certainty calibrated? (guard-rail #4)

For the certainty↔structure loop to work, the confidence must MEAN something: when
the model says 0.8, it should be right ~80% of the time. This plots the reliability
diagram (predicted confidence vs actual accuracy) for the audio chord model at the
family / seventh / exact levels, with the expected calibration error (ECE).

Confidence = max softmax probability, from out-of-fold (grouped-by-song CV)
predictions so it's honest. Figure → docs/plots/certainty_calibration.png.

Usage: .venv/bin/python scripts/plot_calibration.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
PLOTS = REPO / "docs" / "plots"

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def oof(X, y, groups):
    conf = np.zeros(len(y)); correct = np.zeros(len(y), bool)
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))
        conf[te] = p.max(1)
        correct[te] = clf.classes_[p.argmax(1)] == y[te]
    return conf, correct


def reliability(conf, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= 1.0)
        if m.sum() >= 10:
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(m.sum())
            ece += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return np.array(xs), np.array(ys), np.array(ns), ece


def main():
    d = np.load(FEAT, allow_pickle=True)
    X = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    groups = d["song"]
    levels = [("family", d["family"].astype(int), "#4C72B0"),
              ("seventh", d["base7"].astype(int), "#DD8452"),
              ("exact", d["exact"].astype(int), "#55A868")]

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7.5, 9),
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.5, label="perfect calibration")
    for name, y, col in levels:
        conf, correct = oof(X, y, groups)
        xs, ys, ns, ece = reliability(conf, correct)
        ax.plot(xs, ys, "o-", color=col, lw=2, ms=6, label=f"{name}  (ECE {ece:.03f})")
        axh.hist(conf, bins=20, range=(0, 1), histtype="step", color=col, lw=1.6)
    ax.set_xlabel("model confidence (max softmax probability)")
    ax.set_ylabel("actual accuracy")
    ax.set_title("Certainty calibration — does confidence mean what it says?\n"
                 "(on/near the diagonal = calibrated; ECE = expected calibration error)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    axh.set_xlabel("confidence"); axh.set_ylabel("# chords")
    axh.set_title("Confidence distribution", fontsize=10)
    axh.set_xlim(0, 1)
    plt.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "certainty_calibration.png", dpi=140, bbox_inches="tight")
    print(f"→ {PLOTS/'certainty_calibration.png'}")


if __name__ == "__main__":
    main()
