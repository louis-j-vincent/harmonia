"""Train audio→chord models that bridge the real-audio → perfect-MIDI gap.

For each tree level (family / seventh / exact chord) compare, with 5-fold
grouped-by-song cross-validation:

  fixed templates   — the pipeline's current nearest-template baseline (audio)
  audio model       — a classifier trained on Basic Pitch evidence (learns what
                      BP's smeared 3rd/7th actually looks like)
  audio + context   — same, plus the key prior, scale degree, and previous chord
  perfect ceiling   — the same classifier trained on ground-truth MIDI notes

and report how much of the (ceiling − baseline) gap the trained models close.
This is the "teach the audio step to hear the third/seventh" model.

Usage: .venv/bin/python scripts/train_audio_chord_model.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"


def onehot(a, n):
    m = np.zeros((len(a), n))
    m[np.arange(len(a)), a] = 1.0
    return m


def cosine_baseline(train_X, train_y, test_X, n_classes):
    """Nearest-template (row-normalized centroid, cosine) — the fixed-template
    baseline the pipeline uses."""
    cent = np.zeros((n_classes, train_X.shape[1]))
    for c in range(n_classes):
        rows = train_X[train_y == c]
        if len(rows):
            cent[c] = rows.mean(axis=0)
    cn = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
    xn = test_X / (np.linalg.norm(test_X, axis=1, keepdims=True) + 1e-9)
    return np.argmax(xn @ cn.T, axis=1)


def cv_accuracy(X, y, groups, model_fn, n_classes, baseline=False, chroma=None):
    """5-fold grouped CV accuracy. If baseline, use cosine nearest-template on
    `chroma` instead of a trained model."""
    gkf = GroupKFold(n_splits=5)
    accs = []
    split_X = chroma if baseline else X
    for tr, te in gkf.split(split_X, y, groups):
        if baseline:
            pred = cosine_baseline(chroma[tr], y[tr], chroma[te], n_classes)
        else:
            sc = StandardScaler().fit(X[tr])
            model = model_fn()
            model.fit(sc.transform(X[tr]), y[tr])
            pred = model.predict(sc.transform(X[te]))
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def save_final_models(d, audio, groups):
    """Train family/7th/exact models on ALL data and save them + the feature recipe."""
    import joblib
    out = REPO / "data" / "models"
    out.mkdir(parents=True, exist_ok=True)
    bundle = {"feature": "root-relative onset(12)+note(12)+bass(12)+treble(12) = 48-d",
              "labels": {}}
    for lvl, key in [("family", "family"), ("base7", "base7"), ("exact", "exact")]:
        y = d[key].astype(int)
        sc = StandardScaler().fit(audio)
        model = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(audio), y)
        joblib.dump({"scaler": sc, "model": model,
                     "labels": list(d[f"{key}_labels"]) if f"{key}_labels" in d else None},
                    out / f"audio_chord_{lvl}.joblib")
        bundle["labels"][lvl] = list(d[f"{key}_labels"]) if f"{key}_labels" in d else None
    print(f"Saved family/base7/exact models → {out}/audio_chord_*.joblib")


def main():
    d = np.load(FEAT, allow_pickle=True)
    onset, note, bass, treble = d["onset"], d["note"], d["bass"], d["treble"]
    perfect = d["perfect"]
    key_prior, degree, mode, prev_b7 = d["key_prior"], d["degree"], d["mode"], d["prev_b7"]
    groups = d["song"]

    audio = np.hstack([onset, note, bass, treble])                    # 48
    context = np.hstack([key_prior, onehot(degree, 12),
                         mode.reshape(-1, 1), onehot(prev_b7 + 1, 15)])  # +32
    audio_ctx = np.hstack([audio, context])

    def logreg():
        return LogisticRegression(max_iter=2000, C=1.0)

    def mlp():
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, alpha=1e-3)

    levels = [
        ("FAMILY (major/min/dim/aug/sus)", d["family"], 5),
        ("SEVENTH (base 7th chord, 14)", d["base7"], 14),
        ("EXACT chord (15)", d["exact"], 15),
    ]

    print(f"{len(audio)} instances, 5-fold grouped-by-song CV\n")
    header = f"{'level':<34}{'baseline':>10}{'audio-LR':>10}{'audio+ctx':>11}{'aud+ctx-MLP':>13}{'ceiling':>9}{'gap closed':>12}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, y, nc in levels:
        y = y.astype(int)
        base, _ = cv_accuracy(None, y, groups, None, nc, baseline=True, chroma=onset)
        a_lr, _ = cv_accuracy(audio, y, groups, logreg, nc)
        ac_lr, _ = cv_accuracy(audio_ctx, y, groups, logreg, nc)
        ac_mlp, _ = cv_accuracy(audio_ctx, y, groups, mlp, nc)
        ceil, _ = cv_accuracy(perfect, y, groups, logreg, nc)
        best = max(ac_lr, ac_mlp)
        gap = (best - base) / (ceil - base) if ceil > base else 0.0
        results[name] = (base, a_lr, ac_lr, ac_mlp, ceil, gap)
        print(f"{name:<34}{base:>9.1%}{a_lr:>10.1%}{ac_lr:>11.1%}{ac_mlp:>13.1%}"
              f"{ceil:>9.1%}{gap:>11.0%}")

    print("\nReading: 'baseline' = current fixed-template audio; 'ceiling' = same "
          "classifier on\nperfect MIDI notes; 'gap closed' = how far the best trained "
          "audio model moves\nfrom baseline toward the ceiling.")

    # channel ablation: the pipeline currently feeds only the onset channel.
    # How much do the sustain (note) + register-split channels add?
    print("\n— Audio channel value (trained LR, what to feed the model) —")
    channels = {
        "onset only (pipeline default)": onset,
        "onset+note": np.hstack([onset, note]),
        "onset+note+bass+treble (full)": audio,
    }
    for name, y, nc in levels:
        y = y.astype(int)
        cells = []
        for cname, feat in channels.items():
            a, _ = cv_accuracy(feat, y, groups, logreg, nc)
            cells.append(f"{a:.1%}")
        print(f"    {name:<34} " + "   ".join(
            f"{c}={v}" for c, v in zip(["onset", "+note", "+regs"], cells)))

    # progression test: does the previous chord + root motion (the real ii-V-I
    # signal) help, and does it help the seventh more than the family?
    prev_deg, root_interval = d["prev_deg"], d["root_interval"]
    prog = np.hstack([onehot(prev_b7 + 1, 15), onehot(prev_deg + 1, 13),
                      onehot(root_interval, 13)])
    print("\n— Does progression (prev chord + root motion, the ii-V-I signal) add unique info? —")
    for name, y, nc in levels:
        y = y.astype(int)
        without = np.hstack([audio, key_prior, onehot(degree, 12), mode.reshape(-1, 1)])
        with_prog = np.hstack([without, prog])
        a0, _ = cv_accuracy(without, y, groups, logreg, nc)
        a1, _ = cv_accuracy(with_prog, y, groups, logreg, nc)
        print(f"    {name:<34} without {a0:.1%} → with progression {a1:.1%}  (Δ {a1-a0:+.1%})")

    if "--save" in sys.argv:
        save_final_models(d, audio, groups)


if __name__ == "__main__":
    main()
