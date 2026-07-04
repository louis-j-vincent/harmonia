"""
First supervised model for "will the chord change at this beat", built on
top of docs/chord_change_signal_analysis/features.csv (2026-07-04 joint-
signal investigation). Tries several architectures and evaluates with
leave-one-song-out (LOSO) cross-validation -- the right protocol given only
5 songs and the large per-song heterogeneity documented throughout that
investigation (e.g. song 003's near-zero phase-2 chord-change rate, song
001's atypically regular meter).

Feature set is deliberately restricted to signals that don't leak the
answer:
  - EXCLUDED: gt_root, gt_label (define chord_changed directly),
    B_bass_is_root_or_fifth (computed FROM gt_root), C_bigram_* (NaN unless
    chord_changed is already True, and computed from the post-change root --
    see docs/chord_change_signal_analysis/README.md; using bigram plausibility
    as an input to a real system requires scoring candidate next chords, not
    reading off the true one, which this feature table doesn't support yet).
  - EXCLUDED (or transformed): E_detected_period, E_loop_phase are
    per-song-constant (2 distinct period values across all 5 songs) --
    including them raw risks a model just memorizing "which song is this"
    rather than learning a real per-beat signal. E_position_in_loop is
    converted to a period-normalized fraction instead.

Usage:
    .venv/bin/python scripts/train_chord_change_classifier.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT_DIR = Path(__file__).parent.parent / "docs" / "chord_change_signal_analysis"

PHASE_BINS = [0, 1, 2, 3, 4]  # 4 absorbs the rare >=4 inter-downbeat gaps


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    df["phase_clipped"] = df["A_beat_phase"].clip(lower=0, upper=4).astype(int)
    phase_dummies = pd.get_dummies(df["phase_clipped"], prefix="phase")
    for b in PHASE_BINS:
        col = f"phase_{b}"
        if col not in phase_dummies:
            phase_dummies[col] = 0
    phase_dummies = phase_dummies[[f"phase_{b}" for b in PHASE_BINS]]

    period = df["E_detected_period"].replace(0, np.nan)
    loop_frac = (df["E_position_in_loop"] / period).fillna(0.0)

    X = pd.DataFrame({
        "beats_since_change": df["A_beats_since_change"].clip(upper=16),
        "bass_changed": df["B_bass_changed"].astype(int),
        "bass_onset": df["B_bass_onset"].astype(int),
        "onset_density": df["D_onset_density"],
        "chroma_dist": df["D_chroma_cosine_dist"],
        "dist_to_boundary": df["E_dist_to_segment_boundary"].clip(upper=16),
        "loop_position_frac": loop_frac,
    })
    X = pd.concat([X, phase_dummies.astype(int)], axis=1)

    # explicit phase x bass_changed interaction terms, for the linear model
    # variant -- logistic regression can't discover interactions on its own,
    # so this tests whether hand-adding the interaction findings_A_vs_BD.md
    # already validated lets a linear model close the gap to tree models.
    for b in PHASE_BINS:
        X[f"phase_{b}_x_bass_changed"] = phase_dummies[f"phase_{b}"] * X["bass_changed"]

    y = df["chord_changed"].astype(int).values
    return X, y


def loso_evaluate(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, model_fn, use_interactions: bool):
    interaction_cols = [c for c in X.columns if "_x_bass_changed" in c]
    cols = X.columns if use_interactions else [c for c in X.columns if c not in interaction_cols]
    songs = sorted(df["song_id"].unique())
    rows = []
    all_true, all_prob = [], []
    for held_out in songs:
        train_mask = df["song_id"] != held_out
        test_mask = df["song_id"] == held_out
        Xtr, Xte = X.loc[train_mask, cols].values, X.loc[test_mask, cols].values
        ytr, yte = y[train_mask.values], y[test_mask.values]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model = model_fn()
        model.fit(Xtr_s, ytr)
        prob = model.predict_proba(Xte_s)[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append({
            "song": held_out,
            "n": len(yte),
            "accuracy": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, zero_division=0),
            "auc": roc_auc_score(yte, prob) if len(set(yte)) > 1 else np.nan,
            "brier": brier_score_loss(yte, prob),
        })
        all_true.extend(yte)
        all_prob.extend(prob)

    all_true, all_prob = np.array(all_true), np.array(all_prob)
    all_pred = (all_prob >= 0.5).astype(int)
    # clip before log_loss: the hard-decision Dummy baselines emit exact
    # 0/1 probabilities, and log_loss is infinite the moment one of those is
    # wrong -- clipping avoids reporting a meaningless huge/inf number for a
    # baseline that was never meant to be evaluated as a calibrated model.
    prob_clipped = np.clip(all_prob, 1e-3, 1 - 1e-3)
    pooled = {
        "song": "POOLED",
        "n": len(all_true),
        "accuracy": accuracy_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
        "auc": roc_auc_score(all_true, all_prob),
        "brier": brier_score_loss(all_true, all_prob),
        "logloss": log_loss(all_true, prob_clipped),
    }
    return pd.DataFrame(rows), pooled


def main() -> None:
    df = pd.read_csv(OUT_DIR / "features.csv", dtype={"song_id": str})
    X, y = build_design_matrix(df)

    print(f"Rows: {len(df)}, positive rate: {y.mean():.1%}\n")

    baselines = {
        "Majority class": lambda: DummyMajority(),
        "Heuristic (phase==0 or bass_changed)": lambda: DummyHeuristic(),
    }
    models = {
        "Logistic (main effects)": (lambda: LogisticRegression(max_iter=2000, C=1.0), False),
        "Logistic (+ phase x bass interactions)": (lambda: LogisticRegression(max_iter=2000, C=1.0), True),
        "Decision Tree (depth=4)": (lambda: DecisionTreeClassifier(max_depth=4, random_state=0), False),
        "Random Forest (200 trees)": (lambda: RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0), False),
        "Gradient Boosting": (lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0), False),
        "MLP (16, 8)": (lambda: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=3000, random_state=0, alpha=1e-2), False),
    }

    results = []
    print("=== Baselines ===")
    for name, fn in baselines.items():
        per_song, pooled = loso_evaluate(df, X, y, fn, use_interactions=False)
        print(f"{name}: acc={pooled['accuracy']:.3f} f1={pooled['f1']:.3f} auc={pooled['auc']:.3f}")
        results.append((name, per_song, pooled))

    print("\n=== Models (leave-one-song-out CV) ===")
    for name, (fn, use_int) in models.items():
        per_song, pooled = loso_evaluate(df, X, y, fn, use_interactions=use_int)
        print(f"{name}: acc={pooled['accuracy']:.3f} f1={pooled['f1']:.3f} auc={pooled['auc']:.3f} "
              f"brier={pooled['brier']:.3f} logloss={pooled['logloss']:.3f}")
        print(per_song.to_string(index=False))
        results.append((name, per_song, pooled))

    # Feature importances from the two tree ensembles (fit on ALL data, for
    # a descriptive/interpretability view -- not part of the LOSO evaluation)
    print("\n=== Feature importances (fit on all 5 songs, descriptive only) ===")
    cols = [c for c in X.columns if "_x_bass_changed" not in c]
    scaler = StandardScaler().fit(X[cols].values)
    Xs = scaler.transform(X[cols].values)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0).fit(Xs, y)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(Xs, y)
    imp = pd.DataFrame({"feature": cols, "rf_importance": rf.feature_importances_,
                         "gb_importance": gb.feature_importances_}).sort_values("rf_importance", ascending=False)
    print(imp.to_string(index=False))

    write_report(results, imp)


class DummyMajority:
    def fit(self, X, y):
        self.p = y.mean()
        return self

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


class DummyHeuristic:
    """Reproduces the earlier joint-analysis heuristic directly: predict
    change if phase==0 (first one-hot column, index matching phase_0) or
    bass_changed, as a sanity floor the real models should beat."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X, _phase0_idx=7, _bass_idx=1):
        # column order from build_design_matrix: [..., bass_changed(1), ...,
        # phase_0..phase_4 at indices 7..11] -- see main() column ordering
        pred = ((X[:, _bass_idx] > 0) | (X[:, _phase0_idx] > 0)).astype(float)
        return np.stack([1 - pred, pred], axis=1)


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def write_report(results, imp) -> None:
    lines = ["# Chord-change classifier: architecture comparison (2026-07-04)\n"]
    lines.append(
        "Leave-one-song-out cross-validation (5 folds, one song held out each "
        "time) on `features.csv`. Features: beat phase (one-hot), beats since "
        "last change, bass pitch-class change/onset, onset density, chroma "
        "novelty, distance to segment boundary, loop-position fraction. "
        "Excludes anything that leaks the ground-truth chord identity "
        "(`gt_root`, `B_bass_is_root_or_fifth`, `C_bigram_*` -- see script "
        "docstring for why).\n"
    )
    lines.append("| Model | Accuracy | F1 | AUC | Brier | LogLoss |")
    lines.append("|---|---|---|---|---|---|")
    for name, per_song, pooled in results:
        ll = f"{pooled.get('logloss', float('nan')):.3f}" if "logloss" in pooled else "-"
        lines.append(f"| {name} | {pooled['accuracy']:.3f} | {pooled['f1']:.3f} | "
                      f"{pooled['auc']:.3f} | {pooled['brier']:.3f} | {ll} |")

    lines.append("\n## Per-song breakdown (best model)\n")
    best_name, best_per_song, _ = max(
        (r for r in results if "x bass" not in r[0] and "Majority" not in r[0] and "Heuristic" not in r[0]),
        key=lambda r: r[2]["auc"],
    )
    lines.append(f"Best model by pooled AUC: **{best_name}**\n")
    lines.append(_df_to_markdown(best_per_song))

    lines.append("\n## Feature importances (RF / GB, fit on all 5 songs, descriptive)\n")
    lines.append(_df_to_markdown(imp))

    (OUT_DIR / "ml_model_results.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'ml_model_results.md'}")


if __name__ == "__main__":
    main()
