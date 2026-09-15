"""
Same architecture comparison as train_chord_change_classifier.py, but on
the full 909-song symbolic feature table (features_symbolic.csv, ~304K
rows) instead of the 5-song audio table -- this is the "test the ML
models on all the songs" follow-up. Reuses the exact same design-matrix
construction (imported directly, not duplicated) so the two runs are
comparable.

With 909 songs, leave-one-song-out is impractical (909 folds) and
unnecessary (the whole point of n=5 -> n=909 was to get past small-sample
variance) -- uses GroupKFold instead (10 folds, grouped by song_id so no
song's beats appear in both train and test).

Adds one architecture beyond the 5-song run: a small PyTorch MLP trained
on the Mac's M4 GPU (`mps` backend) rather than sklearn's CPU-only
MLPClassifier -- at this scale (~270K training rows/fold) GPU training is
both faster and a more legitimate "different architecture" comparison
point than sklearn's solver.

Usage:
    .venv/bin/python scripts/train_chord_change_classifier_full.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train_chord_change_classifier import DummyHeuristic, DummyMajority, _df_to_markdown, build_design_matrix  # noqa: E402

OUT_DIR = Path(__file__).parent.parent / "docs" / "chord_change_signal_analysis"
N_FOLDS = 10


def torch_mlp_factory(input_dim: int, device: str):
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return MLP(input_dim).to(device)


class TorchMLPClassifier:
    """Thin sklearn-compatible wrapper so it drops into the same
    fit/predict_proba evaluation loop as every other model here."""

    def __init__(self, input_dim: int, device: str, epochs: int = 40, lr: float = 1e-3, batch_size: int = 4096):
        self.input_dim = input_dim
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        self.model = torch_mlp_factory(self.input_dim, self.device)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        n = len(Xt)
        for _epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                logits = self.model(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch

        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            prob = torch.sigmoid(self.model(Xt)).cpu().numpy()
        return np.stack([1 - prob, prob], axis=1)


def group_kfold_evaluate(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, model_fn, use_interactions: bool, n_folds: int = N_FOLDS):
    interaction_cols = [c for c in X.columns if "_x_bass_changed" in c]
    cols = X.columns if use_interactions else [c for c in X.columns if c not in interaction_cols]
    groups = df["song_id"].values
    gkf = GroupKFold(n_splits=n_folds)

    all_true, all_prob = [], []
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        Xtr, Xte = X.iloc[train_idx][cols].values, X.iloc[test_idx][cols].values
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model = model_fn(Xtr_s.shape[1]) if callable(model_fn) and model_fn.__code__.co_argcount == 1 else model_fn()
        model.fit(Xtr_s, ytr)
        prob = model.predict_proba(Xte_s)[:, 1]
        pred = (prob >= 0.5).astype(int)

        fold_rows.append({
            "fold": fold, "n": len(yte),
            "accuracy": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, zero_division=0),
            "auc": roc_auc_score(yte, prob) if len(set(yte)) > 1 else np.nan,
        })
        all_true.extend(yte)
        all_prob.extend(prob)

    all_true, all_prob = np.array(all_true), np.array(all_prob)
    all_pred = (all_prob >= 0.5).astype(int)
    prob_clipped = np.clip(all_prob, 1e-3, 1 - 1e-3)
    pooled = {
        "accuracy": accuracy_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
        "auc": roc_auc_score(all_true, all_prob),
        "brier": brier_score_loss(all_true, all_prob),
        "logloss": log_loss(all_true, prob_clipped),
    }
    return pd.DataFrame(fold_rows), pooled


def main() -> None:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Torch device: {device}")

    df = pd.read_csv(OUT_DIR / "features_symbolic.csv", dtype={"song_id": str})
    X, y = build_design_matrix(df)
    n_features_no_int = len([c for c in X.columns if "_x_bass_changed" not in c])
    print(f"Rows: {len(df)}, songs: {df['song_id'].nunique()}, positive rate: {y.mean():.1%}\n")

    results = []
    print(f"=== Baselines ({N_FOLDS}-fold GroupKFold by song) ===")
    for name, fn in [("Majority class", lambda: DummyMajority()),
                      ("Heuristic (phase==0 or bass_changed)", lambda: DummyHeuristic())]:
        t0 = time.time()
        per_fold, pooled = group_kfold_evaluate(df, X, y, fn, use_interactions=False)
        print(f"{name}: acc={pooled['accuracy']:.3f} f1={pooled['f1']:.3f} auc={pooled['auc']:.3f} ({time.time()-t0:.0f}s)")
        results.append((name, per_fold, pooled))

    print(f"\n=== Models ({N_FOLDS}-fold GroupKFold by song) ===")
    models = {
        "Logistic (main effects)": (lambda: LogisticRegression(max_iter=2000), False),
        "Logistic (+ phase x bass interactions)": (lambda: LogisticRegression(max_iter=2000), True),
        "Decision Tree (depth=6)": (lambda: DecisionTreeClassifier(max_depth=6, random_state=0), False),
        "Random Forest (200 trees)": (lambda: RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=0), False),
        "Gradient Boosting": (lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0), False),
        "PyTorch MLP (32,16) on M4 GPU": (lambda d: TorchMLPClassifier(d, device=device), False),
    }
    for name, (fn, use_int) in models.items():
        t0 = time.time()
        per_fold, pooled = group_kfold_evaluate(df, X, y, fn, use_interactions=use_int)
        print(f"{name}: acc={pooled['accuracy']:.3f} f1={pooled['f1']:.3f} auc={pooled['auc']:.3f} "
              f"brier={pooled['brier']:.3f} logloss={pooled['logloss']:.3f} ({time.time()-t0:.0f}s)")
        results.append((name, per_fold, pooled))

    print("\n=== Feature importances (fit on all songs, descriptive only) ===")
    cols = [c for c in X.columns if "_x_bass_changed" not in c]
    scaler = StandardScaler().fit(X[cols].values)
    Xs = scaler.transform(X[cols].values)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=0).fit(Xs, y)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(Xs, y)
    imp = pd.DataFrame({"feature": cols, "rf_importance": rf.feature_importances_,
                         "gb_importance": gb.feature_importances_}).sort_values("rf_importance", ascending=False)
    print(imp.to_string(index=False))

    write_report(results, imp, len(df), df["song_id"].nunique())


def write_report(results, imp, n_rows, n_songs) -> None:
    lines = [f"# Chord-change classifier: full 909-song corpus (2026-07-04)\n"]
    lines.append(
        f"{N_FOLDS}-fold GroupKFold cross-validation (grouped by song, so no song's "
        f"beats leak across train/test) on `features_symbolic.csv` -- {n_rows} rows, "
        f"{n_songs} songs, fully symbolic (no audio, no Basic Pitch; bass from POP909's "
        "own PIANO-track ground truth, chroma/onset from a MIDI note-onset piano-roll). "
        "Same feature set and leakage exclusions as the 5-song audio run "
        "(`ml_model_results.md`) -- see that file's header for what's excluded and why.\n"
    )
    lines.append("| Model | Accuracy | F1 | AUC | Brier | LogLoss |")
    lines.append("|---|---|---|---|---|---|")
    for name, per_fold, pooled in results:
        ll = f"{pooled.get('logloss', float('nan')):.3f}" if "logloss" in pooled else "-"
        lines.append(f"| {name} | {pooled['accuracy']:.3f} | {pooled['f1']:.3f} | "
                      f"{pooled['auc']:.3f} | {pooled['brier']:.3f} | {ll} |")

    lines.append("\n## Feature importances (RF / GB, fit on all songs, descriptive)\n")
    lines.append(_df_to_markdown(imp))

    (OUT_DIR / "ml_model_results_full_corpus.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'ml_model_results_full_corpus.md'}")


if __name__ == "__main__":
    main()
