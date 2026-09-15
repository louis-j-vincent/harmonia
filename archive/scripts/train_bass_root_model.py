#!/usr/bin/env python3
"""Agent 1 — Bass/Root detection model.

Trains a 12-class root detector on McGill Billboard using:
  - 24-dim bothchroma (bass + treble register, NOT collapsed to 12)
  - context: prev/next functional root (oracle) as one-hot
  - a learned harmonic transition prior P(root | prev_root) as soft features

Two models are trained and compared:
  * chroma-only  (deployable today; no oracle context)
  * +context     (oracle prev/next root; mission spec)

Song-stratified 80/10/10 split (no leakage). Class-weighted CE.
Reports per-note recall + confusion matrix. Saves model, predictions cache,
plots, and a markdown report.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, precision_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FEAT = REPO / "data" / "cache" / "bass_root_features.npz"
MODELS = REPO / "data" / "models"
CACHE = REPO / "data" / "cache"
PLOTS = REPO / "docs" / "plots"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALS = ["maj", "min", "dom", "hdim", "dim"]
torch.manual_seed(0); np.random.seed(0)


def l2norm(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def build_features(d, use_context, trans_prior=None):
    """Return X for the chosen model variant."""
    chroma = d["feats"].astype(np.float32)
    # normalize bass and treble halves independently (kills loudness scale)
    chroma = np.concatenate([l2norm(chroma[:, :12]), l2norm(chroma[:, 12:])], axis=1)
    if not use_context:
        return chroma
    n = len(chroma)
    pr, nr = d["prev_root"], d["next_root"]
    pr_oh = np.zeros((n, 13), np.float32); nr_oh = np.zeros((n, 13), np.float32)
    pr_oh[np.arange(n), np.where(pr < 0, 12, pr)] = 1.0
    nr_oh[np.arange(n), np.where(nr < 0, 12, nr)] = 1.0
    parts = [chroma, pr_oh, nr_oh]
    if trans_prior is not None:
        # P(root | prev_root) as a 12-dim soft feature (uniform if prev unknown)
        tp = np.where(pr[:, None] < 0, 1.0 / 12,
                      trans_prior[np.clip(pr, 0, 11)])
        parts.append(tp.astype(np.float32))
    return np.concatenate(parts, axis=1)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(64, 32)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2)]
            d = h
        layers += [nn.Linear(d, 12)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one(Xtr, ytr, Xva, yva, cw, seed=0, epochs=120, lr=2e-3):
    """Train one MLP; select the epoch maximizing (macro-recall + min-recall)
    on val (rewards both average quality and the weakest-note floor)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=512, shuffle=True)
    Xva_t = torch.tensor(Xva); best_sc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va_pred = model(Xva_t).argmax(1).numpy()
        macro = recall_score(yva, va_pred, average="macro", labels=range(12), zero_division=0)
        floor = recall_score(yva, va_pred, average=None, labels=range(12), zero_division=0).min()
        sc = macro + floor
        if sc > best_sc:
            best_sc = sc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_ensemble(Xtr, ytr, Xva, yva, cw, n_seeds=5):
    return [train_one(Xtr, ytr, Xva, yva, cw, seed=s) for s in range(n_seeds)]


def ens_probs(models, X):
    Xt = torch.tensor(X)
    ps = []
    for m in models:
        m.eval()
        with torch.no_grad():
            ps.append(torch.softmax(m(Xt), 1).numpy())
    return np.mean(ps, 0)


def evaluate(models, X, y=None):
    return ens_probs(models, X).argmax(1)


def main():
    d = np.load(FEAT, allow_pickle=True)
    y = d["roots"].astype(np.int64); sid = d["song_id"]; quals = d["quals"]
    prev_root = d["prev_root"]; next_root = d["next_root"]

    # song-stratified 80/10/10
    songs = np.array(sorted(set(sid))); rng = np.random.RandomState(42); rng.shuffle(songs)
    n = len(songs); ntr, nva = int(.8 * n), int(.1 * n)
    tr_s, va_s, te_s = set(songs[:ntr]), set(songs[ntr:ntr + nva]), set(songs[ntr + nva:])
    trm = np.array([s in tr_s for s in sid]); vam = np.array([s in va_s for s in sid])
    tem = np.array([s in te_s for s in sid])
    print(f"split: train={trm.sum()} val={vam.sum()} test={tem.sum()} chords; "
          f"songs {len(tr_s)}/{len(va_s)}/{len(te_s)}")

    # harmonic transition prior P(root | prev_root) from TRAIN only
    trans = np.ones((12, 12), np.float64)  # laplace
    for r, pr in zip(y[trm], prev_root[trm]):
        if pr >= 0:
            trans[pr, r] += 1
    trans /= trans.sum(1, keepdims=True)

    # class weights (inverse freq, from train)
    counts = np.bincount(y[trm], minlength=12).astype(np.float64)
    cw = (counts.sum() / (12 * np.clip(counts, 1, None)))
    cw = cw / cw.mean()

    results = {}
    preds_cache = {}
    for name, use_ctx in [("chroma_only", False), ("context", True)]:
        X = build_features(d, use_ctx, trans_prior=trans if use_ctx else None)
        n_seeds = 5 if use_ctx else 3
        model = train_ensemble(X[trm], y[trm], X[vam], y[vam], cw, n_seeds=n_seeds)
        te_pred = evaluate(model, X[tem])
        rec = recall_score(y[tem], te_pred, average=None, labels=range(12), zero_division=0)
        prec = precision_score(y[tem], te_pred, average=None, labels=range(12), zero_division=0)
        acc = accuracy_score(y[tem], te_pred)
        results[name] = dict(acc=float(acc), recall=rec.tolist(), precision=prec.tolist(),
                             min_recall=float(rec.min()), mean_recall=float(rec.mean()),
                             pass85=bool(rec.min() >= 0.85))
        print(f"[{name}] test acc={acc:.3f} min-recall={rec.min():.3f} "
              f"mean-recall={rec.mean():.3f} pass(>=.85 all)={rec.min()>=0.85}")
        # cache predictions + probs for ALL splits (Agent 2 needs this)
        for split, m in [("train", trm), ("val", vam), ("test", tem)]:
            probs = ens_probs(model, X[m])
            preds_cache[f"{name}_{split}_pred"] = probs.argmax(1)
            preds_cache[f"{name}_{split}_prob"] = probs.astype(np.float32)
            preds_cache[f"{name}_{split}_true"] = y[m]
            preds_cache[f"{name}_{split}_song"] = sid[m]
        if name == "context":
            best_model, best_X = model, X
            best_test_pred = te_pred

    # ---- save model + config (all ensemble seeds) ----
    MODELS.mkdir(exist_ok=True)
    torch.save({f"seed{i}": m.state_dict() for i, m in enumerate(best_model)},
               MODELS / "bass_detector_v1.pt")
    cfg = dict(feat_dim=int(best_X.shape[1]), hidden=[64, 32], classes=NOTE_NAMES,
               qualities=QUALS, uses_context=True, split="song_80_10_10_seed42",
               trans_prior=trans.tolist(), class_weights=cw.tolist(),
               feature_layout="24 chroma(bass12 L2 | treble12 L2) + prev_root_oh13 + "
                              "next_root_oh13 + trans_prior12",
               results=results)
    (MODELS / "bass_detector_v1.json").write_text(json.dumps(cfg, indent=2))

    # ---- predictions cache ----
    np.savez_compressed(CACHE / "bass_predictions_train_val_test.npz", **preds_cache)

    # ---- confusion matrix + per-note recall plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y[tem], best_test_pred, labels=range(12))
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    im = ax[0].imshow(cmn, cmap="magma", vmin=0, vmax=1)
    ax[0].set(xticks=range(12), yticks=range(12), xticklabels=NOTE_NAMES,
              yticklabels=NOTE_NAMES, xlabel="Predicted root", ylabel="True root",
              title="Row-normalized confusion (context model, test)")
    for i in range(12):
        for j in range(12):
            if cmn[i, j] >= 0.03:
                ax[0].text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                           color="w" if cmn[i, j] < 0.6 else "k", fontsize=7)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    rec_c = np.array(results["context"]["recall"])
    rec_b = np.array(results["chroma_only"]["recall"])
    x = np.arange(12)
    ax[1].bar(x - 0.2, rec_b, 0.4, label="chroma-only", color="#6b8fb5")
    ax[1].bar(x + 0.2, rec_c, 0.4, label="+context", color="#d98a3d")
    ax[1].axhline(0.85, ls="--", c="k", lw=1, label="target 0.85")
    ax[1].set(xticks=x, xticklabels=NOTE_NAMES, ylim=(0, 1.02),
              ylabel="per-note recall", title="Per-note recall (test)")
    ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True, parents=True)
    fig.savefig(PLOTS / "bass_confusion_matrix.png", dpi=130)
    print("saved plot", PLOTS / "bass_confusion_matrix.png")

    # error breakdown by chord quality (context model)
    qual_break = {}
    for qi, qn in enumerate(QUALS):
        m = (quals[tem] == qi)
        if m.sum() == 0:
            continue
        qual_break[qn] = dict(n=int(m.sum()),
                              acc=float(accuracy_score(y[tem][m], best_test_pred[m])))
    results["quality_breakdown"] = qual_break
    (MODELS / "bass_detector_v1.json").write_text(json.dumps(cfg, indent=2))
    with open(REPO / "docs" / "_bass_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("quality breakdown:", qual_break)


if __name__ == "__main__":
    main()
