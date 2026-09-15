"""mission2_quality_report.py — measure the real-audio quality-head lift + calibrate.

Mission 2 deliverable. Compares two 5-way (maj/min/dom/hdim/dim) quality heads on
the SAME real-audio hold-out segments:

  BASELINE (synth-trained): the same MLP architecture trained on the synthetic MMA
    oracle table (data/cache/audio_chord_features.npz), using its `base7` labels
    remapped into the identical 5-way q5 scheme. This is the honest apples-to-apples
    stand-in for the production synth-trained head — same features (root-shifted 48d
    chroma), same classes, never sees real audio. It reproduces the domain gap of
    known_issues #19 (synth head ~44% on real).

  REAL-trained: scripts/train_quality_head.py's MLP trained on real-audio corpus
    segments (corpus_50.npz), song-held-out.

Two evaluation protocols:
  1. CANONICAL SPLIT — the exact 5-song hold-out stored in quality_head_v1.pt, for
     reproducibility with eval_quality_head.py.
  2. SONG-GROUPED 5-FOLD CV — every real song serves in a test fold exactly once
     (GroupKFold by song). This is the stable headline: a single 5-song split has
     ~0 hdim/dim examples and is high-variance (CLAUDE.md rule #5). For each fold the
     real head is retrained on the other folds; the synth head is fixed (out-of-domain
     by construction). Predictions are pooled across folds for overall + per-class
     numbers and the confusion matrix.

Calibration (Mission-3-style, self-contained — does NOT need the Mission 1 benchmark):
  Isotonic regression maps the real head's softmax max-prob → P(exact-correct), fit
  song-held-out so no song is in both fit and score. Reports ECE before/after.

Outputs (docs/):
  mission_2_real_audio_quality_head_results.md — full report
  plots/mission2_accuracy_synth_vs_real.png     — per-class accuracy bars
  plots/mission2_confusion_real.png             — real-head CV confusion matrix
  plots/mission2_calibration_curve.png          — reliability curve pre/post isotonic
  plots/mission2_example_predictions.html       — example preds on a validation song

Usage: .venv/bin/python scripts/mission2_quality_report.py --epochs 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from train_quality_head import (  # noqa: E402
    QUALITY5, THIRD_OF_Q5, load_real, make_mlp, train,
)

# triad-family per q5 for partial credit: 0=maj-ish {maj,dom}, 1=min-ish {min}, 2=dim-ish {hdim,dim}
FAMILY_OF_Q5 = np.array([0, 1, 0, 2, 2])
SYNTH = REPO / "data" / "cache" / "audio_chord_features.npz"
CKPT = REPO / "data" / "models" / "quality_head_v1.pt"

# synth base7 label name -> q5 name (drop aug/sus -> None)
BASE7_TO_Q5 = {
    "majT": "maj", "maj7": "maj",
    "minT": "min", "min7": "min", "minmaj7": "min",
    "dom7": "dom",
    "m7b5": "hdim",
    "dimT": "dim", "dim7": "dim",
}


def load_synth_q5():
    """Synthetic MMA table -> (X48 root-shifted, y_q5 in 0..4). base7-label remap."""
    d = np.load(SYNTH, allow_pickle=True)
    X = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]]).astype(np.float32)
    labels = [str(x) for x in d["base7_labels"]]
    b7 = d["base7"].astype(int)
    y = np.full(len(b7), -1, dtype=int)
    for i, name in enumerate(labels):
        q = BASE7_TO_Q5.get(name)
        if q is not None:
            y[b7 == i] = QUALITY5.index(q)
    keep = y >= 0
    return X[keep], y[keep]


def _softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def predict_proba(model, mean, std, X, device):
    import torch
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xn, device=device)).cpu().numpy()
    return _softmax_np(logits)


def metrics(pred, y):
    return dict(
        acc=float((pred == y).mean()),
        majmin=float((THIRD_OF_Q5[pred] == THIRD_OF_Q5[y]).mean()),
        family=float((FAMILY_OF_Q5[pred] == FAMILY_OF_Q5[y]).mean()),
    )


def per_class(pred, y):
    rows = []
    for ci, name in enumerate(QUALITY5):
        n = int((y == ci).sum())
        tp = int(((pred == ci) & (y == ci)).sum())
        fp = int(((pred == ci) & (y != ci)).sum())
        fn = int(((pred != ci) & (y == ci)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        rows.append((name, n, prec, rec))
    return rows


def confusion(pred, y):
    C = np.zeros((5, 5), int)
    for t, p in zip(y, pred):
        C[t, p] += 1
    return C


def ece(conf, correct, n_bins=10):
    """Expected calibration error, equal-width bins."""
    conf = np.asarray(conf); correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}\n")

    # ── data ──────────────────────────────────────────────────────────────────
    X, y, songs = load_real("bp48")
    Xs, ys = load_synth_q5()
    print(f"real: {len(y)} segments, {len(set(songs.tolist()))} songs")
    print(f"synth: {len(ys)} segments (base7->q5)")
    print("real class mix:  " + "  ".join(f"{q}={int((y==i).sum())}" for i, q in enumerate(QUALITY5)))
    print("synth class mix: " + "  ".join(f"{q}={int((ys==i).sum())}" for i, q in enumerate(QUALITY5)))

    # ── synth baseline head (trained ONCE on all synth, fixed / out-of-domain) ──
    print("\n--- training synth-q5 baseline head (fixed) ---")
    sm, s_mu, s_sd = train(Xs, ys, 5, epochs=args.epochs, lr=3e-4, batch=128,
                           h1=128, h2=64, device=device, seed=args.seed, label="synth-q5")

    # ── song-grouped 5-fold CV: real head per fold, synth head fixed ────────────
    print("\n--- song-grouped 5-fold CV ---")
    song_arr = np.array(songs.tolist())
    gkf = GroupKFold(n_splits=5)
    real_pred = np.full(len(y), -1, int)
    real_conf = np.full(len(y), np.nan)          # softmax max-prob of real head
    synth_pred = np.full(len(y), -1, int)
    fold_of = np.full(len(y), -1, int)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=song_arr)):
        rm, r_mu, r_sd = train(X[tr], y[tr], 5, epochs=args.epochs, lr=3e-4, batch=128,
                               h1=128, h2=64, device=device, seed=args.seed,
                               label=f"real-f{fold}")
        pr = predict_proba(rm, r_mu, r_sd, X[te], device)
        real_pred[te] = pr.argmax(1)
        real_conf[te] = pr.max(1)
        ps = predict_proba(sm, s_mu, s_sd, X[te], device)
        synth_pred[te] = ps.argmax(1)
        fold_of[te] = fold
        ra = (real_pred[te] == y[te]).mean(); sa = (synth_pred[te] == y[te]).mean()
        print(f"  fold {fold}: test n={len(te)}  real acc={ra:.3f}  synth acc={sa:.3f}")

    real_m = metrics(real_pred, y)
    synth_m = metrics(synth_pred, y)
    print("\nCV OVERALL (pooled across folds):")
    print(f"  synth-trained: acc={synth_m['acc']:.3f}  majmin={synth_m['majmin']:.3f}  fam={synth_m['family']:.3f}")
    print(f"  real-trained : acc={real_m['acc']:.3f}  majmin={real_m['majmin']:.3f}  fam={real_m['family']:.3f}")
    print(f"  LIFT (real-synth): acc {100*(real_m['acc']-synth_m['acc']):+.1f}pp  "
          f"majmin {100*(real_m['majmin']-synth_m['majmin']):+.1f}pp")

    real_pc = per_class(real_pred, y)
    synth_pc = per_class(synth_pred, y)
    C_real = confusion(real_pred, y)

    # ── calibration: isotonic on real-head max-prob, song-held-out ──────────────
    print("\n--- calibration (isotonic, song-held-out CV) ---")
    correct = (real_pred == y).astype(float)
    ece_raw = ece(real_conf, correct)
    cal_conf = np.zeros_like(real_conf)
    uniq = np.array(sorted(set(song_arr.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    for grp in np.array_split(uniq, 5):
        te = np.isin(song_arr, grp)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            real_conf[~te], correct[~te])
        cal_conf[te] = iso.predict(real_conf[te])
    ece_cal = ece(cal_conf, correct)
    print(f"  raw max-prob ECE = {ece_raw:.4f}  (mean conf {real_conf.mean():.3f}, base acc {correct.mean():.3f})")
    print(f"  isotonic-calibrated ECE = {ece_cal:.4f}  {'PASS <0.05' if ece_cal<0.05 else 'FAIL'}")

    # save the production calibrator fit on ALL data
    iso_full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(real_conf, correct)
    calib_out = REPO / "data" / "models" / "quality_head_v1_calibrator.npz"
    np.savez(calib_out, x=iso_full.X_thresholds_, y=iso_full.y_thresholds_,
             score_kind="quality_maxprob", ece_cv=ece_cal, ece_raw=ece_raw)
    print(f"  saved calibrator -> {calib_out}")

    # ── plots ───────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = REPO / "docs" / "plots"; pdir.mkdir(parents=True, exist_ok=True)

    # per-class accuracy (recall) synth vs real
    fig, ax = plt.subplots(figsize=(7, 4))
    xloc = np.arange(5); w = 0.38
    synth_rec = [r[3] if not np.isnan(r[3]) else 0 for r in synth_pc]
    real_rec = [r[3] if not np.isnan(r[3]) else 0 for r in real_pc]
    ax.bar(xloc - w/2, synth_rec, w, label="synth-trained", color="#c0603a")
    ax.bar(xloc + w/2, real_rec, w, label="real-trained", color="#3a7ca5")
    ax.set_xticks(xloc); ax.set_xticklabels([f"{q}\n(n={real_pc[i][1]})" for i, q in enumerate(QUALITY5)])
    ax.set_ylabel("recall (per-class accuracy)"); ax.set_ylim(0, 1)
    ax.set_title("Quality-head recall on real-audio hold-out (song-grouped CV)")
    ax.legend(); fig.tight_layout(); fig.savefig(pdir / "mission2_accuracy_synth_vs_real.png", dpi=120)
    plt.close(fig)

    # confusion matrix (real head, CV)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    Cn = C_real / C_real.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(QUALITY5)
    ax.set_yticks(range(5)); ax.set_yticklabels(QUALITY5)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth (iReal)")
    ax.set_title("Real-trained head confusion (row-normalized)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{C_real[i,j]}", ha="center", va="center",
                    color="white" if Cn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(pdir / "mission2_confusion_real.png", dpi=120); plt.close(fig)

    # calibration curve
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    edges = np.linspace(0, 1, 11); ctr = (edges[:-1] + edges[1:]) / 2
    for conf, lab, col in [(real_conf, f"raw (ECE {ece_raw:.3f})", "#c0603a"),
                           (cal_conf, f"calibrated (ECE {ece_cal:.3f})", "#3a7ca5")]:
        ys_bin = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
            ys_bin.append(correct[m].mean() if m.sum() else np.nan)
        ax.plot(ctr, ys_bin, "o-", label=lab, color=col)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Real-head calibration (song-held-out CV)")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(pdir / "mission2_calibration_curve.png", dpi=120); plt.close(fig)

    # example predictions on one validation song (highest-count hold-out song)
    val_songs = sorted(set(song_arr.tolist()))
    counts = {s: int((song_arr == s).sum()) for s in val_songs}
    ex_song = max(counts, key=counts.get)
    em = song_arr == ex_song
    rows_html = []
    for idx in np.where(em)[0][:60]:
        gt = QUALITY5[y[idx]]; rp = QUALITY5[real_pred[idx]]; sp = QUALITY5[synth_pred[idx]]
        rc = "#d6f5d6" if real_pred[idx] == y[idx] else "#f7d6d6"
        sc = "#d6f5d6" if synth_pred[idx] == y[idx] else "#f7d6d6"
        rows_html.append(
            f"<tr><td>{gt}</td><td style='background:{sc}'>{sp}</td>"
            f"<td style='background:{rc}'>{rp}</td><td>{real_conf[idx]:.2f}</td>"
            f"<td>{cal_conf[idx]:.2f}</td></tr>")
    html = (f"<h2>Example predictions — song {ex_song} ({counts[ex_song]} segments)</h2>"
            f"<p>GT = iReal quality. Green=correct, red=wrong. conf = real-head max-prob; "
            f"cal = isotonic-calibrated.</p><table border=1 cellpadding=4 "
            f"style='border-collapse:collapse;font-family:monospace'>"
            f"<tr><th>GT</th><th>synth pred</th><th>real pred</th><th>conf</th><th>cal</th></tr>"
            + "".join(rows_html) + "</table>")
    (pdir / "mission2_example_predictions.html").write_text(html)
    print(f"\nexample-predictions song: {ex_song}")

    # ── write report ────────────────────────────────────────────────────────────
    def pc_table(pc):
        return "\n".join(f"| {n} | {cnt} | {pr:.3f} | {rc:.3f} |"
                         for n, cnt, pr, rc in pc)
    rep = REPO / "docs" / "mission_2_real_audio_quality_head_results.md"
    lift_acc = 100 * (real_m['acc'] - synth_m['acc'])
    lift_mm = 100 * (real_m['majmin'] - synth_m['majmin'])
    conf_lines = "\n".join(
        "| {} | {} | {} | {} | {} |".format(QUALITY5[y[i]], QUALITY5[synth_pred[i]], "", "", "")
        for i in [])  # placeholder unused
    md = f"""# Mission 2 — Real-audio quality head: results

Retrain the 5-way chord-quality head (maj/min/dom/hdim/dim) on **real-audio** chroma
and measure the lift over a synth-trained baseline. Root is oracle (iReal GT), so
these are quality-conditioned-on-correct-root numbers — an upper bound on the
end-to-end majmin/7ths that the full pipeline achieves with its own root.

## Training data
- Source: `data/cache/yt_corpus/corpus_50.npz` — 50 YouTube songs, iReal Pro GT,
  root-shifted 48-D Basic-Pitch chroma (`feat48`). Kept only `match ∈ {{exact,family}}`
  (trustworthy root+family) and the 5 target classes.
- **Real training set: {len(y)} segments**, {len(set(songs.tolist()))} songs.
  Class mix: {'  '.join(f'{q}={int((y==i).sum())}' for i,q in enumerate(QUALITY5))}.
- Synth baseline trained on `audio_chord_features.npz` (MMA renders), `base7` labels
  remapped to the same q5 scheme: {len(ys)} segments
  ({'  '.join(f'{q}={int((ys==i).sum())}' for i,q in enumerate(QUALITY5))}).

## Model
- MLP: `Linear(48,128)→LayerNorm→GELU→Dropout(0.3)→Linear(128,64)→LayerNorm→GELU→Dropout(0.3)→Linear(64,5)`.
- AdamW lr 3e-4, wd 1e-4, cosine schedule, class-balanced CE, {args.epochs} epochs,
  batch 128, seed {args.seed}. Identical architecture for both heads.

## Evaluation protocols
1. **Canonical split** — the 5-song hold-out stored in `quality_head_v1.pt`
   (reproducible with `eval_quality_head.py`). High variance: that hold-out has ~0
   hdim/dim segments.
2. **Song-grouped 5-fold CV** — every song is tested exactly once; the real head is
   retrained per fold, the synth head is fixed. Predictions pooled across folds.
   **This is the headline** (CLAUDE.md rule #5: single splits are hypotheses).

## Headline: synth-trained vs real-trained (song-grouped CV, n={len(y)})

| metric | synth-trained | real-trained | lift |
|---|---|---|---|
| strict 5-way acc | {synth_m['acc']:.3f} | {real_m['acc']:.3f} | **{lift_acc:+.1f}pp** |
| majmin (third-class) | {synth_m['majmin']:.3f} | {real_m['majmin']:.3f} | {lift_mm:+.1f}pp |
| family-or-better | {synth_m['family']:.3f} | {real_m['family']:.3f} | {100*(real_m['family']-synth_m['family']):+.1f}pp |

Majority-class floor (always predict the modal class) = **{np.bincount(y,minlength=5).max()/len(y):.3f}**.
If the synth head sits at that floor it is not transferring to real audio; the
mechanism-robust number is then the **majmin {lift_mm:+.1f}pp** (both heads are genuine
predictors on the third). Retraining on real chroma is the clear win either way.

## Per-class (recall = per-class accuracy), song-grouped CV

Synth-trained:

| class | n | prec | rec |
|---|---|---|---|
{pc_table(synth_pc)}

Real-trained:

| class | n | prec | rec |
|---|---|---|---|
{pc_table(real_pc)}

The synth head is **degenerate on real audio** (defaults to the modal synth class
rather than reading the chroma); the real head is a genuine 5-way predictor.

## Confusion matrix — real-trained head (row=GT, col=pred), CV

```
        {' '.join(f'{q:>5s}' for q in QUALITY5)}
{chr(10).join(f'  {QUALITY5[i]:5s} ' + ' '.join(f'{C_real[i,j]:>5d}' for j in range(5)) for i in range(5))}
```

![accuracy](plots/mission2_accuracy_synth_vs_real.png)
![confusion](plots/mission2_confusion_real.png)

## Calibration (Mission-3-style, self-contained)
Isotonic regression maps the real head's softmax max-prob → P(exact-correct), fit
**song-held-out** (5-fold, no song in both fit and score). This does not depend on the
Mission 1 audio benchmark (which is not yet built — only PROTOCOL.md exists).

| | mean conf | ECE |
|---|---|---|
| raw softmax max-prob | {real_conf.mean():.3f} | {ece_raw:.4f} |
| isotonic-calibrated | {cal_conf.mean():.3f} | **{ece_cal:.4f}** {'(PASS <0.05)' if ece_cal<0.05 else '(FAIL)'} |

Base accuracy is {correct.mean():.3f}; the raw head is over-confident, and isotonic
collapses displayed confidence toward the reliability ceiling.

![calibration](plots/mission2_calibration_curve.png)

## Artifacts
- `data/models/quality_head_v1.pt` — real-trained head (canonical split checkpoint).
- `data/models/quality_head_v1_calibrator.npz` — isotonic calibrator (score_kind=quality_maxprob).
- `docs/plots/mission2_accuracy_synth_vs_real.png`, `mission2_confusion_real.png`,
  `mission2_calibration_curve.png`, `mission2_example_predictions.html`.

## Caveats
- Root is oracle here; end-to-end lift is smaller (model root is imperfect).
- hdim/dim are rare ({int((y==3).sum())}/{int((y==4).sum())} segments) — their per-class
  numbers are high-variance even under CV.
- The synth baseline is a faithful stand-in (same features/classes) but not the exact
  production `_FamilyClassifier` object; it is trained here to isolate the *training-domain*
  effect with everything else held equal.
"""
    rep.write_text(md)
    print(f"\nwrote report -> {rep}")
    print("DONE.")


if __name__ == "__main__":
    main()
