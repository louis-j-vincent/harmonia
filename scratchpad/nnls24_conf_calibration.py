"""Fit + evaluate an isotonic confidence map for the live nnls24 path.

AUDIT (2026-07-19): the deployed `_infer_nnls24` path displays
`conf = p_seg[root]/p_seg.sum()` — an UNCALIBRATED root-mass share. None of
the #26 calibration machinery applies to it (that lives in the BP48 branch the
live path returns before reaching). This script measures how miscalibrated
that score is against RWC GT and fits the missing isotonic map.

Data (all cached on disk, no audio needed):
- data/cache/rwc/rwc_nnls24.npz — 13k GT-block pooled NNLS-24 rows + GT
  root/quality/song_id (the corpus behind every NNLS-24 number this week).
- harmonia/models/nnls24_heads.npz — the deployed root/quality heads.

Score analog: per-block softmax max of the root head (the live path's
mass-share over a segment, at oracle blocks). Target: the chord the user SEES
is right = root AND 7-family both correct (joint), plus root-only reported.

Honest caveats (logged in known_issues):
- oracle GT blocks, not predicted segments (segmentation noise not modeled);
- fitted on NNLS-head correctness; the live default labels with music-x-lab,
  which is MORE accurate (0.874 vs 0.802 root) — so this map UNDERSTATES
  confidence for musx-labeled segments: conservative, never over-confident.

Eval: song-grouped 5-fold OOF ECE (15 bins), raw vs calibrated.
Output: data/models/nnls24_conf_calibration.npz (piecewise-linear curve).

Run: .venv/bin/python scratchpad/nnls24_conf_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.models.nnls_features import NNLS24Heads  # noqa: E402

NPZ = REPO / "data" / "cache" / "rwc" / "rwc_nnls24.npz"
OUT_MODEL = REPO / "data" / "models" / "nnls24_conf_calibration.npz"
OUT_JSON = Path(__file__).with_suffix(".json")

QUAL_CANON = {"maj": "maj", "min": "min", "dom": "dom", "hdim": "hdim",
              "dim": "dim", "aug": "aug", "sus": "sus"}


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= 1.0)
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def main():
    d = np.load(NPZ, allow_pickle=True)
    print("npz fields:", d.files)
    X = d["nnls24"]
    gt_root = d["root"].astype(int)
    gt_qual = np.array([str(q) for q in d["quality"]])
    songs = np.array([str(s) for s in d["song_id"]])

    heads = NNLS24Heads()
    p = heads.root_proba(X)
    pred_root = p.argmax(1)
    conf_raw = p.max(1)
    q_idx = heads.quality_idx(X, pred_root)
    pred_qual = np.array([heads.qualities[i] for i in q_idx])

    ok_root = pred_root == gt_root
    ok_joint = ok_root & (pred_qual == gt_qual)
    print(f"n={len(X)}  root acc={ok_root.mean():.3f}  joint acc={ok_joint.mean():.3f}")
    print(f"RAW: mean conf={conf_raw.mean():.3f}  "
          f"ECE(root)={ece(conf_raw, ok_root):.3f}  "
          f"ECE(joint)={ece(conf_raw, ok_joint):.3f}")

    # song-grouped 5-fold OOF isotonic (target = joint)
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(0)
    uniq = rng.permutation(np.unique(songs))
    folds = np.array_split(uniq, 5)
    conf_cal = np.zeros_like(conf_raw)
    for f in folds:
        te = np.isin(songs, f)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(conf_raw[~te], ok_joint[~te].astype(float))
        conf_cal[te] = iso.predict(conf_raw[te])
    print(f"CAL (OOF): mean conf={conf_cal.mean():.3f}  "
          f"ECE(joint)={ece(conf_cal, ok_joint):.3f}  "
          f"ECE(root)={ece(conf_cal, ok_root):.3f}")

    # reliability table for the writeup
    edges = np.linspace(0, 1, 11)
    table = []
    for i in range(10):
        m = (conf_raw >= edges[i]) & (conf_raw < edges[i + 1] if i < 9 else conf_raw <= 1)
        if m.sum():
            table.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()),
                          "mean_conf": round(float(conf_raw[m].mean()), 3),
                          "acc_joint": round(float(ok_joint[m].mean()), 3),
                          "acc_root": round(float(ok_root[m].mean()), 3)})
    for r in table:
        print(r)

    # final map on ALL data, exported as a piecewise-linear curve for np.interp
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(conf_raw, ok_joint.astype(float))
    grid = np.linspace(0.0, 1.0, 201)
    curve = iso.predict(grid)
    np.savez(OUT_MODEL, x=grid.astype(np.float32), y=curve.astype(np.float32),
             score_kind="nnls_root_softmax_max_oracle_block",
             target="joint_root_and_family",
             fitted_on="rwc_nnls24 13k blocks, song-grouped final fit 2026-07-19",
             oof_ece_joint=ece(conf_cal, ok_joint),
             raw_ece_joint=ece(conf_raw, ok_joint))
    print(f"saved {OUT_MODEL}")

    OUT_JSON.write_text(json.dumps({
        "n": int(len(X)), "root_acc": float(ok_root.mean()),
        "joint_acc": float(ok_joint.mean()),
        "raw_mean_conf": float(conf_raw.mean()),
        "raw_ece_root": ece(conf_raw, ok_root),
        "raw_ece_joint": ece(conf_raw, ok_joint),
        "cal_oof_ece_joint": ece(conf_cal, ok_joint),
        "cal_oof_mean_conf": float(conf_cal.mean()),
        "reliability": table}, indent=1))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
