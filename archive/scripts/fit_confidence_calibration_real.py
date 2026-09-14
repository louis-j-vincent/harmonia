"""Fit the REAL-AUDIO confidence calibration map (Mission 4, issue #19/#26).

Motivation
----------
The synth calibrator (data/cache/confidence_calibration.npz) is fitted on MMA
renders. On real recordings it is badly miscalibrated: the quality head's
displayed confidence has mean ~0.90 while real-audio q5 accuracy is ~0.44
(ECE 0.465), and passing it through the synth map *amplifies* overconfidence
(ECE 0.533). Real-audio confidence is near non-discriminative — even conf≈0.98
predictions are only ~48% correct. So k-selection (fix the k lowest-confidence
chords) is near-random on real audio, and constraint propagation trusts a
badly-scaled number.

What this fits
--------------
An isotonic map on the yt_corpus_50 real-audio segments (iReal GT, DTW-aligned
by build_yt_corpus). Score = the QUALITY head's family max-prob (the production
`confidence_raw`); target = q5 (maj/min/dom/hdim/dim) family correct. The map is
loaded by chord_pipeline_v1._get_conf_calibrator("real") and applied when
infer_chords_v1(..., audio_domain="real") (the server default).

Honest caveats (documented in known_issues.md #19/#26)
------------------------------------------------------
- Proxy score: fit on the baseline LR _FamilyClassifier applied to the cached
  48-dim segment feature, NOT the production ctx/joint confidence_raw. It is
  robust to that distribution mismatch only because the fitted map is nearly
  flat (collapses to the base rate ~0.44 regardless of input).
- root_conf is not folded in (corpus stores only the quality feature). The full
  fused real ECE is likely a bit lower than 0.465 (root_conf < 1 shrinks the
  raw), but the flat-reliability finding is robust.
- DTW alignment noise inflates "incorrect" (some GT is misplaced), so 0.44 is a
  lower bound on true real-audio accuracy.

Usage:
    .venv/bin/python scripts/fit_confidence_calibration_real.py            # fit + save + CV report
    .venv/bin/python scripts/fit_confidence_calibration_real.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import chord_pipeline_v1 as P  # noqa: E402

CORPUS = REPO / "data" / "cache" / "yt_corpus" / "corpus_50.npz"
OUT = REPO / "data" / "cache" / "confidence_calibration_real.npz"


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < n_bins - 1 else (conf <= 1.0))
        if m.sum() >= 20:
            e += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from sklearn.isotonic import IsotonicRegression

    clf = P._FamilyClassifier()
    q5 = list(P._Q5_NAMES)
    d = np.load(CORPUS, allow_pickle=True)
    keep = np.isin(d["quality"], q5)  # drop sus/aug (not in q5)
    X = clf.sc.transform(d["feat48"][keep])
    gt = d["quality"][keep]
    sid = d["song_id"][keep]
    pf = clf.clf.predict_proba(X)
    p7 = clf.b7_clf.predict_proba(X)
    b7lab = [clf.base7_labels[int(c)] for c in clf.b7_clf.classes_]
    raw = pf.max(1).astype(float)  # displayed confidence_raw = family max-prob
    pred = np.array([q5[int(np.exp(P._family_q5_logprobs(pf[i], p7[i], b7lab)).argmax())]
                     for i in range(len(gt))])
    correct = (pred == gt).astype(float)
    print(f"n real q5 chords = {len(gt)}   q5 acc = {correct.mean():.3f}   "
          f"mean raw conf = {raw.mean():.3f}   overconf = {raw.mean() - correct.mean():+.3f}")

    # 5-fold song-held-out CV for an honest ECE
    songs = np.array(sorted(set(sid.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(songs)
    cal_cv = np.zeros_like(raw)
    for fold in np.array_split(songs, 5):
        te = np.isin(sid, fold)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw[~te], correct[~te])
        cal_cv[te] = iso.predict(raw[te])
    ece_raw, ece_cv = ece(raw, correct), ece(cal_cv, correct)
    print(f"raw ECE = {ece_raw:.4f}   |   5-fold song-held-out CV isotonic ECE = {ece_cv:.4f}")
    print(f"calibrated mean = {cal_cv.mean():.3f}  (≈ base rate {correct.mean():.3f})")

    # Also show what the synth map does to the real raw (should be worse)
    if P.CONF_CALIBRATION_PATH.exists():
        s = np.load(P.CONF_CALIBRATION_PATH)
        synth_on_real = np.interp(raw, s["x"].astype(float), s["y"].astype(float))
        print(f"synth map applied to real raw: mean = {synth_on_real.mean():.3f}  "
              f"ECE = {ece(synth_on_real, correct):.4f}  (miscalibrated — amplifies overconfidence)")

    if args.dry_run:
        print("(dry-run: not saved)")
        return
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, correct)
    np.savez(OUT, x=iso.X_thresholds_, y=iso.y_thresholds_,
             meta=json.dumps({
                 "fitted": "2026-07-13", "n": int(len(gt)),
                 "source": "yt_corpus_50 real-audio segments, iReal GT (DTW-aligned)",
                 "score": "quality-head family max-prob (proxy for production confidence_raw)",
                 "target": "q5 family correct",
                 "cv_ece_raw": round(float(ece_raw), 4),
                 "cv_ece_cal": round(float(ece_cv), 4),
                 "base_rate": round(float(correct.mean()), 4),
                 "caveat": ("proxy score (baseline LR _FamilyClassifier on corpus feat48), "
                            "NOT production ctx/joint confidence_raw; root_conf not included; "
                            "DTW GT noise. Real-audio confidence is near non-discriminative "
                            "(flat reliability ~base rate)."),
             }))
    print(f"saved -> {OUT}  ({len(iso.X_thresholds_)} breakpoints)")


if __name__ == "__main__":
    main()
