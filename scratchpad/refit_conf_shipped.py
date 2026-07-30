#!/usr/bin/env python3
"""Audit + re-fit the DISPLAYED-confidence map for the shipped config. 2026-07-30.

WHERE THE DISPLAYED NUMBER ACTUALLY COMES FROM (traced 2026-07-30, because two
earlier readings of this were wrong and each would have shipped a bad change):

    SHIPPED_CONFIG feature_frontend="nnls24"
      -> infer_chords_v1 L4059 returns _infer_nnls24(...)   [L4724's emission is
         DEAD on this path]
      -> NNLS24ChordHead.run_full
      -> harmonia/stages/chord_head.py::_finalize_chords(conf_map=_get_nnls24_conf_map())
      -> confidence = interp(confidence_raw, map)           <- what the app shows

So the live display map IS `harmonia/models/nnls24_conf_calibration.npz`.

Two things that are NOT true, recorded so nobody re-derives them:
  * `data/cache/confidence_calibration{,_real}.npz` (_get_conf_calibrator) is the
    BILLBOARD/legacy path. Neither file exists on disk and that is harmless —
    the live path never calls it.
  * `chord_pipeline_v1._finalize_chords` (L3613) is dead; chord_head has its own.

THE CAVEAT THAT MATTERS FOR DEPLOYMENT: the same npz is ALSO read at
chord_pipeline_v1 L3448 to build `bar_conf` for the Occam bar-compression's
Bayes arbitration — which CHANGES CHORD LABELS. Overwriting the file in place
therefore moves live chord decisions, not just a percentage. Any display-only
recalibration must be a SEPARATE map applied at the display emission.

TARGET (Louis, 2026-07-30): the number should mean P(root + parent family are
right) — partial credit — because you play Cm over Cm7 and you are fine.

Fitted on the 7 verified Brick-0 songs (real audio, shipped config, fold ON),
validated LEAVE-ONE-SONG-OUT. `root_conf` is None on this path, so the "fused"
(conf x root posterior) variant does not exist here and is not attempted.

Run: .venv/bin/python scratchpad/refit_conf_shipped.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.eval.accuracy_score import (SHIPPED_CONFIG, _decode_to_wav,  # noqa: E402
                                          chord_family, chord_from_label,
                                          load_frozen_gt)
from harmonia.models.chord_pipeline_v1 import infer_chords_v1  # noqa: E402

CACHE = REPO / "data" / "cache"
GOLDEN = REPO / "golden" / "brick0"
DEPLOYED = REPO / "harmonia" / "models" / "nnls24_conf_calibration.npz"
OUT = Path(__file__).with_suffix(".json")


def ece(conf, correct, w, bins=15):
    """Duration-weighted expected calibration error."""
    conf, correct, w = np.asarray(conf), np.asarray(correct, float), np.asarray(w, float)
    edges = np.linspace(0, 1, bins + 1)
    e, W = 0.0, w.sum()
    for i in range(bins):
        hi = conf <= 1.0 if i == bins - 1 else conf < edges[i + 1]
        m = (conf >= edges[i]) & hi
        if not m.any():
            continue
        ww = w[m].sum()
        e += ww / W * abs(np.average(conf[m], weights=w[m])
                          - np.average(correct[m], weights=w[m]))
    return float(e)


def gather(gt_path: Path):
    """(raw, shown, correct?, duration) per PREDICTED chord under the shipped config.

    correct = root AND parent family match the GT chord holding most of the
    predicted span — what `accuracy_score` reports as partial credit, attributed
    per predicted chord so it can be regressed on.
    """
    gt = load_frozen_gt(gt_path)
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(gt.resolved_audio_path, Path(tmp))
        chart = infer_chords_v1(wav, cache_dir=CACHE, **SHIPPED_CONFIG)
    rows = []
    for c in chart.chords:
        t0, t1 = float(c["start_s"]), float(c["end_s"])
        dur = t1 - t0
        pred = chord_from_label(t0, t1, str(c["label"]))
        if dur <= 0 or pred.is_nc:
            continue
        best, best_ov = None, 0.0
        for g in gt.gt_chords:
            ov = min(t1, g.t1) - max(t0, g.t0)
            if ov > best_ov:
                best, best_ov = g, ov
        if best is None or best_ov <= 0 or best.is_nc:
            continue
        ok = (pred.root_pc == best.root_pc
              and chord_family(pred.quality) == chord_family(best.quality))
        rows.append((float(c["confidence_raw"]), float(c["confidence"]), bool(ok), dur))
    return rows


def fit_iso(x, y, w):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, y, sample_weight=w)
    return iso


def main():
    from sklearn.metrics import roc_auc_score

    data = {}
    for p in sorted(GOLDEN.glob("*.gt.json")):
        gt = load_frozen_gt(p)
        if not gt.verified or not gt.resolved_audio_path.exists():
            continue
        rows = gather(p)
        if rows:
            data[gt.song_id] = rows

    songs = list(data)
    RAW = np.array([r[0] for s in songs for r in data[s]])
    SHOWN = np.array([r[1] for s in songs for r in data[s]])
    Y = np.array([r[2] for s in songs for r in data[s]], float)
    W = np.array([r[3] for s in songs for r in data[s]])
    G = np.array([s for s in songs for _ in data[s]])
    base = float(np.average(Y, weights=W))

    print("\nper song   shown(app)   partial-acc   AUC(raw)")
    for s in songs:
        m = G == s
        a = (roc_auc_score(Y[m], RAW[m], sample_weight=W[m])
             if len(set(Y[m])) > 1 else float("nan"))
        print(f"  {s:24s} {np.average(SHOWN[m], weights=W[m]):.3f}   "
              f"{np.average(Y[m], weights=W[m]):.3f}   {a:.3f}")
    print(f"\npooled: {len(Y)} chords, {len(songs)} songs, partial acc={base:.3f}")

    # ── 1. is the number honest? ────────────────────────────────────────────
    ece_shown = ece(SHOWN, Y, W)
    ece_const = ece(np.full_like(SHOWN, base), Y, W)
    print(f"\nDEPLOYED map (what the app shows): mean={np.average(SHOWN, weights=W):.3f}  "
          f"ECE={ece_shown:.4f}")
    print(f"CONSTANT base rate               : mean={base:.3f}  ECE={ece_const:.4f}")

    oof = np.zeros_like(RAW, dtype=float)
    for s in songs:
        te = G == s
        oof[te] = fit_iso(RAW[~te], Y[~te], W[~te]).predict(RAW[te])
    ece_refit = ece(oof, Y, W)
    print(f"REFIT on raw (LOSO)              : mean={np.average(oof, weights=W):.3f}  "
          f"ECE={ece_refit:.4f}")

    # ── 2. does the number MEAN anything? ───────────────────────────────────
    # ECE is 0 by construction for a constant equal to the base rate, so it
    # cannot distinguish "well calibrated" from "uninformative". AUC asks the
    # decisive question: does a higher score actually mean more often right?
    auc = roc_auc_score(Y, RAW, sample_weight=W)
    print(f"\nDISCRIMINATION  AUC(raw score) = {auc:.3f}   "
          f"(0.5 = the number carries no information)")
    print("\nraw score -> accuracy, by bin:")
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .9), (.9, 1.01)]:
        m = (RAW >= lo) & (RAW < hi)
        if m.sum():
            print(f"   raw [{lo:.1f},{hi:.1f})  n={m.sum():4d}  "
                  f"shown={np.average(SHOWN[m], weights=W[m]):.3f}  "
                  f"actual={np.average(Y[m], weights=W[m]):.3f}")

    np.savez(Path(__file__).with_name("refit_conf_rows.npz"),
             raw=RAW, shown=SHOWN, y=Y, w=W, g=G)
    OUT.write_text(json.dumps({
        "n_chords": int(len(Y)), "n_songs": len(songs), "partial_acc": base,
        "ece_shown_deployed": ece_shown, "ece_constant": ece_const,
        "ece_refit_loso": ece_refit, "auc_raw": float(auc),
        "mean_shown_deployed": float(np.average(SHOWN, weights=W)),
        "mean_refit_loso": float(np.average(oof, weights=W)),
        "per_song": {s: {"n": len(data[s]),
                         "shown": float(np.average(SHOWN[G == s], weights=W[G == s])),
                         "acc": float(np.average(Y[G == s], weights=W[G == s]))}
                     for s in songs},
    }, indent=1))
    print(f"\nwrote {OUT}")

    if auc < 0.55:
        print("\nVERDICT: the raw score does not rank correct chords above wrong "
              "ones. No monotone map can fix that — the best any calibration can "
              "do is collapse to the base rate, which LOOKS per-chord but is not. "
              "Recalibrating is the wrong fix; the score itself has to change.")


if __name__ == "__main__":
    main()
