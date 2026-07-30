#!/usr/bin/env python3
"""Search for a per-chord confidence score that actually DISCRIMINATES. 2026-07-30.

Context: the deployed displayed confidence has AUC 0.480 over the 7 verified
Brick-0 songs — it does not rank right chords above wrong ones, so no monotone
recalibration can help (see docs/known_issues.md 2026-07-30, and
scratchpad/refit_conf_shipped.py). Louis's call: find a score that discriminates,
then calibrate THAT.

Candidates all come from music-x-lab's own frame posteriors — the thing the
displayed number never looks at. `musx_redecode.frame_posteriors` returns the
5-fold-averaged triad head as (n_frame, 73) on a 23.22 ms grid: column 0 is N,
column i>=1 is root (i-1)%12 with triad type (i-1)//12 in
{maj,min,sus4,sus2,dim,aug}. These are the RAW (pre-fold) posteriors, which is
what we want — an independent second opinion on the chord we chose to display.

Boundary robustness: the re-decode picks a per-song latency (up to ~100 ms) that
is not recovered here, so every span is TRIMMED to its interior before scoring.
That removes both the latency offset and genuine transition blur, at the cost of
dropping very short spans.

Run: .venv/bin/python scratchpad/conf_score_search.py
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
from harmonia.models.musx_redecode import FRAME_DT, frame_posteriors  # noqa: E402

CACHE = REPO / "data" / "cache"
GOLDEN = REPO / "golden" / "brick0"
OUT = Path(__file__).with_suffix(".json")

TRIM = 0.15          # drop this fraction off each end of a span before scoring
MIN_FRAMES = 4       # a span needs this many interior frames to be scored

# triad column i>=1 -> (root_pc, type_idx); type order per musx's chord list
_TRIAD_TYPES = ("maj", "min", "sus4", "sus2", "dim", "aug")
# our parent families -> the musx triad types that can express them
_FAMILY_TO_TYPES = {
    "maj": (0,), "dom": (0,), "min": (1,), "sus": (2, 3),
    "dim": (4,), "hdim": (4,), "aug": (5,),
}


def span_scores(triad: np.ndarray, t0: float, t1: float, root_pc: int, fam: str):
    """Candidate confidence scores for one displayed chord span."""
    a = int(round((t0 + TRIM * (t1 - t0)) / FRAME_DT))
    b = int(round((t1 - TRIM * (t1 - t0)) / FRAME_DT))
    a, b = max(0, a), min(len(triad), b)
    if b - a < MIN_FRAMES:
        return None
    P = triad[a:b]                                   # (n, 73)
    n_chord = P[:, 1:].reshape(len(P), 6, 12)        # (n, type, root)
    root_marg = n_chord.sum(1)                       # (n, 12) root marginal
    p_n = P[:, 0]                                    # P(no chord)

    srt = np.sort(root_marg, axis=1)
    mean_top1 = float(srt[:, -1].mean())
    mean_margin = float((srt[:, -1] - srt[:, -2]).mean())
    p_root = float(root_marg[:, root_pc].mean())     # mass on the root WE display
    agree = float((root_marg.argmax(1) == root_pc).mean())
    ent = root_marg / np.maximum(root_marg.sum(1, keepdims=True), 1e-12)
    entropy = float((-(ent * np.log(ent + 1e-12)).sum(1)).mean())

    types = _FAMILY_TO_TYPES.get(fam, (0,))
    p_chord = float(n_chord[:, types, root_pc].sum(1).mean())   # root AND family

    return {
        "musx_p_root": p_root,          # mass musx puts on the root we display
        "musx_p_chord": p_chord,        # ...on the root AND family we display
        "musx_agree": agree,            # frames whose argmax root == ours
        "musx_top1": mean_top1,         # musx's own peak height (label-blind)
        "musx_margin": mean_margin,     # top - runner-up (label-blind)
        "musx_neg_entropy": -entropy,   # sharpness (label-blind)
        "musx_not_n": float(1.0 - p_n.mean()),
        "n_frames": int(b - a),
    }


def gather(gt_path: Path):
    gt = load_frozen_gt(gt_path)
    audio = gt.resolved_audio_path
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(audio, Path(tmp))
        chart = infer_chords_v1(wav, cache_dir=CACHE, **SHIPPED_CONFIG)
    triad = frame_posteriors(audio)[0]
    rows = []
    for c in chart.chords:
        t0, t1 = float(c["start_s"]), float(c["end_s"])
        pred = chord_from_label(t0, t1, str(c["label"]))
        if t1 <= t0 or pred.is_nc or pred.root_pc is None:
            continue
        best, best_ov = None, 0.0
        for g in gt.gt_chords:
            ov = min(t1, g.t1) - max(t0, g.t0)
            if ov > best_ov:
                best, best_ov = g, ov
        if best is None or best_ov <= 0 or best.is_nc:
            continue
        fam = chord_family(pred.quality)
        s = span_scores(triad, t0, t1, int(pred.root_pc), fam)
        if s is None:
            continue
        s["deployed"] = float(c["confidence"])
        s["raw"] = float(c["confidence_raw"])
        s["dur"] = t1 - t0
        s["ok"] = bool(pred.root_pc == best.root_pc
                       and fam == chord_family(best.quality))
        rows.append(s)
    return rows


def main():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    data = {}
    for p in sorted(GOLDEN.glob("*.gt.json")):
        gt = load_frozen_gt(p)
        if not gt.verified or not gt.resolved_audio_path.exists():
            continue
        r = gather(p)
        if r:
            data[gt.song_id] = r
            print(f"  {gt.song_id:24s} {len(r):4d} scored spans")

    songs = list(data)
    rows = [r for s in songs for r in data[s]]
    Y = np.array([r["ok"] for r in rows], float)
    W = np.array([r["dur"] for r in rows])
    G = np.array([s for s in songs for _ in data[s]])
    base = float(np.average(Y, weights=W))
    print(f"\n{len(rows)} spans, {len(songs)} songs, partial acc={base:.3f}")

    keys = ["deployed", "raw", "musx_p_root", "musx_p_chord", "musx_agree",
            "musx_top1", "musx_margin", "musx_neg_entropy", "musx_not_n"]
    print(f"\n{'score':20s} {'AUC':>7s}  {'per-song AUCs':<44s}")
    aucs = {}
    for k in keys:
        X = np.array([r[k] for r in rows], float)
        a = roc_auc_score(Y, X, sample_weight=W)
        per = []
        for s in songs:
            m = G == s
            per.append(roc_auc_score(Y[m], X[m], sample_weight=W[m])
                       if len(set(Y[m])) > 1 else float("nan"))
        aucs[k] = {"pooled": float(a), "per_song": [float(v) for v in per]}
        print(f"{k:20s} {a:7.3f}  " + " ".join(f"{v:.2f}" for v in per))

    # the one that wins, calibrated leave-one-song-out
    best_k = max((k for k in keys if k not in ("deployed", "raw")),
                 key=lambda k: aucs[k]["pooled"])
    X = np.array([r[best_k] for r in rows], float)
    oof = np.zeros_like(X)
    for s in songs:
        te = G == s
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(X[~te], Y[~te], sample_weight=W[~te])
        oof[te] = iso.predict(X[te])

    def ece(conf, bins=15):
        edges = np.linspace(0, 1, bins + 1)
        e, WT = 0.0, W.sum()
        for i in range(bins):
            hi = conf <= 1.0 if i == bins - 1 else conf < edges[i + 1]
            m = (conf >= edges[i]) & hi
            if m.any():
                e += W[m].sum() / WT * abs(np.average(conf[m], weights=W[m])
                                           - np.average(Y[m], weights=W[m]))
        return float(e)

    dep = np.array([r["deployed"] for r in rows], float)
    print(f"\nbest discriminating score: {best_k} (AUC {aucs[best_k]['pooled']:.3f})")
    print(f"  deployed today      mean={np.average(dep, weights=W):.3f}  ECE={ece(dep):.4f}")
    print(f"  {best_k} calibrated LOSO  mean={np.average(oof, weights=W):.3f}  "
          f"ECE={ece(oof):.4f}")
    print("\n  calibrated score -> actual accuracy, by decile of the new score:")
    q = np.quantile(X, np.linspace(0, 1, 6))
    for lo, hi in zip(q[:-1], q[1:]):
        m = (X >= lo) & (X <= hi)
        if m.sum() > 3:
            print(f"    {best_k} [{lo:.2f},{hi:.2f}]  n={m.sum():4d}  "
                  f"shown={np.average(oof[m], weights=W[m]):.3f}  "
                  f"actual={np.average(Y[m], weights=W[m]):.3f}")

    np.savez(Path(__file__).with_name("conf_score_rows.npz"),
             y=Y, w=W, g=G, oof=oof, best=X, deployed=dep,
             **{k: np.array([r[k] for r in rows], float) for k in keys})
    OUT.write_text(json.dumps({"n": len(rows), "songs": songs, "base": base,
                               "auc": aucs, "best": best_k,
                               "ece_deployed": ece(dep), "ece_best_loso": ece(oof)},
                              indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
