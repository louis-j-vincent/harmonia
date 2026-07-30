#!/usr/bin/env python3
"""Round 2: add the INDEPENDENT second opinion + feature combination. 2026-07-30.

Round 1 (`conf_score_search.py`) found that music-x-lab's own posterior scores
beat the deployed confidence (AUC 0.60 vs 0.48) but are still weak. The reason
is structural: under SHIPPED_CONFIG the displayed root and quality COME FROM
music-x-lab, so every musx-derived score is the model grading its own homework.

The NNLS-24 root head (`ChordStageResult.beat_proba`, a per-beat 12-way root
posterior) never touches the displayed label on this config — it is a genuinely
independent second opinion, and disagreement between two models is the classic
uncertainty signal that self-confidence cannot provide.

`beat_proba` is not on the ChordChart, and rebuilding the beat stage to get it
would risk feeding the head a different grid than production (error pattern #1).
So it is CAPTURED from the live call instead: wrap `NNLS24ChordHead.run` and
stash its result while `infer_chords_v1` runs normally.

Run: .venv/bin/python scratchpad/conf_score_search2.py
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
from harmonia.stages.chord_head import NNLS24ChordHead  # noqa: E402

CACHE = REPO / "data" / "cache"
GOLDEN = REPO / "golden" / "brick0"
OUT = Path(__file__).with_suffix(".json")

TRIM = 0.15
MIN_FRAMES = 4
_TRIAD_TYPES = ("maj", "min", "sus4", "sus2", "dim", "aug")
_FAMILY_TO_TYPES = {"maj": (0,), "dom": (0,), "min": (1,), "sus": (2, 3),
                    "dim": (4,), "hdim": (4,), "aug": (5,)}

FEATURES = ["musx_p_root", "musx_p_chord", "musx_agree", "musx_top1",
            "musx_margin", "musx_neg_entropy", "musx_not_n",
            "nnls_p_root", "nnls_agree", "nnls_margin", "cross_agree",
            "log_dur", "raw"]


def run_capturing(wav: Path):
    """infer_chords_v1 on the shipped config, capturing the NNLS stage result."""
    # run_full INLINES the stages (it never calls self.run), so the hook goes on
    # extract_features — the method that actually produces beat_proba — and the
    # beat grid `bt` is captured from the same call.
    box = {}
    orig = NNLS24ChordHead.extract_features

    def spy(self, audio_path, bt, *a, **kw):
        out = orig(self, audio_path, bt, *a, **kw)
        box["beat_proba"] = np.asarray(out[4], float)
        box["bt"] = np.asarray(bt, float)
        return out

    NNLS24ChordHead.extract_features = spy
    try:
        chart = infer_chords_v1(wav, cache_dir=CACHE, **SHIPPED_CONFIG)
    finally:
        NNLS24ChordHead.extract_features = orig
    return chart, box.get("beat_proba"), box.get("bt")


def span_scores(triad, beat_proba, bt, t0, t1, root_pc, fam):
    a = int(round((t0 + TRIM * (t1 - t0)) / FRAME_DT))
    b = int(round((t1 - TRIM * (t1 - t0)) / FRAME_DT))
    a, b = max(0, a), min(len(triad), b)
    if b - a < MIN_FRAMES:
        return None
    P = triad[a:b]
    n_chord = P[:, 1:].reshape(len(P), 6, 12)
    root_marg = n_chord.sum(1)
    srt = np.sort(root_marg, axis=1)
    ent = root_marg / np.maximum(root_marg.sum(1, keepdims=True), 1e-12)
    types = _FAMILY_TO_TYPES.get(fam, (0,))
    out = {
        "musx_p_root": float(root_marg[:, root_pc].mean()),
        "musx_p_chord": float(n_chord[:, types, root_pc].sum(1).mean()),
        "musx_agree": float((root_marg.argmax(1) == root_pc).mean()),
        "musx_top1": float(srt[:, -1].mean()),
        "musx_margin": float((srt[:, -1] - srt[:, -2]).mean()),
        "musx_neg_entropy": float(-(-(ent * np.log(ent + 1e-12)).sum(1)).mean()),
        "musx_not_n": float(1.0 - P[:, 0].mean()),
    }
    # ── the independent opinion: NNLS-24 root head, per beat ────────────────
    i0 = int(np.searchsorted(bt, t0, side="left"))
    i1 = max(int(np.searchsorted(bt, t1, side="left")), i0 + 1)
    i1 = min(i1, len(beat_proba))
    if i0 >= len(beat_proba):
        return None
    B = beat_proba[i0:i1]
    bs = np.sort(B, axis=1)
    out["nnls_p_root"] = float(B[:, root_pc].mean())
    out["nnls_agree"] = float((B.argmax(1) == root_pc).mean())
    out["nnls_margin"] = float((bs[:, -1] - bs[:, -2]).mean())
    # do the two models point at the same root? (neither is told the other's answer)
    out["cross_agree"] = float((B.argmax(1) == root_marg.argmax(1)[
        np.clip(np.linspace(0, len(root_marg) - 1, len(B)).astype(int),
                0, len(root_marg) - 1)]).mean())
    return out


def gather(gt_path: Path):
    gt = load_frozen_gt(gt_path)
    audio = gt.resolved_audio_path
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(audio, Path(tmp))
        chart, beat_proba, bt = run_capturing(wav)
    if beat_proba is None or bt is None:
        print(f"    !! no NNLS stage captured for {gt.song_id}")
        return []
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
        s = span_scores(triad, beat_proba, bt, t0, t1, int(pred.root_pc), fam)
        if s is None:
            continue
        s["raw"] = float(c["confidence_raw"])
        s["deployed"] = float(c["confidence"])
        s["log_dur"] = float(np.log(t1 - t0))
        s["dur"] = t1 - t0
        s["ok"] = bool(pred.root_pc == best.root_pc
                       and fam == chord_family(best.quality))
        rows.append(s)
    return rows


def main():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    data = {}
    for p in sorted(GOLDEN.glob("*.gt.json")):
        gt = load_frozen_gt(p)
        if not gt.verified or not gt.resolved_audio_path.exists():
            continue
        r = gather(p)
        if r:
            data[gt.song_id] = r
            print(f"  {gt.song_id:24s} {len(r):4d} spans")

    songs = list(data)
    rows = [r for s in songs for r in data[s]]
    Y = np.array([r["ok"] for r in rows], float)
    W = np.array([r["dur"] for r in rows])
    G = np.array([s for s in songs for _ in data[s]])
    print(f"\n{len(rows)} spans, {len(songs)} songs, acc={np.average(Y, weights=W):.3f}")

    def auc_of(X):
        return float(roc_auc_score(Y, X, sample_weight=W))

    print(f"\n{'score':20s} {'AUC':>7s}  per-song")
    single = {}
    for k in ["deployed"] + FEATURES:
        X = np.array([r[k] for r in rows], float)
        per = [roc_auc_score(Y[G == s], X[G == s], sample_weight=W[G == s])
               if len(set(Y[G == s])) > 1 else float("nan") for s in songs]
        single[k] = auc_of(X)
        print(f"{k:20s} {single[k]:7.3f}  " + " ".join(f"{v:.2f}" for v in per))

    # ── combination, leave-one-song-out ─────────────────────────────────────
    M = np.column_stack([[r[k] for r in rows] for k in FEATURES]).astype(float)
    oof = np.zeros(len(Y))
    for s in songs:
        te = G == s
        sc = StandardScaler().fit(M[~te])
        lr = LogisticRegression(max_iter=2000, C=0.5)
        lr.fit(sc.transform(M[~te]), Y[~te], sample_weight=W[~te])
        oof[te] = lr.predict_proba(sc.transform(M[te]))[:, 1]
    per = [roc_auc_score(Y[G == s], oof[G == s], sample_weight=W[G == s])
           if len(set(Y[G == s])) > 1 else float("nan") for s in songs]
    print(f"\n{'COMBINED (LOSO)':20s} {auc_of(oof):7.3f}  "
          + " ".join(f"{v:.2f}" for v in per))

    sc = StandardScaler().fit(M)
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(M), Y, sample_weight=W)
    print("\n  weights (standardised, + = more likely correct):")
    for k, w_ in sorted(zip(FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"    {k:20s} {w_:+.3f}")

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
    print(f"\n  deployed today   mean={np.average(dep, weights=W):.3f}  ECE={ece(dep):.4f}"
          f"  AUC={single['deployed']:.3f}")
    print(f"  COMBINED (LOSO)  mean={np.average(oof, weights=W):.3f}  ECE={ece(oof):.4f}"
          f"  AUC={auc_of(oof):.3f}")
    print("\n  combined score -> actual accuracy, by quintile:")
    q = np.quantile(oof, np.linspace(0, 1, 6))
    for lo, hi in zip(q[:-1], q[1:]):
        m = (oof >= lo) & (oof <= hi)
        if m.sum() > 3:
            print(f"    shown {np.average(oof[m], weights=W[m]):.3f}   "
                  f"actual {np.average(Y[m], weights=W[m]):.3f}   n={m.sum():4d}")

    np.savez(Path(__file__).with_name("conf_score2_rows.npz"),
             y=Y, w=W, g=G, oof=oof, M=M, feats=np.array(FEATURES), dep=dep)
    OUT.write_text(json.dumps({"n": len(rows), "songs": songs,
                               "single_auc": single, "combined_auc": auc_of(oof),
                               "combined_per_song": [float(v) for v in per],
                               "ece_deployed": ece(dep), "ece_combined": ece(oof)},
                              indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
