#!/usr/bin/env python3
"""Re-fit the displayed-confidence map on the SHIPPED config. 2026-07-30.

The deployed map (`harmonia/models/nnls24_conf_calibration.npz`) was fitted on
the NNLS-24 heads at ORACLE GT blocks over 100 RWC songs. Production has since
moved to music-x-lab for root/quality/bass/segmentation, and music-x-lab is more
accurate — so the map now understates. Measured on 4 verified Brick-0 songs:
displayed 0.519 vs partial-credit accuracy 0.669, i.e. ~15 pp low, while
simultaneously OVER-reporting strict on 2 of them.

Louis's call (2026-07-30): the number should mean **P(root + parent family are
right)** — partial credit — because that is how you read a chart: you play Cm over
Cm7 and you are fine, so a missing 7th should not cost the reader's attention.
That is already the target the old script used (its docstring says "7-family" but
its code compares parent families), so the TARGET was never wrong; the FITTING
CONFIG was.

The honest constraint: RWC audio is not on disk, only its cached NNLS features, so
the shipped pipeline cannot be re-run over those 100 songs. This fits on the 7
verified Brick-0 songs instead — correct config, real audio, fold ON, but a much
smaller sample. It is therefore validated LEAVE-ONE-SONG-OUT and only recommended
if LOSO beats the deployed map on the same data. A 7-song curve that only wins
in-sample is an overfit and must be rejected.

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
                                          chord_from_label, family_of, load_gt)
from harmonia.models.chord_pipeline_v1 import infer_chords_v1  # noqa: E402

CACHE = REPO / "data" / "cache"
DEPLOYED = REPO / "harmonia" / "models" / "nnls24_conf_calibration.npz"
OUT = Path(__file__).with_suffix(".json")

SONGS = {
    "stand_by_me": "ben_e_king_stand_by_me_audio",
    "bein_green": "bein_green",
    "georgia_on_my_mind": "ray_charles_georgia_on_my_mind_official_video",
    "close_to_you": "carpenters_close_to_you",
    "every_breath_you_take": "the_police_every_breath_you_take_official_music_video",
    "blue_bossa": "blue_bossa",
    "blue_bossa_backing": "blue_bossa_150bpm_backing_track",
}


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


def gather(song: str, stem: str):
    """(raw score, correct?, duration) per PREDICTED chord, under the shipped config.

    Correctness = root AND parent family, against the GT chord holding the most
    of the predicted span — the same notion `accuracy_score` scores as
    partial_credit, just attributed per predicted chord so it can be regressed on.
    """
    gt = load_gt(REPO / "golden" / "brick0" / f"{song}.gt.json")
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(REPO / "docs" / "audio" / f"{stem}.m4a", Path(tmp))
        chart = infer_chords_v1(wav, cache_dir=CACHE, **SHIPPED_CONFIG)
    rows = []
    for c in chart.chords:
        t0, t1 = float(c["start_s"]), float(c["end_s"])
        dur = t1 - t0
        if dur <= 0 or str(c["label"]).upper().startswith("N"):
            continue
        # the GT chord covering most of this predicted span
        best, best_ov = None, 0.0
        for g in gt.gt_chords:
            ov = min(t1, g.t1) - max(t0, g.t0)
            if ov > best_ov:
                best, best_ov = g, ov
        if best is None or best_ov <= 0:
            continue
        pred = chord_from_label(t0, t1, str(c["label"]))
        ok = (pred.root == best.root
              and family_of(pred.quality) == family_of(best.quality))
        rows.append((float(c["confidence_raw"]), bool(ok), dur))
    return rows


def fit_iso(x, y, w):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, y, sample_weight=w)
    return iso


def main():
    data = {}
    for song, stem in SONGS.items():
        if not (REPO / "docs" / "audio" / f"{stem}.m4a").exists():
            print(f"  skip {song}: no audio")
            continue
        rows = gather(song, stem)
        data[song] = rows
        acc = np.average([r[1] for r in rows], weights=[r[2] for r in rows])
        print(f"  {song:24s} {len(rows):4d} chords   partial acc={acc:.3f}")

    songs = list(data)
    X = np.array([r[0] for s in songs for r in data[s]])
    Y = np.array([r[1] for s in songs for r in data[s]], float)
    W = np.array([r[2] for s in songs for r in data[s]])
    G = np.array([s for s in songs for _ in data[s]])
    print(f"\npooled: {len(X)} chords, {len(songs)} songs, "
          f"partial acc={np.average(Y, weights=W):.3f}")

    dep = np.load(DEPLOYED)
    dep_conf = np.interp(X, dep["x"], dep["y"])
    print(f"\nDEPLOYED map : mean shown={np.average(dep_conf, weights=W):.3f}  "
          f"ECE={ece(dep_conf, Y, W):.4f}")
    print(f"RAW score    : mean={np.average(X, weights=W):.3f}  "
          f"ECE={ece(X, Y, W):.4f}")

    # leave-one-song-out — the only number that decides whether to ship
    oof = np.zeros_like(X)
    for s in songs:
        te = G == s
        iso = fit_iso(X[~te], Y[~te], W[~te])
        oof[te] = iso.predict(X[te])
    print(f"REFIT (LOSO) : mean shown={np.average(oof, weights=W):.3f}  "
          f"ECE={ece(oof, Y, W):.4f}")

    better = ece(oof, Y, W) < ece(dep_conf, Y, W)
    print(f"\n-> LOSO {'BEATS' if better else 'does NOT beat'} the deployed map")

    iso_full = fit_iso(X, Y, W)
    grid = np.linspace(0.0, 1.0, 101)
    curve = iso_full.predict(grid)
    print("\nrefit curve (raw -> P(root+family right)):")
    for r in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        print(f"   raw {r:.1f}  deployed {float(np.interp(r, dep['x'], dep['y'])):.3f}"
              f"   ->  refit {float(np.interp(r, grid, curve)):.3f}")

    OUT.write_text(json.dumps({
        "n_chords": int(len(X)), "n_songs": len(songs),
        "partial_acc": float(np.average(Y, weights=W)),
        "ece_deployed": ece(dep_conf, Y, W), "ece_raw": ece(X, Y, W),
        "ece_refit_loso": ece(oof, Y, W), "loso_beats_deployed": bool(better),
        "mean_shown_deployed": float(np.average(dep_conf, weights=W)),
        "mean_shown_refit_loso": float(np.average(oof, weights=W)),
        "grid": grid.tolist(), "curve": curve.tolist(),
        "per_song": {s: {"n": len(data[s]),
                          "acc": float(np.average([r[1] for r in data[s]],
                                                  weights=[r[2] for r in data[s]]))}
                      for s in songs},
    }, indent=1))
    print(f"\nwrote {OUT}")
    if better:
        np.savez(Path(__file__).with_name("nnls24_conf_calibration_refit.npz"),
                 x=grid, y=curve, target="partial_credit_root_and_family",
                 fitted_on="brick0_7_verified_shipped_config", n=len(X))
        print("wrote scratchpad/nnls24_conf_calibration_refit.npz (NOT deployed)")


if __name__ == "__main__":
    main()
