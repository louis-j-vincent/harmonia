#!/usr/bin/env python3
"""drum_fills_eval.py — precision/recall + same/different-bar calibration for
drum_fills.py, plus the diagnostic PNG (score bar chart + onset envelope +
PR curve) per song.

This Love boundaries are Louis's hand spec (docs/this_love_target_spec.md +
current segmenter agreement). Every Breath You Take / Billie Jean boundaries
are MODEL-DERIVED from pattern_slide.segment()'s chord-SSM blocks -- NOT hand
GT, said explicitly in every printout.

Run:  .venv/bin/python scratchpad/drum_fills_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from drum_fills import component_scores
from pattern_slide import rigid_slots, chord_ssm, segment
from rhythm_ssm import _drum_onset_bands, _hpss_percussive, separate_drums
from section_merge_declined import _load_payload
from harmonia.models.rigid_grid import rigid_grid_for
from harmonia.serving.config import AUDIO_DIR

# This Love -- Louis's hand spec, verbatim from the brief
THIS_LOVE_PRIMARY = [16, 24, 36, 44, 48, 56]      # true section-boundary bars
THIS_LOVE_SECONDARY = [64, 72]                     # weaker (chorus re-entries)
THIS_LOVE_CADENTIAL = [22, 42, 62, 70, 78]         # fill-plausible, NOT a boundary


def average_precision(scores: np.ndarray, positive_bars: set[int], valid_idx: list[int]) -> float:
    order = [i for i in sorted(valid_idx, key=lambda j: -scores[j])]
    n_pos = len(positive_bars)
    if n_pos == 0:
        return float("nan")
    hits, precisions = 0, []
    for rank, idx in enumerate(order, start=1):
        if idx in positive_bars:
            hits += 1
            precisions.append(hits / rank)
    return float(np.mean(precisions)) if precisions else 0.0


def pr_curve(scores: np.ndarray, positive_bars: set[int], valid_idx: list[int]):
    order = sorted(valid_idx, key=lambda j: -scores[j])
    n_pos = max(len(positive_bars), 1)
    hits = 0
    P, R = [], []
    for rank, idx in enumerate(order, start=1):
        if idx in positive_bars:
            hits += 1
        P.append(hits / rank)
        R.append(hits / n_pos)
    return np.array(R), np.array(P)


def report_group_gap(name: str, scores: np.ndarray, pos: set[int], neg_pool: list[int]):
    s = [scores[i] for i in pos if i < len(scores)]
    d = [scores[i] for i in neg_pool]
    def stat(v):
        if not v:
            return "n=0"
        p25, p50, p75 = np.percentile(v, [25, 50, 75])
        return f"{min(v):.3f}-{max(v):.3f} n={len(v)} mean={np.mean(v):.3f} median={p50:.3f} IQR=[{p25:.3f},{p75:.3f}]"
    print(f"    [{name}] fill-bar (positive):     {stat(s)}")
    print(f"    [{name}] other bars (negative):   {stat(d)}")
    if s and d:
        print(f"    [{name}] median-gap (pos - neg) = {np.median(s) - np.median(d):+.3f}")


def eval_song(slug_glob: str, primary_boundaries: list[int] | None = None,
              secondary_boundaries: list[int] | None = None,
              cadential_bars: list[int] | None = None,
              boundaries_are_hand_gt: bool = True, out_png: Path | None = None):
    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{slug_glob}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    tonic = int((P.get("home") or {}).get("tonic", 0)) % 12
    grid = rigid_grid_for(P.get("chords", []), tonic_pc=tonic)
    seq, names, n_bars, bar_sec = rigid_slots(P)
    assert len(grid) - 1 == n_bars

    if primary_boundaries is None:
        S_chord = chord_ssm(names)
        blocks = segment(S_chord, seq, names, log=lambda *a, **k: None)
        primary_boundaries = sorted({b["start"] // 2 for b in blocks if b["start"] > 0})
        boundaries_are_hand_gt = False
    secondary_boundaries = secondary_boundaries or []
    cadential_bars = cadential_bars or []

    primary_fill = {b - 1 for b in primary_boundaries if b - 1 >= 0}
    secondary_fill = {b - 1 for b in secondary_boundaries if b - 1 >= 0}
    all_positive_fill = primary_fill | secondary_fill
    cadential_set = set(cadential_bars)
    valid_idx = [i for i in range(n_bars - 1) if i not in cadential_set]  # exclude last bar (undefined) + cadential

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    comps_sym, source = component_scores(audio_path, grid, return_source=True, causal=False)
    comps_causal, _ = component_scores(audio_path, grid, return_source=True, causal=True)
    comps = comps_causal  # causal is the one we now trust; symmetric kept for the printed comparison

    print(f"\n{'=' * 70}\n{slug}  n_bars={n_bars} bar_sec={bar_sec:.3f}  drum source={source}")
    print(f"boundaries are {'HAND GT' if boundaries_are_hand_gt else 'MODEL-DERIVED (pattern_slide.segment on chord SSM) -- NOT hand GT'}")
    print(f"primary boundaries: {primary_boundaries} -> fill-target bars {sorted(primary_fill)}")
    if secondary_boundaries:
        print(f"secondary boundaries: {secondary_boundaries} -> fill-target bars {sorted(secondary_fill)}")
    if cadential_bars:
        print(f"cadential-tail bars (excluded from neg pool, reported separately): {cadential_bars}")

    k = len(primary_fill)
    chance_p = k / len(valid_idx) if valid_idx else float("nan")
    print(f"chance level: precision@k = k/n_valid = {k}/{len(valid_idx)} = {chance_p:.3f}  "
          f"(AP at chance is also ~{chance_p:.3f} for a random ranking)")
    results = {}
    for name, sc in comps.items():
        sc_sym = comps_sym[name]
        top_k = sorted(valid_idx, key=lambda j: -sc[j])[:k]
        hit = len(set(top_k) & primary_fill)
        prec = hit / k if k else float("nan")
        ap = average_precision(sc, primary_fill, valid_idx)
        ap_sym = average_precision(sc_sym, primary_fill, valid_idx)
        results[name] = dict(top_k=top_k, hit=hit, precision_at_k=prec, ap=ap)
        print(f"\n  -- {name} --  precision@k={prec:.2f} ({hit}/{k})  "
              f"AP(causal)={ap:.3f}  AP(symmetric)={ap_sym:.3f}  [chance={chance_p:.3f}]")
        print(f"    top-{k} bars: {top_k}")
        neg_pool = [i for i in valid_idx if i not in all_positive_fill]
        report_group_gap(name, sc, primary_fill, neg_pool)
        if cadential_bars:
            cad_scores = [sc[b] for b in cadential_bars if b < len(sc)]
            if cad_scores:
                print(f"    [{name}] cadential-tail bars (informational, not scored as FP): "
                      f"{[round(sc[b], 3) for b in cadential_bars if b < len(sc)]} mean={np.mean(cad_scores):.3f}")

    if out_png is not None:
        _plot(slug, comps, grid, n_bars, bar_sec, primary_boundaries, secondary_boundaries,
              cadential_bars, primary_fill, valid_idx, results, source, out_png)
    return dict(slug=slug, comps=comps, results=results, n_bars=n_bars,
                primary_fill=primary_fill, grid=grid)


def _plot(slug, comps, grid, n_bars, bar_sec, primary_boundaries, secondary_boundaries,
          cadential_bars, primary_fill, valid_idx, results, source, out_png):
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    drum_wav = separate_drums(audio_path)
    import librosa
    if drum_wav is not None:
        y, sr = librosa.load(str(drum_wav), sr=22050, mono=True)
    else:
        y, sr = _hpss_percussive(audio_path, sr=22050)
    env, times = _drum_onset_bands(y, sr)

    fig, axes = plt.subplots(3, 1, figsize=(20, 11), gridspec_kw={"height_ratios": [2, 1.4, 1.6]})

    ax = axes[0]
    bars = np.arange(n_bars)
    ax.bar(bars, comps["combined_mean"], width=0.9, color="steelblue", label="combined_mean")
    ax.plot(bars, comps["density"], "o-", ms=2, lw=0.7, color="orange", alpha=0.7, label="density")
    ax.plot(bars, comps["groove_dev"], "o-", ms=2, lw=0.7, color="green", alpha=0.7, label="groove_dev")
    ax.plot(bars, comps["crash"], "o-", ms=2, lw=0.7, color="red", alpha=0.7, label="crash")
    ax.plot(bars, comps["flurry"], "o-", ms=2, lw=0.7, color="purple", alpha=0.7, label="flurry")
    for b in sorted(primary_fill):
        ax.axvline(b, color="black", lw=1.4, alpha=0.85)
    for b in secondary_boundaries or []:
        ax.axvline(b - 1, color="black", lw=1.0, alpha=0.4, ls="--")
    for b in cadential_bars or []:
        ax.axvline(b, color="gray", lw=1.0, alpha=0.5, ls=":")
    ax.set_xlim(-0.5, n_bars - 0.5)
    ax.set_ylabel("fill score")
    ax.legend(fontsize=8, ncol=6, loc="upper right")
    ax.set_title(f"{slug} -- fill-score per bar (black solid = true fill-target bar (boundary-1), "
                 "dashed = secondary, dotted gray = cadential-tail, drums via " + source + ")", fontsize=10)

    ax = axes[1]
    labels = ["low(kick)", "mid(snare)", "high(hats/crash)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i in range(3):
        ax.plot(times, env[i] + i * 1.2, lw=0.4, color=colors[i], label=labels[i])
    for b in grid:
        ax.axvline(b, color="black", lw=0.3, alpha=0.25)
    for b in sorted(primary_fill):
        ax.axvline(grid[b], color="black", lw=1.2, alpha=0.8)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylabel("onset envelope (offset per band)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    for name in ["density", "groove_dev", "crash", "flurry", "combined_mean"]:
        sc = comps[name]
        order = sorted(valid_idx, key=lambda j: -sc[j])
        n_pos = max(len(primary_fill), 1)
        hits, P, R = 0, [], []
        for rank, idx in enumerate(order, start=1):
            if idx in primary_fill:
                hits += 1
            P.append(hits / rank)
            R.append(hits / n_pos)
        ax.plot(R, P, marker=".", ms=3, label=f"{name} (AP={results[name]['ap']:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("precision-recall vs primary fill-target bars", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125)
    plt.close(fig)
    print(f"  saved {out_png}")


if __name__ == "__main__":
    outdir = REPO / "docs" / "research_sessions"
    eval_song("this_love", THIS_LOVE_PRIMARY, THIS_LOVE_SECONDARY, THIS_LOVE_CADENTIAL,
              boundaries_are_hand_gt=True, out_png=outdir / "drum_fills_this_love_2026-07-30.png")
    eval_song("the_police_every_breath",
              out_png=outdir / "drum_fills_the_police_every_breath_2026-07-30.png")
    eval_song("michael_jackson_billie_jean",
              out_png=outdir / "drum_fills_michael_jackson_billie_jean_2026-07-30.png")
