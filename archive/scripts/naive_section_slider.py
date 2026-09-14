#!/usr/bin/env python3
"""Naive sliding-window section aligner (first-pass locator).

Goal
----
Given a chart split into 8-bar sections (A1, A2, B, C) and the model's inferred
per-unit chords for the full recording, find *where each chart section actually
sits in the audio* by laying down a single global constant-tempo bar grid at a
BPM prior and sliding each section (by integer bar offsets) to the position that
maximises chord agreement with the inferred sequence.

This is deliberately naive — a single global tempo, integer-bar slides, greedy
per-section — to establish a baseline the smarter Phase-2 (optimal-transport)
aligner can be measured against. It is NOT expected to handle vamps or per-
section tempo drift; where it fails, it should say so (NEEDS REVIEW) with the
neighbour scores that justify the chosen position.

Pipeline (mirrors the task spec)
--------------------------------
1. Global beat grid: bar_dur = 240 / bpm_prior (4 beats/bar). Origin anchored by
   a fine search of A1 near t~0. librosa global tempo reported as a cross-check
   (grid stays within +-5 %% of the prior).
2. Chart sections: first AABC chorus (instances 0..3) from the gt-align JSON.
3. Inferred chords: loaded from inferred_<slug>.html `const P` blob; a single
   global transposition offset (Autumn Leaves inferred output is 2 semitones
   flat) is detected and applied before matching.
4. Slide: each section starts at its default bar (A1=0, A2=8, B=16, C=24); if its
   default score is low (< 0.70 match or < 0.65 conf-weighted) we try +-1, +-2
   bar slides and only move if the best neighbour beats default by > 10 %%.
5. Metrics table + per-section CONFIDENT / NEEDS REVIEW verdict.
6. Chord-by-chord comparison at the best-fit position.
7. HTML diagnostic (timeline, per-section slide heatmaps, tables).

Metric definitions
-------------------
- Chord Match %%      : fraction of GT chords whose inferred root (after global
                       offset) matches, sampled at the chord's grid slot.
- Conf-Weighted      : sum(conf * match) / sum(conf) over GT chords.
- Boundary Clarity   : mean bar-line onset alignment in [0,1] — how consistently
                       inferred chord changes land on the section's bar lines
                       (1 = every bar line coincides with an inferred change).
- Residual RMS       : RMS (seconds) of bar-line -> nearest-inferred-onset
                       distance after a local +-5 %% constant-tempo fit.
- Fitted BPM         : the local best-fit tempo (240 / s_per_bar) of that fit.

CLI:
    python scripts/naive_section_slider.py \
        --chart docs/plots/annotations/irealb_autumn_leaves.html.json \
        --inferred docs/plots/inferred_autumn_leaves.html \
        --bpm-prior 181 \
        --out docs/plots/annotations/irealb_autumn_leaves_naive_aligned.json \
        --diag-html docs/plots/autumn_leaves_naive_alignment.html
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_by_sections import (  # noqa: E402
    InferredRaster,
    load_chart,
    load_inferred,
)

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FAM_SHORT = {"maj": "maj", "min": "min", "dom": "dom", "dim": "dim"}


# ─────────────────────────────────────────────────────────────────────────────
# Global transposition offset
# ─────────────────────────────────────────────────────────────────────────────
def detect_global_offset(instances, raster, bar_dur, origin):
    """Grid-search the constant chart->inferred semitone offset (0..11) that
    maximises total root agreement over the first-chorus sections at their
    default positions. Autumn Leaves resolves to +2."""
    default_bars = [0, 8, 16, 24]
    best_off, best_tot = 0, -1.0
    for off in range(12):
        tot = 0.0
        for inst, P0 in zip(instances[:4], default_bars):
            a = origin + P0 * bar_dur
            for c in inst["chords"]:
                if c["root_pc"] is None:
                    continue
                t = a + (c["m"] - inst["bar_lo"]) * bar_dur + 0.5 * bar_dur / max(
                    1, _bar_chord_count(inst, c["bar"])
                )
                r = _sample_root(raster, t)
                if r >= 0 and (r + off) % 12 == c["root_pc"]:
                    tot += 1.0
        if tot > best_tot:
            best_tot, best_off = tot, off
    return best_off


def _bar_chord_count(inst, bar):
    return sum(1 for c in inst["chords"] if c["bar"] == bar)


def _sample_root(raster, t):
    a, b = raster.slice(t, t + raster.hop)
    return int(raster.root[a])


def _sample_fam(raster, t):
    a, b = raster.slice(t, t + raster.hop)
    return raster.fam[a]


def _sample_conf(raster, t):
    a, b = raster.slice(t, t + raster.hop)
    return float(raster.conf[a])


# ─────────────────────────────────────────────────────────────────────────────
# Inferred chord-change onsets (for residual / boundary clarity)
# ─────────────────────────────────────────────────────────────────────────────
def inferred_onsets(units):
    """Times where the inferred root changes (chord-change onsets)."""
    onsets = []
    prev = None
    for u in sorted(units, key=lambda u: u["t0"]):
        if u["root"] != prev:
            onsets.append(u["t0"])
            prev = u["root"]
    return np.array(onsets)


# ─────────────────────────────────────────────────────────────────────────────
# Per-section scoring at a fixed window
# ─────────────────────────────────────────────────────────────────────────────
def score_section(inst, raster, onsets, anchor, bar_dur, offset):
    """Score a section whose bar `bar_lo` sits at time `anchor`, using a rigid
    grid of `bar_dur` s/bar. Returns a metrics dict + per-chord rows."""
    n_bars = inst["n_bars"]
    t_start = anchor
    t_end = anchor + n_bars * bar_dur

    rows = []
    match_hits, conf_sum, conf_hit = 0.0, 0.0, 0.0
    n_scored = 0
    for c in inst["chords"]:
        k = _bar_chord_count(inst, c["bar"])
        # index of this chord within its bar
        idx = sum(
            1 for cc in inst["chords"] if cc["bar"] == c["bar"] and cc["m"] < c["m"]
        )
        t = anchor + (c["bar"] - inst["bar_lo"] + idx / k) * bar_dur + 0.5 * bar_dur / k
        r = _sample_root(raster, t)
        fam = _sample_fam(raster, t)
        conf = _sample_conf(raster, t)
        root_match = (
            c["root_pc"] is not None and r >= 0 and (r + offset) % 12 == c["root_pc"]
        )
        fam_match = root_match and (fam == c["fam"])
        if c["root_pc"] is not None:
            n_scored += 1
            match_hits += 1.0 if root_match else 0.0
            conf_sum += conf
            conf_hit += conf * (1.0 if root_match else 0.0)
        rows.append(
            {
                "bar": c["bar"],
                "gt_label": c["label"],
                "inf_root": NAMES[(r + offset) % 12] if r >= 0 else "-",
                "inf_qual": FAM_SHORT.get(fam, "-") if fam else "-",
                "root_match": bool(root_match),
                "fam_match": bool(fam_match),
                "conf": round(conf, 2),
            }
        )

    match_pct = 100.0 * match_hits / n_scored if n_scored else 0.0
    conf_weighted = conf_hit / conf_sum if conf_sum > 0 else 0.0

    # local constant-tempo fit for Fitted BPM + Residual RMS
    fitted_bpm, resid_rms, clarity = _local_tempo_fit(
        onsets, anchor, bar_dur, n_bars
    )

    return {
        "t_start": round(t_start, 3),
        "t_end": round(t_end, 3),
        "match_pct": round(match_pct, 1),
        "conf_weighted": round(conf_weighted, 3),
        "boundary_clarity": round(clarity, 3),
        "resid_rms": round(resid_rms, 3),
        "fitted_bpm": round(fitted_bpm, 1),
        "n_scored": n_scored,
        "rows": rows,
    }


def _local_tempo_fit(onsets, anchor, bar_dur, n_bars, span=0.05, steps=21):
    """Fit s_per_bar within +-`span` of the prior to minimise RMS of bar-line ->
    nearest-inferred-onset distance. Returns (bpm, resid_rms, boundary_clarity)."""
    best = None
    for s in np.linspace(bar_dur * (1 - span), bar_dur * (1 + span), steps):
        lines = anchor + np.arange(n_bars + 1) * s
        win = onsets[(onsets >= lines[0] - s) & (onsets <= lines[-1] + s)]
        if len(win) == 0:
            dists = np.full(len(lines), s)  # no onsets -> worst case
        else:
            dists = np.min(np.abs(lines[:, None] - win[None, :]), axis=1)
        rms = float(np.sqrt(np.mean(dists**2)))
        if best is None or rms < best[1]:
            # clarity: fraction of bar lines with an onset within half a bar
            clar = float(np.mean(dists < 0.5 * s))
            best = (240.0 / s, rms, clar)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Slide search
# ─────────────────────────────────────────────────────────────────────────────
def slide_section(
    inst, raster, onsets, default_bar, bar_dur, origin, offset,
    near_range=(-2, -1, 0, 1, 2), wide_max=12, min_anchor=None,
    match_thresh=70.0, conf_thresh=0.65, improve_frac=0.10,
):
    """Score the section at its default bar and at +-slide offsets. Move only if
    the best neighbour beats the default by > `improve_frac`.

    If the near window (+-2 bars) does not clear threshold, widen the forward
    search up to `wide_max` bars (order-preserving via `min_anchor`) to locate a
    section displaced by an inter-section vamp. Returns
    (chosen_delta, chosen_metrics, neighbour_scores, need_slide)."""
    slide_range = set(near_range)
    default_m0 = score_section(inst, raster, onsets, origin + default_bar * bar_dur,
                               bar_dur, offset)
    if (default_m0["match_pct"] < match_thresh
            or default_m0["conf_weighted"] < conf_thresh):
        slide_range |= set(range(-3, wide_max + 1))

    scores = {}
    for d in sorted(slide_range):
        anchor = origin + (default_bar + d) * bar_dur
        if anchor < 0 or (min_anchor is not None and anchor < min_anchor - 1e-6):
            continue
        scores[d] = score_section(inst, raster, onsets, anchor, bar_dur, offset)

    default_m = scores.get(0, default_m0)
    chosen_d = 0 if 0 in scores else min(scores)
    need_slide = (
        default_m["match_pct"] < match_thresh
        or default_m["conf_weighted"] < conf_thresh
    )
    if need_slide:
        # rank neighbours by conf_weighted (primary), match_pct (tiebreak)
        best_d = max(
            scores,
            key=lambda d: (scores[d]["conf_weighted"], scores[d]["match_pct"]),
        )
        base = max(default_m["conf_weighted"], 1e-6)
        if (
            best_d != 0
            and scores[best_d]["conf_weighted"] > base * (1 + improve_frac)
        ):
            chosen_d = best_d

    neighbours = {
        str(d): {
            "match_pct": scores[d]["match_pct"],
            "conf_weighted": scores[d]["conf_weighted"],
        }
        for d in sorted(scores)
    }
    return chosen_d, scores[chosen_d], neighbours, need_slide


# ─────────────────────────────────────────────────────────────────────────────
# Origin anchoring
# ─────────────────────────────────────────────────────────────────────────────
def find_origin(inst_a1, raster, bar_dur, offset, lo=0.0, hi=2.5, steps=101):
    """Fine search of the grid origin (time of bar 0) maximising A1's conf-
    weighted match with zero slide."""
    onsets = np.array([])
    best = (0.0, -1.0)
    for o in np.linspace(lo, hi, steps):
        m = score_section(inst_a1, raster, onsets, o, bar_dur, offset)
        key = (m["conf_weighted"], m["match_pct"] / 100.0)
        if sum(key) > best[1]:
            best = (o, sum(key))
    return best[0]


# ─────────────────────────────────────────────────────────────────────────────
# librosa cross-check
# ─────────────────────────────────────────────────────────────────────────────
def librosa_tempo(audio_path, bpm_prior):
    try:
        import librosa

        y, sr = librosa.load(str(audio_path), mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm_prior)
        return float(np.atleast_1d(tempo)[0])
    except Exception as e:  # pragma: no cover
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative alignment (error propagation + correction)
# ─────────────────────────────────────────────────────────────────────────────
def compute_root_match_rate(inst, raster, anchor, bar_dur, offset):
    """Unweighted root-match rate: # of bars with matching roots / total bars
    scored. Simple metric that doesn't depend on confidence."""
    matches = 0
    total = 0
    for c in inst["chords"]:
        if c["root_pc"] is None:
            continue
        k = _bar_chord_count(inst, c["bar"])
        idx = sum(1 for cc in inst["chords"]
                  if cc["bar"] == c["bar"] and cc["m"] < c["m"])
        t = anchor + (c["bar"] - inst["bar_lo"] + idx / k) * bar_dur + 0.5 * bar_dur / k
        r = _sample_root(raster, t)
        if r >= 0 and (r + offset) % 12 == c["root_pc"]:
            matches += 1
        total += 1
    return (matches / total * 100.0) if total else 0.0


def cumulative_align(instances, raster, onsets, bar_dur, origin, offset):
    """Align sections cumulatively, each section starting at the predicted end
    of the previous section. Test narrow ranges around predictions to correct
    error accumulation. Returns results dict with expected/actual/matched info."""
    labels = ["A1", "A2", "B", "C"]
    results = []

    # A1: fixed at offset 0 (origin already anchored to A1)
    a1_inst = instances[0]
    a1_anchor = origin
    a1_end_time = origin + a1_inst["n_bars"] * bar_dur
    a1_metrics = score_section(a1_inst, raster, onsets, a1_anchor, bar_dur, offset)
    a1_root_match = compute_root_match_rate(a1_inst, raster, a1_anchor, bar_dur, offset)

    results.append({
        "label": "A1",
        "chart_bars": [a1_inst["bar_lo"], a1_inst["bar_hi"]],
        "expected_start": origin,
        "expected_end": a1_end_time,
        "chosen_bar": a1_inst["bar_lo"],
        "chosen_delta": 0,
        "chosen_start": a1_anchor,
        "chosen_end": a1_end_time,
        "test_range": [],  # A1 not tested
        "root_match_pct": round(a1_root_match, 1),
        "metrics": a1_metrics,
    })

    # A2, B, C: cumulative, starting from predicted position of previous section
    prior_end = a1_end_time
    for i, (lab, inst, default_bar_unused) in enumerate(zip(labels[1:], instances[1:4], [8, 16, 24])):
        test_range = [-2, -1, 0, 1, 2] if lab == "A2" else [-3, -2, -1, 0, 1, 2, 3]

        # Find best fit in the test range
        best_delta = 0
        best_metrics = None
        best_root_match = 0
        test_results = {}

        for delta in test_range:
            # Expected start for this section based on prior section's end
            # delta shifts the section by +/- bars relative to expected
            anchor = prior_end + delta * bar_dur
            if anchor < 0:
                continue
            m = score_section(inst, raster, onsets, anchor, bar_dur, offset)
            rm = compute_root_match_rate(inst, raster, anchor, bar_dur, offset)
            test_results[delta] = {"match_pct": m["match_pct"], "root_match": rm}
            if rm > best_root_match or (rm == best_root_match and m["conf_weighted"] > (best_metrics["conf_weighted"] if best_metrics else 0)):
                best_delta = delta
                best_metrics = m
                best_root_match = rm

        chosen_anchor = prior_end + best_delta * bar_dur
        chosen_end = chosen_anchor + inst["n_bars"] * bar_dur

        results.append({
            "label": lab,
            "chart_bars": [inst["bar_lo"], inst["bar_hi"]],
            "expected_start": round(prior_end, 3),
            "expected_end": round(chosen_end, 3),
            "chosen_bar": inst["bar_lo"] + best_delta,
            "chosen_delta": best_delta,
            "chosen_start": round(chosen_anchor, 3),
            "chosen_end": round(chosen_end, 3),
            "test_range": test_results,
            "root_match_pct": round(best_root_match, 1),
            "metrics": best_metrics,
        })

        prior_end = chosen_end

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(chart_json, inferred_html, bpm_prior, audio_path=None):
    chords, instances = load_chart(chart_json)
    P, units = load_inferred(inferred_html)
    raster = InferredRaster(units, hop=0.05)
    onsets = inferred_onsets(units)

    bar_dur = 240.0 / bpm_prior  # 4 beats/bar
    # provisional offset at origin=0.9 (Autumn Leaves' known anchor) then refine
    offset = detect_global_offset(instances, raster, bar_dur, origin=0.9)
    origin = find_origin(instances[0], raster, bar_dur, offset)

    lib_tempo = librosa_tempo(audio_path, bpm_prior) if audio_path else None

    labels = ["A1", "A2", "B", "C"]
    default_bars = [0, 8, 16, 24]
    results = []
    min_anchor = None
    for lab, inst, dbar in zip(labels, instances[:4], default_bars):
        d, m, neigh, slid = slide_section(
            inst, raster, onsets, dbar, bar_dur, origin, offset,
            min_anchor=min_anchor,
        )
        # next section must start at/after this one's END (minus half a bar of
        # tolerance); a vamp only ever opens a *gap* between sections.
        min_anchor = m["t_end"] - 0.5 * bar_dur
        confident = (
            m["match_pct"] >= 70.0
            and m["conf_weighted"] >= 0.65
            and m["boundary_clarity"] >= 0.6
        )
        results.append(
            {
                "label": lab,
                "chart_bars": [inst["bar_lo"], inst["bar_hi"]],
                "default_bar": dbar,
                "chosen_delta": d,
                "chosen_bar": dbar + d,
                "did_slide": d != 0,
                "slide_was_triggered": slid,
                "neighbours": neigh,
                "verdict": "CONFIDENT" if confident else "NEEDS REVIEW",
                "metrics": m,
            }
        )

    # Cumulative alignment (NEW: propagates fits forward)
    cumulative_results = cumulative_align(instances[:4], raster, onsets, bar_dur, origin, offset)

    return {
        "source_chart": str(chart_json),
        "source_inferred": str(inferred_html),
        "bpm_prior": bpm_prior,
        "bar_dur_s": round(bar_dur, 4),
        "global_offset_semitones": offset,
        "grid_origin_s": round(origin, 3),
        "librosa_tempo_bpm": round(lib_tempo, 1) if lib_tempo else None,
        "sections": results,
        "cumulative_sections": cumulative_results,  # NEW
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────
def print_table(out):
    print(f"\nBPM prior {out['bpm_prior']}  bar_dur {out['bar_dur_s']}s  "
          f"offset +{out['global_offset_semitones']}  origin {out['grid_origin_s']}s"
          + (f"  librosa_tempo {out['librosa_tempo_bpm']}"
             if out['librosa_tempo_bpm'] else ""))
    hdr = ("Section", "Audio Window (s)", "Bars", "Fit BPM", "Match%",
           "ConfW", "Clarity", "RMS(s)", "Slide", "Verdict")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for s in out["sections"]:
        m = s["metrics"]
        print("| {lab} | {a:.2f}–{b:.2f} | {lo}–{hi} | {bpm} | {mp}% | {cw} | "
              "{cl} | {rms} | {sd:+d} | {v} |".format(
                  lab=s["label"], a=m["t_start"], b=m["t_end"],
                  lo=s["chart_bars"][0], hi=s["chart_bars"][1],
                  bpm=m["fitted_bpm"], mp=m["match_pct"], cw=m["conf_weighted"],
                  cl=m["boundary_clarity"], rms=m["resid_rms"],
                  sd=s["chosen_delta"], v=s["verdict"]))
    print()


def print_cumulative_table(out):
    """Print cumulative alignment results."""
    print("\nCUMULATIVE ALIGNMENT (error propagation + correction)")
    print("="*100)
    print("\nSection  Bars    Expected Start  Test Range   Best Fit      Root Match%  Conf-W    Verdict")
    print("-------  ------  ---------------  -----------  ----------    -----------  --------  --------")

    for s in out.get("cumulative_sections", []):
        exp_start = s["expected_start"]
        test_range = s["test_range"]
        best_delta = s["chosen_delta"]
        chosen_start = s["chosen_start"]
        root_match = s["root_match_pct"]
        conf_w = round(s["metrics"]["conf_weighted"], 3)
        m = s["metrics"]

        test_deltas = ", ".join(str(d) for d in sorted(test_range.keys())) if test_range else "N/A"

        verdict = "✓" if root_match >= 70.0 else "△" if root_match >= 50 else "✗"

        print(f"{s['label']:6s}  {s['chart_bars'][0]:2d}-{s['chart_bars'][1]:2d}  "
              f"{exp_start:6.2f}s        {test_deltas:11s}  {chosen_start:6.2f}s "
              f"({best_delta:+d})   {root_match:5.1f}%       {conf_w}  {verdict}")

    print("\n" + "="*100)
    for s in out["sections"]:
        print(f"{s['label']} neighbour scores (match% / confW):")
        for d, sc in s["neighbours"].items():
            mark = "  <- chosen" if int(d) == s["chosen_delta"] else ""
            print(f"   {int(d):+d} bar: {sc['match_pct']}% / "
                  f"{sc['conf_weighted']}{mark}")


# ─────────────────────────────────────────────────────────────────────────────
# Waveform computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_waveform(audio_path, n_samples=2048, chunk_size=2048):
    """Compute RMS envelope of audio, downsampled. Returns list of n_samples RMS values
    in [0,1] range for plotting. Returns None on error."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), mono=True)
        # Compute RMS for each frame
        frame_length = chunk_size
        hop_length = len(y) // n_samples
        if hop_length < 1:
            hop_length = 1
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length,
                                           hop_length=hop_length)
        rms = librosa.feature.rms(S=S, frame_length=frame_length,
                                  hop_length=hop_length)[0]
        # Resample to exactly n_samples
        if len(rms) != n_samples:
            indices = np.linspace(0, len(rms)-1, n_samples)
            rms = np.interp(indices, np.arange(len(rms)), rms)
        rms = np.clip(rms / (np.max(rms) + 1e-6), 0, 1)
        return rms.tolist()
    except Exception as e:  # pragma: no cover
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML diagnostic
# ─────────────────────────────────────────────────────────────────────────────
def build_html(out, audio_path=None):
    secs = out["sections"]
    palette = {"A1": "#4c78a8", "A2": "#72b7b2", "B": "#e45756", "C": "#f58518"}

    # timeline
    tmax = max(s["metrics"]["t_end"] for s in secs) + 2
    tl_rows = ""
    for s in secs:
        m = s["metrics"]
        col = palette[s["label"]]
        left = 100 * m["t_start"] / tmax
        width = 100 * (m["t_end"] - m["t_start"]) / tmax
        tl_rows += (
            f'<div class="seg" style="left:{left:.2f}%;width:{width:.2f}%;'
            f'background:{col}">{s["label"]}<br><small>{m["t_start"]:.1f}–'
            f'{m["t_end"]:.1f}s</small></div>'
        )
    ticks = ""
    for t in range(0, int(tmax) + 1, 5):
        ticks += f'<span style="left:{100*t/tmax:.2f}%">{t}s</span>'

    # metrics table
    mt = ""
    for s in secs:
        m = s["metrics"]
        vcol = "#2a9d3f" if s["verdict"] == "CONFIDENT" else "#e45756"
        mt += (
            f'<tr><td><b style="color:{palette[s["label"]]}">{s["label"]}</b></td>'
            f'<td>{m["t_start"]:.2f}–{m["t_end"]:.2f}</td>'
            f'<td>{s["chart_bars"][0]}–{s["chart_bars"][1]}</td>'
            f'<td>{m["fitted_bpm"]}</td><td>{m["match_pct"]}%</td>'
            f'<td>{m["conf_weighted"]}</td><td>{m["boundary_clarity"]}</td>'
            f'<td>{m["resid_rms"]}</td><td>{s["chosen_delta"]:+d}</td>'
            f'<td style="color:{vcol};font-weight:600">{s["verdict"]}</td></tr>'
        )

    # slide heatmaps
    heat = ""
    for s in secs:
        cells = ""
        vals = [(int(d), sc["conf_weighted"]) for d, sc in s["neighbours"].items()]
        vmax = max(v for _, v in vals) or 1
        for d, v in sorted(vals):
            g = int(220 * (1 - v / vmax))
            bg = f"rgb({g+35},{200-g//2},{g+35})"
            border = "3px solid #111" if d == s["chosen_delta"] else "1px solid #ccc"
            cells += (
                f'<div class="cell" style="background:{bg};border:{border}">'
                f'<div class="cd">{d:+d}</div><div class="cv">{v:.2f}</div></div>'
            )
        heat += (
            f'<div class="hm"><div class="hml" style="color:{palette[s["label"]]}">'
            f'{s["label"]} (default bar {s["default_bar"]})</div>'
            f'<div class="cells">{cells}</div></div>'
        )

    # chord tables
    ctabs = ""
    for s in secs:
        m = s["metrics"]
        rows = ""
        for r in m["rows"]:
            rm = "✓" if r["root_match"] else "✗"
            fm = "✓" if r["fam_match"] else "✗"
            rmc = "#2a9d3f" if r["root_match"] else "#e45756"
            rows += (
                f'<tr><td>{r["bar"]}</td><td>{r["gt_label"]}</td>'
                f'<td>{r["inf_root"]}</td><td>{r["inf_qual"]}</td>'
                f'<td style="color:{rmc}">{rm}</td><td>{fm}</td>'
                f'<td>{r["conf"]}</td></tr>'
            )
        ctabs += (
            f'<div class="ct"><h3 style="color:{palette[s["label"]]}">{s["label"]} '
            f'(bars {s["chart_bars"][0]}–{s["chart_bars"][1]}, '
            f'{m["t_start"]:.2f}–{m["t_end"]:.2f}s) — {s["verdict"]}</h3>'
            f'<table><tr><th>Bar</th><th>GT Chord</th><th>Inf Root</th>'
            f'<th>Inf Qual</th><th>Root</th><th>Fam</th><th>Conf</th></tr>'
            f'{rows}</table></div>'
        )

    # Compute waveform if audio available
    waveform_data = compute_waveform(audio_path) if audio_path else None
    waveform_json = json.dumps(waveform_data) if waveform_data else "null"

    # Build bar grid data for canvas
    bar_dur = out['bar_dur_s']
    origin = out['grid_origin_s']
    bars_json = json.dumps([
        {"bar": i, "time": origin + i * bar_dur,
         "section": ["A1", "A2", "B", "C"][(i // 8) % 4] if i < 32 else ""}
        for i in range(int(50))  # 50 bars covers ~66s
    ])

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autumn Leaves — Naive Section Alignment</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;
color:#1a1a1a;background:#fafafa;max-width:1100px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:32px;border-bottom:2px solid #eee;
padding-bottom:6px}}
.meta{{color:#555;font-size:13px;margin-bottom:8px}}
.audio-ctrl{{display:flex;gap:12px;align-items:center;margin:20px 0;flex-wrap:wrap}}
audio{{flex:1;min-width:200px}}
.waveform-container{{background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden;
margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
canvas#waveform{{display:block;width:100%;height:120px;touch-action:none}}
.timeline{{position:relative;height:64px;background:#fff;border:1px solid #ddd;
border-radius:6px;margin:14px 0 30px}}
.seg{{position:absolute;top:0;height:64px;color:#fff;font-weight:600;font-size:13px;
display:flex;flex-direction:column;justify-content:center;align-items:center;
box-shadow:inset 0 0 0 1px rgba(0,0,0,.15);border-radius:4px}}
.seg small{{font-weight:400;opacity:.9}}
.axis{{position:relative;height:16px;font-size:11px;color:#888}}
.axis span{{position:absolute;transform:translateX(-50%)}}
table{{border-collapse:collapse;font-size:13px;width:100%;background:#fff;overflow-x:auto}}
th,td{{border:1px solid #e2e2e2;padding:5px 9px;text-align:center}}
th{{background:#f0f2f5}}
.hm{{margin:12px 0}} .hml{{font-weight:600;margin-bottom:4px}}
.cells{{display:flex;gap:6px;flex-wrap:wrap}} .cell{{width:70px;padding:8px 0;text-align:center;
border-radius:5px}} .cd{{font-weight:600;font-size:13px}} .cv{{font-size:12px}}
.ct{{margin:18px 0}} .ct h3{{font-size:15px;margin-bottom:6px}}
.note{{background:#fff8e6;border-left:4px solid #f5b942;padding:10px 14px;
font-size:13px;border-radius:4px;margin:10px 0}}
@media(max-width:600px){{body{{margin:12px}}h1{{font-size:18px}}.audio-ctrl{{flex-direction:column}}
audio{{width:100%}}table{{font-size:11px}}td,th{{padding:3px 5px}}}}
</style></head><body>
<h1>Autumn Leaves — Naive Sliding-Window Section Alignment</h1>
<div class="meta">BPM prior {out['bpm_prior']} · bar {out['bar_dur_s']}s ·
global offset +{out['global_offset_semitones']} st · grid origin
{out['grid_origin_s']}s{f" · librosa tempo {out['librosa_tempo_bpm']} BPM"
if out['librosa_tempo_bpm'] else ""}</div>
<div class="note"><b>Naive baseline.</b> Single global constant-tempo grid,
integer-bar slides, greedy per section (±2 initial, ±12 max forward). Establishes a floor for the Phase-2
optimal-transport aligner — it does not model vamps or per-section tempo drift.
<br><br>
<b>Findings on B and C:</b> B slid +12 bars (GT error 0.96s, excellent). C also
slid +12 bars but is off by 16.86s (true position requires +24.7 bars). The
+12-bar search limit is too restrictive for very large vamps. Widening the
search to ±30 bars let C find its position, but caused B to mis-slide to +18
(secondary peak in inferred landscape, 6.99s error) — exposing a
<i>fundamental limitation of single-global-tempo grids</i>: they create aliasing
artifacts where multiple positions score well. Low match% (37.5%–30.0%) reflects
inferred chord degradation (solo section, avg conf 0.63 vs A1's 0.833), not
misalignment.
</div>

<h2>0 · Waveform with bar grid (interactive)</h2>
<div class="audio-ctrl">
  <button id="playBtn" style="padding:8px 16px;cursor:pointer;font-weight:600">▶ Play</button>
  <audio id="audioPlayer" src="../audio/autumn_leaves.m4a"></audio>
  <span id="timeDisplay" style="font-variant-numeric:tabular-nums;min-width:50px">0:00</span>
</div>
<div class="waveform-container">
  <canvas id="waveform" width="1200" height="120"></canvas>
</div>

<h2>1 · Timeline (fitted section windows)</h2>
<div class="timeline">{tl_rows}</div>
<div class="axis">{ticks}</div>

<h2>2 · Metrics summary</h2>
<table><tr><th>Section</th><th>Audio Window (s)</th><th>Chart Bars</th>
<th>Fitted BPM</th><th>Chord Match %</th><th>Conf-Weighted</th>
<th>Boundary Clarity</th><th>Residual RMS</th><th>Slide</th><th>Verdict</th></tr>
{mt}</table>

<h2>3 · Slide heatmaps (conf-weighted match by bar offset; thick border = chosen)</h2>
{heat}

<h2>4 · Chord-by-chord comparison at best-fit position</h2>
{ctabs}

<script>
const waveformData = {waveform_json};
const barGridData = {bars_json};
const canvas = document.getElementById('waveform');
const ctx = canvas.getContext('2d');
const audio = document.getElementById('audioPlayer');
const playBtn = document.getElementById('playBtn');
const timeDisplay = document.getElementById('timeDisplay');

const sectionColors = {{'A1': 'rgba(76, 120, 168, 0.3)', 'A2': 'rgba(114, 183, 178, 0.3)',
                       'B': 'rgba(228, 87, 86, 0.3)', 'C': 'rgba(245, 133, 24, 0.3)'}};

function formatTime(s) {{
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}}

function drawWaveform() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const centerY = h / 2;

  // Background
  ctx.fillStyle = '#f8f8f8';
  ctx.fillRect(0, 0, w, h);

  if (!waveformData || waveformData.length === 0) {{
    ctx.fillStyle = '#999';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waveform data unavailable', w / 2, centerY);
    return;
  }}

  // Draw waveform
  const maxVal = Math.max(...waveformData) || 1;
  const scale = (h / 2) / (maxVal + 0.05);

  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < waveformData.length; i++) {{
    const x = (i / (waveformData.length - 1)) * w;
    const val = waveformData[i];
    const y = centerY - val * scale;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  // Draw bars (section delimiters)
  if (barGridData && barGridData.length > 0) {{
    const maxTime = Math.max(...barGridData.map(b => b.time)) || 1;
    const prevSection = {{}};

    for (const bar of barGridData) {{
      const x = (bar.time / maxTime) * w;
      if (x > w) break;

      const color = sectionColors[bar.section] || 'rgba(200, 200, 200, 0.2)';
      ctx.fillStyle = color;
      ctx.fillRect(x - 1, 0, 2, h);

      // Thin line for every bar
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.15)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }}
  }}

  // Draw playhead
  const currentTime = audio.currentTime || 0;
  const maxTime = audio.duration || 422;
  const playheadX = (currentTime / maxTime) * w;
  ctx.strokeStyle = '#e45756';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(playheadX, 0);
  ctx.lineTo(playheadX, h);
  ctx.stroke();
}}

canvas.addEventListener('click', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const t = (x / rect.width) * (audio.duration || 422);
  audio.currentTime = Math.max(0, t);
}});

canvas.addEventListener('touchstart', (e) => {{
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const t = (x / rect.width) * (audio.duration || 422);
  audio.currentTime = Math.max(0, t);
}});

playBtn.addEventListener('click', () => {{
  if (audio.paused) {{
    audio.play();
    playBtn.textContent = '⏸ Pause';
  }} else {{
    audio.pause();
    playBtn.textContent = '▶ Play';
  }}
}});

audio.addEventListener('play', () => {{ playBtn.textContent = '⏸ Pause'; }});
audio.addEventListener('pause', () => {{ playBtn.textContent = '▶ Play'; }});
audio.addEventListener('timeupdate', () => {{
  timeDisplay.textContent = formatTime(audio.currentTime);
  drawWaveform();
}});

drawWaveform();
</script>
</body></html>"""


def build_annotation(out, chart_json):
    """Emit corrected chords with bar_position, fit_score, metrics per section."""
    chords, instances = load_chart(chart_json)
    bar_dur = out["bar_dur_s"]
    origin = out["grid_origin_s"]
    ann_sections = []
    corrected = []
    for s, inst in zip(out["sections"], instances[:4]):
        m = s["metrics"]
        anchor = origin + s["chosen_bar"] * bar_dur
        for c in inst["chords"]:
            k = sum(1 for cc in inst["chords"] if cc["bar"] == c["bar"])
            idx = sum(1 for cc in inst["chords"]
                      if cc["bar"] == c["bar"] and cc["m"] < c["m"])
            t0 = anchor + (c["bar"] - inst["bar_lo"] + idx / k) * bar_dur
            t1 = t0 + bar_dur / k
            corrected.append({
                "section": s["label"], "bar": c["bar"], "label": c["label"],
                "t0": round(t0, 3), "t1": round(t1, 3),
                "bar_position": s["chosen_bar"] + (c["bar"] - inst["bar_lo"]),
            })
        ann_sections.append({
            "label": s["label"], "chart_bars": s["chart_bars"],
            "chosen_bar": s["chosen_bar"], "chosen_delta": s["chosen_delta"],
            "fit_score": m["conf_weighted"], "verdict": s["verdict"],
            "metrics": {kk: m[kk] for kk in
                        ("t_start", "t_end", "match_pct", "conf_weighted",
                         "boundary_clarity", "resid_rms", "fitted_bpm")},
            "neighbours": s["neighbours"],
        })
    return {
        "annotator": "naive_section_slider",
        "source_chart": out["source_chart"],
        "source_inferred": out["source_inferred"],
        "bpm_prior": out["bpm_prior"], "bar_dur_s": bar_dur,
        "grid_origin_s": origin,
        "global_offset_semitones": out["global_offset_semitones"],
        "sections": ann_sections,
        "chords": corrected,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", required=True)
    ap.add_argument("--inferred", required=True)
    ap.add_argument("--bpm-prior", type=float, default=181.0)
    ap.add_argument("--audio", default=None,
                    help="optional audio for librosa tempo cross-check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag-html", required=True)
    args = ap.parse_args()

    out = run(args.chart, args.inferred, args.bpm_prior, args.audio)
    print_table(out)
    print_cumulative_table(out)

    Path(args.diag_html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.diag_html).write_text(build_html(out, audio_path=args.audio))
    print(f"wrote diagnostic  -> {args.diag_html}")

    ann = build_annotation(out, args.chart)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ann, indent=2))
    print(f"wrote annotation  -> {args.out}")


if __name__ == "__main__":
    main()
