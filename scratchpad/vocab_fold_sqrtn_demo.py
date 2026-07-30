#!/usr/bin/env python3
"""vocab_fold_sqrtn_demo.py — SHOW the √N noise reduction and what it does to the
quality call. This Love, on the real NNLS-24 observations the live decoder reads.

Answers Louis's "where's the demo that shows the comparison for the quality call
with reduction in square root of N of the noise?" — with the measurement, not the
claim:

  1. per-occurrence observations at one slot (N noisy 12-d treble chroma vectors)
  2. their mean, and the MEASURED spread of that mean vs the 1/√N prediction
  3. the quality call before and after: template scores for maj / dom7 / min /
     min7 at the verse downbeat, so "G or G7" becomes a number you can read
  4. the same, per vocabulary item, across the whole song

Uses the CHART-GRADE vocabulary (rigid grid + section_vocab on the decoded
chords), not the in-pipeline provisional chain — which is exactly why the fold
defers on This Love in-pipeline today.

Run: .venv/bin/python scratchpad/vocab_fold_sqrtn_demo.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SERIES = ["#3b6fd4", "#d97a1f", "#1f8a6b", "#8a4fbf"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"

# chord-quality templates over the 12 treble pitch classes, relative to the root
TEMPLATES = {
    "maj":  [0, 4, 7],
    "dom7": [0, 4, 7, 10],
    "min":  [0, 3, 7],
    "min7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
}


def features(audio: Path):
    """(feat24, beat_times) — the exact per-beat observation chord_head reads.

    Beats come from **beatthis**, the project's official tracker (never librosa,
    which locks a 2x tempo octave)."""
    from harmonia.models.chord_pipeline_v1 import _get_beatthis
    from harmonia.stages.chord_head import NNLS24ChordHead
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(audio), "-ar", "44100", str(wav)],
                       check=True, capture_output=True)
        f2b = _get_beatthis()
        if f2b is None:
            raise SystemExit("beatthis unavailable — refusing the librosa fallback")
        bt, _dbs = f2b(str(wav))
        bt = np.asarray(bt, dtype=float)
        # returns (heads, arr, times, feat, beat_proba, key_result)
        _h, _a, _t, feat, _bp, _k = NNLS24ChordHead().extract_features(wav, bt)
    return np.asarray(feat), bt


def chart_vocabulary(slug: str):
    """The chart-grade vocabulary + bar grid: rigid grid, then section_vocab on the
    regridded display bars. This is the good grouping the display layer already
    computes for every song."""
    sys.path.insert(0, str(REPO / "scratchpad"))
    from section_merge_declined import _load_payload
    from harmonia.models.rigid_grid import apply_rigid_grid, rigid_grid_for
    from harmonia.models.section_vocab import form_string, vocab_sections

    P = _load_payload(REPO / "docs" / "plots" / f"inferred_{slug}.html")
    tonic = int((P.get("home") or {}).get("tonic", 0)) % 12
    grid = rigid_grid_for(P.get("chords", []), tonic_pc=tonic)
    if grid is None:
        raise SystemExit("rigid grid deferred")
    rc, n_bars = apply_rigid_grid(P["chords"], grid, beats_per_bar=4,
                                 drop_before_grid=True)
    bars = [[] for _ in range(n_bars)]
    for c in rc:
        b = c.get("bar", 0)
        if 0 <= b < n_bars:
            lv = (c.get("lv") or {}).get("exact") or {}
            bars[b].append({"root": c["root"] % 12, "q": lv.get("q", ""),
                            "beat": c.get("beat", 0)})
    for bar in bars:
        bar.sort(key=lambda e: e["beat"])
    vocab = vocab_sections(bars, n_bars, tonic_pc=tonic, bpb=4)
    return vocab, form_string(vocab), list(grid), n_bars, bars


def beats_of_bar(bt: np.ndarray, edges: list[float], b: int) -> list[int]:
    """Beat indices whose time falls inside bar b. By TIME only — the `beat` label
    is unreliable (a 0.15-bar grid backoff means 0/34 This Love downbeats carry
    beat 0, docs/known_issues.md OPEN #3)."""
    if b + 1 >= len(edges):
        return []
    lo, hi = edges[b], edges[b + 1]
    return [i for i, t in enumerate(bt) if lo <= t < hi]


def score_qualities(chroma: np.ndarray, root: int) -> dict[str, float]:
    """Cosine of a 12-d treble chroma against each quality template at `root`."""
    out = {}
    v = chroma / (np.linalg.norm(chroma) or 1.0)
    for name, ivs in TEMPLATES.items():
        t = np.zeros(12)
        for iv in ivs:
            t[(root + iv) % 12] = 1.0
        out[name] = float(v @ (t / np.linalg.norm(t)))
    return out


def main():
    slug = "maroon_5_this_love"
    audio = REPO / "docs" / "audio" / f"{slug}.m4a"
    feat, bt = features(audio)
    treble = feat[:, 12:]
    print(f"observations: feat {feat.shape} (bass[:12] | treble[12:]), "
          f"{len(bt)} beats")

    vocab, form, edges, n_bars, bars = chart_vocabulary(slug)
    print(f"chart vocabulary: {form}")
    bar_s = float(np.median(np.diff(edges)))
    print(f"bar grid: {n_bars} bars @ {bar_s:.3f}s\n")

    # group occurrences per vocabulary item
    items: dict[str, list[dict]] = {}
    for s in vocab:
        items.setdefault(s["label"], []).append(s)

    print("=== 1. does the noise really fall as 1/sqrt(N)? ===")
    print("For each item slot with N occurrences: SD across occurrences (the noise")
    print("on ONE observation), and the SD of the mean, measured vs predicted.\n")
    print("  item  N   slots   SD(single obs)   SD(mean) measured   predicted SD/sqrtN   ratio")
    rows = []
    for lab, occ in sorted(items.items()):
        d = occ[0]["d_bars"]
        # slot = (bar within item, ordinal beat within bar)
        buckets: dict[tuple, list[np.ndarray]] = {}
        for o in occ:
            for k in range(max(1, (o["bar1"] - o["bar0"]) // d)):
                for bb in range(d):
                    b = o["bar0"] + k * d + bb
                    if b >= n_bars:
                        continue
                    for j, bi in enumerate(beats_of_bar(bt, edges, b)):
                        buckets.setdefault((bb, j), []).append(treble[bi])
        sds, sdm, n_used = [], [], []
        for key, vecs in buckets.items():
            if len(vecs) < 2:
                continue
            M = np.stack(vecs)
            sd = float(M.std(axis=0, ddof=1).mean())
            sds.append(sd)
            sdm.append(sd / np.sqrt(len(vecs)))
            n_used.append(len(vecs))
        if not sds:
            continue
        Nbar = float(np.mean(n_used))
        single = float(np.mean(sds))
        # empirical SD of the mean: split occurrences in half, compare the two means
        halves = []
        for key, vecs in buckets.items():
            if len(vecs) < 4:
                continue
            M = np.stack(vecs)
            h = len(vecs) // 2
            halves.append(np.abs(M[:h].mean(0) - M[h:].mean(0)).mean() / 2)
        emp = float(np.mean(halves)) if halves else float("nan")
        pred = single / np.sqrt(Nbar)
        rows.append((lab, Nbar, len(sds), single, emp, pred))
        print(f"  {lab:4s}  {Nbar:4.1f}  {len(sds):5d}   {single:12.4f}   "
              f"{emp:17.4f}   {pred:18.4f}   {emp/pred if pred else float('nan'):5.2f}")

    print("\n=== 2. the quality call at the verse downbeat (his sheet says G) ===")
    A = items.get("A")
    if A:
        d = A[0]["d_bars"]
        starts = [o["bar0"] + k * d for o in A
                  for k in range(max(1, (o["bar1"] - o["bar0"]) // d))]
        starts = [b for b in starts if b < n_bars]
        per_occ = []
        for b in starts:
            bs = beats_of_bar(bt, edges, b)
            if bs:
                per_occ.append(treble[bs[0]])
        print(f"  {len(per_occ)} occurrences of the verse downbeat "
              f"(bars {starts})\n")
        print("   occ  bar   " + "  ".join(f"{p:>5s}" for p in _PC))
        for b, v in zip(starts, per_occ):
            print(f"   {' ':3s} {b:3d}   " + "  ".join(f"{x:5.2f}" for x in v))
        mean = np.mean(np.stack(per_occ), axis=0)
        print(f"   MEAN      " + "  ".join(f"{x:5.2f}" for x in mean))
        print()
        root = 7   # G
        print("   quality template scores at root G:")
        print("        " + "  ".join(f"{k:>6s}" for k in TEMPLATES))
        singles = [score_qualities(v, root) for v in per_occ]
        for i, sc in enumerate(singles):
            print(f"   occ{i}  " + "  ".join(f"{sc[k]:6.3f}" for k in TEMPLATES))
        msc = score_qualities(mean, root)
        print(f"   MEAN  " + "  ".join(f"{msc[k]:6.3f}" for k in TEMPLATES))
        print()
        # how often does a SINGLE observation pick the right family vs the mean?
        best_single = [max(sc, key=sc.get) for sc in singles]
        print(f"   argmax per single observation: {best_single}")
        print(f"   argmax of the mean:            {max(msc, key=msc.get)}")
        sp = float(np.mean([sc["maj"] - sc["dom7"] for sc in singles]))
        print(f"\n   maj - dom7 margin: {sp:+.3f} averaged over single obs, "
              f"{msc['maj'] - msc['dom7']:+.3f} on the mean")
        sd_margin = float(np.std([sc["maj"] - sc["dom7"] for sc in singles], ddof=1))
        print(f"   SD of that margin across observations: {sd_margin:.3f}  -> "
              f"SD of the mean's margin ~ {sd_margin/np.sqrt(len(singles)):.3f}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for a in ax:
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=8)

    if A and per_occ:
        x = np.arange(12)
        for v in per_occ:
            ax[0].plot(x, v, color=SERIES[0], lw=1, alpha=0.32)
        ax[0].plot(x, mean, color=SERIES[1], lw=2.6, label="mean of "
                                                          f"{len(per_occ)}")
        ax[0].plot([], [], color=SERIES[0], lw=1, alpha=0.5, label="single pass")
        ax[0].set_xticks(x); ax[0].set_xticklabels(_PC, fontsize=7)
        ax[0].set_title("verse downbeat: every pass, and their mean",
                        fontsize=10, color=INK, loc="left")
        ax[0].legend(frameon=False, fontsize=8)
        for pc, nm in ((7, "G"), (11, "B"), (2, "D"), (5, "F=the 7th")):
            ax[0].annotate(nm, (pc, mean[pc]), textcoords="offset points",
                           xytext=(0, 6), fontsize=8, color=INK2, ha="center")

    if rows:
        labs = [r[0] for r in rows]
        xs = np.arange(len(rows))
        ax[1].bar(xs - 0.2, [r[3] for r in rows], 0.38, color=SERIES[0],
                  label="SD, one observation")
        ax[1].bar(xs + 0.2, [r[4] for r in rows], 0.38, color=SERIES[1],
                  label="SD of the mean (measured)")
        ax[1].plot(xs + 0.2, [r[5] for r in rows], "o", ms=8, color=SERIES[2],
                   mec=SURFACE, mew=2, label="predicted SD/√N", zorder=5)
        ax[1].set_xticks(xs); ax[1].set_xticklabels(labs)
        ax[1].set_title("noise on the observation vs the 1/√N prediction",
                        fontsize=10, color=INK, loc="left")
        ax[1].legend(frameon=False, fontsize=8)

    if A and per_occ:
        ks = list(TEMPLATES)
        xs = np.arange(len(ks))
        S = np.array([[sc[k] for k in ks] for sc in singles])
        ax[2].boxplot([S[:, i] for i in range(len(ks))], positions=xs,
                      widths=0.5, showfliers=False)
        ax[2].plot(xs, [msc[k] for k in ks], "o-", color=SERIES[1], lw=2, ms=9,
                   mec=SURFACE, mew=2, label="on the folded mean")
        ax[2].set_xticks(xs); ax[2].set_xticklabels(ks, fontsize=8)
        ax[2].set_title("quality call at root G: single passes vs the mean",
                        fontsize=10, color=INK, loc="left")
        ax[2].legend(frameon=False, fontsize=8)

    plt.tight_layout()
    out = REPO / "scratchpad" / "vocab_fold_sqrtn_this_love.png"
    plt.savefig(out, dpi=125, facecolor=SURFACE)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
