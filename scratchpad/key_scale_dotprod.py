"""Scale-mask dot products on This Love — which flavour of C minor, when?

Premise check for the harmonic-key brick (branch feat/harmonic-key):
given per-beat NNLS chroma, score binary pitch-class masks for each C-minor
scale variant (natural / harmonic / melodic / dorian) by the share of chroma
mass that falls inside the mask.  If the harmonic-vs-natural alternation the
ear hears in This Love is recoverable this way, the discriminating pitch
classes (B vs Bb, A vs Ab) should flip with the G7 bars.

Outputs
-------
scratchpad/key_scale_dotprod_this_love.png   3-panel diagnostic plot
stdout                                        summary numbers

Usage:  .venv/bin/python scratchpad/key_scale_dotprod.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import nnls_features as nf  # noqa: E402
from harmonia.models.beat_grid import bestfit_beat_period  # noqa: E402
from harmonia.theory.key_profiles import infer_key  # noqa: E402

SLUG = "maroon_5_this_love"
AUDIO = REPO / "docs" / "audio" / f"{SLUG}.m4a"
DURATION_S = 205.253  # ffprobe of the m4a; matches nnls cache times[-1]=205.218

# C-first pitch-class indices, tonic C = 0
SCALES = {
    "natural minor": {0, 2, 3, 5, 7, 8, 10},
    "harmonic minor": {0, 2, 3, 5, 7, 8, 11},
    "melodic minor": {0, 2, 3, 5, 7, 9, 11},
    "dorian": {0, 2, 3, 5, 7, 9, 10},
}
# pcs outside every C-minor variant: Db, E, F#
OUT_ALL = {1, 4, 6}

PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# dataviz reference palette (light mode)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SCALE_COLORS = {
    "natural minor": C_BLUE,
    "harmonic minor": C_ORANGE,
    "melodic minor": C_AQUA,
    "dorian": C_YELLOW,
}


def beat_grid() -> np.ndarray:
    raw = np.asarray(
        json.load(open(REPO / "data/cache/raw_beat_times_v2" / f"{SLUG}.json")), float
    )
    tempo_bpm = 60.0 / float(np.median(np.diff(raw)))
    period = bestfit_beat_period(raw, 60.0 / tempo_bpm)
    ang = 2 * np.pi * (raw % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    return np.unique(
        np.concatenate([[0.0], np.arange(phase, DURATION_S + period, period), [DURATION_S]])
    )


def load_chart_chords() -> list[dict]:
    txt = (REPO / "docs/plots" / f"inferred_{SLUG}.html").read_text(errors="ignore")
    m = re.search(r"const\s+P\s*=\s*", txt)
    i = txt.index("{", m.end())
    depth, j, instr, esc = 0, i, False, False
    while j < len(txt):
        c = txt[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        elif c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return json.loads(txt[i : j + 1])["chords"]


def chord_name(ch: dict) -> str:
    if ch.get("nc"):
        return "NC"
    q = ch["lv"]["exact"]["q"]
    name = PC_FLAT[ch["root"]] + q
    if ch.get("bass", -1) >= 0 and ch["bass"] != ch["root"]:
        name += "/" + PC_FLAT[ch["bass"]]
    return name


def main() -> None:
    arr, times = nf.extract_bothchroma(AUDIO)
    bt = beat_grid()
    feat = nf.pool_beats(arr, times, bt)  # (n_beats, 24), C-first, L2 per half
    treble = feat[:, 12:]
    key = infer_key(treble.sum(0))
    print(f"key check: {key.key_name}  conf={key.confidence:.4f}")

    chords = load_chart_chords()

    # Anchor 4-beat bars so bar lines coincide with chord onsets from the chart.
    onsets = np.array([c["t0"] for c in chords if not c.get("nc")])
    best_off, best_hits = 0, -1
    for off in range(4):
        starts = bt[off::4]
        hits = sum(np.abs(starts - t).min() < 0.12 for t in onsets)
        if hits > best_hits:
            best_off, best_hits = off, hits
    print(f"bar anchor: beat offset {best_off} ({best_hits}/{len(onsets)} onsets on bar lines)")

    n_beats = treble.shape[0]
    bar_starts = np.arange(best_off, n_beats - 3, 4)
    bar_chroma = np.stack([treble[s : s + 4].mean(0) for s in bar_starts])  # (n_bars, 12)
    bar_t0 = bt[bar_starts]
    n_bars = len(bar_starts)
    print(f"{n_beats} beats -> {n_bars} bars")

    mass = bar_chroma.sum(1)  # total chroma mass per bar
    mass[mass <= 1e-9] = np.nan  # silent bars (edges) -> NaN, nan-aware below
    frac = {
        name: bar_chroma[:, sorted(pcs)].sum(1) / mass for name, pcs in SCALES.items()
    }
    out_all = bar_chroma[:, sorted(OUT_ALL)].sum(1) / mass
    pc_frac = bar_chroma / mass[:, None]  # per-pc share of bar mass

    # summary numbers
    print("\nmean in-scale share per variant (per-bar):")
    for name in SCALES:
        print(f"  {name:15s} {np.nanmean(frac[name]):.3f}")
    print(f"  {'outside all':15s} {np.nanmean(out_all):.3f}")
    valid = ~np.isnan(mass)
    winners = np.array(list(frac.values()))[:, valid].argmax(0)
    names = list(SCALES)
    print("\nbars won (argmax of the 4 masks):")
    for i, name in enumerate(names):
        print(f"  {name:15s} {(winners == i).sum():3d} / {valid.sum()}")
    b_gt_bb = (pc_frac[valid, 11] > pc_frac[valid, 10]).mean()
    a_gt_ab = (pc_frac[valid, 9] > pc_frac[valid, 8]).mean()
    print(f"\nbars with B > Bb: {b_gt_bb:.0%}   bars with A > Ab: {a_gt_ab:.0%}")

    # cosine variant (Louis's exact formulation) vs the L1 share, for the record
    for name, pcs in SCALES.items():
        m = np.zeros(12)
        m[sorted(pcs)] = 1.0
        cos = bar_chroma[valid] @ m / (
            np.linalg.norm(bar_chroma[valid], axis=1) * np.linalg.norm(m)
        )
        r = np.corrcoef(cos, frac[name][valid])[0, 1]
        print(f"corr(cosine, L1-share) {name:15s} {r:.3f}")

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(20, 9.5), sharex=True, height_ratios=[1.4, 1, 1]
    )
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(n_bars)

    ax = axes[0]
    for name in SCALES:
        ax.plot(x, frac[name], color=SCALE_COLORS[name], lw=2, label=name)
        ax.annotate(
            name, (x[-1], frac[name][-1]), xytext=(6, 0), textcoords="offset points",
            color=SCALE_COLORS[name], fontsize=9, va="center",
        )
    ax.plot(x, out_all, color=MUTED, lw=1.5, ls="--", label="outside all (Db+E+Gb)")
    ax.set_ylabel("share of chroma mass in scale", color=INK2)
    ax.legend(loc="lower right", ncol=5, frameon=False, fontsize=9)
    ax.set_title(
        "This Love — C-minor scale variants vs per-bar NNLS treble chroma",
        color=INK, loc="left", fontsize=13,
    )

    ax = axes[1]
    ax.plot(x, pc_frac[:, 10], color=C_BLUE, lw=2, label="Bb (natural 7th)")
    ax.plot(x, pc_frac[:, 11], color=C_ORANGE, lw=2, label="B (leading tone)")
    ax.set_ylabel("pc share", color=INK2)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
    ax.set_title("7th degree: Bb vs B", color=INK2, loc="left", fontsize=11)

    ax = axes[2]
    ax.plot(x, pc_frac[:, 8], color=C_BLUE, lw=2, label="Ab (b6)")
    ax.plot(x, pc_frac[:, 9], color=C_ORANGE, lw=2, label="A (natural 6)")
    ax.set_ylabel("pc share", color=INK2)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
    ax.set_title("6th degree: Ab vs A", color=INK2, loc="left", fontsize=11)

    # chord labels from the baked chart, one row under the bottom axis
    labels = [""] * n_bars
    for ch in chords:
        b = int(np.searchsorted(bar_t0, ch["t0"] + 0.06, side="right") - 1)
        if 0 <= b < n_bars and not labels[b]:
            labels[b] = chord_name(ch)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=90, fontsize=6.5, color=INK2)
    axes[2].set_xlabel("bar (label = chart's chord at bar start)", color=INK2)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        for b in range(0, n_bars, 4):
            ax.axvline(b, color=GRID, lw=0.6)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=INK2)
        ax.margins(x=0.01)

    out = REPO / "scratchpad" / "key_scale_dotprod_this_love.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
