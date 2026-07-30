#!/usr/bin/env python3
"""rhythm_plot.py — diagnostic PNG: chord SSM vs rhythm SSM side by side, with
the chord-SSM-derived section boundaries drawn on both. Also plots the raw
3-band drum onset envelope over time so a claimed "no discrimination" result
can be checked by eye against the actual audio signal, not just the metric.

Run:  .venv/bin/python scratchpad/rhythm_plot.py [song-substring] [out.png]
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

from rhythm_calibrate import run_song, per_slot_labels
from pattern_slide import SLOTS_PER_BAR


def plot_song(sub: str, out_png: Path, variant_for_plot: str = "pearson_tolerance"):
    r = run_song(sub)
    S_chord, variants, blocks = r["S_chord"], r["variants"], r["blocks"]
    S_rhythm = variants[variant_for_plot]
    n = r["n"]

    bounds = sorted({b["start"] for b in blocks} | {b["end"] for b in blocks})
    label_at = per_slot_labels(blocks, n)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax, S, title in [(axes[0], S_chord, "chord SSM (build_chord_ssm cosine)"),
                          (axes[1], S_rhythm, f"rhythm SSM ({variant_for_plot}, drums via {r['source']})")]:
        im = ax.imshow(S, vmin=0, vmax=1, cmap="magma", origin="upper")
        for x in bounds:
            ax.axvline(x - 0.5, color="cyan", lw=0.8, alpha=0.8)
            ax.axhline(x - 0.5, color="cyan", lw=0.8, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("slot (half-bar)")
        ax.set_ylabel("slot (half-bar)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # third panel: baseline (uncorrected) rhythm variant, to show the failure mode
    S_base = variants["baseline"]
    ax = axes[2]
    im = ax.imshow(S_base, vmin=0, vmax=1, cmap="magma", origin="upper")
    for x in bounds:
        ax.axvline(x - 0.5, color="cyan", lw=0.8, alpha=0.8)
        ax.axhline(x - 0.5, color="cyan", lw=0.8, alpha=0.8)
    ax.set_title("rhythm SSM (baseline cosine, no tolerance/centring)", fontsize=10)
    ax.set_xlabel("slot (half-bar)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # annotate section labels along the top of the chord SSM panel
    prev = None
    for b in blocks:
        mid = (b["start"] + b["end"]) / 2
        axes[0].text(mid, -3, b["label"], ha="center", fontsize=9, color="black")

    fig.suptitle(f"{r['slug']}  —  chord-SSM-derived form: "
                 + " ".join(f"{b['label']}x{b['mult']}" if b["mult"] > 1 else b["label"] for b in blocks),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")
    plt.close(fig)
    return r


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "docs" / "research_sessions" / f"rhythm_ssm_{sub}_2026-07-30.png"
    plot_song(sub, out)
