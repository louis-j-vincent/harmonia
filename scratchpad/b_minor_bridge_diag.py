"""Why does the chart hear B- at ~127s where Louis's ear says G? (bridge diag)

Prints per-chord treble/bass top pitch classes around the bridge and renders
a frame-level chroma heatmap (treble + bass halves) for 118-138s with chord
boundaries, so the mechanism is visible.

Usage:  .venv/bin/python scratchpad/b_minor_bridge_diag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from colour_hmm_this_love import (  # noqa: E402
    AUDIO, PC_FLAT, ROLL, chord_name, load_chart_chords,
)
from harmonia.models import nnls_features as nf  # noqa: E402

INK2, GRID, SURFACE = "#52514e", "#e1e0d9", "#fcfcfb"
T0, T1 = 118.0, 138.0


def main() -> None:
    arr, times = nf.extract_bothchroma(AUDIO)
    chords = load_chart_chords()

    print("chords in the bridge window:")
    for i, ch in enumerate(chords):
        if ch["t1"] < T0 or ch["t0"] > T1:
            continue
        sel = (times >= ch["t0"]) & (times < ch["t1"])
        seg = arr[sel].mean(0)
        bass = np.roll(seg[:12], ROLL)
        treb = np.roll(seg[12:], ROLL)
        bass = bass / max(bass.sum(), 1e-9)
        treb = treb / max(treb.sum(), 1e-9)
        tt = ", ".join(f"{PC_FLAT[p]} {treb[p]:.2f}" for p in np.argsort(treb)[::-1][:4])
        bb = ", ".join(f"{PC_FLAT[p]} {bass[p]:.2f}" for p in np.argsort(bass)[::-1][:4])
        print(f"  #{i:3d} {chord_name(ch):6s} {ch['t0']:6.1f}-{ch['t1']:6.1f}s  "
              f"treble[{tt}]  bass[{bb}]")

    lab = REPO / "data/cache/musx_infer/maroon_5_this_love_submission.lab"
    if lab.exists():
        print("\nmusx second opinion (window):")
        for line in lab.read_text().splitlines():
            p = line.split()
            if len(p) >= 3 and float(p[1]) > T0 and float(p[0]) < T1:
                print(f"  {float(p[0]):7.2f} {float(p[1]):7.2f}  {p[2]}")

    sel = (times >= T0) & (times <= T1)
    tw = times[sel]
    treb = np.stack([np.roll(f[12:], ROLL) for f in arr[sel]]).T  # (12, T)
    bass = np.stack([np.roll(f[:12], ROLL) for f in arr[sel]]).T

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, M, name in ((axes[0], treb, "treble half"), (axes[1], bass, "bass half")):
        ax.imshow(M, aspect="auto", origin="lower", cmap="Blues",
                  extent=[tw[0], tw[-1], -0.5, 11.5])
        ax.set_yticks(range(12))
        ax.set_yticklabels(PC_FLAT, fontsize=8, color=INK2)
        ax.set_title(f"NNLS {name} — bridge window", color=INK2, loc="left",
                     fontsize=11)
        for ch in chords:
            if T0 <= ch["t0"] <= T1:
                ax.axvline(ch["t0"], color="#e34948", lw=0.8, alpha=0.6)
                ax.text(ch["t0"] + 0.05, 11.2, chord_name(ch), fontsize=7,
                        color="#e34948", rotation=90, va="top")
    axes[1].set_xlabel("time (s)", color=INK2)
    out = REPO / "scratchpad" / "b_minor_bridge_diag.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
