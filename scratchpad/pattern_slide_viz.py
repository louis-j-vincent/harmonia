#!/usr/bin/env python3
"""pattern_slide_viz.py — show the SLIDE-ACROSS-X result next to the
SLIDE-ALONG-THE-DIAGONAL (Foote) result, on the same half-bar SSM.

Four stacked panels, one shared x axis = half-bar slot:
  1. the half-bar SSM, with each learned pattern's row-band and the blocks;
  2. slide ACROSS X, DIAGONAL reading  (mean of the slid stripe's diagonal)
     -> the signal the segmenter actually decides on;
  3. slide ACROSS X, FULL-SQUARE reading (Louis's literal dot product)
     -> why it does not work: it never comes down;
  4. slide ALONG THE DIAGONAL (Foote checkerboard novelty) -> last session's
     method, for comparison. Peaks = boundaries, not occurrences.

Run: .venv/bin/python scratchpad/pattern_slide_viz.py [song-substring]
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

from harmonia.models.section_structure import build_chord_ssm
from section_merge_declined import _load_payload
import pattern_slide as ps
from ssm_block_segment import checkerboard_kernel

# validated categorical palette (scripts/validate_palette.js: ALL CHECKS PASS)
SERIES = ["#3b6fd4", "#d97a1f", "#1f8a6b", "#8a4fbf"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"


def foote_novelty(S: np.ndarray, M: int) -> np.ndarray:
    """Slide a checkerboard kernel ALONG THE DIAGONAL (both indices move together)."""
    n = S.shape[0]
    K = checkerboard_kernel(M)
    Sp = np.pad(S, M, mode="edge")
    nov = np.array([float((Sp[i:i + 2 * M, i:i + 2 * M] * K).sum()) for i in range(n)])
    nov = np.clip(nov, 0, None)
    return nov / (nov.max() + 1e-9)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    seq, names, n_bars, bar_sec = ps.rigid_slots(P)
    S = ps.chord_ssm(names)
    n = S.shape[0]

    quiet = []
    blocks = ps.segment(S, seq, names, log=quiet.append)

    # one canonical pattern per letter (the first block that introduced it)
    pats: list[dict] = []
    for b in blocks:
        if b["label"] not in [p["label"] for p in pats]:
            pats.append(b)

    # ── the numbers, printed ──────────────────────────────────────────────────
    print(f"\n=== {slug} — pattern slide, {n_bars} bars @ {bar_sec:.3f}s, "
          f"{n} half-bar slots ===\n")
    for i, p in enumerate(pats):
        print(f"pattern {p['label']}  ({p['d']//2}-bar loop learned at bar "
              f"{p['row0']//2}): {' '.join(p['pattern'])}")
    print(f"\nform: " + " ".join(f"{b['label']}x{b['mult']}" if b["mult"] > 1
                                 else b["label"] for b in blocks))
    print("\nSLIDE ACROSS X — diagonal reading, sampled every bar "
          f"(>= {ps.REP_STRICT} = the pattern is here):")
    for p in pats:
        print(f"\n  pattern {p['label']}:")
        for b0 in range(0, n_bars, 20):
            bars = range(b0, min(b0 + 20, n_bars))
            print("    bar " + " ".join(f"{b:>4d}" for b in bars))
            print("        " + " ".join(
                ("   ." if np.isnan(p["corr"][b * 2]) else f"{p['corr'][b*2]:4.2f}")
                for b in bars))

    # ── the figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                           gridspec_kw={"height_ratios": [2.6, 1, 1, 1]},
                           facecolor=SURFACE)
    for a in ax:
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=8)

    # panel 1: the SSM
    ax[0].imshow(S, cmap="Greys", origin="upper", interpolation="nearest",
                 extent=(0, n, n, 0), aspect="auto", vmin=0, vmax=1)
    for b in blocks:
        ax[0].axvline(b["start"], color=INK, lw=0.8, alpha=0.45)
        ax[0].text((b["start"] + b["end"]) / 2, -5,
                   f"{b['label']}" + (f"×{b['mult']}" if b["mult"] > 1 else ""),
                   ha="center", va="bottom", fontsize=11, fontweight="bold",
                   color=SERIES[[p["label"] for p in pats].index(b["label"])])
    for i, p in enumerate(pats):
        ax[0].add_patch(plt.Rectangle((0, p["row0"]), n, p["d"], fill=False,
                                      edgecolor=SERIES[i], lw=1.6))
        ax[0].text(n + 1, p["row0"] + p["d"] / 2, p["label"], color=SERIES[i],
                   fontsize=10, fontweight="bold", va="center")
    ax[0].set_ylabel("slot (row)", color=INK2, fontsize=9)
    ax[0].set_title(f"{slug} — half-bar SSM.  The boxed row-bands are the patterns; "
                    f"each is slid horizontally across the whole width.",
                    fontsize=10, color=INK, loc="left")

    # panel 2: slide across x, DIAGONAL reading
    for i, p in enumerate(pats):
        ax[1].plot(p["corr"], color=SERIES[i], lw=2.0, label=f"pattern {p['label']}")
        grid = [t for t in range(p["row0"], n, p["d"]) if not np.isnan(p["corr"][t])]
        hits = [t for t in grid if p["corr"][t] >= ps.REP_STRICT]
        ax[1].plot(hits, [p["corr"][t] for t in hits], "o", ms=8,
                   color=SERIES[i], mec=SURFACE, mew=2, zorder=5)
    ax[1].axhline(ps.REP_STRICT, color=INK2, lw=1, ls=(0, (4, 3)))
    ax[1].text(n - 1, ps.REP_STRICT + 0.02, f"repeat threshold {ps.REP_STRICT}",
               ha="right", fontsize=8, color=INK2)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_ylabel("diag match", color=INK2, fontsize=9)
    ax[1].set_title("SLIDE ACROSS X — diagonal of the slid stripe.  Dots = the pattern "
                    "is here. Clean separation; this drives every decision.",
                    fontsize=10, color=INK, loc="left")
    ax[1].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    # panel 3: slide across x, FULL SQUARE reading
    for i, p in enumerate(pats):
        ax[2].plot(p["square"], color=SERIES[i], lw=2.0, label=f"pattern {p['label']}")
    ax[2].axhline(ps.REP_STRICT, color=INK2, lw=1, ls=(0, (4, 3)))
    ax[2].set_ylim(0, 1.08)
    ax[2].set_ylabel("square dot", color=INK2, fontsize=9)
    ax[2].set_title("SLIDE ACROSS X — full d×d dot product (the literal version).  "
                    "Never comes down: it measures texture, not chords. Unusable.",
                    fontsize=10, color=INK, loc="left")
    ax[2].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    # panel 4: slide ALONG THE DIAGONAL (Foote)
    nov = foote_novelty(S, 8)
    ax[3].fill_between(range(n), nov, color=SERIES[3], alpha=0.18)
    ax[3].plot(nov, color=SERIES[3], lw=2.0)
    for b in blocks:
        ax[3].axvline(b["start"], color=INK, lw=0.8, alpha=0.35)
    ax[3].set_ylim(0, 1.08)
    ax[3].set_ylabel("novelty", color=INK2, fontsize=9)
    ax[3].set_xlabel("half-bar slot   (bar = slot / 2)", color=INK2, fontsize=9)
    ax[3].set_title("SLIDE ALONG THE DIAGONAL — Foote checkerboard novelty (last "
                    "session).  Peaks mark WHERE it changes, never WHICH section it is.",
                    fontsize=10, color=INK, loc="left")

    ax[3].set_xlim(0, n)
    ax[3].set_xticks(range(0, n + 1, 16))
    ax[3].set_xticklabels([f"{t}\nbar {t//2}" for t in range(0, n + 1, 16)], fontsize=8)
    plt.tight_layout()
    out = REPO / "scratchpad" / f"pattern_slide_x_vs_diag_{slug}.png"
    plt.savefig(out, dpi=125, facecolor=SURFACE)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
