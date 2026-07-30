"""Lead-sheet view of the per-chord colour decode (This Love).

Lays the baked chart's chords out as bars in rows (8 bars per row, like a
chart) with each chord's background = its decoded scale colour from the
sticky HMM in ``colour_hmm_this_love.py`` (argmax bass mode).  Pale fill =
state held on stickiness (no real contrast-pc evidence in that chord).

Output: scratchpad/colour_chart_this_love.png

Usage:  .venv/bin/python scratchpad/colour_chart_this_love.py
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
    AUDIO, GAIN, HELD_W, STATE_COLORS, challenge_chords, chord_name,
    decode_folded, inflections, load_chart_chords,
)

CRIT = "#d03b3b"  # status colour for challenged chords
from key_scale_dotprod import beat_grid  # noqa: E402
from harmonia.models import nnls_features as nf  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BARS_PER_ROW = 8
BAR_ANCHOR_OFFSET = 2  # from key_scale_dotprod: 74/117 chord onsets on bar lines


def main() -> None:
    arr, times = nf.extract_bothchroma(AUDIO)
    chords = load_chart_chords()
    path, rows, ev6, ev7, chromas, _slotmap, form = decode_folded(arr, times, chords)
    flag_list = inflections(path, rows, ev6, ev7)
    flags = dict(flag_list)
    audits = {i: (best, kind)
              for i, best, _, _, kind in
              challenge_chords(chords, chromas, path, flag_list)}
    w_tot = np.array([GAIN * (t6 + t7) for (t6, _), (t7, _) in zip(ev6, ev7)])
    held = w_tot < HELD_W
    print(f"form: {form}")

    bt = beat_grid()
    bar_t0 = bt[BAR_ANCHOR_OFFSET::4]
    n_bars = len(bar_t0) - 1

    def to_pos(t: float) -> float:
        """time -> fractional bar position (bar index + fraction within bar).

        Chart times are rounded to 10 ms and the bar grid is reconstructed
        independently, so on-the-barline chords land ~10 ms early; snap
        positions within 10% of a bar to the bar line.
        """
        b = int(np.clip(np.searchsorted(bar_t0, t, side="right") - 1, 0, n_bars - 1))
        frac = (t - bar_t0[b]) / (bar_t0[b + 1] - bar_t0[b])
        p = b + float(np.clip(frac, 0.0, 1.0))
        return float(round(p)) if abs(p - round(p)) < 0.1 else p

    n_rows = int(np.ceil(n_bars / BARS_PER_ROW))
    fig, ax = plt.subplots(figsize=(17, 0.75 * n_rows + 1.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, ch in enumerate(chords):
        p0, p1 = to_pos(ch["t0"]), to_pos(ch["t1"])
        if p1 <= p0:
            p1 = p0 + 0.05
        colour = STATE_COLORS[path[i]]
        alpha = 0.18 if held[i] else 0.55
        # split the chord's span at row boundaries
        seg0 = p0
        while seg0 < p1 - 1e-9:
            row = int(seg0 // BARS_PER_ROW)
            row_end = (row + 1) * BARS_PER_ROW
            seg1 = min(p1, row_end)
            x0, x1 = seg0 - row * BARS_PER_ROW, seg1 - row * BARS_PER_ROW
            ax.add_patch(
                plt.Rectangle((x0, -row - 0.42), x1 - x0, 0.84,
                              color=colour, alpha=alpha, linewidth=0)
            )
            if i in flags:  # borrowed-colour inflection: thick underline
                ax.add_patch(
                    plt.Rectangle((x0, -row - 0.42), x1 - x0, 0.12,
                                  color=STATE_COLORS[flags[i]], linewidth=0)
                )
            seg0 = seg1
        row0 = int(p0 // BARS_PER_ROW)
        x_lab = p0 - row0 * BARS_PER_ROW + 0.04
        ax.text(x_lab, -row0 + 0.24, chord_name(ch),
                fontsize=8.5, color=INK, va="top", ha="left")
        if i in audits:  # colour prior challenges the chord itself
            best, kind = audits[i]
            solid = kind == "challenge"
            ax.text(x_lab, -row0 - 0.13, f"{'→' if solid else '?'}{best}",
                    fontsize=8, color=CRIT, va="top", ha="left",
                    fontweight="bold" if solid else "normal")
            ax.add_patch(
                plt.Rectangle((p0 - row0 * BARS_PER_ROW, -row0 - 0.42),
                              p1 - p0, 0.84, fill=False, edgecolor=CRIT,
                              lw=1.8, linestyle="-" if solid else (0, (3, 2)))
            )

    for row in range(n_rows):
        for b in range(BARS_PER_ROW + 1):
            ax.plot([b, b], [-row - 0.42, -row + 0.42], color=INK2, lw=0.8)
        bar0 = row * BARS_PER_ROW
        if bar0 < n_bars:
            ax.text(-0.15, -row, f"bar {bar0 + 1}\n{bar_t0[bar0]:.0f}s",
                    fontsize=7.5, color=MUTED, ha="right", va="center")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55)
               for c in STATE_COLORS.values()]
    ax.legend(handles, list(STATE_COLORS), loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=10)
    ax.set_title(
        "This Love — prevailing scale colour, v4 structure-folded (evidence "
        "pooled across section-slot occurrences); pale = held; underline = "
        "borrowed colour; red frame = chord challenged by the colour prior "
        "(solid →X proposal, dashed ?X suspect)",
        color=INK, loc="left", fontsize=12,
    )
    ax.set_xlim(-1.2, BARS_PER_ROW + 0.1)
    ax.set_ylim(-n_rows + 0.4, 0.75)
    ax.axis("off")

    out = REPO / "scratchpad" / "colour_chart_this_love.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
