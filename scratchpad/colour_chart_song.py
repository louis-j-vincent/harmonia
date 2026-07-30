"""Lead-sheet view of the per-chord colour decode — generic version.

Same rendering as ``colour_chart_this_love.py`` (frozen while the main
session edits it), but slug/tonic/bpb come from argv + the baked payload,
via ``colour_hmm_song``. The bar anchor offset is computed per song
(most chord onsets on bar lines), not hardcoded.

Output: scratchpad/colour_chart_<slug>.png

Usage:  .venv/bin/python scratchpad/colour_chart_song.py <slug> [--tonic N]
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

from colour_hmm_song import (  # noqa: E402
    GAIN, HELD_W, STATE_COLORS, bar_anchor_offset, beat_grid, challenge_chords,
    chord_name, decode_folded, inflections, load_song,
)
from harmonia.models import nnls_features as nf  # noqa: E402

CRIT = "#d03b3b"  # status colour for challenged chords
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BARS_PER_ROW = 8


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("usage: colour_chart_song.py <slug> [--tonic N]")
    slug = args[0]
    tonic_override = None
    if "--tonic" in sys.argv:
        tonic_override = int(sys.argv[sys.argv.index("--tonic") + 1])

    song = load_song(slug, tonic_override)
    audio = REPO / "docs" / "audio" / f"{slug}.m4a"
    arr, times = nf.extract_bothchroma(audio)
    chords = song.chords
    path, rows, ev6, ev7, chromas, _slotmap, form = decode_folded(
        arr, times, chords, song)
    flag_list = inflections(path, rows, ev6, ev7)
    flags = dict(flag_list)
    audits = {i: (best, kind)
              for i, best, _, _, kind in
              challenge_chords(chords, chromas, path, flag_list, song)}
    w_tot = np.array([GAIN * (t6 + t7) for (t6, _), (t7, _) in zip(ev6, ev7)])
    held = w_tot < HELD_W
    print(f"form: {form}")

    bt = beat_grid(slug, float(times[-1]))
    bpb = song.bpb
    off, hits = bar_anchor_offset(bt, chords, bpb)
    n_on = sum(1 for c in chords if not c.get("nc"))
    print(f"bar anchor: beat offset {off} ({hits}/{n_on} onsets on bar lines)")
    bar_t0 = bt[off::bpb]
    n_bars = len(bar_t0) - 1

    def to_pos(t: float) -> float:
        """time -> fractional bar position; snap within 10% of a bar line."""
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
            if i in flags:
                ax.add_patch(
                    plt.Rectangle((x0, -row - 0.42), x1 - x0, 0.12,
                                  color=STATE_COLORS[flags[i]], linewidth=0)
                )
            seg0 = seg1
        row0 = int(p0 // BARS_PER_ROW)
        x_lab = p0 - row0 * BARS_PER_ROW + 0.04
        ax.text(x_lab, -row0 + 0.24, chord_name(ch, song),
                fontsize=8.5, color=INK, va="top", ha="left")
        if i in audits:
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
    tname = song.name_pc(song.tonic)
    ax.set_title(
        f"{slug} — prevailing {tname}-minor scale colour, v4.1 structure-folded; "
        "pale = held; underline = borrowed colour; red frame = chord challenged "
        "by the colour prior (solid →X proposal, dashed ?X suspect)",
        color=INK, loc="left", fontsize=12,
    )
    ax.set_xlim(-1.2, BARS_PER_ROW + 0.1)
    ax.set_ylim(-n_rows + 0.4, 0.75)
    ax.axis("off")

    suffix = f"_tonic{song.tonic}" if song.tonic_overridden else ""
    out = REPO / "scratchpad" / f"colour_chart_{slug}{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
