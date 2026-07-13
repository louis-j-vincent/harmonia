"""Diagnostic plot: librosa vs madmom beat-tracker tempo across docs/audio.

Reads docs/tempo_comparison_madmom.json, renders grouped bars (librosa, madmom)
per song with the reference BPM as a target marker, plus half/double guide lines
around each song's librosa reading so octave jumps are visible at a glance.

Output: docs/plots/tempo_comparison_madmom.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("docs/tempo_comparison_madmom.json")
OUT = Path("docs/plots/tempo_comparison_madmom.png")

C_LIB = "#2a78d6"   # blue   — librosa
C_MAD = "#eb6834"   # orange — madmom
C_REF = "#0b0b0b"   # ink    — reference target
C_GUIDE = "#c9c8c4"  # recessive half/double guide
INK = "#0b0b0b"
MUTED = "#52514e"


def short(name: str) -> str:
    m = {
        "blue_bossa_150bpm_backing_track": "blue bossa\n(150 backing)",
        "the_beatles_the_beatles_let_it_be_official_music_video_remas": "let it be",
        "muppets_kermit_its_not_easy_being_green_original": "kermit\n(being green)",
        "adele_hello_official_music_video": "adele hello",
        "nina_simone_feeling_good_lyric_video": "nina simone\nfeeling good",
    }
    return m.get(name, name.replace("_", " "))


def main() -> None:
    rows = json.loads(DATA.read_text())
    # Sort by |librosa↔madmom octave| descending — worst disagreements first.
    rows.sort(key=lambda r: abs(r["librosa_vs_madmom_oct"]), reverse=True)

    songs = [short(r["song"]) for r in rows]
    lib = np.array([r["librosa_bpm"] for r in rows])
    mad = np.array([r["madmom_bpm"] for r in rows])
    ref = [r["ref_bpm"] for r in rows]

    x = np.arange(len(rows))
    w = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # half/double guides off the librosa reading (recessive) — an octave error
    # is a bar landing on one of these lines.
    for xi, l in zip(x, lib):
        for factor in (0.5, 2.0):
            ax.plot([xi - 0.44, xi + 0.44], [l * factor, l * factor],
                    color=C_GUIDE, lw=1, zorder=1)

    b1 = ax.bar(x - w / 2, lib, w, color=C_LIB, label="librosa", zorder=3)
    b2 = ax.bar(x + w / 2, mad, w, color=C_MAD, label="madmom", zorder=3)

    # reference target markers (only where known)
    ref_plotted = False
    for xi, rv in zip(x, ref):
        if rv is not None:
            ax.plot([xi - 0.46, xi + 0.46], [rv, rv], color=C_REF, lw=2.2,
                    zorder=4, solid_capstyle="round",
                    label="reference" if not ref_plotted else None)
            ref_plotted = True

    # direct value labels
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 2, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(songs, fontsize=8.5, color=INK)
    ax.set_ylabel("tempo (BPM)", fontsize=10, color=INK)
    ax.set_title(
        "Tempo detection: librosa vs madmom  ·  guide lines = ½× / 2× octaves",
        fontsize=13, color=INK, pad=12, loc="left")
    ax.set_ylim(0, max(lib.max(), mad.max()) * 1.18)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d3")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#ececea", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(0.01, 0.005,
             "Sorted by librosa↔madmom octave disagreement. Reference BPM: "
             "blue_bossa backing track is exact (filename); others approximate "
             "(ballad/swing convention) — a factor-of-2 gap is unambiguous even so.",
             fontsize=7.5, color=MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
