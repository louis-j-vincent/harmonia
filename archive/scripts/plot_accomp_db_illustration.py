"""Illustrate one accompaniment-database record: structure, chord timeline, MIDI piano roll.

Usage: .venv/bin/python scripts/plot_accomp_db_illustration.py [--title "All The Things You Are"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pretty_midi
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SECTION_COLORS = {
    "A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52",
    "E": "#8172B2", "F": "#937860", "G": "#DA8BC3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="All The Things You Are")
    ap.add_argument("--db", default=REPO / "data" / "accomp_db" / "db.jsonl", type=Path)
    ap.add_argument("--out", default=REPO / "docs" / "plots" / "accomp_db_illustration.png", type=Path)
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.db)]
    rec = next(r for r in records if args.title.lower() in r["title"].lower())
    print(f"Illustrating: {rec['title']} ({rec['corpus']}) — {rec['form']}, "
          f"key {rec['key']}, groove {rec['groove']} @ {rec['tempo']} BPM")

    mid_path = REPO.parent.parent / rec["midi_path"] if not (REPO / rec["midi_path"]).exists() else REPO / rec["midi_path"]
    mid_path = REPO / rec["midi_path"]
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    duration = pm.get_end_time()

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 11), sharex=True,
        gridspec_kw={"height_ratios": [0.6, 0.6, 3, 2.2]},
    )
    ax_struct, ax_chords, ax_roll, ax_bass = axes

    sec_per_beat = 60.0 / rec["tempo"]
    sec_per_bar = sec_per_beat * rec["beats_per_bar"]

    # ── 1. structure strip ──────────────────────────────────────────────
    section_per_bar = rec["section_per_bar"]
    bar_idx = 0
    while bar_idx < len(section_per_bar):
        label = section_per_bar[bar_idx]
        start = bar_idx
        while bar_idx < len(section_per_bar) and section_per_bar[bar_idx] == label:
            bar_idx += 1
        t0, t1 = start * sec_per_bar, bar_idx * sec_per_bar
        color = SECTION_COLORS.get(label, "#999999")
        ax_struct.add_patch(Rectangle((t0, 0), t1 - t0, 1, facecolor=color, edgecolor="white"))
        ax_struct.text((t0 + t1) / 2, 0.5, f"{label} ({bar_idx - start} bars)",
                        ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax_struct.set_xlim(0, duration)
    ax_struct.set_ylim(0, 1)
    ax_struct.set_yticks([])
    ax_struct.set_title(f"{rec['title']}  —  form: {rec['form']}  |  key {rec['key']}  |  "
                         f"groove {rec['groove']} @ {rec['tempo']} BPM  |  {rec['corpus']}",
                         fontsize=13, fontweight="bold", loc="left")
    ax_struct.set_ylabel("structure", fontsize=9)

    # ── 2. chord timeline ────────────────────────────────────────────────
    events = rec["chord_timeline"]
    for i, ev in enumerate(events):
        t0 = ev["time"]
        t1 = events[i + 1]["time"] if i + 1 < len(events) else duration
        bar_label = rec["section_per_bar"][ev["bar"] - 1]
        color = SECTION_COLORS.get(bar_label, "#999999")
        ax_chords.add_patch(Rectangle((t0, 0), t1 - t0, 1, facecolor=color, alpha=0.25, edgecolor="#cccccc"))
        if t1 - t0 > 0.6:  # skip labels too narrow to read
            ax_chords.text((t0 + t1) / 2, 0.5, ev["ireal"], ha="center", va="center",
                            fontsize=8, rotation=0)
    for bar in range(rec["n_bars"] + 1):
        ax_chords.axvline(bar * sec_per_bar, color="#dddddd", lw=0.5, zorder=0)
    ax_chords.set_xlim(0, duration)
    ax_chords.set_ylim(0, 1)
    ax_chords.set_yticks([])
    ax_chords.set_ylabel("chords\n(iReal)", fontsize=9)

    # ── 3. full piano roll (all melodic tracks) ─────────────────────────
    track_colors = {"Bass": "#C44E52", "Chord": "#4C72B0", "Chord-Guitar": "#55A868",
                     "Melody": "#DA8BC3"}
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        color = track_colors.get(inst.name, "#937860")
        for n in inst.notes:
            ax_roll.add_patch(Rectangle((n.start, n.pitch - 0.4), n.end - n.start, 0.8,
                                         facecolor=color, edgecolor="none", alpha=0.85))
    for bar in range(rec["n_bars"] + 1):
        ax_roll.axvline(bar * sec_per_bar, color="#eeeeee", lw=0.5, zorder=0)
    all_pitches = [n.pitch for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    ax_roll.set_xlim(0, duration)
    ax_roll.set_ylim(min(all_pitches) - 3, max(all_pitches) + 3)
    ax_roll.set_ylabel("pitch (MIDI)\nall tracks", fontsize=9)
    handles = [plt.Line2D([0], [0], color=c, lw=6, label=n) for n, c in track_colors.items()
               if any(i.name == n for i in pm.instruments)]
    ax_roll.legend(handles=handles, loc="upper right", fontsize=8, ncol=len(handles))

    # ── 4. bass stem only, zoomed pitch range, root markers overlaid ────
    bass_notes = rec["bass_notes"]
    for pitch, start, end, vel in bass_notes:
        ax_bass.add_patch(Rectangle((start, pitch - 0.4), end - start, 0.8,
                                     facecolor="#C44E52", edgecolor="none", alpha=0.9))
    # overlay the chart's expected chord root as a dashed line per bar
    from harmonia.data.ireal_corpus import chord_root_pc
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, ev in enumerate(events):
        t0 = ev["time"]
        t1 = events[i + 1]["time"] if i + 1 < len(events) else duration
        pc = chord_root_pc(ev["mma"])
        if pc is None:
            continue
        octave_pitch = 36 + pc  # display root at a fixed low octave for readability
        ax_bass.plot([t0, t1], [octave_pitch, octave_pitch], color="black", lw=1.5,
                     linestyle="--", alpha=0.6, zorder=5)
    for bar in range(rec["n_bars"] + 1):
        ax_bass.axvline(bar * sec_per_bar, color="#eeeeee", lw=0.5, zorder=0)
    if bass_notes:
        bp = [n[0] for n in bass_notes]
        ax_bass.set_ylim(min(bp) - 3, max(max(bp), 48) + 3)
    ax_bass.set_xlim(0, duration)
    ax_bass.set_ylabel("bass stem\n(dashed = chart root)", fontsize=9)
    ax_bass.set_xlabel("time (s)", fontsize=10)

    agreement = rec.get("bass_root_agreement")
    fig.text(0.99, 0.01, f"bass-root agreement (QA): {agreement:.0%}" if agreement else "",
              ha="right", fontsize=9, style="italic")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
