"""
Zoomed beat-level note-probability heatmap with ground-truth chord labels in
an aligned panel below, sharing the same beat-index/time x-axis. Built to
visually audit whether the underlying pitch-class structure is legible
enough to hand-derive scale/transition rules from — see docs/known_issues.md
#1, and the BASIC_PITCH_FRAME_RATE fix (frame-to-beat alignment was silently
corrupted by a 2x frame-rate bug prior to this).

Usage:
    .venv/bin/python scripts/plot_note_probs_vs_gt.py --song 001
    .venv/bin/python scripts/plot_note_probs_vs_gt.py --song 001 --low C2 --high C5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

MIDI_START = 21  # A0
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots"


def midi_label(key_idx: int) -> str:
    midi = MIDI_START + key_idx
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def parse_note_name(s: str) -> int:
    """'C2' -> MIDI number (octave convention: C4 = MIDI 60)."""
    m = re.match(r"([A-G][#b]?)(-?\d+)", s)
    if not m:
        raise ValueError(f"Unparseable note name: {s!r}")
    pc = _NOTE_TO_PC[m.group(1)]
    octave = int(m.group(2))
    return (octave + 1) * 12 + pc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default="001")
    parser.add_argument("--low", default="C2")
    parser.add_argument("--high", default="C5")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor

    song_id = args.song
    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    if not wav.exists():
        wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v000_prog0.wav"
    if not wav.exists():
        print(f"No render found for song {song_id}")
        sys.exit(1)

    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs = bg.quantise_frames(act.frame_times, act.onset_probs)
    B = beat_probs.shape[0]

    print(f"[{wav.name}]")
    print(f"  n_frames={act.n_frames}  computed duration_s={act.duration_s:.1f}s "
          f"(should match the real audio's actual duration)")
    print(f"  n_beats={B}  tempo={bg.tempo_bpm:.1f} BPM  last beat_time={bg.beat_times[-1]:.1f}s")

    low_midi, high_midi = parse_note_name(args.low), parse_note_name(args.high)
    low_k, high_k = low_midi - MIDI_START, high_midi - MIDI_START
    print(f"  key range: {args.low} (idx {low_k}) .. {args.high} (idx {high_k})")

    zoomed = beat_probs[:, low_k:high_k + 1]

    # Chroma: fold the FULL 88-key range (not just the zoomed window) down
    # to 12 pitch classes per beat — this is the "underlying scale" signal:
    # a stable chroma distribution within a GT segment is what a key/scale
    # estimate would be built from.
    chroma = np.zeros((B, 12), dtype=np.float64)
    for k in range(88):
        pc = (MIDI_START + k) % 12
        chroma[:, pc] += beat_probs[:, k]
    chroma_norm = chroma / chroma.sum(axis=1, keepdims=True).clip(min=1e-9)

    gt_song = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)
    gt_chords = gt_song.chord_events if gt_song else []

    fig, (ax_notes, ax_chroma, ax_chords) = plt.subplots(
        3, 1, figsize=(min(B * 0.11 + 2, 34), 11),
        gridspec_kw={"height_ratios": [4, 2, 1]}, sharex=True,
    )
    axes = [ax_notes, ax_chroma, ax_chords]

    beat_max = zoomed.max(axis=1, keepdims=True).clip(min=1e-6)
    display = (zoomed / beat_max).T  # normalise per-beat for visibility

    im = ax_notes.imshow(
        display, aspect="auto", origin="lower", cmap="inferno",
        vmin=0, vmax=1, interpolation="nearest",
        extent=[0, B, 0, display.shape[0]],
    )
    n_keys = high_k - low_k + 1
    c_indices_local = [i for i in range(n_keys) if (MIDI_START + low_k + i) % 12 == 0]
    ax_notes.set_yticks(c_indices_local)
    ax_notes.set_yticklabels([midi_label(low_k + i) for i in c_indices_local], fontsize=8)
    for k in c_indices_local:
        ax_notes.axhline(k, color="white", linewidth=0.3, alpha=0.25)
    ax_notes.set_ylabel("Piano key")
    ax_notes.set_title(
        f"Note probabilities ({args.low}–{args.high}), chroma, and GT chords — POP909 {song_id}",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax_notes, pad=0.01, label="normalised salience")

    # Chroma panel: 12 pitch classes, C at bottom.
    im2 = ax_chroma.imshow(
        chroma_norm.T, aspect="auto", origin="lower", cmap="inferno",
        vmin=0, interpolation="nearest", extent=[0, B, 0, 12],
    )
    ax_chroma.set_yticks(np.arange(12) + 0.5)
    ax_chroma.set_yticklabels(NOTE_NAMES, fontsize=8)
    ax_chroma.set_ylabel("Pitch class")
    plt.colorbar(im2, ax=ax_chroma, pad=0.01, label="share of chroma energy")

    # GT chord track, aligned to the same beat-index x-axis via the real beat
    # grid (ev.start_beat/end_beat are seconds, not beat indices — see the
    # known POP909Parser field-naming gotcha).
    for ev in gt_chords:
        b_start = int(np.searchsorted(bg.beat_times, ev.start_beat, side="left"))
        b_end = int(np.searchsorted(bg.beat_times, ev.end_beat, side="left"))
        b_start = min(b_start, B)
        b_end = min(max(b_end, b_start + 1), B)
        if b_start >= B:
            continue
        color = plt.get_cmap("hsv")(ev.root / 12.0) if ev.root >= 0 else "#888888"
        ax_chords.add_patch(mpatches.Rectangle(
            (b_start, 0), b_end - b_start, 1,
            facecolor=color, edgecolor="white", linewidth=0.3,
        ))
        ax_chords.text(
            (b_start + b_end) / 2, 0.5, ev.label, ha="center", va="center",
            fontsize=6, rotation=90 if (b_end - b_start) < 3 else 0,
            color="black" if ev.root >= 0 else "white",
        )
    ax_chords.set_xlim(0, B)
    ax_chords.set_ylim(0, 1)
    ax_chords.set_yticks([])
    ax_chords.set_ylabel("GT chord")
    ax_chords.set_xlabel("Time →")

    # Shared time gridlines + tick labels on every panel (not just the
    # bottom one) so a specific moment can be traced vertically through all
    # three — this is the point of stacking them.
    step = max(1, B // 40)
    xticks = list(range(0, B, step))
    xticklabels = [f"{bg.beat_times[i]:.1f}s" for i in xticks]
    for ax in axes:
        ax.set_xticks(xticks)
        ax.tick_params(labelbottom=True)
        for x in xticks:
            ax.axvline(x, color="white", linewidth=0.4, alpha=0.15, zorder=0)
    ax_notes.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    ax_chroma.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    ax_chords.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    out_dir = PLOT_ROOT / "inference" / f"pop909_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"note_probs_vs_gt_{args.low}_{args.high}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
