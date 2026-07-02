"""
Combined diagnostic: beat-level note-probability heatmap (zoomed to a piano
key range) stacked directly above the predicted-vs-ground-truth chord
timeline, sharing the same time axis — lets you visually check whether a
chord change in the timeline actually lines up with a real change in the
underlying note evidence, and vice versa.

Combines scripts/plot_note_probs_vs_gt.py's heatmap with
scripts/evaluate.py's plot_chord_timeline() panel.

Usage:
    .venv/bin/python scripts/plot_note_probs_with_chord_timeline.py --song 001
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
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import _cmap_for_root, root_pc_from_label  # noqa: E402

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
    from harmonia.eval.mirex_eval import evaluate_song
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.pipeline import HarmoniaPipeline

    song_id = args.song
    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    if not wav.exists():
        wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v000_prog0.wav"
    if not wav.exists():
        print(f"No render found for song {song_id}")
        sys.exit(1)

    # --- Note-probability heatmap data (beat-level, zoomed key range) ---
    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs = bg.quantise_frames(act.frame_times, act.onset_probs)
    B = beat_probs.shape[0]
    print(f"[{wav.name}] n_beats={B}  tempo={bg.tempo_bpm:.1f} BPM  "
          f"duration_s={act.duration_s:.1f}")

    low_midi, high_midi = parse_note_name(args.low), parse_note_name(args.high)
    low_k, high_k = low_midi - MIDI_START, high_midi - MIDI_START
    zoomed = beat_probs[:, low_k:high_k + 1]
    beat_max = zoomed.max(axis=1, keepdims=True).clip(min=1e-6)
    display = (zoomed / beat_max).T  # (n_keys, B), normalised per beat

    # Beat-time edges for pcolormesh, in seconds — real (non-uniform) beat
    # spacing, not a linear approximation, so this lines up exactly with the
    # chord timeline panel below (which is plotted in real seconds).
    avg_beat_s = float(np.mean(np.diff(bg.beat_times))) if B > 1 else 0.5
    time_edges = np.concatenate([bg.beat_times, [bg.beat_times[-1] + avg_beat_s]])
    key_edges = np.arange(high_k - low_k + 2)

    # --- Chord timeline data (predicted vs ground truth) ---
    gt_song = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)
    ref_intervals = np.array([[ev.start_beat, ev.end_beat] for ev in gt_song.chord_events])
    ref_labels = [ev.label for ev in gt_song.chord_events]

    pipeline = HarmoniaPipeline(prefer_madmom=False, cache_dir=DATA_ROOT / "cache")
    chart = pipeline.run(wav)
    est_intervals = np.array([[c["start_s"], c["end_s"]] for c in chart.chords])
    est_labels = [c["label"] for c in chart.chords]

    score = evaluate_song(chart.chords, ref_intervals, ref_labels)
    print(f"  {score.summary_line()}")

    duration_s = float(max(ref_intervals[-1, 1], est_intervals[-1, 1], time_edges[-1]))

    # --- Combined figure ---
    fig, (ax_notes, ax_chords) = plt.subplots(
        2, 1, figsize=(min(duration_s * 0.09 + 2, 40), 8.5),
        gridspec_kw={"height_ratios": [4, 1.2]}, sharex=True,
    )

    im = ax_notes.pcolormesh(
        time_edges, key_edges, display, cmap="inferno", vmin=0, vmax=1,
        shading="flat",
    )
    n_keys = high_k - low_k + 1
    c_indices_local = [i for i in range(n_keys) if (MIDI_START + low_k + i) % 12 == 0]
    ax_notes.set_yticks([i + 0.5 for i in c_indices_local])
    ax_notes.set_yticklabels([midi_label(low_k + i) for i in c_indices_local], fontsize=8)
    for k in c_indices_local:
        ax_notes.axhline(k, color="white", linewidth=0.3, alpha=0.25)
    ax_notes.set_ylabel("Piano key")
    ax_notes.set_title(
        f"Note probabilities ({args.low}–{args.high}) vs predicted/GT chord timeline — POP909 {song_id}",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax_notes, pad=0.01, label="normalised salience (per beat)")

    rows = [("Predicted", est_intervals, est_labels, 0.0), ("Ground truth", ref_intervals, ref_labels, 1.0)]
    for row_name, intervals, labels, y in rows:
        for (start, end), label in zip(intervals, labels):
            pc = root_pc_from_label(label)
            ax_chords.add_patch(mpatches.Rectangle(
                (start, y), end - start, 0.9,
                facecolor=_cmap_for_root(pc), edgecolor="white", linewidth=0.3,
            ))
            if end - start > duration_s * 0.008:
                ax_chords.text((start + end) / 2, y + 0.45, label, ha="center", va="center",
                                fontsize=6, color="black" if pc >= 0 else "white")
    ax_chords.set_xlim(0, duration_s)
    ax_chords.set_ylim(0, 2.0)
    ax_chords.set_yticks([0.45, 1.45])
    ax_chords.set_yticklabels(["Predicted", "Ground truth"])
    ax_chords.set_xlabel("Time (s)")

    plt.tight_layout()
    out_dir = PLOT_ROOT / "inference" / f"pop909_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"note_probs_with_chord_timeline_{args.low}_{args.high}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
