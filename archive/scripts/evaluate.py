"""
End-to-end evaluation: run the Harmonia pipeline on a POP909 song and score
predicted chords against ground truth.

Usage:
    .venv/bin/python scripts/evaluate.py --song 001

Prints MIREX weighted-overlap accuracy (root/majmin/sevenths/tetrads) and a
duration-weighted root-confusion summary, and saves a predicted-vs-GT chord
timeline to docs/plots/inference/pop909_<song>/chord_timeline.png.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots"


def root_pc_from_label(label: str) -> int:
    """Extract root pitch class (0-11) from a Harte-ish label, -1 for N/unparseable."""
    from harmonia.data.pop909_parser import parse_harte_label

    parsed = parse_harte_label(label)
    if parsed is None:
        return -1
    root, _ = parsed
    return root


def root_name(pc: int) -> str:
    return "N" if pc < 0 else ROOT_NAMES[pc]


# ---------------------------------------------------------------------------
# Confusion summary
# ---------------------------------------------------------------------------

def root_confusion(
    ref_intervals: np.ndarray, ref_labels: list[str],
    est_intervals: np.ndarray, est_labels: list[str],
) -> tuple[np.ndarray, float]:
    """
    Duration-weighted root-pitch-class confusion matrix (13x13: 12 roots + N).

    Returns:
        matrix: (13, 13) seconds of overlap, rows=GT root, cols=predicted root
        total_duration: total scored duration in seconds
    """
    import mir_eval.util as mu

    t_min = min(ref_intervals[0, 0], est_intervals[0, 0])
    t_max = max(ref_intervals[-1, 1], est_intervals[-1, 1])
    ref_intervals, ref_labels = mu.adjust_intervals(
        ref_intervals, ref_labels, t_min=t_min, t_max=t_max, start_label="N", end_label="N"
    )
    est_intervals, est_labels = mu.adjust_intervals(
        est_intervals, est_labels, t_min=t_min, t_max=t_max, start_label="N", end_label="N"
    )
    intervals, ref_l, est_l = mu.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels
    )
    durations = intervals[:, 1] - intervals[:, 0]

    matrix = np.zeros((13, 13))
    for dur, rl, el in zip(durations, ref_l, est_l):
        r = root_pc_from_label(rl)
        e = root_pc_from_label(el)
        matrix[r if r >= 0 else 12, e if e >= 0 else 12] += dur

    return matrix, float(durations.sum())


def print_confusion(matrix: np.ndarray, total: float) -> None:
    labels = ROOT_NAMES + ["N"]
    print(f"\n  Root-pitch-class confusion (seconds, GT rows x predicted cols, total {total:.1f}s)")
    header = "        " + "".join(f"{l:>6}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        row = "".join(f"{matrix[i, j]:6.1f}" for j in range(13))
        print(f"  {l:>4}  {row}")

    diag = np.trace(matrix)
    print(f"\n  Root match rate (any duration weighting): {diag / total:.1%}" if total > 0 else "")

    # top confused (off-diagonal) root pairs
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i in range(13):
        for j in range(13):
            if off[i, j] > 0:
                pairs.append((off[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    print("\n  Top confusions (GT -> predicted):")
    for dur, gt, pred in pairs[:8]:
        print(f"    {gt:>3} -> {pred:<3}  {dur:.1f}s")


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------

def _cmap_for_root(pc: int) -> str:
    if pc < 0:
        return "#888888"
    cmap = plt.get_cmap("hsv")
    return cmap(pc / 12.0)


def plot_chord_timeline(
    ref_intervals: np.ndarray, ref_labels: list[str],
    est_intervals: np.ndarray, est_labels: list[str],
    duration_s: float,
    out: Path,
    title_suffix: str = "",
) -> None:
    width = min(duration_s * 0.09 + 2, 40)
    fig, ax = plt.subplots(figsize=(width, 3.2))

    rows = [("Predicted", est_intervals, est_labels, 0.0), ("Ground truth", ref_intervals, ref_labels, 1.0)]

    for row_name, intervals, labels, y in rows:
        for (start, end), label in zip(intervals, labels):
            pc = root_pc_from_label(label)
            ax.add_patch(mpatches.Rectangle(
                (start, y), end - start, 0.9,
                facecolor=_cmap_for_root(pc), edgecolor="white", linewidth=0.3,
            ))
            if end - start > duration_s * 0.008:
                ax.text((start + end) / 2, y + 0.45, label, ha="center", va="center",
                        fontsize=6, color="black" if pc >= 0 else "white")

    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.45, 1.45])
    ax.set_yticklabels(["Predicted", "Ground truth"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Chord timeline — predicted vs ground truth{title_suffix}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default="001", help="POP909 song ID (e.g. 001)")
    parser.add_argument("--render", default=None, help="Override render filename (e.g. 001_v005_musescoregeneral.wav)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(levelname)s  %(message)s")

    song_id = args.song
    render_name = args.render or f"{song_id}_v005_musescoregeneral.wav"
    wav = DATA_ROOT / "renders" / "pop909" / song_id / render_name
    if not wav.exists():
        # fall back to prog0 if musescoregeneral not found
        wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v000_prog0.wav"
    if not wav.exists():
        print(f"WAV not found: {wav}")
        sys.exit(1)

    from harmonia.data.pop909_parser import POP909Parser

    pop909_dir = DATA_ROOT / "pop909" / "POP909"
    gt_song = POP909Parser(pop909_dir).parse_song(song_id)
    if gt_song is None:
        print(f"No GT annotations for song {song_id}")
        sys.exit(1)
    print(f"GT: {gt_song.n_chords} chord events")

    # NOTE: POP909 chord_midi.txt start/end columns are already in seconds
    # (MIDI-aligned), despite the ChordEvent field names start_beat/end_beat.
    ref_intervals = np.array([[ev.start_beat, ev.end_beat] for ev in gt_song.chord_events])
    ref_labels = [ev.label for ev in gt_song.chord_events]

    print(f"Running Harmonia pipeline on {wav.name}...")
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    chart = infer_chords_v1(wav)
    print(f"Predicted: {len(chart.chords)} chord events, "
          f"key={chart.global_key}, tempo={chart.tempo_bpm:.0f} BPM, style={chart.style}")

    from harmonia.eval.mirex_eval import evaluate_song

    score = evaluate_song(chart.chords, ref_intervals, ref_labels)
    print(f"\nMIREX weighted-overlap accuracy: {score.summary_line()}")

    est_intervals, est_labels = np.array([[c["start_s"], c["end_s"]] for c in chart.chords]), \
        [c["label"] for c in chart.chords]

    matrix, total = root_confusion(ref_intervals, ref_labels, est_intervals, est_labels)
    print_confusion(matrix, total)

    duration_s = float(max(ref_intervals[-1, 1], est_intervals[-1, 1]))
    out_dir = PLOT_ROOT / "inference" / f"pop909_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_chord_timeline(
        ref_intervals, ref_labels, est_intervals, est_labels,
        duration_s, out_dir / "chord_timeline.png",
        title_suffix=f" — POP909 {song_id}",
    )


if __name__ == "__main__":
    main()
