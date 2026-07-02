"""
Per-song inference diagnostic plots.

Usage:
    python scripts/plot_inference_diagnostics.py --song 001

Generates docs/plots/inference/<song_id>/:
    s1_note_probs_beats.png  — (B, 88) beat-level note probability heatmap
    s1_note_probs_frames.png — (F, 88) raw frame-level Basic Pitch output
    chord_timeline.png       — predicted vs GT chord timeline (if GT available)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

NOTE_NAMES  = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MIDI_START  = 21   # A0
PLOT_ROOT   = Path(__file__).parent.parent / "docs" / "plots"
DATA_ROOT   = Path(__file__).parent.parent / "data"


def midi_label(key_idx: int) -> str:
    midi = MIDI_START + key_idx
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


# ── Plot: beat-level note probability heatmap ─────────────────────────────────

def plot_beat_note_probs(
    beat_probs: np.ndarray,     # (B, 88)
    beat_times: np.ndarray,     # (B,)
    gt_chords=None,             # list of ChordEvent or None
    out: Path = None,
    title_suffix: str = "",
):
    """
    Heatmap of beat-level note activations.

    Rows  = 88 piano keys (A0 at bottom, C8 at top).
    Cols  = beats (time →).
    Color = summed Basic Pitch salience within that beat window.

    Ground-truth chord boundaries shown as vertical dashed lines if provided.
    """
    B, K = beat_probs.shape
    assert K == 88

    # Normalise each beat independently so colour reflects relative note
    # salience rather than beat-length-accumulated energy.
    beat_max = beat_probs.max(axis=1, keepdims=True).clip(min=1e-6)
    display = (beat_probs / beat_max).T   # (88, B)  — transpose for imshow

    fig, ax = plt.subplots(figsize=(min(B * 0.12 + 2, 28), 7))

    im = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        vmin=0, vmax=1,
        interpolation="nearest",
        extent=[0, B, 0, 88],
    )

    # Y-axis: label every C note
    c_indices = [k for k in range(88) if (MIDI_START + k) % 12 == 0]
    ax.set_yticks(c_indices)
    ax.set_yticklabels([midi_label(k) for k in c_indices], fontsize=8)
    ax.set_ylabel("Piano key")

    # X-axis: every 4 beats, labelled with time in seconds
    step = max(1, B // 40)
    xticks = list(range(0, B, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{beat_times[i]:.1f}s" for i in xticks],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Beat (time →)")

    # Octave guides
    for k in c_indices:
        ax.axhline(k, color="white", linewidth=0.3, alpha=0.25)

    # Ground-truth chord boundaries.
    # ev.start_beat is already SECONDS (POP909Parser field-naming gotcha),
    # not a beat index -- must convert via the real beat grid (searchsorted),
    # not by rounding the seconds value directly. At this song's tempo
    # (89 BPM, ~0.68s/beat) beats accumulate faster than seconds, so treating
    # seconds as a beat index systematically misplaces every line earlier
    # than it should be, and drops every event past beat_times[-1]/2 or so
    # off the right edge entirely once the (wrong) index exceeds B.
    if gt_chords is not None:
        seen_beats = set()
        for ev in gt_chords:
            b = int(np.searchsorted(beat_times, ev.start_beat, side="left"))
            if 0 < b < B and b not in seen_beats:
                ax.axvline(b, color="#00E5FF", linewidth=0.8, alpha=0.7)
                if ev.label != "N":
                    ax.text(b + 0.1, 86, ev.label, fontsize=6,
                            color="#00E5FF", va="top", rotation=90)
                seen_beats.add(b)

    plt.colorbar(im, ax=ax, label="Normalised salience (per beat)", pad=0.01)
    ax.set_title(
        f"Stage 1 — Beat-level note probabilities{title_suffix}\n"
        f"{B} beats · 88 keys · cyan lines = GT chord boundaries",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Plot: raw frame-level heatmap ─────────────────────────────────────────────

def plot_frame_note_probs(
    note_probs: np.ndarray,     # (F, 88)
    frame_times: np.ndarray,    # (F,)
    beat_times: np.ndarray,     # (B,)
    out: Path = None,
    title_suffix: str = "",
):
    """
    Full-resolution frame-level heatmap from Basic Pitch (~43 Hz).
    Shows the raw model output before beat quantisation.
    Beat grid overlaid as vertical lines.
    """
    F, K = note_probs.shape
    display = note_probs.T  # (88, F)

    # Downsample for display if very long (keep max 2000 frames wide)
    if F > 2000:
        stride = F // 2000
        display = display[:, ::stride]
        frame_times_ds = frame_times[::stride]
    else:
        frame_times_ds = frame_times

    W = display.shape[1]

    fig, ax = plt.subplots(figsize=(min(W * 0.012 + 2, 28), 7))
    im = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        vmin=0, vmax=display.max(),
        interpolation="nearest",
        extent=[0, W, 0, 88],
    )

    # Y-axis: C labels
    c_indices = [k for k in range(88) if (MIDI_START + k) % 12 == 0]
    ax.set_yticks(c_indices)
    ax.set_yticklabels([midi_label(k) for k in c_indices], fontsize=8)
    ax.set_ylabel("Piano key")
    for k in c_indices:
        ax.axhline(k, color="white", linewidth=0.3, alpha=0.2)

    # Beat grid
    for bt in beat_times:
        frame_idx = np.searchsorted(frame_times_ds, bt)
        if frame_idx < W:
            ax.axvline(frame_idx, color="#00E5FF", linewidth=0.5, alpha=0.5)

    # X-axis: time labels
    step = max(1, W // 30)
    xticks = list(range(0, W, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{frame_times_ds[i]:.1f}s" for i in xticks],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Frame (time →)")

    plt.colorbar(im, ax=ax, label="Basic Pitch salience", pad=0.01)
    ax.set_title(
        f"Stage 1 — Frame-level note probabilities (~43 Hz){title_suffix}\n"
        f"{F} frames · cyan lines = beat grid from librosa",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default="001", help="POP909 song ID (e.g. 001)")
    args = parser.parse_args()

    song_id = args.song
    out_dir = PLOT_ROOT / "inference" / f"pop909_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v000_prog0.wav"
    if not wav.exists():
        print(f"WAV not found: {wav}")
        sys.exit(1)

    # Load GT chords
    gt_chords = None
    pop909_dir = DATA_ROOT / "pop909" / "POP909"
    if pop909_dir.exists():
        from harmonia.data.pop909_parser import POP909Parser
        parser_p = POP909Parser(pop909_dir)
        song = parser_p.parse_song(song_id)
        if song:
            gt_chords = song.chord_events
            print(f"  GT: {len(gt_chords)} chord events")

    print("Running Basic Pitch + beat tracker...")
    import logging
    logging.basicConfig(level=logging.WARNING)

    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.rhythm import RhythmAnalyser

    extractor = PitchExtractor()
    rhythm = RhythmAnalyser(prefer_madmom=False)

    activations = extractor.extract(wav)
    beat_grid = rhythm.analyse(wav)
    beat_probs = beat_grid.quantise_frames(activations.frame_times, activations.onset_probs)

    print(f"  Frames: {activations.note_probs.shape}  "
          f"range [{activations.note_probs.min():.3f}, {activations.note_probs.max():.3f}]")
    print(f"  Beats:  {beat_probs.shape}  "
          f"range [{beat_probs.min():.3f}, {beat_probs.max():.3f}]  "
          f"@ {beat_grid.tempo_bpm:.1f} BPM")

    suffix = f" — POP909 {song_id}"

    print("Plot: frame-level onset probabilities...")
    plot_frame_note_probs(
        activations.onset_probs,
        activations.frame_times,
        beat_grid.beat_times,
        out=out_dir / "s1_note_probs_frames.png",
        title_suffix=suffix,
    )

    print("Plot: beat-level note probabilities...")
    plot_beat_note_probs(
        beat_probs,
        beat_grid.beat_times,
        gt_chords=gt_chords,
        out=out_dir / "s1_note_probs_beats.png",
        title_suffix=suffix,
    )

    print(f"\nAll plots → {out_dir}/")


if __name__ == "__main__":
    main()
