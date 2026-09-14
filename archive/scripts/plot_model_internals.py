"""
Diagnostic plots for the Harmonia Bayesian HMM.

Plots generated:
  1. Chord emission profiles (chroma-collapsed) — what notes does each chord "expect"?
  2. Emission matrix slice — piano-key probabilities for C-rooted chords
  3. Key prior — diatonic vs chromatic boost for a given key
  4. Transition matrix (root-movement view) — cycle-of-fifths structure
  5. Jazz progression weights — which II-V-I variants dominate
  6-8. Inference plots (require pipeline output passed as --inference-json)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from harmonia.models.chord_hmm import (
    build_emission_matrix, build_key_prior, build_transition_matrix,
)
from harmonia.theory.chord_vocabulary import (
    build_index, chord_label, ChordQuality, get_vocabulary, CHORD_TEMPLATES,
)
from harmonia.theory.jazz_priors import STYLE_PRIORS, PROGRESSIONS, build_relative_transition_matrix

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MIDI_START = 21

OUT_DIR = Path(__file__).parent.parent / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def midi_to_name(midi: int) -> str:
    return NOTE_NAMES[(midi - 12) % 12] + str(midi // 12 - 1)


def emission_to_chroma(E: np.ndarray) -> np.ndarray:
    """Collapse (C, 88) emission matrix → (C, 12) chroma by summing across octaves."""
    chroma = np.zeros((E.shape[0], 12), dtype=np.float32)
    for k in range(88):
        midi = MIDI_START + k
        pc = midi % 12
        chroma[:, pc] += E[:, k]
    chroma /= chroma.sum(axis=1, keepdims=True).clip(min=1e-9)
    return chroma


# ── Plot 1: Chroma emission profiles for selected chord qualities ──────────────

def plot_emission_chroma(E: np.ndarray, idx_to_chord, save: Path):
    root = 0  # C
    qualities_of_interest = [
        ChordQuality.MAJOR, ChordQuality.MINOR, ChordQuality.DOM7,
        ChordQuality.MAJ7, ChordQuality.MIN7, ChordQuality.HALF_DIM7,
        ChordQuality.MIN_MAJ7, ChordQuality.DIMINISHED,
    ]
    chord_to_idx = {v: k for k, v in enumerate(idx_to_chord)}
    chroma = emission_to_chroma(E)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
    fig.suptitle("Emission profiles (chroma-collapsed) — C root, all qualities\n"
                 "Each bar = P(pitch class active | chord). Noise floor visible on non-chord tones.",
                 fontsize=11)

    for ax, qual in zip(axes.flat, qualities_of_interest):
        key = (root, qual)
        if key not in chord_to_idx:
            ax.set_visible(False)
            continue
        idx = chord_to_idx[key]
        profile = chroma[idx]

        # Colour bars by whether the pitch class is in the chord template
        template = CHORD_TEMPLATES.get(qual)
        in_chord = set()
        if template:
            in_chord = {(root + iv) % 12 for iv in template.weights}

        colors = ["#2196F3" if pc in in_chord else "#BBDEFB" for pc in range(12)]
        bars = ax.bar(range(12), profile, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(12))
        ax.set_xticklabels(
            [NOTE_NAMES[(root + i) % 12] for i in range(12)],
            fontsize=8,
        )
        ax.set_title(f"C:{qual.value}", fontsize=10, fontweight="bold")
        ax.set_ylim(0, profile.max() * 1.2)
        ax.axhline(0.05 / 12, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="noise floor" if ax == axes[0, 0] else "")

    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("P(pitch class | chord)")
    axes[1, 0].set_ylabel("P(pitch class | chord)")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save}")


# ── Plot 2: Piano-key emission for C-rooted chords (88-key view) ───────────────

def plot_emission_piano_keys(E: np.ndarray, idx_to_chord, save: Path):
    root = 0
    qualities = [ChordQuality.MAJOR, ChordQuality.MINOR, ChordQuality.DOM7,
                 ChordQuality.MAJ7, ChordQuality.MIN7]
    chord_to_idx = {v: k for k, v in enumerate(idx_to_chord)}

    # Focus on the musically relevant range: A1–C8 (MIDI 33–96 → key indices 12–75)
    key_range = slice(12, 76)
    midi_range = range(MIDI_START + 12, MIDI_START + 76)

    fig, axes = plt.subplots(len(qualities), 1, figsize=(18, 10), sharex=True)
    fig.suptitle("Emission P(key | chord) across piano range — C-rooted chords\n"
                 "Blue = chord tones; octave Gaussian weighting visible as envelope",
                 fontsize=11)

    for ax, qual in zip(axes, qualities):
        key = (root, qual)
        if key not in chord_to_idx:
            continue
        idx = chord_to_idx[key]
        profile = E[idx, key_range]
        midi_vals = list(midi_range)

        template = CHORD_TEMPLATES.get(qual)
        in_chord = set()
        if template:
            in_chord = {(root + iv) % 12 for iv in template.weights}

        colors = ["#1565C0" if (m % 12) in in_chord else "#90CAF9"
                  for m in midi_vals]
        ax.bar(range(len(midi_vals)), profile, color=colors, width=1.0,
               edgecolor="none")
        ax.set_ylabel(f"C:{qual.value}", fontsize=9, rotation=0, labelpad=50,
                      va="center")
        ax.set_ylim(0, profile.max() * 1.3)

        # Octave markers
        for m_idx, m in enumerate(midi_vals):
            if m % 12 == 0:
                ax.axvline(m_idx, color="gray", linewidth=0.5, alpha=0.4)
                ax.text(m_idx, profile.max() * 1.1, f"C{m//12 - 1}",
                        fontsize=6, ha="center", color="gray")

    # X-axis: note names for every C
    c_positions = [i for i, m in enumerate(midi_range) if m % 12 == 0]
    axes[-1].set_xticks(c_positions)
    axes[-1].set_xticklabels([f"C{(MIDI_START+12+p)//12-1}" for p in c_positions],
                              fontsize=8)
    axes[-1].set_xlabel("Piano key")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save}")


# ── Plot 3: Key prior — diatonic vs chromatic chord weights ────────────────────

def plot_key_prior(idx_to_chord, tonic: int = 0, mode: str = "major",
                   diatonic_boost: float = 3.0, save: Path = None):
    log_prior = build_key_prior(tonic, mode, max_phase=1,
                                diatonic_boost=diatonic_boost)
    prior = np.exp(log_prior)

    # Sort chords by prior weight descending
    labels = [chord_label(r, q) for r, q in idx_to_chord]
    order = np.argsort(prior)[::-1][:40]  # top 40

    colors = ["#1B5E20" if prior[i] > 1.0 else "#EF9A9A" for i in order]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(order)), prior[order], color=colors, edgecolor="white",
           linewidth=0.3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[i] for i in order], rotation=60, ha="right",
                       fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
               label="chromatic baseline (1.0)")
    ax.axhline(diatonic_boost, color="green", linestyle="--", linewidth=0.8,
               label=f"diatonic boost ({diatonic_boost}×)")
    ax.set_ylabel("Prior weight (linear)")
    ax.set_title(f"Key prior — {NOTE_NAMES[tonic]} {mode}  (top 40 chords shown)\n"
                 f"Green = diatonic, Red = chromatic. diatonic_boost={diatonic_boost}",
                 fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save}")


# ── Plot 4: Transition matrix — root-movement view ─────────────────────────────

def plot_transition_matrix(tonic: int = 0, style: str = "jazz_medium_swing",
                           save: Path = None):
    """
    Collapse the (C,C) transition matrix to a (12,12) root-movement matrix
    by summing over all quality pairs. Reveals cycle-of-fifths structure.
    """
    log_A = build_transition_matrix(tonic, max_phase=1, style=style)
    A = np.exp(log_A)  # (C, C)

    idx_to_chord, _ = build_index(1)
    C = len(idx_to_chord)

    # Root-movement matrix: A12[from_root, to_root] = sum over qualities
    A12 = np.zeros((12, 12), dtype=np.float32)
    counts = np.zeros((12, 12), dtype=np.int32)
    for i, (ri, qi) in enumerate(idx_to_chord):
        if qi == ChordQuality.NO_CHORD:
            continue
        for j, (rj, qj) in enumerate(idx_to_chord):
            if qj == ChordQuality.NO_CHORD:
                continue
            A12[ri, rj] += A[i, j]
            counts[ri, rj] += 1
    # Average rather than sum so scale doesn't depend on quality count
    A12 /= counts.clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: full 12×12 heatmap
    ax = axes[0]
    tonic_label = NOTE_NAMES[tonic]
    im = ax.imshow(A12, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    # Reorder labels so tonic is at position 0 (scale-degree view)
    scale_labels = [NOTE_NAMES[(tonic + i) % 12] for i in range(12)]
    ax.set_xticklabels(scale_labels, fontsize=9)
    ax.set_yticklabels(scale_labels, fontsize=9)
    ax.set_xlabel("To root")
    ax.set_ylabel("From root")
    ax.set_title(f"Root-movement transition matrix\n{tonic_label} major, {style}",
                 fontsize=10)
    plt.colorbar(im, ax=ax, label="Avg P(to | from)")

    # Annotate the II-V-I arc: ii(D)→V(G)→I(C) for C major
    ii  = (tonic + 2) % 12
    V   = (tonic + 7) % 12
    I   = tonic % 12
    # Reindex to tonic-relative
    for (fr, to), label in [((ii, V), "ii→V"), ((V, I), "V→I"), ((ii, I), "ii→I")]:
        ax.annotate("", xy=(to, fr), xytext=(fr, to),
                    arrowprops=dict(arrowstyle="-", color="blue", lw=1.5,
                                   linestyle="dashed"))

    # Right: interval histogram (root movement in semitones)
    ax2 = axes[1]
    interval_weights = np.zeros(12)
    for fr in range(12):
        for to in range(12):
            interval = (to - fr) % 12
            interval_weights[interval] += A12[fr, to]
    interval_weights /= interval_weights.sum()

    interval_names = ["unison\n(stay)", "m2↑", "M2↑", "m3↑", "M3↑",
                      "P4↑", "tritone", "P5↑\n(=P4↓)", "m6↑", "M6↑",
                      "m7↑", "M7↑"]
    highlight = [0, 5, 7]  # unison, fourth, fifth — most common in jazz
    colors2 = ["#1565C0" if i in highlight else "#90CAF9" for i in range(12)]
    ax2.bar(range(12), interval_weights, color=colors2, edgecolor="white")
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(interval_names, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Probability mass")
    ax2.set_title("Root-movement interval distribution\n(marginalised over all from-roots)",
                  fontsize=10)
    ax2.annotate("P5 = cycle-of-fifths\nstrongest jazz motion",
                 xy=(7, interval_weights[7]), xytext=(8.5, interval_weights[7] * 0.9),
                 fontsize=8, color="#1565C0",
                 arrowprops=dict(arrowstyle="->", color="#1565C0"))

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save}")


# ── Plot 5: Jazz progression weights by style ──────────────────────────────────

def plot_progression_weights(save: Path):
    styles = list(STYLE_PRIORS.keys())
    prog_names = sorted(PROGRESSIONS.keys())

    # Weight matrix: styles × progressions
    W = np.zeros((len(styles), len(prog_names)))
    for si, sname in enumerate(styles):
        sp = STYLE_PRIORS[sname]
        for pi, pname in enumerate(prog_names):
            W[si, pi] = sp.progression_weights.get(pname, 0.0)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(W, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(prog_names)))
    ax.set_yticks(range(len(styles)))
    ax.set_xticklabels(prog_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(styles, fontsize=9)
    ax.set_title("Jazz progression weights by style\n"
                 "Rows = styles, Cols = progressions. Brighter = stronger prior.",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save}")


# ── Plot 6–8: Inference plots (need pipeline output) ──────────────────────────

def plot_inference(inference_json: Path, pop909_song=None):
    """
    Plots 6–8, generated after running the pipeline on a POP909 song.

    6. Beat-level note probability heatmap (Basic Pitch output) with
       ground-truth chord boundaries overlaid.
    7. Viterbi path: top-5 chord posteriors per beat + MAP path.
    8. Ground-truth vs predicted chord timeline (segment bars).
    """
    with open(inference_json) as f:
        data = json.load(f)

    # -- Plot 8: chord timeline comparison (works from JSON alone)
    fig, axes = plt.subplots(2, 1, figsize=(18, 4), sharex=True)

    def draw_chord_timeline(ax, chords, title, color):
        prev_end = 0.0
        for ch in chords:
            s, e = ch["start_s"], ch["end_s"]
            ax.barh(0, e - s, left=s, height=0.6, color=color,
                    alpha=0.7, edgecolor="white", linewidth=0.5)
            if e - s > 0.5:
                ax.text((s + e) / 2, 0, ch["label"],
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white")
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(ch["end_s"] for ch in chords) if chords else 30)

    draw_chord_timeline(axes[0], data["chords"],
                        f"Predicted — {data.get('global_key','?')} "
                        f"@ {data.get('tempo_bpm','?')} BPM", "#1565C0")

    if pop909_song is not None:
        # Convert beat-based GT to seconds using beat times from pipeline
        gt_chords = []
        for ev in pop909_song.chord_events:
            gt_chords.append({
                "label": ev.label,
                "start_s": float(ev.start_beat),
                "end_s": float(ev.end_beat),
            })
        # Normalise: use beat index as proxy for time (approximate)
        if gt_chords:
            draw_chord_timeline(axes[1], gt_chords,
                                "Ground truth (POP909 — beat index as time proxy)",
                                "#2E7D32")
    else:
        axes[1].set_visible(False)

    axes[-1].set_xlabel("Time (s) / beat index")
    plt.suptitle("Chord timeline: predicted vs ground truth", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    save = OUT_DIR / "06_chord_timeline.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-json", type=Path, default=None,
                        help="Path to chart JSON from pipeline.run() → chart.save_json()")
    parser.add_argument("--tonic", type=int, default=0,
                        help="Tonic pitch class for prior/transition plots (0=C)")
    parser.add_argument("--style", default="jazz_medium_swing")
    args = parser.parse_args()

    print("Building model components...")
    E = build_emission_matrix(max_phase=1, noise_floor=0.05)
    idx_to_chord, _ = build_index(max_phase=1)

    print(f"Emission matrix: {E.shape}  ({len(idx_to_chord)} chords × 88 keys)")
    print(f"Output dir: {OUT_DIR}")
    print()

    print("Plot 1: Chord emission chroma profiles...")
    plot_emission_chroma(E, idx_to_chord,
                         save=OUT_DIR / "01_emission_chroma.png")

    print("Plot 2: Piano-key emission for C-rooted chords...")
    plot_emission_piano_keys(E, idx_to_chord,
                              save=OUT_DIR / "02_emission_piano_keys.png")

    print("Plot 3: Key prior...")
    plot_key_prior(idx_to_chord, tonic=args.tonic, mode="major",
                   save=OUT_DIR / "03_key_prior.png")

    print("Plot 4: Transition matrix (root-movement view)...")
    plot_transition_matrix(tonic=args.tonic, style=args.style,
                           save=OUT_DIR / "04_transition_matrix.png")

    print("Plot 5: Jazz progression weights by style...")
    plot_progression_weights(save=OUT_DIR / "05_progression_weights.png")

    if args.inference_json:
        print("Plot 6: Chord timeline (inference vs ground truth)...")
        plot_inference(args.inference_json)
    else:
        print("Skipping inference plots (pass --inference-json to generate).")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
