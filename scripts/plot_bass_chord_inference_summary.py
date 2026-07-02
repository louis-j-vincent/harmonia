"""
Summary plot for the 1-hour bass-chord-inference sprint (2026-07-02) --
see docs/known_issues.md issue #1's "oracle-segment chord reconstruction"
subsection for the full write-up. Two panels:
  1. Iteration story: root accuracy as terms are added to the scoring
     formula (chroma+key only -> + bass -> + root/fifth heuristic ->
     + tuned weights), oracle GT segment boundaries, all 5 songs pooled.
  2. The slash-chord (inversion) finding: root accuracy split by whether
     the GT label encodes a bass-note inversion ("/5" etc.) -- the single
     largest source of remaining "error", and mostly a ground-truth
     definitional mismatch (labelled functional root vs sounding bass
     note) rather than a model mistake.

Usage:
    .venv/bin/python scripts/plot_bass_chord_inference_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots" / "inference" / "bass_patterns"

# Hard-coded from the sprint's actual runs (scripts/experiment_bass_chord_inference.py),
# all 5-song pooled means at oracle GT segment boundaries.
ITERATION_STEPS = [
    ("chroma + key only\n(w_bass=0)", 53.3),
    ("+ bass\n(fifth_weight=0)", 73.4),
    ("+ root/fifth heuristic\n(fifth_weight=0.4)", 80.3),
    ("+ tuned weights\n(fifth_weight=0.8, w_bass=1.5)", 83.2),
]

SLASH_BREAKDOWN = [
    ("non-inversion labels\n(n=545, ~90%)", 86.8),
    ("inversion labels (e.g. \"/5\")\n(n=63, ~10%)", 38.1),
]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    labels = [s[0] for s in ITERATION_STEPS]
    values = [s[1] for s in ITERATION_STEPS]
    colors = ["#888888", "#1f77b4", "#1f77b4", "#2ca02c"]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("root accuracy (%)")
    ax.set_ylim(0, 95)
    ax.set_title("Root accuracy at ORACLE GT segment boundaries\n(isolates labelling from timing -- see docs/known_issues.md #1)")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1]
    labels = [s[0] for s in SLASH_BREAKDOWN]
    values = [s[1] for s in SLASH_BREAKDOWN]
    bars = ax.bar(labels, values, color=["#2ca02c", "#d62728"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("root accuracy (%)")
    ax.set_ylim(0, 95)
    ax.set_title("Remaining error is concentrated in slash-chord (inversion) labels\n"
                  "GT root = functional root; bass-driven model follows the SOUNDING bass note")

    fig.suptitle("Bass-informed chord reconstruction at oracle boundaries (5 POP909 songs, pooled)", fontsize=12)
    fig.tight_layout()
    out = PLOT_ROOT / "bass_chord_inference_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
