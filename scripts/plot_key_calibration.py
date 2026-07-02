"""
Visual diagnostics for the key-inference calibration fix (docs/known_issues.md
#0 / docs/handoff_2026-07-02_key_inference.md §4).

Produces three figures under docs/plots/inference/key_calibration/:
  1. confidence_by_segment.png — per-song bar chart of every segment's
     confidence, colour-coded by match/mismatch against key_audio.txt GT.
     What "normal" looks like: confidence varies segment-to-segment, and
     mismatches tend to cluster at lower confidence.
  2. evidence_vs_confidence.png — scatter of segment length (beats, the raw
     "amount of evidence") vs confidence, across all 5 songs. Shows the
     calibration property directly: more evidence -> higher confidence,
     not a flat line (which is what the old bug looked like) and not a
     wall at 1.0 (which is what the *second* bug, evidence inflation,
     looked like before it was fixed).
  3. posterior_song001_seg5.png — the full 24-key posterior bar chart for
     the exact segment used as the running example in the handoff (song
     001, 35-beat segment, 38.6s-62.1s). Old code gave every key here
     between 0.041 and 0.043 (visually indistinguishable bars). Annotated
     with that old ceiling for scale.

Usage:
    .venv/bin/python scripts/plot_key_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots" / "inference" / "key_calibration"

SONGS = ["001", "002", "003", "004", "005"]


def _match_color(match: bool) -> str:
    return "#2ca02c" if match else "#d62728"


def plot_confidence_by_segment(all_results: list[dict]) -> None:
    fig, axes = plt.subplots(len(all_results), 1, figsize=(12, 2.2 * len(all_results)), sharex=False)
    for ax, r in zip(axes, all_results):
        segs = r["segments"]
        x = np.arange(len(segs))
        colors = [_match_color(s["match"]) for s in segs]
        ax.bar(x, [s["confidence"] for s in segs], color=colors, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s['n_beats']}b" for s in segs], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("confidence")
        n_match = sum(s["match"] for s in segs)
        ax.set_title(
            f"Song {r['song_id']} — GT {r['gt_key']} — global pred {r['global_key']} "
            f"({'MATCH' if r['global_match'] else 'MISMATCH'}) — "
            f"{n_match}/{len(segs)} segments match GT ({r['duration_weighted_acc']:.0%} duration-weighted)",
            fontsize=9,
        )
    axes[-1].set_xlabel("segment (label = beat count, i.e. amount of evidence)")
    fig.suptitle(
        "Per-segment key-inference confidence, colour-coded by GT match "
        "(green=match, red=mismatch)\nx-axis label = segment length in beats",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = PLOT_ROOT / "confidence_by_segment.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_evidence_vs_confidence(all_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^", "D", "v"]
    for r, marker in zip(all_results, markers):
        segs = r["segments"]
        n_beats = [s["n_beats"] for s in segs]
        conf = [s["confidence"] for s in segs]
        colors = [_match_color(s["match"]) for s in segs]
        ax.scatter(n_beats, conf, c=colors, marker=marker, s=70,
                   edgecolors="black", linewidths=0.5, label=f"song {r['song_id']}", alpha=0.85)

    ax.set_xlabel("segment length (beats) — raw evidence amount")
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Calibration check: confidence should rise with evidence, not sit\n"
        "flat near 1/24≈0.042 (old bug) or saturate at 1.0 (the second bug)"
    )
    ax.axhline(1 / 24, color="gray", linestyle="--", linewidth=1, label="1/24 (old bug's ceiling)")

    # legend: marker shape = song, colour = match/mismatch (separate legends)
    from matplotlib.lines import Line2D
    song_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                            markeredgecolor="black", markersize=9, label=f"song {r['song_id']}")
                    for r, m in zip(all_results, markers)]
    match_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=9, label="matches GT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=9, label="mismatches GT"),
    ]
    leg1 = ax.legend(handles=song_handles, loc="upper left", title="song (shape)", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=match_handles + [Line2D([0], [0], color="gray", linestyle="--", label="1/24 ceiling")],
              loc="lower right", fontsize=8)

    fig.tight_layout()
    out = PLOT_ROOT / "evidence_vs_confidence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_full_posterior_example() -> None:
    """The song-001, 35-beat, 38.6s-62.1s segment used throughout the handoff."""
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.structure import Segmenter
    from harmonia.theory.key_profiles import infer_key, KEY_NAMES

    wav = DATA_ROOT / "renders" / "pop909" / "001" / "001_v005_musescoregeneral.wav"
    pe = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    act = pe.extract(wav)
    rhythm = RhythmAnalyser(prefer_madmom=False)
    bg = rhythm.analyse(wav)
    beat_probs = bg.quantise_frames(act.frame_times, act.onset_probs)
    segments = Segmenter().segment(beat_probs, bg.beat_times)

    # the 35-beat segment, 38.6s-62.1s (see docs/handoff_2026-07-02_key_inference.md §4)
    seg = next(s for s in segments if s.n_beats == 35)
    kp = infer_key(seg.chroma)

    order = np.argsort(kp.probs)[::-1]
    names = [KEY_NAMES[i] for i in order]
    probs = kp.probs[order]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ca02c" if n == kp.key_name else "#1f77b4" for n in names]
    ax.bar(range(24), probs, color=colors)
    ax.set_xticks(range(24))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.axhline(1 / 24, color="gray", linestyle="--", linewidth=1,
               label=f"1/24 = {1/24:.4f} (old bug: every key landed within ±0.001 of this)")
    ax.set_ylabel("posterior probability")
    ax.set_title(
        f"Full 24-key posterior — song 001, segment [{seg.start_time_s:.1f}s-{seg.end_time_s:.1f}s], "
        f"35 beats\nMAP: {kp.key_name} (confidence {kp.confidence:.3f}). "
        f"Old code gave every key 0.041-0.043 here — visually flat."
    )
    ax.legend()
    fig.tight_layout()
    out = PLOT_ROOT / "posterior_song001_seg5.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)

    from validate_key_inference import run_song

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Running pipeline + key inference across all 5 songs...")
    all_results = []
    for song_id in SONGS:
        r = run_song(song_id)
        if r is not None:
            all_results.append(r)

    print("\nPlotting...")
    plot_confidence_by_segment(all_results)
    plot_evidence_vs_confidence(all_results)
    plot_full_posterior_example()


if __name__ == "__main__":
    main()
