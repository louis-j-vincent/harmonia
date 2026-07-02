"""
Diagnostic plot for the Candidate-C periodicity idea (see docs/known_issues.md
issue #1, plan section "Candidate C — periodicity / structure folding").

Plots, for a real POP909 song:
  1. The self-similarity matrix (SSM) as a heatmap, with GT chord boundaries
     overlaid — makes the "L-th off-diagonal" concept concrete on real data.
  2. The periodicity profile score(L) = mean_i SSM[i, i+L] for L in a musically
     plausible range, with the candidate period set {4, 8, 16, 32} beats marked.

Usage:
    .venv/bin/python scripts/plot_periodicity_diagnostic.py --song 001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots"


def periodicity_profile(ssm: np.ndarray, max_lag: int) -> np.ndarray:
    """score(L) = mean_i SSM[i, i+L], for L in [1, max_lag]."""
    B = ssm.shape[0]
    scores = np.zeros(max_lag)
    for L in range(1, max_lag + 1):
        if L >= B:
            break
        scores[L - 1] = np.diagonal(ssm, offset=L).mean()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default="001")
    args = parser.parse_args()
    song_id = args.song

    import logging
    logging.basicConfig(level=logging.WARNING)

    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.structure import build_ssm
    from harmonia.data.pop909_parser import POP909Parser

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v000_prog0.wav"
    if not wav.exists():
        print(f"WAV not found: {wav}")
        sys.exit(1)

    print("Extracting activations + beat grid...")
    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs = bg.quantise_frames(act.frame_times, act.onset_probs)
    B = beat_probs.shape[0]
    beats_per_bar = bg.beats_per_bar()
    print(f"  {B} beats, {beats_per_bar} beats/bar, tempo={bg.tempo_bpm:.1f} BPM")

    ssm = build_ssm(beat_probs)

    gt_chords = None
    pop909_dir = DATA_ROOT / "pop909" / "POP909"
    if pop909_dir.exists():
        song = POP909Parser(pop909_dir).parse_song(song_id)
        if song:
            gt_chords = song.chord_events

    max_lag = min(B - 1, 80)
    scores = periodicity_profile(ssm, max_lag)
    candidate_periods = [beats_per_bar * k for k in (1, 2, 4, 8) if beats_per_bar * k <= max_lag]

    fig, (ax_ssm, ax_score) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.1]}
    )

    # --- Panel 1: SSM heatmap ---
    im = ax_ssm.imshow(ssm, origin="upper", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax_ssm.set_title(f"Self-similarity matrix — POP909 {song_id}\n{B} beats")
    ax_ssm.set_xlabel("beat j")
    ax_ssm.set_ylabel("beat i")
    plt.colorbar(im, ax=ax_ssm, label="cosine similarity", fraction=0.046, pad=0.04)

    # Mark GT chord boundaries as tick lines
    if gt_chords is not None:
        seen = set()
        for ev in gt_chords:
            b = bg.beat_index_at(ev.start_beat)  # ev.start_beat is seconds
            if 0 < b < B and b not in seen:
                ax_ssm.axhline(b, color="cyan", linewidth=0.2, alpha=0.4)
                ax_ssm.axvline(b, color="cyan", linewidth=0.2, alpha=0.4)
                seen.add(b)

    # Highlight the candidate-period off-diagonals with dashed guide lines
    for L in candidate_periods:
        xs = np.arange(0, B - L)
        ys = xs + L
        ax_ssm.plot(ys, xs, linewidth=0.6, alpha=0.6, label=f"L={L}")
    ax_ssm.legend(loc="upper right", fontsize=7, framealpha=0.7)

    # --- Panel 2: periodicity profile ---
    lags = np.arange(1, max_lag + 1)
    ax_score.plot(lags, scores, color="black", linewidth=1)
    ax_score.set_xlabel("lag L (beats)")
    ax_score.set_ylabel("score(L) = mean_i SSM[i, i+L]")
    ax_score.set_title("Periodicity profile (autocorrelation of self-similarity)")
    for L in candidate_periods:
        ax_score.axvline(L, color="tab:orange", linestyle="--", linewidth=1, alpha=0.7)
        ax_score.text(L, scores.max() * 1.02, f"{L}", color="tab:orange", fontsize=8, ha="center")
    xticks = np.arange(0, max_lag + 1, 4)
    ax_score.set_xticks(xticks)
    ax_score.set_xticklabels([str(x) for x in xticks], rotation=90, fontsize=7)
    ax_score.set_xlim(0, max_lag)
    ax_score.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = PLOT_ROOT / "inference" / f"pop909_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ssm_periodicity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    print("\nscore(L) at candidate periods:")
    for L in candidate_periods:
        print(f"  L={L:3d}  score={scores[L-1]:.4f}")
    top5 = np.argsort(scores)[::-1][:5] + 1
    print("\ntop-5 lags overall (unconstrained):")
    for L in top5:
        print(f"  L={L:3d}  score={scores[L-1]:.4f}")


if __name__ == "__main__":
    main()
