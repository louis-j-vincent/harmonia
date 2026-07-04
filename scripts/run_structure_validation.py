"""
Cross-Repeat Harmonic Agreement (CRHA) structure-validation experiment,
the "smallest first experiment" proposed in
docs/structure_trigram_design_2026-07-04.md: does a hypothesized
repeating loop (period, phase) actually correspond to beats that share the
same ground-truth chord, more than a shuffled-label null would predict by
chance? Runs entirely on features_symbolic.csv (gt_label, A_beat_phase
already computed there) -- no MIDI/audio re-parsing, no chroma proxy
needed, exactly the "sidestep the chroma proxy entirely" version the
design doc calls out as the cheapest first check.

For each song:
  - Candidate periods: the fixed set {4, 8, 16, 32, 64} (this codebase's
    standing beats_per_bar=4 x {1,2,4,8,16} convention, matching
    harmonia.models.periodicity.score_periods's own candidate set),
    restricted to values with at least 2 full repeats available.
  - For each candidate period L, phase = the song's first downbeat index
    mod L (is_downbeat reconstructed as A_beat_phase==0 -- exactly
    equivalent to harmonia.models.periodicity.find_loop_phase's own logic,
    verified: `_beat_in_bar_phase` sets phase=0 exactly at is_downbeat).
  - Group beats (excluding unannotated/no-chord beats) by
    (beat_idx - phase) % L. For each group with >=2 members, agreement =
    (count of the most common gt_label in the group) / (group size).
    CRHA(L) = size-weighted mean agreement across all qualifying groups.
  - Null: 200 shuffles of which beat goes in which group (same group-size
    partition, random membership), recomputing the same weighted-agreement
    statistic; null_95 = the 95th percentile across shuffles.
  - Best hypothesis for a song = the period L maximizing
    (CRHA(L) - null_95(L)). "Trustworthy" = that margin exceeds
    TRUST_MARGIN (0.15, matching the design doc's example threshold).

Usage:
    .venv/bin/python scripts/run_structure_validation.py [--limit N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent.parent / "docs" / "chord_change_signal_analysis"
CANDIDATE_PERIODS = [4, 8, 16, 32, 64]
N_SHUFFLES = 200
TRUST_MARGIN = 0.15
RNG_SEED = 0


def weighted_agreement(labels: np.ndarray, group_ids: np.ndarray) -> float:
    """Size-weighted mean of (majority-label fraction) across groups with >=2 members."""
    order = np.argsort(group_ids, kind="stable")
    labels_sorted = labels[order]
    group_sorted = group_ids[order]
    boundaries = np.flatnonzero(np.diff(group_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(group_sorted)]))

    total_weight = 0
    total_score = 0.0
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        _, counts = np.unique(labels_sorted[s:e], return_counts=True)
        total_score += counts.max()
        total_weight += n
    return total_score / total_weight if total_weight > 0 else np.nan


def evaluate_hypothesis(gt_label: np.ndarray, valid_mask: np.ndarray, phase: int, period: int,
                         rng: np.random.RandomState) -> tuple[float, float]:
    B = len(gt_label)
    group_ids_all = (np.arange(B) - phase) % period
    idx = np.where(valid_mask)[0]
    if len(idx) < 2 * period:
        return np.nan, np.nan
    labels = gt_label[idx]
    group_ids = group_ids_all[idx]

    real_score = weighted_agreement(labels, group_ids)
    if np.isnan(real_score):
        return np.nan, np.nan

    null_scores = np.empty(N_SHUFFLES)
    for i in range(N_SHUFFLES):
        shuffled = rng.permutation(group_ids)
        null_scores[i] = weighted_agreement(labels, shuffled)
    null_95 = float(np.nanpercentile(null_scores, 95))
    return real_score, null_95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(OUT_DIR / "features_symbolic.csv", dtype={"song_id": str})
    rng = np.random.RandomState(RNG_SEED)

    song_ids = sorted(df["song_id"].unique())
    if args.limit:
        song_ids = song_ids[:args.limit]

    rows = []
    t0 = time.time()
    for i, song_id in enumerate(song_ids):
        g = df[df.song_id == song_id].sort_values("beat_idx").reset_index(drop=True)
        B = len(g)
        valid_mask = g["gt_root"].values >= 0
        gt_label = g["gt_label"].fillna("__NOCHORD__").values.astype(str)
        is_downbeat = (g["A_beat_phase"].values == 0)
        downbeat_idxs = np.flatnonzero(is_downbeat)
        first_downbeat = int(downbeat_idxs[0]) if len(downbeat_idxs) else 0

        candidates = [L for L in CANDIDATE_PERIODS if L <= B // 2]
        if not candidates:
            continue

        per_song = []
        for L in candidates:
            phase = first_downbeat % L
            score, null95 = evaluate_hypothesis(gt_label, valid_mask, phase, L, rng)
            if np.isnan(score):
                continue
            margin = score - null95
            per_song.append((L, phase, score, null95, margin))

        if not per_song:
            continue
        L, phase, score, null95, margin = max(per_song, key=lambda r: r[-1])
        rows.append({
            "song_id": song_id, "n_beats": B, "n_candidates_tested": len(per_song),
            "best_period": L, "best_phase": phase,
            "crha": score, "null_95": null95, "margin": margin,
            "trustworthy": margin > TRUST_MARGIN,
        })
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(song_ids)} songs, {elapsed:.0f}s elapsed, "
                  f"{elapsed / (i + 1) * (len(song_ids) - i - 1):.0f}s remaining (est.)")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "structure_validation_results.csv", index=False)

    print(f"\nSongs evaluated: {len(result)}/{len(song_ids)}, {time.time() - t0:.0f}s total")
    print(f"Trustworthy (margin > {TRUST_MARGIN}): {result['trustworthy'].sum()} "
          f"({result['trustworthy'].mean():.1%})")
    print("\nMargin distribution:")
    print(result["margin"].describe())
    print("\nBest-period distribution among trustworthy songs:")
    print(result.loc[result.trustworthy, "best_period"].value_counts())
    print("\nBest-period distribution among ALL songs (not just trustworthy):")
    print(result["best_period"].value_counts())


if __name__ == "__main__":
    main()
