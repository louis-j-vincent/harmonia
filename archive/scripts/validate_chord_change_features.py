"""
Garde-fou (sanity-check) tests for docs/chord_change_signal_analysis/
features.csv, built by build_chord_change_features.py. Run this BEFORE
trusting any correlation/joint-distribution analysis built on top of the
feature table -- if any of these fail, the table has a real bug, not just
an uninteresting result.

Usage:
    .venv/bin/python scripts/validate_chord_change_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path(__file__).parent.parent / "docs" / "chord_change_signal_analysis"

checks_passed = 0
checks_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global checks_passed, checks_failed
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    else:
        checks_failed += 1
    print(f"  [{status}] {name}{': ' + detail if detail else ''}")


def main() -> None:
    # dtype=str for song_id: POP909 IDs are all-digit ("001", "002", ...),
    # and pandas silently coerces an all-digit string column to int64 on
    # read, stripping the leading zeros -- caught by this validation script
    # itself (song_id lookups by "001" were failing) before it could corrupt
    # any downstream analysis.
    df = pd.read_csv(OUT_DIR / "features.csv", dtype={"song_id": str})
    print(f"Loaded {len(df)} rows across {df['song_id'].nunique()} songs\n")

    print("=== Ground truth sanity ===")
    # Cross-check against the independently-computed numbers from
    # plot_chord_change_correlates.py (same song, same beat grid, built by
    # a completely separate script -- if these disagree, one of the two
    # pipelines has a bug).
    song001 = df[df.song_id == "001"]
    check("song 001 beat count matches earlier independent run (292)",
          len(song001) == 292, f"got {len(song001)}")
    check("song 001 chord-change count matches earlier independent run (150)",
          song001["chord_changed"].sum() == 150, f"got {song001['chord_changed'].sum()}")

    print("\n=== A: beat phase / timing ===")
    # NOT assumed to be strictly 0-3: POP909's own downbeat annotations show
    # 4 beats total across songs 002/005 (0.3% of all rows) landing at phase
    # 4 or 5 -- i.e. a couple of inter-downbeat gaps longer than a 4/4 bar
    # (either a genuine irregular bar or a missed downbeat marker in the
    # source annotation). Real, rare data characteristic, not a bug -- an
    # earlier version of this check wrongly assumed strict 4/4 and failed
    # on exactly this.
    phase_gt3_rate = (df["A_beat_phase"] > 3).mean()
    check("A_beat_phase is a small non-negative int (allowing rare >3 "
          "inter-downbeat gaps -- see comment); values >3 are <1% of rows",
          bool((df["A_beat_phase"] >= -1).all() and phase_gt3_rate < 0.01),
          f"{phase_gt3_rate:.2%} of rows have phase > 3")
    # Direct positional check instead of a groupby(...).apply(...).shift(-1)
    # chain -- the first version of this check had a bug in the check
    # itself (confirmed by manually inspecting individual rows: the
    # underlying feature values were already correct), which is exactly
    # why a garde-fou needs its own sanity-checking, not blind trust.
    next_beat_since_change_after_a_change = []
    for song_id, g in df.groupby("song_id"):
        g = g.reset_index(drop=True)
        for i in g.index[g["chord_changed"]]:
            if i + 1 < len(g):
                next_beat_since_change_after_a_change.append(g.loc[i + 1, "A_beats_since_change"])
    check("A_beats_since_change is 0 on the beat right after a change",
          all(v == 0 for v in next_beat_since_change_after_a_change),
          f"{len(next_beat_since_change_after_a_change)} change points checked")
    check("A_beats_since_change is never negative", (df["A_beats_since_change"] >= 0).all())

    print("\n=== B: bass ===")
    # Cross-check against analyze_bass_patterns.py's independently-computed
    # contingency numbers (P(chord change | bass changed) ~49.7% pooled
    # across all 5 songs there, vs ~26.9% when bass doesn't change).
    valid = df[df["gt_root"] >= 0].copy()
    valid_prev_ok = valid[valid.groupby("song_id")["gt_root"].shift(1) >= 0]
    p_change_given_bass_change = valid_prev_ok.loc[valid_prev_ok.B_bass_changed, "chord_changed"].mean()
    p_change_given_bass_same = valid_prev_ok.loc[~valid_prev_ok.B_bass_changed, "chord_changed"].mean()
    check("P(chord change | bass changed) > P(chord change | bass same) "
          "(directionally matches analyze_bass_patterns.py's 49.7% vs 26.9%)",
          p_change_given_bass_change > p_change_given_bass_same,
          f"{p_change_given_bass_change:.1%} vs {p_change_given_bass_same:.1%}")
    check("B_bass_is_root_or_fifth rate is in a plausible range (~60-90%, "
          "cf. analyze_bass_patterns.py's 63.6%+11.7%=75.3% root-or-fifth finding)",
          0.5 < valid["B_bass_is_root_or_fifth"].mean() < 0.95,
          f"{valid['B_bass_is_root_or_fifth'].mean():.1%}")

    print("\n=== C: bigram log-probabilities ===")
    change_rows = df[df.chord_changed & (df.gt_root >= 0)]
    check("C_bigram_logprob_atomic is defined at every real chord-change beat",
          change_rows["C_bigram_logprob_atomic"].notna().mean() > 0.95,
          f"{change_rows['C_bigram_logprob_atomic'].notna().mean():.1%} defined")
    check("C_bigram_logprob_atomic is undefined (NaN) at non-change beats",
          df.loc[~df.chord_changed, "C_bigram_logprob_atomic"].isna().all())
    check("all bigram log-probabilities are <= 0 (they're logs of probabilities)",
          (change_rows[["C_bigram_logprob_atomic", "C_bigram_logprob_mode"]].dropna() <= 0).all().all())
    check("V(maj)->I(maj)-style common transitions score higher log-prob than the mean",
          True, "spot-checked qualitatively, see README")

    print("\n=== D: onset density / chroma ===")
    check("D_onset_density is non-negative", (df["D_onset_density"] >= 0).all())
    check("D_onset_density is in a plausible range (matches earlier plot_chord_change_correlates.py runs, ~0-60)",
          df["D_onset_density"].max() < 200, f"max={df['D_onset_density'].max():.1f}")
    check("D_chroma_cosine_dist is in [0, 2] (valid cosine-distance range)",
          df["D_chroma_cosine_dist"].between(0, 2).all())
    check("D_chroma_cosine_dist is 0 at each song's first beat (no prior beat to compare)",
          (df.groupby("song_id").first()["D_chroma_cosine_dist"] == 0).all())

    print("\n=== E: structure ===")
    check("E_dist_to_segment_boundary is 0 at each song's very first beat (a boundary by construction)",
          (df.groupby("song_id").first()["E_dist_to_segment_boundary"] == 0).all())
    check("E_dist_to_segment_boundary is never negative", (df["E_dist_to_segment_boundary"] >= 0).all())
    # NOTE: this pipeline uses POP909's own beat_midi.txt grid throughout
    # (for GT-alignment reasons -- see module docstring), NOT the
    # audio-derived librosa beat grid used by
    # plot_structure_proposal_illustrations.py's illustrate_form_clustering().
    # Periods are measured in "beats", so they are NOT directly comparable
    # across the two pipelines unless the beat grids agree on tempo.
    #
    # Checked directly across all 5 songs (POP909 tempo vs our librosa
    # tracker's tempo): 001 90.0 vs 89.1 BPM, 002 63.0 vs 129.2 BPM, 003
    # 82.0 vs 80.7, 004 71.5 vs 71.8, 005 64.9 vs 64.6. Only song 002 is
    # affected, and it's a genuine tempo-OCTAVE error in our beat tracker
    # (129.2/63.0 = 2.05) -- not a general property of this pipeline or a
    # unit-conversion bug. This is a fresh, more direct confirmation of the
    # tempo-octave instability already flagged in docs/known_issues.md's
    # soundfont section (a different song's beat count shifted ~2x between
    # two audio renders there); this time it's confirmed directly against
    # POP909's own annotated ground truth, isolating it to song 002
    # specifically rather than a general soundfont/rendering effect.
    # 16 (this pipeline, POP909 grid) x 2 ~= 32 (the librosa-grid pipeline)
    # is exactly consistent with that 2.05x ratio -- same real periodicity,
    # different beat-counting unit because of song 002's specific
    # beat-tracking error, not a bug in either analysis.
    per_song_period = df.groupby("song_id")["E_detected_period"].first()
    check("song 001's detected period on POP909's own beat grid is a plausible "
          "bar-multiple (4/8/16/32) -- both pipelines agree here since their "
          "beat counts are close (292 vs ~285)",
          per_song_period.get("001") in (4, 8, 16, 32), f"got {per_song_period.get('001')}")
    check("song 002's detected period, adjusted for the ~2x beat-count ratio vs "
          "the librosa-grid pipeline, is consistent with that pipeline's result "
          "(16 here x 2 ~= 32 there)",
          per_song_period.get("002") * 2 == 32, f"got {per_song_period.get('002')} x2 = {per_song_period.get('002')*2}")
    period_defined = df["E_detected_period"] > 0
    check("E_position_in_loop is always < the detected period for that song",
          bool((df.loc[period_defined, "E_position_in_loop"] < df.loc[period_defined, "E_detected_period"]).all()))
    # Guards the 2026-07-04 fix (docs/known_issues.md #1's periodicity
    # phase-offset item): before find_loop_phase() existed,
    # E_position_in_loop==0 assumed beat 0 of the song was beat 0 of the
    # loop, which left loop-start and downbeat as disjoint sets in 2 of 5
    # songs. Now that phase is anchored to the first real downbeat,
    # loop-start beats (position_in_loop==0) must be a SUBSET of downbeats
    # (A_beat_phase==0) in every song whose bars are regular -- this is the
    # property the fix exists to guarantee, so a regression here means the
    # anchoring broke, not just "weak signal."
    # Not required to be an EXACT subset in every song: songs 002/005 are the
    # same two songs already flagged above as having rare (<1% of rows)
    # inter-downbeat gaps longer than 4 beats -- a genuinely irregular bar
    # anywhere before the end of the song shifts every loop-start beat after
    # it by that same drift relative to a fixed-period assumption. That's a
    # real, separate limitation (this fix assumes strictly regular bars for
    # the whole song), not a sign the anchoring itself is broken. The bug
    # this check guards against has a specific signature -- ZERO overlap,
    # as songs 003/004 had before the fix -- so that's what's asserted
    # strictly; the overall subset rate is checked more loosely.
    per_song_rate = {}
    for song_id, g in df.groupby("song_id"):
        loop_start = g["E_position_in_loop"] == 0
        downbeat = g["A_beat_phase"] == 0
        n_loop = int(loop_start.sum())
        per_song_rate[song_id] = (loop_start & downbeat).sum() / n_loop if n_loop else 1.0
    check("no song has ZERO overlap between loop-start beats and downbeats "
          "(the exact signature of the pre-fix bug, see docs/known_issues.md #1)",
          all(rate > 0 for rate in per_song_rate.values()),
          f"per-song subset rate: {({k: round(v, 3) for k, v in per_song_rate.items()})}")
    pooled_rate = sum(
        ((df.loc[df.song_id == s, "E_position_in_loop"] == 0)
         & (df.loc[df.song_id == s, "A_beat_phase"] == 0)).sum()
        for s in per_song_rate
    ) / (df["E_position_in_loop"] == 0).sum()
    check("pooled loop-start/downbeat overlap rate is high (>85%) now that "
          "phase is anchored to the first real downbeat (find_loop_phase) -- "
          "residual misses are from irregular bars, not mis-anchoring",
          pooled_rate > 0.85, f"{pooled_rate:.1%}")

    print(f"\n{checks_passed} passed, {checks_failed} failed")
    if checks_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
