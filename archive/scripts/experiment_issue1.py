"""
EMISSION-QUALITY probe for issue #1 (chord-change temporal resolution too
coarse). See docs/known_issues.md #1.

ONE metric, deliberately decoder-free:

  per-beat emission argmax root-accuracy vs GT — build the emission matrix,
  take the argmax at every beat, compare roots. No HMM, no Viterbi, no
  segmentation, no key prior. It answers only "does the raw per-beat evidence
  discriminate chords better under this frontend setting?", which is the
  question candidate A was about.

Scope, after the 2026-07-30 cleanup
-----------------------------------
This file used to also carry `run_full_pipeline_variant` and five modes
(`--verify`, `--sweep-full`, `--sweep-duration`, `--sweep-key-prior`,
`--sweep-emission-scoring`) that ran the Gen-1 `HarmoniaPipeline` end-to-end on
POP909 and scored strict MIREX. `HarmoniaPipeline` was deleted as a duplicate
of the live pipeline, so those modes went with it. What that means:

* The knobs those modes swept (`emission_scoring`, `key_prior_weight`,
  `duration_prior`, `self_transition_boost`, ...) are `chord_hmm.ChordInferrer`
  constructor arguments, and `ChordInferrer` is NOT on the live path —
  `chord_pipeline_v1.infer_chords_v1` never constructs one. Sweeping them could
  not have moved any shipped number.
* End-to-end scoring now lives in ONE place: `harmonia.eval.accuracy_score`
  (Brick 0, partial-credit + strict, real audio) via `scripts/evaluate.py`.
  Add end-to-end modes there, not here.
* What this file still uniquely provides is the HMM-free view of emission
  quality. That is why it was kept rather than deleted.

Usage:
    .venv/bin/python scripts/experiment_issue1.py --songs 001 002 003 004 005
    .venv/bin/python scripts/experiment_issue1.py --songs 001 --onset-percentile 95
    .venv/bin/python scripts/experiment_issue1.py --sweep --songs 001 002
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "data"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------

def gt_root_at_time(gt_chords, t: float) -> int:
    """Root pitch class (0-11) of the GT chord active at time t, -1 for N/none."""
    for ev in gt_chords:
        if ev.start_beat <= t < ev.end_beat:  # already seconds, see known gotcha
            return ev.root
    return -2  # no GT coverage at this time (before first / after last event)


# ---------------------------------------------------------------------------
# Metric 1: per-beat emission argmax root-accuracy (bypasses the HMM)
# ---------------------------------------------------------------------------

def per_beat_argmax_root_accuracy(
    beat_probs: np.ndarray,       # (B, 88)
    beat_times: np.ndarray,       # (B,)
    gt_chords,
    emission_matrix: np.ndarray,  # (C, 88)
    idx_to_chord: list,
    normalize: bool = False,
    compress: str | None = None,  # None | "sqrt" | "log1p"
) -> tuple[float, int]:
    """
    Returns (accuracy, n_beats_scored). Only beats where GT has a real chord
    (not N, not uncovered) are scored — root accuracy isn't meaningful for N.

    normalize (L1, per beat) is mathematically inert here — it subtracts a
    per-beat constant from every chord's score uniformly, which can never
    change an argmax. Kept only so the harness can demonstrate that (see
    docs/known_issues.md #1). compress applies a nonlinear, per-element
    transform instead, which *can* change relative weighting within a beat.
    """
    bp = beat_probs.astype(np.float64)
    if compress == "sqrt":
        bp = np.sqrt(bp)
    elif compress == "log1p":
        bp = np.log1p(bp)
    if normalize:
        row_sums = bp.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        bp = bp / row_sums

    scores = bp @ emission_matrix.T  # (B, C)
    pred_idx = scores.argmax(axis=1)

    correct = 0
    scored = 0
    for b, t in enumerate(beat_times):
        gt_root = gt_root_at_time(gt_chords, float(t))
        if gt_root < 0:
            continue  # N or uncovered — skip
        scored += 1
        pred_root, _ = idx_to_chord[pred_idx[b]]
        if pred_root == gt_root:
            correct += 1

    return (correct / scored if scored else 0.0), scored


# ---------------------------------------------------------------------------
# Per-song variant runner
# ---------------------------------------------------------------------------

def compute_beat_probs(
    wav: Path,
    onset_threshold: float,
    onset_percentile: float | None,
    cache_dir: Path,
):
    from harmonia.core.features import FeatureExtractor
    from harmonia.models.rhythm import RhythmAnalyser

    extractor = FeatureExtractor.create("bp48", cache_dir=cache_dir)
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(
        wav, onset_threshold=onset_threshold, onset_percentile=onset_percentile
    )
    bg = rhythm.analyse(wav)
    beat_probs = bg.quantise_frames(act.frame_times, act.onsets)
    return beat_probs, bg


def run_variant(
    song_ids: list[str],
    onset_threshold: float,
    onset_percentile: float | None,
    normalize_emission: bool,
    label: str,
    compress: str | None = None,
) -> None:
    from harmonia.models.chord_hmm import build_emission_matrix
    from harmonia.theory.chord_vocabulary import build_index
    from harmonia.data.pop909_parser import POP909Parser

    pop909_dir = DATA_ROOT / "pop909" / "POP909"
    parser = POP909Parser(pop909_dir)
    E = build_emission_matrix(max_phase=1)
    idx_to_chord, _ = build_index(max_phase=1)
    cache_dir = DATA_ROOT / "cache"

    print(f"\n=== Variant: {label} "
          f"(onset_threshold={onset_threshold}, onset_percentile={onset_percentile}, "
          f"normalize_emission={normalize_emission}, compress={compress}) ===")

    accs = []
    for song_id in song_ids:
        gt = parser.parse_song(song_id)
        if gt is None or not gt.chord_events:
            print(f"  {song_id}: no GT, skipping")
            continue
        wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v000_prog0.wav"
        if not wav.exists():
            print(f"  {song_id}: no wav, skipping")
            continue

        beat_probs, bg = compute_beat_probs(wav, onset_threshold, onset_percentile, cache_dir)
        acc, n_scored = per_beat_argmax_root_accuracy(
            beat_probs, bg.beat_times, gt.chord_events, E, idx_to_chord,
            normalize=normalize_emission, compress=compress,
        )
        accs.append(acc)
        print(f"  {song_id}: per-beat argmax root-accuracy = {acc:.1%} ({n_scored} beats scored)")

    if accs:
        print(f"  MEAN across {len(accs)} songs: {np.mean(accs):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", nargs="+", default=["001", "002", "003", "004", "005"])
    parser.add_argument("--sweep", action="store_true",
                         help="Run the full baseline + A1/A2/A3 comparison sweep (metric 1)")
    parser.add_argument("--onset-threshold", type=float, default=0.3)
    parser.add_argument("--onset-percentile", type=float, default=None)
    parser.add_argument("--normalize-emission", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(levelname)s  %(message)s")

    if args.sweep:
        run_variant(args.songs, onset_threshold=0.3, onset_percentile=None,
                    normalize_emission=False, label="baseline (fixed 0.3, no L1-norm)")
        run_variant(args.songs, onset_threshold=0.3, onset_percentile=None,
                    normalize_emission=True, label="A1: L1-normalize beat_probs")
        for p in (90, 95, 97):
            run_variant(args.songs, onset_threshold=0.3, onset_percentile=p,
                        normalize_emission=False, label=f"A2: percentile={p}")
        run_variant(args.songs, onset_threshold=0.3, onset_percentile=95,
                    normalize_emission=True, label="A1+A2: L1-norm + percentile=95")
        for c in ("sqrt", "log1p"):
            run_variant(args.songs, onset_threshold=0.3, onset_percentile=None,
                        normalize_emission=False, compress=c, label=f"A3: compress={c}")
        return

    run_variant(
        args.songs, onset_threshold=args.onset_threshold,
        onset_percentile=args.onset_percentile,
        normalize_emission=args.normalize_emission, label="custom",
    )


if __name__ == "__main__":
    main()
