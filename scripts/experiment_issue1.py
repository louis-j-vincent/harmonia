"""
A/B testing harness for issue #1 (chord-change temporal resolution too coarse).
See docs/known_issues.md #1 and the approved plan
(~/.claude/plans/proud-plotting-boot.md at time of writing).

Three isolated metrics, each targeting one fix candidate's specific hypothesis
rather than only the confounded end-to-end score:

  1. per-beat emission argmax root-accuracy vs GT — bypasses the HMM/Viterbi
     entirely, so it isolates whether the raw emission signal discriminates
     chords better. Targets candidate A (emission quality).
  2. chord-boundary F-score (mir_eval.segment.detection) — isolates whether
     the *rate* of predicted chord changes matches GT, independent of root/
     quality correctness. Targets candidate B (duration model).
  3. MIREX weighted accuracy (harmonia.eval.mirex_eval.evaluate_song) — full
     downstream sanity check, used for all three candidates.

Usage:
    .venv/bin/python scripts/experiment_issue1.py --songs 001 002 003 004 005
    .venv/bin/python scripts/experiment_issue1.py --songs 001 --onset-percentile 95
    .venv/bin/python scripts/experiment_issue1.py --songs 001 --normalize-emission
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
# Metric 2: chord-boundary F-score
# ---------------------------------------------------------------------------

def boundary_f_score(
    pred_intervals: np.ndarray,
    ref_intervals: np.ndarray,
    window: float = 0.5,
) -> tuple[float, float, float]:
    """Returns (precision, recall, f_measure) via mir_eval.segment.detection."""
    import mir_eval.segment as ms

    if len(pred_intervals) == 0 or len(ref_intervals) == 0:
        return 0.0, 0.0, 0.0
    return ms.detection(ref_intervals, pred_intervals, window=window)


# ---------------------------------------------------------------------------
# Per-song variant runner
# ---------------------------------------------------------------------------

def compute_beat_probs(
    wav: Path,
    onset_threshold: float,
    onset_percentile: float | None,
    cache_dir: Path,
):
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.rhythm import RhythmAnalyser

    extractor = PitchExtractor(cache_dir=cache_dir)
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(
        wav, onset_threshold=onset_threshold, onset_percentile=onset_percentile
    )
    bg = rhythm.analyse(wav)
    beat_probs = bg.quantise_frames(act.frame_times, act.onset_probs)
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


def run_full_pipeline_variant(
    song_ids: list[str],
    onset_percentile: float | None,
    normalize_emission: bool,
    label: str,
    compress_emission: str | None = None,
    duration_prior: dict | None = None,
    boundary_window: float = 0.5,
    key_prior_per_beat: bool = True,
    key_prior_weight: float = 1.0,
    wav_suffix: str = "v000_prog0",
    emission_scoring: str = "dot",
) -> None:
    """
    Runs the actual HarmoniaPipeline (Viterbi included) and reports metrics
    2 (boundary F-score) and 3 (MIREX weighted accuracy).

    wav_suffix: which render to use, e.g. "v000_prog0" (original, low-
        fidelity soundfont) or "v005_musescoregeneral" (the soundfont fix
        adopted in docs/known_issues.md #2 -- use this for any comparison
        against the key_prior_per_beat numbers in
        docs/handoff_2026-07-02_key_inference.md §3, which were measured
        post-soundfont-fix).
    emission_scoring: "dot" (default) or "cosine" — see
        docs/known_issues.md #5.
    """
    from harmonia.pipeline import HarmoniaPipeline
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.eval.mirex_eval import evaluate_song

    pop909_dir = DATA_ROOT / "pop909" / "POP909"
    parser = POP909Parser(pop909_dir)
    pipeline = HarmoniaPipeline(
        prefer_madmom=False,
        cache_dir=DATA_ROOT / "cache",
        normalize_emission=normalize_emission,
        compress_emission=compress_emission,
        onset_percentile=onset_percentile,
        duration_prior=duration_prior,
        key_prior_per_beat=key_prior_per_beat,
        key_prior_weight=key_prior_weight,
        emission_scoring=emission_scoring,
    )

    print(f"\n=== Full-pipeline variant: {label} "
          f"(onset_percentile={onset_percentile}, normalize_emission={normalize_emission}, "
          f"compress_emission={compress_emission}, duration_aware={duration_prior is not None}, "
          f"key_prior_per_beat={key_prior_per_beat}, key_prior_weight={key_prior_weight}, "
          f"wav={wav_suffix}) ===")

    f_scores, root_scores, majmin_scores = [], [], []
    per_song = {}
    for song_id in song_ids:
        gt = parser.parse_song(song_id)
        if gt is None or not gt.chord_events:
            print(f"  {song_id}: no GT, skipping")
            continue
        wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_{wav_suffix}.wav"
        if not wav.exists():
            print(f"  {song_id}: no wav, skipping")
            continue

        ref_intervals = np.array([[ev.start_beat, ev.end_beat] for ev in gt.chord_events])
        ref_labels = [ev.label for ev in gt.chord_events]

        chart = pipeline.run(wav)
        pred_intervals = np.array([[c["start_s"], c["end_s"]] for c in chart.chords])

        p, r, f = boundary_f_score(pred_intervals, ref_intervals, window=boundary_window)
        score = evaluate_song(chart.chords, ref_intervals, ref_labels)
        f_scores.append(f)
        root_scores.append(score.root)
        majmin_scores.append(score.majmin)
        per_song[song_id] = {"root": score.root, "majmin": score.majmin, "boundary_f": f}
        print(f"  {song_id}: n_events={len(chart.chords):3d}  "
              f"boundary P/R/F={p:.2f}/{r:.2f}/{f:.2f}  "
              f"root={score.root:.1%}  majmin={score.majmin:.1%}")

    if f_scores:
        print(f"  MEAN: boundary_F={np.mean(f_scores):.3f}  "
              f"root={np.mean(root_scores):.1%}  majmin={np.mean(majmin_scores):.1%}")
    return per_song


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", nargs="+", default=["001", "002", "003", "004", "005"])
    parser.add_argument("--sweep", action="store_true",
                         help="Run the full baseline + A1/A2 comparison sweep (metric 1 only)")
    parser.add_argument("--sweep-full", action="store_true",
                         help="Run baseline + A1/A2 through the full pipeline (metrics 2+3)")
    parser.add_argument("--sweep-duration", action="store_true",
                         help="Run baseline vs duration-aware decoding (candidate B) through the full pipeline")
    parser.add_argument("--sweep-key-prior", action="store_true",
                         help="Re-check key_prior_per_beat (docs/known_issues.md #0/#3) now that "
                              "infer_key() is calibrated; uses v005_musescoregeneral renders")
    parser.add_argument("--sweep-emission-scoring", action="store_true",
                         help="A/B dot vs cosine emission scoring (docs/known_issues.md #5); "
                              "uses v005_musescoregeneral renders")
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

    if args.sweep_full:
        run_full_pipeline_variant(args.songs, onset_percentile=None,
                                   normalize_emission=False, label="baseline")
        run_full_pipeline_variant(args.songs, onset_percentile=None,
                                   normalize_emission=True, label="A1: L1-normalize")
        run_full_pipeline_variant(args.songs, onset_percentile=97,
                                   normalize_emission=False, label="A2: percentile=97")
        run_full_pipeline_variant(args.songs, onset_percentile=97,
                                   normalize_emission=True, label="A1+A2")
        run_full_pipeline_variant(args.songs, onset_percentile=None, normalize_emission=False,
                                   compress_emission="sqrt", label="A3: compress=sqrt")
        run_full_pipeline_variant(args.songs, onset_percentile=None, normalize_emission=False,
                                   compress_emission="log1p", label="A3: compress=log1p")
        return

    if args.sweep_duration:
        from harmonia.theory.duration_prior import fit_duration_prior

        prior = fit_duration_prior(DATA_ROOT / "pop909" / "POP909")
        run_full_pipeline_variant(args.songs, onset_percentile=None,
                                   normalize_emission=False, label="baseline (geometric)")
        run_full_pipeline_variant(args.songs, onset_percentile=None, normalize_emission=False,
                                   duration_prior=prior, label="B: duration-aware (empirical)")
        return

    if args.sweep_key_prior:
        without = run_full_pipeline_variant(
            args.songs, onset_percentile=None, normalize_emission=False,
            key_prior_per_beat=False, label="key_prior_per_beat=False (baseline)",
            wav_suffix="v005_musescoregeneral",
        )
        wit = run_full_pipeline_variant(
            args.songs, onset_percentile=None, normalize_emission=False,
            key_prior_per_beat=True, key_prior_weight=1.0, label="key_prior_per_beat=True (w=1)",
            wav_suffix="v005_musescoregeneral",
        )
        print("\n=== Per-song delta (with - without) ===")
        for song_id in args.songs:
            if song_id not in without or song_id not in wit:
                continue
            d_root = wit[song_id]["root"] - without[song_id]["root"]
            d_majmin = wit[song_id]["majmin"] - without[song_id]["majmin"]
            print(f"  {song_id}: root {without[song_id]['root']:.1%} -> {wit[song_id]['root']:.1%} "
                  f"({d_root:+.1%})   majmin {without[song_id]['majmin']:.1%} -> "
                  f"{wit[song_id]['majmin']:.1%} ({d_majmin:+.1%})")
        return

    if args.sweep_emission_scoring:
        dot = run_full_pipeline_variant(
            args.songs, onset_percentile=None, normalize_emission=False,
            emission_scoring="dot", label="emission_scoring=dot (baseline)",
            wav_suffix="v005_musescoregeneral",
        )
        cosine = run_full_pipeline_variant(
            args.songs, onset_percentile=None, normalize_emission=False,
            emission_scoring="cosine", label="emission_scoring=cosine",
            wav_suffix="v005_musescoregeneral",
        )
        print("\n=== Per-song delta (cosine - dot) ===")
        for song_id in args.songs:
            if song_id not in dot or song_id not in cosine:
                continue
            d_root = cosine[song_id]["root"] - dot[song_id]["root"]
            d_majmin = cosine[song_id]["majmin"] - dot[song_id]["majmin"]
            print(f"  {song_id}: root {dot[song_id]['root']:.1%} -> {cosine[song_id]['root']:.1%} "
                  f"({d_root:+.1%})   majmin {dot[song_id]['majmin']:.1%} -> "
                  f"{cosine[song_id]['majmin']:.1%} ({d_majmin:+.1%})")
        return

    run_variant(
        args.songs, onset_threshold=args.onset_threshold,
        onset_percentile=args.onset_percentile,
        normalize_emission=args.normalize_emission, label="custom",
    )


if __name__ == "__main__":
    main()
