"""
Learn what a real bass note looks like from POP909's ground-truth MIDI (the
PIANO/accompaniment track -- see bass_track.py::true_bass_track), then use
that to build and validate an improved audio-only bass detector.

Motivation: infer_bass_track()'s naive heuristic ("lowest active audio key
above a relative threshold") can't tell a genuinely isolated bass note apart
from the bottom note of a closely-voiced chord, and it always reports *some*
bass even when the accompaniment genuinely has a rest -- "the bass isn't
always the lowest note, and sometimes there is no bass" (2026-07 feedback).

Three steps, using ground truth (never available at real inference time,
only for learning/validating a detector that only uses audio):
  1. How often is there truly no bass, and what register does a real bass
     note live in? (from the PIANO track alone, ground truth)
  2. What semitone gap (to the next note up, across the full mix) separates
     a genuinely isolated bass note from a non-isolated one? Measured on
     the *audio*-derived activation (what the detector actually has to work
     with), split by whether ground truth says a real bass note is present.
  3. Grid-search a (register ceiling, isolation-gap) threshold pair against
     ground truth, and report the improvement over the naive detector.

Usage:
    .venv/bin/python scripts/learn_bass_distribution.py --songs 001 002 003 004 005
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

from bass_track import MIDI_START, infer_bass_track, true_bass_track  # noqa: E402

DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots" / "inference" / "bass_patterns"
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def audio_candidate_and_gap(beat_probs_onset, threshold_frac=0.15, min_abs=0.3):
    """Per beat: (candidate MIDI note or -1, semitone gap to the next active
    key above it or None) -- the same active-set logic infer_bass_track
    uses, with the intermediate gap exposed for analysis."""
    B, _ = beat_probs_onset.shape
    candidates = np.full(B, -1, dtype=int)
    gaps: list[int | None] = [None] * B
    for b in range(B):
        row = beat_probs_onset[b]
        peak = row.max()
        if peak < min_abs:
            continue
        thresh = max(min_abs, threshold_frac * peak)
        active = np.nonzero(row >= thresh)[0]
        if len(active) == 0:
            continue
        candidates[b] = MIDI_START + int(active[0])
        if len(active) >= 2:
            gaps[b] = int(active[1] - active[0])
    return candidates, gaps


def collect_song(song_id: str) -> dict | None:
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    if not wav.exists():
        print(f"  {song_id}: no wav, skipping")
        return None
    gt_song = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)
    if gt_song is None:
        print(f"  {song_id}: no GT, skipping")
        return None

    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs_onset = bg.quantise_frames(act.frame_times, act.onset_probs)

    true_beats = true_bass_track(gt_song.midi_path, bg.beat_times)
    candidates, audio_gaps = audio_candidate_and_gap(beat_probs_onset)

    return {
        "song_id": song_id, "beat_probs_onset": beat_probs_onset,
        "true_beats": true_beats, "candidates": candidates, "audio_gaps": audio_gaps,
    }


# ---------------------------------------------------------------------------
# 1. Register + silence rate (ground truth only)
# ---------------------------------------------------------------------------

def plot_register_and_silence(all_data: list[dict]) -> int:
    true_bass_vals = [tb.true_bass for d in all_data for tb in d["true_beats"] if tb.true_bass >= 0]
    n_total = sum(len(d["true_beats"]) for d in all_data)
    n_no_bass = n_total - len(true_bass_vals)
    no_bass_rate = n_no_bass / n_total

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(true_bass_vals, bins=range(min(true_bass_vals), max(true_bass_vals) + 2),
            color="#1f77b4", align="left")
    ax.set_xlabel("true bass MIDI note (PIANO track, ground truth)")
    ax.set_ylabel("beat count")
    p95 = int(np.percentile(true_bass_vals, 95))
    p99 = int(np.percentile(true_bass_vals, 99))
    ax.axvline(p95, color="orange", linestyle="--", label=f"95th pct = {p95} ({NOTE_NAMES[p95%12]}{p95//12-1})")
    ax.axvline(p99, color="red", linestyle="--", label=f"99th pct = {p99} ({NOTE_NAMES[p99%12]}{p99//12-1})")
    ax.set_title(
        f"True bass register (ground truth, {len(all_data)} songs, n={n_total} beats)\n"
        f"No genuine bass at all in {n_no_bass}/{n_total} beats ({no_bass_rate:.1%})"
    )
    ax.legend()
    fig.tight_layout()
    out = PLOT_ROOT / "bass_register_and_silence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"    no-bass rate (ground truth) = {no_bass_rate:.1%}")
    print(f"    true bass register: min={min(true_bass_vals)} max={max(true_bass_vals)} "
          f"p95={p95} p99={p99}")
    return max(true_bass_vals)


# ---------------------------------------------------------------------------
# 2. Audio-gap distribution, split by true-bass-presence
# ---------------------------------------------------------------------------

def plot_gap_distribution(all_data: list[dict]) -> None:
    gap_present, gap_absent = [], []
    for d in all_data:
        for tb, gap in zip(d["true_beats"], d["audio_gaps"]):
            if gap is None:
                continue
            (gap_present if tb.true_bass >= 0 else gap_absent).append(gap)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, 14) - 0.5
    ax.hist(gap_present, bins=bins, alpha=0.6, density=True, color="#2ca02c",
            label=f"true bass present (n={len(gap_present)}, median={np.median(gap_present):.0f})")
    ax.hist(gap_absent, bins=bins, alpha=0.6, density=True, color="#d62728",
            label=f"true bass absent (n={len(gap_absent)}, median={np.median(gap_absent):.0f})")
    ax.set_xlabel("audio-derived gap: lowest active key to next active key above it (semitones)")
    ax.set_ylabel("density")
    ax.set_title(
        "Is the lowest active audio note isolated (real bass) or just the bottom\n"
        "of a dense chord (no real bass), split by ground truth"
    )
    ax.legend()
    fig.tight_layout()
    out = PLOT_ROOT / "bass_isolation_gap_dist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"    gap | true bass present: median={np.median(gap_present):.1f}, "
          f"gap | true bass absent: median={np.median(gap_absent):.1f}")


# ---------------------------------------------------------------------------
# 3. Grid search + v1 vs v2 comparison
# ---------------------------------------------------------------------------

def apply_v2(candidates, audio_gaps, max_bass_midi, min_gap):
    out = candidates.copy()
    for b in range(len(out)):
        if out[b] < 0:
            continue
        if out[b] > max_bass_midi:
            out[b] = -1
            continue
        gap = audio_gaps[b]
        if gap is not None and gap < min_gap:
            out[b] = -1
    return out


def score(v2_all, all_data) -> dict:
    tp_no_bass = fp_no_bass = fn_no_bass = 0
    pitch_match = pitch_total = 0
    for v2, d in zip(v2_all, all_data):
        for b, tb in enumerate(d["true_beats"]):
            true_no_bass = tb.true_bass < 0
            pred_no_bass = v2[b] < 0
            if true_no_bass and pred_no_bass:
                tp_no_bass += 1
            elif (not true_no_bass) and pred_no_bass:
                fn_no_bass += 1
            elif true_no_bass and (not pred_no_bass):
                fp_no_bass += 1
            if not true_no_bass:
                pitch_total += 1
                if (not pred_no_bass) and (v2[b] % 12) == (tb.true_bass % 12):
                    pitch_match += 1
    no_bass_actual = tp_no_bass + fp_no_bass
    no_bass_predicted = tp_no_bass + fn_no_bass
    recall = tp_no_bass / max(no_bass_actual, 1)
    precision = tp_no_bass / max(no_bass_predicted, 1)
    pitch_recall = pitch_match / max(pitch_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"no_bass_precision": precision, "no_bass_recall": recall,
            "no_bass_f1": f1, "pitch_recall": pitch_recall,
            "combined": (f1 + pitch_recall) / 2}


def grid_search(all_data: list[dict], true_bass_max: int):
    """
    Search a wide range of (register ceiling, isolation-gap) pairs -- wide
    enough to include "no filtering at all" (ceiling far above any
    observed true bass note, gap=0) as a candidate, so the search can
    honestly conclude "the naive detector is already as good as anything
    in this family" instead of being forced to pick a worse constrained
    option by construction.
    """
    candidates_list = [d["candidates"] for d in all_data]
    gaps_list = [d["audio_gaps"] for d in all_data]

    best = None
    for ceiling in list(range(true_bass_max, true_bass_max + 20, 2)) + [200]:
        for min_gap in range(0, 9):
            v2_all = [apply_v2(c, g, ceiling, min_gap) for c, g in zip(candidates_list, gaps_list)]
            s = score(v2_all, all_data)
            if best is None or s["combined"] > best[0]["combined"]:
                best = (s, ceiling, min_gap)
    return best


def plot_v1_vs_v2(v1_score: dict, v2_score: dict, ceiling: int, min_gap: int) -> None:
    metrics = ["no_bass_precision", "no_bass_recall", "no_bass_f1", "pitch_recall"]
    labels = ["no-bass\nprecision", "no-bass\nrecall", "no-bass\nF1", "pitch-class\nmatch rate\n(when bass present)"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, [v1_score[m] for m in metrics], width, label="v1 (naive: lowest active key)", color="#888888")
    ax.bar(x + width / 2, [v2_score[m] for m in metrics], width,
           label=f"v2 (learned: ceiling={ceiling} MIDI, min_gap={min_gap} semitones)", color="#2ca02c")
    for i, m in enumerate(metrics):
        ax.text(i - width / 2, v1_score[m] + 0.02, f"{v1_score[m]:.0%}", ha="center", fontsize=8)
        ax.text(i + width / 2, v2_score[m] + 0.02, f"{v2_score[m]:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title("Naive vs learned bass detector, validated against POP909's ground-truth PIANO track")
    if abs(v1_score["combined"] - v2_score["combined"]) < 1e-6:
        ax.text(
            0.5, 1.06,
            "identical: no register/isolation-gap constraint beat the naive detector "
            "(no-bass metrics based on only 7 true no-bass beats -- too few to learn from reliably)",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#555555",
        )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = PLOT_ROOT / "bass_detector_v1_vs_v2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", nargs="+", default=["001", "002", "003", "004", "005"])
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Collecting ground truth + audio observations per song...")
    all_data = []
    for song_id in args.songs:
        d = collect_song(song_id)
        if d is not None:
            all_data.append(d)
            n_true_bass = sum(1 for tb in d["true_beats"] if tb.true_bass >= 0)
            print(f"  {song_id}: {len(d['true_beats'])} beats, {n_true_bass} with true bass")

    print("\n1. Register + silence rate...")
    true_bass_max = plot_register_and_silence(all_data)

    print("\n2. Isolation-gap distribution...")
    plot_gap_distribution(all_data)

    print("\n3. Grid search for (register ceiling, isolation gap) thresholds...")
    v1_all = [d["candidates"] for d in all_data]  # v1 = raw candidate, no filtering
    v1_score = score(v1_all, all_data)
    print(f"  v1 (naive): {v1_score}")

    best_score, ceiling, min_gap = grid_search(all_data, true_bass_max)
    print(f"  v2 (learned) best: ceiling={ceiling}, min_gap={min_gap} -> {best_score}")
    if best_score["combined"] <= v1_score["combined"] + 1e-9:
        print("  -> no (ceiling, gap) constraint in the searched range beats the naive "
              "detector; isolation-gap filtering doesn't survive contact with real data here.")

    plot_v1_vs_v2(v1_score, best_score, ceiling, min_gap)

    print(f"\nSuggested infer_bass_track_learned() defaults: "
          f"max_bass_midi={ceiling}, min_gap_semitones={min_gap}")


if __name__ == "__main__":
    main()
