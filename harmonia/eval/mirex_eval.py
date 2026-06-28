"""
MIREX chord evaluation metrics.

Implements the standard MIREX (Music Information Retrieval Evaluation eXchange)
chord evaluation protocol, which compares predicted chord sequences against
reference annotations at multiple levels of strictness:

  root:      only root pitch class must match
  majmin:    root + major/minor distinction must match
  sevenths:  root + major/minor/dominant-7th distinction
  tetrads:   full quality match (root + all chord tones)

Reference: Mauch & Dixon (2010), "Approximate Note Transcription for the
Improved Identification of Difficult Chords", ISMIR.

We use mir_eval (Raffel et al., 2014) under the hood.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class MIREXScore:
    """Chord evaluation results at all MIREX levels."""
    root: float         # root-only accuracy (0–1)
    majmin: float       # major/minor accuracy
    sevenths: float     # seventh-chord accuracy
    tetrads: float      # full tetrad accuracy
    duration_s: float   # total scored duration

    def __repr__(self) -> str:
        return (
            f"MIREXScore("
            f"root={self.root:.3f}, "
            f"majmin={self.majmin:.3f}, "
            f"sevenths={self.sevenths:.3f}, "
            f"tetrads={self.tetrads:.3f}, "
            f"duration={self.duration_s:.1f}s)"
        )

    def summary_line(self) -> str:
        return (
            f"root={self.root:.1%}  majmin={self.majmin:.1%}  "
            f"7ths={self.sevenths:.1%}  tetrads={self.tetrads:.1%}  "
            f"[{self.duration_s:.0f}s]"
        )


@dataclass
class DatasetScore:
    """Aggregated MIREX scores over a dataset."""
    per_song: list[tuple[str, MIREXScore]] = field(default_factory=list)

    @property
    def macro_avg(self) -> MIREXScore:
        """Per-song macro average (each song weighted equally)."""
        if not self.per_song:
            return MIREXScore(0.0, 0.0, 0.0, 0.0, 0.0)
        scores = [s for _, s in self.per_song]
        return MIREXScore(
            root=float(np.mean([s.root for s in scores])),
            majmin=float(np.mean([s.majmin for s in scores])),
            sevenths=float(np.mean([s.sevenths for s in scores])),
            tetrads=float(np.mean([s.tetrads for s in scores])),
            duration_s=float(np.sum([s.duration_s for s in scores])),
        )

    @property
    def micro_avg(self) -> MIREXScore:
        """Duration-weighted micro average."""
        if not self.per_song:
            return MIREXScore(0.0, 0.0, 0.0, 0.0, 0.0)
        scores = [s for _, s in self.per_song]
        total_dur = sum(s.duration_s for s in scores)
        if total_dur == 0:
            return MIREXScore(0.0, 0.0, 0.0, 0.0, 0.0)
        w = np.array([s.duration_s / total_dur for s in scores])
        return MIREXScore(
            root=float(np.dot(w, [s.root for s in scores])),
            majmin=float(np.dot(w, [s.majmin for s in scores])),
            sevenths=float(np.dot(w, [s.sevenths for s in scores])),
            tetrads=float(np.dot(w, [s.tetrads for s in scores])),
            duration_s=total_dur,
        )

    def print(self) -> None:
        print(f"\n{'─'*60}")
        print(f"  MIREX Evaluation — {len(self.per_song)} songs")
        print(f"{'─'*60}")
        macro = self.macro_avg
        micro = self.micro_avg
        print(f"  Macro: {macro.summary_line()}")
        print(f"  Micro: {micro.summary_line()}")
        print(f"{'─'*60}")
        worst = sorted(self.per_song, key=lambda x: x[1].majmin)[:5]
        print("  5 hardest songs (lowest majmin):")
        for name, score in worst:
            print(f"    {name:30s} {score.summary_line()}")
        print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def _chords_to_mireval_format(
    chords: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert Harmonia chord list → mir_eval interval/label format.

    Returns:
        intervals: (N, 2) array of [start_s, end_s]
        labels:    list of N chord label strings (Harte notation)
    """
    if not chords:
        return np.zeros((0, 2)), []

    intervals = np.array([[c["start_s"], c["end_s"]] for c in chords])
    labels = [c["label"] for c in chords]
    return intervals, labels


def _label_to_mireval(label: str) -> str:
    """Convert Harmonia chord label → mir_eval Harte notation."""
    if label == "N":
        return "N"
    # Harmonia uses e.g. "Cmaj7", "G7", "Bbmin7", "Dø7"
    # mir_eval expects "C:maj7", "G:7", "Bb:min7", "D:hdim7"
    replacements = [
        ("maj7", ":maj7"), ("min7", ":min7"), ("7", ":7"),
        ("maj", ":maj"), ("min", ":min"), ("dim7", ":dim7"),
        ("ø7", ":hdim7"), ("°7", ":dim7"), ("mMaj7", ":minmaj7"),
        ("sus4", ":sus4"), ("sus2", ":sus2"),
    ]
    for harmonia_suffix, mireval_suffix in replacements:
        if label.endswith(harmonia_suffix):
            root = label[: -len(harmonia_suffix)]
            return f"{root}{mireval_suffix}"
    # Bare root → major triad
    return f"{label}:maj"


def evaluate_song(
    predicted_chords: list[dict],
    reference_intervals: np.ndarray,
    reference_labels: list[str],
) -> MIREXScore:
    """
    Compute MIREX scores for a single song.

    Args:
        predicted_chords:    list of {"label", "start_s", "end_s"} dicts
                             (from ChordChart.chords)
        reference_intervals: (N, 2) array of reference chord [start, end] times
        reference_labels:    list of N reference chord labels (Harte notation)

    Returns:
        MIREXScore for this song.
    """
    try:
        import mir_eval.chord as mec
    except ImportError:
        raise ImportError("mir_eval not installed. Run: pip install mir_eval")

    if not predicted_chords:
        return MIREXScore(0.0, 0.0, 0.0, 0.0, 0.0)

    # Convert predicted chords to mir_eval format
    pred_intervals = np.array([[c["start_s"], c["end_s"]]
                                for c in predicted_chords])
    pred_labels = [_label_to_mireval(c["label"]) for c in predicted_chords]

    try:
        # Trim to common duration
        duration = min(pred_intervals[-1, 1], reference_intervals[-1, 1])

        scores = {}
        for metric_name, metric_fn in [
            ("root",     mec.root),
            ("majmin",   mec.majmin),
            ("sevenths", mec.sevenths),
            ("tetrads",  mec.tetrads),
        ]:
            durations, corr = mec.evaluate(
                reference_intervals, reference_labels,
                pred_intervals, pred_labels,
                metric_fn,
            )
            scores[metric_name] = float(np.sum(corr * durations) / np.sum(durations)) \
                if np.sum(durations) > 0 else 0.0

        return MIREXScore(
            root=scores["root"],
            majmin=scores["majmin"],
            sevenths=scores["sevenths"],
            tetrads=scores["tetrads"],
            duration_s=float(duration),
        )
    except Exception as e:
        logger.warning(f"mir_eval evaluation failed: {e}")
        return MIREXScore(0.0, 0.0, 0.0, 0.0, 0.0)


def evaluate_pop909(
    pipeline: "HarmoniaPipeline",
    songs: list["POP909Song"],
    max_songs: int | None = None,
) -> DatasetScore:
    """
    Evaluate Harmonia on the POP909 dataset.

    Args:
        pipeline:  HarmoniaPipeline instance.
        songs:     list of POP909Song objects (must have audio_path set).
        max_songs: limit evaluation to N songs (useful for quick tests).

    Returns:
        DatasetScore with per-song and aggregate MIREX scores.
    """
    from harmonia.pipeline import HarmoniaPipeline

    if max_songs is not None:
        songs = songs[:max_songs]

    songs_with_audio = [s for s in songs if s.audio_path is not None]
    logger.info(f"Evaluating {len(songs_with_audio)} songs with audio...")

    dataset_score = DatasetScore()

    for i, song in enumerate(songs_with_audio):
        logger.info(f"[{i+1}/{len(songs_with_audio)}] {song.song_id}")
        try:
            chart = pipeline.run(song.audio_path)

            # Build reference in mir_eval format from POP909 beat annotations
            if len(song.beat_times) == 0:
                logger.warning(f"  No beat times for {song.song_id}, skipping")
                continue

            # Convert beat-aligned chord events to time-aligned
            ref_intervals = []
            ref_labels = []
            for ev in song.chord_events:
                start_beat = int(ev.start_beat)
                end_beat = int(ev.end_beat)
                if start_beat >= len(song.beat_times):
                    continue
                t_start = song.beat_times[min(start_beat, len(song.beat_times) - 1)]
                t_end = (song.beat_times[min(end_beat, len(song.beat_times) - 1)]
                         if end_beat < len(song.beat_times)
                         else song.beat_times[-1] + 0.5)
                ref_intervals.append([t_start, t_end])
                ref_labels.append(f"{ev.label}")  # already Harte-ish

            if not ref_intervals:
                continue

            ref_intervals_arr = np.array(ref_intervals)
            score = evaluate_song(chart.chords, ref_intervals_arr, ref_labels)
            dataset_score.per_song.append((song.song_id, score))
            logger.info(f"  {score.summary_line()}")

        except Exception as e:
            logger.error(f"  Failed on {song.song_id}: {e}")

    return dataset_score
