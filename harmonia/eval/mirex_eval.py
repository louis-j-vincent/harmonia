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

from harmonia.theory.chord_vocabulary import SEMITONE_NAMES, ChordQuality

logger = logging.getLogger(__name__)

# ChordQuality -> mir_eval shorthand (mir_eval.chord.QUALITIES). mir_eval has
# no shorthand for altered/suspended-7th qualities, so those fall back to the
# closest supported quality (documented inline) rather than an invalid label.
_QUALITY_TO_MIREVAL: dict[ChordQuality, str] = {
    ChordQuality.MAJOR: "maj",
    ChordQuality.MINOR: "min",
    ChordQuality.DIMINISHED: "dim",
    ChordQuality.AUGMENTED: "aug",
    ChordQuality.SUS2: "sus2",
    ChordQuality.SUS4: "sus4",
    ChordQuality.MAJ7: "maj7",
    ChordQuality.MIN7: "min7",
    ChordQuality.DOM7: "7",
    ChordQuality.MIN_MAJ7: "minmaj7",
    ChordQuality.HALF_DIM7: "hdim7",
    ChordQuality.DIM7: "dim7",
    ChordQuality.AUG_MAJ7: "aug",       # no augmented-major-7th shorthand
    ChordQuality.AUG7: "aug",           # no augmented-7th shorthand
    ChordQuality.DOM7SUS4: "sus4",      # no dominant-7-sus4 shorthand
    ChordQuality.MAJ9: "maj9",
    ChordQuality.MIN9: "min9",
    ChordQuality.DOM9: "9",
    ChordQuality.DOM7B9: "7",           # no altered-9th shorthand
    ChordQuality.DOM7S9: "7",
    ChordQuality.DOM9SUS4: "sus4",
    ChordQuality.MAJ9S11: "maj9",
    ChordQuality.MIN11: "min11",
    ChordQuality.DOM7S11: "7",
    ChordQuality.DOM7B9S11: "7",
    ChordQuality.MAJ13: "maj13",
    ChordQuality.MIN13: "min13",
    ChordQuality.DOM13: "13",
    ChordQuality.DOM13B9: "13",
}
# Sort longest-first so e.g. "C#" is matched before "C".
_ROOT_NAMES_BY_LENGTH = sorted(SEMITONE_NAMES, key=len, reverse=True)


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
    """Aggregated MIREX scores over a dataset.

    Currently UNCALLED: its only consumer was `evaluate_pop909`, deleted
    2026-07-30 with `HarmoniaPipeline` (it took a pipeline instance). Kept
    because it is a generic MIREXScore aggregator with no pipeline coupling.
    Note `harmonia.eval.accuracy_score` has its own, separate `DatasetScore`
    — that is the one the live Brick-0 scorer uses.
    """
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
    """Convert a Harmonia chord label (e.g. "Cmaj7", "Dø7", "Bbmin") to
    mir_eval Harte notation (e.g. "C:maj7", "D:hdim7", "Bb:min").

    Harmonia labels are built as f"{root_name}{quality.value}" (see
    chord_vocabulary.chord_label), so root and quality are parsed exactly
    against the vocabulary rather than guessed with endswith() heuristics —
    a previous version of this function checked a generic "7" suffix before
    more specific ones like "mMaj7"/"°7"/"ø7", silently mangling any
    minor-major-7th, diminished-7th, or half-diminished-7th chord into an
    invalid Harte string and crashing mir_eval for the whole song.
    """
    if label == "N":
        return "N"

    root_name = next((r for r in _ROOT_NAMES_BY_LENGTH if label.startswith(r)), None)
    if root_name is None:
        logger.debug(f"Unparseable root in label {label!r}, defaulting to maj")
        return f"{label}:maj"

    quality_str = label[len(root_name):]
    if quality_str == "":
        return f"{root_name}:maj"

    quality = next((q for q in ChordQuality if q.value == quality_str), None)
    if quality is None:
        logger.debug(f"Unknown quality {quality_str!r} in label {label!r}, defaulting to maj")
        return f"{root_name}:maj"

    mireval_quality = _QUALITY_TO_MIREVAL.get(quality, "maj")
    return f"{root_name}:{mireval_quality}"


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

        # mir_eval.chord.evaluate() already merges intervals, applies MIREX's
        # duration weighting, and computes every comparison metric in one pass.
        scores = mec.evaluate(
            reference_intervals, reference_labels,
            pred_intervals, pred_labels,
        )

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
