"""Chord proximity graph built from db.jsonl bigrams.

Exposes a 12×12 root-transition count matrix (transposition-invariant, keyed on
interval mod 12) and a neighbour lookup combining empirical transition probs with
circle-of-fifths distance as a prior for unseen transitions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO / "data" / "accomp_db" / "db.jsonl"

# Circle-of-fifths distance (min of clockwise / counter-clockwise steps in 5ths)
_COF = np.zeros(12)
for _i in range(12):
    # position on circle of fifths: (i * 7) % 12 maps semitones to fifths position
    _steps_cw = 0
    _pos = 0
    while _pos != _i:
        _pos = (_pos + 7) % 12
        _steps_cw += 1
    _COF[_i] = min(_steps_cw, 12 - _steps_cw)

COF_DISTANCE = _COF.copy()  # COF_DISTANCE[interval] = min fifths distance


def build_transition_matrix(db_path: str | Path | None = None) -> np.ndarray:
    """Build a 12×12 root-interval transition count matrix from db.jsonl.

    Matrix[i][j] = count of transitions where prev_root=i, curr_root=j.
    But we store it transposition-invariant: shape (12,) keyed on
    (curr_root - prev_root) % 12. Returns shape (12, 12) for generality:
    row i = transitions FROM root i, col j = transitions TO root j.
    Actually returns a (12,12) matrix where entry [r1][r2] = count of
    transitions from any chord with root r1 to any chord with root r2,
    across the whole corpus.
    """
    if db_path is None:
        db_path = DB_PATH
    db_path = Path(db_path)

    # We'll build an interval histogram (12,) for efficiency,
    # then expand to (12,12) assuming transposition invariance.
    interval_counts = np.zeros(12, dtype=np.float64)

    with open(db_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            timeline = rec.get("chord_timeline", [])
            if len(timeline) < 2:
                continue

            prev_root = _parse_root(timeline[0].get("mma", ""))
            if prev_root is None:
                continue

            for entry in timeline[1:]:
                curr_root = _parse_root(entry.get("mma", ""))
                if curr_root is None:
                    continue
                if curr_root == prev_root:
                    prev_root = curr_root
                    continue  # skip self-transitions
                interval = (curr_root - prev_root) % 12
                interval_counts[interval] += 1
                prev_root = curr_root

    # Expand to 12x12: mat[r1][r2] = interval_counts[(r2-r1)%12]
    mat = np.zeros((12, 12), dtype=np.float64)
    for r1 in range(12):
        for r2 in range(12):
            mat[r1, r2] = interval_counts[(r2 - r1) % 12]
    return mat


# Note name mapping for MMA chord parsing
_NOTE_MAP = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}


def _parse_root(mma: str) -> int | None:
    """Extract root pitch class (0-11) from an MMA chord string like 'Bbm7'."""
    if not mma:
        return None
    i = 0
    if i >= len(mma) or mma[i] not in _NOTE_MAP:
        return None
    root = _NOTE_MAP[mma[i]]
    i += 1
    if i < len(mma) and mma[i] == '#':
        root = (root + 1) % 12
        i += 1
    elif i < len(mma) and mma[i] == 'b':
        root = (root - 1) % 12
        i += 1
    return root


class ChordGraph:
    """Chord proximity graph for neighbour lookup."""

    def __init__(self, db_path: str | Path | None = None):
        self.trans_mat = build_transition_matrix(db_path)
        # Normalise rows to get transition probabilities (interval-based)
        self._interval_probs = np.zeros(12)
        total = self.trans_mat[0].sum()  # all rows are identical (transposition-invariant)
        if total > 0:
            self._interval_probs = self.trans_mat[0] / total

    def neighbours(self, root_pc: int, fam_idx: int, k: int = 8) -> list[tuple[int, int, float]]:
        """Return top-k (root_pc, fam_idx, score) neighbours.

        Scoring combines:
          (a) empirical root-transition probability
          (b) circle-of-fifths distance prior (closer = higher score)
          (c) same-root quality changes (parallel quality neighbours)

        Returns list sorted by descending score, excluding the input chord itself.
        """
        candidates = []
        n_families = 5

        for r2 in range(12):
            interval = (r2 - root_pc) % 12
            if interval == 0:
                # Same root, different quality
                for f2 in range(n_families):
                    if f2 == fam_idx:
                        continue
                    # Quality change score: modest (empirical: ~4% of transitions are same-root)
                    score = 0.04 + 0.02 * (1.0 / (1.0 + COF_DISTANCE[0]))
                    candidates.append((r2, f2, score))
            else:
                # Root motion
                trans_prob = self._interval_probs[interval]
                cof_prior = 1.0 / (1.0 + COF_DISTANCE[interval])
                # Blend: 70% empirical, 30% CoF prior
                score = 0.7 * trans_prob + 0.3 * cof_prior * 0.2
                for f2 in range(n_families):
                    candidates.append((r2, f2, score))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[2])
        return candidates[:k]


if __name__ == "__main__":
    print("Building transition matrix from", DB_PATH)
    mat = build_transition_matrix()
    # interval counts (row 0 = all intervals since transposition-invariant)
    intervals = mat[0]
    total = intervals.sum()
    print(f"\nTotal non-self transitions: {int(total)}")
    print("\nTop-5 root transitions (by interval):")
    NOTE = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    ranked = sorted(range(12), key=lambda i: -intervals[i])
    for rank, iv in enumerate(ranked[:5], 1):
        pct = 100 * intervals[iv] / total if total > 0 else 0
        # Show as "up by N semitones" with note name example
        print(f"  {rank}. interval +{iv} (e.g. C->{NOTE[iv]}): "
              f"{int(intervals[iv])} ({pct:.1f}%)")
