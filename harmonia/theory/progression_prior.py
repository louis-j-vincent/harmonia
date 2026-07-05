"""Learned chord-progression prior for the HMM transition matrix.

The scale-relative progression "language model" (see scripts/train_progression_lm.py),
reduced to the order-1 form the HMM can use: a bigram P(next | prev) over
(scale-degree, family) states, fitted once from the corpus and applied as a
low-weight multiplicative boost on the hand-coded transition matrix — the same
"priors regularize, don't override" design as every other prior in the codebase.

Wired into ChordInferrer via `progression_prior_weight` (default 0.0 = no change).
Fit/refresh the table with:  python -m harmonia.theory.progression_prior
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harmonia.theory.chord_tree import family_of
from harmonia.theory.chord_vocabulary import ChordQuality

FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
_FI = {f: i for i, f in enumerate(FAMILIES)}
# chord_tree.Family enum values ("maj"/"min"/…) → the long names used here
_FAM_ENUM_TO_NAME = {"maj": "major", "min": "minor", "dim": "diminished",
                     "aug": "augmented", "sus": "suspended", "N": None}
_TABLE_PATH = Path(__file__).parent / "progression_bigram.json"


def _state(degree: int, family: str) -> int:
    return degree * len(FAMILIES) + _FI[family]


def fit(db_path: Path, out_path: Path = _TABLE_PATH) -> None:
    """Fit P(next|prev) over (degree, family) from the corpus and save log-probs."""
    from harmonia.data.ireal_corpus import chord_root_pc  # noqa: PLC0415

    # bucket→family, mirroring build_audio_chord_features
    from scripts.analyze_accomp_priors import parse_key  # type: ignore  # noqa: PLC0415
    fam_of_bucket = {
        "maj": "major", "maj7": "major", "6": "major", "dom7": "major", "dom7alt": "major",
        "min": "minor", "min7": "minor", "m6": "minor", "minmaj7": "minor",
        "dim": "diminished", "dim7": "diminished", "m7b5": "diminished",
        "aug": "augmented", "aug7": "augmented", "augmaj7": "augmented",
        "sus2": "suspended", "sus4": "suspended", "7sus4": "suspended",
    }
    from scripts.analyze_accomp_emission import parse_chord  # type: ignore  # noqa: PLC0415

    ns = 12 * len(FAMILIES)
    counts = np.ones((ns, ns)) * 0.2   # Laplace
    for line in open(db_path):
        rec = json.loads(line)
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic = k[0]
        seq = []
        for ev in rec["chord_timeline"]:
            p = parse_chord(ev["mma"])
            if p is None or p[1] not in fam_of_bucket:
                continue
            seq.append(_state((p[0] - tonic) % 12, fam_of_bucket[p[1]]))
        for a, b in zip(seq, seq[1:]):
            if a != b:
                counts[a, b] += 1
    logp = np.log(counts / counts.sum(1, keepdims=True))
    out_path.write_text(json.dumps({"families": FAMILIES, "logp": logp.tolist()}))
    print(f"Fitted progression bigram ({ns} states) → {out_path}")


def load(path: Path = _TABLE_PATH) -> np.ndarray:
    """(ns, ns) log P(next|prev) over (degree, family) states."""
    return np.array(json.loads(path.read_text())["logp"], dtype=np.float32)


def transition_log_boost(idx_to_chord, tonic: int, logp: np.ndarray) -> np.ndarray:
    """(C, C) additive log-boost for the HMM transition matrix, keyed on this tonic."""
    C = len(idx_to_chord)
    state = np.full(C, -1)
    for i, (root, q) in enumerate(idx_to_chord):
        fam = _FAM_ENUM_TO_NAME.get(family_of(q).value)
        if q == ChordQuality.NO_CHORD or fam is None:
            continue
        state[i] = _state((root - tonic) % 12, fam)
    boost = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        if state[i] < 0:
            continue
        for j in range(C):
            if state[j] >= 0:
                boost[i, j] = logp[state[i], state[j]]
    return boost


if __name__ == "__main__":
    import sys
    fit(Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2] / "data" / "accomp_db" / "db.jsonl")
