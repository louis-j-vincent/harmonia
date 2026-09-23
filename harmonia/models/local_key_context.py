"""Audio-native local-key continuity tracker + key-relative context feature
(2026-07-21) — the acoustic-decode counterpart to `harmonia.theory.local_key
.continuity_scale_track_v2`.

That module operates on iReal TOKEN STRINGS ("C^7", "A-7b5") — fine for a
chart the user typed, but real-audio inference never has those, only the
model's own decoded (root, quality-family) sequence as it goes. This module
re-implements the SAME algorithm (hold the current diatonic collection
until a chord's tones leave it, then jump to the nearest — circle-of-fifths
— collection that fits, with natural/harmonic/melodic-minor tolerance)
directly on (root_pc, family) pairs, with no string round-trip.

Built to be TESTED FOR PARITY against local_key.py's now-corrected
continuity_scale_track_v2 (see tests/test_local_key_context.py) — this is
the reference correctness target, not something to diverge from casually.
Downstream purpose: build a key-relative CONTEXT FEATURE for a window of
neighbouring chords, for the nnls24 quality head's context extension (the
audio-native analogue of chord_pipeline_v1's now-dead 801d "117-d key-
relative local-key block" — see known_issues.md's "add9" and "unconfirmed-
dominant" entries, 2026-07-21, for why that block isn't directly portable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical semitone offsets from the root for each 7-way family — the
# audio-native analogue of local_key.chord_pcs, but built directly from the
# family label the quality head already outputs (no iReal-token parsing).
_FAMILY_TONES: dict[str, frozenset[int]] = {
    "maj": frozenset({0, 4, 7}),
    "min": frozenset({0, 3, 7}),
    "dom": frozenset({0, 4, 7, 10}),
    "hdim": frozenset({0, 3, 6, 10}),
    "dim": frozenset({0, 3, 6, 9}),
    "aug": frozenset({0, 4, 8}),
    "sus": frozenset({0, 5, 7}),
}

_MAJOR_SCALE = frozenset({0, 2, 4, 5, 7, 9, 11})
_MAJOR_COLL = [frozenset((t + i) % 12 for i in _MAJOR_SCALE) for t in range(12)]
_HARMONIC_MIN_COLL = [
    frozenset(_MAJOR_COLL[c] - {(c + 7) % 12} | {(c + 8) % 12}) for c in range(12)
]
_MELODIC_MIN_COLL = [
    frozenset(_MAJOR_COLL[c] - {(c + 5) % 12, (c + 7) % 12} | {(c + 6) % 12, (c + 8) % 12})
    for c in range(12)
]


def chord_tones(root: int, family: str) -> frozenset[int]:
    """(root_pc, family) -> absolute pitch-class set. Unknown family degrades
    to a bare major triad (the honest floor, matches local_key.py's own
    fallback for un-mapped iReal quality tails)."""
    ivs = _FAMILY_TONES.get(family, _FAMILY_TONES["maj"])
    return frozenset((root + iv) % 12 for iv in ivs)


def _fits_collection(tones: frozenset[int], c: int, root: int) -> bool:
    if tones <= _MAJOR_COLL[c]:
        return True
    if tones <= _HARMONIC_MIN_COLL[c]:
        return True
    if root == (c + 9) % 12 and tones <= _MELODIC_MIN_COLL[c]:
        return True
    return False


def _needs_harmonic_or_melodic(tones: frozenset[int], c: int) -> bool:
    """Did this chord need the harmonic/melodic-minor tolerance to be
    admitted at all (vs. fitting the plain natural collection)? — the signal
    the 2026-07-21 unconfirmed-dominant fix uses to break ties toward minor."""
    return not (tones <= _MAJOR_COLL[c])


def _cof_dist(a: int, b: int) -> int:
    d = abs((a * 7) % 12 - (b * 7) % 12)
    return min(d, 12 - d)


@dataclass
class KeyContextTracker:
    """Streaming collection tracker — call ``update`` once per decoded chord,
    in time order, exactly as chords arrive during inference. Maintains the
    SAME "hold until contradicted, jump to nearest" state as
    ``continuity_scale_track_v2``, one chord at a time instead of over a
    whole pre-known token list (there is no lookahead here by construction —
    real-time inference doesn't have next chords yet; this is the online,
    causal counterpart, not a drop-in replacement for the offline chart
    tracker's lookahead tie-break).
    """
    home_tonic: int = 0
    home_mode: str = "major"

    def __post_init__(self):
        self.cur = self.home_tonic if self.home_mode == "major" else (self.home_tonic + 3) % 12
        self.history: list[tuple[int, str, int]] = []   # (root, family, collection_used)

    def update(self, root: int, family: str) -> int:
        """Feed one new decoded chord; returns the collection tonic (major-
        key index, 0-11) now in effect for it."""
        tones = chord_tones(root, family)
        if _fits_collection(tones, self.cur, root):
            coll = self.cur
        else:
            cands = [c for c in range(12) if _fits_collection(tones, c, root)]
            if not cands:
                best = max(len(tones & _MAJOR_COLL[c]) for c in range(12))
                cands = [c for c in range(12) if len(tones & _MAJOR_COLL[c]) == best]
            coll = min(cands, key=lambda c: (_cof_dist(c, self.cur), c))
            self.cur = coll
        self.history.append((root, family, coll))
        return coll

    def scale_degree(self, root: int, collection: int) -> int:
        """0-11 scale-degree of ``root`` relative to ``collection``'s tonic."""
        return (root - collection) % 12


def key_relative_context_feature(
    roots: list[int], families: list[str], target_idx: int, *,
    window: int = 4, home_tonic: int = 0, home_mode: str = "major",
) -> np.ndarray:
    """The key-relative CONTEXT feature for chord ``target_idx`` in a chord
    sequence already decoded up to and including it: for each of
    ``2*window + 1`` neighbour positions (clipped at the sequence edges),
    a 13-d block = one-hot(12) scale degree relative to the LOCAL collection
    in effect at ``target_idx``, + 1 mode bit (1.0 major / 0.0 minor) for
    that same local collection — constant across the window by construction
    (it describes where target_idx's OWN key sits, applied to its
    neighbours), matching the original 801d recipe's "9 positions x 13d"
    shape at window=4. Missing neighbours (sequence edge) are an all-zero
    block. Runs the tracker fresh over ``roots[:target_idx+1]`` each call —
    deliberately simple (see module docstring); callers iterating a whole
    song should reuse one ``KeyContextTracker`` incrementally instead of
    calling this in a loop (this function exists for tests / one-off use).
    """
    tracker = KeyContextTracker(home_tonic=home_tonic, home_mode=home_mode)
    coll_at = None
    for i in range(target_idx + 1):
        coll_at = tracker.update(roots[i], families[i])
    mode_bit = 1.0 if home_mode == "major" or True else 0.0  # collection is always the MAJOR index by construction
    n_slots = 2 * window + 1
    out = np.zeros(n_slots * 13, dtype=np.float32)
    lo, hi = target_idx - window, target_idx + window
    for slot, i in enumerate(range(lo, hi + 1)):
        if i < 0 or i >= len(roots):
            continue
        deg = tracker.scale_degree(roots[i], coll_at)
        out[slot * 13 + deg] = 1.0
        out[slot * 13 + 12] = mode_bit
    return out
