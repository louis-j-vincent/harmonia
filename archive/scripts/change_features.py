"""Per-beat features for the chord-change detector + duration-aware boundary decode.

Shared by train_change_detector.py (fit) and pipeline_v0.py (infer). All features
are computable from audio alone (Basic Pitch per-beat activations); the labels
(GT chord changes) are only needed at training time.
"""

from __future__ import annotations

import numpy as np

MIDI_START = 21
FEATURE_NAMES = ["nov_prev", "nov_next", "nov_win", "bass_nov", "bass_changed",
                 "onset_density", "onset_ratio", "treble_nov"]


def _chroma(v88, lo=0, hi=200):
    c = np.zeros(12)
    for k in range(88):
        m = k + MIDI_START
        if lo <= m < hi:
            c[m % 12] += v88[k]
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def beat_change_features(onset_beats: np.ndarray) -> np.ndarray:
    """(n_beats, F) per-beat change features from the onset beat-activation matrix."""
    n = len(onset_beats)
    ch = np.stack([_chroma(onset_beats[b]) for b in range(n)])
    bass = np.stack([_chroma(onset_beats[b], 0, 52) for b in range(n)])
    treb = np.stack([_chroma(onset_beats[b], 60, 200) for b in range(n)])
    dens = onset_beats.sum(1)
    dens_run = np.convolve(dens, np.ones(5) / 5, mode="same") + 1e-9
    bass_pc = np.array([int(bass[b].argmax()) if bass[b].sum() > 1e-9 else -1 for b in range(n)])
    F = []
    for b in range(n):
        prev = ch[b - 1] if b > 0 else ch[b]
        nxt = ch[b + 1] if b < n - 1 else ch[b]
        win = ch[max(0, b - 2):b].mean(0) if b >= 2 else ch[b]
        win = win / (np.linalg.norm(win) + 1e-9)
        F.append([
            1 - float(ch[b] @ prev),                              # novelty vs prev
            1 - float(ch[b] @ nxt),                               # novelty vs next
            1 - float(ch[b] @ win),                               # novelty vs recent window
            1 - float(bass[b] @ (bass[b - 1] if b > 0 else bass[b])),   # bass novelty
            float(b > 0 and bass_pc[b] != bass_pc[b - 1] and bass_pc[b] >= 0),  # bass PC changed
            float(dens[b]),                                       # onset density
            float(dens[b] / dens_run[b]),                         # onset density spike
            1 - float(treb[b] @ (treb[b - 1] if b > 0 else treb[b])),   # treble novelty
        ])
    return np.array(F, dtype=np.float32)


def decode_boundaries(change_proba: np.ndarray, min_gap: int = 2,
                      threshold: float = 0.5) -> list[int]:
    """Place boundaries at high P(change) beats, respecting a minimum segment length
    (the duration prior: jazz chords last ~2 beats). Greedy by descending proba."""
    n = len(change_proba)
    chosen = [0]
    for b in np.argsort(-change_proba):
        if b == 0 or change_proba[b] < threshold:
            continue
        if all(abs(int(b) - c) >= min_gap for c in chosen):
            chosen.append(int(b))
    return sorted(chosen) + [n]
