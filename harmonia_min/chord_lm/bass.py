"""harmonia_min/chord_lm/bass.py — the bass as evidence for the chord LM.

Measured 2026-08-02 (Louis's ear on three contested half-bars, then the corpus):
**85% of the LM's disagreements change the ROOT**, and on the 19 contested
half-bars of our five charts the bass plane backs the chart's root 16 times out
of 19 — contradicting the LM's proposed root almost everywhere. The one case
where the bass sided with the LM is the one case Louis's ear did too.

So the LM must be conditioned on the bass, not compete with it:

    root / bass          <- acoustic; the bass plane already knows
    quality + family     <- the LM; nothing else has the grammar
    harmonic rhythm      <- the LM (change vs hold), 91.1% accurate

This module is level 1 of docs/chord_lm_bass_extension_2026-08-02.md: fuse at
decode time, no retraining.

    score(chord) = log P_LM(chord | context) + w * log P_bass(bass_of(chord))

THE BASS PLANE'S COLUMN CONVENTION IS DOCUMENTED NOWHERE and was determined
empirically: **column 0 is N, column i>=1 is pitch class i-1** (93.8% agreement
with the decoded triad root on Let It Be; the alternative reading scores 0.2%).

`bass_of(chord)` is deliberately NOT a hard root lookup. A C/E sounds an E, and
pop plays first inversions constantly; scoring only the root would punish every
chord played the way the record plays it. So the term is a small mixture over
the chord's own tones, root-weighted.

The mapping being explicit matters: `vocab.py` roots are FUNCTIONAL (D-7/A is a
D:min) while the bass plane hears the SOUNDING bass. Fusing the two silently is
CLAUDE.md error-pattern #3 — a disagreement blamed on the model when the two
sides encode different things.
"""
from __future__ import annotations

import numpy as np

from .vocab import FAMILIES, NC, N_CHORD, VOCAB_SIZE, chord_id

#: How much of a chord's bass mass sits on its root, third and fifth.
#: Root-dominant because most chords are in root position; the two small terms
#: exist so an inversion is merely less likely, never impossible.
BASS_WEIGHTS = (0.80, 0.10, 0.10)

#: Semitones of (third, fifth) above the root, per family.
_TONES = {
    "maj": (4, 7), "min": (3, 7), "dom": (4, 7), "dim": (3, 6),
    "hdim": (3, 6), "sus": (5, 7), "aug": (4, 8),
}


def chord_bass_matrix() -> np.ndarray:
    """(13, VOCAB_SIZE) — P(bass pitch class | chord token).

    Row 0 is the bass plane's N column; row 1+pc is pitch class pc. Columns for
    REP / PAD / BOS / EOS / MASK stay zero: REP has no bass of its own, it
    borrows the chord it stands for, and the caller resolves that.
    """
    M = np.zeros((13, VOCAB_SIZE), dtype=np.float32)
    w_r, w_3, w_5 = BASS_WEIGHTS
    for root in range(12):
        for fam in FAMILIES:
            tok = chord_id(root, fam)
            third, fifth = _TONES[fam]
            M[1 + root, tok] += w_r
            M[1 + (root + third) % 12, tok] += w_3
            M[1 + (root + fifth) % 12, tok] += w_5
    M[0, NC] = 1.0
    return M


_M = chord_bass_matrix()


def token_logp(bass_vec: np.ndarray, *, floor: float = 1e-4) -> np.ndarray:
    """(13,) bass posterior -> (VOCAB_SIZE,) log P(bass | token).

    `floor` keeps a chord whose bass the model did not hear merely unlikely
    rather than impossible — the bass plane is confident but not infallible, and
    a hard zero would let one bad frame veto the grammar outright.
    """
    b = np.asarray(bass_vec, dtype=np.float32)
    s = b.sum()
    if s <= 0:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    p = _M.T @ (b / s)
    return np.log(np.maximum(p, floor))


def slot_bass(bass_plane: np.ndarray, bar_grid, n_slots: int,
              slots_per_bar: int = 2, frame_dt: float | None = None
              ) -> np.ndarray:
    """(n_slots, 13) — the bass posterior averaged over each half-bar.

    `bar_grid` is the chart's own `barGrid`, so the slots line up with the token
    grid exactly. Slots past the end of the grid get a uniform row, which
    `token_logp` turns into a flat (uninformative) term rather than a wrong one.
    """
    from harmonia_min import musx as _musx
    dt = frame_dt or _musx.FRAME_DT
    out = np.zeros((n_slots, 13), dtype=np.float32)
    n_frames = bass_plane.shape[0]
    for i in range(n_slots):
        bar, slot = divmod(i, slots_per_bar)
        if bar + 1 >= len(bar_grid):
            out[i] = 1.0 / 13
            continue
        bw = bar_grid[bar + 1] - bar_grid[bar]
        t0 = bar_grid[bar] + slot * bw / slots_per_bar
        t1 = t0 + bw / slots_per_bar
        a = max(0, int(round(t0 / dt)))
        b = min(n_frames, int(round(t1 / dt)))
        out[i] = bass_plane[a:b].mean(0) if b > a else 1.0 / 13
    return out


def fuse(lm_logp: np.ndarray, bass_slots: np.ndarray, tokens: list[int],
         absolute: list[int | None], weight: float) -> np.ndarray:
    """(T, V) fused log-probs, renormalised.

    REP is scored with the bass of the chord it stands for — otherwise the one
    token that means "the same chord as before" would be the only one the bass
    could never support, and every hold would be fused away.
    """
    T = lm_logp.shape[0]
    out = np.array(lm_logp, dtype=np.float32, copy=True)
    if weight <= 0:
        return out
    from .vocab import REP
    for i in range(T):
        term = token_logp(bass_slots[i])
        prev = absolute[i - 1] if i > 0 else None
        if prev is not None and 0 <= prev < N_CHORD:
            term[REP] = term[prev]         # a hold sounds the previous bass
        out[i] += weight * term
    out -= out.max(axis=1, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=1, keepdims=True)
    return np.log(np.maximum(out, 1e-12))
