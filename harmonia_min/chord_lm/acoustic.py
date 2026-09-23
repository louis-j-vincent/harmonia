"""harmonia_min/chord_lm/acoustic.py — the chord model's own posterior, per slot.

The gate in `intervene.py` is binary: the LM speaks only where the pipeline's
scalar confidence falls below a threshold. That throws away almost everything
the acoustic model knows — it keeps one number per chord and discards the shape
of the distribution behind it.

This module exposes that shape so the three sources can be combined properly:

    log P(chord) = w_ac * log P_acoustic + log P_LM + w_bass * log P_bass

which is what "the LM should step in where the chord model is unsure" means
once you stop thresholding: where the acoustic posterior is flat the LM decides,
where it is peaked the LM cannot overrule it, and no threshold has to be chosen.
It is also the joint decode blog post 17 concluded the progression prior needed
("its information belongs in a joint decode as a real transition factor"), after
the same information failed as a post-hoc reranker.

WHAT THE TRIAD PLANE CAN AND CANNOT SAY. musx's 73 columns are N plus
root x {maj, min, sus4, sus2, dim, aug} — there is no dominant and no
half-diminished, because those are a triad PLUS a seventh, and the seventh lives
in a different plane. So this term constrains the ROOT and the TRIAD, and is
deliberately indifferent between:

    maj  vs  dom      (same major triad)
    dim  vs  hdim     (same diminished triad)

which is precisely the distinction the LM exists to make. The division of
labour falls out of the representation rather than being imposed on it.
"""
from __future__ import annotations

import numpy as np

from .vocab import FAMILIES, NC, VOCAB_SIZE, chord_id

#: our family -> musx triad type(s). Type index is 1-based, matching
#: `musx.frame_posteriors`: {1 maj, 2 min, 3 sus4, 4 sus2, 5 dim, 6 aug}.
#: `sus` collects both suspensions; `dom` rides on the major triad and `hdim`
#: on the diminished one, so each of those pairs shares an acoustic column.
_FAMILY_TYPES = {
    "maj": (1,), "dom": (1,), "min": (2,), "sus": (3, 4),
    "dim": (5,), "hdim": (5,), "aug": (6,),
}


def triad_matrix() -> np.ndarray:
    """(73, VOCAB_SIZE) — P(triad-plane column | chord token)."""
    M = np.zeros((73, VOCAB_SIZE), dtype=np.float32)
    for root in range(12):
        for fam in FAMILIES:
            tok = chord_id(root, fam)
            types = _FAMILY_TYPES[fam]
            for t in types:
                M[1 + (t - 1) * 12 + root, tok] = 1.0 / len(types)
    M[0, NC] = 1.0
    return M


_M = triad_matrix()


def token_logp(triad_vec: np.ndarray, *, floor: float = 1e-4) -> np.ndarray:
    """(73,) triad posterior -> (VOCAB_SIZE,) log P(acoustic | token).

    `floor` keeps an unheard chord unlikely rather than impossible: one bad
    frame must not be able to veto the grammar outright.
    """
    v = np.asarray(triad_vec, dtype=np.float32)
    s = v.sum()
    if s <= 0:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)
    return np.log(np.maximum(_M.T @ (v / s), floor))


def slot_triad(triad_plane: np.ndarray, bar_grid, n_slots: int,
               slots_per_bar: int = 2, frame_dt: float | None = None
               ) -> np.ndarray:
    """(n_slots, 73) — the triad posterior averaged over each half-bar."""
    from harmonia_min import musx as _musx
    dt = frame_dt or _musx.FRAME_DT
    out = np.zeros((n_slots, 73), dtype=np.float32)
    n_frames = triad_plane.shape[0]
    for i in range(n_slots):
        bar, slot = divmod(i, slots_per_bar)
        if bar + 1 >= len(bar_grid):
            out[i] = 1.0 / 73
            continue
        bw = bar_grid[bar + 1] - bar_grid[bar]
        t0 = bar_grid[bar] + slot * bw / slots_per_bar
        t1 = t0 + bw / slots_per_bar
        a = max(0, int(round(t0 / dt)))
        b = min(n_frames, int(round(t1 / dt)))
        out[i] = triad_plane[a:b].mean(0) if b > a else 1.0 / 73
    return out


def joint(lm_logp: np.ndarray, triad_slots: np.ndarray | None,
          bass_slots: np.ndarray | None, tokens: list[int],
          absolute: list[int | None], *, w_ac: float, w_bass: float
          ) -> np.ndarray:
    """(T, V) joint log-posterior over the three sources, renormalised.

    REP is scored with the acoustic and bass evidence of the chord it stands
    for: it means "the same chord is still sounding", so the evidence for it is
    the evidence for that chord, not a category of its own.
    """
    from . import bass as _bass
    from .vocab import N_CHORD, REP
    out = np.array(lm_logp, dtype=np.float32, copy=True)
    T = out.shape[0]
    for i in range(T):
        prev = absolute[i - 1] if i > 0 else None
        held = prev if (prev is not None and 0 <= prev < N_CHORD) else None
        if triad_slots is not None and w_ac > 0:
            term = token_logp(triad_slots[i])
            if held is not None:
                term[REP] = term[held]
            out[i] += w_ac * term
        if bass_slots is not None and w_bass > 0:
            term = _bass.token_logp(bass_slots[i])
            if held is not None:
                term[REP] = term[held]
            out[i] += w_bass * term
    out -= out.max(axis=1, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=1, keepdims=True)
    return np.log(np.maximum(out, 1e-12))
