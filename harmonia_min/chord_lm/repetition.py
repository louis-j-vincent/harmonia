"""harmonia_min/chord_lm/repetition.py — a within-song copy prior.

Measured 2026-08-01: on half-bars whose previous 4 bars already occurred
verbatim earlier in the same song (35.5% of all slots), the transformer scores
72.1% while a rule that just copies what followed last time scores 91.3%. Charts
are literally built out of repeats — AABA, chorus 2 = chorus 1 — so that is a
large, structural, and completely free signal the model is not fully taking.

This is the classic neural-cache remedy (Grave et al. 2017, "Improving Neural
Language Models with a Continuous Cache"), specialised to exact symbolic
matching, which is available here and is not available in text.

    P(slot) = (1 - lambda) * P_model(slot) + lambda * P_copy(slot)

`P_copy` backs off over match length: it looks for the longest earlier context
that matches the current one, and distributes mass over whatever followed those
occurrences. `lambda` is zero wherever no match exists, so the cache can only
speak when it has something to say — it never dilutes the model on novel
material.

This is a prior over REPETITION, not over harmony, and it is deliberately
separate from the LM: it is exactly the lever Louis has been pointing at, and
keeping it explicit means we can see how much of the model's score is grammar
and how much is bookkeeping.
"""
from __future__ import annotations

import numpy as np

from .vocab import VOCAB_SIZE


class RepetitionCache:
    """Exact-match within-song copy distribution, with backoff over match length.

    `weights` maps match length (in slots) to the confidence given to a match of
    that length. Longer match = stronger evidence that we are inside a literal
    repeat rather than a coincidence. Only the LONGEST available match is used.
    """

    def __init__(self, widths: tuple[int, ...] = (16, 12, 8, 6, 4, 2),
                 alpha: float = 0.002):
        """`alpha` must stay tiny — it is spread over all 90 tokens.

        At alpha=0.05 the smoothing contributes 0.05*90 = 4.5 pseudo-counts,
        so a signature seen ONCE before gave the copied chord p = 1.05/5.5 =
        0.19. The cache's top-1 was already 94% at a 16-slot match, but its
        probability mass was so flat that mixing it in moved almost nothing
        (+0.6pp). The confidence of a match is expressed by `lambda`, fitted on
        held-out data; it must not be pre-emptively destroyed here.
        """
        self.widths = tuple(sorted(widths, reverse=True))
        self.alpha = alpha

    def distributions(self, seq: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(T, V) copy distribution and (T,) match length for each position.

        Position i predicts seq[i] from seq[:i], so row 0 is always empty.
        A match length of 0 means "nothing to copy from".
        """
        T = len(seq)
        out = np.zeros((T, VOCAB_SIZE), dtype=np.float32)
        matched = np.zeros(T, dtype=np.int32)
        # signature -> list of tokens that followed it, built causally
        tables: dict[int, dict[tuple, list[int]]] = {w: {} for w in self.widths}
        for i in range(T):
            for w in self.widths:
                if i < w:
                    continue
                sig = tuple(seq[i - w:i])
                nxt = tables[w].get(sig)
                if nxt:
                    counts = np.full(VOCAB_SIZE, self.alpha, dtype=np.float32)
                    for t in nxt:
                        counts[t] += 1.0
                    out[i] = counts / counts.sum()
                    matched[i] = w
                    break
            # record AFTER predicting, so a position never sees itself
            for w in self.widths:
                if i >= w:
                    tables[w].setdefault(tuple(seq[i - w:i]), []).append(seq[i])
        return out, matched


def mix(model_logprobs: np.ndarray, copy_probs: np.ndarray, matched: np.ndarray,
        lam: dict[int, float] | float) -> np.ndarray:
    """Blend model and copy distributions; returns log-probs.

    `lam` may be a scalar or a per-match-length dict — a 16-slot match deserves
    more weight than a 2-slot one, and fitting one weight per length is 6
    parameters on a validation set of 20k slots.
    """
    p_model = np.exp(model_logprobs)
    out = p_model.copy()
    for i in range(len(matched)):
        w = int(matched[i])
        if w == 0:
            continue
        l = lam.get(w, 0.0) if isinstance(lam, dict) else float(lam)
        if l <= 0:
            continue
        out[i] = (1.0 - l) * p_model[i] + l * copy_probs[i]
    out = np.maximum(out, 1e-12)
    out /= out.sum(1, keepdims=True)
    return np.log(out)


def fit_lambdas(model_lp: list[np.ndarray], copy_p: list[np.ndarray],
                matched: list[np.ndarray], targets: list[np.ndarray],
                widths: tuple[int, ...] = (16, 12, 8, 6, 4, 2),
                grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                           0.7, 0.8, 0.9)) -> dict[int, float]:
    """Pick one lambda per match length by minimising held-out NLL.

    Fitted independently per length: the positions with a 16-slot match are
    disjoint from those with an 8-slot match (longest match wins), so the
    choices do not interact.
    """
    best: dict[int, float] = {}
    for w in widths:
        rows_m, rows_c, tg = [], [], []
        for lp, cp, mt, y in zip(model_lp, copy_p, matched, targets):
            sel = mt == w
            if sel.any():
                rows_m.append(lp[sel])
                rows_c.append(cp[sel])
                tg.append(y[sel])
        if not rows_m:
            best[w] = 0.0
            continue
        M = np.exp(np.concatenate(rows_m))
        C = np.concatenate(rows_c)
        Y = np.concatenate(tg)
        idx = np.arange(len(Y))
        scores = []
        for l in grid:
            p = (1 - l) * M[idx, Y] + l * C[idx, Y]
            scores.append(-np.log(np.maximum(p, 1e-12)).mean())
        best[w] = grid[int(np.argmin(scores))]
    return best
