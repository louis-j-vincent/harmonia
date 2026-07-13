"""Segment-level JOINT decode over (root × quality) — audit build-order step 2.

Replaces the greedy classifier→reranker chain (argmax root, then quality at that
root, then a post-hoc progression *override*) with a single principled MAP
inference: root and quality are decided TOGETHER, coupled across segments by the
fitted scale-relative progression bigram as a TRANSITION FACTOR.

Motivation (audit agent D): the true root is in beat_seq_v4's top-2 for 85.9% of
root-error segments (top-3 95.2%), but the greedy pipeline commits to the top-1
root before quality is ever computed. Letting (root, quality) be chosen jointly,
with a progression prior linking neighbouring segments, is the headline lever.

State space per segment: K candidate roots (top-K of the segment-summed beat
posterior) × 5 qualities (maj/min/dom/hdim/dim) = up to 15 states. Emission and
transition are scored in log space; an exact Viterbi over the segment chain gives
the MAP (root, q5) path, and log forward–backward gives per-segment marginals
(for the confidence display — returned even if unused now).

Design notes / v1 approximations (documented per CLAUDE.md rule #4):
  * The classifier re-run for a NON-argmax candidate root uses the GREEDY
    neighbour roots as its context (ctx_rt) — we vary only the current segment's
    root, not the whole neighbourhood jointly. A full joint-context decode is a
    later step; here the context is fixed to the top-1 path.
  * Emission root term uses ``log(mean-per-beat posterior at that root)`` — the
    log of the *mean* beat posterior, NOT a sum of per-beat log-probs, which
    would double-count sticky (repeated) beats and over-reward long segments.
  * Triad-vs-seventh is preserved exactly as ``rerank_progression_qualities``
    does: the joint decode picks the q5 *family*, and the seventh-vs-triad bit is
    taken from the classifier's own Harte call at the chosen root.
  * transition_weight=0.0 reduces to per-segment argmax of the JOINT emission
    (over the root×q5 grid) — which differs from the greedy baseline in two
    documented ways: (a) it may pick a top-2 root when that root's quality
    evidence outweighs the small root-posterior gap, and (b) it uses the
    argmax of the q5 log-probs rather than the family-head argmax. It is
    therefore *close to* but not byte-identical with the greedy path.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from harmonia.theory.progression_prior import _state as _prog_state_raw
from harmonia.theory.progression_prior import load as _load_bigram

# q5 index (maj/min/dom/hdim/dim) → progression_prior family name.  Mirrors the
# fam_of_bucket map used to FIT the bigram: dom7 is a "major"-family degree,
# m7b5 (hdim) is "diminished".
_Q5_TO_PROG_FAMILY = {
    0: "major",       # maj
    1: "minor",       # min
    2: "major",       # dom  (dominant is a major-family scale degree)
    3: "diminished",  # hdim
    4: "diminished",  # dim
}

_NEG_INF = -1e30


def prog_state(degree: int, q5idx: int) -> int:
    """Transition-matrix state index for a (scale-degree, q5-quality) pair.

    ``degree`` is ``(root - tonic) % 12``; ``q5idx`` in 0..4.  Delegates to the
    canonical ``progression_prior._state`` so the indexing is identical to how
    the bigram was fitted (state = degree * 5 + family_index).
    """
    return _prog_state_raw(degree % 12, _Q5_TO_PROG_FAMILY[q5idx])


def transition_logscore(prev_root: int, prev_q5: int, cur_root: int, cur_q5: int,
                        tonic: int, bigram_logp: np.ndarray) -> float:
    """Scale-relative bigram log P(cur | prev), keyed on the global tonic.

    Transposition-invariant by construction (P1): both roots enter only as
    ``(root - tonic) % 12``, so transposing tonic and both roots together leaves
    the score unchanged.
    """
    si = prog_state((prev_root - tonic) % 12, prev_q5)
    sj = prog_state((cur_root - tonic) % 12, cur_q5)
    return float(bigram_logp[si, sj])


def _logsumexp(a: np.ndarray) -> float:
    m = float(a.max())
    if m == _NEG_INF or not np.isfinite(m):
        return _NEG_INF
    return m + float(np.log(np.exp(a - m).sum()))


def load_bigram() -> np.ndarray:
    """(60, 60) log P(next | prev) over (degree, family) states."""
    return _load_bigram()


def joint_decode(
    segs: list[tuple[int, int]],
    beat_proba: np.ndarray,
    classify_fn: Callable[[int, int], tuple[str, str, float, np.ndarray]],
    tonic: int,
    *,
    K: int = 3,
    transition_weight: float = 1.0,
    bigram_logp: np.ndarray | None = None,
) -> dict:
    """Exact segment-level joint Viterbi over (root × q5) with a progression prior.

    Args:
        segs: the EXISTING segmentation (beat index ranges); unchanged — we only
            relabel, never re-segment.
        beat_proba: (n_beats, 12) per-beat root posterior.
        classify_fn: ``classify_fn(idx, root) -> (fam_h, sev_h, conf, q5_logp)``
            re-runs the family/ctx classifier for segment ``idx`` assuming
            ``root``.  ``q5_logp`` is the (5,) q5 log-prob vector; ``sev_h`` its
            Harte seventh form (carries the triad-vs-seventh bit).
        tonic: global key tonic pitch class (0..11) from infer_key.
        K: number of candidate roots per segment (top-K of the segment-summed
            beat posterior).
        transition_weight: weight on the bigram transition factor (0 = emission
            only).
        bigram_logp: (60,60) transition table (defaults to the fitted bigram).

    Returns a dict with per-segment lists (length T = len(segs)):
        roots, q5, sev_h, fam_h, conf (MAP-state marginal posterior),
        q5_logp (the chosen root's q5 log-probs, for suggestions),
        marginals (list of (S_t,) posterior arrays, aligned to `states`),
        states (list of (root, q5) tuples per segment).
    """
    if bigram_logp is None:
        bigram_logp = load_bigram()
    T = len(segs)

    # ── Per-segment candidate states + emission log-scores ────────────────────
    seg_states: list[list[tuple[int, int]]] = []   # (root, q5) per state
    seg_emis: list[np.ndarray] = []                 # (S_t,) emission log-score
    seg_cand: list[dict] = []                       # per candidate-root: sev_h/fam_h/q5_logp
    for idx, (s, e) in enumerate(segs):
        p_mean = beat_proba[s:e].mean(0)            # (12,) mean per-beat posterior
        tot = float(p_mean.sum())
        p_norm = p_mean / tot if tot > 1e-9 else np.full(12, 1.0 / 12)
        cand_roots = [int(r) for r in np.argsort(p_norm)[::-1][:K]]
        log_proot = {r: float(np.log(max(p_norm[r], 1e-12))) for r in cand_roots}

        cand_info: dict[int, dict] = {}
        states: list[tuple[int, int]] = []
        emis: list[float] = []
        for r in cand_roots:
            fam_h, sev_h, conf, q5_logp = classify_fn(idx, r)
            cand_info[r] = {"fam_h": fam_h, "sev_h": sev_h, "conf": conf,
                            "q5_logp": np.asarray(q5_logp, dtype=np.float64)}
            for q in range(5):
                states.append((r, q))
                emis.append(log_proot[r] + float(q5_logp[q]))
        seg_states.append(states)
        seg_emis.append(np.asarray(emis, dtype=np.float64))
        seg_cand.append(cand_info)

    # ── Precompute transition matrices between consecutive segments ───────────
    def _trans(prev_states, cur_states) -> np.ndarray:
        M = np.empty((len(prev_states), len(cur_states)), dtype=np.float64)
        for a, (r1, q1) in enumerate(prev_states):
            si = prog_state((r1 - tonic) % 12, q1)
            for b, (r2, q2) in enumerate(cur_states):
                sj = prog_state((r2 - tonic) % 12, q2)
                M[a, b] = transition_weight * float(bigram_logp[si, sj])
        return M

    trans_mats = [None] + [_trans(seg_states[t - 1], seg_states[t]) for t in range(1, T)]

    # ── Viterbi (MAP path) ────────────────────────────────────────────────────
    delta = [seg_emis[0].copy()]
    back: list[np.ndarray] = [np.full(len(seg_states[0]), -1, dtype=np.int64)]
    for t in range(1, T):
        M = trans_mats[t]                            # (S_{t-1}, S_t)
        scores = delta[t - 1][:, None] + M           # (S_{t-1}, S_t)
        bp = scores.argmax(0)
        delta.append(seg_emis[t] + scores[bp, np.arange(scores.shape[1])])
        back.append(bp.astype(np.int64))
    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    path = path[::-1]

    # ── Log forward–backward (marginals) ──────────────────────────────────────
    alpha = [seg_emis[0].copy()]
    for t in range(1, T):
        M = trans_mats[t]
        prev = alpha[t - 1]
        a = np.array([_logsumexp(prev + M[:, b]) for b in range(M.shape[1])])
        alpha.append(seg_emis[t] + a)
    beta: list[np.ndarray | None] = [None] * T
    beta[T - 1] = np.zeros(len(seg_states[T - 1]))
    for t in range(T - 2, -1, -1):
        M = trans_mats[t + 1]                        # (S_t, S_{t+1})
        nxt = seg_emis[t + 1] + beta[t + 1]
        beta[t] = np.array([_logsumexp(M[a, :] + nxt) for a in range(M.shape[0])])
    logZ = _logsumexp(alpha[T - 1])
    marginals = []
    for t in range(T):
        lm = alpha[t] + beta[t] - logZ
        marginals.append(np.exp(lm))

    # ── Assemble output ───────────────────────────────────────────────────────
    from harmonia.models.chord_pipeline_v1 import _Q5IDX_TO_HARTE, _SEVENTH_HARTE

    roots, q5s, sev_hs, fam_hs, confs, q5_logps = [], [], [], [], [], []
    for t in range(T):
        r, q = seg_states[t][path[t]]
        info = seg_cand[t][r]
        triad, seventh = _Q5IDX_TO_HARTE[q]
        sev_h = seventh if info["sev_h"] in _SEVENTH_HARTE else triad
        roots.append(r)
        q5s.append(q)
        sev_hs.append(sev_h)
        fam_hs.append(info["fam_h"])
        confs.append(float(marginals[t][path[t]]))
        q5_logps.append(info["q5_logp"])

    return {
        "roots": roots, "q5": q5s, "sev_h": sev_hs, "fam_h": fam_hs,
        "conf": confs, "q5_logp": q5_logps,
        "marginals": marginals, "states": seg_states, "path": path,
    }
