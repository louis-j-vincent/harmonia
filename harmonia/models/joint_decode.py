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
    (over the root×q5 grid). With K=1 this reproduces the greedy labels exactly
    (see the greedy anchor below); with K>1 it may pick a top-2 root when that
    root's quality evidence outweighs the small root-posterior gap.
  * GREEDY ANCHOR (gate-failure fix, 2026-07-13): the raw q5 log-probs from
    ``_family_q5_logprobs`` fold the aug+sus family mass onto ``maj``, so a
    chord the family/seventh heads decide is (say) minor can have ``maj`` as
    its q5 argmax — using that vector raw as the emission regressed majmin
    14pp at w=0 with IDENTICAL roots (first gate run, 2026-07-13). The fix is
    local to this module (``_family_q5_logprobs`` keeps its behaviour for the
    suggestions display): per candidate root, if the q5 argmax disagrees with
    the classifier's own greedy call (``_harte_to_q5idx(sev_h)``), the greedy
    class's log-prob is raised to ``max + eps`` — the classifier's actual
    decision is treated as at-least-as-likely as the aug/sus-contaminated
    argmax. The rest of the vector's geometry (the evidence the transition
    prior argues against) is untouched.
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


def _harte_to_q5idx_lazy(sev_h: str):
    """Lazy import of chord_pipeline_v1._harte_to_q5idx (avoids import cycles)."""
    from harmonia.models.chord_pipeline_v1 import _harte_to_q5idx
    return _harte_to_q5idx(sev_h)


# Default clamp strength for a user chord-confirm (Mission 3). The bonus is an
# additive log-score, so to be "effectively hard" it must dominate the WORST
# emission gap it might have to overcome. A confirmed root the acoustics give ~0
# mass sits at the posterior floor log(1e-12) ≈ −27.6 relative to a fully-pinned
# rival; adding the q5 gap (a few nats) the total gap is ~30 nats. +40 nats
# (exp(40) ≈ 2.4e17) therefore dominates even confirming a root the model never
# considered, while staying FINITE: the state remains in the same log-space as
# every other factor, so forward–backward and the transition prior still operate
# normally. That finiteness is deliberate — a −inf/delta clamp would break the
# marginals and forbid the decoder from ever disagreeing, whereas we WANT a
# confirmed chord to still let its NEIGHBOURS re-decode through the transition
# factor (propagation). Confirm = dominant evidence, not a hard freeze. (~+20
# was the initial spec; it is enough for the common case of correcting among the
# acoustic top-K, but not for asserting an unsupported root, so we use +40.)
CLAMP_NATS = 40.0


def joint_decode(
    segs: list[tuple[int, int]],
    beat_proba: np.ndarray,
    classify_fn: Callable[[int, int], tuple[str, str, float, np.ndarray]],
    tonic: int,
    *,
    K: int = 3,
    transition_weight: float = 1.0,
    bigram_logp: np.ndarray | None = None,
    local_tonic: "np.ndarray | list[int] | None" = None,
    q5_bonus: "Callable[[int, int], np.ndarray] | None" = None,
    constraints: "list[dict | None] | None" = None,
    pool_groups: "list[list[int]] | None" = None,
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
        local_tonic: optional per-segment (length T) LOCAL key tonic pc, used to
            re-reference the transition state to the local key instead of the
            single global ``tonic`` (H1, issue #27). Both endpoints of a
            transition are keyed on their OWN local tonic, so a ii-V-I inside a
            tonicization maps onto the bigram's diatonic diagonal (deg 2→7→0)
            rather than looking chromatic w.r.t. the global tonic. ``None``
            (default) reproduces the global-tonic behaviour bit-for-bit.
        q5_bonus: optional callback ``(seg_idx, root) -> (5,)`` returning an
            ADDITIVE log-score per q5 quality, folded into the EMISSION for that
            candidate root (H2 shallow fusion, issue #27): the ASR-style
            ``log P_acoustic + λ·log P_LM`` fusion of the ProgressionEncoder's
            per-quality log-prob, entering BEFORE the joint argmax (jointly with
            the root choice) rather than as a post-hoc rerank. Root-dependent by
            design — the encoder's context intervals are relative to the
            candidate centre root. ``None`` (default) is a no-op.
        constraints: optional per-segment (length T) list of USER chord-confirm
            factors (Mission 3). Each entry is ``None`` or a dict
            ``{"root": int, "q5": int | None, "bonus": float}``: the confirmed
            root is FORCE-INCLUDED as a candidate (even if outside the acoustic
            top-K, so a user can assert a root the model never considered), and
            an additive log-bonus (default ``CLAMP_NATS`` ≈ +20 nats) is added to
            the emission of the matching state(s) — ``(root, q5)`` if ``q5`` is
            given, else every ``(root, *)``. This is a soft-but-dominant delta
            prior: it pins the confirmed slot yet, being finite and in the SAME
            log-space as the transition factor, PROPAGATES to the neighbours
            through ``transition_weight`` rather than freezing them. ``None``
            (default) is a no-op (bit-identical to production).
        pool_groups: optional list of segment-index groups to TIE (Mission 3
            section-merge / P3 parallelism-as-denoising). Each group lists the
            segment indices the user asserts are the SAME chord (e.g. the k-th
            slot of two merged sections). Their emission log-scores are SUMMED
            over a shared (unioned) candidate-root state space — "superimposed
            observations, variance ↓ ~1/N" — and the group is force-tied to one
            decoded label. ``None`` (default) is a no-op.

    Returns a dict with per-segment lists (length T = len(segs)):
        roots, q5, sev_h, fam_h, conf (MAP-state marginal posterior),
        q5_logp (the chosen root's q5 log-probs, for suggestions),
        marginals (list of (S_t,) posterior arrays, aligned to `states`),
        states (list of (root, q5) tuples per segment).
    """
    if bigram_logp is None:
        bigram_logp = load_bigram()
    T = len(segs)

    # ── Pass 0: candidate roots per segment (top-K + user-forced + pool-union) ─
    p_norms: list[np.ndarray] = []
    seg_cand_roots: list[list[int]] = []
    for idx, (s, e) in enumerate(segs):
        p_mean = beat_proba[s:e].mean(0)            # (12,) mean per-beat posterior
        tot = float(p_mean.sum())
        p_norm = p_mean / tot if tot > 1e-9 else np.full(12, 1.0 / 12)
        p_norms.append(p_norm)
        cand_roots = [int(r) for r in np.argsort(p_norm)[::-1][:K]]
        con = constraints[idx] if constraints is not None else None
        if con is not None and con.get("root") is not None:
            fr = int(con["root"]) % 12
            if fr not in cand_roots:
                cand_roots.append(fr)            # force-include the confirmed root
        seg_cand_roots.append(cand_roots)
    # Pooled (tied) segments must share ONE candidate-root state space so their
    # emission log-scores are elementwise-summable — union the roots per group.
    if pool_groups:
        for group in pool_groups:
            union = sorted(set().union(*[set(seg_cand_roots[i]) for i in group]))
            for i in group:
                seg_cand_roots[i] = list(union)

    # ── Pass 1: classify + emission log-scores (with user clamp bonuses) ───────
    seg_states: list[list[tuple[int, int]]] = []   # (root, q5) per state
    seg_emis: list[np.ndarray] = []                 # (S_t,) emission log-score
    seg_cand: list[dict] = []                       # per candidate-root: sev_h/fam_h/q5_logp
    for idx, (s, e) in enumerate(segs):
        p_norm = p_norms[idx]
        cand_roots = seg_cand_roots[idx]
        log_proot = {r: float(np.log(max(p_norm[r], 1e-12))) for r in cand_roots}
        con = constraints[idx] if constraints is not None else None
        c_root = int(con["root"]) % 12 if (con and con.get("root") is not None) else None
        c_q5 = con.get("q5") if con else None
        c_bonus = float(con.get("bonus", CLAMP_NATS)) if con else 0.0

        cand_info: dict[int, dict] = {}
        states: list[tuple[int, int]] = []
        emis: list[float] = []
        for r in cand_roots:
            fam_h, sev_h, conf, q5_logp = classify_fn(idx, r)
            q5_logp = np.asarray(q5_logp, dtype=np.float64)
            # Greedy anchor (see module docstring): the classifier's own call
            # must be the emission argmax — undoes the aug/sus→maj folding.
            q5_emis = q5_logp.copy()
            g = _harte_to_q5idx_lazy(sev_h)
            if g is not None and int(q5_emis.argmax()) != g:
                q5_emis[g] = float(q5_emis.max()) + 1e-3
            cand_info[r] = {"fam_h": fam_h, "sev_h": sev_h, "conf": conf,
                            "q5_logp": q5_logp}
            bonus = q5_bonus(idx, r) if q5_bonus is not None else None
            for q in range(5):
                states.append((r, q))
                e = log_proot[r] + float(q5_emis[q])
                if bonus is not None:
                    e += float(bonus[q])
                # User chord-confirm clamp (Mission 3): dominant additive log-bonus.
                if c_root is not None and r == c_root and (c_q5 is None or q == c_q5):
                    e += c_bonus
                emis.append(e)
        seg_states.append(states)
        seg_emis.append(np.asarray(emis, dtype=np.float64))
        seg_cand.append(cand_info)

    # ── Pass 1b: pool tied segments' emission log-scores (section-merge, P3) ───
    # Members of a group now share an identical (root, q5) state ordering, so the
    # pooled emission is a plain elementwise SUM — the superimposed-observations
    # likelihood. Each member is assigned the pooled vector (so it decodes with
    # the combined evidence); the group is force-tied to one label after Viterbi.
    if pool_groups:
        for group in pool_groups:
            pooled = np.sum([seg_emis[i] for i in group], axis=0)
            for i in group:
                seg_emis[i] = pooled.copy()

    # ── Precompute transition matrices between consecutive segments ───────────
    def _tonic_at(t: int) -> int:
        return tonic if local_tonic is None else int(local_tonic[t]) % 12

    def _trans(prev_states, cur_states, ton_p: int, ton_c: int) -> np.ndarray:
        M = np.empty((len(prev_states), len(cur_states)), dtype=np.float64)
        for a, (r1, q1) in enumerate(prev_states):
            si = prog_state((r1 - ton_p) % 12, q1)
            for b, (r2, q2) in enumerate(cur_states):
                sj = prog_state((r2 - ton_c) % 12, q2)
                M[a, b] = transition_weight * float(bigram_logp[si, sj])
        return M

    trans_mats = [None] + [
        _trans(seg_states[t - 1], seg_states[t], _tonic_at(t - 1), _tonic_at(t))
        for t in range(1, T)
    ]

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

    # Force-tie pooled groups to ONE label (section-merge): members share an
    # identical state ordering and pooled emission, so pin every member to the
    # state that maximises the pooled emission (the superimposed-observation MAP).
    if pool_groups:
        for group in pool_groups:
            tied = int(seg_emis[group[0]].argmax())
            for i in group:
                path[i] = tied

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
