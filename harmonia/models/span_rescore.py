"""span_rescore.py — post-hoc span-lattice re-score for lock propagation
(branch feat/chord-context-prior; design: docs/design_chord_context_prior.md).

Pure-numpy core: pools acoustic evidence onto the CLIENT's currently displayed
chord spans (boundaries never move), builds a small per-span candidate set,
and decodes root+quality with a second-order (pair-state) Viterbi so a
chord-context prior — conditioned on BOTH neighbours — can pull an ambiguous
span toward a user-locked one. No Flask here; harmonia/serving/api.py wires
this to the network.

ACOUSTIC BACKENDS (2026-07-30 redirect: musx is this project's
state-of-the-art chord recognizer, not the in-house NNLS-24 heads)
--------------------------------------------------------------------
1. ``musx_probs`` (PRIMARY) — music-x-lab's own 5-fold-averaged frame
   posteriors (``harmonia.models.musx_redecode.frame_posteriors``), mean-pooled
   per span on their native 23.22 ms grid.  Cache-hit only in this module: a
   cache miss does NOT trigger a fresh ~10-30s musx run inline in a request —
   the caller degrades to ``nnls_heads`` instead (see ``compute_acoustic_logp``).
2. ``nnls_heads`` (FALLBACK) — the trained NNLS-24 root+quality heads
   (``harmonia.models.nnls_features``) over VAMP bothchroma, mean-pooled per
   span.  Same heads ``harmonia/stages/chord_head.py`` uses live; root and
   quality are scored as if independent (v1 approximation — the heads are a
   root->quality CASCADE at inference, not literally independent; noted again
   at ``acoustic_logp_nnls``).

Both produce a per-span (12 roots x 5 QUAL5) log-posterior over the SAME
60-candidate space, so ``lattice_rescore`` never needs to know which backend
supplied it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent

# QUAL5 = ["maj", "min", "dom", "hdim", "dim"] — reused, not redefined (must
# stay byte-identical to progression_encoder's order: the context prior's
# token space and this module's candidate index (root*5+qual) both depend on
# it, and chord_pipeline_v1._Q5_NAMES is the same tuple under a different name).
from harmonia.models.progression_encoder import QUAL5  # noqa: E402

N_ROOTS = 12
N_QUAL5 = len(QUAL5)          # 5
N_CANDIDATES = N_ROOTS * N_QUAL5   # 60, index = root_pc * 5 + qual5_idx


def idx_of(root_pc: int, qual5_idx: int) -> int:
    return int(root_pc) % N_ROOTS * N_QUAL5 + int(qual5_idx)


def token_of(idx: int) -> tuple[int, int]:
    """60-index -> (root_pc, qual5_idx)."""
    return divmod(int(idx), N_QUAL5)


# ═══════════════════════════════════════════════════════════════════════════
# Backend 1 (PRIMARY): music-x-lab frame posteriors
# ═══════════════════════════════════════════════════════════════════════════
# TriadTypes enum order in the vendored clone (complex_chord.py), EXCLUDING
# 'none'/'power'/'one' which the trained 73-wide triad head never predicts —
# verified against harmonia/third_party/.../data/submission_chord_list.txt
# (the model's legal-vocabulary file): maj/min/dim/aug/sus4/sus2 are all
# present as bare triads, 'power'/'one' never appear.
_TRIAD_TYPES = ("maj", "min", "sus4", "sus2", "dim", "aug")
# SeventhTypes enum order (complex_chord.py): none, add_7 (natural 7 / maj7
# interval), add_b7 (flat 7 — dominant/min7), add_bb7 (diminished 7th).
_SEV_TYPES = ("none", "maj7", "b7", "bb7")

# (triad type, seventh type) -> QUAL5 family. Built from the SAME semantics as
# progression_encoder.FINE_TO_QUAL5 (maj7->maj, dom7->dom, m7b5->hdim, dim7->
# dim, sus2/sus4(bare)->maj, 7sus4->dom, aug->maj, aug7->dom) rather than
# invented fresh — the fine-grained key spellings differ (musx says "7", the
# fine vocab says "dom7") but the family assignment is identical by chord
# theory, and is verified against submission_chord_list.txt: maj, min, 7,
# maj7, min7, dim, dim7, hdim7, aug, sus2, sus4, sus4(b7) are the ONLY
# (triad,seventh) combinations the model was ever trained to predict. Cells
# for combinations absent from that vocabulary (e.g. an augmented major 7th)
# get a defensive same-triad-family fallback below — their posterior mass is
# ~0 in practice, never trained-for.
_TRIAD_SEV_TO_QUAL5 = {
    ("maj", "none"): "maj", ("maj", "maj7"): "maj",
    ("maj", "b7"): "dom",   ("maj", "bb7"): "maj",
    ("min", "none"): "min", ("min", "maj7"): "min",
    ("min", "b7"): "min",   ("min", "bb7"): "min",
    ("sus4", "none"): "maj", ("sus4", "maj7"): "maj",
    ("sus4", "b7"): "dom",   ("sus4", "bb7"): "maj",
    ("sus2", "none"): "maj", ("sus2", "maj7"): "maj",
    ("sus2", "b7"): "dom",   ("sus2", "bb7"): "maj",
    ("dim", "none"): "dim", ("dim", "maj7"): "dim",
    ("dim", "b7"): "hdim",  ("dim", "bb7"): "dim",
    ("aug", "none"): "maj", ("aug", "maj7"): "maj",
    ("aug", "b7"): "dom",   ("aug", "bb7"): "maj",
}


def _build_fold_tensor() -> np.ndarray:
    """(6 triad types, 4 seventh types, 5 QUAL5) one-hot selection tensor."""
    fold = np.zeros((len(_TRIAD_TYPES), len(_SEV_TYPES), N_QUAL5), dtype=np.float64)
    q5_idx = {q: i for i, q in enumerate(QUAL5)}
    for ti, t in enumerate(_TRIAD_TYPES):
        for si, s in enumerate(_SEV_TYPES):
            fam = _TRIAD_SEV_TO_QUAL5[(t, s)]
            fold[ti, si, q5_idx[fam]] = 1.0
    return fold


_FOLD = _build_fold_tensor()   # (6,4,5)


def musx_cache_path(audio_path: Path | str) -> Path:
    """Where ``frame_posteriors`` would look for this stem — WITHOUT loading.

    Mirrors ``musx_redecode.frame_posteriors``'s own cache key exactly
    (``Path(audio_path).resolve().stem``) so callers can check existence
    before deciding whether a cache hit is even possible.
    """
    from harmonia.models.musx_redecode import _PROB_CACHE
    return _PROB_CACHE / f"{Path(audio_path).resolve().stem}.npz"


def pool_span_musx(
    probs: list[np.ndarray], spans: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool music-x-lab's triad(73) + s7(4) frame posteriors per span.

    ``probs`` is exactly ``musx_redecode.frame_posteriors``'s return
    (``[triad, bass, s7, s9, s11, s13]``); only triad and s7 are read here (s9/
    s11/s13 don't affect QUAL5 — a dominant 9th still folds to "dom").  Frame
    ``f`` covers ``[f*FRAME_DT, (f+1)*FRAME_DT)`` — the SAME convention
    ``musx_posterior_fold._bar_frame_index`` uses, reproduced here rather than
    imported since that helper is bar-keyed, not span-keyed.

    Returns ``(pooled_triad (n_spans,73), pooled_s7 (n_spans,4))``.  An empty
    span (shorter than one frame, or past the end of the posteriors) falls
    back to the single nearest frame, mirroring
    ``nnls_features.pool_beats``'s own empty-interval rule.
    """
    from harmonia.models.musx_redecode import FRAME_DT

    triad, s7 = probs[0], probs[2]
    n_frame = triad.shape[0]
    n = len(spans)
    out_triad = np.zeros((n, triad.shape[1]), dtype=np.float64)
    out_s7 = np.zeros((n, s7.shape[1]), dtype=np.float64)
    for i, (t0, t1) in enumerate(spans):
        f0 = max(0, int(round(t0 / FRAME_DT)))
        f1 = min(n_frame, int(round(t1 / FRAME_DT)))
        if f1 <= f0:
            j = int(np.clip(round(0.5 * (t0 + t1) / FRAME_DT), 0, max(n_frame - 1, 0)))
            out_triad[i] = triad[j]
            out_s7[i] = s7[j]
        else:
            out_triad[i] = triad[f0:f1].mean(0)
            out_s7[i] = s7[f0:f1].mean(0)
    return out_triad, out_s7


def acoustic_logp_musx(
    pooled_triad: np.ndarray, pooled_s7: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(pooled triad, pooled s7) -> (log-posterior (n,60), n_mass (n,)).

    ``n_mass`` is the pooled "no chord at all" mass (triad column 0) — kept
    OUT of the 60-candidate space per the design doc, reported separately so a
    caller can flag a likely-silent span without it competing for a (root,
    quality) slot it doesn't have.

    Root and triad-TYPE are NOT independent in this model — the 73-wide triad
    head predicts them JOINTLY (one-of-73, incl. "no chord") — so this is an
    exact marginal-times-conditional combination for that pair, only the
    seventh factor is treated as combining independently of root (which is
    also the vendored architecture's own factorization: complex_chord.py
    trains triad/seventh/ninth/... as separate categorical heads).
    """
    n = pooled_triad.shape[0]
    n_mass = np.clip(pooled_triad[:, 0].astype(np.float64), 0.0, 1.0)
    # Column i>=1 of the 73-wide triad posterior is root=(i-1)%12, type=
    # (i-1)//12 (root is the FAST-varying axis, type is SLOW) — verified
    # against scripts/check_span_rescore_sanity.py catching exactly this
    # (root agreement 0.0 on This Love before the fix): a naive
    # ``.reshape(n, 12, 6)`` implicitly assumes the opposite (root slow, type
    # fast) and silently transposes every chord's root+type. Reshape as
    # (type, root) — the true flat order — then move the axes to (root, type).
    chord = (pooled_triad[:, 1:].astype(np.float64)
            .reshape(n, len(_TRIAD_TYPES), N_ROOTS).transpose(0, 2, 1))
    sev = pooled_s7.astype(np.float64)
    # joint[n,r,q] = sum_t sum_s chord[n,r,t] * sev[n,s] * FOLD[t,s,q]
    joint = np.einsum("nrt,ns,tsq->nrq", chord, sev, _FOLD)
    denom = np.clip(1.0 - n_mass, 1e-9, None)          # == joint.sum((1,2)) up to fp error
    joint = joint / denom[:, None, None]
    logp = np.log(np.clip(joint, 1e-12, None)).reshape(n, N_CANDIDATES)
    return logp, n_mass


# ═══════════════════════════════════════════════════════════════════════════
# Backend 2 (FALLBACK): NNLS-24 trained heads
# ═══════════════════════════════════════════════════════════════════════════
def pool_span_features(
    bothchroma: np.ndarray, frame_times: np.ndarray, spans: list[tuple[float, float]],
) -> np.ndarray:
    """Mean-pool raw VAMP bothchroma frames within each span's [t0,t1).

    Same feature convention the trained heads expect (nnls_features.py
    module docstring, verified against a real cached npz here — see
    scripts/check_span_rescore_sanity.py): roll each 12-half by 9 so index 0
    is C, L2-normalise each half independently, stack bass|treble -> 24-d.
    ``bothchroma``/``frame_times`` are exactly ``nnls_features.
    extract_bothchroma``'s return (index 0 = A, NOT yet rolled). Empty spans
    fall back to the nearest single frame (matches ``pool_beats``).
    """
    from harmonia.models.nnls_features import _l2, _ROLL_TO_C

    n = len(spans)
    out = np.zeros((n, 24), dtype=np.float32)
    for i, (t0, t1) in enumerate(spans):
        m = (frame_times >= t0) & (frame_times < t1)
        if not m.any():
            j = int(np.argmin(np.abs(frame_times - 0.5 * (t0 + t1))))
            seg = bothchroma[j]
        else:
            seg = bothchroma[m].mean(0)
        bass = np.roll(seg[:12], _ROLL_TO_C)
        treb = np.roll(seg[12:], _ROLL_TO_C)
        out[i] = np.concatenate([_l2(bass), _l2(treb)])
    return out


def _fold7_to_qual5(qual_p7: np.ndarray) -> np.ndarray:
    """(n,7) NNLS-24 quality posterior (heads.qualities order: maj, min, dom,
    hdim, dim, aug, sus) -> (n,5) QUAL5. Bare aug/sus fold to maj — same
    convention as ``_TRIAD_SEV_TO_QUAL5`` above (aug->maj, sus2/sus4->maj);
    this head's vocabulary has no separate 7sus/aug7 class to route to dom."""
    out = np.array(qual_p7[:, :5], copy=True, dtype=np.float64)
    out[:, 0] += qual_p7[:, 5] + qual_p7[:, 6]   # aug, sus -> maj
    return out


def acoustic_logp_nnls(feat24: np.ndarray, heads) -> np.ndarray:
    """NNLS-24 heads -> (n,60) log-posterior over 12 roots x QUAL5.

    v1 INDEPENDENCE ASSUMPTION (design doc, explicit): log p(r,q) = log
    p_root[r] + log p_quality[q]. The trained quality head is actually a
    root->quality CASCADE (rotated by the root argmax before scoring, see
    NNLS24Heads.quality_proba) — quality is conditioned on ONE root estimate,
    not scored per-candidate-root. Treating it as if independent of the final
    root choice is an approximation, not exact (unlike the musx backend's
    triad/seventh split, which IS the model's real factorization).
    """
    root_p = np.clip(heads.root_proba(feat24).astype(np.float64), 1e-12, None)
    root_idx = root_p.argmax(1)
    qual_p7 = heads.quality_proba(feat24, root_idx)
    qual_p5 = np.clip(_fold7_to_qual5(qual_p7), 1e-12, None)
    logp = np.log(root_p)[:, :, None] + np.log(qual_p5)[:, None, :]
    return logp.reshape(len(feat24), N_CANDIDATES)


# ═══════════════════════════════════════════════════════════════════════════
# Backend dispatch
# ═══════════════════════════════════════════════════════════════════════════
def compute_acoustic_logp(
    audio_for_cache: Path, spans: list[tuple[float, float]], *,
    fallback_audio: Path | None = None,
) -> dict:
    """Pick musx_probs (cache-hit only) else nnls_heads; never a fresh musx run.

    ``audio_for_cache``: a path whose ``.stem`` is the ORIGINAL analysis
    stem (the youtube video id for a server-analyzed chart) — this is the
    cache key both ``musx_redecode.frame_posteriors`` and ``nnls_features.
    extract_bothchroma`` use, and it is NOT the same file
    ``_chart_audio_path`` returns (that's a re-transcoded, re-slugged copy for
    browser playback — see api.py's endpoint for the stem-resolution). The
    path need not exist on disk for a cache HIT (checked by stem only).
    ``fallback_audio``: the REAL, readable audio file to fall back to for a
    fresh (slower) nnls_heads extraction if NEITHER cache hits.

    Returns dict(logp=(n,60), backend="musx_probs"|"nnls_heads", n_mass=arr
    or None, cache_hit=bool).
    """
    musx_cache = musx_cache_path(audio_for_cache)
    if musx_cache.exists():
        from harmonia.models.musx_redecode import frame_posteriors
        logger.info("span_rescore: musx_probs cache HIT (%s)", musx_cache.name)
        probs = frame_posteriors(audio_for_cache)   # cache hit, no audio touch
        pooled_triad, pooled_s7 = pool_span_musx(probs, spans)
        logp, n_mass = acoustic_logp_musx(pooled_triad, pooled_s7)
        return {"logp": logp, "backend": "musx_probs", "n_mass": n_mass,
                "cache_hit": True}

    logger.info("span_rescore: musx_probs cache MISS (%s) — falling back to "
                "nnls_heads (never triggering a fresh musx run inline)",
                musx_cache.name)
    from harmonia.models import nnls_features as nf
    heads = nf.get_heads()
    if heads is None:
        raise RuntimeError("span_rescore: nnls24_heads.npz unavailable and no "
                           "musx_probs cache — cannot compute acoustic evidence")

    nnls_cache_hit = (nf._CACHE_DIR / f"{Path(audio_for_cache).stem}.npz").exists()
    audio_for_extract = audio_for_cache if nnls_cache_hit else (fallback_audio or audio_for_cache)
    logger.info("span_rescore: nnls_infer cache %s (using %s)",
               "HIT" if nnls_cache_hit else "MISS", Path(audio_for_extract).name)
    arr, times = nf.extract_bothchroma(audio_for_extract)
    feat24 = pool_span_features(arr, times, spans)
    logp = acoustic_logp_nnls(feat24, heads)
    return {"logp": logp, "backend": "nnls_heads", "n_mass": None,
            "cache_hit": nnls_cache_hit}


# ═══════════════════════════════════════════════════════════════════════════
# Context scorer: the Phase 1 seam (stubbed until harmonia/models/
# chord_context_prior.py lands — DO NOT create/edit that module here)
# ═══════════════════════════════════════════════════════════════════════════
class _UniformContextScorer:
    """lam-inert stand-in: a flat distribution adds the SAME constant to every
    candidate's score, so it can never change an argmax — exactly "lam
    effectively inert" until Phase 1 lands."""

    def score_candidates(self, prev_token, next_token):
        return np.full(N_CANDIDATES, 1.0 / N_CANDIDATES, dtype=np.float64)


def load_context_scorer(corpus: str = "pooled"):
    """The real Phase-1/genre-arm prior if importable, else the uniform stub.

    ``corpus`` selects the table (genre-arm experiment, docs/
    context_prior_phase1_results.md): "jazz" | "pop" | "pooled" (default,
    unchanged zero-arg behaviour). Routing WHICH corpus to request per chart
    is the caller's job (api.py's genre routing, task 1) — this function only
    loads whichever table it's told to.

    Contract (design doc "Phase 2 wiring"): ``score_candidates(prev_token,
    next_token) -> (60,)`` normalized over 12 roots x QUAL5, tokens =
    (root_pc, qual5_idx) or None (directed-bigram fallback, used at the ends
    of the lattice). Phase 1 owns chord_context_prior.py entirely — this
    module only ever imports its public loader.
    """
    try:
        from harmonia.models.chord_context_prior import load_context_prior
        return load_context_prior(corpus=corpus)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        logger.info("span_rescore: chord_context_prior unavailable (%s) — "
                    "uniform stub, lam inert", exc)
        return _UniformContextScorer()


# ═══════════════════════════════════════════════════════════════════════════
# The lattice: exact second-order (pair-state) Viterbi
# ═══════════════════════════════════════════════════════════════════════════
_START = object()   # sentinel: "no left neighbour" / "no right neighbour"


def _candidates_for(acoustic_logp_row: np.ndarray, displayed_idx: int,
                    lock_idx: int | None, K: int) -> list[int]:
    if lock_idx is not None:
        return [lock_idx]
    top_k = np.argsort(acoustic_logp_row)[::-1][:K].tolist()
    cands = set(top_k)
    cands.add(int(displayed_idx))
    return sorted(cands)


_UNIFORM_VEC = np.full(N_CANDIDATES, 1.0 / N_CANDIDATES, dtype=np.float64)
_warned_scorer_failure = False   # log the fallback once per process, not per span


def _ctx_logp(context_scorer, prev_idx, next_idx, cache: dict) -> np.ndarray:
    global _warned_scorer_failure
    key = (prev_idx, next_idx)
    if key in cache:
        return cache[key]
    prev_tok = None if prev_idx is _START or prev_idx is None else token_of(prev_idx)
    next_tok = None if next_idx is _START or next_idx is None else token_of(next_idx)
    # Defensive beyond the load-time try/except in load_context_scorer():
    # Phase 1 (chord_context_prior.py) is being built concurrently and its
    # interface can be mid-flight (e.g. 2026-07-30: load_context_prior()
    # returns a raw fitted dict + a MODULE-LEVEL score_candidates(prev, next,
    # model=...) function, not yet the object-with-method the design doc's
    # wiring contract specifies). A scorer that doesn't (yet) match the
    # contract must degrade to lam-inert uniform, not crash the endpoint —
    # same "never break the live path" rule as chord_head.py's musx fallbacks.
    try:
        vec = np.clip(np.asarray(context_scorer.score_candidates(prev_tok, next_tok),
                                 dtype=np.float64), 1e-12, None)
        if vec.shape != (N_CANDIDATES,):
            raise ValueError(f"score_candidates returned shape {vec.shape}, want "
                             f"({N_CANDIDATES},)")
    except Exception as exc:  # noqa: BLE001 — see comment above
        if not _warned_scorer_failure:
            logger.warning("span_rescore: context_scorer.score_candidates failed "
                           "(%s) — falling back to uniform (lam inert) for the "
                           "rest of this process", exc)
            _warned_scorer_failure = True
        vec = _UNIFORM_VEC
    logvec = np.log(vec)
    cache[key] = logvec
    return logvec


def lattice_rescore(
    acoustic_logp: np.ndarray,
    displayed: list[tuple[int, int]],
    locks: list[tuple[int, int] | None],
    context_scorer,
    *, lam: float = 1.0, K: int = 6, delta: float = 0.0,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Exact second-order Viterbi -> (chosen (root,qual5) per span, margins).

    Objective: sum_i [ log p_acoustic(c_i) + incumbent_bonus(i, c_i) ] + lam *
    sum_i log P_ctx(c_i | c_{i-1}, c_{i+1}).  Candidates per span = acoustic
    top-K union the displayed chord union {lock} if locked; a locked span's
    candidate set is EXACTLY {lock} (never overridden). Boundaries are the
    caller's — this function only chooses labels, never moves a span edge.

    ``delta`` (incumbent-stickiness bonus, task 2 of the lock-propagation
    tuning brief): ``incumbent_bonus(i, c) = delta`` when ``c`` equals span
    ``i``'s DISPLAYED (currently-shown) label AND span ``i`` is unlocked, else
    0. Rationale: without it, a NO-lock call still re-scores every span from
    scratch and flips ties to the acoustic argmax — unrequested churn (measured
    ~28/119 spans on "This Love"). A flat per-span bonus on the incumbent label
    breaks exact ties toward "leave it alone" and requires real acoustic+context
    evidence to overcome before flipping, without ever touching locked spans
    (already clamped to {lock} regardless of delta) or acting like a prior on
    which unlocked chord is CORRECT — it only privileges "what's already
    displayed", so genuine lock-driven propagation (acoustic+context strongly
    preferring another candidate) still goes through once it clears delta.

    DP state after span i is the PAIR (c_{i-1}, c_i) (``_START`` stands in for
    "no left neighbour" at i=0). The context term for c_{i-1} needs BOTH its
    neighbours, so it is scored on the transition INTO state i (which is the
    first point both are known); the last span's context term (next=None) is
    added once, after the loop, to whichever final pair-state wins.

    Returns ``margins[i]`` = (score of the chosen candidate at i) - (score of
    the runner-up candidate at i), holding the REST of the winning path fixed
    — a cheap per-span confidence signal, not a true marginal posterior.
    """
    n = len(acoustic_logp)
    if n == 0:
        return [], []
    idx_displayed = [idx_of(*d) for d in displayed]
    idx_locks = [idx_of(*l) if l is not None else None for l in locks]
    candidates = [_candidates_for(acoustic_logp[i], idx_displayed[i], idx_locks[i], K)
                 for i in range(n)]
    ctx_cache: dict = {}

    def _incumbent(i: int, c: int) -> float:
        if delta == 0.0 or idx_locks[i] is not None:
            return 0.0
        return delta if c == idx_displayed[i] else 0.0

    if n == 1:
        c0 = candidates[0]
        scores = [acoustic_logp[0][c] + lam * _ctx_logp(context_scorer, _START, _START,
                                                        ctx_cache)[c] + _incumbent(0, c)
                 for c in c0]
        order = np.argsort(scores)[::-1]
        best = c0[order[0]]
        margin = (scores[order[0]] - scores[order[1]]) if len(c0) > 1 else float("inf")
        return [token_of(best)], [float(margin)]

    # dp[(a, b)] = best cumulative score of a path ending in pair-state (a, b),
    # i.e. c_{i-1}=a, c_i=b, for the CURRENT i. back[(a,b)] = predecessor a'
    # (the c_{i-2} that pair (a,b) extends).
    dp: dict[tuple, float] = {(_START, b): float(acoustic_logp[0][b]) + _incumbent(0, b)
                              for b in candidates[0]}
    back: list[dict[tuple, object]] = [dict()]   # back[i][(a,b)] -> predecessor of a

    for i in range(1, n):
        new_dp: dict[tuple, float] = {}
        new_back: dict[tuple, object] = {}
        acoustic_i = acoustic_logp[i]
        # left_cands = candidates for c_{i-2}, the OLD state's left component
        # (just _START when i==1, since there is no span -1).
        left_cands = candidates[i - 2] if i >= 2 else [_START]
        for c in candidates[i]:
            best_for_c: dict = {}   # b (=c_{i-1}) -> (score, predecessor a=c_{i-2})
            for a in left_cands:
                # b's that actually reached dp paired with this a (dp is the
                # full candidates[i-2] x candidates[i-1] cross product by
                # construction, so this is exactly candidates[i-1] every time,
                # but keying off dp keeps it correct even if that ever stops
                # holding).
                relevant_bs = [b for (aa, b) in dp if aa is a]
                if not relevant_bs:
                    continue
                # ctx_vec scores the MIDDLE token c_{i-1} now that both its
                # neighbours (a=c_{i-2}, c=c_i) are fixed.
                ctx_vec = _ctx_logp(context_scorer, a, c, ctx_cache)
                for b in relevant_bs:
                    score = (dp[(a, b)] + float(acoustic_i[c]) + lam * float(ctx_vec[b])
                            + _incumbent(i, c))
                    cur = best_for_c.get(b)
                    if cur is None or score > cur[0]:
                        best_for_c[b] = (score, a)
            for b, (score, a) in best_for_c.items():
                new_dp[(b, c)] = score
                new_back[(b, c)] = a
        dp = new_dp
        back.append(new_back)

    # finalize: score the LAST span's context term (next=None), directed
    # bigram — this is the only place c_{n-1}'s own context ever gets added.
    final_scores = {}
    for (a, b), score in dp.items():
        ctx_vec = _ctx_logp(context_scorer, a, _START, ctx_cache)
        final_scores[(a, b)] = score + lam * float(ctx_vec[b])
    best_state = max(final_scores, key=final_scores.get)

    # backtrack
    path = [None] * n
    a, b = best_state
    path[n - 1] = b
    for i in range(n - 1, 0, -1):
        a_prev = back[i][(a, b)]
        path[i - 1] = a
        b = a
        a = a_prev
    # path[i] now holds c_i for every i (0-indexed)

    # margins: hold the winning path's neighbours fixed, compare candidates at i
    margins = [0.0] * n
    for i in range(n):
        prev_c = path[i - 1] if i > 0 else _START
        next_c = path[i + 1] if i < n - 1 else _START
        ctx_vec = _ctx_logp(context_scorer, prev_c, next_c, ctx_cache)
        scored = sorted(
            (float(acoustic_logp[i][c]) + lam * float(ctx_vec[c]) + _incumbent(i, c)
             for c in candidates[i]),
            reverse=True)
        margins[i] = (scored[0] - scored[1]) if len(scored) > 1 else float("inf")

    chosen = [token_of(c) for c in path]
    return chosen, margins


def _candidate_score(
    acoustic_row: np.ndarray, prev_tok, next_tok, displayed_tok: tuple[int, int],
    candidate_tok: tuple[int, int], context_scorer, lam: float, delta: float, ctx_cache: dict,
) -> float:
    """Combined score of ONE candidate at ONE span, given its (already fixed)
    neighbour tokens -- the exact same per-candidate formula the DP inside
    ``lattice_rescore`` accumulates (``acoustic_logp[i][c] + lam*ctx_vec[b] +
    incumbent(i, c)``, see that function's inner loop), evaluated standalone
    for a margin comparison between two SPECIFIC candidates rather than
    accumulated over a whole path. ``prev_tok``/``next_tok`` are (root, qual5)
    tuples or ``None`` (no neighbour, the lattice's own edge convention).
    ``delta`` applies exactly like ``lattice_rescore``'s own incumbent bonus:
    only when ``candidate_tok == displayed_tok`` (this helper is only ever
    called for UNLOCKED spans -- a locked span's candidate set is {lock},
    margin-gating never applies there -- so the "not locked" half of that
    bonus's condition is always true here).
    """
    prev_idx = _START if prev_tok is None else idx_of(*prev_tok)
    next_idx = _START if next_tok is None else idx_of(*next_tok)
    ctx_vec = _ctx_logp(context_scorer, prev_idx, next_idx, ctx_cache)
    cand_idx = idx_of(*candidate_tok)
    incumbent = delta if candidate_tok == displayed_tok else 0.0
    return float(acoustic_row[cand_idx]) + lam * float(ctx_vec[cand_idx]) + incumbent


def differential_rescore(
    acoustic_logp: np.ndarray,
    displayed: list[tuple[int, int]],
    locks: list[tuple[int, int] | None],
    context_scorer,
    *, lam: float = 1.0, K: int = 6, delta: float = 0.0, margin_gate: float = 0.0,
) -> tuple[list[tuple[int, int]], list[bool], list[float]]:
    """Lock-attributable-only rescore (2026-07-30 redesign, replacing the
    "vs displayed" comparison every earlier version of this module used).

    THE BUG THIS FIXES: comparing the locked lattice's output directly against
    the DISPLAYED chart conflates two different things --
      (a) genuine effects of the lock (what we want to report), and
      (b) the lattice's own unconditional disagreement with the displayed
          chart (real, but present even with ZERO locks -- ~15-30% of spans
          on real songs in the tuning sweep, because acoustic evidence is
          often too flat for the context prior not to dominate, and a
          repeated progression fragment gets the SAME reading everywhere the
          trigram fires, independent of any lock).
    Task 3's simulated-lock sweep measured "corruption" this conflated way and
    concluded no config was both safe and effective; re-diagnosed 2026-07-30
    (docs/lock_propagation_tuning.md "Differential re-analysis"): locking ONE
    span in a 189-span song changed 64 spans vs displayed, but only 8 of those
    64 differ from what a ZERO-lock call at the SAME (lam, delta, K) already
    picks -- i.e. 56/64 were pre-existing churn, not lock-caused.

    THE FIX (same shape as the old, working ``/api/reinfer``: decode base
    with no constraints, cons WITH constraints, diff cons vs base -- never vs
    the original display): run ``lattice_rescore`` TWICE at the identical
    (lam, delta, K) -- once with ``locks`` all None (the baseline), once with
    the real ``locks`` -- and only let a span's value move away from
    ``displayed`` if EITHER it is itself locked, OR the two runs disagree
    there (a genuine, lock-caused effect). Every span where baseline and
    locked agree keeps its DISPLAYED value, no matter what either run's own
    opinion of that span is.

    A direct consequence: a request with **zero** locks has an identical
    baseline and "locked" run by construction, so nothing ever changes --
    the old no-lock-churn problem (task 2) is now solved structurally, not by
    tuning delta down the effect (delta's role shifts: see the tuning doc's
    differential re-analysis for whether it is even still needed).

    MARGIN GATE (2026-07-30, docs/lock_propagation_tuning.md "Margin gate"):
    ``margin_gate`` (nats, default 0.0 = off) screens PROPAGATED changes only
    (an unlocked span where the locked run's choice differs from baseline's;
    the locked span itself is never gated -- the user asked for that one
    explicitly). Hypothesis: a lot of the remaining wrong-lock corruption is
    low-margin flips -- spans where the locked and baseline runs barely
    disagree -- while genuine, confident context patterns (e.g. a V7 inside
    ii-?-I) should clear a much wider margin. For each propagated span j, the
    margin is ``score(locked_run's neighbours, locked_chosen[j]) -
    score(same neighbours, baseline_chosen[j])`` -- i.e. BOTH candidates are
    scored inside the LOCKED run's own solution (its actual chosen neighbours
    at j-1/j+1 held fixed), using the identical acoustic + lam*context(+delta
    incumbent) formula the DP itself accumulates (``_candidate_score``). This
    directly answers "how much better does the locked run like its own
    answer over the alternative baseline was proposing", not a generic
    top-1-vs-runner-up margin. A propagated change only survives if this gap
    is >= margin_gate nats; otherwise the span reverts to ``displayed``
    (exactly as if baseline and locked had agreed).

    Returns ``(final, changed, margins)``: ``final[i]`` is the span's value
    after applying only lock-attributable, margin-gated changes;
    ``changed[i]`` is ``final[i] != displayed[i]`` (convenience -- callers
    would otherwise recompute this themselves); ``margins`` is the LOCKED
    run's per-span margin (an approximation for spans where ``final``
    reverted to ``displayed`` -- that margin describes the locked run's own
    candidate, not a confidence in displayed, but this is a display-only
    confidence proxy, not a scored quantity, and documented as such at its
    call site in api.py).
    """
    n = len(acoustic_logp)
    no_locks: list[tuple[int, int] | None] = [None] * n
    baseline_chosen, _ = lattice_rescore(acoustic_logp, displayed, no_locks, context_scorer,
                                        lam=lam, K=K, delta=delta)
    locked_chosen, locked_margins = lattice_rescore(acoustic_logp, displayed, locks, context_scorer,
                                                    lam=lam, K=K, delta=delta)
    ctx_cache: dict = {}
    final: list[tuple[int, int]] = []
    changed: list[bool] = []
    for j in range(n):
        if locks[j] is not None:
            value = locked_chosen[j]                       # never margin-gated
        elif locked_chosen[j] == baseline_chosen[j]:
            value = displayed[j]                           # not lock-attributable at all
        elif margin_gate <= 0.0:
            value = locked_chosen[j]                        # gate off -> old behaviour
        else:
            prev_tok = locked_chosen[j - 1] if j > 0 else None
            next_tok = locked_chosen[j + 1] if j < n - 1 else None
            score_new = _candidate_score(acoustic_logp[j], prev_tok, next_tok, displayed[j],
                                         locked_chosen[j], context_scorer, lam, delta, ctx_cache)
            score_base = _candidate_score(acoustic_logp[j], prev_tok, next_tok, displayed[j],
                                          baseline_chosen[j], context_scorer, lam, delta, ctx_cache)
            value = locked_chosen[j] if (score_new - score_base) >= margin_gate else displayed[j]
        final.append(value)
        changed.append(value != displayed[j])
    return final, changed, locked_margins
