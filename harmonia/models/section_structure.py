"""Section-structure inference (issue #22): recover AABA / A-B-Bridge section
boundaries from a *symbolic* chord sequence rather than from acoustic novelty.

Why symbolic, not acoustic
--------------------------
The production pipeline segments with a cosine-*novelty* detector (gmerge) tuned
at the chord level.  Novelty detects local contrast, which in jazz is high
*everywhere* (a ii–V every two bars), so it fragments the song into chord cells
and never sees the 8/16-bar section grid.  Section structure lives in
*repetition* — bar i ≈ bar i+16 — not in local contrast.

A premise check (scripts/premise_check_chord_ssm.py, 8 genuine AABA standards)
confirmed the mechanism:

  * On the symbolic chord SSM the bridge B is correctly *less* similar to A than
    the two A sections are to each other (bridge-contrast +0.05..+0.11 on 6/8
    tunes; chord-SSM beats the acoustic SSM 7/8).
  * On the *acoustic* (Basic-Pitch) SSM the same bridge-contrast is ~0 (±0.003):
    the audio self-similarity carries essentially no section signal on these
    metronomic renders.  This is exactly issue #22's diagnosis.
  * Across 371 AABA tunes the bridge is correctly the odd-one-out in 85% of
    cases (mean margin +0.08) — real, but weak and noisy, so the detector must
    lean on the strong *form-length prior*, not on a raw novelty peak.

What this module does NOT solve
-------------------------------
  * It infers section *lengths* and *boundaries*, not section *labels* (which
    span is "A" vs "B").  Grouping repeated sections (labelling) is left to a
    downstream pass.
  * It assumes a single global base section length for the whole tune; a tune
    that changes metre or mixes 8- and 12-bar sections (e.g. a 12-bar-blues
    bridge) will be snapped to one length.
  * The chord sequence is taken as given; if the upstream per-beat root/quality
    predictions are wrong the SSM inherits that noise (the premise numbers above
    are the GT-chord ceiling, not the end-to-end figure).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "build_chord_ssm",
    "detect_section_boundaries",
    "label_sections",
    "estimate_base_period_bars",
    "build_progression_model",
    "load_progression_model",
    "correct_section_phase",
    "apply_phase_shift",
    "librosa_laplacian_sections",
    "is_degenerate_sections",
    "barlocked_sections",
]


def build_chord_ssm(chord_sequence: list[tuple[int | None, int]], n_pitches: int = 12) -> np.ndarray:
    """Cosine self-similarity matrix of a per-beat symbolic chord sequence.

    Args:
        chord_sequence: one ``(root, quality_idx)`` tuple per beat.  ``root`` is
            a pitch class ``0..n_pitches-1`` **already relative to the tonic**
            (the caller subtracts the key so the representation is key-invariant
            and two tunes in different keys share a geometry).  A ``root < 0``
            marks a no-chord / unknown beat and yields an all-zero row.
        n_pitches: size of the root one-hot block (12 for 12-TET).

    Representation per beat: ``[root one-hot (n_pitches) | quality one-hot]``.
    Root and quality are separate one-hot blocks so that two chords sharing a
    root but differing in quality (Cmaj7 vs C7) are partially — not fully —
    dissimilar, and two chords sharing a quality but not a root likewise.

    Returns:
        ``(n_beats, n_beats)`` float32 cosine-similarity matrix in ``[0, 1]``.
        Empty (all-zero) rows have similarity 0 to everything.
    """
    n_beats = len(chord_sequence)
    if n_beats == 0:
        return np.zeros((0, 0), dtype=np.float32)

    n_qual = max((q for _, q in chord_sequence if q >= 0), default=0) + 1
    feat = np.zeros((n_beats, n_pitches + n_qual), dtype=np.float32)
    for b, (root, qual) in enumerate(chord_sequence):
        if root is None or root < 0:
            continue
        feat[b, root % n_pitches] = 1.0
        if qual is not None and qual >= 0:
            feat[b, n_pitches + qual] = 1.0

    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    feat_n = feat / np.clip(norm, 1e-9, None)
    ssm = feat_n @ feat_n.T
    return np.clip(ssm, 0.0, 1.0).astype(np.float32)


def _repetition_score(ssm: np.ndarray, lag: int) -> float:
    """Mean of the ``lag``-th off-diagonal of the SSM = strength of repetition
    at period ``lag`` beats (an autocorrelation of beat-to-beat similarity)."""
    if lag <= 0 or lag >= ssm.shape[0]:
        return 0.0
    return float(np.diagonal(ssm, offset=lag).mean())


def estimate_base_period_bars(
    chord_ssm: np.ndarray,
    beats_per_bar: int = 4,
    form_lengths: tuple[int, ...] = (4, 8, 16, 32, 64),
    rep_floor: float = 0.25,
) -> int | None:
    """Base section/loop length in bars from the SSM repetition score.

    Same logic as step 1 of :func:`detect_section_boundaries` (smallest form
    length whose lag-repetition score clears ``rep_floor``; argmax fallback),
    factored out so phase correction can operate at the same grain the boundary
    detector chose.  Adds ``4`` to the default candidate set because pop/rock
    loops are frequently a 4-bar cycle (e.g. Let It Be's C-G-Am-F), shorter than
    the 8-bar jazz A-section floor.  Returns ``None`` when the song is shorter
    than the smallest candidate section.
    """
    n_beats = chord_ssm.shape[0]
    cands = [L for L in form_lengths if 0 < L * beats_per_bar < n_beats]
    if not cands:
        return None
    scores = {L: _repetition_score(chord_ssm, L * beats_per_bar) for L in cands}
    above = [L for L in sorted(cands) if scores[L] >= rep_floor]
    return above[0] if above else max(scores, key=lambda L: scores[L])


def detect_section_boundaries(
    chord_ssm: np.ndarray,
    beats_per_bar: int = 4,
    form_lengths: tuple[int, ...] = (8, 16, 32, 64),
    merge_threshold: float = 0.60,
    rep_floor: float = 0.25,
    min_section_bars: int = 4,
) -> list[int]:
    """Beat indices of inferred section boundaries.

    Algorithm (form-length prior + repetition merge):

    1. **Pick a base section length.**  Score each candidate section length
       ``L`` (in bars, from ``form_lengths``) by the SSM repetition score at
       lag ``L·beats_per_bar`` — how self-similar the song is when shifted by
       exactly one candidate section.  Take the *smallest* ``L`` whose score
       clears ``rep_floor`` (fall back to the argmax if none does).  Smallest,
       not strongest, because sections nest: an AABA with a 16-bar first half
       (iReal ``A16 B8 A8``) scores *higher* at lag 16 (the tail of the A16
       aligns with the closing A) than at lag 8, yet its true grain is 8 bars —
       step 3 rebuilds the 16-bar block by merging, but a 16-bar base can never
       recover the interior 8-bar boundary it skipped.  Scoring only standard
       jazz form lengths *is* the form-length prior: an unconstrained lag sweep
       is dominated by the trivial lag≈1 peak (issue #1 over-smoothing).

    2. **Lay a uniform section grid** at every multiple of ``L*`` bars.

    3. **Merge adjacent blocks that repeat.**  Two adjacent ``L*``-bar blocks
       whose mean cross-similarity exceeds ``merge_threshold`` are the same
       section played twice (e.g. the two 8-bar A phrases that iReal marks as a
       single ``A16``); merge them so the returned boundaries match the
       iReal-form convention of one boundary per *label* change, not per phrase.

    Returns a sorted list of interior boundary beat indices (0 and the final
    beat are **not** included — callers add the song ends themselves).  Returns
    ``[]`` when the song is shorter than the smallest candidate section.

    NOT solved here: section *phase* (this assumes section 0 starts at beat 0 —
    a pickup/intro offset is not detected, cf. periodicity.find_loop_phase) and
    section *labelling* (which block is A vs B).
    """
    n_beats = chord_ssm.shape[0]
    if n_beats < min(form_lengths) * beats_per_bar:
        return []

    # 1. base section length
    cands = [L for L in form_lengths if 0 < L * beats_per_bar < n_beats]
    if not cands:
        return []
    scores = {L: _repetition_score(chord_ssm, L * beats_per_bar) for L in cands}
    above = [L for L in sorted(cands) if scores[L] >= rep_floor]
    base_L = above[0] if above else max(scores, key=lambda L: scores[L])
    step = base_L * beats_per_bar

    # 2. uniform grid at multiples of the base length
    grid = list(range(0, n_beats, step))
    if grid[-1] < n_beats:
        grid.append(n_beats)
    blocks = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)]

    # 3. merge adjacent repeated blocks
    merged: list[list[int]] = [list(blocks[0])]
    for s, e in blocks[1:]:
        ps, pe = merged[-1]
        # compare only the overlapping-length prefix so a short trailing block
        # (a tag/coda) does not spuriously merge on partial overlap
        w = min(pe - ps, e - s)
        if w > 0:
            cross = chord_ssm[ps:ps + w, s:s + w]
            sim = float(np.mean(np.diagonal(cross)))  # slot-aligned similarity
        else:
            sim = 0.0
        if sim >= merge_threshold:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Absorb runt blocks (< min_section_bars) into a neighbour.  On detected
    # (tempo-grid) beats n_beats is rarely an exact multiple of the section
    # step, so the final grid block is a short ragged tail — fold it back rather
    # than emit a spurious section boundary a few beats before the song end.
    min_len = min_section_bars * beats_per_bar
    i = 0
    while len(merged) > 1 and i < len(merged):
        s, e = merged[i]
        if e - s < min_len:
            if i == len(merged) - 1:          # tail -> extend previous
                merged[i - 1][1] = e
                merged.pop(i)
            else:                              # merge into following block
                merged[i + 1][0] = s
                merged.pop(i)
        else:
            i += 1

    # interior boundaries only
    return [b for _, b in merged[:-1]]


def label_sections(
    chord_ssm: np.ndarray,
    boundary_beats: list[int],
    sim_threshold: float = 0.70,
) -> list[str]:
    """Assign A/B/C labels to detected sections by clustering chord-SSM fingerprints.

    Two sections get the same label iff their mean pairwise chord-SSM similarity
    exceeds ``sim_threshold``.

    Algorithm:
        1. For each section *i* (beats ``boundary_beats[i]..boundary_beats[i+1]``),
           compute its **fingerprint** = L2-normalised mean of that section's rows
           in ``chord_ssm`` — a (n_beats,) summary of what chords appear when.
        2. Build a (n_sections × n_sections) cosine-similarity matrix ``S`` from
           the fingerprints (dot-product of L2-normalised vectors = cosine).
        3. **Greedy assignment**: first section → "A".  For each subsequent
           section, compare its fingerprint against the *representative* of each
           existing label (the first section that received that label) in
           alphabetical order.  If ``S[new, rep_A] > sim_threshold`` → assign "A";
           else if ``S[new, rep_B] > threshold`` → "B"; … else open a new letter.

    Returns a list of single-letter strings, one per section, e.g.
    ``['A', 'A', 'B', 'A']``.  Returns ``[]`` when there are no sections.

    NOT solved here: multi-phrase merging (an "A16" merged section is still
    labelled as one "A", not two separate A8 labels — the caller controls
    boundary resolution via ``detect_section_boundaries``).
    """
    n_sections = len(boundary_beats) - 1
    if n_sections <= 0:
        return []

    # 1. Fingerprints: mean of each section's SSM rows, L2-normalised
    fingerprints: list[np.ndarray] = []
    for i in range(n_sections):
        s, e = boundary_beats[i], boundary_beats[i + 1]
        fp = chord_ssm[s:e].mean(axis=0).astype(np.float64)
        n = np.linalg.norm(fp)
        fingerprints.append(fp / n if n > 1e-9 else fp)

    # 2. Similarity matrix (cosine, since fingerprints are unit vectors)
    fps = np.array(fingerprints, dtype=np.float64)  # (n_sec, n_beats)
    S = fps @ fps.T  # (n_sec, n_sec), values in [-1, 1]

    # 3. Greedy label assignment
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labels: list[str] = [""] * n_sections
    rep_idx: list[int] = []  # representative (first occurrence) for each letter

    labels[0] = letters[0]
    rep_idx.append(0)

    for i in range(1, n_sections):
        assigned = False
        for j, rep in enumerate(rep_idx):
            if float(S[i, rep]) > sim_threshold:
                labels[i] = letters[j]
                assigned = True
                break
        if not assigned:
            next_j = len(rep_idx)
            labels[i] = letters[next_j % len(letters)]
            rep_idx.append(i)

    return labels


# ── Section-phase correction (issue #22, cycle-shift bug) ──────────────────────
#
# detect_section_boundaries recovers the section *length* but assumes phase 0
# (section 0 starts at beat 0).  On real audio a loop can start on the wrong bar
# of its cycle — e.g. Let It Be's C-G-Am-F (I-V-vi-IV in C) came out phased so
# the tonic C, which opens each 4-bar cycle, landed *last*.  periodicity.
# find_loop_phase anchors phase on ``is_downbeat``, which is POP909 ground truth
# and unavailable on YouTube audio, so it does not fix the production task.
#
# This module recovers phase from the *harmonic progression likelihood* instead:
# for a detected period of P bars, the P candidate phases are scored by how
# probable the resulting per-period chord progression is under a key-relative
# **trigram** language model (bigram backoff) with a start (BOS) distribution,
# and the most probable phase is chosen.  The start distribution — peaked on the
# tonic (I / i), because pop/jazz sections overwhelmingly open on the tonic — is
# what breaks the cyclic symmetry that transitions alone cannot: rotating a loop
# leaves every consecutive pair (and every interior triple) intact, so only the
# per-period BOS + boundary-cut effects differentiate phases.
#
# Trigram, not bigram (issue #21): the #21 premise-check showed jazz's resolving
# ii–V–I motif is a *trigram* pattern bigrams miss (63.8% cloze, MARGINAL), which
# is why the ProgressionEncoder was trained with ±6 context.  Commit d5bedb5's
# phase fix rebuilt a bigram without noting that; this raises the order to a
# trigram (Witten-Bell backoff to the bigram for sparse contexts) while keeping
# the tonic-peaked BOS prior — that prior, not the model order, is the phase
# symmetry-breaker, so it is unchanged.
#
# Why NOT the ProgressionEncoder (issue #21) here: that encoder is a *quality*
# cloze model (root-relative, transpose-invariant) and emits no next-root / start
# signal, so its summed log-probs are invariant to a whole-loop rotation — it
# carries no phase information.  A trigram LM with a key-relative BOS distribution
# does, so we use that (task Option B).

_Q5 = ["maj", "min", "dom", "hdim", "dim"]
_N_Q5 = 5


def _key_to_tonic_pc(key_str: str) -> int | None:
    """Pitch class of an iReal key string like 'C', 'Bb', 'E-' (minor), 'F#-'.

    The trailing '-' marks minor mode and does not move the tonic pitch class.
    """
    if not key_str:
        return None
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    tok = key_str.strip()
    pc = base.get(tok[0].upper())
    if pc is None:
        return None
    if len(tok) > 1 and tok[1] in "#b":
        pc += 1 if tok[1] == "#" else -1
    return pc % 12


def build_progression_model(
    db_path: str | Path | None = None,
    corpora: tuple[str, ...] = ("jazz1460", "pop400", "blues50"),
    cache_path: str | Path | None = None,
    alpha: float = 0.5,
) -> dict | None:
    """Build a key-relative **trigram** progression LM (with bigram backoff).

    Returns a dict with:
        ``start`` : (12, 5) log-prob of a section-opening chord as
                    ``(root relative to tonic, q5-family)``.
        ``trans`` : (5, 12, 5) log-prob of ``(q_prev) -> (delta_root, q_next)``,
                    the bigram table (add-``alpha`` smoothed, log-normalised).  It
                    is kept both for backward compatibility and as the trigram's
                    backoff distribution.
        ``tri``   : (5, 12, 5, 12, 5) *raw counts* of a key-relative chord triple
                    ``(q_prev2, delta21, q_prev1) -> (delta_n, q_next)`` where all
                    root deltas are taken relative to the **middle** chord's root
                    (``delta21 = r_prev2 - r_prev1``, ``delta_n = r_n - r_prev1``),
                    so the table is transpose-invariant exactly like the bigram.
        ``ctx``   : (5, 12, 5) trigram context counts (``tri.sum`` over the last
                    two axes) — the backoff weight per context.

    Why trigram, not bigram (issue #21 / #22): the issue-#21 premise-check found
    bigrams miss jazz's ii–V–I motif (63.8% cloze, MARGINAL < 70%) *precisely
    because* the resolving third chord is conditionally informative given the two
    predecessors.  The prior phase-correction model (commit d5bedb5) rebuilt a
    bigram table without noting that; this upgrades it to a trigram with a
    Witten-Bell-style backoff to the bigram for unseen/rare contexts.  The
    **start prior is unchanged** (peaked on the tonic) — it, not the model order,
    is what breaks the cyclic rotation symmetry.

    The start distribution uses each song's *first* chord (a tonic proxy for a
    section opener).  Persists to ``cache_path`` (npz) when given.  Returns
    ``None`` if the corpus or its parser is unavailable (phase correction then
    degrades to a no-op).
    """
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    db_path = Path(db_path) if db_path else repo / "data" / "accomp_db" / "db.jsonl"
    if not db_path.exists():
        return None
    try:
        sys.path.insert(0, str(repo / "scripts"))
        from analyze_accomp_emission import song_chord_spans  # noqa: E402

        from harmonia.models.progression_encoder import fine_to_q5
    except Exception:
        return None

    import json

    start = np.full((12, _N_Q5), alpha, dtype=np.float64)
    trans = np.full((_N_Q5, 12, _N_Q5), alpha, dtype=np.float64)
    tri = np.zeros((_N_Q5, 12, _N_Q5, 12, _N_Q5), dtype=np.float64)

    for line in open(db_path):
        rec = json.loads(line)
        if rec.get("corpus") not in corpora:
            continue
        tonic = _key_to_tonic_pc(rec.get("key", ""))
        seq: list[tuple[int, int]] = []
        for _t0, _t1, root, qual in song_chord_spans(rec):
            q5 = fine_to_q5(qual)
            if q5 is None:
                continue
            seq.append((root % 12, q5))
        if len(seq) < 2:
            continue
        if tonic is not None:
            r0, q0 = seq[0]
            start[(r0 - tonic) % 12, q0] += 1.0
        for (ri, qi), (rj, qj) in zip(seq, seq[1:]):
            trans[qi, (rj - ri) % 12, qj] += 1.0
        # trigrams: (a, b, c) key-relative to the middle chord b's root
        for (ra, qa), (rb, qb), (rc, qc) in zip(seq, seq[1:], seq[2:]):
            tri[qa, (ra - rb) % 12, qb, (rc - rb) % 12, qc] += 1.0

    start_lp = np.log(start / start.sum())
    trans_lp = np.log(trans / trans.sum(axis=(1, 2), keepdims=True))
    ctx = tri.sum(axis=(3, 4))
    model = {
        "start": start_lp.astype(np.float32),
        "trans": trans_lp.astype(np.float32),
        "tri": tri.astype(np.float32),
        "ctx": ctx.astype(np.float32),
    }

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **model)
    return model


def load_progression_model(
    cache_path: str | Path | None = None, rebuild: bool = False
) -> dict | None:
    """Load the cached trigram progression LM, building + caching it if missing.

    Defaults the cache to ``data/cache/chord_progression_model.npz``.  Returns
    ``None`` if it can neither be loaded nor built (caller then skips phase
    correction).  A cache missing the trigram tables (an old bigram-only ``.npz``)
    is rebuilt so callers always get the trigram model.
    """
    repo = Path(__file__).resolve().parent.parent.parent
    cache_path = (
        Path(cache_path) if cache_path
        else repo / "data" / "cache" / "chord_progression_model.npz"
    )
    if cache_path.exists() and not rebuild:
        try:
            z = np.load(cache_path)
            if "tri" in z.files:
                return {k: z[k] for k in z.files}
        except Exception:
            pass
    return build_progression_model(cache_path=cache_path)


_BACKOFF_K = 5.0  # Witten-Bell-style pseudo-count: trigram weight = c / (c + K)


def _next_logprob(
    prev2: tuple[int, int] | None,
    prev1: tuple[int, int],
    cur: tuple[int, int],
    model: dict,
) -> float:
    """Log P(cur | prev1[, prev2]) under the trigram LM with bigram backoff.

    The trigram estimate is linearly interpolated with the (smoothed) bigram,
    ``p = λ·p_trigram_MLE + (1-λ)·p_bigram`` with ``λ = c / (c + K)`` where ``c``
    is the trigram *context* count — an unseen context (c=0) backs off cleanly to
    the pure bigram, a well-attested context trusts the trigram.  When ``model``
    carries no trigram table (an old bigram-only cache) this is the plain bigram.
    All root deltas are relative to ``prev1``'s root, matching the build.
    """
    trans = model["trans"]
    q1 = prev1[1]
    dn = (cur[0] - prev1[0]) % 12
    qn = cur[1]
    p_bi = float(np.exp(trans[q1, dn, qn]))
    tri = model.get("tri")
    if tri is None or prev2 is None:
        return float(np.log(max(p_bi, 1e-12)))
    ctx = model["ctx"]
    q2 = prev2[1]
    d21 = (prev2[0] - prev1[0]) % 12
    c = float(ctx[q2, d21, q1])
    if c <= 0.0:
        return float(np.log(max(p_bi, 1e-12)))
    lam = c / (c + _BACKOFF_K)
    p_ml = float(tri[q2, d21, q1, dn, qn]) / c
    p = lam * p_ml + (1.0 - lam) * p_bi
    return float(np.log(max(p, 1e-12)))


def _period_logprob(period: list[tuple[int, int] | None], model: dict) -> float:
    """Log-likelihood of one period's chord progression under the trigram LM.

    ``period`` is a per-bar list of ``(root_rel_to_tonic, q5)`` or ``None`` for a
    no-chord / unknown bar.  The first known chord is scored by the start (BOS)
    distribution; the second by the bigram; each subsequent chord by the trigram
    (with bigram backoff).  A ``None`` bar resets the context (both predecessors),
    so an unknown bar cannot corrupt a specific transition.
    """
    start = model["start"]
    lp = 0.0
    prev1: tuple[int, int] | None = None
    prev2: tuple[int, int] | None = None
    for bar in period:
        if bar is None:
            prev1 = prev2 = None
            continue
        r, q = (bar[0] % 12, bar[1])
        if prev1 is None:
            lp += float(start[r, q])
        else:
            lp += _next_logprob(prev2, prev1, (r, q), model)
        prev2 = prev1
        prev1 = (r, q)
    return lp


def _beats_to_bars(
    chord_sequence: list[tuple[int, int] | None], beats_per_bar: int
) -> list[tuple[int, int] | None]:
    """Reduce a per-beat ``(root_rel, q5)`` sequence to one chord per bar (mode).

    The dominant (most frequent) chord in each bar is the bar's representative;
    ties break toward the bar's first chord.  ``None``/negative-root beats are
    ignored in the vote; a bar with no valid chord becomes ``None``.
    """
    bars: list[tuple[int, int] | None] = []
    for b0 in range(0, len(chord_sequence), beats_per_bar):
        window = chord_sequence[b0:b0 + beats_per_bar]
        counts: dict[tuple[int, int], int] = {}
        order: list[tuple[int, int]] = []
        for ch in window:
            if ch is None:
                continue
            r, q = ch
            if r is None or r < 0 or q is None or q < 0:
                continue
            key = (int(r) % 12, int(q))
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1
        if not counts:
            bars.append(None)
        else:
            best = max(order, key=lambda k: (counts[k], -order.index(k)))
            bars.append(best)
    return bars


def correct_section_phase(
    chord_sequence: list[tuple[int, int] | None],
    period_bars: int,
    beats_per_bar: int,
    model: dict,
) -> int:
    """Best phase offset (in *bars*, 0..period_bars-1) for the section grid.

    Tests every candidate phase of a ``period_bars``-bar loop and returns the one
    that maximises the summed per-period progression log-likelihood under
    ``model`` (a trigram LM from :func:`load_progression_model`).  ``0`` means the
    detected phase-0 grid is already best (or the input is too short / model
    missing).  The returned ``shift`` is such that placing section boundaries at
    bars ``shift, shift+period_bars, ...`` aligns each section to open on the
    most-probable (typically tonic) chord.

    ``chord_sequence`` is per-beat ``(root_rel_to_tonic, q5)`` (or ``None``); it
    is reduced to one chord per bar internally.  The bar sequence is treated as
    cyclic (a loop), so each candidate phase is a rotation.
    """
    if model is None or period_bars is None or period_bars < 2:
        return 0
    bars = _beats_to_bars(chord_sequence, beats_per_bar)
    n_bars = len(bars)
    if n_bars < 2 * period_bars:
        return 0
    n_periods = n_bars // period_bars

    best_shift, best_lp = 0, -np.inf
    for shift in range(period_bars):
        rot = bars[shift:] + bars[:shift]
        total = 0.0
        for j in range(n_periods):
            total += _period_logprob(rot[j * period_bars:(j + 1) * period_bars], model)
        if total > best_lp:
            best_lp, best_shift = total, shift
    return best_shift


def apply_phase_shift(
    boundary_beats: list[int], shift_bars: int, beats_per_bar: int, n_beats: int
) -> list[int]:
    """Shift a detected interior-boundary list by ``shift_bars`` bars.

    Slides every boundary forward by ``shift_bars * beats_per_bar`` beats and
    adds a boundary at the shift itself, so the leading ``shift_bars`` bars become
    a partial pickup section and every full section afterwards is phase-aligned.
    Boundaries that fall on 0 or at/after ``n_beats`` are dropped.  Returns a
    sorted, de-duplicated interior-boundary list (0 and ``n_beats`` excluded).
    """
    if shift_bars <= 0:
        return boundary_beats
    off = shift_bars * beats_per_bar
    shifted = {off} | {b + off for b in boundary_beats}
    return sorted(b for b in shifted if 0 < b < n_beats)


# ── Acoustic fallback: librosa-Laplacian section detection (McFee & Ellis 2014) ─
#
# The symbolic detector above (build_chord_ssm -> detect_section_boundaries)
# produces EMPTY output on ~half of real recordings (11/21 cached charts,
# docs/known_issues.md ★ STRUCTURE 2026-07-17) because its form-length prior
# `rep_floor` gate rejects the noisy per-beat chord estimates that real audio
# yields.  Its whole premise — "the acoustic SSM carries essentially no section
# signal" — was established on *metronomic MMA renders* and is FALSE for real
# recordings, which do carry measurable off-diagonal recurrence structure
# (0.016-0.030 density on 3 test songs).
#
# This function is the canonical online remedy: librosa's affinity recurrence
# matrix + Laplacian spectral clustering (McFee & Ellis, ISMIR 2014), the
# off-diagonal-stripe repetition detector.  Its output is the SAME shape as the
# symbolic path's `sections_out` dicts, so it plugs straight into the existing
# `P.sections`/`sectionChips` UI with no UI change.  It is intended as a
# FALLBACK ONLY — run it when detect_section_boundaries returned nothing, never
# in place of the symbolic path.
#
# Ported from scratchpad/librosa_struct.py (the research-agent repro).  Adds the
# min-section merge + sanity bounds the raw method lacks (it over-segments) so a
# bad result degrades to [] (status-quo empty) rather than to visible garbage.
#
# KNOWN LIMITATION (NOT fixed here, out of scope — issue #1): librosa's own beat
# tracker can lock a 2x tempo octave.  When it does, the beat *times* (and hence
# section start/end seconds) stay correct — only the derived `n_bars` count
# doubles.  We sanity-bound `n_bars` to a plausible musical range and reject an
# implausible whole result, so the blast radius of a tempo error is a wrong bar
# COUNT on otherwise-correctly-placed boundaries, never a garbage chart.

def librosa_laplacian_sections(
    audio_path: str | Path,
    *,
    n_types: int = 6,
    beats_per_bar: int = 4,
    min_section_bars: int = 4,
    max_sections: int = 24,
    max_section_bars: int = 64,
    sr: int = 22050,
) -> list[dict]:
    """Acoustic section detection fallback → list of section dicts.

    Returns a list of ``{"start_s", "end_s", "n_bars", "label"}`` dicts (the
    exact shape produced by ``chord_pipeline_v1`` §10b), with A/B/C letter
    labels where the SAME letter marks a detected repeat (verse2 == verse1).
    Returns ``[]`` (i.e. defers to the status-quo empty state) on any failure
    or when the result is not musically plausible — so this can only ever
    *add* sections where there were none, never make a chart worse.

    Args:
        audio_path: path to the recording (loaded mono at ``sr``).
        n_types: number of section *types* (k-means clusters) to seek.
        beats_per_bar: assumed metre for the bars<->beats conversion (4/4).
        min_section_bars: sections shorter than this are merged into a
            neighbour (kills the raw method's over-segmentation / runts).
        max_sections: reject the whole result if more sections than this
            survive (an over-fragmented, untrustworthy segmentation).
        max_section_bars: clamp each section's reported ``n_bars`` to this
            (bounds the blast radius of a 2x tempo-octave lock, issue #1).
        sr: load sample rate.
    """
    try:
        import numpy as _np
        import librosa
        from scipy.ndimage import median_filter
        from sklearn.cluster import KMeans
    except Exception:
        return []

    try:
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
        dur = len(y) / sr
        if dur < 20.0:  # too short to have a meaningful section structure
            return []

        # beat-synchronous CQT (harmonic content, 3 bins/semitone)
        C = _np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12 * 3, n_bins=7 * 12 * 3))
        _tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        if beats is None or len(beats) < 4 * beats_per_bar:
            return []
        Csync = librosa.util.sync(C, beats, aggregate=_np.median)
        n = Csync.shape[1]
        if n < 8:
            return []

        # affinity recurrence (off-diagonal stripes = repeats) + diagonal enhance.
        # The median filter must run in TIME-LAG space (librosa.segment.timelag_filter),
        # not raw time-time: repeat structure shows up as constant-lag diagonal
        # stripes, and smoothing along a raw row mixes unrelated time offsets,
        # destroys R's symmetry (eigh then silently reads only one triangle), and
        # does not enhance stripes at all. This is librosa's own documented usage
        # (see librosa.segment.timelag_filter docstring / ISMIR14 example).
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
        Rf = librosa.segment.timelag_filter(median_filter)(R, size=(1, 7))
        # sequence (path) matrix linking consecutive beats
        path_dist = _np.sum(_np.diff(Csync, axis=1) ** 2, axis=0)
        sigma = _np.median(path_dist)
        path_sim = _np.exp(-path_dist / (sigma + 1e-9))
        R_path = _np.diag(path_sim, 1) + _np.diag(path_sim, -1)
        deg_path = _np.sum(R_path, axis=1)
        deg_rec = _np.sum(Rf, axis=1)
        mu = deg_path.dot(deg_path + deg_rec) / (_np.sum((deg_path + deg_rec) ** 2) + 1e-9)
        A = mu * Rf + (1 - mu) * R_path
        # symmetric normalized Laplacian eigenvectors
        Dinv = _np.diag(1.0 / (_np.sum(A, axis=1) + 1e-9) ** 0.5)
        L = _np.eye(A.shape[0]) - Dinv.dot(A).dot(Dinv)
        _evals, evecs = _np.linalg.eigh(L)
        k = int(min(n_types, evecs.shape[1], n))
        if k < 2:
            return []
        X = librosa.util.normalize(evecs[:, :k], norm=2, axis=1)
        seg_ids = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        # temporal smoothing kills per-beat flicker (~16-beat majority window)
        seg_ids = median_filter(seg_ids, size=17, mode="nearest")

        bt = librosa.frames_to_time(beats, sr=sr)

        # beat-frame labels -> contiguous (start_beat, end_beat, label) runs
        runs: list[list[int]] = []  # [start_beat, end_beat, label]
        cur = int(seg_ids[0])
        start = 0
        for i in range(1, len(seg_ids)):
            if int(seg_ids[i]) != cur:
                runs.append([start, i, cur])
                cur = int(seg_ids[i])
                start = i
        runs.append([start, len(seg_ids), cur])

        # merge runt sections (< min_section_bars) into their more-similar
        # neighbour (here: the neighbour they abut; prefer the previous run so
        # a short lead-in folds into what precedes it)
        min_beats = min_section_bars * beats_per_bar
        merged: list[list[int]] = []
        for run in runs:
            s, e, lab = run
            if (e - s) < min_beats and merged:
                merged[-1][1] = e  # absorb into previous, keep prev label
            elif (e - s) < min_beats and not merged:
                merged.append(run)  # first run too short: keep, next may absorb
            else:
                merged.append(run)
        # second pass: a still-too-short first run folds forward
        if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_beats:
            merged[1][0] = merged[0][0]
            merged.pop(0)

        if not merged or len(merged) > max_sections:
            return []

        # relabel cluster ids -> A/B/C by first appearance (repeat = same letter)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        id_to_letter: dict[int, str] = {}
        sections: list[dict] = []
        for s, e, lab in merged:
            if lab not in id_to_letter:
                id_to_letter[lab] = letters[len(id_to_letter) % len(letters)]
            t0 = float(bt[s]) if s < len(bt) else dur
            t1 = float(bt[e]) if e < len(bt) else dur
            if t1 <= t0:
                continue
            n_bars = int(round((e - s) / beats_per_bar))
            n_bars = max(1, min(n_bars, max_section_bars))  # bound tempo-error blast radius
            sections.append({
                "start_s": round(t0, 3),
                "end_s": round(t1, 3),
                "n_bars": n_bars,
                "label": id_to_letter[lab],
            })

        # final plausibility gate: need >=2 sections and a non-degenerate map
        if len(sections) < 2:
            return []
        return sections
    except Exception:
        return []


# ── Degenerate-symbolic detector (fallback re-target, 2026-07-17) ───────────────
#
# The §10b symbolic detector is non-empty on ~all real recordings under current
# code (the old "52% empty" premise was stale, docs/known_issues.md ★ STRUCTURE
# 2026-07-17), but its `label_sections` cosine clustering COLLAPSES on noisy
# real-audio chord fingerprints: it either assigns nearly every section the SAME
# letter (no repeat variety) or the boundary detector over-fragments the song.
# Either way the section-REPEAT feature is broken.  This predicate flags that
# state so §10c can re-target the librosa-Laplacian fallback from "symbolic empty"
# (never happens) to "symbolic degenerate" (the real failure).
#
# Thresholds — each anchored to an OBSERVED pathology + a 3-song iReal-GT
# validation (V-measure vs iReal *A/*B/*C form; docs/known_issues.md entry):
#   * distinct-label collapse (<=1 letter): Autumn Leaves symbolic "AAAA" —
#     objectively zero repeat structure.  librosa V_F 0.18 > symbolic 0.00.
#   * dominant-label collapse (one letter >=85% of >=4 sections): Goodbye Yellow
#     Brick Road symbolic "A"+11xB (0.92).  librosa V_F 0.64 > symbolic 0.19.
#     The 0.85 cut is deliberately ABOVE Chain Of Fools' "ABBBBBABBBB" (B=0.82),
#     which is NOT collapsed — there symbolic V_F 0.50 BEATS librosa 0.40, so the
#     gate correctly leaves it on the symbolic path.  0.85 sits inside the task's
#     stated 80-90% band; a milder 0.75-style collapse (e.g. Happy "AAAABAAB") is
#     left to the symbolic path on purpose — this gate is high-precision, it fires
#     only when the collapse is severe enough that "no structure" is the honest
#     read.
#   * gross over-segmentation (>=18 sections): ABBA Chiquitita 27, Commodores
#     Easy 20 (docs/known_issues.md) — the form-level grouping is gone.  No
#     validated well-formed chart reached 18 sections; a real pop/jazz form is
#     ~4-12 sections.
#
# Returns False (NOT degenerate) for a healthy multi-label chart, and for a
# trivially short chart (<2 sections) where "repeat structure" is undefined and
# there is nothing for the fallback to improve.

def is_degenerate_sections(
    sections: list[dict],
    *,
    max_label_frac: float = 0.85,
    dominant_min_sections: int = 4,
    overseg_min_sections: int = 18,
) -> bool:
    """True iff a symbolic ``sections_out`` list has no usable repeat structure.

    A section list is degenerate when its A/B/C labelling has collapsed to (near-)
    one letter — so it encodes no repeat variety — or when it has over-segmented
    into implausibly many sections.  Both states mean the section-REPEAT feature
    is broken and the acoustic librosa-Laplacian fallback should replace it.

    Args:
        sections: list of ``{"start_s","end_s","n_bars","label"}`` dicts.
        max_label_frac: a single label covering this fraction or more of the
            sections (with at least ``dominant_min_sections`` sections) is a
            dominant-label collapse.
        dominant_min_sections: minimum section count for the dominant-label rule
            (with only 2-3 sections a shared letter can be a genuine repeat).
        overseg_min_sections: this many sections or more is gross
            over-segmentation.

    Returns:
        ``True`` if degenerate (fallback should fire), ``False`` otherwise —
        including for ``< 2`` sections, where repeat structure is undefined.
    """
    n = len(sections)
    if n < 2:
        return False
    labels = [s.get("label", "?") for s in sections]
    distinct = len(set(labels))
    if distinct <= 1:
        return True  # total collapse: every section one letter
    if n >= overseg_min_sections:
        return True  # gross over-segmentation: form grouping lost
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    if n >= dominant_min_sections and max(counts.values()) / n >= max_label_frac:
        return True  # dominant-label collapse: no repeat variety
    return False


# ── Bar-grid-locked, repetition-first section pass (2026-07-19) ─────────────────
#
# Motivation (docs/known_issues.md ★ STRUCTURE 2026-07-18 "GRID PHASE
# MISALIGNMENT"): on the LIVE nnls24 path `_section_fallback` fills sections with
# the librosa-Laplacian ACOUSTIC detector, which is phase-blind w.r.t. the bar
# grid.  On a two-chord vamp (Mayer Hawthorne "Just Ain't Gonna Work Out") the
# chord-content SSM is near-uniform, so acoustic novelty fires at production
# changes / vocal entries and boundaries land MID-PHRASE (user report 2026-07-19:
# chips at bars 11/29/45 are off the 4-bar grid).
#
# This pass works from the (good) predicted CHORD sequence instead of audio, on
# the 4-bar-locked grid, encoding the user's two structural principles:
#   1. A = the PREDOMINANT (most-repeated) section — the largest content cluster.
#   2. INTRO = the prefix before the first occurrence of A, when it differs.
# All boundaries are 4-bar-multiples of the (bestfit) bar grid BY CONSTRUCTION,
# so nothing can land mid-phrase.  This does NOT solve:
#   * metre changes / mixed section lengths (a single global L per song);
#   * a song whose true A really is not the most frequent section (rare —
#     principle 1 is the user's spec);
#   * it inherits upstream chord noise: garbage chords -> garbage sections
#     (same caveat as the symbolic §10b detector).  Concretely, if the SAME
#     material is predicted with different chords in different repeats (noisy
#     quality head), a single section CLUSTER can SPLIT into two labels — the
#     Mayer vamp splits under the raw nnls24 quality head (first post-intro
#     block -> B) but not under the deployed music-x-lab quality (-> A@bar4).
#     The A/B *letters* are therefore as reliable as the chord input; the
#     Intro split + 4-bar-locked boundaries are robust to this.
#   * a near-single-chord loop (Chain Of Fools, Let It Be's I-V-vi-IV) has no
#     chord-content contrast to separate sections -> collapses to one label;
#     the caller (`is_degenerate_sections` gate) defers those to the acoustic
#     detector, whose timbre signal is the honest last resort there.

def _complete_linkage(sim: np.ndarray, threshold: float) -> list[int]:
    """Threshold complete-linkage clustering on a similarity matrix.

    Two clusters merge iff their *minimum* pairwise similarity (complete linkage)
    is >= ``threshold``.  Returns a cluster id per item (0-based, in first-
    appearance order).  Complete linkage (not single/average) is the validated
    choice for grouping repeated sections without chaining unrelated blocks
    through a bridge (docs/known_issues.md ★ STRUCTURE: it correctly groups
    Autumn Leaves' A/A repeat where pairwise ranking fails).
    """
    n = sim.shape[0]
    clusters: list[list[int]] = [[i] for i in range(n)]
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        best = (-1.0, -1, -1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                # complete linkage: worst (min) similarity across the two groups
                mn = min(sim[i, j] for i in clusters[a] for j in clusters[b])
                if mn > best[0]:
                    best = (mn, a, b)
        if best[0] >= threshold:
            a, b = best[1], best[2]
            clusters[a] = clusters[a] + clusters[b]
            clusters.pop(b)
            merged = True
    # assign ids in first-appearance order
    order = sorted(range(len(clusters)), key=lambda c: min(clusters[c]))
    lab = [0] * n
    for new_id, c in enumerate(order):
        for i in clusters[c]:
            lab[i] = new_id
    return lab


def _as_bar_matrix(bar_feat) -> np.ndarray:
    """Coerce the per-bar input to a float ``(n_bars, D)`` matrix.

    Accepts either a ready-made ``(n_bars, D)`` array (the intended production
    input: per-bar 12-d NNLS root POSTERIORS, rows summing to ~1) OR a list of
    ``(root, _qual)`` / ``root`` / ``None`` bar tuples (a convenience for tests
    and offline chord-only drivers), which is expanded to a 12-d root one-hot —
    a valid vertex of the same probability simplex the soft posterior lives on.
    """
    if isinstance(bar_feat, np.ndarray):
        return bar_feat.astype(np.float64)
    n = len(bar_feat)
    mat = np.zeros((n, 12), dtype=np.float64)
    for b, ch in enumerate(bar_feat):
        if ch is None:
            continue
        r = ch[0] if isinstance(ch, (tuple, list)) else ch
        if r is not None and r >= 0:
            mat[b, int(r) % 12] = 1.0
    return mat



def _derive_loop_period(fn: np.ndarray, max_loop: int = 8, tol: float = 0.85) -> int:
    """Smallest bar-lag >= 2 whose recurrence is within ``tol`` of the strongest
    (the MINIMAL repeating loop unit — user's 2026-07-19 principle).  Lag 1 is
    excluded: on real audio the per-bar root posterior has runs of one chord, so
    lag-1 persistence is trivially high and is NOT the musical loop.  Returns 2 as
    a safe floor when nothing recurs.
    """
    n = fn.shape[0]
    hi = min(max_loop, n // 2)
    if hi < 2:
        return 2
    ssm = fn @ fn.T
    lag = {k: float(np.diagonal(ssm, offset=k).mean()) for k in range(2, hi + 1)}
    best = max(lag.values())
    for k in range(2, hi + 1):
        if lag[k] >= tol * best:
            return k
    return 2


def barlocked_sections(
    bar_feat,
    bar_times: list[tuple[float, float]],
    *,
    tonic_pc: int | None = None,
    max_loop: int = 8,
    max_k: int = 5,
    intro_sim: float = 0.5,
    smooth_units: float = 2.5,
    min_run_units: int = 2,
    max_families: int = 4,
) -> list[dict]:
    """Derived-grain, loop-family section detection from the per-bar CHORD GRID.

    Replaces the earlier FIXED 4/8-bar block clustering (which the user rejected:
    it mixed the song's two 2-bar loops -- Emaj7|F#m7 = A and G#m7|F#m7 = B --
    inside single blocks).  New design credits the user's 2026-07-19 principle:
    *detect the minimal repeating loop unit first, then take maximal contiguous
    runs of one loop family*.

    Pipeline (chord grid only -- per-bar NNLS root POSTERIOR, never acoustic):
      1. **Derive the loop period** ``p`` from the bar-level lag-recurrence
         profile (:func:`_derive_loop_period`) -- per song, not a fixed grain.
      2. **Intro** = leading bars whose content does not recur later (max cosine
         to any bar >= 2 ahead below ``intro_sim``): the pre-loop junk.
      3. **Mean-centre** the per-bar posteriors over the body (removes the loop's
         SHARED chord -- e.g. the F#m7 common to both Mayer loops -- so the
         DISCRIMINATIVE content, Emaj7 vs G#m7, dominates), then smooth over ~p
         bars so a shared bar is pulled into its loop family.
      4. **Cluster bars into loop families** (k-means, k chosen by silhouette but
         biased to FEWER families) and **majority-smooth** the labels over ~p
         bars, merging turnaround/blip bars into the surrounding loop.
      5. **Sections = maximal runs** of one family; runs shorter than
         ``min_run_units`` loop units (turnarounds, e.g. A^7/B7 cadence bars) are
         absorbed into the neighbour they cadence toward.  Boundaries are snapped
         to the loop-unit (``p``-bar) grid, so none lands inside a loop unit.
      6. **A = the family with the most total bars**; B/C by first appearance;
         intro labelled ``Intro``; at most ``max_k`` labels.

    ``bar_feat`` is a per-bar ``(n_bars, 12)`` root-posterior matrix (or a list of
    ``(root, _q)`` / ``root`` / ``None`` tuples -> one-hot, for tests).
    ``bar_times`` gives each bar's ``(t0, t1)`` seconds on the renderer's uniform
    grid.  Returns ``[{start_s,end_s,n_bars,label}, ...]`` or ``[]`` (defer to the
    acoustic fallback) when there is no usable loop structure.
    """
    feat = _as_bar_matrix(bar_feat)
    n = feat.shape[0]
    if n < 6 or n != len(bar_times):
        return []
    fn = feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-9, None)

    # 1. derive the loop period (minimal repeating unit)
    p = _derive_loop_period(fn, max_loop=max_loop)

    # 2. intro = leading bars whose content is not part of the song's LOOPS.
    # Two deterministic (grid-robust) signals, whichever runs longer:
    #   (a) leading bars whose argmax root is not among the few dominant loop
    #       roots (the pre-song junk: Fdim/F7/... in Mayer, roots outside the
    #       Emaj7/F#m7/G#m7 loop set) -- this is stable to sub-bar grid noise,
    #       unlike a pure recurrence threshold which flipped on the live grid;
    #   (b) leading bars whose content does not recur >=2 bars later.
    from collections import Counter as _C
    arg_all = np.argmax(feat, axis=1)
    rc = _C(int(r) for r in arg_all)
    # dominant loop roots: those each covering >=8% of bars (a real loop chord),
    # capped to the top 4 (a loop has few distinct roots).
    thr = max(2, int(0.08 * n))
    loop_roots = {r for r, c in rc.most_common(4) if c >= thr}
    intro_root = 0
    for b in range(n):
        if int(arg_all[b]) not in loop_roots:
            intro_root = b + 1
        else:
            break
    ssm = fn @ fn.T
    intro_rec = 0
    for b in range(n):
        later = ssm[b, b + 2:]
        if later.size and float(later.max()) < intro_sim:
            intro_rec = b + 1
        else:
            break
    intro_end = max(intro_root, intro_rec)
    # Snap the intro length UP to a whole loop unit (p bars) so the first real
    # section starts on a loop boundary (an odd-bar A would split a 2-bar loop).
    if 0 < intro_end < n:
        intro_end = min(((intro_end + p - 1) // p) * p, n - p)
    body = fn[intro_end:]
    if len(body) < 4:
        return []

    # 3. mean-centre over the body + smooth over ~p bars
    Xc = fn - body.mean(0, keepdims=True)
    w = max(1, int(round(smooth_units)))
    ker = np.ones(w) / w
    Xs = np.column_stack([np.convolve(Xc[:, j], ker, mode="same") for j in range(12)])

    # 4. assign body bars to loop families by DETERMINISTIC nearest-centroid
    # seeding (not unsupervised k-means, which flips under sub-bar beat-grid noise
    # — 2026-07-19 gate failure).  Seed two centroids from actual musical roots:
    #   * A-seed = mean-centred posteriors of bars whose argmax root is the TONIC
    #     (the tonic-loop: Emaj7|F#m7 in Mayer), when the key is known;
    #   * B-seed = ...of bars whose argmax root is the strongest NON-tonic,
    #     non-shared root (G#m7 in Mayer).
    # Each bar is then labelled by the nearer seed (cosine on the smoothed
    # mean-centred vector).  Grid-robust because the seeds are anchored to root
    # identity, not to a random init that a 1-bar phase shift can re-cluster.
    try:
        from scipy.ndimage import median_filter
    except Exception:
        return []
    kmax = min(max_families, max_k - 1, len(body) - 1)
    if kmax < 1:
        return []
    arg = np.argmax(feat, axis=1)            # raw per-bar argmax root
    body_idx = list(range(intro_end, n))
    from collections import Counter as _Counter
    root_counts = _Counter(int(arg[b]) for b in body_idx)
    # candidate seed roots, most frequent first
    ranked = [r for r, _ in root_counts.most_common()]
    if tonic_pc is not None and (tonic_pc % 12) in root_counts:
        a_root = tonic_pc % 12
    else:
        a_root = ranked[0] if ranked else 0
    # B-seed root = the most frequent root that is neither the A root nor the
    # single SHARED root (the root present in ~both loops, i.e. very common and
    # adjacent to many other roots).  Simplest robust proxy: the most frequent
    # root != a_root whose bars are NOT overwhelmingly the global majority.
    b_root = next((r for r in ranked if r != a_root), None)
    if b_root is None:
        lab = np.full(n, 0, dtype=int); lab[:intro_end] = -1
    else:
        def _seed(root):
            rows = [b for b in body_idx if int(arg[b]) == root]
            return Xs[rows].mean(0) if rows else None
        sa, sb = _seed(a_root), _seed(b_root)
        if sa is None or sb is None:
            lab = np.full(n, 0, dtype=int); lab[:intro_end] = -1
        else:
            def _cos(u, v):
                return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
            lab = np.full(n, -1, dtype=int)
            for b in body_idx:
                lab[b] = 0 if _cos(Xs[b], sa) >= _cos(Xs[b], sb) else 1
    # majority-smooth labels over ~p bars (merge blips into the local loop)
    win = max(3, 2 * p + 1)
    body_lab = lab[intro_end:]
    if len(body_lab) >= win and set(body_lab) != {-1}:
        lab[intro_end:] = median_filter(body_lab, size=win, mode="nearest")

    # 5. runs + absorb short (< min_run_units*p) turnaround runs
    def _runs(a):
        out = []
        cur, st = a[0], 0
        for b in range(1, n):
            if a[b] != cur:
                out.append([st, b, int(cur)])
                cur, st = a[b], b
        out.append([st, n, int(cur)])
        return out

    cents = {}
    for l in set(int(x) for x in lab if x >= 0):
        cents[l] = Xs[[b for b in range(n) if lab[b] == l]].mean(0)
    min_run = max(1, min_run_units) * p
    while True:
        runs = _runs(lab)
        short = [(e - s, i) for i, (s, e, l) in enumerate(runs)
                 if l != -1 and (e - s) < min_run]
        if not short or len([r for r in runs if r[2] != -1]) <= 1:
            break
        short.sort()
        _, i = short[0]
        s, e, _l = runs[i]
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs) and runs[j][2] != -1]
        if not nb:
            break
        seg_c = Xs[s:e].mean(0)
        j = max(nb, key=lambda j: float(
            seg_c @ cents[runs[j][2]] /
            (np.linalg.norm(seg_c) * np.linalg.norm(cents[runs[j][2]]) + 1e-9)))
        lab[s:e] = runs[j][2]

    # 6. snap interior boundaries to the loop-unit (p-bar) grid
    runs = _runs(lab)
    snapped = [list(runs[0])]
    for s, e, l in runs[1:]:
        s2 = int(round(s / p) * p)
        s2 = min(max(s2, snapped[-1][0] + 1), n - 1)
        snapped[-1][1] = s2
        snapped.append([s2, e, l])
    runs = [r for r in snapped if r[1] > r[0]]
    # re-coalesce adjacent same-label after snapping
    merged = [runs[0]]
    for s, e, l in runs[1:]:
        if l == merged[-1][2]:
            merged[-1][1] = e
        else:
            merged.append([s, e, l])
    runs = merged

    # 7. labels: Intro; A = family with most bars; then B/C by first appearance
    from collections import Counter
    fam_bars = Counter()
    for s, e, l in runs:
        if l >= 0:
            fam_bars[l] += e - s
    if not fam_bars:
        return []
    # A = the loop family containing the TONIC chord when the key is known (the
    # user's Emaj7|F#m7 loop in E major); this is grid-independent, unlike a raw
    # most-bars vote which flips A/B under beat-grid noise.  Remaining families
    # ranked by total bars.
    if tonic_pc is not None:
        fam_tonic = {l: float(feat[[b for b in range(n) if lab[b] == l], tonic_pc % 12].sum())
                     for l in fam_bars}
        a_fam = max(fam_tonic, key=lambda l: fam_tonic[l])
        order = [a_fam] + [l for l, _ in fam_bars.most_common() if l != a_fam]
    else:
        order = [l for l, _ in fam_bars.most_common()]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    fam_letter = {l: letters[i] for i, l in enumerate(order)}
    labels = ["Intro" if l == -1 else fam_letter.get(l, "?") for s, e, l in runs]

    # enforce k<=max_k distinct labels (incl. Intro)
    def _distinct(ls):
        return list(dict.fromkeys(ls))
    guard = 0
    while len(_distinct(labels)) > max_k and guard < 26:
        guard += 1
        freq = Counter(l for l in labels if l not in ("Intro", "A"))
        if not freq:
            break
        rare = min(freq, key=lambda k: freq[k])
        for i, l in enumerate(labels):
            if l == rare:
                labels[i] = "A"

    # 8. emit section dicts (times from the uniform bar grid)
    out = []
    for (s, e, _l), lab_str in zip(runs, labels):
        t0 = float(bar_times[s][0])
        t1 = float(bar_times[min(e, n) - 1][1])
        if t1 <= t0:
            continue
        if out and out[-1]["label"] == lab_str:
            out[-1]["end_s"] = round(t1, 3)
            out[-1]["n_bars"] += int(e - s)
            continue
        out.append({"start_s": round(t0, 3), "end_s": round(t1, 3),
                    "n_bars": int(e - s), "label": lab_str})
    if len(out) < 2:
        return []
    return out
