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
