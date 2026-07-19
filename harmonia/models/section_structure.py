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


def barlocked_sections(
    bar_feat,
    bar_times: list[tuple[float, float]],
    *,
    form_lengths: tuple[int, ...] = (8, 16, 32),
    rep_floor: float = 0.22,
    step_bars: int = 4,
    block_sim: float = 0.62,
    intro_sim: float = 0.55,
    max_k: int = 5,
    min_section_bars: int = 4,
) -> list[dict]:
    """Repetition-first, 4-bar-locked section detection from the CHORD GRID.

    The similarity input is the per-bar **probabilistic root** representation and
    a **scalar-product** self-similarity — NOT acoustic/timbre features (user
    directive 2026-07-19: acoustic/rhythm/production changes must not move section
    boundaries; the chord grid alone is authoritative for intro/A/B).  This reuses
    the validated per-bar-root-softmax + scalar-product prior art
    (docs/known_issues.md ★ STRUCTURE 2026-07-18 "PROBABILISTIC root-only ... TIES
    full-chord"; ``scratchpad/bar_distance_matrix.py`` / ``real_root_proba.py``).
    Root-only (dropping the noisier quality head) is deliberate — it is what makes
    the A/B *letters* robust to quality-head noise, the one fragility of the
    earlier root+quality variant.

    Similarity is the COSINE of the per-bar root distributions (each row L2-
    normalised, then a scalar product).  The normalisation is not decorative: on
    real audio a *raw* dot product of the softmaxes conflates per-bar confidence
    with similarity (a confident bar dots higher with everything), which
    compresses the same/different-section separation and makes complete-linkage
    over-fragment — verified across the 6 real-audio gate songs, where raw-dot
    loses the Mayer intro/A split that cosine recovers.

    Args:
        bar_feat: per-bar 12-d root posterior matrix ``(n_bars, 12)`` (rows ~sum
            to 1), OR a list of per-bar ``(root, _q)`` / ``root`` / ``None`` tuples
            (expanded to a root one-hot).  Because the whole song shares one key,
            a global tonic-roll is a constant rotation that leaves every pairwise
            dot product unchanged — so no key-normalisation is required here.
        bar_times: one ``(t0, t1)`` seconds tuple per bar (bestfit bar grid).
        form_lengths: candidate global section lengths in bars (form-length prior).
        rep_floor: minimum lag-repetition score (dot-product scale) for a form
            length to be accepted (else argmax fallback).
        step_bars: bar-grid quantum; ALL boundaries are multiples of this.
        block_sim: complete-linkage threshold (mean-posterior dot-product scale)
            for grouping equal-length blocks into section clusters.
        intro_sim: a leading bar-run is INTRO only if its per-bar dot-product to
            the A consensus is BELOW this (principle 2: intro must differ from A).
        max_k: cap on distinct labels incl. Intro (hard user rule k<=5).
        min_section_bars: sections shorter than this fold into a neighbour.

    Returns:
        A list of ``{"start_s","end_s","n_bars","label"}`` dicts, or ``[]`` when
        the song is too short / the grid can't support a section.
    """
    feat = _as_bar_matrix(bar_feat)
    n = feat.shape[0]
    if n < min(form_lengths) or n != len(bar_times):
        return []

    # L2-normalise each per-bar root distribution: the similarity is then the
    # COSINE (a normalised scalar product) of the chord-grid root posteriors.
    # Normalising removes the per-bar confidence confound of the raw softmax (a
    # confident bar would otherwise dot higher with everything), which on real
    # audio is what compresses the same/different-section separation and makes
    # complete-linkage over-fragment.  One-hot rows are already unit-norm, so the
    # chord-only convenience path is unchanged.
    fn = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.clip(fn, 1e-9, None)
    ssm = feat @ feat.T  # bar-level cosine SSM (probabilistic root)

    # 1. base section length L (form-length prior; bar-level lag = L bars)
    L = estimate_base_period_bars(ssm.astype(np.float32), beats_per_bar=1,
                                  form_lengths=form_lengths, rep_floor=rep_floor)
    if L is None or L >= n:
        L = min(form_lengths)
    if L >= n:
        return []

    # 2. enumerate equal-length (L-bar) blocks on the step-bar grid; a block is
    # summarised by its MEAN per-bar posterior (still a distribution), compared by
    # plain dot product (no L2 renorm — keep the probabilistic-root geometry).
    starts = list(range(0, n - L + 1, step_bars))
    if len(starts) < 2:
        return []
    block_feat = np.array([feat[s:s + L].mean(0) for s in starts])
    block_feat = block_feat / np.clip(np.linalg.norm(block_feat, axis=1, keepdims=True), 1e-9, None)
    bsim = block_feat @ block_feat.T

    # 3. cluster blocks (complete linkage); A = the largest cluster (principle 1)
    blab = _complete_linkage(bsim, block_sim)
    from collections import Counter
    counts = Counter(blab)
    # predominant cluster: most blocks, ties -> earliest first occurrence
    best_c = max(counts, key=lambda c: (counts[c],
                 -min(i for i, l in enumerate(blab) if l == c)))
    a_block_idx = [i for i, l in enumerate(blab) if l == best_c]
    first_a_bar = starts[a_block_idx[0]]
    a_consensus = block_feat[a_block_idx].mean(0)
    a_consensus = a_consensus / max(np.linalg.norm(a_consensus), 1e-9)

    # 4. intro = the leading run of bars whose CONTENT differs from the A
    # consensus (principle 2), measured per-bar so a half-vamp opening block
    # cannot hide the intro inside the A cluster.  The raw leading-mismatch run
    # is snapped UP to the step-bar grid (a detected intro is at least one 4-bar
    # phrase; boundaries stay phrase-locked).  ``first_a_bar`` is then that
    # snapped intro length (0 = song opens on A).
    per_bar_a = feat @ a_consensus  # dot product of each bar to the A consensus
    lead = 0
    while lead < len(per_bar_a) and per_bar_a[lead] < intro_sim:
        lead += 1
    if 0 < lead < n:
        # snap up to a whole step-bar phrase, but never past the first A block
        intro_bars = min(((lead + step_bars - 1) // step_bars) * step_bars,
                         starts[a_block_idx[0]] if a_block_idx[0] > 0 else n)
        intro_bars = max(intro_bars, step_bars)
        first_a_bar = intro_bars
    else:
        intro_bars = 0
        first_a_bar = 0

    # 5. build the 4-bar-locked section grid: [intro] + L-bar tiles from first_a
    spans: list[tuple[int, int]] = []
    if intro_bars > 0:
        spans.append((0, intro_bars))
    s = first_a_bar
    while s < n:
        e = min(s + L, n)
        spans.append((s, e))
        s = e
    # fold a ragged/short trailing tile into the previous section
    if len(spans) > 1 and (spans[-1][1] - spans[-1][0]) < min_section_bars:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()

    # 6. label sections: cluster their mean-posterior fingerprints (cosine)
    fps = np.array([feat[s0:e0].mean(0) for (s0, e0) in spans])
    fps = fps / np.clip(np.linalg.norm(fps, axis=1, keepdims=True), 1e-9, None)
    fsim = fps @ fps.T
    slab = _complete_linkage(fsim, block_sim)

    # which section cluster is A: the cluster of the FIRST non-intro section
    # (principles 1+2 unified: the intro precedes the first A, and the song
    # settles onto its predominant material right after the intro — user's
    # "next 8 bars = A").  Anchoring A to the first post-intro section GUARANTEES
    # "first A at bar <intro_end>" on the phrase grid, and on real songs that
    # first section is also the most-repeated one (verified A-is-most-frequent on
    # the gate corpus).  This is more robust than a global most-frequent vote,
    # which under soft-posterior noise can hand A to a late-appearing fragment.
    intro_present = intro_bars > 0
    first_sect_idx = 1 if intro_present else 0
    a_cluster = slab[first_sect_idx] if first_sect_idx < len(slab) else slab[0]

    # 7. assign letters: A to the A cluster, then B/C/... by first appearance;
    # the intro span (if any) is labelled "Intro" regardless of its cluster.
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cluster_letter: dict[int, str] = {a_cluster: "A"}
    next_letter = 1
    labels: list[str] = []
    for i, (s0, e0) in enumerate(spans):
        if intro_present and i == 0:
            labels.append("Intro")
            continue
        c = slab[i]
        if c not in cluster_letter:
            cluster_letter[c] = letters[next_letter % len(letters)]
            next_letter += 1
        labels.append(cluster_letter[c])

    # 8. enforce k<=max_k distinct labels (incl. Intro): collapse the rarest
    # non-A/non-Intro letters into the nearest surviving letter by fingerprint.
    def _distinct(labs):
        return list(dict.fromkeys(labs))
    guard = 0
    while len(_distinct(labels)) > max_k and guard < 26:
        guard += 1
        # count letter frequency among mergeable (not Intro, not A)
        freq: dict[str, int] = {}
        for lab in labels:
            if lab in ("Intro", "A"):
                continue
            freq[lab] = freq.get(lab, 0) + 1
        if not freq:
            break
        rare = min(freq, key=lambda k: freq[k])
        # merge every 'rare' section into its most-similar surviving section
        surv = [i for i, lab in enumerate(labels) if lab not in ("Intro", rare)]
        for i, lab in enumerate(labels):
            if lab != rare:
                continue
            if surv:
                j = max(surv, key=lambda j: float(fps[i] @ fps[j]))
                labels[i] = labels[j]
            else:
                labels[i] = "A"

    # 8b. coalesce ADJACENT same-label sections into one span (two consecutive
    # 8-bar A tiles are one "A" section, not two) — this keeps the chip count at
    # the FORM level and, on an all-one-label song (a single-chord loop), collapses
    # to a single span, which the len<2 guard below turns into "" (defer to the
    # acoustic detector, whose timbre signal is the honest last resort there).
    merged_spans: list[tuple[int, int]] = []
    merged_labels: list[str] = []
    for (s0, e0), lab in zip(spans, labels):
        if merged_labels and merged_labels[-1] == lab:
            ps0, _ = merged_spans[-1]
            merged_spans[-1] = (ps0, e0)
        else:
            merged_spans.append((s0, e0))
            merged_labels.append(lab)
    spans, labels = merged_spans, merged_labels

    # 9. bar spans -> section dicts (times from the bar grid)
    out: list[dict] = []
    for (s0, e0), lab in zip(spans, labels):
        t0 = float(bar_times[s0][0])
        t1 = float(bar_times[min(e0, n) - 1][1])
        if t1 <= t0:
            continue
        out.append({"start_s": round(t0, 3), "end_s": round(t1, 3),
                    "n_bars": int(e0 - s0), "label": lab})
    if len(out) < 2:
        return []
    return out
