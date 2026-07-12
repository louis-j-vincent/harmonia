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

import numpy as np

__all__ = ["build_chord_ssm", "detect_section_boundaries", "label_sections"]


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
