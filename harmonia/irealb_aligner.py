"""irealb_aligner.py — align an iReal Pro GT chord sequence to audio-inferred chords.

The core use case: an inferred chord chart has accurate timestamps but imperfect
labels. The iReal Pro GT has correct chord labels but no timestamps. DTW alignment
transfers the timestamps from the inferred chart to the iReal chart.

Algorithm
---------
1. Parse + deduplicate the inferred sequence  (H3)
2. Parse the iReal sequence (already in iReal token format)
3. Extract compact iReal form (A×2, B×1, …) from section labels — not blindly tiled
4. Find the key offset: try all 12 semitone transpositions (H1, same as tab_aligner)
5. Estimate repeat count: from form structure + duration, constrained by SSM boundaries
6. Build harmonic SSM of inferred sequence; detect section boundaries in audio  (H5)
7. DTW per-section when sections are detectable, else global DTW  (H6)
8. Assign timestamps: interpolate within merged segments  (H4)
9. Return per-iReal-chord {label, t0, t1, section, bar}
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from harmonia.tab_aligner import (
    _chord_dist,
    _dtw,
    _family,
    _parse_ireal,
    _ROOTS,
    _chord_tones,
)


# ── Compact iReal form parser ─────────────────────────────────────────────────
# Parses the raw chord_string (BEFORE pyRealParser expansion) to recover the
# true repeat structure.  pyRealParser expands `{ ... }` to 2× repetitions and
# collapses `[ n1 ... ] [ n2 ... ]` first/second endings, hiding this info.
#
# iReal bracket semantics:
#   { ... }   = repeat bracket → section inside plays TWICE
#   [ ... ]   = regular bracket (no automatic repetition, may have n1/n2 endings)
#   *A *B …   = section markers (may appear before { or [ )
#
# Returns: list of SectionBlock(label, n_bars_unique, n_plays)
# Example Autumn Leaves → [SectionBlock("A", 8, 2), SectionBlock("B", 8, 1), SectionBlock("C", 8, 1)]

class SectionBlock(NamedTuple):
    label: str      # "A", "B", …  (or "?" for unmarked section)
    n_bars: int     # unique bars in this section (before repeat expansion)
    n_plays: int    # how many times it plays (2 if inside {}, else 1)


def parse_form_compact(chord_string: str) -> list[SectionBlock]:
    """Extract the compact form structure from a raw iReal chord_string.

    Handles both ordering variants of iReal notation:
      Variant A: *X { content }  — section marker comes before repeat bracket
      Variant B: { *X content }  — repeat bracket opens before section marker

    Returns list of SectionBlock(label, n_bars_unique, n_plays).
    """
    cs = chord_string
    # Tokenise into a flat stream: (*X, {, }, [, ], |, content_char)
    # We use a left-to-right state machine.
    tokens = re.findall(r'\*[A-Z]|[{}\[\]|]|[^{}\[\]*|]+', cs)

    blocks: list[SectionBlock] = []
    current_label = "?"
    pending_repeat = False   # True when we've seen { but not yet a *X marker
    in_repeat = False        # True when we're inside a {} block
    bar_count = 0
    has_content = False

    def _flush(label: str, n_bars: int, n_plays: int) -> None:
        if n_bars > 0:
            blocks.append(SectionBlock(label=label, n_bars=n_bars, n_plays=n_plays))

    for tok in tokens:
        if re.fullmatch(r'\*[A-Z]', tok):
            # New section starts.  Flush the current accumulation if it had bars.
            # (bars before *X are an unlabelled intro/preamble — keep them if non-zero)
            if has_content and bar_count > 0:
                _flush(current_label, bar_count, 2 if in_repeat else 1)
                bar_count = 0
                has_content = False
            current_label = tok[1]
            # If we saw a pending { before this *X, the repeat belongs to this section
            if pending_repeat:
                in_repeat = True
                pending_repeat = False

        elif tok == '{':
            # Flush any accumulated bars first (section content before this repeat)
            if has_content and bar_count > 0:
                _flush(current_label, bar_count, 1)
                bar_count = 0
                has_content = False
            # Mark: next *X (if any) or current section gets n_plays=2
            pending_repeat = True

        elif tok == '}':
            # Flush current section with repeat
            if has_content and bar_count > 0:
                n_plays = 2 if (in_repeat or pending_repeat) else 1
                _flush(current_label, bar_count, n_plays)
                bar_count = 0
                has_content = False
            in_repeat = False
            pending_repeat = False

        elif tok == '|':
            bar_count += 1

        elif tok in ('[', ']'):
            pass  # bracket markers — don't affect repeat count

        else:
            # Content token: count as chord content for the bar counter
            clean = re.sub(r'n[12]|T\d\d|[UWQNS\s]', '', tok)
            if clean:
                has_content = True

    # Flush final section
    if has_content and bar_count > 0:
        _flush(current_label, bar_count, 2 if in_repeat else 1)

    # Bar-count heuristic: '|' are separators → N separators ≈ N+1 bars, but iReal
    # bars without a trailing | only get counted when there's content.
    # Add 1 to each block that doesn't end with '|' — handled implicitly by `has_content`.
    # Adjust: each block's n_bars counts the separators; add 1 for the last bar.
    corrected = []
    for b in blocks:
        # pyRealParser expands { A×8 } to 16 bars. Our token count yields (n_seps+1).
        # But the final bar of a section often has no trailing '|', so we rely on
        # has_content to ensure at least 1 bar. Add 1 bar for the trailing content.
        corrected.append(SectionBlock(b.label, b.n_bars + 1, b.n_plays))
    return corrected


def one_chorus_bars(form: list[SectionBlock]) -> int:
    """Total bars in one complete play-through (accounting for repeats)."""
    return sum(b.n_bars * b.n_plays for b in form)


# ── Bloc B — Harmonic SSM + chorus boundary detection ─────────────────────────
# Build a self-similarity matrix from the inferred chord sequence (root PCs only,
# no quality — quality inference is noisy, root is more reliable for structure).
# Detect diagonal stripes at the expected chorus lag to find repeat boundaries.

def _harmonic_windows(
    inferred: list,  # list[InferredChord]
    window_secs: float,
) -> np.ndarray:
    """Project inferred chords into fixed-duration windows (one row = one window).

    Each row is a 12-dim L1-normalized root-PC histogram.
    Silent / N.C. chords (pc < 0) are ignored.
    """
    if not inferred or window_secs <= 0:
        return np.zeros((0, 12))
    t_start = inferred[0].t0
    total_dur = inferred[-1].t1 - t_start
    n_win = max(1, int(total_dur / window_secs) + 1)
    acc = np.zeros((n_win, 12), dtype=np.float32)
    for ch in inferred:
        if ch.pc < 0:
            continue
        ch_t0 = ch.t0 - t_start
        ch_t1 = ch.t1 - t_start
        w0 = int(ch_t0 / window_secs)
        w1 = int(ch_t1 / window_secs)
        for w in range(max(0, w0), min(n_win, w1 + 1)):
            wt0 = w * window_secs
            wt1 = (w + 1) * window_secs
            overlap = max(0.0, min(ch_t1, wt1) - max(ch_t0, wt0))
            if overlap > 0:
                acc[w, ch.pc % 12] += overlap
    row_sums = acc.sum(1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return acc / row_sums


def _harmonic_ssm(windows: np.ndarray) -> np.ndarray:
    """N×N cosine similarity matrix from harmonic windows.  Values in [0, 1]."""
    if len(windows) == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(windows, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    W = windows / norms
    return np.clip(W @ W.T, 0.0, 1.0)


def _find_chorus_boundaries(
    inferred: list,          # list[InferredChord]
    one_chorus_secs: float,
    bpm: float,
    beats_per_bar: int,
) -> list[int]:
    """Detect chorus start indices in `inferred` using harmonic SSM.

    Returns a sorted list of indices into `inferred` where a new chorus begins.
    Always includes 0.  Uses SSM diagonal similarity at the expected chorus lag
    to confirm/adjust the period; falls back to pure duration-based division if
    the SSM signal is weak.
    """
    if not inferred or one_chorus_secs <= 0:
        return [0]

    total_dur = inferred[-1].t1 - inferred[0].t0
    if total_dur < one_chorus_secs * 0.8:
        return [0]   # less than one chorus in the audio

    # Window = one bar (coarse but robust to tempo rubato)
    bar_secs = beats_per_bar * 60.0 / bpm
    window_secs = max(bar_secs, 1.0)

    wins = _harmonic_windows(inferred, window_secs)
    if len(wins) < 4:
        return [0]

    ssm = _harmonic_ssm(wins)
    n = len(wins)
    expected_lag = max(1, round(one_chorus_secs / window_secs))

    # Score each lag ±3 around the expected lag; pick the one with the best
    # mean diagonal similarity (= best self-repeat at that period)
    best_lag, best_score = expected_lag, -1.0
    for lag in range(max(1, expected_lag - 3), min(n, expected_lag + 4)):
        if n - lag <= 0:
            continue
        score = float(np.mean([ssm[i, i + lag] for i in range(n - lag)]))
        if score > best_score:
            best_score, best_lag = score, lag

    confirmed_lag_secs = best_lag * window_secs
    n_choruses = max(1, round(total_dur / confirmed_lag_secs))

    t0 = inferred[0].t0
    boundaries = [0]
    for k in range(1, n_choruses):
        target_t = t0 + k * confirmed_lag_secs
        idx = min(range(len(inferred)), key=lambda i: abs(inferred[i].t0 - target_t))
        if idx > boundaries[-1] + 1:
            boundaries.append(idx)

    return boundaries


# ── Bloc C — Rhythmic onset-flux boundary refinement ─────────────────────────
# High-frequency onset flux (> ~3 kHz) peaks at drum fills / cymbal crashes that
# mark section transitions (e.g. A→B).  We use it to snap SSM-estimated boundaries
# to the nearest strong rhythmic event within a ±half-bar search window.

def _refine_boundaries_rhythmic(
    boundaries: list[int],
    inferred: list,           # list[InferredChord]  (for timing)
    onset_flux_times: np.ndarray,   # shape (T,)  — times for each flux value
    onset_flux: np.ndarray,         # shape (T,)  — high-freq onset flux
    bar_secs: float,
) -> list[int]:
    """Snap each boundary to the nearest strong rhythmic event (within ±bar/2).

    If no flux data is available (empty arrays), returns boundaries unchanged.
    """
    if len(onset_flux) == 0 or len(inferred) == 0:
        return boundaries

    refined = [0]
    search_window = bar_secs * 0.5

    for bi in boundaries[1:]:
        # Target time for this boundary
        target_t = inferred[bi].t0
        # Find flux frames within search window
        mask = np.abs(onset_flux_times - target_t) < search_window
        if not mask.any():
            refined.append(bi)
            continue
        # Pick the time with highest flux in window
        best_t = float(onset_flux_times[mask][np.argmax(onset_flux[mask])])
        # Map back to nearest inferred chord index
        idx = min(range(len(inferred)), key=lambda i: abs(inferred[i].t0 - best_t))
        if idx > refined[-1] + 1:
            refined.append(idx)
        else:
            refined.append(bi)

    return refined


# ── Quality-family distance (H1) ──────────────────────────────────────────────
# Stricter than full Jaccard: same root + same family = 0, same root +
# different family = 0.4, different root = 1.0. This is more robust to
# extension differences (Cm7 vs Cm9 → same family → 0 cost).

def _family_dist(pc_a: int, q_a: str, pc_b: int, q_b: str) -> float:
    """Distance in [0, 1]: root mismatch = 1.0, family mismatch = 0.4, match = 0.0."""
    if pc_a < 0 or pc_b < 0:
        return 1.0
    if pc_a != pc_b:
        return 1.0
    if _family(q_a) == _family(q_b):
        return 0.0
    return 0.4


def _dtw_family(
    seq_a: list[tuple[int, str]],
    seq_b: list[tuple[int, str]],
) -> tuple[float, list[tuple[int, int]]]:
    """DTW with family-level distance metric."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf"), []
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = _family_dist(seq_a[i-1][0], seq_a[i-1][1],
                             seq_b[j-1][0], seq_b[j-1][1])
            dp[i][j] = d + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        diag, left, up = dp[i-1][j-1], dp[i][j-1], dp[i-1][j]
        best = min(diag, left, up)
        if best == diag:   i -= 1; j -= 1
        elif best == left: j -= 1
        else:              i -= 1
    path.reverse()
    return dp[n][m] / max(n, m), path


# ── Inferred chord deduplication (H3) ─────────────────────────────────────────

@dataclass
class InferredChord:
    """One deduplicated segment from the inferred chart."""
    pc: int
    quality: str
    label: str
    t0: float           # start time (seconds)
    t1: float           # end time (seconds)
    orig_indices: list[int] = field(default_factory=list)  # indices in P.chords


def dedup_inferred(p_chords: list[dict]) -> list[InferredChord]:
    """Collapse consecutive identical-root chords in the inferred sequence.

    We merge on root pitch class only (not quality) because our model's
    quality classification is noisy, and a run of "C" / "Cm7" / "C7"
    over a C section should count as one segment.
    """
    merged: list[InferredChord] = []
    for i, c in enumerate(p_chords):
        pc = c.get("root", -1)
        q  = c.get("lv", {}).get("seventh", {}).get("q", "")
        if q.startswith(":"):          # pipeline uses ':min7' etc.; tab_aligner uses 'min7'
            q = q[1:]
        t0 = float(c.get("t0") or 0)
        t1 = float(c.get("t1") or t0 + 1)
        label = c.get("label", "")
        if not label:
            sharp = ["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"]
            label = (sharp[pc] if 0 <= pc < 12 else "N") + q
        if merged and merged[-1].pc == pc:
            merged[-1].t1 = t1
            merged[-1].orig_indices.append(i)
        else:
            merged.append(InferredChord(pc=pc, quality=q, label=label,
                                        t0=t0, t1=t1, orig_indices=[i]))
    return merged


# ── iReal sequence extraction ─────────────────────────────────────────────────

@dataclass
class IRealChord:
    """One chord event from the iReal Pro GT chart."""
    label: str          # raw iReal token (e.g. "C-7", "F7", "Bb^7")
    pc: int             # root pitch class
    quality: str
    bar: int            # 1-indexed bar number
    section: str        # "A", "B", etc.
    beat_in_bar: int    # 0-indexed beat within bar
    dur_beats: float    # duration in beats


def extract_ireal_seq(mma_chart) -> list[IRealChord]:
    """Flatten an MMAChart timeline into a list of IRealChord."""
    chords: list[IRealChord] = []
    bpb = mma_chart.beats_per_bar
    for bar_no, section, slots in mma_chart.timeline:
        for k, (beat_offset, ireal_token, _) in enumerate(slots):
            next_beat = slots[k + 1][0] if k + 1 < len(slots) else bpb
            dur = max(next_beat - beat_offset, 1)
            import re
            label = re.sub(r"[npWNQUSr]+$", "", ireal_token).strip() or "N.C."
            pc, quality = _parse_ireal(label)
            chords.append(IRealChord(
                label=label, pc=pc, quality=quality,
                bar=bar_no, section=section,
                beat_in_bar=beat_offset, dur_beats=float(dur),
            ))
    return chords


# ── Key search (reuses H1 from tab_aligner) ───────────────────────────────────

def _best_transpose_family(
    inferred: list[InferredChord],
    ireal: list[IRealChord],
) -> tuple[int, float]:
    """Try all 12 transpositions of the iReal sequence; return (offset, cost)."""
    inf_pairs = [(c.pc, c.quality) for c in inferred]
    irl_pairs = [(c.pc, c.quality) for c in ireal]
    best_offset, best_cost = 0, float("inf")
    for offset in range(12):
        transposed = [((pc + offset) % 12 if pc >= 0 else -1, q)
                      for pc, q in irl_pairs]
        cost, _ = _dtw_family(inf_pairs, transposed)
        if cost < best_cost:
            best_cost, best_offset = cost, offset
    return best_offset, best_cost


# ── Repeat tiling (H2) ────────────────────────────────────────────────────────

def _tile_ireal(ireal: list[IRealChord], n: int) -> list[IRealChord]:
    """Repeat the iReal sequence n times (for multi-chorus videos)."""
    if n <= 1:
        return ireal
    total_bars = (ireal[-1].bar if ireal else 0)
    result = []
    for rep in range(n):
        for ch in ireal:
            import copy
            c = copy.copy(ch)
            c.bar = c.bar + rep * total_bars
            result.append(c)
    return result


def _estimate_repeats(
    inferred: list[InferredChord],
    ireal: list[IRealChord],
    bpm: float,
    beats_per_bar: int,
) -> int:
    """Estimate how many times the form repeats in the video."""
    if not inferred or not ireal:
        return 1
    inferred_dur = inferred[-1].t1 - inferred[0].t0
    total_ireal_beats = sum(c.dur_beats for c in ireal)
    ireal_dur = total_ireal_beats * (60.0 / bpm)
    if ireal_dur <= 0:
        return 1
    n = round(inferred_dur / ireal_dur)
    return max(1, min(n, 6))


# ── Timestamp assignment (H4) ─────────────────────────────────────────────────

def _assign_timestamps(
    ireal: list[IRealChord],
    inferred: list[InferredChord],
    dtw_path: list[tuple[int, int]],
    transpose: int,
    bpm: float,
    beats_per_bar: int = 4,
) -> list[dict]:
    """Assign t0/t1 to each iReal chord from the DTW alignment.

    When multiple iReal chords map to the same inferred segment, we interpolate
    timestamps within that segment using BPM (H4). When one iReal chord spans
    multiple inferred segments, it gets the union of their time range.
    """
    spb = 60.0 / bpm  # seconds per beat

    # Build ireal_idx → [inferred_idx, ...]
    irl_to_inf: dict[int, list[int]] = {i: [] for i in range(len(ireal))}
    for inf_idx, irl_idx in dtw_path:
        irl_to_inf[irl_idx].append(inf_idx)

    # Build inferred_idx → [ireal_idx, ...] for interpolation
    inf_to_irl: dict[int, list[int]] = {}
    for irl_idx, inf_list in irl_to_inf.items():
        for inf_idx in inf_list:
            inf_to_irl.setdefault(inf_idx, []).append(irl_idx)

    result: list[dict] = []
    sharp = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    for irl_idx, ch in enumerate(ireal):
        inf_indices = irl_to_inf[irl_idx]

        # Transpose label for display
        transposed_pc = (ch.pc + transpose) % 12 if ch.pc >= 0 else -1
        if transposed_pc >= 0:
            transposed_label = sharp[transposed_pc] + ch.quality
        else:
            transposed_label = ch.label

        if not inf_indices:
            # Gap: no inferred chord aligned here
            result.append({
                "label":   transposed_label,
                "t0":      None,
                "t1":      None,
                "bar":     ch.bar - 1,
                "section": ch.section,
                "match":   "gap",
            })
            continue

        # Multiple inferred segments → use union of time range
        t0 = min(inferred[i].t0 for i in inf_indices)
        t1 = max(inferred[i].t1 for i in inf_indices)

        # If multiple iReal chords share the same inferred segment, interpolate
        # within that segment using BPM (H4).
        # We use absolute beat position rather than the DTW co-mapping set, which
        # can include off-root chords routed through the same inferred slot.
        inf_idx = inf_indices[0]  # primary segment
        co_irl = sorted(inf_to_irl.get(inf_idx, [irl_idx]))
        # Filter co_irl to only chords with the same root as the inferred segment
        # (avoids polluting the interpolation with DTW-warped mismatches)
        inf_pc_here = inferred[inf_idx].pc
        co_irl_filt = [i for i in co_irl
                       if ireal[i].pc < 0 or ireal[i].pc == inf_pc_here]
        if not co_irl_filt:
            co_irl_filt = co_irl   # fallback: use all
        if len(co_irl_filt) > 1 and irl_idx in co_irl_filt:
            seg = inferred[inf_idx]
            seg_dur = seg.t1 - seg.t0
            # Clip co_irl_filt to chords whose cumulative beats fit the segment.
            # This prevents DTW warp-holes (many iReal bars into one inferred slot)
            # from inflating total_beats and collapsing individual durations.
            # Snap to bar boundary: round the raw estimate to the nearest bpb multiple
            raw_beats = seg_dur * bpm / 60.0
            expected_beats = max(beats_per_bar, round(raw_beats / beats_per_bar) * beats_per_bar)
            trimmed: list[int] = []
            acc = 0.0
            for i in co_irl_filt:
                trimmed.append(i)
                acc += ireal[i].dur_beats
                if acc >= expected_beats:   # stop once we've filled the segment
                    break
            # Keep irl_idx in the set even if trimmed would have excluded it
            if irl_idx not in trimmed:
                trimmed = co_irl_filt   # fallback to full filtered set
            co_irl_filt = trimmed
            total_beats = sum(ireal[i].dur_beats for i in co_irl_filt)
            beat_before = sum(ireal[i].dur_beats for i in co_irl_filt
                              if i < irl_idx)
            frac_start = beat_before / total_beats if total_beats > 0 else 0.0
            frac_end   = (beat_before + ch.dur_beats) / total_beats if total_beats > 0 else 1.0
            t0 = seg.t0 + frac_start * seg_dur
            t1 = seg.t0 + frac_end   * seg_dur

        # Compute match quality
        if inf_indices:
            inf_pc = inferred[inf_indices[0]].pc
            inf_q  = inferred[inf_indices[0]].quality
            dist = _family_dist(inf_pc, inf_q, transposed_pc, ch.quality)
            if dist == 0.0:    match = "exact"
            elif dist < 1.0:   match = "family"
            else:              match = "mismatch"
        else:
            match = "gap"

        result.append({
            "label":   transposed_label,
            "t0":      round(t0, 3),
            "t1":      round(t1, 3),
            "bar":     ch.bar - 1,
            "section": ch.section,
            "match":   match,
        })

    return result


# ── Main entry point ──────────────────────────────────────────────────────────

@dataclass
class IRealbAlignmentResult:
    transpose_semitones: int
    dtw_cost: float
    n_repeats: int
    chords: list[dict]          # [{label, t0, t1, bar, section, match}]
    exact_frac: float           # fraction of iReal chords with exact root+family match
    family_frac: float
    mismatch_frac: float


def align_irealb_to_inferred(
    mma_chart,
    p_chords: list[dict],
    *,
    bpm_override: float | None = None,
    onset_flux_times: np.ndarray | None = None,
    onset_flux: np.ndarray | None = None,
) -> IRealbAlignmentResult:
    """Align an iReal MMAChart to the inferred P.chords from the pipeline.

    Args:
        mma_chart:  MMAChart from tune_to_mma() — the iReal Pro GT.
                    Must have .chord_string set (populated by tune_to_mma).
        p_chords:   P.chords list from the inferred chart page (with t0, t1,
                    root pitch class, and lv quality dict)
        bpm_override: override the MMAChart's BPM
        onset_flux_times: optional array of times for Bloc C rhythmic refinement
        onset_flux:       optional high-freq onset flux array (same length)

    Returns:
        IRealbAlignmentResult with per-chord {label, t0, t1, section, match}
    """
    bpm = float(bpm_override or mma_chart.tempo)
    bpb = mma_chart.beats_per_bar
    bar_secs = bpb * 60.0 / bpm

    # Step 1: build inferred sequence (deduplicated on root)
    inferred = dedup_inferred(p_chords)

    # Step 2: extract iReal sequence (expanded by pyRealParser)
    ireal_base = extract_ireal_seq(mma_chart)

    if not inferred or not ireal_base:
        return IRealbAlignmentResult(0, float("inf"), 1, [], 0.0, 0.0, 0.0)

    # Step 3: Bloc A — one-chorus duration from the expanded timeline.
    # mma_chart.timeline is already fully expanded by pyRealParser (including {} repeats),
    # so it represents exactly one full play-through of the form.
    # parse_form_compact is preserved for future per-section matching (Bloc D).
    chorus_bars = len(mma_chart.timeline)
    one_chorus_secs = chorus_bars * bpb * 60.0 / bpm

    # Step 4: find best key offset (on base, untiled iReal sequence)
    transpose, _ = _best_transpose_family(inferred, ireal_base)

    # Step 5: Bloc B — SSM-based chorus boundary detection.
    # Find where each chorus starts in the audio, guided by the form structure.
    boundaries = _find_chorus_boundaries(inferred, one_chorus_secs, bpm, bpb)

    # Step 6: Bloc C — refine boundaries with high-freq onset flux (if available).
    if onset_flux_times is not None and onset_flux is not None and len(onset_flux) > 0:
        boundaries = _refine_boundaries_rhythmic(
            boundaries, inferred,
            np.asarray(onset_flux_times), np.asarray(onset_flux),
            bar_secs,
        )

    n_choruses_detected = len(boundaries)

    # Step 7: per-chorus DTW.
    # Split the inferred sequence at detected boundaries and align each chorus
    # independently against the full iReal form.  This prevents DTW from warping
    # across chorus boundaries (the main failure mode with global DTW).
    inf_pairs_full = [(c.pc, c.quality) for c in inferred]
    irl_pairs_t    = [((c.pc + transpose) % 12 if c.pc >= 0 else -1, c.quality)
                      for c in ireal_base]

    all_chords: list[dict] = []
    overall_cost = 0.0

    for ci, b_start in enumerate(boundaries):
        b_end = boundaries[ci + 1] if ci + 1 < len(boundaries) else len(inferred)
        chorus_inferred = inferred[b_start:b_end]
        if not chorus_inferred:
            continue

        chorus_inf_pairs = inf_pairs_full[b_start:b_end]

        # DTW this chorus against iReal form; try ±1 tile to handle partial choruses
        best_cost_c = float("inf")
        best_path_c: list[tuple[int, int]] = []
        best_ireal_c = ireal_base

        for n in range(1, 3):
            tiled = _tile_ireal(ireal_base, n)
            irl_t = [((c.pc + transpose) % 12 if c.pc >= 0 else -1, c.quality)
                     for c in tiled]
            cost_c, path_c = _dtw_family(chorus_inf_pairs, irl_t)
            if cost_c < best_cost_c:
                best_cost_c = cost_c
                best_path_c = path_c
                best_ireal_c = tiled

        overall_cost += best_cost_c
        chords_c = _assign_timestamps(
            best_ireal_c, chorus_inferred, best_path_c,
            transpose, bpm, beats_per_bar=bpb,
        )
        all_chords.extend(chords_c)

    # Fallback: if per-chorus gave nothing, run global DTW (old behaviour)
    if not all_chords:
        n_reps = _estimate_repeats(inferred, ireal_base, bpm, bpb)
        best_cost_g = float("inf")
        best_path_g: list[tuple[int, int]] = []
        best_n_g = n_reps
        for n in range(max(1, n_reps - 1), min(7, n_reps + 2)):
            tiled = _tile_ireal(ireal_base, n)
            irl_t = [((c.pc + transpose) % 12 if c.pc >= 0 else -1, c.quality)
                     for c in tiled]
            cost_g, path_g = _dtw_family(inf_pairs_full, irl_t)
            if cost_g < best_cost_g:
                best_cost_g = cost_g
                best_path_g = path_g
                best_n_g = n
        ireal_final = _tile_ireal(ireal_base, best_n_g)
        all_chords = _assign_timestamps(ireal_final, inferred, best_path_g, transpose, bpm,
                                        beats_per_bar=bpb)
        overall_cost = best_cost_g
        n_choruses_detected = best_n_g

    # Compute match stats
    exact_n  = sum(1 for c in all_chords if c["match"] == "exact")
    family_n = sum(1 for c in all_chords if c["match"] == "family")
    miss_n   = sum(1 for c in all_chords if c["match"] == "mismatch")
    total_ng = max(exact_n + family_n + miss_n, 1)
    avg_cost = round(overall_cost / max(n_choruses_detected, 1), 4)

    return IRealbAlignmentResult(
        transpose_semitones=transpose,
        dtw_cost=avg_cost,
        n_repeats=n_choruses_detected,
        chords=all_chords,
        exact_frac=round(exact_n  / total_ng, 3),
        family_frac=round(family_n / total_ng, 3),
        mismatch_frac=round(miss_n  / total_ng, 3),
    )
