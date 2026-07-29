"""SSM diagonal-block section detector — fallback for the fixed-lag
``chart_model._sections_by_largest_unit`` (2026-07-29).

Motivation (docs/known_issues.md 2026-07-29 "This Love section detection"): the
production largest-unit detector tests chord recurrence at a FIXED 8/16-bar lag
and returns ``None`` when that recurrence is weak — which then falls back to the
crude changepoint segmentation (on This Love: one cut -> "B then A"). But This
Love HAS clear sections; they just don't line up on a rigid 8/16-bar grid.

This detector reads the DIAGONAL BLOCKS of the bar self-similarity matrix the
way the eye does (user idea, 2026-07-29):
  1. per-bar (root_rel_to_tonic, coarse-quality) -> bar SSM (build_chord_ssm)
  2. GAUSSIAN-BLUR the SSM over ~one loop: a section is tiled by small repeating
     squares (a 2-bar loop over an 8-bar chorus); blurring fuses the small
     checkerboard into one solid diagonal block.
  3. Foote checkerboard-kernel novelty along the diagonal -> peak-pick boundaries
  4. label: cross-block rectangle similarity RELATIVE to each block's own
     self-similarity, greedy first-appearance clustering -> A/B/C.

Boundary-FIRST (cut, then cluster), so it survives songs whose sections do not
repeat at a fixed lag — the exact case the fixed-lag detector gives up on.

Scope / NOT solved (be honest, CLAUDE.md #4/#5):
  * Validated by eye on ~5 songs (This Love -> A B A B A C B B, correctly
    isolating the Fm7-Ebmaj7 bridge; Falling -> A B A B A recovering a real
    Am-E7 vs F-A^7 contrast). NOT corpus-scored yet — that is the follow-up.
  * FALLBACK ONLY: only runs when ``_sections_by_largest_unit`` returned None,
    so its blast radius is bounded to songs that are ALREADY producing the crude
    changepoint fallback. Returns ``None`` (defer to changepoint) on anything it
    can't segment into >=2 blocks. Kill-switch ``HARMONIA_SSM_BLOCK=0``.
  * Root-relative representation: sections live in ROOT motion, not chord colour
    (a note-content/chord-tone SSM over-merges diatonic material — measured).
  * Letters are content clusters, not ear-validated verse/chorus/bridge names.
"""
from __future__ import annotations

import os

import numpy as np

from harmonia.models.section_structure import build_chord_ssm

__all__ = ["ssm_block_sections"]

# iReal quality token -> coarse family index (root carries most of the signal;
# a light quality split keeps Cmaj vs Cm partially distinct in the SSM).
_QMAP = {
    "": 0, "6": 0, "^7": 0, "maj7": 0, "add9": 0,
    "-": 1, "-7": 1, "-6": 1, "-^7": 1, "m": 1, "min7": 1,
    "7": 2, "9": 2, "13": 2, "7b9": 2,
    "h7": 3, "-7b5": 3,
    "o": 4, "o7": 4,
}


def _bar_root_qual(bars: list[list[dict]], n_bars: int, tonic_pc: int) -> list[tuple[int, int]]:
    """Per-bar ``(root_rel_to_tonic, qual_idx)``; a held/empty bar inherits the
    previous bar's chord (a held chord still sounds). ``(-1, -1)`` = no chord."""
    seq: list[tuple[int, int]] = []
    prev: tuple[int, int] = (-1, -1)
    for b in range(n_bars):
        bar = bars[b] if b < len(bars) else []
        ch = next((e for e in bar if e.get("q") != "N" and int(e.get("root", -1)) >= 0), None)
        if ch is None:
            seq.append(prev)
        else:
            r = (int(ch["root"]) - tonic_pc) % 12
            seq.append((r, _QMAP.get(ch.get("q", ""), 0)))
            prev = seq[-1]
    return seq


def _checkerboard_kernel(m: int) -> np.ndarray:
    taper = np.exp(-0.5 * (np.linspace(-2, 2, 2 * m)) ** 2)
    g = np.outer(taper, taper)
    sign = np.ones((2 * m, 2 * m))
    sign[:m, m:] = -1
    sign[m:, :m] = -1
    return sign * g


def _novelty(S: np.ndarray, m: int) -> np.ndarray:
    n = S.shape[0]
    K = _checkerboard_kernel(m)
    Sp = np.pad(S, m, mode="edge")
    nov = np.array([float(np.sum(Sp[i:i + 2 * m, i:i + 2 * m] * K)) for i in range(n)])
    nov = np.clip(nov, 0, None)
    return nov / (nov.max() + 1e-9)


def _pick_boundaries(nov: np.ndarray, min_gap: int, rel_thresh: float) -> list[int]:
    n = len(nov)
    cand = [i for i in range(1, n - 1)
            if nov[i] >= rel_thresh and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    cand.sort(key=lambda i: -nov[i])
    kept: list[int] = []
    for i in cand:
        if all(abs(i - k) >= min_gap for k in kept):
            kept.append(i)
    return sorted(kept)


def _snap_to_grid(bounds: list[int], nov: np.ndarray, phrase: int, tol: int) -> list[int]:
    """Refine coarse boundaries onto the PHRASE grid (user 2026-07-29): a real
    section edge lands on a phrase downbeat (a multiple of ``phrase`` bars from
    bar 0), not a random bar. Snap each boundary to its nearest multiple within
    ``tol`` bars, but VERIFY first — only move it if the checkerboard novelty at
    the snapped position still carries real boundary evidence (>= half the
    original peak), so a snap never invents an edge where the SSM doesn't back it.
    NB: fixes the phrase-scale grid only; a sub-bar pickup/anacrusis (This Love's
    "G|Cm" levée) is a separate beat-1 phase problem, not solved here."""
    if phrase <= 1:
        return bounds
    out: list[int] = []
    for b in bounds:
        cand = int(round(b / phrase) * phrase)
        if cand != b and 0 < cand < len(nov) and abs(cand - b) <= tol and nov[cand] >= 0.5 * nov[b]:
            out.append(cand)
        else:
            out.append(b)
    return sorted(set(out))


def _align_repeats(edges: list[int], labels: list[str], bars: list[list[dict]],
                   tol: int) -> list[int]:
    """Make every occurrence of a letter OPEN ON THE SAME CHORD (user 2026-07-29:
    "the A bars always open on G"). Independent boundary detection can put two
    occurrences of the same section a bar out of phase (one opening on the V
    pickup, another on the i). For each letter, take the FIRST occurrence's
    opening chord as the reference and nudge every other occurrence (<= ``tol``
    bars) onto the nearest bar whose displayed downbeat chord has that same root.
    Uses the DISPLAY bars (``bars[b][0]``) — the exact chord the app shows — so
    what we align to is what you see, not a forward-filled reconstruction."""
    n = edges[-1]
    starts = edges[:-1]

    def oroot(b: int) -> int:
        if 0 <= b < len(bars) and bars[b] and bars[b][0].get("q") != "N":
            return int(bars[b][0].get("root", -1)) % 12
        return -1

    from collections import defaultdict
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[lab].append(i)

    new_starts = list(starts)
    for lab, idxs in by_label.items():
        if len(idxs) < 2:
            continue
        target = oroot(starts[idxs[0]])
        if target < 0:
            continue
        for i in idxs[1:]:
            s = starts[i]
            cands = [s + d for d in range(-tol, tol + 1)
                     if 0 < s + d < n and oroot(s + d) == target]
            if cands:
                new_starts[i] = min(cands, key=lambda x: abs(x - s))
    return sorted({0, *[s for s in new_starts if s > 0], n})


def _split_intro_outro(labels: list[str]) -> list[str]:
    """Split a one-off leading/trailing section off as Intro / Outro (user
    2026-07-29: "always split the intro and the outro off"). Conservative: only
    relabels the first/last section when its letter appears exactly once in the
    form, so a genuine repeated section is never mistaken for an intro."""
    if len(labels) < 3:
        return labels
    from collections import Counter
    cnt = Counter(labels)
    out = list(labels)
    if cnt[out[0]] == 1:
        out[0] = "Intro"
    if cnt[out[-1]] == 1 and out[-1] != "Intro":
        out[-1] = "Outro"
    return out


def _merge_runts(edges: list[int], min_bars: int) -> list[int]:
    segs = [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]
    i = 0
    while len(segs) > 1 and i < len(segs):
        s, e = segs[i]
        if e - s < min_bars:
            if i == len(segs) - 1:
                segs[i - 1][1] = e; segs.pop(i)
            else:
                segs[i + 1][0] = s; segs.pop(i)
        else:
            i += 1
    return [segs[0][0]] + [s[1] for s in segs]


def _block_sim(S: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    r = S[a[0]:a[1], b[0]:b[1]]
    return float(r.mean()) if r.size else 0.0


def _label(S: np.ndarray, edges: list[int], sim_thresh: float) -> list[str]:
    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    self_sim = [_block_sim(S, s, s) + 1e-9 for s in segs]
    letters = "ABCDEFGHIJ"
    labs: list[str] = []
    reps: list[int] = []
    for i, seg in enumerate(segs):
        hit = None
        for k, ri in enumerate(reps):
            denom = (self_sim[i] * self_sim[ri]) ** 0.5
            if _block_sim(S, seg, segs[ri]) / denom >= sim_thresh:
                hit = k
                break
        if hit is None:
            labs.append(letters[len(reps) % len(letters)])
            reps.append(i)
        else:
            labs.append(letters[hit])
    return labs


def ssm_block_sections(
    bars: list[list[dict]],
    n_bars: int,
    tonic_pc: int = 0,
    *,
    sigma: float = 2.0,
    kernel_bars: int = 6,
    min_gap: int = 4,
    min_section_bars: int = 4,
    rel_thresh: float = 0.15,
    sim_thresh: float = 0.80,
    snap_phrase: int = 4,
    snap_tol: int = 2,
) -> "list[dict] | None":
    """Diagonal-block sections from ``bars`` -> section dicts (same shape as
    ``_sections_by_largest_unit``: ``id/label/tag/reps/bars/spans/barRanges``,
    ``barRanges`` end INCLUSIVE), or ``None`` to defer to the changepoint path.

    ``bars`` is the per-display-bar list of chord-entry dicts (each ``{"root",
    "q", ...}``); ``tonic_pc`` keys the root-relative representation.
    """
    if os.environ.get("HARMONIA_SSM_BLOCK", "1") == "0":
        return None
    if n_bars < 3 * min_section_bars:      # too short for a meaningful block form
        return None
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return None

    seq = _bar_root_qual(bars, n_bars, tonic_pc)
    S = build_chord_ssm([(r, q) for r, q in seq])
    if S.shape[0] != n_bars:
        return None
    Sb = gaussian_filter(S.astype(np.float64), sigma=sigma, mode="nearest")
    nov = _novelty(Sb, kernel_bars)
    bounds = _pick_boundaries(nov, min_gap, rel_thresh)
    bounds = _snap_to_grid(bounds, nov, snap_phrase, snap_tol)   # refine onto phrase grid
    edges = _merge_runts([0] + bounds + [n_bars], min_bars=min_section_bars)
    if len(edges) < 3:                      # <2 sections -> nothing gained
        return None
    labels = _label(S, edges, sim_thresh)
    # Phase-lock repeated sections to a consistent opening chord, then split a
    # one-off leading/trailing block off as Intro/Outro.
    aligned = _align_repeats(edges, labels, bars, snap_tol)
    if len(aligned) == len(edges):          # shift kept the section count
        edges = aligned
    labels = _split_intro_outro(labels)

    out: list[dict] = []
    for i, lab in enumerate(labels):
        b0, b1 = edges[i], edges[i + 1]      # b1 exclusive
        sec_bars = bars[b0:b1]
        if not sec_bars:
            continue
        out.append({
            "id": lab, "label": lab, "tag": "", "reps": 1,
            "bars": sec_bars,
            "barRanges": [[b0, b1 - 1]],     # inclusive end, matches raw path
        })
    return out if len(out) >= 2 else None
