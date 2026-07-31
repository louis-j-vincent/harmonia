"""harmonia_min/sections.py — section boundaries + A/B/C labels from the RAW
NNLS chroma at HALF-BAR granularity. Label strip only: NO folding (milestone
scoped with Louis 2026-07-31 — detection first, repli later).

v2 (Louis's corrections, 2026-07-31, after the This Love diagnostic):
  * v1 built the SSM from DECODED CHORD TONES per BAR. That erased the
    structure: every This Love bar collapsed to the same C-minor-family
    vector, the SSM turned into a uniform fine checkerboard, and the novelty
    fired on turnarounds. "Le SSM a une structure, mais tu la lis mal."
  * v2 reads the structure the way it is actually written: the substrate is
    the raw NNLS bothchroma pooled per HALF-BAR (texture: voicings, bass
    movement, harmonic rhythm — not just which chord family), and the
    blurred checkerboard runs on THAT. The blur at half-bar resolution pools
    the fast alternation into section-scale blocks.
  * Cuts are detected at half-bar resolution, then snapped to bar lines
    (ChartModel sections are bar-ranged).

Letters: average-linkage agglomeration of segment-mean chroma features.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# constants — few, documented, not a config surface. Grain is HALF-BARS.
KERNEL_HB = 16         # checkerboard half-width (half-bars) = 8 bars context
BLUR_SIGMA = 1.5       # gaussian blur (half-bars) applied to the SSM
PEAK_FRAC = 0.5        # keep peaks >= this fraction of the strongest one
MIN_SEG_BARS = 2       # refuse degenerate slivers (NOT an 8-bar prior)
LABEL_COS = 0.96       # cross/self block ratio above this = same letter
                       # (This Love measured: same-type pairs 0.98-1.00,
                       #  verse-vs-chorus 0.91-0.94 — the gap is real)


def halfbar_features(grid: list[float], arr, times) -> np.ndarray:
    """(2*n_bars, 24) mean raw NNLS bothchroma per half-bar, L2-normalised
    per half (bass and treble each unit-norm, so neither half dominates)."""
    edges = []
    for b in range(len(grid) - 1):
        mid = 0.5 * (grid[b] + grid[b + 1])
        edges += [(grid[b], mid), (mid, grid[b + 1])]
    F = np.zeros((len(edges), 24))
    for i, (t0, t1) in enumerate(edges):
        sel = (times >= t0) & (times < t1)
        if not sel.any():
            j = int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))
            v = arr[j]
        else:
            v = arr[sel].mean(0)
        for h in (slice(0, 12), slice(12, 24)):
            n = np.linalg.norm(v[h])
            F[i, h] = v[h] / n if n > 1e-9 else 0.0
    return F / np.sqrt(2.0)          # whole row ~unit norm when both halves live


def _blur(S: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, S)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)


def _novelty(S: np.ndarray, kw: int) -> np.ndarray:
    """Checkerboard-kernel novelty along the diagonal."""
    n = len(S)
    kernel = np.zeros((2 * kw, 2 * kw))
    kernel[:kw, :kw] = kernel[kw:, kw:] = 1.0
    kernel[:kw, kw:] = kernel[kw:, :kw] = -1.0
    g = np.exp(-0.5 * ((np.arange(2 * kw) - (kw - 0.5)) / (0.6 * kw)) ** 2)
    kernel *= np.outer(g, g)
    nov = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - kw), min(n, i + kw)
        sub = S[a:b, a:b]
        ka, kb = kw - (i - a), kw + (b - i)
        nov[i] = float((sub * kernel[ka:kb, ka:kb]).sum())
    return nov


def detect_sections(grid: list[float], arr, times, bars=None) -> list[dict]:
    """[{b0, b1, label}] over BAR indices — contiguous, covering, unfolded.

    Detection runs at half-bar grain on the raw-chroma SSM; each accepted cut
    is snapped to its nearest bar line.
    """
    n_bars = len(grid) - 1
    if n_bars < 2 * MIN_SEG_BARS:
        return [{"b0": 0, "b1": n_bars - 1, "label": "A"}]
    F = halfbar_features(grid, arr, times)
    S = F @ F.T
    n = len(S)                                   # = 2 * n_bars

    nov = _novelty(_blur(S, BLUR_SIGMA), KERNEL_HB)
    # Edge half-bars see a truncated, unbalanced kernel — their values are
    # artifacts and inflated the old mean+z·σ threshold past every real peak
    # (This Love: threshold 20.8 over real section peaks at 11–13, found
    # nothing). Mask them, then keep interior local maxima that reach
    # PEAK_FRAC of the strongest — a per-song-adaptive threshold — with a
    # weak absolute floor so a structureless song doesn't get noise cuts.
    kw = KERNEL_HB
    interior = nov[kw:n - kw]
    if len(interior) == 0:
        return [{"b0": 0, "b1": n_bars - 1, "label": "A"}]
    floor = interior.mean() + 0.5 * interior.std()
    cand = [i for i in range(kw, n - kw)
            if nov[i] == max(nov[max(0, i - 3):i + 4])
            and nov[i] >= max(PEAK_FRAC * interior.max(), floor)]

    # ── half-bar peak → BAR cut. Bars are THE reference unit for sections
    # (Louis, 2026-07-31): the SSM runs finer for detection, but a boundary
    # is "which bar belongs to which side", decided on the chart's own bars.
    # Rules, in priority order:
    #   1. NEVER split a recurring 2-bar cell (Louis: the chorus's
    #      "Cm Fm | Bb Eb" sub-section). A cell = a pair of bars with real
    #      onsets whose chord signature recurs >=2 times in the song; a cut
    #      may not land between its two bars. Held/N.C. bars never form cells.
    #   2. A section never OPENS on a held bar ("%") — the hold belongs to
    #      the phrase that is closing (This Love: "Ab G | %" chorus tail).
    #   3. Section lengths round to the nearest MULTIPLE OF 2 bars (strong
    #      preference, applied left-to-right; yields to rules 1-2 when they
    #      conflict — never silently splits a cell to make a length even).
    #   4. Mid-bar peaks (odd h) mean the change happens DURING that bar →
    #      the nominal first full bar is the next one (ceil; round() was
    #      banker's rounding, a literal coin flip on .5).
    def _sig(b: int, include_carries: bool = True):
        """Bar signature = the harmony SOUNDING in it. Since the "%" removal
        (Louis, 2026-07-31 evening) held bars carry a written copy of their
        sounding chord — the signature uses it, so 'held G' == 'attacked G'
        for overlap/failsafe purposes. N.C. bars sign as ("%",)."""
        if bars is None or not bars[b]:
            return ("%",)
        out = []
        for c in bars[b]:
            if c.get("carry") and not include_carries:
                continue
            if c.get("nc"):
                return ("%",)
            q = c["q"]
            fam = ("m" if q.startswith("-") else
                   "d" if q[:1] in ("h", "o") else
                   "s" if "sus" in q else
                   "a" if q.startswith("+") else "M")
            out.append((c["root"], fam))
        return tuple(out) or ("%",)

    def _is_held(b: int) -> bool:
        """A bar with no real onset: empty, or carry-only (the written copy
        of the previous chord that replaced the old "%")."""
        return (bars is not None and 0 <= b < n_bars
                and (not bars[b] or all(c.get("carry") for c in bars[b])))

    from collections import defaultdict
    sig2 = {b: (_sig(b), _sig(b + 1)) for b in range(n_bars - 1)}
    pos2 = defaultdict(list)
    for b, pair in sig2.items():
        if _is_held(b) or _is_held(b + 1):
            continue                              # holds never anchor a cell
        pos2[pair].append(b)
    # A CELL is a recurring pair that TILES locally (occurrences <= 4 bars
    # apart — the chorus's "Cm Fm | Bb Eb" recurs every 2 bars). A recurring
    # TRANSITION (verse-end → chorus-start also recurs, but 20 bars apart,
    # once per section pass) is NOT a cell — treating it as one forbade the
    # legitimate section cut and swallowed whole choruses (measured).
    _cells = {pair for pair, ps in pos2.items()
              if len(ps) >= 2 and ("%",) not in pair
              and min(b2 - b1 for b1, b2 in zip(ps, ps[1:])) <= 4}

    def _splits_cell(c: int) -> bool:
        if bars is None or not (0 < c < n_bars):
            return False
        # a held bar can't BIND a cell at this location: since holds write
        # their sounding chord, "G(held) | Cm" signs like the verse's tiling
        # "G/B | Cm" cell and wrongly forbade the validated cut (measured)
        if _is_held(c - 1) or _is_held(c):
            return False
        return sig2.get(c - 1) in _cells

    def _opens_held(c: int) -> bool:
        return _is_held(c)

    def _tail_penalty(b0: int) -> float:
        """An opening whose SECOND bar is held ("X | %") is cadence-tail
        material, not a section start — two This Love A's aligned with each
        other on exactly that (consistent but wrong) before this penalty."""
        return 0.3 if (b0 + 1 < n_bars and not _is_held(b0)
                       and _is_held(b0 + 1)) else 0.0

    def _eff_even(prev: int, c: int) -> int:
        """Rule 3 parity on the EFFECTIVE length: trailing held bars do not
        count (Louis's validated B section = 8 attack bars + a held cadence
        tail = 9 written bars — still 'even'). Interior holds DO count (they
        sit inside phrases: 'Fm7 | %' is a real 2-bar unit)."""
        t = c - 1
        while t > prev and _is_held(t):
            t -= 1
        return (t + 1 - prev) % 2

    cuts = []
    for h in cand:
        base = (h + 1) // 2                       # rule 4: ceil(h/2)
        prev = cuts[-1] if cuts else 0
        best, best_score = None, None
        for c in range(base - 1, base + 3):       # small bar-level search
            if not (prev + MIN_SEG_BARS <= c <= n_bars - 1):
                continue
            if _splits_cell(c) or _opens_held(c):
                continue
            # Priority: stay on the peak (a sharp boundary — Close's Db
            # modulation lands EXACTLY on its bar — must not move); tail and
            # evenness only arbitrate candidates equally near it. "Arrondis
            # au multiple de 2" rounds AMBIGUOUS lengths, it never drags a
            # confident cut (measured: evenness-first moved Close +2 bars).
            score = (abs(c - base),
                     _tail_penalty(c),
                     _eff_even(prev, c))
            if best_score is None or score < best_score:
                best, best_score = c, score
        if best is not None:
            cuts.append(best)
    bounds = [0] + cuts + [n_bars]
    segs = [{"b0": a, "b1": b - 1} for a, b in zip(bounds, bounds[1:])]

    # ── letters from the OFF-DIAGONAL repetition blocks ──────────────────
    # Segment-mean chroma cosine failed (every This Love segment is C-minor
    # material → one letter for the whole song). What actually says "these
    # two segments are the same section type" is the cross-block of the
    # blurred SSM — the red off-diagonal blocks in the diagnostic plot. Two
    # segments share a letter when their cross-block similarity comes close
    # to their own internal similarity (a correlation-style ratio).
    Sb = _blur(S, BLUR_SIGMA)
    k = len(segs)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ri = slice(2 * segs[i]["b0"], 2 * (segs[i]["b1"] + 1))
            rj = slice(2 * segs[j]["b0"], 2 * (segs[j]["b1"] + 1))
            M[i, j] = float(Sb[ri, rj].mean())
    groups: list[list[int]] = []
    for i in range(k):
        best, best_r = None, LABEL_COS
        for gi, g in enumerate(groups):
            r = float(np.mean([M[i, j] / np.sqrt(max(M[i, i] * M[j, j], 1e-12))
                               for j in g]))
            if r > best_r:
                best, best_r = gi, r
        if best is None:
            groups.append([i])
        else:
            groups[best].append(i)
    for gi, g in enumerate(groups):
        letter = chr(ord("A") + gi) if gi < 26 else f"S{gi}"
        for i in g:
            segs[i]["label"] = letter

    # ── FAILSAFE (Louis, 2026-07-31): same-letter sections must AGREE ────────
    # "Compare les deux A — s'ils ne se recoupent pas, on a mal coupé."
    # For each letter group, shift each member's opening (±2 bars, respecting
    # the cell/held rules and its neighbours) to maximise agreement of the
    # first 4 bar-signatures with its group mates; log what remains disagreed.
    def _open_sig(b0: int, k: int = 4):
        return tuple(_sig(b) for b in range(b0, min(b0 + k, n_bars)))

    def _agree(a, b):
        return sum(x == y for x, y in zip(a, b)) / max(1, min(len(a), len(b)))


    for g in groups:
        if len(g) < 2:
            continue
        for i in g:
            s = segs[i]
            if s["b0"] == 0:
                continue                          # pinned at the song start
            prev_s = segs[segs.index(s) - 1]
            mates = [segs[j] for j in g if j != i]

            def _prefix_match(b0: int) -> int:
                """Strict-equality opening evidence (up to 4 bars) vs the
                best-matching mate. HELD bars never count as evidence — the
                held-G cadence tails of two sections match each other and
                rebuilt the consensus-on-the-cadence trap the moment holds
                started signing as their sounding chord (measured)."""
                best = 0
                for mt in mates:
                    k = 0
                    for i in range(4):
                        x, y = b0 + i, mt["b0"] + i
                        if x >= n_bars or y >= n_bars or _sig(x) != _sig(y) \
                                or _sig(x) == ("%",):
                            break
                        if not _is_held(x) and not _is_held(y):
                            k += 1
                    best = max(best, k)
                return best

            # "se recoupent" taken literally (Louis): shift ONLY to a position
            # whose opening strictly equals a sibling's opening on >= 2 bars,
            # and only if the current position has no such match. Fraction-
            # based scoring shifted sections on noise (measured, twice).
            if _prefix_match(s["b0"]) >= 2:
                continue
            best_d, best_k = 0, 1
            for d in (-3, -2, -1, 1, 2, 3):
                nb = s["b0"] + d
                if not (prev_s["b0"] + MIN_SEG_BARS <= nb <= s["b1"] - 1):
                    continue
                if _splits_cell(nb) or _opens_held(nb) or _tail_penalty(nb):
                    continue                      # a cadence-tail opening is
                    # never a legitimate shift target
                k = _prefix_match(nb)
                if k > best_k or (k == best_k and best_d and abs(d) < abs(best_d)):
                    best_d, best_k = d, k
            if best_d:
                logger.info("sections failsafe: %s shifted %+d bars — opening "
                            "now equals a sibling on %d bars",
                            s["label"], best_d, best_k)
                s["b0"] += best_d
                prev_s["b1"] = s["b0"] - 1
        # report residual disagreement — the failsafe signal itself
        openings = [_open_sig(segs[j]["b0"]) for j in g]
        pair_a = [_agree(openings[x], openings[y])
                  for x in range(len(g)) for y in range(x + 1, len(g))]
        if pair_a and min(pair_a) < 0.5:
            logger.warning("sections failsafe: letter %s members agree only "
                           "%.2f on their openings — cuts suspect",
                           segs[g[0]]["label"], min(pair_a))
    logger.info("sections v2: %d segments, %d letter groups (half-bar grain)",
                len(segs), len(groups))
    return segs
