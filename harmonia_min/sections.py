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
BLUR_SIGMA = 0.0       # gaussian blur (half-bars) applied to the SSM.
                       # WAS 1.5. Measured 2026-08-05 on 285 Billboard tracks
                       # / 2781 GT starts (random tie-break, trivial
                       # "never move" baseline 10.8% = chance on a 9-wide
                       # window): removing the blur takes the novelty cue from
                       # 27.6% to 30.1% exact-bar, and the runs+peaks fusion
                       # from 40.2% to 41.8%. Louis called it, though not for
                       # the reason he gave — letter separability barely moves
                       # (AUC 0.886 blurred vs 0.893 raw). What the blur really
                       # costs is the PEAK POSITION: it smears the boundary it
                       # is supposed to locate.
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
    if sigma <= 0:
        return S                      # BLUR_SIGMA=0 means "don't"
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


TILE_QUANTILE = 0.95     # A bar "tiles" at period P when cos(bar, bar±P) ranks
                         # in the top 5% of THIS SONG's own off-diagonal
                         # similarities. Replaces the fixed TILE_MIN = 0.80
                         # (2026-08-05).
                         #
                         # Why: 0.80 is not a threshold on "does this repeat",
                         # it is a threshold on how harmonically homogeneous a
                         # song happens to be. Measured — it sits at the 43rd
                         # percentile of Billie Jean's matrix and the 93rd of
                         # Sunny's, and passes 12.5%–46.0% of the matrix across
                         # the corpus (p10–p90). Meanwhile the CLEAR peaks (top
                         # quartile of topographic prominence) sit at the 95.7 /
                         # 95.8 / 95.8th percentile of their own song on the
                         # three reliable grids — the peaks are at a stable
                         # RANK, so rank is the right unit.
                         #
                         # Honest about what this buys: on 285 Billboard tracks
                         # (60/40 split by song, knob tuned on one half, random
                         # tie-break) accuracy is a WASH — 37.0% vs 37.6% alone,
                         # 42.9% vs 42.5% fused, and a song-level bootstrap puts
                         # zero inside both CIs. What it buys is STABILITY: the
                         # share of matrix accepted goes from 26.0% ± 17.2 to
                         # 9.9% ± 0.4, i.e. the same amount of evidence on every
                         # song instead of varying four-fold, and slightly fewer
                         # songs collapse to chance.
                         #
                         # Set to 0.95 by Louis on the "clear peaks live at the
                         # 95th percentile" reading. The sweep (120 tracks, tune
                         # split, standalone) does NOT pick it on the mean —
                         # q0.85 36.5% / q0.90 34.6% / q0.95 34.9% / fixed 36.0%
                         # — but q0.95 has the best SONG MEDIAN of the family
                         # (40.0% vs 33.3% for every other q and for the fixed
                         # threshold). It helps the typical song and hurts a few
                         # badly; the mean hides that and the median shows it.
                         # Which one matters is a product call, and the product
                         # is judged one song at a time by ear.
                         # Limitation: the held-out split was run with the
                         # sweep's own pick (q0.85) frozen, so there is no
                         # held-out number for 0.95 — only the tune split.
                         #
                         # Consequence, stated: a quantile always passes its top
                         # 5%, so a through-composed song now gets some runs
                         # where a fixed floor gave it none. The fused number
                         # says that is not harmful on aggregate; it is a real
                         # behaviour change all the same.
                         # (Direct prominence-peak detection was the other
                         # candidate and measured FALSE: 29.8% vs 37.6%, because
                         # noisy rows have many prominent maxima and it keeps
                         # MORE of the matrix, not less.)
TILE_LAG_MIN = 2         # |lag| below this is trivially self-similar and would
                         # drag the quantile up.
RUN_COVERAGE_MIN = 0.5   # below this fraction of bars in runs, the song is
                         # not loop-built (Close to You) — fall back to the
                         # novelty-cut path unchanged.


def tile_threshold(Vb: np.ndarray, n_bars: int) -> float:
    """This song's own TILE_QUANTILE-th off-diagonal similarity.

    Scale-free by construction: the same share of the matrix is accepted on
    every song, whatever its overall harmonic homogeneity.
    """
    if n_bars < 2 * TILE_LAG_MIN + 2:
        return 1.0                               # too short to say anything
    S = Vb @ Vb.T
    i = np.arange(n_bars)
    off = S[np.abs(i[:, None] - i[None, :]) >= TILE_LAG_MIN]
    if not off.size:
        return 1.0
    return float(np.quantile(off, TILE_QUANTILE))


def tiling_runs(Vb: np.ndarray, n_bars: int) -> list[dict]:
    """Option A (Louis, 2026-08-01): maximal contiguous intervals where one
    cell tiles — [{b0, b1, period}]. A section boundary is where a run ends
    and another begins; sections then START on their cell's first bar, and
    the multiples-of-2 rule holds locally (multiples of the CELL, counted
    from the section start — not parity from bar 0 of the song).
    """
    thr = tile_threshold(Vb, n_bars)
    period_of = [0] * n_bars                     # 0 = no run
    for P in (2, 4, 8):                          # P2 priority: finest first
        for b in range(n_bars):
            if period_of[b]:
                continue
            fwd = b + P < n_bars and float(Vb[b] @ Vb[b + P]) >= thr
            bwd = b - P >= 0 and float(Vb[b] @ Vb[b - P]) >= thr
            if fwd or bwd:
                period_of[b] = P
    runs, b = [], 0
    while b < n_bars:
        P = period_of[b]
        e = b
        while e + 1 < n_bars and period_of[e + 1] == P:
            e += 1
        if P and e - b + 1 >= 2 * P:             # at least two tiles
            runs.append({"b0": b, "b1": e, "period": P})
        elif runs and P == 0:
            pass                                  # orphan zone: handled later
        b = e + 1
    return runs


SECTION_MODE_ENV = "HARMONIA_SECTIONS"   # "voice" (default) | "harmonic" | "chroma"


def detect_sections(grid: list[float], arr, times, bars=None,
                    triad=None, audio=None) -> list[dict]:
    """[{b0, b1, label}] over BAR indices — contiguous, covering, unfolded.

    Three implementations live behind this one name:

    * **voice** (default since 2026-08-08, Louis: « mets-moi cette technique en
      prod ») — `voice_sections.py`. The voice does the searching: the intro
      ends where the singing starts, an 8-bar block finds its own repeats, then
      4-bar blocks fill the holes, then letters naming the same music are
      merged. Needs `triad` AND `audio`.
    * **harmonic** (the default from 2026-08-05 to 2026-08-08) — the repetition
      dictionary in `harmonic_sections.py`, built on musx chord posteriors
      projected onto the 12 pitch classes. Needs `triad`.
    * **chroma** (the algorithm before those) — checkerboard novelty + tiling
      runs on the raw NNLS bothchroma at half-bar grain. Reachable with
      `HARMONIA_SECTIONS=chroma`, and used by report pages that spy on this
      function to capture the pipeline's grid without passing `triad`.

    WHAT FINALLY MOVED THE DEFAULT, because until 2026-08-08 the objection was
    the honest one: `voice` had been measured against Louis's own section
    annotations but never against what `harmonic` produces on those same songs,
    so promoting it would have been an unmeasured change to every grid he owns.
    `scripts/section_metric.py` made that comparison possible, and it is not
    close — **0.769 for voice against 0.599 for harmonic** over his seventeen
    validated songs, winning on 13 of 17. Three of the four losses are within
    a rounding of nothing (-0.009, -0.013, -0.023); the only real one is Blue
    Lights (0.829 -> 0.543), where `harmonic` reads the 4-bar chorus that our
    8-bar blocks straddle.

    The cost is a demucs voice separation plus a pyin pitch track on the first
    pass — cached afterwards. It buys nothing back and it is paid anyway,
    because section detection is 96-98 % of analysis time either way and
    `analyze_steps` already yields the playable chart BEFORE it runs.

    The older paths are kept reachable rather than deleted: `harmonic` is the
    only one measured on the songs Louis has not annotated, and `chroma` is the
    only one that does not depend on the bar grid being metrically right.
    """
    import os
    mode = os.environ.get(SECTION_MODE_ENV, "voice").lower()
    if mode == "voice":
        if triad is not None and audio is not None:
            from harmonia_min.voice_sections import detect_sections as _vd
            return _vd(grid, triad, bars, audio)
        # Depuis que `voice` est le DÉFAUT, ce repli n'est plus le choix d'un
        # appelant curieux : c'est la voie normale qui échoue. Il vaut donc
        # 0,599 au lieu de 0,769, et il doit s'entendre.
        logger.error("sections: mode=voice (le défaut) demande triad= ET "
                     "audio=, et l'appelant a passé triad=%s audio=%s — on "
                     "retombe sur le détecteur harmonic, qui vaut 0.599 contre "
                     "0.769 sur les morceaux annotés de Louis.",
                     "None" if triad is None else "ok",
                     "None" if audio is None else "ok")
        mode = "harmonic"
    if mode == "harmonic":
        if triad is not None:
            from harmonia_min.harmonic_sections import detect_sections as _hd
            return _hd(grid, triad, bars)
        # Not silent: a caller asking for the shipped mode without the
        # posteriors gets the OTHER algorithm, and must be told so.
        logger.warning("sections: mode=harmonic but no musx posteriors were "
                       "passed — running the CHROMA detector instead. This is "
                       "not what ships; pass triad= or set %s=chroma.",
                       SECTION_MODE_ENV)
    elif mode != "chroma":
        raise ValueError(f"{SECTION_MODE_ENV}={mode!r} — expected 'harmonic', "
                         "'chroma' or 'voice'")
    return _detect_sections_chroma(grid, arr, times, bars)


def _detect_sections_chroma(grid: list[float], arr, times,
                            bars=None) -> list[dict]:
    """The pre-2026-08-05 detector: half-bar raw-chroma SSM, checkerboard
    novelty peaks ∪ tiling-run edges, letters from off-diagonal blocks."""
    n_bars = len(grid) - 1
    if n_bars < 2 * MIN_SEG_BARS:
        return [{"b0": 0, "b1": n_bars - 1, "label": "A"}]
    F = halfbar_features(grid, arr, times)
    S = F @ F.T
    n = len(S)                                   # = 2 * n_bars

    # ── option A (Louis, 2026-08-01): cuts at TILING-RUN edges ──────────────
    # A boundary is where one cell's tiling stops and another starts; each
    # section then STARTS on its cell's first bar (Louis: the old bug was
    # "on commence mal la section" — parity holds naturally inside a section
    # once it starts right). Fallback to novelty cuts for through-composed
    # songs (low run coverage, e.g. Close to You).
    from harmonia_min.folding import _bar_vecs
    Vb = _bar_vecs(F, n_bars)
    runs = tiling_runs(Vb, n_bars)
    coverage = sum(r["b1"] - r["b0"] + 1 for r in runs) / max(1, n_bars)
    # UNION, not either/or (Louis, 2026-08-05: « les pics recouvrent très bien,
    # ils devraient corriger les tuilages, pas l'un ou l'autre mais les deux
    # ensemble »). The old code picked ONE source — run edges when they covered
    # >= RUN_COVERAGE_MIN of the song, novelty otherwise — and then deleted
    # every novelty candidate sitting inside a run. Measured on 285 Billboard
    # tracks: run edges alone place 35.8% of starts on the exact bar, peaks
    # alone 30.1%, both together 41.8%; boundary F 0.232 / 0.217 / 0.281. The
    # deleted candidate set was worth +4.9pp of F on its own.
    # `run_cuts` is now always a list (empty when no run qualifies), so a
    # through-composed song simply contributes no run edges instead of
    # switching the whole algorithm to another branch.
    run_cuts = sorted({c for r in runs for c in (r["b0"], r["b1"] + 1)
                       if 0 < c < n_bars})
    logger.info("sections: %d runs (coverage %.0f%%), %d run edges + novelty "
                "peaks (union)", len(runs), coverage * 100, len(run_cuts))

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

    # RIGID rule (Louis, 2026-07-31): sections are ALWAYS multiples of 2
    # bars — once the cut location is decided, snap to the nearest multiple
    # of 2. With the section grid anchored at bar 0, that means every cut
    # lands on an EVEN bar index. Cells stay a hard constraint; the old
    # "never open on a held bar" rule is DELETED — it was a "%"-display
    # artifact, and Louis's structure has A OPENING on the held G (the same
    # role as A1's opening G/B). Ties between the two nearest even bars are
    # broken by the novelty curve itself.
    # Both sources feed the same cut list. Novelty candidates are NO LONGER
    # dropped where a run covers them — that deletion is what the union
    # replaces, and it was measured to cost 4.9pp of boundary F. A peak inside
    # a run now splits it, which is the point: the peaks correct the tiling.
    cuts = list(run_cuts)
    for h in cand:
        base = (h + 1) / 2.0
        # prev must be the nearest cut BELOW this candidate — in run mode,
        # cuts is pre-filled with ALL run edges, so cuts[-1] was the LAST
        # edge of the song and every in-gap novelty candidate died on the
        # range check (the dead code that swallowed This Love's bridge —
        # root-caused and fix verified by the 2026-08-01 causal audit).
        below = [c for c in cuts if c <= base]
        prev = max(below) if below else 0
        c_lo = int(base // 2) * 2                 # nearest even bars around base
        options = [c_lo, c_lo + 2] if c_lo != base else [c_lo]
        best, best_score = None, None
        for c in options:
            if not (prev + MIN_SEG_BARS <= c <= n_bars - 1):
                continue
            if _splits_cell(c):
                continue
            hb = min(n - 1, 2 * c)
            score = (abs(c - base), -nov[hb])
            if best_score is None or score < best_score:
                best, best_score = c, score
        if best is not None and best not in cuts:
            cuts.append(best)
    bounds = [0] + sorted(set(cuts)) + [n_bars]
    segs = [{"b0": a, "b1": b - 1} for a, b in zip(bounds, bounds[1:])
            if b - a > 0]
    # orphan fragments attach to the CLOSING section on their left: anything
    # shorter than the preceding run's period is tail material (the held Ab
    # after She Will Be Loved's chorus; This Love's 2-bar verse endings that
    # fall out of the P4 tiling because they vary between passes)
    # One path now that run_cuts is always a list: a song with no runs simply
    # has an empty `period_at`, so only genuine slivers (< MIN_SEG_BARS) merge
    # — the behaviour the old `else` branch had.
    period_at = {}
    for r in runs:
        for b in range(r["b0"], r["b1"] + 1):
            period_at[b] = r["period"]
    merged_orph = []
    for sg in segs:
        plen = period_at.get(merged_orph[-1]["b0"], MIN_SEG_BARS) \
            if merged_orph else MIN_SEG_BARS
        if merged_orph and sg["b0"] not in period_at \
                and sg["b1"] - sg["b0"] + 1 <= max(plen, MIN_SEG_BARS):
            merged_orph[-1]["b1"] = sg["b1"]
        else:
            merged_orph.append(sg)
    segs = merged_orph

    # ── cadence-tail openings roll into the CLOSING section ─────────────────
    # A section must not OPEN on (attack, held) — that pair is the previous
    # phrase's cadence + resonance (This Love's "Ab G | (G)", She Will Be
    # Loved's "Ab | (Ab)"). If starting 2 bars later (cell-preserving) gives
    # an (attack, attack) opening, shift the boundary and let the closing
    # section absorb its own tail.
    def _held_b(b):
        return bars is not None and 0 <= b < n_bars \
            and (not bars[b] or all(c.get("carry") for c in bars[b]))
    for i in range(1, len(segs)):
        sg = segs[i]
        b0 = sg["b0"]
        if sg["b1"] - b0 + 1 > 4 and not _held_b(b0) and _held_b(b0 + 1) \
                and not _held_b(b0 + 2) and not _held_b(b0 + 3):
            sg["b0"] = b0 + 2
            segs[i - 1]["b1"] = b0 + 1
            logger.info("sections: cadence-tail opening at bar %d rolled "
                        "into the closing section (+2)", b0)

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

    # ── held boundary bar joins the section it OPENS (Louis, 2026-08-01:
    # « le B de This Love finit au 23, pas au 24 ») ────────────────────────────
    # A held bar ending section X whose SOUNDING chord equals the OPENING of
    # any same-letter sibling of the NEXT section belongs to the next section
    # (This Love: the held G = A1's G/B opening → opens A2/A3). Counter-case
    # kept correct: She Will Be Loved's held Ab matches no verse opening →
    # stays with its chorus. Deterministic, bar-granularity, sig-based.
    by_letter = {}
    for sg in segs:
        by_letter.setdefault(sg["label"], []).append(sg)
    for i in range(1, len(segs)):
        prev_s, cur = segs[i - 1], segs[i]
        e = prev_s["b1"]
        if e <= prev_s["b0"] or not _is_held(e):
            continue
        sibling_opens = {_sig(sg["b0"]) for sg in by_letter[cur["label"]]
                         if sg is not cur}
        if _sig(e) in sibling_opens and _sig(e) != ("%",):
            logger.info("sections: held bar %d (sig matches a %s-opening) "
                        "moves from %s to open %s", e, cur["label"],
                        prev_s["label"], cur["label"])
            cur["b0"] = e
            prev_s["b1"] = e - 1

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
            for d in (-2, 2):      # parity-preserving shifts only
                nb = s["b0"] + d
                if not (prev_s["b0"] + MIN_SEG_BARS <= nb <= s["b1"] - 1):
                    continue
                if _splits_cell(nb):
                    continue
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
    # ── merge adjacent NEVER-REPEATED letters (Louis, 2026-08-01): This
    # Love's C[6b] + D[2b] are each played once — a one-shot run of unique
    # material is ONE section (here: an 8-bar bridge), not two. Only adjacent
    # singletons fuse; a repeated letter never absorbs anything.
    from collections import Counter as _Ct
    counts = _Ct(s["label"] for s in segs)
    def _short(x):
        return x["b1"] - x["b0"] + 1 < 8
    merged = []
    for s in segs:
        if merged and counts[s["label"]] == 1 \
                and counts[merged[-1]["label"]] == 1 \
                and (_short(s) or _short(merged[-1])):
            # guard (measured on Close to You): two ≥8-bar one-shots are REAL
            # standalone sections (its two key halves) — raw singleton-fusion
            # swallowed the 1:38 modulation. Only fragments (<8 bars) glue.
            merged[-1]["b1"] = s["b1"]
        else:
            merged.append(s)
    segs = merged
    logger.info("sections v2: %d segments, %d letter groups (half-bar grain)",
                len(segs), len(groups))
    return segs
