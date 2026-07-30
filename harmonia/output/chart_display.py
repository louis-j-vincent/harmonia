"""Regrid display refinement — clean iReal-style sections from the RIGID grid.

The rigid-grid regridder (``harmonia.models.rigid_grid``, opt-in ``HARMONIA_REGRID
=1``) re-quantises the decoded chords onto the true bar grid recovered from their
onsets. On This Love the result is a very clean 4-bar-phrase structure
(``A×4 B A×3 B A C B×3``), but the standard 8-bar-block section detector
(``_sections_by_largest_unit`` / ``ssm_block_sections``) can't recover it: the
song's sections change on ODD multiples of the 4-bar phrase (A×3, a single A,
an 8-bar bridge), so an 8-bar blocking straddles every one of those boundaries
and mis-cuts the A/C region.

This module is the display-layer answer, used ONLY on the regrid path and ONLY
when it finds a confident, clean, repeating phrase structure (otherwise it
returns ``None`` and the standard detector output stands, byte-identical). It
re-derives the section list at the natural phrase grain and applies, in one
pass, the four display rules Louis's This Love lead sheet encodes
(``docs/this_love_target_spec.md``):

  1. **per-section chords-per-bar** — a section whose harmonic rhythm is < ~1.5
     chords/bar is collapsed to one chord per bar (the downbeat), dropping the
     mid-bar passing chords (This Love's A verse: ``G Cm Fm Dø``, not
     ``G Cm Fm7 Dø·Fdim``); a busier section keeps its ≤2 split (the B chorus,
     ``Cm Fm | Bb Eb``);
  2. **fold internal loop repeats** into one phrase shown once + a ``×N`` badge
     (fuzzy, so slight per-pass decode wobble still folds);
  3. **1st/2nd endings** — a phrase whose repeat diverges only in its last 1-2
     bars carries the two tails as ``endings`` (reusing ``_detect_endings``), so
     the B chorus and the C bridge each show a bracketed 1./2. ending;
  4. a compact **form string** (``A×4 B A×3 B A C B×3``).

Everything here is gated behind the regrid flag (the caller only invokes it when
``rigid_grid_for`` actually fired) and behind its own confidence check, so a
song with regrid off — or one where no clean phrase structure is found — is
completely unaffected.
"""

from __future__ import annotations

# Imported lazily by ``chart_model.to_chart_model`` (regrid hook), so chart_model
# is fully initialised by the time this top-level import runs — no cycle.
from harmonia.output.chart_model import _bar_key, _detect_endings, _span_of

__all__ = ["regrid_display_sections", "bar_spans_for_sections",
           "snap_bar_spans_to_beats"]

_RANK_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SUP = "¹²³⁴⁵⁶⁷⁸⁹"
# 4/4-pop phrase lengths, finest → coarsest (odd grains create false structure on
# a straight-4 grid and are excluded).
_P_CANDIDATES = (2, 4, 8, 16)
_MATCH_FRAC = 0.75          # per-bar downbeat-root agreement to call two blocks equal
_ENDING_FRAC = 0.7          # shared-prefix agreement for a 1st/2nd-ending pair
_COVERAGE_MIN = 0.7         # ≥ this share of blocks must sit in a repeated cluster
_MAX_CLUSTERS = 6           # a real pop/rock form has few section types
_DOMINANT_MAX = 0.6         # if ONE cluster covers ≥ this, it's a single-loop song →
                            #   defer (the standard detector's intro/section logic is
                            #   better there; this path is for BALANCED verse/chorus/
                            #   bridge forms the 8-bar detector mis-cuts, like This Love)
_DENSITY_THRESH = 1.5       # < this many chords/bar (section-wide) → 1 chord/bar


# ── bar / block primitives ────────────────────────────────────────────────────

def _downbeat_root(bar: list[dict]) -> "int | None":
    """The earliest (downbeat) real-chord root in a bar, or ``None`` (empty / N.C.)."""
    best = None
    for c in bar:
        if c.get("q") == "N" or c.get("nc"):
            continue
        beat = c.get("beat", 0)
        if best is None or beat < best[0]:
            best = (beat, c["root"] % 12)
    return best[1] if best else None


def _song_root_seq(bars: list[list[dict]], n_bars: int) -> list[int]:
    """One downbeat-root per bar for the whole song; a held/empty/N bar inherits
    the previous bar's root (the chord is still sounding), so recurrence sees the
    real harmony rather than a gap."""
    seq: list[int] = []
    prev = -1
    for b in range(n_bars):
        bar = bars[b] if b < len(bars) else []
        r = _downbeat_root(bar)
        seq.append(r if r is not None else prev)
        prev = seq[-1]
    return seq


def _seq_match(a: list[int], b: list[int], frac: float = _MATCH_FRAC) -> bool:
    k = min(len(a), len(b))
    if k == 0:
        return False
    same = sum(1 for i in range(k) if a[i] == b[i])
    return same / k >= frac


def _real_chords(bar: list[dict]) -> list[dict]:
    return [c for c in bar if c.get("q") != "N" and not c.get("nc")]


def _density_one_per_bar(block_bars: list[list[dict]]) -> bool:
    """A section is 1-chord-per-bar when it averages < _DENSITY_THRESH real chords
    per bar (This Love's A verse ≈ 1.2, its B chorus ≈ 2.0)."""
    n_ch = sum(len(_real_chords(bar)) for bar in block_bars)
    n_bar = len(block_bars) or 1
    return n_ch / n_bar < _DENSITY_THRESH


def _apply_density(block_bars: list[list[dict]], one_per_bar: bool) -> list[list[dict]]:
    """Rewrite a block's bars to the chosen harmonic-rhythm: keep only the
    downbeat (earliest) chord per bar when ``one_per_bar`` (drops mid-bar passing
    chords), else keep the ≤2 chords in time order. Copies; never mutates."""
    out: list[list[dict]] = []
    for bar in block_bars:
        real = sorted(_real_chords(bar), key=lambda c: c.get("beat", 0))
        if not real:
            out.append([])                       # held / empty — filled later
        elif one_per_bar:
            out.append([dict(real[0])])
        else:
            out.append([dict(c) for c in real[:2]])
    return out


def _fill_held(block_bars: list[list[dict]]) -> list[list[dict]]:
    """Forward-fill empty (held) bars with the previous bar's chords, and
    backfill any leading empties from the first chorded bar — so a folded
    representative phrase has no blank cells (This Love's held A@24 / bridge tail)."""
    out = [list(b) for b in block_bars]
    last: "list[dict] | None" = None
    for i, b in enumerate(out):
        if b:
            last = b
        elif last is not None:
            out[i] = [dict(c) for c in last]
    first_real = next((b for b in out if b), None)
    if first_real is not None:
        for i, b in enumerate(out):
            if b:
                break
            out[i] = [dict(c) for c in first_real]
    return out


# ── segmentation at a fixed phrase length P ────────────────────────────────────

class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def _segment_at(bars: list[list[dict]], n_bars: int, P: int):
    """Cut the song into P-bar blocks, cluster them, group consecutive same-
    cluster blocks into runs. Returns ``(runs, blocks, block_root, coverage,
    n_clusters)`` or ``None``. ``runs`` = list of block-index lists (in order)."""
    blocks = [(i, min(i + P, n_bars)) for i in range(0, n_bars, P)]
    nb = len(blocks)
    if nb < 2:
        return None
    seq = _song_root_seq(bars, n_bars)
    seqs = [seq[b0:b1] for (b0, b1) in blocks]
    block_bars = [[bars[b] if b < len(bars) else [] for b in range(b0, b1)]
                  for (b0, b1) in blocks]
    uf = _UF(nb)
    # 1. cluster blocks whose downbeat-root sequences agree on ≥ _MATCH_FRAC of bars
    for i in range(nb):
        for j in range(i + 1, nb):
            if _seq_match(seqs[i], seqs[j]):
                uf.union(i, j)
    # 2. ENDING-PAIR merge: two ADJACENT blocks in different clusters that share
    # their prefix but diverge only in the last 1-2 bars are a 1st/2nd-ending pair
    # of ONE section (This Love's C bridge — its two passes agree on only 2/4 bars,
    # below _MATCH_FRAC, so step 1 leaves them split). ``_detect_endings`` firing
    # is exactly the "shared prefix, tail-only divergence" test we want.
    for i in range(nb - 1):
        if uf.find(i) != uf.find(i + 1):
            if len(block_bars[i]) == len(block_bars[i + 1]) == P and \
                    _detect_endings([block_bars[i], block_bars[i + 1]], frac=_ENDING_FRAC):
                uf.union(i, i + 1)
    roots = [uf.find(i) for i in range(nb)]
    sizes: dict[int, int] = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1
    coverage = sum(1 for r in roots if sizes[r] >= 2) / nb
    n_clusters = len(sizes)
    runs: list[list[int]] = []
    for i in range(nb):
        if runs and roots[i] == roots[runs[-1][0]]:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs, blocks, roots, coverage, n_clusters


# ── run → section ──────────────────────────────────────────────────────────────

def _root_of(bar: list[dict]) -> "int | None":
    real = [c for c in bar if c.get("q") != "N" and not c.get("nc")]
    return real[0]["root"] % 12 if real else None


def _ending_tail(rep_even: list[list[dict]], rep_odd: list[list[dict]]) -> int:
    """1st/2nd-ending tail length (1 or 2) between a run's representative even
    (1st-pass) and odd (2nd-pass) blocks, else 0.

    A true 1st/2nd ending diverges only at the phrase's END and that divergence
    is a real HARMONIC change — a ROOT moves (This Love's B chorus ``Bb Eb`` →
    ``Ab G``; its C bridge ``G7 Cm`` → ``G G7``). Three tests, tuned to pass
    those while rejecting decode noise:

      * the last bar must differ (the ending reaches the phrase end);
      * ≥ one of the last 1-2 divergent bars must change ROOT — this rejects the
        A verse, whose passes differ only by a quality wobble on the SAME root
        (``Ddim`` vs ``Dø``, both rooted on D), which is decode noise, not an
        ending;
      * the shared prefix must agree on ROOTS (≥ _ENDING_FRAC), tolerating a
        stray prefix quality wobble (a decoded ``Cm7`` where the pair has ``Cm``).

    The tail is the contiguous divergent region at the end, capped at 2 (the
    lead-sheet norm; a longer divergence is a different section, not an ending).
    """
    P = len(rep_even)
    if P != len(rep_odd) or P < 2:
        return 0
    qdiff = [_bar_key(rep_even[i]) != _bar_key(rep_odd[i]) for i in range(P)]
    rdiff = [_root_of(rep_even[i]) != _root_of(rep_odd[i]) for i in range(P)]
    if not qdiff[P - 1]:
        return 0                                     # ending must reach the last bar
    d = 0                                            # contiguous divergent bars from end
    while d < P and qdiff[P - 1 - d]:
        d += 1
    tail = min(d, 2)
    if not any(rdiff[i] for i in range(P - tail, P)):
        return 0                                     # pure quality wobble → not an ending
    pref = range(P - tail)
    same = sum(1 for i in pref if _root_of(rep_even[i]) == _root_of(rep_odd[i]))
    return tail if same / max(len(pref), 1) >= _ENDING_FRAC else 0


def _section_from_run(run: list[int], blocks, bars: list[list[dict]],
                      root: int) -> dict:
    """Build one section dict from a run of same-cluster P-blocks: apply the
    section's density, fold the passes to one representative + ``×reps``, and
    attach ``endings`` when the passes are 1st/2nd-ending pairs."""
    P = blocks[run[0]][1] - blocks[run[0]][0]
    raw = [[bars[b] if b < len(bars) else [] for b in range(blocks[i][0], blocks[i][1])]
           for i in run]
    one_pb = _density_one_per_bar([bar for blk in raw for bar in blk])
    dens = [_fill_held(_apply_density(blk, one_pb)) for blk in raw]
    spans = [_span_of([bars[b] if b < len(bars) else []
                       for b in range(blocks[i][0], blocks[i][1])]) for i in run]
    branges = [[blocks[i][0], blocks[i][1] - 1] for i in run]
    even = [k for k in range(len(run)) if k % 2 == 0]
    odd = [k for k in range(len(run)) if k % 2 == 1]
    sec = {"id": "", "label": "", "tag": "", "reps": 0, "bars": [], "spans": spans,
           "barRanges": branges, "_root": root}

    # The representative phrase MUST be the run's FIRST block: the app replays it
    # onto each pass by offsetting by ``spans[k][0] - spans[0][0]``, so its chord
    # times have to live in block-0's time frame (and, for endings, the variants'
    # ``passes[0]`` must index the block their bars came from). even[0]=0, odd[0]=1.
    #
    # 1st/2nd ending? the passes come in even/odd pairs (blk0/blk2/… a "1st
    # ending", blk1/blk3/… a "2nd ending") when the 1st and 2nd pass share a root
    # prefix but diverge in the last 1-2 bars (This Love's B chorus & C bridge).
    # Identical passes (the A verse) give tail 0 → each block is its own rep.
    tail = _ending_tail(dens[0], dens[1]) if len(dens) >= 2 and P >= 2 else 0
    if tail:
        sec["reps"] = len(even)                      # phrase-cycles, not raw blocks
        sec["bars"] = dens[0]                         # representative = the 1st pass
        variants = [{"passes": even, "bars": dens[0][P - tail:]},
                    {"passes": odd, "bars": dens[1][P - tail:]}]
        sec["endings"] = {"tail": tail, "variants": variants}
    else:
        sec["reps"] = len(run)
        sec["bars"] = dens[0]
    return sec


def _label_and_intro(sections: list[dict]) -> None:
    """Assign letters by first-appearance (A = first distinct phrase, B = next…),
    with a leading one-off phrase that never recurs collapsed to ``Intro`` — the
    same convention as ``_sections_by_largest_unit``. Mutates in place."""
    roots_in_order: list[int] = []
    for s in sections:
        if s["_root"] not in roots_in_order:
            roots_in_order.append(s["_root"])
    # dominant phrase = the section-root covering the most sections
    from collections import Counter
    counts = Counter(s["_root"] for s in sections)
    dominant = max(counts, key=lambda r: counts[r])
    first_dom = next(i for i, s in enumerate(sections) if s["_root"] == dominant)
    lead = {sections[i]["_root"] for i in range(first_dom)}
    recurs = any(sections[i]["_root"] in lead for i in range(first_dom, len(sections)))
    intro_upto = first_dom if (first_dom > 0 and not recurs) else 0
    intro_roots = {sections[i]["_root"] for i in range(intro_upto)}
    rest_order = [r for r in roots_in_order if r not in intro_roots]
    letter_of = {r: (_RANK_ALPHA[i] if i < len(_RANK_ALPHA) else "?")
                 for i, r in enumerate(rest_order)}
    for i, s in enumerate(sections):
        if i < intro_upto:
            s["label"] = s["id"] = "Intro"
        else:
            s["label"] = s["id"] = letter_of[s["_root"]]
    # occurrence tags (A¹ A² …) for non-adjacent repeats of the same letter — the
    # UI shows these; the compact form string (built separately) does not.
    by_label: dict[str, list[dict]] = {}
    for s in sections:
        by_label.setdefault(s["label"], []).append(s)
    for label, group in by_label.items():
        if label != "Intro" and len(group) > 1:
            for i, s in enumerate(group):
                s["id"] = f"{label}{i + 1}"
                s["tag"] = _SUP[i] if i < len(_SUP) else str(i + 1)


def _form_string(sections: list[dict]) -> str:
    parts = []
    for s in sections:
        if s["label"] == "Intro":
            parts.append("Intro")
        else:
            parts.append(s["label"] + (f"×{s['reps']}" if s["reps"] > 1 else ""))
    return " ".join(parts)


# ── entry point ────────────────────────────────────────────────────────────────

def _bar_seconds(bars: list[list[dict]]) -> float:
    """Median seconds per bar, from consecutive chorded bars' first onsets. Used
    only to re-time a held chord copied into an empty bar; 2.0 if unknowable."""
    firsts = [(i, _real_chords(b)[0]["t0"]) for i, b in enumerate(bars)
              if _real_chords(b)]
    gaps = [(t1 - t0) / (i1 - i0) for (i0, t0), (i1, t1) in zip(firsts, firsts[1:])
            if i1 > i0 and t1 > t0]
    if not gaps:
        return 2.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def _section_from_vocab(sec: dict, bars: list[list[dict]], n_bars: int) -> dict:
    """One ``section_vocab`` section → the app's section shape.

    The vocabulary item is the representative phrase and its repetitions inside
    the section become the passes, so a 16-bar run of a 4-bar loop renders as one
    4-bar phrase with ``×4`` rather than sixteen bars of chart.
    """
    d, b0 = sec["d_bars"], sec["bar0"]
    reps = sec["reps"]
    blocks = [(b0 + k * d, min(b0 + (k + 1) * d, sec["bar1"])) for k in range(reps)]
    raw = [[bars[b] if b < len(bars) else [] for b in range(x, y)] for x, y in blocks]
    # A block whose FIRST bar carries no chord onset opens on a chord held over
    # from the previous bar. `_fill_held` can only look inside the block, so it
    # backfills from the block's SECOND bar instead — which printed This Love's
    # repeated verse as `Cm | Cm | Fm | Dø`, losing the G it actually opens on.
    # Seed it with the chord genuinely sounding at the downbeat: the LAST real
    # chord before the block, not the first chord of the preceding bar.
    bar_sec = _bar_seconds(bars)
    for k, (x, _y) in enumerate(blocks):
        if raw[k] and not _real_chords(raw[k][0]):
            held = next((_real_chords(bars[b])[-1] for b in range(x - 1, -1, -1)
                         if b < len(bars) and _real_chords(bars[b])), None)
            if held is None:
                continue
            # Re-time the copy to THIS bar. Copying the held chord's own t0/t1
            # verbatim reaches back into the previous section, and `spans` is what
            # drives the playhead — the audit measured a 3.78 s overlap at two
            # transitions, so the highlight sat on the verse while the bridge was
            # already sounding. Place it in the empty bar it actually fills.
            nxt = next((_real_chords(b)[0]["t0"] for b in raw[k] if _real_chords(b)),
                       None)
            t0 = (nxt - bar_sec) if nxt is not None else float(held.get("t0", 0.0))
            raw[k] = [[dict(held, beat=0, t0=t0,
                            t1=nxt if nxt is not None else t0 + bar_sec)]] + raw[k][1:]
    one_pb = _density_one_per_bar([bar for blk in raw for bar in blk])
    return {
        "id": "", "label": "", "tag": "", "reps": reps,
        "bars": _fill_held(_apply_density(raw[0], one_pb)),
        "spans": [_span_of(blk) for blk in raw],
        "barRanges": [[x, y - 1] for x, y in blocks],
        "_vocab": sec["label"],
    }


_MIN_SECTION_BARS = 8   # Louis, 2026-07-30: "as a fixed rule, we will say that a
                        # section needs to be at least 8 bars"
_MAX_ENDING_TAIL = 2    # "if 2 sections differ only by their last 2 bars, collapse
                        # them into one with 1st and 2nd ending"


def _min_section_bars() -> int:
    """The minimum bars a rendered section may have. Louis wants this tweakable
    ("sections defined as a minimum of 4 or 8 bars, should be an option we can
    tweak"), so it is read at call time from ``HARMONIA_MIN_SECTION_BARS``."""
    import os
    try:
        v = int(os.environ.get("HARMONIA_MIN_SECTION_BARS", _MIN_SECTION_BARS))
    except ValueError:
        return _MIN_SECTION_BARS
    return v if v >= 1 else _MIN_SECTION_BARS


def _bar_roots(bar: list[dict]) -> tuple:
    """A bar's roots in beat order — the identity used to compare two phrases.
    Roots only, deliberately: the decoder wobbles qualities between passes (Cm vs
    Cm7, Bb vs Bb7) and two passes of one chorus must still read as the same
    phrase."""
    return tuple(c["root"] % 12 for c in sorted(_real_chords(bar),
                                                key=lambda c: c.get("beat", 0)))


def _ending_split(a: list[list[dict]], b: list[list[dict]],
                  max_tail: int = _MAX_ENDING_TAIL) -> int:
    """If phrases ``a`` and ``b`` are the same phrase with different endings,
    return how many trailing bars differ; else 0.

    Louis's general rule, 2026-07-30: "if 2 sections differ only by their last 2
    bars, then collapse them into one with 1st and 2nd ending". Applied a priori to
    every pair of rendered sections, not special-cased per song.
    """
    if len(a) != len(b) or len(a) <= max_tail:
        return 0
    ra, rb = [_bar_roots(x) for x in a], [_bar_roots(x) for x in b]
    if ra == rb:
        return 0                                  # identical: not an ending pair
    n = len(ra)
    first_diff = next(i for i in range(n) if ra[i] != rb[i])
    tail = n - first_diff
    return tail if tail <= max_tail else 0


def _collapse_endings(sections: list[dict]) -> list[dict]:
    """Fuse rendered sections that differ only in their last bars into ONE section
    carrying 1st/2nd endings, labelled ``B1`` / ``B2`` after the base letter.

    On This Love this fuses the chorus-with-C-tail and the chorus-with-E-tail:
    their first 7 bars agree bar for bar and only the last differs
    (``Ab G7`` vs ``Ab``), so the chart shows one 8-bar B with two endings rather
    than two nearly-identical choruses.
    """
    out: list[dict] = []
    for sec in sections:
        host = next((h for h in out
                     if h["_base"] == sec["_base"]
                     and _ending_split(h["bars"], sec["bars"])), None)
        if host is None:
            out.append(sec)
            continue
        tail = _ending_split(host["bars"], sec["bars"])
        base = host["_base"]
        n_host = len(host["barRanges"])
        if "endings" not in host:
            host["endings"] = {"tail": tail, "variants": [
                {"label": f"{base}1", "passes": list(range(n_host)),
                 "bars": host["bars"][len(host["bars"]) - tail:]}]}
        host["endings"]["variants"].append({
            "label": f"{base}{len(host['endings']['variants']) + 1}",
            "passes": [n_host + k for k in range(len(sec["barRanges"]))],
            "bars": sec["bars"][len(sec["bars"]) - tail:]})
        host["barRanges"] += sec["barRanges"]
        host["spans"] += sec["spans"]
        host["reps"] += sec["reps"]
    return out


def _group_to_min_bars(vocab: list[dict], log_short: list | None = None,
                       min_bars: int | None = None):
    """A-POSTERIORI rendering pass (Louis: "keep the section detection as is, then
    add this as an a posteriori rendering step").

    Two rules, applied to the play-order section list without touching detection:

    * a section shorter than ``_MIN_SECTION_BARS`` absorbs the following
      section(s) — but only ones that are THEMSELVES short, i.e. tails. That is
      what turns This Love's ``B×3`` (6 bars) + ``C`` (2 bars) into one 8-bar
      chorus, exactly as Louis specified, while refusing to swallow the 8-bar
      bridge into the 4-bar verse that precedes it.
    * a section at or above the minimum is chunked into units of whole loops, each
      at least the minimum long, so ``A×4`` (16 bars of a 4-bar loop) renders as
      two 8-bar A's rather than one 16-bar block. A trailing chunk below the
      minimum is folded back into the previous one, so ``A×3`` (12 bars) stays a
      single 12-bar unit rather than 8 + a stranded 4.

    A unit that is STILL short — This Love's lone 4-bar verse at bars 44-47, whose
    only neighbour is the full-length bridge — is emitted as-is and appended to
    ``log_short``. Forcing it to the minimum would mean merging a verse into a
    bridge, which is worse than being one section short.

    Returns a list of units ``{parts: [section...], bar0, bar1, d_bars, label}``.
    """
    min_bars = _min_section_bars() if min_bars is None else min_bars
    units: list[dict] = []
    i = 0
    while i < len(vocab):
        s = vocab[i]
        span = s["bar1"] - s["bar0"]
        if span < min_bars:
            parts, bar1, j = [s], s["bar1"], i + 1
            while (bar1 - s["bar0"]) < min_bars and j < len(vocab):
                nxt = vocab[j]
                if (nxt["bar1"] - nxt["bar0"]) >= min_bars:
                    break                     # a full section, not a tail
                parts.append(nxt)
                bar1 = nxt["bar1"]
                j += 1
            units.append({"parts": parts, "bar0": s["bar0"], "bar1": bar1,
                          "d_bars": s["d_bars"], "label": s["label"]})
            i = j
            continue
        d = max(1, s["d_bars"])
        chunk = d * max(1, -(-min_bars // d))               # whole loops, >= minimum
        edges = list(range(s["bar0"], s["bar1"], chunk)) + [s["bar1"]]
        if len(edges) > 2 and (edges[-1] - edges[-2]) < min_bars:
            edges.pop(-2)                     # fold a short tail chunk back
        for x, y in zip(edges, edges[1:]):
            units.append({"parts": [s], "bar0": x, "bar1": y,
                          "d_bars": d, "label": s["label"]})
        i += 1
    if log_short is not None:
        log_short.extend(u for u in units
                         if (u["bar1"] - u["bar0"]) < min_bars)
    return units


def _vocab_display_sections(bars: list[list[dict]], n_bars: int, *,
                            tonic_pc: int = 0, bpb: int = 4):
    """The vocabulary detector (``harmonia.models.section_vocab``) in the app's
    section shape. Louis adopted this as THE section detector on 2026-07-30.

    Letters come from the vocabulary itself, not from first-appearance ranking, so
    a section that recurs later keeps its own letter, and a leading one-off phrase
    that never recurs is still collapsed to ``Intro`` for display.
    """
    from harmonia.models.section_vocab import form_string, vocab_sections

    vocab = vocab_sections(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    if not vocab:
        return None

    # ── a-posteriori rendering: >= 8-bar units, each written ONCE ─────────────
    # Louis, 2026-07-30: "if you've already written the A section, you don't write
    # it again, each section is written only once, and at the bottom of the chart
    # there is a kind of timeline that tells us the total structure".
    #
    # Units are keyed by (item, the tails merged into it), NOT by their exact bar
    # count or chords. That is what makes the fold actually fold: A's occurrences
    # run 8, 12 and 4 bars long and their qualities wobble between passes (G vs
    # G7, Fm7 vs Fm), so keying on content would emit A, A¹, A² — three "different"
    # verses that are one verse the decoder heard three ways. Keying on the tail
    # DOES keep the two choruses apart, which is wanted: `B×3 C` and `B×3 E` are
    # genuinely different endings.
    units = _group_to_min_bars(vocab)
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for u in units:
        k = (u["label"], tuple(p["label"] for p in u["parts"][1:]))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(u)

    sections = []
    for k in order:
        us = groups[k]
        first = us[0]
        # The written phrase: for a unit carrying merged tails, the unit itself.
        # For a pure loop, the loop repeated up to the minimum — so the canonical
        # A is 8 bars even where it happens to play 4 or 12.
        if k[1]:
            wb0, wb1 = first["bar0"], first["bar1"]
        else:
            d = max(1, first["d_bars"])
            span = d * max(1, -(-_min_section_bars() // d))
            wb0 = first["bar0"]
            wb1 = min(first["bar0"] + span, max(u["bar1"] for u in us))
        sec = _section_from_vocab(
            {"label": k[0], "bar0": wb0, "bar1": wb1,
             "d_bars": wb1 - wb0, "reps": 1}, bars, n_bars)
        # every occurrence, so playback and highlighting still cover the song
        sec["barRanges"] = sorted([u["bar0"], u["bar1"] - 1] for u in us)
        sec["spans"] = sorted(_span_of([bars[b] if b < len(bars) else []
                                       for b in range(u["bar0"], u["bar1"])])
                              for u in us)
        sec["reps"] = len(us)
        sec.pop("_vocab", None)
        sec["_base"] = k[0]
        sec["label"] = k[0] if not k[1] else f"{k[0]}→{''.join(k[1])}"
        sections.append(sec)

    # a priori rule: two phrases differing only in their last bars are ONE section
    # with 1st/2nd endings
    sections = _collapse_endings(sections)
    for i, s in enumerate(sections):
        s["label"] = s.pop("_base") if "endings" in s else s["label"]
        s["id"] = f"{s['label']}{i}"
        s["tag"] = s["label"]
        s["barRanges"] = sorted(s["barRanges"])
        s["spans"] = sorted(s["spans"])
    return sections, form_string(vocab)


def _clip_spans_in_play_order(sections: list[dict]) -> None:
    """Make the section spans partition time, so the playhead highlights exactly
    one section at a time. Mutates in place.

    ``_span_of`` measures a block by its chords' t0..t1, and a chord's t1 runs
    until the NEXT chord starts — so a chord held across a section boundary makes
    the outgoing section's span overrun the incoming one. On This Love the chorus
    tail's G7 genuinely rings through bar 24, which the next verse also owns, and
    the audit measured the resulting highlight sitting on the verse while the
    bridge was already sounding. Sustain and ownership are different questions:
    the chart's highlight follows BARS, so each span ends where the next begins.

    PLAY ORDER MEANS TIME ORDER (fixed 2026-07-30). This used to walk the spans
    in section-then-pass order, which was play order only while each section was
    a contiguous block of the song. The chart is now MINIMAL — every section is
    written once and its passes are scattered through the track (This Love: A's
    four passes are interleaved with B's five) — so that walk compared A's last
    pass against B's first and clipped nothing where it mattered. Sort by start
    time. Degenerate spans (a chart with no audio has every span ``[0, 0]``) are
    left alone: there is nothing to clip and no playhead to confuse.
    """
    flat = sorted(((si, k) for si, s in enumerate(sections)
                   for k in range(len(s["spans"]))),
                  key=lambda x: sections[x[0]]["spans"][x[1]][0])
    for (si, k), (nsi, nk) in zip(flat, flat[1:]):
        cur, nxt = sections[si]["spans"][k], sections[nsi]["spans"][nk]
        if cur[1] > nxt[0] > cur[0]:
            sections[si]["spans"][k] = [cur[0], nxt[0]]


def regrid_display_sections(bars: list[list[dict]], n_bars: int, *,
                            tonic_pc: int = 0, bpb: int = 4):
    """Re-derive clean display sections + a form string from rigid-grid bars.

    Returns ``(sections, form)``, or ``None`` to defer to the standard section
    detector (no regression). ``sections`` matches the ChartModel section shape
    the app consumes (``id/label/tag/reps/spans/barRanges/bars`` + optional
    ``endings``).

    Two detectors, in order:

    1. **the VOCABULARY detector** (``harmonia.models.section_vocab``), adopted
       2026-07-30 — Louis's call, because it reproduces his This Love lead sheet,
       which the block clustering never did. It defers on single-loop and
       through-composed songs, where it has nothing to say.
    2. **fixed-phrase block clustering** (``_block_display_sections``), the
       2026-07-29 original, kept as the fallback for whatever the vocabulary
       detector declines.
    """
    if n_bars < 8:
        return None
    vocab = _vocab_display_sections(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    if vocab is not None:
        return vocab
    return _block_display_sections(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)


def _block_display_sections(bars: list[list[dict]], n_bars: int, *,
                            tonic_pc: int = 0, bpb: int = 4):
    """Fixed-phrase block clustering — the original regrid detector, now the
    fallback under ``regrid_display_sections``. Cuts the song into P-bar blocks
    for P in ``_P_CANDIDATES``, clusters them by downbeat-root agreement, and
    folds runs into sections with ``×N`` and 1st/2nd endings.

    Unlike the vocabulary detector it cannot represent a section whose length is
    not a fixed phrase multiple, which is why it never reproduced This Love.
    """
    if n_bars < 8:
        return None
    # Scan phrase lengths FINEST → coarsest and act on the first CLEAN grain
    # (few section types, most blocks repeating). The first clean grain is the
    # true phrase length, so its verdict is final:
    #   * one cluster dominates (≥ _DOMINANT_MAX) → single-loop song → DEFER (the
    #     standard detector handles its intro/section nuance better);
    #   * blocks merely alternate A B A B with no section folding (max_run < 2) →
    #     this grain is HALF the true phrase → try the next (coarser) one;
    #   * else a balanced, folding multi-section form (This Love) → FIRE here.
    chosen = None
    for P in _P_CANDIDATES:
        if P * 2 > n_bars:
            continue
        res = _segment_at(bars, n_bars, P)
        if res is None:
            continue
        runs, blocks, roots, coverage, n_clusters = res
        if n_clusters > _MAX_CLUSTERS or coverage < _COVERAGE_MIN:
            continue                                 # not a clean grain — try coarser
        nb = len(roots)
        sizes: dict[int, int] = {}
        for r in roots:
            sizes[r] = sizes.get(r, 0) + 1
        if max(sizes.values()) / nb >= _DOMINANT_MAX:
            return None                              # single-loop song — defer
        if max((len(run) for run in runs), default=0) < 2:
            continue                                 # alternating half-phrase — coarser
        chosen = res
        break
    if chosen is None:
        return None
    runs, blocks, roots, coverage, n_clusters = chosen
    sections = [_section_from_run(run, blocks, bars, roots[run[0]]) for run in runs]
    _label_and_intro(sections)
    form = _form_string(sections)
    for s in sections:
        s.pop("_root", None)
    return sections, form


# ── the playhead's audio-seconds -> rendered-bar map ─────────────────────────
# The chart is MINIMAL: each distinct section is written ONCE and replayed on
# every pass, so an ~80-bar song renders as ~25 bars. Something therefore has to
# say, for every RENDERED bar and every pass, which real audio seconds that bar
# occupies. That map is the playhead — and until 2026-07-30 it lived in the
# client, which built it from CHORD SUSTAIN times. Two failures followed
# (Louis: "the play head does n'importe quoi ... it skips sections, doesn't play
# the first bars when 2 consecutive bars share the same chord, and ends up being
# on the wrong chords and being either early or late"):
#
#   * ``app_shell`` reconstructed pass k as ``c.t0 + (span_k[0] - span_0[0])``
#     — a RIGID TRANSLATION of the written phrase onto each repeat's start. That
#     is only correct when every pass lasts as long as the written phrase, and
#     ``_group_to_min_bars`` deliberately emits units of different bar counts
#     (This Love's verse occurs as 8, 8, 12 and 4 bars, written once as 8). On
#     This Love the 12-bar pass ran out 8.2 s early (playhead frozen on the last
#     verse bar) and the 4-bar pass overran 12 s into the bridge (playhead on
#     verse chords while the bridge sounded). 14 of 59 baked charts have a pass
#     that is >10% off the written length, up to 327%.
#   * a HELD ("%") bar — a bar with no chord because the previous chord sustains
#     through it — was timed ``[previous chord's t1, next chord's t0]``. The
#     previous chord already runs to the end of the held bar, so that is a
#     ZERO-LENGTH span: the bar could never light and its predecessor stayed lit
#     for both bars. All 376 held bars across 25 charts were dead this way.
#
# Both are the same mistake — reading BAR boundaries off chord SUSTAIN. A bar
# ends where the NEXT bar begins, not where its chord stops ringing.
#
# What this map does NOT fix: the number of rendered bars still comes from the
# section detector, so a pass whose real length disagrees with the written
# phrase (This Love's 12-bar verse written as 8) is time-COMPRESSED or STRETCHED
# to fit. The playhead then stays inside the right section and reaches its end
# exactly, but individual bars inside that pass do not line up with the audio's
# real bar lines. Fixing that needs the detector to write the right number of
# bars; it is not a playhead problem.


def _run_widths(bars: list[list[dict]], t_start: float, t_end: float) -> list[float]:
    """Relative widths of a consecutive run of rendered bars.

    A bar's width is ``next bar's ONSET - this bar's ONSET`` — where the bar
    starts, not how long its chord rings. A held bar carries no chord and so no
    onset; it takes an equal share of the gap between the nearest known onsets
    on either side, which is exactly what a bar of "same chord again" occupies.

    An onset is only believed when it is STRICTLY LATER than the last believed
    one. Two rendered bars can carry the same onset — the display fold can write
    one song bar as two grid bars (This Love's bridge tail) — and taking that at
    face value gives the first of them a width of zero. Treating the repeat as
    unknown instead makes the pair split their shared stretch evenly, which is
    what the ear expects and what the grid draws.
    """
    n = len(bars)
    if n <= 0:
        return []
    edges: list[float | None] = [None] * (n + 1)
    edges[0] = float(t_start)
    edges[n] = float(t_end)
    last = edges[0]
    for i, bar in enumerate(bars):
        if i and bar:
            onsets = [float(c.get("t0", 0.0)) for c in bar]
            if onsets and last < min(onsets) < edges[n]:
                edges[i] = min(onsets)
                last = edges[i]
    # held bars: split the gap between the enclosing known edges evenly
    i = 0
    while i <= n:
        if edges[i] is not None:
            i += 1
            continue
        a = i - 1
        j = i
        while j <= n and edges[j] is None:
            j += 1
        if j > n:                       # nothing known to the right — shouldn't happen
            for k in range(i, n + 1):
                edges[k] = edges[a]
            break
        lo, hi = edges[a], edges[j]
        for k in range(i, j):
            edges[k] = lo + (hi - lo) * (k - a) / (j - a)
        i = j
    for i in range(1, n + 1):           # a decoded onset can precede its own bar
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    widths = [edges[i + 1] - edges[i] for i in range(n)]
    if sum(widths) <= 0:
        return [1.0] * n
    # a bar squeezed to nothing by the monotonicity clamp still has to be
    # reachable — give it a floor of a twentieth of the mean bar
    floor = (sum(widths) / n) * 0.05
    return [max(w, floor) for w in widths]


def _render_blocks(sec: dict) -> list[tuple[list[list[dict]], list[int], int]]:
    """``[(bars, pass indices, reference pass)]`` in the order the client renders.

    The shared prefix first (played by every pass), then each 1st/2nd-ending
    variant's tail bars (played only by that variant's passes). Mirrors
    ``app_shell.html``'s ``loadModel`` emit order exactly.
    """
    all_bars = list(sec.get("bars") or [])
    reps = len(sec.get("spans") or [])
    endings = sec.get("endings") or {}
    variants = list(endings.get("variants") or [])
    tail = int(endings.get("tail") or 0) if variants else 0
    if tail <= 0 or tail >= len(all_bars):
        return [(all_bars, list(range(reps)), 0)]
    blocks: list[tuple[list[list[dict]], list[int], int]] = [
        (all_bars[:len(all_bars) - tail], list(range(reps)), 0)]
    for v in variants:
        passes = [p for p in (v.get("passes") or []) if 0 <= p < reps]
        blocks.append((list(v.get("bars") or []), passes, passes[0] if passes else 0))
    return blocks


def bar_spans_for_sections(sections: list[dict]) -> None:
    """Attach ``sec["barSpans"]`` — the ONE map the playhead runs on. In place.

    ``barSpans[r][slot] = [t0, t1]``: rendered bar ``r`` (prefix bars first, then
    each ending variant's tail bars) during pass ``slot`` — for a prefix bar the
    slot is the pass index, for a variant bar it is the index into that
    variant's ``passes``. Same indexing the client already used for ``tspans``.

    Guarantees, per section and per pass k:
      * the bars that pass k plays tile ``sec["spans"][k]`` end to end — first
        bar starts at the span's start, last ends at its end, no gaps, no
        overlaps, strictly increasing;
      * every bar has positive duration, held bars included;
      * a pass whose own span is empty (a chart with no audio has every span
        ``[0, 0]``) gets ``None`` in every slot rather than a row of dead
        zero-length bars — "this pass has no time", said once, explicitly.

    Section spans are clipped into play order first, so two sections never claim
    the same instant and the playhead is inside exactly one bar at any time.
    """
    _clip_spans_in_play_order(sections)
    for sec in sections:
        spans = [list(sp) for sp in (sec.get("spans") or [])]
        blocks = _render_blocks(sec)
        n_rendered = sum(len(b) for b, _, _ in blocks)
        if not spans or not n_rendered:
            sec["barSpans"] = [[] for _ in range(n_rendered)]
            continue
        # Relative widths per block, each read in ITS OWN reference pass's clock.
        # Only the proportions survive: the per-pass layout below renormalises,
        # so mixing clocks between the prefix and a variant tail is harmless.
        widths: list[list[float]] = []
        for bars, passes, ref in blocks:
            ref_span = spans[ref] if ref < len(spans) else spans[0]
            first_onset = None
            for bar in bars:
                if bar:
                    first_onset = min(float(c.get("t0", 0.0)) for c in bar)
                    break
            t_start = first_onset if first_onset is not None else ref_span[0]
            widths.append(_run_widths(bars, t_start, ref_span[1]))
        # A prefix ends where its own pass's ending tail begins, not at the
        # pass's end — otherwise the prefix would claim the tail's time too.
        if len(blocks) > 1:
            for bi, (bars, passes, ref) in enumerate(blocks[1:], start=1):
                if ref not in blocks[0][1]:
                    continue
                onset = next((min(float(c.get("t0", 0.0)) for c in bar)
                              for bar in bars if bar), None)
                if onset is None or blocks[0][2] != 0 or ref != 0:
                    continue
                widths[0] = _run_widths(blocks[0][0], spans[0][0], onset)
                break
        out: list[list[list[float]]] = [[] for _ in range(n_rendered)]
        base = 0
        offsets = []
        for bars, _, _ in blocks:
            offsets.append(base)
            base += len(bars)
        for k, (T0, T1) in enumerate(spans):
            seq: list[tuple[int, float]] = []
            for bi, (bars, passes, _) in enumerate(blocks):
                if k not in passes:
                    continue
                for j in range(len(bars)):
                    seq.append((offsets[bi] + j, widths[bi][j]))
            total = sum(w for _, w in seq)
            if float(T1) - float(T0) <= 0 or total <= 0:
                # No time to give out — say so once instead of emitting a row of
                # zero-length bars that pollute the client's lookup index.
                for r, _ in seq:
                    out[r].append(None)
                continue
            scale = (float(T1) - float(T0)) / total
            cur = float(T0)
            for n, (r, w) in enumerate(seq):
                nxt = float(T1) if n == len(seq) - 1 else cur + w * scale
                out[r].append([cur, nxt])
                cur = nxt
        sec["barSpans"] = out


def snap_bar_spans_to_beats(sections: list[dict], beat_times: list[float],
                            _max_shift_beats: float = 1.0) -> None:
    """Pull rendered-bar EDGES onto real detected beats. In place, order-safe.

    The reason the fold-reconstructed onsets need this at all is unchanged from
    the 2026-07-20 boundary-snap study (corpus mean 84 -> 27 ms): the repeats of
    a folded phrase are not identically timed, so a reconstructed onset lands
    near, not on, the beat. What is new is the ORDER GUARD. A pass can now be
    time-compressed (This Love writes a 4-bar verse as 8 bars, giving rendered
    bars shorter than one beat), and a naive nearest-beat snap would pull two
    consecutive edges onto the SAME beat and delete the bar between them —
    re-creating the dead-bar bug this whole map exists to fix. An edge therefore
    only moves if it stays strictly between its neighbours.
    """
    bt = sorted(float(b) for b in (beat_times or []))
    if len(bt) < 2:
        return
    import bisect

    period = (bt[-1] - bt[0]) / (len(bt) - 1)

    def nearest(t: float) -> float:
        i = bisect.bisect_left(bt, t)
        cands = [bt[j] for j in (i - 1, i) if 0 <= j < len(bt) and abs(bt[j] - t) <= period * _max_shift_beats]
        return min(cands, key=lambda v: abs(v - t)) if cands else t

    for sec in sections:
        spans = sec.get("spans") or []
        blocks = _render_blocks(sec)
        offsets, base = [], 0
        for bars, _, _ in blocks:
            offsets.append(base)
            base += len(bars)
        bs = sec.get("barSpans") or []
        for k in range(len(spans)):
            # the (rendered bar, slot) chain this pass plays, in time order
            chain: list[tuple[int, int]] = []
            for bi, (bars, passes, _) in enumerate(blocks):
                if k not in passes:
                    continue
                slot = passes.index(k)
                for j in range(len(bars)):
                    r = offsets[bi] + j
                    if r < len(bs) and slot < len(bs[r]):
                        chain.append((r, slot))
            if len(chain) < 2:
                continue
            if any(bs[r][s] is None for r, s in chain):
                continue                    # a pass with no time — nothing to snap
            edges = [bs[chain[0][0]][chain[0][1]][0]] + [bs[r][s][1] for r, s in chain]
            # interior edges only: the ends are the section span, which owns the
            # boundary with the neighbouring section and must not move
            for i in range(1, len(edges) - 1):
                cand = nearest(edges[i])
                if edges[i - 1] < cand < edges[i + 1]:
                    edges[i] = cand
            for i, (r, s) in enumerate(chain):
                bs[r][s] = [edges[i], edges[i + 1]]
