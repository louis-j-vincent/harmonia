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

__all__ = ["regrid_display_sections"]

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
    for k, (x, _y) in enumerate(blocks):
        if raw[k] and not _real_chords(raw[k][0]):
            held = next((_real_chords(bars[b])[-1] for b in range(x - 1, -1, -1)
                         if b < len(bars) and _real_chords(bars[b])), None)
            if held is not None:
                raw[k] = [[dict(held, beat=0)]] + raw[k][1:]
    one_pb = _density_one_per_bar([bar for blk in raw for bar in blk])
    return {
        "id": "", "label": "", "tag": "", "reps": reps,
        "bars": _fill_held(_apply_density(raw[0], one_pb)),
        "spans": [_span_of(blk) for blk in raw],
        "barRanges": [[x, y - 1] for x, y in blocks],
        "_vocab": sec["label"],
    }


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
    sections = [_section_from_vocab(s, bars, n_bars) for s in vocab]
    seen: dict[str, int] = {}
    for s in sections:
        seen[s["_vocab"]] = seen.get(s["_vocab"], 0) + 1
    for i, s in enumerate(sections):
        lab = s.pop("_vocab")
        one_off_head = i == 0 and seen[lab] == 1
        s["label"] = "Intro" if one_off_head else lab
        s["id"] = f"{s['label']}{i}"
        s["tag"] = s["label"]
    form = form_string(vocab)
    return sections, form


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
