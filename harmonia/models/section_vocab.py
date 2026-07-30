"""Vocabulary section detection — learn a song's repeating patterns, then cover
the song with them.

Designed with Louis on 2026-07-30 and adopted as the section detector for the
regrid path, replacing the fixed-phrase block clustering in
``harmonia.output.chart_display`` (which he judged unusable). Validated against
``docs/this_love_target_spec.md``, his hand-written lead sheet for Maroon 5's
"This Love"; that song's accepted output is::

    A×4  B×3 C  A×3  B×3 C  A  D  B×3 C  B×3 E  B×3 E

    A  4-bar  G Cm Fm7 Dø          8 occurrences   (the verse loop)
    B  2-bar  Cm Fm | Bb Eb       15 occurrences   (the chorus loop)
    C  2-bar  Cm F△7 | Ab G7       3 occurrences   (the chorus tail)
    D  8-bar  Fm Eb | Bm Cm7 …     1 occurrence    (the bridge)
    E  2-bar  Cm7 F | Ab Ab        2 occurrences   (the outro tail)

``B×3 C`` is the 8-bar chorus: the 2-bar loop three times, then its cadential
tail. The tail is a first-class vocabulary item rather than a "1st/2nd ending"
special case — that was the design decision that made the rest fall out.

## How it works

1. **Half-bar slots.** Each bar is split into ``SLOTS_PER_BAR`` slots and a held
   chord is forward-filled across its slots. Half-bar resolution is what lets a
   2-chords-per-bar chorus expose its true 2-bar loop while a 1-chord-per-bar
   verse exposes a 4-bar one; at one slot per bar the chorus loop is invisible.

2. **The SSM** (``chord_ssm``) is cosine similarity between weighted chord-tone
   vectors, so **Bb is closer to Gm than to F** — Louis's standing requirement.
   See that function for why the root-one-hot alternatives are wrong.

3. **Slide ACROSS X, read the stripe's diagonal** (``stripe_diag``). Rows are
   pinned to one pattern, only the columns move: "where else in the song does
   THIS pattern come back?". This is NOT Foote novelty, which slides a kernel
   ALONG the diagonal and answers the different question "does the texture change
   here?" — that finds boundaries but can never say which section it is. Reading
   the whole slid square instead of its diagonal also fails: the square is
   phase-invariant and scores This Love's chorus 0.70 against its verse, because
   it measures texture rather than chords.

4. **Pass 1 — discover and claim.** From the first uncovered bar, find the
   smallest whole-bar lag that reproduces the span (``minimal_period``), then
   slide that pattern over the WHOLE song and claim every bar-aligned peak. A
   pattern is never restricted to multiples of its own length from its own start;
   that "own grid" assumption made a section starting one bar off unrecoverable.

5. **Pass 1b — degraded copies.** A repetition whose chords the decoder got
   slightly wrong scores in the measured band 0.76–0.84, above the 0.70 ceiling
   that genuinely different sections reach. Re-offering known patterns at
   ``CONTINUE`` before minting new letters is what keeps This Love's 3rd and 4th
   verse passes (bars 28 and 32, where ``G`` decoded as ``C`` and ``Bm``) from
   becoming a bogus section.

6. **Pass 2 — mine the holes.** Whatever no pattern claimed is itself a candidate
   pattern: take the hole's content, slide it, and give it a letter if it recurs.
   This is how the chorus tail is found.

## What this does NOT solve

* **It inherits the bar grid.** Everything here assumes ``harmonia.models.
  rigid_grid`` recovered the right bar length and phase. The grid octave is
  ambiguous corpus-wide (held chords double it, two-chords-per-bar halves it) and
  is NOT resolved here; a wrong grid produces a confidently wrong vocabulary.
* **No drum or rhythm evidence.** A drum-fill prior (``groove_dev`` in
  ``docs/research_sessions/rhythm_ssm_2026-07-30.md``) is the intended next
  input, as a small additive boundary prior — not wired yet.
* **Thresholds are calibrated on ONE song.** They are measured, not guessed, but
  This Love is a single song and therefore a hypothesis (CLAUDE.md #5). They have
  deliberately not been tuned on others yet.
* **No context merge.** A rule fusing two same-length items that always follow
  the same item was built and rejected — it would silently fuse two genuinely
  distinct sections that happen to share length and predecessor.
"""

from __future__ import annotations

import numpy as np

from harmonia.theory.local_key import _FLAT_NAMES, chord_pcs

__all__ = ["SLOTS_PER_BAR", "build_slots", "chord_ssm", "stripe_diag",
           "minimal_period", "vocab_sections"]

SLOTS_PER_BAR = 2
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── thresholds: MEASURED, not guessed ────────────────────────────────────────
# Calibrated 2026-07-30 by scoring `stripe_diag` against spans whose identity is
# known from docs/this_love_target_spec.md:
#
#   same section, clean repeat          0.893 – 1.000
#   same section, a chord mis-decoded   0.763 – 0.843
#   DIFFERENT section                   0.168 – 0.697   (max = chorus vs bridge,
#                                                        which really share Fm/Eb)
#
# There is a genuinely empty band between 0.697 and 0.763, and CONTINUE sits in
# it. Re-derive these numbers before changing them.
SAME_CHORD = 0.90     # an SSM cell this high is "the same chord"
REP_STRICT = 0.85     # a clean repeat of the pattern
CONTINUE = 0.75       # ...still the same section, with a chord mis-decoded
PERIOD_FLOOR = 0.75   # a period must reproduce its span at least this well
PERIOD_MARGIN = 0.30  # accept the SMALLEST period scoring within this of the best
_MAX_ITERS = 40       # runaway guard on the discovery loop


# ── slots ─────────────────────────────────────────────────────────────────────

def _token(root: int, q: str) -> str:
    return _FLAT_NAMES[int(root) % 12] + (q or "")


def build_slots(bars: list[list[dict]], n_bars: int, *, tonic_pc: int = 0,
                bpb: int = 4) -> tuple[list[str], list[int], list[bool]]:
    """DISPLAY bars → ``(tokens, roots, known)``, one entry per half-bar slot.

    A chord lands in the slot containing its beat and the earliest chord owns the
    slot. Then two DIFFERENT kinds of gap are treated differently — conflating
    them is what made This Love's third verse pass split off as a phantom section:

    * **a slot with no event at all** is a HELD chord, still sounding, so it
      inherits the previous chord and stays ``known``;
    * **a slot whose event is a no-chord** (``q == "N"`` / ``nc``) is the decoder
      saying *I don't know here*. It is marked ``known = False`` and inherits
      NOTHING. Filling it from the previous bar invents evidence: bar 28 of This
      Love is a no-chord, and forward-filling it with bar 27's passing ``Bb△7``
      scored the verse repeat below threshold, minting a bogus section — while the
      honest answer is that this slot should not vote either way.

    ``roots`` are tonic-relative and used ONLY to reject a candidate pattern that
    holds a single distinct chord — never for similarity, which is always
    chord-tone based.
    """
    n = n_bars * SLOTS_PER_BAR
    slot: list[dict | None] = [None] * n
    unknown = [False] * n
    per_slot_beats = bpb / SLOTS_PER_BAR
    for b in range(min(n_bars, len(bars))):
        for c in sorted(bars[b], key=lambda c: c.get("beat", 0)):
            k = min(int(c.get("beat", 0) / per_slot_beats), SLOTS_PER_BAR - 1)
            s = b * SLOTS_PER_BAR + k
            if c.get("q") == "N" or c.get("nc"):
                if slot[s] is None:
                    unknown[s] = True
                continue
            if slot[s] is None:
                slot[s], unknown[s] = c, False
    cur = None
    for s in range(n):
        if slot[s] is not None:
            cur = slot[s]
        elif not unknown[s]:
            slot[s] = cur
    tokens = [_token(c["root"], c.get("q", "")) if c else "" for c in slot]
    roots = [((int(c["root"]) - tonic_pc) % 12) if c else -1 for c in slot]
    known = [bool(t) for t in tokens]
    return tokens, roots, known


# ── the matrix ────────────────────────────────────────────────────────────────

def chord_ssm(tokens: list[str]) -> np.ndarray:
    """Cosine similarity between weighted chord-tone vectors — the project's SSM.

    Louis's standing requirement (2026-07-30): **Bb must be closer to Gm than to
    F**. Verified with ``chord_pcs`` weights (root 2.0 / 3rd 1.5 / 5th 1.0 /
    7th 0.8)::

        Bb ↔ Gm   0.701     relatives are CLOSE (they share Bb and D)
        Bb ↔ F    0.276     a fifth apart is FAR (they share only F)
        G  ↔ G7   0.959     the same chord with its quality wobbling by the
        Dø ↔ Dm7♭5 0.959    decoder still reads as the same chord
        G  ↔ Cm   0.264     verse and chorus openings stay far apart
        Cm ↔ Bb   0.212

    Two alternatives are WRONG here, recorded so they are not retried:
    ``section_structure.build_chord_ssm`` (root one-hot | quality one-hot, cosine)
    scores a root-match/quality-mismatch at only 0.5, which dragged This Love's
    true 4-bar verse period down to 0.69 and made it undetectable. Forcing that
    function's quality slot constant to fake "root only" leaves a **0.5 floor**
    under every cell, halving the dynamic range of every downstream curve — a
    silent scale bug that still produces plausible numbers (CLAUDE.md #1). A true
    root-only matrix was also measured as a second weighted vote and added no
    separation while actively costing coverage, so it is not used.
    """
    n = len(tokens)
    F = np.zeros((n, 12))
    for i, tok in enumerate(tokens):
        if not tok:
            continue
        for pc, w in chord_pcs(tok).items():
            F[i, pc % 12] += w
    F /= np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-9, None)
    return np.clip(F @ F.T, 0.0, 1.0).astype(np.float32)


def stripe_diag(S: np.ndarray, row0: int, d: int,
                known: list[bool] | None = None) -> np.ndarray:
    """Slide across X with the rows pinned to ``[row0, row0+d)`` and read the slid
    stripe's DIAGONAL: ``out[t] = mean_i S[row0+i, t+i]``.

    1.0 means the d slots starting at t reproduce the pattern slot for slot. NaN
    where the window would run off the end.

    Slots the decoder could not name are ABSTAINED from rather than scored: a
    comparison involving an unknown slot is dropped from the mean, so a missing
    chord neither helps nor hurts. If fewer than half the slots are comparable the
    result is NaN — too little evidence to claim anything.
    """
    n = S.shape[0]
    out = np.full(n, np.nan)
    for t in range(0, n - d + 1):
        vals = [S[row0 + i, t + i] for i in range(d)
                if known is None or (known[row0 + i] and known[t + i])]
        if len(vals) * 2 >= d:
            out[t] = float(np.mean(vals))
    return out


# ── the minimal repeating pattern ─────────────────────────────────────────────

def minimal_period(S: np.ndarray, roots: list[int], t0: int, hi: int,
                   known: list[bool] | None = None) -> int | None:
    """Smallest whole-bar lag that reproduces the span starting at ``t0``, searched
    within ``[t0, hi)``. ``None`` means nothing reproduces it, which is itself
    informative: the span is a one-off (a bridge, or a cadential tail) and the
    caller should treat the span itself as the pattern.

    Three constraints on a candidate lag, each one paid for by a real failure:

    * **a whole number of bars** — odd slot counts produced junk patterns like a
      3-bar ``Cm F△7 Ab G7 G7 G7``;
    * **longer than the held run** — a chord held h slots trivially matches itself
      at every lag ≤ h, which is how a held ``Cm Cm`` was accepted as a perfect
      1-bar "pattern" with a band score of 1.00;
    * **at least two distinct chords** — one chord is not a pattern.

    The winner is the SMALLEST lag scoring within ``PERIOD_MARGIN`` of the best,
    not the best: a multiple of the true period always scores at least as well as
    the period itself, so picking the argmax picks a multiple.
    """
    n = S.shape[0]
    j = t0 + 1
    while j < n and S[t0, j] >= SAME_CHORD:
        j += 1
    held = j - t0

    lo = max(SLOTS_PER_BAR, (held // SLOTS_PER_BAR + 1) * SLOTS_PER_BAR)
    band: dict[int, float] = {}
    for d in range(lo, (hi - t0) + 1, SLOTS_PER_BAR):
        if t0 + 2 * d > hi:
            break
        if len(set(roots[t0:t0 + d])) < 2:
            continue
        vals = [S[t0 + i, t0 + i + d] for i in range(d)
                if known is None or (known[t0 + i] and known[t0 + i + d])]
        if len(vals) * 2 < d:
            continue                      # too little named harmony to judge
        band[d] = float(np.mean(vals))
    if not band:
        return None
    cut = max(PERIOD_FLOOR, max(band.values()) - PERIOD_MARGIN)
    return next((d for d in sorted(band) if band[d] >= cut), None)


# ── cover ─────────────────────────────────────────────────────────────────────

def _claim(corr: np.ndarray, d: int, n_bars: int, covered: list, label: str,
           thresh: float) -> list[int]:
    """Claim every bar-aligned peak scoring ≥ ``thresh`` that lands on free bars.
    Strongest first, so a clean occurrence wins a contested span over a weak one."""
    n = len(corr)
    cand = [t for t in range(0, n - d + 1, SLOTS_PER_BAR)
            if not np.isnan(corr[t]) and corr[t] >= thresh - 1e-9]
    cand.sort(key=lambda t: (-corr[t], t))
    got = []
    for t in cand:
        b0, b1 = t // SLOTS_PER_BAR, min((t + d) // SLOTS_PER_BAR, n_bars)
        if any(covered[b] is not None for b in range(b0, b1)):
            continue
        for b in range(b0, b1):
            covered[b] = label
        got.append(t)
    return sorted(got)


def _holes(covered: list, n_bars: int) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < n_bars:
        if covered[i] is None:
            j = i
            while j < n_bars and covered[j] is None:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def vocab_sections(bars: list[list[dict]], n_bars: int, *, tonic_pc: int = 0,
                   bpb: int = 4) -> list[dict] | None:
    """Learn the song's pattern vocabulary and cover the song with it.

    Returns a list of contiguous sections, in play order::

        {"label": "A", "bar0": 0, "bar1": 16, "d_bars": 4, "reps": 4,
         "tokens": ["G", "G", "C-", ...]}

    ``bar1`` is EXCLUSIVE. ``d_bars`` is the vocabulary item's own length and
    ``reps`` how many times it repeats inside this section. Returns ``None`` when
    the song is too short or nothing repeats at all.
    """
    if n_bars < 8:
        return None
    tokens, roots, known = build_slots(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    if not any(tokens):
        return None
    S = chord_ssm(tokens)
    n = len(tokens)

    covered: list[str | None] = [None] * n_bars
    pats: list[dict] = []
    tried: set[int] = set()

    # One loop, and the ORDER inside it is the whole design: the vocabulary we
    # already have always gets first refusal — strictly, then at the looser
    # measured threshold — before a new letter is minted. Minting first was a real
    # bug: This Love's 3rd verse pass scores 0.822, below REP_STRICT, so a "claim
    # strict everywhere, then loosen at the end" ordering let a brand-new pattern
    # discovered at the next uncovered bar swallow bars 28-35 before the verse got
    # its second chance, splitting A×3 into A / C / A.
    for _ in range(_MAX_ITERS):
        for thresh in (REP_STRICT, CONTINUE):
            for p in pats:
                p["spans"] = sorted(set(p["spans"] + _claim(
                    p["corr"], p["d"], n_bars, covered, p["label"], thresh)))

        start = next((b for b in range(n_bars)
                      if covered[b] is None and b not in tried), None)
        if start is None or len(pats) >= len(_LETTERS):
            break
        tried.add(start)

        # Explain THIS hole: the pattern may be at most the hole itself. When
        # nothing inside the hole repeats, the hole IS the pattern — that is how a
        # cadential tail or a one-off bridge becomes a first-class vocabulary item
        # instead of being absorbed into whatever preceded it.
        end = start
        while end < n_bars and covered[end] is None:
            end += 1
        t0, span = start * SLOTS_PER_BAR, (end - start) * SLOTS_PER_BAR
        d = minimal_period(S, roots, t0, t0 + span, known) or span
        corr = stripe_diag(S, t0, d, known)
        label = _LETTERS[len(pats)]
        got = _claim(corr, d, n_bars, covered, label, REP_STRICT)
        if not got:
            continue
        pats.append({"label": label, "row0": t0, "d": d, "corr": corr,
                     "spans": got})

    # ── when to stay quiet ───────────────────────────────────────────────────
    # Two songs this detector has nothing useful to say about, and must defer on
    # rather than answer confidently (both were caught as regressions by
    # tests/test_chart_display.py::TestGating):
    #
    #  * ONE vocabulary item — a song that is a single repeating loop has no
    #    section structure to report. "The whole song is A×10" is true and
    #    useless; the standard detector's intro/section handling is better there.
    #  * NOTHING RECURS — on a through-composed song no lag reproduces anything,
    #    so the hole-is-the-pattern fallback would make the ENTIRE SONG one
    #    "pattern" that trivially matches itself once. That is not a finding.
    if len(pats) < 2 or max(len(p["spans"]) for p in pats) < 2:
        return None
    for b in range(n_bars):              # anything still free joins its neighbour
        if covered[b] is None:
            covered[b] = covered[b - 1] if b else pats[0]["label"]

    by_label = {p["label"]: p for p in pats}
    out: list[dict] = []
    for b in range(n_bars):
        if out and out[-1]["label"] == covered[b] and out[-1]["bar1"] == b:
            out[-1]["bar1"] = b + 1
        else:
            out.append({"label": covered[b], "bar0": b, "bar1": b + 1})
    for s in out:
        p = by_label.get(s["label"])
        d_bars = (p["d"] // SLOTS_PER_BAR) if p else (s["bar1"] - s["bar0"])
        s["d_bars"] = max(1, d_bars)
        s["reps"] = max(1, (s["bar1"] - s["bar0"]) // s["d_bars"])
        s["tokens"] = tokens[p["row0"]:p["row0"] + p["d"]] if p else []
    return out


def form_string(sections: list[dict]) -> str:
    """``A×4 B×3 C A×3 …`` — the compact form, one term per contiguous section."""
    return " ".join(f"{s['label']}×{s['reps']}" if s["reps"] > 1 else s["label"]
                    for s in sections)
