"""harmonia_min/chord_lm/grid.py — chord chart <-> metrical token grid.

The conversion the chord LM trains on: a chart (bars, each holding 1..n chord
symbols) becomes one token per metrical slot, `slots_per_bar` slots per bar.
Default 2 = half-bar. `slots_per_bar=4` gives the quarter-bar grid.

Placing chords inside a bar
---------------------------
iReal's own chord string loses the within-bar cell positions once pyRealParser
flattens it ('G7 Ab7 G7' arrives as three adjacent symbols with no spacing), so
the position has to be reconstructed. The rule here: chords divide the bar's
beats as evenly as possible, with the remainder given to the EARLIER chords —
3 chords in 4/4 becomes 2+1+1, which is the common jazz reading (the first
chord is the one that gets held).

How much this can possibly matter, over the 2,401-tune iReal corpus:

    1 chord/bar  69.2%   exact at half-bar grain
    2 chords/bar 27.3%   exact at half-bar grain
    3 chords/bar  2.5%   third chord dropped at half-bar grain
    4 chords/bar  0.9%   2nd and 4th dropped at half-bar grain
    >4            0.03%  clipped to the first `beats_per_bar`

So 96.6% of bars round-trip exactly at half-bar, and the 2+1+1-vs-1+1+2 choice
is confined to the 2.5%. At quarter-bar grain, 97.4% round-trip exactly.

What this does NOT solve
------------------------
* The dropped chords in 3- and 4-chord bars are gone, not approximated — a
  half-bar grid cannot represent them. Use `slots_per_bar=4` if they matter.
* Pickup bars / partial bars are treated as full bars; iReal does not mark them
  reliably and the corpus has no beat-level ground truth to check against.
* Section labels (A/B/C) are carried alongside the tokens but are NOT part of
  the token stream. Chordonomicon (arXiv 2410.22046) reports that structural
  part annotations help next-chord prediction; that is an open extension here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .vocab import NC, REP, is_chord, parse_ireal_token, token_name


@dataclass
class GriddedChart:
    """A chart on a metrical grid. `tokens` is what the LM consumes."""

    title: str
    tokens: list[int]                 # len == n_bars * slots_per_bar
    slots_per_bar: int
    beats_per_bar: int
    sections: list[str]               # one label per BAR (len == n_bars)
    key: str | None = None
    style: str | None = None
    n_unmapped: int = 0               # chord symbols the vocab could not map
    n_symbols: int = 0                # chord symbols seen in total

    @property
    def n_bars(self) -> int:
        return len(self.tokens) // self.slots_per_bar

    def render(self, bars_per_line: int = 4) -> str:
        """Human-readable grid, e.g. `| C:maj  %     | A:min  D:dom |`."""
        cells = []
        for b in range(self.n_bars):
            sl = self.tokens[b * self.slots_per_bar:(b + 1) * self.slots_per_bar]
            cells.append(" ".join(f"{token_name(t):<7}" for t in sl))
        lines = []
        for i in range(0, len(cells), bars_per_line):
            lines.append("| " + " | ".join(cells[i:i + bars_per_line]) + " |")
        return "\n".join(lines)


def place_in_bar(n_chords: int, beats_per_bar: int) -> list[int]:
    """Beat index at which each of `n_chords` chords starts, remainder-to-front.

    3 chords in 4/4 -> [0, 2, 3]  (durations 2, 1, 1)
    2 chords in 4/4 -> [0, 2]
    4 chords in 4/4 -> [0, 1, 2, 3]
    2 chords in 3/4 -> [0, 2]     (durations 2, 1)
    """
    n = min(n_chords, beats_per_bar)
    base, rem = divmod(beats_per_bar, n)
    starts, pos = [], 0
    for i in range(n):
        starts.append(pos)
        pos += base + (1 if i < rem else 0)
    return starts


def bar_to_slots(chord_tokens: list[int], slots_per_bar: int,
                 beats_per_bar: int) -> list[int | None]:
    """One bar's chord tokens -> `slots_per_bar` slots (None = chord unmapped).

    A slot takes the chord sounding at its START. Chords that begin and end
    strictly inside a slot are dropped (see module docstring).
    """
    if not chord_tokens:
        return [None] * slots_per_bar
    starts = place_in_bar(len(chord_tokens), beats_per_bar)
    used = chord_tokens[:len(starts)]
    out: list[int | None] = []
    for s in range(slots_per_bar):
        beat = s * beats_per_bar / slots_per_bar
        # last chord whose onset is at or before this slot's start
        idx = max(i for i, st in enumerate(starts) if st <= beat)
        out.append(used[idx])
    return out


def apply_repeats(slots: list[int | None]) -> tuple[list[int], int]:
    """Absolute per-slot chords -> the REP-compressed token stream.

    REP means "the same sounding chord as the last chord token that was written
    out". NC does not reset that memory (`C % NC %` decodes to C, C, N.C., C),
    and the first slot is never REP. Unmapped slots (None) inherit the previous
    chord — they are counted by the caller, not silently invented.

    Returns (tokens, n_rep).
    """
    out: list[int] = []
    last: int | None = None
    n_rep = 0
    for tok in slots:
        if tok is None:
            tok = last
        if tok is None:
            tok = NC
        if tok == NC:
            out.append(NC)
            continue
        if tok == REP:                       # iReal's own 'p' glyph
            out.append(REP if last is not None else NC)
            n_rep += 1
            continue
        if last is not None and tok == last:
            out.append(REP)
            n_rep += 1
        else:
            out.append(tok)
            last = tok
    return out, n_rep


def expand_repeats(tokens: list[int]) -> list[int | None]:
    """Inverse of `apply_repeats`: REP -> the chord it stands for.

    A leading REP (which `apply_repeats` never emits) decodes to None.
    """
    out: list[int | None] = []
    last: int | None = None
    for tok in tokens:
        if tok == REP:
            out.append(last)
        elif tok == NC:
            out.append(NC)
        else:
            out.append(tok)
            if is_chord(tok):
                last = tok
    return out


def chart_to_grid(measures: list[tuple[str, list[str]]], *, title: str = "",
                  slots_per_bar: int = 2, beats_per_bar: int = 4,
                  key: str | None = None, style: str | None = None) -> GriddedChart:
    """`[(section_label, [ireal_chord_token, ...]), ...]` -> a GriddedChart.

    This is the single entry point: everything upstream (iReal playlists, app
    charts, pipeline output) only has to produce that list-of-bars shape.
    """
    per_slot: list[int | None] = []
    sections: list[str] = []
    n_unmapped = n_symbols = 0
    for label, raw in measures:
        toks: list[int] = []
        for sym in raw:
            n_symbols += 1
            t = parse_ireal_token(sym)
            if t is None:
                n_unmapped += 1
                continue
            toks.append(t)
        per_slot.extend(bar_to_slots(toks, slots_per_bar, beats_per_bar))
        sections.append(label)
    tokens, _ = apply_repeats(per_slot)
    return GriddedChart(title=title, tokens=tokens, slots_per_bar=slots_per_bar,
                        beats_per_bar=beats_per_bar, sections=sections, key=key,
                        style=style, n_unmapped=n_unmapped, n_symbols=n_symbols)


def grid_to_bars(chart: GriddedChart) -> list[list[int | None]]:
    """A GriddedChart back to per-bar absolute chord ids (REP expanded)."""
    absolute = expand_repeats(chart.tokens)
    s = chart.slots_per_bar
    return [absolute[i * s:(i + 1) * s] for i in range(chart.n_bars)]
