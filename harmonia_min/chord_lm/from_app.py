"""harmonia_min/chord_lm/from_app.py — our own charts -> the same token grid.

The iReal side (corpus.py) has to *reconstruct* within-bar positions, because
pyRealParser drops them. Our charts do not: every chord in
`harmonia_min/state/charts/*.json` carries an explicit `beat` inside its bar, so
this conversion is exact — no placement heuristic, no dropped chords beyond what
the grid resolution itself cannot hold.

That asymmetry is the point of having both loaders: the LM trains on iReal
(high-trust chords, reconstructed rhythm) and is applied to our charts (uncertain
chords, exact rhythm).

Chord fields consumed, as written by harmonia_min.labels.to_chord:
    root  0-11 pitch class of the FUNCTIONAL root
    q     iReal quality tail ("", "-7", "^7", "7sus4", ...)
    bass  sounding bass pc, or -1 — IGNORED here on purpose (see vocab.py:
          the LM's roots are functional, a D-7/A is a D-7)
    nc    True for an explicit no-chord
    beat  beat index within the bar
"""
from __future__ import annotations

import json
from pathlib import Path

from .grid import GriddedChart, apply_repeats
from .vocab import NC, parse_app_chord


def _bar_to_slots(bar: list[dict], slots_per_bar: int, beats_per_bar: int,
                  carry_in: int | None) -> tuple[list[int | None], int | None, int]:
    """One bar's chord dicts -> slots. Returns (slots, chord carried out, n_unmapped)."""
    events: list[tuple[float, int | None]] = []
    n_unmapped = 0
    for ch in bar:
        beat = float(ch.get("beat", 0))
        if ch.get("nc"):
            events.append((beat, NC))
            continue
        tok = parse_app_chord(int(ch["root"]), str(ch.get("q", "")))
        if tok is None:
            n_unmapped += 1
            continue
        events.append((beat, tok))
    events.sort(key=lambda e: e[0])

    slots: list[int | None] = []
    current = carry_in
    for s in range(slots_per_bar):
        start = s * beats_per_bar / slots_per_bar
        end = (s + 1) * beats_per_bar / slots_per_bar
        # the chord sounding at the slot's start: the latest onset <= start
        at_start = [t for b, t in events if b <= start]
        if at_start:
            current = at_start[-1]
        elif any(start < b < end for b, _ in events):
            # Nothing sounds at the boundary (leading edge of the chart, no carry)
            # but something starts inside the slot — take it rather than emitting
            # a hold of nothing. Once something IS sounding at the boundary it
            # wins, so a beat-4 chord in a half-bar grid is dropped: that is the
            # grid's resolution limit and matches grid.bar_to_slots exactly.
            current = next(t for b, t in events if start < b < end)
        slots.append(current)
    if events:
        current = events[-1][1]
    return slots, current, n_unmapped


def app_chart_to_grid(chart: dict, *, slots_per_bar: int = 2) -> GriddedChart:
    """A harmonia_min chart dict -> GriddedChart.

    Sections are emitted in the order their `barRanges` start, each section's
    bars once. `reps` is NOT expanded: the token stream mirrors the chart as
    displayed, which is the object a user edits and the LM should score.
    """
    beats_per_bar = int(chart.get("bpb", 4) or 4)
    sections = sorted(
        chart.get("sections", []),
        key=lambda s: (s.get("barRanges") or [[10**9]])[0][0],
    )
    per_slot: list[int | None] = []
    labels: list[str] = []
    carry: int | None = None
    n_unmapped = n_symbols = 0
    for sec in sections:
        label = sec.get("label") or sec.get("id") or "?"
        for bar in sec.get("bars", []):
            n_symbols += len(bar)
            slots, carry, bad = _bar_to_slots(bar, slots_per_bar, beats_per_bar, carry)
            n_unmapped += bad
            per_slot.extend(slots)
            labels.append(label)
    tokens, _ = apply_repeats(per_slot)
    return GriddedChart(
        title=chart.get("title") or chart.get("file", ""),
        tokens=tokens, slots_per_bar=slots_per_bar, beats_per_bar=beats_per_bar,
        sections=labels, key=chart.get("keyName"), style=None,
        n_unmapped=n_unmapped, n_symbols=n_symbols,
    )


def load_app_charts(state_dir: Path | str = "harmonia_min/state/charts",
                    *, slots_per_bar: int = 2) -> list[GriddedChart]:
    out = []
    for p in sorted(Path(state_dir).glob("*.json")):
        try:
            out.append(app_chart_to_grid(json.loads(p.read_text()),
                                         slots_per_bar=slots_per_bar))
        except Exception as e:  # a malformed chart must not kill the batch
            print(f"  skip {p.name}: {type(e).__name__}: {e}")
    return out
