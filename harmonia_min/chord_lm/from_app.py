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


def _lay_out_bars(chart: dict) -> tuple[list[list[dict] | None], list[str]]:
    """Absolute per-bar chord lists over the whole song, repetitions EXPANDED.

    Display folding (shipped 2026-08-02) stores a section once as its repeating
    TEMPLATE plus the bar ranges where it recurs: Let It Be became one section
    with `reps: 3`, `barRanges [[0,35],[36,55],[56,69]]` and four bars in
    `bars`. Emitting each section's bars once — correct before folding, when
    every bar was written out — silently yielded a 4-bar song instead of a
    70-bar one, and the LM lost the very repetition it exists to exploit.

    So occurrences are expanded here: the template is laid down at each range's
    start and cycled to fill it. The token stream is then the chart AS HEARD,
    on the same bar indices as `barGrid`, which is also what makes the
    timestamps in a suggestion correct.

    Known loss: the chart keeps only the template, so genuine per-occurrence
    variation (different endings, a turnaround on the last pass) is reproduced
    as the template. That is a property of the stored chart, not of this code.
    """
    n_bars = int(chart.get("nBars") or 0)
    sections = sorted(chart.get("sections", []),
                      key=lambda s: (s.get("barRanges") or [[10 ** 9]])[0][0])
    if not n_bars:
        n_bars = sum(len(s.get("bars", [])) for s in sections)
    out: list[list[dict] | None] = [None] * n_bars
    labels: list[str] = ["?"] * n_bars
    for sec in sections:
        tmpl = sec.get("bars", []) or []
        if not tmpl:
            continue
        label = sec.get("label") or sec.get("id") or "?"
        ranges = sec.get("barRanges") or [[0, len(tmpl) - 1]]
        for rng in ranges:
            b0, b1 = int(rng[0]), int(rng[1])
            for k, b in enumerate(range(b0, min(b1, n_bars - 1) + 1)):
                if 0 <= b < n_bars:
                    out[b] = tmpl[k % len(tmpl)]
                    labels[b] = label
    return out, labels


def app_chart_to_grid(chart: dict, *, slots_per_bar: int = 2) -> GriddedChart:
    """A harmonia_min chart dict -> GriddedChart, repetitions expanded.

    Bar indices match `chart["barGrid"]`, so slot i sits at bar `i //
    slots_per_bar` of the real timeline.
    """
    beats_per_bar = int(chart.get("bpb", 4) or 4)
    laid, labels = _lay_out_bars(chart)
    per_slot: list[int | None] = []
    carry: int | None = None
    n_unmapped = n_symbols = 0
    for bar in laid:
        bar = bar or []
        n_symbols += len(bar)
        slots, carry, bad = _bar_to_slots(bar, slots_per_bar, beats_per_bar, carry)
        n_unmapped += bad
        per_slot.extend(slots)
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
