"""irealb_export.py — export a Harmonia ChartModel to an irealb:// URL (iReal Pro).

Reverse direction of irealb_fetcher.py (which only imports FROM iReal — this
project never needed to write the format until the app's "Share" button,
2026-07-20). iReal Pro's per-song obfuscation is a pure character-swap
involution (blocks of 50, first/last 5 and positions 10-23 swapped with their
mirror) — pyRealParser's ``Tune._unscramble_chord_string`` applies exactly
that transform, so calling it on PLAIN text scrambles it (verified by
round-trip: unscramble(unscramble(x)) == x for any x, since swaps are self-
inverse and the chunking is length-based, hence identical in both directions).
There is deliberately no separate "scramble" function — reusing the existing
one means the two directions can never drift apart.
"""
from __future__ import annotations

import urllib.parse

from pyRealParser.pyRealParser import Tune

NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _key_string(tonic_pc: int, mode: str) -> str:
    return NOTE_SHARP[tonic_pc % 12] + ("-" if mode == "minor" else "")


def _bar_token(bar: list[dict], is_first_of_section: bool, section_letter: str) -> str:
    """One bar's iReal token: a section marker prefix (only the section's
    first bar), then the chord(s) — 'x' for a held bar (repeat previous),
    'n' for an explicit no-chord bar, or 'Root'+iReal-quality-token per chord
    (space-separated for a split/2-chord bar)."""
    prefix = f"*{section_letter}" if is_first_of_section else ""
    if not bar:
        return prefix + "x"
    if len(bar) == 1 and (bar[0].get("q") or "") == "N":
        return prefix + "n"
    toks = [NOTE_SHARP[c["root"] % 12] + (c.get("q") or "") for c in bar]
    return prefix + " ".join(toks)


def chart_model_to_irealb_url(
    model: dict, *, composer: str = "Harmonia", style: str = "Medium Swing",
) -> str:
    """``model``: a ChartModel dict (harmonia.output.chart_model.to_chart_model).

    Returns a single-tune ``irealb://`` URL that iReal Pro can import directly
    (paste into the app, or open the link on a device with iReal Pro
    installed). Folded repeats (a section rendered once but played ``reps``
    times) are unrolled into the literal bar sequence — iReal's own repeat-
    bracket notation would be a nicer read, but a flat, faithful sequence is
    simpler and never round-trips incorrectly.
    """
    title = model.get("title") or "Untitled"
    key = _key_string(model["key"]["tonic"], model["key"]["mode"])

    bar_tokens: list[str] = ["T44"]
    for sec in model["sections"]:
        letter = (sec.get("label") or "A")[0]
        for _rep in range(max(1, sec.get("reps", 1))):
            for i, bar in enumerate(sec["bars"]):
                bar_tokens.append(_bar_token(bar, i == 0, letter))
    chord_string = "|".join(bar_tokens) + "|Z"

    scrambled = Tune._unscramble_chord_string(chord_string)
    fields = [title, composer, style, key, Tune._chords_prefix + scrambled]
    raw = "=".join(fields) + "="
    return "irealb://" + urllib.parse.quote(raw, safe="=")
