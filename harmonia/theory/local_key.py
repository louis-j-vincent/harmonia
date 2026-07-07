"""Symbolic local-key estimation for chord charts.

Given the chords in a section (as iReal tokens), accumulate a pitch-class
histogram from the chord tones and run the existing Krumhansl key matcher
(``key_profiles.infer_key``) to label the section's key/scale. Used to colour-
code which sections of a chart sit in which key.

Deliberately symbolic (chart-only, no audio): the chart is a clean chord
sequence, so chord tones are exact and a Krumhansl match on them is a solid,
cheap key estimate — no need to touch the noisy audio chroma here.
"""

from __future__ import annotations

import re

import numpy as np

from .key_profiles import infer_key

# conventional key-name spelling (flats for flat-side keys, etc.)
_MAJOR_NAMES = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "Gb",
                7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
_MINOR_NAMES = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#",
                7: "G", 8: "G#", 9: "A", 10: "Bb", 11: "B"}


def key_name(tonic: int, mode: str) -> str:
    names = _MAJOR_NAMES if mode == "major" else _MINOR_NAMES
    return f"{names[tonic % 12]} {mode}"


_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"b": -1, "#": 1, "": 0}
_TOKEN_RE = re.compile(r"^([A-G])([b#]?)(.*)$")


_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_FLAT_MAJ_TONICS = {0, 1, 3, 5, 6, 8, 10}


def prefer_flats(tonic: int, mode: str) -> bool:
    """Should this key be spelled with flats? (relative-major convention)."""
    maj = tonic if mode == "major" else (tonic + 3) % 12
    return maj % 12 in _FLAT_MAJ_TONICS


def transpose_token(token: str, semitones: int, flats: bool) -> str:
    """Transpose an iReal token by ``semitones``, spelling roots with flats or
    sharps. Quality tail is unchanged; a slash bass is transposed too."""
    root, q, bass = parse_token(token)
    names = _FLAT_NAMES if flats else _SHARP_NAMES
    out = names[(root + semitones) % 12] + q
    if bass is not None:
        out += "/" + names[(bass + semitones) % 12]
    return out


def parse_token(token: str) -> tuple[int, str, int | None]:
    """iReal token → (root_pc, quality_tail, bass_pc | None)."""
    head, _, bass = token.partition("/")
    m = _TOKEN_RE.match(head.strip())
    if not m:
        return 0, "", None
    letter, acc, qual = m.groups()
    root = (_LETTER[letter] + _ACC[acc]) % 12
    bass_pc = None
    if bass:
        bm = _TOKEN_RE.match(bass.strip())
        if bm:
            bass_pc = (_LETTER[bm.group(1)] + _ACC[bm.group(2)]) % 12
    return root, qual, bass_pc


def chord_pcs(token: str) -> dict[int, float]:
    """Chord tones of an iReal token as {pitch_class: weight}. Root and third
    weighted highest (they pin the key); 7th/extensions add lighter colour."""
    root, q, bass = parse_token(token)
    ivs: dict[int, float] = {0: 2.0}

    # third + fifth from the triad quality
    if q.startswith("sus"):
        ivs[5] = 1.0 if "2" not in q else 0.0
        ivs[2 if "2" in q else 7] = 1.0
        ivs[7] = 1.0
    else:
        if q[:1] in ("-", "h") or q.startswith("o") or q.startswith("dim") or q.startswith("m"):
            ivs[3] = 1.5                      # minor third
        elif q.startswith("+") or q.startswith("aug"):
            ivs[4] = 1.5
        else:
            ivs[4] = 1.5                      # major third
        if q.startswith(("o", "dim", "h")) or "b5" in q:
            ivs[6] = 1.0
        elif q.startswith(("+", "aug")) or "#5" in q:
            ivs[8] = 1.0
        else:
            ivs[7] = 1.0

    # seventh / sixth
    if "^" in q or "maj7" in q or "M7" in q:
        ivs[11] = 0.8
    elif q.startswith(("o", "dim")) and "7" in q:
        ivs[9] = 0.8                          # dim7 → diminished 7th
    elif "6" in q and "b6" not in q:
        ivs[9] = 0.8
    elif "7" in q or q.startswith(("-", "h", "m")):
        ivs.setdefault(10, 0.8)

    out: dict[int, float] = {}
    for iv, w in ivs.items():
        out[(root + iv) % 12] = out.get((root + iv) % 12, 0.0) + w
    if bass is not None:
        out[bass] = out.get(bass, 0.0) + 1.0
    return out


def estimate_key(tokens: list[str], weights: list[float] | None = None) -> dict:
    """Estimate the key/scale of a chord run. Returns
    {tonic, mode, name, conf}. ``weights`` (e.g. chord durations) default to 1."""
    weights = weights or [1.0] * len(tokens)
    chroma = np.zeros(12)
    for tok, w in zip(tokens, weights):
        for pc, cw in chord_pcs(tok).items():
            chroma[pc] += cw * w
    # tonic cue: tonal phrases tend to begin and (especially) end on the tonic,
    # which disambiguates a key from its dominant / relative (both share notes).
    if tokens:
        chroma[parse_token(tokens[-1])[0]] += 3.0
        chroma[parse_token(tokens[0])[0]] += 1.5
    if chroma.sum() == 0:
        return {"tonic": 0, "mode": "major", "name": "C major", "conf": 0.0}
    kp = infer_key(chroma)
    return {"tonic": kp.tonic, "mode": kp.mode,
            "name": key_name(kp.tonic, kp.mode), "conf": kp.confidence}


def section_keys(chords: list[dict], section_per_bar: list[str]) -> dict[str, dict]:
    """One key estimate per distinct section label, from all its chords.

    ``chords`` items need ``bar`` and ``symbol`` (iReal token). Returns
    {section_label: {tonic, mode, name, conf}}.
    """
    by_label: dict[str, list[str]] = {}
    for c in chords:
        bar = c["bar"]
        lab = section_per_bar[bar] if 0 <= bar < len(section_per_bar) else "?"
        by_label.setdefault(lab, []).append(c["symbol"])
    return {lab: estimate_key(toks) for lab, toks in by_label.items()}
