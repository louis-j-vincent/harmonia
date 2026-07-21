"""Billboard McGill dataset — chord notation translator to Harmonia format."""

from __future__ import annotations

import re
from typing import Optional, Tuple

_NOTE_TO_PC = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

_PC_TO_NOTE = {v: k for k, v in _NOTE_TO_PC.items()}


def note_to_pitch_class(note: str) -> Optional[int]:
    """Convert note name (C, C#, Db, F#, etc.) to pitch class (0-11)."""
    if not note or note == "N":
        return None

    base = note[0].upper()
    if base not in _NOTE_TO_PC:
        return None

    pc = _NOTE_TO_PC[base]

    if len(note) > 1:
        acc = note[1:].lower()
        if acc == "#":
            pc = (pc + 1) % 12
        elif acc == "b":
            pc = (pc - 1) % 12
        else:
            return None

    return pc


BILLBOARD_TO_Q5 = {
    "maj": "maj", "maj7": "maj", "maj6": "maj", "maj9": "maj",
    "maj13": "maj", "add9": "maj", "add11": "maj", "add2": "maj",
    "aug": "maj", "augmaj7": "maj", "": "maj",
    "min": "min", "min7": "min", "min6": "min", "min9": "min",
    "min13": "min", "minmaj7": "min", "-": "min",
    "7": "dom", "9": "dom", "11": "dom", "13": "dom",
    "7b9": "dom", "7#9": "dom", "7b5": "dom", "7#5": "dom",
    "7alt": "dom", "9b5": "dom", "9#5": "dom", "13b9": "dom",
    "13#9": "dom", "m7b5": "hdim", "hdim": "hdim", "hdim7": "hdim",
    "dim": "dim", "dim7": "dim", "sus2": "maj", "sus4": "maj",
    "7sus4": "dom", "7sus": "dom", "7susb9": "dom", "N": None,
}


def parse_billboard_chord(label: str) -> Tuple[Optional[int], Optional[str]]:
    """Convert Billboard chord notation to Harmonia format.
    
    Args:
        label: Billboard chord label (e.g., "C:maj", "Bb:7", "N")
    
    Returns:
        Tuple of (root_pitch_class: 0-11 or None, quality or None)
    
    Examples:
        >>> parse_billboard_chord("C:maj")
        (0, 'maj')
        >>> parse_billboard_chord("Bb:min7")
        (10, 'min')
        >>> parse_billboard_chord("N")
        (None, None)
    """
    label = label.strip()

    if label == "N" or label == "":
        return (None, None)

    parts = label.split(":")
    if len(parts) != 2:
        return (None, None)

    note_str, quality_str = parts
    note_str = note_str.strip()
    quality_str = quality_str.strip()

    root = note_to_pitch_class(note_str)
    if root is None:
        return (None, None)

    q5_quality = BILLBOARD_TO_Q5.get(quality_str)
    if q5_quality is None:
        return (None, None)

    return (root, q5_quality)


def billboard_chord_list_to_harmonia(billboard_chords) -> list[dict]:
    """Convert mirdata Billboard ChordData to Harmonia ground truth format."""
    result = []

    for i, (start, end) in enumerate(billboard_chords.intervals):
        label = billboard_chords.labels[i]
        root, quality = parse_billboard_chord(label)

        result.append(
            {
                "t0": float(start),
                "t1": float(end),
                "root": root,
                "quality": quality,
                "label": label,
                "source": "billboard",
            }
        )

    return result


def count_unmapped_qualities(track_chords) -> dict[str, int]:
    """Count Billboard qualities that don't map to Harmonia Q5."""
    unmapped = {}
    for label in track_chords.labels:
        if label == "N":
            continue
        parts = label.split(":")
        if len(parts) != 2:
            continue
        quality_str = parts[1].strip()
        if quality_str not in BILLBOARD_TO_Q5:
            unmapped[quality_str] = unmapped.get(quality_str, 0) + 1
    return unmapped
