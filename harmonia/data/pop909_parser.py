"""
POP909 dataset parser.

POP909 (Wang et al., 2020) contains 909 Chinese and Western pop songs with:
  - Piano arrangement MIDI (melody + bridge + accompaniment tracks)
  - Beat annotations (.beat files)
  - Chord annotations (.chord files) — our primary label source

Dataset: https://github.com/music-x-lab/POP909-Dataset
Paper:   https://arxiv.org/abs/2008.07142

Label format (chord files):
    Each line: <start_beat> <end_beat> <chord_label>
    Chord labels follow the standard Harte notation:
        C:maj, G:min7, Bb:dom7, N (no chord), etc.

This parser converts POP909 chord labels → our ChordQuality taxonomy
and aligns them with beat times for training data construction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harmonia.theory.chord_vocabulary import ChordQuality, SEMITONE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pitch class parsing
# ---------------------------------------------------------------------------

_NOTE_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

def parse_root(root_str: str) -> int | None:
    """Parse a root note string to pitch class (0–11). Returns None on failure."""
    return _NOTE_TO_PC.get(root_str)


# ---------------------------------------------------------------------------
# Chord quality string → ChordQuality
# ---------------------------------------------------------------------------

_QUALITY_MAP: dict[str, ChordQuality] = {
    # Triads
    "maj":      ChordQuality.MAJOR,
    "min":      ChordQuality.MINOR,
    "dim":      ChordQuality.DIMINISHED,
    "aug":      ChordQuality.AUGMENTED,
    "sus2":     ChordQuality.SUS2,
    "sus4":     ChordQuality.SUS4,
    # 7ths
    "maj7":     ChordQuality.MAJ7,
    "min7":     ChordQuality.MIN7,
    "7":        ChordQuality.DOM7,
    "dom7":     ChordQuality.DOM7,
    "minmaj7":  ChordQuality.MIN_MAJ7,
    "hdim7":    ChordQuality.HALF_DIM7,
    "dim7":     ChordQuality.DIM7,
    "aug7":     ChordQuality.AUG7,
    "7sus4":    ChordQuality.DOM7SUS4,
    # 9ths
    "maj9":     ChordQuality.MAJ9,
    "min9":     ChordQuality.MIN9,
    "9":        ChordQuality.DOM9,
    "7b9":      ChordQuality.DOM7B9,
    "7#9":      ChordQuality.DOM7S9,
    # Aliases common in pop annotations
    "maj6":     ChordQuality.MAJOR,    # approximate: treat as major triad
    "min6":     ChordQuality.MINOR,
    "1":        ChordQuality.MAJOR,    # sometimes used for bare root
}

def parse_quality(quality_str: str) -> ChordQuality | None:
    """Map a Harte-notation quality string to ChordQuality. None if unknown."""
    return _QUALITY_MAP.get(quality_str.lower())


# ---------------------------------------------------------------------------
# Harte chord label parser
# ---------------------------------------------------------------------------

_HARTE_RE = re.compile(
    r"^(?P<root>[A-Ga-g][b#]?)"       # root: e.g. C, Bb, F#
    r"(?::(?P<quality>[^/\(]+))?",     # optional :quality (no slash or parens)
)

def parse_harte_label(label: str) -> tuple[int, ChordQuality] | None:
    """
    Parse a Harte-notation chord label into (root_pc, ChordQuality).

    Examples:
        "C:maj"    → (0, MAJOR)
        "Bb:min7"  → (10, MIN7)
        "G:7"      → (7, DOM7)
        "N"        → (-1, NO_CHORD)
        "X"        → None  (unknown, skip)

    Bass inversions (/bass_note) are ignored — we model root position chords.
    """
    label = label.strip()

    if label in ("N", "n", "NC", "no_chord"):
        return (-1, ChordQuality.NO_CHORD)

    if label in ("X", "x", "?"):
        return None  # ambiguous, skip

    m = _HARTE_RE.match(label)
    if not m:
        logger.debug(f"Unparseable chord label: {label!r}")
        return None

    root_str = m.group("root").capitalize()
    quality_str = (m.group("quality") or "maj").lower()

    root_pc = parse_root(root_str)
    if root_pc is None:
        logger.debug(f"Unknown root: {root_str!r}")
        return None

    quality = parse_quality(quality_str)
    if quality is None:
        # Fall back: if we can identify root but not quality, return major
        logger.debug(f"Unknown quality {quality_str!r} in {label!r}, defaulting to maj")
        quality = ChordQuality.MAJOR

    return (root_pc, quality)


# ---------------------------------------------------------------------------
# POP909 song record
# ---------------------------------------------------------------------------

@dataclass
class ChordEvent:
    start_beat: float
    end_beat: float
    root: int               # 0–11, or -1 for N
    quality: ChordQuality
    label: str              # original string

    @property
    def is_no_chord(self) -> bool:
        return self.quality == ChordQuality.NO_CHORD

    def duration_beats(self) -> float:
        return self.end_beat - self.start_beat


@dataclass
class POP909Song:
    song_id: str
    chord_events: list[ChordEvent]
    beat_times: np.ndarray      # shape (B,) — time in seconds of each beat
    midi_path: Path
    audio_path: Path | None     # may not exist if only MIDI available

    @property
    def n_chords(self) -> int:
        return len(self.chord_events)

    def chord_at_beat(self, beat: float) -> ChordEvent | None:
        """Return the chord event active at the given beat number."""
        for ev in self.chord_events:
            if ev.start_beat <= beat < ev.end_beat:
                return ev
        return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class POP909Parser:
    """
    Parses the POP909 dataset directory structure.

    Expected layout:
        pop909/
          001/
            001.mid
            chord_midi.txt    ← chord annotations (beat-aligned)
            beat_midi.txt     ← beat times in seconds
          002/
            ...
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"POP909 directory not found: {self.dataset_dir}\n"
                "Download from: https://github.com/music-x-lab/POP909-Dataset"
            )

    def _parse_beat_file(self, beat_path: Path) -> np.ndarray:
        """Parse beat file → array of beat times in seconds."""
        times = []
        with open(beat_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        times.append(float(parts[0]))
                    except ValueError:
                        pass
        return np.array(times)

    def _parse_chord_file(self, chord_path: Path) -> list[ChordEvent]:
        """Parse chord annotation file → list of ChordEvents."""
        events: list[ChordEvent] = []
        with open(chord_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    start_beat = float(parts[0])
                    end_beat = float(parts[1])
                    label = parts[2]
                except ValueError:
                    continue

                parsed = parse_harte_label(label)
                if parsed is None:
                    continue

                root, quality = parsed
                events.append(ChordEvent(
                    start_beat=start_beat,
                    end_beat=end_beat,
                    root=root,
                    quality=quality,
                    label=label,
                ))

        return events

    def parse_song(self, song_id: str) -> POP909Song | None:
        """Parse a single song by ID (e.g. '001')."""
        song_dir = self.dataset_dir / song_id

        midi_path = song_dir / f"{song_id}.mid"
        chord_path = song_dir / "chord_midi.txt"
        beat_path = song_dir / "beat_midi.txt"

        if not midi_path.exists():
            logger.warning(f"MIDI not found: {midi_path}")
            return None
        if not chord_path.exists():
            logger.warning(f"Chord file not found: {chord_path}")
            return None

        beat_times = np.array([])
        if beat_path.exists():
            beat_times = self._parse_beat_file(beat_path)

        chord_events = self._parse_chord_file(chord_path)

        # Try to find rendered audio (user may have rendered MIDI → WAV)
        audio_path = None
        for ext in [".wav", ".mp3", ".flac"]:
            candidate = song_dir / f"{song_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        return POP909Song(
            song_id=song_id,
            chord_events=chord_events,
            beat_times=beat_times,
            midi_path=midi_path,
            audio_path=audio_path,
        )

    def parse_all(
        self,
        max_songs: int | None = None,
        require_audio: bool = False,
    ) -> list[POP909Song]:
        """
        Parse all songs in the dataset.

        Args:
            max_songs:     limit number of songs (useful for quick testing).
            require_audio: if True, skip songs without rendered audio.

        Returns:
            List of POP909Song objects.
        """
        song_dirs = sorted(
            d for d in self.dataset_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

        if max_songs is not None:
            song_dirs = song_dirs[:max_songs]

        songs: list[POP909Song] = []
        for song_dir in song_dirs:
            song = self.parse_song(song_dir.name)
            if song is None:
                continue
            if require_audio and song.audio_path is None:
                logger.debug(f"Skipping {song.song_id} (no audio)")
                continue
            songs.append(song)

        logger.info(f"Parsed {len(songs)} POP909 songs.")
        return songs

    def chord_statistics(self, songs: list[POP909Song]) -> dict:
        """Compute chord quality distribution across the dataset."""
        from collections import Counter
        counter: Counter = Counter()
        for song in songs:
            for ev in song.chord_events:
                counter[ev.quality.value] += 1
        total = sum(counter.values())
        return {k: v / total for k, v in counter.most_common()}
