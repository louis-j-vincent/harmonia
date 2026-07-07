"""
Guitar Pro tab parser — alternative labeled-data source for Harmonia evaluation.

Parses .gp, .gpx, .gp5 files (and .gp3/.gp4) via PyGuitarPro and extracts chord
events as {label, start_s, end_s} dicts compatible with ChordChart.chords
(minus the duration_beats and confidence fields, which we don't have from tabs).

Usage::

    from harmonia.tab_parser import parse_guitar_pro
    events = parse_guitar_pro("MySong.gp5")
    for ev in events:
        print(f"{ev['start_s']:6.2f}s  {ev['label']}")

Chord name normalization
------------------------
Guitar Pro stores chord names as free-text strings entered by the tab author.
Common variants we normalise:

  Root spellings: Bb / Bb  →  A#  (we keep sharps as SEMITONE_NAMES uses)
                  Db→C#, Eb→D#, Fb→E, Gb→F#, Ab→G#, Cb→B
  Quality tokens (case-insensitive after root extraction):
    ""  / "M" / "maj" / "Δ" / "△"         →  maj
    "m" / "mi" / "min" / "-"               →  min
    "7"                                     →  7
    "maj7" / "M7" / "Δ7" / "△7" / "^7"    →  maj7
    "m7" / "mi7" / "min7" / "-7"           →  min7
    "dim" / "°" / "o"                      →  dim
    "dim7" / "°7" / "o7"                   →  °7
    "aug" / "+"                            →  aug
    "aug7" / "+7"                          →  aug7
    "m7b5" / "ø7" / "ø" / "h7"            →  ø7  (half-dim)
    "sus2"                                  →  sus2
    "sus4" / "sus"                          →  sus4
    "7sus4" / "7sus"                       →  7sus4
    "mMaj7" / "mM7" / "m^7" / "minMaj7"   →  mMaj7
    "maj9" / "M9" / "Δ9"                   →  maj9
    "min9" / "m9" / "-9"                   →  min9
    "9"                                     →  9
    "min11" / "m11" / "-11"               →  min11
    "maj13" / "M13"                        →  maj13
    "min13" / "m13"                        →  min13
    "13"                                    →  13

Anything unrecognised is left as-is (with a warning) so we don't silently discard
unusual chords from the data — the caller can filter or inspect the unknowns.

NOT handled (ambiguous or rare; document for future work):
- Add-9 / add4 / add11 chords — no clean mapping in current vocabulary
- Slash chords (C/G etc.) — root is taken as the chord root, bass ignored
- Altered dominants (7alt, 7#5, 7b9 etc.) — too many sub-variants to map
  reliably from free text without more context
- Chord diagram data (root/type/extension fields on Chord) — these exist in
  newer GP formats but are often incomplete or inconsistent with the name string,
  so we use the name string as the authoritative source
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import with clear error
# ---------------------------------------------------------------------------

def _require_guitarpro():
    try:
        import guitarpro
        return guitarpro
    except ImportError:
        raise ImportError(
            "PyGuitarPro is required to parse Guitar Pro files.\n"
            "Install it with:  pip install PyGuitarPro"
        ) from None


# ---------------------------------------------------------------------------
# Root normalisation: flat → sharp, using SEMITONE_NAMES ordering
# ---------------------------------------------------------------------------

# Map every flat spelling (and enharmonic oddities) to a sharp/natural name
_FLAT_TO_SHARP: dict[str, str] = {
    "Cb": "B",
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}

# All valid natural/sharp root names (from chord_vocabulary.SEMITONE_NAMES)
_NATURAL_ROOTS = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}


def _normalise_root(raw_root: str) -> str | None:
    """Return a canonical root name (C, C#, …, B) or None if unrecognised."""
    # Capitalise first letter to handle "c#", "db", etc.
    root = raw_root[0].upper() + raw_root[1:] if raw_root else raw_root
    if root in _FLAT_TO_SHARP:
        return _FLAT_TO_SHARP[root]
    if root in _NATURAL_ROOTS:
        return root
    return None


# ---------------------------------------------------------------------------
# Quality normalisation
# ---------------------------------------------------------------------------

# Map (lowercase stripped quality suffix) → Harmonia ChordQuality.value
# Ordered from longest to shortest so greedy matching works top-to-bottom.
_QUALITY_MAP: list[tuple[re.Pattern, str]] = [
    # --- Minor-major 7th ---
    (re.compile(r"^(mmaj7|mm7|mmaj7|minmaj7|m\^7|m△7)$", re.I), "mMaj7"),
    # --- Augmented major 7th ---
    (re.compile(r"^(augmaj7|\+maj7|\+\^7|\+△7)$", re.I), "augMaj7"),
    # --- Augmented 7 ---
    (re.compile(r"^(aug7|\+7|7\+|7#5)$", re.I), "aug7"),
    # --- Half-diminished ---
    (re.compile(r"^(m7b5|ø7|ø|h7|halfdim7|halfdim|ø7)$", re.I), "ø7"),
    # --- Fully diminished 7 ---
    (re.compile(r"^(dim7|°7|o7|°7)$", re.I), "°7"),
    # --- 7sus4 ---
    (re.compile(r"^(7sus4|7sus)$", re.I), "7sus4"),
    # --- Major 7 ---
    (re.compile(r"^(maj7|M7|△7|Δ7|\^7|△maj7)$"), "maj7"),
    # --- Minor 7 ---
    (re.compile(r"^(min7|mi7|m7|-7)$", re.I), "min7"),
    # --- Dominant 7 ---
    (re.compile(r"^7$"), "7"),
    # --- Sus2 ---
    (re.compile(r"^sus2$", re.I), "sus2"),
    # --- Sus4 ---
    (re.compile(r"^(sus4|sus)$", re.I), "sus4"),
    # --- Diminished triad ---
    (re.compile(r"^(dim|°|o)$", re.I), "dim"),
    # --- Augmented triad ---
    (re.compile(r"^(aug|\+)$", re.I), "aug"),
    # --- Major 9 ---
    (re.compile(r"^(maj9|M9|△9|Δ9|\^9)$"), "maj9"),
    # --- Minor 9 ---
    (re.compile(r"^(min9|mi9|m9|-9)$", re.I), "min9"),
    # --- Dominant 9 ---
    (re.compile(r"^9$"), "9"),
    # --- Minor 11 ---
    (re.compile(r"^(min11|mi11|m11|-11)$", re.I), "min11"),
    # --- Major 13 ---
    (re.compile(r"^(maj13|M13)$"), "maj13"),
    # --- Minor 13 ---
    (re.compile(r"^(min13|mi13|m13|-13)$", re.I), "min13"),
    # --- Dominant 13 ---
    (re.compile(r"^13$"), "13"),
    # --- Minor triad (after all m+X patterns exhausted) ---
    (re.compile(r"^(min|mi|m|-)$", re.I), "min"),
    # --- Major triad (empty string, or explicit major marker) ---
    (re.compile(r"^(maj|M|Δ|△|)?$"), "maj"),
]


def _normalise_quality(raw_quality: str) -> str:
    """Map a raw quality suffix to a Harmonia quality value string.

    Returns the raw quality unchanged (with a warning) if not recognised.
    """
    q = raw_quality.strip()
    for pattern, canonical in _QUALITY_MAP:
        if pattern.match(q):
            return canonical
    warnings.warn(f"tab_parser: unrecognised chord quality {raw_quality!r} — kept as-is")
    return raw_quality


# ---------------------------------------------------------------------------
# Full chord name parsing: "Am7" → ("A", "min7") → "Amin7"
# ---------------------------------------------------------------------------

# Regex: root (letter + optional # or b) then optional quality suffix
# Also strip slash-chord bass note (C/G → C)
_CHORD_RE = re.compile(
    r"^([A-Ga-g][#b]?)"  # root: note letter + optional sharp or flat
    r"(.*?)"              # quality: everything after root
    r"(?:/[A-Ga-g][#b]?)?$"  # optional slash bass note, discarded
)


def normalise_chord_name(raw: str) -> str | None:
    """
    Normalise a raw chord name string to Harmonia label format.

    Returns None for empty strings or strings that can't be parsed as chords.
    Returns the normalised label (e.g. "Am7" → "Amin7", "Gbmaj7" → "F#maj7").

    Note: Harmonia labels are <Root><quality.value>, e.g. "Cmaj", "Amin7",
    "F#7", "Bb" is written "A#maj" (sharps only).
    """
    raw = raw.strip()
    if not raw or raw.upper() in ("N", "NC", "N.C.", "N/C", "NO CHORD", "-"):
        return "N"

    m = _CHORD_RE.match(raw)
    if not m:
        warnings.warn(f"tab_parser: cannot parse chord name {raw!r} — skipped")
        return None

    raw_root, raw_quality = m.group(1), m.group(2)

    root = _normalise_root(raw_root)
    if root is None:
        warnings.warn(f"tab_parser: unrecognised root {raw_root!r} in {raw!r} — skipped")
        return None

    quality = _normalise_quality(raw_quality)

    # Harmonia label convention: <Root><quality.value>
    # For major triad, quality.value is "maj" → label is e.g. "Cmaj"
    return f"{root}{quality}"


# ---------------------------------------------------------------------------
# Tick-to-seconds conversion with tempo-change tracking
# ---------------------------------------------------------------------------

_QUARTER_TIME = 960  # Duration.quarterTime — ticks per quarter note


def _ticks_to_seconds(
    ticks: int,
    tempo_map: list[tuple[int, float]],
) -> float:
    """Convert an absolute tick position to seconds using a tempo map.

    tempo_map: sorted list of (tick_offset, bpm) pairs — first entry must be (0, initial_bpm).
    """
    t_s = 0.0
    for i, (tick_start, bpm) in enumerate(tempo_map):
        tick_end = tempo_map[i + 1][0] if i + 1 < len(tempo_map) else None
        if tick_end is not None and ticks >= tick_end:
            # Accumulate full segment
            t_s += (tick_end - tick_start) / _QUARTER_TIME * (60.0 / bpm)
        else:
            # Ticks lands in this segment
            t_s += (ticks - tick_start) / _QUARTER_TIME * (60.0 / bpm)
            break
    return t_s


def _build_tempo_map(song) -> list[tuple[int, float]]:
    """Walk all measures/beats and collect (absolute_tick, bpm) change points.

    Guitar Pro stores tempo changes in BeatEffect.mixTableChange.tempo on
    individual beats. We also honour the song-level default tempo.
    """
    tempo_map: list[tuple[int, float]] = [(0, float(song.tempo))]

    for track in song.tracks[:1]:  # one track is enough — tempo is global
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    if beat.effect.mixTableChange is not None:
                        mtc = beat.effect.mixTableChange
                        if mtc.tempo is not None:
                            new_bpm = float(mtc.tempo.value)
                            if beat.start is not None:
                                # Avoid duplicate ticks from multiple voices
                                if not tempo_map or tempo_map[-1][0] != beat.start:
                                    tempo_map.append((beat.start, new_bpm))

    tempo_map.sort(key=lambda x: x[0])
    return tempo_map


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_guitar_pro(path: str | Path) -> list[dict]:
    """Parse a Guitar Pro file and return a list of chord events.

    Each event is a dict with keys:
        label    (str)   — normalised chord label, e.g. "Amin7", "F#maj7", "N"
        start_s  (float) — chord onset in seconds
        end_s    (float) — chord offset in seconds (= onset of next chord event)
        raw_name (str)   — original chord name from the tab (for debugging)
        track    (int)   — 1-indexed track number the chord was found on
        measure  (int)   — 1-indexed measure number

    Strategy
    --------
    Guitar Pro attaches chord diagrams to beats via beat.effect.chord.  We walk
    every track, every measure, every voice (voice 0 = main), every beat and
    collect (absolute_tick, chord_name) pairs.  We then deduplicate (same chord
    persisting across adjacent beats) and convert ticks → seconds.

    Only beats where beat.effect.isChord is True carry a chord annotation;
    gaps (no chord on a beat) inherit the most recent chord (Guitar Pro's
    display convention: a chord box applies until the next chord box).

    If multiple tracks annotate different chords at the same tick we report
    the first track's annotation.  In practice only the chord-melody /
    rhythm track in a GP file has chord boxes.

    Raises
    ------
    ImportError  — if PyGuitarPro is not installed
    FileNotFoundError — if path doesn't exist
    guitarpro.GPException — if the file can't be parsed
    """
    guitarpro = _require_guitarpro()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Guitar Pro file not found: {path}")

    logger.info(f"tab_parser: parsing {path.name}")
    song = guitarpro.parse(str(path))
    logger.info(
        f"  {song.title!r}  tempo={song.tempo} BPM  "
        f"{len(song.measureHeaders)} measures  {len(song.tracks)} tracks"
    )

    tempo_map = _build_tempo_map(song)
    logger.debug(f"  tempo map: {tempo_map}")

    # Collect raw chord events: (tick, raw_name, track_idx, measure_idx)
    # keyed by tick — first track wins
    raw_events: dict[int, tuple[str, int, int]] = {}

    for track_idx, track in enumerate(song.tracks):
        for measure in track.measures:
            # Walk voice 0 (primary voice) — voice 1 is usually decorative
            voice = measure.voices[0]
            active_chord: str | None = None
            for beat in voice.beats:
                if beat.effect.isChord and beat.effect.chord.name:
                    active_chord = beat.effect.chord.name
                if active_chord is not None and beat.start is not None:
                    tick = beat.start
                    if tick not in raw_events:
                        raw_events[tick] = (
                            active_chord, track_idx + 1, measure.number
                        )

    if not raw_events:
        logger.warning("tab_parser: no chord annotations found in file")
        return []

    # Sort by tick and compute timing
    sorted_ticks = sorted(raw_events.keys())

    # Compute total duration: last beat start + last beat's duration
    # Use the last measure's end tick as a reasonable approximation
    last_measure_end = song.measureHeaders[-1].end if song.measureHeaders else sorted_ticks[-1] + _QUARTER_TIME * 4
    total_ticks = max(last_measure_end, sorted_ticks[-1] + _QUARTER_TIME)
    total_s = _ticks_to_seconds(total_ticks, tempo_map)

    events: list[dict] = []
    prev_label: str | None = None

    for i, tick in enumerate(sorted_ticks):
        raw_name, track_num, measure_num = raw_events[tick]
        label = normalise_chord_name(raw_name)
        if label is None:
            continue  # unparseable — skip (warning already emitted)

        start_s = _ticks_to_seconds(tick, tempo_map)
        end_tick = sorted_ticks[i + 1] if i + 1 < len(sorted_ticks) else total_ticks
        end_s = _ticks_to_seconds(end_tick, tempo_map)

        # Collapse consecutive identical chords (tab authors sometimes annotate
        # every repeat of a chord pattern separately)
        if events and events[-1]["label"] == label and abs(events[-1]["end_s"] - start_s) < 0.01:
            events[-1]["end_s"] = end_s
            continue

        events.append({
            "label": label,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "raw_name": raw_name,
            "track": track_num,
            "measure": measure_num,
        })
        prev_label = label

    logger.info(f"  {len(events)} chord events extracted")
    return events


# ---------------------------------------------------------------------------
# Utility: print a chord timeline (mirrors ChordChart.print())
# ---------------------------------------------------------------------------

def print_chord_timeline(events: list[dict], title: str = "") -> None:
    """Print a formatted chord timeline to stdout."""
    if title:
        print(f"\n{'━'*60}")
        print(f"  {title}")
    else:
        print(f"\n{'━'*60}")
    print(f"  {len(events)} chord events")
    print(f"{'━'*60}")
    print(f"  {'CHORD':<12} {'START':>7}  {'END':>7}  {'RAW NAME'}")
    print(f"  {'─'*50}")
    for ev in events:
        print(
            f"  {ev['label']:<12} {ev['start_s']:>7.2f}  {ev['end_s']:>7.2f}  "
            f"{ev.get('raw_name', '')}"
        )
    print(f"{'━'*60}\n")
