"""iReal Pro chart corpus → structured chord charts → MMA accompaniment input.

Pipeline (validated end-to-end 2026-07-03, see docs/accompaniment_db_2026-07-03.md):

    iReal playlist file (irealb:// URLs)
      → pyRealParser.Tune                     (chords, style, key, time signature)
      → sectionized_measures()                (per-bar section labels survive flattening)
      → split_chords() + to_mma_chord()       (iReal tokens → MMA chord names)
      → tune_to_mma()                         (.mma chart: groove, tempo, per-beat chords)

The .mma file is rendered to multi-track MIDI by MMA (Musical MIDI Accompaniment),
which emits *named* tracks (Bass, Chord, Drum, …) — so the bass line is extractable
by construction, not by inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pyRealParser import Tune
from pyRealParser.pyRealParser import Tune as _TuneCls

# ── iReal quality → MMA quality ────────────────────────────────────────────────
# MMA's chord table (MMA/chordtable.py) has 163 qualities; this covers every
# quality observed in the jazz1460 / pop400 / blues50 corpora. Unmapped tokens
# are reported by tune_to_mma, not silently dropped.
IREAL_TO_MMA: dict[str, str] = {
    "": "", "5": "5", "2": "(add9)", "add9": "(add9)",
    "^": "M7", "^7": "M7", "^9": "M9", "^13": "M13",
    "^7#11": "M7#11", "^9#11": "M9#11", "^7#5": "M7#5",
    "6": "6", "69": "69",
    "-": "m", "-6": "m6", "-69": "m69", "-7": "m7", "-9": "m9", "-11": "m11",
    "-b6": "m#5", "-#5": "m#5", "-^7": "mM7", "-^9": "mM7(add9)",
    "-7b5": "m7b5", "-7#5": "m7#5",
    "h": "m7b5", "h7": "m7b5", "h9": "m9b5",
    "o": "dim", "o7": "dim7",
    "7": "7", "9": "9", "11": "11", "13": "13",
    "7b9": "7b9", "7#9": "7#9", "7#11": "7#11", "9#11": "9#11",
    "7b5": "7b5", "7#5": "7#5", "+": "aug", "7+": "7#5",
    "7b13": "7b13", "7#9#5": "7#5#9", "7b9b5": "7b5b9", "7b9#5": "7b9#5",
    "7b9#9": "7b9", "7b9b13": "7b9b13", "7b9#11": "7b9#11", "7#9b5": "7b5#9",
    "7#9#11": "7#9#11", "7alt": "7alt", "alt": "7alt",
    "13b9": "13b9", "13#9": "13#9", "13#11": "13#11",
    "9b5": "9b5", "9#5": "9#5",
    "sus": "sus4", "7sus": "7sus4", "9sus": "9sus4", "13sus": "13sus4",
    "7susb9": "7susb9", "7b9sus": "7b9sus",
    # pyRealParser's annotation cleanup strips single 'l's, mangling 'alt' → 'at'
    "at": "7alt", "7at": "7alt",
    "o^7": "dim7(addM7)", "7b13sus": "7sus4", "susadd3": "sus4", "7susadd3": "7sus4",
}

_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_CHORD_RE = re.compile(r"([A-G][b#]?)([^A-G]*?)(?:/([A-G][b#]?))?$")

# ── style → (MMA groove, default BPM) ──────────────────────────────────────────
# Ordered: first substring match wins. Time signature overrides (3/4 → waltz etc.)
# are applied before this table.
STYLE_GROOVES: list[tuple[str, str, int]] = [
    ("up tempo swing", "Swing", 220),
    ("medium up swing", "Swing", 180),
    ("medium swing", "Swing", 140),
    ("slow swing", "Swing", 100),
    ("ballad", "Ballad", 70),
    ("swing", "Swing", 140),
    ("bossa", "BossaNova", 130),
    ("samba", "Samba", 180),
    ("latin", "BossaNova", 140),
    ("even 8ths", "8Beat", 120),
    ("even 8th", "8Beat", 120),
    ("rock", "JazzRock", 120),
    ("rnb", "Blues", 100),
    ("funk", "8Beat", 105),
    ("shuffle", "ShuffleBoggie", 130),
    ("slow blues", "SlowBlues", 65),
    ("blues", "Blues", 110),
    ("afro", "Afro", 120),
    ("calypso", "BossaNova", 150),
    ("waltz", "JazzWaltz", 140),
]


@dataclass
class MMAChart:
    """A tune converted to MMA input, plus the aligned ground-truth timeline."""

    title: str
    composer: str
    style: str
    key: str
    time_signature: tuple[int, int]
    groove: str
    tempo: int
    beats_per_bar: int
    mma_text: str
    # one entry per rendered bar: (bar_number_1indexed, section_label,
    #                              [(beat_offset_0indexed, ireal_token, mma_chord), ...])
    timeline: list[tuple[int, str, list[tuple[int, str, str]]]]
    unmapped_tokens: list[str] = field(default_factory=list)
    # raw iReal chord_string (before pyRealParser expansion) — needed by
    # irealb_aligner.parse_form_compact to recover true repeat structure
    chord_string: str = ""

    @property
    def form(self) -> str:
        """Compact form string, e.g. 'A8 B8 C8 D12'."""
        runs: list[list] = []
        for _, label, _ in self.timeline:
            if not runs or runs[-1][0] != label:
                runs.append([label, 1])
            else:
                runs[-1][1] += 1
        return " ".join(f"{lab}{n}" for lab, n in runs)

    @property
    def section_per_bar(self) -> list[str]:
        return [label for _, label, _ in self.timeline]


def load_playlist(path: Path) -> list[Tune]:
    """Parse an iReal playlist file (one or more irealb:// URLs) into Tunes."""
    text = Path(path).read_text()
    urls = re.findall(r"irealb(?:ook)?://[^\s\"<>]+", text)
    tunes: list[Tune] = []
    for url in urls:
        tunes.extend(Tune.parse_ireal_url(url))
    return tunes


def sectionized_measures(tune: Tune) -> list[tuple[str, str]]:
    """Flatten a tune's chord string like pyRealParser does, but keep section labels.

    pyRealParser's _remove_annotations strips ``*A`` markers before flattening.
    We pre-replace them with an ``@A`` sentinel that survives every regex in
    _get_measures (bar-split, repeat filling, slash filling), then peel it off
    the flattened measures. Repeated sections correctly re-emit their label
    because repeat expansion copies the sentinel along with the chords.
    """
    cs = re.sub(r"\*(\w)", lambda m: "@" + m.group(1) + " ", tune.chord_string)
    measures = _TuneCls._get_measures(cs)
    out: list[tuple[str, str]] = []
    label = "A"
    pending: str | None = None
    for measure in measures:
        if pending is not None:
            label = pending
            pending = None
        m = re.match(r"@(\w)", measure)
        if m:
            label = m.group(1)
            measure = measure[2:]
        # a sentinel glued mid/end of a measure (repeat expansion can drop the
        # bar separator) labels the *next* measure, not this one
        if "@" in measure:
            marks = re.findall(r"@(\w)", measure)
            if marks:
                pending = marks[-1]
            measure = re.sub(r"@\w?", "", measure)
        if measure.strip():
            out.append((label, measure))
    return out


def split_chords(measure: str) -> list[str]:
    """Split a flattened iReal measure like ``'Eh7A7b9'`` into ``['Eh7', 'A7b9']``.

    A chord starts at an uppercase root letter A–G, except when that letter is
    a slash-bass (preceded by ``/``). ``n`` (N.C.), ``p`` (repeat-previous) and
    ``W`` (invisible root / bass-only symbol) also start tokens.
    """
    starts = [
        j
        for j, ch in enumerate(measure)
        if ch in "ABCDEFGnpW" and (j == 0 or measure[j - 1] != "/")
    ]
    return [
        measure[st : starts[k + 1] if k + 1 < len(starts) else len(measure)]
        for k, st in enumerate(starts)
    ]


def to_mma_chord(token: str, valid_qualities: set[str] | None = None) -> str | None:
    """Convert one iReal chord token to an MMA chord name.

    Returns ``'z'`` for explicit no-chord, ``None`` if the token can't be mapped
    (caller decides whether to substitute or fail).
    """
    token = token.strip()
    if token in ("n", "N.C.", "p", "W"):
        return "z"
    # strip leaked repeat-ending / segno / coda / bar-repeat markers
    token = re.sub(r"N\d", "", token)
    token = re.sub(r"[UQSr]+$", "", token)
    if token.startswith("W/"):
        # bass-note-only symbol: render as a power chord on the bass note
        m = re.match(r"W/([A-G][b#]?)", token)
        return m.group(1) + "5" if m else None
    m = _CHORD_RE.match(token)
    if m is None:
        return None
    root, quality, bass = m.groups()
    quality = quality.replace("W", "").strip()
    mma_quality = IREAL_TO_MMA.get(quality)
    if mma_quality is None:
        # pass through qualities MMA already understands natively
        if valid_qualities is not None and quality in valid_qualities:
            mma_quality = quality
        else:
            return None
    if valid_qualities is not None and mma_quality and mma_quality not in valid_qualities:
        return None
    out = root + mma_quality
    if bass:
        out += "/" + bass
    return out


def chord_root_pc(mma_chord: str) -> int | None:
    """Pitch class of an MMA chord's root, or None for 'z' (no chord)."""
    if not mma_chord or mma_chord == "z":
        return None
    pc = _NOTE_TO_PC[mma_chord[0]]
    if len(mma_chord) > 1:
        if mma_chord[1] == "#":
            pc += 1
        elif mma_chord[1] == "b":
            pc -= 1
    return pc % 12


def style_to_groove(style: str | None, time_signature: tuple[int, int] | None) -> tuple[str, int]:
    """Map an iReal style string + time signature to an (MMA groove, BPM)."""
    num = time_signature[0] if time_signature else 4
    if num == 3:
        return "JazzWaltz", 140
    if num == 5:
        return "Jazz54", 140
    if num == 6:
        return "68Swing", 100
    if num == 12:
        return "Ballad128", 70
    s = (style or "").lower()
    for key, groove, bpm in STYLE_GROOVES:
        if key in s:
            return groove, bpm
    return "Swing", 140


def tune_to_mma(
    tune: Tune,
    valid_qualities: set[str] | None = None,
    tempo: int | None = None,
    rnd_seed: int = 42,
) -> MMAChart:
    """Convert a parsed Tune into an MMA chart + aligned ground-truth timeline.

    Chords within a bar are distributed evenly across beats (iReal semantics).
    Unmappable tokens are rendered as 'z' (rest) and reported in
    ``unmapped_tokens`` so callers can decide to reject the song.
    """
    groove, default_bpm = style_to_groove(tune.style, tune.time_signature)
    bpm = tempo or default_bpm
    beats_per_bar = tune.time_signature[0] if tune.time_signature else 4

    lines = [
        f"// {tune.title} ({tune.style})",
        f"RndSeed {rnd_seed}",
        f"Time {beats_per_bar}",
        f"Tempo {bpm}",
        f"Groove {groove}",
    ]
    timeline: list[tuple[int, str, list[tuple[int, str, str]]]] = []
    unmapped: list[str] = []
    barno = 0
    for label, measure in sectionized_measures(tune):
        tokens = split_chords(measure)
        if not tokens:
            continue
        slots: list[tuple[int, str, str]] = []
        per = max(beats_per_bar // max(len(tokens), 1), 1)
        for i, tok in enumerate(tokens[:beats_per_bar]):
            chord = to_mma_chord(tok, valid_qualities)
            if chord is None:
                unmapped.append(tok)
                chord = "z"
            slots.append((i * per, tok, chord))
        cells = []
        for b in range(beats_per_bar):
            active = [c for pos, _, c in slots if pos == b]
            cells.append(active[0] if active else "/")
        barno += 1
        lines.append(f"{barno} " + " ".join(cells))
        timeline.append((barno, label, slots))

    return MMAChart(
        title=tune.title,
        composer=tune.composer,
        style=tune.style,
        key=tune.key,
        time_signature=tune.time_signature,
        groove=groove,
        tempo=bpm,
        beats_per_bar=beats_per_bar,
        mma_text="\n".join(lines) + "\n",
        timeline=timeline,
        unmapped_tokens=unmapped,
        chord_string=tune.chord_string,
    )
