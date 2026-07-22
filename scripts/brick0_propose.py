"""Brick 0 — per-song frozen-GT *PROPOSAL* builder (batch 1), SECTION-BASED.

This is the PROPOSAL half of Brick 0 (the scorer/schema half is
`harmonia/eval/accuracy_score.py` + `golden/brick0/SCHEMA.md`, commit 759643d).
It emits, per song, a `golden/brick0/<song>.gt.json` with **`verified=false`**
plus a self-contained ear-verification HTML under `docs/brick0_review/`. It
NEVER freezes/verifies — a human (Louis) hand-verifies by ear and flips
`verified` afterwards.

WHY THIS REWRITE (2026-07-22, section-based re-proposal).
---------------------------------------------------------
The first proposer tiled the whole chart form RIGIDLY from one bar-1 anchor.
Louis ear-verified batch 1 and found a systemic flaw: rigid whole-form tiling
breaks on (a) INTROS — the song starts late but the chart is forced to t=0, so
everything smears — and (b) TURNAROUNDS / VAMPS — extra material between
choruses that isn't in the chart, which rigid tiling has nothing to absorb.
The fix (memory `feedback_chart_alignment_sections`):

  1. Deconstruct the chart into ordered SECTIONS, each with its chord sequence
     AND per-chord CHART durations (bars/beats).
  2. Detect the first section's onset (SKIP the intro) — the intro is a leading
     low-agreement gap, the chart must NOT be assumed to start at audio t=0.
  3. Align sections IN ORDER WITH LEEWAY (ordered-with-gaps DP): each section
     anchored where its harmony matches best, constrained to follow the
     previous section, but with GAPS allowed between sections for
     turnarounds/vamps. Inter-section gaps are left UNLABELED.
  4. Within each anchored section, lay chords at their CHART durations on the
     Beat This! beats (a 2-beat chord spans 2 beats; NOT distributed evenly).

THE ONE NON-CIRCULAR SIGNAL — per-region HARMONIC AGREEMENT.
-----------------------------------------------------------
Everything is driven by ONE signal computed from raw audio only: the
beat-synchronous PEARSON correlation (mean-centred over the 12 pitch classes)
between the audio's CQT chroma and the chart chord's binary chord-tone
template. This is `harmonic_agreement()`. It NEVER touches the model's decode
(the circularity CLAUDE.md #3 kills). It is used THREE ways:

  * as the ALIGNMENT OBJECTIVE — the section DP maximises total agreement, so
    intros/vamps fall out as low-agreement regions the DP skips/leaves
    unlabeled rather than being force-fit;
  * as the per-song + per-REGION QUALITY/CONFIDENCE score — a well-aligned song
    has high, stable agreement; a misaligned region shows a LOCAL DROP. We
    report the whole-song agreement, the WORST region + its time, and colour
    the HTML chord ribbon by agreement so the ear goes straight to the weak
    spot. A song can NOT be high-confidence if agreement is low somewhere;
  * to BREAK TRANSPOSE TIES — the correct transpose maximises WHOLE-SONG
    agreement (over the full section alignment), replacing the old single-frame
    chroma peak that dead-tied a key against its dominant (Close To You / Blue
    Bossa backing). We pick the transpose whose full alignment agreement wins.

NEAR-CONSTANT TEMPO — aligner v3 (Louis round 5, the governing prior).
-----------------------------------------------------------------------
v2 let the aligner DEFORM tempo WITHIN a song (per-section free placement on the
DRIFTING Beat This! beat array + per-section sub-beat refinement + per-occurrence
cross-rep nudging) to chase local agreement. That overfit: it drifted Autumn
Leaves' start 0.6s->9s and mis-timed Close To You's bridge. Fix: ONE rigid
chart-duration grid at ONE constant tempo for the WHOLE song. The Beat This! grid
is used ONLY for the global TEMPO (median beat period) + phase candidates, never
for per-beat warping — chords are laid on a CONSTANT lattice `t = phase + k*bp`.
The alignment's real degrees of freedom collapse to: (1) a GLOBAL tempo-octave
(orig/half/double, one per song), (2) a GLOBAL anchor/phase (sub-beat refined
ONCE for the whole song, not per-section), and (3) where/how-long the PAUSE-GAPS
are (intros/vamps/turnarounds = discrete gaps where the grid pauses then RESUMES
AT THE SAME TEMPO, opened only when they clearly explain more of the song).
Sub-beat refinement and cross-rep now adjust the global anchor + flag divergences
— they NEVER warp the tempo. Kept from v2: content-first section anchoring (never
flux), melodic-pickup!=downbeat, cross-rep divergence FLAGGING, the iReal parse
fixes, and the per-song `human_anchor` data field.

GAP DISCIPLINE — aligner v4 (Louis round 6). Gaps are EXCEPTIONS, not tempo
-------------------------------------------------------------------------
compensation. v3 still opened a pile of sub-second "pause-gaps" (~0.6s after each
A section) to reach the next match — that is COMPENSATING for a constant tempo a
hair too FAST (each section ends early). v4 kills that two ways: (1) the tempo is
FINE-TUNED — `select_fine_tempo` sweeps the ONE global tempo in small % steps
around the coarse octave period and picks the tempo at which sections butt up
CONTIGUOUSLY (max min-gap agreement, MIN small-gap pressure); gap-insertion
pressure IS the signal that the tempo is wrong. (2) The placement DP (`_dp_min_gap`)
STRUCTURALLY forbids small gaps: an inter-section gap is EITHER exactly 0 beats
(contiguous) OR >= `_min_gap_beats` (a real multi-second vamp that also has to earn
`_GAP_OPEN` agreement to open). So most songs have ZERO internal gaps; the only
survivor is a genuine sustained vamp/turnaround (Autumn Leaves' ~10-12s). A uniform
lateness across a whole song (Blue Bossa) is a global-anchor offset (shift the
`human_anchor`), never a per-section gap; a rubato ballad that one constant tempo
cannot cover (Georgia) is FLAGGED via low coverage, never papered over with gaps.

TEMPO DRIFT — aligner v5 (Louis round 8, "LONG LIVE TAKES DRIFT"). A long take
------------------------------------------------------------------------------
drifts a LITTLE (Blue Bossa's band accelerates ~170->172 BPM over 8.5 min, so NO
single constant tempo fits). v5 REFINES the constant-tempo prior without breaking
it: it adds ONE new degree of freedom, a single GLOBAL low-order (<= quadratic)
offset model whose LOCAL SLOPE sets a per-section BPM that can only TRACK the drift,
never JUMP. The trigger (`detect_drift`) is the "best local time shift" ramp: per
chorus (or section) slide the placed chords together by a trial offset that maximises
harmonic agreement over the span; that offset = the local timing error of the rigid
grid, and a MONOTONE ramp = tempo drift. Gates (the guardrail): a materially-large
(>= ~1.5 beats), monotone (|Pearson|>=0.6), well-fit ramp -> `drift` (warp the grid by
the smooth offset model, RE-PLACE, report a smooth per-section BPM series); a ~flat
ramp -> `flat` (hold ONE constant tempo — constant songs untouched); an erratic ramp
-> `erratic` (rubato/granularity — left FLAGGED, never chased). This is the tempo dual
of v4's gap discipline; the v2 free-per-section warp stays banned. On batch 1 it fires
on Blue Bossa ALONE (offset ramp span 2.1s->0.7s, 2nd-half agreement 0.256->0.308,
per-section 170.2->172.8 BPM) and leaves the 7 constant/rubato/gap songs identical.

NON-CIRCULARITY CONTRACT. Chord labels <- iReal chart ONLY (keep `/bass` ->
SOUNDING bass via `corpus_schema.sounding_bass_pc`). Beat grid <- an INDEPENDENT
Beat This! pass (tempo + phase ONLY). Everything acoustic <- raw librosa CQT
chroma. The model's own chord decode is used NOWHERE.

Every proposed field carries {confidence in [0,1], top alternative, EVIDENCE}
with real run numbers. Per-song aggregate = MIN of the three machine-guessed
fields (transpose / alignment / start-anchor) — weakest-link — additionally
tempered so a low worst-region agreement caps the aggregate. The verification
queue is sorted ASCENDING by that aggregate so Louis spends ear-time on the
ambiguous songs first, and any region whose agreement stays low even after the
re-fit is flagged as genuine ambiguity.

Run:  .venv/bin/python scripts/brick0_propose.py
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Make `harmonia` importable when run as `python scripts/brick0_propose.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("brick0_propose")

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "golden" / "brick0"
REVIEW = REPO / "docs" / "brick0_review"
MANIFEST = REPO / "data" / "real_audio_benchmark" / "brick0_batch1.json"

NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ── Batch 1: on-disk audio (docs/audio/*.m4a) x iReal chart (data/ireal/*.txt).
BATCH1 = [
    dict(song_id="autumn_leaves", title="Autumn Leaves",
         audio="docs/audio/autumn_leaves.m4a",
         ireal_file="jazz1460", tune_title="Autumn Leaves",
         # Louis's ear (round 5): the head opens at ~0.6s (trumpet pickup lands the
         # first comping onset there); v2 drifted the start to 9s because the
         # per-section free-tempo grid deformed. Seed the anchor and hold ONE
         # constant tempo from here (per-song DATA field, like Blue Bossa's 12s).
         human_anchor=0.6),
    dict(song_id="blue_bossa", title="Blue Bossa",
         audio="docs/audio/blue_bossa.m4a",
         ireal_file="jazz1460", tune_title="Blue Bossa",
         # The ONLY human-provided constant in the whole batch, and a per-song
         # DATA field (not baked into any code path): the 9-min jam is chroma-flat
         # so pure agreement can't resolve the head onset. Louis round 6 REVISED his
         # ear from ~12s to ~11s — the v3 fit sat ~1s LATE uniformly (a global anchor
         # offset, not per-section) — so seed 11s and snap/sub-beat-refine from there.
         # Every other song self-aligns with no human input.
         human_anchor=11.0),
    dict(song_id="blue_bossa_backing", title="Blue Bossa (150bpm backing track)",
         audio="docs/audio/blue_bossa_150bpm_backing_track.m4a",
         ireal_file="jazz1460", tune_title="Blue Bossa"),
    dict(song_id="georgia_on_my_mind", title="Georgia On My Mind (Ray Charles)",
         audio="docs/audio/ray_charles_georgia_on_my_mind_official_video.m4a",
         ireal_file="jazz1460", tune_title="Georgia On My Mind"),
    dict(song_id="bein_green", title="Bein' Green",
         audio="docs/audio/bein_green.m4a",
         ireal_file="jazz1460", tune_title="Bein' Green"),
    dict(song_id="close_to_you", title="Close To You (Carpenters)",
         audio="docs/audio/carpenters_close_to_you.m4a",
         ireal_file="pop400", tune_title="Close To You (They Long To Be)"),
    dict(song_id="stand_by_me", title="Stand By Me (Ben E. King)",
         audio="docs/audio/ben_e_king_stand_by_me_audio.m4a",
         ireal_file="pop400", tune_title="Stand By Me"),
    dict(song_id="every_breath_you_take", title="Every Breath You Take (The Police)",
         audio="docs/audio/the_police_every_breath_you_take_official_music_video.m4a",
         ireal_file="pop400", tune_title="Every Breath You Take"),
]

# ── iReal quality token -> shipped schema quality vocabulary ─────────────────
IREAL_Q_TO_SCHEMA: dict[str, str] = {
    "": "maj", "add9": "maj", "2": "maj", "5": "maj",
    "^": "maj7", "^7": "maj7", "^9": "maj9", "^13": "maj7",
    "6": "6", "69": "6",
    "-": "min", "-7": "min7", "-9": "min9", "-6": "min6", "-69": "min6",
    "-^7": "minmaj7", "-^9": "minmaj7",
    "h": "hdim7", "h7": "hdim7", "h9": "hdim7",
    "o": "dim", "o7": "dim7",
    "7": "7", "9": "9", "11": "11", "13": "13",
    "7b9": "7", "7#9": "7", "7b5": "7", "7#5": "7", "7b13": "7",
    "7#11": "7", "7alt": "7", "alt": "7", "at": "7", "7#9#5": "7", "7b9b5": "7",
    "+": "aug", "7+": "7",
    "sus": "sus4", "7sus": "7sus4", "9sus": "7sus4", "13sus": "7sus4",
}

# Chord-tone pitch-class sets (root-relative) for the acoustic agreement match.
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9], "hdim7": [0, 3, 6, 10], "aug": [0, 4, 8],
    "sus2": [0, 2, 7], "sus4": [0, 5, 7], "7sus4": [0, 5, 7, 10],
    "6": [0, 4, 7, 9], "maj6": [0, 4, 7, 9], "min6": [0, 3, 7, 9],
    "9": [0, 4, 7, 10, 2], "maj9": [0, 4, 7, 11, 2], "min9": [0, 3, 7, 10, 2],
    "11": [0, 4, 7, 10, 2, 5], "13": [0, 4, 7, 10, 2, 9], "minmaj7": [0, 3, 7, 11],
}
# Quality families used by the Close-To-You dominant-preservation fix.
_DOM_QUALITIES = {"7", "9", "11", "13"}

_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# root(1) + optional quality(2) + optional slash-bass(3), anchored at end.
_CHORD_RE = re.compile(r"([A-G][b#]?)([^A-G]*?)(?:/([A-G][b#]?))?$")


def note_to_pc(name: str) -> int:
    pc = _NOTE_TO_PC[name[0]]
    for c in name[1:]:
        if c == "#":
            pc += 1
        elif c == "b":
            pc -= 1
    return pc % 12


# ── chart parsing ────────────────────────────────────────────────────────────

@dataclass
class BarChord:
    beat_offset: int          # 0-indexed beat within the bar
    root_pc: int              # written-key root pc
    quality: str              # schema quality token
    bass_pc: int              # written-key sounding-bass pc
    bass_name: str | None     # note letter for a slash bass, else None
    raw: str                  # raw iReal token (for debugging)
    bare_major: bool = False  # token was a plain root letter (-> maj); used by
                              #   the dominant-preservation fix


@dataclass
class Chart:
    title: str
    key: str
    beats_per_bar: int
    tempo: int
    bars: list[tuple[str, list[BarChord]]]   # (section_label, [BarChord ...])
    unmapped: list[str] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def section_runs(self) -> list[tuple[str, int]]:
        runs: list[list] = []
        for label, _ in self.bars:
            if not runs or runs[-1][0] != label:
                runs.append([label, 1])
            else:
                runs[-1][1] += 1
        return [(lab, n) for lab, n in runs]


def parse_token(tok: str) -> BarChord | None:
    """Parse one iReal token -> BarChord (written key), or None for no-chord."""
    tok = tok.strip()
    if tok in ("n", "N.C.", "p", "W", ""):
        return None
    tok = re.sub(r"N\d", "", tok)
    tok = re.sub(r"[UQSrl]+$", "", tok)
    m = _CHORD_RE.match(tok)
    if m is None:
        return BarChord(0, 0, "?" + tok, 0, None, tok)
    root, qual, bass = m.groups()
    qual = (qual or "").replace("W", "").strip()
    root_pc = note_to_pc(root)
    schema_q = IREAL_Q_TO_SCHEMA.get(qual)
    if schema_q is None:
        schema_q = "?" + qual
    bare_major = (qual == "" and not bass)   # plain "D", "Eb" -> maj triad
    if bass:
        bass_pc = note_to_pc(bass)
        bass_name = bass
    else:
        bass_pc = root_pc
        bass_name = None
    return BarChord(0, root_pc, schema_q, bass_pc, bass_name, tok, bare_major)


# ── iReal stacked-alternate ("suggested") chord stripping ────────────────────
# iReal Pro notates a reharmonisation SUGGESTION as SMALL chords stacked in the
# same bar as the main (normal-size) chord: e.g. Georgia's `G/B sBb-7 Eb7` = the
# MAIN chord `G/B` for the bar, with a small-print alternate `Bb-7 Eb7` (== Cm7
# F7 at Ray's +2 — exactly Louis's "main A7/C# vs suggestion Cm7 F7"). It is one
# OR the other, never both. pyRealParser strips the `s`/`l` SIZE markers but
# KEEPS the chord, so the alternate got MERGED into the main line (the bug Louis
# heard). Fix: PER BAR, drop the small chords ONLY when a normal-size chord also
# occupies the bar. If the WHOLE bar is small (e.g. Bein' Green's 1st-ending
# walkdown `sBb^ sAb/Eb Gb/Db F7/C`), the small print IS the real content — keep
# it. iReal size is a toggle (`s`=small, `l`=large); we reset to large per bar.
_SMALL_MARK = re.compile(r"(?<!su)s(?!us)")          # 's' small marker, not 'sus'
_LARGE_MARK = re.compile(r"(?<![A-Za-z])l(?=[A-G])")  # 'l' large marker before a root
_ROOT_RE = re.compile(r"[A-G]")


def _clean_measure_alternates(seg: str) -> str:
    """Within one iReal measure string, drop small-print alternate chords iff a
    normal-size chord shares the bar (else keep — the small print is real)."""
    masked = re.sub(r"<[^>]*>", lambda m: " " * len(m.group()), seg)   # comments
    masked = re.sub(r"\([^)]*\)", lambda m: " " * len(m.group()), masked)  # paren-alts
    smalls = [m.start() for m in _SMALL_MARK.finditer(masked)]
    if not smalls:
        return seg
    larges = [m.start() for m in _LARGE_MARK.finditer(masked)]
    spans = [(s, min([l for l in larges if l > s], default=len(seg))) for s in smalls]
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    in_span = lambda i: any(a <= i < b for a, b in merged)
    if not any(not in_span(m.start()) for m in _ROOT_RE.finditer(masked)):
        return seg                                   # whole bar is small -> keep
    out, prev = [], 0
    for a, b in merged:
        out.append(seg[prev:a]); prev = b
    out.append(seg[prev:])
    return "".join(out)


def strip_stacked_alternates(chord_string: str) -> tuple[str, int]:
    """Remove small-print stacked ALTERNATE chords from a raw iReal chord string,
    bar by bar (see `_clean_measure_alternates`). Returns (cleaned, n_bars_hit)."""
    parts = re.split(r"([|\[\]{}])", chord_string)
    n = 0
    for i, p in enumerate(parts):
        if p in "|[]{}" or p == "":
            continue
        c = _clean_measure_alternates(p)
        if c != p:
            n += 1
        parts[i] = c
    return "".join(parts), n


def parse_chart(ireal_file: str, tune_title: str) -> Chart:
    """Parse a tune from a data/ireal/*.txt playlist into a Chart (written key).

    Reuses ireal_corpus.tune_to_mma for section labelling + repeat expansion +
    within-bar beat distribution, then re-labels each slot's quality with the
    shipped-vocab mapping above. Stacked-alternate ("suggested") small chords are
    stripped first (Georgia/Close-To-You). pyRealParser already expands N1/N2
    first/second endings correctly (Close To You: 1st ending G^7, 2nd ending G7)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # pyRealParser prints "Parsed <t>"
        from harmonia.data.ireal_corpus import load_playlist, tune_to_mma
        tunes = {t.title: t for t in load_playlist(REPO / "data" / "ireal" / f"{ireal_file}.txt")}
        tune = tunes[tune_title]
        cleaned, n_alt = strip_stacked_alternates(tune.chord_string)
        if n_alt:
            tune.chord_string = cleaned
            log.info("  stripped stacked-alternate chords from %d bar(s)", n_alt)
        mma = tune_to_mma(tune)

    bars: list[tuple[str, list[BarChord]]] = []
    unmapped: list[str] = []
    for _barno, label, slots in mma.timeline:
        chords: list[BarChord] = []
        for beat_offset, ireal_tok, _mma in slots:
            bc = parse_token(ireal_tok)
            if bc is None:
                continue
            bc.beat_offset = beat_offset
            if bc.quality.startswith("?"):
                unmapped.append(bc.quality[1:] or ireal_tok)
            chords.append(bc)
        bars.append((label, chords))
    chart = Chart(title=mma.title, key=mma.key, beats_per_bar=mma.beats_per_bar,
                  tempo=mma.tempo, bars=bars, unmapped=unmapped)
    _fix_flattened_dominants(chart)
    return chart


def _fix_flattened_dominants(chart: Chart) -> int:
    """Dominant-preservation fix (Louis, Close To You).

    iReal writers sometimes notate a sustained dominant as a bare root letter
    (``D``) and only spell the extension on the *next* cell (``D9``). The bare
    token parses to a plain major triad, silently DROPPING the dominant 7th —
    exactly the ``Dmaj``/``D#maj`` -> ``D9``/``D#9`` loss Louis heard on Close
    To You. Rule (narrow, so real major chords are untouched): a BARE-root major
    immediately followed — in play order — by a dominant chord on the SAME root
    is promoted to that dominant's quality. Returns the number of promotions.
    """
    flat = [bc for _lab, chords in chart.bars for bc in chords]
    n = 0
    for i, bc in enumerate(flat[:-1]):
        if not bc.bare_major or bc.quality != "maj":
            continue
        nxt = flat[i + 1]
        if nxt.root_pc == bc.root_pc and nxt.quality in _DOM_QUALITIES:
            bc.quality = nxt.quality
            bc.raw += f"->[{nxt.quality} via {nxt.raw}]"
            n += 1
    if n:
        log.info("  dominant-preservation fix: promoted %d bare-major token(s) "
                 "to the following same-root dominant", n)
    return n


# ── sections: chord sequence + CHART durations, and a chord-tone template ────

@dataclass
class Section:
    label: str
    n_bars: int
    beat_chords: list[BarChord | None]   # one entry per beat (chart duration)
    template: np.ndarray                 # (n_bars*bpb, 12) chord-tone, transposed
    content_key: tuple = ()              # transpose-invariant chord signature;
                                         #   equal keys == the SAME section (used
                                         #   to group repetitions for cross-rep)

    @property
    def n_beats(self) -> int:
        return len(self.beat_chords)


def _section_beatgrid(bars: list[tuple[str, list[BarChord]]], bpb: int,
                      transpose: int) -> tuple[list[BarChord | None], np.ndarray]:
    """Per-beat chord (chart duration) + chord-tone template for a run of bars.

    A chord holds from its beat_offset to the next chord's offset (within-bar
    durations), and across whole bars with no change (multi-bar holds) — so the
    CHART rhythm is respected, never distributed evenly."""
    beat_chords: list[BarChord | None] = []
    tmpl: list[np.ndarray] = []
    last: BarChord | None = None
    for _lab, chords in bars:
        cur: list[BarChord | None] = [None] * bpb
        if chords:
            offs = [c.beat_offset for c in chords] + [bpb]
            for i, c in enumerate(chords):
                for b in range(offs[i], min(offs[i + 1], bpb)):
                    cur[b] = c
        for b in range(bpb):
            if cur[b] is None:
                cur[b] = last          # carry a hold across empty beats/bars
            else:
                last = cur[b]
            beat_chords.append(cur[b])
            v = np.zeros(12)
            if cur[b] is not None:
                ivs = QUALITY_INTERVALS.get(cur[b].quality, [0, 4, 7])
                for iv in ivs:
                    v[(cur[b].root_pc + transpose + iv) % 12] = 1.0
            tmpl.append(v)
    return beat_chords, np.asarray(tmpl)


def _bar_sig(chords: list[BarChord]) -> tuple:
    return tuple((c.beat_offset, c.root_pc, c.quality, c.bass_pc, c.bass_name)
                 for c in chords)


_MIN_UNIT_BARS = 4     # never split below this (avoids fragmenting 1-2 bar vamps)


def _minimal_period(bars: list[tuple[str, list[BarChord]]]) -> int:
    """Smallest bar-count d (>= _MIN_UNIT_BARS) dividing len(bars) s.t. the chord
    content repeats with period d (so a 16-bar run that is an 8-bar phrase twice
    -> 8). Stand By Me's 16-bar 'A' is really the 8-bar I-vi-IV-V twice; splitting
    it makes the true repeating unit explicit (fixes '3rd A mislabeled B' + feeds
    cross-rep). The >=4-bar floor stops a 4-bar `Cadd9 C` intro from shattering
    into four 1-bar sections (which wrecked Close To You's coverage)."""
    sigs = [_bar_sig(ch) for _l, ch in bars]
    n = len(sigs)
    for d in range(_MIN_UNIT_BARS, n):
        if n % d == 0 and all(sigs[i] == sigs[i % d] for i in range(n)):
            return d
    return n


def deconstruct_sections(chart: Chart, transpose: int) -> list[Section]:
    """One chorus of the chart -> ordered Section instances. Each maximal
    same-label run of bars is split into its MINIMAL repeating sub-unit, then each
    unit gets a transpose-invariant `content_key` and a CONTENT-based canonical
    label (identical chord content -> same letter, so genuinely-repeated sections
    share a label regardless of the chart's cosmetic *A/*B/*C markers)."""
    runs: list[tuple[str, list[tuple[str, list[BarChord]]]]] = []
    for lab, chords in chart.bars:
        if not runs or runs[-1][0] != lab:
            runs.append((lab, []))
        runs[-1][1].append((lab, chords))
    # split each run into its minimal repeating sub-unit
    units: list[list[tuple[str, list[BarChord]]]] = []
    for _lab, bars in runs:
        d = _minimal_period(bars)
        for k in range(0, len(bars), d):
            units.append(bars[k:k + d])
    out: list[Section] = []
    keymap: dict[tuple, str] = {}
    for bars in units:
        bc, tmpl = _section_beatgrid(bars, chart.beats_per_bar, transpose)
        ckey = tuple((c.root_pc, c.quality, c.bass_pc) if c is not None else None
                     for c in bc)
        if ckey not in keymap:
            i = len(keymap)
            keymap[ckey] = chr(ord("A") + i) if i < 26 else f"S{i}"
        out.append(Section(keymap[ckey], len(bars), bc, tmpl, ckey))
    return out


# ── audio: Beat This! grid + beat-synchronous chroma ────────────────────────

def decode_wav(audio: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / (audio.stem + ".wav")
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
                        "-ac", "1", "-ar", "22050", str(wav)], check=True)
    return wav


def audio_duration(audio: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def beat_this_full(wav: Path) -> dict:
    """INDEPENDENT Beat This! pass -> full beat grid + downbeats (never the
    model's decode grid). Reuses the module's lazy singleton + regularity."""
    from beat_this.inference import load_audio
    from harmonia.models import downbeat_anchor as da

    a2f, pp = da._get_beat_this()
    signal, sr = load_audio(str(wav))
    beat_logits, downbeat_logits = a2f(signal, sr)
    beats, downbeats = pp(beat_logits, downbeat_logits)
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    beat_period = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    return dict(beats=beats, downbeats=downbeats, beat_period=beat_period,
                beat_regularity=da._regularity(beats),
                downbeat_regularity=da._regularity(downbeats))


def load_chroma_frames(wav: Path) -> tuple[np.ndarray, np.ndarray]:
    """Raw librosa CQT chroma (idx0==C) + frame times. Independent of the model."""
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)   # ~43 fps
    times = librosa.frames_to_time(np.arange(ch.shape[1]), sr=sr, hop_length=512)
    return ch.T, times                                            # (nframes,12)


def _const_grid(bp: float, phase: float, audio_dur: float, pad: int = 4) -> np.ndarray:
    """The CONSTANT-tempo lattice `t = phase + k*bp` spanning the whole song
    (v3 governing prior). This REPLACES the drifting Beat This! beat array as the
    timing spine — one rigid grid at one tempo, so no within-song tempo warp is
    possible. `phase` is the global sub-beat anchor (in [0, bp))."""
    n = int(np.ceil(audio_dur / bp)) + pad if bp > 0 else pad
    return phase + np.arange(n) * bp


def beat_sync_chroma(frames: np.ndarray, ftimes: np.ndarray,
                     beats: np.ndarray) -> np.ndarray:
    """Mean chroma within each beat interval [beats[j], beats[j+1]) -> (nbeats,12)."""
    n = len(beats)
    out = np.zeros((n, 12))
    for j in range(n):
        t0 = beats[j]
        t1 = beats[j + 1] if j + 1 < n else t0 + (beats[j] - beats[j - 1] if j > 0 else 0.5)
        lo = int(np.searchsorted(ftimes, t0))
        hi = int(np.searchsorted(ftimes, t1))
        if hi > lo:
            out[j] = frames[lo:hi].mean(axis=0)
    return out


# ── the ONE non-circular signal: per-beat harmonic agreement ─────────────────

def _centre_norm(mat: np.ndarray) -> np.ndarray:
    """Row-wise mean-centre over the 12 pcs then L2-normalise (for Pearson)."""
    c = mat - mat.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return c / nrm


def harmonic_agreement(template: np.ndarray, beat_chroma_cn: np.ndarray,
                       p: int) -> float:
    """Mean per-beat Pearson correlation between a chord-tone template placed at
    beat ``p`` and the (pre-centre-normalised) audio beat-chroma. The one
    non-circular alignment/quality signal. Returns -inf if it runs past the grid."""
    L = len(template)
    if p < 0 or p + L > len(beat_chroma_cn):
        return float("-inf")
    tcn = _centre_norm(template)
    seg = beat_chroma_cn[p:p + L]
    return float((tcn * seg).sum() / L)


def _agreement_curve(template: np.ndarray, cn: np.ndarray) -> np.ndarray:
    """agr[p] for every start beat p (mean per-beat Pearson); -inf past the end.

    Vectorised via the diagonal-sum identity
    ``agr[p] = (1/L) * sum_j <tcn[j], cn[p+j]>`` where the inner products form the
    (L, N) matrix ``D = tcn @ cn.T`` and ``agr[p] = mean_j D[j, p+j]`` — the same
    number the naive per-p loop computed, but as ~L vectorised slice-adds instead
    of N dot products (the fine-tempo sweep calls this dozens of times/song)."""
    L = len(template)
    N = len(cn)
    m = N - L + 1
    out = np.full(N, float("-inf"))
    if m <= 0:
        return out
    tcn = _centre_norm(template)
    D = tcn @ cn.T                       # (L, N) per-beat inner products
    acc = np.zeros(m)
    for j in range(L):
        acc += D[j, j:j + m]
    out[:m] = acc / L
    return out


# ── sub-beat refinement: agreement as an OPTIMISER, not just a detector ───────
# (Louis round 2). The coarse section fit lands on the Beat This! beat grid, but
# a section can be right-to-the-beat yet ~0.5s early on the TRUE downbeat (Stand
# By Me). We fine-tune each section's start with a continuous sub-beat offset that
# MAXIMISES that section's harmonic agreement, re-deriving beat-synchronous chroma
# at the shifted interval boundaries from the raw CQT frames (never the model).

def _span_chroma(frames: np.ndarray, ftimes: np.ndarray,
                 spans: list[tuple[float, float]]) -> np.ndarray:
    """Mean CQT chroma over each arbitrary (t0,t1) interval -> (len(spans),12)."""
    out = np.zeros((len(spans), 12))
    for i, (t0, t1) in enumerate(spans):
        lo = int(np.searchsorted(ftimes, t0))
        hi = int(np.searchsorted(ftimes, t1))
        if hi > lo:
            out[i] = frames[lo:hi].mean(axis=0)
    return out


def _beat_spans(beats: np.ndarray, beat_period: float, start_beat: int,
                L: int, offset_s: float) -> list[tuple[float, float]]:
    """The L per-beat (t0,t1) intervals of a section placed at ``start_beat`` and
    shifted by ``offset_s`` seconds (sub-beat)."""
    return [(_beat_time(beats, start_beat + j, beat_period) + offset_s,
             _beat_time(beats, start_beat + j + 1, beat_period) + offset_s)
            for j in range(L)]


def _agr_at_offset(tcn: np.ndarray, frames: np.ndarray, ftimes: np.ndarray,
                   beats: np.ndarray, beat_period: float, start_beat: int,
                   offset_s: float) -> float:
    """Mean per-beat Pearson of a (pre-centre-normed) template placed at
    start_beat + offset_s. -inf if the shifted span leaves the audio."""
    L = len(tcn)
    spans = _beat_spans(beats, beat_period, start_beat, L, offset_s)
    if spans[0][0] < 0 or spans[-1][1] > (ftimes[-1] if len(ftimes) else 0) + 1e-6:
        return float("-inf")
    cn = _centre_norm(_span_chroma(frames, ftimes, spans))
    return float((tcn * cn).sum() / L)


def _refine_offset(sec: "Section", frames: np.ndarray, ftimes: np.ndarray,
                   beats: np.ndarray, beat_period: float, start_beat: int,
                   span_s: float, step_s: float) -> tuple[float, float]:
    """Search sub-beat offsets in [-span_s, +span_s]; return (best_agr, best_off)
    maximising the section's harmonic agreement."""
    tcn = _centre_norm(sec.template)
    best_a, best_o = float("-inf"), 0.0
    o = -span_s
    while o <= span_s + 1e-9:
        a = _agr_at_offset(tcn, frames, ftimes, beats, beat_period, start_beat, o)
        if a > best_a:
            best_a, best_o = a, o
        o += step_s
    if not np.isfinite(best_a):                      # fell off the grid at 0
        best_a, best_o = _agr_at_offset(
            _centre_norm(sec.template), frames, ftimes, beats, beat_period,
            start_beat, 0.0), 0.0
    return best_a, best_o


def _position_agr(sec: "Section", frames: np.ndarray, ftimes: np.ndarray,
                  beats: np.ndarray, beat_period: float, start_beat: int,
                  offset_s: float) -> list[float]:
    """Per-RLE-chord (per-position) agreement for a placed+shifted section. The
    position list is identical across occurrences of the same section (same
    content_key), so these vectors are directly comparable slot-for-slot — the
    cross-repetition matrix rows."""
    bc = sec.beat_chords
    L = sec.n_beats
    spans = _beat_spans(beats, beat_period, start_beat, L, offset_s)
    cn = _centre_norm(_span_chroma(frames, ftimes, spans))
    tcn = _centre_norm(sec.template)
    out: list[float] = []
    k = 0
    while k < L:
        j = k + 1
        while j < L and _same_chord(bc[j], bc[k]):
            j += 1
        seg = slice(k, j)
        if bc[k] is not None and np.any(tcn[seg]):
            out.append(float((tcn[seg] * cn[seg]).sum() / (j - k)))
        else:
            out.append(float("nan"))                 # no-chord position
        k = j
    return out


# ── ordered-with-gaps section alignment (maximises harmonic agreement) ───────

_START_WINDOW_S = 40.0     # intros beyond this are vanishingly rare in this corpus
_START_MARGIN = 0.90       # earliest peak within 90% of the in-window best
_GAP_COST = 0.010          # per-beat penalty on inter-section gaps (mild: vamps ok)
_MIN_FIT = 0.12            # drop a placed section whose agreement is ~noise

# ── gap DISCIPLINE (aligner v4, Louis round 6) ───────────────────────────────
# Small inter-section gaps are FORBIDDEN — a pile of little gaps is the SYMPTOM of
# a constant tempo that is slightly wrong (each section ends a hair early/late), and
# the cure is to FINE-TUNE THE TEMPO so sections butt up contiguously, NOT to paper
# over the drift with gaps (see select_fine_tempo). Under the min-gap DP the only
# inter-section discontinuity allowed is a SINGLE large, sustained VAMP: a gap must
# be either exactly 0 beats (contiguous) or >= _min_gap_beats (a real multi-second
# turnaround, e.g. Autumn Leaves' ~10-12s vamp), never a sub-second catch-up.
_MIN_VAMP_S = 4.0          # a legitimate gap must span at least this many seconds
_GAP_OPEN = 0.60           # fixed agreement-mass a large gap must EARN to open
                           #   (blocks noise gaps; a real vamp recovers far more)
_GAP_SKIP_W = 0.0          # per-beat cost to skip CHART-explainable material (fill).
                           #   DISABLED: best-single-chord fill can't tell a real vamp
                           #   (Autumn's turnaround fill~0.62) from a drift-resync
                           #   region (Blue Bossa fill~0.45) — both high. The real
                           #   discriminator is the fine tempo (min gap pressure) +
                           #   _GAP_OPEN (a real vamp EARNS far more than a drift skip).
                           #   Kept as a wired knob for later per-beat gap gating.


def _min_gap_beats(beat_period: float) -> int:
    """Smallest inter-section gap (in beats) that counts as a real vamp rather than
    a forbidden tempo-drift catch-up — _MIN_VAMP_S converted to this song's grid."""
    return max(2, int(round(_MIN_VAMP_S / beat_period))) if beat_period > 0 else 2


def _first_section_start(agr0: np.ndarray, beats: np.ndarray,
                         seed_s: float | None = None) -> tuple[int, dict]:
    """First-section onset by EXPLICIT MULTI-HYPOTHESIS (Louis's steer): every
    beat offset in the start window is a candidate onset (a superset of the Beat
    This! downbeats); we score each by the section's harmonic agreement, take the
    local-peak candidates, and CHOOSE the EARLIEST peak within _START_MARGIN of
    the best-in-window (first strong occurrence of the head, i.e. skip the
    intro; a later cleaner chorus must not win the start). We REPORT the top
    candidate landscape + the winner's margin over the runner-up — a small
    margin is an ambiguous start (low confidence, ear needed).

    If ``seed_s`` is given (a human-provided anchor where chroma is blind, e.g.
    Blue Bossa's ~12s in the chroma-flat jam), we SNAP the coarse onset to the
    nearest beat to that seed and let sub-beat refinement take it from there."""
    if seed_s is not None:
        pick = int(np.argmin(np.abs(beats - seed_s))) if len(beats) else 0
        return pick, dict(
            peak=float(agr0[pick]) if np.isfinite(agr0[pick]) else 0.0,
            margin=1.0, start_margin=1.0, window_best=float("nan"),
            chosen_time=round(float(beats[pick]), 2), seeded=True,
            candidates=[dict(t=round(float(beats[pick]), 2),
                             agr=round(float(agr0[pick]), 3)
                             if np.isfinite(agr0[pick]) else None, chosen=True)])
    n = int(np.searchsorted(beats, _START_WINDOW_S))
    n = min(max(n, 1), len(agr0))
    seg = np.where(np.isfinite(agr0[:n]), agr0[:n], -9.0)
    finite = seg[seg > -8]
    if len(finite) == 0:
        return 0, dict(peak=0.0, margin=0.0, start_margin=0.0, candidates=[])
    mx, md = float(finite.max()), float(np.median(finite))
    sep = 4
    # candidate onsets = local peaks (dedup within ~3.5s so choruses don't spam)
    cand: list[tuple[int, float]] = []
    for p in range(len(seg)):
        if seg[p] <= -8:
            continue
        lo, hi = max(0, p - sep), min(len(seg), p + sep + 1)
        if seg[p] >= seg[lo:hi].max() - 1e-9:
            if all(abs(beats[p] - beats[q]) > 3.5 for q, _ in cand):
                cand.append((p, float(seg[p])))
    if not cand:
        cand = [(int(seg.argmax()), mx)]
    # winner = earliest candidate within _START_MARGIN of the window best
    thr = _START_MARGIN * mx
    pick = next((p for p, v in cand if v >= thr), cand[0][0])
    pick_v = float(seg[pick])
    others = [v for p, v in cand if p != pick]
    runner = max(others) if others else pick_v
    start_margin = pick_v - runner            # >0 confident; <0 a stronger rival exists
    rel_margin = (pick_v - md) / (abs(pick_v) + 1e-6)
    landscape = sorted(cand, key=lambda x: -x[1])[:4]
    return pick, dict(
        peak=pick_v, margin=rel_margin, start_margin=start_margin, window_best=mx,
        chosen_time=round(float(beats[pick]), 2),
        candidates=[dict(t=round(float(beats[p]), 2), agr=round(v, 3),
                         chosen=bool(p == pick)) for p, v in landscape])


@dataclass
class Placement:
    chorus: int
    sec_idx: int          # index into one-chorus section list
    label: str
    start_beat: int
    n_beats: int
    agreement: float
    offset_s: float = 0.0     # sub-beat refinement offset (s) applied to start
    occ: int = 0              # occurrence index among same-content sections


def _suffix_max_arg(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(suffmax, sargmax) length N+1: suffmax[p] = max_{p'>=p} a[p'] and sargmax[p]
    = the SMALLEST such p' (the ``>=`` left-tie-break the old backward scan used);
    index N is the empty-suffix sentinel (-1e9, -1). Vectorised replacement for the
    O(N) python scan (the fine-tempo sweep runs this ~200x/song)."""
    N = len(a)
    suff = np.empty(N + 1)
    suff[N] = -1e9
    if N:
        suff[:N] = np.maximum.accumulate(a[::-1])[::-1]     # suffix max
    sarg = np.full(N + 1, -1, dtype=int)
    if N:
        records = np.nonzero(a >= suff[:N])[0]              # leftmost achievers
        idx = np.searchsorted(records, np.arange(N), side="left")
        ok = idx < len(records)
        sarg[:N][ok] = records[idx[ok]]
    return suff, sarg


def _dp_linear_gap(order: list, N: int, p0: int, gap_cost: float) -> list[Placement]:
    """LINEAR-gap DP (any-size gaps, mild per-beat cost). Used ONLY as the
    diagnostic that MEASURES gap-insertion PRESSURE for the fine-tempo search —
    NOT for the final GT, which forbids small catch-up gaps (see _dp_min_gap)."""
    M = len(order)
    U = np.zeros(N + 1)                       # V(M, *) = 0
    Bstore: list = [None] * M
    ramp = gap_cost * np.arange(N)
    for i in range(M - 1, -1, -1):
        _c, _si, s, agr = order[i]
        L = s.n_beats
        Bp = np.full(N, -1e9)
        m = N - L + 1
        if m > 0:
            f = np.where(np.isfinite(agr[:m]), agr[:m], -1e9)
            Bp[:m] = np.where(f > -8, f + U[L:L + m] - ramp[:m], -1e9)
        suffmax, sarg = _suffix_max_arg(Bp)
        Vi = np.maximum(0.0, ramp + suffmax[:N])
        U = np.concatenate([Vi, [0.0]])
        Bstore[i] = sarg
    placements: list[Placement] = []
    p_lb = p0
    for i in range(M):
        c, si, s, agr = order[i]
        sarg = Bstore[i]
        pstar = p0 if i == 0 else (int(sarg[p_lb]) if p_lb < N else -1)
        if pstar < 0 or pstar + s.n_beats > N:
            break
        placements.append(Placement(c, si, s.label, pstar, s.n_beats, float(agr[pstar])))
        p_lb = pstar + s.n_beats
        if p_lb >= N - 1:
            break
    return placements


def _chart_fill(sections: list[Section], cn: np.ndarray) -> np.ndarray:
    """Per-beat CHART-EXPLAINABILITY: fill[j] = the best per-beat agreement any chart
    chord could achieve at beat j (max over the chart's distinct chord-tone templates
    of the centre-normed dot with the audio beat-chroma). HIGH where the recording is
    playing chart material, LOW over non-chart vamps/turnarounds/solo-breaks. This is
    what tells a real vamp (low fill — skip it freely) from a drift-resync gap that
    would jump OVER live chart material (high fill — forbidden)."""
    tmpls = np.vstack([s.template for s in sections])
    tmpls = np.unique(tmpls[tmpls.any(axis=1)], axis=0)     # distinct chords, drop N.C.
    if not len(tmpls):
        return np.zeros(len(cn))
    return (_centre_norm(tmpls) @ cn.T).max(axis=0)         # (N,) best chord per beat


def _dp_min_gap(order: list, N: int, p0: int, skip_cost: np.ndarray,
                min_gap_beats: int, gap_open: float) -> list[Placement]:
    """GAP-DISCIPLINE DP (aligner v4). Between two consecutive sections the gap is
    forced to be EITHER exactly 0 beats (contiguous — the constant-tempo default)
    OR at least ``min_gap_beats`` (a real, sustained VAMP). Sub-second / small
    catch-up gaps — the symptom of a slightly-wrong tempo — are structurally
    IMPOSSIBLE. A large gap additionally must EARN a fixed ``gap_open`` of agreement
    to open AND PAY ``skip_cost`` for every beat it skips, so it opens only over
    genuinely non-chart material (Autumn's turnaround, where skip_cost~0) and NEVER
    jumps over live chart material to resync accumulated drift (Blue Bossa, where the
    skipped beats carry real chords -> high skip_cost blocks the gap).

    Backward DP over the tiled section order. ``V(i,q)`` = best value placing
    sections i.. with section i starting at a feasible beat given the previous
    section ended at q. Branches: contiguous (start == q) or large-gap
    (start >= q + min_gap_beats, charged ``sum(skip_cost[q:start]) + gap_open``).
    Dropping the tail (value 0) is always allowed. The first section is pinned to
    the intro-skip / human-seed anchor ``p0``; the rest chain off it."""
    M = len(order)
    U = np.zeros(N + 1)                       # V(M, *) = 0
    choice: list = [None] * M                 # choice[i][q] = start beat, or -1 (drop)
    qidx = np.arange(N)
    C = np.concatenate([[0.0], np.cumsum(skip_cost)])       # C[p]-C[q] = cost to skip [q,p)
    for i in range(M - 1, -1, -1):
        _c, _si, s, agr = order[i]
        L = s.n_beats
        m = N - L + 1
        Bp = np.full(N, -1e9)                 # value of placing section i AT p (+ rest)
        if m > 0:
            f = np.where(np.isfinite(agr[:m]), agr[:m], -1e9)
            Bp[:m] = np.where(f > -8, f + U[L:L + m], -1e9)
        Sp = Bp - C[:N]                       # large-gap branch (variable skip cost)
        suffS, sargS = _suffix_max_arg(Sp)
        # branch 1 — contiguous (gap == 0): value = Bp[q] where the section fits
        cont = np.where(qidx < m, Bp, -1e9)
        # branch 2 — large gap (start >= q + min_gap_beats): pay skip cost + gap_open
        kk = np.minimum(qidx + min_gap_beats, N)
        large = C[qidx] + suffS[kk] - gap_open
        large_arg = sargS[kk]
        # branch 0 — drop the remaining tail (value 0)
        stack = np.vstack([np.zeros(N), cont, large])          # rows: drop, cont, large
        sel = np.argmax(stack, axis=0)
        Vi = stack[sel, qidx]
        pstar = np.where(sel == 1, qidx, np.where(sel == 2, large_arg, -1))
        choice[i] = pstar
        U = np.concatenate([Vi, [0.0]])
    placements: list[Placement] = []
    q = p0
    for i in range(M):
        c, si, s, agr = order[i]
        pstar = p0 if i == 0 else (int(choice[i][q]) if q < N else -1)
        if pstar < 0 or pstar + s.n_beats > N:
            break
        placements.append(Placement(c, si, s.label, pstar, s.n_beats, float(agr[pstar])))
        q = pstar + s.n_beats
        if q >= N - 1:
            break
    return placements


def align_sections(sections: list[Section], cn: np.ndarray, beats: np.ndarray,
                   seed_s: float | None = None, gap_cost: float = _GAP_COST,
                   min_gap_beats: int = 0, gap_open: float = _GAP_OPEN
                   ) -> tuple[list[Placement], dict]:
    """Place the chart's sections (tiled over enough choruses) on the beat grid,
    IN ORDER, maximising total harmonic agreement. First section onset = intro-skip
    / human seed; trailing ~noise placements (agreement < _MIN_FIT) are dropped.

    ``min_gap_beats`` selects the gap model:
      * ``> 0`` — the v4 GAP-DISCIPLINE DP: inter-section gaps are 0 or >= this many
        beats (a real vamp), never small catch-up gaps. This builds the final GT.
      * ``<= 0`` — the legacy linear-gap DP: any-size gaps at ``gap_cost``/beat, used
        only to MEASURE gap-insertion pressure while fine-tuning the global tempo.
    Returns the placements + a diagnostics dict."""
    N = len(cn)
    chorus_beats = sum(s.n_beats for s in sections) or 1
    maxchor = max(1, int(np.ceil(N / chorus_beats)) + 1)
    curves = [_agreement_curve(s.template, cn) for s in sections]  # per-type, reused
    order: list[tuple[int, int, Section, np.ndarray]] = []
    for c in range(maxchor):
        for si, s in enumerate(sections):
            order.append((c, si, s, curves[si]))

    # first-section onset (intro skip / human seed) anchors the whole chain
    p0, start_diag = _first_section_start(curves[0], beats, seed_s)
    if min_gap_beats > 0:
        # per-beat cost of OPENING a gap over beat j = a small base (prefer shorter
        # gaps) + the chart-explainability there (forbid skipping live chart material)
        skip_cost = gap_cost + _GAP_SKIP_W * np.maximum(_chart_fill(sections, cn), 0.0)
        placements = _dp_min_gap(order, N, p0, skip_cost, min_gap_beats, gap_open)
    else:
        placements = _dp_linear_gap(order, N, p0, gap_cost)

    # tail-trim: drop trailing placements whose agreement is ~noise (outro/silence).
    # (Mid-song dips are KEPT+flagged — a real chart!=recording divergence, e.g.
    # Georgia's reharm, must be placed on the rigid grid, not dropped.)
    while placements and placements[-1].agreement < _MIN_FIT:
        placements.pop()

    _assign_occurrences(placements, sections)
    labeled = sum(p.n_beats for p in placements)
    avg = float(np.mean([p.agreement for p in placements])) if placements else 0.0
    # coverage-weighted whole-song score = agreement mass per TOTAL beat (the fair
    # cross-grid / cross-transpose / cross-tempo objective; unlabeled beats -> 0).
    song_score = (sum(max(p.agreement, 0.0) * p.n_beats for p in placements) / N
                  if N else 0.0)
    diag = dict(n_placed=len(placements), n_choruses_est=len(placements) / max(len(sections), 1),
                avg_agreement=avg, song_score=round(song_score, 4),
                labeled_beats=labeled, total_beats=N,
                coverage=labeled / N if N else 0.0, start=start_diag,
                chorus_beats=chorus_beats)
    return placements, diag


def _assign_occurrences(placements: list[Placement], sections: list[Section]) -> None:
    """Number each placement among placements of the SAME section content_key
    (its repetition index), in time order — the row index of the cross-rep matrix."""
    seen: dict[tuple, int] = {}
    for pl in placements:
        ck = sections[pl.sec_idx].content_key
        pl.occ = seen.get(ck, 0)
        seen[ck] = pl.occ + 1


# ── transpose: pick the one whose full section-alignment agreement wins ───────

def chart_profile(chart: Chart) -> np.ndarray:
    prof = np.zeros(12)
    bpb = chart.beats_per_bar
    for _label, chords in chart.bars:
        if not chords:
            continue
        offs = [c.beat_offset for c in chords] + [bpb]
        for i, c in enumerate(chords):
            dur = max(offs[i + 1] - offs[i], 1)
            for iv in QUALITY_INTERVALS.get(c.quality, [0, 4, 7]):
                prof[(c.root_pc + iv) % 12] += dur
            prof[c.bass_pc % 12] += dur
    n = prof.sum()
    return prof / n if n else prof


def _candidate_transposes(prof: np.ndarray, mean_chroma: np.ndarray, k: int = 5
                          ) -> tuple[list[int], np.ndarray]:
    """Prune to the k best-by-mean-chroma transposes (+ the notated key t=0);
    whole-song agreement then DECIDES among them (the fifth-vs-dominant near-ties
    are both in this shortlist, so the acoustic decision is what breaks them)."""
    scores = np.array([
        float(np.dot(np.roll(prof, t), mean_chroma) /
              ((np.linalg.norm(prof) * np.linalg.norm(mean_chroma)) or 1.0))
        for t in range(12)])
    cand = list(np.argsort(scores)[::-1][:k])
    if 0 not in cand:
        cand.append(0)
    return [int(t) for t in cand], scores


def _grid_beat_period(beats: np.ndarray) -> float:
    """Constant-lattice beat period (median inter-beat) — used to size the min-gap
    vamp threshold from the grid handed in."""
    return float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5


def _aligned_variant(secs: list[Section], cn: np.ndarray, beats: np.ndarray,
                     seed_s: float | None) -> tuple[list[Placement], dict]:
    """v4 gap-discipline placement (replaces v3's gapped-vs-contiguous vote). Runs
    the min-gap DP: sections butt up contiguously (the constant-tempo default) and
    the ONLY inter-section discontinuity allowed is a single large, sustained vamp
    (>= _min_gap_beats). Small catch-up gaps are structurally impossible, so the
    old continuity guard against spurious gaps is no longer needed — a clean song
    stays contiguous by construction, and only Autumn Leaves' real turnaround opens
    a gap. Gap PRESSURE is handled upstream by fine-tuning the global tempo."""
    mgb = _min_gap_beats(_grid_beat_period(beats))
    pl, d = align_sections(secs, cn, beats, seed_s, gap_cost=_GAP_COST,
                           min_gap_beats=mgb, gap_open=_GAP_OPEN)
    d["continuity"] = dict(chosen="min-gap", min_gap_beats=mgb,
                           song_score=d["song_score"], coverage=round(d["coverage"], 3))
    return pl, d


def propose_transpose(chart: Chart, cn: np.ndarray, beats: np.ndarray,
                      mean_chroma: np.ndarray, seed_s: float | None = None
                      ) -> tuple[dict, list[Placement], dict]:
    """Transpose = argmax over candidate transposes of the WHOLE-SONG coverage-
    weighted alignment agreement (non-circular). Returns (transpose proposal, the
    winning placements, its diagnostics) so the alignment is computed once."""
    prof = chart_profile(chart)
    cand, chroma_scores = _candidate_transposes(prof, mean_chroma)
    results = []
    for t in cand:
        secs = deconstruct_sections(chart, t)
        placements, diag = _aligned_variant(secs, cn, beats, seed_s)
        results.append((t, diag["song_score"], diag["coverage"], placements, diag))
    results.sort(key=lambda r: (-r[1], -r[2]))
    best_t, best_score, _cov, best_pl, best_diag = results[0]
    best_agr = best_diag["avg_agreement"]
    second = results[1] if len(results) > 1 else None
    gap = (best_score - second[1]) if second else best_score
    # confidence: how decisively the coverage-weighted score prefers this transpose
    conf = float(np.clip(0.30 + 5.0 * gap, 0.2, 0.9))
    alt_t = int(second[0]) if second else best_t
    ev = (f"whole-song coverage-weighted agreement {best_score:.3f} (per-chord "
          f"r={best_agr:.3f}, cov {best_diag['coverage']*100:.0f}%) @ t={best_t:+d} "
          f"({NOTE_SHARP[best_t]}) vs next-best {second[1] if second else 0:.3f} "
          f"@ t={alt_t:+d} (margin {gap:+.3f}); chart key {chart.key}; "
          f"candidates(chroma-ranked)={cand}; EAR NEEDED")
    prop = dict(transpose_semitones=best_t, confidence=round(conf, 3),
                alternative=dict(transpose=alt_t,
                                 agreement=round(second[1], 3) if second else None),
                evidence=ev, _agreement=round(best_agr, 4), _score=round(best_score, 4),
                _all=[(int(t), round(a, 3)) for t, a, *_ in results])
    return prop, best_pl, best_diag


# ── tempo-octave hypotheses: never trust a bare BPM ──────────────────────────
# Beat This! can double-time lock (Autumn Leaves 187 BPM), mapping chord
# durations to HALF their real length so chords "switch randomly". We add the
# halved and doubled beat grids as HYPOTHESES and pick the octave with the
# highest WHOLE-SONG agreement — the acoustic decision, not the BPM number.

# Tempo-plausibility gate: a DOUBLE-TIME lock only makes sense when Beat This!
# reports a suspiciously FAST tempo (>_FAST_BPM, e.g. Autumn Leaves 187 -> 91),
# and a HALF-time lock only when it reports a suspiciously SLOW one. This stops a
# rubato ballad whose grid is merely noisy (Georgia ~65 BPM) from being "halved"
# to an absurd 32 BPM just because coarser windows smooth the chroma.
_FAST_BPM = 140.0          # above this, suspect a double-time lock -> try 'half'
_SLOW_BPM = 80.0           # below this, suspect a half-time lock  -> try 'double'


def tempo_grid_hypotheses(beats: np.ndarray, beat_period: float) -> dict:
    """{name: (beat_grid, beat_period)}. 'half' (both phases: the wrong phase
    lands on off-beats) is offered only when the detected tempo is implausibly
    FAST; 'double' only when implausibly SLOW. Always includes 'orig'."""
    grids = {"orig": (beats, beat_period)}
    bpm = 60.0 / beat_period if beat_period else 0.0
    if len(beats) >= 4 and bpm >= _FAST_BPM:
        for ph in (0, 1):
            half = beats[ph::2]
            grids[f"half{ph}"] = (half, float(np.median(np.diff(half)))
                                  if len(half) > 1 else beat_period * 2)
    if len(beats) >= 4 and 0 < bpm <= _SLOW_BPM:
        mids = (beats[:-1] + beats[1:]) / 2.0
        dbl = np.sort(np.concatenate([beats, mids]))
        grids["double"] = (dbl, float(np.median(np.diff(dbl))) if len(dbl) > 1
                           else beat_period / 2)
    return grids


# (v3's stand-alone `select_tempo_octave` — coarse-period octave vote — is gone:
# v4 folds the octave choice into `select_octave_and_tempo`, which judges each
# octave on its FINE-TUNED fit so a drifty true octave isn't wrongly halved.)


# ── FINE global-tempo search: gap PRESSURE is the signal, tempo is the knob ────
# (Louis round 6.) The octave picks the coarse period from Beat This!. A constant
# lattice at a tempo a HAIR too fast/slow makes every section end early/late; a
# low-cost aligner would paper that drift over with a pile of sub-second catch-up
# GAPS — the exact symptom to kill. So we do NOT compensate with gaps: we sweep the
# ONE global tempo in small % steps and pick the tempo at which the sections butt up
# CONTIGUOUSLY — max coverage-weighted min-gap agreement, tie-broken by MIN residual
# small-gap pressure (the correct tempo drives small gaps to ~0). Still ONE constant
# tempo for the whole song — a global tweak, never per-section warping. A genuine
# multi-second vamp (Autumn's turnaround) survives as the one allowed large gap.
_FINE_MULTS = tuple(round(1.0 + d, 4) for d in np.arange(-0.06, 0.0601, 0.004))
_FINE_PHASE_FRACS = (0.0, 0.5)          # per-tempo phase probes -> kill grid-phase noise
_FINE_AGR_GUARD = 0.80    # tempo must reach >= this frac of the sweep's best min-gap
                          #   agreement to be eligible (rejects degenerate tempos)
_FINE_GAP_TOL = 3.0       # total-gap tolerance (s) defining the "fits well" basin
                          #   around the minimum-gap tempo
_FINE_ZERO_GAP_S = 0.5    # <= this total gap counts as a GAPLESS tempo


def _gap_seconds(placements: list[Placement], beat_period: float,
                 vamp_s: float = _MIN_VAMP_S) -> tuple[float, float, int, int]:
    """(small_gap_s, large_gap_s, n_small, n_large) over inter-section gaps, split
    at the vamp threshold. small = the forbidden catch-up pressure; large = real
    vamps. Reads whatever gaps the (diagnostic) linear aligner opened."""
    small = large = 0.0
    ns = nl = 0
    for a, b in zip(placements[:-1], placements[1:]):
        gb = b.start_beat - (a.start_beat + a.n_beats)
        if gb <= 0:
            continue
        g = gb * beat_period
        if g < vamp_s:
            small += g; ns += 1
        else:
            large += g; nl += 1
    return round(small, 2), round(large, 2), ns, nl


def _phase_base(seed_s: float | None, bp: float, beats: np.ndarray) -> float:
    return (seed_s % bp) if seed_s is not None else (
        float(beats[0]) % bp if len(beats) else 0.0)


def select_fine_tempo(chart: Chart, transpose: int, frames: np.ndarray,
                      ftimes: np.ndarray, coarse_bp: float, seed_s: float | None,
                      audio_dur: float, beats0: np.ndarray
                      ) -> tuple[float, float, dict]:
    """Fine-tune the global constant tempo around ``coarse_bp`` (fixed transpose /
    octave). Returns (best_beat_period, best_phase, report).

    OBJECTIVE (Louis round 6): *minimise total gap length while keeping agreement
    high* — the correct tempo makes the sections butt up. Over tempos clearing
    ``_FINE_AGR_GUARD`` of the sweep's best min-gap agreement (rejects washed-out
    tempos):
      * if ANY tempo tiles the song GAPLESSLY (total gap <= _FINE_ZERO_GAP_S) — the
        usual case — take the BEST-AGREEMENT such tempo (the tight, correct fit);
      * else NO constant tempo is gapless, i.e. a real VAMP always lurks (Autumn's
        turnaround) — take the SLOWEST tempo within _FINE_GAP_TOL of the min total
        gap, so the vamp OPENS as one clean gap instead of being smeared/rushed.
    (Maximising agreement alone would smear the chart across a real vamp at a
    compromise tempo; blindly taking the slowest would over-slow the gapless songs.)"""
    sections = deconstruct_sections(chart, transpose)
    # A human seed pins the DOWNBEAT phase (seed % bp); probing a half-beat-off
    # phase would break that pin (Autumn's 0.6s start), so seeded songs use pf=0.
    phase_fracs = (0.0,) if seed_s is not None else _FINE_PHASE_FRACS
    rows: list[dict] = []
    for mult in _FINE_MULTS:
        bp = coarse_bp * mult
        mgb = _min_gap_beats(bp)
        base = _phase_base(seed_s, bp, beats0)
        ph_best = None                              # (score, phase, coverage)
        for pf in phase_fracs:
            phase = (base + pf * bp) % bp
            gbeats = _const_grid(bp, phase, audio_dur)
            cn = _centre_norm(beat_sync_chroma(frames, ftimes, gbeats))
            _pl, d = align_sections(sections, cn, gbeats, seed_s, min_gap_beats=mgb)
            if ph_best is None or d["song_score"] > ph_best[0]:
                ph_best = (d["song_score"], phase, d["coverage"])
        score, phase, cov = ph_best
        # gap PRESSURE at this (bp, best-phase): the low-cost linear aligner reveals
        # how much the tempo wants to be papered over with small catch-up gaps.
        gbeats = _const_grid(bp, phase, audio_dur)
        cn = _centre_norm(beat_sync_chroma(frames, ftimes, gbeats))
        gpl, _gd = align_sections(sections, cn, gbeats, seed_s, gap_cost=_GAP_COST,
                                  min_gap_beats=0)
        small_s, large_s, ns, nl = _gap_seconds(gpl, bp)
        rows.append(dict(mult=mult, bp=round(bp, 4), bpm=round(60.0 / bp, 1),
                         phase=round(phase, 4), mingap_agr=round(score, 4),
                         coverage=round(cov, 3), small_gap_s=small_s,
                         large_gap_s=large_s, n_small=ns, n_large=nl))
    # SELECT (agreement-guarded): best-agreement GAPLESS tempo if one exists, else
    # slowest tempo in the low-total-gap basin (a real vamp lurks -> let it open).
    for r in rows:
        r["total_gap_s"] = round(r["small_gap_s"] + r["large_gap_s"], 2)
    max_agr = max(r["mingap_agr"] for r in rows)
    guard = _FINE_AGR_GUARD * max_agr
    eligible = [r for r in rows if r["mingap_agr"] >= guard] or rows
    min_total = min(r["total_gap_s"] for r in eligible)
    gapless = [r for r in eligible if r["total_gap_s"] <= _FINE_ZERO_GAP_S]
    if gapless:
        br = max(gapless, key=lambda r: r["mingap_agr"])        # tight best-fit
        mode = "best-agreement gapless"
    else:
        basin = [r for r in eligible if r["total_gap_s"] <= min_total + _FINE_GAP_TOL]
        br = max(basin, key=lambda r: r["mult"])                # slowest -> vamp opens
        mode = "slowest low-gap (vamp lurks)"
    bp_star, phase_star = br["bp"], br["phase"]
    report = dict(coarse_bpm=round(60.0 / coarse_bp, 1),
                  chosen_bpm=round(60.0 / bp_star, 1),
                  chosen_beat_period_s=round(bp_star, 4),
                  chosen_mult=round(bp_star / coarse_bp, 4),
                  max_mingap_agr=round(max_agr, 4), min_total_gap_s=round(min_total, 2),
                  select_mode=mode,
                  objective=f"best-agreement GAPLESS tempo (total<= {_FINE_ZERO_GAP_S}s) if one "
                            f"clears {_FINE_AGR_GUARD:.2f}x best agreement, else SLOWEST within "
                            f"min+{_FINE_GAP_TOL:.0f}s total gap (let a real vamp open, don't smear)",
                  chosen=br, sweep=rows)
    return bp_star, phase_star, report


def select_octave_and_tempo(chart: Chart, transpose: int, frames: np.ndarray,
                            ftimes: np.ndarray, grids: dict, seed_s: float | None,
                            audio_dur: float) -> tuple[str, float, float, dict, dict]:
    """Joint GLOBAL tempo-octave + FINE tempo (v4). The octave MUST be judged on the
    FINE-TUNED fit, not the raw Beat This! period: at the raw period a drifty octave
    (Blue Bossa's true 172 BPM) is forced into gaps and looks worse than its coarse
    half (86 BPM), so a coarse-period octave vote wrongly halves the tempo. Here each
    octave hypothesis is fine-tuned first, then compared on its BEST achievable
    min-gap agreement (max over its tempo sweep — the fair octave discriminator).
    Returns (octave_name, beat_period, phase, fine_report, per_octave_scores)."""
    results: dict[str, tuple] = {}
    for name, (bts, bp_oct) in grids.items():
        if name == "double" and len(grids["orig"][0]) > 1500:
            continue                                # double-time of a fast long jam: skip
        try:
            bp_s, ph_s, rep = select_fine_tempo(chart, transpose, frames, ftimes,
                                                bp_oct, seed_s, audio_dur, bts)
        except Exception:                           # a degenerate grid -> skip
            continue
        results[name] = (rep["max_mingap_agr"], bp_s, ph_s, rep)
    if not results:
        bp = grids["orig"][1]
        return "orig", bp, _phase_base(seed_s, bp, grids["orig"][0]), {}, {"orig": 0.0}
    scores = {n: round(v[0], 4) for n, v in results.items()}
    best = max(results, key=lambda n: results[n][0])
    # keep 'orig' unless another octave beats it by a clear agreement margin
    if best != "orig" and "orig" in results and \
            results["orig"][0] >= results[best][0] - 0.015:
        best = "orig"
    _a, bp_s, ph_s, rep = results[best]
    rep = dict(rep); rep["octave"] = best; rep["octave_scores"] = scores
    return best, bp_s, ph_s, rep, scores


# ── refinement orchestration: GLOBAL phase + cross-repetition consistency ─────

def refine_global_phase(placements: list[Placement], sections: list[Section],
                        frames: np.ndarray, ftimes: np.ndarray, gbeats: np.ndarray,
                        beat_period: float) -> tuple[float, float]:
    """CONSTANT-TEMPO refinement (v3): fine-tune the ONE global sub-beat phase of
    the whole-song lattice (a single offset applied to EVERY beat), maximising the
    coverage-weighted whole-song agreement. This replaces v2's per-section sub-beat
    search — which moved sections independently and thereby warped the tempo. The
    grid stays rigid; only its global phase slides. Returns (best_delta_s, best_agr).
    """
    span, step = 0.5 * beat_period, beat_period / 16.0
    tcns = [(_centre_norm(sections[pl.sec_idx].template), pl) for pl in placements]

    def whole_song_agr(delta: float) -> float:
        tot, wsum = 0.0, 0
        for tcn, pl in tcns:
            a = _agr_at_offset(tcn, frames, ftimes, gbeats, beat_period,
                               pl.start_beat, delta)
            if np.isfinite(a):
                tot += max(a, 0.0) * pl.n_beats
                wsum += pl.n_beats
        return tot / wsum if wsum else float("-inf")

    best_d, best_a = 0.0, whole_song_agr(0.0)
    d = -span
    while d <= span + 1e-9:
        a = whole_song_agr(d)
        if a > best_a:
            best_a, best_d = a, d
        d += step
    return best_d, best_a


def rescore_placements(placements: list[Placement], sections: list[Section],
                       frames: np.ndarray, ftimes: np.ndarray, gbeats: np.ndarray,
                       beat_period: float) -> None:
    """Re-measure each placement's agreement on the (phase-shifted) constant grid.
    Placements keep offset_s=0 — the phase lives in `gbeats`, not per-section."""
    for pl in placements:
        tcn = _centre_norm(sections[pl.sec_idx].template)
        a = _agr_at_offset(tcn, frames, ftimes, gbeats, beat_period,
                           pl.start_beat, 0.0)
        if np.isfinite(a):
            pl.agreement = a


# ── TEMPO-DRIFT detector + piecewise-constant-per-section tempo (aligner v5) ──
# Louis round 8 ("LONG LIVE TAKES DRIFT", Blue Bossa DIAGNOSED). A long take drifts
# a LITTLE: the band accelerates/decelerates over minutes so NO single constant tempo
# fits (Blue Bossa ~170->172 BPM over 8.5 min). This REFINES the constant-tempo prior,
# it does NOT break it. The ONLY new degree of freedom is a single GLOBAL, low-order
# (<= quadratic) offset model whose LOCAL SLOPE sets a per-section BPM that can only
# TRACK the drift, never JUMP — the v2 free-per-section warp stays banned. The tempo
# is CONSTANT WITHIN a section and updates section-to-section along one smooth ramp.
#
# THE METRIC — "best local time shift". For each chorus (or section) placed on the
# constant grid, slide its chords together by a trial offset delta (+- _DRIFT_SPAN_S,
# fine steps) and take the delta MAXIMISING harmonic agreement over that span. delta =
# the local timing error of the rigid grid; its slope d(delta)/dt = the fractional
# tempo error; a CHANGING slope = tempo drift. Non-circular throughout (CHART
# chord-tones vs raw-audio CQT chroma, NEVER the model decode).
#
# THE GUARDRAIL (Louis, critical): fire ONLY on a MONOTONE, materially-large,
# well-fit ramp. A ~flat ramp => no drift (stay one constant tempo — constant songs
# untouched, no regression). An ERRATIC ramp => rubato/granularity => leave FLAGGED,
# never chased. The per-section BPMs come from ONE global fit, so they vary as smoothly
# as that fit — a section's BPM cannot chase local agreement (the banned free warp).
_DRIFT_SPAN_S = 2.5          # +- search window for the local time shift (s)
_DRIFT_STEP_S = 0.05         # fine step of the shift search (s)
_DRIFT_MIN_CHORUSES = 4      # >= this many choruses -> per-chorus ramp; else per-section
_DRIFT_MIN_UNITS = 4         # need at least this many ramp points to fit a drift
# CLASSIFICATION gates. Tuned on batch 1: Blue Bossa (drift, per-chorus |r|=0.89,
# span 2.1s) FIRES; Blue Bossa backing (span 0.2s), Close To You (|r|=0.27 erratic)
# and Georgia (|r|=0.15, rubato) do NOT.
_DRIFT_R_MIN = 0.60          # |Pearson| the monotone ramp must clear
_DRIFT_SPEAR_MIN = 0.55      # |Spearman| (rank monotonicity) the ramp must clear
_DRIFT_SPAN_MIN_S = 0.5      # ramp span floor (s) — below this the drift is negligible
_DRIFT_SPAN_BEATS = 1.5      # ...and it must exceed this many beats (tempo-relative)
_DRIFT_RESID_RATIO = 0.55    # resid_std/span below this => a real trend, not jitter
_DRIFT_MAX_SLOPE = 0.05      # |d(delta)/dt| cap (5%) — reject absurd fits (sanity)


def _local_shift(items: list[tuple[np.ndarray, int, int]], frames: np.ndarray,
                 ftimes: np.ndarray, beats: np.ndarray, beat_period: float,
                 span_s: float = _DRIFT_SPAN_S, step_s: float = _DRIFT_STEP_S
                 ) -> tuple[float, float, float]:
    """Best local time shift over a group of (tcn, start_beat, n_beats) items: the
    delta in [-span, +span] MAXIMISING the beat-weighted mean per-beat harmonic
    agreement, plus that best agr and the zero-shift (baseline) agr. Non-circular."""
    wsum = sum(n for _t, _b, n in items) or 1

    def agr(delta: float) -> float | None:
        tot = 0.0
        for tcn, sb, n in items:
            a = _agr_at_offset(tcn, frames, ftimes, beats, beat_period, sb, delta)
            if not np.isfinite(a):
                return None
            tot += a * n
        return tot / wsum

    best_a, best_d = -1e9, 0.0
    d = -span_s
    while d <= span_s + 1e-9:
        a = agr(d)
        if a is not None and a > best_a:
            best_a, best_d = a, d
        d += step_s
    base = agr(0.0)
    return best_d, (best_a if best_a > -1e8 else float("nan")), (
        base if base is not None else float("nan"))


def offset_ramp(placements: list[Placement], sections: list[Section],
                frames: np.ndarray, ftimes: np.ndarray, beats: np.ndarray,
                beat_period: float) -> tuple[list[tuple], bool]:
    """The one drift signal: the per-chorus (>= _DRIFT_MIN_CHORUSES choruses) or
    per-section best-local-time-shift ramp, as [(t_center, delta, agr_best, n_beats)]
    in time order. delta = the local timing error of the rigid constant grid."""
    by_ch: dict[int, list[Placement]] = {}
    for pl in placements:
        by_ch.setdefault(pl.chorus, []).append(pl)
    per_chorus = len(by_ch) >= _DRIFT_MIN_CHORUSES
    if per_chorus:
        groups = [sorted(v, key=lambda p: p.start_beat) for _k, v in sorted(by_ch.items())]
    else:
        groups = [[pl] for pl in sorted(placements, key=lambda p: p.start_beat)]
    rows: list[tuple] = []
    for pls in groups:
        items = [(_centre_norm(sections[pl.sec_idx].template), pl.start_beat, pl.n_beats)
                 for pl in pls]
        d, a, _base = _local_shift(items, frames, ftimes, beats, beat_period)
        t0 = _beat_time(beats, pls[0].start_beat, beat_period)
        t1 = _beat_time(beats, pls[-1].start_beat + pls[-1].n_beats, beat_period)
        nb = sum(pl.n_beats for pl in pls)
        rows.append((round((t0 + t1) / 2, 3), round(d, 3),
                     round(a, 4) if np.isfinite(a) else 0.0, nb))
    return rows, per_chorus


def detect_drift(rows: list[tuple], beat_period: float) -> dict:
    """Classify the offset ramp -> {classification, apply, coeffs, stats}. The
    guardrail: 'drift' (apply piecewise tempo) requires a MONOTONE (|Pearson| >=
    _DRIFT_R_MIN, |Spearman| >= _DRIFT_SPEAR_MIN), MATERIALLY-LARGE (span >=
    _DRIFT_SPAN_MIN_S and >= _DRIFT_SPAN_BEATS beats), WELL-FIT (resid_std/span <=
    _DRIFT_RESID_RATIO) ramp. Else 'flat' (span too small -> stay one constant tempo)
    or 'erratic' (rubato/granularity -> leave FLAGGED, do not chase)."""
    n = len(rows)
    if n < _DRIFT_MIN_UNITS:
        return dict(classification="insufficient", apply=False, n_units=n,
                    coeffs=[0.0, 0.0, 0.0])
    t = np.array([r[0] for r in rows], float)
    d = np.array([r[1] for r in rows], float)
    w = np.clip(np.array([r[2] for r in rows], float), 0.0, None) + 1e-3   # agr weights
    W = np.sqrt(w)[:, None]
    A1 = np.vstack([t, np.ones_like(t)]).T
    b1 = np.linalg.lstsq(A1 * W, (d * W[:, 0]), rcond=None)[0]
    lin = A1 @ b1
    span = float(d.max() - d.min())
    pear = float(np.corrcoef(t, d)[0, 1]) if n > 1 else 0.0
    rt = np.argsort(np.argsort(t)).astype(float)
    rd = np.argsort(np.argsort(d)).astype(float)
    spear = float(np.corrcoef(rt, rd)[0, 1])
    resid_std = float((d - lin).std())
    resid_ratio = resid_std / span if span > 1e-6 else 9.9
    span_beats = span / beat_period if beat_period else 0.0
    # quadratic (constant-acceleration = ONE linear tempo ramp) only with enough pts;
    # else the plain line. This is the low-order fit that guardrails per-section BPM.
    if n >= 6:
        A2 = np.vstack([t * t, t, np.ones_like(t)]).T
        coeffs = np.linalg.lstsq(A2 * W, (d * W[:, 0]), rcond=None)[0]
    else:
        coeffs = np.array([0.0, b1[0], b1[1]])
    monotone = abs(pear) >= _DRIFT_R_MIN and abs(spear) >= _DRIFT_SPEAR_MIN
    large = span >= _DRIFT_SPAN_MIN_S and span_beats >= _DRIFT_SPAN_BEATS
    clean = resid_ratio <= _DRIFT_RESID_RATIO
    sane = abs(b1[0]) <= _DRIFT_MAX_SLOPE
    if not large:
        cls = "flat"
    elif monotone and clean and sane:
        cls = "drift"
    else:
        cls = "erratic"
    return dict(classification=cls, apply=(cls == "drift"),
                coeffs=[float(x) for x in coeffs], slope_s_per_s=round(float(b1[0]), 5),
                span_s=round(span, 3), span_beats=round(span_beats, 2),
                pearson=round(pear, 3), spearman=round(spear, 3),
                resid_std=round(resid_std, 3), resid_ratio=round(resid_ratio, 3),
                n_units=n, ramp=[[a, b, c] for a, b, c, _ in rows])


def _drift_offset(coeffs: list[float], t) -> np.ndarray:
    """The smooth global offset model delta_fit(t) = c2 t^2 + c1 t + c0 (seconds)."""
    c2, c1, c0 = coeffs
    return c2 * np.asarray(t) ** 2 + c1 * np.asarray(t) + c0


def drift_grid(beats: np.ndarray, coeffs: list[float]) -> np.ndarray:
    """Warp the constant grid by the smooth global offset model: wbeats[k] =
    beats[k] + delta_fit(beats[k]). Because delta_fit is ONE low-order (<= quadratic)
    global curve, the induced beat period varies only smoothly (a single tempo ramp),
    NOT a free per-beat warp — that is the guardrail, structural. Monotone by
    construction for |delta'| < 1 (enforced defensively)."""
    t = np.asarray(beats, float)
    w = t + _drift_offset(coeffs, t)
    return np.maximum.accumulate(w)


def section_bpms(placements: list[Placement], wbeats: np.ndarray,
                 beat_period: float) -> list[dict]:
    """Per-section (placement) mean BPM on the drift-corrected grid — the 'mean
    drifted BPM per section' Louis asked to keep. Each section's tempo is the mean
    over its own span; it tracks the single global ramp and cannot jump."""
    out: list[dict] = []
    for pl in sorted(placements, key=lambda p: p.start_beat):
        t0 = _beat_time(wbeats, pl.start_beat, beat_period)
        t1 = _beat_time(wbeats, pl.start_beat + pl.n_beats, beat_period)
        bpm = pl.n_beats * 60.0 / (t1 - t0) if t1 > t0 else None
        out.append(dict(chorus=pl.chorus, label=pl.label, occ=pl.occ,
                        t0=round(t0, 2), n_beats=pl.n_beats,
                        bpm=round(bpm, 2) if bpm else None))
    return out


def _matrix_stats(mat: list[list[float]]) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-position (column) mean + std across occurrences (rows), ignoring nan,
    and the mean per-position std (the cross-repetition variance signal)."""
    A = np.array(mat, dtype=float)
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(A, axis=0)
        col_std = np.nanstd(A, axis=0)
    mvar = float(np.nanmean(col_std)) if col_std.size else 0.0
    return col_mean, col_std, mvar


def cross_rep_analysis(placements: list[Placement], sections: list[Section],
                       frames: np.ndarray, ftimes: np.ndarray, beats: np.ndarray,
                       beat_period: float) -> list[dict]:
    """Cross-repetition consistency (Louis round 2) — FLAG-ONLY under constant
    tempo (v3). For each section repeated >=2x, build a position x occurrence
    agreement matrix (like-for-like: same chord at the same slot across passes, so
    chord identity is controlled for). v2 NUDGED a low-scoring occurrence toward
    its best sibling; that moved sections independently and warped the tempo, so v3
    DROPS the nudge — under a rigid constant-tempo grid every occurrence sits where
    the tempo puts it. The matrix is kept purely as a DIAGNOSTIC / DIVERGENCE flag:

      * a position whose score VARIES across occurrences => a candidate misalignment
        (reported as cross-rep variance; the ear decides — no auto-nudge).
      * a position CONSISTENTLY LOW across ALL occurrences (low mean, low spread)
        while neighbours are high => the RECORDING diverging from the chart (a
        repeated substitution/modulation) -> FLAG (chart!=recording), never touch.

    Does NOT mutate placements. Returns one report dict per repeated section."""
    groups: dict[tuple, list[Placement]] = {}
    for pl in placements:
        groups.setdefault(sections[pl.sec_idx].content_key, []).append(pl)

    DIV_MEAN = 0.16           # a position mean below this is "low"
    DIV_STD = 0.07            # ...and this stable across occurrences -> divergence
    reports: list[dict] = []

    for ck, pls in groups.items():
        if len(pls) < 2:
            continue
        sec = sections[pls[0].sec_idx]
        pls = sorted(pls, key=lambda p: p.start_beat)

        M1 = [_position_agr(sec, frames, ftimes, beats, beat_period,
                            pl.start_beat, pl.offset_s) for pl in pls]
        overall = [float(np.nanmean(r)) if np.any(np.isfinite(r)) else float("nan")
                   for r in M1]
        best_ov = float(np.nanmax(overall))
        col_mean, col_std, var_after = _matrix_stats(M1)
        # positional labels + representative time (from first occurrence)
        pos_labels = _section_positions(sec)
        overall_mean = float(np.nanmean(col_mean)) if col_mean.size else 0.0
        divergences = []
        for pi, (lab, _dur) in enumerate(pos_labels):
            if pi >= len(col_mean):
                break
            m, sd = float(col_mean[pi]), float(col_std[pi])
            if m < DIV_MEAN and sd < DIV_STD and m < overall_mean - 0.10:
                t = round(_beat_time(beats, pls[0].start_beat, beat_period)
                          + pls[0].offset_s + _position_beat(sec, pi) * beat_period, 1)
                divergences.append(dict(pos=pi, label=lab, mean=round(m, 3),
                                        std=round(sd, 3), t_first=t,
                                        note="chart!=recording (uniformly low, well-aligned)"))
        reports.append(dict(
            label=sec.label, content_key_len=len(ck), n_occ=len(pls),
            var_before=round(var_after, 4), var_after=round(var_after, 4),
            occ_overall=[round(o, 3) if np.isfinite(o) else None for o in overall],
            best_occ=round(best_ov, 3), nudges=[], divergences=divergences,
            matrix=[[round(x, 3) if np.isfinite(x) else None for x in row] for row in M1],
            positions=[lab for lab, _ in pos_labels]))
    return reports


def _section_positions(sec: Section) -> list[tuple[str, int]]:
    """RLE the section's per-beat chords into (label, n_beats) positions (the
    columns of the cross-rep matrix)."""
    bc = sec.beat_chords
    L = sec.n_beats
    out: list[tuple[str, int]] = []
    k = 0
    while k < L:
        j = k + 1
        while j < L and _same_chord(bc[j], bc[k]):
            j += 1
        c = bc[k]
        lab = f"{NOTE_SHARP[c.root_pc]}:{c.quality}" if c is not None else "N"
        out.append((lab, j - k))
        k = j
    return out


def _position_beat(sec: Section, pos: int) -> int:
    """Beat offset (within the section) of RLE position `pos`."""
    bc = sec.beat_chords
    L = sec.n_beats
    k = 0
    p = 0
    while k < L:
        j = k + 1
        while j < L and _same_chord(bc[j], bc[k]):
            j += 1
        if p == pos:
            return k
        p += 1
        k = j
    return 0


# ── GT timeline construction (chart durations on the Beat This! beats) ───────

def _beat_time(beats: np.ndarray, k: int, beat_period: float) -> float:
    """Time of beat index k, extrapolating past the tracked grid."""
    if 0 <= k < len(beats):
        return float(beats[k])
    if len(beats):
        return float(beats[-1]) + (k - (len(beats) - 1)) * beat_period
    return k * beat_period


def build_gt_chords(placements: list[Placement], sections: list[Section],
                    beats: np.ndarray, beat_period: float, transpose: int,
                    frames: np.ndarray, ftimes: np.ndarray, audio_dur: float
                    ) -> list[dict]:
    """Lay each placed section's chart-duration chords on the Beat This! beats
    (run-length encoding the per-beat chord grid so a 2-beat chord spans 2
    beats), SHIFTED by the placement's sub-beat offset, apply the transpose,
    attach per-chord harmonic agreement (measured at the shifted span), and merge
    adjacent identical spans. Inter-section gaps are simply not emitted (left
    unlabeled -> scored as no-chord)."""
    from harmonia.data.corpus_schema import sounding_bass_pc

    raw: list[dict] = []
    for pl in placements:
        sec = sections[pl.sec_idx]
        bc = sec.beat_chords
        L = sec.n_beats
        off = pl.offset_s
        tcn = _centre_norm(sec.template)
        # RLE the per-beat chord grid into constant spans
        k = 0
        while k < L:
            c = bc[k]
            j = k + 1
            while j < L and _same_chord(bc[j], c):
                j += 1
            if c is not None:
                gb0 = pl.start_beat + k
                gb1 = pl.start_beat + j
                t0 = _beat_time(beats, gb0, beat_period) + off
                t1 = min(_beat_time(beats, gb1, beat_period) + off, audio_dur)
                if t1 > t0:
                    root_pc = (c.root_pc + transpose) % 12
                    bass_pc = (c.bass_pc + transpose) % 12
                    root_name = NOTE_SHARP[root_pc]
                    if c.bass_name is not None and bass_pc != root_pc:
                        label = f"{root_name}:{c.quality}/{NOTE_SHARP[bass_pc]}"
                    else:
                        label = f"{root_name}:{c.quality}"
                    sb = sounding_bass_pc(label, root_pc)
                    spans = _beat_spans(beats, beat_period, gb0, j - k, off)
                    scn = _centre_norm(_span_chroma(frames, ftimes, spans))
                    agr = float((tcn[k:j] * scn).sum() / (j - k)) if j > k else float("nan")
                    raw.append(dict(t0=round(t0, 3), t1=round(t1, 3), root_pc=root_pc,
                                    quality=c.quality,
                                    bass_pc=(sb if sb is not None else root_pc),
                                    label=label,
                                    agr=round(agr, 3) if np.isfinite(agr) else None))
            k = j
    raw.sort(key=lambda c: c["t0"])
    # resolve the small (<~0.5s) overlaps that different sub-beat offsets create
    # at section boundaries: trim the earlier chord's end to the later's start
    # (the later section's refined onset wins); drop any span that collapses.
    dedup: list[dict] = []
    for c in raw:
        if dedup and c["t0"] < dedup[-1]["t1"] - 1e-3:
            if c["t0"] > dedup[-1]["t0"] + 1e-3:
                dedup[-1]["t1"] = round(c["t0"], 3)
            else:
                dedup.pop()                       # earlier span fully overlapped
        if c["t1"] > c["t0"] + 1e-4:
            dedup.append(c)
    # merge adjacent identical (root_pc, quality, bass_pc, label) spans that abut
    merged: list[dict] = []
    for c in dedup:
        if merged and (merged[-1]["root_pc"], merged[-1]["quality"],
                       merged[-1]["bass_pc"], merged[-1]["label"]) == \
                (c["root_pc"], c["quality"], c["bass_pc"], c["label"]) and \
                abs(merged[-1]["t1"] - c["t0"]) < 1e-3:
            merged[-1]["t1"] = c["t1"]
        else:
            merged.append(dict(c))
    return merged


def _same_chord(a: BarChord | None, b: BarChord | None) -> bool:
    if a is None or b is None:
        return a is b
    return (a.root_pc == b.root_pc and a.quality == b.quality
            and a.bass_pc == b.bass_pc and a.bass_name == b.bass_name)


# ── per-region quality report + flux corroborator ───────────────────────────

def region_quality(gt_chords: list[dict]) -> dict:
    """Whole-song + worst-region harmonic agreement over the LABELED spans.
    The 'is this song well aligned?' detector: a misaligned region shows a local
    drop. Worst region = the labeled chord with the lowest agreement + its time."""
    scored = [(c["agr"], c["t0"], c["t1"], c["label"]) for c in gt_chords
              if c.get("agr") is not None]
    if not scored:
        return dict(overall=0.0, worst=None, worst_time=None, worst_label=None,
                    frac_low=1.0)
    durs = np.array([t1 - t0 for _a, t0, t1, _l in scored])
    agrs = np.array([a for a, *_ in scored])
    overall = float(np.average(agrs, weights=durs))
    wi = int(np.argmin(agrs))
    frac_low = float(np.average((agrs < 0.12).astype(float), weights=durs))
    # PER-SONG NORMALISATION (Louis): absolute r is NOT comparable across songs
    # (a well-aligned song can sit at 0.50, a misaligned one at 0.45). Use THIS
    # song's own best-aligned span as the achievable CEILING and report the
    # whole-song fraction-of-achievable. (The position-matched cross-rep view is
    # the sharper signal; this is the coarse per-song scalar.)
    ceiling = float(np.percentile(agrs, 90)) if len(agrs) >= 5 else float(agrs.max())
    ceiling = max(ceiling, 1e-3)
    norm = float(np.clip(overall / ceiling, 0.0, 1.0))
    return dict(overall=round(overall, 3),
                worst=round(float(agrs[wi]), 3),
                worst_time=round(float(scored[wi][1]), 1),
                worst_label=scored[wi][3],
                frac_low=round(frac_low, 3),
                ceiling=round(ceiling, 3), overall_norm=round(norm, 3))


def boundary_flux_agreement(gt_chords: list[dict], frames: np.ndarray,
                            ftimes: np.ndarray) -> dict:
    """Secondary corroborator (durations): do proposed chord-CHANGE times land
    near audio chroma-flux peaks? Reported, not folded into confidence."""
    flux = np.zeros(len(frames))
    d = np.diff(frames, axis=0)
    flux[1:] = np.maximum(d, 0).sum(axis=1)
    if flux.max() > 0:
        flux = flux / flux.max()
    # local flux peaks
    peaks = []
    for i in range(1, len(flux) - 1):
        if flux[i] >= flux[i - 1] and flux[i] > flux[i + 1] and flux[i] > 0.15:
            peaks.append(ftimes[i])
    peaks = np.asarray(peaks)
    if len(peaks) == 0 or len(gt_chords) < 2:
        return dict(boundary_flux_agree=None, n_boundaries=max(0, len(gt_chords) - 1))
    bounds = [c["t0"] for c in gt_chords[1:]]
    hit = sum(1 for b in bounds if np.min(np.abs(peaks - b)) <= 0.30)
    return dict(boundary_flux_agree=round(hit / len(bounds), 3),
                n_boundaries=len(bounds))


# ── confidence model (tempered by agreement) ─────────────────────────────────

def _alignment_confidence(diag: dict, rq: dict) -> tuple[float, str]:
    avg = diag["avg_agreement"]
    base = float(np.clip(0.20 + 1.45 * (avg - 0.10), 0.2, 0.9))
    # temper by coverage and by the EXTENT of low agreement (frac of duration
    # below the noise floor) — a large low-agreement fraction means the fit is
    # broadly off, whereas a single transient dip should not tank the song.
    base *= float(np.clip(0.7 + 0.3 * diag["coverage"], 0.7, 1.0))
    frac_low = rq.get("frac_low", 0.0)
    base *= float(np.clip(1.0 - 1.2 * frac_low, 0.4, 1.0))
    if frac_low > 0.25:                      # broadly-low agreement caps hard
        base = min(base, 0.40)
    ev = (f"section-DP whole-song agreement r={avg:.3f}; placed {diag['n_placed']} "
          f"sections (~{diag['n_choruses_est']:.1f} choruses); coverage "
          f"{diag['coverage']*100:.0f}%; {int(frac_low*100)}% of labeled span low; "
          f"worst-region r={rq.get('worst')}@{rq.get('worst_time')}s ({rq.get('worst_label')})")
    return round(base, 3), ev


def _start_confidence(diag: dict) -> tuple[float, str]:
    s = diag["start"]
    if s.get("seeded"):                          # human-provided anchor (trusted)
        return 0.70, (f"HUMAN-SEEDED start at {s.get('chosen_time')}s (Louis) then "
                      f"sub-beat-refined; chroma too flat here to self-resolve")
    peak = s.get("peak", 0.0)
    margin = s.get("margin", 0.0)
    start_margin = s.get("start_margin", 0.0)
    conf = float(np.clip(0.10 + 1.5 * (peak - 0.12), 0.1, 0.85))
    conf *= float(np.clip(0.4 + 1.4 * margin, 0.4, 1.0))
    # a stronger rival onset later in the window (start_margin<0) = ambiguous
    # start -> cap confidence (this flags the chroma-blind intros).
    if start_margin < 0:
        conf = min(conf, float(np.clip(0.45 + 4.0 * start_margin, 0.12, 0.45)))
    cand = s.get("candidates", [])
    land = " ".join(f"{c['t']}s:r{c['agr']}{'*' if c['chosen'] else ''}" for c in cand)
    return round(conf, 3), (
        f"multi-hypothesis start: chose {s.get('chosen_time')}s (r={peak:.3f}); "
        f"candidates[{land}]; winner margin over runner-up {start_margin:+.3f}; "
        f"rel-margin {margin:.2f} (intro = leading gap)")


# ── verification HTML (self-contained, embedded m4a data URI) ───────────────

def build_html(song: dict, gt_chords: list[dict], sections_view: list[dict],
               prop: dict, agg: dict, bar_starts: list[dict], grid: str,
               region: dict, audio_b64: str) -> str:
    payload = json.dumps(dict(
        title=song["title"], gt_chords=gt_chords, bar_starts=bar_starts,
        sections=sections_view, transpose=prop["transpose"], form=prop["form"],
        anchor=prop["anchor"], aggregate=agg, region=region))
    tr, an = prop["transpose"], prop["anchor"]
    fo = prop["form"]
    return _HTML_TEMPLATE.replace("__TITLE__", song["title"]) \
        .replace("__AGG__", f"{agg['aggregate_confidence']:.2f}") \
        .replace("__WEAK__", agg["weak_field"]) \
        .replace("__GRID__", grid) \
        .replace("__TR__", f"{tr['transpose_semitones']:+d} (conf {tr['confidence']:.2f}) — {tr['evidence']}") \
        .replace("__FO__", f"{fo['one_chorus_form']} (conf {fo['confidence']:.2f}) — {fo['evidence']}") \
        .replace("__AN__", f"t={an['bar1_anchor_time']:.2f}s (conf {an['confidence']:.2f}) — {an['evidence']}") \
        .replace("__REG__", f"overall r={region['overall']} • worst r={region['worst']} @ {region['worst_time']}s ({region['worst_label']}) • {int(region['frac_low']*100)}% low") \
        .replace("__NCH__", str(len(gt_chords))) \
        .replace("__AUDIO_B64__", audio_b64) \
        .replace("__PAYLOAD__", payload)


_HTML_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Brick0 verify — __TITLE__</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#14161b;color:#e6e8ec}
 header{padding:14px 20px;background:#1c1f27;border-bottom:1px solid #2a2e39;position:sticky;top:0;z-index:5}
 h1{margin:0 0 6px;font-size:19px}
 .agg{font-size:14px;margin:4px 0 10px}
 .agg b{color:#ffcf5c}
 .field{font-size:12px;line-height:1.5;margin:3px 0;color:#c3c7d1}
 .field span{color:#7fd1ff;font-weight:600}
 .weak{color:#ff8a8a!important;font-weight:700}
 #now{font-size:60px;font-weight:800;text-align:center;padding:14px}
 #nowsub{text-align:center;color:#9aa0ad;margin-top:-8px;font-size:13px}
 #sections{display:flex;overflow-x:auto;gap:3px;padding:8px 10px 2px;background:#0f1116}
 .sec{flex:0 0 auto;padding:5px 9px;border-radius:5px;font-size:12px;cursor:pointer;
   border:1px solid #2a2e39;white-space:nowrap}
 .sec.gap{background:#241c1c;color:#c98a8a;border-style:dashed}
 .sec b{font-size:13px}
 #ribbon{display:flex;overflow-x:auto;gap:2px;padding:8px 10px 12px;background:#0f1116}
 .cell{flex:0 0 auto;min-width:56px;padding:6px 8px;border-radius:5px;background:#232733;
   text-align:center;font-size:13px;cursor:pointer;border:1px solid transparent;
   border-bottom:3px solid #444}
 .cell.active{background:#2f6df6;border-color:#7fb0ff;color:#fff}
 .cell small{display:block;color:#8b91a0;font-size:10px}
 .cell.active small{color:#cfe0ff}
 .controls{padding:10px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 audio{width:100%;max-width:640px}
 label{font-size:13px;user-select:none}
 .barflash{display:inline-block;width:14px;height:14px;border-radius:50%;background:#333;margin-left:8px;vertical-align:middle}
 .barflash.on{background:#ffcf5c;box-shadow:0 0 10px #ffcf5c}
 .legend{font-size:11px;color:#8b91a0;padding:0 20px 6px}
</style></head><body>
<header>
 <h1>Brick 0 — verify by ear: __TITLE__</h1>
 <div class="agg">aggregate confidence <b>__AGG__</b> &nbsp; weak field: <span class="weak">__WEAK__</span> &nbsp; | &nbsp; __NCH__ GT chords &nbsp; <span class="barflash" id="flash"></span></div>
 <div class="field"><span>grid</span> __GRID__</div>
 <div class="field"><span>transpose</span> __TR__</div>
 <div class="field"><span>form (section-fit)</span> __FO__</div>
 <div class="field"><span>bar-1 anchor / intro-skip</span> __AN__</div>
 <div class="field"><span>harmonic agreement</span> __REG__</div>
</header>
<div class="controls">
 <audio id="au" controls src="data:audio/mp4;base64,__AUDIO_B64__"></audio>
 <label><input type="checkbox" id="click" checked> metronome click at bar downbeats (accent = section start)</label>
</div>
<div id="now">—</div>
<div id="nowsub">click a section/cell to seek • chord cells are colour-coded by harmonic agreement (green=good, red=weak — that's where your ear is needed) • dashed = unlabeled gap (intro/vamp)</div>
<div class="legend">sections (in play order; gaps = non-chart material left unlabeled):</div>
<div id="sections"></div>
<div id="ribbon"></div>
<script>
const D = __PAYLOAD__;
const au = document.getElementById('au');
const ribbon = document.getElementById('ribbon');
const secbar = document.getElementById('sections');
const now = document.getElementById('now');
const flash = document.getElementById('flash');
const clickBox = document.getElementById('click');
function agrColor(a){
  if(a===null||a===undefined) return '#444';
  const x=Math.max(0,Math.min(1,(a-0.05)/0.45));           // 0.05..0.50 -> 0..1
  const r=Math.round(200*(1-x)+40*x), g=Math.round(60*(1-x)+180*x);
  return 'rgb('+r+','+g+',70)';
}
// sections strip (with gaps)
let prevEnd=null;
D.sections.forEach(s=>{
  if(prevEnd!==null && s.t0-prevEnd>0.4){
    const g=document.createElement('div'); g.className='sec gap';
    g.textContent='⏸ gap '+(s.t0-prevEnd).toFixed(1)+'s';
    g.onclick=()=>{au.currentTime=prevEnd+0.01;au.play();}; secbar.appendChild(g);
  }
  const el=document.createElement('div'); el.className='sec';
  el.style.borderBottom='3px solid '+agrColor(s.agr);
  el.innerHTML='<b>'+s.label+'</b> '+s.t0.toFixed(0)+'–'+s.t1.toFixed(0)+'s <small>r'+(s.agr!=null?s.agr.toFixed(2):'?')+'</small>';
  el.onclick=()=>{au.currentTime=s.t0+0.01;au.play();}; secbar.appendChild(el);
  prevEnd=s.t1;
});
// chord ribbon (colour by agreement)
D.gt_chords.forEach((c,i)=>{
  const el=document.createElement('div'); el.className='cell'; el.dataset.i=i;
  el.style.borderBottomColor=agrColor(c.agr);
  el.innerHTML=c.label.replace(':','')+'<small>'+c.t0.toFixed(1)+'s</small>';
  el.onclick=()=>{au.currentTime=c.t0+0.01; au.play();};
  ribbon.appendChild(el);
});
const cells=[...ribbon.children];
let actx=null;
function ping(accent){
  if(!clickBox.checked) return;
  actx=actx||new (window.AudioContext||window.webkitAudioContext)();
  const o=actx.createOscillator(), g=actx.createGain();
  o.frequency.value=accent?1760:880; o.connect(g); g.connect(actx.destination);
  const t=actx.currentTime; g.gain.setValueAtTime(0.001,t);
  g.gain.exponentialRampToValueAtTime(accent?0.5:0.28,t+0.001);
  g.gain.exponentialRampToValueAtTime(0.001,t+0.06);
  o.start(t); o.stop(t+0.07);
}
let bi=0, ci=-1;
function resync(){const t=au.currentTime; bi=D.bar_starts.findIndex(b=>b.t>t); if(bi<0) bi=D.bar_starts.length;}
au.addEventListener('seeking',resync); au.addEventListener('play',()=>{resync();});
function frame(){
  const t=au.currentTime;
  while(bi<D.bar_starts.length && D.bar_starts[bi].t<=t){
    ping(D.bar_starts[bi].accent);
    flash.className='barflash on'; setTimeout(()=>flash.className='barflash',90); bi++;
  }
  let idx=-1;
  for(let i=0;i<D.gt_chords.length;i++){ if(D.gt_chords[i].t0<=t && t<D.gt_chords[i].t1){idx=i;break;} }
  if(idx!==ci){
    if(ci>=0) cells[ci].classList.remove('active');
    if(idx>=0){ cells[idx].classList.add('active');
      now.textContent=D.gt_chords[idx].label.replace(':','');
      cells[idx].scrollIntoView({inline:'center',block:'nearest',behavior:'smooth'});
    } else now.textContent='—';
    ci=idx;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>
"""


# ── main per-song processing ─────────────────────────────────────────────────

def process(song: dict, workdir: Path) -> dict:
    audio = REPO / song["audio"]
    log.info("=== %s (%s) ===", song["song_id"], song["title"])
    chart = parse_chart(song["ireal_file"], song["tune_title"])
    if chart.unmapped:
        log.warning("%s: unmapped quality tokens %s", song["song_id"], set(chart.unmapped))
    dur = audio_duration(audio)

    wav = decode_wav(audio, workdir)
    bt = beat_this_full(wav)
    beats, downbeats = bt["beats"], bt["downbeats"]
    beat_period = bt["beat_period"]
    bar_period = beat_period * chart.beats_per_bar
    est_bpm = 60.0 / beat_period if beat_period else 0.0
    log.info("  Beat This!: %d beats (reg %.2f) / %d downbeats (reg %.2f); "
             "beat=%.3fs bar=%.3fs (~%.0f BPM); dur=%.1fs",
             len(beats), bt["beat_regularity"], len(downbeats),
             bt["downbeat_regularity"], beat_period, bar_period, est_bpm, dur)

    frames, ftimes = load_chroma_frames(wav)
    mean_chroma = frames.mean(axis=0)
    s = mean_chroma.sum()
    mean_chroma = mean_chroma / s if s else mean_chroma
    seed_s = song.get("human_anchor")

    # ── TRANSPOSE at the ORIG coarse constant grid (a PITCH decision, ~independent
    # of the tempo octave; done once so the octave+tempo search can hold it fixed).
    coarse_bp = beat_period
    coarse_phase = _phase_base(seed_s, coarse_bp, beats)
    coarse_grid = _const_grid(coarse_bp, coarse_phase, dur)
    cn0 = _centre_norm(beat_sync_chroma(frames, ftimes, coarse_grid))
    tr, _pl0, _d0 = propose_transpose(chart, cn0, coarse_grid, mean_chroma, seed_s)
    transpose = tr["transpose_semitones"]

    # ── JOINT GLOBAL TEMPO-OCTAVE + FINE TEMPO (v4): gaps are the SIGNAL, tempo is
    # the knob. Each octave hypothesis is FINE-TUNED (sweep the constant tempo in
    # small % steps) and judged on its best min-gap contiguous agreement — so a
    # drifty true octave (Blue Bossa 172 BPM) is not wrongly halved just because its
    # raw period needs gaps. The chosen tempo makes sections butt up CONTIGUOUSLY;
    # we never paper a hair-wrong tempo over with sub-second catch-up gaps.
    grids = tempo_grid_hypotheses(beats, beat_period)
    octave, beat_period, phase, tempo_report, octave_scores = select_octave_and_tempo(
        chart, transpose, frames, ftimes, grids, seed_s, dur)
    bar_period = beat_period * chart.beats_per_bar
    est_bpm = 60.0 / beat_period if beat_period else 0.0
    gbeats = _const_grid(beat_period, phase, dur)
    cn = _centre_norm(beat_sync_chroma(frames, ftimes, gbeats))
    _mid = tempo_report.get("sweep", [{}])[len(tempo_report.get("sweep", [{}])) // 2]
    log.info("  TEMPO: octave '%s' (scores %s); FINE %.1f -> %.1f BPM (x%.4f); "
             "small-gap pressure %.1fs -> %.1fs; %d large vamp-gap(s) at chosen tempo; "
             "phase %.3fs; %d lattice beats", octave, octave_scores,
             tempo_report.get("coarse_bpm", est_bpm), tempo_report.get("chosen_bpm", est_bpm),
             tempo_report.get("chosen_mult", 1.0), _mid.get("small_gap_s", 0.0),
             tempo_report.get("chosen", {}).get("small_gap_s", 0.0),
             tempo_report.get("chosen", {}).get("n_large", 0), phase, len(gbeats))

    # final placement at the tuned constant tempo (min-gap DP: contiguous by
    # default, only a real sustained vamp opens a gap).
    sections = deconstruct_sections(chart, transpose)
    placements, diag = _aligned_variant(sections, cn, gbeats, seed_s)

    # ── REFINEMENT (v3): agreement as an OPTIMISER but NEVER a tempo-warper.
    # (1) ONE global sub-beat phase for the whole grid (not per-section);
    # (2) cross-rep is FLAG-ONLY (no per-occurrence nudge). Tempo stays rigid.
    dphase, _pa = refine_global_phase(placements, sections, frames, ftimes,
                                      gbeats, beat_period)
    if abs(dphase) > 1e-6:
        gbeats = gbeats + dphase
        phase = (phase + dphase) % beat_period
        rescore_placements(placements, sections, frames, ftimes, gbeats, beat_period)
    beats = gbeats                            # downstream timing = the constant grid

    # ── TEMPO DRIFT (v5): detect a MONOTONE per-chorus/section offset ramp and, ONLY
    # then, apply a piecewise-constant-per-section tempo that TRACKS the drift. The
    # ramp is measured on the constant grid; a flat/erratic ramp leaves the grid
    # untouched (constant songs + rubato do not regress). On a real drift we warp the
    # grid by ONE smooth global (<= quadratic) offset model, then RE-PLACE the sections
    # on the corrected grid (better-aligned beats recover coverage/agreement the drift
    # had cost the tail). Non-circular; the v2 free-per-section warp stays banned.
    ramp_rows, per_chorus_ramp = offset_ramp(placements, sections, frames, ftimes,
                                             beats, beat_period)
    drift = detect_drift(ramp_rows, beat_period)
    drift["granularity"] = "chorus" if per_chorus_ramp else "section"
    per_section_bpm: list[dict] = []
    if drift["apply"]:
        beats = drift_grid(gbeats, drift["coeffs"])
        cn_w = _centre_norm(beat_sync_chroma(frames, ftimes, beats))
        placements, diag = _aligned_variant(sections, cn_w, beats, seed_s)
        rescore_placements(placements, sections, frames, ftimes, beats, beat_period)
        per_section_bpm = section_bpms(placements, beats, beat_period)
        bpms = [s["bpm"] for s in per_section_bpm if s["bpm"]]
        log.info("  TEMPO DRIFT: %s ramp (%s-grain, span %.2fs, r=%.2f, slope %.4f s/s) "
                 "-> piecewise per-section tempo %.1f..%.1f BPM over %d sections; re-placed",
                 drift["classification"], drift["granularity"], drift["span_s"],
                 drift["pearson"], drift["slope_s_per_s"],
                 (min(bpms) if bpms else 0.0), (max(bpms) if bpms else 0.0),
                 len(per_section_bpm))
    else:
        log.info("  TEMPO DRIFT: %s ramp (%s-grain, span %.2fs, r=%.2f, resid_ratio %.2f) "
                 "-> NO drift applied (one constant tempo held)", drift["classification"],
                 drift["granularity"], drift.get("span_s", 0.0),
                 drift.get("pearson", 0.0), drift.get("resid_ratio", 0.0))

    xrep = cross_rep_analysis(placements, sections, frames, ftimes, beats, beat_period)
    n_nudge = 0
    n_div = sum(len(r["divergences"]) for r in xrep)
    subbeat_offsets = [round(dphase, 3)]      # the single global phase offset
    log.info("  global phase offset %.3fs; cross-rep: %d repeated section(s), "
             "%d divergence flag(s) (flag-only, no tempo warp)",
             dphase, len(xrep), n_div)

    gt_chords = build_gt_chords(placements, sections, beats, beat_period,
                                transpose, frames, ftimes, dur)
    region = region_quality(gt_chords)
    flux = boundary_flux_agreement(gt_chords, frames, ftimes)

    # anchor (intro-skip) + alignment(form) confidences, tempered by agreement
    anchor_t = float(_beat_time(beats, placements[0].start_beat, beat_period)
                     + placements[0].offset_s) if placements else 0.0
    start_conf, start_ev = _start_confidence(diag)
    align_conf, align_ev = _alignment_confidence(diag, region)

    an = dict(bar1_anchor_time=round(anchor_t, 3), confidence=start_conf,
              alternative=dict(bar1_anchor_time=round(float(downbeats[0]), 3)
                               if len(downbeats) else 0.0,
                               note="first Beat This! downbeat (covers a chroma-blind intro)"),
              evidence=start_ev + f"; anchor t={anchor_t:.2f}s; PHASE is a guess — EAR NEEDED")

    # one-chorus form from the CONTENT-labelled section deconstruction (minimal
    # repeating units, identical content -> same letter)
    one_runs: list[list] = []
    for sec in sections:
        if not one_runs or one_runs[-1][0] != sec.label:
            one_runs.append([sec.label, sec.n_bars])
        else:
            one_runs[-1][1] += sec.n_bars
    sections_per_chorus = len(sections)
    n_chor = max(1, round(len(placements) / max(sections_per_chorus, 1)))
    fo = dict(n_choruses=n_chor, bars_per_chorus=chart.n_bars,
              section_order=[lab for lab, _ in one_runs],
              repeat_counts={lab: sum(n for l2, n in one_runs if l2 == lab)
                             for lab, _ in one_runs},
              one_chorus_form=" ".join(f"{lab}{n}" for lab, n in one_runs),
              intro_bars=0, outro_bars=0,
              confidence=align_conf, alternative=dict(n_choruses=max(1, n_chor - 1)),
              evidence=align_ev)

    # sections view (for HTML) + bar starts (metronome) — all sub-beat shifted
    sections_view = []
    bar_starts = []
    for pl in placements:
        sec = sections[pl.sec_idx]
        off = pl.offset_s
        t0 = _beat_time(beats, pl.start_beat, beat_period) + off
        t1 = min(_beat_time(beats, pl.start_beat + pl.n_beats, beat_period) + off, dur)
        occ_lab = f"{pl.label}{pl.occ + 1}"
        sections_view.append(dict(label=occ_lab, t0=round(t0, 3), t1=round(t1, 3),
                                  agr=round(pl.agreement, 3), offset_s=round(off, 3)))
        for b in range(0, pl.n_beats, chart.beats_per_bar):
            t = _beat_time(beats, pl.start_beat + b, beat_period) + off
            if t < dur:
                bar_starts.append(dict(t=round(t, 3), accent=bool(b == 0)))
    bar_starts.sort(key=lambda x: x["t"])

    fields = {"transpose": tr["confidence"], "form": fo["confidence"],
              "bar1_anchor": an["confidence"]}
    weak = min(fields, key=fields.get)
    agg_conf = min(fields.values())
    if region["overall"] < 0.20:             # global agreement caps the aggregate
        agg_conf = min(agg_conf, 0.35)
    agg = dict(aggregate_confidence=round(agg_conf, 3), weak_field=weak,
               field_confidences=fields,
               harmonic_agreement=region)

    oct_note = "" if octave == "orig" else f" • TEMPO-OCTAVE '{octave}' (scores {octave_scores})"
    grid_str = (f"CONSTANT ~{est_bpm:.0f} BPM (beat {beat_period:.2f}s, bar {bar_period:.2f}s @ "
                f"{chart.beats_per_bar}/4, phase {phase:.2f}s) • beat-reg {bt['beat_regularity']:.2f} • "
                f"downbeat-reg {bt['downbeat_regularity']:.2f}{oct_note}")
    # refinement summary (v3: GLOBAL constant-tempo grid + global phase, no warp)
    refinement = dict(
        constant_tempo=dict(bpm=round(est_bpm, 1) if beat_period else None,
                            beat_period_s=round(beat_period, 3),
                            phase_s=round(phase, 3), global_phase_offset_s=round(dphase, 3),
                            note="ONE rigid grid at ONE FINE-TUNED tempo for the whole "
                                 "song; only DoF = global octave + fine tempo + global "
                                 "phase + rare large vamp-gaps (no small catch-up gaps)"),
        tempo_octave=dict(chosen=octave, scores=octave_scores),
        fine_tempo=tempo_report,               # v4: gap-pressure-minimising tempo sweep
        tempo_drift=dict(                      # v5: monotone-offset-ramp drift model
            classification=drift["classification"], applied=drift["apply"],
            granularity=drift.get("granularity"), n_units=drift.get("n_units"),
            span_s=drift.get("span_s"), span_beats=drift.get("span_beats"),
            pearson=drift.get("pearson"), spearman=drift.get("spearman"),
            resid_ratio=drift.get("resid_ratio"), slope_s_per_s=drift.get("slope_s_per_s"),
            offset_coeffs=drift.get("coeffs"), per_section_bpm=per_section_bpm,
            offset_ramp=drift.get("ramp"),
            note="piecewise-constant tempo per section from ONE global <=quadratic "
                 "offset fit; applied only on a monotone materially-large well-fit "
                 "ramp (flat=>hold one tempo, erratic=>rubato flagged)"),
        continuity=diag.get("continuity"),
        sub_beat_offsets_s=subbeat_offsets,        # the single global phase offset
        sub_beat_median_abs=round(float(np.median(np.abs(subbeat_offsets))), 3) if subbeat_offsets else 0.0,
        sub_beat_max_abs=round(float(np.max(np.abs(subbeat_offsets))), 3) if subbeat_offsets else 0.0,
        cross_repetition=xrep,
        n_nudges=n_nudge, n_divergences=n_div, seed_s=seed_s)

    # ---- write golden/brick0/<song>.gt.json (verified=false) ----
    gt_json = dict(
        song_id=song["song_id"], title=song["title"], audio_path=song["audio"],
        chart_source=dict(ireal_file=song["ireal_file"], index=None,
                          tune_title=song["tune_title"]),
        transpose_semitones=transpose,
        form=dict(section_order=fo["section_order"], repeat_counts=fo["repeat_counts"],
                  intro_bars=fo["intro_bars"], outro_bars=fo["outro_bars"],
                  n_choruses=fo["n_choruses"], bars_per_chorus=fo["bars_per_chorus"],
                  one_chorus_form=fo["one_chorus_form"]),
        bar1_anchor_time=an["bar1_anchor_time"],
        downbeat_times=[b["t"] for b in bar_starts],
        gt_chords=[{k: c[k] for k in ("t0", "t1", "root_pc", "quality", "bass_pc", "label")}
                   for c in gt_chords],
        verified=False,
        proposal=dict(
            aggregate=agg,
            transpose=tr, form=fo, bar1_anchor=an,
            section_alignment=dict(
                sections=[dict(chorus=pl.chorus, sec_idx=pl.sec_idx, label=sv["label"],
                               occ=pl.occ, t0=sv["t0"], t1=sv["t1"],
                               agreement=sv["agr"], offset_s=sv["offset_s"])
                          for pl, sv in zip(placements, sections_view)],
                n_placed=diag["n_placed"], coverage=round(diag["coverage"], 3),
                avg_agreement=round(diag["avg_agreement"], 3),
                start_candidates=diag["start"].get("candidates", []),
                start_margin=round(diag["start"].get("start_margin", 0.0), 3),
                gaps_unlabeled=_describe_gaps(sections_view)),
            refinement=refinement,
            harmonic_agreement=region,
            # machine-readable score bundle for the (separate) CALIBRATION step
            # that will learn the absolute aligned-vs-mismatched threshold from
            # deliberately-corrupted variants + Louis's batch-1 ear-corrections.
            agreement_detail=dict(
                signal="per-beat Pearson(audio CQT chroma, chart chord-tones); "
                       "non-circular (never the model decode)",
                whole_song=round(region["overall"], 4),
                worst_region=dict(agr=region["worst"], t=region["worst_time"],
                                  label=region["worst_label"]),
                per_transpose=tr["_all"],
                per_section=[dict(label=pl.label, t0=sv["t0"], t1=sv["t1"],
                                  agr=sv["agr"]) for pl, sv in
                             zip(placements, sections_view)],
                per_chord=[dict(t0=c["t0"], t1=c["t1"], label=c["label"],
                                agr=c.get("agr")) for c in gt_chords],
                start_candidates=diag["start"].get("candidates", []),
                start_margin=round(diag["start"].get("start_margin", 0.0), 3)),
            boundary_flux=flux,
            beat_this=dict(n_beats=len(beats), n_downbeats=len(downbeats),
                           beat_regularity=round(bt["beat_regularity"], 3),
                           downbeat_regularity=round(bt["downbeat_regularity"], 3),
                           beat_period=round(beat_period, 3),
                           bar_period=round(bar_period, 3),
                           est_bpm=round(est_bpm, 1) if beat_period else None),
            audio_duration_s=round(dur, 2),
            unmapped_quality_tokens=sorted(set(chart.unmapped)),
            builder="scripts/brick0_propose.py (constant-tempo grid + fine-tempo gap-discipline + piecewise-per-section tempo-drift, aligner v5, 2026-07-22)",
        ),
    )
    gt_path = GOLDEN / f"{song['song_id']}.gt.json"
    gt_path.write_text(json.dumps(gt_json, indent=2))
    log.info("  wrote %s (%d chords, agreement r=%.3f, worst r=%.3f@%.0fs)",
             gt_path.relative_to(REPO), len(gt_chords), region["overall"],
             region["worst"] if region["worst"] is not None else 0.0,
             region["worst_time"] or 0.0)

    # ---- write verification HTML ----
    audio_b64 = base64.b64encode(audio.read_bytes()).decode()
    html = build_html(song, gt_chords, sections_view,
                      dict(transpose=tr, form=fo, anchor=an), agg, bar_starts,
                      grid_str, region, audio_b64)
    html_path = REVIEW / f"{song['song_id']}.html"
    html_path.write_text(html)
    log.info("  wrote %s (%.1f MB)", html_path.relative_to(REPO), len(html) / 1e6)

    return dict(
        song_id=song["song_id"], title=song["title"], audio_path=song["audio"],
        ireal_file=song["ireal_file"], tune_title=song["tune_title"],
        transpose=tr, form={k: fo[k] for k in
                            ("n_choruses", "one_chorus_form", "confidence",
                             "alternative", "evidence")},
        bar1_anchor=an, aggregate=agg, n_chords=len(gt_chords),
        audio_duration_s=round(dur, 2), grid=grid_str, region=region,
        flux=flux, coverage=round(diag["coverage"], 3),
        refinement=refinement,
        gt_path=str(gt_path.relative_to(REPO)),
        html_path=str(html_path.relative_to(REPO)),
    )


def _describe_gaps(sections_view: list[dict]) -> list[dict]:
    gaps = []
    for a, b in zip(sections_view[:-1], sections_view[1:]):
        if b["t0"] - a["t1"] > 0.4:
            gaps.append(dict(after=a["label"], before=b["label"],
                             t0=a["t1"], t1=b["t0"], dur=round(b["t0"] - a["t1"], 1)))
    return gaps


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="brick0_"))
    rows = []
    for song in BATCH1:
        try:
            rows.append(process(song, workdir))
        except Exception as exc:  # noqa: BLE001
            log.exception("FAILED %s: %s", song["song_id"], exc)
    rows.sort(key=lambda r: r["aggregate"]["aggregate_confidence"])  # low -> high

    MANIFEST.write_text(json.dumps(dict(
        batch="brick0_batch1", n_songs=len(rows), verified=False,
        note="ALL verified=false — machine PROPOSALS for human ear-verification "
             "(section-based re-proposal, 2026-07-22). Sorted ascending by "
             "aggregate confidence (weakest first).",
        songs=rows), indent=2))
    log.info("wrote manifest %s", MANIFEST.relative_to(REPO))

    write_queue_md(rows)
    write_index_html(rows)
    return 0


def write_queue_md(rows: list[dict]) -> None:
    lines = ["# Brick 0 — batch 1 verification QUEUE (verified=false)",
             "",
             "Machine PROPOSALS for Louis to hand-verify by ear — **SECTION-BASED "
             "re-proposal** (2026-07-22). **Nothing is frozen/verified.** Sorted "
             "ascending by aggregate confidence (= min of transpose/form/anchor, "
             "tempered by harmonic agreement) — **do the top ones first**.",
             "",
             "The alignment now deconstructs the chart into SECTIONS and fits each "
             "where the raw-audio chroma agrees best (non-circular; never the "
             "model's decode), skipping the intro and leaving turnaround/vamp gaps "
             "UNLABELED. `harmonic agreement r` (per-beat Pearson of audio chroma "
             "vs chart chord-tones) is the self-diagnostic: high+stable = well "
             "aligned; a local drop = a misaligned region needing your ear.",
             ""]
    for i, r in enumerate(rows, 1):
        a = r["aggregate"]
        tr, fo, an = r["transpose"], r["form"], r["bar1_anchor"]
        rq = r["region"]
        fl = r["flux"].get("boundary_flux_agree")
        lines += [
            f"## {i}. {r['title']}  —  aggregate **{a['aggregate_confidence']:.2f}**"
            f"  (weak: **{a['weak_field']}**)",
            f"- audio {r['audio_duration_s']:.0f}s • {r['n_chords']} GT chords • "
            f"coverage {int(r['coverage']*100)}% • HTML `{r['html_path']}` • GT `{r['gt_path']}`",
            f"- **harmonic agreement** overall r={rq['overall']} • worst r={rq['worst']} "
            f"@ {rq['worst_time']}s ({rq['worst_label']}) • {int(rq['frac_low']*100)}% low"
            + (f" • boundary-flux {fl}" if fl is not None else ""),
            f"- grid: {r['grid']}",
            f"- **transpose** {tr['transpose_semitones']:+d}, conf {tr['confidence']:.2f}; "
            f"alt {tr['alternative']['transpose']:+d} — {tr['evidence']}",
            f"- **form/section-fit** {fo['one_chorus_form']} x{fo['n_choruses']}, "
            f"conf {fo['confidence']:.2f} — {fo['evidence']}",
            f"- **bar-1 anchor / intro-skip** t={an['bar1_anchor_time']:.2f}s, "
            f"conf {an['confidence']:.2f}; alt t={an['alternative']['bar1_anchor_time']:.2f}s "
            f"— {an['evidence']}",
            "",
        ]
    (REVIEW / "_QUEUE.md").write_text("\n".join(lines))
    log.info("wrote %s", (REVIEW / "_QUEUE.md").relative_to(REPO))


def write_index_html(rows: list[dict]) -> None:
    def cls(c):
        return "low" if c < 0.4 else ("mid" if c < 0.6 else "ok")
    trs = []
    for i, r in enumerate(rows, 1):
        a = r["aggregate"]; c = a["aggregate_confidence"]
        rq = r["region"]; tr = r["transpose"]; an = r["bar1_anchor"]
        prop = (f"{tr['transpose_semitones']:+d} → {NOTE_SHARP[tr['transpose_semitones']]}, "
                f"start {an['bar1_anchor_time']:.0f}s, agree r={rq['overall']}")
        flag = ""
        if rq["overall"] < 0.25 or c < 0.4:
            flag = " <span class='flag'>ear needed</span>"
        trs.append(
            f"<tr><td class='n'>{i}</td>"
            f"<td><a class='song' href='./{r['song_id']}.html'>{r['title']}</a></td>"
            f"<td class='conf {cls(c)}'>{c:.2f}</td>"
            f"<td class='weak'>{a['weak_field']}</td>"
            f"<td class='prop'>{prop}{flag}</td>"
            f"<td class='prop'>worst r={rq['worst']} @ {rq['worst_time']}s</td></tr>")
    body = _INDEX_TEMPLATE.replace("__ROWS__", "\n".join(trs))
    (REVIEW / "index.html").write_text(body)
    log.info("wrote %s", (REVIEW / "index.html").relative_to(REPO))


_INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brick 0 — Batch 1 Verification Queue</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 2rem clamp(1rem,4vw,3rem); max-width: 1150px; margin-inline: auto;
         background: #faf9f7; color: #1c1a17; }
  @media (prefers-color-scheme: dark) { body { background:#17150f; color:#ece7dd; } }
  h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
  .sub { opacity: .7; margin: 0 0 1.5rem; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: .55rem .7rem; border-bottom: 1px solid rgba(128,128,128,.25); vertical-align: top; }
  th { font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; opacity: .65; }
  td.n { font-variant-numeric: tabular-nums; opacity:.5; width:1.5rem; }
  a.song { font-weight: 600; font-size: 1.05rem; text-decoration: none; color: inherit; border-bottom: 2px solid #c98a2b; }
  a.song:hover { background: rgba(201,138,43,.15); }
  .conf { font-variant-numeric: tabular-nums; font-weight: 700; }
  .low { color: #c0392b; } .mid { color: #c98a2b; } .ok { color: #2e8b57; }
  .weak { font-size:.85rem; opacity:.8; }
  .flag { color:#c0392b; font-weight:600; }
  .prop { font-size:.9rem; opacity:.85; }
  .box { background: rgba(201,138,43,.10); border-left: 3px solid #c98a2b; padding: .8rem 1rem; border-radius: 4px; margin: 1.5rem 0; }
  .box h2 { font-size: .95rem; margin: 0 0 .4rem; }
  code { background: rgba(128,128,128,.18); padding: .1rem .35rem; border-radius: 3px; font-size: .88em; }
</style></head>
<body>
<h1>🎧 Brick 0 — Batch 1 Verification Queue (section-based)</h1>
<p class="sub">8 songs, all <code>verified: false</code>. The chart is now fitted SECTION-BY-SECTION to the audio (intro skipped, vamp gaps left unlabeled), driven + measured by non-circular <strong>harmonic agreement</strong> (raw-audio chroma vs chart chord-tones — never the model). Sorted <strong>lowest-confidence first</strong>. Click a song to open its listen-and-verify page; chord cells are colour-coded by agreement so your ear goes to the weak spots.</p>
<table>
<thead><tr><th class="n">#</th><th>Song</th><th>Conf</th><th>Weak field</th><th>Proposal</th><th>Worst region</th></tr></thead>
<tbody>
__ROWS__
</tbody>
</table>
<div class="box">
<h2>How to read this</h2>
<p><strong>harmonic agreement r</strong> = per-beat Pearson correlation of the raw audio chroma against the chart chord's pitch-classes, averaged over each labeled span. High + stable ⇒ the chart lands on the music. A local drop (the "worst region") ⇒ a misaligned or genuinely ambiguous spot — that's where your ear is needed. Dashed strips on a song page are unlabeled gaps (intro / turnaround vamp).</p>
<p><strong>To sign off:</strong> for each song tell me <code>accept</code> or <code>correct → &lt;what&gt;</code>. On <em>accept</em> I flip <code>verified: true</code> and freeze that <code>golden/brick0/&lt;song&gt;.gt.json</code>; on a correction I re-propose. Only <code>verified: true</code> songs enter the scored benchmark.</p>
</div>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
