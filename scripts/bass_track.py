"""
Shared utilities for bass-note and rolling-key-track inference.

Exploratory, not wired into the pipeline: the working hypothesis (2026-07
discussion, see docs/known_issues.md #1) is that bass motion carries useful
signal about *when* a chord actually changes, distinct from *what* it
changes to. A walking bass line moves every beat without necessarily
implying a new chord; a bass note landing on a new pitch class in step with
a change elsewhere is more likely a real chord change. This module builds
the two raw per-beat observations needed to check that empirically
(inferred bass note, rolling key estimate) -- nothing here feeds
harmonia/models/chord_hmm.py yet. See scripts/plot_bass_and_key_tracks.py
(per-song visual check) and scripts/analyze_bass_patterns.py (cross-song
empirical distributions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIDI_START = 21  # A0, piano key index 0


# ---------------------------------------------------------------------------
# Bass note inference
# ---------------------------------------------------------------------------

def infer_bass_track(
    beat_probs_onset: np.ndarray,   # (B, 88) -- quantised from onset_probs, NOT
                                     # note_probs. note_probs is a near-constant,
                                     # barely-discriminating sustain signal across
                                     # nearly the whole 88-key range (the exact
                                     # pitfall already documented in
                                     # docs/known_issues.md's "Resolved" section --
                                     # confirmed again here: every register lit up
                                     # within a narrow 3x band, so a relative
                                     # threshold on it locks onto the lowest piano
                                     # key almost every beat instead of a real bass
                                     # note). onset_probs is sparse and genuinely
                                     # discriminative, same reason the rest of the
                                     # pipeline uses it for beat_probs.
    threshold_frac: float = 0.15,
    min_abs: float = 0.3,
) -> np.ndarray:
    """
    Per-beat inferred bass note: the lowest piano key whose onset activation
    clears threshold_frac * (that beat's own peak activation), with an
    absolute floor (min_abs) to reject beats with no real onset at all.

    Returns (B,) int array of MIDI note numbers, or -1 where no key clears
    the threshold (silence, or a held note with no new onset this beat --
    a real limitation of onset-based detection, not a bug: a bass note that
    sustains without being re-struck won't be picked up on the beats after
    its initial onset).
    """
    B, n_keys = beat_probs_onset.shape
    bass_midi = np.full(B, -1, dtype=int)
    for b in range(B):
        row = beat_probs_onset[b]
        peak = row.max()
        if peak < min_abs:
            continue
        thresh = max(min_abs, threshold_frac * peak)
        active = np.nonzero(row >= thresh)[0]
        if len(active) == 0:
            continue
        bass_midi[b] = MIDI_START + int(active[0])
    return bass_midi


def infer_bass_track_learned(
    beat_probs_onset: np.ndarray,
    threshold_frac: float = 0.15,
    min_abs: float = 0.3,
    min_gap_semitones: int = 0,
    max_bass_midi: int = 68,
) -> np.ndarray:
    """
    Bass detector with thresholds learned from real ground truth (POP909's
    PIANO/accompaniment track -- see scripts/learn_bass_distribution.py),
    rather than guessed. Two candidate corrections were tested against
    ground truth across all 5 available songs; only one held up:

    1. Register ceiling (max_bass_midi, default 68): genuine bass notes in
       this corpus stay within MIDI 37-61 (C#2-C#4), never higher --
       max_bass_midi=68 gives comfortable margin above the observed range
       while still rejecting implausibly-high false candidates. Low risk,
       no measured downside.
    2. Isolation gap (min_gap_semitones, default 0 = disabled): the
       hypothesis was that a real bass note sits further below the next
       note up than the bottom of a closely-voiced chord does -- and
       ground truth *does* show a real difference in the predicted
       direction (median gap 5 semitones when a true bass note is present
       vs 2 when it's genuinely absent). But a full grid search over
       (ceiling, min_gap) against ground truth found no nonzero min_gap
       that improved pitch-class match rate or no-bass detection over
       min_gap=0 (i.e. no filtering) -- see
       docs/plots/inference/bass_patterns/bass_detector_v1_vs_v2.png. Two
       likely reasons: true "no bass" beats are rare in this corpus (7 out
       of 1584 beats, 0.4% -- not enough real negatives to learn a
       reliable single-beat threshold from), and the dominant error mode
       turned out to be something a gap threshold can't fix at all: even
       when there IS a fresh onset and a real bass note present, the
       audio's lowest active key names the wrong pitch class about half
       the time (52% match rate) -- that's a raw pitch-detection accuracy
       problem in the bass register, not a "confused with a nearby chord
       tone" problem this kind of post-hoc filtering can correct. Kept as
       a parameter (not deleted) in case a larger ground-truth sample
       changes this conclusion later.
    """
    B, n_keys = beat_probs_onset.shape
    bass_midi = np.full(B, -1, dtype=int)
    for b in range(B):
        row = beat_probs_onset[b]
        peak = row.max()
        if peak < min_abs:
            continue
        thresh = max(min_abs, threshold_frac * peak)
        active = np.nonzero(row >= thresh)[0]
        if len(active) == 0:
            continue
        candidate = MIDI_START + int(active[0])
        if candidate > max_bass_midi:
            continue
        if len(active) >= 2:
            gap = int(active[1]) - int(active[0])
            if gap < min_gap_semitones:
                continue
        bass_midi[b] = candidate
    return bass_midi


def forward_fill_bass(bass_midi: np.ndarray, max_gap: int = 8) -> np.ndarray:
    """
    Carry the last confidently-detected bass note forward across beats with
    no new onset (a held note is presumed to still be sounding until a new
    onset supersedes it), up to `max_gap` beats -- beyond that, a gap this
    long more likely means real silence than an unusually long held note, so
    it's left as -1 rather than fabricating a bass note out of nothing.
    """
    filled = bass_midi.copy()
    last, gap = -1, 0
    for b in range(len(filled)):
        if filled[b] >= 0:
            last, gap = filled[b], 0
        elif last >= 0 and gap < max_gap:
            filled[b] = last
            gap += 1
        else:
            gap += 1
    return filled


# ---------------------------------------------------------------------------
# Ground-truth bass from symbolic MIDI (POP909's PIANO track), for learning
# and validating the audio-based detector above -- not used at inference
# time (the real pipeline only ever has audio), only to calibrate it.
# ---------------------------------------------------------------------------

def _load_track_notes(midi_path, track_name: str | None) -> list[tuple[float, float, int]]:
    """
    (start_s, end_s, pitch) for one named POP909 track, or all non-drum
    tracks combined if track_name is None. POP909 MIDI files have exactly
    three non-drum instruments: MELODY (main tune), BRIDGE (secondary/
    counter-melody), PIANO (the full accompaniment -- this is the track
    that carries any real bass line, since MELODY/BRIDGE are foreground
    melodic voices, not harmony/bass).
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if track_name is None:
        notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                notes.extend((n.start, n.end, n.pitch) for n in inst.notes)
        return notes

    inst = next((i for i in pm.instruments
                 if not i.is_drum and i.name and i.name.strip().upper() == track_name), None)
    if inst is None:
        # Fallback: lowest-average-pitch non-drum track (covers MIDI files
        # that don't use POP909's standard track naming).
        candidates = [i for i in pm.instruments if not i.is_drum and i.notes]
        if not candidates:
            return []
        inst = min(candidates, key=lambda i: np.mean([n.pitch for n in i.notes]))
    return [(n.start, n.end, n.pitch) for n in inst.notes]


def _notes_active_per_beat(notes: list[tuple[float, float, int]], beat_times: np.ndarray) -> list[list[int]]:
    """For each beat interval [beat_times[b], beat_times[b+1)), the pitches of any note overlapping it."""
    B = len(beat_times)
    result: list[list[int]] = [[] for _ in range(B)]
    if B == 0:
        return result
    beat_dur = beat_times[-1] - beat_times[-2] if B > 1 else 1.0
    for b in range(B):
        t0 = beat_times[b]
        t1 = beat_times[b + 1] if b + 1 < B else t0 + beat_dur
        for (s, e, p) in notes:
            if s < t1 and e > t0:
                result[b].append(p)
    return result


@dataclass
class TrueBassBeat:
    true_bass: int          # lowest PIANO-track pitch this beat, or -1 if PIANO is silent (no bass)
    gap_to_next: int | None # semitones from true_bass to the next-lowest pitch across ALL tracks
                             # (None if true_bass is -1, or no other note sounds this beat)


def true_bass_track(midi_path, beat_times: np.ndarray) -> list[TrueBassBeat]:
    """
    Ground-truth bass per beat, from POP909's symbolic MIDI rather than
    audio: the PIANO (accompaniment) track's lowest note, or -1 when PIANO
    has no note sounding at all this beat -- a real, unambiguous "no bass"
    case (e.g. a bar where only the melody plays solo), which "lowest
    active audio bin" can never represent since something is almost always
    faintly active somewhere in real audio.
    """
    piano_notes = _load_track_notes(midi_path, "PIANO")
    all_notes = _load_track_notes(midi_path, None)
    piano_active = _notes_active_per_beat(piano_notes, beat_times)
    all_active = _notes_active_per_beat(all_notes, beat_times)

    out = []
    for piano_pitches, all_pitches in zip(piano_active, all_active):
        if not piano_pitches:
            out.append(TrueBassBeat(true_bass=-1, gap_to_next=None))
            continue
        bass = min(piano_pitches)
        above = sorted(p for p in all_pitches if p > bass)
        gap = (above[0] - bass) if above else None
        out.append(TrueBassBeat(true_bass=bass, gap_to_next=gap))
    return out


# ---------------------------------------------------------------------------
# Rolling key track
# ---------------------------------------------------------------------------

@dataclass
class KeyTrackPoint:
    tonic: int
    mode: str
    key_name: str
    confidence: float


def rolling_key_track(beat_probs: np.ndarray, window: int = 8) -> list[KeyTrackPoint]:
    """
    Per-beat rolling key estimate: a centred window of +/- `window` beats,
    each beat's chroma L1-normalised individually before summing (same
    evidence-counting convention as structure.py::_make_segment -- each
    beat counts as one unit of evidence, not however loud/polyphonic it
    happens to be -- see docs/known_issues.md #0), then infer_key() on the
    windowed sum. This is *not* how the pipeline infers key (that's one
    estimate per structural segment) -- it's a smoother, denser diagnostic
    signal for visually checking key stability beat-by-beat.
    """
    from harmonia.models.structure import _beat_chroma
    from harmonia.theory.key_profiles import infer_key

    per_beat_chroma = _beat_chroma(beat_probs, norm="l1")  # (B, 12)
    B = per_beat_chroma.shape[0]
    track: list[KeyTrackPoint] = []
    for b in range(B):
        lo, hi = max(0, b - window), min(B, b + window + 1)
        chroma = per_beat_chroma[lo:hi].sum(axis=0)
        kp = infer_key(chroma)
        track.append(KeyTrackPoint(kp.tonic, kp.mode, kp.key_name, kp.confidence))
    return track


# ---------------------------------------------------------------------------
# Run-length encoding (shared by plotting and the run-length distribution)
# ---------------------------------------------------------------------------

@dataclass
class Run:
    start_b: int
    end_b: int          # exclusive
    label: str
    extra: dict = field(default_factory=dict)

    @property
    def n_beats(self) -> int:
        return self.end_b - self.start_b


def compress_to_runs(labels: list, extras: list[dict] | None = None) -> list[Run]:
    """Run-length encode a per-beat categorical sequence into contiguous Runs."""
    runs: list[Run] = []
    if not labels:
        return runs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            extra = extras[start] if extras else {}
            runs.append(Run(start, i, labels[start], extra))
            start = i
    return runs
