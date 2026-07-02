"""
Empirical chord-duration prior, fit from POP909 ground truth.

The chord HMM's only duration signal is `self_transition_boost`, a constant
added to the transition matrix's diagonal. That implies a **geometric**
distribution over how long a chord lasts (constant per-beat hazard of
switching, independent of how long you've already stayed) — memoryless,
unlike real harmonic rhythm, which clusters around a typical duration.

This module fits the *actual* empirical distribution directly from ground
truth instead of assuming a shape, for use by a duration-aware (semi-Markov
style) decoder. See docs/known_issues.md #1 for the full write-up.

Fit from all 909 POP909 songs' text annotations (~125k chord events) — no
audio or model inference needed, so this is cheap and can use the full
dataset even though only 5 songs have rendered audio for pipeline testing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from harmonia.data.pop909_parser import ChordEvent, POP909Parser
from harmonia.theory.chord_vocabulary import ChordQuality

logger = logging.getLogger(__name__)

DEFAULT_MAX_DURATION_BEATS = 32


def _duration_beats(ev: ChordEvent, beat_times: np.ndarray) -> int | None:
    """
    Convert a ChordEvent's (start, end) — already in **seconds**, per the
    known POP909Parser field-naming gotcha, not beat indices — to a duration
    in beats via that song's own beat grid (beat_midi.txt, exact).

    Returns None if the event collapses to zero beats (annotation artifact).
    """
    start_idx = int(np.searchsorted(beat_times, ev.start_beat, side="left"))
    end_idx = int(np.searchsorted(beat_times, ev.end_beat, side="left"))
    d = end_idx - start_idx
    return d if d >= 1 else None


def fit_duration_prior(
    pop909_dir: Path,
    max_duration_beats: int = DEFAULT_MAX_DURATION_BEATS,
) -> dict[str, np.ndarray]:
    """
    Fit empirical PMFs over chord duration in beats, separately for real
    chords and for NO_CHORD (N durations behave differently — typically
    short intros/outros/bridges — so are tracked separately).

    Args:
        pop909_dir:         path to data/pop909/POP909 (all 909 songs).
        max_duration_beats:  durations longer than this are clipped into the
                             last bin (rare — see fit stats logged at INFO).

    Returns:
        {"chord": (max_duration_beats,) PMF, "no_chord": (max_duration_beats,) PMF}
        Index d (0-based) represents duration = d+1 beats.
    """
    parser = POP909Parser(pop909_dir)
    songs = parser.parse_all(require_audio=False)

    chord_durations: list[int] = []
    no_chord_durations: list[int] = []
    clipped = 0

    for song in songs:
        if len(song.beat_times) == 0:
            continue
        for ev in song.chord_events:
            d = _duration_beats(ev, song.beat_times)
            if d is None:
                continue
            if d > max_duration_beats:
                clipped += 1
            d = min(d, max_duration_beats)
            if ev.quality == ChordQuality.NO_CHORD:
                no_chord_durations.append(d)
            else:
                chord_durations.append(d)

    logger.info(
        f"Fit duration prior from {len(songs)} songs: "
        f"{len(chord_durations)} chord events, {len(no_chord_durations)} N events "
        f"({clipped} clipped to {max_duration_beats} beats)"
    )
    if chord_durations:
        logger.info(
            f"  chord duration: mean={np.mean(chord_durations):.2f} beats, "
            f"median={np.median(chord_durations):.1f}"
        )

    def to_pmf(durations: list[int]) -> np.ndarray:
        counts = np.zeros(max_duration_beats, dtype=np.float64)
        for d in durations:
            counts[d - 1] += 1
        total = counts.sum()
        if total == 0:
            return np.ones(max_duration_beats) / max_duration_beats
        return counts / total

    return {
        "chord": to_pmf(chord_durations),
        "no_chord": to_pmf(no_chord_durations),
    }


def load_or_fit_duration_prior(
    pop909_dir: Path,
    cache_path: Path | None = None,
    max_duration_beats: int = DEFAULT_MAX_DURATION_BEATS,
) -> dict[str, np.ndarray]:
    """Same as fit_duration_prior, cached to an .npz so it's not refit on
    every pipeline run (fitting itself only takes ~1s, but this keeps
    startup deterministic and avoids re-parsing 909 files repeatedly)."""
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        return {"chord": data["chord"], "no_chord": data["no_chord"]}

    prior = fit_duration_prior(pop909_dir, max_duration_beats)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, chord=prior["chord"], no_chord=prior["no_chord"])

    return prior


def log_duration_prior(pmf: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """log(PMF) with a small floor so unseen durations aren't -inf."""
    return np.log(np.clip(pmf, floor, None))
