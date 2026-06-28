"""
Scale-agnostic jazz priors for harmonic inference.

All progressions are encoded as sequences of (interval_from_tonic, ChordQuality)
pairs — purely relative. They are instantiated at inference time by transposing
to the inferred key. This means one prior covers all 12 transpositions.

The model is also tempo-aware: harmonic rhythm (how often chords change)
depends on style × tempo. A II-V-I at 280 BPM bebop spans 2 beats;
the same progression in a ballad at 60 BPM spans 8 bars.

Architecture:
    StylePrior  →  P(tempo)
                →  P(harmonic_rhythm | tempo)
                →  P(progression | style)         [scale-agnostic]
                →  instantiate to absolute chords given inferred key
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from harmonia.theory.chord_vocabulary import ChordQuality


# ---------------------------------------------------------------------------
# Relative chord: interval from tonic (semitones) + quality
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelativeChord:
    """
    A chord expressed relative to the tonic.

    interval: semitones above the tonic (0–11).
              0  = I  (tonic)
              2  = II (supertonic)
              4  = III (mediant)
              5  = IV (subdominant)
              7  = V  (dominant)
              9  = VI (submediant)
              11 = VII (leading tone)
              — chromatic degrees —
              1  = bII  (Neapolitan / tritone sub root)
              3  = bIII (minor / modal interchange)
              6  = bV / #IV (tritone of I)
              8  = bVI (modal interchange)
              10 = bVII (subtonic, Mixolydian)
    quality: ChordQuality
    """
    interval: int   # semitones from tonic, 0–11
    quality: ChordQuality

    def to_absolute(self, tonic: int) -> tuple[int, ChordQuality]:
        """Instantiate to absolute (root, quality) given tonic pitch class."""
        return ((tonic + self.interval) % 12, self.quality)

    def __repr__(self) -> str:
        degree = _INTERVAL_TO_ROMAN.get(self.interval, f"+{self.interval}")
        return f"{degree}{self.quality.value}"


_INTERVAL_TO_ROMAN: dict[int, str] = {
    0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III",
    5: "IV", 6: "bV", 7: "V", 8: "bVI", 9: "VI",
    10: "bVII", 11: "VII",
}


# ---------------------------------------------------------------------------
# Relative progression
# ---------------------------------------------------------------------------

@dataclass
class RelativeProgression:
    """
    A named harmonic progression expressed in relative terms.

    chords:          ordered sequence of RelativeChords.
    weight:          prior probability weight (unnormalised). Higher = more likely.
    typical_beats:   list of beat-counts for each chord slot. Tempo-scaled at inference.
                     None means infer from harmonic rhythm prior.
    style_tags:      which musical styles this progression is common in.
    description:     human-readable explanation.
    """
    name: str
    chords: tuple[RelativeChord, ...]
    weight: float
    typical_beats: tuple[float, ...] | None = None
    style_tags: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def instantiate(self, tonic: int) -> list[tuple[int, ChordQuality]]:
        """Return list of (root, quality) pairs for the given tonic."""
        return [rc.to_absolute(tonic) for rc in self.chords]


# ---------------------------------------------------------------------------
# Core jazz/harmonic progressions (all scale-agnostic)
# ---------------------------------------------------------------------------

_RC = RelativeChord  # alias

PROGRESSIONS: dict[str, RelativeProgression] = {

    # ------------------------------------------------------------------ #
    # II-V-I family — the backbone of jazz                                #
    # ------------------------------------------------------------------ #

    "ii_V_I_major": RelativeProgression(
        name="ii-V-I (major)",
        chords=(
            _RC(2,  ChordQuality.MIN7),   # iim7
            _RC(7,  ChordQuality.DOM7),   # V7
            _RC(0,  ChordQuality.MAJ7),   # Imaj7
        ),
        weight=1.0,
        typical_beats=(2.0, 2.0, 4.0),
        style_tags=frozenset({"jazz", "bop", "swing", "latin"}),
        description="The most fundamental jazz cadence. iim7 → V7 → Imaj7.",
    ),

    "ii_V_I_minor": RelativeProgression(
        name="ii-V-I (minor)",
        chords=(
            _RC(2,  ChordQuality.HALF_DIM7),  # iiø7
            _RC(7,  ChordQuality.DOM7),        # V7 (or 7b9)
            _RC(0,  ChordQuality.MIN7),        # im7 (or imMaj7)
        ),
        weight=0.85,
        typical_beats=(2.0, 2.0, 4.0),
        style_tags=frozenset({"jazz", "bop", "swing", "latin"}),
        description="Minor ii-V-I. iiø7 → V7(b9) → im7.",
    ),

    "ii_V_only": RelativeProgression(
        name="ii-V (unresolved)",
        chords=(
            _RC(2,  ChordQuality.MIN7),
            _RC(7,  ChordQuality.DOM7),
        ),
        weight=0.65,
        typical_beats=(2.0, 2.0),
        style_tags=frozenset({"jazz", "bop"}),
        description="Unresolved ii-V — common on A sections before resolution.",
    ),

    # ------------------------------------------------------------------ #
    # Tritone substitution                                                #
    # ------------------------------------------------------------------ #

    "tritone_sub_major": RelativeProgression(
        name="Tritone sub → I (major)",
        chords=(
            _RC(2,  ChordQuality.MIN7),   # iim7
            _RC(1,  ChordQuality.DOM7),   # bII7 (tritone sub of V7)
            _RC(0,  ChordQuality.MAJ7),   # Imaj7
        ),
        weight=0.7,
        typical_beats=(2.0, 2.0, 4.0),
        style_tags=frozenset({"jazz", "bop", "reharmonisation"}),
        description="bII7 shares the 3rd and 7th with V7 (enharmonic). "
                    "Chromatic bass descent: II → bII → I.",
    ),

    "tritone_sub_bare": RelativeProgression(
        name="Tritone sub direct (bII7 → I)",
        chords=(
            _RC(1,  ChordQuality.DOM7),   # bII7
            _RC(0,  ChordQuality.MAJ7),   # Imaj7
        ),
        weight=0.5,
        typical_beats=(2.0, 4.0),
        style_tags=frozenset({"jazz", "reharmonisation"}),
        description="Direct tritone substitution without preceding ii.",
    ),

    # ------------------------------------------------------------------ #
    # Cycle of fifths                                                     #
    # ------------------------------------------------------------------ #

    "cycle_of_fifths_4": RelativeProgression(
        name="Cycle of fifths (4 chords)",
        chords=(
            _RC(9,  ChordQuality.MIN7),   # vim7
            _RC(2,  ChordQuality.MIN7),   # iim7
            _RC(7,  ChordQuality.DOM7),   # V7
            _RC(0,  ChordQuality.MAJ7),   # Imaj7
        ),
        weight=0.75,
        typical_beats=(2.0, 2.0, 2.0, 2.0),
        style_tags=frozenset({"jazz", "pop", "swing", "standards"}),
        description="Root movement by descending 5ths (ascending 4ths). "
                    "Rhythm changes B section archetype.",
    ),

    "rhythm_changes_bridge": RelativeProgression(
        name="Rhythm changes bridge (III-VI-II-V)",
        chords=(
            _RC(4,  ChordQuality.DOM7),   # III7 (V/vi)
            _RC(9,  ChordQuality.DOM7),   # VI7 (V/ii)
            _RC(2,  ChordQuality.DOM7),   # II7 (V/V)
            _RC(7,  ChordQuality.DOM7),   # V7
        ),
        weight=0.6,
        typical_beats=(2.0, 2.0, 2.0, 2.0),
        style_tags=frozenset({"jazz", "bop", "standards"}),
        description="All dominant 7ths in cycle of fifths. "
                    "III7 → VI7 → II7 → V7, then resolves to I.",
    ),

    # ------------------------------------------------------------------ #
    # Secondary dominants                                                 #
    # ------------------------------------------------------------------ #

    "secondary_dom_to_ii": RelativeProgression(
        name="V/ii → ii",
        chords=(
            _RC(4,  ChordQuality.DOM7),   # III7 = V/ii
            _RC(2,  ChordQuality.MIN7),   # iim7
        ),
        weight=0.55,
        typical_beats=(2.0, 2.0),
        style_tags=frozenset({"jazz", "pop", "standards"}),
        description="Secondary dominant tonicising the ii chord.",
    ),

    "secondary_dom_to_IV": RelativeProgression(
        name="V/IV → IV",
        chords=(
            _RC(0,  ChordQuality.DOM7),   # I7 = V/IV
            _RC(5,  ChordQuality.MAJ7),   # IVmaj7
        ),
        weight=0.5,
        typical_beats=(2.0, 2.0),
        style_tags=frozenset({"jazz", "blues", "gospel"}),
        description="I7 treating IV as temporary tonic. Very common in blues/gospel.",
    ),

    # ------------------------------------------------------------------ #
    # Modal interchange / borrowed chords                                #
    # ------------------------------------------------------------------ #

    "borrowed_bVII": RelativeProgression(
        name="bVII → I (Mixolydian borrow)",
        chords=(
            _RC(10, ChordQuality.MAJOR),  # bVII
            _RC(0,  ChordQuality.MAJOR),  # I
        ),
        weight=0.5,
        typical_beats=(2.0, 2.0),
        style_tags=frozenset({"pop", "rock", "modal"}),
        description="Borrowed from Mixolydian. Very common in pop and rock.",
    ),

    "borrowed_bVI": RelativeProgression(
        name="bVI → bVII → I",
        chords=(
            _RC(8,  ChordQuality.MAJOR),  # bVI
            _RC(10, ChordQuality.MAJOR),  # bVII
            _RC(0,  ChordQuality.MAJOR),  # I
        ),
        weight=0.45,
        typical_beats=(2.0, 2.0, 4.0),
        style_tags=frozenset({"pop", "rock", "modal"}),
        description="Classic pop / rock plagal cadence chain.",
    ),

    "iv_minor_borrow": RelativeProgression(
        name="iv minor (borrowed from parallel minor)",
        chords=(
            _RC(5,  ChordQuality.MINOR),  # iv (minor subdominant)
            _RC(0,  ChordQuality.MAJ7),   # Imaj7
        ),
        weight=0.5,
        typical_beats=(2.0, 4.0),
        style_tags=frozenset({"jazz", "pop", "gospel"}),
        description="Minor iv → I. Strong emotional pull, borrowed from parallel minor.",
    ),

    # ------------------------------------------------------------------ #
    # Blues                                                               #
    # ------------------------------------------------------------------ #

    "blues_I_IV_V": RelativeProgression(
        name="12-bar blues (I-IV-V)",
        chords=(
            _RC(0,  ChordQuality.DOM7),   # I7
            _RC(0,  ChordQuality.DOM7),   # I7
            _RC(0,  ChordQuality.DOM7),   # I7
            _RC(0,  ChordQuality.DOM7),   # I7
            _RC(5,  ChordQuality.DOM7),   # IV7
            _RC(5,  ChordQuality.DOM7),   # IV7
            _RC(0,  ChordQuality.DOM7),   # I7
            _RC(0,  ChordQuality.DOM7),   # I7
            _RC(7,  ChordQuality.DOM7),   # V7
            _RC(5,  ChordQuality.DOM7),   # IV7
            _RC(0,  ChordQuality.DOM7),   # I7
            _RC(7,  ChordQuality.DOM7),   # V7 (turnaround)
        ),
        weight=0.7,
        typical_beats=(4.0,) * 12,
        style_tags=frozenset({"blues", "jazz", "rock", "shuffle"}),
        description="12-bar blues. Structural prior — locks in 12-bar grid when detected.",
    ),

    # ------------------------------------------------------------------ #
    # Pedal / static harmony                                              #
    # ------------------------------------------------------------------ #

    "pedal_tonic": RelativeProgression(
        name="Pedal on tonic",
        chords=(
            _RC(0,  ChordQuality.MAJ7),
            _RC(0,  ChordQuality.MAJ7),
        ),
        weight=0.4,
        typical_beats=(4.0, 4.0),
        style_tags=frozenset({"jazz", "modal", "ambient"}),
        description="Static tonic — common in modal jazz (Coltrane, Miles Kind of Blue era).",
    ),
}


# ---------------------------------------------------------------------------
# Style priors — tempo + harmonic rhythm + progression weights
# ---------------------------------------------------------------------------

@dataclass
class StylePrior:
    """
    A musical style encapsulates:
      - tempo range (BPM)
      - typical harmonic rhythm in beats per chord change
      - probability weights over progressions

    All fields feed into the hierarchical Bayesian model:
        P(style) → P(tempo | style) → P(harmonic_rhythm | tempo, style)
                                    → P(progression | style)
    """
    name: str
    tempo_range: tuple[float, float]          # (min_bpm, max_bpm)
    tempo_mode: float                          # most likely BPM within style
    harmonic_rhythm_beats: tuple[float, float] # (min, max) beats per chord
    harmonic_rhythm_mode: float                # most likely beats per chord
    progression_weights: dict[str, float]      # keys are PROGRESSIONS keys
    description: str = ""

    def tempo_log_prior(self, bpm: float) -> float:
        """Log-prior on tempo for this style (log-normal approximation)."""
        import math
        lo, hi = self.tempo_range
        if bpm < lo or bpm > hi:
            return -np.inf
        mu = math.log(self.tempo_mode)
        sigma = 0.3
        return -0.5 * ((math.log(bpm) - mu) / sigma) ** 2


STYLE_PRIORS: dict[str, StylePrior] = {

    "jazz_ballad": StylePrior(
        name="Jazz Ballad",
        tempo_range=(30.0, 80.0),
        tempo_mode=55.0,
        harmonic_rhythm_beats=(2.0, 8.0),
        harmonic_rhythm_mode=4.0,
        progression_weights={
            "ii_V_I_major": 1.0,
            "ii_V_I_minor": 0.8,
            "tritone_sub_major": 0.7,
            "cycle_of_fifths_4": 0.5,
            "iv_minor_borrow": 0.4,
        },
        description="Slow, expressive. Chords breathe for 2–8 beats. Heavy reharmonisation.",
    ),

    "jazz_medium_swing": StylePrior(
        name="Medium Swing",
        tempo_range=(90.0, 180.0),
        tempo_mode=130.0,
        harmonic_rhythm_beats=(1.0, 4.0),
        harmonic_rhythm_mode=2.0,
        progression_weights={
            "ii_V_I_major": 1.0,
            "ii_V_I_minor": 0.75,
            "ii_V_only": 0.7,
            "cycle_of_fifths_4": 0.65,
            "rhythm_changes_bridge": 0.55,
            "secondary_dom_to_ii": 0.5,
            "tritone_sub_major": 0.6,
        },
        description="The bread and butter swing tempo. 2-beat harmonic changes common.",
    ),

    "bebop": StylePrior(
        name="Bebop",
        tempo_range=(180.0, 340.0),
        tempo_mode=240.0,
        harmonic_rhythm_beats=(0.5, 2.0),
        harmonic_rhythm_mode=1.0,
        progression_weights={
            "ii_V_I_major": 1.0,
            "ii_V_only": 0.9,
            "ii_V_I_minor": 0.75,
            "rhythm_changes_bridge": 0.8,
            "tritone_sub_major": 0.7,
            "tritone_sub_bare": 0.5,
            "secondary_dom_to_ii": 0.65,
        },
        description="Very fast. Rapid ii-Vs, often unresolved. 1-beat chord changes at peak.",
    ),

    "latin_jazz": StylePrior(
        name="Latin Jazz",
        tempo_range=(120.0, 220.0),
        tempo_mode=165.0,
        harmonic_rhythm_beats=(1.0, 4.0),
        harmonic_rhythm_mode=2.0,
        progression_weights={
            "ii_V_I_major": 0.9,
            "ii_V_I_minor": 0.85,
            "cycle_of_fifths_4": 0.7,
            "pedal_tonic": 0.4,
        },
        description="Clave-based. Modal vamps common. Pedal tonic sections frequent.",
    ),

    "pop": StylePrior(
        name="Pop",
        tempo_range=(80.0, 140.0),
        tempo_mode=110.0,
        harmonic_rhythm_beats=(2.0, 8.0),
        harmonic_rhythm_mode=4.0,
        progression_weights={
            "ii_V_I_major": 0.5,
            "borrowed_bVII": 0.8,
            "borrowed_bVI": 0.7,
            "iv_minor_borrow": 0.6,
            "secondary_dom_to_IV": 0.4,
            "cycle_of_fifths_4": 0.4,
        },
        description="4-beat chord changes common. Modal interchange (bVII, bVI) very frequent.",
    ),

    "blues": StylePrior(
        name="Blues / Shuffle",
        tempo_range=(60.0, 160.0),
        tempo_mode=100.0,
        harmonic_rhythm_beats=(4.0, 8.0),
        harmonic_rhythm_mode=4.0,
        progression_weights={
            "blues_I_IV_V": 1.0,
            "secondary_dom_to_IV": 0.6,
        },
        description="12-bar structure dominant. All chords typically dominant 7ths.",
    ),

    "modal_jazz": StylePrior(
        name="Modal Jazz",
        tempo_range=(60.0, 200.0),
        tempo_mode=120.0,
        harmonic_rhythm_beats=(4.0, 32.0),
        harmonic_rhythm_mode=8.0,
        progression_weights={
            "pedal_tonic": 1.0,
            "borrowed_bVII": 0.5,
            "ii_V_only": 0.3,
        },
        description="Static harmony, slow or no chord changes. Miles Davis / Coltrane territory.",
    ),
}


# ---------------------------------------------------------------------------
# Transition prior matrix (relative)
# ---------------------------------------------------------------------------

def build_relative_transition_matrix(
    n_pitch_classes: int = 12,
) -> dict[tuple[int, int], float]:
    """
    Returns unnormalised log-weights for chord root transitions
    expressed as (from_interval, to_interval) pairs (both relative to tonic).

    These encode voice-leading tendencies regardless of key:
      - Descending 5th (ascending 4th) motion: the cycle of fifths pull
      - Semitone resolution: tritone sub / chromatic approach
      - Tritone: dominant ↔ tritone sub symmetry
    """
    weights: dict[tuple[int, int], float] = {}

    for from_iv in range(12):
        for to_iv in range(12):
            diff = (to_iv - from_iv) % 12
            # Descending 5th (= ascending 4th = diff of 5)
            if diff == 5:
                w = 0.9
            # Descending semitone (chromatic resolution)
            elif diff == 11:
                w = 0.7
            # Tritone movement (6 semitones)
            elif diff == 6:
                w = 0.5
            # Ascending step (2 semitones)
            elif diff == 2:
                w = 0.3
            # Staying on same root (chord quality change)
            elif diff == 0:
                w = 0.4
            else:
                w = 0.1
            weights[(from_iv, to_iv)] = w

    return weights


# ---------------------------------------------------------------------------
# Helper: most likely style given a detected tempo
# ---------------------------------------------------------------------------

def infer_style_posteriors(tempo_bpm: float) -> dict[str, float]:
    """
    Returns normalised posterior P(style | tempo) assuming uniform style prior.
    Useful for initialising the model before full inference.
    """
    log_probs = {
        name: sp.tempo_log_prior(tempo_bpm)
        for name, sp in STYLE_PRIORS.items()
    }
    # Normalise in log space
    max_lp = max(log_probs.values())
    probs = {k: np.exp(v - max_lp) for k, v in log_probs.items()}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}
