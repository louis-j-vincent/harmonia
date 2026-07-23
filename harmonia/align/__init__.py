"""harmonia.align — timing/beat instruments for the Bayesian fusion aligner.

Stage 1 of the fusion aligner (design: docs/fusion_aligner_design.md) lives here:
the standalone DRUM beat/tempo tracker (`drum_pattern`), which becomes the drum
observation term (stream #2) of the Stage-2 fusion DBN. Alongside it:
the BASS-salience instrument (`bass_salience`, stream #4 + the sounding-bass GT
target) and the global-phase DOWNBEAT resolver (`downbeat`), which fuses
harmonic rhythm + chart + bass + the drum strong-beat pair into one downbeat
phase.
"""
from harmonia.align.drum_pattern import (
    OnsetEnvelopes,
    AnchorWindow,
    DrumBeatTrack,
    percussive_onset_envelopes,
    select_drum_anchor,
    drum_reliability,
    estimate_period_in_octave,
    track_drum_beats,
    track_from_audio,
)
from harmonia.align.bass_salience import (
    BassChroma,
    bass_chroma,
    bass_pc_over_span,
    bass_reliability,
    bass_pc_series,
)
from harmonia.align.downbeat import (
    DownbeatResult,
    resolve_downbeat,
    resolve_phase,
    constant_lattice,
    beat_chroma_flux,
    harmonic_change_evidence,
    bass_beat_evidence,
)

__all__ = [
    "OnsetEnvelopes",
    "AnchorWindow",
    "DrumBeatTrack",
    "percussive_onset_envelopes",
    "select_drum_anchor",
    "drum_reliability",
    "estimate_period_in_octave",
    "track_drum_beats",
    "track_from_audio",
    # bass salience (stream #4)
    "BassChroma",
    "bass_chroma",
    "bass_pc_over_span",
    "bass_reliability",
    "bass_pc_series",
    # global-phase downbeat resolver
    "DownbeatResult",
    "resolve_downbeat",
    "resolve_phase",
    "constant_lattice",
    "beat_chroma_flux",
    "harmonic_change_evidence",
    "bass_beat_evidence",
]
