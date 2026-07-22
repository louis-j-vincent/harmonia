"""harmonia.align — timing/beat instruments for the Bayesian fusion aligner.

Stage 1 of the fusion aligner (design: docs/fusion_aligner_design.md) lives here:
the standalone DRUM beat/tempo tracker (`drum_pattern`), which becomes the drum
observation term (stream #2) of the Stage-2 fusion DBN.
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
]
