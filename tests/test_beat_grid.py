"""Beat-grid period fitting (known_issues "BAR-GRID vs REAL-MUSIC DRIFT").

The load-bearing assumption under every chart is the constant-tempo grid
``bt = arange(phase, dur, period)``. These tests pin the new bestfit period
estimator against synthetic beats with known ground truth (CLAUDE.md rule #1:
unit-test the basic assumption against an external reference).
"""

import inspect

import numpy as np
import pytest

from harmonia.models.chord_pipeline_v1 import _bestfit_beat_period, infer_chords_v1


TRUE_PERIOD = 60.0 / 123.0  # an awkward, non-round tempo


def _beats(n=400, period=TRUE_PERIOD, t0=0.37, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = t0 + period * np.arange(n)
    if jitter:
        t = t + rng.normal(0.0, jitter, size=n)
    return t


class TestBestfitBeatPeriod:
    def test_recovers_true_period_from_biased_init(self):
        # librosa-style error: init period 1.5% off the whole-song average.
        beats = _beats()
        init = TRUE_PERIOD * 1.015
        fit = _bestfit_beat_period(beats, init)
        assert abs(fit - TRUE_PERIOD) / TRUE_PERIOD < 1e-6

    def test_recovers_under_realistic_onset_jitter(self):
        # 15 ms onset jitter, 3.5-minute song: fit error must stay below the
        # 0.5% systematic error it exists to remove.
        beats = _beats(jitter=0.015)
        fit = _bestfit_beat_period(beats, TRUE_PERIOD * 1.02)
        assert abs(fit - TRUE_PERIOD) / TRUE_PERIOD < 0.005

    def test_robust_to_missed_beats(self):
        # Drop 5% of beats: indices must advance by 2 across each gap, so the
        # slope is unaffected (a naive arange index would dilate the period).
        beats = _beats()
        keep = np.ones(len(beats), bool)
        keep[10::20] = False
        fit = _bestfit_beat_period(beats[keep], TRUE_PERIOD * 1.01)
        assert abs(fit - TRUE_PERIOD) / TRUE_PERIOD < 1e-6

    def test_octave_locked_init_is_left_alone(self):
        # A 2x-wrong init (tempo octave-lock, known_issues #1) is out of the
        # estimator's scope: it must return the init unchanged, not "fix" it.
        beats = _beats()
        init = TRUE_PERIOD * 2.0
        assert _bestfit_beat_period(beats, init) == init

    def test_too_few_beats_is_a_no_op(self):
        beats = _beats(n=5)
        init = TRUE_PERIOD * 1.02
        assert _bestfit_beat_period(beats, init) == init

    def test_accumulated_drift_removed(self):
        # The user-visible symptom: with a 1.5%-biased period, beat 400 of the
        # grid is ~6 beats (1.5 bars) off; with the fitted period, < 0.1 beat.
        beats = _beats()
        init = TRUE_PERIOD * 1.015
        fit = _bestfit_beat_period(beats, init)
        n = len(beats)
        drift_init = abs(init * (n - 1) - TRUE_PERIOD * (n - 1)) / TRUE_PERIOD
        drift_fit = abs(fit * (n - 1) - TRUE_PERIOD * (n - 1)) / TRUE_PERIOD
        assert drift_init > 4.0
        assert drift_fit < 0.1


def test_default_beat_period_mode_is_librosa():
    # The flag ships default-off: the production grid stays bit-identical
    # (CLAUDE.md rule #6) until the staged rollout says otherwise.
    sig = inspect.signature(infer_chords_v1)
    assert sig.parameters["beat_period_mode"].default == "librosa"
