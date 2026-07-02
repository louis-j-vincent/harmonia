"""
Unit tests for harmonia/theory/duration_prior.py.

Uses the real POP909 dataset (text annotations only, no audio/model
inference needed — see docs/known_issues.md #1) since that's the whole
point of this module: an empirical fit from real ground truth, not a
synthetic stand-in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harmonia.theory.duration_prior import (
    fit_duration_prior,
    load_or_fit_duration_prior,
    log_duration_prior,
)

POP909_DIR = Path(__file__).parent.parent / "data" / "pop909" / "POP909"

pytestmark = pytest.mark.skipif(
    not POP909_DIR.exists(), reason="POP909 dataset not present locally"
)


@pytest.fixture(scope="module")
def default_prior():
    """Fitting parses all 909 songs (~1s) — do it once per test module."""
    return fit_duration_prior(POP909_DIR)


class TestFitDurationPrior:

    def test_pmfs_are_normalised(self, default_prior):
        assert default_prior["chord"].sum() == pytest.approx(1.0)
        assert default_prior["no_chord"].sum() == pytest.approx(1.0)

    def test_chord_duration_is_sane(self, default_prior):
        """Real chord durations in pop music: a beat or a few, not dozens."""
        D = len(default_prior["chord"])
        mean_duration = np.sum((np.arange(D) + 1) * default_prior["chord"])
        assert 1.0 <= mean_duration <= 6.0

    def test_chord_duration_is_not_geometric(self, default_prior):
        """A geometric distribution is always maximised at its minimum value
        (duration=1). The real distribution peaking at duration=2 is the
        empirical justification for using an explicit-duration decoder
        instead of a memoryless self-transition boost."""
        assert default_prior["chord"][1] > default_prior["chord"][0]  # P(d=2) > P(d=1)

    def test_no_chord_duration_is_mostly_one_beat(self, default_prior):
        """N events (intros/outros/pickups) should be short."""
        assert default_prior["no_chord"][0] > 0.9  # P(d=1) > 90%

    def test_respects_max_duration_beats(self):
        prior = fit_duration_prior(POP909_DIR, max_duration_beats=10)
        assert len(prior["chord"]) == 10
        assert len(prior["no_chord"]) == 10


class TestLoadOrFitDurationPrior:

    def test_caches_to_disk(self, tmp_path):
        cache_path = tmp_path / "duration_prior.npz"
        assert not cache_path.exists()

        prior1 = load_or_fit_duration_prior(POP909_DIR, cache_path=cache_path)
        assert cache_path.exists()

        prior2 = load_or_fit_duration_prior(POP909_DIR, cache_path=cache_path)
        np.testing.assert_array_equal(prior1["chord"], prior2["chord"])


class TestLogDurationPrior:

    def test_no_negative_infinity_for_zero_probability_bins(self):
        pmf = np.array([0.5, 0.5, 0.0, 0.0])
        log_pmf = log_duration_prior(pmf)
        assert np.isfinite(log_pmf).all()
