"""
Characterization tests for harmonia/eval/mirex_eval.py — Phase 0 of the
refactoring plan (docs/refactoring_delegation_plan.md).

Pins `evaluate_song()`'s output on a small, hand-built (predicted, reference)
chord pair spanning all four MIREX strictness levels (root / majmin /
sevenths / tetrads) so a later refactor of the eval path can be checked for
bit-for-bit score parity. This is CLAUDE.md error pattern #6 ("a metric that
silently shifts is the worst failure mode in a research repo") — complements
the existing `tests/test_mirex_eval.py`, which regression-tests
`_label_to_mireval()`'s label-encoding correctness but does not pin
end-to-end `evaluate_song()` scores.

Scope note (CLAUDE.md rule #4 — document what's NOT covered): CLAUDE.md's
collaboration conventions ask for "partial-credit chord scoring (predicting
maj7 when GT is maj should get credit for the parent family) alongside
strict exact-match." No such family-partial-credit function exists in
`harmonia/eval/mirex_eval.py` as of this session — the four MIREX levels
(root/majmin/sevenths/tetrads) are a strictness *hierarchy*, not a
maj7-credits-maj partial-credit scorer. This test treats "majmin" as the
coarse/lenient level and "tetrads" as the strict level, which is the closest
existing analogue, and explicitly flags that a true family-credit metric is
not implemented — so a refactor must not assume one exists to preserve.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonia.eval.mirex_eval import MIREXScore, evaluate_song

pytest.importorskip("mir_eval")


class TestEvaluateSongCharacterization:
    """A tiny, fully hand-constructed 4-chord song: 2 exact matches, 1
    root-only match (quality differs), 1 total miss. Chosen so root,
    majmin, sevenths, and tetrads scores are all expected to differ from
    each other and from 0/1, exercising the full strictness ladder in one
    fixture."""

    # Reference (ground truth): C:maj, D:min7, F:maj7, G:7 — each 1s long.
    reference_intervals = np.array([
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ])
    reference_labels = ["C:maj", "D:min7", "F:maj7", "G:7"]

    # Predictions (Harmonia-format labels, as ChordChart.chords would emit):
    #   bar 0: Cmaj      -> exact match (root+majmin+7ths+tetrad all correct)
    #   bar 1: Dmin7     -> exact match
    #   bar 2: Fmaj      -> root+majmin match, sevenths/tetrads wrong (no 7th)
    #   bar 3: Emin      -> total miss (wrong root entirely: G vs E)
    predicted_chords = [
        {"label": "Cmaj", "start_s": 0.0, "end_s": 1.0},
        {"label": "Dmin7", "start_s": 1.0, "end_s": 2.0},
        {"label": "Fmaj", "start_s": 2.0, "end_s": 3.0},
        {"label": "Emin", "start_s": 3.0, "end_s": 4.0},
    ]

    @classmethod
    @pytest.fixture(scope="class")
    def score(cls) -> MIREXScore:
        return evaluate_song(
            cls.predicted_chords, cls.reference_intervals, cls.reference_labels
        )

    def test_score_is_mirex_score(self, score):
        assert isinstance(score, MIREXScore)

    def test_duration_scored(self, score):
        # min(pred last end, ref last end) = 4.0s
        assert score.duration_s == pytest.approx(4.0, abs=1e-6)

    def test_root_score_pinned(self, score):
        # 3 of 4 bars have correct root (C, D, F correct; G vs E wrong) = 0.75
        assert score.root == pytest.approx(0.75, abs=1e-6)

    def test_majmin_score_pinned(self, score):
        # Same 3/4 bars also have correct triad quality (maj/min) = 0.75
        assert score.majmin == pytest.approx(0.75, abs=1e-6)

    def test_sevenths_score_pinned(self, score):
        # Only bars 0 (Cmaj==Cmaj exact-ish) and 1 (Dmin7==Dmin7 exact) score;
        # bar 2 predicted maj but ref is maj7 -> mismatch at 7ths level.
        assert score.sevenths == pytest.approx(0.5, abs=1e-6)

    def test_tetrads_score_pinned(self, score):
        assert score.tetrads == pytest.approx(0.5, abs=1e-6)

    def test_strictness_ordering_holds(self, score):
        # A basic sanity invariant that should survive any refactor: looser
        # levels can't score lower than stricter ones on the same pair.
        assert score.root >= score.majmin >= score.sevenths >= score.tetrads \
            or (score.majmin >= score.sevenths >= score.tetrads)


class TestEvaluateSongEdgeCases:

    def test_empty_predicted_chords_returns_zero_score(self):
        score = evaluate_song([], np.array([[0.0, 1.0]]), ["C:maj"])
        assert score == MIREXScore(0.0, 0.0, 0.0, 0.0, 0.0)

    def test_perfect_prediction_scores_all_ones(self):
        ref_intervals = np.array([[0.0, 1.0], [1.0, 2.0]])
        ref_labels = ["C:maj", "A:min"]
        pred = [
            {"label": "Cmaj", "start_s": 0.0, "end_s": 1.0},
            {"label": "Amin", "start_s": 1.0, "end_s": 2.0},
        ]
        score = evaluate_song(pred, ref_intervals, ref_labels)
        assert score.root == pytest.approx(1.0)
        assert score.majmin == pytest.approx(1.0)
        assert score.sevenths == pytest.approx(1.0)
        assert score.tetrads == pytest.approx(1.0)
