"""Vocabulary-based folding of chord OBSERVATIONS
(``harmonia.models.periodicity.fold_by_vocabulary``).

Louis's brief, 2026-07-30: once the section vocabulary is known, every
occurrence of a vocabulary item is another observation of the SAME music, so
average the per-beat note activations (88-dim) across occurrences BEFORE
decoding — √N signal-to-noise on the raw evidence — instead of voting on
decoded chord labels afterwards (label voting throws away the confidence
geometry: it can never learn that a 7th was weakly present in every pass).

Three things are pinned here, in the order they can silently break:

1. ``TestExactMean`` — the averaging really is the arithmetic mean of the right
   cells. This is the load-bearing check: the vocabulary is indexed in BARS and
   ``beat_probs`` in ABSOLUTE BEATS, so an off-by-one in the bar→beat map still
   produces plausible-looking smoothed output (CLAUDE.md error pattern #1).
2. ``TestEndingsAreNotSmeared`` — a 1st/2nd ending pair (This Love's C
   ``Cm F△7 | Ab G7`` and E ``Cm7 F | Ab Ab``) must NOT blend, and the old
   fixed-period ``fold_beat_probs`` at the matching period MUST blend them.
   That contrast is the entire justification for this function existing.
3. ``TestRagged`` — occurrences of one item can contain different numbers of
   beats (the bar grid is rigid, beat detection is not). Nothing may be
   silently zipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonia.models.periodicity import fold_beat_probs, fold_by_vocabulary

BPB = 4          # beats per bar
BAR_S = 1.0      # one bar = 1 s, so one beat = 0.25 s
K = 88           # piano keys — beat_probs is (B, 88)


def _uniform_grid(n_bars: int, bpb: int = BPB):
    """``(beat_times, bar_bounds_sec)`` for a perfectly uniform grid."""
    n_beats = n_bars * bpb
    beat_times = np.arange(n_beats) * (BAR_S / bpb)
    bar_bounds = [b * BAR_S for b in range(n_bars + 1)]
    return beat_times, bar_bounds


def _ramp_probs(n_beats: int) -> np.ndarray:
    """``beat_probs[i, :] = i`` — every beat carries its own absolute index, so
    the expected fold of any slot is just the mean of its beat indices and an
    off-by-one alignment error is arithmetically visible."""
    return np.tile(np.arange(n_beats, dtype=np.float64)[:, None], (1, K))


# ── 1. the mean is the mean, of exactly the right cells ──────────────────────

class TestExactMean:
    """Each occurrence of item A carries a distinct known value at slot i; the
    folded output at every one of those positions must equal the exact
    arithmetic mean of those values."""

    def _song(self):
        # 12 bars:  A×3 (4-bar item, bars 0-12)  →  occurrences at beats 0/16/32
        beat_times, bar_bounds = _uniform_grid(12)
        sections = [{"label": "A", "bar0": 0, "bar1": 12, "d_bars": 4, "reps": 3}]
        return _ramp_probs(48), beat_times, sections, bar_bounds

    def test_slot_value_is_the_exact_arithmetic_mean(self):
        bp, bt, sections, bounds = self._song()
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        # slot s of a 4-bar item = beats {s, 16+s, 32+s}; mean = 16 + s.
        for s in range(16):
            expected = (s + (16 + s) + (32 + s)) / 3.0
            assert expected == pytest.approx(16.0 + s)
            for occ_start in (0, 16, 32):
                np.testing.assert_allclose(out[occ_start + s], expected)

    def test_off_by_one_alignment_would_fail_this(self):
        """Sanity on the sanity check: the expected values differ by 1 per slot,
        so a one-beat slot shift cannot accidentally satisfy the assertion."""
        bp, bt, sections, bounds = self._song()
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        assert out[0, 0] == pytest.approx(16.0)
        assert out[1, 0] == pytest.approx(17.0)

    def test_two_items_fold_independently(self):
        # A×3 (4-bar) over bars 0-12, then B×2 (3-bar) over bars 12-18.
        beat_times, bar_bounds = _uniform_grid(18)
        bp = _ramp_probs(72)
        sections = [
            {"label": "A", "bar0": 0, "bar1": 12, "d_bars": 4, "reps": 3},
            {"label": "B", "bar0": 12, "bar1": 18, "d_bars": 3, "reps": 2},
        ]
        out = fold_by_vocabulary(bp, beat_times, sections, bar_bounds)
        for s in range(16):                                   # A: beats 0/16/32
            np.testing.assert_allclose(out[s], 16.0 + s)
        for s in range(12):                                   # B: beats 48/60
            np.testing.assert_allclose(out[48 + s], 54.0 + s)

    def test_shape_and_dtype_match_fold_beat_probs(self):
        bp, bt, sections, bounds = self._song()
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        assert out.shape == bp.shape
        assert out.dtype == fold_beat_probs(bp, 16).dtype

    def test_input_is_not_mutated(self):
        bp, bt, sections, bounds = self._song()
        before = bp.copy()
        fold_by_vocabulary(bp, bt, sections, bounds)
        np.testing.assert_array_equal(bp, before)

    def test_a_single_occurrence_is_the_identity(self):
        """Mean of one observation is that observation — a one-off bridge must
        come back untouched, not smoothed toward anything."""
        beat_times, bar_bounds = _uniform_grid(8)
        bp = _ramp_probs(32)
        sections = [{"label": "D", "bar0": 0, "bar1": 8, "d_bars": 8, "reps": 1}]
        out = fold_by_vocabulary(bp, beat_times, sections, bar_bounds)
        np.testing.assert_array_equal(out, bp)

    def test_beats_in_no_vocabulary_item_are_left_untouched(self):
        """A section with fewer bars than its own item length has NO complete
        occurrence — those bars must pass through, not be half-folded."""
        beat_times, bar_bounds = _uniform_grid(11)
        bp = _ramp_probs(44)
        sections = [
            {"label": "A", "bar0": 0, "bar1": 8, "d_bars": 4, "reps": 2},
            {"label": "B", "bar0": 8, "bar1": 11, "d_bars": 4, "reps": 1},  # partial
        ]
        out = fold_by_vocabulary(bp, beat_times, sections, bar_bounds)
        for s in range(16):                                   # A: beats 0/16
            np.testing.assert_allclose(out[s], 8.0 + s)
        np.testing.assert_array_equal(out[32:44], bp[32:44])   # B: pass-through

    def test_beats_outside_the_bar_grid_are_left_untouched(self):
        """A pickup beat before the first bar edge, and beats after the last."""
        beat_times = np.concatenate([[-0.3], np.arange(32) * 0.25, [8.4]])
        bar_bounds = [b * BAR_S for b in range(9)]             # 8 bars, 0-8 s
        bp = _ramp_probs(len(beat_times))
        sections = [{"label": "A", "bar0": 0, "bar1": 8, "d_bars": 4, "reps": 2}]
        out = fold_by_vocabulary(bp, beat_times, sections, bar_bounds)
        np.testing.assert_array_equal(out[0], bp[0])           # the pickup
        np.testing.assert_array_equal(out[-1], bp[-1])         # past the grid
        for s in range(16):                                    # beats 1..32
            np.testing.assert_allclose(out[1 + s], (s + (16 + s)) / 2.0 + 1.0)

    def test_empty_and_degenerate_inputs_defer(self):
        beat_times, bar_bounds = _uniform_grid(8)
        bp = _ramp_probs(32)
        np.testing.assert_array_equal(
            fold_by_vocabulary(bp, beat_times, [], bar_bounds), bp)
        np.testing.assert_array_equal(
            fold_by_vocabulary(bp, beat_times, None, bar_bounds), bp)
        np.testing.assert_array_equal(
            fold_by_vocabulary(bp, beat_times,
                               [{"label": "A", "bar0": 0, "bar1": 8, "d_bars": 0}],
                               bar_bounds), bp)
        assert fold_by_vocabulary(np.zeros((0, K)), np.zeros(0),
                                  [], bar_bounds).shape == (0, K)


# ── 2. the whole point: a 1st/2nd ending must not blend ──────────────────────

class TestEndingsAreNotSmeared:
    """This Love's chorus tail C (``Cm F△7 | Ab G7``) and outro tail E
    (``Cm7 F | Ab Ab``) are separate vocabulary items precisely so their
    observations never mix. Form here:

        B×3 C   B×3 C   B×3 E      (24 bars, 2-bar items)

    C's two occurrences sit 8 bars apart, and E sits 8 bars after the second C —
    so the fixed-period folder at period = 8 bars = 32 beats averages C1, C2 and
    E into one another. That is exactly the failure the vocabulary folder fixes.
    """

    C_MARK, E_MARK = 10, 20        # two distinguishing keys: G7's F vs Ab's Ab

    def _song(self):
        beat_times, bar_bounds = _uniform_grid(24)
        bp = np.zeros((96, K))
        sections = []
        for b0 in (0, 8, 16):
            sections.append({"label": "B", "bar0": b0, "bar1": b0 + 6,
                             "d_bars": 2, "reps": 3})
        for b0, lab in ((6, "C"), (14, "C"), (22, "E")):
            sections.append({"label": lab, "bar0": b0, "bar1": b0 + 2,
                             "d_bars": 2, "reps": 1})
            mark = self.C_MARK if lab == "C" else self.E_MARK
            bp[b0 * BPB:(b0 + 2) * BPB, mark] = 1.0
        sections.sort(key=lambda s: s["bar0"])
        return bp, beat_times, sections, bar_bounds

    def test_vocabulary_folding_keeps_the_two_tails_apart(self):
        bp, bt, sections, bounds = self._song()
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        for beat in range(24, 32):                     # C1 (bars 6-8)
            assert out[beat, self.C_MARK] == pytest.approx(1.0)
            assert out[beat, self.E_MARK] == pytest.approx(0.0)
        for beat in range(88, 96):                     # E (bars 22-24)
            assert out[beat, self.E_MARK] == pytest.approx(1.0)
            assert out[beat, self.C_MARK] == pytest.approx(0.0)

    def test_the_two_C_occurrences_DO_fold_into_each_other(self):
        """Not smearing E must not come at the price of not folding C at all —
        C's own two observations still have to average."""
        bp, bt, sections, bounds = self._song()
        bp = bp.copy()
        bp[24:32, 40] = 1.0            # a note present only in C's FIRST pass
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        # C1 = bars 6-8 = beats 24-32;  C2 = bars 14-16 = beats 56-64.
        for beat in list(range(24, 32)) + list(range(56, 64)):
            assert out[beat, 40] == pytest.approx(0.5)     # 1 of 2 passes
        assert out[88, 40] == pytest.approx(0.0)           # E never sees it

    def test_the_old_fixed_period_folder_DOES_smear_them(self):
        """RED-FIRST justification: `fold_beat_probs` at the matching period
        blends C with E, destroying both tails."""
        bp, bt, _sections, _bounds = self._song()
        old = fold_beat_probs(bp, period=8 * BPB)
        for beat in range(24, 32):
            assert old[beat, self.C_MARK] == pytest.approx(2.0 / 3.0)
            assert old[beat, self.E_MARK] == pytest.approx(1.0 / 3.0)
        for beat in range(88, 96):
            assert old[beat, self.E_MARK] == pytest.approx(1.0 / 3.0)
            assert old[beat, self.C_MARK] == pytest.approx(2.0 / 3.0)


# ── 3. ragged occurrences: never silently zipped ─────────────────────────────

class TestRagged:
    """Alignment is per BAR (item-relative bar index, then ordinal beat within
    that bar), so a beat-detection wobble is contained inside the one bar where
    it happened instead of shifting every later beat of the occurrence.
    A slot supplied by only one occurrence folds to itself — an identity, never
    a zip against a different musical position."""

    def _ragged_song(self):
        # A×3, 2-bar item, 6 bars — but bar 2 (the start of occurrence 2)
        # received a 5th beat from the tracker.
        bar_bounds = [b * BAR_S for b in range(7)]
        times = []
        for b in range(6):
            n = 5 if b == 2 else 4
            times += [b * BAR_S + j * (BAR_S / n) for j in range(n)]
        beat_times = np.array(times)
        bp = _ramp_probs(len(beat_times))
        sections = [{"label": "A", "bar0": 0, "bar1": 6, "d_bars": 2, "reps": 3}]
        return bp, beat_times, sections, bar_bounds

    def test_ragged_bar_does_not_shift_the_rest_of_the_occurrence(self):
        bp, bt, sections, bounds = self._ragged_song()
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        # Beats: bar0 0-3, bar1 4-7, bar2 8-12 (5!), bar3 13-16, bar4 17-20,
        # bar5 21-24. Occurrence starts: bars 0/2/4 → beats 0/8/17.
        # Item-bar 1, ordinal 0 = beats 4, 13, 21 — all three still line up,
        # even though occurrence 2 carries one extra beat before them.
        np.testing.assert_allclose(out[4], (4 + 13 + 21) / 3.0)
        np.testing.assert_allclose(out[13], (4 + 13 + 21) / 3.0)
        np.testing.assert_allclose(out[21], (4 + 13 + 21) / 3.0)

    def test_the_extra_beat_folds_to_itself(self):
        """Ordinal 4 of item-bar 0 exists in exactly one occurrence, so it is
        its own mean — untouched, not averaged against someone else's beat."""
        bp, bt, sections, bounds = self._ragged_song()
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        np.testing.assert_array_equal(out[12], bp[12])

    def test_matched_ordinals_in_the_ragged_bar_still_fold(self):
        bp, bt, sections, bounds = self._ragged_song()
        out = fold_by_vocabulary(bp, bt, sections, bounds)
        # item-bar 0, ordinal 0 = beats 0, 8, 17
        np.testing.assert_allclose(out[0], (0 + 8 + 17) / 3.0)
        np.testing.assert_allclose(out[8], (0 + 8 + 17) / 3.0)
        np.testing.assert_allclose(out[17], (0 + 8 + 17) / 3.0)

    def test_a_bar_with_no_beats_at_all_is_harmless(self):
        beat_times = np.array([0.1, 0.2, 2.1, 2.2])      # bar 1 and bar 3 empty
        bar_bounds = [0.0, 1.0, 2.0, 3.0, 4.0]
        bp = _ramp_probs(4)
        sections = [{"label": "A", "bar0": 0, "bar1": 4, "d_bars": 2, "reps": 2}]
        out = fold_by_vocabulary(bp, beat_times, sections, bar_bounds)
        np.testing.assert_allclose(out[0], 1.0)          # beats 0 and 2
        np.testing.assert_allclose(out[1], 2.0)          # beats 1 and 3


# ── 4. no downbeat-on-beat-0 assumption (docs/known_issues.md OPEN #3) ───────

def test_alignment_survives_a_grid_phase_that_never_puts_a_beat_on_the_edge():
    """`rigid_grid.py` backs every bar edge off by 0.15 bar, so on This Love no
    downbeat lands on beat 0 of its bar (histogram {1:75, 2:1, 3:43}). The
    bar→beat map here is purely by TIME, so a constant phase offset shifts which
    beats a bar owns but keeps every occurrence's slot alignment consistent —
    which is all folding needs."""
    n_bars, bpb = 8, 4
    bar_bounds = [b * BAR_S - 0.15 * BAR_S for b in range(n_bars + 1)]
    beat_times = np.arange(n_bars * bpb) * (BAR_S / bpb)   # beats still at k/4
    bp = _ramp_probs(n_bars * bpb)
    sections = [{"label": "A", "bar0": 0, "bar1": 8, "d_bars": 4, "reps": 2}]
    out = fold_by_vocabulary(bp, beat_times, sections, bar_bounds)
    # Bar 0 spans [-0.15, 0.85) → owns beats 0,1,2,3 exactly as before; the
    # backoff is smaller than one beat, so the ownership is unchanged and both
    # occurrences fold slot-for-slot.
    for s in range(16):
        np.testing.assert_allclose(out[s], (s + (16 + s)) / 2.0)


# ── 5. the pipeline flag is off by default ──────────────────────────────────

def test_pipeline_flag_defaults_off(monkeypatch):
    """With `HARMONIA_VOCAB_FOLD` unset the live pipeline must fold nothing —
    the OFF path has to stay byte-identical to before this feature existed.

    The flag is read in `chord_pipeline_v1`, which since 2026-07-30 is the ONLY
    inference path (the second one, `harmonia/pipeline.py::HarmoniaPipeline`,
    was deleted as a duplicate nothing shipped went through). `harmonia/
    pipeline.py` now holds only the shared `ChordChart`/`PipelineConfig`
    types."""
    from harmonia.models import chord_pipeline_v1 as cp

    monkeypatch.delenv("HARMONIA_VOCAB_FOLD", raising=False)
    assert cp._vocab_fold_enabled() is False
    monkeypatch.setenv("HARMONIA_VOCAB_FOLD", "1")
    assert cp._vocab_fold_enabled() is True
    monkeypatch.setenv("HARMONIA_VOCAB_FOLD", "0")
    assert cp._vocab_fold_enabled() is False


def test_folding_is_a_no_op_when_the_vocabulary_declines(monkeypatch):
    """`vocab_sections` returns None on single-loop and through-composed songs.
    The live glue must then leave the evidence exactly as it found it."""
    from harmonia.models import chord_pipeline_v1 as cp

    bt = np.arange(33) * 0.25
    onset_b = np.random.default_rng(0).random((32, K))
    out = cp._vocab_fold_arrays([], bt, onset_b)
    np.testing.assert_array_equal(out[0], onset_b)
