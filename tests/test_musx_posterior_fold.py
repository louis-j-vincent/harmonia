"""Two-pass music-x-lab decode: vocabulary fold on ITS OWN frame posteriors.

Red-first (2026-07-30). The NNLS-24 vocabulary fold measured exactly +0.00 pp on
the shipped config because music-x-lab supplies segmentation/root/quality/bass
there, so the folded NNLS observation is barely read
(``docs/research_sessions/vocabulary_fold_2026-07-30.md`` §5). This module folds
UPSTREAM of music-x-lab's own decoder instead, so the shipped config can move.

The four things these tests pin (in the order the brief asks for them):
  1. the fold is the arithmetic MEAN of the right frames, and an off-by-one
     frame shift is arithmetically visible;
  2. a 1st/2nd ending is not smeared (This Love's ``C`` vs ``E``);
  3. the disagreement guard skips slots whose occurrences are not the same
     music, and the bass stream is never folded;
  4. flag OFF is identity.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models import musx_posterior_fold as mpf

DT = mpf.FRAME_DT

# A bar of exactly 43 frames, so every bar boundary lands on a frame edge and
# the expected fold of a slot can be written down by hand (no reuse of the
# module's own bar->frame map inside the assertion).
FPB = 43
BAR = FPB * DT


def _bounds(n_bars: int, t0: float = 0.0) -> list[float]:
    return [t0 + b * BAR for b in range(n_bars + 1)]


def _probs(n_frame: int, k_triad: int = 73) -> list[np.ndarray]:
    """A uniform 6-stream posterior stack of the real shapes."""
    shapes = (k_triad, 13, 4, 4, 3, 3)
    return [np.full((n_frame, k), 1.0 / k, dtype=np.float32) for k in shapes]


def _one_hot(n_frame: int, k: int, col: int) -> np.ndarray:
    a = np.zeros((n_frame, k), dtype=np.float32)
    a[:, col] = 1.0
    return a


class TestFoldIsTheMean:
    """1. The fold is the arithmetic mean of the RIGHT frames."""

    def test_exact_mean_of_the_three_occurrences(self):
        # 12 bars = A x 3 of a 4-bar item. Occurrences start at frames
        # 0 / 172 / 344 (4 bars x 43 frames), so slot s of the item must fold to
        # the mean of x at frames {s, 172 + s, 344 + s}.
        n_bars, d = 12, 4
        n_frame = n_bars * FPB
        probs = _probs(n_frame)
        # Every frame carries its own index: triad col 0 = f / n_frame, col 1 =
        # the complement, so each row is still a distribution and an off-by-one
        # is arithmetically visible (neighbouring frames differ by 1/n_frame).
        x = np.arange(n_frame, dtype=np.float64) / n_frame
        probs[0] = np.zeros((n_frame, 73), dtype=np.float64)
        probs[0][:, 0] = x
        probs[0][:, 1] = 1.0 - x
        sections = [{"label": "A", "bar0": 0, "bar1": 12, "d_bars": d, "reps": 3}]

        out, rep = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), agree_min=0.0, report=True)

        occ = [0, d * FPB, 2 * d * FPB]
        for s in range(d * FPB):
            want = float(np.mean([x[o + s] for o in occ]))
            for o in occ:
                assert out[0][o + s, 0] == pytest.approx(want, abs=1e-12)
        assert rep["n_slots_folded"] > 0

    def test_an_off_by_one_frame_shift_fails(self):
        n_bars, d = 12, 4
        n_frame = n_bars * FPB
        probs = _probs(n_frame)
        x = np.arange(n_frame, dtype=np.float64) / n_frame
        probs[0] = np.zeros((n_frame, 73), dtype=np.float64)
        probs[0][:, 0] = x
        probs[0][:, 1] = 1.0 - x
        sections = [{"label": "A", "bar0": 0, "bar1": 12, "d_bars": d, "reps": 3}]
        out, _ = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), agree_min=0.0, report=True)

        occ = [0, d * FPB, 2 * d * FPB]
        # The SHIFTED expectation (slot s folded from frames o + s + 1) must NOT
        # reproduce the output — otherwise the test could not detect a slip.
        bad = 0
        for s in range(0, d * FPB - 1):
            shifted = float(np.mean([x[o + s + 1] for o in occ]))
            if out[0][s, 0] == pytest.approx(shifted, abs=1e-12):
                bad += 1
        assert bad == 0

    def test_input_is_not_mutated(self):
        n_frame = 12 * FPB
        probs = _probs(n_frame)
        probs[0] = _one_hot(n_frame, 73, 3)
        snap = [p.copy() for p in probs]
        sections = [{"label": "A", "bar0": 0, "bar1": 12, "d_bars": 4, "reps": 3}]
        mpf.fold_frame_posteriors(probs, sections, _bounds(12), agree_min=0.0)
        for a, b in zip(probs, snap):
            assert np.array_equal(a, b)

    def test_rows_still_sum_to_one(self):
        n_frame = 12 * FPB
        probs = _probs(n_frame)
        rng = np.random.default_rng(0)
        for i, k in enumerate((73, 13, 4, 4, 3, 3)):
            a = rng.random((n_frame, k)).astype(np.float32)
            probs[i] = (a / a.sum(1, keepdims=True)).astype(np.float32)
        sections = [{"label": "A", "bar0": 0, "bar1": 12, "d_bars": 4, "reps": 3}]
        out, _ = mpf.fold_frame_posteriors(
            probs, sections, _bounds(12), agree_min=0.0, report=True)
        for a in out:
            assert np.allclose(a.sum(1), 1.0, atol=1e-5)


class TestEndingsAreNotSmeared:
    """2. This Love's chorus tail C and outro tail E must never mix."""

    def test_first_and_second_ending_stay_separate(self):
        # B B B C | B B B C | B B B E  — 2-bar items, 24 bars.
        n_bars = 24
        n_frame = n_bars * FPB
        probs = _probs(n_frame)
        tri = np.zeros((n_frame, 73), dtype=np.float64)
        C_COL, E_COL, B_COL = 5, 6, 7

        def bar_frames(b):
            return slice(b * FPB, (b + 1) * FPB)

        sections = []
        for start, label in ((0, "B"), (6, "C"), (8, "B"), (14, "C"),
                             (16, "B"), (22, "E")):
            n = 6 if label == "B" else 2
            sections.append({"label": label, "bar0": start, "bar1": start + n,
                             "d_bars": 2, "reps": n // 2})
            col = {"B": B_COL, "C": C_COL, "E": E_COL}[label]
            for b in range(start, start + n):
                tri[bar_frames(b), col] = 1.0
        probs[0] = tri

        out, _ = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), agree_min=0.0, report=True)

        c_frames = np.r_[6 * FPB:8 * FPB, 14 * FPB:16 * FPB]
        e_frames = np.arange(22 * FPB, 24 * FPB)
        assert out[0][c_frames, C_COL] == pytest.approx(1.0)
        assert out[0][c_frames, E_COL] == pytest.approx(0.0)
        assert out[0][e_frames, E_COL] == pytest.approx(1.0)
        assert out[0][e_frames, C_COL] == pytest.approx(0.0)

    def test_evidence_present_in_only_one_pass_of_C_still_averages(self):
        # Not refusing to fold: a column present in only C's FIRST pass folds to
        # 0.5 across C's two occurrences, and E never sees it at all.
        n_bars = 24
        n_frame = n_bars * FPB
        probs = _probs(n_frame)
        tri = np.zeros((n_frame, 73), dtype=np.float64)
        sections = []
        for start, label in ((0, "B"), (6, "C"), (8, "B"), (14, "C"),
                             (16, "B"), (22, "E")):
            n = 6 if label == "B" else 2
            sections.append({"label": label, "bar0": start, "bar1": start + n,
                             "d_bars": 2, "reps": n // 2})
        tri[:, 0] = 1.0
        tri[6 * FPB:8 * FPB, 0] = 0.0
        tri[6 * FPB:8 * FPB, 9] = 1.0          # only C's first pass
        probs[0] = tri
        out, _ = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), agree_min=0.0, report=True)
        assert out[0][6 * FPB + 5, 9] == pytest.approx(0.5)
        assert out[0][14 * FPB + 5, 9] == pytest.approx(0.5)
        assert out[0][22 * FPB + 5, 9] == pytest.approx(0.0)


class TestDisagreementGuard:
    """3. Folding amplifies a grouping error — the guard must catch it."""

    def _three_occ(self, cols=(3, 3, 3)):
        """A x 3 of a 4-bar item; ``cols`` is the one-hot triad each pass plays."""
        n_bars, d = 12, 4
        n_frame = n_bars * FPB
        probs = _probs(n_frame)
        tri = np.zeros((n_frame, 73), dtype=np.float64)
        for r, col in enumerate(cols):
            tri[r * d * FPB:(r + 1) * d * FPB, col] = 1.0
        probs[0] = tri
        sections = [{"label": "A", "bar0": 0, "bar1": 12, "d_bars": d, "reps": 3}]
        return probs, sections, n_bars

    def test_agreeing_slots_fold(self):
        probs, sections, n_bars = self._three_occ()
        out, rep = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), report=True)
        assert rep["n_slots_guarded"] == 0
        assert rep["n_slots_folded"] == 4 * FPB

    def test_disagreeing_slots_are_skipped_untouched(self):
        # three occurrences, three DIFFERENT chords -> mean pairwise cosine 0.0,
        # i.e. "these spans are not the same music". Guarded at the default.
        probs, sections, n_bars = self._three_occ((3, 40, 55))
        out, rep = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), report=True)
        assert rep["n_slots_guarded"] == 4 * FPB
        assert rep["n_slots_folded"] == 0
        assert np.array_equal(out[0], probs[0])

    def test_one_dissenting_pass_out_of_three_is_blocked_at_the_default(self):
        # 2 of 3 agree -> mean pairwise cosine (1 + 0 + 0) / 3 = 0.333, BELOW the
        # calibrated default 0.60, so it is refused. With n=3 one bad pass is a
        # third of the evidence; the 7-song A/B says refusing is worth +0.9 pp
        # partial-credit over folding it (0.60 vs 0.30). It folds at 0.30.
        probs, sections, n_bars = self._three_occ((3, 3, 40))
        _, strict = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), report=True)
        _, loose = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), agree_min=0.30, report=True)
        assert strict["n_slots_guarded"] == 4 * FPB
        assert loose["n_slots_guarded"] == 0

    def test_the_same_single_dissenter_is_tolerated_with_more_occurrences(self):
        # 8 occurrences, 1 mis-decoded: (||sum u||^2 - n) / (n(n-1))
        #   = (7^2 + 1^2 - 8) / 56 = 0.75 >= 0.60 -> folds.
        # The guard is strict when the evidence is thin and tolerant when it is
        # thick, which is the behaviour we want and is NOT a tuned special case.
        n_bars, d = 16, 2
        n_frame = n_bars * FPB
        probs = _probs(n_frame)
        tri = np.zeros((n_frame, 73), dtype=np.float64)
        tri[:, 3] = 1.0
        tri[7 * d * FPB:8 * d * FPB, 3] = 0.0
        tri[7 * d * FPB:8 * d * FPB, 40] = 1.0
        probs[0] = tri
        sections = [{"label": "A", "bar0": 0, "bar1": 16, "d_bars": d, "reps": 8}]
        _, rep = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), report=True)
        assert rep["n_slots_guarded"] == 0
        assert rep["agree"].min() == pytest.approx(0.75)

    def test_guard_threshold_is_honoured(self):
        probs, sections, n_bars = self._three_occ((3, 3, 40))
        _, rep = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), agree_min=0.30, report=True)
        assert rep["n_slots_guarded"] == 0        # 0.333 >= 0.30 -> folds

    def test_bass_stream_is_never_folded(self):
        n_bars, d = 12, 4
        n_frame = n_bars * FPB
        probs = _probs(n_frame)
        probs[0] = _one_hot(n_frame, 73, 3)
        bass = np.zeros((n_frame, 13), dtype=np.float64)
        for r in range(3):                       # a different inversion each pass
            bass[r * d * FPB:(r + 1) * d * FPB, r + 1] = 1.0
        probs[1] = bass
        sections = [{"label": "A", "bar0": 0, "bar1": 12, "d_bars": d, "reps": 3}]
        out, _ = mpf.fold_frame_posteriors(
            probs, sections, _bounds(n_bars), agree_min=0.0, report=True)
        assert np.array_equal(out[1], probs[1])
        assert mpf.FOLDED_STREAMS == (0, 2, 3, 4, 5)


class TestDeferrals:
    """A no-op is always safe; a wrong grouping averages different music."""

    @pytest.mark.parametrize("sections", [None, []])
    def test_no_sections_is_identity(self, sections):
        n_frame = 12 * FPB
        probs = _probs(n_frame)
        out, rep = mpf.fold_frame_posteriors(
            probs, sections, _bounds(12), report=True)
        for a, b in zip(out, probs):
            assert np.array_equal(a, b)
        assert rep["n_slots_folded"] == 0

    def test_no_bars_is_identity(self):
        probs = _probs(12 * FPB)
        out, _ = mpf.fold_frame_posteriors(probs, [], [], report=True)
        for a, b in zip(out, probs):
            assert np.array_equal(a, b)

    def test_single_occurrence_item_is_identity(self):
        n_frame = 12 * FPB
        probs = _probs(n_frame)
        probs[0] = _one_hot(n_frame, 73, 3)
        sections = [{"label": "D", "bar0": 0, "bar1": 8, "d_bars": 8, "reps": 1}]
        out, rep = mpf.fold_frame_posteriors(
            probs, sections, _bounds(12), agree_min=0.0, report=True)
        assert np.array_equal(out[0], probs[0])
        assert rep["n_slots_folded"] == 0


class TestFlagOff:
    """4. The kill switch, and byte-identity when it is thrown.

    The default flipped OFF -> ON on 2026-07-30 on the strength of Brick-0 over
    all 7 verified songs (partial +2.12 pp, strict +1.43 pp, LOSO +1.87/+1.17,
    reproduced independently twice). What still has to hold is that
    ``HARMONIA_MUSX_FOLD=0`` restores the one-pass decode exactly.
    """

    def test_enabled_defaults_on(self, monkeypatch):
        monkeypatch.delenv("HARMONIA_MUSX_FOLD", raising=False)
        assert mpf.enabled() is True

    def test_zero_is_the_kill_switch(self, monkeypatch):
        monkeypatch.setenv("HARMONIA_MUSX_FOLD", "0")
        assert mpf.enabled() is False

    @pytest.mark.parametrize("val,want", [("1", True), ("on", True),
                                          ("0", False), ("", False)])
    def test_enabled_reads_the_env(self, monkeypatch, val, want):
        monkeypatch.setenv("HARMONIA_MUSX_FOLD", val)
        assert mpf.enabled() is want

    def test_chord_head_gates_the_fold_on_the_flag(self, monkeypatch):
        """The wiring in chord_head must be gated on ``enabled()`` alone."""
        import inspect

        from harmonia.stages import chord_head
        src = inspect.getsource(chord_head.NNLS24ChordHead.label_stage)
        assert "musx_posterior_fold" in src
        i = src.index("musx_posterior_fold")
        assert "enabled()" in src[i:i + 400]


class TestDegradesToPassOne:
    """Production rule: the fold is an ENHANCEMENT of a decode we already have.
    Anything that goes wrong after pass 1 must cost the fold, never the
    re-decode — otherwise ``chord_head``'s single except-block swallows pass 1
    too and the song silently falls back to NNLS root-change segmentation.
    """

    LAB1 = [(0.0, 2.0, "C:maj"), (2.0, 4.0, "G:7")]

    def _patch(self, monkeypatch, **kw):
        from harmonia.models import musx_redecode as mxr
        monkeypatch.setattr(mxr, "frame_posteriors",
                            lambda *a, **k: _probs(8 * FPB))
        monkeypatch.setattr(mxr, "redecode", lambda *a, **k: (self.LAB1, 0.12))
        for name, fn in kw.items():
            monkeypatch.setattr(mpf, name, fn)

    def test_vocabulary_declining_returns_pass_one(self, monkeypatch):
        self._patch(monkeypatch, vocab_from_chords=lambda *a, **k: None)
        lab, lat, st = mpf.two_pass_redecode("x.wav", [0.0, 1.0, 2.0])
        assert lab == self.LAB1 and lat == 0.12
        assert st["deferred"] and "vocabulary" in st["reason"]

    def test_vocabulary_raising_returns_pass_one(self, monkeypatch):
        def boom(*a, **k):
            raise ValueError("nope")

        self._patch(monkeypatch, vocab_from_chords=boom)
        lab, lat, st = mpf.two_pass_redecode("x.wav", [0.0, 1.0, 2.0])
        assert lab == self.LAB1
        assert st["reason"] == "vocabulary raised ValueError"

    def test_fold_raising_returns_pass_one(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("nope")

        self._patch(monkeypatch,
                    vocab_from_chords=lambda *a, **k: ([{"label": "A", "bar0": 0,
                                                         "bar1": 8, "d_bars": 4,
                                                         "reps": 2}],
                                                       _bounds(8), 8),
                    fold_frame_posteriors=boom)
        lab, lat, st = mpf.two_pass_redecode("x.wav", [0.0, 1.0, 2.0])
        assert lab == self.LAB1
        assert st["reason"] == "fold raised RuntimeError"

    def test_guard_blocking_everything_returns_pass_one(self, monkeypatch):
        self._patch(monkeypatch,
                    vocab_from_chords=lambda *a, **k: ([{"label": "A", "bar0": 0,
                                                         "bar1": 8, "d_bars": 4,
                                                         "reps": 2}],
                                                       _bounds(8), 8))
        lab, lat, st = mpf.two_pass_redecode("x.wav", [0.0, 1.0, 2.0],
                                            agree_min=1.01)
        assert lab == self.LAB1
        assert st["reason"] == "disagreement guard blocked every slot"


class TestLabelsToChords:
    """Pass 1's chords, in the shape the chart path (rigid_grid/vocab) reads."""

    def test_roots_and_ireal_qualities(self):
        lab = [(0.0, 1.0, "C:maj"), (1.0, 2.0, "A:min7"), (2.0, 3.0, "G:7"),
               (3.0, 4.0, "F:maj7/3"), (4.0, 5.0, "B:hdim7")]
        got = mpf.chords_from_labels(lab)
        assert [c["root"] for c in got] == [0, 9, 7, 5, 11]
        assert [c["q"] for c in got] == ["", "-7", "7", "^7", "h7"]
        assert [(c["t0"], c["t1"]) for c in got][0] == (0.0, 1.0)

    def test_no_chord_becomes_an_explicit_nc_slot(self):
        got = mpf.chords_from_labels([(0.0, 1.0, "N"), (1.0, 2.0, "C:maj")])
        assert got[0]["nc"] is True
        assert got[0]["q"] == "N"
        assert got[1]["nc"] is False
