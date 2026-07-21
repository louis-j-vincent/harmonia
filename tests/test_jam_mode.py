"""Jam Mode (2026-07-20) — loop-detection + vote-refinement unit tests.

fast_draft_decode/JamSession need real audio + the NNLS-24 heads (see
scratchpad/jam_premise_check.py for the end-to-end real-audio validation);
these tests cover the pure-Python pieces that don't need either, so they run
fast and everywhere.
"""
from __future__ import annotations

from harmonia.models.jam_mode import LoopVotes, detect_loop_period


class TestDetectLoopPeriod:
    def test_clean_period_detected(self):
        seq = (["A:maj", "A:maj", "D:maj", "D:maj"] * 6)
        p, score = detect_loop_period(seq, min_p=2)
        assert p == 4
        assert score == 1.0

    def test_noisy_sequence_abstains(self):
        import random
        rng = random.Random(0)
        seq = [rng.choice(["A:maj", "B:min", "C:maj", "D:min", "E:maj"]) for _ in range(40)]
        assert detect_loop_period(seq) is None

    def test_prefers_smallest_period_occam(self):
        # A period-4 loop trivially also "matches" at period 8, 12, … — must
        # return the smallest, not the first the caller happens to test.
        seq = ["A:maj", "B:min"] * 20
        p, _ = detect_loop_period(seq, min_p=2)
        assert p == 2


class TestLoopVotes:
    def test_majority_vote_converges_despite_noise(self):
        # 3 clean reps of a 2-slot loop + 1 noisy rep with one wrong label —
        # majority should still recover the clean pattern. add() is called
        # ONCE with the full sequence, matching real usage (JamSession.update
        # builds one fresh LoopVotes per poll from that poll's full window —
        # see its module docstring on why votes are never accumulated
        # incrementally across polls).
        votes = LoopVotes(period=2)
        reps = [["A:maj", "D:maj"]] * 3 + [["A:maj", "G:maj"]]
        votes.add([lbl for rep in reps for lbl in rep])
        best = votes.best()
        assert [c["label"] for c in best] == ["A:maj", "D:maj"]
        assert best[0]["confidence"] == 1.0    # unanimous
        assert best[1]["confidence"] == 0.75   # 3/4 — the noisy rep's miss shows up

    def test_n_reps_counts_full_cycles_only(self):
        votes = LoopVotes(period=4)
        votes.add(["A", "B", "C", "D", "A", "B"])   # 1.5 cycles
        assert votes.n_reps() == 1

    def test_empty_slot_yields_no_chord_zero_confidence(self):
        votes = LoopVotes(period=3)
        votes.add(["A", "B"])   # slot 2 never observed
        best = votes.best()
        assert best[2] == {"label": "N", "confidence": 0.0}
