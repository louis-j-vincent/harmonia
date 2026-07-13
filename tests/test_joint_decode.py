"""Unit tests for the segment-level joint (root × quality) decode.

Follows CLAUDE.md rule #1: unit-test the load-bearing bigram-transition lookup
against a hand-computed case BEFORE trusting any end-to-end number, plus a tiny
synthetic Viterbi with a known answer and the P1 transposition-invariance check.
"""
import numpy as np
import pytest

from harmonia.models import joint_decode as J
from harmonia.theory.progression_prior import FAMILIES, _FI, _state


def test_prog_state_roundtrip():
    """prog_state must reproduce progression_prior._state for every (deg, q5)."""
    q5_fam = {0: "major", 1: "minor", 2: "major", 3: "diminished", 4: "diminished"}
    for deg in range(12):
        for q5, fam in q5_fam.items():
            assert J.prog_state(deg, q5) == _state(deg, fam)
    # degree wraps mod 12
    assert J.prog_state(14, 0) == J.prog_state(2, 0)


def test_transition_lookup_matches_hand_computed():
    """transition_logscore must read the exact bigram cell for the state pair."""
    logp = J.load_bigram()
    tonic = 5  # F
    # prev = ii min at degree 2, cur = V dom at degree 7 (a ii–V in the key)
    prev_root, prev_q5 = (tonic + 2) % 12, 1      # degree 2, minor
    cur_root, cur_q5 = (tonic + 7) % 12, 2        # degree 7, dom (major-family)
    si = _state(2, "minor")
    sj = _state(7, "major")
    expected = float(logp[si, sj])
    got = J.transition_logscore(prev_root, prev_q5, cur_root, cur_q5, tonic, logp)
    assert got == pytest.approx(expected)


def test_transition_transposition_invariance():
    """P1: transposing tonic AND both roots by the same amount is a no-op."""
    logp = J.load_bigram()
    base = J.transition_logscore(2, 1, 7, 2, tonic=0, bigram_logp=logp)
    for shift in range(1, 12):
        r1, r2, ton = (2 + shift) % 12, (7 + shift) % 12, shift % 12
        got = J.transition_logscore(r1, 1, r2, 2, tonic=ton, bigram_logp=logp)
        assert got == pytest.approx(base), f"not invariant under +{shift}"


def _uniform_bigram():
    """A flat transition table so decode is driven purely by emissions."""
    return np.log(np.full((60, 60), 1.0 / 60.0, dtype=np.float64))


def test_viterbi_emission_only_known_answer():
    """weight=0 → per-segment joint-emission argmax; hand-checkable answer.

    Two segments. beat_proba pins root 0 hard in seg0, root 7 hard in seg1.
    classify_fn makes q5=1 (min) the clear winner in seg0, q5=2 (dom) in seg1.
    With transition_weight=0 the MAP must be [(0,'min'), (7,'7')].
    """
    beat_proba = np.zeros((2, 12), dtype=np.float32)
    beat_proba[0, 0] = 1.0
    beat_proba[1, 7] = 1.0
    segs = [(0, 1), (1, 2)]

    def classify_fn(idx, root):
        lp = np.full(5, np.log(0.02))
        if idx == 0:
            lp[1] = np.log(0.9)   # min
        else:
            lp[2] = np.log(0.9)   # dom
        lp = lp - np.log(np.exp(lp).sum())
        # greedy sev_h consistent with the peaked q5 (triad form for seg0)
        sev = "min" if idx == 0 else "7"
        return "min", sev, 0.9, lp

    out = J.joint_decode(segs, beat_proba, classify_fn, tonic=0,
                         K=1, transition_weight=0.0)
    assert out["roots"] == [0, 7]
    assert out["q5"] == [1, 2]
    assert out["sev_h"] == ["min", "7"]  # dom's canonical form is "7"


def test_w0_k1_reproduces_greedy_on_contaminated_q5():
    """Greedy anchor: w=0, K=1 must reproduce the greedy labels EXACTLY even
    when the q5 log-probs are aug/sus-contaminated (argmax=maj) while the
    classifier's own greedy call is minor — the first-gate failure mode.
    """
    beat_proba = np.zeros((2, 12), dtype=np.float32)
    beat_proba[0, 2] = 1.0   # D
    beat_proba[1, 7] = 1.0   # G
    segs = [(0, 1), (1, 2)]

    def classify_fn(idx, root):
        # p_fam=[.30 maj,.40 min,.05 dim,.15 aug,.10 sus] folded through
        # _family_q5_logprobs → maj gets .30+.15+.10=.55 > min .40:
        # contaminated argmax=maj, but the family head (and greedy sev_h)
        # says MINOR.
        p = np.array([0.55, 0.40, 0.02, 0.02, 0.01])
        lp = np.log(p / p.sum())
        return ("min", "min7", 0.4, lp) if idx == 0 else ("min", "min", 0.4, lp)

    out = J.joint_decode(segs, beat_proba, classify_fn, tonic=0,
                         K=1, transition_weight=0.0)
    assert out["roots"] == [2, 7]
    assert out["q5"] == [1, 1]
    assert out["sev_h"] == ["min7", "min"]   # greedy labels, seventh bit kept


def test_viterbi_transition_overrides_flat_emission():
    """A strong transition prior must flip a near-tie root toward the grammar.

    seg0 root 0 (I). seg1: roots 5 and 7 are a near tie in beat_proba, and the
    quality evidence is flat. A bigram that strongly prefers I→V (degree 7)
    over I→IV (degree 5) must select root 7 once the transition weight is on.
    """
    beat_proba = np.zeros((2, 12), dtype=np.float32)
    beat_proba[0, 0] = 1.0
    beat_proba[1, 5] = 0.51
    beat_proba[1, 7] = 0.49
    segs = [(0, 1), (1, 2)]

    def classify_fn(idx, root):
        lp = np.log(np.full(5, 1.0 / 5))  # totally flat quality
        return "maj", "maj", 0.5, lp

    logp = np.log(np.full((60, 60), 1e-6, dtype=np.float64))
    # strongly prefer I(maj,deg0) → V(maj,deg7)
    si = _state(0, "major")
    logp[si, _state(7, "major")] = np.log(0.9)
    logp[si, _state(5, "major")] = np.log(0.001)

    # weight 0: emission-only → root 5 (higher beat_proba)
    out0 = J.joint_decode(segs, beat_proba, classify_fn, tonic=0,
                          K=2, transition_weight=0.0, bigram_logp=logp)
    assert out0["roots"][1] == 5
    # weight high: transition prior flips seg1 to root 7 (the V)
    out1 = J.joint_decode(segs, beat_proba, classify_fn, tonic=0,
                          K=2, transition_weight=5.0, bigram_logp=logp)
    assert out1["roots"][1] == 7


def test_marginals_are_distributions():
    """Forward–backward marginals per segment must sum to 1."""
    beat_proba = np.zeros((3, 12), dtype=np.float32)
    beat_proba[0, 0] = 1.0
    beat_proba[1, 5] = 0.6
    beat_proba[1, 7] = 0.4
    beat_proba[2, 0] = 1.0
    segs = [(0, 1), (1, 2), (2, 3)]

    def classify_fn(idx, root):
        lp = np.log(np.full(5, 1.0 / 5))
        return "maj", "maj", 0.5, lp

    out = J.joint_decode(segs, beat_proba, classify_fn, tonic=0,
                         K=2, transition_weight=1.0)
    for m in out["marginals"]:
        assert m.sum() == pytest.approx(1.0, abs=1e-6)
    assert len(out["roots"]) == 3
