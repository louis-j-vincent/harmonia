"""Mission 5 Part A — tests for the LLM-prior glue into the joint decode.

Red-first (CLAUDE.md): the unit test pins the factor math the glue MUST produce
from a hand-built analysis JSON (tonic gate, per-root q5 bonus, repeat→segment
pooling) before trusting any pipeline number. The integration test drives the
real ``infer_chords_v1`` and checks the q5_bonus actually reaches the emission
(labels move toward an extreme prior) — the end-to-end wiring the audit found
missing.
"""
from pathlib import Path

import numpy as np
import pytest

from harmonia.models.chord_pipeline_v1 import (
    LLM_KEY_TRUST,
    apply_llm_priors,
    bars_to_segment_groups,
)

REPO = Path(__file__).resolve().parents[1]

# Q5 index convention (matches Q5_NAMES in llm_chord_priors / semi_markov_decode).
MAJ, MIN, DOM, HDIM, DIM = 0, 1, 2, 3, 4


def _analysis(*, tonic_pc, confidence, chord_priors=None,
              sections=None, repeats=None):
    return {
        "key": "G minor",
        "mode": "minor",
        "tonic_pc": tonic_pc,
        "structure": {
            "form": "",
            "sections": sections or [],
            "repeats": repeats or [],
        },
        "chord_priors": chord_priors or {},
        "transition_priors": {},
        "confidence": confidence,
    }


# ── Seam 1: tonic KEY_TRUST gate ──────────────────────────────────────────────

def test_tonic_used_when_confidence_high():
    an = _analysis(tonic_pc=7, confidence=LLM_KEY_TRUST + 0.05)
    out = apply_llm_priors(an, segs=[(0, 4)], beat_times=np.arange(5),
                           inferred_tonic=0)
    assert out["tonic"] == 7  # analyst tonic wins


def test_inferred_tonic_used_when_confidence_low():
    an = _analysis(tonic_pc=7, confidence=LLM_KEY_TRUST - 0.05)
    out = apply_llm_priors(an, segs=[(0, 4)], beat_times=np.arange(5),
                           inferred_tonic=3)
    assert out["tonic"] == 3  # gate falls back to the audio-inferred tonic


# ── Seam 2: per-root q5 quality bonus callback ────────────────────────────────

def test_q5_bonus_favours_prior_quality_for_present_root():
    # root 2 (D): prior mass on dominant → q5_bonus argmax must be DOM.
    cp = {"2": {"maj": 0.05, "min": 0.05, "dom": 0.80, "hdim": 0.05, "dim": 0.05}}
    an = _analysis(tonic_pc=7, confidence=0.9, chord_priors=cp)
    out = apply_llm_priors(an, segs=[(0, 4)], beat_times=np.arange(5),
                           inferred_tonic=7)
    row = out["q5_bonus"](0, 2)
    assert row.shape == (5,)
    assert int(np.argmax(row)) == DOM
    assert row[DOM] > 0 and row[MAJ] < 0  # centred bonus: winner +, others −


def test_q5_bonus_zero_for_absent_root():
    cp = {"2": {"maj": 0.2, "min": 0.2, "dom": 0.2, "hdim": 0.2, "dim": 0.2}}
    an = _analysis(tonic_pc=7, confidence=0.9, chord_priors=cp)
    out = apply_llm_priors(an, segs=[(0, 4)], beat_times=np.arange(5),
                           inferred_tonic=7)
    assert np.allclose(out["q5_bonus"](0, 9), np.zeros(5))  # root 9 not in prior


def test_q5_bonus_strength_scales_with_confidence():
    cp = {"2": {"maj": 0.05, "min": 0.05, "dom": 0.80, "hdim": 0.05, "dim": 0.05}}
    hi = apply_llm_priors(_analysis(tonic_pc=7, confidence=0.9, chord_priors=cp),
                          segs=[(0, 4)], beat_times=np.arange(5), inferred_tonic=7)
    lo = apply_llm_priors(_analysis(tonic_pc=7, confidence=0.3, chord_priors=cp),
                          segs=[(0, 4)], beat_times=np.arange(5), inferred_tonic=7)
    # honesty knob: a less-confident analyst tilts the emission less.
    assert hi["q5_bonus"](0, 2)[DOM] > lo["q5_bonus"](0, 2)[DOM] > 0


# ── Seam 3: repeat bar-spans → tied segment groups ────────────────────────────

def test_bars_to_segment_groups_slotwise():
    # 4 bars, 4 beats/bar; one segment per bar. Sections (1-2) ∥ (3-4).
    segs = [(0, 4), (4, 8), (8, 12), (12, 16)]
    pool_group_bars = [[(1, 2), (3, 4)]]
    groups = bars_to_segment_groups(pool_group_bars, segs, np.arange(17),
                                    beats_per_bar=4)
    # slot 0: bar1(seg0) ∥ bar3(seg2); slot 1: bar2(seg1) ∥ bar4(seg3).
    assert [sorted(g) for g in groups] == [[0, 2], [1, 3]]


def test_pool_groups_flow_through_apply():
    segs = [(0, 4), (4, 8), (8, 12), (12, 16)]
    sections = [{"label": "a", "start_bar": 1, "end_bar": 2},
                {"label": "a", "start_bar": 3, "end_bar": 4}]
    an = _analysis(tonic_pc=7, confidence=0.9, sections=sections,
                   repeats=[[0, 1]])
    out = apply_llm_priors(an, segs=segs, beat_times=np.arange(17),
                           inferred_tonic=7)
    assert [sorted(g) for g in out["pool_groups"]] == [[0, 2], [1, 3]]


def test_no_group_when_single_span():
    # a lone span (no parallel partner) yields no tie.
    assert bars_to_segment_groups([[(1, 2)]], [(0, 4), (4, 8)],
                                  np.arange(9), beats_per_bar=4) == []


# ── offline analyst → glue smoke test (real chart) ────────────────────────────

def test_autumn_leaves_offline_analysis_produces_factors():
    """The offline analyst on a real chart yields a usable factor bundle:
    a minor tonic, a non-empty per-root quality prior, and (Autumn Leaves)
    a pooled repeat group mapping onto segments."""
    from scripts.llm_chord_priors import load_chart, offline_analyze
    pl = REPO / "data" / "ireal" / "jazz1460.txt"
    if not pl.exists():
        pytest.skip("jazz1460 playlist not available")
    chart = load_chart("Autumn Leaves", pl)
    an = offline_analyze(chart)
    # 32-bar AABC form; one segment per bar for the mapping check.
    n_bars = len(chart.sections)
    segs = [(4 * i, 4 * (i + 1)) for i in range(n_bars)]
    out = apply_llm_priors(an, segs=segs, beat_times=np.arange(4 * n_bars + 1),
                           inferred_tonic=0)
    f = out["factors"]
    assert f.confidence >= LLM_KEY_TRUST        # clean diatonic chart → trusted
    assert out["tonic"] == f.tonic              # so the analyst tonic is used
    assert len(f.quality_bonus) > 0             # per-root quality prior present
    assert len(out["pool_groups"]) > 0          # repeat strain → tied segments


# ── Integration: q5_bonus reaches the decoder end-to-end ──────────────────────

_RENDER = REPO / "data" / "renders" / "pop909" / "001" / "001_v005_musescoregeneral.wav"


@pytest.mark.slow
@pytest.mark.skipif(not _RENDER.exists(), reason="POP909 render 001 not available")
def test_llm_priors_shift_labels_end_to_end():
    """Driving infer_chords_v1 with an extreme dominant-everywhere prior must
    push the decoded qualities toward DOM vs the use_llm_priors=False run — proof
    the q5_bonus is wired into the real emission (not a no-op)."""
    from collections import Counter

    from harmonia.models.chord_pipeline_v1 import _HARTE_TO_Q5NAME, infer_chords_v1

    cd = REPO / "data" / "cache"

    def dom_frac(chart):
        c = Counter(_HARTE_TO_Q5NAME.get(ch["label"].split(":")[-1], "?")
                    for ch in chart.chords)
        return c.get("dom", 0) / max(sum(c.values()), 1)

    base = infer_chords_v1(_RENDER, cache_dir=cd, use_llm_priors=False)
    an = _analysis(
        tonic_pc=6, confidence=0.95,
        chord_priors={str(pc): {"maj": 0.02, "min": 0.02, "dom": 0.92,
                                "hdim": 0.02, "dim": 0.02} for pc in range(12)},
    )
    guided = infer_chords_v1(_RENDER, cache_dir=cd, use_llm_priors=True,
                             llm_analysis=an, llm_max_nats=8.0)
    assert dom_frac(guided) > dom_frac(base) + 0.20  # prior visibly moved labels
