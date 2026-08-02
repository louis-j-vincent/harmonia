"""The intervention gate: two-sided, proposal-only, and safe when absent."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from harmonia_min.chord_lm import intervene, vocab

CHARTS = Path("harmonia_min/state/charts")


def _chart(bpb=4, bars=None, confs=None):
    bars = bars or [[(0, "", 0)], [(7, "", 0)], [(9, "-", 0)], [(5, "", 0)]]
    confs = confs or [0.9] * len(bars)
    return {
        "title": "t", "bpb": bpb, "nBars": len(bars),
        "barGrid": [2.0 * i for i in range(len(bars) + 1)],
        "sections": [{"label": "A", "barRanges": [[0, len(bars) - 1]], "bars": [
            [{"root": r, "q": q, "bass": -1, "nc": False, "beat": b, "c": c}
             for (r, q, b) in bar]
            for bar, c in zip(bars, confs)]}],
    }


def test_slot_confidence_is_one_value_per_slot():
    ch = _chart(confs=[0.9, 0.2, 0.5, 0.7])
    conf = intervene._slot_confidence(ch, 2)
    assert len(conf) == 8
    # each bar's single chord sits at beat 0, so it owns both half-bars
    assert list(np.round(conf, 3)) == [0.9, 0.9, 0.2, 0.2, 0.5, 0.5, 0.7, 0.7]


def test_slot_confidence_follows_a_mid_bar_chord_change():
    ch = _chart(bars=[[(0, "", 0), (7, "7", 2)]], confs=[0.9])
    ch["sections"][0]["bars"][0][1]["c"] = 0.3
    conf = intervene._slot_confidence(ch, 2)
    assert list(np.round(conf, 3)) == [0.9, 0.3]


def test_propose_returns_empty_without_a_model(monkeypatch, tmp_path):
    monkeypatch.setenv("HARMONIA_CHORD_LM", str(tmp_path / "absent.pt"))
    monkeypatch.setattr(intervene, "_MODEL", None)
    assert intervene.propose(_chart()) == []


def test_propose_is_empty_on_a_chart_too_short_for_context(monkeypatch):
    monkeypatch.setattr(intervene, "_MODEL", object())
    monkeypatch.setattr(intervene, "load_model", lambda device="cpu": object())
    assert intervene.propose(_chart(bars=[[(0, "", 0)]])) == []


def test_name_resolves_rep_against_what_it_stands_for():
    c = vocab.chord_id(0, "maj")
    assert intervene._name(c, None) == "C:maj"
    assert intervene._name(vocab.REP, c) == "% (C:maj)"
    assert intervene._name(vocab.REP, None) == "%"
    assert intervene._name(vocab.NC, c) == "N.C."


@pytest.mark.skipif(not Path("data/models/chord_lm_rope_masked.pt").exists(),
                    reason="chord LM checkpoint not present")
@pytest.mark.skipif(not CHARTS.exists(), reason="no state charts")
def test_gate_is_two_sided_on_a_real_chart():
    """Raising the pipeline ceiling can only ADD proposals; raising the LM floor
    can only remove them. If either stops mattering the gate is one-sided."""
    chart = json.loads(sorted(CHARTS.glob("*.json"))[0].read_text())
    loose = intervene.propose(chart, lm_min=0.60, pipe_max=2.0)
    tight = intervene.propose(chart, lm_min=0.60, pipe_max=0.50)
    assert len(tight) <= len(loose)
    stricter = intervene.propose(chart, lm_min=0.95, pipe_max=2.0)
    assert len(stricter) <= len(loose)


@pytest.mark.skipif(not Path("data/models/chord_lm_rope_masked.pt").exists(),
                    reason="chord LM checkpoint not present")
@pytest.mark.skipif(not CHARTS.exists(), reason="no state charts")
def test_suggestions_never_repeat_what_the_chart_already_says():
    chart = json.loads(sorted(CHARTS.glob("*.json"))[0].read_text())
    for s in intervene.propose(chart, lm_min=0.70, pipe_max=0.70):
        assert s.proposed != s.shown
        assert len(s.alternatives) == intervene.TOP_K
        assert s.lm_conf >= 0.70 and s.pipe_conf <= 0.70


def test_pipeline_flag_is_off_by_default():
    """The chart must carry no suggestions unless the env flag asks for them —
    measured net was 0 on the only verified corpus, so nothing ships on."""
    assert os.environ.get("HARMONIA_CHORD_LM_SUGGEST") != "1", (
        "the flag is set in this environment; the default-off assertion is "
        "meaningless here")
    src = Path("harmonia_min/pipeline.py").read_text()
    assert 'os.environ.get("HARMONIA_CHORD_LM_SUGGEST") == "1"' in src
    assert "lmSuggestions" in src
