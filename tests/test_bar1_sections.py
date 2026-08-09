"""tests/test_bar1_sections.py — Set bar 1 must ANCHOR the structure, red-first.

2026-08-09 report (D-major chart, gbO7qQliXT8): after the user marked bar 1,
the voice detector re-derived its own sung start ONE bar past the mark, its
1-bar post-mark "intro" stub was RENAMED to a letter by _force_bar1_sections,
and the chart came out `intro[0,0] A[1,1] B[2,9]×9` — a one-bar A.

Contract pinned here:
* a post-mark intro tail MERGES FORWARD into the section that follows (the
  mark is the start of A, by definition of the tool) — never its own letter;
* the sub-detection is told to anchor at the mark (`form_start=0` threads
  from the pipeline through sections.detect_sections to voice_sections).
"""
from __future__ import annotations

from harmonia_min.pipeline import _force_bar1_sections


def test_postmark_intro_tail_merges_into_next_letter():
    # what the voice detector actually produced on the D-major chart,
    # already offset to absolute bars by the pipeline
    segs = [{"b0": 1, "b1": 1, "label": "intro"},
            {"b0": 2, "b1": 9, "label": "A"},
            {"b0": 10, "b1": 17, "label": "A"}]
    out = _force_bar1_sections(segs, 1)
    assert out[0] == {"b0": 0, "b1": 0, "label": "intro"}   # pre-mark intro
    # the tail joined A: the marked bar STARTS the first letter section
    assert out[1]["b0"] == 1 and out[1]["b1"] == 9
    assert out[1]["label"] == "A"
    assert [s["label"] for s in out[1:]] == ["A", "A"]


def test_lone_postmark_intro_becomes_the_form():
    out = _force_bar1_sections([{"b0": 3, "b1": 10, "label": "intro"}], 3)
    assert out[-1] == {"b0": 3, "b1": 10, "label": "A"}
    assert out[0] == {"b0": 0, "b1": 2, "label": "intro"}


def test_straddling_section_still_cut_at_mark():
    segs = [{"b0": 0, "b1": 7, "label": "A"}, {"b0": 8, "b1": 15, "label": "B"}]
    out = _force_bar1_sections(segs, 4)
    assert out[0] == {"b0": 0, "b1": 3, "label": "intro"}
    assert out[1]["b0"] == 4 and out[1]["label"] == "A"
    assert out[2]["label"] == "B"


def test_dispatcher_threads_form_start(monkeypatch):
    from harmonia_min import sections as S
    seen = {}

    def fake_voice(grid, triad, bars=None, audio=None, form_start=None):
        seen["form_start"] = form_start
        return [{"b0": 0, "b1": len(grid) - 2, "label": "A"}]

    monkeypatch.setenv("HARMONIA_SECTIONS", "voice")
    import harmonia_min.voice_sections as V
    monkeypatch.setattr(V, "detect_sections", fake_voice)
    S.detect_sections([0.0, 1.0, 2.0], None, None, bars=[[], []],
                      triad=object(), audio="x.m4a", form_start=0)
    assert seen["form_start"] == 0
