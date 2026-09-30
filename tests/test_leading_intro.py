"""tests/test_leading_intro.py — the chart starts on the first bar with music.

Louis, 2026-09-30: « on commence toujours une mesure trop tôt j'ai
l'impression, à chaque fois le chart commence par un % ou un N.C. ce qui n'a
aucun sens ». Measured on the library (HEAD bake, 46 warm charts): 13 print
« N.C. » or « % » as their first bar. For the 10 without hand sections the
cause is the same: the bar grid starts on the tracker's first beat, the music
(or Louis's bar-1 mark) comes one to seven bars later, and since SongFormer
was dropped (bbb0ab0, 2026-09-19) the chart is ONE section « A » that holds
those empty lead-in bars. Before that, `_force_bar1_sections` put everything
before the mark in an « intro » section, which chart.js hides in Read when it
has no chord (a hole in the form strip). `_one_section` lost that split.

Contract pinned here (`pipeline._one_section`):
* leading bars without a single real chord become a leading « intro »
  section (chart.js then hides it in Read, keeps it in Annotate);
* with a bar-1 mark, the bars before the mark are the intro — the chordless
  head is split off first, pre-mark bars WITH chords stay a visible intro;
* the mark is sovereign: a chordless bar AT or after the mark is never hidden;
* nothing is cut from the grid: bar indices, barGrid and every bar's chords
  are untouched (hand sections and marks are stored as absolute bar indices).
"""
from __future__ import annotations

from harmonia.pipeline import _one_section


def _nc(b):
    return [{"root": 0, "q": "", "nc": True, "bar": b, "beat": 0}]


def _ch(b, root=0):
    return [{"root": root, "q": "", "nc": False, "bar": b, "beat": 0}]


def _grid(n):
    return [2.0 * i for i in range(n + 1)]


def _shape(secs):
    return [(s["label"], s["barRanges"][0][0], s["barRanges"][0][1]) for s in secs]


def test_leading_nc_and_empty_bars_become_intro():
    # Hot N Cold / Billie Jean shape: N.C. | (empty) | first chord
    bars = [_nc(0), [], _ch(2), _ch(3, 7), _ch(4)]
    secs = _one_section(bars, _grid(5), 5)
    assert _shape(secs) == [("intro", 0, 1), ("A", 2, 4)]
    # every bar still present, same objects, same order
    assert [b for s in secs for b in s["bars"]] == bars
    assert secs[0]["spans"] == [[0.0, 4.0]] and secs[1]["spans"] == [[4.0, 10.0]]
    assert len(secs[0]["barSpans"]) == 2 and len(secs[1]["barSpans"]) == 3


def test_music_from_bar_zero_stays_one_section():
    bars = [_ch(0), _ch(1, 5), _ch(2)]
    assert _shape(_one_section(bars, _grid(3), 3)) == [("A", 0, 2)]


def test_all_nc_song_stays_one_section():
    bars = [_nc(0), [], _nc(2)]
    assert _shape(_one_section(bars, _grid(3), 3)) == [("A", 0, 2)]


def test_mark_splits_premark_music_into_visible_intro():
    # N.C. | C | G | [mark] F ... : the chordless bar is the hidden head,
    # C G before the mark is a real (printed) intro, the mark starts A
    bars = [_nc(0), _ch(1), _ch(2, 7), _ch(3, 5), _ch(4)]
    secs = _one_section(bars, _grid(5), 5, bar1_bar=3)
    assert _shape(secs) == [("intro", 0, 0), ("intro", 1, 2), ("A", 3, 4)]


def test_mark_is_sovereign_over_a_silent_marked_bar():
    # Sam Smith shape: the mark is bar 1, musx hears nothing there yet — the
    # marked bar stays in A (musx is late, not the grid)
    bars = [_nc(0), _nc(1), _ch(2), _ch(3)]
    secs = _one_section(bars, _grid(4), 4, bar1_bar=1)
    assert _shape(secs) == [("intro", 0, 0), ("A", 1, 3)]


def test_mark_at_zero_or_absent_changes_nothing_when_music_starts():
    bars = [_ch(0), _ch(1)]
    assert _shape(_one_section(bars, _grid(2), 2, bar1_bar=0)) == [("A", 0, 1)]
