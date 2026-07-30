"""The audio-seconds -> rendered-bar map the playhead runs on.

Bug (Louis, 2026-07-30): "the play head does n'importe quoi ... it skips
sections, doesn't play the first bars when 2 consecutive bars share the same
chord, and ends up being on the wrong chords and being either early or late".

The chart is MINIMAL: each distinct section is written ONCE and replayed on
every pass, so the grid has ~25 rendered bars for an ~80-bar song. The client
therefore has to reconstruct, for every rendered bar and every pass, the real
audio seconds during which that bar sounds. Two invariants make the playhead
well-defined, and both were violated:

1. **Each pass is exactly tiled.** The rendered bars of pass k must partition
   that pass's own span ``sec["spans"][k]`` — no gap (playhead freezes on the
   last bar), no overrun (playhead sits on this section's bars while the NEXT
   section sounds). The old client translated pass 0's chord times rigidly onto
   each pass start (``c.t0 + span_k[0] - span_0[0]``), which only works when
   every pass has the same duration. On This Love the four passes of A last
   20.2 / 20.2 / 27.8 / 7.6 s, so pass 3 left an 8.2 s hole and pass 4 overran
   12 s into the bridge.
2. **Every rendered bar gets real time on every pass it plays.** A held ("%")
   bar carries no chord because the previous chord sustains through it. Timing
   it as ``[previous chord's end, next chord's start]`` makes it ZERO-length —
   it can never light, and the previous bar stays lit for both bars.

Both are the same disease: bar times were read off chord SUSTAIN instead of BAR
boundaries. These tests assert the invariants on the authoritative map the
client consumes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harmonia.output.chart_display import bar_spans_for_sections, snap_bar_spans_to_beats

TOL = 1e-6


# ── the authoritative map, exactly as the client reads it ────────────────────
def rendered_bar_spans(sec: dict) -> list[list[list[float]]]:
    """``[per rendered bar][per pass slot] -> [t0, t1]`` for one section.

    Mirrors the order ``app_shell.html``'s ``loadModel`` emits rendered bars:
    the shared prefix first, then each 1st/2nd-ending variant's tail bars. A
    prefix bar has one slot per pass; a variant bar has one slot per pass in
    ``variant["passes"]``.
    """
    return sec["barSpans"]


def pass_slots(sec: dict) -> list[tuple[list[int], list[int]]]:
    """``[(rendered bar indices, pass indices)]`` per rendered block."""
    n_bars = len(sec["bars"])
    reps = len(sec["spans"])
    endings = sec.get("endings") or {}
    tail = endings.get("tail") or 0
    n_prefix = n_bars - tail if tail else n_bars
    blocks = [(list(range(n_prefix)), list(range(reps)))]
    nxt = n_prefix
    for v in (endings.get("variants") or []) if tail else []:
        k = len(v["bars"])
        blocks.append((list(range(nxt, nxt + k)),
                       [p for p in v["passes"] if 0 <= p < reps]))
        nxt += k
    return blocks


def check_section(sec: dict, where: str) -> list[str]:
    fails: list[str] = []
    bs = rendered_bar_spans(sec)
    blocks = pass_slots(sec)
    n_rendered = sum(len(b) for b, _ in blocks)
    if len(bs) != n_rendered:
        return [f"{where}: barSpans has {len(bs)} rendered bars, grid has {n_rendered}"]

    # per pass: the bars that pass plays must tile its own span end to end
    for k, (t0, t1) in enumerate(sec["spans"]):
        seq: list[list[float]] = []
        for bar_idxs, passes in blocks:
            if k not in passes:
                continue
            slot = passes.index(k)
            for bi in bar_idxs:
                if slot >= len(bs[bi]):
                    fails.append(f"{where} pass {k}: rendered bar {bi} has no span")
                    continue
                seq.append(bs[bi][slot])
        if not seq:
            continue
        if all(sp is None for sp in seq):
            # a pass with no time at all (chart with no audio: every span [0,0])
            continue
        if any(sp is None for sp in seq):
            fails.append(f"{where} pass {k}: some bars have time and some don't")
            continue
        if abs(seq[0][0] - t0) > 1e-3:
            fails.append(f"{where} pass {k}: starts at {seq[0][0]:.3f}, span starts {t0:.3f}")
        if abs(seq[-1][1] - t1) > 1e-3:
            fails.append(f"{where} pass {k}: ends at {seq[-1][1]:.3f}, span ends {t1:.3f}")
        for i, (a, b) in enumerate(zip(seq, seq[1:])):
            if abs(a[1] - b[0]) > 1e-3:
                fails.append(f"{where} pass {k}: bar {i}->{i + 1} "
                             f"{'gap' if b[0] > a[1] else 'overlap'} "
                             f"{abs(b[0] - a[1]):.3f}s at t={a[1]:.2f}")
        for i, sp in enumerate(seq):
            if sp[1] - sp[0] <= TOL:
                fails.append(f"{where} pass {k}: bar {i} is {sp[1] - sp[0]:.4f}s long "
                             f"— it can never light")
    return fails


def flat_intervals(sections: list[dict]) -> list[tuple[float, float, str, int]]:
    out = []
    for sec in sections:
        bs = rendered_bar_spans(sec)
        for bi, slots in enumerate(bs):
            for sp in slots:
                if sp is not None:
                    out.append((sp[0], sp[1], sec["id"], bi))
    out.sort()
    return out


# ── synthetic fixtures (never skip) ──────────────────────────────────────────
def _chord(bar: int, t0: float, t1: float, root: int = 0, beat: int = 0) -> dict:
    """A chord ON THE BAR LINE by default — `beat=0` is what makes t0 a bar line."""
    return {"root": root, "q": "", "c": 0.5, "bass": -1, "bar": bar, "beat": beat,
            "t0": t0, "t1": t1}


def _grid(n_bars: int, step: float, t0: float = 0.0) -> list[float]:
    """A steady bar grid: n_bars + 1 bar lines."""
    return [t0 + step * i for i in range(n_bars + 1)]


def test_passes_of_unequal_length_still_tile_their_own_span():
    """A folded section written once but played four times, the passes covering
    8, 8, 12 and 4 SONG bars. Each pass is timed from the grid over its own bars,
    so it tiles its own stretch exactly however many bars it really has."""
    grid = _grid(48, 2.5)
    sec = {
        "id": "A0", "label": "A", "tag": "", "reps": 4,
        "bars": [[_chord(i, grid[i], grid[i + 1], i % 12)] for i in range(8)],
        "barRanges": [[0, 7], [8, 15], [24, 35], [44, 47]],
        "spans": [[0, 0]] * 4,
    }
    bar_spans_for_sections([sec], bar_grid=grid)
    assert check_section(sec, "A0") == []
    # spans are re-stated from the grid, so each pass ends where its bars do
    assert sec["spans"][2] == pytest.approx([grid[24], grid[36]])
    assert sec["spans"][3] == pytest.approx([grid[44], grid[48]])


def test_a_bar_line_is_never_read_off_a_MID_BAR_chord():
    """The round-3 bug, and the reason this file exists in its current form.

    Louis: *"clairement un decalage quand je suis sur le deuxieme A x2"*, and
    then the diagnosis: *"une fois qu'on a les accords snappes sur le grid, on
    se fie au grid et on defile dessus"*. Sam Smith's verse bar 30 carries a
    single chord on BEAT 2. Timing the bar from that chord's onset — which is
    what taking ``min(chord.t0)`` does — starts the bar half a bar late. A bar
    line comes from the grid; a mid-bar chord is not one.
    """
    grid = _grid(4, 2.0)
    sec = {"id": "A0", "label": "A", "tag": "", "reps": 1,
           # bar 1's only chord sits on beat 2, an ENTIRE half bar after its line
           "bars": [[_chord(0, 0.0, 2.0)], [_chord(1, 3.0, 4.0, 5, beat=2)],
                    [_chord(2, 4.0, 6.0, 7)], [_chord(3, 6.0, 8.0, 9)]],
           "barRanges": [[0, 3]], "spans": [[0, 0]]}
    bar_spans_for_sections([sec], bar_grid=grid)
    assert check_section(sec, "A0") == []
    assert sec["barSpans"][1][0] == pytest.approx([2.0, 4.0]), (
        f"bar 1 must start on its GRID line 2.0, not at its beat-2 chord 3.0: "
        f"{sec['barSpans'][1][0]}")


def test_a_held_bar_gets_real_time_so_it_can_light():
    """Two consecutive bars share one chord: the second is a held ("%") bar. It
    has no chord of its own and needs none — the grid gives it a bar line."""
    grid = _grid(8, 2.0)
    sec = {"id": "A0", "label": "A", "tag": "", "reps": 2,
           "bars": [[_chord(0, 0.0, 4.0)], [], [_chord(2, 4.0, 6.0, 5)],
                    [_chord(3, 6.0, 8.0, 7)]],
           "barRanges": [[0, 3], [4, 7]], "spans": [[0, 0], [0, 0]]}
    bar_spans_for_sections([sec], bar_grid=grid)
    assert check_section(sec, "A0") == []
    assert sec["barSpans"][1][0] == pytest.approx([2.0, 4.0])
    assert all(sp[1] - sp[0] > 0.5 for sp in sec["barSpans"][1])


def test_endings_tile_each_pass_including_the_short_variant():
    """1st/2nd endings: the prefix is shared, each pass plays only its own tail."""
    grid = _grid(12, 2.0)
    prefix = [[_chord(i, grid[i], grid[i + 1], i % 12)] for i in range(3)]
    sec = {"id": "B1", "label": "B", "tag": "", "reps": 3,
           "bars": prefix + [[_chord(3, grid[3], grid[4], 8)]],
           "barRanges": [[0, 3], [4, 7], [8, 11]], "spans": [[0, 0]] * 3,
           "endings": {"tail": 1, "variants": [
               {"label": "B1", "passes": [0, 1], "bars": [[_chord(3, grid[3], grid[4], 8)]]},
               {"label": "B2", "passes": [2], "bars": [[_chord(11, grid[11], grid[12], 3)]]}]}}
    bar_spans_for_sections([sec], bar_grid=grid)
    assert check_section(sec, "B1") == []


def test_sections_can_never_claim_the_same_instant():
    """Both sections are timed by the same monotone grid, so overlap is not
    something to detect and repair — it cannot be constructed."""
    grid = _grid(12, 2.5)
    a = {"id": "A0", "label": "A", "tag": "", "reps": 2,
         "bars": [[_chord(i, grid[i], grid[i + 1])] for i in range(4)],
         "barRanges": [[0, 3], [8, 11]], "spans": [[0, 0], [0, 0]]}
    d = {"id": "D1", "label": "D", "tag": "", "reps": 1,
         "bars": [[_chord(4 + i, grid[4 + i], grid[5 + i])] for i in range(4)],
         "barRanges": [[4, 7]], "spans": [[0, 0]]}
    bar_spans_for_sections([a, d], bar_grid=grid)
    iv = flat_intervals([a, d])
    for (t0, t1, sid, bi), (n0, n1, nsid, nbi) in zip(iv, iv[1:]):
        assert n0 >= t1 - 1e-6, f"{nsid}/rb{nbi} starts inside {sid}/rb{bi}"


def test_no_grid_means_no_time_rather_than_invented_time():
    """A chart with no audio has nothing to follow. Say so, once."""
    sec = {"id": "A0", "label": "A", "tag": "", "reps": 1,
           "bars": [[_chord(i, 0.0, 0.0)] for i in range(4)],
           "barRanges": [[0, 3]], "spans": [[0.0, 0.0]]}
    bar_spans_for_sections([sec], bar_grid=None)
    assert sec["barSpans"] == [[None]] * 4


def test_grid_is_anchored_on_downbeats_and_ignores_mid_bar_chords():
    from harmonia.output.chart_display import _bar_grid
    bar_chords = {0: [_chord(0, 0.0, 2.0)],
                  1: [_chord(1, 3.1, 4.0, beat=2)],       # mid-bar only — not an anchor
                  2: [_chord(2, 4.0, 6.0)],
                  4: [_chord(4, 8.0, 10.0)]}
    g = _bar_grid(bar_chords, 5, 4)
    assert g is not None
    assert g[0] == pytest.approx(0.0)
    assert g[2] == pytest.approx(4.0)
    assert g[4] == pytest.approx(8.0)
    assert g[1] == pytest.approx(2.0), f"bar 1 interpolated on the grid, got {g[1]}"
    assert g == sorted(g)


def test_beat_snap_never_collapses_a_bar():
    """Snapping bar edges to real detected beats must keep every bar non-empty."""
    grid = _grid(8, 2.5)
    sec = {"id": "A0", "label": "A", "tag": "", "reps": 2,
           "bars": [[_chord(i, grid[i], grid[i + 1])] for i in range(8)],
           "barRanges": [[0, 7], [8, 15]], "spans": [[0, 0], [0, 0]]}
    bar_spans_for_sections([sec], bar_grid=_grid(16, 2.5))
    snap_bar_spans_to_beats([sec], [round(0.6 * i, 4) for i in range(400)])
    assert check_section(sec, "A0") == []


# ── the real payloads ────────────────────────────────────────────────────────
_THIS_LOVE = Path("docs/plots/inferred_maroon_5_this_love.html")
_NORAH = Path("docs/plots/inferred_norah_jones_don_t_know_why.html")


@pytest.mark.parametrize("chart", [_THIS_LOVE, _NORAH])
def test_real_chart_playhead_map_is_well_defined(chart):
    if not chart.exists():
        pytest.skip(f"{chart} not baked")
    os.environ.setdefault("HARMONIA_REGRID", "1")
    from harmonia.serving.render import _chart_model_for

    model = _chart_model_for(chart.name, include_gt=False)
    sections = model["sections"]
    assert sections, "no sections"
    assert model.get("barGrid"), "no bar grid — the playhead has nothing to follow"
    grid = model["barGrid"]
    assert grid == sorted(grid), "the bar grid must be monotone"

    fails: list[str] = []
    for sec in sections:
        fails += check_section(sec, f"{model['file']}:{sec['id']}")
    iv = flat_intervals(sections)
    for (t0, t1, sid, bi), (n0, n1, nsid, nbi) in zip(iv, iv[1:]):
        if n0 < t1 - 1e-3:
            fails.append(f"{nsid}/rb{nbi} starts {t1 - n0:.2f}s inside {sid}/rb{bi}")
    assert fails == [], "\n".join(fails[:40])


@pytest.mark.skipif(not _NORAH.exists(), reason="chart not baked")
def test_every_pass_follows_the_grid_over_its_own_song_bars():
    """The round-3 contract, on Louis's song: a pass's rendered bars must be
    exactly the grid times of the song bars that pass covers."""
    os.environ.setdefault("HARMONIA_REGRID", "1")
    from harmonia.serving.render import _chart_model_for

    model = _chart_model_for(_NORAH.name, include_gt=False)
    grid = model["barGrid"]
    bad = []
    for sec in model["sections"]:
        for k, (b0, b1) in enumerate(sec.get("barRanges") or []):
            row = [r for r in sec["barSpans"] if k < len(r) and r[k]]
            if not row:
                continue
            first, last = row[0][k] if False else None, None
        for k, (b0, b1) in enumerate(sec.get("barRanges") or []):
            got0 = sec["barSpans"][0][k] if k < len(sec["barSpans"][0]) else None
            if got0 and abs(got0[0] - grid[b0]) > 1e-3:
                bad.append(f"{sec['id']} pass{k}: starts {got0[0]:.3f}, "
                           f"grid says {grid[b0]:.3f}")
    assert bad == [], "\n".join(bad[:20])
