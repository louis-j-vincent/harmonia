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
def _chord(bar: int, t0: float, t1: float, root: int = 0) -> dict:
    return {"root": root, "q": "", "c": 0.5, "bass": -1, "bar": bar, "beat": 1,
            "t0": t0, "t1": t1}


def test_passes_of_unequal_length_still_tile_their_own_span():
    """A folded section whose passes are NOT the same length as the written one.

    This Love's verse: four occurrences of 20.2 / 20.2 / 27.8 / 7.6 s, written
    once as 8 bars. Rigid translation of pass 0's chord times leaves pass 3
    8.2 s short and runs pass 4 12 s past its own end into the bridge.
    """
    sec = {
        "id": "A0", "label": "A", "tag": "", "reps": 4,
        "bars": [[_chord(i, 1.0 + 2.5 * i, 1.0 + 2.5 * (i + 1), i % 12)] for i in range(8)],
        "barRanges": [[0, 7], [8, 15], [24, 35], [44, 47]],
        "spans": [[1.0, 21.0], [21.0, 41.0], [64.0, 92.0], [114.0, 122.0]],
    }
    bar_spans_for_sections([sec])
    assert check_section(sec, "A0") == []


def test_a_held_bar_gets_real_time_so_it_can_light():
    """Two consecutive bars share one chord: the second is a held ("%") bar.

    The old timing gave it ``[previous chord's t1, next chord's t0]`` = a
    zero-length span, so it never lit and the first bar stayed lit for both.
    """
    sec = {
        "id": "A0", "label": "A", "tag": "", "reps": 2,
        # bar 0 chord runs through bar 1; bar 1 is empty (held)
        "bars": [[_chord(0, 0.0, 4.0)], [], [_chord(2, 4.0, 6.0, 5)], [_chord(3, 6.0, 8.0, 7)]],
        "barRanges": [[0, 3], [4, 7]],
        "spans": [[0.0, 8.0], [8.0, 16.0]],
    }
    bar_spans_for_sections([sec])
    assert check_section(sec, "A0") == []
    held = sec["barSpans"][1]
    assert all(sp[1] - sp[0] > 0.5 for sp in held), f"held bar still dead: {held}"


def test_endings_tile_each_pass_including_the_short_variant():
    """1st/2nd endings: the prefix is shared, each pass plays only its own tail."""
    prefix = [[_chord(i, 0.0 + 2.0 * i, 2.0 * (i + 1), i % 12)] for i in range(3)]
    sec = {
        "id": "B1", "label": "B", "tag": "", "reps": 3,
        "bars": prefix + [[_chord(3, 6.0, 10.0, 8)]],
        "barRanges": [[0, 3], [4, 7], [8, 11]],
        "spans": [[0.0, 10.0], [10.0, 20.0], [20.0, 28.0]],
        "endings": {"tail": 1, "variants": [
            {"label": "B1", "passes": [0, 1], "bars": [[_chord(3, 6.0, 10.0, 8)]]},
            {"label": "B2", "passes": [2], "bars": [[_chord(11, 26.0, 28.0, 3)]]},
        ]},
    }
    bar_spans_for_sections([sec])
    assert check_section(sec, "B1") == []


def test_sections_never_overlap_in_time():
    """The playhead must be inside exactly one section at any instant."""
    a = {"id": "A0", "label": "A", "tag": "", "reps": 2,
         "bars": [[_chord(i, 1.0 + 2.5 * i, 1.0 + 2.5 * (i + 1))] for i in range(4)],
         "barRanges": [[0, 3], [8, 11]], "spans": [[1.0, 11.0], [21.0, 25.0]]}
    d = {"id": "D1", "label": "D", "tag": "", "reps": 1,
         "bars": [[_chord(4, 11.0 + 2.5 * i, 11.0 + 2.5 * (i + 1))] for i in range(4)],
         "barRanges": [[4, 7]], "spans": [[11.0, 21.0]]}
    bar_spans_for_sections([a, d])
    iv = flat_intervals([a, d])
    for (t0, t1, sid, bi), (n0, n1, nsid, nbi) in zip(iv, iv[1:]):
        assert n0 >= t1 - 1e-3, (f"{nsid}/rb{nbi} starts {t0 - n0:.2f}s inside "
                                 f"{sid}/rb{bi} (t={n0:.2f})")


def test_beat_snap_never_collapses_a_bar():
    """Snapping bar edges to real beats must keep every bar non-empty and ordered.

    A heavily compressed pass (This Love's 4-bar verse written as 8 bars) has
    rendered bars SHORTER than one beat, so a naive nearest-beat snap would pull
    two edges onto the same beat and kill the bar in between.
    """
    sec = {"id": "A0", "label": "A", "tag": "", "reps": 2,
           "bars": [[_chord(i, 1.0 + 2.5 * i, 1.0 + 2.5 * (i + 1))] for i in range(8)],
           "barRanges": [[0, 7], [44, 47]], "spans": [[1.0, 21.0], [114.0, 121.6]]}
    bar_spans_for_sections([sec])
    beats = [round(0.6 * i, 4) for i in range(400)]
    snap_bar_spans_to_beats([sec], beats)
    assert check_section(sec, "A0") == []


# ── the real payload ─────────────────────────────────────────────────────────
_THIS_LOVE = Path("docs/plots/inferred_maroon_5_this_love.html")


@pytest.mark.skipif(not _THIS_LOVE.exists(), reason="This Love chart not baked")
def test_this_love_playhead_map_is_well_defined():
    os.environ.setdefault("HARMONIA_REGRID", "1")
    from harmonia.serving.render import _chart_model_for

    model = _chart_model_for(_THIS_LOVE.name, include_gt=False)
    sections = model["sections"]
    assert sections, "no sections"

    fails: list[str] = []
    for sec in sections:
        fails += check_section(sec, f"{model['file']}:{sec['id']}")
    iv = flat_intervals(sections)
    for (t0, t1, sid, bi), (n0, n1, nsid, nbi) in zip(iv, iv[1:]):
        if n0 < t1 - 1e-3:
            fails.append(f"{nsid}/rb{nbi} starts {t1 - n0:.2f}s inside {sid}/rb{bi} "
                         f"(t={n0:.2f})")
    assert fails == [], "\n".join(fails[:40])


# ── what the client used to do (kept so the bug can't come back quietly) ─────
def legacy_bar_spans(sec: dict) -> list[list[list[float]]]:
    """The pre-2026-07-30 client reconstruction, for comparison only.

    ``app_shell.html`` translated the written phrase's chord times rigidly onto
    each pass's start and took a bar's extent from its chords' sustain:

        allPass = c => s.spans.map(sp => [c.t0 + (sp[0]-base), c.t1 + (sp[0]-base)])
        bar.tspans[r] = [min over chords of span[r][0], max of span[r][1]]
        held bar      = [previous bar's t1, next chorded bar's t0]
    """
    spans = sec["spans"]
    base = spans[0][0]
    endings = sec.get("endings") or {}
    tail = endings.get("tail") or 0
    all_bars = sec["bars"]
    out: list[list[list[float]]] = []
    blocks = [(all_bars[:len(all_bars) - tail] if tail else all_bars, list(range(len(spans))), base)]
    for v in (endings.get("variants") or []) if tail else []:
        blocks.append((v["bars"], v["passes"], spans[v["passes"][0]][0]))
    for bars, passes, b0 in blocks:
        for bar in bars:
            if not bar:
                out.append([[0.0, 0.0]])          # held: collapses to nothing
                continue
            row = []
            for p in passes:
                off = spans[p][0] - b0
                row.append([min(c["t0"] for c in bar) + off,
                            max(c["t1"] for c in bar) + off])
            out.append(row)
    return out


def test_the_old_reconstruction_really_was_broken():
    """Red-first evidence. The same two fixtures, timed the way the client used
    to time them: the long pass runs out early, the short pass overruns its own
    section, and the held bar is zero-length."""
    sec = {
        "id": "A0", "label": "A", "tag": "", "reps": 4,
        "bars": [[_chord(i, 1.0 + 2.5 * i, 1.0 + 2.5 * (i + 1), i % 12)] for i in range(8)],
        "barRanges": [[0, 7], [8, 15], [24, 35], [44, 47]],
        "spans": [[1.0, 21.0], [21.0, 41.0], [64.0, 92.0], [114.0, 122.0]],
    }
    old = legacy_bar_spans(sec)
    assert abs(old[-1][2][1] - 92.0) > 5.0, "expected the 28s pass to run out early"
    assert old[-1][3][1] > 122.0 + 5.0, "expected the 8s pass to overrun its section"

    held_sec = {
        "id": "A0", "label": "A", "tag": "", "reps": 1,
        "bars": [[_chord(0, 0.0, 4.0)], [], [_chord(2, 4.0, 6.0, 5)]],
        "barRanges": [[0, 2]], "spans": [[0.0, 6.0]],
    }
    assert legacy_bar_spans(held_sec)[1] == [[0.0, 0.0]], "expected a dead held bar"

    # and the same inputs through the map that replaced it (one chart each —
    # they are unrelated fixtures, and the map clips a chart's spans into play
    # order, which is meaningless across two different songs)
    bar_spans_for_sections([sec])
    bar_spans_for_sections([held_sec])
    assert check_section(sec, "A0") == []
    assert all(sp[1] - sp[0] > 0.5 for sp in held_sec["barSpans"][1])


# ── round 2 (Louis, 2026-07-30, after the first fix shipped) ─────────────────
def form_runs(sections: list[dict]) -> list[dict]:
    """Port of ``app_shell.html``'s ``formRuns()``.

    One chip per section OCCURRENCE in BAR order, consecutive repeats of the
    same letter folded to ×N. Bar order is what the chart reads like, so that
    is the right order to DRAW them in — but it is not necessarily time order,
    which is what the highlight has to be looked up in.
    """
    occ = []
    for si, s in enumerate(sections):
        for k, r in enumerate(s.get("barRanges") or []):
            sp = (s.get("spans") or [])[k] if k < len(s.get("spans") or []) else None
            occ.append({"b0": r[0], "b1": r[1], "label": s["label"],
                        "t0": sp[0] if sp else None, "t1": sp[1] if sp else None})
    occ.sort(key=lambda o: o["b0"])
    runs: list[dict] = []
    for o in occ:
        L = runs[-1] if runs else None
        if L and L["label"] == o["label"] and L["b1"] + 1 == o["b0"]:
            L["b1"] = o["b1"]
            L["n"] += 1
            if o["t1"] is not None:
                L["t1"] = o["t1"]
        else:
            runs.append({**o, "n": 1})
    return runs


def chip_at(runs: list[dict], t: float) -> int:
    """What the form strip SHOULD light at time t: the chip containing t."""
    live = [(i, r) for i, r in enumerate(runs)
            if r["t0"] is not None and r["t1"] is not None and r["t1"] > r["t0"]]
    for i, r in live:
        if r["t0"] <= t < r["t1"]:
            return i
    prev = [i for i, r in live if r["t0"] <= t]
    return prev[-1] if prev else -1


def test_repeated_chords_share_the_bar_evenly():
    """Louis: "when multiple chords repeat say 2 bars of F7, only the last one
    is highlighted."

    Consecutive rendered bars can carry the SAME chord with onsets a few
    milliseconds apart (the display fold re-emits one decoded chord per grid
    bar). Believing those onsets literally gives every bar but the last a
    ~10 ms sliver and hands the last one the whole remaining stretch — and a
    bar shorter than one `timeupdate` tick (~250 ms) is never observed at all.
    """
    # four bars of F7 whose onsets differ by 11 ms, then a real chord
    bars = [[_chord(i, 10.0 + 0.011 * i, 18.0, 5)] for i in range(4)]
    bars.append([_chord(4, 18.0, 20.0, 7)])
    sec = {"id": "A0", "label": "A", "tag": "", "reps": 1, "bars": bars,
           "barRanges": [[0, 4]], "spans": [[10.0, 20.0]]}
    bar_spans_for_sections([sec])
    assert check_section(sec, "A0") == []
    widths = [sp[1] - sp[0] for row in sec["barSpans"] for sp in row]
    nominal = (20.0 - 10.0) / 5
    assert min(widths) > nominal * 0.3, (
        f"a repeated-chord bar collapsed to {min(widths):.3f}s "
        f"(nominal {nominal:.2f}s): {[round(w, 3) for w in widths]}")


def test_form_chip_lookup_survives_chips_out_of_time_order():
    """Louis: "no highlighting of the timeframe that explains the chord
    sequences (Ax3 B Ax2 ..) on top."

    Chips are drawn in BAR order. A section whose span is broken upstream (an
    iReal import with no times gives ``[0, 0]``) can therefore sit LAST in the
    strip while claiming t0 = 0, and "the last chip whose t0 <= t" then matches
    it for every t — the highlight sticks there and never tracks the music.
    """
    sections = [
        {"id": "A0", "label": "A", "barRanges": [[0, 7], [16, 23]],
         "spans": [[0.0, 20.0], [40.0, 60.0]]},
        {"id": "B1", "label": "B", "barRanges": [[8, 15]], "spans": [[20.0, 40.0]]},
        {"id": "C2", "label": "C", "barRanges": [[24, 31]], "spans": [[0.0, 0.0]]},
    ]
    runs = form_runs(sections)
    assert [r["label"] for r in runs] == ["A", "B", "A", "C"]
    # the old rule: last chip with t0 <= t — the dead C chip wins everywhere
    def legacy(t):
        cur = -1
        for i, r in enumerate(runs):
            if r["t0"] is not None and t >= r["t0"]:
                cur = i
        return cur
    assert legacy(10.0) == 3, "expected the old rule to stick on the dead chip"
    # the rule that replaced it
    assert chip_at(runs, 10.0) == 0
    assert chip_at(runs, 30.0) == 1
    assert chip_at(runs, 50.0) == 2


@pytest.mark.skipif(not _THIS_LOVE.exists(), reason="This Love chart not baked")
def test_no_rendered_bar_is_too_short_to_see():
    """A bar the playhead can never be observed on is a bar that never lights.
    `timeupdate` fires ~4x/second, so anything under ~250 ms is invisible."""
    os.environ.setdefault("HARMONIA_REGRID", "1")
    from harmonia.serving.render import _chart_model_for

    model = _chart_model_for(_THIS_LOVE.name, include_gt=False)
    bad = []
    for sec in model["sections"]:
        for r, row in enumerate(sec.get("barSpans") or []):
            for k, sp in enumerate(row):
                if sp and sp[1] - sp[0] < 0.25:
                    bad.append(f"{sec['id']} rb{r} pass{k}: {sp[1] - sp[0]:.3f}s")
    assert bad == [], "\n".join(bad[:20])
