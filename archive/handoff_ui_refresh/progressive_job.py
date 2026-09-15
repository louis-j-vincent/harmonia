"""progressive_job.py — the job payload the two-phase loading screen needs.

DROP-IN for harmonia_min/server.py. Read harmonia_min/pipeline.py:analyze()
before this file; the design follows what that function can actually publish.

⚠ CORRECTION, 2026-08-07. An earlier draft of this file assumed bars stream in
one at a time. They do not. `_musx.redecode()` is a single blocking call that
returns EVERY segment at once, and the bar layout that follows it is pure
integer arithmetic over those segments — microseconds. There is no fold-by-fold
progress in harmonia_min (the `musx_fold` / `n_folds` fields the analysing
screen still reads belong to the OLD pipeline and are never set here). Anything
that animates bars arriving one by one would be a fake progress bar.

The real shape of a run
───────────────────────
Two waits, one hand-off between them:

    PHASE 1  beats.track            seconds
             musx.frame_posteriors  MINUTES on a fresh song, cache-hit on a
                                    library one — this is the whole wait
             musx.redecode          seconds
             bar layout             instant
        └─ at the end of this phase the RAW CHART exists, complete: every bar,
           every chord, one section, no letters. It is readable and playable.

    PHASE 2  detect_sections + fold_letter_groups + minimal_fold
             harmonic_key.analyze_harmony
        └─ letters, repeats, endings and colours land on the SAME grid.

So the loading screen shows a genuine wait, then the whole chart at once, then
the letters folding in over it. Bars never trickle.

Note `HARMONIA_RAW_CHART=1` already skips phase 2 entirely (pipeline.py, Louis
2026-08-07). This payload makes that same raw chart the normal intermediate
state instead of an env-flag mode.

Job fields
──────────
Additive; nothing existing changes shape.

    phase : "listening" | "decoding" | "raw" | "sections" | "done"
        Drives the whole screen. "raw" is the moment the chart appears.

    raw_model : dict | None
        A ChartModel — the exact shape /api/chart-model/<file> returns — built
        from the phase-1 result with ONE section, reps=1, every bar written
        out. None until phase "raw". The shell renders it with the same
        loadModel()/buildIReal() path it uses for the final chart, which is
        what guarantees nothing shifts when the real model replaces it.

    n_bars, n_chords : int | None
        For the header line. Available with raw_model, not before.

    sections_found : int | None
        Set when phase becomes "done"; the footer says "Found N sections".
"""

from __future__ import annotations

from typing import Any


def raw_chart_model(bars: list[list[dict]], grid: list[float], bpb: int,
                    *, file_key: str, title: str, key: dict, key_name: str,
                    audio_url: str, beat_times: list[float]) -> dict[str, Any]:
    """The phase-1 chart: one section, unfolded, every bar written out.

    This is deliberately the same construction pipeline.py already does under
    HARMONIA_RAW_CHART=1 — keep the two in sync, or extract that block and call
    it from both. The invariant that matters: bar indices, `beat`, and every
    (t0, t1) here are the ones the final model will carry. The shell keeps its
    scroll position and playhead across the swap only because of that.
    """
    n_bars = len(bars)
    return {
        "file": file_key, "title": title, "video_id": "",
        "audio_url": audio_url,
        "key": key, "keyName": key_name,
        "bpb": bpb, "nBars": n_bars,
        "barGrid": grid, "beatTimes": beat_times,
        "form": None, "fold": {"raw_chart": True},
        "sections": [{
            "id": "S0", "label": "A", "tag": "", "reps": 1,
            "spans": [[grid[0], grid[n_bars]]],
            "barRanges": [[0, n_bars - 1]],
            "bars": bars,
            "barSpans": [[[grid[b], grid[b + 1]]] for b in range(n_bars)],
        }],
        "meta": {"raw": True},
    }


def wire_into_analyze(report) -> None:
    """Where each phase is published, in pipeline.py:analyze() order.

    `report` is analyze()'s existing progress callback — no new plumbing.

        report(0, phase="listening")                     # before beats.track
        report(1, phase="decoding", tempo_bpm=bd["bpm"]) # after beats.track

        # after the bar-layout block (the `prev`/carry loop), BEFORE
        # detect_sections — this is the hand-off:
        report(3, phase="raw",
               raw_model=raw_chart_model(bars, grid, bpb, ...),
               n_bars=n_bars,
               n_chords=sum(1 for bar in bars for c in bar
                            if not c["nc"] and not c.get("carry")))

        report(3, phase="sections")                      # entering detect_sections
        report(4, phase="done", sections_found=len(sections),
               n_sections=len(sections))

    Under HARMONIA_RAW_CHART=1 the "sections" phase never fires and the job
    goes straight from "raw" to "done" with sections_found=1. The screen
    handles that: the footer button simply appears sooner.
    """


# ── acceptance ─────────────────────────────────────────────────────────────
# 1. `phase` reaches "raw" strictly before detect_sections is called, and
#    raw_model is non-None from that point on.
# 2. raw_model has exactly one section, reps=1, and len(bars) == nBars.
# 3. Bar indices, `beat`, and every (t0, t1) in raw_model are IDENTICAL to the
#    final model's for the same song. Diff them in a test on This Love — if
#    they differ, the grid visibly jumps when the letters land, which is the
#    one thing this screen exists to avoid.
# 4. No field here implies per-bar streaming. If you find yourself appending to
#    a `bars` list across polls, stop: that is the earlier, wrong design.
