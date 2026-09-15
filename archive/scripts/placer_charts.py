"""Build the FINAL CHARTS with the placer's section starts, for the real app.

Louis: « montre-moi direct les grilles finales. » The audition page compares two
audio excerpts; this compares two GRIDS, rendered by the app's own renderer, so
what he judges is the actual product and not a diagram of it.

Method: run the real `pipeline.analyze()` twice per song. The first pass spies on
`detect_sections` to capture the true grid/bars/segments; the placer then shifts
the starts; the second pass patches `detect_sections` to RETURN those shifted
segments, so folding, letters, bar spans and everything downstream are the
pipeline's own work on the new boundaries. The result is saved as a separate
chart (`min_<stem>__placer`) so the original stays untouched and both sit side
by side in the library.

    python scripts/placer_charts.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from harmonia_min import sections as hs                    # noqa: E402
from repeat_placer_prototype import bar_sym, place, SONGS   # noqa: E402

CHARTS = HERE / "harmonia_min/state/charts"


def build(stem: str, title: str):
    from harmonia_min import pipeline as _pl
    audio = HERE / f"docs/audio/{stem}.m4a"
    real = hs.detect_sections
    cap: dict = {}

    def spy(grid, arr, times, bars=None, **_kw):
        out = real(grid, arr, times, bars, **_kw)
        cap.update(grid=grid, bars=copy.deepcopy(bars), segs=copy.deepcopy(out))
        return out

    hs.detect_sections = spy
    try:
        base = _pl.analyze(audio, title=title, file_key=f"min_{stem}",
                           audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = real

    grid, bars, segs = cap["grid"], cap["bars"], cap["segs"]
    n_bars = len(grid) - 1
    S = [bar_sym(bars, b, n_bars) for b in range(n_bars)]
    prop, detail, dropped = place(segs, S, n_bars)
    if not prop:
        print(f"— {stem}: placer moves nothing, no alternative chart written")
        return base, None, prop

    # Apply the shifts, then re-close the ranges so the sections still tile the
    # song end to end (a start moving left/right must take the previous
    # section's end with it, or the chart loses or duplicates bars).
    new = copy.deepcopy(segs)
    for i, d in prop.items():
        new[i]["b0"] += d
    # The first section is pinned at bar 0 because sections must cover the
    # song — but if the placer says this letter's pattern really starts later,
    # forcing bar 0 to carry the letter TRUNCATES it: Don't Know Why wrote an
    # A of 4 bars where it used to write 8, because the fold takes its block
    # from the first occurrence. Those leading bars are a pickup/intro, so give
    # them their own section instead of a mutilated A.
    new[0]["b0"] = 0
    for i in range(len(new) - 1):
        new[i]["b1"] = new[i + 1]["b0"] - 1
    new[-1]["b1"] = n_bars - 1
    new = [s for s in new if s["b1"] >= s["b0"]]
    # `minimal_fold` writes a letter's block from its FIRST occurrence, so a
    # truncated first occurrence poisons the whole letter. That is what happens
    # here: occurrence 0 is pinned at bar 0 while occurrence 1 moves left onto
    # bar 4, leaving A[0..3] — and Don't Know Why then wrote a 4-bar A where it
    # used to write 8. Musically those leading bars are an intro, not a
    # mutilated A: relabel them so the letter is written from a full occurrence.
    if len(new) > 1 and new[0]["label"] == new[1]["label"] \
            and (new[0]["b1"] - new[0]["b0"]) < (new[1]["b1"] - new[1]["b0"]):
        new[0] = {**new[0], "label": "Intro"}

    hs.detect_sections = lambda *a, **k: copy.deepcopy(new)
    try:
        alt = _pl.analyze(audio, title=f"{title} — PLACEUR",
                          file_key=f"min_{stem}__placer",
                          audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = real

    json.dump(alt, open(CHARTS / f"min_{stem}__placer.json", "w"))
    return base, alt, prop


def describe(tag, m):
    tot = sum(len(s["bars"]) for s in m["sections"])
    desc = "  ".join(f"{s['label']}×{s['reps']}({len(s['bars'])} mes.)"
                     for s in m["sections"])
    print(f"    {tag:<10} {tot:>3} mesures écrites | {desc}")


def main():
    for stem, title in SONGS:
        pretty = title.split("—")[-1].strip()
        print(f"\n=== {title}")
        base, alt, prop = build(stem, pretty)
        describe("ACTUEL", base)
        if alt:
            describe("PLACEUR", alt)
            print(f"    départs déplacés : {prop}")
            print(f"    -> /?open=min_{stem}__placer")


if __name__ == "__main__":
    main()
