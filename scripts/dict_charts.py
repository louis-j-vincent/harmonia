"""Real charts written from the dictionary sections, openable in the app.

« Génère-moi les charts pour que je puisse les lire. »

Runs the real `pipeline.analyze()` twice per song: once to capture the bar grid,
once with `detect_sections` patched to return the sections the dictionary method
produced — so the folding, the letters, the bar spans and the playhead map are
all the pipeline's own work on the new boundaries. Saved as `min_<stem>__dict`
so the original stays untouched beside it in the library.

    python scripts/dict_charts.py
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
from harmonia_min import sections as hs                        # noqa: E402
import harmonic_method as HM                                   # noqa: E402
from dictionary_harmonic import build                          # noqa: E402
from sections_from_dict import sections_from                   # noqa: E402

CHARTS = HERE / "harmonia_min/state/charts"
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why")]


def build_chart(stem, title):
    from harmonia_min import pipeline as _pl
    audio = HERE / f"docs/audio/{stem}.m4a"
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None, **_kw):
        out = real(grid, arr, times, bars, **_kw)
        cap.update(grid=grid, segs=copy.deepcopy(out))
        return out

    hs.detect_sections = spy
    try:
        base = _pl.analyze(audio, title=title, file_key=f"min_{stem}",
                           audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = real

    grid = cap["grid"]
    n = len(grid) - 1
    S = HM.ssm(audio, grid)
    entries, _ = build(S, n)
    secs = sections_from(S, n, entries)
    new = [{"b0": s["b0"], "b1": s["b1"], "label": s["letter"]} for s in secs]

    hs.detect_sections = lambda *a, **k: copy.deepcopy(new)
    try:
        alt = _pl.analyze(audio, title=f"{title} — DICO",
                          file_key=f"min_{stem}__dict",
                          audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = real

    json.dump(alt, open(CHARTS / f"min_{stem}__dict.json", "w"))
    return base, alt


def describe(tag, m):
    tot = sum(len(s["bars"]) for s in m["sections"])
    return (f"{tag:<8} {tot:>3} mesures écrites | " +
            "  ".join(f"{s['label']}×{s['reps']}({len(s['bars'])})"
                      for s in m["sections"]))


def main():
    for stem, title in SONGS:
        base, alt = build_chart(stem, title)
        print(f"\n=== {title}")
        print("   " + describe("ACTUEL", base))
        print("   " + describe("DICO", alt))
        print(f"   -> /?open=min_{stem}__dict")


if __name__ == "__main__":
    main()
