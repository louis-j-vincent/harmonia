"""Raw charts IN THE APP (Louis, 2026-08-07: « le chart, version harmonia »).

Runs analyze() under HARMONIA_RAW_CHART=1 (single section, no section
detection, no repli — the first-pass decode written bar for bar, granularity
unrestricted), then splits the single section at the bar the VOICE RULE
picks (demucs vocals → first moving-pitch onset → bar, rounded up on the
2-bar grid — scripts/melody_ssm doctrine) into an "Intro" block and the
body, and writes the ChartModel into harmonia_min/state/charts/raw_<stem>.json
so it appears in the app library on :7772.

Run: .venv/bin/python scratchpad/raw_app_charts.py [stem ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scratchpad"))

os.environ["HARMONIA_RAW_CHART"] = "1"

SONGS = [
    ("let_it_be_remastered_2009", "Let It Be · brut"),
    ("bein_green", "Bein Green · brut"),
    ("ray_charles_georgia_on_my_mind_official_video", "Georgia · brut"),
]


def build(stem: str, title: str):
    from harmonia_min import pipeline as _pl
    from raw_chart_pages import voice_form_start
    model = _pl.analyze(REPO / f"docs/audio/{stem}.m4a", title=title,
                        file_key=f"raw_{stem}",
                        audio_url=f"/audio/{stem}.m4a")
    grid = model["barGrid"]
    fs, onset = voice_form_start(stem, grid)
    if fs > 0:
        s0 = model["sections"][0]
        n_last = model["nBars"] - 1
        model["sections"] = [
            {"id": "S0", "label": "Intro", "tag": "intro", "reps": 1,
             "spans": [[grid[0], grid[fs]]],
             "barRanges": [[0, fs - 1]],
             "bars": s0["bars"][:fs], "barSpans": s0["barSpans"][:fs]},
            {"id": "S1", "label": "A", "tag": "", "reps": 1,
             "spans": [[grid[fs], grid[-1]]],
             "barRanges": [[fs, n_last]],
             "bars": s0["bars"][fs:], "barSpans": s0["barSpans"][fs:]},
        ]
    out = REPO / "harmonia_min" / "state" / "charts" / f"raw_{stem}.json"
    out.write_text(json.dumps(model), encoding="utf-8")
    print(f"wrote {out.name}  formStart={fs} (sing "
          f"{'-' if onset is None else round(onset, 1)}s)  "
          f"nBars={model['nBars']}")


if __name__ == "__main__":
    only = sys.argv[1:]
    for stem, title in SONGS:
        if only and stem not in only:
            continue
        try:
            build(stem, title)
        except Exception as e:
            print(f"{stem}: REFUSÉ — {type(e).__name__}: {e}")
