#!/usr/bin/env python3
"""vocab_fold_demo.py — HARMONIA_VOCAB_FOLD on vs off, on one song, side by side.

The fold averages the per-beat observations across every occurrence of a learned
section before anything decodes, so the noise on the evidence falls as ~1/sqrt(N).
This runs the LIVE decoder (`chord_pipeline_v1.infer_chords_v1`) twice on the same
audio — flag off, then flag on — and reports what actually changed, bar by bar,
against Louis's hand lead sheet where one exists.

Run: .venv/bin/python scratchpad/vocab_fold_demo.py [audio-stem]
     (default: maroon_5_this_love)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CACHE = REPO / "data" / "cache"
_PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_Q = {"maj": "", "min": "-", "dom7": "7", "maj7": "^7", "min7": "-7",
      "dim": "o", "hdim7": "h7", "aug": "+", "sus": "sus", "N": "N"}


def run(wav: Path, fold: bool):
    """One decode of the live pipeline with the flag set as asked."""
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    prev = os.environ.get("HARMONIA_VOCAB_FOLD")
    os.environ["HARMONIA_VOCAB_FOLD"] = "1" if fold else "0"
    t0 = time.time()
    try:
        chart = infer_chords_v1(wav, cache_dir=CACHE)
    finally:
        if prev is None:
            os.environ.pop("HARMONIA_VOCAB_FOLD", None)
        else:
            os.environ["HARMONIA_VOCAB_FOLD"] = prev
    return chart, time.time() - t0


def label(c: dict) -> str:
    q = c.get("q5") or c.get("q") or ""
    return _PC[int(c["root"]) % 12] + _Q.get(q, q if isinstance(q, str) else "")


def chords_of(chart) -> list[dict]:
    ch = getattr(chart, "chords", None)
    if ch is None and isinstance(chart, dict):
        ch = chart.get("chords")
    return list(ch or [])


def at_time(chords: list[dict], t: float) -> dict | None:
    """The chord sounding at t."""
    best = None
    for c in chords:
        s = float(c.get("start_s", c.get("t0", 0.0)))
        e = float(c.get("end_s", c.get("t1", s)))
        if s <= t < e:
            return c
        if s <= t:
            best = c
    return best


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "maroon_5_this_love"
    audio = REPO / "docs" / "audio" / f"{stem}.m4a"
    if not audio.exists():
        raise SystemExit(f"no audio at {audio}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(audio), "-ar", "44100", str(wav)],
                       check=True, capture_output=True)
        print(f"decoding {stem} twice (flag OFF, then ON)...")
        off, t_off = run(wav, False)
        print(f"  OFF done in {t_off:.0f}s")
        on, t_on = run(wav, True)
        print(f"  ON  done in {t_on:.0f}s")

    c_off, c_on = chords_of(off), chords_of(on)
    print(f"\nchords: OFF {len(c_off)}   ON {len(c_on)}")

    # Bar grid from the rigid-grid recovery, so the comparison is per BAR and
    # lines up with the chart Louis reads.
    from harmonia.models.rigid_grid import rigid_grid_for
    payload = [{"root": int(c["root"]), "t0": float(c.get("start_s", 0.0)),
                "t1": float(c.get("end_s", 0.0)),
                "lv": {"exact": {"q": _Q.get(c.get("q5", ""), "")}}}
               for c in c_off]
    grid = rigid_grid_for(payload, tonic_pc=0)
    if grid is None:
        print("rigid grid deferred — falling back to a flat 2.524 s bar")
        grid = [0.732 + 2.524 * i for i in range(81)]
    bar_t = list(grid)

    diffs = []
    rows = []
    for b in range(len(bar_t) - 1):
        t = bar_t[b] + 0.05
        a, z = at_time(c_off, t), at_time(c_on, t)
        la = label(a) if a else "·"
        lz = label(z) if z else "·"
        rows.append((b, la, lz))
        if la != lz:
            diffs.append((b, la, lz))

    print(f"\nbars where the two decodes DISAGREE: {len(diffs)} / {len(rows)}")
    print("\n bar |  OFF   ->  ON")
    for b, la, lz in diffs:
        print(f" {b:3d} | {la:>6s}  ->  {lz}")

    # The verse downbeat is Louis's actual question: G or G7?
    print("\n--- the G-vs-G7 question (verse downbeats; his lead sheet says G) ---")
    VERSE = [0, 4, 8, 12, 24, 28, 32, 44]
    for b in VERSE:
        if b >= len(rows):
            continue
        _, la, lz = rows[b]
        mark = "" if la == lz else "   <- changed"
        print(f"  bar {b:2d}:  OFF {la:>6s}   ON {lz:>6s}{mark}")

    out = REPO / "scratchpad" / f"vocab_fold_demo_{stem}.json"
    out.write_text(json.dumps(
        {"stem": stem, "n_off": len(c_off), "n_on": len(c_on),
         "bars": [{"bar": b, "off": a, "on": z} for b, a, z in rows],
         "diffs": [{"bar": b, "off": a, "on": z} for b, a, z in diffs]}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
