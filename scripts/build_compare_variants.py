#!/usr/bin/env python3
"""Precompute the A/B/C chord variants for the /compare ear-test page.

Three recognisers on the SAME audio timeline, so Louis can swipe between them
and judge by ear (his call, not a metric on a possibly-broken dataset):

    A  pipe actuel   -- infer_chords_v1, shipped defaults (musx_redecode seg,
                        function_family ON, Occam, synthetic beat lattice)
    B  musx brut     -- musx_bass.musx_labels: the vendored frame-level Viterbi
                        (use_beats=False, 23 ms frames), NO Harmonia head, NO
                        beat grid -- the "persistence OFF" arm Louis liked on
                        Georgia (free of the synthetic clock)
    C  beat grid snap-- infer_chords_v1 + HARMONIA_REAL_BEAT_GRID=snap: decode
                        as A, then re-lay the final onsets onto the DETECTED
                        beats (escapes the synthetic lattice, keeps the labels)

Writes docs/research_sessions/compare_variants.json; the /compare route reads it.
Audio is served live by /compare/audio/<sid> from the path recorded here.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import logging; logging.disable(logging.WARNING)

from harmonia.eval.accuracy_score import _decode_to_wav, load_frozen_gt, SHIPPED_CONFIG
from harmonia.models.chord_pipeline_v1 import infer_chords_v1
from harmonia.models import musx_bass

OUT = REPO / "docs" / "research_sessions" / "compare_variants.json"
WORK = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
            "Code-harmonia/836b2458-dce2-4a3a-9e3a-d95036c79686/scratchpad/cmpwav")
WORK.mkdir(parents=True, exist_ok=True)

TITLES = {
    "georgia_on_my_mind": "Georgia On My Mind (Ray Charles)",
    "bein_green": "Bein' Green",
    "stand_by_me": "Stand By Me (Ben E. King)",
    "blue_bossa": "Blue Bossa",
    "close_to_you": "Close To You (Carpenters)",
}
SONGS = sys.argv[1:] or ["georgia_on_my_mind", "bein_green", "stand_by_me", "blue_bossa"]


def chords_of_chart(ch):
    return [{"label": c["label"], "start_s": round(float(c["start_s"]), 3),
             "end_s": round(float(c["end_s"]), 3)} for c in ch.chords]


def infer(wav, grid):
    os.environ["HARMONIA_REAL_BEAT_GRID"] = grid
    os.environ["HARMONIA_FUNCTION_FAMILY"] = "1"
    cfg = {k: v for k, v in SHIPPED_CONFIG.items() if k != "function_family"}
    return infer_chords_v1(wav, cache_dir=REPO / "data" / "cache", **cfg)


def musx_raw(wav, duration):
    labs = musx_bass.musx_labels(wav)            # [(t0,t1,label)] frame-level
    return [{"label": l, "start_s": round(float(t0), 3), "end_s": round(float(t1), 3)}
            for t0, t1, l in labs if t1 > t0]


out = []
for sid in SONGS:
    t0 = time.time()
    gt = load_frozen_gt(REPO / "golden" / "brick0" / f"{sid}.gt.json")
    audio = Path(gt.resolved_audio_path)
    wav = _decode_to_wav(audio, WORK)
    A = infer(wav, "off")
    C = infer(wav, "snap")
    dur = float(A.duration_s)
    entry = {
        "sid": sid, "title": TITLES.get(sid, sid.replace("_", " ")),
        "duration_s": round(dur, 2),
        "audio_path": str(audio),           # served live by /compare/audio/<sid>
        "variants": {
            "A": {"name": "pipe actuel", "chords": chords_of_chart(A)},
            "B": {"name": "musx brut", "chords": musx_raw(wav, dur)},
            "C": {"name": "beat grid snap", "chords": chords_of_chart(C)},
        },
    }
    out.append(entry)
    na, nb, nc = (len(entry["variants"][k]["chords"]) for k in "ABC")
    print(f"[{sid}] A={na} B={nb} C={nc} chords  dur={dur:.0f}s  ({time.time()-t0:.0f}s)",
          flush=True)

OUT.write_text(json.dumps(out, indent=1))
print(f"\nwrote {OUT.relative_to(REPO)}  ({len(out)} songs)")
