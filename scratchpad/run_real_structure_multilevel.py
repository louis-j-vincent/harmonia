"""run_real_structure_multilevel.py — Task 2 deployment: regenerate the
real-audio structure checkpoint with THREE levels (phrase/section/form)
instead of Call 1's single section-only level, using the new proba-root
VARIABLE-SPAN encoder (keynorm_proba_varspan.pt) via hierarchy_real.py.
Still QUALITATIVE ONLY (no GT section labels exist for real audio) — no
V-measure number is fabricated here, same discipline as Call 1.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_root_proba import per_bar_root_proba
from adaptive_proba import load_proba_encoder
from hierarchy_real import predict_multilevel

REPO = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO / "docs" / "audio"
OUT = Path(__file__).resolve().parent / "real_structure_multilevel.json"

SONGS = [
    ("autumn_leaves", AUDIO_DIR / "autumn_leaves.m4a"),
    ("abba_chiquitita", AUDIO_DIR / "abba_chiquitita_official_lyric_video.m4a"),
    ("aretha_chain_of_fools", AUDIO_DIR / "aretha_franklin_chain_of_fools_official_lyric_video.m4a"),
]


def runs_from_labels(labels, bar_times):
    runs = []
    cur = labels[0]; start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != cur:
            runs.append({"label": cur, "bar_start": start, "bar_end": i,
                        "t_start": round(float(bar_times[start]), 2),
                        "t_end": round(float(bar_times[min(i, len(bar_times) - 1)]), 2)})
            if i < len(labels):
                cur = labels[i]; start = i
    return runs


def main():
    model, ck = load_proba_encoder("scratchpad/keynorm_proba_varspan.pt")
    print("loaded keynorm_proba_varspan.pt")
    results = {}
    for name, path in SONGS:
        if not path.exists():
            print("MISSING audio for %s (%s) — skipping" % (name, path))
            continue
        print("\n=== %s ===" % name)
        bar_proba, bar_times, tempo = per_bar_root_proba(path)
        levels = predict_multilevel(bar_proba, model)
        song_out = {"tempo_bpm": tempo, "n_bars": len(bar_proba),
                    "est_tonic_pc": levels["est_tonic_pc"],
                    "bar_times": [round(float(t), 3) for t in bar_times],
                    "levels": {}}
        for lvl in ("phrase", "section", "form"):
            labs = levels[lvl]
            runs = runs_from_labels(labs, bar_times)
            n_distinct = len(set(labs))
            print("  [%s] %d runs, %d distinct labels" % (lvl, len(runs), n_distinct))
            song_out["levels"][lvl] = {"runs": runs, "n_distinct": n_distinct}
        results[name] = song_out
    OUT.write_text(json.dumps(results, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
