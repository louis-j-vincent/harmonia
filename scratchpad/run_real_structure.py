"""run_real_structure.py — Stage B3b: real-audio structure segmentation,
end to end, using the Stage B2 proba-input learned encoder
(scratchpad/keynorm_proba_rootonly_s8.pt) fed the REAL pipeline's per-bar
root softmax (scratchpad/real_root_proba.py). QUALITATIVE checkpoint only —
no GT section labels exist for real audio in this repo, so there is no V_F
number here, by design (see the task brief). Writes a JSON per song for the
deploy step to render.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_root_proba import per_bar_root_proba
from symstruct_proba import (BlockEncoder, predict_learned_union_proba_bars,
                             estimate_tonic_pc)

REPO = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO / "docs" / "audio"
OUT = Path(__file__).resolve().parent / "real_structure_results.json"

SONGS = [
    ("autumn_leaves", AUDIO_DIR / "autumn_leaves.m4a"),
    ("abba_chiquitita", AUDIO_DIR / "abba_chiquitita_official_lyric_video.m4a"),
    ("aretha_chain_of_fools", AUDIO_DIR / "aretha_franklin_chain_of_fools_official_lyric_video.m4a"),
]


def load_model():
    ckpt = torch.load(Path(__file__).resolve().parent / "keynorm_proba_rootonly_s8.pt",
                      map_location="cpu", weights_only=False)
    model = BlockEncoder(root_mode="proba", qual_mode="none")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt["tau_star"])


def main():
    model, tau = load_model()
    print("loaded model, tau*=%.2f" % tau)
    results = {}
    for name, path in SONGS:
        if not path.exists():
            print("MISSING audio for %s (%s) — skipping" % (name, path))
            continue
        print("\n=== %s ===" % name)
        bar_proba, bar_times, tempo = per_bar_root_proba(path)
        tonic = estimate_tonic_pc(bar_proba)
        shift = (-tonic) % 12
        labels = predict_learned_union_proba_bars(
            bar_proba, model, "cpu", size=8, tau=tau, keynorm_shift=shift)
        n_sections = len(set(labels))
        print("  tempo=%.1f bpm  n_bars=%d  est_tonic_pc=%d  n_sections=%d"
              % (tempo, len(bar_proba), tonic, n_sections))
        # compact run-length summary
        runs = []
        cur = labels[0]; start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != cur:
                runs.append({"label": cur, "bar_start": start, "bar_end": i,
                            "t_start": round(float(bar_times[start]), 2),
                            "t_end": round(float(bar_times[min(i, len(bar_times)-1)]), 2)})
                if i < len(labels):
                    cur = labels[i]; start = i
        print("  %d runs: %s" % (len(runs), " ".join(
            "%s(%d-%d)" % (r["label"], r["bar_start"], r["bar_end"]) for r in runs)))
        results[name] = {
            "tempo_bpm": tempo, "n_bars": len(bar_proba),
            "est_tonic_pc": tonic, "n_sections": n_sections,
            "tau": tau, "runs": runs,
            "bar_times": [round(float(t), 3) for t in bar_times],
            "labels": labels,
        }
    OUT.write_text(json.dumps(results, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
