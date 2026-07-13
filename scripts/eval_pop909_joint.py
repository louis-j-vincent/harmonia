"""POP909 gate for the joint (root × quality) decode (audit build-order step 2).

Mirrors scripts/eval_pop909_reranker_ab.py but compares the greedy baseline arm
against use_joint_decode=True on the 5 rendered POP909 songs (v005 renders).

Usage: .venv/bin/python scripts/eval_pop909_joint.py --weight 1.0
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.pop909_parser import POP909Parser  # noqa: E402
from harmonia.eval.mirex_eval import evaluate_song  # noqa: E402
from harmonia.models.chord_pipeline_v1 import infer_chords_v1  # noqa: E402

DATA_ROOT = REPO / "data"
SONGS = ["001", "002", "003", "004", "005"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()

    parser = POP909Parser(DATA_ROOT / "pop909" / "POP909")
    agg = {arm: {"root": [], "majmin": [], "7ths": []} for arm in ("greedy", "joint")}
    for song_id in SONGS:
        wav = (DATA_ROOT / "renders" / "pop909" / song_id
               / f"{song_id}_v005_musescoregeneral.wav")
        if not wav.exists():
            print(f"  {song_id}: render missing, skipped")
            continue
        gt = parser.parse_song(song_id)
        ref_int = np.array([[ev.start_beat, ev.end_beat] for ev in gt.chord_events])
        ref_lab = [ev.label for ev in gt.chord_events]
        for arm in ("greedy", "joint"):
            kw = dict(use_joint_decode=True, joint_K=args.K,
                      joint_transition_weight=args.weight) if arm == "joint" else {}
            chart = infer_chords_v1(wav, cache_dir=DATA_ROOT / "cache", **kw)
            sco = evaluate_song(chart.chords, ref_int, ref_lab)
            agg[arm]["root"].append(sco.root)
            agg[arm]["majmin"].append(sco.majmin)
            agg[arm]["7ths"].append(sco.sevenths)
            print(f"  {song_id} {arm:>6}: root={sco.root:.1%} "
                  f"majmin={sco.majmin:.1%} 7ths={sco.sevenths:.1%}", flush=True)

    print(f"\n=== POP909 (5 songs, v005) — greedy vs joint (w={args.weight}) ===")
    print(f"{'arm':<7} {'root':>7} {'majmin':>7} {'7ths':>7} {'n':>3}")
    for arm in ("greedy", "joint"):
        a = agg[arm]
        if not a["root"]:
            continue
        print(f"{arm:<7} {np.mean(a['root']):>7.1%} {np.mean(a['majmin']):>7.1%} "
              f"{np.mean(a['7ths']):>7.1%} {len(a['root']):>3}")


if __name__ == "__main__":
    main()
