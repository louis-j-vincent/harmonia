"""Eval / weight-sweep for the joint (root × quality) decode on jazz1460.

Real-path harness (mirrors eval_two_pass_801d.py's scoring) that drives
infer_chords_v1 with use_joint_decode=True and reports MIREX root/majmin/7ths.
Used for (a) the transition_weight sweep on the FIT split (idx 20..50, NEVER the
eval split) and (b) the held-out GATE (idx 70..95) vs the greedy baseline arm.

Usage:
  sweep (fit):  .venv/bin/python scripts/eval_joint_decode.py --start 20 --n 10 \
                    --weights 0.0 0.5 1.0 2.0
  gate (held):  .venv/bin/python scripts/eval_joint_decode.py --start 70 --n 25 \
                    --weights 1.0 --baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from eval_two_pass_801d import score_song
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--baseline", action="store_true",
                    help="also run the greedy (use_joint_decode=False) arm")
    ap.add_argument("--local-key", action="store_true",
                    help="re-reference the joint transition to the per-chord local key (H1)")
    ap.add_argument("--fusion", action="store_true",
                    help="H2: --weights are ProgressionEncoder shallow-fusion λ (transition off)")
    ap.add_argument("--fusion-iters", type=int, default=1)
    ap.add_argument("--subtract-prior", action="store_true",
                    help="H3: subtract the encoder's marginal (density-ratio fusion)")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.n]
    print(f"jazz songs: {len(held)} (idx {args.start}..{args.start + args.n})  K={args.K}")

    arms = []
    if args.baseline:
        arms.append(("greedy", None))
    arms += [(f"joint w={w}", w) for w in args.weights]
    agg = {name: {"root": [], "majmin": [], "7ths": [],
                  "fam": defaultdict(lambda: [0, 0])} for name, _ in arms}

    with tempfile.TemporaryDirectory() as cd:
        cache = Path(cd)
        for i, rec in enumerate(held):
            spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                     if t1 > t0 and q in BUCKET_FAMILY]
            if not spans:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp,
                                RenderConfig(soundfont_path=sf2))
                for name, w in arms:
                    if w is None:
                        chart = P.infer_chords_v1(tmp, cache_dir=cache)
                    elif args.fusion:
                        chart = P.infer_chords_v1(
                            tmp, cache_dir=cache, use_joint_decode=True,
                            joint_K=args.K, joint_transition_weight=0.0,
                            joint_progression_fusion=(w > 0.0),
                            joint_progression_weight=w,
                            joint_fusion_iters=args.fusion_iters,
                            joint_fusion_subtract_prior=args.subtract_prior)
                    else:
                        chart = P.infer_chords_v1(
                            tmp, cache_dir=cache, use_joint_decode=True,
                            joint_K=args.K, joint_transition_weight=w,
                            joint_local_key_transition=args.local_key)
                    res = score_song(chart, spans)
                    if res is None:
                        continue
                    agg[name]["root"].append(res[0])
                    agg[name]["majmin"].append(res[1])
                    agg[name]["7ths"].append(res[2])
                    for fam, (c, n) in res[3].items():
                        agg[name]["fam"][fam][0] += c
                        agg[name]["fam"][fam][1] += n
            finally:
                tmp.unlink(missing_ok=True)
            print(f"  [{i+1}/{len(held)}] {rec['song_id']}", flush=True)

    fams = ["maj", "min", "dom", "hdim", "dim"]
    print(f"\n=== joint decode — jazz1460 idx {args.start}..{args.start + args.n} ===")
    print(f"{'arm':<12} {'root':>6} {'majmin':>7} {'7ths':>6} {'n':>4}   "
          + "  ".join(f"{f:>5}" for f in fams))
    print("-" * 78)
    for name, _ in arms:
        a = agg[name]
        if not a["root"]:
            print(f"{name:<12}  (no scored songs)")
            continue
        fam_str = []
        for f in fams:
            c, n = a["fam"][f]
            fam_str.append(f"{(c / n):>5.0%}" if n else f"{'—':>5}")
        print(f"{name:<12} {np.mean(a['root']):>6.1%} {np.mean(a['majmin']):>7.1%} "
              f"{np.mean(a['7ths']):>6.1%} {len(a['root']):>4}   " + "  ".join(fam_str))


if __name__ == "__main__":
    main()
