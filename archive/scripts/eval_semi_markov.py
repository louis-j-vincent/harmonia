"""Eval / sweep for the per-beat semi-Markov (explicit-duration) decode (#27 M2).

Baseline arm = current production (segment joint decode). Semi-Markov arms sweep
duration-prior weight and (optionally) the v3 quality-emission weight.

  fit sweep:  .venv/bin/python scripts/eval_semi_markov.py --start 20 --n 10 \
                  --dur-weights 0 0.25 0.5 1.0 --baseline
  gate:       .venv/bin/python scripts/eval_semi_markov.py --start 70 --n 25 \
                  --dur-weights 1.0 --baseline
"""
from __future__ import annotations
import argparse, json, sys, tempfile, warnings, time
from collections import defaultdict
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from eval_two_pass_801d import score_song, MMA_TO_Q5
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def _root_err_intervals(chart, spans):
    """Count root errors by semitone interval (pred - gt) % 12 at span centres."""
    ivals = defaultdict(int)
    for t0, t1, r, q in spans:
        lab = None
        for c in chart.chords:
            if c["start_s"] <= 0.5 * (t0 + t1) < c["end_s"]:
                lab = c["label"]; break
        if lab is None or ":" not in lab:
            continue
        pr = P.NOTE.index(lab.split(":")[0]) if lab.split(":")[0] in P.NOTE else None
        if pr is None:
            continue
        d = (pr - (r % 12)) % 12
        if d != 0:
            ivals[d] += 1
    return ivals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--dur-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--qual-weight", type=float, default=0.0)
    ap.add_argument("--per-quality-dur", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="also run production joint decode")
    ap.add_argument("--pop909", action="store_true", help="eval on POP909 5-song instead")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.n]
    print(f"jazz songs: {len(held)} (idx {args.start}..{args.start+args.n}) "
          f"qual_w={args.qual_weight} per_q_dur={args.per_quality_dur}")

    arms = []
    if args.baseline:
        arms.append(("prod(joint)", None))
    arms += [(f"sm dur={w}", w) for w in args.dur_weights]
    agg = {name: {"root": [], "majmin": [], "7ths": [], "nchord": [], "meandur": [],
                  "fam": defaultdict(lambda: [0, 0]), "rerr": defaultdict(int)}
           for name, _ in arms}
    gt_meandur = []
    t0 = time.time()

    with tempfile.TemporaryDirectory() as cd:
        cache = Path(cd)
        for i, rec in enumerate(held):
            spans = [(a, b, r % 12, q) for a, b, r, q in song_chord_spans(rec)
                     if b > a and q in BUCKET_FAMILY]
            if not spans:
                continue
            spb = 60.0 / rec["tempo"]
            gt_meandur.append(np.mean([(b - a) / spb for a, b, _, _ in spans]))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
                for name, w in arms:
                    if w is None:
                        # explicit baseline: production joint decode w/o semi-Markov
                        chart = P.infer_chords_v1(tmp, cache_dir=cache,
                                                  use_semi_markov=False)
                    else:
                        chart = P.infer_chords_v1(
                            tmp, cache_dir=cache, use_joint_decode=True,
                            use_semi_markov=True, semi_markov_dur_weight=w,
                            semi_markov_qual_weight=args.qual_weight,
                            semi_markov_per_quality_dur=args.per_quality_dur)
                    res = score_song(chart, spans)
                    if res is None:
                        continue
                    agg[name]["root"].append(res[0]); agg[name]["majmin"].append(res[1])
                    agg[name]["7ths"].append(res[2])
                    agg[name]["nchord"].append(len(chart.chords))
                    agg[name]["meandur"].append(np.mean([c["duration_beats"] for c in chart.chords]))
                    for fam, (c, n) in res[3].items():
                        agg[name]["fam"][fam][0] += c; agg[name]["fam"][fam][1] += n
                    for d, cnt in _root_err_intervals(chart, spans).items():
                        agg[name]["rerr"][d] += cnt
            finally:
                tmp.unlink(missing_ok=True)
            print(f"  [{i+1}/{len(held)}] {rec['song_id']} ({time.time()-t0:.0f}s)", flush=True)

    fams = ["maj", "min", "dom", "hdim", "dim"]
    print(f"\n=== semi-Markov — jazz idx {args.start}..{args.start+args.n}  "
          f"GT mean chord dur={np.mean(gt_meandur):.2f} beats ===")
    print(f"{'arm':<13} {'root':>6} {'majmin':>7} {'7ths':>6} {'n':>3} {'nchd':>5} {'mdur':>5}  "
          + "  ".join(f"{f:>5}" for f in fams))
    print("-" * 92)
    for name, _ in arms:
        a = agg[name]
        if not a["root"]:
            print(f"{name:<13} (none)"); continue
        fam_str = []
        for f in fams:
            c, n = a["fam"][f]
            fam_str.append(f"{(c/n):>5.0%}" if n else f"{'—':>5}")
        print(f"{name:<13} {np.mean(a['root']):>6.1%} {np.mean(a['majmin']):>7.1%} "
              f"{np.mean(a['7ths']):>6.1%} {len(a['root']):>3} {np.mean(a['nchord']):>5.0f} "
              f"{np.mean(a['meandur']):>5.1f}  " + "  ".join(fam_str))
    print("\nroot-error intervals (pred-gt semitones):")
    for name, _ in arms:
        rr = agg[name]["rerr"]; tot = sum(rr.values())
        if not tot: continue
        top = sorted(rr.items(), key=lambda x: -x[1])[:4]
        p5 = 100 * (rr.get(7, 0) + rr.get(5, 0)) / tot
        print(f"  {name:<13} n_err={tot:4d}  5th-apart(±5/7)={p5:.0f}%  "
              + " ".join(f"{d:+d}:{c}" for d, c in top))


if __name__ == "__main__":
    main()
