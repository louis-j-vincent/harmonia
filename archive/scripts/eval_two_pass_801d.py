"""End-to-end eval of the 801d two-pass key-relative ctx classifier (#20/#23).

Drives the REAL production pipeline (infer_chords_v1) on held-out jazz1460 songs,
comparing ctx_classifier_variant="684d" (current) vs "801d_two_pass" (new). Unlike
eval_irealb_e2e.py (which bypasses the ctx model and uses the raw family LR), this
exercises the actual ctx classifier + two-pass local-key reclassification that
ships — the honest "realizable prod gain", not the bootstrap upper bound.

Reports per variant:
  * MIREX root / majmin / sevenths (mir_eval, full-chart weighted overlap);
  * per-GT-family quality accuracy over the (maj/min/dom/hdim/dim) taxonomy —
    the breakdown that showed the bootstrap's +7.6pp minor-family gain.

The local-key/progression rerankers are OFF by default so the comparison
isolates the ctx classifier itself (the two-pass 801d is "SEUL", per the brief).

Usage:
    .venv/bin/python scripts/eval_two_pass_801d.py --n 25 --start 70
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
import soundfile as sf  # noqa: F401  (kept for parity / future use)

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import mir_eval

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from train_beat_seq_model_v3 import quality5, QUALITY5
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NOTE = P.NOTE
Q5_HARTE = ["maj", "min", "7", "maj7", "dim"]  # QUALITY5 order -> Harte

# MMA/iReal quality bucket -> the pipeline's 5-way q5 family (P._Q5_NAMES).
MMA_TO_Q5 = {
    "maj": "maj", "maj7": "maj", "6": "maj",
    "min": "min", "min7": "min", "m6": "min", "minmaj7": "min",
    "dom7": "dom", "dom7alt": "dom", "7": "dom", "9": "dom", "aug7": "dom",
    "aug": "maj", "augmaj7": "maj",
    "dim": "dim", "dim7": "dim",
    "m7b5": "hdim", "hdim7": "hdim",
    "sus2": "maj", "sus4": "maj", "7sus4": "dom",
}


def _pred_label_at(chart, t: float) -> str | None:
    """The predicted chord label covering time ``t`` (or the last before it)."""
    lab = None
    for c in chart.chords:
        if c["start_s"] <= t < c["end_s"]:
            return c["label"]
        if c["start_s"] <= t:
            lab = c["label"]
    return lab


def _pred_q5(label: str | None) -> str | None:
    if not label or ":" not in label:
        return None
    sev = label.split(":", 1)[1]
    idx = P._harte_to_q5idx(sev)
    return P._Q5_NAMES[idx] if idx is not None else None


def score_song(chart, spans):
    """Return (root, majmin, sevenths, per-family {fam: (correct, total)})."""
    ref_int, ref_lab = [], []
    fam_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t0, t1, r, q in spans:
        q5 = quality5(q)
        ref_int.append([t0, t1])
        ref_lab.append(f"{NOTE[r]}:{Q5_HARTE[q5] if q5 is not None else 'maj'}")
        # per-family quality accuracy (pipeline's maj/min/dom/hdim/dim taxonomy)
        gt_fam = MMA_TO_Q5.get(q)
        if gt_fam is not None:
            pred_fam = _pred_q5(_pred_label_at(chart, 0.5 * (t0 + t1)))
            fam_hits[gt_fam][1] += 1
            if pred_fam == gt_fam:
                fam_hits[gt_fam][0] += 1

    est_int = [[c["start_s"], c["end_s"]] for c in chart.chords if c["end_s"] > c["start_s"]]
    est_lab = [c["label"] for c in chart.chords if c["end_s"] > c["start_s"]]
    try:
        sco = mir_eval.chord.evaluate(np.array(ref_int), ref_lab,
                                      np.array(est_int), est_lab)
    except ValueError:
        return None
    return sco["root"], sco["majmin"], sco["sevenths"], fam_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=70)
    ap.add_argument("--variants", nargs="+", default=["684d", "801d_two_pass"])
    ap.add_argument("--use-progression-prior", action="store_true",
                    help="also apply the #21 progression rerank (off by default to "
                         "isolate the ctx classifier)")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")

    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.n]
    print(f"held-out jazz songs: {len(held)} (index {args.start}..{args.start + args.n})")
    print(f"variants: {args.variants}  progression_prior={args.use_progression_prior}")

    agg = {v: {"root": [], "majmin": [], "7ths": [],
               "fam": defaultdict(lambda: [0, 0])} for v in args.variants}

    with tempfile.TemporaryDirectory() as cache_dir:
        cache = Path(cache_dir)
        for i, rec in enumerate(held):
            print(f"  [{i + 1}/{len(held)}] {rec['song_id']}", flush=True)
            spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                     if t1 > t0 and q in BUCKET_FAMILY]
            if not spans:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp,
                                RenderConfig(soundfont_path=sf2))
                for v in args.variants:
                    chart = P.infer_chords_v1(
                        tmp, cache_dir=cache,
                        ctx_classifier_variant=v,
                        use_progression_prior=args.use_progression_prior,
                        use_local_key_prior=False,
                    )
                    res = score_song(chart, spans)
                    if res is None:
                        continue
                    agg[v]["root"].append(res[0])
                    agg[v]["majmin"].append(res[1])
                    agg[v]["7ths"].append(res[2])
                    for fam, (c, n) in res[3].items():
                        agg[v]["fam"][fam][0] += c
                        agg[v]["fam"][fam][1] += n
            finally:
                tmp.unlink(missing_ok=True)

    fams = ["maj", "min", "dom", "hdim", "dim"]
    print("\n=== 801d two-pass vs 684d — end-to-end on held-out jazz1460 ===")
    print(f"{'variant':<15} {'root':>6} {'majmin':>7} {'7ths':>6} {'n':>4}   "
          + "  ".join(f"{f:>5}" for f in fams))
    print("-" * 78)
    for v in args.variants:
        a = agg[v]
        if not a["root"]:
            print(f"{v:<15}  (no scored songs)")
            continue
        fam_str = []
        for f in fams:
            c, n = a["fam"][f]
            fam_str.append(f"{(c / n):>5.0%}" if n else f"{'—':>5}")
        print(f"{v:<15} {np.mean(a['root']):>6.1%} {np.mean(a['majmin']):>7.1%} "
              f"{np.mean(a['7ths']):>6.1%} {len(a['root']):>4}   " + "  ".join(fam_str))
    # family counts (support)
    if args.variants:
        v0 = args.variants[0]
        print("support (GT chords/family): " + "  ".join(
            f"{f}={agg[v0]['fam'][f][1]}" for f in fams))


if __name__ == "__main__":
    main()
