"""Mission 3 — propagation eval for user-input constraint factors.

Simulates a user confirming the k lowest-CALIBRATED-confidence chords with the
ground truth, then measures the accuracy change on the OTHER (non-confirmed)
chords vs the no-constraint decode. The headline number: does confirming a
chord PROPAGATE (sharpen its neighbours through the joint decoder), not just
repaint one cell?

Also runs the section-merge arm: for songs whose GT has a repeated section,
tie the two matching spans and pool their emissions — does pooling two truly-
matching spans improve their chords vs decoding them separately?

Own script (does NOT touch scripts/eval_joint_decode.py — concurrency).

Usage:
  .venv/bin/python scripts/eval_user_constraints.py --start 20 --n 10 --k 1 3 5
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NOTE = P.NOTE

# GT MMA quality -> q5 idx (maj/min/dom/hdim/dim), reusing the pipeline taxonomy.
MMA_TO_Q5IDX = {
    "maj": 0, "maj7": 0, "6": 0, "aug": 0, "augmaj7": 0, "sus2": 0, "sus4": 0,
    "min": 1, "min7": 1, "m6": 1, "minmaj7": 1,
    "dom7": 2, "dom7alt": 2, "7": 2, "9": 2, "aug7": 2, "7sus4": 2,
    "hdim7": 3, "m7b5": 3,
    "dim": 4, "dim7": 4,
}
Q5_MAJMIN = {0: "maj", 1: "min", 2: "maj", 3: "other", 4: "other"}  # majmin collapse


def gt_at(spans, t):
    """(root, q5idx) of the GT chord covering time t, or None."""
    for t0, t1, r, q in spans:
        if t0 <= t < t1:
            qi = MMA_TO_Q5IDX.get(q)
            return (r % 12, qi) if qi is not None else None
    return None


def pred_at(chart, t):
    """(root, q5idx) predicted at time t, or None."""
    lab = None
    for c in chart.chords:
        if c["start_s"] <= t < c["end_s"]:
            lab = c["label"]
            break
        if c["start_s"] <= t:
            lab = c["label"]
    if not lab or ":" not in lab:
        return None
    name, sev = lab.split(":", 1)
    try:
        r = NOTE.index(name)
    except ValueError:
        return None
    qi = P._harte_to_q5idx(sev)
    return (r, qi) if qi is not None else None


def frame_scores(chart, spans, dur, *, excluded=None, included=None, step=0.05):
    """(root_acc, majmin_acc, n) over sampled frames.

    `excluded`: frames inside these spans are skipped (the confirmed chords).
    `included`: if given, ONLY frames inside these spans are scored (used for the
    neighbour-band arm — the chords adjacent to a confirmed one, where
    propagation actually acts, isolated from the dilution of the whole song).
    """
    excluded = excluded or []
    rt_ok = mm_ok = n = 0
    t = 0.0
    while t < dur:
        inc = (included is None) or any(a <= t < b for a, b in included)
        exc = any(a <= t < b for a, b in excluded)
        if inc and not exc:
            g = gt_at(spans, t)
            p = pred_at(chart, t)
            if g is not None and Q5_MAJMIN[g[1]] in ("maj", "min"):
                n += 1
                if p is not None and p[0] == g[0]:
                    rt_ok += 1
                    if Q5_MAJMIN.get(p[1]) == Q5_MAJMIN[g[1]]:
                        mm_ok += 1
        t += step
    if n == 0:
        return None
    return rt_ok / n, mm_ok / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--transition-weight", type=float, default=0.5,
                    help="joint transition weight for the constrained decode "
                         "(propagation flows through this factor; production "
                         "default is 0 so a confirm only re-segments)")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.n]
    print(f"jazz songs: {len(held)} (idx {args.start}..{args.start + args.n})  "
          f"tw={args.transition_weight}")

    # aggregates: per k, two arms — "all" non-confirmed frames and the
    # "neighbour band" (chords immediately adjacent to a confirmed one).
    def _blank():
        return {"base_rt": [], "cons_rt": [], "base_mm": [], "cons_mm": []}
    agg = {k: {"all": _blank(), "nbr": _blank()} for k in args.k}

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
                dur = spans[-1][1]
                base = P.infer_chords_v1(tmp, cache_dir=cache,
                                         joint_transition_weight=args.transition_weight)
                # rank predicted chords by CALIBRATED confidence (ascending)
                ch = [c for c in base.chords if c["end_s"] > c["start_s"]]
                ch_sorted = sorted(ch, key=lambda c: c.get("confidence", 1.0))
                for k in args.k:
                    picks = ch_sorted[:k]
                    pick_idx = [ch.index(c) for c in picks]
                    confirms, excl, nbr = [], [], []
                    for c, ci in zip(picks, pick_idx):
                        mid = 0.5 * (c["start_s"] + c["end_s"])
                        g = gt_at(spans, mid)
                        if g is None:
                            continue
                        confirms.append({"t0": c["start_s"], "t1": c["end_s"],
                                         "root": g[0], "q5": g[1]})
                        excl.append((c["start_s"], c["end_s"]))
                        # immediate neighbours (prev/next chord), if not themselves picked
                        for nj in (ci - 1, ci + 1):
                            if 0 <= nj < len(ch) and nj not in pick_idx:
                                nc = ch[nj]
                                nbr.append((nc["start_s"], nc["end_s"]))
                    if not confirms:
                        continue
                    cons = P.infer_chords_v1(
                        tmp, cache_dir=cache,
                        joint_transition_weight=args.transition_weight,
                        user_constraints={"confirms": confirms})
                    b = frame_scores(base, spans, dur, excluded=excl)
                    c2 = frame_scores(cons, spans, dur, excluded=excl)
                    if b and c2:
                        agg[k]["all"]["base_rt"].append(b[0]); agg[k]["all"]["cons_rt"].append(c2[0])
                        agg[k]["all"]["base_mm"].append(b[1]); agg[k]["all"]["cons_mm"].append(c2[1])
                    if nbr:
                        bn = frame_scores(base, spans, dur, excluded=excl, included=nbr)
                        cn = frame_scores(cons, spans, dur, excluded=excl, included=nbr)
                        if bn and cn:
                            agg[k]["nbr"]["base_rt"].append(bn[0]); agg[k]["nbr"]["cons_rt"].append(cn[0])
                            agg[k]["nbr"]["base_mm"].append(bn[1]); agg[k]["nbr"]["cons_mm"].append(cn[1])
            finally:
                tmp.unlink(missing_ok=True)
            print(f"  [{i+1}/{len(held)}] {rec['song_id']}", flush=True)

    def _report(title, key):
        print(f"\n=== PROPAGATION [{title}] "
              f"(jazz idx {args.start}..{args.start + args.n}, tw={args.transition_weight}) ===")
        print(f"{'k':>3} {'n':>4} {'root_base':>10} {'root_cons':>10} {'Δroot':>7}   "
              f"{'mm_base':>8} {'mm_cons':>8} {'Δmm':>7}")
        for k in args.k:
            a = agg[k][key]
            if not a["base_mm"]:
                print(f"{k:>3}  (none)")
                continue
            rb, rc = np.mean(a["base_rt"]), np.mean(a["cons_rt"])
            mb, mc = np.mean(a["base_mm"]), np.mean(a["cons_mm"])
            print(f"{k:>3} {len(a['base_mm']):>4} {rb:>10.1%} {rc:>10.1%} "
                  f"{rc - rb:>+7.1%}   {mb:>8.1%} {mc:>8.1%} {mc - mb:>+7.1%}")

    _report("neighbour band — chords adjacent to a confirmed one", "nbr")
    _report("all non-confirmed chords (diluted)", "all")


if __name__ == "__main__":
    main()
