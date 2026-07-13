"""Mission 3 — section-merge (P3 pooling) eval.

For jazz songs whose GT contains a truly-repeated chord run, tie the two
matching spans (SectionMerge) so the joint decode POOLS their per-segment
emission log-scores, and measure whether the pooled chords beat decoding the
two spans separately. This is the "superimposed observations, variance ↓ ~1/N"
claim, gated by a GT-verified true repeat (never a blind average).

Own script (does not touch scripts/eval_joint_decode.py — concurrency).

Usage:
  .venv/bin/python scripts/eval_user_merge.py --start 20 --n 40 --min-len 4
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
MMA_TO_Q5IDX = {
    "maj": 0, "maj7": 0, "6": 0, "aug": 0, "augmaj7": 0, "sus2": 0, "sus4": 0,
    "min": 1, "min7": 1, "m6": 1, "minmaj7": 1,
    "dom7": 2, "dom7alt": 2, "7": 2, "9": 2, "aug7": 2, "7sus4": 2,
    "hdim7": 3, "m7b5": 3, "dim": 4, "dim7": 4,
}
Q5_MAJMIN = {0: "maj", 1: "min", 2: "maj", 3: "other", 4: "other"}


def find_repeat(spans, min_len):
    """Largest pair of non-overlapping equal-length GT chord runs that are
    identical in (root, q5). Returns (i, j, L) chord-index ranges or None."""
    seq = [(r % 12, MMA_TO_Q5IDX.get(q)) for _, _, r, q in spans]
    n = len(seq)
    best = None
    for L in range(n // 2, min_len - 1, -1):
        for i in range(0, n - L):
            for j in range(i + L, n - L + 1):
                if seq[i:i + L] == seq[j:j + L] and None not in seq[i:i + L]:
                    return (i, j, L)
    return best


def pred_at(chart, t):
    lab = None
    for c in chart.chords:
        if c["start_s"] <= t < c["end_s"]:
            lab = c["label"]; break
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


def score_span(chart, spans, t0, t1, step=0.05):
    rt = mm = n = 0
    t = t0
    while t < t1:
        g = None
        for a, b, r, q in spans:
            if a <= t < b:
                g = (r % 12, MMA_TO_Q5IDX.get(q)); break
        if g is not None and g[1] is not None and Q5_MAJMIN[g[1]] in ("maj", "min"):
            n += 1
            p = pred_at(chart, t)
            if p and p[0] == g[0]:
                rt += 1
                if Q5_MAJMIN.get(p[1]) == Q5_MAJMIN[g[1]]:
                    mm += 1
        t += step
    return (rt, mm, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--transition-weight", type=float, default=0.0)
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.n]

    base_rt = base_mm = mrg_rt = mrg_mm = tot = 0
    n_songs = 0
    with tempfile.TemporaryDirectory() as cd:
        cache = Path(cd)
        for i, rec in enumerate(held):
            spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                     if t1 > t0 and q in BUCKET_FAMILY]
            rep = find_repeat(spans, args.min_len) if spans else None
            if not rep:
                continue
            ci, cj, L = rep
            aA = (spans[ci][0], spans[ci + L - 1][1])
            aB = (spans[cj][0], spans[cj + L - 1][1])
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp,
                                RenderConfig(soundfont_path=sf2))
                base = P.infer_chords_v1(
                    tmp, cache_dir=cache,
                    joint_transition_weight=args.transition_weight)
                merged = P.infer_chords_v1(
                    tmp, cache_dir=cache,
                    joint_transition_weight=args.transition_weight,
                    user_constraints={"merges": [{"spans": [list(aA), list(aB)]}]})
                bs = [score_span(base, spans, *s) for s in (aA, aB)]
                ms = [score_span(merged, spans, *s) for s in (aA, aB)]
                ntot = sum(s[2] for s in bs)
                if ntot == 0:
                    continue
                base_rt += sum(s[0] for s in bs); base_mm += sum(s[1] for s in bs)
                mrg_rt += sum(s[0] for s in ms); mrg_mm += sum(s[1] for s in ms)
                tot += ntot
                n_songs += 1
                print(f"  {rec['song_id']}: repeat L={L} "
                      f"base_mm={sum(s[1] for s in bs)}/{ntot} "
                      f"merged_mm={sum(s[1] for s in ms)}/{ntot}", flush=True)
            finally:
                tmp.unlink(missing_ok=True)

    print(f"\n=== SECTION-MERGE pooling (jazz idx {args.start}..{args.start+args.n}, "
          f"{n_songs} songs w/ a GT repeat, tw={args.transition_weight}) ===")
    if tot:
        print(f"  frames scored: {tot}")
        print(f"  root  : base {base_rt/tot:.1%}  merged {mrg_rt/tot:.1%}  "
              f"Δ {(mrg_rt-base_rt)/tot:+.1%}")
        print(f"  majmin: base {base_mm/tot:.1%}  merged {mrg_mm/tot:.1%}  "
              f"Δ {(mrg_mm-base_mm)/tot:+.1%}")
    else:
        print("  (no songs with a qualifying GT repeat)")


if __name__ == "__main__":
    main()
