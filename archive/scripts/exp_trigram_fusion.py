"""Trigram-in-the-slot falsification check (#27 Mission 1 addendum).

Does a scale-relative TRIGRAM P(s_i | s_{i-2}, s_{i-1}) over ((root−tonic)%12, q5)
states — add-k smoothed, backed off to bigram→unigram — escape the bigram/encoder
label-bias diagnosis when fused into the joint decode as an iterated q5 emission
bonus? Score-only experiment on the FIT split (jazz1460 idx 20..30); production
files untouched — the fusion enters via a runtime monkeypatch of
``harmonia.models.joint_decode.joint_decode`` (the pipeline late-imports it), so
semi-Markov boundaries (Mission 2, default ON) are exactly production's.

Trigram fit EXCLUDES both the fit split (idx 20..30) and the gate split
(idx 70..95) under the same jz filter eval_joint_decode.py uses.

Usage: .venv/bin/python scripts/exp_trigram_fusion.py --weights 0.1 0.25 0.5
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
from analyze_accomp_priors import parse_key
from build_audio_chord_features import BUCKET_FAMILY
from eval_two_pass_801d import score_song
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P
from harmonia.models import joint_decode as JD
from harmonia.models.progression_encoder import fine_to_q5

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NS = 60  # 12 degrees × 5 q5 classes


def st(deg: int, q5: int) -> int:
    return (deg % 12) * 5 + q5


# ── n-gram fit over (deg, q5) states ──────────────────────────────────────────
def gt_state_seq(rec) -> list[int] | None:
    k = parse_key(rec["key"])
    if k is None:
        return None
    tonic = k[0]
    seq = []
    for _t0, _t1, root, qual in song_chord_spans(rec):
        q5 = fine_to_q5(qual)
        if q5 is None or qual not in BUCKET_FAMILY:
            continue
        s = st((root - tonic) % 12, q5)
        if not seq or seq[-1] != s:          # collapse repeats, like the bigram fit
            seq.append(s)
    return seq if len(seq) >= 3 else None


class NGram:
    """Add-k trigram with interpolated backoff to bigram → unigram (all in log)."""

    def __init__(self, seqs: list[list[int]], k3: float = 0.2, k2: float = 0.2):
        self.uni = np.full(NS, 1.0)
        self.bi = defaultdict(float)
        self.bi_ctx = defaultdict(float)
        self.tri = defaultdict(float)
        self.tri_ctx = defaultdict(float)
        self.k3, self.k2 = k3, k2
        for s in seqs:
            for a in s:
                self.uni[a] += 1
            for a, b in zip(s, s[1:]):
                self.bi[(a, b)] += 1
                self.bi_ctx[a] += 1
            for a, b, c in zip(s, s[1:], s[2:]):
                self.tri[(a, b, c)] += 1
                self.tri_ctx[(a, b)] += 1
        self.uni_p = self.uni / self.uni.sum()

    def p_bi(self, a: int, c: int) -> float:
        n = self.bi_ctx.get(a, 0.0)
        return (self.bi.get((a, c), 0.0) + self.k2 * NS * self.uni_p[c]) / (n + self.k2 * NS)

    def lp_tri(self, a: int, b: int, c: int) -> float:
        n = self.tri_ctx.get((a, b), 0.0)
        p = (self.tri.get((a, b, c), 0.0) + self.k3 * NS * self.p_bi(b, c)) / (n + self.k3 * NS)
        return float(np.log(p))

    def lp_uni(self, c: int) -> float:
        return float(np.log(self.uni_p[c]))


def coverage_stats(model: NGram, fit_seqs: list[list[int]]) -> dict:
    """How often do the FIT split's GT trigram contexts/events exist in the table?"""
    ctx_seen = ev_seen = tot = 0
    for s in fit_seqs:
        for a, b, c in zip(s, s[1:], s[2:]):
            tot += 1
            ctx_seen += (a, b) in model.tri_ctx
            ev_seen += (a, b, c) in model.tri
    return {"n_test_trigrams": tot,
            "ctx_coverage": ctx_seen / max(1, tot),
            "event_coverage": ev_seen / max(1, tot),
            "distinct_tri": len(model.tri), "distinct_bi": len(model.bi),
            "tri_tokens": int(sum(model.tri.values()))}


# ── fusion wrapper (runtime patch; production untouched) ──────────────────────
_ORIG_JOINT = JD.joint_decode
CFG = {"lam": 0.0, "mode": "left", "ratio": False, "model": None}


def _patched_joint_decode(segs, beat_proba, classify_fn, tonic, **kw):
    dec = _ORIG_JOINT(segs, beat_proba, classify_fn, tonic, **kw)
    lam, model = CFG["lam"], CFG["model"]
    if lam <= 0.0 or model is None or len(segs) < 3:
        return dec
    roots, q5s = list(dec["roots"]), list(dec["q5"])
    N = len(roots)
    dstate = [st((roots[i] - tonic) % 12, q5s[i]) for i in range(N)]
    full = CFG["mode"] == "full"
    ratio = CFG["ratio"]

    def bonus(i: int, root: int) -> np.ndarray:
        out = np.zeros(5, dtype=np.float64)
        deg = (root - tonic) % 12
        for q in range(5):
            si = st(deg, q)
            terms = 0
            v = 0.0
            if i >= 2:
                v += model.lp_tri(dstate[i - 2], dstate[i - 1], si); terms += 1
            if full and 1 <= i <= N - 2:
                v += model.lp_tri(dstate[i - 1], si, dstate[i + 1]); terms += 1
            if full and i <= N - 3:
                v += model.lp_tri(si, dstate[i + 1], dstate[i + 2]); terms += 1
            if ratio:
                v -= terms * model.lp_uni(si)
            out[q] = lam * v
        return out

    return _ORIG_JOINT(segs, beat_proba, classify_fn, tonic,
                       **{**kw, "q5_bonus": bonus})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    excl = {r["song_id"] for r in jz[20:31]} | {r["song_id"] for r in jz[70:96]}
    train = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["song_id"] not in excl]
    train_seqs = [s for s in (gt_state_seq(r) for r in train) if s]
    model = NGram(train_seqs)
    CFG["model"] = model

    fit_seqs = [s for s in (gt_state_seq(r) for r in jz[args.start:args.start + args.n]) if s]
    cov = coverage_stats(model, fit_seqs)
    print(f"trigram fit: {len(train_seqs)} songs (excl fit+gate) | "
          f"distinct tri {cov['distinct_tri']} / bi {cov['distinct_bi']} | "
          f"tri tokens {cov['tri_tokens']}")
    print(f"fit-split GT coverage: context {cov['ctx_coverage']:.1%}, "
          f"exact trigram {cov['event_coverage']:.1%} of {cov['n_test_trigrams']}")

    # infer_chords_v1 late-imports joint_decode from the module at call time, so
    # patching the module attribute routes every decode through the wrapper.
    JD.joint_decode = _patched_joint_decode

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    held = jz[args.start:args.start + args.n]

    arms = [("baseline w=0", 0.0, "left", False)]
    for mode in ("left", "full"):
        for ratio in (False, True):
            for w in args.weights:
                arms.append((f"{mode}{'-ratio' if ratio else ''} λ={w}", w, mode, ratio))
    agg = {name: {"root": [], "majmin": [], "7ths": [],
                  "fam": defaultdict(lambda: [0, 0])} for name, *_ in arms}

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
                for name, w, mode, ratio in arms:
                    CFG.update(lam=w, mode=mode, ratio=ratio)
                    chart = P.infer_chords_v1(tmp, cache_dir=cache)
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
    print(f"\n=== trigram fusion — jazz1460 idx {args.start}..{args.start + args.n} "
          f"(semi-Markov ON) ===")
    print(f"{'arm':<18} {'root':>6} {'majmin':>7} {'7ths':>6} {'n':>3}   "
          + "  ".join(f"{f:>5}" for f in fams))
    print("-" * 84)
    for name, *_ in arms:
        a = agg[name]
        if not a["root"]:
            continue
        fam_str = []
        for f in fams:
            c, n = a["fam"][f]
            fam_str.append(f"{(c / n):>5.0%}" if n else f"{'—':>5}")
        print(f"{name:<18} {np.mean(a['root']):>6.1%} {np.mean(a['majmin']):>7.1%} "
              f"{np.mean(a['7ths']):>6.1%} {len(a['root']):>3}   " + "  ".join(fam_str))


if __name__ == "__main__":
    main()
