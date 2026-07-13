"""Addendum #2 (#27): ENTROPY-GATED trigram fusion + ii-V-I slice metric.

The user's counter-example: "when there is a 2-5-1, we know the 5 has a dom 7."
Two flaws in the always-on sweeps: (a) a global λ lets corpus-marginal losses
swamp cadence wins; (b) MIREX majmin maps dom7→maj, so the V:maj→dom fix is
INVISIBLE in majmin — it lives in 7ths + dom-recall. This experiment:

1. Slice metric: GT ii-V (deg2,min)→(deg7,dom) instances (global tonic), with or
   without the (deg0,maj) resolution; count V chords production mislabels at q5.
2. Gated fusion: trigram bonus fires at position i ONLY when (i) the trigram
   predictive over s_i given decoded (s_{i-2}, s_{i-1}) is SHARP (entropy < thr),
   (ii) the centre's acoustic q5 max-prob < τ (uncertain), (iii) both left
   context chords have decoded conf ≥ 0.5. Everywhere else λ=0.

Score-only on the fit split; production untouched (same runtime monkeypatch).

Usage: .venv/bin/python scripts/exp_trigram_gated.py
"""
from __future__ import annotations

import argparse
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

import json

from analyze_accomp_emission import song_chord_spans
from analyze_accomp_priors import parse_key
from build_audio_chord_features import BUCKET_FAMILY
from eval_two_pass_801d import _pred_label_at, score_song
from exp_trigram_fusion import DB, NS, NGram, gt_state_seq, st
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P
from harmonia.models import joint_decode as JD
from harmonia.models.progression_encoder import QUAL5, fine_to_q5

NOTE_PC = {n: i for i, n in enumerate(P.NOTE)}


# ── ii-V(-I) slice from GT spans ──────────────────────────────────────────────
def iiV_instances(rec) -> list[dict]:
    """GT ii-V instances, KEY-AGNOSTIC (catches tonicized ii-Vs, the common jazz
    case): adjacent (root, min) → (root+5, dom); ``strict`` marks the global-key
    (deg2,min)→(deg7,dom) subset; ``has_I`` a following (V_root+5, maj|min)."""
    k = parse_key(rec["key"])
    tonic = k[0] if k is not None else None
    merged: list[list] = []   # [t0, t1, root_pc, q5]
    for t0, t1, root, qual in song_chord_spans(rec):
        q5 = fine_to_q5(qual)
        if q5 is None or qual not in BUCKET_FAMILY or t1 <= t0:
            continue
        if merged and merged[-1][2] == root % 12 and merged[-1][3] == q5:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1, root % 12, q5])
    out = []
    for j in range(len(merged) - 1):
        a, b = merged[j], merged[j + 1]
        if a[3] == 1 and b[3] == 2 and b[2] == (a[2] + 5) % 12:  # min → dom up a 4th
            has_I = (j + 2 < len(merged)
                     and merged[j + 2][2] == (b[2] + 5) % 12
                     and merged[j + 2][3] in (0, 1))
            strict = (tonic is not None and (a[2] - tonic) % 12 == 2)
            out.append({"t0": b[0], "t1": b[1], "root": b[2],
                        "has_I": has_I, "strict": strict})
    return out


def slice_score(chart, insts: list[dict]) -> tuple[int, int, int]:
    """(n_V, q5_dom_hits, root_and_dom_hits) at each V chord's midpoint."""
    nV = dom = rootdom = 0
    for inst in insts:
        t = 0.5 * (inst["t0"] + inst["t1"])
        lab = _pred_label_at(chart, t)
        if not lab or ":" not in lab:
            nV += 1
            continue
        r, sev = lab.split(":", 1)
        q5 = P._harte_to_q5idx(sev)
        nV += 1
        if q5 == 2:
            dom += 1
            if NOTE_PC.get(r) == inst["root"]:
                rootdom += 1
    return nV, dom, rootdom


# ── gated fusion wrapper ──────────────────────────────────────────────────────
_ORIG_JOINT = JD.joint_decode
CFG = {"lam": 0.0, "H_thr": 0.0, "tau": 0.0, "ctx_conf": 0.5, "model": None,
       "fires": 0, "positions": 0, "H_samples": [], "diag": None,
       # Mission 3: when True the centre's uncertainty gate tests the FUSED
       # score (joint root+quality marginal, dec["conf"][i]) instead of the
       # quality-only p_max — i.e. gate on HONEST uncertainty (issue #29). The
       # hypothesis: quality p_max is dishonestly high on real audio (conf 0.93
       # on errors), so the τ gate never opens where it should; the fused score
       # shrinks on weak-root chords, letting the trigram finally fire there.
       "fused_gate": False}


def _patched_joint_decode(segs, beat_proba, classify_fn, tonic, **kw):
    dec = _ORIG_JOINT(segs, beat_proba, classify_fn, tonic, **kw)
    model = CFG["model"]
    if model is None or len(segs) < 3:
        return dec
    roots, q5s, confs = list(dec["roots"]), list(dec["q5"]), list(dec["conf"])
    q5_logps = list(dec["q5_logp"])
    N = len(roots)
    dstate = [st((roots[i] - tonic) % 12, q5s[i]) for i in range(N)]

    # per-position predictive entropy + gate decision (root-independent)
    gate_open = np.zeros(N, dtype=bool)
    diag = []
    for i in range(N):
        p_max = float(np.exp(np.max(np.asarray(q5_logps[i], dtype=np.float64))))
        H = None
        if i >= 2:
            pred = np.array([np.exp(model.lp_tri(dstate[i - 2], dstate[i - 1], c))
                             for c in range(NS)])
            pred = pred / pred.sum()
            H = float(-(pred * np.log(pred + 1e-12)).sum())
            CFG["H_samples"].append(H)
            CFG["positions"] += 1
            # centre uncertainty: fused (honest, joint root+quality) or quality-only
            unc = confs[i] if CFG["fused_gate"] else p_max
            if (H < CFG["H_thr"] and unc < CFG["tau"]
                    and confs[i - 1] >= CFG["ctx_conf"]
                    and confs[i - 2] >= CFG["ctx_conf"]):
                gate_open[i] = True
                CFG["fires"] += 1
        # label exactly as the pipeline builds it (for coalesce-index mapping)
        diag.append({"label": f"{P.NOTE[roots[i]]}:{dec['sev_h'][i]}",
                     "p_max": p_max, "H": H, "q5": q5s[i],
                     "ctx_ok": (i >= 2 and confs[i - 1] >= CFG["ctx_conf"]
                                and confs[i - 2] >= CFG["ctx_conf"])})
    CFG["diag"] = diag
    if CFG["lam"] <= 0.0 or not gate_open.any():
        return dec

    lam = CFG["lam"]

    def bonus(i: int, root: int) -> np.ndarray:
        out = np.zeros(5, dtype=np.float64)
        if not gate_open[i]:
            return out
        deg = (root - tonic) % 12
        for q in range(5):
            out[q] = lam * model.lp_tri(dstate[i - 2], dstate[i - 1], st(deg, q))
        return out

    return _ORIG_JOINT(segs, beat_proba, classify_fn, tonic,
                       **{**kw, "q5_bonus": bonus})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--fused-gate", action="store_true",
                    help="Mission 3: gate centre uncertainty on the FUSED score "
                         "(joint root+quality marginal) instead of quality p_max "
                         "— re-test the trigram against HONEST uncertainty (#29).")
    args = ap.parse_args()
    CFG["fused_gate"] = args.fused_gate
    if args.fused_gate:
        print("fused-gate ON: τ tests the joint root+quality marginal (honest "
              "uncertainty), not quality-only p_max.")

    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    excl = {r["song_id"] for r in jz[20:31]} | {r["song_id"] for r in jz[70:96]}
    train = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["song_id"] not in excl]
    CFG["model"] = NGram([s for s in (gt_state_seq(r) for r in train) if s])
    JD.joint_decode = _patched_joint_decode

    held = jz[args.start:args.start + args.n]
    all_inst = [iiV_instances(r) for r in held]
    n_inst = sum(len(x) for x in all_inst)
    n_strict = sum(1 for x in all_inst for i in x if i["strict"])
    n_withI = sum(1 for x in all_inst for i in x if i["has_I"])
    print(f"fit split: {len(held)} songs | GT ii-V instances: {n_inst} "
          f"(strict global-key ii: {n_strict}, with I resolution: {n_withI})")

    # arm grid: (name, lam, H_thr, tau)
    arms = [("baseline", 0.0, 4.2, 1.01)]   # gates 'open' for stats but lam=0
    for lam in (0.5, 1.0):
        for H_thr in (1.0, 1.75, 2.5):
            for tau in (0.65, 0.8):
                arms.append((f"λ={lam} H<{H_thr} τ<{tau}", lam, H_thr, tau))

    agg = {name: {"root": [], "majmin": [], "7ths": [],
                  "fam": defaultdict(lambda: [0, 0]),
                  "slice": [0, 0, 0], "fires": 0, "pos": 0}
           for name, *_ in arms}

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    v_diag: list[dict] = []
    with tempfile.TemporaryDirectory() as cd:
        cache = Path(cd)
        for i, rec in enumerate(held):
            spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                     if t1 > t0 and q in BUCKET_FAMILY]
            insts = iiV_instances(rec)
            if not spans:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp,
                                RenderConfig(soundfont_path=sf2))
                for name, lam, H_thr, tau in arms:
                    CFG.update(lam=lam, H_thr=H_thr, tau=tau,
                               fires=0, positions=0, diag=None)
                    chart = P.infer_chords_v1(tmp, cache_dir=cache)
                    agg[name]["fires"] += CFG["fires"]
                    agg[name]["pos"] += CFG["positions"]
                    # baseline arm: per-V diagnosis (why would/wouldn't the gate open)
                    if name == "baseline" and CFG["diag"] is not None:
                        d = CFG["diag"]
                        groups, glabels = [], []
                        for pi, rec_d in enumerate(d):
                            if glabels and glabels[-1] == rec_d["label"]:
                                groups[-1].append(pi)
                            else:
                                groups.append([pi])
                                glabels.append(rec_d["label"])
                        if len(groups) == len(chart.chords):
                            for inst in insts:
                                t = 0.5 * (inst["t0"] + inst["t1"])
                                gidx = next((g for g, c in enumerate(chart.chords)
                                             if c["start_s"] <= t < c["end_s"]), None)
                                if gidx is None:
                                    continue
                                pos = groups[gidx][0]
                                v_diag.append({**d[pos], "song": rec["song_id"]})
                        else:
                            print(f"    (diag skipped: {len(groups)} groups vs "
                                  f"{len(chart.chords)} chords)", flush=True)
                    res = score_song(chart, spans)
                    if res is not None:
                        agg[name]["root"].append(res[0])
                        agg[name]["majmin"].append(res[1])
                        agg[name]["7ths"].append(res[2])
                        for fam, (c, n) in res[3].items():
                            agg[name]["fam"][fam][0] += c
                            agg[name]["fam"][fam][1] += n
                    sv = slice_score(chart, insts)
                    for k in range(3):
                        agg[name]["slice"][k] += sv[k]
            finally:
                tmp.unlink(missing_ok=True)
            print(f"  [{i+1}/{len(held)}] {rec['song_id']}", flush=True)

    Hs = np.array(CFG["H_samples"])
    print(f"\npredictive-entropy percentiles (nats, uniform={np.log(NS):.2f}): "
          f"p10={np.percentile(Hs,10):.2f} p25={np.percentile(Hs,25):.2f} "
          f"p50={np.percentile(Hs,50):.2f} p75={np.percentile(Hs,75):.2f}")
    print(f"\n=== gated trigram — jazz idx {args.start}..{args.start+args.n} "
          f"(semi-Markov ON) ===")
    print(f"{'arm':<20} {'fire%':>6} {'V-dom':>7} {'V-r+d':>7} {'7ths':>6} "
          f"{'majmin':>7} {'root':>6} {'min':>5} {'hdim':>5}")
    print("-" * 84)
    for name, *_ in arms:
        a = agg[name]
        if not a["root"]:
            continue
        nV, dom, rdom = a["slice"]
        fire = a["fires"] / max(1, a["pos"])
        mn = a["fam"]["min"]
        hd = a["fam"]["hdim"]
        print(f"{name:<20} {fire:>6.1%} {dom}/{nV:>4} {rdom}/{nV:>4} "
              f"{np.mean(a['7ths']):>6.1%} {np.mean(a['majmin']):>7.1%} "
              f"{np.mean(a['root']):>6.1%} "
              f"{(mn[0]/mn[1] if mn[1] else 0):>5.0%} "
              f"{(hd[0]/hd[1] if hd[1] else 0):>5.0%}")

    # ── per-V diagnosis (baseline decode): WHY would the gate (not) open? ──────
    if v_diag:
        err = [d for d in v_diag if d["q5"] != 2]
        ok = [d for d in v_diag if d["q5"] == 2]
        print(f"\nV-chord diagnosis (baseline): {len(v_diag)} matched, "
              f"{len(err)} q5 errors / {len(ok)} correct")
        for tag, grp in (("ERRORS", err), ("correct", ok)):
            if not grp:
                continue
            pm = np.array([d["p_max"] for d in grp])
            hh = np.array([d["H"] for d in grp if d["H"] is not None])
            ctx = np.mean([d["ctx_ok"] for d in grp])
            print(f"  {tag:<8} p_max med={np.median(pm):.2f} "
                  f"(≥0.8: {np.mean(pm >= 0.8):.0%}, ≥0.65: {np.mean(pm >= 0.65):.0%}) | "
                  f"trigram H med={np.median(hh):.2f} (<2.5: {np.mean(hh < 2.5):.0%}, "
                  f"<1.75: {np.mean(hh < 1.75):.0%}) | ctx_conf ok {ctx:.0%}")
        for d in err:
            print(f"    ERR {d['song']}: decoded {d['label']} "
                  f"(q5={QUAL5[d['q5']]}) p_max={d['p_max']:.2f} "
                  f"H={d['H'] if d['H'] is None else round(d['H'], 2)} "
                  f"ctx_ok={d['ctx_ok']}")


if __name__ == "__main__":
    main()
