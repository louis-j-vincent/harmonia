"""Level 1: does fusing the bass posterior into the LM actually pay?

Real chain, no simulation: GuitarSet audio -> harmonia_min.pipeline.analyze ->
the LM's cloze distribution -> fused with the musx bass plane -> scored against
GuitarSet's verified ground truth (audio and GT ship together, so the bar grid
comes from beat tracking and the labels from a human, and nothing is circular).

Three rows, and the third is the one that decides whether this is worth
building on:

    LM alone        the grammar guessing a root it cannot hear
    bass alone      the acoustic evidence with no grammar at all
    LM + bass       the fusion

If "bass alone" matches "LM + bass", the LM is contributing nothing and levels
2-3 of the design rest on a false premise. That control is the point of the
script.

    .venv/bin/python scripts/chord_lm_bass_fusion_eval.py --max 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chord_conf_auc_guitarset import load_gt                    # noqa: E402
from chord_lm_prod_eval import ANN, AUD, SLOTS, gt_tokens_on_grid  # noqa: E402
from harmonia_min import musx as _musx                          # noqa: E402
from harmonia_min import pipeline as minpipe                    # noqa: E402
from harmonia_min.chord_lm import bass as bassmod, model as M, vocab  # noqa: E402
from harmonia_min.chord_lm.from_app import app_chart_to_grid    # noqa: E402
from harmonia_min.chord_lm.grid import expand_repeats           # noqa: E402
from harmonia_min.chord_lm.intervene import cloze_probs         # noqa: E402


def collect(stems, net, device):
    recs = []
    for i, jp in enumerate(stems, 1):
        stem = jp.stem
        hits = list(AUD.glob(f"{stem}*.wav"))
        if not hits:
            continue
        gt = load_gt(jp)
        if not gt:
            continue
        try:
            chart = minpipe.analyze(hits[0], title=stem, file_key=stem)
        except Exception as e:
            print(f"  {stem}: FAILED {type(e).__name__}", flush=True)
            continue
        g = app_chart_to_grid(chart, slots_per_bar=SLOTS)
        n = len(g.tokens)
        if n < 8:
            continue
        lm = np.log(np.maximum(cloze_probs(net, g.tokens, device), 1e-12))
        bplane = _musx.frame_posteriors(hits[0])[1]
        bslots = bassmod.slot_bass(bplane, chart.get("barGrid") or [], n, SLOTS)
        absolute = expand_repeats(g.tokens)
        truth = gt_tokens_on_grid(gt, chart, n)
        recs.append({"song": stem, "tokens": g.tokens, "lm": lm,
                     "bass": bslots, "abs": absolute, "truth": truth})
        print(f"  [{i}/{len(stems)}] {stem}: {n} slots", flush=True)
    return recs


def score_quality_only(recs, weight):
    """Root FROZEN to what the chart wrote; the LM picks only the family.

    Two independent measurements say this is the right shape. The harmonic-prior
    session measured 44/44 of Blue Bossa's true errors keeping the root and only
    changing the quality, and their root-free corrector scored 0% top-1. This
    session measured 85% of the LM's disagreements changing the root, and the
    bass plane contradicting it there on Louis's ear-verified cases.

    So the question is not "which chord" but "which quality on this root" — and
    unlike the scale-snap, the LM needs no key: it learned the grammar from
    2,172 charts, which is exactly the step (the Db bridge) their tonic tracker
    misses.
    """
    ok = tot = changed = 0
    for r in recs:
        fused = bassmod.fuse(r["lm"], r["bass"], r["tokens"], r["abs"], weight)
        for k in range(len(r["tokens"])):
            t, shown = r["truth"][k], r["abs"][k]
            if t is None or shown is None or not vocab.is_chord(shown):
                continue
            root = vocab.split_chord(shown)[0]
            cands = [vocab.chord_id(root, f) for f in vocab.FAMILIES]
            best = max(cands, key=lambda c: fused[k, c])
            tot += 1
            ok += int(best == t)
            changed += int(best != shown)
    return ok / max(tot, 1), changed, tot


def score(recs, weight):
    """Top-1 and top-3 accuracy of the fused distribution, on ABSOLUTE chords."""
    ok1 = ok3 = tot = 0
    for r in recs:
        fused = bassmod.fuse(r["lm"], r["bass"], r["tokens"], r["abs"], weight)
        order = np.argsort(-fused, axis=1)
        for k in range(len(r["tokens"])):
            t = r["truth"][k]
            if t is None or r["abs"][k] is None:
                continue
            tot += 1
            cand = []
            for j in range(4):
                tok = int(order[k, j])
                a = r["abs"][k - 1] if (tok == vocab.REP and k > 0) else tok
                if a is not None and vocab.is_chord(a) and a not in cand:
                    cand.append(a)
            if cand and cand[0] == t:
                ok1 += 1
            if t in cand[:3]:
                ok3 += 1
    return ok1 / max(tot, 1), ok3 / max(tot, 1), tot


def score_bass_only(recs):
    ok1 = tot = 0
    for r in recs:
        for k in range(len(r["tokens"])):
            t = r["truth"][k]
            if t is None or r["abs"][k] is None:
                continue
            tot += 1
            lp = bassmod.token_logp(r["bass"][k])
            if int(np.argmax(lp[:vocab.N_CHORD])) == t:
                ok1 += 1
    return ok1 / max(tot, 1), tot


def score_pipeline(recs):
    ok = tot = 0
    for r in recs:
        for k in range(len(r["tokens"])):
            t = r["truth"][k]
            if t is None or r["abs"][k] is None:
                continue
            tot += 1
            ok += int(r["abs"][k] == t)
    return ok / max(tot, 1), tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--model", default="chord_lm_rope_masked")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = M.ChordLM.from_checkpoint(Path("data/models") / f"{args.model}.pt", device)

    jamses = sorted(ANN.glob("*_comp.jams"))
    step = max(1, len(jamses) // args.max)
    jamses = jamses[::step][:args.max]
    print(f"{len(jamses)} comping excerpts\n")
    t0 = time.time()
    recs = collect(jamses, net, device)
    songs = sorted({r["song"] for r in recs})
    tune = set(songs[::2])
    tun = [r for r in recs if r["song"] in tune]
    held = [r for r in recs if r["song"] not in tune]
    print(f"\n{len(recs)} excerpts ({time.time()-t0:.0f}s) — "
          f"tune {len(tun)}, held {len(held)}")

    pa, n = score_pipeline(held)
    ba, _ = score_bass_only(held)
    print("\n" + "=" * 62)
    print(f"HELD-OUT — {n} half-bar slots")
    print("=" * 62)
    print(f"  {'pipeline chord (what we ship)':<34} {pa:.4f}")
    print(f"  {'bass alone, no grammar':<34} {ba:.4f}")

    # Selection uses TUNE only; the held column is printed for transparency,
    # never to choose. Rule: the SMALLEST w within one standard error of the
    # best tune score. Plain argmax picked w=12, the grid's edge, on a curve
    # that creeps up monotonically on tune while held-out is already falling
    # (0.5412 at w=3 -> 0.5354 at w=12, top-3 0.681 -> 0.652). A monotone
    # tuning curve and a bounded grid is how you select a boundary artifact.
    GRID = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
    print(f"\n  {'weight w':>9} {'tune top-1':>11} {'tune top-3':>11}"
          f" | {'held top-1':>11} {'held top-3':>11}")
    tune_scores = {}
    for w in GRID:
        a1, a3, nt = score(tun, w)
        h1, h3, _ = score(held, w)
        tune_scores[w] = (a1, nt)
        print(f"  {w:>9.2f} {a1:>11.4f} {a3:>11.4f} | {h1:>11.4f} {h3:>11.4f}")
    best_w = max(tune_scores, key=lambda k: tune_scores[k][0])
    b1, nt = tune_scores[best_w]
    se = (b1 * (1 - b1) / max(nt, 1)) ** 0.5
    within = [w for w in GRID if tune_scores[w][0] >= b1 - se]
    best_w = min(within)
    print(f"\n  best tune score {b1:.4f} at w={max(tune_scores, key=lambda k: tune_scores[k][0])};"
          f" 1 SE = {se:.4f}")
    print(f"  -> smallest w within 1 SE: w = {best_w}")
    a1_0, a3_0, _ = score(held, 0.0)
    a1_w, a3_w, _ = score(held, best_w)
    print("\n  HELD-OUT:")
    print(f"    LM alone (w=0)     top-1 {a1_0:.4f}   top-3 {a3_0:.4f}")
    print(f"    LM + bass (w={best_w})  top-1 {a1_w:.4f}   top-3 {a3_w:.4f}"
          f"   ({100*(a1_w-a1_0):+.2f} pp)")
    print(f"    bass alone         top-1 {ba:.4f}")
    verdict = ("the LM ADDS to the bass" if a1_w > ba + 0.005 else
               "the LM adds NOTHING over the bass alone")
    print(f"\n  -> {verdict}")

    # The product question: with the two-sided gate, does the fusion change
    # more chords for the better than for the worse?
    print("\n" + "=" * 62)
    print("THE GATE — net chords fixed minus broken, held-out")
    print("=" * 62)
    print(f"  {'rule':<26} {'fixed':>6} {'broke':>6} {'net':>6} {'prec':>6} {'acted':>6}")
    for label, w in (("LM alone", 0.0), (f"LM + bass (w={best_w})", best_w)):
        fixed = broke = acted = 0
        for r in held:
            fused = bassmod.fuse(r["lm"], r["bass"], r["tokens"], r["abs"], w)
            pr = np.exp(fused)
            for k in range(len(r["tokens"])):
                t, shown = r["truth"][k], r["abs"][k]
                if t is None or shown is None:
                    continue
                tok = int(np.argmax(fused[k]))
                prop = r["abs"][k - 1] if (tok == vocab.REP and k > 0) else tok
                if prop is None or not vocab.is_chord(prop) or prop == shown:
                    continue
                if float(pr[k, tok]) < 0.80:
                    continue
                acted += 1
                fixed += int(prop == t and shown != t)
                broke += int(prop != t and shown == t)
        prec = fixed / max(fixed + broke, 1)
        print(f"  {label:<26} {fixed:>6} {broke:>6} {fixed-broke:>+6} {prec:>6.2f} {acted:>6}")

    # ── root frozen: the LM chooses only the quality ────────────────────────
    print("\n" + "=" * 62)
    print("ROOT FROZEN — the LM picks only the family (held-out)")
    print("=" * 62)
    base_ok = base_tot = 0
    for r in held:
        for k in range(len(r["tokens"])):
            t, shown = r["truth"][k], r["abs"][k]
            if t is None or shown is None or not vocab.is_chord(shown):
                continue
            base_tot += 1
            base_ok += int(shown == t)
    print(f"  {'chart as shipped':<34} {base_ok/max(base_tot,1):.4f}  ({base_tot} slots)")
    for w in (0.0, best_w):
        a, ch, n = score_quality_only(held, w)
        tag = "LM alone" if w == 0 else f"LM + bass (w={w})"
        print(f"  {'root frozen, ' + tag:<34} {a:.4f}  "
              f"({ch} qualities changed, {100*ch/max(n,1):.1f}%)")
    # and the ceiling: how much is even available on this axis?
    ceil_ok = 0
    for r in held:
        for k in range(len(r["tokens"])):
            t, shown = r["truth"][k], r["abs"][k]
            if t is None or shown is None or not vocab.is_chord(shown):
                continue
            ceil_ok += int(vocab.split_chord(shown)[0] == vocab.split_chord(t)[0])
    print(f"  {'ceiling (root already right)':<34} {ceil_ok/max(base_tot,1):.4f}"
          "   <- nothing on this axis can beat it")

    Path("docs/research_sessions").mkdir(parents=True, exist_ok=True)
    Path("docs/research_sessions/chord_lm_bass_fusion.json").write_text(json.dumps(
        {"n_slots": n, "pipeline": pa, "bass_only": ba, "best_w": best_w,
         "lm_only": {"top1": a1_0, "top3": a3_0},
         "lm_bass": {"top1": a1_w, "top3": a3_w}}, indent=1))


if __name__ == "__main__":
    main()
