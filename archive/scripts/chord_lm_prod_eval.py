"""The intervention rule, measured for real — no simulation anywhere.

Everything up to now used a SIMULATED pipeline (corrupted iReal charts, a
synthetic confidence at a target AUC). This runs the real thing end to end:

    real audio  ->  harmonia_min.pipeline.analyze  ->  real chords, real `c`
                ->  the chord LM's cloze proposal at every half-bar
                ->  scored against GuitarSet's verified ground truth

GuitarSet is the only corpus on this disk whose audio and expert chord GT ship
together, so the bar grid comes from beat tracking and the GT from human
annotation — nothing is aligned using the model's own output, and the
measurement is not circular.

Thresholds are fitted on a TUNE half of the excerpts and reported on the HELD
half, so the numbers are not the thresholds' own training score.

    .venv/bin/python scripts/chord_lm_prod_eval.py --max 60
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

from build_jaah_corpus import parse_jaah                         # noqa: E402
from chord_conf_auc_guitarset import load_gt, gt_at              # noqa: E402
from harmonia_min import pipeline as minpipe                     # noqa: E402
from harmonia_min.chord_lm import model as M, vocab              # noqa: E402
from harmonia_min.chord_lm.from_app import app_chart_to_grid     # noqa: E402
from harmonia_min.chord_lm.grid import expand_repeats            # noqa: E402
from harmonia_min.chord_lm.intervene import _slot_confidence, cloze_probs  # noqa: E402

ANN = Path("data/cache/guitarset/annotation")
AUD = Path("data/cache/guitarset/audio")
SLOTS = 2


def gt_tokens_on_grid(gt, chart, n_slots: int) -> list[int | None]:
    """GT chord at the START of each half-bar of the PIPELINE's own bar grid.

    Sampling GT on the pipeline's grid is what makes shown/truth comparable
    slot-by-slot. The grid comes from beat tracking, not from chord labels, so
    this introduces no circularity.
    """
    bpb = int(chart.get("bpb", 4) or 4)
    grid = chart.get("barGrid") or []
    out: list[int | None] = []
    for i in range(n_slots):
        bar, slot = divmod(i, SLOTS)
        if bar + 1 >= len(grid):
            out.append(None)
            continue
        bw = grid[bar + 1] - grid[bar]
        t = grid[bar] + slot * bw / SLOTS
        root, fam = gt_at(gt, t + 1e-3)
        out.append(None if root is None or fam not in vocab.FAM_IDX
                   else vocab.chord_id(root, fam))
    return out


def collect(stems, net, device):
    """Per-slot (shown, truth, lm top-k, lm conf, pipe conf) over excerpts."""
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
        if len(g.tokens) < 8:
            continue
        pipe = _slot_confidence(chart, SLOTS)
        n = min(len(pipe), len(g.tokens))
        probs = cloze_probs(net, g.tokens[:n], device)
        order = np.argsort(-probs, axis=1)
        shown_abs = expand_repeats(g.tokens[:n])
        truth = gt_tokens_on_grid(gt, chart, n)
        for k in range(n):
            if truth[k] is None or shown_abs[k] is None:
                continue
            top = int(order[k, 0])
            prop_abs = shown_abs[k - 1] if (top == vocab.REP and k > 0) else top
            if not vocab.is_chord(prop_abs if prop_abs is not None else -1):
                continue
            recs.append({
                "song": stem, "slot": k,
                "shown": int(shown_abs[k]), "truth": int(truth[k]),
                "prop": int(prop_abs),
                "lm": float(probs[k, top]), "pipe": float(pipe[k]),
                "top3": [int(order[k, j]) for j in range(3)],
                "top3_abs": [int(shown_abs[k - 1]) if (int(order[k, j]) == vocab.REP
                                                       and k > 0)
                             else int(order[k, j]) for j in range(3)],
            })
        print(f"  [{i}/{len(stems)}] {stem}: {n} slots", flush=True)
    return recs


def score(recs, lm_min, pipe_max):
    fixed = broke = acted = 0
    for r in recs:
        if r["prop"] == r["shown"]:
            continue
        if r["lm"] < lm_min or r["pipe"] > pipe_max:
            continue
        acted += 1
        if r["prop"] == r["truth"] and r["shown"] != r["truth"]:
            fixed += 1
        elif r["prop"] != r["truth"] and r["shown"] == r["truth"]:
            broke += 1
    return fixed, broke, acted


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
    print(f"{len(jamses)} comping excerpts, model {args.model}\n")

    t0 = time.time()
    recs = collect(jamses, net, device)
    songs = sorted({r["song"] for r in recs})
    tune = set(songs[::2])                        # every other excerpt
    held = [r for r in recs if r["song"] not in tune]
    tun = [r for r in recs if r["song"] in tune]
    print(f"\n{len(recs)} scored slots over {len(songs)} excerpts "
          f"({time.time()-t0:.0f}s) — tune {len(tun)}, held {len(held)}")

    shown_ok = np.mean([r["shown"] == r["truth"] for r in recs])
    top1 = np.mean([r["prop"] == r["truth"] for r in recs])
    top3 = np.mean([r["truth"] in r["top3_abs"] for r in recs])
    print("\n" + "=" * 70)
    print("WHERE WE STAND (all slots)")
    print("=" * 70)
    print(f"  pipeline chord correct        {shown_ok:.4f}")
    print(f"  LM top-1 correct              {top1:.4f}")
    print(f"  truth in LM top-3             {top3:.4f}")

    print("\n" + "=" * 70)
    print("THE RULE — thresholds fitted on TUNE, reported on HELD")
    print("=" * 70)
    grid_lm = np.arange(0.50, 1.00, 0.05)
    grid_pipe = list(np.arange(0.30, 1.01, 0.05)) + [2.0]   # 2.0 = ignore pipe
    best, best_net = None, -10 ** 9
    for a in grid_lm:
        for b in grid_pipe:
            f, br, _ = score(tun, a, b)
            if f - br > best_net:
                best, best_net = (a, b), f - br
    lm_min, pipe_max = best

    rows = []
    f1, b1, a1 = score(held, lm_min, 2.0)              # LM-only, same LM gate
    f2, b2, a2 = score(held, lm_min, pipe_max)         # two-sided
    rows.append(("LM only (ignore pipeline `c`)", lm_min, 2.0, f1, b1, a1))
    rows.append(("two-sided (fitted)", lm_min, pipe_max, f2, b2, a2))

    print(f"  {'rule':<32} {'LM>=':>5} {'pipe<=':>7} {'fixed':>6} {'broke':>6}"
          f" {'net':>6} {'prec':>6} {'acted':>6}")
    for name, a, b, f, br, act in rows:
        prec = f / max(f + br, 1)
        bs = "any" if b > 1 else f"{b:.2f}"
        print(f"  {name:<32} {a:>5.2f} {bs:>7} {f:>6} {br:>6} {f-br:>+6} "
              f"{prec:>6.2f} {act:>6}")

    n_held_slots = len(held)
    print(f"\n  held-out: {n_held_slots} slots; the two-sided rule speaks on "
          f"{a2} of them ({100*a2/max(n_held_slots,1):.1f}%)")
    if f2 + b2:
        acc_before = np.mean([r["shown"] == r["truth"] for r in held])
        acc_after = acc_before + (f2 - b2) / n_held_slots
        print(f"  chord accuracy on held-out: {acc_before:.4f} -> {acc_after:.4f} "
              f"({100*(acc_after-acc_before):+.2f} pp)")

    Path("docs/research_sessions").mkdir(parents=True, exist_ok=True)
    Path("docs/research_sessions/chord_lm_prod_eval.json").write_text(json.dumps(
        {"lm_min": float(lm_min), "pipe_max": float(pipe_max),
         "n_slots": len(recs), "shown_acc": float(shown_ok),
         "lm_top1": float(top1), "lm_top3": float(top3),
         "held": {"fixed": f2, "broke": b2, "acted": a2, "n": n_held_slots},
         "lm_only": {"fixed": f1, "broke": b1, "acted": a1}}, indent=1))
    print("\nwrote docs/research_sessions/chord_lm_prod_eval.json")


if __name__ == "__main__":
    main()
