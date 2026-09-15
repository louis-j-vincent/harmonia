"""When should the chord LM overrule the pipeline? (Louis, 2026-08-01)

The plan: let the LM step in where we are unsure of a chord. That needs two
numbers nobody has measured yet — how trustworthy the LM's OWN confidence is,
and how well it holds up when its context is the pipeline's noisy output rather
than a clean chart.

Everything the LM has been scored on so far gave it a PERFECT chart minus one
slot. That is not the deployment condition. In deployment its neighbours are
whatever the pipeline emitted, which is wrong maybe 20-30% of the time. This
script measures the drop.

  A. CALIBRATION. Bin cloze predictions by the model's own max probability;
     within each bin, how often is it actually right? A usable gate needs
     "p >= 0.9" to mean something close to 90%.

  B. NOISY CONTEXT. Corrupt a fraction of slots to simulate pipeline errors,
     then ask the LM to fill each slot with the corrupted chart as context.
     Three quantities matter, and only one of them is "accuracy":
       repair   — at CORRUPTED slots, does the LM propose the true chord?
       damage   — at CORRECT slots, how often does it confidently propose a
                  change? Every one of those is a regression if we act on it.
       detect   — can confidence separate the two?

  C. THE RULE. Sweep the confidence threshold and report net chords fixed minus
     chords broken. A threshold is only worth shipping where that number is
     positive with margin.

Corruption is a SIMULATION, not the pipeline's real error distribution — we have
no verified audio-aligned ground truth to fit that from (see the audit note in
docs/). Two models are used: a realistic one (confusions the pipeline actually
tends to make: maj<->dom, relative minor, fifths, dropped/added changes) and a
uniform-random one as a worst case.

    .venv/bin/python scripts/chord_lm_intervention.py --model chord_lm_rope_masked
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import corpus, data, model as M, vocab


# ── corruption models ───────────────────────────────────────────────────────
def _realistic_corrupt(tok: int, rng: random.Random) -> int:
    """One plausible pipeline error on a token.

    Weighted toward the confusions the project keeps hitting: major/dominant
    family slips, relative-minor and fifth root slips, and spurious or missing
    chord changes (REP <-> chord).
    """
    if tok == vocab.REP:
        return vocab.REP          # handled by the caller (needs a real chord)
    if not vocab.is_chord(tok):
        return tok
    root, fam = vocab.split_chord(tok)
    r = rng.random()
    if r < 0.35:                                     # family slip
        alt = {"maj": "dom", "dom": "maj", "min": "hdim", "hdim": "min",
               "dim": "hdim", "sus": "maj", "aug": "maj"}[fam]
        return vocab.chord_id(root, alt)
    if r < 0.60:                                     # relative major/minor
        return vocab.chord_id(root + (9 if fam == "maj" else 3),
                              "min" if fam == "maj" else "maj")
    if r < 0.85:                                     # up or down a fifth
        return vocab.chord_id(root + rng.choice((5, 7)), fam)
    return vocab.chord_id(root + rng.choice((1, -1, 2, -2)), fam)  # neighbour root


def corrupt(seq: list[int], rate: float, rng: random.Random,
            realistic: bool = True) -> tuple[list[int], np.ndarray]:
    """Return (corrupted sequence, boolean mask of which slots were changed)."""
    out = list(seq)
    bad = np.zeros(len(seq), dtype=bool)
    for i in range(len(seq)):
        if rng.random() >= rate:
            continue
        old = out[i]
        if old == vocab.REP:
            # a spurious CHANGE where the chart holds: invent a chord
            new = vocab.chord_id(rng.randrange(12), rng.choice(vocab.FAMILIES))
        elif not vocab.is_chord(old):
            continue
        elif rng.random() < 0.25:
            new = vocab.REP if i > 0 else old     # a MISSED change
        else:
            new = (_realistic_corrupt(old, rng) if realistic
                   else vocab.chord_id(rng.randrange(12),
                                       rng.choice(vocab.FAMILIES)))
        if new != old:
            out[i] = new
            bad[i] = True
    return out, bad


# ── scoring ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def cloze(net: M.ChordLM, seq: list[int], device: str, batch: int = 128
          ) -> np.ndarray:
    """(T, V) probabilities, each slot predicted with itself masked."""
    T = len(seq)
    base = torch.tensor(seq, dtype=torch.long)
    out = np.zeros((T, vocab.VOCAB_SIZE), dtype=np.float32)
    for s0 in range(0, T, batch):
        pos = list(range(s0, min(s0 + batch, T)))
        x = base.unsqueeze(0).repeat(len(pos), 1).clone()
        for r, p in enumerate(pos):
            x[r, p] = vocab.MASK
        x = x.to(device)
        slot_ix = (torch.arange(T, device=device) % net.cfg.slots_per_bar
                   ).expand(len(pos), T)
        lp = F.log_softmax(net(x, slot_ix), -1)
        rows = torch.arange(len(pos), device=device)
        out[s0:s0 + len(pos)] = lp[rows, torch.tensor(pos, device=device)
                                   ].exp().cpu().numpy()
    return out


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
        n = int(sel.sum())
        if n == 0:
            continue
        acc, mean_c = float(correct[sel].mean()), float(conf[sel].mean())
        rows.append((lo, hi, n, mean_c, acc))
        ece += n / len(conf) * abs(acc - mean_c)
    return rows, ece


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a correct slot > score of an incorrect one)."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


THRESHOLDS = (0.99, 0.95, 0.9, 0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chord_lm_rope_masked")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.99, 0.95, 0.9, 0.8])
    args = ap.parse_args()
    global THRESHOLDS
    THRESHOLDS = tuple(args.thresholds)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = M.ChordLM.from_checkpoint(Path("data/models") / f"{args.model}.pt", device)
    print(f"model {args.model}  (rope={net.cfg.rope}, causal={net.cfg.causal})\n")

    splits = corpus.load_splits(slots_per_bar=2)
    te = [c for ch in splits.test for c in data.chunk(ch, 256) if len(c) >= 16]
    print(f"{len(te)} held-out test charts\n")

    # ── A. calibration on CLEAN context ─────────────────────────────────────
    print("=" * 78)
    print("A. IS THE LM'S CONFIDENCE MEANINGFUL?  (clean context — best case)")
    print("=" * 78)
    conf, correct = [], []
    probs_clean = []
    for s in te:
        p = cloze(net, s, device)
        probs_clean.append(p)
        pred = p.argmax(1)
        conf.append(p.max(1))
        correct.append(pred == np.array(s))
    conf = np.concatenate(conf); correct = np.concatenate(correct)
    rows, ece = reliability(conf, correct)
    print(f"  {'confidence bin':<16} {'n':>7} {'mean conf':>10} {'accuracy':>10}")
    for lo, hi, n, mc, acc in rows:
        print(f"  {f'{lo:.1f}-{hi:.1f}':<16} {n:>7} {mc:>10.3f} {acc:>10.3f}")
    print(f"\n  ECE {ece:.4f}   AUC(confidence separates right from wrong) "
          f"{auc(conf, correct):.4f}")
    print(f"  overall accuracy {correct.mean():.4f}")

    # ── B/C. noisy context ──────────────────────────────────────────────────
    for realistic in (True, False):
        kind = "realistic pipeline-like errors" if realistic else "uniform-random errors"
        print("\n" + "=" * 78)
        print(f"B. LM UNDER NOISY CONTEXT — {kind}")
        print("=" * 78)
        print(f"  {'corrupt':>8} {'repair':>8} {'damage':>8} {'AUC':>7}   "
              + "".join(f"{'net@'+str(t):>9}" for t in THRESHOLDS))
        for rate in args.rates:
            rng = random.Random(args.seed)
            rep_ok = rep_n = 0
            dmg = ok_n = 0
            sc, lab = [], []
            fixed = {t: 0 for t in THRESHOLDS}
            broke = {t: 0 for t in THRESHOLDS}
            for s in te:
                cs, bad = (list(s), np.zeros(len(s), bool)) if rate == 0 \
                    else corrupt(s, rate, rng, realistic)
                p = cloze(net, cs, device)
                pred = p.argmax(1)
                cf = p.max(1)
                truth = np.array(s)
                shown = np.array(cs)
                # repair: at corrupted slots, does the LM name the TRUE chord?
                rep_ok += int((pred[bad] == truth[bad]).sum()); rep_n += int(bad.sum())
                good = ~bad
                # damage: at correct slots, does it propose something else?
                dmg += int((pred[good] != truth[good]).sum()); ok_n += int(good.sum())
                sc.append(cf); lab.append(pred == truth)
                for t in fixed:
                    act = (cf >= t) & (pred != shown)     # we would override here
                    fixed[t] += int((act & (pred == truth) & (shown != truth)).sum())
                    broke[t] += int((act & (pred != truth) & (shown == truth)).sum())
            sc = np.concatenate(sc); lab = np.concatenate(lab)
            print(f"  {rate:>8.0%} {rep_ok/max(rep_n,1):>8.3f} {dmg/max(ok_n,1):>8.3f} "
                  f"{auc(sc, lab):>7.3f}   "
                  + "".join(f"{fixed[t]-broke[t]:>+9d}" for t in THRESHOLDS))
        print("  repair = true chord proposed at a corrupted slot")
        print("  damage = something else proposed at an already-correct slot")
        print("  net@T  = (chords fixed) - (chords broken) if we override "
              "wherever LM conf >= T and it disagrees")


if __name__ == "__main__":
    main()
