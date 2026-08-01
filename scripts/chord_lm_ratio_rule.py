"""The intervention rule: LM confidence vs pipeline confidence. (Louis, 2026-08-01)

Louis's framing: the pipeline's confidence does not have to be *calibrated*, it
only has to be *meaningful* — monotone enough that low means "probably wrong".
Then the decision is a ratio between the two confidences, not a threshold on
either alone.

That poses a question this script answers: **how meaningful does the pipeline's
confidence have to be before the ratio rule beats using the LM's confidence
alone?** The answer is a spec for the confidence work — a number to build toward
rather than "make it better".

Method. We have real ground truth for the LM side (held-out iReal charts) but
none for the pipeline side (see the audit: data/real_audio_benchmark is
verified=false). So the pipeline is simulated at a controlled quality:

  * a fraction of slots is corrupted, standing in for pipeline errors;
  * each slot gets a "pipeline confidence" drawn from a latent Gaussian whose
    separation is set to hit a TARGET AUC. AUC 0.5 = the score carries no
    information; AUC 0.9 = it separates right from wrong very well. This is a
    monotone, deliberately UNCALIBRATED score, exactly the kind Louis described.

Rules compared, thresholds tuned on VAL songs and reported on TEST:

  lm-only     override when  LM_conf >= t
  ratio       override when  log LM_conf - log pipe_conf >= t
  two-sided   override when  LM_conf >= t1 AND pipe_conf <= t2

Also reported: top-k containment — is the truth in the LM's top 3 even when its
top 1 is wrong? That decides whether the LM should propose a *shortlist* to the
UI instead of a single override.

    .venv/bin/python scripts/chord_lm_ratio_rule.py
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmonia_min.chord_lm import corpus, data, model as M, vocab
from chord_lm_intervention import auc, cloze, corrupt


def simulated_pipeline_conf(correct: np.ndarray, target_auc: float,
                            rng: np.random.Generator) -> np.ndarray:
    """A monotone, uncalibrated confidence with a prescribed discriminative power.

    Latent Gaussian: correct slots ~ N(d,1), wrong ~ N(0,1). For that model
    AUC = Phi(d/sqrt2), so d = sqrt2 * Phi^-1(AUC). Squashed through a logistic
    into (0,1) — the squash is monotone, so it changes the calibration but not
    the AUC, which is the whole point of the exercise.
    """
    from statistics import NormalDist
    d = math.sqrt(2) * NormalDist().inv_cdf(min(max(target_auc, 0.5001), 0.9999))
    z = rng.standard_normal(len(correct)) + d * correct
    return 1.0 / (1.0 + np.exp(-z))


def evaluate_rule(shown, truth, lm_pred, lm_conf, pipe_conf, mask) -> tuple[int, int]:
    """(chords fixed, chords broken) for the slots where `mask` says override."""
    act = mask & (lm_pred != shown)
    fixed = int((act & (lm_pred == truth) & (shown != truth)).sum())
    broke = int((act & (lm_pred != truth) & (shown == truth)).sum())
    return fixed, broke


def collect(net, charts, device, rate, seed, auc_target, rng_np):
    rng = random.Random(seed)
    S, T, P, C, K, Q = [], [], [], [], [], []
    for s in charts:
        cs, _bad = (list(s), None) if rate == 0 else corrupt(s, rate, rng, True)
        p = cloze(net, cs, device)
        order = np.argsort(-p, axis=1)
        S.append(np.array(cs)); T.append(np.array(s))
        P.append(order[:, 0]); C.append(p.max(1)); K.append(order[:, :5])
        Q.append(p)
    S, T, P, C = map(np.concatenate, (S, T, P, C))
    K = np.concatenate(K, 0)
    pipe = simulated_pipeline_conf((S == T).astype(float), auc_target, rng_np)
    return S, T, P, C, K, pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chord_lm_rope_masked")
    ap.add_argument("--rate", type=float, default=0.20,
                    help="simulated pipeline error rate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = M.ChordLM.from_checkpoint(Path("data/models") / f"{args.model}.pt", device)

    splits = corpus.load_splits(slots_per_bar=2)
    va = [c for ch in splits.val for c in data.chunk(ch, 256) if len(c) >= 16]
    te = [c for ch in splits.test for c in data.chunk(ch, 256) if len(c) >= 16]
    print(f"model {args.model} | val {len(va)} charts | test {len(te)} charts")
    print(f"simulated pipeline error rate: {args.rate:.0%}\n")

    # ── top-k containment, clean context and noisy context ──────────────────
    print("=" * 78)
    print("TOP-K — is the truth in the LM's shortlist even when its top-1 misses?")
    print("=" * 78)
    for label, rate in (("clean context", 0.0), (f"{args.rate:.0%} corrupted context",
                                                 args.rate)):
        rng_np = np.random.default_rng(args.seed)
        S, T, P, C, K, _ = collect(net, te, device, rate, args.seed, 0.8, rng_np)
        print(f"\n  {label}  ({len(T)} slots)")
        for k in (1, 2, 3, 5):
            hit = (K[:, :k] == T[:, None]).any(1).mean()
            print(f"    truth in top-{k}: {100*hit:6.2f}%")
        # the useful conditional: when top-1 is wrong, how often is top-3 right?
        wrong1 = K[:, 0] != T
        if wrong1.any():
            rescue = (K[wrong1][:, :3] == T[wrong1, None]).any(1).mean()
            print(f"    when top-1 is WRONG ({wrong1.mean()*100:.1f}% of slots), "
                  f"truth is still in top-3 {100*rescue:.2f}% of the time")

    # ── rules ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("THE RULE — how good must the pipeline's confidence be to be worth using?")
    print("=" * 78)
    print("  'AUC' = how well the pipeline's confidence separates its own right")
    print("  answers from its wrong ones. 0.5 = worthless, 0.9 = very good.")
    print("  Thresholds tuned on VAL, reported on TEST. net = fixed - broken.\n")
    print(f"  {'pipe AUC':>9} | {'lm-only':>22} | {'ratio':>22} | {'two-sided':>22}")
    print(f"  {'':>9} | {'t':>6} {'net':>7} {'prec':>7} | {'t':>6} {'net':>7} {'prec':>7}"
          f" | {'t1/t2':>6} {'net':>7} {'prec':>7}")
    print("  " + "-" * 76)

    for pauc in (0.50, 0.60, 0.70, 0.80, 0.90):
        rng_v = np.random.default_rng(args.seed + 1)
        rng_t = np.random.default_rng(args.seed + 2)
        Sv, Tv, Pv, Cv, Kv, PIv = collect(net, va, device, args.rate, args.seed + 10,
                                          pauc, rng_v)
        St, Tt, Pt, Ct, Kt, PIt = collect(net, te, device, args.rate, args.seed + 20,
                                          pauc, rng_t)

        def best(make_mask, grid):
            best_t, best_net = None, -10**9
            for t in grid:
                f, b = evaluate_rule(Sv, Tv, Pv, Cv, PIv, make_mask(Cv, PIv, t))
                if f - b > best_net:
                    best_t, best_net = t, f - b
            return best_t

        def report(make_mask, t):
            f, b = evaluate_rule(St, Tt, Pt, Ct, PIt, make_mask(Ct, PIt, t))
            prec = f / max(f + b, 1)
            return f - b, prec

        g1 = np.arange(0.50, 1.00, 0.01)
        t_lm = best(lambda c, p, t: c >= t, g1)
        net_lm, pr_lm = report(lambda c, p, t: c >= t, t_lm)

        g2 = np.arange(-1.0, 5.0, 0.1)
        ratio = lambda c, p, t: (np.log(np.maximum(c, 1e-9))
                                 - np.log(np.maximum(p, 1e-9))) >= t
        t_r = best(ratio, g2)
        net_r, pr_r = report(ratio, t_r)

        g3 = [(a, b) for a in np.arange(0.5, 1.0, 0.05) for b in np.arange(0.1, 0.95, 0.05)]
        two = lambda c, p, t: (c >= t[0]) & (p <= t[1])
        t2 = best(two, g3)
        net_2, pr_2 = report(two, t2)

        print(f"  {pauc:>9.2f} | {t_lm:>6.2f} {net_lm:>+7d} {pr_lm:>7.2f}"
              f" | {t_r:>6.1f} {net_r:>+7d} {pr_r:>7.2f}"
              f" | {t2[0]:.2f}/{t2[1]:.2f} {net_2:>+7d} {pr_2:>7.2f}")

    print("\n  prec = of the chords we changed, the fraction that were improvements.")


if __name__ == "__main__":
    main()
