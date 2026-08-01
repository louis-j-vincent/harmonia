"""harmonia_min/chord_lm/evaluate.py — metrics + n-gram baselines.

Raw token accuracy on a metrical grid is a trap: 44.5% of half-bar slots are
REP, so "always say REP" already scores 44.5% while knowing no harmony at all.
Every number here is therefore split into the two questions the grid actually
poses:

    harmonic rhythm — does the chord change on this half-bar?
    identity        — given that it changes, to what?

`chord_acc_given_change` restricts the argmax to chord tokens, so it answers the
identity question alone. That is the number that transfers to the pipeline: it
is the ceiling on how much a chord grammar can rescue an acoustically ambiguous
slot.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .vocab import N_CHORD, NC, PAD, REP, VOCAB_SIZE, is_chord, split_chord


@dataclass
class Metrics:
    n: int = 0
    ppl: float = float("nan")
    token_acc: float = 0.0
    change_acc: float = 0.0          # binary REP vs change
    change_f1: float = 0.0           # F1 on the "a change happens" class
    chord_acc_at_change: float = 0.0     # unconditional, at change slots
    chord_acc_given_change: float = 0.0  # argmax restricted to chord tokens
    root_acc_given_change: float = 0.0
    family_acc_given_change: float = 0.0
    n_change: int = 0

    def table_row(self, name: str) -> str:
        return (f"{name:<26} {self.ppl:7.3f} {100*self.token_acc:7.2f} "
                f"{100*self.change_acc:7.2f} {100*self.change_f1:7.2f} "
                f"{100*self.chord_acc_at_change:7.2f} "
                f"{100*self.chord_acc_given_change:7.2f} "
                f"{100*self.root_acc_given_change:7.2f} "
                f"{100*self.family_acc_given_change:7.2f}")


HEADER = (f"{'model':<26} {'ppl':>7} {'tok%':>7} {'chg%':>7} {'chgF1':>7} "
          f"{'id%':>7} {'id|chg%':>7} {'root%':>7} {'fam%':>7}")


class MetricAccumulator:
    """Feeds on (log-probs, targets) chunks; PAD targets are ignored."""

    def __init__(self):
        self.nll = 0.0
        self.n = 0
        self.tok_ok = 0
        self.chg_tp = self.chg_fp = self.chg_fn = self.chg_ok = 0
        self.n_change = 0
        self.id_ok = self.id_cond_ok = self.root_ok = self.fam_ok = 0

    def add(self, logprobs: np.ndarray, targets: np.ndarray) -> None:
        """logprobs (N, V) already log-softmaxed; targets (N,)."""
        keep = targets != PAD
        logprobs, targets = logprobs[keep], targets[keep]
        if targets.size == 0:
            return
        n = targets.size
        self.n += n
        self.nll += float(-logprobs[np.arange(n), targets].sum())
        pred = logprobs.argmax(1)
        self.tok_ok += int((pred == targets).sum())

        tgt_change = targets != REP
        pred_change = pred != REP
        self.chg_ok += int((tgt_change == pred_change).sum())
        self.chg_tp += int((tgt_change & pred_change).sum())
        self.chg_fp += int((~tgt_change & pred_change).sum())
        self.chg_fn += int((tgt_change & ~pred_change).sum())

        # identity: only where the target is an actual chord (REP and NC excluded)
        sel = (targets < N_CHORD)
        if not sel.any():
            return
        t = targets[sel]
        self.n_change += int(sel.sum())
        self.id_ok += int((pred[sel] == t).sum())
        cond = logprobs[sel][:, :N_CHORD].argmax(1)
        self.id_cond_ok += int((cond == t).sum())
        self.root_ok += int(((cond // 7) == (t // 7)).sum())
        self.fam_ok += int(((cond % 7) == (t % 7)).sum())

    def result(self) -> Metrics:
        if self.n == 0:
            return Metrics()
        prec = self.chg_tp / max(self.chg_tp + self.chg_fp, 1)
        rec = self.chg_tp / max(self.chg_tp + self.chg_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        nc = max(self.n_change, 1)
        return Metrics(
            n=self.n,
            ppl=math.exp(self.nll / self.n),
            token_acc=self.tok_ok / self.n,
            change_acc=self.chg_ok / self.n,
            change_f1=f1,
            chord_acc_at_change=self.id_ok / nc,
            chord_acc_given_change=self.id_cond_ok / nc,
            root_acc_given_change=self.root_ok / nc,
            family_acc_given_change=self.fam_ok / nc,
            n_change=self.n_change,
        )


# ── n-gram baselines ────────────────────────────────────────────────────────
class NGramLM:
    """Interpolated n-gram over half-bar tokens, counted in all 12 keys.

    Counting every training sequence in all 12 transpositions is what makes this
    a *fair* baseline against a transposition-augmented transformer: both see
    the same key-invariant statistics, so any gap is about context length and
    not about one of them having memorised jazz's favourite keys.
    """

    def __init__(self, order: int = 3, alpha: float = 0.1):
        self.order = order
        self.alpha = alpha
        # sparse: context tuple -> {token: count}. A dense (V,) array per context
        # would be 648 MB at order 4 on this corpus; observed contexts are few.
        self.counts: list[dict[tuple, dict[int, float]]] = [{} for _ in range(order)]
        self.totals: list[dict[tuple, float]] = [{} for _ in range(order)]
        self.lambdas = np.array([1.0 / order] * order)
        self._uniform = np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE, dtype=np.float32)

    def fit(self, seqs: list[list[int]], *, transpositions: int = 12) -> "NGramLM":
        from .vocab import transpose
        for seq in seqs:
            for k in range(transpositions):
                s = [transpose(t, k) for t in seq] if k else seq
                for i, tok in enumerate(s):
                    for o in range(self.order):
                        if i < o:
                            continue
                        ctx = tuple(s[i - o:i])
                        d = self.counts[o].setdefault(ctx, {})
                        d[tok] = d.get(tok, 0.0) + 1.0
                        self.totals[o][ctx] = self.totals[o].get(ctx, 0.0) + 1.0
        return self

    def _dist(self, ctx: tuple, o: int) -> np.ndarray:
        key = ctx[len(ctx) - o:] if o else ()
        d = self.counts[o].get(key)
        if d is None:
            return self._uniform
        out = np.full(VOCAB_SIZE, self.alpha, dtype=np.float32)
        for t, c in d.items():
            out[t] += c
        return out / (self.totals[o][key] + self.alpha * VOCAB_SIZE)

    def _prob(self, ctx: tuple, o: int, tok: int) -> float:
        """Single-token probability without materialising the (V,) row."""
        key = ctx[len(ctx) - o:] if o else ()
        d = self.counts[o].get(key)
        if d is None:
            return 1.0 / VOCAB_SIZE
        return ((d.get(tok, 0.0) + self.alpha)
                / (self.totals[o][key] + self.alpha * VOCAB_SIZE))

    def logprobs(self, seq: list[int]) -> np.ndarray:
        """(len(seq), V) log P(token at i | tokens < i)."""
        out = np.zeros((len(seq), VOCAB_SIZE), dtype=np.float32)
        for i in range(len(seq)):
            ctx = tuple(seq[max(0, i - self.order + 1):i])
            p = np.zeros(VOCAB_SIZE, dtype=np.float32)
            for o in range(self.order):
                p += self.lambdas[o] * (self._uniform if i < o else self._dist(ctx, o))
            out[i] = np.log(np.maximum(p, 1e-12))
        return out

    def tune_lambdas(self, seqs: list[list[int]], n_iter: int = 30) -> "NGramLM":
        """EM on held-out sequences for the interpolation weights."""
        cache = [[(tuple(seq[max(0, i - self.order + 1):i]), tok)
                  for i, tok in enumerate(seq)] for seq in seqs]
        for _ in range(n_iter):
            num = np.zeros(self.order)
            for ex in cache:
                for i, (ctx, tok) in enumerate(ex):
                    comp = np.array([
                        (1.0 / VOCAB_SIZE) if i < o else self._prob(ctx, o, tok)
                        for o in range(self.order)])
                    w = self.lambdas * comp
                    tot = w.sum()
                    if tot > 0:
                        num += w / tot
            if num.sum() > 0:
                self.lambdas = num / num.sum()
        return self


class AlwaysRepeat:
    """The floor: every slot holds the previous chord."""

    @staticmethod
    def logprobs(seq: list[int]) -> np.ndarray:
        out = np.full((len(seq), VOCAB_SIZE), -20.0, dtype=np.float32)
        out[:, REP] = 0.0
        return np.log(np.exp(out) / np.exp(out).sum(1, keepdims=True))


class ClozeBigram:
    """Both-sided control for the masked model: P(slot | left neighbour, right).

    Without this, the masked transformer's cloze score has no honest reference —
    `AlwaysRepeat` only sees the left. Counted in all 12 keys like NGramLM, with
    backoff (prev, next) -> prev -> unigram.
    """

    def __init__(self, alpha: float = 0.1):
        self.pair: dict[tuple[int, int], dict[int, float]] = {}
        self.pair_tot: dict[tuple[int, int], float] = {}
        self.prev: dict[int, dict[int, float]] = {}
        self.prev_tot: dict[int, float] = {}
        self.uni = np.full(VOCAB_SIZE, 1.0, dtype=np.float64)
        self.alpha = alpha

    def fit(self, seqs: list[list[int]], *, transpositions: int = 12) -> "ClozeBigram":
        from .vocab import transpose
        for seq in seqs:
            for k in range(transpositions):
                s = [transpose(t, k) for t in seq] if k else seq
                for i, tok in enumerate(s):
                    self.uni[tok] += 1
                    if i == 0 or i == len(s) - 1:
                        continue
                    key = (s[i - 1], s[i + 1])
                    d = self.pair.setdefault(key, {})
                    d[tok] = d.get(tok, 0.0) + 1.0
                    self.pair_tot[key] = self.pair_tot.get(key, 0.0) + 1.0
                    dp = self.prev.setdefault(s[i - 1], {})
                    dp[tok] = dp.get(tok, 0.0) + 1.0
                    self.prev_tot[s[i - 1]] = self.prev_tot.get(s[i - 1], 0.0) + 1.0
        self.uni /= self.uni.sum()
        return self

    def logprobs_cloze(self, seq: list[int]) -> np.ndarray:
        """(len(seq), V): distribution for each slot, given its two neighbours."""
        out = np.zeros((len(seq), VOCAB_SIZE), dtype=np.float32)
        for i in range(len(seq)):
            p = None
            if 0 < i < len(seq) - 1:
                d = self.pair.get((seq[i - 1], seq[i + 1]))
                if d is not None and self.pair_tot[(seq[i - 1], seq[i + 1])] >= 12:
                    p = np.full(VOCAB_SIZE, self.alpha)
                    for t, c in d.items():
                        p[t] += c
                    p /= p.sum()
            if p is None and i > 0:
                d = self.prev.get(seq[i - 1])
                if d is not None:
                    p = np.full(VOCAB_SIZE, self.alpha)
                    for t, c in d.items():
                        p[t] += c
                    p /= p.sum()
            if p is None:
                p = self.uni
            out[i] = np.log(np.maximum(p, 1e-12))
        return out


class UnigramLM:
    def __init__(self):
        self.logp = np.full(VOCAB_SIZE, -math.log(VOCAB_SIZE), dtype=np.float32)

    def fit(self, seqs: list[list[int]]) -> "UnigramLM":
        c = np.ones(VOCAB_SIZE, dtype=np.float64) * 0.1
        for s in seqs:
            for t in s:
                c[t] += 1
        self.logp = np.log(c / c.sum()).astype(np.float32)
        return self

    def logprobs(self, seq: list[int]) -> np.ndarray:
        return np.tile(self.logp, (len(seq), 1))
