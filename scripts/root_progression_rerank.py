#!/usr/bin/env python3
"""Root lever build (premise-check): a functional root-MOTION prior + Viterbi,
re-ranking musx's per-span root posteriors on jazz.

The jazz root ceiling is fifth/fourth ambiguity; musx's top-2 contains the true
root 15.7% of the time (ceiling 0.76 vs 0.61 baseline) but no LOCAL signal
disambiguates. This tries the SEQUENCE signal: jazz root motion is dominated by
Δ=+5 (up a fourth, the ii-V-I cycle: 44% of transitions in iReal jazz1460), so
the fifth-away decoy (Δ=+7) breaks the progression. A Viterbi over musx root
posteriors with a transposition-invariant root-motion transition prior (learned
from 1460 iReal jazz charts) should pick the functionally-correct root.

  emission[t][r]  = log(musx root-marginal over span t)
  transition[i][j]= log P(Δ=(j-i) mod 12)  from iReal jazz roots
  score[t][j]     = emission[t][j] + max_i(score[t-1][i] + lam * transition[i][j])

Measured on CLEAN JAAH (oracle boundaries, to isolate root re-ranking from
segmentation). Sweeps lam; reports vs baseline (argmax musx) and the top-2
ceiling. Read-only, cached probs.

Usage:
    .venv/bin/python scripts/root_progression_rerank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.musx_redecode import frame_posteriors, FRAME_DT  # noqa: E402
from scripts.build_jaah_corpus import parse_jaah, load_lab, LABS_DIR   # noqa: E402
from harmonia.data.ireal_corpus import (                               # noqa: E402
    load_playlist, sectionized_measures, split_chords, to_mma_chord,
    chord_root_pc,
)

RS = REPO / "docs" / "research_sessions"
JAAH_AUD = REPO / "data" / "cache" / "jaah" / "audio"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"
MUSX_LAT = 0.113
IREAL = REPO / "data" / "ireal"


def motion_prior(playlists=("jazz1460.txt",), smooth=0.5):
    """P(Δroot mod 12) from iReal charts (transposition-invariant)."""
    counts = np.full(12, smooth)
    for pl in playlists:
        for t in load_playlist(IREAL / pl):
            roots = []
            for _, meas in sectionized_measures(t):
                for tok in split_chords(meas):
                    mma = to_mma_chord(tok)
                    if mma is None:
                        continue
                    pc = chord_root_pc(mma)
                    if pc is not None:
                        roots.append(pc)
            for a, b in zip(roots[:-1], roots[1:]):
                counts[(b - a) % 12] += 1
    return counts / counts.sum()


def span_marginals(slug, vid):
    """(list of (12,) musx root-marginals over oracle GT spans, list of gt roots),
    or None if wrong-take (oracle root < 0.35)."""
    triad = frame_posteriors(JAAH_AUD / f"{vid}.wav")[0]
    am = triad.argmax(1); nf = len(am)
    rows = load_lab(LABS_DIR / f"{slug}.lab")
    marg, gts = [], []
    dur = rc = 0.0
    for t0, t1, lab in rows:
        gr, gf, _ = parse_jaah(lab.split("/")[0])
        if gr is None:
            continue
        a = max(0, int(round((t0 + MUSX_LAT) / FRAME_DT)))
        b = min(nf, int(round((t1 + MUSX_LAT) / FRAME_DT)))
        if b <= a:
            continue
        m = triad[a:b].mean(0)
        r = np.zeros(12)
        for i in range(1, 73):
            r[(i - 1) % 12] += m[i]
        if r.sum() <= 0:
            continue
        marg.append(r / r.sum())
        gts.append(gr)
        dur += (t1 - t0); rc += (t1 - t0) * (int(r.argmax()) == gr)
    if dur == 0 or rc / dur < 0.35:
        return None
    return marg, gts


def viterbi(marg, logT, lam):
    """Best root sequence. marg: list of (12,) emissions; logT: (12,12) log
    transition (row=prev, col=next)."""
    n = len(marg)
    em = np.log(np.stack(marg) + 1e-9)          # (n,12)
    score = em[0].copy()
    back = np.zeros((n, 12), dtype=int)
    for t in range(1, n):
        prev = score[:, None] + lam * logT       # (12 prev, 12 next)
        best_prev = prev.argmax(0)
        score = em[t] + prev[best_prev, np.arange(12)]
        back[t] = best_prev
    path = np.zeros(n, dtype=int)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def main():
    prior = motion_prior()
    logT = np.log(prior)[None, :].repeat(12, 0)   # T[i][j] depends on (j-i)
    T = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            T[i, j] = np.log(prior[(j - i) % 12])
    print("root-motion prior (top): " +
          ", ".join(f"Δ{d}={100*prior[d]:.0f}%" for d in np.argsort(-prior)[:4]))

    pins = json.loads((RS / "jaah_source_pins.json").read_text())
    scores = json.loads((RS / "jaah_benchmark_scores.json").read_text())["scores"]
    songs = []
    for s in scores:
        vid = pins.get(s["slug"])
        if not vid or not (PROB_CACHE / f"{vid}.npz").exists():
            continue
        r = span_marginals(s["slug"], vid)
        if r:
            songs.append(r)
    print(f"clean JAAH songs: {len(songs)}\n")

    def acc(lam):
        tot = ok = 0
        for marg, gts in songs:
            path = viterbi(marg, T, lam)
            for p, g in zip(path, gts):
                tot += 1; ok += (p == g)
        return ok / tot

    base = sum(int(m.argmax()) == g for marg, gts in songs
               for m, g in zip(marg, gts)) / \
        sum(len(gts) for _, gts in songs)
    ceil = 0.763
    print(f"  baseline (argmax musx)   {base:.3f}")
    print("  Viterbi re-rank by lam:")
    best = (0.0, base)
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        a = acc(lam)
        mark = "  <-- best" if a > best[1] else ""
        if a > best[1]:
            best = (lam, a)
        print(f"    lam={lam:<4} root {a:.3f}  ({a-base:+.3f}){mark}")
    print(f"\n  best: lam={best[0]} -> {best[1]:.3f}  "
          f"(baseline {base:.3f}, top-2 ceiling {ceil:.3f}; "
          f"captured {100*(best[1]-base)/(ceil-base):.0f}% of headroom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
