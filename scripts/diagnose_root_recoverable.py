#!/usr/bin/env python3
"""ROOT lever premise-check: is the GT root recoverable from musx's own posterior?

The jazz root ceiling is fifth/fourth ambiguity (rootless voicings, V<->I), not
bass. A harmonic-context prior (key/progression) could only help if the correct
root is already NEAR THE TOP of musx's frame posterior — a prior re-ranks the
model's own candidates, it can't conjure a root the model never considered.

So, on clean JAAH oracle-boundary spans, marginalise the 73-way triad posterior
to a 12-way ROOT distribution (sum over the 6 triad types), average over the
span, and find the RANK of the GT root:
  rank 1  -> musx already right.
  rank 2  -> recoverable by a light prior (the win a context model can capture).
  rank>=3 -> the frame model barely considered it; a prior won't save it.

top-2 recall = the root accuracy ceiling a perfect prior over musx's top-2 could
reach. If it's well above the current 0.62 oracle root, the context lever is
worth building; if not, we need external signal (a stronger model / the bass).

Read-only, cached probs.

Usage:
    .venv/bin/python scripts/diagnose_root_recoverable.py
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

RS = REPO / "docs" / "research_sessions"
JAAH_AUD = REPO / "data" / "cache" / "jaah" / "audio"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"
MUSX_LAT = 0.113


def root_marginal(triad_span):
    """(12,) root distribution: sum the 73-way triad posterior over the 6 triad
    types per root, averaged over the span's frames. Col 0 (N) dropped."""
    m = triad_span.mean(0)                    # (73,)
    r = np.zeros(12)
    for i in range(1, 73):
        r[(i - 1) % 12] += m[i]
    return r


def main():
    pins = json.loads((RS / "jaah_source_pins.json").read_text())
    scores = json.loads((RS / "jaah_benchmark_scores.json").read_text())["scores"]

    # duration-weighted rank tallies over CHORD spans of clean songs
    rank_dur = {}
    tot = 0.0
    top1 = top2 = 0.0
    n_songs = 0
    for s in scores:
        slug = s["slug"]; vid = pins.get(slug)
        if not vid or not (PROB_CACHE / f"{vid}.npz").exists():
            continue
        wav = JAAH_AUD / f"{vid}.wav"
        triad = frame_posteriors(wav)[0]
        am = triad.argmax(1); nf = len(am)
        rows = load_lab(LABS_DIR / f"{slug}.lab")
        # per-song oracle root to skip wrong takes
        dur = rc = 0.0; spans = []
        for t0, t1, lab in rows:
            gr, gf, _ = parse_jaah(lab.split("/")[0])
            if gr is None:
                continue
            a = max(0, int(round((t0 + MUSX_LAT) / FRAME_DT)))
            b = min(nf, int(round((t1 + MUSX_LAT) / FRAME_DT)))
            if b <= a:
                continue
            rr = [((int(x) - 1) % 12) for x in am[a:b] if x != 0]
            if not rr:
                continue
            pr = int(np.bincount(rr).argmax())
            dur += (t1 - t0); rc += (t1 - t0) * (pr == gr)
            spans.append((t0, t1, gr, a, b))
        if dur == 0 or rc / dur < 0.35:
            continue
        n_songs += 1
        for t0, t1, gr, a, b in spans:
            rm = root_marginal(triad[a:b])
            order = np.argsort(-rm)            # roots by posterior mass desc
            rank = int(np.where(order == gr)[0][0]) + 1
            d = t1 - t0
            tot += d
            rank_dur[rank] = rank_dur.get(rank, 0.0) + d
            top1 += d * (rank == 1)
            top2 += d * (rank <= 2)

    print(f"=== GT-root rank in musx root-marginal, clean JAAH (n={n_songs}) ===")
    print(f"chord-span duration analysed: {tot:.0f}s\n")
    for rk in sorted(rank_dur):
        print(f"  rank {rk:<2} {100*rank_dur[rk]/tot:>5.1f}%")
    print(f"\n  top-1 (musx already right)         : {100*top1/tot:.1f}%")
    print(f"  top-2 recall (context-lever ceiling): {100*top2/tot:.1f}%")
    print(f"  headroom a perfect top-2 prior buys : "
          f"{100*(top2-top1)/tot:+.1f}pp over musx top-1")
    print("\nRead: a big rank-2 share => the GT root IS in musx's candidates, a "
          "key/progression prior can re-rank to it (lever viable). A big rank>=3 "
          "share => the frame model lacks it, need external signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
