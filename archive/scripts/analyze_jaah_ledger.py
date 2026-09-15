#!/usr/bin/env python3
"""Analyse the JAAH Tinder ledger (docs/research_sessions/jaah_tinder_ledger.json).

Same idea as scripts/analyze_tinder_ledger.py but for the JAAH benchmark: the GT
here is JAAH's absolute-timestamp Harte labels, scored at root + 7-family
(parse_jaah) — the same convention build_jaah_benchmark.py scores with. Turns
Louis's swipes/corrections/seg-flags on real jazz into:

  1. Verdict tally per song + overall.
  2. Truth confusion: with the human-entered true chord (or the implied truth
     from a right/wrong swipe), is the PREDICTOR or the JAAH GT closer to the
     truth? On JAAH the GT is trustworthy, so most 'wrong' verdicts should
     confirm the model erred — the exception is the interesting case.
  3. Segmentation flags: cards where Louis said the BOUNDARY is wrong — the
     dense-bebop failure mode the density diagnostic points at.
  4. Predictor error taxonomy on 'wrong' verdicts.

Read-only.

Usage:
    .venv/bin/python scripts/analyze_jaah_ledger.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.build_jaah_corpus import parse_jaah          # noqa: E402

LEDGER = REPO / "docs" / "research_sessions" / "jaah_tinder_ledger.json"


def rf(label):
    """(root_pc, family) via JAAH's own parser; (None,None) for no-chord."""
    r, f, _ = parse_jaah(label)
    return r, f


def root_eq(a, b):
    ra, _ = rf(a)
    rb, _ = rf(b)
    if ra is None or rb is None:
        return ra is None and rb is None
    return ra == rb


def fam_eq(a, b):
    ra, fa = rf(a)
    rb, fb = rf(b)
    if ra is None or rb is None:
        return ra is None and rb is None
    return ra == rb and fa == fb


def truth_label(rec):
    v = rec.get("verdict")
    if v == "correct" and rec.get("correction"):
        return rec["correction"]["label"]
    if v == "right":
        return rec["pred"]
    if v == "wrong":
        return rec["gt"]
    return None


def main():
    if not LEDGER.exists():
        print(f"no JAAH ledger yet at {LEDGER.relative_to(REPO)} — swipe /jaah first.")
        return 0
    recs = list(json.loads(LEDGER.read_text()).values())
    if not recs:
        print("JAAH ledger empty.")
        return 0

    per_song = defaultdict(Counter)
    overall = Counter()
    for r in recs:
        per_song[r["song"]][r.get("verdict", "?")] += 1
        overall[r.get("verdict", "?")] += 1

    print(f"=== {len(recs)} judged JAAH cards ===\n")
    print("VERDICTS")
    print(f"  {'song':<20} {'wrong':>6} {'right':>6} {'correct':>8} {'skip':>5} {'seg':>4}")
    for s in sorted(per_song):
        c = per_song[s]
        seg = sum(1 for r in recs if r["song"] == s and r.get("seg_issue"))
        print(f"  {s:<20} {c['wrong']:>6} {c['right']:>6} {c['correct']:>8} "
              f"{c['skip']:>5} {seg:>4}")
    segn = sum(1 for r in recs if r.get("seg_issue"))
    print(f"  {'TOTAL':<20} {overall['wrong']:>6} {overall['right']:>6} "
          f"{overall['correct']:>8} {overall['skip']:>5} {segn:>4}\n")

    # truth confusion
    n = pr = pf = gr = gf = 0
    for r in recs:
        tl = truth_label(r)
        if tl is None:
            continue
        n += 1
        pr += root_eq(r["pred"], tl)
        pf += fam_eq(r["pred"], tl)
        gr += root_eq(r["gt"], tl)
        gf += fam_eq(r["gt"], tl)
    if n:
        print(f"TRUTH CONFUSION (n={n})")
        print(f"  predictor matches truth: root {pr}/{n} ({100*pr/n:.0f}%)  "
              f"family {pf}/{n} ({100*pf/n:.0f}%)")
        print(f"  JAAH GT matches truth  : root {gr}/{n} ({100*gr/n:.0f}%)  "
              f"family {gf}/{n} ({100*gf/n:.0f}%)")
        who = "predictor" if pr > gr else "JAAH GT" if gr > pr else "tie"
        print(f"  → {who} closer to truth on these disagreements\n")

    # segmentation flags
    seg = [r for r in recs if r.get("seg_issue")]
    if seg:
        print(f"SEGMENTATION FLAGS ({len(seg)}) — the dense-harmonic-rhythm failure mode")
        for r in sorted(seg, key=lambda r: (r["song"], r["from"])):
            print(f"  {r['song']:<20} {r['from']:>7.2f} {r['to']:>7.2f}  "
                  f"{r['gt']:>10} vs {r['pred']}")
        print()

    # error taxonomy on 'wrong'
    err = Counter()
    for r in recs:
        if r.get("verdict") != "wrong":
            continue
        gr_, gf_ = rf(r["gt"])
        pr_, pf_ = rf(r["pred"])
        if pr_ is None and gr_ is not None:
            err["pred N.C. (missed a chord)"] += 1
        elif gr_ is None and pr_ is not None:
            err["pred a chord over no-chord"] += 1
        elif gr_ != pr_:
            err["wrong root"] += 1
        elif gf_ != pf_:
            err["right root, wrong family"] += 1
        else:
            err["boundary/other"] += 1
    if err:
        print("PREDICTOR ERROR TAXONOMY (verdict=wrong)")
        for k, v in err.most_common():
            print(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
