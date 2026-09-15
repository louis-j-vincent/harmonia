#!/usr/bin/env python3
"""Analyse the pred-vs-GT Tinder ledger (docs/research_sessions/tinder_ledger.json).

For every judged card we know: gt label, pred label, the verdict
(wrong/right/correct/skip), and — when the verdict is `correct` — the TRUE chord
Louis dialled in. This turns the swipes into three actionable tables:

  1. Verdict tally per song and overall (how often the predictor beats the GT).
  2. Truth confusion: with the human-entered true chord, does PRED match truth?
     does GT match truth? — the only way to tell whether a disagreement is a
     model error or a GT error, adjudicated by ear.
  3. GT-repair leads: every span where the true chord differs from the frozen
     GT label (verdict `right` ⇒ true≈pred; verdict `correct` ⇒ true=entered).

Uses harmonia.eval.accuracy_score's own matchers so root/family verdicts match
the scorer exactly. Read-only.

Usage:
    .venv/bin/python scripts/analyze_tinder_ledger.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.eval.accuracy_score import (          # noqa: E402
    chord_from_label, chord_family,
)

LEDGER = REPO / "docs" / "research_sessions" / "tinder_ledger.json"


def root_eq(a, b) -> bool:
    if a.is_nc or b.is_nc:
        return a.is_nc and b.is_nc
    return a.root_pc == b.root_pc


def fam_eq(a, b) -> bool:
    if a.is_nc or b.is_nc:
        return a.is_nc and b.is_nc
    return chord_family(a.quality) == chord_family(b.quality)


def truth_label(rec) -> str | None:
    """The true chord for a card, if the ledger implies one:
       - verdict 'correct' -> the entered label
       - verdict 'right'   -> the predictor (predictor judged correct)
       - verdict 'wrong'   -> the GT stands (predictor wrong, GT assumed right)
       - else -> unknown."""
    v = rec.get("verdict")
    if v == "correct" and rec.get("correction"):
        return rec["correction"]["label"]
    if v == "right":
        return rec["pred"]
    if v == "wrong":
        return rec["gt"]
    return None


def main() -> int:
    if not LEDGER.exists():
        print(f"no ledger yet at {LEDGER.relative_to(REPO)} — swipe some cards first.")
        return 0
    recs = list(json.loads(LEDGER.read_text()).values())
    if not recs:
        print("ledger is empty.")
        return 0

    # 1) verdict tally
    per_song = defaultdict(Counter)
    overall = Counter()
    for r in recs:
        per_song[r["song"]][r.get("verdict", "?")] += 1
        overall[r.get("verdict", "?")] += 1

    print(f"=== {len(recs)} judged cards ===\n")
    print("VERDICTS  (right = predictor beats GT; correct = true chord entered)")
    hdr = f"  {'song':<26} {'wrong':>6} {'right':>6} {'correct':>8} {'skip':>5}"
    print(hdr)
    for s in sorted(per_song):
        c = per_song[s]
        print(f"  {s:<26} {c['wrong']:>6} {c['right']:>6} {c['correct']:>8} {c['skip']:>5}")
    print(f"  {'TOTAL':<26} {overall['wrong']:>6} {overall['right']:>6} "
          f"{overall['correct']:>8} {overall['skip']:>5}\n")

    # 2) truth confusion (needs an implied truth)
    dur = 0.0
    pred_root = pred_fam = gt_root = gt_fam = n_truth = 0
    for r in recs:
        tl = truth_label(r)
        if tl is None:
            continue
        n_truth += 1
        truth = chord_from_label(0, 1, tl)
        pr = chord_from_label(0, 1, r["pred"])
        gt = chord_from_label(0, 1, r["gt"])
        pred_root += root_eq(pr, truth)
        pred_fam += fam_eq(pr, truth)
        gt_root += root_eq(gt, truth)
        gt_fam += fam_eq(gt, truth)
    if n_truth:
        print(f"TRUTH CONFUSION  (n={n_truth} cards with an implied true chord)")
        print(f"  predictor matches truth : root {pred_root}/{n_truth} "
              f"({100*pred_root/n_truth:.0f}%)  family {pred_fam}/{n_truth} "
              f"({100*pred_fam/n_truth:.0f}%)")
        print(f"  frozen GT matches truth : root {gt_root}/{n_truth} "
              f"({100*gt_root/n_truth:.0f}%)  family {gt_fam}/{n_truth} "
              f"({100*gt_fam/n_truth:.0f}%)")
        verdict = ("predictor" if pred_root > gt_root else
                   "GT" if gt_root > pred_root else "tie")
        print(f"  → on these disagreements the {verdict} is closer to the truth\n")

    # 3) GT-repair leads
    leads = []
    for r in recs:
        tl = truth_label(r)
        if tl is None:
            continue
        gt = chord_from_label(0, 1, r["gt"])
        truth = chord_from_label(0, 1, tl)
        if not (root_eq(gt, truth) and fam_eq(gt, truth)):
            tag = " [SEG]" if r.get("seg_issue") else ""
            leads.append((r["song"], r["from"], r["to"], r["gt"], tl,
                          (r.get("verdict") or "") + tag))
    if leads:
        print(f"GT-REPAIR LEADS  ({len(leads)} spans where true chord ≠ frozen GT)")
        print(f"  {'song':<22} {'from':>7} {'to':>7}  {'GT':>10} -> {'TRUE':<10} via")
        for s, a, b, g, t, v in sorted(leads):
            print(f"  {s:<22} {a:>7.2f} {b:>7.2f}  {g:>10} -> {t:<10} {v}")

    # 3b) segmentation flags — cards where the boundaries themselves are wrong,
    # so any label disagreement there is (partly) a boundary artifact, not a
    # clean label error. Louis's ear flags these directly.
    seg = [r for r in recs if r.get("seg_issue")]
    if seg:
        seg_by_song = Counter(r["song"] for r in seg)
        print(f"\nSEGMENTATION FLAGS  ({len(seg)} cards where the boundary is wrong)")
        for s, n in seg_by_song.most_common():
            print(f"  {n:>4}  {s}")
        print("  (label metrics on these are confounded by the boundary error — "
              "treat as segmentation targets, not label errors)")
        print(f"  {'song':<22} {'from':>7} {'to':>7}  {'GT':>10}  {'PRED':>10}")
        for r in sorted(seg, key=lambda r: (r["song"], r["from"])):
            print(f"  {r['song']:<22} {r['from']:>7.2f} {r['to']:>7.2f}  "
                  f"{r['gt']:>10}  {r['pred']:>10}")

    # 4) error taxonomy on 'wrong' verdicts (predictor's mistakes)
    err = Counter()
    for r in recs:
        if r.get("verdict") != "wrong":
            continue
        gt = chord_from_label(0, 1, r["gt"])
        pr = chord_from_label(0, 1, r["pred"])
        if pr.is_nc and not gt.is_nc:
            err["pred N.C. (missed a chord)"] += 1
        elif gt.is_nc and not pr.is_nc:
            err["pred a chord over silence"] += 1
        elif not root_eq(gt, pr):
            err["wrong root"] += 1
        elif not fam_eq(gt, pr):
            err["right root, wrong family"] += 1
        else:
            err["extension-only"] += 1
    if err:
        print("\nPREDICTOR ERROR TAXONOMY  (verdict=wrong)")
        for k, v in err.most_common():
            print(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
