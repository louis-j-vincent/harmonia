"""validate_chart_alignment.py — Mission 6 Part 3/4: 20-song injected-slip harness.

Calibrates + validates ``harmonia.models.alignment_validator`` on a controlled set
of real iReal charts with hand-injected structural slips, and runs the CLAUDE.md
stopping-criterion gate (≥80% slip-recall @ ≤10% false-positive, ≥80% localisation).

Why synthetic-but-real: the 3-pilot premise check already proved the signal on
*real audio* (docs/mission_6_premise_check_results.md). This harness is about
*threshold calibration*, which needs many labelled slips — so we take real iReal
harmonic structure (jazz1460/pop400), synthesise a looped multi-chorus "inferred"
track from the GT (the realistic case: videos loop the form), and inject the three
slip types with a known ground-truth victim. The alignment itself is perfect by
construction, so what is measured is purely the *validator's* discrimination.

Run:  PYTHONPATH=. python scripts/validate_chart_alignment.py [--n 20] [--seed 0]
Exit: 0 if the gate PASSES, 1 if it FAILS.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from harmonia.data.ireal_corpus import load_playlist, tune_to_mma
    from harmonia.tab_aligner import _parse_ireal, _ROOTS
    from harmonia.models.alignment_validator import (
        validate_alignment, _section_instances,
    )

BAR_SECS = 2.0  # synthetic: constant 2 s / bar (120 BPM, 4/4) — timing is not tested


# ── build a looped multi-chorus (result, inferred_segments) from a tune ─────────
def _bar_chords(mma):
    """One (section_label, root_pc, quality) per chart bar (first slot's chord)."""
    out = []
    for _bar_no, section, slots in mma.timeline:
        if not slots:
            out.append((section, -1, ""))
            continue
        token = re.sub(r"[npWNQUSr]+$", "", slots[0][1]).strip()
        pc, q = _parse_ireal(token) if token else (-1, "")
        out.append((section, pc, q))
    return out


def build_song(tune, n_chorus=3, noise=0.10, rng=None):
    """Return (result, segs, bars) for a looped n_chorus rendering of the tune.

    result.chords carries the GT labels/sections/times (a perfect alignment).
    segs is the inferred content (== GT + light noise), one segment per bar.
    """
    rng = rng or np.random.default_rng(0)
    mma = tune_to_mma(tune)
    one = _bar_chords(mma)
    tiled = one * n_chorus

    chords, segs = [], []
    for b, (section, pc, q) in enumerate(tiled):
        t0 = b * BAR_SECS
        t1 = t0 + BAR_SECS
        label = (_ROOTS[pc % 12] + q) if pc >= 0 else "N.C."
        chords.append({"label": label, "section": section, "bar": b,
                       "t0": t0, "t1": t1, "match": "exact"})
        i_pc, i_q = pc, q
        if pc >= 0 and rng.random() < noise:          # realistic inference noise
            if rng.random() < 0.5:
                i_pc = (pc + int(rng.choice([-1, 1, 2, -2]))) % 12
            else:
                i_q = rng.choice(["maj7", "7", "min7", "min", "maj"])
        segs.append((i_pc, i_q, t0, t1))
    return SimpleNamespace(chords=chords), segs, tiled


# ── slip injectors (mutate a COPY of segs; GT result stays fixed) ───────────────
def inject_typeA_rotate(result, segs, rng):
    """Rotate content at the SECTION-INSTANCE level by one (global chorus slip).

    Each instance receives its predecessor instance's content (cyclically), fitted
    to its own bar length.  On the periodic multi-chorus form this preserves
    within-label consistency (every A-instance now holds the same *wrong* label's
    content) while collapsing every section's inferred↔GT family — the canonical
    whole-chorus / section-miscount slip that Signal 1 alone cannot see."""
    insts = _section_instances(result.chords)
    blocks = [[(segs[b][0], segs[b][1]) for b in range(s["bar0"], s["bar1"] + 1)]
              for s in insts]
    rolled = [blocks[-1]] + blocks[:-1]
    new = list(segs)
    for s, content in zip(insts, rolled):
        span = list(range(s["bar0"], s["bar1"] + 1))
        for k, b in enumerate(span):
            if content:
                r, q = content[k % len(content)]
                new[b] = (r, q, segs[b][2], segs[b][3])
    return new, None  # global → no single victim


def inject_typeB_swap(result, segs, rng):
    """Overwrite one repeated-label instance's inferred content with a different
    section's content (localised slipped repeat). Returns (segs, victim_name)."""
    insts = _section_instances(result.chords)
    labels = [s["label"] for s in insts]
    counts = {l: labels.count(l) for l in set(labels)}
    rep = [i for i, s in enumerate(insts) if counts[s["label"]] >= 2]
    if not rep:
        return segs, None
    vi = int(rng.choice(rep))
    victim = insts[vi]
    donor = next((insts[j] for j in range(len(insts))
                  if insts[j]["label"] != victim["label"]), None)
    if donor is None:
        return segs, None
    new = list(segs)
    d0, d1 = donor["bar0"], donor["bar1"]
    donor_content = [(segs[b][0], segs[b][1]) for b in range(d0, d1 + 1)]
    for k, b in enumerate(range(victim["bar0"], victim["bar1"] + 1)):
        if donor_content:
            r, q = donor_content[k % len(donor_content)]
            new[b] = (r, q, segs[b][2], segs[b][3])
    return new, f"{victim['label']}#{victim['instance']}"


def inject_typeC_shift(result, segs, rng):
    """Shift all inferred content by 2 bars (phase offset)."""
    rolled = segs[2:] + segs[:2]
    new = [(r, q, s0, s1) for (r, q, _o0, _o1), (_r, _q, s0, s1) in zip(rolled, segs)]
    return new, None


# ── the 20-song gate ────────────────────────────────────────────────────────────
def run_eval(n_songs=20, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        tunes = []
        for pl in ("jazz1460", "pop400"):
            tunes += load_playlist(ROOT / f"data/ireal/{pl}.txt")
    rng.shuffle(tunes)

    rows = []
    clean_scores, slip_scores = [], []
    for tune in tunes:
        if len(rows) >= n_songs:
            break
        try:
            result, segs, _ = build_song(tune, rng=rng)
        except Exception:
            continue
        insts = _section_instances(result.chords)
        labels = [s["label"] for s in insts]
        if len(set(labels)) < 2 or all(labels.count(l) < 2 for l in set(labels)):
            continue                                   # need repeats + >=2 labels
        clean = validate_alignment(result, segs)
        if clean.verdict == "UNVERIFIABLE":
            continue                                   # low-contrast tune: not usable
        if np.isnan(clean.align_score):
            continue

        stype = rng.choice(["A", "B", "C"], p=[0.25, 0.50, 0.25])
        injector = {"A": inject_typeA_rotate, "B": inject_typeB_swap,
                    "C": inject_typeC_shift}[stype]
        slip_segs, victim = injector(result, segs, rng)
        slipped = validate_alignment(result, slip_segs)

        clean_flag = clean.verdict in ("SUSPECT", "MISALIGNED")
        slip_flag = slipped.verdict in ("SUSPECT", "MISALIGNED")
        localized = (stype == "B" and victim is not None
                     and victim in slipped.suspect_sections)
        clean_scores.append(clean.align_score)
        slip_scores.append(slipped.align_score if not np.isnan(slipped.align_score)
                           else 0.0)
        rows.append(dict(
            title=(tune.title or "?")[:28], stype=stype, victim=victim,
            clean_verdict=clean.verdict, clean_score=clean.align_score,
            clean_flag=clean_flag, slip_verdict=slipped.verdict,
            slip_score=slipped.align_score, slip_flag=slip_flag,
            z=slipped.z_outlier, suspect=slipped.suspect_sections, localized=localized,
        ))

    n = len(rows)
    slip_recall = np.mean([r["slip_flag"] for r in rows]) if n else 0.0
    fp_clean = np.mean([r["clean_flag"] for r in rows]) if n else 1.0
    typeB = [r for r in rows if r["stype"] == "B" and r["slip_flag"]]
    loc_acc = np.mean([r["localized"] for r in typeB]) if typeB else 0.0

    # ROC-AUC of (1 - align_score) separating clean vs slipped (Mann-Whitney)
    auc = _auc([1 - s for s in slip_scores], [1 - s for s in clean_scores])

    if verbose:
        _print_table(rows)
        print(f"\nn={n} songs   type mix: "
              f"A={sum(r['stype']=='A' for r in rows)} "
              f"B={sum(r['stype']=='B' for r in rows)} "
              f"C={sum(r['stype']=='C' for r in rows)}")
        print(f"slip recall (flagged)      : {slip_recall:.1%}  (want >=80%)")
        print(f"false positive on clean    : {fp_clean:.1%}  (want <=10%)")
        print(f"localisation acc (Type B)  : {loc_acc:.1%}  (want >=80%, n={len(typeB)})")
        print(f"ROC-AUC (score sep)        : {auc:.3f}")

    return dict(n=n, misaligned_recall=float(slip_recall), fp_on_clean=float(fp_clean),
                suspect_accuracy=float(loc_acc), auc=float(auc), rows=rows)


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _print_table(rows):
    print(f"\n{'title':30s} {'ty':2s} {'clean':>6s} {'slip':>6s} "
          f"{'flag':>4s} {'z':>6s} {'loc':>4s} suspect")
    for r in rows:
        cs = f"{r['clean_score']:.2f}"
        ss = f"{r['slip_score']:.2f}" if not np.isnan(r['slip_score']) else " nan"
        z = f"{r['z']:.2f}" if not np.isnan(r['z']) else "  -"
        loc = "Y" if r["localized"] else ("-" if r["stype"] != "B" else "N")
        print(f"{r['title']:30s} {r['stype']:2s} {cs:>6s} {ss:>6s} "
              f"{'Y' if r['slip_flag'] else 'N':>4s} {z:>6s} {loc:>4s} "
              f"{','.join(r['suspect'][:3])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_eval(args.n, args.seed)
    recall, fp, loc = res["misaligned_recall"], res["fp_on_clean"], res["suspect_accuracy"]
    print("\n" + "=" * 60)
    if recall >= 0.80 and fp <= 0.10 and loc >= 0.80:
        print("PASS: ready to integrate into server (display banner).")
        return 0
    print("FAIL: iterate signal weights / thresholds.")
    print(f"  slip recall        : {recall:.1%} (want >=80%)")
    print(f"  false pos (clean)  : {fp:.1%} (want <=10%)")
    print(f"  localisation acc   : {loc:.1%} (want >=80%)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
