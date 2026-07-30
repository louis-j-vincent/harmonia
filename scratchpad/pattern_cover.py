#!/usr/bin/env python3
"""pattern_cover.py — Louis's two-pass cover-then-mine-the-holes segmentation.
2026-07-30. Supersedes the single left-to-right walk in `pattern_slide.segment`.

Why it replaces the walk (Louis, 2026-07-30):

  "once we've filled in most of the song with A B (and C if relevant), then look
   at the holes that are not covered by any of these, and see if we can detect
   patterns in these, like the ending of C that goes C F Ab G, and find other
   peaks where it happens. For now, this would maybe be a separate D."

So there is NO hard-coded 1st/2nd-ending rule any more. The chorus tail
`C F | Ab G` is not a special case bolted onto B — it is simply a hole left over
after B claimed its loops, and pass 2 discovers it as its own pattern D with its
own peak list across the whole song. A display layer can later notice "D always
follows B" and render it as a 2nd ending; the segmenter does not need to know.

The other change is WHERE a pattern is allowed to claim bars. The old walk only
ever tested a pattern at multiples of its own length counted from its own start
("its own grid"), so a section starting one bar off was unrecoverable — and a
competing pattern scoring 0.807 somewhere was never even asked. Here every
pattern is slid over the WHOLE song and claims every bar-aligned peak it finds,
which is what Louis proposed from the beginning.

Scores fuse several matrices (Louis: "combine them, heavy weighting on the chord
SSM"). `drums` is a reserved slot for the rhythm SSM being built in parallel.

Run: .venv/bin/python scratchpad/pattern_cover.py [song-substring]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

import pattern_slide as ps
from section_merge_declined import _load_payload

SPB = ps.SLOTS_PER_BAR

# Heavy weight on harmony, a real but minority vote from bare roots, and a slot
# held open for the drum-pattern SSM. Weights are renormalised over whatever
# matrices are actually supplied, so dropping one does not change the scale.
# MEASURED 2026-07-30 against docs/this_love_target_spec.md ground truth, comparing
# 0.7*chord+0.3*root vs max(chord,root) vs chord alone. Separation gaps came out
# essentially identical (A +0.34/+0.32/+0.32, B +0.23/+0.23/+0.23, bridge
# +0.53/+0.46/+0.46) — so root adds NO measurable separation here. Worse, averaging
# root in actively costs coverage: root is a binary judge, so a verse repeat with
# two mis-decoded chords (bar 28) scores 0.763 on chord tones but only 0.625 on
# roots, and the average 0.722 falls below CONTINUE, splitting off a bogus section.
# Hence chord-tone carries the vote until a cue with independent information exists.
# `drums` stays reserved: the rhythm SSM (docs/research_sessions/rhythm_ssm_2026-07-30.md)
# is a null on this song's programmed drums but beats harmony on live-drum songs, so
# its weight has to be per-song, not a constant.
WEIGHTS = {"chord": 1.00, "root": 0.00, "drums": 0.00}


def root_ssm(seq) -> np.ndarray:
    """1.0 iff two slots share a root. A SECOND opinion, never the only one —
    the project's SSM is `pattern_slide.chord_ssm` (Bb closer to Gm than to F).
    Measured on This Love it separates A and B better than chord-tone does, and C
    worse, so the two are complementary and both are weighted in."""
    r = np.array([x[0] for x in seq])
    return ((r[:, None] == r[None, :]) & (r[:, None] >= 0)).astype(np.float32)


def stripe_diag(S: np.ndarray, row0: int, d: int) -> np.ndarray:
    """Slide ACROSS X with the rows pinned to [row0, row0+d): the mean of the slid
    stripe's DIAGONAL. 1.0 = these d slots reproduce the pattern slot for slot."""
    n = S.shape[0]
    out = np.full(n, np.nan)
    for t in range(0, n - d + 1):
        out[t] = float(np.mean([S[row0 + i, t + i] for i in range(d)]))
    return out


def fused_diag(mats: dict, row0: int, d: int, weights=WEIGHTS) -> np.ndarray:
    """Weighted mean of each matrix's stripe-diagonal slide."""
    used = {k: w for k, w in weights.items() if w > 0 and k in mats}
    tot = sum(used.values()) or 1.0
    acc = None
    for k, w in used.items():
        c = stripe_diag(mats[k], row0, d) * (w / tot)
        acc = c if acc is None else acc + c
    return acc


def band_score(mats: dict, t0: int, d: int, weights=WEIGHTS) -> float:
    """Does the span at t0 reproduce itself d slots later? (the period test)"""
    used = {k: w for k, w in weights.items() if w > 0 and k in mats}
    tot = sum(used.values()) or 1.0
    n = next(iter(mats.values())).shape[0]
    if t0 + 2 * d > n:
        return 0.0
    return sum(w / tot * float(np.mean([mats[k][t0 + i, t0 + i + d] for i in range(d)]))
               for k, w in used.items())


def minimal_period(mats: dict, seq, t0: int, hi_slot: int, log=print):
    """Smallest whole-bar lag that reproduces the span at t0, searched only within
    [t0, hi_slot). Returns None when nothing reproduces it — which is itself
    meaningful: it means this span is a one-off (a bridge, or a cadential tail),
    and the caller should treat the span itself as the pattern."""
    S = mats["chord"]
    n = S.shape[0]
    j = t0 + 1
    while j < n and S[t0, j] >= ps.HI:
        j += 1
    held = j - t0

    lo = max(SPB, (held // SPB + 1) * SPB)
    band = {}
    for d in range(lo, (hi_slot - t0) + 1, SPB):
        if t0 + 2 * d > hi_slot:
            break
        if len({r for r, _ in seq[t0:t0 + d]}) < 2:
            continue                      # one chord held is not a pattern
        band[d] = band_score(mats, t0, d)
    if not band:
        return None, {}
    best = max(band.values())
    cut = max(ps.PERIOD_FLOOR, best - ps.PERIOD_MARGIN)
    d = next((k for k in sorted(band) if band[k] >= cut), None)
    top = sorted(band.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    log(f"      period search: " + ", ".join(f"{k//SPB}bar:{v:.2f}" for k, v in top)
        + f"   cut={cut:.2f} -> "
        + (f"{d//SPB}-bar loop" if d else "NO repeating loop (one-off span)"))
    return d, band


def claim(mats, seq, names, row0, d, n_bars, covered, label, log=print,
          thresh=None, corr=None):
    """Slide the pattern over the WHOLE song and claim every bar-aligned peak that
    scores >= ``thresh`` and lands on still-free bars. Returns the claimed spans."""
    thresh = ps.REP_STRICT if thresh is None else thresh
    if corr is None:
        corr = fused_diag(mats, row0, d)
    n = len(corr)
    cand = [t for t in range(0, n - d + 1, SPB)
            if not np.isnan(corr[t]) and corr[t] >= thresh - 1e-9]
    cand.sort(key=lambda t: (-corr[t], t))
    got = []
    for t in cand:
        b0, b1 = t // SPB, (t + d) // SPB
        if any(covered[b] is not None for b in range(b0, min(b1, n_bars))):
            continue                      # overlaps something already claimed
        for b in range(b0, min(b1, n_bars)):
            covered[b] = label
        got.append((t, float(corr[t])))
    got.sort()
    log(f"      {label} = {d//SPB}-bar [{' '.join(names[row0:row0+d])}] claims "
        f"{len(got)} span(s): " + ", ".join(f"bar {t//SPB}({c:.2f})" for t, c in got))
    return corr, got


CONTEXT_FLOOR = 0.50   # relaxed similarity gate for a context-based merge; the
                       # CONTEXT is the discriminator, similarity only vetoes absurdity


def merge_by_context(mats, sections, pats, covered, n_bars, log=print):
    """Merge two vocabulary items that are the same length and ALWAYS occur in the
    same place in the form. Louis's idea, 2026-07-30.

    This Love's chorus tail is decoded three different ways — `Cm F△7 | Ab G7`,
    `Cm7 F△7 | Ab Ab`, `Cm F | F Ab` — because the final G is sometimes not decoded
    at all (bars 63 and 71 carry a single chord onset). By similarity alone they
    cannot be merged: the best degraded copy scores 0.740 against the clean one,
    while genuinely DIFFERENT sections reach 0.697, leaving 0.05 of room — far
    too little to threshold safely.

    Context settles it without touching the floor: every occurrence of all three
    lands immediately after a run of B. Two items of equal length whose every
    occurrence shares the same single predecessor are the same item, and
    similarity is demoted to a sanity veto (>= CONTEXT_FLOOR) rather than the
    decision.

    RISK, stated because this is the kind of rule that over-merges silently: two
    genuinely distinct sections that happen to always follow the same section and
    happen to be the same length WILL be fused. The length + unique-predecessor +
    floor conjunction makes that unlikely in pop forms, but it is not impossible;
    a form where a verse and a pre-chorus are both 2 bars and both always follow
    the intro would break it.
    """
    def preds(label):
        out = set()
        for i, s in enumerate(sections):
            if s["label"] != label:
                continue
            out.add(sections[i - 1]["label"] if i else "<start>")
        return out

    by_label = {p["label"]: p for p in pats}
    order = [p["label"] for p in pats]
    remap: dict[str, str] = {}
    for qi in range(len(order) - 1, 0, -1):
        q = order[qi]
        if q in remap or q not in by_label:
            continue
        for p in order[:qi]:
            if p in remap or p not in by_label:
                continue
            P, Q = by_label[p], by_label[q]
            if P["d"] != Q["d"]:
                continue
            cp, cq = preds(p), preds(q)
            if len(cp) != 1 or cp != cq:
                continue
            d = P["d"]
            sim = float(np.mean([mats["chord"][P["row0"] + i, Q["row0"] + i]
                                for i in range(d)]))
            if sim < CONTEXT_FLOOR:
                log(f"    {q} and {p} share the context {cp} and the length but only "
                    f"score {sim:.2f} < {CONTEXT_FLOOR} — NOT merging")
                continue
            log(f"    {q} -> {p}: same length ({d//SPB} bars), every occurrence of both "
                f"follows {list(cp)[0]}, similarity {sim:.2f} passes the {CONTEXT_FLOOR} "
                f"veto. Same item, decoded differently.")
            remap[q] = p
            break
    if not remap:
        log("    nothing to merge on context")
    for b in range(n_bars):
        while covered[b] in remap:
            covered[b] = remap[covered[b]]
    return remap


def holes(covered, n_bars):
    """Maximal runs of still-unclaimed bars."""
    out, i = [], 0
    while i < n_bars:
        if covered[i] is None:
            j = i
            while j < n_bars and covered[j] is None:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def cover(mats, seq, names, n_bars, log=print, context_merge=False):
    n = n_bars * SPB
    covered: list[str | None] = [None] * n_bars
    pats: list[dict] = []
    letters = "ABCDEFGHIJ"

    log("\n  PASS 1 — discover patterns from the front, each claiming every "
        "occurrence it has anywhere in the song")
    tried: set[int] = set()
    guard = 0
    while guard < 40:
        guard += 1
        start = next((b for b in range(n_bars)
                      if covered[b] is None and b not in tried), None)
        if start is None:
            break
        tried.add(start)
        t0 = start * SPB
        log(f"\n    uncovered from bar {start} ({names[t0]}):")
        # a pattern discovered here may run to the end of the song
        d, _ = minimal_period(mats, seq, t0, n, log=log)
        if d is None:
            # NOT a reason to stop: a later part of the song may still hold a loop.
            # Skip this bar and keep looking; pass 2 will mine what is left.
            log("      -> no loop from here; skipping (pass 2 will mine it)")
            continue
        label = letters[len(pats)]
        corr, got = claim(mats, seq, names, t0, d, n_bars, covered, label, log=log)
        if not got:
            log("      -> claimed nothing new; skipping")
            continue
        pats.append({"label": label, "row0": t0, "d": d, "corr": corr, "spans": got,
                     "pattern": names[t0:t0 + d], "origin": "loop"})

    # ── PASS 1b — the DEGRADED copies ────────────────────────────────────────
    # A repetition whose chords the decoder got slightly wrong scores in the
    # measured band 0.76-0.84: too low for REP_STRICT (0.85), but well above the
    # highest score any genuinely DIFFERENT section reaches (0.697). Re-offer every
    # known pattern at CONTINUE before minting new letters, or the verse's 3rd/4th
    # passes (This Love bars 28 and 32, where `G` decoded as `C` and `Bm`) become a
    # bogus new section, and the chorus tail gets one letter per decode variant.
    log(f"\n  PASS 1b — re-offer known patterns at the looser CONTINUE={ps.CONTINUE} "
        f"(measured: degraded repeats score 0.76-0.84, different sections top out at 0.70)")
    for p in pats:
        _, got = claim(mats, seq, names, p["row0"], p["d"], n_bars, covered,
                       p["label"], log=log, thresh=ps.CONTINUE, corr=p["corr"])
        p["spans"] = sorted(p["spans"] + got)

    hl = holes(covered, n_bars)
    log(f"\n  PASS 2 — mine the {len(hl)} hole(s) no pattern covered: "
        + ", ".join(f"bars {a}-{b-1}" for a, b in hl))
    for a, b in hl:
        if covered[a] is not None:
            continue                       # a previous hole's pattern took it
        t0, span = a * SPB, (b - a) * SPB
        log(f"\n    hole bars {a}-{b-1} ({b-a} bars): {' '.join(names[t0:t0+span])}")
        # Does the hole contain its own internal loop? If not, the hole ITSELF is
        # the pattern — that is how `C F | Ab G` becomes a first-class pattern.
        d, _ = minimal_period(mats, seq, t0, t0 + span, log=log)
        if d is None:
            d = span
            log(f"      -> the hole itself is the pattern ({d//SPB} bars)")
        label = letters[len(pats)]
        corr, got = claim(mats, seq, names, t0, d, n_bars, covered, label, log=log)
        if not got:
            continue
        # a hole pattern also gets the degraded-copy pass, so the chorus tail is ONE
        # letter whether or not its final G was decoded
        _, more = claim(mats, seq, names, t0, d, n_bars, covered, label, log=log,
                        thresh=ps.CONTINUE, corr=corr)
        pats.append({"label": label, "row0": t0, "d": d, "corr": corr,
                     "spans": sorted(got + more), "pattern": names[t0:t0 + d],
                     "origin": "hole" if d == span else "hole-loop"})

    left = [b for b in range(n_bars) if covered[b] is None]
    if left:
        log(f"\n  still unclaimed after both passes: bars {left} — attaching each to "
            f"the previous bar's section")
        for b in left:
            covered[b] = covered[b - 1] if b else (pats[0]["label"] if pats else "A")

    def build():
        """consecutive same-label bars -> one section"""
        out = []
        for b in range(n_bars):
            if out and out[-1]["label"] == covered[b] and out[-1]["bar1"] == b:
                out[-1]["bar1"] = b + 1
            else:
                out.append({"label": covered[b], "bar0": b, "bar1": b + 1})
        return out

    sections = build()

    # ── PASS 3 — context merge, OFF BY DEFAULT ───────────────────────────────
    # Louis's call, 2026-07-30: the over-merge risk documented in
    # `merge_by_context` is not worth taking. Keeping the tail variants as
    # separate vocabulary items is also musically truer — the last chorus loops
    # back into B rather than returning to A, so its tail is a different item.
    # The code stays, gated, because the rule may earn its place once there is a
    # corpus to measure the false-merge rate against.
    if context_merge:
        log("\n  PASS 3 — context merge (same length + same predecessor everywhere)")
        remap = merge_by_context(mats, sections, pats, covered, n_bars, log=log)
        if remap:
            pats = [p for p in pats if p["label"] not in remap]
            sections = build()

    by_label = {p["label"]: p for p in pats}
    for s in sections:
        p = by_label.get(s["label"])
        s["d_bars"] = (p["d"] // SPB) if p else (s["bar1"] - s["bar0"])
        s["reps"] = max(1, (s["bar1"] - s["bar0"]) // max(1, s["d_bars"]))
        s["pattern"] = p["pattern"] if p else []
    return sections, pats, covered


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    seq, names, n_bars, bar_sec = ps.rigid_slots(P)
    mats = {"chord": ps.chord_ssm(names), "root": root_ssm(seq)}

    print(f"\n=== {slug} — {n_bars} bars @ {bar_sec:.3f}s, {n_bars*SPB} half-bar slots ===")
    print(f"fusing: " + ", ".join(f"{k} x{WEIGHTS[k]:.2f}" for k in mats if WEIGHTS[k] > 0)
          + f"   (drums slot reserved, weight {WEIGHTS['drums']:.2f})")

    sections, pats, covered = cover(mats, seq, names, n_bars)

    print("\n--- patterns found ---")
    for p in pats:
        print(f"  {p['label']}  {p['d']//SPB}-bar, learned at bar {p['row0']//SPB}, "
              f"{len(p['spans'])} occurrence(s) [{p['origin']}]")
        print(f"       {' '.join(p['pattern'])}")
        print(f"       at bars " + ", ".join(f"{t//SPB}({c:.2f})" for t, c in p["spans"]))
    print("\n--- sections ---")
    for s in sections:
        rep = f" x{s['reps']}" if s["reps"] > 1 else ""
        print(f"  {s['label']}  bars {s['bar0']:2d}-{s['bar1']-1:2d}  "
              f"({s['d_bars']}-bar loop{rep})")
    print("\n  form: " + " ".join(
        f"{s['label']}x{s['reps']}" if s["reps"] > 1 else s["label"] for s in sections))
    return mats, seq, names, n_bars, sections, pats, slug


if __name__ == "__main__":
    main()
