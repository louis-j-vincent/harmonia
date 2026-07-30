"""Detection-granularity variants of `section_vocab.vocab_sections`.

RESEARCH CODE — nothing here is imported by the app. It re-implements the cover
loop with a pluggable *pattern-length chooser* so the shipped file stays
untouched while the granularity question is measured.

The shipped detector chooses each pattern's length with `minimal_period`, which
is free to return any whole number of bars. Louis's proposal is to fix the unit
instead — 2 bars, then 4, then 8 — and to treat a pattern that survives at the
larger scale as a section and one that only exists at 2 bars as a fragment.

Four arms:

* ``auto``  — the shipped chooser (`minimal_period`). The baseline.
* ``fix2`` / ``fix4`` / ``fix8`` — every pattern is forced to that many bars.
* ``hier``  — enumerate candidate patterns at ALL of {2,4,8} bars first (the
  "log the unique combos" step), then cover COARSE-TO-FINE: an 8-bar pattern
  that recurs claims its bars before any 4-bar one is offered, and 2-bar
  patterns only ever explain what is left.
"""
from __future__ import annotations

import numpy as np

from harmonia.models.section_vocab import (
    CONTINUE, REP_STRICT, SLOTS_PER_BAR, _LETTERS, _claim, build_slots,
    chord_ssm, minimal_period, stripe_diag,
)

_MAX_ITERS = 40


# ── the shipped cover loop, with the length chooser pulled out ────────────────

def _cover(bars, n_bars, tonic_pc, bpb, choose_d):
    """`vocab_sections` verbatim except that the pattern length for a hole comes
    from ``choose_d(S, roots, t0, hi, known, span)`` instead of `minimal_period`."""
    if n_bars < 8:
        return None
    tokens, roots, known = build_slots(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    if not any(tokens):
        return None
    S = chord_ssm(tokens)

    covered: list[str | None] = [None] * n_bars
    pats: list[dict] = []
    tried: set[int] = set()

    for _ in range(_MAX_ITERS):
        for thresh in (REP_STRICT, CONTINUE):
            for p in pats:
                p["spans"] = sorted(set(p["spans"] + _claim(
                    p["corr"], p["d"], n_bars, covered, p["label"], thresh)))
        start = next((b for b in range(n_bars)
                      if covered[b] is None and b not in tried), None)
        if start is None or len(pats) >= len(_LETTERS):
            break
        tried.add(start)
        end = start
        while end < n_bars and covered[end] is None:
            end += 1
        t0, span = start * SLOTS_PER_BAR, (end - start) * SLOTS_PER_BAR
        d = choose_d(S, roots, t0, t0 + span, known, span)
        corr = stripe_diag(S, t0, d, known)
        label = _LETTERS[len(pats)]
        got = _claim(corr, d, n_bars, covered, label, REP_STRICT)
        if not got:
            continue
        pats.append({"label": label, "row0": t0, "d": d, "corr": corr,
                     "spans": got})
    return _finish(S, tokens, known, covered, pats, n_bars)


def _finish(S, tokens, known, covered, pats, n_bars):
    """The shipped tail of `vocab_sections`: gating, hole fill, re-award, runs."""
    if len(pats) < 2 or max((len(p["spans"]) for p in pats), default=0) < 2:
        return None
    for b in range(n_bars):
        if covered[b] is None:
            covered[b] = covered[b - 1] if b else pats[0]["label"]
    by_label = {p["label"]: p for p in pats}

    def _runs():
        runs = []
        for b in range(n_bars):
            if runs and runs[-1]["label"] == covered[b] and runs[-1]["bar1"] == b:
                runs[-1]["bar1"] = b + 1
            else:
                runs.append({"label": covered[b], "bar0": b, "bar1": b + 1})
        return runs

    for s in _runs():
        cur = by_label.get(s["label"])
        if cur is None:
            continue
        t = s["bar0"] * SLOTS_PER_BAR
        rivals = [p for p in pats if p["d"] == cur["d"]]
        if len(rivals) < 2:
            continue

        def _score(p, _t=t, _d=cur["d"]):
            v = stripe_diag(S, p["row0"], _d, known)[_t]
            return -1.0 if np.isnan(v) else float(v)

        best = max(rivals, key=_score)
        if best["label"] != s["label"] and _score(best) > _score(cur) + 1e-6:
            for b in range(s["bar0"], s["bar1"]):
                covered[b] = best["label"]

    out = _runs()
    still = {s["label"] for s in out}
    pats = [p for p in pats if p["label"] in still]
    by_label = {p["label"]: p for p in pats}
    for s in out:
        p = by_label.get(s["label"])
        d_bars = (p["d"] // SLOTS_PER_BAR) if p else (s["bar1"] - s["bar0"])
        s["d_bars"] = max(1, d_bars)
        s["reps"] = max(1, (s["bar1"] - s["bar0"]) // s["d_bars"])
        s["tokens"] = tokens[p["row0"]:p["row0"] + p["d"]] if p else []
    return out


# ── choosers ─────────────────────────────────────────────────────────────────

def _auto(S, roots, t0, hi, known, span):
    return minimal_period(S, roots, t0, hi, known) or span


def _fixed(k_bars: int):
    """Force every pattern to ``k_bars``. Falls back to the hole when the hole is
    shorter than the unit — a 2-bar tail cannot be written as an 8-bar phrase."""
    def choose(S, roots, t0, hi, known, span, _k=k_bars * SLOTS_PER_BAR):
        return _k if _k <= span else span
    return choose


def _coarse_first(S, roots, t0, hi, known, span, scales=(8, 4, 2)):
    """Louis's "combo of the two": keep the shipped cover loop, but pick the hole's
    pattern length COARSE-FIRST — the largest unit in ``scales`` that both fits the
    hole and actually recurs elsewhere in the song wins; otherwise fall back to the
    shipped `minimal_period`. A unit that only recurs at 2 bars stays a fragment."""
    for k in scales:
        d = k * SLOTS_PER_BAR
        if d > span:
            continue
        corr = stripe_diag(S, t0, d, known)
        peaks = [t for t in range(0, len(corr) - d + 1, SLOTS_PER_BAR)
                 if not np.isnan(corr[t]) and corr[t] >= REP_STRICT - 1e-9]
        occ, last = 0, -10**9
        for t in peaks:
            if t - last >= d:
                occ += 1
                last = t
        if occ >= 2:
            return d
    return minimal_period(S, roots, t0, hi, known) or span


CHOOSERS = {"auto": _auto, "fix2": _fixed(2), "fix4": _fixed(4), "fix8": _fixed(8),
            "coarse1st": _coarse_first}


def vocab_variant(bars, n_bars, *, tonic_pc=0, bpb=4, arm="auto"):
    if arm == "hier":
        return hier_sections(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    return _cover(bars, n_bars, tonic_pc, bpb, CHOOSERS[arm])


# ── (a) multi-scale: enumerate at every scale, then cover coarse-to-fine ──────

SCALES = (2, 4, 8)


def scale_inventory(bars, n_bars, *, tonic_pc=0, bpb=4, scales=SCALES):
    """Louis's "log the unique N-bar combos" step, for every N in ``scales``.

    At each scale: every bar-aligned window is a candidate pattern; windows are
    greedily clustered by chord-tone similarity (`stripe_diag` ≥ REP_STRICT), so
    a cluster is "one distinct N-bar combo" and its size is how often it recurs.

    Returns ``{n_bars_unit: [{"row0": slot, "count": int, "members": [slots]}]}``,
    sorted by count descending.
    """
    tokens, roots, known = build_slots(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    S = chord_ssm(tokens)
    out = {}
    for k in scales:
        d = k * SLOTS_PER_BAR
        starts = [b * SLOTS_PER_BAR for b in range(n_bars - k + 1)]
        unassigned = set(starts)
        clusters = []
        for t in starts:
            if t not in unassigned:
                continue
            corr = stripe_diag(S, t, d, known)
            members = [u for u in sorted(unassigned)
                       if not np.isnan(corr[u]) and corr[u] >= REP_STRICT - 1e-9]
            if t not in members:
                members.append(t)
            # non-overlapping occurrences only — a held chord matches itself at
            # every offset and would otherwise report a bogus recurrence count
            occ, last = [], -10**9
            for u in sorted(members):
                if u - last >= d:
                    occ.append(u)
                    last = u
            for u in members:
                unassigned.discard(u)
            clusters.append({"row0": t, "count": len(occ), "members": occ})
        clusters.sort(key=lambda c: (-c["count"], c["row0"]))
        out[k] = clusters
    return out


def survival(inv, scales=SCALES):
    """How many distinct patterns at each scale actually RECUR (count ≥ 2), and
    what share of the finer scale's recurring patterns are contained in a
    recurring pattern one scale up ("survives to the larger scale")."""
    rows = []
    for i, k in enumerate(scales):
        rec = [c for c in inv[k] if c["count"] >= 2]
        surv = None
        if i + 1 < len(scales):
            up = scales[i + 1]
            big = [c for c in inv[up] if c["count"] >= 2]
            spans = [(m, m + up * SLOTS_PER_BAR) for c in big for m in c["members"]]
            def inside(t, d=k * SLOTS_PER_BAR):
                return any(a <= t and t + d <= b for a, b in spans)
            if rec:
                surv = sum(1 for c in rec
                           if all(inside(m) for m in c["members"])) / len(rec)
        rows.append({"scale": k, "n_distinct": len(inv[k]), "n_recurring": len(rec),
                     "survives_to_next": surv})
    return rows


def hier_sections(bars, n_bars, *, tonic_pc=0, bpb=4, scales=SCALES):
    """Cover the song coarse-to-fine from the multi-scale inventory.

    Largest scale first: every pattern that recurs ≥2× claims its free bars,
    strongest (most recurrent) first. Then the next scale down explains what is
    left, and so on. Whatever survives to the end is a fragment and becomes its
    own one-off item — which is the honest reading of "a pattern that only exists
    at 2 bars is a fragment".
    """
    if n_bars < 8:
        return None
    tokens, roots, known = build_slots(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    if not any(tokens):
        return None
    S = chord_ssm(tokens)
    inv = scale_inventory(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb, scales=scales)

    covered: list[str | None] = [None] * n_bars
    pats: list[dict] = []
    for k in sorted(scales, reverse=True):
        d = k * SLOTS_PER_BAR
        for c in inv[k]:
            if c["count"] < 2 or len(pats) >= len(_LETTERS):
                continue
            # The shipped loop's load-bearing ordering: the vocabulary we already
            # have gets first refusal — strictly, then at the looser measured
            # threshold — BEFORE a new letter is minted. Dropping it was worth
            # +2.4 letters/chart in the first sweep.
            for thresh in (REP_STRICT, CONTINUE):
                for p in pats:
                    p["spans"] = sorted(set(p["spans"] + _claim(
                        p["corr"], p["d"], n_bars, covered, p["label"], thresh)))
            corr = stripe_diag(S, c["row0"], d, known)
            label = _LETTERS[len(pats)]
            got = _claim(corr, d, n_bars, covered, label, REP_STRICT)
            if len(got) < 2:                     # lost its recurrence to earlier claims
                for b in range(n_bars):
                    if covered[b] == label:
                        covered[b] = None
                continue
            pats.append({"label": label, "row0": c["row0"], "d": d, "corr": corr,
                         "spans": got})
    for thresh in (REP_STRICT, CONTINUE):
        for p in pats:
            p["spans"] = sorted(set(p["spans"] + _claim(
                p["corr"], p["d"], n_bars, covered, p["label"], thresh)))
    # leftovers: the hole is its own pattern (a bridge / a tail / an intro)
    b = 0
    while b < n_bars and len(pats) < len(_LETTERS):
        if covered[b] is not None:
            b += 1
            continue
        e = b
        while e < n_bars and covered[e] is None:
            e += 1
        t0, d = b * SLOTS_PER_BAR, (e - b) * SLOTS_PER_BAR
        corr = stripe_diag(S, t0, d, known)
        label = _LETTERS[len(pats)]
        got = _claim(corr, d, n_bars, covered, label, REP_STRICT)
        if got:
            pats.append({"label": label, "row0": t0, "d": d, "corr": corr,
                         "spans": got})
        b = e
    return _finish(S, tokens, known, covered, pats, n_bars)
