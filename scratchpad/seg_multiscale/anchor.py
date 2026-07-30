"""Boundary-anchored, backwards-propagated sections.

Louis, 2026-07-30: *"la BONNE façon de segmenter une section, c'est de partir de
la zone de changement avec la section suivante (donnée par le combo SSM flou +
SSM sharp), puis de retourner en répétitions arrière — c'est ça qui nous évite
de commencer une section sur le mauvais beat 0"*.

The shipped detector grows FORWARD: it takes the first uncovered bar as a
pattern start, so if that bar is the wrong phase every section downstream is
shifted. This module never chooses a start. It chooses BOUNDARIES — which are
locally detectable events — and *derives* each start by stepping backwards in
whole repetitions from the boundary.

Three steps:

1. **Coarse** — Foote novelty on the Gaussian-blurred chord-tone SSM gives change
   ZONES (``blur.change_zones``), accurate to about ±σ.
2. **Sharp** — inside each zone, the un-blurred SSM picks the exact bar line
   (``blur.pin``).
3. **Backwards** — from each pinned boundary B, find the smallest whole-bar
   period p such that the p bars ENDING at B repeat the p bars before them, then
   walk back B-p, B-2p, … for as long as the repetition holds.

The union of the walk-backs is the section-start set. What is left over at the
front and the back is intro/outro material (see ``absorb_edges``).
"""
from __future__ import annotations

import numpy as np

from harmonia.models.section_vocab import (
    CONTINUE, REP_STRICT, SLOTS_PER_BAR, _LETTERS, build_slots, chord_ssm,
    stripe_diag,
)

from blur import change_zones, gaussian_blur, lag_profile, novelty, peaks, pin

MIN_PERIOD_BARS = 2
MAX_PERIOD_BARS = 32


def _match(S, a: int, b: int, d_slots: int, known) -> float:
    """Mean similarity between the d-slot blocks starting at slots ``a`` and ``b``.
    NaN when too few named slots to judge — never 0, which would read as
    'measured, and different'."""
    vals = [S[a + i, b + i] for i in range(d_slots)
            if known is None or (known[a + i] and known[b + i])]
    if len(vals) * 2 < d_slots:
        return float("nan")
    return float(np.mean(vals))


PERIOD_MARGIN = 0.10


def backward_period(S, roots, known, B: int, free, thresh: float = REP_STRICT,
                    glob=None, glob_margin: float = 0.06):
    """Smallest whole-bar p such that bars ``[B-p, B)`` repeat ``[B-2p, B-p)``.

    Read backwards from the boundary on purpose: this is the ONE place where the
    phase is known for free, because a boundary is where the music changed and
    the material immediately before it is by definition the section that is
    ending.

    Two guards carried over verbatim from `minimal_period`, each paid for by a
    real failure there: the block must hold **≥ 2 distinct chords** (one held
    chord matches itself at every lag), and the winner is the SMALLEST p scoring
    within ``PERIOD_MARGIN`` of the best, never the argmax — a multiple of the
    true period always scores at least as well as the period.
    """
    band: dict[int, float] = {}
    for p in range(MIN_PERIOD_BARS, min(MAX_PERIOD_BARS, B // 2) + 1):
        a0, b0 = B - 2 * p, B - p
        if a0 < 0 or not all(free[x] for x in range(a0, B)):
            break
        if len(set(roots[b0 * SLOTS_PER_BAR:B * SLOTS_PER_BAR])) < 2:
            continue
        v = _match(S, a0 * SLOTS_PER_BAR, b0 * SLOTS_PER_BAR, p * SLOTS_PER_BAR, known)
        if not np.isnan(v):
            band[p] = v
    if not band or max(band.values()) < thresh:
        return None
    cut = max(thresh, max(band.values()) - PERIOD_MARGIN)
    ok = [p for p in sorted(band) if band[p] >= cut]
    if glob is not None and ok:
        # GLOBAL-PERIOD PRIOR. A period is a property of the SONG, not of one
        # boundary: a candidate that also shows up as a strong lag in the song's
        # own self-match profile is far likelier to be the real unit than one that
        # only happens to fit here. Keep the candidates within `glob_margin` of the
        # best global score, then take the smallest of those (never the argmax —
        # a multiple always scores at least as well).
        g = {p: float(glob[p * SLOTS_PER_BAR]) for p in ok
             if p * SLOTS_PER_BAR < len(glob)}
        if g:
            hi = max(g.values())
            ok = [p for p in ok if g.get(p, -1) >= hi - glob_margin] or ok
    return ok[0] if ok else None


def anchor_sections(bars, n_bars, *, tonic_pc=0, bpb=4, sigma_bars=4.0,
                    zones_from: str = "blur", global_period: bool = False,
                    debug: dict | None = None):
    """Boundaries first, starts derived. Returns the same shape as
    ``section_vocab.vocab_sections`` so it drops into the same scorer."""
    if n_bars < 8:
        return None
    tokens, roots, known = build_slots(bars, n_bars, tonic_pc=tonic_pc, bpb=bpb)
    if not any(tokens):
        return None
    S = chord_ssm(tokens)

    # 1-2. coarse zones -> sharp bar lines. The song end is always a boundary:
    # the last section ends there whether or not anything "changes", and it is
    # the STRONGEST anchor in the song because it needs no detection at all.
    # ABLATION HANDLE. `zones_from` decides where the coarse change zones come
    # from: the blurred matrix (Louis's proposal), the sharp matrix with the same
    # kernel, or nowhere at all — in which case the ONLY anchor is the end of the
    # song, which needs no detection. If "none" scores like "blur", the blur is
    # not what is doing the work.
    nov_b = novelty(gaussian_blur(S, sigma_bars), sigma_bars)
    if zones_from == "none":
        zones = []
    elif zones_from == "sharp":
        zones = peaks(novelty(S, sigma_bars), sigma_bars)
    else:
        zones = change_zones(S, sigma_bars)
    pinned = {pin(S, z, n_bars, radius_bars=max(1.0, sigma_bars / 2)) for z in zones}
    pinned.discard(0)
    order = sorted(pinned, key=lambda b: -nov_b[min(b * SLOTS_PER_BAR, len(nov_b) - 1)])
    order = [n_bars] + [b for b in order if b != n_bars]

    # 3. backwards propagation, with CLAIM discipline. Without it the walk-backs
    # from different boundaries are different grids laid over each other and the
    # union of their starts shreds the song (This Love: 24 starts, 17 letters).
    # A walk-back may only ever claim FREE bars, and a boundary whose own tail is
    # already claimed is skipped — the stronger boundary already explained it.
    free = [True] * n_bars
    blocks: list[tuple[int, int]] = []
    periods: dict[int, int] = {}
    glob = lag_profile(S, known) if global_period else None
    for B in order:
        p = backward_period(S, roots, known, B, free, glob=glob)
        if p is None:
            continue
        periods[B] = p
        t = B
        while t - p >= 0 and all(free[x] for x in range(t - p, t)):
            blocks.append((t - p, t))
            for x in range(t - p, t):
                free[x] = False
            if t - 2 * p < 0:
                break
            v = _match(S, (t - 2 * p) * SLOTS_PER_BAR, (t - p) * SLOTS_PER_BAR,
                       p * SLOTS_PER_BAR, known)
            if np.isnan(v) or v < CONTINUE:
                break
            t -= p
    # leftovers become their own blocks — an intro, a bridge, a tag
    b = 0
    while b < n_bars:
        if not free[b]:
            b += 1
            continue
        e = b
        while e < n_bars and free[e]:
            e += 1
        blocks.append((b, e))
        b = e
    blocks.sort()
    if debug is not None:
        debug.update({"zones": zones, "pinned": sorted(pinned), "periods": periods,
                      "blocks": blocks, "S": S, "known": known, "tokens": tokens})
    if len(blocks) < 2:
        return None

    # ── label the blocks: equal-length blocks that match get the same letter ──
    labels: list[str | None] = [None] * len(blocks)
    nxt = 0
    for i, (a0, a1) in enumerate(blocks):
        if labels[i] is not None:
            continue
        if nxt >= len(_LETTERS):
            break
        labels[i] = _LETTERS[nxt]
        nxt += 1
        da = (a1 - a0) * SLOTS_PER_BAR
        for j in range(i + 1, len(blocks)):
            b0, b1 = blocks[j]
            if labels[j] is not None or (b1 - b0) != (a1 - a0):
                continue
            v = _match(S, a0 * SLOTS_PER_BAR, b0 * SLOTS_PER_BAR, da, known)
            if not np.isnan(v) and v >= CONTINUE:
                labels[j] = labels[i]
    for i in range(len(labels)):
        if labels[i] is None:
            labels[i] = _LETTERS[min(nxt, len(_LETTERS) - 1)]
            nxt += 1

    # merge adjacent same-letter blocks into one run, as `vocab_sections` does
    out: list[dict] = []
    for (a, b), lab in zip(blocks, labels):
        if out and out[-1]["label"] == lab and out[-1]["bar1"] == a:
            out[-1]["bar1"] = b
            out[-1]["reps"] += 1
        else:
            out.append({"label": lab, "bar0": a, "bar1": b, "reps": 1,
                        "d_bars": b - a})
    for s in out:
        s["d_bars"] = max(1, (s["bar1"] - s["bar0"]) // max(1, s["reps"]))
        s["reps"] = max(1, (s["bar1"] - s["bar0"]) // s["d_bars"])
        t0 = s["bar0"] * SLOTS_PER_BAR
        s["tokens"] = tokens[t0:t0 + s["d_bars"] * SLOTS_PER_BAR]
    if len(out) < 2:
        return None
    return out


# ── (c) intro / outro absorption ─────────────────────────────────────────────

def absorb_edges(sections, n_bars, *, min_intro_share=0.0):
    """Push non-conforming material at the START or the END into Intro / Outro.

    Louis: *"just shove the uneven repeats that don't coincide with the rest of
    the song's logic repeat patterns in the intro and outro"*.

    A section is *conforming* when its letter is the dominant one or its letter
    recurs at least twice anywhere in the song. Leading non-conforming sections
    become ``Intro``; trailing ones ``Outro``.

    **A non-conforming section in the MIDDLE is a bridge and is left alone.**
    Sweeping it anywhere would delete real music. The count of those is returned
    so it can be reported rather than hidden.
    """
    if not sections:
        return sections, {"intro": 0, "outro": 0, "middle": 0}
    occ: dict[str, int] = {}
    for s in sections:
        occ[s["label"]] = occ.get(s["label"], 0) + 1
    conforming = [i for i, s in enumerate(sections) if occ[s["label"]] >= 2]
    if not conforming:
        return sections, {"intro": 0, "outro": 0, "middle": 0}
    first, last = conforming[0], conforming[-1]
    middle = sum(1 for i in range(first, last + 1) if occ[sections[i]["label"]] < 2)
    out = [dict(s) for s in sections]
    for i in range(first):
        out[i]["label"] = "Intro"
    for i in range(last + 1, len(out)):
        out[i]["label"] = "Outro"
    return out, {"intro": first, "outro": len(out) - last - 1, "middle": middle}
