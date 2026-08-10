"""harmonia_min/folding.py — sub-section folding, phase 1: OBSERVATION STACKING.

Louis's spec (2026-07-31): the fold's purpose is double — (1) stack the bars
sitting at the same position of a repeating sub-section so each chord gets
MANY observations instead of one (noise ↓ ~√n), (2) display synthesis
(cosmetic, NOT built yet). This module is (1) only:

  detect the internal period of each section (This Love's A = a 4-bar loop,
  B = a 2-bar cell) → stack same-position bars across every occurrence and
  every same-letter section → AVERAGE their musx frame posteriors → run the
  musx beat-grid decode a SECOND time on the averaged template → write the
  template's chords back onto every contributing bar.

Hard-coded exception (Louis): the LAST bar/cell of a section sometimes
changes to transition into the next section — any member too far from its
stack centroid is left UNFOLDED (variant), it keeps its own first-pass decode.

Thresholds MEASURED (2026-07-31/08-01 studies, This Love / Let It Be / Close):
  * PERIOD_MIN_SCORE = 0.80 — real loops score 0.86–0.94 (This Love A 0.944,
    B 0.859, Let It Be A 0.904); Close to You's through-composed sections
    score 0.65–0.77 and stay honestly UNFOLDED.
  * OUTLIER_Z = 3.0 — Louis's individual-vs-collective deviation rule (see
    the constant below); true variants measure z=5.7–38, normal members ≤1.7.
  * STACK_COHERENCE = 0.85 — median pairwise cos per position (bimodal guard).
Two-song-family calibration = hypothesis (CLAUDE.md rule #5).
"""
from __future__ import annotations

import logging
import os

import numpy as np

from harmonia_min import musx as _musx
from harmonia_min.labels import to_chord

logger = logging.getLogger(__name__)

PERIOD_MIN_SCORE = 0.80
OUTLIER_Z = 3.0          # Louis's refined rule (2026-08-01): compare each
                         # member's INDIVIDUAL deviation from the stack
                         # centroid to the COLLECTIVE norm (median + MAD of
                         # all members' deviations). Above median +
                         # OUTLIER_Z*MAD = VARIANT, keeps its own first-pass
                         # decode. Transition bars (section ends) are where
                         # variants are EXPECTED, but the test runs on every
                         # member — interior cadence cells (This Love's Ab G
                         # inside the final B, z≈38) are caught too.
                         # Measured: true variants z=5.7–38, normal members
                         # z≤1.7. Replaces the absolute cosine member gate
                         # AND the VAR_MAX position skip.
STACK_COHERENCE = 0.85   # per-position MEDIAN PAIRWISE cos of the gated
                         # stack must reach this, else the letter does NOT
                         # fold. Mean-to-centroid was tautological after the
                         # member gate; the median pairwise separates cleanly
                         # (measured: This Love positions 0.88-0.94, Let It
                         # Be's verse+chorus composite 0.74-0.81, Stand By
                         # Me 0.80 — the mixed stacks that rewrote real
                         # content, "Am F" -> "F C", both fall under 0.85).
PERIODS = (2, 4, 8)
CV_MAX = 0.51            # Louis's validated metric (2026-08-01): CV =
                         # std/mean of the RAW half-bar chroma across the
                         # gated stack members, per half-bar — dimensionless,
                         # volume-invariant (measured: var/mean scaled 2x
                         # with a 2x gain; CV didn't). Threshold = 5th
                         # percentile of 400 deliberately-MIXED stacks over
                         # 5 songs → false-merge rate 5.0% by construction
                         # (Louis's requirement: <5%). Corpus-wide 52% of
                         # same-written-chord stacks pass; ALL currently
                         # validated folds (This Love A/B, CV 0.31-0.48)
                         # pass. A position failing on either half-bar is
                         # NOT squashed.


def _bar_vecs(F: np.ndarray, n_bars: int) -> np.ndarray:
    """(n_bars, 48) unit vectors from the half-bar features (2 rows per bar)."""
    V = np.array([F[2 * b:2 * b + 2].reshape(-1) for b in range(n_bars)])
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def section_period(Vb: np.ndarray, b0: int, b1: int) -> tuple[int | None, float]:
    """Smallest confident repeating period of a section, or (None, best)."""
    L = b1 - b0 + 1
    scores = {}
    for P in PERIODS:
        if L < 2 * P:
            continue
        scores[P] = float(np.mean([Vb[b] @ Vb[b + P]
                                   for b in range(b0, b1 + 1 - P)]))
    if not scores:
        return None, 0.0
    best = max(scores.values())
    if best < PERIOD_MIN_SCORE:
        return None, best
    pick = min(P for P, v in scores.items() if v >= max(PERIOD_MIN_SCORE,
                                                        0.95 * best))
    return pick, scores[pick]


def _resample(mat: np.ndarray, n_out: int) -> np.ndarray:
    """Linear time-resample of a (T, k) posterior block to (n_out, k)."""
    T = mat.shape[0]
    if T == 0:
        return np.zeros((n_out, mat.shape[1]), dtype=mat.dtype)
    if T == 1:
        return np.repeat(mat, n_out, axis=0)
    xs = np.linspace(0, T - 1, n_out)
    lo = np.floor(xs).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (xs - lo)[:, None]
    return (1 - w) * mat[lo] + w * mat[hi]


def fold_letter_groups(sections, bars, grid, probs, bpb: int,
                       arr=None, times=None) -> dict:
    """Stack + re-decode + redistribute, per letter group. Mutates `bars`
    IN PLACE (each bar list object is shared with the section slices).

    Returns the fold report: {letter: {period, positions: n_obs list,
    variants: [bar indices], changed: [bar indices]}} — the data a future
    validation UI ("interface ludique") will present.
    """
    from harmonia_min.sections import halfbar_features
    n_bars = len(grid) - 1
    F = halfbar_features(grid, arr, times)   # same substrate as detection
    Vb = _bar_vecs(F, n_bars)

    def _raw_half(b: int, half: int):
        mid = 0.5 * (grid[b] + grid[b + 1])
        t0, t1 = ((grid[b], mid), (mid, grid[b + 1]))[half]
        sel = (times >= t0) & (times < t1)
        return arr[sel].mean(0) if sel.any() else \
            arr[int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))]
    # frame slices per bar on the musx grid
    def bar_probs(b):
        a = max(0, int(round(grid[b] / _musx.FRAME_DT)))
        z = min(probs[0].shape[0], int(round(grid[b + 1] / _musx.FRAME_DT)))
        return [p[a:z] for p in probs]

    med_bar = float(np.median(np.diff(grid)))
    Lf = max(bpb, int(round(med_bar / _musx.FRAME_DT)))   # frames per bar

    # ── group sections by letter, pick the group period ─────────────────────
    groups: dict[str, list[dict]] = {}
    for s in sections:
        groups.setdefault(s["label"], []).append(s)
    report = {}
    for letter, secs in groups.items():
        picks = []
        for s in secs:
            b0, b1 = s["barRanges"][0]
            P, sc = section_period(Vb, b0, b1)
            if P is not None:
                picks.append(P)
        if not picks:
            report[letter] = {"period": None, "reason": "no confident loop"}
            continue
        P = int(np.bincount(picks).argmax())      # group consensus period

        # ── stacks: position k -> member bar indices ─────────────────────────
        pos_members: list[list[int]] = [[] for _ in range(P)]
        section_last = set()
        for s in secs:
            b0, b1 = s["barRanges"][0]
            section_last.add(b1)
            for b in range(b0, b1 + 1):
                pos_members[(b - b0) % P].append(b)

        variants, gated = [], [[] for _ in range(P)]
        for k in range(P):
            mem = pos_members[k]
            if len(mem) < 3:
                gated[k] = mem
                continue
            cen = np.mean([Vb[b] for b in mem], axis=0)
            cen /= max(np.linalg.norm(cen), 1e-9)
            d = {b: 1.0 - float(Vb[b] @ cen) for b in mem}
            med = float(np.median(list(d.values())))
            mad = float(np.median(np.abs(np.array(list(d.values())) - med))) + 1e-6
            for b in mem:
                if (d[b] - med) / mad <= OUTLIER_Z:
                    gated[k].append(b)
                else:
                    variants.append(b)            # écart individuel anormal vs
                                                  # l'écart collectif → variante,
                                                  # garde son décodage 1ʳᵉ passe
        if sum(len(g) for g in gated) < 2 * P:
            report[letter] = {"period": P, "reason": "too few gated members"}
            continue
        coh = []
        for k in range(P):
            g = gated[k]
            if len(g) < 2:
                continue
            pw = [float(Vb[a] @ Vb[b]) for i, a in enumerate(g) for b in g[i + 1:]]
            coh.append(float(np.median(pw)))
        if not coh or min(coh) < STACK_COHERENCE:
            report[letter] = {"period": P,
                              "reason": f"stack incoherent (min median pairwise "
                                        f"{min(coh):.2f})" if coh else "stacks too thin"}
            continue

        # CV squash-verifier (Louis's metric, calibrated FP<5%): a position
        # whose gated members vary too much on either half-bar is NOT
        # squashed — every occurrence keeps its own decode.
        cv_skip = set()
        for k in range(P):
            g = gated[k]
            if len(g) < 2:
                continue
            for half in (0, 1):
                X = np.array([_raw_half(b, half) for b in g])
                c_v = float(np.sqrt(X.var(0).mean())
                            / max(X.mean(0).mean(), 1e-9))
                if c_v > CV_MAX:
                    cv_skip.add(k)
        if cv_skip:
            logger.info("fold %s: positions %s not squashed (CV > %.2f)",
                        letter, sorted(cv_skip), CV_MAX)

        # Audit note (2026-08-01): a position with ZERO gated members borrows
        # one rejected variant bar as decode CONTEXT only — the write loop
        # below iterates gated[k], so such a position is never rewritten.
        # Accepted risk: the variant's posteriors mildly colour the
        # neighbouring positions' transitions in the template decode.
        pos_chords = _template_chords(
            [g or [pos_members[k][0]] for k, g in enumerate(gated)],
            bar_probs, len(probs), Lf, bpb, P)
        if pos_chords is None:
            report[letter] = {"period": P, "reason": "template decoded empty"}
            continue

        n_obs = [len(g) for g in gated]
        changed = []
        for k in range(P):
            if not pos_chords[k] or k in cv_skip:
                continue                          # empty or CV-refused slot
            for b in gated[k]:
                if _write_position(bars, grid, b, pos_chords[k], bpb, n_obs[k]):
                    changed.append(b)
        report[letter] = {"period": P, "n_obs": n_obs,
                          "variants": sorted(set(variants)),
                          "changed": sorted(set(changed))}
        logger.info("fold %s: P=%d, obs/pos %s, %d variants, %d bars changed",
                    letter, P, n_obs, len(set(variants)), len(set(changed)))
    return report




def _template_chords(pos_members, bar_probs, n_probs, Lf, bpb, P):
    """Average each position's member-bar posteriors, decode the P-bar
    template once (tiled ×3 against Viterbi edge effects), return the
    per-position chord lists (sustains write a carry at beat 0), or None
    if the template decodes empty."""
    tmpl = []
    for k in range(P):
        mems = [bar_probs(b) for b in pos_members[k]]
        avg = [np.mean([_resample(m[i], Lf) for m in mems], axis=0)
               for i in range(n_probs)]
        tmpl.append(avg)
    cat = [np.concatenate([tmpl[k][i] for k in range(P)] * 3)
           for i in range(n_probs)]
    step = Lf * _musx.FRAME_DT / bpb
    beats = [i * step for i in range(3 * P * bpb + 1)]
    # Half-bar transitions cost the same as bar transitions (15), quarter-
    # bar stays expensive (100): Louis 2026-08-01 — the aggregated posterior
    # showed Ddim lasting only HALF its window but the default mid-bar cost
    # (45) glued it to a full bar. Consistent with the half-bar snap rule.
    # NOTE (2026-08-07): since the half-bar-only zeroing, "expensive (100)"
    # had silently become FORBIDDEN — the grade-4 beats were dropped before
    # the penalty could apply. Restriction lifted the same day (Louis: « on ne
    # met plus de restrictions sur la granularité »): the documented cost is
    # the real one again, so the template decode cannot erase sub-half-bar
    # chords the first pass found. HARMONIA_QUARTER_BAR=off restores the
    # half-bar-only template decode.
    _q = None if os.environ.get("HARMONIA_QUARTER_BAR", "").strip().lower() \
        in ("off", "0", "false") else "all"
    lab, _ = _musx.redecode(beats, cat, downbeat_times=beats[::bpb],
                            beat_trans_penalty=(15.0, 15.0, 100.0),
                            quarter_beats=_q)
    T0, T1 = P * Lf * _musx.FRAME_DT, 2 * P * Lf * _musx.FRAME_DT
    # The window that keeps the MIDDLE copy of the ×3-tiled template needs a
    # tolerance of half a musx frame, not 1e-6 s (Louis, 2026-08-09: « pk sur
    # This Love on loupe le Cm du refrain ? »). `redecode` returns boundaries
    # snapped to musx's own 23.2 ms frame grid, so the chord that starts
    # exactly ON the window edge lands up to half a frame BEFORE it — the
    # chorus's Cm came out of the Viterbi at 5.052 s against T0 = 5.062 s and
    # was thrown away for 10 ms, leaving the bar showing only its second
    # chord. Cost of the 1e-6 tolerance, measured across the served library:
    # 18 chords heard with 0.85–0.94 confidence and silently dropped, ALL of
    # them on the first beat of fold position 0 — the only slot that sits on
    # the window edge. The decode itself was never wrong: on This Love the
    # Viterbi returns Cm Fm Bb Eb exactly, aggregation and all.
    edge_tol = _musx.FRAME_DT / 2
    events = []
    for t0, t1, l in lab:
        if t0 < T0 - edge_tol or t0 >= T1 - edge_tol:
            continue
        beat = int(round((max(t0, T0) - T0) / step))
        # ACOUSTIC confidence, measured on the averaged template this decode
        # actually read — the same quantity, on the same scale, as an unfolded
        # chord's. Averaging n members legitimately raises it, so a folded
        # chord scores higher for the right reason instead of being handed
        # `0.5 + 0.08*n_obs`, which is a repetition count wearing a posterior's
        # clothes (measured cost: 2 points of AUC, see musx.label_confidence).
        conf = round(_musx.label_confidence(cat[0], t0, min(t1, T1), l), 3)
        events.append((beat // bpb, beat % bpb, l, conf))
    if not events or all(l == "N" for _, _, l, _ in events):
        return None
    pos_chords: list[list[dict]] = [[] for _ in range(P)]
    cur = None
    for k in range(P):
        evk = [e for e in events if e[0] == k]
        if (not evk or evk[0][1] > 0) and cur is not None:
            pos_chords[k].append({**cur, "beat": 0, "carry": True})
        for _, beat, l, conf in evk:
            ch = to_chord(l)
            entry = ({"root": 0, "q": "", "bass": -1, "nc": True,
                      "beat": beat, "c": conf}
                     if ch is None else {**ch, "nc": False, "beat": beat,
                                         "c": conf})
            pos_chords[k].append(entry)
            cur = {kk: vv for kk, vv in entry.items()
                   if kk not in ("beat", "carry")}
    return pos_chords


def _write_position(bars, grid, b, chords_k, bpb, n_obs):
    """Rewrite bar b with a template position's chords (real bar times).

    `c` comes per-chord from the template decode (an acoustic posterior on the
    averaged evidence); `n_obs` is carried ALONGSIDE it, never folded into it.
    Two chords in one bar can be supported to different degrees and must be
    allowed to say so — the old signature took one confidence for the whole bar
    because that confidence was a property of the section, not of the chord.
    """
    bw = grid[b + 1] - grid[b]
    new = []
    for j, e in enumerate(chords_k):
        t0 = grid[b] + e["beat"] / bpb * bw
        nxt = (chords_k[j + 1]["beat"] / bpb * bw
               if j + 1 < len(chords_k) else bw)
        new.append({"root": e["root"], "q": e["q"], "bass": e["bass"],
                    "nc": e["nc"], "carry": bool(e.get("carry")),
                    "beat": e["beat"], "bar": b,
                    "c": float(e.get("c", 0.5)), "n_obs": n_obs,
                    "folded": True,
                    "t0": round(t0, 3), "t1": round(grid[b] + nxt, 3)})
    changed = [(c["root"], c["q"], c.get("carry", False)) for c in bars[b]] \
        != [(c["root"], c["q"], c.get("carry", False)) for c in new]
    bars[b][:] = new
    return changed


# ── phase 2: DISPLAY fold (Louis, 2026-08-01) ────────────────────────────────

def display_fold(sections: list[dict], bars, grid, probs=None, bpb=4,
                 loop_folded: set | None = None) -> list[dict]:
    """Write a repeated section ONCE ×N — the ChartModel reps/spans/barSpans
    contract app_shell already renders — with the divergent tail (≤2 last
    bars) carried per variant as `endings` ("le débordement en dessous").

    UNDER-FOLD doctrine (Louis, 2026-07-30): only same-letter sections of the
    SAME length whose bars agree everywhere except the last ≤2 fold together;
    a pass of a different length stays written out (This Love: B8+B8 fold ×2,
    the 24-bar final B stays its own block).
    """
    def barsig(b):
        return tuple((c["root"], c["q"], c["nc"]) for c in bars[b])

    used = [False] * len(sections)
    out = []
    for i, sec in enumerate(sections):
        if used[i]:
            continue
        b0, b1 = sec["barRanges"][0]
        L = b1 - b0 + 1
        group = [i]
        for j in range(i + 1, len(sections)):
            if used[j] or sections[j]["label"] != sec["label"]:
                continue
            c0, c1 = sections[j]["barRanges"][0]
            if c1 - c0 + 1 != L:
                continue
            diff = [r for r in range(L) if barsig(b0 + r) != barsig(c0 + r)]
            if all(r >= L - 2 for r in diff):
                group.append(j)
        if len(group) == 1:
            out.append(sec)
            used[i] = True
            continue
        for j in group:
            used[j] = True
        ranges = [sections[j]["barRanges"][0] for j in group]
        # ── CROSS-PASS observation stacking (Louis, 2026-08-01): the folded
        # passes' musx posteriors are averaged position by position and
        # re-decoded, so the folded block shows a CONSENSUS of all passes —
        # not pass 0's chords. Skipped when the letter already loop-folded in
        # phase 1 (its bars are already a template consensus pooled across
        # every occurrence, including unfolded-length passes like This Love's
        # 24-bar final B — a broader pool than these passes alone). The
        # divergent tail (endings) keeps each pass's own decode.
        tail_probe = 0
        for c0, _ in ranges[1:]:
            diff = [r for r in range(L) if barsig(b0 + r) != barsig(c0 + r)]
            if diff:
                tail_probe = max(tail_probe, L - min(diff))
        if probs is not None and sec["label"] not in (loop_folded or set()):
            med_bar = float(np.median(np.diff(grid)))
            Lf = max(bpb, int(round(med_bar / _musx.FRAME_DT)))

            def bar_probs(b):
                a = max(0, int(round(grid[b] / _musx.FRAME_DT)))
                z = min(probs[0].shape[0],
                        int(round(grid[b + 1] / _musx.FRAME_DT)))
                return [pp[a:z] for pp in probs]

            Pp = L - tail_probe
            if Pp < 1:
                Pp = 0                            # nothing shared: no stack
            members = [[c0 + r for c0, _ in ranges] for r in range(Pp)]
            pos_chords = _template_chords(members, bar_probs, len(probs),
                                          Lf, bpb, Pp) if Pp >= 1 else None
            if pos_chords is not None:
                k_obs = len(ranges)
                nch = 0
                for r in range(Pp):
                    if not pos_chords[r]:
                        continue
                    for c0, _ in ranges:
                        nch += _write_position(bars, grid, c0 + r,
                                               pos_chords[r], bpb, k_obs)
                logger.info("display fold %s: cross-pass stack ×%d, %d bars "
                            "rewritten to consensus", sec["label"], k_obs, nch)
        # tail depth = deepest divergence vs pass 0, within the last 2 bars
        tail = 0
        for c0, _ in ranges[1:]:
            diff = [r for r in range(L) if barsig(b0 + r) != barsig(c0 + r)]
            if diff:
                tail = max(tail, L - min(diff))
        prefix_len = L - tail
        folded = {
            "id": sec["id"], "label": sec["label"], "tag": "",
            "reps": len(group),
            "spans": [[grid[c0], grid[c1 + 1]] for c0, c1 in ranges],
            "barRanges": [[c0, c1] for c0, c1 in ranges],
            "bars": bars[b0:b0 + prefix_len]
                    + bars[b0 + prefix_len:b1 + 1],   # prefix + rep tail
            "barSpans": [[[grid[c0 + r], grid[c0 + r + 1]] for c0, _ in ranges]
                         for r in range(prefix_len)],
        }
        if tail:
            # variants: passes grouped by their tail content; each variant's
            # bars come from ITS first pass (absolute times of that pass)
            bytail: dict[tuple, list[int]] = {}
            for pi, (c0, _) in enumerate(ranges):
                key = tuple(barsig(c0 + r) for r in range(prefix_len, L))
                bytail.setdefault(key, []).append(pi)
            variants = []
            for key, passes in sorted(bytail.items(), key=lambda kv: kv[1][0]):
                v0 = ranges[passes[0]][0]
                variants.append({"passes": passes,
                                 "bars": bars[v0 + prefix_len:v0 + L]})
                for r in range(prefix_len, L):
                    folded["barSpans"].append(
                        [[grid[ranges[pi][0] + r], grid[ranges[pi][0] + r + 1]]
                         for pi in passes])
            folded["endings"] = {"tail": tail, "variants": variants}
            # representative block keeps the FULL bar list (prefix + pass-0
            # tail) — an endings-aware renderer slices bars[:-tail] itself
        logger.info("display fold: %s ×%d (tail %d, %d variant(s))",
                    sec["label"], len(group), tail,
                    len(folded.get("endings", {}).get("variants", [])) or 1)
        out.append(folded)
    return out


# ── phase 2b: MINIMAL fold into the EXISTING app UI (Louis, 2026-08-02) ─────

def minimal_fold(sections, bars, grid, fold_report) -> list[dict]:
    """One ChartModel section per LETTER — the validated minimal folding,
    rendered by the UNCHANGED app_shell (Louis: the folding logic is right,
    the old UI stays). Golden rule: chords written once.

    Block per letter (validated on /min): the fold cell padded to >=4 bars;
    if pass tails diverge — equal-length passes: the whole representative
    pass (This Love B = 3 cells + cadence); unequal: cell + divergent tail
    cell (This Love A = 8 bars, « 8 suffisent largement »).
    reps = number of passes; spans/barRanges per pass (chronological);
    barSpans[r] maps rendered bar r onto each pass PROPORTIONALLY (the old
    app's own contract for length-mismatched passes). The FORM strip then
    shows the sequence — the timeline that fait foi.
    """
    def sig(b):
        return tuple((c["root"], c["q"], c["nc"]) for c in bars[b]) \
            if 0 <= b < len(bars) else ()

    def t_at(b0, x):                      # grid time at fractional bar b0+x
        i = int(x)
        f = x - i
        lo = grid[min(len(grid) - 1, b0 + i)]
        hi = grid[min(len(grid) - 1, b0 + i + 1)]
        return round(lo + f * (hi - lo), 4)

    by_letter: dict[str, list] = {}
    order: list[str] = []
    for s in sections:
        L = s["label"]
        if L not in by_letter:
            order.append(L)
        by_letter.setdefault(L, []).extend(
            [tuple(r) for r in s["barRanges"]])
    out = []
    def _evidence(rng):
        """How many REAL chord onsets a pass carries (carries and N.C. don't
        count) — what makes a pass worth showing as the letter's block."""
        c0, c1 = rng
        return sum(1 for b in range(c0, c1 + 1) for c in (bars[b] if 0 <= b < len(bars) else [])
                   if not c["nc"] and not c.get("carry"))

    for L in order:
        ranges = sorted(by_letter[L])
        # The block shown for a letter used to be `ranges[0]` — the
        # chronologically first pass, whatever it contained. On a song that
        # fades in, that is the WORST pass: Stand By Me's first A holds 2
        # onsets over 8 bars (the fade-in) while its eight later A's are
        # identical and complete (F#m D E A), so the chart rendered ~empty
        # while the song's chords were sitting in passes we never displayed
        # (Louis, 2026-08-10: « je vois plein d'accords, et sur le chart j'ai
        # juste des NC partout »). Pick the pass with the most real onsets
        # instead; ties keep the earliest, so every already-correct chart is
        # untouched (a letter whose passes agree has equal evidence).
        # `ranges` STAYS chronological — spans/barRanges/the playhead map all
        # depend on it. Only which pass is WRITTEN changes.
        b0, b1 = max(ranges, key=lambda r: (_evidence(r), -ranges.index(r)))
        # Only a fold that was ACCEPTED may drive the display. Every refusal
        # in fold_letter_groups still records the period it was testing
        # ({"period": P, "reason": ...} at the "too few gated members",
        # "stack incoherent" and "template decoded empty" exits) — reading
        # `period` without checking `reason` silently rebuilt the chart from a
        # loop the stacker had just rejected. Norah Jones' A was refused at
        # "stack incoherent (min median pairwise 0.61)" and displayed anyway:
        # P=2 wrote `Bb Bb | Eb^7 D` twice over a first pass that really goes
        # `Bb Bb | Eb^7 D | G-7 C7 | F7 F7`, so bars 2-7 never reached the
        # chart and 72.6% of A's playing time was misrepresented (2026-08-02).
        # Under-fold, never over-fold: a refused fold writes its bars out.
        _rep = fold_report.get(L) or {}
        P = None if _rep.get("reason") else _rep.get("period")
        cell = [bars[b] for b in range(b0, min(b1, b0 + P - 1) + 1)] if P \
            else [bars[b] for b in range(b0, b1 + 1)]
        ref_tail = [sig(ranges[0][1] - k) for k in (1, 0)]
        div = next(((c0, c1) for c0, c1 in ranges[1:]
                    if [sig(c1 - k) for k in (1, 0)] != ref_tail), None)
        lens = {c1 - c0 + 1 for c0, c1 in ranges}
        # Bars of the representative pass that the fold itself flagged as NOT
        # matching the cell. They are real music the cell does not contain, so
        # the written block has to reach them (Louis, 2026-08-05, on Norah
        # Jones: « tu me bouffes la répétition de la fin du A ; tu peux
        # techniquement l'intégrer au A mais il faut l'ÉCRIRE sur le chart, car
        # c'est pas clair là »). Her A was three passes of 22 / 8 / 20 bars
        # written as 4 bars × 3, with 13 bars — the 2-bar tag among them —
        # flagged as variants and never rendered. Under-fold, never over-fold.
        var_in_pass = [v for v in (_rep.get("variants") or []) if b0 <= v <= b1]
        if P and div and len(lens) == 1:
            block_rng = list(range(b0, b1 + 1))
        elif P:
            block_rng = list(range(b0, b0 + len(cell)))
            while len(block_rng) < min(4, b1 - b0 + 1):
                block_rng += list(range(b0, b0 + len(cell)))
            block_rng = block_rng[:max(len(cell), min(4, b1 - b0 + 1))]
            if var_in_pass:
                end = min(b1, max(var_in_pass))
                if end >= b0 + len(block_rng):
                    block_rng = list(range(b0, end + 1))
        else:
            block_rng = list(range(b0, b1 + 1))
        Lb = len(block_rng)
        is_pass_block = bool(P and div and len(lens) == 1)
        tail_len = 0                              # Louis 2026-08-02: le bloc
        tail_sigs = []                            # = la cellule, échelle de
                                                  # répétition minimale (×N
                                                  # vivent dans la strip)
        # ── playhead map BY CONTENT (fix 2026-08-02: the proportional map
        # stretched an 8-bar block over a 16-bar pass — by the cell's 2nd
        # repetition the highlight sat on the wrong rows). A cell row lights
        # at EVERY repetition of the cell inside every pass (multiple time
        # windows per row — the UI's tspans accept any number); tail rows
        # light only on the pass whose ending matches the divergent tail.
        rows = [[] for _ in range(Lb)]
        for c0, c1 in ranges:
            Lk = c1 - c0 + 1
            pass_tail = bool(tail_len) and \
                [sig(b) for b in range(c1 - tail_len + 1, c1 + 1)] == tail_sigs
            for b in range(c0, c1 + 1):
                if pass_tail and b > c1 - tail_len:
                    r = len(cell) + (b - (c1 - tail_len + 1))
                elif P and not is_pass_block:
                    r = (b - c0) % (len(cell) if tail_len else Lb)
                else:
                    r = min(Lb - 1, int((b - c0) * Lb / Lk))
                rows[r].append([grid[b], grid[min(len(grid) - 1, b + 1)]])
        out.append({
            "id": f"L{L}", "label": L, "tag": "", "reps": len(ranges),
            "spans": [[grid[c0], grid[min(len(grid) - 1, c1 + 1)]]
                      for c0, c1 in ranges],
            "barRanges": [[c0, c1] for c0, c1 in ranges],
            "bars": [bars[b] for b in block_rng],
            "barSpans": rows,
        })
    return out
