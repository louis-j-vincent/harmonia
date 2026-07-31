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

Thresholds MEASURED (2026-07-31 study, This Love / Let It Be / Close to You):
  * PERIOD_MIN_SCORE = 0.80 — real loops score 0.86–0.94 (This Love A 0.944,
    B 0.859, Let It Be A 0.904); Close to You's through-composed sections
    score 0.65–0.77 and stay honestly UNFOLDED.
  * STACK_COS = 0.85 — true stack members sit at 0.91+ (median 0.97),
    cross-position bars at ~0.68–0.75, and This Love's transition cell
    (Cm F7 | Ab G vs the Cm Fm | Bb Eb cell) lands at 0.74 → excluded,
    exactly the variant behaviour Louis described.
Two-song-family calibration = hypothesis (CLAUDE.md rule #5).
"""
from __future__ import annotations

import logging

import numpy as np

from harmonia_min import musx as _musx
from harmonia_min.labels import to_chord

logger = logging.getLogger(__name__)

PERIOD_MIN_SCORE = 0.80
STACK_COS = 0.85
STACK_COHERENCE = 0.85   # per-position MEDIAN PAIRWISE cos of the gated
                         # stack must reach this, else the letter does NOT
                         # fold. Mean-to-centroid was tautological after the
                         # member gate; the median pairwise separates cleanly
                         # (measured: This Love positions 0.88-0.94, Let It
                         # Be's verse+chorus composite 0.74-0.81, Stand By
                         # Me 0.80 — the mixed stacks that rewrote real
                         # content, "Am F" -> "F C", both fall under 0.85).
PERIODS = (2, 4, 8)


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
            if len(mem) < 2:
                gated[k] = mem
                continue
            cen = np.mean([Vb[b] for b in mem], axis=0)
            cen /= max(np.linalg.norm(cen), 1e-9)
            for b in mem:
                if float(Vb[b] @ cen) >= STACK_COS:
                    gated[k].append(b)
                else:
                    variants.append(b)            # transition bar / outlier:
                                                  # keeps its first-pass decode
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

        # ── averaged template posteriors, decoded ONCE (tiled ×3 to kill
        # Viterbi edge effects; the middle copy is read back) ────────────────
        tmpl = []
        for k in range(P):
            mems = [bar_probs(b) for b in gated[k]] or [bar_probs(pos_members[k][0])]
            avg = [np.mean([_resample(m[i], Lf) for m in mems], axis=0)
                   for i in range(len(probs))]
            tmpl.append(avg)
        cat = [np.concatenate([tmpl[k][i] for k in range(P)] * 3)
               for i in range(len(probs))]
        step = Lf * _musx.FRAME_DT / bpb
        beats = [i * step for i in range(3 * P * bpb + 1)]
        downs = beats[::bpb]
        lab, _ = _musx.redecode(beats, cat, downbeat_times=downs)
        # keep the middle tile, in template-local time
        T0, T1 = P * Lf * _musx.FRAME_DT, 2 * P * Lf * _musx.FRAME_DT
        events = []                               # (pos, beat_in_bar, label)
        for t0, t1, l in lab:
            if t0 < T0 - 1e-6 or t0 >= T1 - 1e-6:
                continue
            beat = int(round((t0 - T0) / step))
            events.append((beat // bpb, beat % bpb, l, min(t1, T1) - t0))
        if not events or all(l == "N" for _, _, l, _ in events):
            report[letter] = {"period": P, "reason": "template decoded empty"}
            continue

        # per-position chord lists (sustains write a carry at beat 0)
        pos_chords: list[list[dict]] = [[] for _ in range(P)]
        cur = None
        for k in range(P):
            evk = [e for e in events if e[0] == k]
            if (not evk or evk[0][1] > 0) and cur is not None:
                pos_chords[k].append({**cur, "beat": 0, "carry": True})
            for _, beat, l, dur in evk:
                ch = to_chord(l)
                if ch is None:
                    entry = {"root": 0, "q": "", "bass": -1, "nc": True,
                             "beat": beat}
                else:
                    entry = {**ch, "nc": False, "beat": beat}
                pos_chords[k].append(entry)
                cur = {kk: vv for kk, vv in entry.items()
                       if kk not in ("beat", "carry")}

        # ── redistribute onto every gated member bar ─────────────────────────
        n_obs = [len(g) for g in gated]
        changed = []
        for k in range(P):
            if not pos_chords[k]:
                continue                          # never write an empty bar
            conf = round(min(0.97, 0.5 + 0.08 * n_obs[k]), 3)
            for b in gated[k]:
                bw = grid[b + 1] - grid[b]
                new = []
                for j, e in enumerate(pos_chords[k]):
                    t0 = grid[b] + e["beat"] / bpb * bw
                    nxt = (pos_chords[k][j + 1]["beat"] / bpb * bw
                           if j + 1 < len(pos_chords[k]) else bw)
                    new.append({"root": e["root"], "q": e["q"],
                                "bass": e["bass"], "nc": e["nc"],
                                "carry": bool(e.get("carry")),
                                "beat": e["beat"], "bar": b,
                                "c": conf, "n_obs": n_obs[k], "folded": True,
                                "t0": round(t0, 3), "t1": round(grid[b] + nxt, 3)})
                old_sig = [(c["root"], c["q"], c.get("carry", False))
                           for c in bars[b]]
                new_sig = [(c["root"], c["q"], c.get("carry", False))
                           for c in new]
                if old_sig != new_sig:
                    changed.append(b)
                bars[b][:] = new                  # in place: sections see it
        report[letter] = {"period": P, "n_obs": n_obs,
                          "variants": sorted(set(variants)),
                          "changed": sorted(set(changed))}
        logger.info("fold %s: P=%d, obs/pos %s, %d variants, %d bars changed",
                    letter, P, n_obs, len(set(variants)), len(set(changed)))
    return report
