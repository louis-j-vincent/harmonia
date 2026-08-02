"""Measure candidate MERGE criteria on Billboard GT (890 tracks).

Question (Louis, 2026-08-02): which measurable criterion separates section
occurrences that may be written ONCE (a real repeat) from spans that must be
written out (genuinely different music)?

Decision unit = a PAIR of spans the folder would consider merging.
  positive      : same SALAMI letter, same prime level  (A vs A)
  prime         : same base letter, different prime      (A vs A')
  hard negative : different letter, chord-content Jaccard >= HARD_JAC
  easy negative : different letter, otherwise

Loss is asymmetric: a false merge destroys real music, a missed merge only
costs redundancy. Headline number per criterion = the threshold that keeps
false merges <= 2% of negatives, and the recall on positives there.

Two arms:
  LABEL arm    — criteria computed from Billboard's own Harte chord labels
                 (upper bound; isolates the criterion from decoder noise)
  CHROMA arm   — same spans, same criteria shapes, but computed on McGill's
                 NNLS `bothchroma.csv` (real audio features, no decoder).
                 The shipped constants (PERIOD_MIN_SCORE / STACK_COHERENCE /
                 CV_MAX) live on this substrate.

Outputs: scratchpad/fold_criteria/{pairs_label.npz,pairs_chroma.npz,summary.json}
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HARD_JAC = 0.6          # chord-set overlap that makes a different-letter pair "hard"
N_SLOT = 64             # time-normalised slots for the aligned criteria
MIN_SPAN = 8.0          # seconds; ignore stubs (intros/silence)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "scratchpad", "fold_criteria")


# ── chord labels → chord-tone vectors ────────────────────────────────────────
def chordtone_seq(labels, intervals):
    """(N,12) unit chord-tone vectors + intervals, one per labelled chord."""
    import mir_eval.chord as C
    V, iv, names = [], [], []
    for lab, (a, b) in zip(labels, intervals):
        if b - a <= 1e-6:
            continue
        try:
            root, bmap, _ = C.encode(lab)
        except Exception:
            root, bmap = -1, np.zeros(12)
        v = np.zeros(12)
        if root >= 0:
            for s in np.nonzero(bmap)[0]:
                v[(root + s) % 12] = 1.0
        V.append(v)
        iv.append((a, b))
        names.append(lab.split("/")[0])
    if not V:
        return np.zeros((0, 12)), np.zeros((0, 2)), []
    return np.array(V), np.array(iv), names


def dedup(names, iv):
    out_n, out_i = [], []
    for n, (a, b) in zip(names, iv):
        if out_n and out_n[-1] == n:
            out_i[-1][1] = b
        else:
            out_n.append(n)
            out_i.append([a, b])
    return out_n, np.array(out_i)


def slice_span(V, iv, t0, t1, names=None):
    m = (iv[:, 1] > t0) & (iv[:, 0] < t1)
    if not m.any():
        return None
    sv = V[m]
    si = np.clip(iv[m], t0, t1)
    sn = [n for n, k in zip(names, m) if k] if names is not None else None
    return sv, si, sn


def to_slots(V, iv, t0, t1, n=N_SLOT):
    """Time-normalise a span's chord-tone track onto n equal slots."""
    edges = np.linspace(t0, t1, n + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    idx = np.searchsorted(iv[:, 1], mid, side="right")
    idx = np.clip(idx, 0, len(V) - 1)
    return V[idx]


def onset_sig(iv, t0, t1, n=32, sigma=1.0):
    """Harmonic-rhythm signature: WHERE the chord changes fall, chord identity
    discarded. Gaussian-smoothed binary onset histogram on n normalised slots."""
    x = np.zeros(n)
    for a in iv[:, 0]:
        if a <= t0 + 1e-6 or a >= t1 - 1e-6:
            continue
        k = int((a - t0) / (t1 - t0) * n)
        x[min(n - 1, max(0, k))] += 1.0
    g = np.exp(-0.5 * (np.arange(-3, 4) / sigma) ** 2)
    x = np.convolve(x, g, mode="same")
    return x


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))


def slot_cos(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    ok = den > 1e-9
    return float(np.mean(num[ok] / den[ok])) if ok.any() else 0.0


def lev(a, b):
    """Levenshtein on token lists."""
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ── criteria ─────────────────────────────────────────────────────────────────
def label_criteria(sp1, sp2):
    (V1, i1, n1), (t0a, t1a) = sp1
    (V2, i2, n2), (t0b, t1b) = sp2
    d1, d2 = t1a - t0a, t1b - t0b
    dn1, di1 = dedup(n1, i1)
    dn2, di2 = dedup(n2, i2)
    S1, S2 = to_slots(V1, i1, t0a, t1a), to_slots(V2, i2, t0b, t1b)

    # tiling coverage: write span1 ONCE at its own natural length, loop it
    # over span2 (exactly what minimal_fold does), measure agreed TIME.
    m = int(np.ceil(d2 / max(d1, 1e-6)))
    grid = np.linspace(t0b, t1b, 257)[:-1]
    tiled = ((grid - t0b) % d1) + t0a
    ii = np.clip(np.searchsorted(i1[:, 1], tiled, side="right"), 0, len(V1) - 1)
    jj = np.clip(np.searchsorted(i2[:, 1], grid, side="right"), 0, len(V2) - 1)
    per = np.array([cos(V1[a], V2[b]) for a, b in zip(ii, jj)])
    tile_cov = float(np.mean(per >= 0.99))

    bg1 = set(zip(dn1, dn1[1:]))
    bg2 = set(zip(dn2, dn2[1:]))
    return {
        "len_agree": min(d1, d2) / max(d1, d2),
        "nchord_agree": min(len(dn1), len(dn2)) / max(len(dn1), len(dn2), 1),
        "seq_sim": 1.0 - lev(dn1, dn2) / max(len(dn1), len(dn2), 1),
        "ct_bag_cos": cos((V1 * (i1[:, 1] - i1[:, 0])[:, None]).sum(0),
                          (V2 * (i2[:, 1] - i2[:, 0])[:, None]).sum(0)),
        "ct_align": slot_cos(S1, S2),
        "tile_cov": tile_cov,
        "hr_sig": cos(onset_sig(i1, t0a, t1a), onset_sig(i2, t0b, t1b)),
        "bigram_jac": (len(bg1 & bg2) / max(len(bg1 | bg2), 1)),
        "chordset_jac": len(set(dn1) & set(dn2)) / max(len(set(dn1) | set(dn2)), 1),
    }


def chroma_criteria(C, dt, t0a, t1a, t0b, t1b):
    """Same shapes on real NNLS chroma. `C` is (T,12) L1-normalised treble."""
    def blk(t0, t1):
        a, b = int(t0 / dt), max(int(t0 / dt) + 1, int(t1 / dt))
        return C[a:min(b, len(C))]
    A, B = blk(t0a, t1a), blk(t0b, t1b)
    if len(A) < 4 or len(B) < 4:
        return None

    def res(X, n=N_SLOT):
        idx = np.clip((np.linspace(0, len(X), n + 1)[:-1]).astype(int), 0, len(X) - 1)
        e = np.clip((np.linspace(0, len(X), n + 1)[1:]).astype(int), 1, len(X))
        return np.array([X[i:j].mean(0) for i, j in zip(idx, e)])
    Sa, Sb = res(A), res(B)
    # shipped-shape criteria
    L = min(len(A), len(B))                      # UNstretched lag comparison
    lag = slot_cos(A[:L], B[:L])                 # = section_period's Vb[b]@Vb[b+P]
    stack = slot_cos(Sa, Sb)                     # = STACK_COHERENCE pairwise
    X = np.stack([Sa, Sb])                       # = CV_MAX (std/mean per slot)
    cv = float(np.mean(np.sqrt(X.var(0)) .mean(1) / np.maximum(X.mean(0).mean(1), 1e-9)))
    # chroma harmonic rhythm: novelty curve = 1 - cos(frame_t, frame_{t-1})
    def nov(X):
        d = 1.0 - np.array([cos(X[i], X[i - 1]) for i in range(1, len(X))])
        return np.interp(np.linspace(0, len(d) - 1, 32), np.arange(len(d)), d)
    return {"chroma_bag_cos": cos(A.mean(0), B.mean(0)),
            "chroma_align": stack,
            "chroma_lag_unstretched": lag,
            "chroma_cv_inv": 1.0 - min(cv, 1.0),
            "chroma_hr": cos(nov(A), nov(B))}


# ── corpus loop ──────────────────────────────────────────────────────────────
def base_letter(l):
    return l.rstrip("'")


def main(n_chroma=400, limit=None):
    import mirdata
    os.makedirs(OUT, exist_ok=True)
    bb = mirdata.initialize("billboard")
    ids = list(bb.track_ids)
    if limit:
        ids = ids[:limit]
    rows, chroma_rows = [], []
    t_start = time.time()
    n_ch_done = 0
    for ti, tid in enumerate(ids):
        tr = bb.track(tid)
        try:
            sec, ch = tr.sections, tr.chords_full
        except Exception:
            continue
        if sec is None or ch is None or len(sec.labels) < 2:
            continue
        V, iv, names = chordtone_seq(ch.labels, ch.intervals)
        if len(V) < 4:
            continue
        spans = []
        for (a, b), lab in zip(sec.intervals, sec.labels):
            if b - a < MIN_SPAN or not lab or not lab[0].isalpha():
                continue
            if base_letter(lab) in ("Z",) or len(base_letter(lab)) > 1:
                continue          # 'fadeout', 'silence', instrument words
            s = slice_span(V, iv, a, b, names)
            if s is None or len(s[0]) < 2:
                continue
            spans.append((lab, a, b, s))
        if len(spans) < 2:
            continue
        want_chroma = n_ch_done < n_chroma
        C = dt = None
        if want_chroma:
            try:
                import pandas as pd
                raw = pd.read_csv(tr.bothchroma_path, header=None).values
                t = raw[:, 1].astype(float)
                X = raw[:, 2:26].astype(float)
                X = X[:, 12:] + X[:, :12]          # treble + bass (chord tones)
                X = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
                C, dt = X, float(np.median(np.diff(t)))
                n_ch_done += 1
            except Exception as e:
                C = None
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                l1, a1, b1, s1 = spans[i]
                l2, a2, b2, s2 = spans[j]
                if l1 == l2:
                    cls = 0                                   # positive
                elif base_letter(l1) == base_letter(l2):
                    cls = 1                                   # prime variant
                else:
                    cls = 2                                   # negative (typed below)
                f = label_criteria((s1, (a1, b1)), (s2, (a2, b2)))
                if cls == 2 and f["chordset_jac"] >= HARD_JAC:
                    cls = 3                                   # HARD negative
                rec = {"tid": tid, "cls": cls, "l1": l1, "l2": l2,
                       "t": [round(a1, 2), round(b1, 2), round(a2, 2), round(b2, 2)],
                       "d1": b1 - a1, "d2": b2 - a2, **f}
                rows.append(rec)
                if C is not None:
                    cf = chroma_criteria(C, dt, a1, b1, a2, b2)
                    if cf:
                        chroma_rows.append({"tid": tid, "cls": cls, **cf, **f})
        if ti % 100 == 0:
            print(f"  {ti}/{len(ids)} tracks, {len(rows)} pairs, "
                  f"{len(chroma_rows)} chroma pairs, {time.time()-t_start:.0f}s",
                  flush=True)
    json.dump(rows, open(os.path.join(OUT, "pairs_label.json"), "w"))
    json.dump(chroma_rows, open(os.path.join(OUT, "pairs_chroma.json"), "w"))
    print(f"DONE {len(rows)} label pairs, {len(chroma_rows)} chroma pairs, "
          f"{time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main(n_chroma=int(sys.argv[1]) if len(sys.argv) > 1 else 400,
         limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
