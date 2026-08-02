"""Norah verdict: run every candidate criterion on Don't Know Why's SIX
occurrences of letter A, and check each against the threshold measured on
Billboard. Which criterion would have refused the fold?

Chroma arm uses the CURRENT chart bar grid (66 bars) recomputed from audio —
no stale artifact. Symbolic arm uses the per-half-bar decode tokens in
scratchpad/ssm_norah_meta.json (67-bar run; first 132 half-bars align, both
grids start at 0.14 s) and is flagged as approximate.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from fold_criteria_billboard import (chroma_criteria, cos, dedup,  # noqa: E402
                                     label_criteria, onset_sig, slot_cos)

CHART = os.path.join(HERE, "harmonia_min/state/charts/min_norah_jones_don_t_know_why.json")
AUDIO = os.path.join(HERE, "docs/audio/norah_jones_don_t_know_why.m4a")
META = os.path.join(HERE, "scratchpad/ssm_norah_meta.json")

QMAP = {"": [0, 4, 7], "^7": [0, 4, 7, 11], "7": [0, 4, 7, 10],
        "-7": [0, 3, 7, 10], "-": [0, 3, 7], "h7": [0, 3, 6, 10],
        "o7": [0, 3, 6, 9], "6": [0, 4, 7, 9], "-6": [0, 3, 7, 9],
        "+": [0, 4, 8], "sus": [0, 5, 7], "7sus": [0, 5, 7, 10],
        "^": [0, 4, 7, 11], "9": [0, 4, 7, 10, 2], "-9": [0, 3, 7, 10, 2]}
PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
      "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
      "B": 11}


def tok_vec(t):
    for n in (2, 1):
        if t[:n] in PC:
            r, q = PC[t[:n]], t[n:]
            break
    else:
        return np.zeros(12)
    v = np.zeros(12)
    for s in QMAP.get(q, QMAP.get(q.rstrip("913"), [0, 4, 7])):
        v[(r + s) % 12] = 1.0
    return v


def main():
    ch = json.load(open(CHART))
    grid = np.array(ch["barGrid"], float)
    ranges = ch["sections"][0]["barRanges"]
    print("A occurrences (bars):", ranges,
          "lengths:", [b - a + 1 for a, b in ranges])

    # ── symbolic arm from the half-bar decode tokens ─────────────────────────
    toks = json.load(open(META))["tokens"][:2 * (len(grid) - 1)]
    hb = np.linspace(0, 1, 1)  # placeholder
    # build (V, iv, names) at half-bar resolution on the CURRENT grid
    V, iv, nm = [], [], []
    for i, t in enumerate(toks):
        b, half = divmod(i, 2)
        if b + 1 >= len(grid):
            break
        t0 = grid[b] + half * 0.5 * (grid[b + 1] - grid[b])
        t1 = grid[b] + (half + 1) * 0.5 * (grid[b + 1] - grid[b])
        V.append(tok_vec(t))
        iv.append([t0, t1])
        nm.append(t)
    V, iv = np.array(V), np.array(iv)

    def sym_span(a, b):
        t0, t1 = grid[a], grid[b + 1]
        m = (iv[:, 1] > t0) & (iv[:, 0] < t1)
        return (V[m], np.clip(iv[m], t0, t1), [n for n, k in zip(nm, m) if k]), (t0, t1)

    # ── chroma arm on the current grid ───────────────────────────────────────
    from harmonia_min.nnls_features import extract_bothchroma
    from pathlib import Path
    arr, times = extract_bothchroma(Path(AUDIO))
    arr = np.asarray(arr)
    if arr.shape[1] >= 24:
        X = arr[:, 12:24] + arr[:, :12]
    else:
        X = arr[:, :12]
    X = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
    dt = float(np.median(np.diff(times)))
    print("chroma", X.shape, "dt %.4f" % dt, "span %.1fs" % times[-1])

    rows = []
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            a1, b1 = ranges[i]
            a2, b2 = ranges[j]
            s1, t1_ = sym_span(a1, b1)
            s2, t2_ = sym_span(a2, b2)
            f = label_criteria((s1, t1_), (s2, t2_))
            cf = chroma_criteria(X, dt, t1_[0], t1_[1], t2_[0], t2_[1]) or {}
            rows.append({"pair": f"A{i+1}({b1-a1+1}b) vs A{j+1}({b2-a2+1}b)",
                         **f, **cf})
    keys = [k for k in rows[0] if k != "pair"]
    print("\n%-22s " % "pair" + " ".join(f"{k[:11]:>11}" for k in keys))
    for r in rows:
        print("%-22s " % r["pair"] + " ".join(f"{r[k]:11.3f}" for k in keys))
    med = {k: float(np.median([r[k] for r in rows])) for k in keys}
    print("%-22s " % "MEDIAN" + " ".join(f"{med[k]:11.3f}" for k in keys))
    json.dump({"rows": rows, "median": med, "ranges": ranges},
              open(os.path.join(HERE, "scratchpad/fold_criteria/norah.json"), "w"),
              indent=1)

    # first pass' true 4-bar cell vs what was written (2-bar cell x2)
    print("\nwritten block = source bars", [0, 1, 0, 1],
          "=", " | ".join(f"{nm[2*b]} {nm[2*b+1]}" for b in (0, 1, 0, 1)))
    print("true first pass bars 0-7 =",
          " | ".join(f"{nm[2*b]} {nm[2*b+1]}" for b in range(8)))


if __name__ == "__main__":
    main()


def norah_policies():
    """What each POLICY would write for Don't Know Why's letter A, and how much
    of the song's harmony it would misrepresent. Uses the half-bar decode
    tokens as the truth track (our own decode, not GT — Norah has no GT)."""
    import numpy as np
    ch = json.load(open(CHART))
    grid = np.array(ch["barGrid"], float)
    ranges = ch["sections"][0]["barRanges"]
    toks = json.load(open(META))["tokens"][:2 * (len(grid) - 1)]
    hb_v = np.array([tok_vec(t) for t in toks])          # (132,12) half-bars

    def track(a, b, n=256):
        """chord-tone track of bars [a,b] on n samples."""
        idx = np.linspace(2 * a, 2 * (b + 1), n, endpoint=False).astype(int)
        return hb_v[np.clip(idx, 0, len(hb_v) - 1)]

    def harm(tmpl, a, b, n=256):
        true = track(a, b, n)
        # tile tmpl at its own BAR length over the occurrence
        Lt = tmpl.shape[0]
        occ_bars = b - a + 1
        tb = tmpl_bars[id(tmpl)]
        phase = (np.linspace(0, occ_bars, n, endpoint=False) % tb) / tb
        ti = np.clip((phase * Lt).astype(int), 0, Lt - 1)
        A, B = tmpl[ti], true
        num = (A * B).sum(1)
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
        ok = den > 1e-9
        ag = np.zeros(n, bool)
        ag[ok] = num[ok] / den[ok] >= 0.99
        return float(1 - ag.mean())

    tmpl_bars = {}

    def build(a, b, nbars=None):
        nb = nbars or (b - a + 1)
        t = track(a, a + nb - 1)
        tmpl_bars[id(t)] = nb
        return t

    total = sum(b - a + 1 for a, b in ranges)
    pols = {
        "SHIPPED (minimal_fold: 2-bar cell x2)":
            [(build(0, 3, 4), ranges)],                       # bars 0,1,0,1 ~ 4 bars
        "letter (first occurrence, 8 bars)":
            [(build(*ranges[0]), ranges)],
        "equal_len / align>=0.85 (A3+A4 merge, rest apart)":
            [(build(*r), [r]) for r in ranges if r not in (ranges[2], ranges[3])]
            + [(build(*ranges[2]), [ranges[2], ranges[3]])],
        "write_out (every occurrence)":
            [(build(*r), [r]) for r in ranges],
    }
    print("\n%-52s %8s %8s" % ("policy for Norah letter A", "bars", "harm"))
    for name, blocks in pols.items():
        written = sum(tmpl_bars[id(t)] for t, _ in blocks)
        h = sum((b - a + 1) * harm(t, a, b) for t, rs in blocks for a, b in rs)
        print("%-52s %8d %7.1f%%" % (name, written, 100 * h / total))
    # the shipped block is really bars 0,1,0,1 — rebuild it explicitly
    t01 = np.concatenate([track(0, 1, 128), track(0, 1, 128)])
    tmpl_bars[id(t01)] = 4
    h = sum((b - a + 1) * harm(t01, a, b) for a, b in ranges)
    print("%-52s %8d %7.1f%%" % ("SHIPPED, exact block [bar0 bar1 bar0 bar1]",
                                 4, 100 * h / total))


if __name__ == "__main__":
    norah_policies()
