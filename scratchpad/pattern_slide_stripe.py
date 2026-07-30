#!/usr/bin/env python3
"""pattern_slide_stripe.py — the stripe-diagonal slide ONLY, chord-tone SSM vs
root SSM, so we can see whether the root matrix carries extra information.

Also prints the A-vs-C competition table: at every bar in the A x3 run, which
known pattern scores highest. That is the "how do you differentiate?" question.

Run: .venv/bin/python scratchpad/pattern_slide_stripe.py [song-substring]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

import pattern_slide as ps
from section_merge_declined import _load_payload

SERIES = ["#3b6fd4", "#d97a1f", "#1f8a6b", "#8a4fbf"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"


def root_only_ssm(seq) -> np.ndarray:
    """COMPARISON ONLY (not the project's SSM — see pattern_slide.chord_ssm and the
    standing rule). 1.0 iff two slots share a root, 0.0 otherwise."""
    r = np.array([x[0] for x in seq])
    return ((r[:, None] == r[None, :]) & (r[:, None] >= 0)).astype(np.float32)


def stripe_diag(S: np.ndarray, row0: int, d: int) -> np.ndarray:
    """mean_i S[row0+i, t+i] for every t — the slid stripe's diagonal, nothing else."""
    n = S.shape[0]
    out = np.full(n, np.nan)
    for t in range(0, n - d + 1):
        out[t] = float(np.mean([S[row0 + i, t + i] for i in range(d)]))
    return out


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    seq, names, n_bars, bar_sec = ps.rigid_slots(P)
    Sc = ps.chord_ssm(names)          # the project's SSM (chord tones)
    Sr = root_only_ssm(seq)           # comparison
    n = Sc.shape[0]

    quiet = []
    blocks = ps.segment(Sc, seq, names, log=quiet.append)
    pats = []
    for b in blocks:
        if b["label"] not in [p["label"] for p in pats]:
            pats.append(b)

    curves = {}
    for M, tag in ((Sc, "chord"), (Sr, "root")):
        for p in pats:
            curves[(tag, p["label"])] = stripe_diag(M, p["row0"], p["d"])

    # ── the A-vs-C competition, printed ───────────────────────────────────────
    print(f"\n=== {slug} — which pattern wins at each bar? ===")
    print("(stripe-diagonal on the chord-tone SSM; the winner is argmax over patterns)\n")
    hdr = "  bar  " + "  ".join(f"{p['label']:>6s}" for p in pats) + "   winner   true"
    truth = {}
    for b in blocks:
        for bar in range(b["start"] // 2, b["end"] // 2):
            truth[bar] = b["label"]
    print(hdr)
    for bar in range(n_bars):
        t = bar * 2
        vals = []
        for p in pats:
            c = curves[("chord", p["label"])][t]
            vals.append(-1.0 if np.isnan(c) else float(c))
        win = pats[int(np.argmax(vals))]["label"]
        tr = truth.get(bar, "?")
        flag = "" if win == tr else "   <-- argmax disagrees with the walk"
        row = "  ".join(f"{v:6.3f}" if v >= 0 else "     ." for v in vals)
        print(f"  {bar:3d}  {row}   {win}        {tr}{flag}")

    # ── how much does root add? ───────────────────────────────────────────────
    print("\n=== does the ROOT SSM carry extra information? ===")
    print("Measured ON-PHASE ONLY — a pattern can only ever claim bars on its own grid")
    print("(row0, row0+d, row0+2d, ...). Scoring it at off-phase bars is meaningless:")
    print("mid-loop positions legitimately match nothing, which is why the argmax table")
    print("above 'disagrees' so often. PHASE is what discriminates, not amplitude.\n")
    print("  pattern   chord-tone                  root-only                   better")
    for p in pats:
        line = f"  {p['label']:<9s}"
        gaps = {}
        step = p["d"] // 2
        for tag in ("chord", "root"):
            c = curves[(tag, p["label"])]
            same, other = [], []
            for bar in range(n_bars):
                t = bar * 2
                if np.isnan(c[t]) or (bar - p["row0"] // 2) % step != 0:
                    continue          # off-phase: not this pattern's to claim
                (same if truth.get(bar) == p["label"] else other).append(float(c[t]))
            lo, hi = (min(same) if same else 0.0), (max(other) if other else 0.0)
            gaps[tag] = lo - hi
            line += f" same>={lo:.2f} other<={hi:.2f} gap {lo-hi:+.2f}   "
        line += "root" if gaps["root"] > gaps["chord"] else "chord-tone"
        print(line)

    # ── the figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True, facecolor=SURFACE)
    for a in ax:
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=8)
        a.set_ylim(0, 1.06)
        for b in blocks:
            a.axvline(b["start"], color=INK, lw=0.8, alpha=0.30)

    for k, (tag, title) in enumerate((
            ("chord", "CHORD-TONE SSM (the project's SSM: Bb closer to Gm than to F)"),
            ("root", "ROOT-ONLY SSM (comparison: same root = 1, else 0)"))):
        for i, p in enumerate(pats):
            c = curves[(tag, p["label"])]
            ax[k].plot(c, color=SERIES[i], lw=2.0, label=f"pattern {p['label']}")
            grid = [t for t in range(p["row0"], n, p["d"]) if not np.isnan(c[t])]
            hits = [t for t in grid if c[t] >= ps.REP_STRICT]
            ax[k].plot(hits, [c[t] for t in hits], "o", ms=8, color=SERIES[i],
                       mec=SURFACE, mew=2, zorder=5)
        ax[k].axhline(ps.REP_STRICT, color=INK2, lw=1, ls=(0, (4, 3)))
        ax[k].set_ylabel("stripe diagonal", color=INK2, fontsize=9)
        ax[k].set_title(f"SLIDE ACROSS X, stripe diagonal only — {title}",
                        fontsize=10, color=INK, loc="left")
        ax[k].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    for b in blocks:
        lbl = b["label"] + (f"×{b['mult']}" if b["mult"] > 1 else "")
        ax[0].text((b["start"] + b["end"]) / 2, 1.08, lbl, ha="center", fontsize=11,
                   fontweight="bold",
                   color=SERIES[[p["label"] for p in pats].index(b["label"])])
    ax[1].set_xlim(0, n)
    ax[1].set_xticks(range(0, n + 1, 8))
    ax[1].set_xticklabels([f"{t//2}" for t in range(0, n + 1, 8)], fontsize=8)
    ax[1].set_xlabel("bar", color=INK2, fontsize=9)
    plt.tight_layout()
    out = REPO / "scratchpad" / f"pattern_slide_stripe_{slug}.png"
    plt.savefig(out, dpi=125, facecolor=SURFACE)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
