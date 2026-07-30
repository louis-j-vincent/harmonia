"""Diagnostic plots for the segmentation-granularity study.

Colour: Okabe-Ito (published CVD-safe categorical set), assigned in FIXED order
per arm so an arm keeps its colour across every panel. The SSM is a single-hue
sequential ramp — never a rainbow. One axis per panel; no dual scales.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from anchor import anchor_sections                                  # noqa: E402
from blur import gaussian_blur, novelty, lag_profile                # noqa: E402
from harness import all_charts                                      # noqa: E402
from harmonia.models.section_vocab import (                         # noqa: E402
    SLOTS_PER_BAR, build_slots, chord_ssm, form_string, vocab_sections)

INK, INK2, GRID, SURF = "#161615", "#5c5b58", "#e2e1dc", "#fdfdfc"
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
         "#F0E442", "#000000"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#fdfdfc", "#a8c8de", "#2b6d92",
                                                "#0d2f42"])

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "font.size": 8.5, "axes.titlesize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def _clean(ax):
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# ── 1. the arms ──────────────────────────────────────────────────────────────

def plot_arms():
    rows = json.loads((HERE / "sweep_min4.json").read_text())
    arms = ["auto", "fix2", "fix4", "fix8", "hier", "coarse1st",
            "anchor4", "anchorS4", "anchorN4"]
    label = {"auto": "auto  (shipped)", "fix2": "fixed 2 bars",
             "fix4": "fixed 4 bars", "fix8": "fixed 8 bars",
             "hier": "hierarchy 8→4→2", "coarse1st": "coarsest recurring first",
             "anchor4": "boundary-anchored (blur σ=4)",
             "anchorS4": "  ″  sharp zones", "anchorN4": "  ″  no zones"}
    ok = [r for r in rows if all(r.get(a) and "error" not in r[a] for a in arms)]
    panels = [("sane", "M3  form a musician would call sane", "share of charts", True),
              ("n_written", "M1  sections the chart writes", "count", False),
              ("rec_cov", "M4  song covered by a RECURRING section", "share of bars", True),
              ("frag", "M5  fragments (< 4 bars)", "count", False)]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.6))
    y = np.arange(len(arms))
    for ax, (key, title, xlab, pct) in zip(axes, panels):
        vals = [np.mean([r[a][key] for r in ok]) for a in arms]
        cols = [OKABE[0] if a == "auto" else
                (OKABE[1] if a.startswith("anchor") else OKABE[2]) for a in arms]
        ax.barh(y, vals, color=cols, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i,
                    f"{v:.0%}" if pct else f"{v:.2f}", va="center", fontsize=8,
                    color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([label[a] for a in arms] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlab)
        ax.set_xlim(0, max(vals) * 1.22)
        _clean(ax)
    fig.suptitle(f"Section-detection granularity — {len(ok)} charts where every arm "
                 "fires.  Blue = shipped, green = fixed/hierarchical grain, "
                 "orange = boundary-anchored.", x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "seg_arms.png", dpi=150)
    print("wrote seg_arms.png")


# ── 2. Mayer: the 25-bar unit ────────────────────────────────────────────────

def _strip(ax, secs, n_bars, y, cmap_of, h=0.8):
    for s in secs:
        ax.add_patch(plt.Rectangle((s["bar0"], y - h / 2), s["bar1"] - s["bar0"], h,
                                   facecolor=cmap_of(s["label"]), edgecolor=SURF,
                                   lw=1.4))
        if s["bar1"] - s["bar0"] >= 3:
            ax.text((s["bar0"] + s["bar1"]) / 2, y,
                    s["label"] + (f"×{s['reps']}" if s["reps"] > 1 else ""),
                    ha="center", va="center", fontsize=8, color="white",
                    weight="bold")


def plot_song(name, fname, title):
    c = [x for x in all_charts() if name in x["file"].name][0]
    tok, roots, known = build_slots(c["bars"], c["n_bars"],
                                    tonic_pc=c["tonic_pc"], bpb=c["bpb"])
    S = chord_ssm(tok)
    n = c["n_bars"]
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1, 0.78],
                          hspace=0.5, wspace=0.30)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(S, cmap=SEQ, vmin=0, vmax=1, origin="upper",
              extent=(0, n, n, 0), interpolation="nearest")
    ax.set_title("chord-tone SSM (sharp)  —  Bb is closer to Gm than to F",
                 loc="left")
    ax.set_xlabel("bar")
    ax.set_ylabel("bar")

    axl = fig.add_subplot(gs[0, 1])
    p = lag_profile(S, known)
    lags = np.arange(2, min(len(p) // SLOTS_PER_BAR, n))
    vals = [p[int(b) * SLOTS_PER_BAR] for b in lags]
    axl.plot(lags, vals, color=OKABE[0], lw=1.8)
    b = int(lags[int(np.argmax(vals))])
    axl.plot([b], [max(vals)], "o", ms=8, color=OKABE[1])
    axl.annotate(f"period = {b} bars\nself-match {max(vals):.3f}",
                 (b, max(vals)), textcoords="offset points", xytext=(8, -14),
                 fontsize=8.5, color=INK)
    axl.set_title("how well the song matches itself N bars later", loc="left")
    axl.set_xlabel("lag (bars)")
    axl.set_ylabel("mean similarity")
    axl.grid(color=GRID, lw=0.7)
    axl.set_axisbelow(True)

    axs = fig.add_subplot(gs[1, 1])
    shipped = vocab_sections(c["bars"], n, tonic_pc=c["tonic_pc"], bpb=c["bpb"])
    anc = anchor_sections(c["bars"], n, tonic_pc=c["tonic_pc"], bpb=c["bpb"],
                          sigma_bars=4)
    letters = sorted({s["label"] for s in (shipped or []) + (anc or [])})
    cmap_of = {l: OKABE[i % len(OKABE)] for i, l in enumerate(letters)}.get
    if shipped:
        _strip(axs, shipped, n, 1, lambda l: cmap_of(l) or INK2)
    if anc:
        _strip(axs, anc, n, 0, lambda l: cmap_of(l) or INK2)
    axs.set_xlim(0, n)
    axs.set_ylim(-0.75, 1.95)
    axs.set_yticks([1, 0])
    axs.set_yticklabels(["forward", "anchored"], fontsize=8.5)
    for yy, secs in ((1, shipped), (0, anc)):
        axs.text(0, yy + 0.52, form_string(secs) if secs else "defer",
                 fontsize=8, color=INK2, va="bottom")
    axs.set_xlabel("bar")
    axs.set_title("what each detector writes", loc="left")
    for sp in ("left",):
        axs.spines[sp].set_visible(False)
    axs.tick_params(left=False)

    fig.suptitle(title, x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / fname, dpi=150)
    print("wrote", fname)


# ── 3. the blur prior ────────────────────────────────────────────────────────

def plot_blur():
    rows = json.loads((HERE / "blur_eval.json").read_text())
    sig = [2, 3, 4, 6, 8]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axes[0]
    for j, (k, lab) in enumerate((("prf", "every letter change"),
                                  ("prf_major", "only changes between two\nsections ≥ σ bars"))):
        P = [np.mean([r[f"s{s}"][k][0] for r in rows]) for s in sig]
        ax.plot(sig, P, "-o", color=OKABE[j], lw=2, ms=7, label=lab)
    base = np.mean([r["sharp"]["prf"][0] for r in rows])
    ax.axhline(base, color=INK2, ls="--", lw=1.4)
    ax.text(8, base + .012, "no blur at all", ha="right", fontsize=8, color=INK2)
    ax.set_title("PRECISION of the blurred-SSM boundaries", loc="left")
    ax.set_xlabel("σ (bars)")
    ax.set_ylabel("share of emitted boundaries that are real")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    for j, (k, lab) in enumerate((("prf", "every letter change"),
                                  ("prf_major", "only changes between two\nsections ≥ σ bars"))):
        R = [np.mean([r[f"s{s}"][k][1] for r in rows]) for s in sig]
        ax.plot(sig, R, "-o", color=OKABE[j], lw=2, ms=7, label=lab)
    ax.axhline(np.mean([r["sharp"]["prf"][1] for r in rows]), color=INK2,
               ls="--", lw=1.4)
    ax.text(8, np.mean([r["sharp"]["prf"][1] for r in rows]) + .012,
            "no blur at all", ha="right", fontsize=8, color=INK2)
    ax.set_title("RECALL of the real boundaries", loc="left")
    ax.set_xlabel("σ (bars)")
    ax.set_ylabel("share of real boundaries found")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=.7)
    ax.set_axisbelow(True)

    ax = axes[2]
    c = [x for x in all_charts() if "don_t_know_why" in x["file"].name][0]
    tok, roots, known = build_slots(c["bars"], c["n_bars"],
                                    tonic_pc=c["tonic_pc"], bpb=c["bpb"])
    S = chord_ssm(tok)
    bars = np.arange(S.shape[0]) / SLOTS_PER_BAR
    ns = novelty(S, 4.0)
    nb = novelty(gaussian_blur(S, 4.0), 4.0)
    ax.plot(bars, ns / (ns.max() or 1), color=INK2, lw=1.2, label="sharp SSM")
    ax.plot(bars, nb / (nb.max() or 1), color=OKABE[1], lw=2.2, label="blurred σ=4")
    v = vocab_sections(c["bars"], c["n_bars"], tonic_pc=c["tonic_pc"], bpb=c["bpb"])
    for s in (v or [])[1:]:
        ax.axvline(s["bar0"], color=OKABE[0], lw=1.1, alpha=.55)
    ax.plot([], [], color=OKABE[0], lw=1.1, label="detector's letter changes")
    ax.set_title("Don't Know Why — where each novelty curve peaks", loc="left")
    ax.set_xlabel("bar")
    ax.set_ylabel("novelty (scaled)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color=GRID, lw=.7)
    ax.set_axisbelow(True)

    fig.suptitle("The blurred-SSM boundary prior, 39 charts — blurring does NOT "
                 "make the emitted boundaries more reliable", x=0.005, ha="left",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "blur_prior.png", dpi=150)
    print("wrote blur_prior.png")


if __name__ == "__main__":
    plot_arms()
    plot_song("just_ain", "mayer_anchor.png",
              "Mayer Hawthorne — \"Just Ain't Gonna Work Out\":  one 25-bar unit, "
              "played twice, after a 6-bar intro")
    plot_song("don_t_know_why", "norah_anchor.png",
              "Norah Jones — \"Don't Know Why\":  the 4-bar phrase the forward "
              "detector finds 11×, vs the anchored 16-bar reading")
    plot_blur()
