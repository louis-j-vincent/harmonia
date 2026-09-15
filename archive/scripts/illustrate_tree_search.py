"""Illustration of the greedy top-down chord tree search.

Renders a single figure showing:
  1. The query chroma (a real segment from the corpus or a synthetic one)
  2. L0: all 5 family scores — winner highlighted, losers greyed
  3. L1: only the 3 base7 children of the winning family are scored
  4. L2: only the 1-2 exact children of the winning base7 are scored
  5. The final prediction label + confidence

The pruned branches are shown faded with a ✕, making it visually clear that
we never score them at all.

Output: docs/plots/tree_search_illustration.png

Usage:
    .venv/bin/python scripts/illustrate_tree_search.py
    .venv/bin/python scripts/illustrate_tree_search.py --quality dom7 --root 5
    .venv/bin/python scripts/illustrate_tree_search.py --synthetic   # no cache needed
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

TREE_CACHE = REPO / "data" / "cache" / "chord_tree_ltas.npz"
OUT        = REPO / "docs" / "plots" / "tree_search_illustration.png"

NOTE   = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
DEGREE = ["R", "b2", "2", "b3", "3", "4", "b5", "5", "#5", "6", "b7", "7"]

TREE = {
    "major":      {"majT": ["maj", "6"],  "maj7": ["maj7"], "dom7": ["dom7", "dom7alt"]},
    "minor":      {"minT": ["min", "m6"], "min7": ["min7"], "minmaj7": ["minmaj7"]},
    "diminished": {"dimT": ["dim"],       "dim7": ["dim7"], "m7b5": ["m7b5"]},
    "augmented":  {"augT": ["aug"],       "aug7": ["aug7"], "augmaj7": ["augmaj7"]},
    "suspended":  {"susT": ["sus2","sus4"], "7sus4": ["7sus4"]},
}
FAM_COLORS = {
    "major": "#58d4ff", "minor": "#a65fd4", "diminished": "#e34948",
    "augmented": "#e0a03b", "suspended": "#1baf7a",
}
EXACT_DISPLAY = {
    "maj": "maj", "6": "6", "maj7": "Δ7", "dom7": "7", "dom7alt": "7alt",
    "min": "min", "m6": "m6", "min7": "min7", "minmaj7": "mΔ7",
    "dim": "dim", "dim7": "°7", "m7b5": "ø7",
    "aug": "aug", "aug7": "aug7", "augmaj7": "augΔ7",
    "sus2": "sus2", "sus4": "sus4", "7sus4": "7sus4",
}
CHORD_INTERVALS = {
    "maj": [0,4,7],       "6": [0,4,7,9],       "maj7": [0,4,7,11],
    "dom7": [0,4,7,10],   "dom7alt": [0,4,7,10], "min": [0,3,7],
    "m6": [0,3,7,9],      "min7": [0,3,7,10],    "minmaj7": [0,3,7,11],
    "dim": [0,3,6],       "dim7": [0,3,6,9],     "m7b5": [0,3,6,10],
    "aug": [0,4,8],       "aug7": [0,4,8,10],    "augmaj7": [0,4,8,11],
    "sus2": [0,2,7],      "sus4": [0,5,7],       "7sus4": [0,5,7,10],
}

BG   = "#0d1520"
PANEL= "#111e2e"
DIM  = "#1e2c3a"
MUTED= "#3a4a5a"
TEXT = "#e2e8f0"
SUB  = "#6a80a0"


# ── LL helpers ────────────────────────────────────────────────────────────────

def diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std)**2) - np.sum(np.log(std)))

def max_ll_over_keys(x, mu, std):
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = diag_ll(np.roll(x, -r), mu, std)
        if ll > best_ll: best_ll, best_r = ll, r
    return best_ll, best_r


# ── synthetic query (no cache needed) ────────────────────────────────────────

def synthetic_query(quality: str, root_pc: int, noise: float = 0.15) -> np.ndarray:
    # Build at ABSOLUTE pitches (root_pc + each interval).
    # The distributions are learned in root-shifted space, and max_ll_over_keys
    # internally tries roll(x, -r) for all r — so the algorithm itself finds
    # the root. We never pre-shift the query.
    ivs = CHORD_INTERVALS[quality]
    x = np.zeros(12)
    for iv in ivs:
        x[(root_pc + iv) % 12] += 1.0
    x /= x.sum()
    rng = np.random.default_rng(42)
    x = x + rng.uniform(0, noise, 12)
    return (x / np.linalg.norm(x)).astype(np.float32)


def synthetic_distributions():
    """Build simple Gaussian distributions from chord tone vectors."""
    dist = {}
    rng = np.random.default_rng(0)
    for fam, b7_dict in TREE.items():
        fam_vecs = []
        for b7, exacts in b7_dict.items():
            b7_vecs = []
            for ex in exacts:
                ivs = CHORD_INTERVALS[ex]
                base = np.zeros(12)
                for iv in ivs: base[iv % 12] += 1.0
                base /= base.sum()
                # generate 30 noisy samples
                samples = base + rng.uniform(0, 0.12, (30, 12))
                samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
                dist[f"exact_{ex}_mu"]  = samples.mean(0)
                dist[f"exact_{ex}_std"] = samples.std(0) + 1e-4
                b7_vecs.extend(samples)
            arr = np.stack(b7_vecs)
            dist[f"b7_{b7}_mu"]  = arr.mean(0)
            dist[f"b7_{b7}_std"] = arr.std(0) + 1e-4
            fam_vecs.extend(b7_vecs)
        arr = np.stack(fam_vecs)
        dist[f"fam_{fam}_mu"]  = arr.mean(0)
        dist[f"fam_{fam}_std"] = arr.std(0) + 1e-4
    return dist


def load_dist(synthetic: bool):
    if synthetic or not TREE_CACHE.exists():
        if not synthetic:
            print("  (tree cache not found — using synthetic distributions)")
        return synthetic_distributions()
    d = np.load(TREE_CACHE)
    return {k: d[k] for k in d.files}


# ── run the search, record all scores ────────────────────────────────────────

def run_search(x12, dist):
    # L0: score all families
    fam_scores = {}; fam_roots = {}
    for fam in TREE:
        ll, r = max_ll_over_keys(x12, dist[f"fam_{fam}_mu"], dist[f"fam_{fam}_std"])
        fam_scores[fam] = ll; fam_roots[fam] = r
    best_fam  = max(fam_scores, key=fam_scores.__getitem__)
    pred_root = fam_roots[best_fam]

    # L1: only winner's children
    b7_scores = {}
    for b7 in TREE[best_fam]:
        ll, _ = max_ll_over_keys(x12, dist[f"b7_{b7}_mu"], dist[f"b7_{b7}_std"])
        b7_scores[b7] = ll
    best_b7 = max(b7_scores, key=b7_scores.__getitem__)

    # L2: only winner's children
    ex_scores = {}
    for ex in TREE[best_fam][best_b7]:
        ll, _ = max_ll_over_keys(x12, dist[f"exact_{ex}_mu"], dist[f"exact_{ex}_std"])
        ex_scores[ex] = ll
    best_ex = max(ex_scores, key=ex_scores.__getitem__)

    return {
        "fam_scores": fam_scores, "fam_roots": fam_roots,
        "best_fam": best_fam, "pred_root": pred_root,
        "b7_scores": b7_scores, "best_b7": best_b7,
        "ex_scores": ex_scores, "best_ex": best_ex,
    }


# ── figure ────────────────────────────────────────────────────────────────────

def _norm(scores: dict) -> dict:
    vals = np.array(list(scores.values()))
    lo, hi = vals.min(), vals.max()
    return {k: float((v - lo) / (hi - lo + 1e-12)) for k, v in scores.items()}


def _bar_panel(ax, scores: dict, winner: str, fam_color: str,
               pruned: bool = False, title: str = "", show_x: bool = True):
    """Draw a horizontal bar chart for a dict of {label: score}."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(DIM)
    ax.tick_params(colors=SUB, labelsize=6.5)

    norm = _norm(scores)
    labels = list(scores.keys())
    n = len(labels)
    ys = np.arange(n)

    for i, lab in enumerate(labels):
        is_winner = lab == winner
        is_pruned = pruned

        if is_pruned:
            bar_col = "#1a2535"
            text_col = "#2a3a4a"
        elif is_winner:
            bar_col = fam_color
            text_col = TEXT
        else:
            bar_col = fam_color + "44"
            text_col = SUB

        width = norm[lab]
        ax.barh(i, width, color=bar_col, height=0.62,
                edgecolor=fam_color if is_winner else DIM,
                linewidth=1.4 if is_winner else 0.4)

        disp = EXACT_DISPLAY.get(lab, lab)
        ax.text(-0.03, i, disp, ha="right", va="center",
                fontsize=7, color=text_col, fontfamily="monospace")

        if is_winner:
            ax.text(width + 0.03, i, "◀  chosen",
                    ha="left", va="center", fontsize=6.5,
                    color=fam_color, fontweight="bold")
        elif is_pruned:
            ax.text(width + 0.03, i, "✕",
                    ha="left", va="center", fontsize=7, color=MUTED)

    ax.set_xlim(-0.35, 1.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([]); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["lo", "", "hi"], fontsize=5.5, color=MUTED)
    if show_x:
        ax.set_xlabel("normalised LL", fontsize=6, color=SUB, labelpad=2)
    if title:
        ax.set_title(title, fontsize=8, color=TEXT, pad=4)


def draw(x12, res, gt_quality: str, gt_root: int, out: Path):
    best_fam = res["best_fam"]
    best_b7  = res["best_b7"]
    best_ex  = res["best_ex"]
    pred_root = res["pred_root"]
    fam_col  = FAM_COLORS[best_fam]

    # Figure layout:
    #  col 0 (narrow): query chroma + annotation column
    #  col 1: L0 family panel  (5 bars)
    #  col 2: L1 base7 panel   (2-3 bars, rest pruned)
    #  col 3: L2 exact panel   (1-2 bars, rest pruned)

    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.07,
                        wspace=0.48)

    gs = fig.add_gridspec(3, 4, height_ratios=[1.4, 4.5, 0.5],
                          width_ratios=[1.6, 2.0, 1.6, 1.3])

    # ── title ─────────────────────────────────────────────────────────────────
    pred_tok = NOTE[pred_root] + EXACT_DISPLAY.get(best_ex, best_ex)
    gt_tok   = NOTE[gt_root]   + EXACT_DISPLAY.get(gt_quality, gt_quality)
    quality_ok = (best_ex  == gt_quality)
    root_ok    = (pred_root == gt_root)
    if quality_ok and root_ok:
        verdict = f"✓ both correct  —  GT: {gt_tok}  |  pred: {pred_tok}"
        verdict_col = "#33cc77"
    elif quality_ok:
        verdict = f"◐ quality correct, root wrong  —  GT: {gt_tok}  |  pred: {pred_tok}"
        verdict_col = "#e0a03b"
    elif root_ok:
        verdict = f"◑ root correct, quality wrong  —  GT: {gt_tok}  |  pred: {pred_tok}"
        verdict_col = "#e0a03b"
    else:
        verdict = f"✗ both wrong  —  GT: {gt_tok}  |  pred: {pred_tok}"
        verdict_col = "#ff5555"

    fig.suptitle(
        "Greedy top-down chord quality tree search",
        color=TEXT, fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.5, 0.924, verdict, ha="center", fontsize=11,
             color=verdict_col, fontweight="bold")

    # ── query chroma ──────────────────────────────────────────────────────────
    ax_q = fig.add_subplot(gs[:2, 0])
    ax_q.set_facecolor(PANEL)
    for sp in ax_q.spines.values(): sp.set_color(DIM)

    ivs = CHORD_INTERVALS.get(gt_quality, [])
    bar_colors = []
    for i in range(12):
        if i == 0:
            bar_colors.append(fam_col)       # root always highlighted
        elif i in [iv % 12 for iv in ivs]:
            bar_colors.append(fam_col + "99")
        else:
            bar_colors.append("#253447")

    ax_q.bar(range(12), x12, color=bar_colors, width=0.75,
             edgecolor="#1a2535", linewidth=0.4)
    ax_q.set_xticks(range(12))
    ax_q.set_xticklabels(NOTE, fontsize=6.5, color=SUB)
    ax_q.tick_params(axis="y", colors=MUTED, labelsize=5)
    ax_q.set_xlim(-0.6, 11.6)
    ax_q.set_title("Query chroma\n(absolute pitch, L2-normed)", fontsize=8,
                   color=TEXT, pad=4)

    # chord tone tick marks (absolute pitch)
    for iv in ivs:
        ax_q.axvline((gt_root + iv) % 12, color=fam_col, lw=1.0, alpha=0.5, ymin=0, ymax=1)

    ax_q.text(0.5, -0.18,
              f"GT: {gt_tok}",
              ha="center", transform=ax_q.transAxes,
              fontsize=9, color=fam_col, fontweight="bold")

    # ── step labels (row 0, cols 1-3) ─────────────────────────────────────────
    step_labels = [
        ("L0  —  score all 5 families",
         "5 LL evaluations\n(12 root shifts each)"),
        (f"L1  —  score only children of  '{best_fam}'",
         f"{len(TREE[best_fam])} LL evaluations\n(other families: skipped ✕)"),
        (f"L2  —  score only children of  '{best_b7}'",
         f"{len(TREE[best_fam][best_b7])} LL evaluation(s)\n(other base7s: skipped ✕)"),
    ]
    for col, (title, sub) in enumerate(step_labels, start=1):
        ax_h = fig.add_subplot(gs[0, col])
        ax_h.set_facecolor(BG); ax_h.axis("off")
        ax_h.text(0.5, 0.72, title, ha="center", va="center",
                  fontsize=8.5, color=TEXT, fontweight="bold",
                  transform=ax_h.transAxes)
        ax_h.text(0.5, 0.22, sub, ha="center", va="center",
                  fontsize=7, color=SUB, style="italic",
                  transform=ax_h.transAxes)

    # ── L0 family bars ────────────────────────────────────────────────────────
    ax_l0 = fig.add_subplot(gs[1, 1])
    _bar_panel(ax_l0, res["fam_scores"], best_fam, fam_col,
               pruned=False, title="", show_x=True)

    # ── L1 base7 bars — show all base7 nodes but only score winner's children ─
    # Build a full base7 score dict: scored nodes get real values,
    # unscored (different family) get a placeholder at the minimum
    all_b7_scores = {}
    min_scored = min(res["b7_scores"].values())
    for fam, b7_dict in TREE.items():
        for b7 in b7_dict:
            if fam == best_fam:
                all_b7_scores[b7] = res["b7_scores"].get(b7, min_scored)
            else:
                all_b7_scores[b7] = min_scored - 1   # push below all scored

    ax_l1 = fig.add_subplot(gs[1, 2])
    ax_l1.set_facecolor(PANEL)
    for sp in ax_l1.spines.values(): sp.set_color(DIM)
    ax_l1.tick_params(colors=SUB, labelsize=6.5)

    # Draw two groups: scored (this family) and unscored (other families), separated
    scored_b7s  = list(TREE[best_fam].keys())
    other_b7s   = [b7 for fam in TREE for b7 in TREE[fam] if fam != best_fam]
    all_b7_order = scored_b7s + ["···"] + other_b7s
    norm_b7 = _norm({b: all_b7_scores[b] for b in scored_b7s})

    n_scored = len(scored_b7s)
    n_other  = len(other_b7s)
    total_y  = n_scored + 1 + n_other

    for i, lab in enumerate(scored_b7s):
        is_winner = lab == best_b7
        y = total_y - 1 - i
        width = norm_b7[lab]
        ax_l1.barh(y, width, color=fam_col if is_winner else fam_col+"44",
                   height=0.62,
                   edgecolor=fam_col if is_winner else DIM,
                   linewidth=1.4 if is_winner else 0.4)
        ax_l1.text(-0.03, y, lab, ha="right", va="center",
                   fontsize=7, color=TEXT if is_winner else SUB,
                   fontfamily="monospace")
        if is_winner:
            ax_l1.text(width + 0.03, y, "◀  chosen",
                       ha="left", va="center", fontsize=6.5,
                       color=fam_col, fontweight="bold")

    # separator
    sep_y = total_y - 1 - n_scored
    ax_l1.text(0.5, sep_y, "── not scored (wrong family) ──",
               ha="center", va="center", fontsize=6, color=MUTED, style="italic")

    for j, lab in enumerate(other_b7s):
        y = total_y - 1 - n_scored - 1 - j
        fam_of_b7 = next(fam for fam, ch in TREE.items() if lab in ch)
        col_dim = FAM_COLORS[fam_of_b7] + "22"
        ax_l1.barh(y, 0.05, color=col_dim, height=0.62,
                   edgecolor=DIM, linewidth=0.4)
        ax_l1.text(-0.03, y, lab, ha="right", va="center",
                   fontsize=6.5, color="#2a3a4a", fontfamily="monospace")
        ax_l1.text(0.08, y, "✕", ha="left", va="center",
                   fontsize=7, color=MUTED)

    ax_l1.set_xlim(-0.35, 1.6); ax_l1.set_ylim(-0.6, total_y - 0.4)
    ax_l1.set_yticks([]); ax_l1.set_xticks([0, 0.5, 1.0])
    ax_l1.set_xticklabels(["lo", "", "hi"], fontsize=5.5, color=MUTED)
    ax_l1.set_xlabel("normalised LL", fontsize=6, color=SUB, labelpad=2)

    # ── L2 exact bars ─────────────────────────────────────────────────────────
    all_exact_scores = {}
    min_ex = min(res["ex_scores"].values()) - 1
    for fam, b7_dict in TREE.items():
        for b7, exacts in b7_dict.items():
            for ex in exacts:
                if fam == best_fam and b7 == best_b7:
                    all_exact_scores[ex] = res["ex_scores"].get(ex, min_ex)
                else:
                    all_exact_scores[ex] = min_ex

    scored_exs = list(TREE[best_fam][best_b7])
    other_exs  = [ex for fam, ch in TREE.items()
                  for b7, exacts in ch.items()
                  for ex in exacts
                  if not (fam == best_fam and b7 == best_b7)]
    norm_ex = _norm({e: all_exact_scores[e] for e in scored_exs})

    ax_l2 = fig.add_subplot(gs[1, 3])
    ax_l2.set_facecolor(PANEL)
    for sp in ax_l2.spines.values(): sp.set_color(DIM)
    ax_l2.tick_params(colors=SUB, labelsize=6.5)

    n_scored_ex = len(scored_exs)
    n_other_ex  = len(other_exs)
    total_y2    = n_scored_ex + 1 + n_other_ex

    for i, ex in enumerate(scored_exs):
        is_winner = ex == best_ex
        y = total_y2 - 1 - i
        w = norm_ex[ex]
        ax_l2.barh(y, w, color=fam_col if is_winner else fam_col+"44",
                   height=0.62,
                   edgecolor=fam_col if is_winner else DIM,
                   linewidth=1.4 if is_winner else 0.4)
        disp = EXACT_DISPLAY.get(ex, ex)
        ax_l2.text(-0.03, y, disp, ha="right", va="center",
                   fontsize=7, color=TEXT if is_winner else SUB,
                   fontfamily="monospace")
        if is_winner:
            ax_l2.text(w + 0.03, y, "◀", ha="left", va="center",
                       fontsize=7, color=fam_col, fontweight="bold")

    sep_y2 = total_y2 - 1 - n_scored_ex
    ax_l2.text(0.5, sep_y2, "── not scored ──",
               ha="center", va="center", fontsize=6, color=MUTED, style="italic")

    for j, ex in enumerate(other_exs):
        y = total_y2 - 1 - n_scored_ex - 1 - j
        fam_of = next(fam for fam, ch in TREE.items()
                      for b7, exs in ch.items() if ex in exs)
        ax_l2.barh(y, 0.05, color=FAM_COLORS[fam_of]+"22",
                   height=0.62, edgecolor=DIM, linewidth=0.4)
        disp = EXACT_DISPLAY.get(ex, ex)
        ax_l2.text(-0.03, y, disp, ha="right", va="center",
                   fontsize=6, color="#2a3a4a", fontfamily="monospace")
        ax_l2.text(0.08, y, "✕", ha="left", va="center",
                   fontsize=6.5, color=MUTED)

    ax_l2.set_xlim(-0.35, 1.6); ax_l2.set_ylim(-0.6, total_y2 - 0.4)
    ax_l2.set_yticks([]); ax_l2.set_xticks([0, 0.5, 1.0])
    ax_l2.set_xticklabels(["lo", "", "hi"], fontsize=5.5, color=MUTED)
    ax_l2.set_xlabel("normalised LL", fontsize=6, color=SUB, labelpad=2)

    # ── final answer banner ───────────────────────────────────────────────────
    ax_ans = fig.add_subplot(gs[2, 1:])
    ax_ans.set_facecolor(BG); ax_ans.axis("off")

    root_note = NOTE[pred_root]
    path_str = f"{best_fam}  →  {best_b7}  →  {best_ex}"
    ans_str  = f"Prediction:  {root_note}{EXACT_DISPLAY.get(best_ex, best_ex)}"
    n_evals  = f"Total LL evaluations: {len(TREE)} + {len(TREE[best_fam])} + {len(TREE[best_fam][best_b7])} = {len(TREE)+len(TREE[best_fam])+len(TREE[best_fam][best_b7])}  (vs {sum(len(e) for ch in TREE.values() for e in ch.values())} × 12 = {sum(len(e) for ch in TREE.values() for e in ch.values())*12} for exhaustive)"

    ax_ans.text(0.01, 0.65, path_str, ha="left", va="center",
                fontsize=9, color=SUB, fontfamily="monospace",
                transform=ax_ans.transAxes)
    ax_ans.text(0.01, 0.2, ans_str, ha="left", va="center",
                fontsize=11, color=fam_col, fontweight="bold",
                transform=ax_ans.transAxes)
    ax_ans.text(0.99, 0.35, n_evals, ha="right", va="center",
                fontsize=7.5, color=MUTED, style="italic",
                transform=ax_ans.transAxes)

    # draw arrows connecting panels
    for x_frac in [0.395, 0.62]:
        fig.patches.append(mpatches.FancyArrowPatch(
            (x_frac, 0.48), (x_frac + 0.07, 0.48),
            transform=fig.transFigure,
            arrowstyle="->", color=fam_col + "88",
            mutation_scale=12, lw=1.2,
        ))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"→ {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality",   default="dom7",
                    help="GT chord quality to illustrate")
    ap.add_argument("--root",      type=int, default=5,
                    help="GT root (0=C, 5=F, 9=A, …)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic distributions (no cache required)")
    ap.add_argument("--noise",     type=float, default=0.15,
                    help="amount of noise added to the query chroma")
    ap.add_argument("--out",       default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT

    print(f"Building query: {args.quality} root={NOTE[args.root]} noise={args.noise}")
    dist = load_dist(args.synthetic)
    x12  = synthetic_query(args.quality, args.root, args.noise)

    print("Running tree search...")
    res = run_search(x12, dist)
    print(f"  L0 winner: {res['best_fam']}  (predicted root: {NOTE[res['pred_root']]})")
    print(f"  L1 winner: {res['best_b7']}")
    print(f"  L2 winner: {res['best_ex']}")

    draw(x12, res, args.quality, args.root, out)
