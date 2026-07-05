"""Visual representations of the hierarchies we built (for the blog).

1. The CHORD tree: root → family (maj/min/dim/aug/sus) → seventh → exact quality,
   annotated with the audio-model accuracy achievable at each depth.
2. The PATTERN-abstraction funnel: how each progression encoding shrinks the
   vocabulary of 3-chord patterns.
3. The STYLE hierarchy: broad genre → fine feel, with each style's chord mix.

Figures → docs/plots/hierarchy_*.png
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.theory.chord_tree import base_seventh_of, family_of  # noqa: E402
from harmonia.theory.chord_vocabulary import ChordQuality, chord_label  # noqa: E402

PLOTS = REPO / "docs" / "plots"
FAM_COLORS = {"maj": "#4C72B0", "min": "#55A868", "dim": "#C44E52",
              "aug": "#8172B2", "sus": "#DD8452", "N": "#999999"}

# Explicit chord tree WITH 6th chords, and the characteristic scale/mode each
# chord draws from (chord-scale theory). family -> {seventh-node: [exact leaves]}
CHORD_TREE = {
    "maj": ("MAJOR", {
        "maj (triad)": ["maj", "6"],
        "maj7": ["maj7", "maj9", "maj9#11", "maj13"],
        "7 (dom)": ["7", "9", "7b9", "7#9", "7#11", "13", "13b9"],
    }),
    "min": ("MINOR", {
        "min (triad)": ["min", "m6"],
        "min7": ["min7", "min9", "min11", "min13"],
        "min-maj7": ["mMaj7"],
    }),
    "dim": ("DIM", {
        "dim (triad)": ["dim"],
        "dim7": ["°7"],
        "m7b5 (ø)": ["ø7"],
    }),
    "aug": ("AUG", {
        "aug (triad)": ["aug"],
        "augMaj7": ["augMaj7"],
        "aug7": ["aug7"],
    }),
    "sus": ("SUS", {
        "sus (triad)": ["sus2", "sus4"],
        "7sus4": ["7sus4", "9sus4"],
    }),
}
# characteristic scale/mode per exact chord
CHORD_MODE = {
    "maj": "Ionian", "6": "Ionian", "maj7": "Ionian", "maj9": "Ionian",
    "maj9#11": "Lydian", "maj13": "Ionian",
    "7": "Mixolydian", "9": "Mixolydian", "13": "Mixolydian",
    "7b9": "Phrygian dom", "7#9": "Altered", "7#11": "Lydian dom",
    "13b9": "H-W dim", "7alt": "Altered",
    "min": "Aeolian", "m6": "Dorian", "min7": "Dorian", "min9": "Dorian",
    "min11": "Dorian", "min13": "Dorian", "mMaj7": "Melodic minor",
    "dim": "Diminished", "°7": "W-H dim", "ø7": "Locrian",
    "aug": "Whole tone", "augMaj7": "Lydian aug", "aug7": "Whole tone",
    "sus2": "Mixolydian", "sus4": "Mixolydian", "7sus4": "Mixolydian",
    "9sus4": "Mixolydian",
}
MODE_COLORS = {
    "Ionian": "#4C72B0", "Lydian": "#5B8FF0", "Mixolydian": "#DD8452",
    "Dorian": "#55A868", "Aeolian": "#3F8A5C", "Phrygian dom": "#E1794A",
    "Lydian dom": "#C97A3A", "Altered": "#8B0000", "H-W dim": "#B0446A",
    "W-H dim": "#C44E52", "Locrian": "#A0405A", "Melodic minor": "#2E7D57",
    "Diminished": "#C44E52", "Whole tone": "#8172B2", "Lydian aug": "#9A86C4",
}


def box(ax, x, y, text, color, w=0.19, h=0.72, fs=9, bold=False):
    """w in x-data-units (axis 0..~0.9), h in y-data-units (axis 0..n leaves)."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1,
                           edgecolor="white", facecolor=color, alpha=0.95))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold" if bold else "normal")


def q_label(q):
    return chord_label(0, q).replace("C", "", 1) or "maj"


def plot_chord_tree():
    fam_order = ["maj", "min", "dim", "aug", "sus"]
    leaves = []  # (fam, seventh_node, exact)
    for fam in fam_order:
        _, sevenths = CHORD_TREE[fam]
        for sv, exacts in sevenths.items():
            for q in exacts:
                leaves.append((fam, sv, q))
    n = len(leaves)
    fig, ax = plt.subplots(figsize=(17, 0.4 * n + 1.6))
    ys = {leaf: n - 1 - i for i, leaf in enumerate(leaves)}

    xr, xf, xs, xe, xm = 0.03, 0.22, 0.44, 0.68, 0.87
    box(ax, xr, (n - 1) / 2, "CHORD", "#333333", w=0.05, h=2.0, fs=11, bold=True)
    for fam in fam_order:
        _, sevenths = CHORD_TREE[fam]
        fam_leaves = [ys[l] for l in leaves if l[0] == fam]
        yf = sum(fam_leaves) / len(fam_leaves)
        col = FAM_COLORS[fam]
        box(ax, xf, yf, CHORD_TREE[fam][0], col, w=0.10, h=1.6, fs=11, bold=True)
        ax.plot([xr + 0.028, xf - 0.052], [(n - 1) / 2, yf], color=col, lw=1.5, alpha=0.5)
        for sv, exacts in sevenths.items():
            sv_leaves = [ys[(fam, sv, q)] for q in exacts]
            ysv = sum(sv_leaves) / len(sv_leaves)
            box(ax, xs, ysv, sv, col, w=0.13, h=0.78, fs=9)
            ax.plot([xf + 0.05, xs - 0.068], [yf, ysv], color=col, lw=1.2, alpha=0.45)
            for q in exacts:
                ye = ys[(fam, sv, q)]
                box(ax, xe, ye, q, col, w=0.13, h=0.72, fs=8)
                ax.plot([xs + 0.068, xe - 0.068], [ysv, ye], color=col, lw=1, alpha=0.4)
                # scale/mode tag on the leaf (user request)
                mode = CHORD_MODE.get(q, "")
                mc = MODE_COLORS.get(mode, "#888")
                ax.text(xm, ye, mode, ha="center", va="center", fontsize=7.5,
                        style="italic", color=mc, fontweight="bold")

    heads = [(xf, "FAMILY", "audio 94% · ceiling 99%"),
             (xs, "SEVENTH", "audio 88% · ceiling 99%"),
             (xe, "EXACT", "audio 84% · ceiling 98%"),
             (xm, "SCALE / MODE", "chord-scale")]
    for x, h, acc in heads:
        ax.text(x, n - 0.1, h, ha="center", fontsize=11, fontweight="bold")
        ax.text(x, n - 0.6, acc, ha="center", fontsize=8.5, style="italic", color="#555")
    ax.text(xr, n - 0.1, "root", ha="center", fontsize=10, fontweight="bold", color="#555")

    ax.set_xlim(0, 0.96); ax.set_ylim(-1, n + 0.3)
    ax.axis("off")
    ax.set_title("The chord tree — family → seventh → exact chord, with the scale each chord draws from",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(PLOTS / "hierarchy_chord_tree.png", dpi=140, bbox_inches="tight")
    print(f"→ {PLOTS/'hierarchy_chord_tree.png'}")


def plot_chord_modes():
    """Second view: chords grouped by the SCALE/MODE they draw from (colored by mode)."""
    by_mode = defaultdict(list)
    for fam in ["maj", "min", "dim", "aug", "sus"]:
        for _, exacts in CHORD_TREE[fam][1].items():
            for q in exacts:
                by_mode[CHORD_MODE.get(q, "?")].append((q, fam))
    # order modes brightest→darkest-ish by a rough scale-brightness ranking
    mode_order = ["Lydian", "Ionian", "Mixolydian", "Lydian dom", "Dorian",
                  "Aeolian", "Melodic minor", "Phrygian dom", "Altered", "Locrian",
                  "H-W dim", "W-H dim", "Diminished", "Whole tone", "Lydian aug"]
    modes = [m for m in mode_order if m in by_mode] + [m for m in by_mode if m not in mode_order]

    total = sum(len(v) for v in by_mode.values())
    fig, ax = plt.subplots(figsize=(13, 0.5 * total + 2))
    y = total + len(modes)
    ax.text(0.5, y + 0.5, "The same chords, grouped by their parent SCALE / MODE",
            ha="center", fontsize=13, fontweight="bold")
    for mode in modes:
        col = MODE_COLORS.get(mode, "#888")
        chords = by_mode[mode]
        box(ax, 0.16, y, f"{mode}", col, w=0.26, h=0.82, fs=11, bold=True)
        # brightness caption
        for j, (q, fam) in enumerate(chords):
            box(ax, 0.44 + j * 0.135, y, q, FAM_COLORS[fam], w=0.12, h=0.82, fs=9)
        y -= 1.5
    ax.text(0.5, y + 0.4, "box colour = chord family (blue maj · green min · red dim · "
            "purple aug · orange sus); row = the scale those chords imply",
            ha="center", fontsize=9, style="italic", color="#444")
    ax.set_xlim(0, 1.45); ax.set_ylim(y, total + len(modes) + 1.2)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(PLOTS / "hierarchy_chord_modes.png", dpi=140, bbox_inches="tight")
    print(f"→ {PLOTS/'hierarchy_chord_modes.png'}")


def plot_pattern_funnel():
    levels = [
        ("literal\n(degree, 7th)", 157, 11077, "#4C72B0"),
        ("tritone-folded\ndominants", 139, 9289, "#55A868"),
        ("degree + family\n(drop the 7th)", 60, 6879, "#DD8452"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    xmax = 11077
    for i, (name, types, tri, col) in enumerate(levels):
        y = len(levels) - 1 - i
        wnorm = tri / xmax
        ax.add_patch(FancyBboxPatch((0.5 - wnorm / 2, y - 0.3), wnorm, 0.6,
                                    boxstyle="round,pad=0.005", facecolor=col, alpha=0.9,
                                    edgecolor="white"))
        ax.text(0.5, y, f"{name}\n{tri:,} distinct 3-chord patterns · {types} chord types",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if i < len(levels) - 1:
            ax.annotate("", xy=(0.5, y - 0.35), xytext=(0.5, y - 0.65),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
    ax.text(0.5, len(levels) - 0.15, "abstracting progressions concentrates the patterns",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.9, "family level = 38% fewer patterns, +6 pts next-chord accuracy,\n"
            "and far more robust with little data — the same family layer as the chord tree",
            ha="center", fontsize=9, style="italic", color="#444")
    ax.set_xlim(0, 1); ax.set_ylim(-1.3, len(levels) + 0.2); ax.axis("off")
    ax.set_title("The pattern hierarchy — a tree of progressions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "hierarchy_pattern_funnel.png", dpi=140, bbox_inches="tight")
    print(f"→ {PLOTS/'hierarchy_pattern_funnel.png'}")


def plot_style_tree():
    # broad genre → chord-family mix (from experiment_style_prior output, hardcoded here)
    styles = {
        "jazz": dict(major=23, minor=30, dominant=40, other=7),
        "pop": dict(major=52, minor=28, dominant=17, other=3),
        "blues": dict(major=26, minor=3, dominant=70, other=1),
        "country": dict(major=70, minor=6, dominant=23, other=1),
    }
    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = ["major", "minor", "dominant", "other"]
    cols = {"major": "#4C72B0", "minor": "#55A868", "dominant": "#C44E52", "other": "#999999"}
    x0 = 0.5
    box(ax, x0, 3.3, "STYLE", "#333333", w=0.9, h=0.42, fs=11, bold=True)
    xs = [1.4, 2.4, 3.4, 4.4]
    for x, (name, mix) in zip(xs, styles.items()):
        ax.plot([x0 + 0.06, x], [3.3, 2.5], color="#aaa", lw=1.2)
        ax.text(x, 2.62, name, ha="center", fontsize=11, fontweight="bold")
        bottom = 0
        for f in order:
            h = mix[f] / 100 * 1.9
            ax.bar(x, h, bottom=bottom, width=0.5, color=cols[f],
                   edgecolor="white", label=f if x == xs[0] else None)
            if mix[f] >= 12:
                ax.text(x, bottom + h / 2, f"{mix[f]}%", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            bottom += h
    ax.set_xlim(0, 5); ax.set_ylim(0, 3.7)
    ax.axis("off")
    ax.legend(loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("The style hierarchy — each genre has a very different chord-quality mix\n"
                 "(a style prior sharpens the quality decision, especially on weak audio)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "hierarchy_style.png", dpi=140, bbox_inches="tight")
    print(f"→ {PLOTS/'hierarchy_style.png'}")


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_chord_tree()
    plot_chord_modes()
    plot_pattern_funnel()
    plot_style_tree()
