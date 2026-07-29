#!/usr/bin/env python3
"""ssm_block_segment.py — 2026-07-29, branch feat/ssm-block-segmentation.

A NEW section detector that finds the DIAGONAL BLOCKS of the bar-level SSM the
way a human reads the matrix — instead of testing chord recurrence at a fixed
8/16-bar lag (the current `_sections_by_largest_unit`, which returns None and
falls back to garbage on This Love, docs/known_issues.md 2026-07-29).

The insight (Louis, 2026-07-29): a section is a *square of consistent texture*
on the diagonal. Boundaries are where the texture changes; two blocks with the
same texture are the same letter. The subtlety — a section block is itself made
of smaller repeating squares (a 2-bar Cm–Bb loop tiling an 8-bar chorus) — is
handled by BLURRING the SSM over ~one loop before looking for the big squares:
the small checkerboard fuses into a solid block, and Foote's checkerboard-kernel
novelty then finds the section edges cleanly.

Pipeline:
  1. per-bar (root_rel_to_tonic, coarse-quality) -> bar SSM (build_chord_ssm).
  2. GAUSSIAN-BLUR the SSM (sigma ~ loop length) -> big squares become solid.
  3. Foote checkerboard novelty along the diagonal -> peak-pick boundaries.
  4. label segments: mean cross-block similarity, complete-linkage threshold
     -> A/B/C (a new block that matches an earlier block gets its letter).

Run:  .venv/bin/python scratchpad/ssm_block_segment.py [song-substring]
Prints the segmentation and writes a diagnostic PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from harmonia.models.section_structure import build_chord_ssm
from section_merge_declined import _load_payload, _bar_times, _PC

# coarse quality families for the SSM's quality block (root carries most signal)
_QMAP = {"": 0, "6": 0, "^7": 0, "maj7": 0, "add9": 0,            # major-ish
         "-": 1, "-7": 1, "-6": 1, "-^7": 1, "m": 1, "min7": 1,   # minor-ish
         "7": 2, "9": 2, "13": 2, "7b9": 2,                        # dominant
         "h7": 3, "-7b5": 3,                                       # half-dim
         "o": 4, "o7": 4}                                          # dim


def _q_idx(tok: str) -> int:
    return _QMAP.get(tok, 0)


def bar_chord_seq(P: dict):
    """Per-bar (root_rel_to_tonic, q_idx) forward-filled, + bar downbeat times."""
    n = P["nBars"]
    tonic = (P.get("home") or {}).get("tonic", 0)
    ev = sorted(P["chords"], key=lambda c: (c["bar"], c.get("beat", 0)))
    rep = [None] * n
    cur, ci = None, 0
    for b in range(n):
        while ci < len(ev) and (ev[ci]["bar"], ev[ci].get("beat", 0)) <= (b, 0):
            cur, ci = ev[ci], ci + 1
        rep[b] = cur
    seq = [((c["root"] - tonic) % 12, _q_idx(c["lv"]["exact"]["q"])) if c else (-1, -1)
           for c in rep]
    names = [_PC[c["root"] % 12] + c["lv"]["exact"]["q"] if c else "·" for c in rep]
    return seq, names, _bar_times(P["chords"], n)


def gaussian_blur_ssm(S: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(S.astype(np.float64), sigma=sigma, mode="nearest")


def checkerboard_kernel(M: int) -> np.ndarray:
    """(2M x 2M) Gaussian-tapered checkerboard (Foote 2000): +1 on the two
    same-side quadrants, -1 on the cross quadrants, so it fires where the block
    to the upper-left and lower-right are internally similar but dissimilar to
    each other = a diagonal boundary."""
    g = np.outer(*[np.exp(-0.5 * (np.linspace(-2, 2, 2 * M)) ** 2)] * 2)
    sign = np.ones((2 * M, 2 * M))
    sign[:M, M:] = -1
    sign[M:, :M] = -1
    return sign * g


def novelty_curve(S: np.ndarray, M: int) -> np.ndarray:
    """Foote novelty: correlate the checkerboard kernel along the diagonal."""
    n = S.shape[0]
    K = checkerboard_kernel(M)
    nov = np.zeros(n)
    Sp = np.pad(S, M, mode="edge")
    for i in range(n):
        nov[i] = float(np.sum(Sp[i:i + 2 * M, i:i + 2 * M] * K))
    nov = np.clip(nov, 0, None)
    return nov / (nov.max() + 1e-9)


def pick_boundaries(nov: np.ndarray, min_gap: int, rel_thresh: float) -> list[int]:
    """Local maxima above ``rel_thresh`` (fraction of max), NMS by ``min_gap``."""
    n = len(nov)
    cand = [i for i in range(1, n - 1)
            if nov[i] >= rel_thresh and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    cand.sort(key=lambda i: -nov[i])
    kept: list[int] = []
    for i in cand:
        if all(abs(i - k) >= min_gap for k in kept):
            kept.append(i)
    return sorted(kept)


def _merge_runts(edges: list[int], min_bars: int) -> list[int]:
    """Drop boundaries that create a segment shorter than ``min_bars`` (a runt
    at the song head/tail, or a double-fired novelty peak) by absorbing it."""
    segs = [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]
    i = 0
    while len(segs) > 1 and i < len(segs):
        s, e = segs[i]
        if e - s < min_bars:
            if i == len(segs) - 1:
                segs[i - 1][1] = e; segs.pop(i)
            else:
                segs[i + 1][0] = s; segs.pop(i)
        else:
            i += 1
    return [segs[0][0]] + [s[1] for s in segs]


def _block_sim(S: np.ndarray, a: tuple, b: tuple) -> float:
    """Mean of the whole off-diagonal RECTANGLE between two blocks (raw SSM) —
    high iff the two spans share their chords wherever they line up. Robust to
    unequal lengths and phase (unlike a slot-aligned diagonal)."""
    (s0, e0), (s1, e1) = a, b
    r = S[s0:e0, s1:e1]
    return float(r.mean()) if r.size else 0.0


def label_segments(S: np.ndarray, edges: list[int], sim_thresh: float) -> list[str]:
    """Same letter iff cross-block similarity >= sim_thresh, judged RELATIVE to
    each block's own internal self-similarity (the user's 'compare a new block to
    the ones I've already seen'). A block matches an earlier representative when
    their rectangle mean is at least ``sim_thresh`` of the harmonic mean of the
    two blocks' self-similarities — so 'as similar to each other as each is to
    itself' = same section, scale-free across dense vs sparse textures."""
    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    self_sim = [_block_sim(S, s, s) + 1e-9 for s in segs]
    letters = "ABCDEFGHIJ"
    labs: list[str] = []
    reps: list[int] = []  # indices of representative segments
    for i, seg in enumerate(segs):
        hit = None
        for k, ri in enumerate(reps):
            denom = (self_sim[i] * self_sim[ri]) ** 0.5   # harmonic-ish scale
            if _block_sim(S, seg, segs[ri]) / denom >= sim_thresh:
                hit = k; break
        if hit is None:
            labs.append(letters[len(reps) % len(letters)]); reps.append(i)
        else:
            labs.append(letters[hit])
    return labs


def segment(P: dict, sigma: float = 2.0, kernel_bars: int = 6,
            min_gap: int = 4, rel_thresh: float = 0.15, sim_thresh: float = 0.80):
    seq, names, bt = bar_chord_seq(P)
    S = build_chord_ssm(seq)
    Sb = gaussian_blur_ssm(S, sigma)
    nov = novelty_curve(Sb, kernel_bars)
    bounds = pick_boundaries(nov, min_gap, rel_thresh)
    edges = _merge_runts([0] + bounds + [len(seq)], min_bars=4)
    labels = label_segments(S, edges, sim_thresh)   # label on RAW SSM
    segs = [{"bar0": edges[i], "bar1": edges[i + 1], "label": labels[i]}
            for i in range(len(labels))]
    return {"S": S, "Sb": Sb, "nov": nov, "bounds": edges[1:-1], "segs": segs,
            "names": names, "bar_times": bt}


def _diag_plot(P, res, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S, Sb, nov, segs = res["S"], res["Sb"], res["nov"], res["segs"]
    n = S.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4),
                           gridspec_kw={"width_ratios": [1, 1, 0.5]})
    for a, M, t in ((ax[0], S, "raw chord SSM"), (ax[1], Sb, f"blurred (σ={2.0})")):
        a.imshow(M, cmap="Greys", origin="upper", interpolation="nearest")
        a.set_title(t, fontsize=10)
        for s in res["bounds"]:
            a.axvline(s, color="#8a2b2b", lw=0.9); a.axhline(s, color="#8a2b2b", lw=0.9)
        a.set_xlabel("bar")
    for seg in segs:
        ax[1].text((seg["bar0"] + seg["bar1"]) / 2, -2, seg["label"],
                   ha="center", color="#1f8a5b", fontsize=11, fontweight="bold")
    ax[2].plot(nov, np.arange(n), color="#c58a2e")
    ax[2].invert_yaxis(); ax[2].set_title("checkerboard\nnovelty", fontsize=10)
    for s in res["bounds"]:
        ax[2].axhline(s, color="#8a2b2b", lw=0.7, alpha=0.6)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("saved", out_path)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    P = _load_payload(f)
    res = segment(P)
    slug = f.stem.replace("inferred_", "")
    print(f"\n{slug}  ({P['nBars']} bars, key {P.get('keyName')})")
    print("boundaries (bars):", res["bounds"])
    form = "".join(f"{s['label']}" for s in res["segs"])
    print("form:", " ".join(f"{s['label']}[{s['bar0']}-{s['bar1']}]" for s in res["segs"]))
    print("form string:", form)
    # debug: relative cross-block similarity matrix (what labeling clusters on)
    S = res["S"]; segs = [(s["bar0"], s["bar1"]) for s in res["segs"]]
    ss = [_block_sim(S, s, s) + 1e-9 for s in segs]
    print("\nrelative block-similarity matrix (>=sim_thresh -> same letter):")
    hdr = "      " + " ".join(f"{s['label']}{s['bar0']:>2}" for s in res["segs"])
    print(hdr)
    for i, si in enumerate(segs):
        row = " ".join(f"{_block_sim(S, si, sj) / (ss[i]*ss[j])**0.5:4.2f}"
                       for j, sj in enumerate(segs))
        print(f"  {res['segs'][i]['label']}{si[0]:>2}  {row}")
    _diag_plot(P, res, str(REPO / "scratchpad" / f"ssm_blocks_{slug}.png"),
               f"{slug} — SSM block segmentation")


if __name__ == "__main__":
    main()
