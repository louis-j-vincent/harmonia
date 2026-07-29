#!/usr/bin/env python3
"""ssm_template_scan.py — 2026-07-29, branch feat/ssm-block-segmentation.

Louis's matched-filter variant of the block segmenter: instead of Foote's FIXED
checkerboard kernel, use the section's OWN opening texture as the kernel and
slide it along the diagonal. While the w×w diagonal window still looks like the
template (high cosine) we're in the same section; when it drops, that's a
boundary and the template resets to the new window. Because the same template
also lights up wherever that texture RECURS, one pass gives boundaries AND
letters (a new section reuses an earlier letter if its template matches).

This file demonstrates + compares to the checkerboard version in
ssm_block_segment.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from ssm_block_segment import bar_chord_seq, gaussian_blur_ssm
from harmonia.models.section_structure import build_chord_ssm
from section_merge_declined import _load_payload


def _win(M: np.ndarray, p: int, w: int) -> np.ndarray:
    return M[p:p + w, p:p + w].ravel()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def template_response(S: np.ndarray, start: int, w: int) -> np.ndarray:
    """Matched-filter response: cosine of the w×w window at each diagonal
    position against the template taken at ``start``. High = same texture as
    section ``start`` (so its recurrences light up too)."""
    n = S.shape[0]
    tmpl = _win(S, start, w)
    resp = np.full(n, np.nan)
    for p in range(0, n - w + 1):
        resp[p] = _cos(_win(S, p, w), tmpl)
    return resp


def segment_matched(S: np.ndarray, w: int = 4, drop: float = 0.80,
                    min_len: int = 4, same_thresh: float = 0.85):
    """Sequential region-growing with a self-template + one-pass labelling.

    Grow the current section while the window still matches its template
    (cosine >= ``drop`` of the section's running-max match); on a drop, open a
    new section, reset the template, and give it an EARLIER section's letter if
    its opening template matches (cosine >= ``same_thresh``), else a new letter.
    """
    n = S.shape[0]
    letters = "ABCDEFGHIJ"
    bounds: list[int] = []
    seg_start = 0
    tmpl = _win(S, 0, w)
    run_max = 1.0
    seg_templates: list[tuple[str, np.ndarray]] = []
    seg_letters: list[str] = []

    def open_section(start: int):
        t = _win(S, start, w)
        hit = next((lab for lab, tt in seg_templates if _cos(t, tt) >= same_thresh), None)
        lab = hit or letters[len({l for l, _ in seg_templates}) % len(letters)]
        seg_templates.append((lab, t))
        seg_letters.append(lab)
        return t

    open_section(0)  # first section
    recent: list[float] = []          # trailing window of raw scores (loop-smoothing)
    below = 0                          # consecutive smoothed bars below threshold
    for p in range(1, n - w + 1):
        recent.append(_cos(_win(S, p, w), tmpl))
        if len(recent) > 3:
            recent.pop(0)
        s = sum(recent) / len(recent)  # smoothed over ~one loop -> kills the jag
        if s >= drop * run_max:
            run_max = max(run_max, s); below = 0
            continue
        below += 1
        # require a SUSTAINED drop (2 smoothed bars) so a 1-bar dip isn't a cut
        if (p - seg_start) >= min_len and below >= 2:
            bounds.append(p - 1)      # texture changed -> boundary
            seg_start = p - 1
            tmpl = open_section(p - 1)
            run_max = 1.0; recent = []; below = 0
    edges = [0] + bounds + [n]
    segs = [{"bar0": edges[i], "bar1": edges[i + 1], "label": seg_letters[i]}
            for i in range(len(seg_letters))]
    return segs, bounds


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    f = next((REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    P = _load_payload(f)
    seq, names, bt = bar_chord_seq(P)
    # RAW SSM: the matched filter keys on INTERNAL texture (the chorus's 2-bar
    # checkerboard vs the verse's pattern) — blurring would erase exactly that.
    # (Foote's checkerboard wants blur; the self-template wants raw.)
    S = build_chord_ssm(seq)
    n = S.shape[0]
    slug = f.stem.replace("inferred_", "")

    segs, bounds = segment_matched(S)
    print(f"{slug} ({n} bars) — matched-filter (self-template) form:")
    print("  ", " ".join(f"{s['label']}[{s['bar0']}-{s['bar1']}]" for s in segs))
    print("   form string:", "".join(s["label"] for s in segs))

    # plot: blurred SSM + the OPENING-template response curve (the key picture:
    # high in section 0's texture, drops at its boundary, re-lights where it recurs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    resp0 = template_response(S, 0, 4)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 0.6]})
    ax[0].imshow(S, cmap="Greys", origin="upper", interpolation="nearest")
    for b in bounds:
        ax[0].axvline(b, color="#8a2b2b", lw=0.9); ax[0].axhline(b, color="#8a2b2b", lw=0.9)
    for s in segs:
        ax[0].text((s["bar0"] + s["bar1"]) / 2, -2, s["label"], ha="center",
                   color="#1f8a5b", fontsize=11, fontweight="bold")
    ax[0].set_title(f"{slug} — matched-filter segmentation", fontsize=10)
    ax[0].set_xlabel("bar")
    ax[1].plot(np.arange(n), resp0, color="#c58a2e")
    ax[1].axhline(0.80, color="#999", lw=0.6, ls="--")
    for b in bounds:
        ax[1].axvline(b, color="#8a2b2b", lw=0.6, alpha=0.5)
    ax[1].set_title("response of the OPENING template\n(slid along the diagonal)", fontsize=9)
    ax[1].set_xlabel("bar"); ax[1].set_ylim(0, 1.02)
    plt.tight_layout()
    out = REPO / "scratchpad" / f"ssm_matched_{slug}.png"
    plt.savefig(out, dpi=120)
    print("   saved", out)


if __name__ == "__main__":
    main()
