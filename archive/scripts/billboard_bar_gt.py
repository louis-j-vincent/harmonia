"""Bar-level section ground truth for the 890 McGill Billboard tracks — free.

Why this exists: every structure measurement in this repo so far was on 6-13 of
our own songs. `mirdata.initialize('billboard')` gives 890 tracks that carry
BOTH (a) precomputed NNLS `bothchroma` (`track.chroma`, (T,25) = time + 12 bass
+ 12 treble — the exact substrate `harmonia_min/sections.py` runs on) and
(b) a BAR-NOTATED annotation. No audio download, no feature extraction.

The annotation lines look like

    46.537891156 <TAB> B, bridge, | G:min7 | C:maj/5 | Bb:maj/5 | F:maj | G:maj |

so counting `|` gives the bar count of the line and `xN` multiplies it. Bar
times are then linearly interpolated between consecutive line timestamps.

WHAT THIS DOES NOT SOLVE (rule #4): the interpolation assumes a constant tempo
*within* an annotation line. A track with a ritardando or a pickup contributes
some GT error. Measured 2026-08-04: 112/120 sampled tracks are self-consistent
(>80% of lines within 12% of the song's median bar length) — so screen with
`bar_grid_consistency()` and drop the rest before using a track.

Usage:
    from billboard_bar_gt import parse, bar_grid
    meta, lines = parse(track.salami_path)
    grid = bar_grid(lines, duration)      # [(t0, bar_dur, n_bars, letter), ...]

Run directly for the self-consistency audit over the first N tracks.
"""
import re
import sys

import numpy as np


def parse(path):
    """-> (meta dict from the '# key: value' header, [line dicts])."""
    lines, meta = [], {}
    for raw in open(path):
        raw = raw.rstrip("\n")
        m = re.match(r"^#\s*(\w+):\s*(.*)$", raw)
        if m:
            meta[m.group(1)] = m.group(2)
            continue
        m = re.match(r"^([0-9.]+)\t(.*)$", raw)
        if not m:
            continue
        t, body = float(m.group(1)), m.group(2)
        lm = re.match(r"^([A-Z])'*,\s*", body)
        bars = body.count("|")
        nbar = max(0, bars - 1) if bars else 0
        rep = re.search(r"x(\d+)", body)
        if rep and nbar:
            nbar *= int(rep.group(1))
        lines.append({"t": t, "letter": lm.group(1) if lm else None,
                      "nbar": nbar, "body": body})
    return meta, lines


def bar_grid(lines, dur):
    """-> [(line_start_time, bar_duration, n_bars, section_letter_or_None)].

    A section STARTS at the first bar of any line carrying a letter.
    """
    out = []
    for i, L in enumerate(lines):
        t1 = lines[i + 1]["t"] if i + 1 < len(lines) else dur
        if L["nbar"] > 0:
            out.append((L["t"], (t1 - L["t"]) / L["nbar"], L["nbar"], L["letter"]))
    return out


def bar_edges(grid):
    """-> (edges array of len n_bars+1, [bar indices where a section starts])."""
    edges, starts = [], []
    for (t0, bd, n, letter) in grid:
        if letter:
            starts.append(len(edges))
        for j in range(n):
            edges.append(t0 + j * bd)
    if not edges:
        return np.zeros(0), []
    edges.append(edges[-1] + grid[-1][1])
    return np.asarray(edges), starts


def bar_grid_consistency(grid):
    """Fraction of lines whose bar duration is within 12% of the song median.
    Screen tracks with `>= 0.8` before trusting the grid."""
    bd = np.array([x[1] for x in grid])
    if len(bd) == 0:
        return 0.0
    med = np.median(bd)
    return float(np.mean(np.abs(bd - med) / med < 0.12)) if med > 0 else 0.0


def main(n_tracks=120):
    import mirdata
    ts = mirdata.initialize("billboard").load_tracks()
    ok = tot = 0
    barlens = []
    for k in list(ts)[:n_tracks]:
        t = ts[k]
        try:
            _, lines = parse(t.salami_path)
            grid = bar_grid(lines, float(t.chroma[-1, 0]))
        except Exception:
            continue
        if not grid:
            continue
        tot += 1
        ok += bar_grid_consistency(grid) > 0.8
        barlens.append(np.median([x[1] for x in grid]))
    print(f"tracks parsed {tot}, bar-grid self-consistent: {ok}")
    print(f"median bar duration {np.median(barlens):.3f}s "
          f"(range {np.min(barlens):.2f}-{np.max(barlens):.2f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
