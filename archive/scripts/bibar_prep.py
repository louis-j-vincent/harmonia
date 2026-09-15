"""Cache the Billboard bar-level substrate once, so the bi-bar sweeps are fast.

Per track we keep exactly what any section-placement cue could need:
  Cb   (n_bars, 24)  mean NNLS bothchroma per BAR (bass 12 + treble 12)
  sym  (n_bars,)     one majmin chord symbol per bar (sampled at the bar midpoint)
  starts             bar indices where the annotator STARTS a lettered section
  letters            the SALAMI letter of each start
  edges              (n_bars+1,) bar times, from `billboard_bar_gt.bar_grid`

Screened with `bar_grid_consistency() >= 0.8` (docs: 112/120 sampled tracks pass).

    python scripts/bibar_prep.py 200
    -> <scratchpad>/bibar_prep.npz
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from billboard_bar_gt import bar_grid, bar_grid_consistency, parse  # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")


def prep_track(t):
    _, lines = parse(t.salami_path)
    ch = np.asarray(t.chroma)
    dur = float(ch[-1, 0])
    g = bar_grid(lines, dur)
    if len(g) < 4 or bar_grid_consistency(g) < 0.8:
        return None
    edges, starts, letters = [], [], []
    for (t0, bd, n, L) in g:
        if L:
            starts.append(len(edges))
            letters.append(L)
        for j in range(n):
            edges.append(t0 + j * bd)
    edges.append(edges[-1] + g[-1][1])
    edges = np.asarray(edges)
    nb = len(edges) - 1
    if nb < 32 or len(starts) < 3:
        return None

    times, arr = ch[:, 0], ch[:, 1:]           # (T,), (T, 24)

    def pool(t0, t1):
        sel = (times >= t0) & (times < t1)
        if sel.any():
            return arr[sel].mean(0)
        return arr[int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))]

    Cb = np.array([pool(edges[b], edges[b + 1]) for b in range(nb)])
    # half-bars too: `sections.py` detects at HALF-bar grain, and reusing a
    # bar-constant copy instead would understate the shipped baseline.
    Hb = np.zeros((2 * nb, 24))
    for b in range(nb):
        m = 0.5 * (edges[b] + edges[b + 1])
        Hb[2 * b] = pool(edges[b], m)
        Hb[2 * b + 1] = pool(m, edges[b + 1])

    cd = t.chords_majmin
    iv, lab = cd.intervals, cd.labels
    mids = 0.5 * (edges[:-1] + edges[1:])
    idx = np.searchsorted(iv[:, 0], mids, "right") - 1
    sym = np.array([lab[j] if 0 <= j < len(lab) else "N" for j in idx])
    return dict(Cb=Cb, Hb=Hb, sym=sym, starts=np.array(starts, int),
                letters=np.array(letters), edges=edges, nb=nb)


def main(n=250):
    import mirdata
    ts = mirdata.initialize("billboard").load_tracks()
    out, keys = {}, []
    for k in list(ts)[:n]:
        try:
            r = prep_track(ts[k])
        except Exception:
            r = None
        if r is None:
            continue
        keys.append(k)
        for f, v in r.items():
            out[f"{k}/{f}"] = v
    out["__keys__"] = np.array(keys)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "bibar_prep.npz", **out)
    nbs = [out[f"{k}/nb"] for k in keys]
    ss = sum(len(out[f"{k}/starts"]) for k in keys)
    print(f"kept {len(keys)}/{n} tracks · {ss} section starts · "
          f"median {np.median(nbs):.0f} bars")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
