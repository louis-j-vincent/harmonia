"""Measure Louis's binary bi-bar SSM against the shipped cue, on the SAME
Billboard bar-GT starts, with the SAME metric as the lit review.

Numbers to beat (docs/research_sessions/structure_literature_2026-08-04.md §1.2,
694 starts / 80 tracks): 34.7% exact-bar for our chroma novelty, 35.7% for CBM.
Every cue below is re-run here on this set, so the baseline is recomputed,
never quoted.

Two guards that are NOT optional:
  * placement ties are broken AT RANDOM (a distance tie-break gives a
    saturated all-ones matrix a free 100%);
  * every accuracy is printed next to the SSM's off-band density, so a
    saturated matrix is visible rather than inferred.

    python scripts/bibar_sweep.py [n_tracks]
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from bibar_binary_ssm import (bibar, bin_ssm, bin_ssm_rot, binarise,  # noqa: E402
                              chordstring_score, chordtone_binary, cont_ssm,
                              density, novelty_score, placement, run_edge_score)

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")


def load(n=None):
    z = np.load(CACHE / "bibar_prep.npz", allow_pickle=True)
    keys = list(z["__keys__"])[:n]
    return [dict(Cb=z[f"{k}/Cb"], Hb=z[f"{k}/Hb"], sym=z[f"{k}/sym"],
                 starts=z[f"{k}/starts"], letters=z[f"{k}/letters"],
                 nb=int(z[f"{k}/nb"]), key=k) for k in keys]


def line(name, errs, dens=None):
    e = np.array(errs)
    d = f"  dens {np.mean(dens)*100:4.1f}%" if dens else ""
    print(f"  {name:<44s} exact {np.mean(e == 0)*100:5.1f}%   "
          f"|err|<=1 {np.mean(np.abs(e) <= 1)*100:5.1f}%   "
          f"med|e| {np.median(np.abs(e)):.1f}{d}   n={len(e)}")
    return float(np.mean(e == 0) * 100)


def run(songs, build):
    """build(song) -> M. Returns (errs, densities)."""
    errs, dens = [], []
    for s in songs:
        M = build(s)
        dens.append(density(M))
        errs += placement(run_edge_score(M), s["starts"], s["nb"])
    return errs, dens


def main(n=None):
    songs = load(n)
    print(f"{len(songs)} tracks, "
          f"{sum(len(s['starts']) for s in songs)} annotated section starts\n")
    out = {}

    print("REFERENCES (recomputed on THIS set, not quoted)")
    out["novelty"] = line("chroma checkerboard novelty (what we ship)",
                          [e for s in songs
                           for e in placement(novelty_score(s["Hb"]),
                                              s["starts"], s["nb"])])
    out["chordstring"] = line("chord-string 8-bar repeat (lit review #1)",
                              [e for s in songs
                               for e in placement(chordstring_score(s["sym"]),
                                                  s["starts"], s["nb"])])

    print("\nBINARY BI-BAR LAG MATRIX (Louis, 2026-08-05)")
    grid = []
    for mode, kb, kt in (("topk", 1, 3), ("topk", 2, 4), ("topk", 3, 5),
                         ("quantile", 0, 0), ("relmax", 0, 0)):
        for join in ("concat", "union"):
            for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                tag = (f"{mode}{f'(kb{kb},kt{kt})' if mode == 'topk' else ''}"
                       f"/{join}/tau{tau}")
                e, d = run(songs, lambda s, m=mode, a=kb, b=kt, j=join, t=tau:
                           bin_ssm(bibar(binarise(s["Cb"], m, a, b), j), t))
                grid.append((np.mean(np.array(e) == 0) * 100, tag, e, d))
    grid.sort(key=lambda x: -x[0])
    for acc, tag, e, d in grid[:6]:
        line(tag, e, d)
    print(f"  ... {len(grid)-6} weaker settings hidden "
          f"(worst {grid[-1][0]:.1f}%, {grid[-1][1]})")
    out["binary_best"], out["binary_best_tag"] = grid[0][0], grid[0][1]

    print("\nCONTROL — identical lag-run reading on CONTINUOUS cosine")
    print("  (isolates BINARISING: same unit, same threshold shape, same reader)")
    cg = []
    for join in ("concat", "union"):
        for tau in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
            e, d = run(songs, lambda s, j=join, t=tau: cont_ssm(s["Cb"], j, t))
            cg.append((np.mean(np.array(e) == 0) * 100,
                       f"continuous/{join}/tau{tau}", e, d))
    cg.sort(key=lambda x: -x[0])
    for acc, tag, e, d in cg[:4]:
        line(tag, e, d)
    out["cont_best"], out["cont_best_tag"] = cg[0][0], cg[0][1]

    print("\nBINARY BI-BAR on CHORD TONES (symbolic binarisation)")
    sg = []
    for join in ("concat", "union"):
        for tau in (0.6, 0.7, 0.8, 0.9, 1.0):
            e, d = run(songs, lambda s, j=join, t=tau:
                       bin_ssm(bibar(chordtone_binary(s["sym"]), j), t))
            sg.append((np.mean(np.array(e) == 0) * 100,
                       f"chordtone/{join}/tau{tau}", e, d))
    sg.sort(key=lambda x: -x[0])
    for acc, tag, e, d in sg[:4]:
        line(tag, e, d)
    out["chordtone_best"], out["chordtone_best_tag"] = sg[0][0], sg[0][1]

    print("\nTRANSPOSITION INVARIANCE — max over the 12 semitone rotations")
    print("  (Sunny: a repeat a semitone up scores 0.797 raw, 0.973 rotated)")
    rg = []
    best_tag = out["binary_best_tag"]
    m0, rest = best_tag.split("(")[0], best_tag
    kb, kt = (int(rest.split("kb")[1].split(",")[0]),
              int(rest.split("kt")[1].split(")")[0])) if "kb" in rest else (2, 4)
    join = rest.split("/")[1]
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        e, d = run(songs, lambda s, t=tau:
                   bin_ssm_rot(bibar(binarise(s["Cb"], m0, kb, kt), join), t))
        rg.append((np.mean(np.array(e) == 0) * 100,
                   f"ROT {m0}/{join}/tau{tau}", e, d))
    rg.sort(key=lambda x: -x[0])
    for acc, tag, e, d in rg[:4]:
        line(tag, e, d)
    out["rot_best"], out["rot_best_tag"] = rg[0][0], rg[0][1]
    print(f"  best WITHOUT rotation on the same family: "
          f"{out['binary_best']:.1f}% ({best_tag})")

    np.savez(CACHE / "bibar_sweep.npz",
             **out,
             grid=np.array([(a, t) for a, t, _, _ in grid], dtype=object),
             cgrid=np.array([(a, t) for a, t, _, _ in cg], dtype=object),
             sgrid=np.array([(a, t) for a, t, _, _ in sg], dtype=object),
             rgrid=np.array([(a, t) for a, t, _, _ in rg], dtype=object))
    print("\nsaved", CACHE / "bibar_sweep.npz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
