"""Head-to-head, Louis 2026-08-05: « soit on fait par rapport à un percentile
de la matrice SSM, style 90 %, soit on détecte les pics directement, à voir
lequel des deux généralise le mieux. »

Two families replace the fixed `TILE_MIN = 0.80` of `harmonia_min/sections.py`:

  (A) PER-SONG PERCENTILE — a bar pair counts as "similar" when its cosine is
      above the q-th percentile of THAT SONG's own off-diagonal SSM values.
      Still one knob, but scale-free per song: a homogeneous song and a varied
      song get thresholds that mean the same thing.
  (B) DIRECT PEAK DETECTION — no absolute threshold anywhere. On each SSM row,
      `scipy.signal.find_peaks` with a PROMINENCE floor expressed as a fraction
      of that row's own dynamic range (max - median over |lag| >= 2).

Generalisation test (both families have a knob, so best-vs-best is not a test):
  1. songs split 60/40 by SONG, fixed seed;
  2. each family's knob chosen on the TUNE split only;
  3. both reported on the HELD-OUT split, honest random tie-break, with the
     trivial "never move" baseline and the shipped TILE_MIN = 0.80;
  4. per-song SPREAD of the operating point reported next to the mean.

Harness = the 285-track Billboard bar-GT set (`scripts/bibar_prep.py`), same
metric as `scripts/tiling_v2.py`. Nothing under harmonia_min/ is touched.

    python scripts/pattern_quantile_billboard.py [n_tracks]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from bibar_binary_ssm import density, placement, run_edge_score  # noqa: E402
from bibar_sweep import load                                     # noqa: E402
from tiling_v2 import bar_vecs, nov_bar, _z, _pad                # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")

TILE_MIN = 0.80
MIN_LAG = 2          # |lag| >= 2 defines "off-diagonal" everywhere here
Q_GRID = (0.80, 0.85, 0.90, 0.95, 0.975)
P_GRID = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50)


# ── the two families, as binary bar-level SSMs ──────────────────────────────
_SSM: dict = {}
_NOV: dict = {}


def bar_ssm(s) -> np.ndarray:
    """Continuous bar-level SSM — the matrix `tiling_runs` reads, rebuilt on
    the Billboard half-bar chroma (same code path as scripts/tiling_v2.py)."""
    k = s["key"]
    if k not in _SSM:
        V = bar_vecs(s["Hb"])
        _SSM[k] = V @ V.T
    return _SSM[k]


def nov(s):
    if s["key"] not in _NOV:
        _NOV[s["key"]] = _pad(nov_bar(s["Hb"], 0.0), s["nb"])
    return _NOV[s["key"]]


def _offdiag(S):
    n = len(S)
    i, j = np.indices((n, n))
    return S[np.abs(i - j) >= MIN_LAG]


def M_fixed(S, tau=TILE_MIN):
    """What ships: one constant for every song."""
    return S >= tau


def M_quantile(S, q):
    """(A) per-song percentile. Density is q-invariant by construction:
    exactly (1-q) of the off-diagonal mass survives, on EVERY song."""
    return S >= float(np.quantile(_offdiag(S), q))


def M_peaks(S, prom_frac, sym=True):
    """(B) direct peak detection, per row, no absolute threshold.

    prominence floor = prom_frac * (row max - row median), both computed on
    that row's own |lag| >= 2 values. Symmetrised with OR because a repeat is
    symmetric in time and `run_edge_score` reads both signs of the lag.
    """
    n = len(S)
    M = np.zeros((n, n), bool)
    for b in range(n):
        row = S[b]
        keep = np.abs(np.arange(n) - b) >= MIN_LAG
        if keep.sum() < 3:
            continue
        span = float(row[keep].max() - np.median(row[keep]))
        pk, _ = find_peaks(row, prominence=max(prom_frac * span, 1e-6))
        pk = pk[np.abs(pk - b) >= MIN_LAG]
        M[b, pk] = True
    return (M | M.T) if sym else M


BUILDERS = {
    "fixed": lambda S, k: M_fixed(S, k),
    "quantile": lambda S, k: M_quantile(S, k),
    "peaks": lambda S, k: M_peaks(S, k),
}


# ── scoring ─────────────────────────────────────────────────────────────────
def cue_scores(songs, family, knob, fuse=False):
    """-> (per-song list of signed bar errors, per-song density)."""
    per_song, dens = [], []
    for s in songs:
        S = bar_ssm(s)
        M = BUILDERS[family](S, knob) if family else np.zeros_like(S, bool)
        v = run_edge_score(M, "edgesum") if family else np.zeros(s["nb"])
        v = _pad(v, s["nb"])
        if fuse:
            v = _z(v) + _z(nov(s))
        per_song.append(placement(v, s["starts"], s["nb"]))
        dens.append(density(M) if family else 0.0)
    return per_song, dens


def summarise(per_song):
    """mean exact-bar over STARTS, plus the spread over SONGS."""
    flat = np.array([e for es in per_song for e in es])
    ps = np.array([np.mean(np.array(es) == 0) * 100
                   for es in per_song if len(es)])
    return dict(exact=float((flat == 0).mean() * 100),
                pm1=float((np.abs(flat) <= 1).mean() * 100),
                med=float(np.median(np.abs(flat))),
                n_starts=int(len(flat)), n_songs=int(len(ps)),
                song_p25=float(np.percentile(ps, 25)),
                song_med=float(np.median(ps)),
                song_p75=float(np.percentile(ps, 75)),
                song_below_chance=float((ps <= 10.8).mean() * 100))


def row(tag, r, dens=None):
    d = f"   dens {np.mean(dens)*100:5.1f}%±{np.std(dens)*100:4.1f}" if dens \
        else ""
    print(f"  {tag:<42s} exact {r['exact']:5.1f}%  |e|<=1 {r['pm1']:5.1f}%  "
          f"song med {r['song_med']:5.1f}% [p25 {r['song_p25']:4.1f} p75 "
          f"{r['song_p75']:5.1f}]  <=chance {r['song_below_chance']:4.1f}%"
          f"{d}")


def main(n=None):
    songs = load(n)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(songs))
    cut = int(round(0.6 * len(songs)))
    tune = [songs[i] for i in idx[:cut]]
    held = [songs[i] for i in idx[cut:]]
    print(f"{len(songs)} Billboard tracks -> tune {len(tune)} / held-out "
          f"{len(held)} (seed 0, split by SONG)")
    print(f"tune starts {sum(len(s['starts']) for s in tune)}, held-out "
          f"{sum(len(s['starts']) for s in held)}")
    print("ties broken AT RANDOM (docs/known_issues.md ★★); trivial "
          "'never move' printed below\n")
    res = {"split": [len(tune), len(held)]}

    for fuse in (False, True):
        tagf = "FUSED with un-blurred novelty" if fuse else \
            "STANDALONE (run-edge cue only)"
        print(f"══ TUNE split — {tagf} ══")
        base_t, base_d = cue_scores(tune, None, None, fuse)
        row("trivial: never move", summarise(base_t))
        pst, dst = cue_scores(tune, "fixed", TILE_MIN, fuse)
        row(f"TILE_MIN = {TILE_MIN} (what ships)", summarise(pst), dst)
        best = {}
        for fam, grid in (("quantile", Q_GRID), ("peaks", P_GRID)):
            for k in grid:
                ps, dd = cue_scores(tune, fam, k, fuse)
                r = summarise(ps)
                row(f"({'A' if fam == 'quantile' else 'B'}) {fam} {k}", r, dd)
                if fam not in best or r["exact"] > best[fam][1]["exact"]:
                    best[fam] = (k, r)
        print()

        print(f"══ HELD-OUT split — {tagf} ══  "
              f"(knobs frozen: quantile q={best['quantile'][0]}, "
              f"peaks prom={best['peaks'][0]})")
        out = {}
        hb, _ = cue_scores(held, None, None, fuse)
        out["never_move"] = summarise(hb)
        row("trivial: never move", out["never_move"])
        hf, hfd = cue_scores(held, "fixed", TILE_MIN, fuse)
        out["fixed"] = summarise(hf)
        out["fixed"]["dens"] = [float(np.mean(hfd)), float(np.std(hfd))]
        row(f"TILE_MIN = {TILE_MIN} (what ships)", out["fixed"], hfd)
        for fam in ("quantile", "peaks"):
            k = best[fam][0]
            hp, hpd = cue_scores(held, fam, k, fuse)
            out[fam] = summarise(hp)
            out[fam]["knob"] = k
            out[fam]["dens"] = [float(np.mean(hpd)), float(np.std(hpd))]
            row(f"({'A' if fam == 'quantile' else 'B'}) {fam} {k}", out[fam],
                hpd)
        res["fused" if fuse else "standalone"] = out
        print()

    # ── the operating point itself: what absolute cosine does q=0.90 mean? ──
    print("══ OPERATING POINT SPREAD — the thing the constant gets wrong ══")
    tv = []
    for s in songs:
        S = bar_ssm(s)
        od = _offdiag(S)
        tv.append([float(np.quantile(od, q)) for q in Q_GRID] +
                  [float((od >= TILE_MIN).mean())])
    tv = np.array(tv)
    for i, q in enumerate(Q_GRID):
        print(f"  cosine at the {q:.3f} percentile: median {np.median(tv[:,i]):.3f}"
              f"  [p10 {np.percentile(tv[:,i],10):.3f}, "
              f"p90 {np.percentile(tv[:,i],90):.3f}]")
    d = tv[:, -1] * 100
    print(f"  share of the matrix above the FIXED 0.80: median {np.median(d):.1f}%"
          f"  [p10 {np.percentile(d,10):.1f}%, p90 {np.percentile(d,90):.1f}%]"
          f"  -> the constant means a different thing on every song")
    res["op_point"] = dict(q_grid=list(Q_GRID),
                           q_cos=tv[:, :len(Q_GRID)].tolist(),
                           fixed_density=d.tolist())

    json.dump(res, open(CACHE / "pattern_quantile.json", "w"), indent=1)
    print("\nsaved", CACHE / "pattern_quantile.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
