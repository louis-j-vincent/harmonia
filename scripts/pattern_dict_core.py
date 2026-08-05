"""Shared core for the "pattern dictionary" study (Louis, 2026-08-05).

Two questions, one substrate:

  (Q1) « il y a des pics clairs qui doivent correspondre à un quartile des
        valeurs de la matrice SSM » — at what QUANTILE of a song's own
        off-diagonal SSM distribution do the visible row peaks sit?
  (Q2) « en cumulant les demi-barres adjacentes on trouve un signal robuste »
        — bar grain vs half-bar grain vs summed-adjacent-half-bars.
  (Q3) the pattern dictionary: take the first repetition motif, slide it
        along the X AXIS of the SSM (not along the diagonal), dot-product with
        the square it lands on, plot that against offset.

Nothing under harmonia_min/ is touched. The bar-level SSM is the one
`scripts/ssm_rows_plot.bar_ssm` already builds by spying on the live pipeline;
it is cached to the scratchpad so the pipeline runs once per song.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")

SONGS = [
    ("maroon_5_this_love", "This Love", True),
    ("norah_jones_don_t_know_why", "Don't Know Why", True),
    ("bobby_hebb_sunny_official_audio", "Sunny", True),
    ("michael_jackson_billie_jean_official_video", "Billie Jean", False),
    ("ray_charles_georgia_on_my_mind_official_video", "Georgia On My Mind",
     False),
]


# ── substrate, cached ───────────────────────────────────────────────────────
def song_data(stem: str) -> dict:
    """{Sb (bar SSM), Sh (half-bar SSM), F (half-bar feats), n, segs}.

    Runs `harmonia_min.pipeline.analyze` ONCE per song and pickles the result:
    the bar SSM here is byte-identical to the one `tiling_runs` reads.
    """
    fp = CACHE / f"patdict_{stem}.pkl"
    if fp.exists():
        return pickle.loads(fp.read_bytes())
    import copy
    from harmonia_min import sections as hs
    from harmonia_min import pipeline as _pl
    from harmonia_min.folding import _bar_vecs
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None):
        out = real(grid, arr, times, bars)
        cap.update(grid=grid, arr=arr, times=times, segs=copy.deepcopy(out),
                   bars=copy.deepcopy(bars))
        return out

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x",
                    file_key="x", audio_url="")
    finally:
        hs.detect_sections = real
    F = hs.halfbar_features(cap["grid"], np.asarray(cap["arr"]), cap["times"])
    n = len(cap["grid"]) - 1
    Vb = _bar_vecs(F, n)
    d = dict(Sb=Vb @ Vb.T, Sh=F @ F.T, Vb=Vb, F=F, n=n, segs=cap["segs"],
             grid=list(cap["grid"]))
    fp.write_bytes(pickle.dumps(d))
    return d


# ── off-diagonal distribution + quantile helpers ────────────────────────────
def offdiag(S: np.ndarray, min_lag: int = 2) -> np.ndarray:
    """All SSM values with |lag| >= min_lag — the song's own background."""
    n = len(S)
    i, j = np.indices((n, n))
    return S[np.abs(i - j) >= min_lag]


def q_of(v: np.ndarray, x: float) -> float:
    """Quantile (0-1) that value x occupies in the distribution v."""
    return float(np.searchsorted(np.sort(v), x) / max(1, len(v)))


# ── peak picking on one SSM row ─────────────────────────────────────────────
def row_peaks(row: np.ndarray, b: int, min_lag: int = 2,
              prom_frac: float = 0.25) -> list[dict]:
    """The peaks a human sees on row `b` when it is drawn as a 1-D signal.

    Definition, deliberately crude and stated in full so it can be argued with:
      * a candidate is a strict local maximum at |lag| >= min_lag;
      * its PROMINENCE is its height minus the higher of the two minima
        reached before hitting a higher point on either side (the standard
        topographic definition);
      * it is kept when prominence >= prom_frac * (row max off-diag - row
        median off-diag), i.e. a fraction of the row's own dynamic range.
    No absolute threshold enters anywhere — that is the point of the exercise.
    """
    n = len(row)
    ok = np.abs(np.arange(n) - b) >= min_lag
    idx = np.flatnonzero(ok)
    if len(idx) < 3:
        return []
    lo, hi = float(np.median(row[idx])), float(row[idx].max())
    span = max(hi - lo, 1e-9)
    out = []
    for k in idx:
        if k - 1 < 0 or k + 1 >= n:
            continue
        if not (row[k] >= row[k - 1] and row[k] >= row[k + 1]):
            continue
        if row[k] == row[k - 1] and row[k] == row[k + 1]:
            continue
        # topographic prominence
        left = row[:k]
        right = row[k + 1:]
        lmin = row[k]
        for x in left[::-1]:
            if x > row[k]:
                break
            lmin = min(lmin, x)
        else:
            lmin = min(lmin, left.min()) if len(left) else lmin
        rmin = row[k]
        for x in right:
            if x > row[k]:
                break
            rmin = min(rmin, x)
        else:
            rmin = min(rmin, right.min()) if len(right) else rmin
        prom = row[k] - max(lmin, rmin)
        if prom >= prom_frac * span:
            out.append(dict(j=int(k), lag=int(k - b), val=float(row[k]),
                            prom=float(prom)))
    return out


def all_peaks(S: np.ndarray, min_lag: int = 2, prom_frac: float = 0.25):
    return [p | {"b": b} for b in range(len(S))
            for p in row_peaks(S[b], b, min_lag, prom_frac)]


# ── (Q2) three grains of the same row signal ────────────────────────────────
def grain_signals(Sh: np.ndarray, n_bars: int):
    """-> dict of (name -> (matrix, unit_in_bars)).

    half : the raw half-bar SSM (2n x 2n), lag counted in half-bars.
    bar  : the bar-level SSM used by tiling_runs (concatenated half-bar pairs
           -> equivalently the mean of the 2x2 aligned half-bar block).
    sum2 : Louis's « en cumulant les demi-barres adjacentes » — the half-bar
           SSM summed over ADJACENT half-bars at the same lag, i.e.
           T[i, j] = Sh[i, j] + Sh[i+1, j+1], still on the half-bar stride.
           This keeps half-bar resolution (stride 1) while pooling two of them.
    """
    n = len(Sh)
    T = np.zeros_like(Sh)
    T[:n - 1, :n - 1] = 0.5 * (Sh[:n - 1, :n - 1] + Sh[1:, 1:])
    T[n - 1, :] = Sh[n - 1, :]
    T[:, n - 1] = Sh[:, n - 1]
    Sb2 = 0.5 * (Sh[0::2, 0::2] + Sh[1::2, 1::2])[:n_bars, :n_bars]
    return {"half": (Sh, 0.5), "sum2": (T, 0.5), "bar": (Sb2, 1.0)}


def peak_contrast(S: np.ndarray, min_lag_units: int, prom_frac: float = 0.25):
    """How much do the peaks stand out from the background, on this matrix?

    Returns (mean peak height - median background) / std(background), plus the
    mean prominence in the same std units and the peak count per row. Higher =
    cleaner signal. Scale-free, so the three grains are comparable.
    """
    bg = offdiag(S, min_lag_units)
    med, sd = float(np.median(bg)), float(np.std(bg)) or 1e-9
    ps = all_peaks(S, min_lag_units, prom_frac)
    if not ps:
        return dict(z_height=0.0, z_prom=0.0, per_row=0.0, n=0)
    h = np.array([p["val"] for p in ps])
    pr = np.array([p["prom"] for p in ps])
    return dict(z_height=float((h.mean() - med) / sd),
                z_prom=float(pr.mean() / sd),
                per_row=len(ps) / len(S), n=len(ps))


# ── (Q3) the pattern dictionary ─────────────────────────────────────────────
def first_motif(S: np.ndarray, min_lag: int = 2, prom_frac: float = 0.25,
                q_keep: float = None):
    """Louis: « le premier bloc de répétition est le premier motif de
    répétition qui démarre sur le temps 1 d'une mesure ».

    On the BAR-level SSM every index is already a bar downbeat, so "starts on
    beat 1" is automatic; what has to be chosen is WHICH bar and WHICH length.
    Rule: scan bars in order; the motif is the first bar b that carries a kept
    peak at a positive lag L >= 2, and the motif is bars [b, b+L-1] — L is the
    period the peak announces, so the motif is exactly one period long.

    `q_keep` (optional) additionally requires the peak value to sit above that
    quantile of the song's own off-diagonal distribution — the adaptive
    threshold this study is testing.
    """
    bg = offdiag(S, min_lag)
    thr = float(np.quantile(bg, q_keep)) if q_keep is not None else -np.inf
    for b in range(len(S)):
        ps = [p for p in row_peaks(S[b], b, min_lag, prom_frac)
              if p["lag"] >= min_lag and p["val"] >= thr]
        if not ps:
            continue
        p = max(ps, key=lambda p: (p["prom"], p["val"]))
        L = p["lag"]
        if b + 2 * L <= len(S):
            return dict(b0=b, L=int(L), lag_peak=p)
    return None


def slide_dot(S: np.ndarray, b0: int, L: int, normalise: bool = True,
              center: bool = False):
    """« faire glisser ce motif sur l'AXE DES X — pas sur la diagonale — et
    faire le produit scalaire avec le carré sur lequel il tombe. »

    The pattern is the L x L block P = S[b0:b0+L, b0:b0+L] (the motif's own
    self-similarity square, sitting on the diagonal). Sliding along X keeps the
    ROWS fixed at the motif and moves the COLUMN window:

        f(d) = <P, S[b0:b0+L, d:d+L]>       for d = 0 .. n-L

    d = b0 is the self-match (f is maximal there by construction). A peak at
    d = c says "the material at bars c..c+L-1 relates to the motif the same way
    the motif relates to itself" -> another occurrence.

    `normalise` divides by ||P|| * ||block|| so f is a cosine in [-1, 1] and is
    comparable across offsets and songs.

    `center` subtracts each block's own mean before the dot product, turning
    the cosine into a CORRELATION.

    *** RETRACTED 2026-08-05 (docs/research_sessions/pattern_algo_2026-08-05.md):
    the paragraph below said centring was "strictly the better reading". It was
    judged on peak CONTRAST alone, never on whether the peaks landed on real
    occurrences. Against an independent reference (the per-bar chord string) the
    RAW dot product wins - F 0.67 vs 0.53, same recall, half the false peaks.
    The contrast numbers below are correct; the conclusion drawn from them is
    not. ***

    Measured 2026-08-05 on the five songs: this
    is strictly the better reading. The raw cosine sits in [0.92, 1.00] because
    every block of a pop SSM shares the same overall similarity level — that
    common level is what the centring removes. Peak contrast (peak height minus
    background, in background sigmas): 1.36 -> 1.67 (This Love), 1.89 -> 2.02
    (Don't Know Why), 0.97 -> 1.16 (Sunny), and -0.09 -> 1.16 on Billie Jean,
    where the un-centred version carries no usable signal at all.
    """
    n = len(S)
    P = S[b0:b0 + L, b0:b0 + L]
    if center:
        P = P - P.mean()
    pn = np.linalg.norm(P)
    ds = np.arange(0, n - L + 1)
    f = np.zeros(len(ds))
    for k, d in enumerate(ds):
        B = S[b0:b0 + L, d:d + L]
        if center:
            B = B - B.mean()
        v = float((P * B).sum())
        if normalise:
            bn = np.linalg.norm(B)
            v = v / max(pn * bn, 1e-12)
        f[k] = v
    return ds, f


def slide_peaks(ds, f, exclude, min_sep: int = 2, prom_frac: float = 0.25):
    """Local maxima of the sliding dot product, excluding a band around the
    motif itself. Same prominence rule as `row_peaks`, so the two answers to
    "which threshold" are directly comparable."""
    keep = np.abs(ds - exclude) >= min_sep
    lo, hi = float(np.median(f[keep])), float(f[keep].max())
    span = max(hi - lo, 1e-9)
    out = []
    for k in range(1, len(f) - 1):
        if not keep[k]:
            continue
        if not (f[k] >= f[k - 1] and f[k] >= f[k + 1]):
            continue
        lmin = f[k]
        for x in f[:k][::-1]:
            if x > f[k]:
                break
            lmin = min(lmin, x)
        rmin = f[k]
        for x in f[k + 1:]:
            if x > f[k]:
                break
            rmin = min(rmin, x)
        prom = f[k] - max(lmin, rmin)
        if prom >= prom_frac * span:
            out.append(dict(d=int(ds[k]), val=float(f[k]), prom=float(prom)))
    return out
