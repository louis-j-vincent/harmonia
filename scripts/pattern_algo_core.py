"""Louis's pattern-dictionary algorithm, implemented literally (2026-08-05).

His words, in order:

  1. « on détecte le premier motif de répétition en agrégeant / faisant glisser
     les premières lignes de la matrice »
  2. « on extrait le premier carré de ce premier motif »
  3. « produit scalaire le long de x -> la courbe »
  4. « des pics de la courbe on déduit quelles autres sections répètent ce
     motif »
  5. « c'est l'entrée n°1 d'un dictionnaire de motifs de répétition »
  6. « on relance la même analyse sur ce qui reste, pour trouver le motif n°2 »

  refinement — « les blocs trouvés ne sont PAS retirés de la matrice. Ils vont
  dans une boîte à part et restent candidats pour les autres motifs. À la fin
  chacun est attribué au motif qui a donné le plus grand produit scalaire, mais
  seulement si ce maximum est nettement au-dessus des autres ; si rien ne
  ressort, le bloc devient sa propre section. »

Nothing under `harmonia_min/` is touched. The bar-level SSM is the live one
(`scripts/pattern_dict_core.song_data`, which spies on `pipeline.analyze`).

Three things are settled here BY MEASUREMENT, not by assertion:
  A. which sliding statistic (raw / cosine / mean-centred / binary diagonal),
  B. what "clearly above the others" means and how sensitive the result is,
  C. which peak selector finds the true repeats without inventing extras.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

MIN_LAG = 2          # |lag| < 2 is trivially self-similar
LMAX = 32            # longest motif we will look for, in bars


# ══ 0. background distribution helpers ══════════════════════════════════════
def offdiag(S: np.ndarray, min_lag: int = MIN_LAG) -> np.ndarray:
    n = len(S)
    i, j = np.indices((n, n))
    return S[np.abs(i - j) >= min_lag]


def robust_z(f: np.ndarray) -> np.ndarray:
    """z against the curve's OWN median/MAD — robust to the handful of peaks.

    Used for cross-pattern comparison: pattern A may sit generally higher than
    pattern B (a motif made of homogeneous material correlates with everything),
    and comparing raw levels would hand every block to A. In z units the
    question becomes "how unusual is this offset FOR THIS PATTERN", which is
    the comparison Louis's assignment step actually needs.
    """
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med)))
    sd = 1.4826 * mad if mad > 1e-9 else (float(np.std(f)) or 1e-9)
    return (f - med) / sd


# ══ 1. the first repetition pattern, from the first rows ════════════════════
def lag_profile(S: np.ndarray, n_rows: int | None = None,
                min_lag: int = MIN_LAG, lmax: int = LMAX) -> np.ndarray:
    """« en agrégeant les premières lignes » — mean similarity at each lag,
    averaged over the first `n_rows` rows of the matrix.

    profile[L] = mean_b S[b, b+L] over the first n_rows bars. The first clear
    peak of this profile is the song's repetition PERIOD as announced by its
    opening. n_rows=None uses every row (the whole-song reading); the
    first-rows reading is what Louis described and is what the report shows.
    """
    n = len(S)
    n_rows = n if n_rows is None else min(n_rows, n)
    prof = np.zeros(lmax + 1)
    for L in range(min_lag, lmax + 1):
        v = [S[b, b + L] for b in range(n_rows) if b + L < n]
        prof[L] = float(np.mean(v)) if v else -np.inf
    return prof


FIRST_ROWS = 16          # « les premières lignes » — the song's OPENING.
                         # Measured 2026-08-05 on the five grid-valid songs:
                         # aggregating the first 16 rows gives 4 / 4 / 16 / 2 /
                         # 8 bars (This Love, Don't Know Why, Sunny, Billie
                         # Jean, Every Breath), which is the musically right
                         # cycle on four of the five. Aggregating EVERY row
                         # gives 20 / 4 / 4 / 4 / 4 — it collapses Sunny's
                         # 16-bar form and Every Breath's 8-bar verse to the
                         # shortest sub-loop, because late-song material
                         # (solos, outro) drags the profile. Louis's "first
                         # rows" is not an approximation of the whole-matrix
                         # reading; it is a better statistic than it.


def song_period(S: np.ndarray, n_rows: int = FIRST_ROWS,
                min_lag: int = MIN_LAG, lmax: int = LMAX) -> tuple[int, np.ndarray]:
    """Step 1, literally: aggregate the first rows, take the strongest lag."""
    n = len(S)
    prof = lag_profile(S, min(n_rows, n), min_lag, min(lmax, max(min_lag, n - 2)))
    L = int(np.argmax(prof[:min(lmax, max(min_lag, n - 2)) + 1]))
    return max(L, min_lag), prof


def first_pattern(S: np.ndarray, q: float = 0.95, min_lag: int = MIN_LAG,
                  lmax: int = LMAX, n_rows: int = FIRST_ROWS,
                  forbidden: np.ndarray | None = None,
                  L: int | None = None) -> dict | None:
    """Steps 1+2: the period L from the aggregated first rows, then the FIRST
    square of length L whose whole block-diagonal actually repeats.

    "Block-diagonal", not one cell: `mean_i S[b+i, b+L+i] >= q-quantile of the
    song's own off-diagonal values`. One high cell is a coincidence; L of them
    in a row is a repeat. The quantile is `sections.TILE_QUANTILE`'s logic — a
    rank, not a constant, because the constant means 4x different things across
    songs (measured, docs/known_issues.md).

    `L` fixes the motif length (the whole dictionary uses ONE length — see
    `build_dictionary`); `forbidden` (bool per bar) blocks bars already covered
    when re-running for pattern #2, #3 ... — step 6.
    """
    n = len(S)
    thr = float(np.quantile(offdiag(S, min_lag), q))
    L0, prof = song_period(S, n_rows, min_lag, lmax)
    L = int(L or L0)
    best = None
    for b in range(n):
        if forbidden is not None and forbidden[b]:
            continue
        if b + 2 * L > n:
            continue
        sc = float(np.mean([S[b + i, b + L + i] for i in range(L)]))
        if sc >= thr:
            return dict(b0=b, L=L, score=sc, thr=thr, prof=prof)
        if best is None or sc > best["score"]:
            best = dict(b0=b, L=L, score=sc, thr=thr, prof=prof)
    return best


# ══ 2. the four sliding statistics ══════════════════════════════════════════
def slide(S: np.ndarray, b0: int, L: int, stat: str = "centered"):
    """« on fait glisser le carré le long de l'axe des X — pas de la diagonale
    — et on fait le produit scalaire avec le carré sur lequel on tombe. »

    Rows FROZEN at the motif, columns slid:
        f(d) = <P, S[b0:b0+L, d:d+L]>,  P = S[b0:b0+L, b0:b0+L]

    stat:
      raw       plain dot product. Louis's argument: sliding along x with the
                rows fixed means the block's overall LEVEL is itself the
                harmonic-similarity signal, and normalising divides it out.
      cosine    divided by ||P||.||B||.
      centered  each block minus its own mean, then cosine -> a Pearson
                correlation between the two blocks.
      diag      Louis's other idea: slide a BINARY DIAGONAL instead of the
                extracted square. mean_i S[b0+i, d+i]. See `diag_orientation`.
      antidiag  the same with the anti-diagonal, kept only to show it is wrong.
    """
    n = len(S)
    P = S[b0:b0 + L, b0:b0 + L]
    Pc = P - P.mean()
    pn, pcn = np.linalg.norm(P), np.linalg.norm(Pc)
    ds = np.arange(0, n - L + 1)
    f = np.zeros(len(ds))
    for k, d in enumerate(ds):
        B = S[b0:b0 + L, d:d + L]
        if stat == "raw":
            f[k] = float((P * B).sum())
        elif stat == "cosine":
            f[k] = float((P * B).sum() / max(pn * np.linalg.norm(B), 1e-12))
        elif stat == "centered":
            Bc = B - B.mean()
            f[k] = float((Pc * Bc).sum() / max(pcn * np.linalg.norm(Bc), 1e-12))
        elif stat == "diag":
            f[k] = float(np.mean(np.diag(B)))
        elif stat == "antidiag":
            f[k] = float(np.mean(np.diag(np.fliplr(B))))
        else:
            raise ValueError(stat)
    return ds, f


def diag_orientation() -> str:
    """« attention à mettre la diagonale dans le bon sens » — settled by what
    the block MEANS, then confirmed by measurement in the report.

    B[i,k] = sim(bar b0+i, bar d+k). If bars d..d+L-1 are a repeat of the motif
    then bar d+k sounds like bar b0+k, so B is large where i == k: the MAIN
    diagonal (top-left to bottom-right), i.e. the identity. The anti-diagonal
    would say bar d+k sounds like bar b0+L-1-k — the motif played BACKWARDS,
    which does not happen in this music.

    Note what this makes the statistic: sliding a binary identity is exactly
    `mean_i S[b0+i, d+i]`, the mean of the SSM's sub-diagonal at lag d-b0 over
    L bars — i.e. the LAG-RUN reading already measured at 36.8% placement
    (docs/known_issues.md, tiling_v2). Louis's two ideas meet here.
    """
    return "main diagonal (identity), not anti-diagonal"


# ══ 3. peak selectors (Task C) ══════════════════════════════════════════════
def _local_maxima(f: np.ndarray, min_sep: int = 1) -> list[int]:
    n = len(f)
    out = []
    for k in range(n):
        lo, hi = max(0, k - min_sep), min(n, k + min_sep + 1)
        if f[k] >= f[lo:hi].max() and (k == 0 or f[k] > f[k - 1] or
                                       f[k] >= f[max(0, k - min_sep):k].max()):
            if not out or k - out[-1] > min_sep or f[k] > f[out[-1]]:
                out.append(k)
    # de-duplicate plateaus: keep the first index of each run of equal maxima
    ded = []
    for k in out:
        if ded and k - ded[-1] <= min_sep:
            if f[k] > f[ded[-1]]:
                ded[-1] = k
            continue
        ded.append(k)
    return ded


def _prominence(f: np.ndarray, k: int) -> float:
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
    return float(f[k] - max(lmin, rmin))


# Louis's rule, validated by him on the 5-song plots (2026-08-05, after the
# 70/75/80/85/90/95 sweep in /reports/peak_rule_sweep.html): a peak must clear
# BOTH a local-baseline margin AND this fraction of the INITIAL peak — the
# curve's value at the motif matched against itself, which is the ceiling by
# construction. The two halves do different jobs: the margin says "this is a
# peak, not a plateau", the fraction says "the match is strong enough to be the
# same music". Chosen by eye on the plots, not by a corpus score — the corpus
# metric cannot separate these thresholds and he trusts the per-song reading.
INITIAL_PEAK_FRAC = 0.90


def pick_peaks(ds: np.ndarray, f: np.ndarray, method: str = "prominence",
               param: float = 0.25, min_sep: int = 2,
               exclude: int | None = None, base_win: int = 8,
               initial_frac: float | None = INITIAL_PEAK_FRAC,
               initial_at: int | None = None) -> list[dict]:
    """Local maxima of the sliding curve, kept by one of four rules.

    `initial_frac` applies Louis's second condition on top of whichever rule:
    the peak must also reach that fraction of the curve at `initial_at` (the
    motif's own position; defaults to the curve maximum, which is the same
    thing whenever the motif matches itself best).

      prominence  topographic prominence >= param * (max - median) of f.
      quantile    value >= the param-quantile of f's own distribution.
      margin      value - median(f in a +-base_win window) >= param * MAD(f),
                  i.e. a LOCAL baseline instead of a global one.
      period      no selection at all: every multiple of L from the motif.
                  (Passed in by the caller as `ds` already filtered — the
                  trivial periodic baseline the others must beat.)
    """
    cand = _local_maxima(f, min_sep)
    if exclude is not None:
        cand = [k for k in cand if abs(int(ds[k]) - exclude) >= min_sep]
    if not cand:
        return []
    med, mx = float(np.median(f)), float(f.max())
    span = max(mx - med, 1e-9)
    mad = 1.4826 * float(np.median(np.abs(f - med))) or 1e-9
    out = []
    for k in cand:
        if method == "prominence":
            ok = _prominence(f, k) >= param * span
        elif method == "quantile":
            ok = f[k] >= float(np.quantile(f, param))
        elif method == "margin":
            a, b = max(0, k - base_win), min(len(f), k + base_win + 1)
            ok = (f[k] - float(np.median(f[a:b]))) >= param * mad
        elif method == "all":
            ok = True
        else:
            raise ValueError(method)
        if ok and initial_frac is not None:
            ref = float(f[initial_at]) if initial_at is not None else mx
            ok = f[k] >= initial_frac * ref
        if ok:
            out.append(dict(d=int(ds[k]), val=float(f[k]), k=int(k),
                            prom=_prominence(f, k)))
    return out


# ══ 4. the dictionary (Task B) ══════════════════════════════════════════════
REDUNDANT = 0.80         # two motifs are the SAME dictionary entry when
                         # their sliding curves are this correlated once
                         # aligned on their motif starts.


def _same_pattern(f_new, b_new, f_old, b_old, thr: float = REDUNDANT) -> bool:
    """Is this candidate motif just an existing entry moved by a bar or two?

    Scanning bar by bar produces phase-shifted copies of the same loop: on This
    Love the motifs at bars 0, 5, 6 and 7 are the same 4-bar cycle read from
    four different starting points, and their occurrence sets are the same set
    shifted by one — so a set-overlap test does NOT catch them (Jaccard stays
    low). Their CURVES, however, are the same curve translated by the same
    offset. Aligning on `b_new - b_old` and correlating catches all four.
    """
    d = b_new - b_old
    n = min(len(f_new), len(f_old))
    lo, hi = max(0, d), min(n, n + d)
    if hi - lo < 8:
        return False
    a = np.asarray(f_new[lo:hi], float)
    b = np.asarray(f_old[lo - d:hi - d], float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return False
    return float(np.corrcoef(a, b)[0, 1]) >= thr


def build_dictionary(S: np.ndarray, stat: str = "centered", q: float = 0.95,
                     peak_method: str = "prominence", peak_param: float = 0.25,
                     max_patterns: int = 6, tau: float = 1.0,
                     assign_stat: str | None = None,
                     removal: str = "none") -> dict:
    """Steps 1-6 plus the refinement.

    Loop: find the first pattern among bars NOT YET USED AS A MOTIF START, cut
    its square, slide it, take the peaks -> occurrences. Occurrences are NOT
    removed from the matrix (Louis's refinement) — they only mark bars as
    "already explained by something", which stops the next motif from starting
    inside them, but they stay candidates for every later pattern.

    Final assignment. Every candidate block start `d` seen by any pattern gets
    one score per pattern, all converted to robust z of that pattern's own
    curve. Assigned to argmax if `z_best - z_second >= tau`; otherwise it is
    AMBIGUOUS and becomes its own section.
    """
    n = len(S)
    assign_stat = assign_stat or stat
    used = np.zeros(n, bool)      # bars a pattern already covers (motif start
                                  # blocker only — the matrix is untouched)
    # ONE length for the whole dictionary — the song's period, read off the
    # aggregated first rows. Re-deriving a length per entry was tried and is
    # degenerate: once the 4-bar cycle is taken, the leftovers' strongest lag
    # is 2, and Billie Jean fills the dictionary with six 2-bar motifs that
    # all match everywhere. A dictionary of different-length blocks also makes
    # the final arbitration compare things that are not the same object.
    Lfix, prof0 = song_period(S)
    patterns, seek = [], 0
    for _ in range(max_patterns):
        if used.all():
            break
        if removal == "none":
            # THE LITERAL READING of « les blocs trouvés ne sont PAS retirés de
            # la matrice ». Only the motif's own square is consumed; every bar
            # stays a legal start for the next motif, and every block stays a
            # candidate for every pattern. A new motif is rejected only if its
            # occurrence set is essentially the one an existing entry already
            # has (Jaccard >= REDUNDANT), which is what stops the loop.
            thr = float(np.quantile(offdiag(S, MIN_LAG), q))
            mot = None
            for b in range(seek, n - 2 * Lfix + 1):
                s = float(np.mean([S[b + i, b + Lfix + i] for i in range(Lfix)]))
                if s >= thr:
                    mot = dict(b0=b, L=Lfix, score=s, thr=thr, prof=prof0)
                    seek = b + 1
                    break
            if mot is None:
                break
        elif not patterns:
            mot = first_pattern(S, q=q, forbidden=used, L=Lfix)
            if mot is None or mot["score"] < mot["thr"]:
                break
        else:
            # « on relance la même analyse sur ce qui reste ». Two readings
            # were tried and the obvious one is measured WRONG:
            #   * keep the song's 95th percentile -> no second entry ever
            #     qualifies (1 entry on 4 of the 5 songs), because the leftover
            #     material is by definition the material that repeats LESS;
            #   * recompute the 95th percentile on the residual sub-matrix ->
            #     WORSE, not better: on This Love the residual bars are the
            #     outro, which is more self-similar than the song average, so
            #     the residual threshold goes UP (0.945 -> 0.967).
            # What works is the same "rank, not constant" idea applied inside
            # the leftovers: motif #2 is the residual bar whose block-diagonal
            # stands out AMONG THE RESIDUAL BARS' OWN scores (median + 1
            # robust sigma), and it must recur at least twice.
            sc = {}
            for b in range(n):
                if used[b] or b + 2 * Lfix > n:
                    continue
                sc[b] = float(np.mean([S[b + i, b + Lfix + i]
                                       for i in range(Lfix)]))
            if len(sc) < 3:
                break
            v = np.array(list(sc.values()))
            med = float(np.median(v))
            sd = 1.4826 * float(np.median(np.abs(v - med))) or float(np.std(v))
            thr_r = med + 1.0 * (sd or 1e-9)
            cand = [b for b, s in sc.items() if s >= thr_r]
            if not cand:
                break
            b0 = min(cand)
            mot = dict(b0=b0, L=Lfix, score=sc[b0], thr=thr_r, prof=prof0)
        b0, L = mot["b0"], mot["L"]
        ds, f = slide(S, b0, L, stat)
        pk = pick_peaks(ds, f, peak_method, peak_param, min_sep=max(2, L // 2),
                        exclude=b0)
        # the motif's own position always belongs to it
        occ = sorted({b0} | {p["d"] for p in pk})
        _, fa = (ds, f) if assign_stat == stat else slide(S, b0, L, assign_stat)
        if any(_same_pattern(f, b0, p["f"], p["b0"]) for p in patterns):
            if removal == "none":
                continue          # a phase-shifted copy — keep scanning
            break
        patterns.append(dict(b0=b0, L=L, score=mot["score"], thr=mot["thr"],
                             prof=mot["prof"], ds=ds, f=f, fz=robust_z(fa),
                             fa=fa, peaks=pk, occ=occ))
        for d in occ:
            used[d:min(n, d + L)] = True
        if len(patterns) >= max_patterns:
            break

    if not patterns:
        return dict(patterns=[], blocks=[], tau=tau, stat=stat)

    # ── the separate box: every candidate block, scored by EVERY pattern ─────
    starts = sorted({d for p in patterns for d in p["occ"]})
    blocks = []
    for d in starts:
        z, v = [], []
        for p in patterns:
            k = d if d < len(p["fz"]) else len(p["fz"]) - 1
            z.append(float(p["fz"][k]))
            v.append(float(p["fa"][k]))
        z = np.array(z)
        order = np.argsort(-z)
        best, second = int(order[0]), (int(order[1]) if len(order) > 1 else None)
        margin = float(z[best] - z[second]) if second is not None else np.inf
        owner = int(best) if margin >= tau else None
        blocks.append(dict(d=int(d), z=z.tolist(), v=v, best=best,
                           second=second, margin=margin, owner=owner,
                           L=patterns[best]["L"]))
    return dict(patterns=patterns, blocks=blocks, tau=tau, stat=stat)


def dictionary_sections(dic: dict, n: int) -> list[dict]:
    """The dictionary rendered as a covering segmentation, so it can be put
    side by side with what `sections.py` produces.

    Each assigned block spans [d, d+L) and carries the letter of its pattern;
    each ambiguous block spans [d, d+L) and gets its own fresh letter; the gaps
    between blocks become unlabelled sections of their own.
    """
    if not dic["patterns"]:
        return [dict(b0=0, b1=n - 1, label="A")]
    segs = []
    amb = 0
    for bl in dic["blocks"]:
        L = dic["patterns"][bl["best"]]["L"] if bl["owner"] is None else \
            dic["patterns"][bl["owner"]]["L"]
        if bl["owner"] is None:
            amb += 1
            lab = f"?{amb}"
        else:
            lab = chr(ord("A") + bl["owner"])
        segs.append(dict(b0=bl["d"], b1=min(n - 1, bl["d"] + L - 1), label=lab))
    segs.sort(key=lambda s: s["b0"])
    # resolve overlaps: an earlier block never eats a later one's start
    out = []
    for s in segs:
        if out and s["b0"] <= out[-1]["b1"]:
            out[-1]["b1"] = s["b0"] - 1
            if out[-1]["b1"] < out[-1]["b0"]:
                out.pop()
        out.append(dict(s))
    # fill gaps
    full, cur = [], 0
    for s in out:
        if s["b0"] > cur:
            full.append(dict(b0=cur, b1=s["b0"] - 1, label="·"))
        full.append(s)
        cur = s["b1"] + 1
    if cur < n:
        full.append(dict(b0=cur, b1=n - 1, label="·"))
    return full


def boundaries(segs) -> list[int]:
    return sorted({s["b0"] for s in segs if s["b0"] > 0})


# ══ 5. contrast: how well a curve separates its peaks from its background ═══
def contrast(f: np.ndarray, peaks: list[dict]) -> float:
    """(mean peak height - median background) / sigma(background), background =
    everything that is not within 1 of a peak. Scale-free, so raw / cosine /
    centred / diagonal are directly comparable despite living on different
    scales — which is the whole point of Task A."""
    if not peaks:
        return 0.0
    mask = np.ones(len(f), bool)
    for p in peaks:
        mask[max(0, p["k"] - 1):p["k"] + 2] = False
    bg = f[mask]
    if len(bg) < 4:
        return 0.0
    sd = float(np.std(bg)) or 1e-9
    h = np.mean([p["val"] for p in peaks])
    return float((h - float(np.median(bg))) / sd)
