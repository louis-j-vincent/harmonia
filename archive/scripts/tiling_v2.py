"""Louis's four directives on tiling runs (2026-08-05), implemented + measured.

  (1) « pas l'un ou l'autre mais les deux ensemble » — the novelty PEAKS must
      CORRECT the tiling runs, not replace them (today `detect_sections` picks
      one mode and discards novelty candidates wherever a run covers).
  (2) « repérer la boucle de répétition minimale au sein de chaque carré de
      tuilage, et un carré est forcément un multiple de cette boucle. »
  (3) « utiliser la matrice SSM non floutée. »
  (4) « plus on s'éloigne dans le temps entre les répétitions, moins elles se
      ressemblent » — so a fixed TILE_MIN = 0.80 is wrong for far-apart repeats.

Harness: the 285-track Billboard bar-GT set (`scripts/bibar_prep.py`), same
metric as `scripts/bibar_final.py`, TIES BROKEN AT RANDOM, trivial "never move"
baseline printed next to every number (see the ★★ RETRACTED entry in
docs/known_issues.md — a distance tie-break gives a flat cue a free 100%).

Nothing under harmonia_min/ is touched.

    python scripts/tiling_v2.py [n_tracks]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from bibar_binary_ssm import (cont_ssm, novelty_score, placement,  # noqa: E402
                              run_edge_score)
from bibar_sweep import load  # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")

TILE_MIN = 0.80          # harmonia_min/sections.py's constant, unchanged
PERIODS = (2, 4, 8, 16)


# ── substrate ───────────────────────────────────────────────────────────────
def bar_vecs(Hb: np.ndarray) -> np.ndarray:
    """(n_bars, 48) unit bar vectors — `folding._bar_vecs(halfbar_features)`
    rebuilt on the Billboard half-bar chroma so the harness and the shipped
    code use the SAME feature."""
    F = Hb.copy()
    for h in (slice(0, 12), slice(12, 24)):
        nrm = np.linalg.norm(F[:, h], axis=1, keepdims=True)
        F[:, h] = F[:, h] / np.maximum(nrm, 1e-9)
    F /= np.sqrt(2.0)
    nb = len(F) // 2
    V = np.array([F[2 * b:2 * b + 2].reshape(-1) for b in range(nb)])
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def nov_bar(Hb: np.ndarray, sigma: float = 1.5, kw: int = 16) -> np.ndarray:
    """Checkerboard novelty per BAR. sigma <= 0 == NO gaussian blur (Louis 3)."""
    from harmonia_min.sections import _blur, _novelty
    F = Hb.copy()
    for h in (slice(0, 12), slice(12, 24)):
        nrm = np.linalg.norm(F[:, h], axis=1, keepdims=True)
        F[:, h] = F[:, h] / np.maximum(nrm, 1e-9)
    F /= np.sqrt(2.0)
    S = F @ F.T
    if sigma > 0:
        S = _blur(S, sigma)
    return _novelty(S, kw)[::2]


def chg(Vb: np.ndarray) -> np.ndarray:
    """Per-bar harmonic CHANGE: 1 - cos(bar b-1, bar b). This is what carries
    the loop PHASE — a loop's first bar is where the big change happens, and
    it happens at the same phase every time round (CLAUDE.md rule #4: period
    detection has never given phase; this is the missing half)."""
    d = np.zeros(len(Vb))
    d[1:] = 1.0 - np.einsum("ij,ij->i", Vb[:-1], Vb[1:])
    return d


# ── (4) DISTANCE-AWARE THRESHOLD ────────────────────────────────────────────
def tau_lag(lag: int, base: float = TILE_MIN, beta: float = 0.0) -> float:
    """Similarity floor that RELAXES with the lag between the two repeats.
    beta = 0 recovers the shipped fixed TILE_MIN exactly."""
    return base - beta * np.log2(max(lag, 1) / 2.0)


# ── (2) MINIMAL LOOP INSIDE A SQUARE ────────────────────────────────────────
def minimal_loop(Vb: np.ndarray, b0: int, b1: int, beta: float = 0.0,
                 base: float = TILE_MIN) -> tuple[int, float]:
    """Smallest L such that the passage [b0,b1] tiles at L. Louis: the loop is
    detected INSIDE the square and may be FINER than the P that built the run.
    Returns (L, score); L = 0 when nothing tiles."""
    L_ = b1 - b0 + 1
    for L in PERIODS:
        if L_ < 2 * L:
            break
        s = float(np.mean([Vb[b] @ Vb[b + L] for b in range(b0, b1 + 1 - L)]))
        if s >= tau_lag(L, base, beta):
            return L, s
    return 0, 0.0


def loop_phase(chg_: np.ndarray, nov: np.ndarray, b0: int, b1: int,
               L: int) -> int:
    """WHERE the loop starts, modulo L, read off the change/novelty pattern.
    Louis (1): the peaks decide where the snap lands. Score each residue class
    by the evidence sitting on it, take the argmax; absolute bar index, so the
    phase is comparable across the whole song."""
    if L <= 1:
        return 0
    z = _z(chg_) + _z(nov)
    best, bs = 0, -1e18
    for ph in range(L):
        idx = [b for b in range(b0, b1 + 1) if b % L == ph]
        if not idx:
            continue
        v = float(np.mean(z[idx]))
        if v > bs:
            best, bs = ph, v
    return best


# ── (1)+(2)+(4) THE PROTOTYPE ───────────────────────────────────────────────
def tiling_runs_v2(Vb: np.ndarray, n_bars: int, nov: np.ndarray,
                   beta: float = 0.0, base: float = TILE_MIN,
                   snap: bool = True) -> list[dict]:
    """v2 of `harmonia_min.sections.tiling_runs`.

    v1: one pass over P in (2,4,8) with a FIXED TILE_MIN, runs kept when their
        length >= 2P, nothing forces the length to be a multiple of anything.
    v2: (4) the floor relaxes with P; (2) each run's MINIMAL loop L is detected
        inside the square and the square is snapped to a multiple of L;
        (1) the novelty peaks choose WHERE the snap lands (the phase).

    -> [{b0, b1, period, L, phase, b0_v1, b1_v1}]
    """
    period_of = [0] * n_bars
    for P in PERIODS:
        t = tau_lag(P, base, beta)
        for b in range(n_bars):
            if period_of[b]:
                continue
            fwd = b + P < n_bars and float(Vb[b] @ Vb[b + P]) >= t
            bwd = b - P >= 0 and float(Vb[b] @ Vb[b - P]) >= t
            if fwd or bwd:
                period_of[b] = P
    runs, b = [], 0
    while b < n_bars:
        P = period_of[b]
        e = b
        while e + 1 < n_bars and period_of[e + 1] == P:
            e += 1
        if P and e - b + 1 >= 2 * P:
            runs.append({"b0": b, "b1": e, "period": P})
        b = e + 1

    c = chg(Vb)
    out = []
    for r in runs:
        b0, b1, P = r["b0"], r["b1"], r["period"]
        L, sc = minimal_loop(Vb, b0, b1, beta, base)
        r.update(b0_v1=b0, b1_v1=b1, L=L, loop_score=sc, phase=None)
        if snap and L >= 2:
            ph = loop_phase(c, nov, b0, b1, L)
            r["phase"] = ph
            # (2) the square STARTS on the loop phase and its LENGTH is a
            # multiple of L. Start moves to the nearest in-run bar of the right
            # residue; the end is trimmed down to the last complete loop.
            nb0 = b0 + ((ph - b0) % L)
            if nb0 > b1:
                nb0 = b0
            n_loops = (b1 - nb0 + 1) // L
            if n_loops >= 2:
                r["b0"], r["b1"] = nb0, nb0 + n_loops * L - 1
        out.append(r)
    return out


def phase_map(Vb: np.ndarray, n_bars: int, nov: np.ndarray, beta: float,
              base: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar (L, phase) from the v2 runs; bars outside every run inherit a
    SONG-GLOBAL loop, so the cue is defined everywhere."""
    runs = tiling_runs_v2(Vb, n_bars, nov, beta, base)
    Lm = np.zeros(n_bars, int)
    Pm = np.zeros(n_bars, int)
    for r in runs:
        if r["L"] >= 2 and r["phase"] is not None:
            Lm[r["b0"]:r["b1"] + 1] = r["L"]
            Pm[r["b0"]:r["b1"] + 1] = r["phase"]
    if (Lm == 0).any():
        gL, _ = minimal_loop(Vb, 0, n_bars - 1, beta, base)
        if gL >= 2:
            gp = loop_phase(chg(Vb), nov, 0, n_bars - 1, gL)
            Lm[Lm == 0] = gL
            Pm[Lm == gL] = np.where(Pm[Lm == gL] == 0, gp, Pm[Lm == gL])
    return Lm, Pm


def loop_congruence_score(Vb, n_bars, nov, beta=0.0, base=TILE_MIN):
    """+1 where a bar sits on its local loop's phase, 0 elsewhere. DELIBERATELY
    a step function: it is a CONSTRAINT, not a ranking, so it must be summed
    with a continuous cue that breaks its ties (Louis: runs AND peaks)."""
    Lm, Pm = phase_map(Vb, n_bars, nov, beta, base)
    return np.array([1.0 if Lm[b] >= 2 and b % Lm[b] == Pm[b] else 0.0
                     for b in range(n_bars)])


def _z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / max(v.std(), 1e-9)


def _pad(v, n):
    v = np.asarray(v, float)
    return v[:n] if len(v) >= n else np.concatenate([v, np.zeros(n - len(v))])


# ═══ MEASUREMENT ════════════════════════════════════════════════════════════
def premise_decay(songs):
    """(4) PREMISE: does similarity between REAL repeats decay with distance?
    Same-position bars of two same-letter GT sections, binned by bar distance."""
    bins = {}
    for s in songs:
        Vb = bar_vecs(s["Hb"])
        st, le = list(s["starts"]), list(s["letters"])
        ends = st[1:] + [s["nb"]]
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                if le[i] != le[j]:
                    continue
                n = min(ends[i] - st[i], ends[j] - st[j], 16)
                lag = st[j] - st[i]
                k = int(np.clip(np.log2(max(lag, 1)), 1, 7))
                for o in range(n):
                    a, b = st[i] + o, st[j] + o
                    if a < len(Vb) and b < len(Vb):
                        bins.setdefault(k, []).append(float(Vb[a] @ Vb[b]))
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(bins.items())}


def premise_multiple(songs, beta=0.0, base=TILE_MIN):
    """(2) PREMISE, two ways:
    (a) GT sections: is the ANNOTATED section length a multiple of the minimal
        loop we detect inside it?   (b) our own v1 runs: same question."""
    gt_ok = gt_n = rn_ok = rn_n = 0
    for s in songs:
        Vb = bar_vecs(s["Hb"])
        nov = nov_bar(s["Hb"], 0.0)
        st = list(s["starts"])
        ends = st[1:] + [s["nb"]]
        for a, b in zip(st, ends):
            if b - a < 4:
                continue
            L, _ = minimal_loop(Vb, a, b - 1, beta, base)
            if L >= 2:
                gt_n += 1
                gt_ok += (b - a) % L == 0
        for r in tiling_runs_v2(Vb, s["nb"], nov, beta, base, snap=False):
            if r["L"] >= 2:
                rn_n += 1
                rn_ok += (r["b1_v1"] - r["b0_v1"] + 1) % r["L"] == 0
    return dict(gt=(gt_ok, gt_n), runs=(rn_ok, rn_n))


def letter_auc(songs, sigma):
    """(3b) Louis's specific claim: the blur destroys the ability to tell two
    SIMILAR-BUT-DIFFERENT sections apart. AUC of the cross/self block ratio
    (`sections.py`'s own letter statistic) on GT same-letter vs different."""
    from harmonia_min.sections import _blur
    accs = []
    for s in songs:
        st, le = list(s["starts"]), list(s["letters"])
        if len(st) < 3:
            continue
        F = s["Hb"].copy()
        for h in (slice(0, 12), slice(12, 24)):
            F[:, h] /= np.maximum(np.linalg.norm(F[:, h], axis=1,
                                                 keepdims=True), 1e-9)
        F /= np.sqrt(2.0)
        S = F @ F.T
        if sigma > 0:
            S = _blur(S, sigma)
        ends = st[1:] + [s["nb"]]
        k = len(st)
        M = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                ri = slice(2 * st[i], min(len(S), 2 * ends[i]))
                rj = slice(2 * st[j], min(len(S), 2 * ends[j]))
                blk = S[ri, rj]
                M[i, j] = float(blk.mean()) if blk.size else 0.0
        pos, neg = [], []
        for i in range(k):
            for j in range(i + 1, k):
                r = M[i, j] / np.sqrt(max(M[i, i] * M[j, j], 1e-12))
                (pos if le[i] == le[j] else neg).append(r)
        if pos and neg:
            accs.append(np.mean([(p > q) + 0.5 * (p == q)
                                 for p in pos for q in neg]))
    return float(np.mean(accs)), len(accs)


def main(n=None):
    songs = load(n)
    ns = sum(len(s["starts"]) for s in songs)
    print(f"{len(songs)} Billboard tracks · {ns} annotated section starts")
    print("ties broken AT RANDOM everywhere (docs/known_issues.md ★★)\n")
    res = {}

    # ── premises first (CLAUDE.md rule #2) ──────────────────────────────────
    print("PREMISE (4) — does similarity decay with the distance between "
          "repeats?\n  (same-position bars of two same-letter GT sections)")
    dec = premise_decay(songs)
    for k, (m, c) in dec.items():
        print(f"    lag {2**k:>3d}-{2**(k+1)-1:<3d} bars   mean cos {m:.3f}"
              f"   (n={c})")
    res["decay"] = {str(k): v for k, v in dec.items()}

    print("\nPREMISE (2) — is a section's length a MULTIPLE of the minimal "
          "loop inside it?")
    for tag, beta in (("fixed  TILE_MIN=0.80", 0.0), ("lag-relaxed beta=0.03",
                                                      0.03)):
        pm = premise_multiple(songs, beta)
        g, gn = pm["gt"]
        r, rn = pm["runs"]
        print(f"    {tag:<24s} GT sections {g}/{gn} = {100*g/max(1,gn):4.1f}%"
              f"   |  our v1 runs {r}/{rn} = {100*r/max(1,rn):4.1f}%")
        res[f"multiple_{beta}"] = dict(gt=[g, gn], runs=[r, rn])

    # ── (3) the blur ────────────────────────────────────────────────────────
    print("\n(3) THE GAUSSIAN BLUR — measured, not obeyed")
    res["blur"] = {}
    for sig in (0.0, 0.5, 1.0, 1.5, 2.5):
        a, na = letter_auc(songs, sig)
        res["blur"][str(sig)] = dict(auc=a, n=na)
        print(f"    sigma {sig:<4} letter AUC (same vs diff) {a:.3f} (n={na})")

    # ── placement table ─────────────────────────────────────────────────────
    print("\nPLACEMENT — exact bar, +-4 bar window")
    rows = []

    def cue(name, fn):
        er = []
        for s in songs:
            er += placement(fn(s), s["starts"], s["nb"])
        er = np.array(er)
        r = dict(name=name, exact=float((er == 0).mean() * 100),
                 pm1=float((np.abs(er) <= 1).mean() * 100),
                 med=float(np.median(np.abs(er))), n=int(len(er)))
        rows.append(r)
        print(f"    {name:<52s} exact {r['exact']:5.1f}%   "
              f"|e|<=1 {r['pm1']:5.1f}%   med {r['med']:.1f}")
        return r

    def _fuse(s, sigma=1.5, w=0.0, beta=0.0, base=TILE_MIN):
        nb = s["nb"]
        nv = nov_bar(s["Hb"], sigma)
        e = run_edge_score(cont_ssm(s["Cb"], "concat", 0.93), "edgesum")
        v = _z(_pad(e, nb)) + _z(_pad(nv, nb))
        if w:
            Vb = bar_vecs(s["Hb"])
            v = v + w * loop_congruence_score(Vb, nb, _pad(nv, nb), beta, base)
        return v

    cue("trivial: never move (flat score)", lambda s: np.zeros(s["nb"]))
    cue("chroma novelty, BLURRED sigma=1.5 (SHIPPED)",
        lambda s: nov_bar(s["Hb"], 1.5))
    cue("chroma novelty, UN-BLURRED (Louis 3)", lambda s: nov_bar(s["Hb"], 0.0))
    cue("lag-runs, continuous (yesterday)",
        lambda s: run_edge_score(cont_ssm(s["Cb"], "concat", 0.93), "edgesum"))
    cue("FUSION lag-runs + blurred novelty (yesterday, 40.2%)",
        lambda s: _fuse(s, 1.5, 0.0))
    cue("FUSION lag-runs + UN-blurred novelty",
        lambda s: _fuse(s, 0.0, 0.0))
    print("    -- loop-congruence alone (a CONSTRAINT: many ties by design) --")
    cue("loop congruence only, beta=0 (step, random ties)",
        lambda s: loop_congruence_score(bar_vecs(s["Hb"]), s["nb"],
                                        nov_bar(s["Hb"], 0.0)))
    print("    -- v2 = runs AND peaks AND the minimal-loop constraint --")
    for w in (0.5, 1.0, 1.5, 2.0, 3.0):
        cue(f"v2  fusion + {w}*loop-congruence (blur on)",
            lambda s, w=w: _fuse(s, 1.5, w))
    for w in (0.25, 0.5, 0.75, 1.0, 2.0):
        cue(f"v2  fusion + {w}*loop-congruence (NO blur)",
            lambda s, w=w: _fuse(s, 0.0, w))
    print("    -- (4) distance-aware TILE_MIN inside the v2 cue --")
    for beta in (0.02, 0.04, 0.08):
        cue(f"v2  w=2.0, beta={beta} (floor relaxes with lag)",
            lambda s, b=beta: _fuse(s, 1.5, 2.0, b))
    for base in (0.70, 0.85):
        cue(f"v2  w=2.0, beta=0, base={base}",
            lambda s, b=base: _fuse(s, 1.5, 2.0, 0.0, b))

    # ── WHY does the constraint not pay? separate the RULE from the PHASE ───
    # If GT starts obey "start on the local loop phase" but our phase estimate
    # is wrong, the idea is right and the estimator is the bug. An ORACLE phase
    # (the residue class of the GT start itself) bounds the headroom.
    print("\nDIAGNOSTIC — is the RULE wrong, or only our PHASE estimate?")
    hit = tot = 0
    for s in songs:
        Vb = bar_vecs(s["Hb"])
        nv = _pad(nov_bar(s["Hb"], 0.0), s["nb"])
        Lm, Pm = phase_map(Vb, s["nb"], nv, 0.0, TILE_MIN)
        for sb in s["starts"]:
            if sb < 8 or sb > s["nb"] - 9 or Lm[sb] < 2:
                continue
            tot += 1
            hit += sb % Lm[sb] == Pm[sb]
    print(f"    GT starts landing on our detected loop phase: "
          f"{hit}/{tot} = {100*hit/max(1,tot):.1f}%  "
          f"(chance ~ 1/L, i.e. 25-50%)")
    res["phase_hit"] = [hit, tot]

    def _oracle(s, w=2.0):
        """ORACLE phase: the residue class is taken from the FIRST GT start of
        the song (not each start), so it is one leaked number per song, and it
        answers 'if the phase were right, would the constraint pay?'"""
        nb = s["nb"]
        v = _fuse(s, 0.0, 0.0)
        Vb = bar_vecs(s["Hb"])
        nv = _pad(nov_bar(s["Hb"], 0.0), nb)
        Lm, _ = phase_map(Vb, nb, nv, 0.0, TILE_MIN)
        ph0 = int(s["starts"][0])
        con = np.array([1.0 if Lm[b] >= 2 and b % Lm[b] == ph0 % Lm[b] else 0.0
                        for b in range(nb)])
        return v + w * con
    cue("ORACLE phase from song's 1st GT start, w=2 (headroom)", _oracle)

    res["rows"] = rows
    json.dump(res, open(CACHE / "tiling_v2.json", "w"), indent=1,
              default=lambda o: int(o) if isinstance(o, np.integer)
              else float(o))
    print("\nsaved", CACHE / "tiling_v2.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
