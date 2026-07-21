"""Importable prototype: section clustering = harmony (pos-agree) + distinctive-chord
veto + energy-confirmer arbitration. Kept out of chart_model.py (concurrent WIP);
wire in later. Deterministic, no audio deps here (energy passed in as per-block scalar)."""
from collections import Counter
import numpy as np

MATCH, PHASE_STRICT, MAXLAG = 0.6, 0.80, 1

# Energy-arbitration operating point (see cluster() docstring). Exported as
# module constants so a SECOND consumer (chart_model._sections_by_largest_unit,
# which ports the energy branch into its own tuned single-linkage loop rather
# than calling cluster()) shares one source of truth for the thresholds and the
# z-score/CoV trust gate below — no duplicated, silently-drifting constants.
E_SAME, E_DIFF, E_MIN_COV = 0.4, 0.8, 0.10


def energy_zscores(block_energy, *, use_energy=True, min_cov=E_MIN_COV):
    """Per-song z-scores of per-block energy, or ``None`` when energy is untrusted.

    The right notion of "unusually different energy" is relative to THIS song's
    own dynamic range (a z-score), not an absolute or relative-to-median measure
    (real audio varies 10-15% inside a section, so a relative threshold false-
    splits ~90% of same-section pairs — measured). Returns ``None`` (energy
    silently skipped, caller falls back to harmony+veto) unless BOTH robustness
    gates pass:
      * ``use_energy`` on AND ``block_energy`` supplied;
      * ``len >= 4``          — a stable per-song distribution (few blocks → degenerate);
      * ``CoV(e) >= min_cov`` — the song HAS real dynamics (on a flat vamp the
        z-score would amplify pure noise into false splits).
    """
    if not use_energy or block_energy is None:
        return None
    e = np.asarray(block_energy, float)
    if e.size < 4:
        return None
    if e.mean() > 1e-9 and (e.std() / e.mean()) >= min_cov:
        return (e - e.mean()) / (e.std() + 1e-9)
    return None


def _om(a, b):
    k = min(len(a), len(b))
    return sum(1 for x, y in zip(a[:k], b[:k]) if x == y) / k if k else 0.0

def sim(a, b):
    if not a or not b:
        return 0.0
    base = (sum(1 for x, y in zip(a, b) if x == y) / len(a)
            if len(a) == len(b) else _om(a, b))
    best = base
    for lag in range(1, MAXLAG + 1):
        for m in (_om(a[lag:], b), _om(a, b[lag:])):
            if m >= PHASE_STRICT:
                best = max(best, m)
    return best

def veto(a, b, min_recur=2, min_frac=0.2):
    ca, cb = Counter(a), Counter(b)
    sa, sb = set(a), set(b)
    for r, n in ca.items():
        if n >= min_recur and n >= min_frac * len(a) and r not in sb:
            return True
    for r, n in cb.items():
        if n >= min_recur and n >= min_frac * len(b) and r not in sa:
            return True
    return False

def cluster(block_roots, block_energy=None, use_veto=True, use_energy=True,
            e_same=E_SAME, e_diff=E_DIFF):
    """single-linkage; returns per-block integer cluster id.

    DEFAULT OPERATING POINT (user error-preference, confirmed 2026-07-21):
    "je préfère l'erreur 2 à l'erreur 1, donc privilégie plus de sections que
    moins" — UNDER-split (a real repeat split into two letters) is preferred over
    OVER-merge (a genuinely different section hidden under an existing letter).
    So the arbiter is biased toward MORE sections / NOT merging when the evidence
    is ambiguous:
      * use_veto/use_energy ON by default (they only ever BLOCK merges → more sections);
      * e_diff=0.8 (aggressive): a modest energy difference is enough to split a
        harmonically-identical pair (real-audio gate: 52% over-merge / 28.5%
        under-split vs harmony-only 71.8%/16.4% — trades over→under, as wanted);
      * e_same=0.4 (strict): the energy-similarity override that RE-ALLOWS a
        veto-blocked merge fires only when energy is CLEARLY the same — so an
        AMBIGUOUS energy (0.4 ≤ |dz| ≤ 0.8) leaves the veto's split STANDING
        (default = don't merge under ambiguity, per the user's principle).

    energy arbitration (calibrated evidence, not blind override):
      - override veto (allow merge) when energy is SIMILAR  (|dz|<e_same) -> same section varied
      - block a harmony-merge when energy is VERY DIFFERENT (|dz|>e_diff) -> diff section, harmony silent
    """
    nb = len(block_roots)
    # Per-song z-score with the nb>=4 / CoV>=E_MIN_COV trust gate (see
    # energy_zscores). ``None`` when energy is untrusted → harmony(+veto) only.
    z = energy_zscores(block_energy, use_energy=use_energy)
    parent = list(range(nb))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def dz(i, j):
        return abs(z[i] - z[j]) if z is not None else 0.0
    for i in range(nb):
        for j in range(i + 1, nb):
            s = sim(block_roots[i], block_roots[j])
            if s < MATCH:
                continue                      # harmony says different -> no merge
            merge = True
            if use_veto and veto(block_roots[i], block_roots[j]):
                merge = False                 # distinctive-chord veto blocks
                if use_energy and dz(i, j) < e_same:
                    merge = True               # ...unless energy says same section (varied)
            if merge and use_energy and dz(i, j) > e_diff:
                merge = False                 # harmony-same but energy strongly differs -> split
            if merge:
                parent[find(i)] = find(j)
    # normalize ids
    ids = {}
    out = []
    for i in range(nb):
        r = find(i)
        out.append(ids.setdefault(r, len(ids)))
    return out
