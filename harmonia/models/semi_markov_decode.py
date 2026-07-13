"""Per-beat semi-Markov (explicit-duration) decode over root × q5 — Mission 2.

Gen-1's HSMM attempt (docs/known_issues.md #1, Candidate B) failed because the
per-SEGMENT emission couldn't discriminate qualities: forcing the true ~2-beat
harmonic rhythm just exposed a weak emission more often. Gen-2 removes that
limitation — a trained per-beat ROOT posterior (beat_seq_v4, 96% per-beat on
clean renders) and a per-beat quality head (beat_seq_v3) exist — so an
explicit-duration decoder now has genuine per-beat evidence to place boundaries
on.

What this decoder decides vs. what it defers
---------------------------------------------
The headline lever (Mission 1's diagnosis) is ROOT: the segment grammar slot is
saturated, and 46–51% of root errors are 5th-apart acoustic confusions where a
single strong-but-wrong beat drags a whole span to the wrong root. An explicit
duration prior (jazz1460 puts ~0 mass on 1- and 3-beat chords; 57% on 2, 30% on
4) resists carving a spurious 1-beat segment around that beat, forcing it to
merge into the neighbouring 2/4-beat span whose SUMMED emission favours the true
root. So the decode decides ROOTS + BOUNDARIES.

QUALITY is NOT trusted to the decode. The v3 per-beat quality head is only 51.7%
q5-exact per-beat (premise check a), vs the segment ctx classifier's ~86% majmin.
So the caller re-runs the strong classifier on the DECODED segments for the final
quality label. The v3 head enters here only as an optional, down-weighted
per-beat quality EMISSION whose job is boundary placement (split where quality
changes even if the root doesn't) — never the final label.

Label-bias discipline (Mission 1 / Korzeniowski & Widmer)
---------------------------------------------------------
The duration prior carries its own label bias: maj/min have more long-duration
(4/8-beat) mass than dom/hdim/dim, so a per-quality duration prior would inject a
"long ⇒ major" snap analogous to the major-snap that killed the H2 grammar
fusion. Default here is therefore a QUALITY-INDEPENDENT (pooled) duration prior:
it shapes boundaries/roots and injects ZERO quality bias. A per-quality variant
is available but enters as a density ratio log[P(d|q)/P(d)] so it contributes
only each quality's duration shape *relative to the pooled marginal*, not the
marginal itself.

Reuses the FROZEN, tested `chord_hmm.viterbi_duration_aware` (O(T·D·C²)) as the
core recursion — states are (root, q5) with root-major indexing s = r*5 + q.
"""
from __future__ import annotations

import numpy as np

from harmonia.models.chord_hmm import viterbi_duration_aware

_NEG = -30.0  # log floor (exp(-30) ≈ 1e-13), matches clip below
Q5_NAMES = ("maj", "min", "dom", "hdim", "dim")


def build_emission(
    beat_proba: np.ndarray,          # (T, 12) per-beat root posterior
    qual_proba: np.ndarray | None,   # (T, 5) per-beat q5 posterior, or None
    *,
    qual_weight: float = 0.0,
) -> np.ndarray:
    """(T, 60) per-beat log-emission over states s = root*5 + q5.

    Separable: E[t, r*5+q] = log P_root(t,r) + qual_weight · log P_q5(t,q).
    With qual_weight=0 (default) quality is inert in the decode — boundaries and
    roots are driven by the root posterior + duration prior alone.
    """
    T = beat_proba.shape[0]
    root_e = np.log(np.clip(beat_proba, 1e-13, None))          # (T, 12)
    if qual_proba is not None and qual_weight != 0.0:
        qual_e = qual_weight * np.log(np.clip(qual_proba, 1e-13, None))  # (T, 5)
    else:
        qual_e = np.zeros((T, 5), dtype=np.float64)
    E = np.empty((T, 60), dtype=np.float64)
    for r in range(12):
        E[:, r * 5:(r + 1) * 5] = root_e[:, r][:, None] + qual_e
    return E


def build_log_duration(
    dur_pmf: dict,                   # {"pooled": (D,), "per_q5": (5, D)}
    *,
    dur_weight: float = 1.0,
    per_quality: bool = False,
    floor: float = 1e-6,
) -> np.ndarray:
    """(60, D) log-duration per state.

    Default (per_quality=False): a single POOLED log-PMF tiled across all 60
    states — zero quality bias (the label-bias discipline). ``dur_weight`` is a
    prior temperature: 0 ⇒ all-zeros ⇒ the duration term is inert and the decode
    reduces to plain per-beat argmax (see the degenerate-case unit test).

    per_quality=True: adds each q5's DENSITY RATIO log[P(d|q)/P(d)] on top of the
    pooled term, so a quality contributes only its duration shape relative to the
    pooled marginal — not the marginal quality frequency itself.
    """
    pooled = np.asarray(dur_pmf["pooled"], dtype=np.float64)
    D = pooled.shape[0]
    log_pooled = np.log(np.clip(pooled, floor, None))          # (D,)
    LD = np.tile(log_pooled, (60, 1))                          # (60, D)
    if per_quality:
        per = np.asarray(dur_pmf["per_q5"], dtype=np.float64)  # (5, D)
        log_per = np.log(np.clip(per, floor, None))
        ratio = log_per - log_pooled[None, :]                 # (5, D) density ratio
        for r in range(12):
            LD[r * 5:(r + 1) * 5, :] += ratio
    return dur_weight * LD


def semi_markov_decode(
    beat_proba: np.ndarray,
    *,
    dur_pmf: dict,
    qual_proba: np.ndarray | None = None,
    qual_weight: float = 0.0,
    dur_weight: float = 1.0,
    per_quality_duration: bool = False,
) -> dict:
    """Exact explicit-duration Viterbi over (root × q5), per beat.

    Args:
        beat_proba: (T, 12) per-beat root posterior (beat_seq_v4).
        dur_pmf: {"pooled": (D,), "per_q5": (5, D)} duration PMFs (jazz1460 fit).
        qual_proba: optional (T, 5) per-beat q5 posterior (v3 head); a
            down-weighted boundary signal only.
        qual_weight: weight on the quality emission (0 = root-only decode).
        dur_weight: prior temperature on the duration term (0 = plain per-beat).
        per_quality_duration: use the density-ratio per-quality duration shape.

    Returns dict:
        beat_root (T,) int, beat_q5 (T,) int, segments list[(s,e,root,q5)].
    """
    T = beat_proba.shape[0]
    E = build_emission(beat_proba, qual_proba, qual_weight=qual_weight)      # (T,60)
    LD = build_log_duration(dur_pmf, dur_weight=dur_weight,
                            per_quality=per_quality_duration)                # (60,D)
    log_trans = np.zeros((60, 60), dtype=np.float64)   # no bigram (Mission 1 dead end)
    log_init = np.zeros(60, dtype=np.float64)

    path, _ = viterbi_duration_aware(E, log_trans, log_init, LD)             # (T,)
    beat_root = (path // 5).astype(int)
    beat_q5 = (path % 5).astype(int)

    segments: list[tuple[int, int, int, int]] = []
    s = 0
    for t in range(1, T + 1):
        if t == T or path[t] != path[s]:
            segments.append((s, t, int(beat_root[s]), int(beat_q5[s])))
            s = t
    return {"beat_root": beat_root, "beat_q5": beat_q5, "segments": segments}
