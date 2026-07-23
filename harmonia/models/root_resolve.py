"""Key-weighted fifth-disagreement root resolver — kill-switched brick, default OFF.

WHY THIS EXISTS
---------------
On the frozen Brick-0 benchmark, 42% of chord-ROOT errors are fifth-related (P5 23%
+ P4 19%) — the classic root-vs-fifth / functional-neighbour confusion.  Under the
shipped config (quality_frontend='musx') the FINAL root is music-x-lab's ``mx_root``;
the trained NNLS-24 root head (``beat_proba`` argmax) is used only for segmentation +
bass-veto.  A premise screen (2026-07-23, georgia + blue_bossa) found that where the
final root is wrong by a fifth, the NNLS argmax ALREADY holds the true root in ~42%
of that duration (music-x-lab overrode a correct NNLS root) — a partially recoverable
signal.  This brick resolves such fifth-disagreements TOWARD the NNLS root, but only
when the NNLS posterior is confident AND the NNLS root is diatonic to the inferred key
(so V-vs-I style ambiguities are broken by acoustic evidence, not blind smoothing).

WHAT THIS IS *NOT* (CLAUDE.md #4 — state the unsolved remainder)
---------------------------------------------------------------
- NOT a general transition/Viterbi smoother.  A pure key/LM prior CANNOT fix a
  fifth-confusion: both roots (V and I) are diatonic, and the fifth is the commonest
  diatonic root motion, so a smoothness/key prior has no basis to prefer one over the
  other.  Only acoustic evidence (the NNLS posterior) disambiguates — hence this uses it.
- Ceiling is ~1.6pp pooled root (premise screen), one-song-dominated (blue_bossa), and
  only 42% of fifth-wrong duration is recoverable; realized gain is lower and must be
  MEASURED (precision risk: swapping can break currently-correct spans where musx is
  right but NNLS is a fifth off).  Do NOT wire ON without the measured on/off delta.
- Touches ONLY roots that disagree with the NNLS argmax by a P4/P5.  Other error modes
  (m3/TT/m7 = 34% of root errors) are untouched.
- Does NOT change quality or the sounding-bass note (bass is re-slashed against the
  new root only for the /bass display).

INTEGRATION HOOK (default OFF; wiring needs an edit to the concurrently-owned
chord_pipeline_v1.py, so it is NOT wired here)
-----------------------------------------------------------------------------------
Inside ``_label_segments``, after ``root`` is chosen and before the label is built::

    from harmonia.models.root_resolve import resolve_root
    root = resolve_root(root, nnls_root, p_seg, key_diatonic_pcs)  # env HARMONIA_ROOT_RESOLVE

``resolve_root`` returns ``root`` unchanged when the env flag is off (exact no-op).
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["resolve_root", "enabled", "resolver_params"]

_MAJ = (0, 2, 4, 5, 7, 9, 11)
_NATMIN = (0, 2, 3, 5, 7, 8, 10)


def key_diatonic_pcs(tonic_pc: int, is_minor: bool) -> set[int]:
    """Diatonic pitch classes of a major/natural-minor key."""
    scale = _NATMIN if is_minor else _MAJ
    return {(tonic_pc + i) % 12 for i in scale}


def enabled() -> bool:
    return os.environ.get("HARMONIA_ROOT_RESOLVE", "").strip().lower() in ("1", "on", "true")


def resolver_params() -> tuple[float, float]:
    """(tau, margin) from env, with defaults tuned on the premise screen."""
    try:
        tau = float(os.environ.get("HARMONIA_ROOT_RESOLVE_TAU", "0.35"))
    except ValueError:
        tau = 0.35
    try:
        margin = float(os.environ.get("HARMONIA_ROOT_RESOLVE_MARGIN", "0.10"))
    except ValueError:
        margin = 0.10
    return tau, margin


def resolve_root(
    final_root: int,
    nnls_root: int,
    p_seg: np.ndarray,
    key_diatonic: set[int] | None = None,
    *,
    force: bool = False,
    tau: float | None = None,
    margin: float | None = None,
) -> int:
    """Resolve a fifth-disagreement between the final (musx) root and the NNLS root.

    Returns ``nnls_root`` iff ALL hold, else ``final_root`` unchanged:
      * the brick is enabled (env HARMONIA_ROOT_RESOLVE) or ``force=True``;
      * final_root and nnls_root differ by a P4 or P5 ((diff % 12) in {5, 7});
      * the per-segment NNLS posterior mass on nnls_root >= tau, and its margin over
        final_root >= margin (confident acoustic preference);
      * nnls_root is diatonic to the inferred key (or no key supplied).

    Reference behaviour is a no-op (returns final_root) when disabled.
    """
    if not (force or enabled()):
        return final_root
    if final_root == nnls_root:
        return final_root
    if (final_root - nnls_root) % 12 not in (5, 7):
        return final_root
    _tau, _margin = resolver_params()
    if tau is not None:
        _tau = tau
    if margin is not None:
        _margin = margin
    p = np.asarray(p_seg, dtype=float)
    s = float(p.sum())
    if s <= 0:
        return final_root
    p = p / s
    if p[nnls_root] < _tau:
        return final_root
    if (p[nnls_root] - p[final_root]) < _margin:
        return final_root
    if key_diatonic is not None and nnls_root not in key_diatonic:
        return final_root
    return int(nnls_root)
