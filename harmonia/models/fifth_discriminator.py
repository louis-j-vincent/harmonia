"""One-note V-vs-I root discriminator — kill-switched brick, default OFF.

MUSIC THEORY (Louis, 2026-07-23)
--------------------------------
The major scale of a key (I) and the scale of its dominant (V) differ by EXACTLY one
pitch class: the 4th degree of the tonic (natural) vs its sharpened form in the
dominant (the dominant's leading tone).  In C major vs G major that one note is
F-natural (4th of C) vs F# (leading tone of G).  So for two candidate roots a fifth
apart — R_low and R_high = R_low + 7 — the LOCAL CHROMA energy on:

    (R_low + 5) % 12   [natural 4th of R_low]  favors R_low  (tonic)
    (R_low + 6) % 12   [= leading tone of R_high] favors R_high (dominant)

This is the acoustic discriminator that a FLAT diatonic key prior cannot supply (V and
I are both diatonic) — the exact gap that sank the plain root_resolve brick.

SCOPE / WHAT THIS IS *NOT* (CLAUDE.md #4 — premise-screened 2026-07-23)
----------------------------------------------------------------------
- The premise HOLDS only for MAJOR-key tonic-vs-dominant confusions: on georgia
  (major) the note discriminator was 88% correct on fifth-wrong spans (vs 38% trust-NNLS,
  38% trust-bass) with a clean sign split.  It FAILS on blue_bossa (C minor + Db bridge,
  fast ii-V): 52% ≈ chance, no sign separation, and a |disc| gate does not rescue it.
- Therefore this brick is DELIBERATELY NARROW: it fires ONLY when the inferred key is
  MAJOR **and** one of the two fifth-apart candidates equals the key tonic (a genuine
  tonic-vs-dominant ambiguity) **and** |disc| >= a confidence floor.  On minor keys /
  non-tonic fifth pairs it ABSTAINS (returns the final root unchanged), so it cannot
  re-introduce the blue_bossa-style damage that made root_resolve −4.3pp.
- Even at best this recovers a small, one-song-dominated slice (georgia's fifth-wrong
  mass is ~0.45pp pooled).  It is a faithful capture of a CORRECT mechanism on a rare
  case, not a large lever.  Measure the on/off delta before trusting it.
- Chooses only between the two EXISTING candidate roots (final=musx vs nnls-argmax); it
  invents no new root, and does not touch quality or the sounding bass.

INTEGRATION HOOK (default OFF; wiring needs an edit to the concurrently-owned
chord_pipeline_v1.py, so it is NOT wired here)
-----------------------------------------------------------------------------------
Inside ``_label_segments``, after ``root`` (final) and ``nnls_root`` are known::

    from harmonia.models.fifth_discriminator import discriminate_fifth_root
    root = discriminate_fifth_root(root, nnls_root, seg_treble_chroma,
                                   tonic_pc, is_minor)   # env HARMONIA_FIFTH_DISC

Returns ``root`` unchanged when disabled (exact no-op).
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["enabled", "disc_threshold", "discriminate_fifth_root"]


def enabled() -> bool:
    return os.environ.get("HARMONIA_FIFTH_DISC", "").strip().lower() in ("1", "on", "true")


def disc_threshold() -> float:
    try:
        return float(os.environ.get("HARMONIA_FIFTH_DISC_THR", "0.03"))
    except ValueError:
        return 0.03


def _fifth_low(a: int, b: int) -> tuple[int, int] | None:
    """(R_low, R_high) for two pcs a fifth apart (R_high = R_low+7), else None."""
    if (b - a) % 12 == 7:
        return a, b
    if (a - b) % 12 == 7:
        return b, a
    return None


def discriminate_fifth_root(
    final_root: int,
    nnls_root: int,
    treble_chroma: np.ndarray,
    tonic_pc: int | None = None,
    is_minor: bool = False,
    *,
    force: bool = False,
    thr: float | None = None,
) -> int:
    """Resolve a fifth-disagreement between final (musx) and NNLS roots via the
    one-note (4th-degree) discriminator.  Returns one of {final_root, nnls_root}.

    Fires only if ALL hold (else returns ``final_root`` unchanged):
      * enabled (env HARMONIA_FIFTH_DISC) or ``force=True``;
      * final_root and nnls_root are a P4/P5 apart;
      * the key is MAJOR (not is_minor) and ``tonic_pc`` is one of the two candidates
        (genuine tonic-vs-dominant ambiguity), when ``tonic_pc`` is provided;
      * |disc| >= thr, where disc = chroma[R_low+5] - chroma[R_low+6].

    Decision: disc > 0 -> R_low (tonic); disc < 0 -> R_high (dominant).
    """
    if not (force or enabled()):
        return final_root
    if final_root == nnls_root:
        return final_root
    fl = _fifth_low(final_root, nnls_root)
    if fl is None:
        return final_root
    R_low, R_high = fl
    if is_minor:
        return final_root
    if tonic_pc is not None and tonic_pc not in (R_low, R_high):
        return final_root
    c = np.asarray(treble_chroma, dtype=float)
    s = float(c.sum())
    if s <= 0:
        return final_root
    c = c / s
    disc = c[(R_low + 5) % 12] - c[(R_low + 6) % 12]
    _thr = disc_threshold() if thr is None else thr
    if abs(disc) < _thr:
        return final_root
    return int(R_low if disc > 0 else R_high)
