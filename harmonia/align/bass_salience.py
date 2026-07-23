"""Stream #4 of the Bayesian fusion aligner — the BASS-SALIENCE instrument.

This is the standalone bass-pitch instrument the BASS premise-check (2026-07-23,
see ``docs/fusion_aligner_design.md``) validated. It reads the SOUNDING BASS
pitch-class per span from the low end of the spectrum and self-reports a per-span
RELIABILITY that auto-downweights the places where the bass carries no usable
pitch (walking jazz bass), and stays high where the bass states clear roots
(pop / soul). It serves two downstream roles:

  * it IS the ``sounding-bass`` GT target (``corpus_schema.sounding_bass_pc``);
  * it contributes the reliability-weighted BASS term to the global-phase
    downbeat resolver (``harmonia.align.downbeat``) — "bass root lands on beat 1",
    weighted by its OWN per-song recurrence so it never dominates on swing.

WHY THE RECIPE IS WHAT IT IS (load-bearing, from the premise check).
--------------------------------------------------------------------
* **Duration-INTEGRATE over the span, NEVER sample the downbeat instant.** The
  bass ATTACK transient is broadband and masks the pitch for ~1 beat; the naive
  "read the pitch at the beat onset" cratered to 15-30%. Reading the argmax of
  the chroma integrated over the whole span (dominated by the sustain) recovered
  it. `bass_pc_over_span` therefore averages the folded chroma over ``[t0, t1)``.
* **Reliability = bass-band ENERGY x argmax CONCENTRATION (peakiness).** A
  walking bass spreads energy across every pitch class within a span -> a FLAT
  folded chroma -> low concentration -> low reliability (Autumn auto-downweights);
  a held/repeated root -> a PEAKY chroma -> high reliability (Stand By Me). This
  self-detection is the whole point: the fusion trusts the bass exactly where the
  bass is trustworthy, with no per-song tuning.
* **Fifth / harmonic guard.** 28-48% of the premise's bass misses were the chord
  3rd or (mostly) the 5th winning the argmax — the played root's perfect fifth /
  third resonates strongly one region up while its own fundamental sits lowest.
  When the argmax pc is a fifth (or major/minor third) ABOVE a pitch class that
  carries MORE energy in the lowest octave, we prefer that lower fundamental —
  recovering a chunk of the jazz misses without hurting the clean pop roots.

FEATURE RECIPE (frozen): C1..B3 CQT (36 bins, 12 bins/octave, sr 22050, hop 512),
magnitude, folded to a 12-PC **flat** bass chroma (plain octave sum, no
perceptual weighting). C1 (32.70 Hz) .. B3 (~247 Hz) is the bass register; the
folded chroma is the sounding-bass evidence.

API
---
* ``bass_chroma(audio) -> BassChroma``  — whole-song flat bass chroma + per-frame
  bass-band energy + frame times (the object the other calls consume).
* ``bass_pc_over_span(bass, t0, t1) -> (pc, reliability)`` — the duration-
  integrated sounding-bass pitch class over a span + its reliability in [0, 1].
* ``bass_reliability(bass, t0, t1) -> float`` — just the reliability term.

NON-CIRCULARITY. Pure raw-audio CQT; nothing here touches the model's chord
decode. Audio-free unit tests drive it with synthetic ``BassChroma`` objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

# ── frozen feature-extraction constants ──────────────────────────────────────
SR = 22050
HOP = 512
N_OCTAVES = 3               # C1..B3
BINS_PER_OCTAVE = 12
N_BINS = N_OCTAVES * BINS_PER_OCTAVE   # 36
FMIN_NOTE = "C1"            # 32.70 Hz — bottom of the bass register

# ── guard defaults ───────────────────────────────────────────────────────────
_GUARD_INTERVALS = (7, 4, 3)   # argmax may be the 5th / maj-3rd / min-3rd of root
_GUARD_LOW_RATIO = 1.0         # lower fundamental must carry >= this x the argmax's
                               #   low-octave energy to override
_GUARD_MIN_FRAC = 0.5          # ...and the candidate's folded mass must be >= this
                               #   fraction of the argmax's (a real 2nd peak)


@dataclass
class BassChroma:
    """Whole-song flat bass chroma (the object the span calls consume).

    Attributes
    ----------
    chroma : (F, 12) float
        Per-frame flat bass chroma (folded octave sum of the C1..B3 CQT
        magnitude, idx 0 == C). NOT normalised — magnitudes are comparable
        across frames so a silent frame reads as low energy.
    low_chroma : (F, 12) float
        Same fold but from the LOWEST octave only (C1..B1) — the fundamental
        evidence the fifth/third guard uses to find the true low root.
    times : (F,) float
        Frame times (s).
    energy : (F,) float
        Per-frame total bass-band energy (sum over the 36 CQT bins).
    energy_ref : float
        Song-level energy scale (95th percentile of ``energy``) used to
        normalise a span's energy into [0, 1] for the reliability term.
    sr, hop : int
    """
    chroma: np.ndarray
    low_chroma: np.ndarray
    times: np.ndarray
    energy: np.ndarray
    energy_ref: float
    sr: int = SR
    hop: int = HOP


# ═════════════════════════════════════════════════════════════════════════════
# Whole-song extraction
# ═════════════════════════════════════════════════════════════════════════════

def bass_chroma(audio: Union[str, np.ndarray], sr: int = SR,
                hop: int = HOP) -> BassChroma:
    """Compute the whole-song flat bass chroma from a wav path or a mono signal.

    C1..B3 CQT magnitude -> folded to a 12-PC flat bass chroma (plain octave
    sum). Also returns the lowest-octave (C1..B1) fold for the guard and the
    per-frame bass-band energy for the reliability term.
    """
    import librosa

    if isinstance(audio, str):
        y, _sr = librosa.load(audio, sr=sr, mono=True)
    else:
        y = np.asarray(audio, dtype=np.float32)
        if sr != SR:                              # keep the frozen analysis rate
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            sr = SR
    fmin = librosa.note_to_hz(FMIN_NOTE)
    C = np.abs(librosa.cqt(y=np.ascontiguousarray(y, dtype=np.float32), sr=sr,
                           hop_length=hop, fmin=fmin, n_bins=N_BINS,
                           bins_per_octave=BINS_PER_OCTAVE))            # (36, F)
    C = C.astype(float)
    F = C.shape[1]
    blocks = C.reshape(N_OCTAVES, BINS_PER_OCTAVE, F)     # (octave, pc, frame)
    chroma = blocks.sum(axis=0).T                          # (F, 12) flat fold
    low_chroma = blocks[0].T                               # (F, 12) C1..B1 only
    energy = C.sum(axis=0)                                  # (F,) bass-band energy
    times = librosa.frames_to_time(np.arange(F), sr=sr, hop_length=hop)
    energy_ref = float(np.percentile(energy, 95)) if F else 1.0
    return BassChroma(chroma=chroma, low_chroma=low_chroma, times=times,
                      energy=energy, energy_ref=max(energy_ref, 1e-9),
                      sr=sr, hop=hop)


# ═════════════════════════════════════════════════════════════════════════════
# Per-span sounding-bass pitch class + reliability
# ═════════════════════════════════════════════════════════════════════════════

def _span_slice(times: np.ndarray, t0: float, t1: float) -> slice:
    lo = int(np.searchsorted(times, t0))
    hi = int(np.searchsorted(times, t1))
    if hi <= lo:                                  # sub-frame span: take one frame
        hi = min(lo + 1, len(times))
    return slice(lo, hi)


def _concentration(vec: np.ndarray) -> float:
    """Peakiness of a 12-PC chroma in [0, 1]: how far the argmax's mass share
    exceeds the flat 1/12 baseline. A walking bass (energy on every pc) -> ~0; a
    held root (one dominant pc) -> ~1. This IS the auto-downweight."""
    s = float(vec.sum())
    if s <= 0:
        return 0.0
    top = float(vec.max()) / s
    return float(np.clip((top - 1.0 / 12.0) / (1.0 - 1.0 / 12.0), 0.0, 1.0))


def _fifth_guard(folded: np.ndarray, low: np.ndarray, pc: int) -> int:
    """Fifth/third harmonic guard (see module docstring).

    If the argmax ``pc`` is a perfect fifth / major-3rd / minor-3rd ABOVE a pitch
    class that (a) carries at least as much LOWEST-octave (fundamental) energy and
    (b) is itself a substantial peak in the folded chroma, return that lower
    fundamental instead. Prefers the fifth (the dominant harmonic confusion)."""
    top = float(folded[pc])
    if top <= 0:
        return pc
    for delta in _GUARD_INTERVALS:
        cand = (pc - delta) % 12
        if (low[cand] >= _GUARD_LOW_RATIO * low[pc]
                and low[cand] > 0
                and folded[cand] >= _GUARD_MIN_FRAC * top):
            return cand
    return pc


def bass_pc_over_span(bass: BassChroma, t0: float, t1: float,
                      guard: bool = True) -> tuple[int, float]:
    """Sounding-bass pitch class over ``[t0, t1)`` + its reliability in [0, 1].

    The pitch class is the argmax of the DURATION-INTEGRATED (mean) folded bass
    chroma over the span — never a single instant, so the attack transient can't
    mask the pitch (the load-bearing recipe fix). ``guard`` applies the
    fifth/third harmonic guard. Reliability = span bass-band energy (vs the
    song's 95th-pct energy) x argmax concentration; ~0 on a walking/absent bass,
    high on a clear held root.
    """
    sl = _span_slice(bass.times, t0, t1)
    if sl.start >= sl.stop:
        return 0, 0.0
    folded = bass.chroma[sl].mean(axis=0)
    if folded.sum() <= 0:
        return int(np.argmax(folded)), 0.0
    pc = int(np.argmax(folded))
    if guard:
        low = bass.low_chroma[sl].mean(axis=0)
        pc = _fifth_guard(folded, low, pc)
    energy_norm = float(np.clip(bass.energy[sl].mean() / bass.energy_ref, 0.0, 1.0))
    rel = energy_norm * _concentration(folded)
    return pc, float(rel)


def bass_reliability(bass: BassChroma, t0: float, t1: float) -> float:
    """The per-span reliability term alone (see `bass_pc_over_span`)."""
    return bass_pc_over_span(bass, t0, t1, guard=False)[1]


def bass_pc_series(bass: BassChroma, boundaries: np.ndarray,
                   guard: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised convenience: the sounding-bass pc + reliability for every span
    ``[boundaries[k], boundaries[k+1])``. Returns ``(pcs, reliabilities)`` of
    length ``len(boundaries) - 1``. Used by the downbeat resolver to read the
    bass root per beat."""
    b = np.asarray(boundaries, float)
    n = max(len(b) - 1, 0)
    pcs = np.zeros(n, dtype=int)
    rels = np.zeros(n, dtype=float)
    for k in range(n):
        pcs[k], rels[k] = bass_pc_over_span(bass, b[k], b[k + 1], guard=guard)
    return pcs, rels
