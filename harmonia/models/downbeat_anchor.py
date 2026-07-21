"""SOTA downbeat-phase anchor (2026-07-21).

User's own framing: "the beat tracker already works great, the problem is
finding beat 1 of the song." The existing ``_flux_downbeat_phase`` (chord_
pipeline_v1.py) tackles this from harmonic content (where chord CHANGES
cluster); this module tackles it with Beat This! (Foscarin, Schlüter,
Widmer, ISMIR 2024, "Accurate Beat Tracking Without DBN Postprocessing") —
a transformer-based downbeat tracker, added 2026-07-21 (``pip install
beat_this``) as a purpose-built, independent second opinion.

Confidence design (screened cheaply first, CLAUDE.md rule #2 — see
scratchpad/downbeat_triangulation.py and beat_this's own framewise output,
inspected directly before writing any of this): Beat This!'s per-frame peak
probability at its OWN picked downbeats turned out NOT to discriminate well
(0.97 on a clean pop song vs 0.90 on a rubato jazz piano cover — barely
separated, peak-picking finds *a* locally-confident peak even when the
overall track is unreliable). The regularity of the INTER-DOWNBEAT SPACING
does: a confidently-tracked song's downbeats land on an almost perfectly
constant period (Hot n Cold: 1.80–1.84s throughout, bar one genuine 20s gap
during a spoken bridge); a rubato jazz piano performance's downbeats
scattered from 1.5s to 4.2s with no consistent period at all. That gap
(≈97% of intervals within 15% of the median vs ≈30%) is the confidence
signal this module actually uses.

An earlier version of this module cross-checked Beat This! against madmom's
RNN+DBN downbeat tracker (also already in this repo, ``harmonia.models.
rhythm``) for every song — works, but madmom takes ~30-35s per song on real
audio vs Beat This!'s ~4-9s, and on the one case where it would have
mattered (a hard, rubato-heavy song) the 8-song triangulation showed madmom
and Beat This! disagree with EACH OTHER too (25-40% agreement) — i.e. paying
madmom's cost doesn't reliably rescue exactly the cases it would be paid
for. So the default path is Beat This! alone, gated by its own internal
regularity; madmom stays available via ``cross_check_with_madmom`` for
manual/offline comparison, not in the hot path.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_A2F = None       # lazy singleton — model load is expensive, reuse across calls
_PP = None


def _get_beat_this():
    global _A2F, _PP
    if _A2F is None:
        from beat_this.inference import Audio2Frames, Postprocessor
        _A2F = Audio2Frames(checkpoint_path="final0")
        _PP = Postprocessor(type="minimal", fps=50)
    return _A2F, _PP


def _regularity(times: np.ndarray, tol_frac: float = 0.15) -> float:
    """Fraction of inter-event intervals within ``tol_frac`` of the MEDIAN
    interval — robust to the odd real gap (e.g. a spoken bridge with no
    beat), unlike a coefficient-of-variation (which one huge outlier can
    dominate and invert the intended ranking; confirmed while screening this
    metric, see module docstring)."""
    if len(times) < 5:
        return 0.0
    iv = np.diff(times)
    med = np.median(iv)
    if med <= 0:
        return 0.0
    return float(np.mean(np.abs(iv - med) <= tol_frac * med))


def beat_this_downbeats(audio_path: "Path | str") -> "tuple[np.ndarray, float]":
    """(downbeat_times, regularity_confidence ∈ [0,1]) via Beat This!."""
    from beat_this.inference import load_audio

    a2f, pp = _get_beat_this()
    signal, sr = load_audio(str(audio_path))
    beat_logits, downbeat_logits = a2f(signal, sr)
    _beats, downbeats = pp(beat_logits, downbeat_logits)
    downbeats = np.asarray(downbeats, dtype=float)
    return downbeats, _regularity(downbeats)


def sota_downbeat_phase(
    audio_path: "Path | str", bar_period: float, beats_per_bar: int = 4,
    min_confidence: float = 0.85,
) -> "tuple[int, float] | None":
    """Same return contract as ``chord_pipeline_v1._flux_downbeat_phase``
    (``phi ∈ [0, beats_per_bar)``, a confidence-like ``ratio``) so it slots
    into the exact same call site as a drop-in first attempt. Returns
    ``None`` (abstain) when Beat This!'s downbeat spacing isn't regular
    enough to trust — the caller's existing flux/structure fallback chain
    handles that case unchanged, and cheaply (no madmom cost paid).

    ``ratio`` is returned as a large sentinel (999.0) on success: the
    caller's only use of ``ratio`` is a ``< 1.05`` weak-comb tie-break, which
    doesn't apply here — the sentinel just skips that check cleanly.
    """
    try:
        downbeats, conf = beat_this_downbeats(audio_path)
    except Exception as exc:  # noqa: BLE001 — never break analyse over this anchor
        logger.warning("sota_downbeat_phase: beat_this failed (%s)", exc)
        return None
    if len(downbeats) < 5 or conf < min_confidence:
        logger.info("sota_downbeat_phase: low regularity (conf=%.2f, n=%d) — "
                   "abstaining, caller falls back to flux/structure", conf, len(downbeats))
        return None

    beat_period = bar_period / beats_per_bar
    ang = 2 * np.pi * (downbeats % bar_period) / bar_period
    phase_s = float((np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi))
                    * bar_period / (2 * np.pi))
    phi = int(round(phase_s / beat_period)) % beats_per_bar
    logger.info("sota_downbeat_phase: beat_this confident (conf=%.2f, n=%d downbeats) "
               "-> phi=%d beats", conf, len(downbeats), phi)
    return phi, 999.0


def cross_check_with_madmom(audio_path: "Path | str", tol_s: float = 0.07) -> dict:
    """Offline/manual comparison only (NOT in the hot path — madmom costs
    ~30-35s/song vs Beat This!'s ~4-9s, see module docstring for why paying
    that cost by default isn't justified). Returns agreement stats between
    Beat This! and madmom's RNN+DBN downbeat tracker."""
    from harmonia.models.rhythm import RhythmAnalyser

    downbeats, conf = beat_this_downbeats(audio_path)
    grid = RhythmAnalyser(prefer_madmom=True).analyse(str(audio_path))
    md = np.asarray(grid.downbeat_times, dtype=float)

    def agreement(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0.0
        return float(np.mean([np.min(np.abs(b - t)) <= tol_s for t in a]))

    return {
        "beat_this_confidence": conf, "n_beat_this": len(downbeats), "n_madmom": len(md),
        "agree_bt_to_md": agreement(downbeats, md), "agree_md_to_bt": agreement(md, downbeats),
    }
