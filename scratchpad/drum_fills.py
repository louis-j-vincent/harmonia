#!/usr/bin/env python3
"""drum_fills.py — per-BAR drum-fill / novelty score, 2026-07-30 (reframe of
rhythm_ssm.py after the identity-SSM null on This Love).

Louis's reframe, verbatim: "the rythmic SSM isn't enough of itself, what we
need more than pattern differences, is just to detect fill ins, and small
drum changes right before rythm changes, which is quite common."

So this is NOVELTY against a LOCAL baseline, not identity against the whole
song. A drummer plays a fill in the bar before a section change and often
crashes on the new section's downbeat. Prediction that inverts the rhythm_ssm
null: This Love's groove is near-uniform 8th/16th activity everywhere (that
uniformity is WHY the identity SSM carried no section signal — see
docs/research_sessions/rhythm_ssm_2026-07-30.md) — but that same uniformity
should make a fill an unusually clean local outlier against a flat baseline.

Four components, scored separately (report which one carries the signal, do
not just ship a blend):
  1. onset-density excess   — total onset energy this bar vs local ±radius median.
  2. groove-template deviation — 1 - cosine(this bar's onset patch, LOCAL
     median patch of the ±radius neighbourhood) -- reuses rhythm_ssm's patch
     machinery but with a local, not global, reference.
  3. crash spike — high-band (cymbal) transient at a bar's OWN downbeat,
     re-indexed to the PRECEDING bar (a crash at the downbeat of bar b+1 is
     evidence for a boundary AT b+1, i.e. it is fill-evidence FOR bar b, same
     convention as components 1/2/4).
  4. snare/tom flurry — mid-band onset PEAK COUNT excess (a fill is a run of
     several closely-spaced hits, not just more energy).

Output convention: ``fill_scores()[b]`` is high when bar ``b`` looks like a
fill / is preceded by a local-outlier crash, i.e. it predicts a section
BOUNDARY AT ``b + 1``. The last bar has no ``b+1`` and is always 0.

Run:  .venv/bin/python scratchpad/drum_fills.py [song-substring]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from rhythm_ssm import (DEFAULT_STEM_CACHE, _drum_onset_bands, _hpss_percussive,
                         build_slot_patches, separate_drums)

__all__ = ["fill_scores", "component_scores"]


# ── per-bar aggregations ────────────────────────────────────────────────────

def _bar_patches(env: np.ndarray, times: np.ndarray, bar_bounds_sec: list[float],
                  sub_steps: int = 16) -> np.ndarray:
    """(n_bars, 3*sub_steps) L2-normalised whole-bar onset patch (reuses
    rhythm_ssm.build_slot_patches with the bar grid itself as the slot grid,
    i.e. slots_per_bar=1)."""
    V = build_slot_patches(env, times, bar_bounds_sec, sub_steps=sub_steps, shifts=(0.0,))
    return V[:, 0, :]


def _bar_sums(env: np.ndarray, times: np.ndarray, bar_bounds_sec: list[float]) -> np.ndarray:
    """Total (all-band) onset energy integrated within each bar."""
    e = env.sum(axis=0)
    n_bars = len(bar_bounds_sec) - 1
    out = np.zeros(n_bars)
    for b in range(n_bars):
        t0, t1 = bar_bounds_sec[b], bar_bounds_sec[b + 1]
        m = (times >= t0) & (times < t1)
        out[b] = float(e[m].sum())
    return out


def _bar_peak_counts(env_band: np.ndarray, times: np.ndarray, bar_bounds_sec: list[float],
                      rel_height: float = 0.25) -> np.ndarray:
    """Count of discrete onset peaks (not just summed energy) in each bar, on
    one band -- a flurry is many CLOSE-TOGETHER hits, not just more energy."""
    from scipy.signal import find_peaks
    n_bars = len(bar_bounds_sec) - 1
    thresh = rel_height * float(env_band.max() or 1.0)
    peaks, _ = find_peaks(env_band, height=thresh, distance=2)
    ptimes = times[peaks]
    out = np.zeros(n_bars)
    for b in range(n_bars):
        t0, t1 = bar_bounds_sec[b], bar_bounds_sec[b + 1]
        out[b] = int(((ptimes >= t0) & (ptimes < t1)).sum())
    return out


def _downbeat_peak(env_band: np.ndarray, times: np.ndarray, bar_bounds_sec: list[float],
                    pre_sec: float = 0.06, post_sec: float = 0.15) -> np.ndarray:
    """Max energy of one band in a short window straddling EACH bar's own
    downbeat (its start time) -- the crash usually lands right on/just after
    the downbeat, occasionally a hair early."""
    n_bars = len(bar_bounds_sec) - 1
    out = np.zeros(n_bars)
    for b in range(n_bars):
        t0 = bar_bounds_sec[b]
        m = (times >= t0 - pre_sec) & (times <= t0 + post_sec)
        out[b] = float(env_band[m].max()) if m.any() else 0.0
    return out


def _local_median(x: np.ndarray, radius: int = 4, exclude_self: bool = True,
                   causal: bool = False) -> np.ndarray:
    """``causal=True``: the reference window for bar i is ONLY bars
    ``[i-radius, i)`` -- the established pattern BEFORE i -- never bars after
    i. Root-cause fix (2026-07-30): a SYMMETRIC window straddling a real
    boundary blends two different patterns into one "local median", so every
    bar within ``radius`` of a boundary (on EITHER side) reads as elevated
    deviation, not just the true fill bar -- confirmed on Every Breath You
    Take, where the symmetric-window groove_dev top-4 included bar 12 (the
    boundary's OWN first bar, not bar 11 the fill target) and the whole
    8-16 span alternated 0/1 with no relation to the true boundary. A causal
    window can't straddle a boundary it hasn't reached yet."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        if causal:
            lo, hi = max(0, i - radius), i
        else:
            lo, hi = max(0, i - radius), min(n, i + radius + 1)
        idx = [j for j in range(lo, hi) if not (exclude_self and j == i)]
        out[i] = float(np.median(x[idx])) if idx else float(x[i])
    return out


def _robust_excess(x: np.ndarray, radius: int = 4, causal: bool = False) -> np.ndarray:
    """(x - local median) / local MAD, clipped at 0 (only POSITIVE excess is
    fill evidence -- a quiet bar is not a fill), rescaled to [0,1] by the
    song's own 95th percentile of the excess (a soft cap so one freak outlier
    doesn't crush the rest of the scale)."""
    med = _local_median(x, radius, causal=causal)
    mad = _local_median(np.abs(x - med), radius, causal=causal)
    z = (x - med) / (mad * 1.4826 + 1e-9)
    z = np.clip(z, 0.0, None)
    cap = float(np.percentile(z, 95)) if z.max() > 0 else 1.0
    cap = max(cap, 1e-6)
    return np.clip(z / cap, 0.0, 1.0)


# ── the four components + combination ───────────────────────────────────────

def component_scores(audio_path, bar_bounds_sec: list[float], *, radius: int = 4,
                      sub_steps: int = 16, causal: bool = False,
                      cache_dir: Path = DEFAULT_STEM_CACHE,
                      device: str = "mps", sr: int = 22050, return_source: bool = False):
    """Returns a dict of the 4 raw components (each ``(n_bars,)`` in [0,1]) plus
    ``combined_mean`` and ``combined_max``. See module docstring for the output
    convention (score[b] predicts a boundary at b+1).

    ``causal``: use a BACKWARD-ONLY local reference window (bars
    ``[b-radius, b)``) instead of the symmetric ``[b-radius, b+radius]``
    window. See ``_local_median``'s docstring -- a symmetric window straddles
    a real boundary and blends two different patterns into one "local
    normal", which floods the whole neighbourhood (both sides) with false
    deviation. Default False for backward compatibility; the calibration in
    docs/research_sessions/rhythm_ssm_2026-07-30.md found causal=True clearly
    better and it should be treated as the real default going forward.
    """
    import librosa
    audio_path = Path(audio_path)
    drum_wav = separate_drums(audio_path, cache_dir=cache_dir, device=device)
    source = "demucs"
    if drum_wav is not None:
        y, srr = librosa.load(str(drum_wav), sr=sr, mono=True)
    else:
        y, srr = _hpss_percussive(audio_path, sr=sr)
        source = "hpss_fallback"

    env, times = _drum_onset_bands(y, srr)  # (3, n_frames): low/mid/high
    n_bars = len(bar_bounds_sec) - 1

    # 1. onset-density excess (all bands combined)
    dens = _bar_sums(env, times, bar_bounds_sec)
    c1 = _robust_excess(dens, radius, causal=causal)

    # 2. groove-template deviation vs the LOCAL median patch
    P = _bar_patches(env, times, bar_bounds_sec, sub_steps=sub_steps)
    c2 = np.zeros(n_bars)
    for b in range(n_bars):
        if causal:
            lo, hi = max(0, b - radius), b
        else:
            lo, hi = max(0, b - radius), min(n_bars, b + radius + 1)
        idx = [j for j in range(lo, hi) if j != b]
        if not idx:
            continue
        med_patch = np.median(P[idx], axis=0)
        nrm = float(np.linalg.norm(med_patch))
        if nrm < 1e-9:
            continue
        cos = float(np.dot(P[b], med_patch) / nrm)  # P[b] already unit norm
        c2[b] = float(np.clip(1.0 - cos, 0.0, 1.0))

    # 3. crash spike at the NEXT bar's downbeat -> fill-evidence for THIS bar
    crash_own = _downbeat_peak(env[2], times, bar_bounds_sec)  # high band
    crash_next = np.zeros(n_bars)
    crash_next[:-1] = crash_own[1:]
    c3 = _robust_excess(crash_next, radius, causal=causal)

    # 4. snare/tom flurry: mid-band peak COUNT excess
    counts = _bar_peak_counts(env[1], times, bar_bounds_sec)
    c4 = _robust_excess(counts, radius, causal=causal)

    for c in (c1, c2, c3, c4):
        if n_bars > 0:
            c[-1] = 0.0  # no b+1 to predict

    combined_mean = np.clip((c1 + c2 + c3 + c4) / 4.0, 0.0, 1.0)
    combined_max = np.clip(np.max(np.stack([c1, c2, c3, c4]), axis=0), 0.0, 1.0)

    out = {"density": c1, "groove_dev": c2, "crash": c3, "flurry": c4,
           "combined_mean": combined_mean, "combined_max": combined_max}
    if return_source:
        return out, source
    return out


def fill_scores(audio_path, bar_bounds_sec: list[float], *, variant: str = "groove_dev",
                 causal: bool = True, **kw) -> np.ndarray:
    """One fill/novelty score per bar in [0,1]. High score[b] = bar b looks
    like a fill (or is followed by an outlier crash) -> predicts a section
    boundary AT bar b+1.

    ``variant``: which of ``component_scores()``'s 6 keys to return. Default
    ``"groove_dev"`` (mean-centred local-neighbourhood cosine deviation,
    causal window) -- the ONLY component with a positive same/different median
    gap on all 3 calibration songs (This Love +0.07, Every Breath You Take
    +0.09, Billie Jean +0.08; AP 1.7-4.3x chance). ``combined_mean`` /
    ``combined_max`` are available but did NOT beat groove_dev alone on any
    of the 3 songs -- "do not just ship a blend" turned out to be the right
    call here. ``causal=True`` (backward-only local reference) is required
    for groove_dev to work at all -- the symmetric-window version had a
    NEGATIVE gap on Every Breath You Take (see component_scores docstring).
    Full numbers: docs/research_sessions/rhythm_ssm_2026-07-30.md ("Fill
    detection").
    """
    comps = component_scores(audio_path, bar_bounds_sec, causal=causal, **kw)
    return comps[variant]


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    from pattern_slide import rigid_slots
    from section_merge_declined import _load_payload
    from harmonia.models.rigid_grid import rigid_grid_for
    from harmonia.serving.config import AUDIO_DIR

    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    tonic = int((P.get("home") or {}).get("tonic", 0)) % 12
    grid = rigid_grid_for(P.get("chords", []), tonic_pc=tonic)
    _seq, _names, n_bars, bar_sec = rigid_slots(P)
    assert len(grid) - 1 == n_bars

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    comps, source = component_scores(audio_path, grid, return_source=True)
    print(f"=== {slug} === n_bars={n_bars} bar_sec={bar_sec:.3f} drum source={source}")
    for name, arr in comps.items():
        top = np.argsort(-arr)[:8]
        print(f"  {name:14s} top bars: " + ", ".join(f"{b}:{arr[b]:.2f}" for b in top))


if __name__ == "__main__":
    main()
