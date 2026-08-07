#!/usr/bin/env python3
"""rhythm_ssm.py — a rhythmic-pattern SSM from a separated DRUMS stem, 2026-07-30.

Motivating idea (Louis): the chord SSM (``harmonia.models.section_structure.
build_chord_ssm`` / ``pattern_slide.chord_ssm``) segments sections from harmony,
but harmony is ambiguous — two different sections can share chords (This Love's
chorus and bridge both sit on Fm/Eb). A verse groove and a chorus groove differ
even when the chords match, and a fill/crash marks a boundary independent of
harmony. This module builds a SECOND self-similarity matrix, from the drum
pattern, on the SAME half-bar slot grid ``pattern_slide.rigid_slots`` produces,
so the two matrices are element-wise comparable — a second, near-independent
vote on where the sections are.

Pipeline:
  1. Separate drums with demucs (htdemucs, two-stems). Falls back to
     ``librosa.decompose.hpss`` (percussive component) if demucs is unavailable
     — WEAKER, and the fallback path prints a warning saying so.
  2. Featurise each half-bar slot as a small 2-D patch: a 3-band onset-strength
     envelope (kick/snare/hats-ish mel-band split, via
     ``librosa.onset.onset_strength(..., channels=...)``) sampled at
     ``sub_steps`` fixed sub-slot positions. Patch = (3, sub_steps), flattened
     and L2-normalised (onset strength is >=0, so cosine similarity of these
     vectors is already a true [0,1] scale, same convention as chord_ssm).
  3. Compare slots: cosine on the raw patch is the baseline. Drum hits jitter
     around the grid, so ALSO compute a tolerance-shifted version (resample the
     patch at a few sub-slot time offsets, take the max cosine over the shift
     grid) and a Gaussian-smoothed version. All three are exposed; calibration
     below reports which wins.

Run:  .venv/bin/python scratchpad/rhythm_ssm.py [song-substring]
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

# Big binary separation artifacts do NOT belong in the tracked repo scratchpad/
# dir. They used to live in a SESSION tmp scratchpad, whose path was hardcoded:
# 3.2 GB of demucs stems sitting in /private/tmp under a session id that died
# months ago. macOS sweeps /private/tmp, and `HARMONIA_SECTIONS=voice` now needs
# these stems on every analysis — one sweep and every song silently repays a
# minute of source separation. Moved 2026-08-07 to data/cache/, next to the
# PitchExtractor npz cache, which is a symlink into ~/harmonia and survives.
# Override with HARMONIA_STEM_CACHE.
import os
DEFAULT_STEM_CACHE = Path(os.environ.get(
    "HARMONIA_STEM_CACHE",
    str(Path(__file__).resolve().parent.parent / "data" / "cache" / "stems"),
))

__all__ = ["rhythm_ssm", "separate_drums", "build_slot_patches", "slot_bounds_from_bars"]


# ── step 1: drum separation ────────────────────────────────────────────────

def separate_drums(audio_path: Path, cache_dir: Path = DEFAULT_STEM_CACHE,
                    device: str = "mps") -> Path | None:
    """Run demucs (htdemucs, two-stems=drums) if not already cached. Returns the
    path to ``drums.wav``, or ``None`` if demucs is unavailable/fails (caller
    should fall back to HPSS and say so — do not silently substitute)."""
    audio_path = Path(audio_path)
    stem = audio_path.stem
    out = cache_dir / "htdemucs" / stem / "drums.wav"
    if out.exists():
        return out
    try:
        import demucs  # noqa: F401
    except ImportError:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "--two-stems=drums", "-n", "htdemucs",
           "-d", device, "-o", str(cache_dir), str(audio_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        warnings.warn(f"demucs subprocess failed: {e}")
        return None
    if r.returncode != 0 or not out.exists():
        warnings.warn(f"demucs did not produce {out}: rc={r.returncode}\n{r.stderr[-2000:]}")
        return None
    return out


def _hpss_percussive(audio_path: Path, sr: int = 22050):
    """Fallback when demucs is unavailable: librosa HPSS percussive component.
    WEAKER than a true drum stem -- bleeds bass/vocal transients, no isolation
    of melodic percussion from a full mix. Caller must report this was used."""
    import librosa
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    _, y_perc = librosa.effects.hpss(y)
    return y_perc, sr


# ── step 2: band-split onset envelope + per-slot patches ──────────────────

def _drum_onset_bands(drum_y: np.ndarray, sr: int, n_mels: int = 40,
                       band_edges=(0, 13, 26, 40)):
    """3-band onset-strength envelope (~kick / snare-mid / hats-high), via
    librosa's own mel-channel split. Returns (env (n_bands, n_frames), times)."""
    import librosa
    env = librosa.onset.onset_strength_multi(
        y=drum_y, sr=sr, n_mels=n_mels, channels=list(band_edges),
        aggregate=np.median,
    )  # (n_bands, n_frames)
    hop_length = 512
    times = librosa.frames_to_time(np.arange(env.shape[1]), sr=sr, hop_length=hop_length)
    # per-band max-normalise so kick energy doesn't drown out hats
    env = env / np.clip(env.max(axis=1, keepdims=True), 1e-9, None)
    return env.astype(np.float32), times


def slot_bounds_from_bars(bar_bounds_sec: list[float], slots_per_bar: int = 2) -> list[float]:
    """Half-bar (or ``slots_per_bar``-way) slot edges from bar edges, by even
    subdivision of each bar span -- the same convention ``pattern_slide.
    rigid_slots`` uses implicitly (beat/(bpb/slots_per_bar))."""
    edges = []
    for i in range(len(bar_bounds_sec) - 1):
        a, b = bar_bounds_sec[i], bar_bounds_sec[i + 1]
        for s in range(slots_per_bar):
            edges.append(a + (b - a) * s / slots_per_bar)
    edges.append(bar_bounds_sec[-1])
    return edges


def _interp_patch(env: np.ndarray, times: np.ndarray, t0: float, t1: float,
                   sub_steps: int, shift_frac: float = 0.0) -> np.ndarray:
    """Sample the (n_bands, n_frames) envelope at ``sub_steps`` evenly spaced
    points within ``[t0, t1)``, optionally shifted by ``shift_frac`` of the slot
    duration (for tolerance testing). Returns (n_bands, sub_steps)."""
    dur = t1 - t0
    ts = t0 + shift_frac * dur + (np.arange(sub_steps) + 0.5) / sub_steps * dur
    out = np.stack([np.interp(ts, times, env[b], left=0.0, right=0.0)
                     for b in range(env.shape[0])])
    return out.astype(np.float32)


def build_slot_patches(env: np.ndarray, times: np.ndarray, slot_edges: list[float],
                        sub_steps: int = 8, shifts=(0.0,)) -> np.ndarray:
    """Returns ``(n_slots, n_shifts, n_bands*sub_steps)`` L2-normalised patch
    vectors, one per slot per tolerance-shift."""
    n_slots = len(slot_edges) - 1
    n_bands = env.shape[0]
    dim = n_bands * sub_steps
    V = np.zeros((n_slots, len(shifts), dim), dtype=np.float32)
    for i in range(n_slots):
        t0, t1 = slot_edges[i], slot_edges[i + 1]
        for si, sh in enumerate(shifts):
            patch = _interp_patch(env, times, t0, t1, sub_steps, shift_frac=sh)
            v = patch.reshape(-1)
            n = np.linalg.norm(v)
            V[i, si] = v / n if n > 1e-9 else v
    return V


def _gaussian_smooth_env(env: np.ndarray, times: np.ndarray, sigma_sec: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    if len(times) < 2:
        return env
    dt = float(np.median(np.diff(times)))
    sigma_frames = max(sigma_sec / dt, 1e-6)
    return gaussian_filter1d(env, sigma=sigma_frames, axis=1, mode="nearest")


# ── step 3: slot-vs-slot similarity ────────────────────────────────────────

def _cosine_gram(V: np.ndarray) -> np.ndarray:
    """(n_slots, n_shifts, dim) -> (n_slots, n_slots) with a max over BOTH
    slots' shift grids (symmetric tolerance)."""
    n, k, d = V.shape
    flat = V.reshape(n * k, d)
    G = np.clip(flat @ flat.T, 0.0, 1.0).astype(np.float32)   # (n*k, n*k)
    G = G.reshape(n, k, n, k)
    return G.max(axis=(1, 3)).astype(np.float32)


def _mean_center_renormalise(V: np.ndarray) -> np.ndarray:
    """Subtract the SONG-AVERAGE patch (the shared "always kick on 1&3, hats on
    every 8th" backbone that is common to almost every slot in a steady 4/4
    groove) before re-normalising -- i.e. Pearson correlation instead of raw
    cosine on non-negative onset vectors.

    Hypothesis (from docs/known_issues.md's chroma-DTW finding: "raw cosine
    sits at a ~0.5 DC floor for ANY alignment... mean-centring is the biggest
    single win"): raw cosine on onset-strength patches is dominated by the
    shared always-on backbone, inflating similarity EVERYWHERE and erasing the
    very deviations (fills, syncopation swaps) that mark a section change.
    Mean-centring removes the shared DC term so the SSM measures agreement on
    the DEVIATIONS from the average groove, not the average groove itself.
    Cosine on a mean-centred, arbitrary-sign vector is genuinely in [-1,1], so
    the caller must clip/rescale to keep the chord_ssm [0,1] convention.
    """
    n, k, d = V.shape
    mean_patch = V.reshape(n * k, d).mean(axis=0, keepdims=True)  # (1, d)
    Vc = V - mean_patch[None, :, :]
    norm = np.linalg.norm(Vc, axis=2, keepdims=True)
    return Vc / np.clip(norm, 1e-9, None)


def _pearson_gram(V: np.ndarray) -> np.ndarray:
    """Like ``_cosine_gram`` but on mean-centred vectors, then rescaled from
    [-1,1] to a true [0,1] scale ((r+1)/2) -- same convention chord_ssm uses."""
    Vc = _mean_center_renormalise(V)
    n, k, d = Vc.shape
    flat = Vc.reshape(n * k, d)
    G = np.clip(flat @ flat.T, -1.0, 1.0)
    G = (G + 1.0) / 2.0
    G = G.reshape(n, k, n, k)
    return G.max(axis=(1, 3)).astype(np.float32)


def rhythm_ssm(audio_path, bar_bounds_sec: list[float], slots_per_bar: int = 2,
               *, sub_steps: int = 8, variant: str = "smooth",
               tol_fracs=(-0.15, -0.075, 0.0, 0.075, 0.15),
               smooth_sigma_sec: float = 0.05,
               cache_dir: Path = DEFAULT_STEM_CACHE, device: str = "mps",
               sr: int = 22050, return_debug: bool = False):
    """(n_slots, n_slots) float32 in [0,1] -- how similar the DRUM pattern in
    each half-bar slot is to every other, on the SAME slot grid
    ``pattern_slide.rigid_slots`` produces (n_slots = (len(bar_bounds_sec)-1) *
    slots_per_bar).

    ``variant``: "baseline" (plain cosine, no jitter tolerance), "tolerance"
    (max cosine over a small time-shift grid -- handles drum-hit jitter around
    the grid), "smooth" (Gaussian-smoothed envelope, plain cosine -- the
    default; best or tied-best same/diff median-gap on 2/3 calibration songs,
    see docs/research_sessions/rhythm_ssm_2026-07-30.md), "pearson" (mean-
    centred/Pearson cosine, no tolerance), or "pearson_tolerance" (mean-centred
    + tolerance-shifted). NO variant dominates on all songs tested -- ranking
    is unstable across only 3 songs, treat "smooth" as a weak default, not a
    validated winner. ``return_debug=True`` returns all variants plus the
    drum-separation source used.
    """
    audio_path = Path(audio_path)
    drum_wav = separate_drums(audio_path, cache_dir=cache_dir, device=device)
    source = "demucs"
    if drum_wav is not None:
        import librosa
        drum_y, srr = librosa.load(str(drum_wav), sr=sr, mono=True)
    else:
        warnings.warn("demucs unavailable/failed -- falling back to librosa HPSS "
                       "percussive component (WEAKER: bleeds non-drum transients)")
        drum_y, srr = _hpss_percussive(audio_path, sr=sr)
        source = "hpss_fallback"

    env, times = _drum_onset_bands(drum_y, srr)
    slot_edges = slot_bounds_from_bars(bar_bounds_sec, slots_per_bar)

    V_base = build_slot_patches(env, times, slot_edges, sub_steps=sub_steps, shifts=(0.0,))
    S_baseline = _cosine_gram(V_base)

    V_tol = build_slot_patches(env, times, slot_edges, sub_steps=sub_steps, shifts=tol_fracs)
    S_tolerance = _cosine_gram(V_tol)

    env_smooth = _gaussian_smooth_env(env, times, smooth_sigma_sec)
    V_smooth = build_slot_patches(env_smooth, times, slot_edges, sub_steps=sub_steps, shifts=(0.0,))
    S_smooth = _cosine_gram(V_smooth)

    # mean-centred (Pearson) variants -- see _mean_center_renormalise docstring
    S_pearson = _pearson_gram(V_base)
    S_pearson_tol = _pearson_gram(V_tol)

    result = {"baseline": S_baseline, "tolerance": S_tolerance, "smooth": S_smooth,
              "pearson": S_pearson, "pearson_tolerance": S_pearson_tol}
    if return_debug:
        return result, source
    return result[variant]


# ── demo / smoke test ──────────────────────────────────────────────────────

def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "this_love"
    from pattern_slide import rigid_slots, chord_ssm
    from section_merge_declined import _load_payload
    from harmonia.models.rigid_grid import rigid_grid_for
    from harmonia.serving.config import AUDIO_DIR

    f = next(Path(REPO / "docs" / "plots").glob(f"inferred_*{sub}*.html"))
    slug = f.stem.replace("inferred_", "")
    P = _load_payload(f)
    tonic = int((P.get("home") or {}).get("tonic", 0)) % 12
    grid = rigid_grid_for(P.get("chords", []), tonic_pc=tonic)
    seq, names, n_bars, bar_sec = rigid_slots(P)
    assert len(grid) - 1 == n_bars, (len(grid) - 1, n_bars)
    S_chord = chord_ssm(names)

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    print(f"=== {slug} === n_bars={n_bars} bar_sec={bar_sec:.3f} audio={audio_path}")
    variants, source = rhythm_ssm(audio_path, grid, return_debug=True)
    print(f"drum source: {source}")
    for name, S in variants.items():
        print(f"  {name}: shape={S.shape} mean={S.mean():.3f} "
              f"offdiag_mean={(S.sum() - np.trace(S)) / (S.size - S.shape[0]):.3f}")
    return S_chord, variants, names, slug, bar_sec, grid


if __name__ == "__main__":
    main()
