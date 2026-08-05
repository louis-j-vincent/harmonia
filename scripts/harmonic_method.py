"""THE METHOD — now a thin shim over the PRODUCTION module.

The implementation moved to `harmonia_min/harmonic_sections.py` on 2026-08-05,
when Louis put it in production (« push en prod et remplacer la pipeline
actuelle par cette version améliorée »). Nothing is defined twice: this file
only adds the audio-path convenience the report pages use — `ssm(audio, grid)`
rather than the production `ssm(triad, grid)` — and re-exports the rest by name
so every existing `import harmonic_method as HM` keeps working untouched.

Read `harmonia_min/harmonic_sections.py` for the method itself, its constants,
the measured head-to-head comparisons, and what it does NOT solve.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
from harmonia_min import musx as mx, sections as hs             # noqa: E402
from harmonia_min.harmonic_sections import (                    # noqa: E402,F401
    FRAME_DT, LAG_MIN, LAG_MAX, PHASE_QUANTILE, CONT_QUANTILE,
    MARGIN_SIGMA, MARGIN_WIN, INITIAL_PEAK_FRAC, TRIAD_TONES,
    chord_tone_matrix, off_diagonal, period_and_phase, slide, square_slide,
    peaks, diag_match,
    harmonic_vectors as _vectors_from_triad,
    ssm as _ssm_from_triad,
)


def harmonic_vectors(audio_path, grid):
    """(n_bars, 12) — the production function, fed from an audio path."""
    return _vectors_from_triad(mx.frame_posteriors(audio_path)[0], grid)


def ssm(audio_path, grid):
    """The harmonic SSM from an audio path (musx posteriors are cached)."""
    return _ssm_from_triad(mx.frame_posteriors(audio_path)[0], grid)


def run(audio_path, grid) -> dict:
    """The whole method on one song. Returns everything a report needs."""
    S = ssm(audio_path, grid)
    L, b0 = period_and_phase(S)
    curve = slide(S, L, b0)
    return {"S": S, "L": L, "b0": b0, "curve": curve,
            "peaks": peaks(curve, L, b0)}


if __name__ == "__main__":
    from harmonia_min import pipeline as _pl
    for stem in ("maroon_5_this_love", "norah_jones_don_t_know_why",
                 "bobby_hebb_sunny_official_audio"):
        real = hs.detect_sections
        cap = {}

        def spy(grid, arr, times, bars=None, **_kw):
            out = real(grid, arr, times, bars, **_kw)
            cap.update(grid=grid)
            return out

        hs.detect_sections = spy
        try:
            _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x",
                        file_key="x", audio_url="")
        finally:
            hs.detect_sections = real
        r = run(HERE / f"docs/audio/{stem}.m4a", cap["grid"])
        print(f"{stem[:34]:<36} période {r['L']:>2}  phase {r['b0']:>3}  "
              f"{len(r['peaks'])} pics aux mesures {list(r['peaks'])}")
