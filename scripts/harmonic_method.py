"""THE METHOD — locked 2026-08-05 by Louis. One source, no drift.

« Lock et push la méthode actuelle car je pense qu'elle est parfaite quand la
matrice SSM est bonne. »

This module is the single definition of the repetition-detection method that
works. Everything else — report pages, future experiments — must import from
here rather than re-implement, because the method was previously spread across
five scripts and had already drifted twice (a margin using MAD instead of σ, a
peak floor taken against the curve maximum instead of the motif's own value).

    audio
      → bar grid (Beat This! downbeats, guarded)
      → musx chord posteriors, averaged per bar
      → projected onto the 12 PITCH CLASSES through the chord-tone matrix
         (so the dot product IS harmonic overlap: B♭ major {D F B♭} and G minor
          {D G B♭} share two notes, A♭ major {C E♭ A♭} shares none)
      → SSM = cosine between those 12-d bar vectors
      → dominant lag L = the period; first strong bar b0 = the phase
      → slide the motif along X, DIAGONAL reading (bar-to-bar); the square
        reading is deprecated and raises
      → keep a peak iff  local-baseline margin  AND  ≥ 90 % of the initial peak

Validated by Louis on the plots, song by song: perfect on This Love and Don't
Know Why. It fails on Sunny, Beat It and Close to You — and that is a SSM
problem upstream, not a method problem. See `docs/known_issues.md`,
"THE HARMONIC METHOD IS LOCKED".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
from harmonia_min import musx as mx, sections as hs      # noqa: E402

# ── the constants, all of them ──────────────────────────────────────────────
FRAME_DT = 23.22e-3      # musx posterior frame step
LAG_MIN, LAG_MAX = 2, 16 # candidate periods, in bars
PHASE_QUANTILE = 0.90    # a bar is "strong" at lag L when it ranks here
MARGIN_SIGMA = 0.5       # a peak must clear the local median by this × σ
MARGIN_WIN = 3           # the local window is ± MARGIN_WIN × L bars
INITIAL_PEAK_FRAC = 0.90 # …AND reach this fraction of the initial peak.
                         # Chosen by Louis on the 70/75/80/85/90/95 sweep,
                         # by eye, over five songs — the corpus metric cannot
                         # separate these thresholds at all.

# musx triad plane: col 0 = N; col i≥1 is root (i−1)%12, type (i−1)//12+1 in
# {maj, min, sus4, sus2, dim, aug}. Chord tones as semitone offsets:
TRIAD_TONES = {1: (0, 4, 7), 2: (0, 3, 7), 3: (0, 5, 7),
               4: (0, 2, 7), 5: (0, 3, 6), 6: (0, 4, 8)}


def chord_tone_matrix(n_cols: int) -> np.ndarray:
    """(n_cols, 12) — each chord class as its pitch-class indicator."""
    M = np.zeros((n_cols, 12))
    for i in range(1, n_cols):
        root, typ = (i - 1) % 12, (i - 1) // 12 + 1
        for s in TRIAD_TONES.get(typ, (0, 4, 7)):
            M[i, (root + s) % 12] = 1.0
    return M


def _unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def harmonic_vectors(audio_path, grid) -> np.ndarray:
    """(n_bars, 12) — the chord posterior of each bar, as pitch classes."""
    triad = mx.frame_posteriors(audio_path)[0]
    M = chord_tone_matrix(triad.shape[1])
    n = len(grid) - 1
    P = []
    for b in range(n):
        a = int(grid[b] / FRAME_DT)
        z = max(a + 1, int(grid[b + 1] / FRAME_DT))
        seg = triad[a:min(z, len(triad))]
        P.append(seg.mean(0) if len(seg) else np.zeros(triad.shape[1]))
    return _unit(np.array(P) @ M)


def ssm(audio_path, grid) -> np.ndarray:
    V = harmonic_vectors(audio_path, grid)
    return V @ V.T


def period_and_phase(S: np.ndarray) -> tuple[int, int]:
    """(L, b0) — the dominant repetition period, and the first bar where it is
    strong, which is the phase the motif is read from."""
    n = len(S)
    best, L = -1.0, LAG_MIN
    for Lx in range(LAG_MIN, min(LAG_MAX, n - 2) + 1):
        m = float(np.mean([S[b, b + Lx] for b in range(n - Lx)]))
        if m > best:
            best, L = m, Lx
    i = np.arange(n)
    off = S[np.abs(i[:, None] - i[None, :]) >= LAG_MIN]
    thr = float(np.quantile(off, PHASE_QUANTILE))
    b0 = next((b for b in range(n - L) if S[b, b + L] >= thr), 0)
    return L, b0


def slide(S: np.ndarray, L: int, b0: int) -> np.ndarray:
    """DIAGONAL reading: f(c) = mean over i of S[b0+i, c+i].

    "Is the material at c the same music as the motif, bar for bar."

    This replaced the sliding SQUARE everywhere on 2026-08-05 (Louis: « la
    diagonale est clairement mieux que le carré, on switch pour celui-là
    partout »). Head-to-head on the same blocks with the same peak rule
    (/reports/diag_vs_square.html):

        This Love, 4-bar cycle    square 7 occurrences   diagonal 7   identical
        This Love, 8-bar chorus   square 12              diagonal 4
        Don't Know Why, 4-bar     square 10              diagonal 10  identical
        Don't Know Why, 8-bar B   square 2               diagonal 2   identical

    The square's 12 on the chorus are 14, 16, 34, 36, 56, 58, 60, 62, 64, 66,
    68, 70 — overlapping junk every two bars, impossible for an 8-bar motif.
    The diagonal returns 16, 36, 56, 64. Same signal, cleaner reading; the two
    only diverge on long motifs, where the square's off-diagonal cells swamp it.
    """
    n = len(S)
    return np.array([diag_match(S, b0, c, L) for c in range(0, n - L + 1)])


def square_slide(S: np.ndarray, L: int, b0: int, *,
                 i_understand_this_is_deprecated: bool = False) -> np.ndarray:
    """DEPRECATED 2026-08-05 — DO NOT USE. Kept only so the historical
    comparison page can still draw it.

    The sliding square. On long motifs it returns overlapping occurrences that
    cannot exist (an 8-bar motif recurring every 2 bars), because its
    off-diagonal cells dominate the product. Use `slide()`.
    """
    if not i_understand_this_is_deprecated:
        raise RuntimeError(
            "square_slide() is DEPRECATED and must not be reconnected. It "
            "returns impossible overlapping occurrences on long motifs (This "
            "Love's 8-bar chorus: 12 hits, four of them 2 bars apart) — see "
            "/reports/diag_vs_square.html and docs/known_issues.md. Use "
            "harmonic_method.slide(), which is the DIAGONAL reading. If you "
            "genuinely need the old behaviour for a historical figure, pass "
            "i_understand_this_is_deprecated=True and say so in your report.")
    n = len(S)
    P = S[b0:b0 + L, b0:b0 + L]
    return np.array([float((P * S[b0:b0 + L, c:c + L]).sum()) / (L * L)
                     for c in range(0, n - L + 1)])


def peaks(curve: np.ndarray, L: int, b0: int) -> np.ndarray:
    """THE RULE: local-baseline margin AND ≥ INITIAL_PEAK_FRAC of the initial
    peak — the curve where the motif sits on itself."""
    ref = float(curve[b0])
    win = max(4, MARGIN_WIN * L)
    cand, _ = find_peaks(curve)
    sd = float(np.std(curve))
    out = []
    for k in cand:
        a, z = max(0, k - win), min(len(curve), k + win + 1)
        if (curve[k] - float(np.median(curve[a:z]))) < MARGIN_SIGMA * sd:
            continue
        if curve[k] < INITIAL_PEAK_FRAC * ref:
            continue
        out.append(int(k))
    return np.array(out, int)


def diag_match(S, a: int, b: int, L: int) -> float:
    """Bar-to-bar similarity of two blocks, aligned: mean of S[a+i, b+i].

    This is NOT the sliding square. Measured on Don't Know Why's B section
    (2026-08-05): bars 22–29 and 38–45 are the same music, and their bar-to-bar
    diagonal is 0.984 — while the full 8×8 block average is 0.529. The gap is
    not noise: B is a THROUGH-COMPOSED 8 bars (Gm7 | C7 | F7 | Dm7 …), so its
    own internal block averages 0.533 — every bar differs from every other. A
    sliding square therefore scores a perfect match of a non-repetitive block
    barely above chance, and the dictionary misses it. The diagonal finds it
    instantly.

    Rule of thumb this establishes: the SQUARE finds material that repeats
    INSIDE itself (loops, cells); the DIAGONAL finds material that repeats
    ELSEWHERE regardless of its internal structure. A dictionary needs both.
    """
    L = min(L, S.shape[0] - max(a, b))
    if L <= 0:
        return 0.0
    return float(np.mean([S[a + i, b + i] for i in range(L)]))


def run(audio_path, grid) -> dict:
    """The whole method on one song. Returns everything a report needs."""
    S = ssm(audio_path, grid)
    L, b0 = period_and_phase(S)
    curve = slide(S, L, b0)
    return {"S": S, "L": L, "b0": b0, "curve": curve,
            "peaks": peaks(curve, L, b0)}


if __name__ == "__main__":
    import copy
    from harmonia_min import pipeline as _pl
    for stem in ("maroon_5_this_love", "norah_jones_don_t_know_why",
                 "bobby_hebb_sunny_official_audio"):
        real = hs.detect_sections
        cap = {}

        def spy(grid, arr, times, bars=None):
            out = real(grid, arr, times, bars)
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
