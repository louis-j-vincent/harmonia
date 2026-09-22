"""Pure-numpy port of musx's CQT front-end (`extractors/cqt.py:CQTV2`).

WHY THIS EXISTS: `third_party/musx_ismir2019/extractors/cqt.py:CQTV2.extract`
calls `librosa.core.hybrid_cqt(y, bins_per_octave=36, fmin=note_to_hz('F#0'),
n_bins=288, tuning=None, hop_length=512)` -- librosa is not installable on
Android (Chaquopy's wheel index has no `librosa`/`vamp`/`resampy`, see
`docs/portage_mobile/p0_t3_deps_natives.md`), so this reproduces the whole
chain (STFT, CQ wavelet filter bank, pseudo-CQT, recursive octave-by-octave
downsampling, tuning estimation) with numpy only -- no scipy, no numba.

Ported (not adapted) from librosa 0.11.0's `librosa/core/constantq.py`,
`librosa/filters.py`, `librosa/core/pitch.py`, reading the exact source
installed in this repo's `.venv`, with these choices FIXED (matching what
`CQTV2` actually calls, verified against the real clone -- not librosa's
defaults):
  - gamma=0 everywhere (never the ERB path) -- `cqt()` passes `gamma=0`
    explicitly to `vqt()`, and `pseudo_cqt`'s/`hybrid_cqt`'s own
    `wavelet_lengths` calls never pass `gamma` either (default is 0).
  - tuning=None -> IS estimated (`librosa.estimate_tuning`, a piptrack +
    histogram-mode procedure) -- CQTV2 passes `tuning=None`, not librosa's
    own default of `0.0`. Measured on 3 songs (2026-09-23): 0.10-0.44 bins
    (of 1/36 octave each), non-negligible.
  - filter_scale=1, norm=1 (L1), sparsity=0.01, window='hann', scale=True,
    pad_mode='constant', res_type='soxr_hq' (librosa's default -- CQTV2
    never overrides it).
  - No early downsampling occurs for these parameters (verified:
    filter_cutoff ~5.9 kHz vs Nyquist 11025 Hz -> `__early_downsample_count`
    returns 0), so the recursive loop always starts at the original
    22050 Hz.

THE ONE PIECE NOT REPRODUCED EXACTLY: librosa's `res_type='soxr_hq'` calls
the compiled `soxr` C library (a variable-rate resampler) for the ~7 cascaded
2x octave downsamplings that the RECURSIVE branch needs (238 of 288 bins --
everything except the top ~1.4 octaves, which use the resample-free
pseudo-CQT branch, see `hybrid_cqt`). `soxr`'s exact filter taps are not
public/portable. This module substitutes a linear-phase Kaiser-windowed-sinc
half-band FIR decimator (`_HALFBAND_TAPS` below), which is a legitimate
bandlimited resampler but not bit-identical to soxr. See
`docs/portage_mobile/p2b_cqt_musx.md` for the measured parity impact
(per-octave, and on the downstream chord argmax -- the number that matters).

Entry point: `cqt_musx(y, sr=22050)` -> `(n_frames, 252)` float32, exactly
the tensor `chordnet_ismir_naive.ChordNet.inference` slices out of the full
288-bin CQT (columns 18:270, see `SLICE_LO`/`SLICE_HI` below) before feeding
its CNN. `y` MUST already be mono float32 at `MUSX_SR` (22050 Hz) -- this
module does no decoding/resampling of the *input* audio, only the internal
per-octave ones the CQT algorithm itself requires.
"""
from __future__ import annotations

import numpy as np

MUSX_SR = 22050
MUSX_HOP = 512
FMIN_F_SHARP_0 = 23.12465141947715  # librosa.note_to_hz('F#0'), float64
BINS_PER_OCTAVE = 36
N_BINS = 288
FILTER_SCALE = 1.0
NORM = 1.0
SPARSITY = 0.01

# ChordNet.inference() slice (chordnet_ismir_naive.py:180-181): SHIFT_HIGH=6,
# SHIFT_STEP=3 -> the "no augmentation shift" 252-wide window out of 288.
SLICE_LO = 18
SLICE_HI = 270


# ── windows, no scipy ────────────────────────────────────────────────────

def _hann_periodic(n: int) -> np.ndarray:
    """scipy.signal.get_window('hann', n, fftbins=True) -- periodic Hann."""
    if n == 1:
        return np.ones(1, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)


def _window_bandwidth_hann(n: int = 1000) -> float:
    """librosa.filters.window_bandwidth('hann') -- ENBW in FFT bins."""
    w = _hann_periodic(n)
    tiny = np.finfo(w.dtype).tiny
    return float(n * np.sum(w ** 2) / (np.sum(w) ** 2 + tiny))


_ENBW_HANN = _window_bandwidth_hann()


# ── CQ filter-bank math (librosa/filters.py, gamma=0 fixed) ────────────

def _relative_bandwidth(freqs: np.ndarray) -> np.ndarray:
    bpo = np.empty_like(freqs)
    logf = np.log2(freqs)
    bpo[0] = 1.0 / (logf[1] - logf[0])
    bpo[-1] = 1.0 / (logf[-1] - logf[-2])
    bpo[1:-1] = 2.0 / (logf[2:] - logf[:-2])
    alpha = (2.0 ** (2.0 / bpo) - 1.0) / (2.0 ** (2.0 / bpo) + 1.0)
    return alpha


def _wavelet_lengths(freqs: np.ndarray, sr: float, filter_scale: float = FILTER_SCALE):
    alpha = _relative_bandwidth(freqs)
    Q = filter_scale / alpha
    f_cutoff = float(np.max(freqs * (1.0 + 0.5 * _ENBW_HANN / Q)))
    lengths = Q * sr / freqs
    return lengths, f_cutoff, alpha


def _pad_center(x: np.ndarray, size: int) -> np.ndarray:
    n = len(x)
    total = size - n
    left = total // 2
    right = total - left
    return np.pad(x, (left, right), mode="constant")


def _wavelet_basis(freqs: np.ndarray, sr: float, filter_scale: float = FILTER_SCALE):
    lengths, _, _ = _wavelet_lengths(freqs, sr, filter_scale)
    filt_list = []
    for ilen, freq in zip(lengths, freqs):
        n = np.arange(-ilen // 2, ilen // 2, dtype=np.float64)
        sig = np.exp(1j * (n * 2 * np.pi * freq / sr))
        sig = sig * _hann_periodic(len(n))
        sig = sig / np.sum(np.abs(sig))  # util.normalize(norm=1)
        filt_list.append(sig)
    max_len = int(2.0 ** np.ceil(np.log2(max(lengths))))
    basis = np.stack([_pad_center(f, max_len) for f in filt_list]).astype(np.complex128)
    return basis, lengths


def _sparsify_rows(x: np.ndarray, quantile: float = SPARSITY) -> np.ndarray:
    if quantile <= 0:
        return x
    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mag_sort = np.sort(mags, axis=1)
    cumulative = np.cumsum(mag_sort / norms, axis=1)
    threshold_idx = np.argmin(cumulative < quantile, axis=1)
    out = np.zeros_like(x)
    for i, j in enumerate(threshold_idx):
        thresh = mag_sort[i, j]
        keep = mags[i] >= thresh
        out[i, keep] = x[i, keep]
    return out


def _vqt_filter_fft(sr: float, freqs: np.ndarray, hop_length: int | None,
                     filter_scale: float = FILTER_SCALE):
    basis, lengths = _wavelet_basis(freqs, sr, filter_scale)
    n_fft = basis.shape[1]
    if hop_length is not None:
        floor_n = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))
        if n_fft < floor_n:
            n_fft = floor_n
    basis = basis * (lengths[:, None] / float(n_fft))
    fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, : n_fft // 2 + 1]
    fft_basis = _sparsify_rows(fft_basis, SPARSITY)
    return fft_basis, n_fft, lengths


# ── STFT, no scipy (rectangular or hann "window" arg, center=True, zero pad) ─

def _stft(y: np.ndarray, n_fft: int, hop_length: int, window: str) -> np.ndarray:
    win = np.ones(n_fft) if window == "ones" else _hann_periodic(n_fft)
    pad = n_fft // 2
    y_pad = np.pad(y, (pad, pad), mode="constant")
    n_frames = 1 + (len(y_pad) - n_fft) // hop_length
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = y_pad[idx] * win  # (n_frames, n_fft)
    D = np.fft.rfft(frames, n=n_fft, axis=1)  # (n_frames, n_fft//2+1)
    return D.T  # (n_fft//2+1, n_frames)


def _cqt_response(y: np.ndarray, n_fft: int, hop_length: int, fft_basis: np.ndarray,
                   window: str = "ones", phase: bool = True) -> np.ndarray:
    D = _stft(y, n_fft, hop_length, window)
    if not phase:
        D = np.abs(D)
    return fft_basis.dot(D)  # (n_bins, n_frames)


# ── half-band decimator (numpy stand-in for soxr_hq, see module docstring) ──

def _make_halfband_taps(num_zeros: int = 32, beta: float = 10.0) -> np.ndarray:
    fc = 0.25  # cutoff at target Nyquist, in cycles per ORIGINAL sample
    n = np.arange(-num_zeros, num_zeros + 1, dtype=np.float64)
    h = 2 * fc * np.sinc(2 * fc * n)
    h = h * np.kaiser(len(n), beta)
    return h / np.sum(h)


_HALFBAND_TAPS = _make_halfband_taps()


def _num_two_factors(x: int) -> int:
    if x <= 0:
        return 0
    n = 0
    while x % 2 == 0:
        n += 1
        x //= 2
    return n


def _early_downsample_count(nyquist: float, filter_cutoff: float, hop_length: int, n_octaves: int) -> int:
    """librosa.core.constantq.__early_downsample_count -- MISSED in the first
    port (2026-09-23): for the full/recursive branch's actual call shape
    (n_bins=238, not 288), this evaluates to 1, not 0 -- the 288-bin split
    computation's cutoff (~5.9 kHz) does NOT carry over to the 238-bin
    sub-call, which has a lower max frequency and therefore a lower cutoff
    (~2.2 kHz) that DOES trigger one early 2x downsample before the octave
    loop even starts. Omitting it produced a systematic ~0.3-0.5% relative
    error through the ENTIRE full branch (confirmed present even at octave
    depth 0, i.e. with no per-octave resampling involved) -- traced by
    isolating basis/STFT/dot-product against real librosa (all matched to
    ~1e-7) and finding this was the only remaining divergent input.
    """
    downsample_count1 = max(0, int(np.ceil(np.log2(nyquist / filter_cutoff))) - 1 - 1)
    num_twos = _num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)
    return min(downsample_count1, downsample_count2)


def _early_downsample(y: np.ndarray, sr: float, hop_length: int, n_octaves: int,
                       nyquist: float, filter_cutoff: float):
    count = _early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
    if count > 0:
        factor = 2 ** count
        hop_length //= factor
        for _ in range(count):
            y = _resample_half(y)
        sr = sr / float(factor)
    return y, sr, hop_length


def _resample_half(y: np.ndarray) -> np.ndarray:
    """orig_sr=2, target_sr=1, res_type~soxr_hq (approx), fix=True, scale=True."""
    taps = _HALFBAND_TAPS
    pad = len(taps) // 2
    y_pad = np.pad(y, (pad, pad), mode="reflect")
    filtered = np.convolve(y_pad, taps, mode="valid")  # len == len(y)
    y_ds = filtered[::2]
    n_target = int(np.ceil(len(y) * 0.5))
    if len(y_ds) < n_target:
        y_ds = np.pad(y_ds, (0, n_target - len(y_ds)))
    elif len(y_ds) > n_target:
        y_ds = y_ds[:n_target]
    y_ds = y_ds * np.sqrt(2.0)  # scale=True: y_hat /= sqrt(ratio), ratio=0.5
    return y_ds.astype(np.float64)


# ── tuning estimation (librosa.estimate_tuning, no numba) ──────────────

def _parabolic_shift(x: np.ndarray) -> np.ndarray:
    """librosa.core.pitch._pi_stencil, vectorized along axis 0. Edges -> 0."""
    xm1 = np.roll(x, 1, axis=0)
    xp1 = np.roll(x, -1, axis=0)
    a = xp1 + xm1 - 2 * x
    b = (xp1 - xm1) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(np.abs(b) < np.abs(a), -b / np.where(a == 0, 1.0, a), 0.0)
    shift[0, :] = 0.0
    shift[-1, :] = 0.0
    return shift


def _localmax(x: np.ndarray) -> np.ndarray:
    xm1 = np.concatenate([[-np.inf], x[:-1]])
    xp1 = np.concatenate([x[1:], [-np.inf]])
    return (x > xm1) & (x >= xp1)


def estimate_tuning(y: np.ndarray, sr: float = MUSX_SR, bins_per_octave: int = BINS_PER_OCTAVE,
                     n_fft: int = 2048, resolution: float = 0.01,
                     fmin: float = 150.0, fmax: float = 4000.0, threshold: float = 0.1) -> float:
    hop_length = n_fft // 4
    D = _stft(y, n_fft, hop_length, window="hann")
    S = np.abs(D)  # (n_fft//2+1, n_frames)

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    avg = np.gradient(S, axis=0)
    shift = _parabolic_shift(S)
    dskew = 0.5 * avg * shift

    ref_value = threshold * np.max(S, axis=0, keepdims=True)
    freq_mask = (fft_freqs >= fmin) & (fft_freqs < min(fmax, sr / 2.0))
    freq_mask = freq_mask[:, None]

    is_peak = _localmax_axis0(S * (S > ref_value))
    idx = np.nonzero(freq_mask & is_peak)
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)
    pitches[idx] = (idx[0] + shift[idx]) * sr / n_fft
    mags[idx] = S[idx] + dskew[idx]

    pitch_mask = pitches > 0
    if not pitch_mask.any():
        return 0.0
    mag_threshold = np.median(mags[pitch_mask])
    freqs = pitches[(mags >= mag_threshold) & pitch_mask]
    freqs = freqs[freqs > 0]
    if freqs.size == 0:
        return 0.0

    hz_to_octs = np.log2(freqs / (440.0 / 16))  # librosa.hz_to_octs default tuning=0, A440
    residual = np.mod(bins_per_octave * hz_to_octs, 1.0)
    residual[residual >= 0.5] -= 1.0
    n_bins_hist = int(np.ceil(1.0 / resolution)) + 1
    bins = np.linspace(-0.5, 0.5, n_bins_hist)
    counts, edges = np.histogram(residual, bins)
    return float(edges[np.argmax(counts)])


def _localmax_axis0(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=bool)
    for j in range(x.shape[1]):
        out[:, j] = _localmax(x[:, j])
    return out


# ── the public entry point ──────────────────────────────────────────────

def song_cqt_288(y: np.ndarray, sr: float = MUSX_SR, hop_length: int = MUSX_HOP,
                  tuning: float | None = None) -> np.ndarray:
    """Reproduces CQTV2.extract's `abs(librosa.core.hybrid_cqt(...)).T`.

    Returns (n_frames, 288) float32. `tuning=None` (default, matches CQTV2's
    call) estimates it from `y`; pass a float to skip that (e.g. to isolate
    the CQT-vs-resampler question from the tuning-estimation question).
    """
    y = np.asarray(y, dtype=np.float64)
    if tuning is None:
        tuning = estimate_tuning(y, sr=sr, bins_per_octave=BINS_PER_OCTAVE)
    fmin = FMIN_F_SHARP_0 * 2.0 ** (tuning / BINS_PER_OCTAVE)

    # cqt_frequencies(n_bins, fmin, bins_per_octave) -- geometric series
    all_freqs = fmin * 2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE)
    lengths, _, _ = _wavelet_lengths(all_freqs, sr, FILTER_SCALE)
    pseudo_mask = 2.0 ** np.ceil(np.log2(lengths)) < 2 * hop_length
    n_pseudo = int(np.sum(pseudo_mask))
    n_full = N_BINS - n_pseudo

    cols = None

    # ---- pseudo branch: top n_pseudo bins, single STFT, no resampling ----
    pseudo_result = None
    if n_pseudo > 0:
        freqs_pseudo = all_freqs[pseudo_mask]
        fft_basis, n_fft, lengths_p = _vqt_filter_fft(sr, freqs_pseudo, hop_length)
        fft_basis = np.abs(fft_basis)
        C = _cqt_response(y, n_fft, hop_length, fft_basis, window="hann", phase=False)
        C = C / np.sqrt(n_fft)  # scale=True
        pseudo_result = C
        cols = C.shape[1]

    # ---- full branch: bottom n_full bins, recursive octave downsampling --
    full_result = None
    if n_full > 0:
        freqs_full = fmin * 2.0 ** (np.arange(n_full) / BINS_PER_OCTAVE)
        n_octaves = int(np.ceil(n_full / BINS_PER_OCTAVE))
        n_filters = min(BINS_PER_OCTAVE, n_full)

        _, filter_cutoff, _ = _wavelet_lengths(freqs_full, sr, FILTER_SCALE)
        nyquist = sr / 2.0
        my_y, my_sr, my_hop = _early_downsample(y, sr, hop_length, n_octaves, nyquist, filter_cutoff)
        resp = []
        for i in range(n_octaves):
            if i == 0:
                sl = slice(-n_filters, None)
            else:
                lo = max(0, n_full - n_filters * (i + 1))
                hi = n_full - n_filters * i
                sl = slice(lo, hi)
            freqs_oct = freqs_full[sl]
            fft_basis, n_fft, _ = _vqt_filter_fft(my_sr, freqs_oct, None)
            fft_basis = fft_basis * np.sqrt(sr / my_sr)
            resp.append(_cqt_response(my_y, n_fft, my_hop, fft_basis, window="ones", phase=True))
            if my_hop % 2 == 0:
                my_hop //= 2
                my_sr /= 2.0
                my_y = _resample_half(my_y)

        max_col = min(r.shape[1] for r in resp)
        full = np.zeros((n_full, max_col), dtype=np.complex128)
        end = n_full
        for r in resp:
            n_oct = r.shape[0]
            if end < n_oct:
                full[:end, :] = r[-end:, :max_col]
            else:
                full[end - n_oct:end, :] = r[:, :max_col]
            end -= n_oct
        full = np.abs(full)
        lengths_full, _, _ = _wavelet_lengths(freqs_full, sr, FILTER_SCALE)
        full = full / np.sqrt(lengths_full[:, None])  # scale=True
        full_result = full
        cols = full.shape[1] if cols is None else min(cols, full.shape[1])

    # ---- trim_stack: pseudo occupies the TOP n_pseudo rows, full the rest ---
    out = np.zeros((N_BINS, cols), dtype=np.float64)
    if pseudo_result is not None:
        out[N_BINS - n_pseudo:, :] = pseudo_result[:, :cols]
    if full_result is not None:
        out[: n_full, :] = full_result[:, :cols]

    return out.T.astype(np.float32)  # (n_frames, 288)


def cqt_musx(y: np.ndarray, sr: float = MUSX_SR, tuning: float | None = None) -> np.ndarray:
    """The (n_frames, 252) tensor ChordNet.inference actually consumes."""
    full = song_cqt_288(y, sr=sr, tuning=tuning)
    return full[:, SLICE_LO:SLICE_HI]
