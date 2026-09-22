"""Pure-numpy reproduction of beat_this's audio -> log-mel front-end.

Why this exists: ExecuTorch/XNNPACK cannot lower `torch.stft` (it produces a
complex64 tensor mid-graph, and XNNPACK has no complex kernels -- confirmed
in `mobile/tools/export_beat_this.py`'s LogMelSpect export attempt, which
fails with a `SpecViolationError` on `aten.view_copy`/`aten.abs` in
complex64). So the mel-spectrogram can never live inside the exported
`.pte` graph; on the phone it has to run as plain numpy before the network.

This module reproduces, bin-for-bin, `beat_this.preprocessing.LogMelSpect`
(`beat_this` 1.1.0, read from
`~/harmonia/.venv-mobile/lib/python3.12/site-packages/beat_this/preprocessing.py`):

    LogMelSpect(sample_rate=22050, n_fft=1024, hop_length=441, f_min=30,
                f_max=11000, n_mels=128, mel_scale="slaney",
                normalized="frame_length", power=1, log_multiplier=1000)

which is `torchaudio.transforms.MelSpectrogram(...)` (that config) followed
by `log1p(1000 * mel)`. The exact recipe, read from torchaudio 2.11.0
source (`torchaudio/transforms/_transforms.py`,
`torchaudio/functional/functional.py`), each point independently verified
against torchaudio behaviour (see `parity_beats.py`):

  1. STFT: n_fft=1024, hop_length=441, win_length=1024 (== n_fft, so no
     window-centering pad is needed inside the frame), window =
     `torch.hann_window(1024)` which is the PERIODIC Hann window
     (w[n] = 0.5 - 0.5*cos(2*pi*n/N), n=0..N-1 -- NOT the symmetric/
     N-1-denominator version `np.hanning` gives), center=True with
     pad_mode="reflect" (`np.pad(..., mode="reflect")`, pads n_fft//2=512
     samples on each side), onesided=True (513 freq bins).
  2. magnitude, not power: `power=1` -> `|STFT|`, no squaring.
  3. normalized="frame_length" is torch.stft's OWN `normalized=True`, which
     divides every frame's FFT by sqrt(n_fft) -- this is NOT torchaudio's
     separate "window" L2-energy normalization (that one divides by
     `window.pow(2).sum().sqrt()` and is NOT used here: LogMelSpect never
     sets it, `window_norm` stays False for the "frame_length" string).
  4. mel filterbank: `mel_scale="slaney"` selects the SLANEY hz<->mel WARP
     formula (linear below 1000 Hz, log above -- see `_hz_to_mel_slaney`
     below), 128 mels between f_min=30 Hz and f_max=11000 Hz, applied to
     513 STFT bins (n_fft//2+1). Crucially `norm` (torchaudio's SEPARATE
     "slaney" AREA normalization, dividing each triangle by its bandwidth)
     is NEVER passed by LogMelSpect, so it defaults to `None` -- filters
     are triangles peaking at 1.0, NOT area-normalized. Mixing this up
     (i.e. using librosa's default `mel(..., norm='slaney')`, which DOES
     area-normalize) silently produces a different spectrogram that still
     "looks like" a mel-spectrogram -- exactly CLAUDE.md rule 1's shape of
     bug, so it gets a dedicated regression check in `parity_beats.py`.
  5. `log1p(1000 * mel)`.

Output: `log_mel_spectrogram(y, sr=22050) -> (T, 128) float32`, T = number
of hops = `1 + len(y) // hop_length` (audio must already be 22050 Hz mono
float32 -- resampling/mixdown is the caller's job, same division of labour
as `beat_this.inference.Audio2Frames.signal2spect`).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 441
WIN_LENGTH = 1024
F_MIN = 30.0
F_MAX = 11000.0
N_MELS = 128
FPS = SAMPLE_RATE / HOP_LENGTH  # 50.0 exactly -- beat_this's Postprocessor assumes this


def _hann_periodic(n: int) -> np.ndarray:
    """torch.hann_window(n) default (periodic=True): 0.5 - 0.5*cos(2*pi*k/n),
    k=0..n-1. NOT np.hanning(n) (that's the symmetric window, denominator
    n-1, and would silently mismatch by ~1 part in n_fft at the edges)."""
    k = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


def _hz_to_mel_slaney(freq: np.ndarray | float) -> np.ndarray:
    """torchaudio's `_hz_to_mel(freq, mel_scale="slaney")`: linear below
    1000 Hz (f_sp = 200/3 Hz/mel), log-spaced above (matches Slaney's
    Auditory Toolbox / librosa's htk=False formula)."""
    freq = np.asarray(freq, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15.0
    logstep = np.log(6.4) / 27.0
    mel = freq / f_sp
    log_region = freq >= min_log_hz
    mel = np.where(log_region, min_log_mel + np.log(np.maximum(freq, 1e-12) / min_log_hz) / logstep, mel)
    return mel


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    freq = f_sp * mel
    log_region = mel >= min_log_mel
    freq = np.where(log_region, min_log_hz * np.exp(logstep * (mel - min_log_mel)), freq)
    return freq


def mel_filterbank(n_freqs: int = N_FFT // 2 + 1, f_min: float = F_MIN, f_max: float = F_MAX,
                    n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """(n_freqs, n_mels) triangular filterbank, torchaudio's
    `melscale_fbanks(..., norm=None, mel_scale="slaney")` -- UNNORMALIZED
    triangles (peak 1.0), Slaney hz<->mel warp. `norm=None` is the part
    that's easy to get silently wrong (librosa's own default area-
    normalizes); LogMelSpect never opts into that normalization."""
    all_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)
    m_min = _hz_to_mel_slaney(f_min)
    m_max = _hz_to_mel_slaney(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    f_pts = _mel_to_hz_slaney(m_pts)  # (n_mels+2,)

    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels+1,)
    slopes = f_pts[None, :] - all_freqs[:, None]  # (n_freqs, n_mels+2)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    return fb.astype(np.float32)


_FB = mel_filterbank()  # built once at import time (128, ) x (513,) -> cheap
_WINDOW = _hann_periodic(WIN_LENGTH).astype(np.float64)


def stft_magnitude(y: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                    win_length: int = WIN_LENGTH) -> np.ndarray:
    """(T, n_fft//2+1) magnitude spectrogram, matching
    `torch.stft(y, n_fft, hop_length, win_length, window=hann_periodic,
    center=True, pad_mode="reflect", normalized=True, onesided=True,
    return_complex=True).abs()`.

    center=True pads n_fft//2 samples of reflection on each side (numpy's
    `mode="reflect"`, i.e. NOT repeating the edge sample -- that would be
    `mode="symmetric"`). `normalized=True` divides each frame's FFT by
    sqrt(n_fft) (torch.stft's own normalization, applied BEFORE windowing
    energy is considered -- verified against torch.stft source, see module
    docstring point 3).
    """
    y = np.asarray(y, dtype=np.float64)
    pad = n_fft // 2
    y_padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(y_padded) - n_fft) // hop_length
    # sliding_window_view over the padded signal, one frame per hop
    frames = np.lib.stride_tricks.sliding_window_view(y_padded, n_fft)[::hop_length][:n_frames]
    windowed = frames * _WINDOW[None, :]
    spec = np.fft.rfft(windowed, n=n_fft, axis=-1)  # (T, n_fft//2+1) complex128
    spec /= np.sqrt(n_fft)  # torch.stft(normalized=True)
    return np.abs(spec)


def log_mel_spectrogram(y: np.ndarray, sample_rate: int = SAMPLE_RATE,
                         log_multiplier: float = 1000.0) -> np.ndarray:
    """waveform (T_samples,) at 22050 Hz mono float -> (T_frames, 128) float32,
    bin-for-bin equal (target <1e-3 max abs diff) to
    `beat_this.preprocessing.LogMelSpect()(torch.tensor(y))`."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz input, got {sample_rate} "
                          "(resample before calling -- this module does not resample)")
    mag = stft_magnitude(y)  # (T, 513) float64
    mel = mag @ _FB.astype(np.float64)  # (T, 128)
    logmel = np.log1p(log_multiplier * mel)
    return logmel.astype(np.float32)
