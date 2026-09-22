"""Pure-numpy port of the Mauch & Dixon NNLS-Chroma VAMP plugin (bothchroma output).

Ported (not adapted) from the C++ GPL source, github.com/c4dm/nnls-chroma
(Matthias Mauch, QMUL C4DM), commit "master" as of 2026-09-23:
  NNLSBase.cpp / NNLSBase.h / NNLSChroma.cpp / chromamethods.cpp / nnls.c
and from the Vamp host SDK (vamp-plugins/vamp-plugin-sdk):
  src/vamp-hostsdk/PluginInputDomainAdapter.cpp, Window.h, vamp/frames.py.
No scipy. NNLS is a hand-rolled active-set Lawson-Hanson solver (nnls_lh
below), the same algorithm nnls.c itself is a translation of (Lawson &
Hanson, "Solving Least Squares Problems", 1974) -- not the exact C
translation byte-for-byte, but the same textbook algorithm, run in float64.

WHY THIS EXISTS: `harmonia/nnls_features.py` calls
`vamp.collect(y, SR, "nnls-chroma:nnls-chroma", output="bothchroma")` with
NO parameter overrides -- every plugin default therefore IS the spec. Read
from `~/Library/Audio/Plug-Ins/Vamp/nnls-chroma.n3` (RDF plugin description)
and confirmed against source:
    useNNLS=1  rollon=0  tuningmode=0 (global)  whitening=1.0  s=0.7
    chromanormalize=0 (none -- nnls_features.py does its own L2-per-half)
Frame geometry (from vampyhost, SR=44100): stepSize=2048, blockSize=16384
(preferred sizes, unconditionally used since nnls_features.py passes none).
`vamp/frames.py` extracts frame i as raw samples [i*step, i*step+block),
NOT centred/pre-padded -- frame 0 starts at sample 0. `vamp.collect`'s
"matrix" mode assigns output times as `arange(T) * step/SR`, so the only
thing that matters for alignment is frame ORDER, not the plugin's internal
per-frame Vamp::RealTime timestamp.

WHAT THIS DOES NOT PORT: `rollon` (bass noise gate) and local tuning mode
(`tuningmode=1`) are implemented for `rollon` (still exercised since it's
cheap) but ONLY the tuningmode=0 (global) path is implemented for tuning,
since that is the default and the only path `nnls_features.py` exercises.
Passing a different `vamp.collect(..., parameters={...})` dict anywhere
would silently stop matching this port -- grep for one before trusting a
mismatch away as "float noise".

Precision: the plugin does its arithmetic in C++ `float` (32-bit)
throughout (`m_dict`, `magnitude`, `b[]`, `x[]` are all `float`). This port
runs internally in float64 for solver stability and casts to float32 only
at the final bothchroma output -- expect the two to agree to noticeably
better than float32 ULP, not bit-identically; see
docs/portage_mobile/p0_t2_nnls_numpy.md for the measured parity.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 44100
STEP_SIZE = 2048
BLOCK_SIZE = 16384
N_FFT = BLOCK_SIZE // 2          # 8192 -- magnitude bins actually used (Nyquist bin dropped)
N_BPS = 3                        # bins per semitone
N_OCTAVE = 7
N_NOTE = N_OCTAVE * 12 * N_BPS + 2 * (N_BPS // 2 + 1)   # 256
N_SEMITONE_NOTES = 12 * N_OCTAVE                         # 84 (dictionary / semitonespectrum columns)
S_PARAM = 0.7
WHITENING = 1.0
ROLLON = 0.0
TUNINGMODE_GLOBAL = True

# ── bass/treble register-weighting windows (verbatim from chromamethods.h,
#    84 values each, one per semitone across the 7-octave note range) ──────
BASSWINDOW = np.array([
    0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509, 0.348162,
    0.430105, 0.514023, 0.597545, 0.678311, 0.754038, 0.822586, 0.882019, 0.930656,
    0.967124, 0.990393, 0.999803, 0.995091, 0.976388, 0.944223, 0.899505, 0.843498,
    0.777785, 0.704222, 0.624888, 0.542025, 0.457975, 0.375112, 0.295778, 0.222215,
    0.156502, 0.100495, 0.055777, 0.023612, 0.004909, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
], dtype=np.float64)
TREBLEWINDOW = np.array([
    0.000350, 0.003144, 0.008717, 0.017037, 0.028058, 0.041719, 0.057942, 0.076638,
    0.097701, 0.121014, 0.146447, 0.173856, 0.203090, 0.233984, 0.266366, 0.300054,
    0.334860, 0.370590, 0.407044, 0.444018, 0.481304, 0.518696, 0.555982, 0.592956,
    0.629410, 0.665140, 0.699946, 0.733634, 0.766016, 0.796910, 0.826144, 0.853553,
    0.878986, 0.902299, 0.923362, 0.942058, 0.958281, 0.971942, 0.982963, 0.991283,
    0.996856, 0.999650, 0.999650, 0.996856, 0.991283, 0.982963, 0.971942, 0.958281,
    0.942058, 0.923362, 0.902299, 0.878986, 0.853553, 0.826144, 0.796910, 0.766016,
    0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696, 0.481304,
    0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366, 0.233984, 0.203090,
    0.173856, 0.146447, 0.121014, 0.097701, 0.076638, 0.057942, 0.041719, 0.028058,
    0.017037, 0.008717, 0.003144, 0.000350,
], dtype=np.float64)


# ── cheap trig helpers matching the C++ source's naming ─────────────────────

def _cospuls(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    out = np.zeros_like(x)
    m = np.abs(x - centre) <= 0.5 * width
    out[m] = np.cos((x[m] - centre) * 2 * np.pi / width) * 0.5 + 0.5
    return out


def _pitch_cospuls(x: np.ndarray, centre: float, bins_per_octave: int) -> np.ndarray:
    with np.errstate(divide="ignore"):
        warped = -bins_per_octave * (np.log2(centre) - np.log2(np.maximum(x, 1e-12)))
    out = _cospuls(warped, 0.0, 2.0)
    c = np.log(2.0) / bins_per_octave
    out = np.where(x > 0, out / (c * np.maximum(x, 1e-12)), 0.0)
    return out


# ── one-time (audio-independent) precomputation: log-freq kernel + dict ────

def _cq_frequencies(binspersemitone: int = N_BPS, minoctave: int = 0, maxoctave: int = N_OCTAVE):
    """440*2^((MIDI+k/bps-69)/12) grid used by BOTH logFreqMatrix and dictionaryMatrix."""
    min_midi = 21 + minoctave * 12 - 1
    max_midi = 21 + maxoctave * 12
    oob = 1.0 / binspersemitone
    freqs = []
    for i in range(min_midi, max_midi):
        for k in range(binspersemitone):
            freqs.append(440.0 * 2.0 ** (0.083333333333 * (i + oob * k - 69)))
    freqs.append(440.0 * 2.0 ** (0.083333 * (max_midi - 69)))
    return np.asarray(freqs, dtype=np.float64)  # length N_NOTE


def build_logfreq_kernel(sr: int = SR, blocksize: int = BLOCK_SIZE,
                          oversampling: int = 80) -> np.ndarray:
    """(N_NOTE, N_FFT) sparse-ish mapping from linear FFT magnitude bins to
    the log-frequency (pitch) spectrum. Port of chromamethods.cpp
    `logFreqMatrix`. Returned as `K` such that `nm = K @ magnitude`
    (transposed relative to the C++ `outmatrix[iFFT + nFFT*iCQ]` layout, so
    it can be used directly as a matmul).
    """
    n_fft = blocksize // 2
    bin_spacing = sr * 1.0 / blocksize          # fft_f[1]
    fft_width = sr * 2.0 / blocksize            # = 2 * bin_spacing
    cq_f = _cq_frequencies()                    # (N_NOTE,)

    # fixed 2*oversampling-length window shape, independent of iFFT (see
    # module docstring derivation: fft_activation only ever depends on the
    # OFFSET from the bin's own centre frequency, not on iFFT itself).
    os_idx = np.arange(2 * oversampling, dtype=np.float64)
    os_freq_offset = os_idx * (bin_spacing / oversampling)  # 0 .. 2*bin_spacing
    fft_activation = _cospuls(os_freq_offset, bin_spacing, fft_width)  # (160,)

    fft_f = np.arange(n_fft, dtype=np.float64) * bin_spacing

    K = np.zeros((N_NOTE, n_fft), dtype=np.float64)

    lo = cq_f * (2.0 ** -0.168) - fft_width
    hi = cq_f * (2.0 ** 0.084) + fft_width

    for i_fft in range(1, n_fft):
        f = fft_f[i_fft]
        cand = np.nonzero((hi > f) & (lo < f))[0]
        if cand.size == 0:
            continue
        freqs = f + os_freq_offset - bin_spacing  # (160,) absolute freqs around this bin
        # (n_cand, 160) pitch-cospuls activations
        act = _pitch_cospuls(np.broadcast_to(freqs, (cand.size, freqs.size)),
                              cq_f[cand][:, None], N_BPS * 12)
        contrib = (act * fft_activation[None, :]).sum(axis=1)  # (n_cand,)
        K[cand, i_fft] += contrib

    return K


def build_dictionary(s_param: float = S_PARAM) -> np.ndarray:
    """(N_NOTE, N_SEMITONE_NOTES) NNLS note dictionary. Port of
    chromamethods.cpp `dictionaryMatrix` -- one harmonic comb per candidate
    semitone (84 of them across 7 octaves), 20 harmonics, amplitude s^(h-1),
    each harmonic a width-`N_BPS` raised-cosine bump in note-index space.
    """
    dm = np.zeros((N_NOTE, N_SEMITONE_NOTES), dtype=np.float64)
    note_idx = np.arange(1, N_NOTE + 1, dtype=np.float64)  # "iNote+1" in the C++
    for i_out in range(N_SEMITONE_NOTES):
        for i_harm in range(1, 21):
            floatbin = (i_out + 1) * N_BPS + 1 + N_BPS * 12 * np.log2(i_harm)
            curr_amp = s_param ** (i_harm - 1)
            mask = np.abs(note_idx - floatbin) < 2
            if not mask.any():
                continue
            dm[mask, i_out] += _cospuls(note_idx[mask], floatbin, float(N_BPS)) * curr_amp
    return dm


def _hann_periodic(n: int) -> np.ndarray:
    """Window.h `cosinewin` with a0=a1=0.5, a2=a3=0 -- PERIODIC Hann (denominator n,
    not n-1), i.e. NOT numpy's default `np.hanning` (which is symmetric)."""
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / n)


def _hamming_half_octave() -> np.ndarray:
    """`hw` in NNLSBase::initialise: Hamming window of length nBPS*6+1=19,
    normalised to sum 1 -- the "running mean/std" kernel used for whitening.
    """
    n = N_BPS * 6 + 1
    i = np.arange(n, dtype=np.float64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))
    return w / w.sum()


def _special_convolution(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Port of SpecialConvolution: a 'valid' convolution of x (len N_NOTE)
    against an odd-length kernel, placed centred, edges replicated from the
    nearest valid value (NOT zero-padded). x is (T, N_NOTE); vectorised over
    T via correlate-per-row through a Toeplitz-free sliding-window matmul.
    """
    lk = kernel.size
    half = lk // 2
    n = x.shape[-1]
    # 'valid' correlation with the kernel reversed matches C++'s convolution
    # loop `s += convolvee[n-m]*kernel[m]` (this is a true convolution).
    ker_rev = kernel[::-1]
    T = x.shape[0]
    valid_len = n - lk + 1
    out = np.empty((T, n), dtype=np.float64)
    core = np.empty((T, valid_len), dtype=np.float64)
    for j in range(valid_len):
        core[:, j] = x[:, j:j + lk] @ ker_rev
    out[:, half:half + valid_len] = core
    out[:, :half] = core[:, [0]]
    out[:, half + valid_len:] = core[:, [-1]]
    return out


# ── Lawson-Hanson NNLS (own implementation, float64, no scipy) ─────────────

def nnls_lh(A: np.ndarray, b: np.ndarray, max_iter: int | None = None,
            tol: float = 1e-10) -> np.ndarray:
    """min ||Ax - b||_2 s.t. x >= 0. Classic active-set Lawson-Hanson (1974)
    algorithm -- the same one `nnls.c` is a C translation of. A is (m, n).
    """
    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n
    x = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=bool)   # passive (unconstrained) set
    w = A.T @ (b - A @ x)
    outer = 0
    while (not P.all()) and w[~P].max(initial=-np.inf) > tol and outer < max_iter:
        outer += 1
        cand = np.where(~P)[0]
        j = cand[np.argmax(w[cand])]
        P[j] = True
        for _inner in range(max_iter):
            Ap = A[:, P]
            z_p, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            if (z_p > 1e-12).all():
                x[P] = z_p
                break
            neg = P.copy()
            idx_p = np.where(P)[0]
            neg_mask = z_p <= 1e-12
            if not neg_mask.any():
                x[P] = z_p
                break
            alpha = np.min(x[idx_p[neg_mask]] /
                            (x[idx_p[neg_mask]] - z_p[neg_mask]))
            x[P] = x[P] + alpha * (z_p - x[P])
            P[idx_p[np.where(np.isclose(x[idx_p], 0.0, atol=1e-9))[0]]] = False
        else:
            x[P] = z_p
        x[~P] = 0.0
        w = A.T @ (b - A @ x)
    return x


# ── batched FISTA NNLS (P0-T2b speedup): same fixed dictionary A for every
#    frame, solved for ALL frames at once as one (n, T) matrix -- replaces
#    the per-frame Python loop over `nnls_lh` (was 27s of the 32s total).
#    Per-frame restriction to "signifIndex" columns (candidate notes with
#    nonzero local energy) becomes a per-frame boolean MASK on the shared
#    dictionary instead of a reduced matrix: forcing x_j=0 for masked-off j
#    at every iteration is mathematically identical to solving the reduced
#    problem over the surviving columns (fixing a variable at 0 == dropping
#    it), so this is not an approximation of the per-frame restriction, only
#    of the NNLS solve itself (early-stopped projected gradient vs. an exact
#    active-set solve to 1e-10). ─────────────────────────────────────────

def nnls_batch_fista(A: np.ndarray, AtA: np.ndarray, B: np.ndarray, mask: np.ndarray,
                      max_iter: int, step: float) -> np.ndarray:
    """min_X >=0  sum_t ||A x_t - b_t||^2, all T columns of B solved at once
    (accelerated projected gradient / FISTA, Beck & Teboulle 2009). A is
    (m, n) fixed for every frame; B is (T, m); mask is (T, n) bool (True =
    column allowed nonzero for that frame, i.e. that frame's "signifIndex").
    Returns X (T, n). `step` = 1/L, L = largest eigenvalue of AtA (global,
    conservative Lipschitz bound -- valid for every per-frame masked
    submatrix too, since a PSD principal submatrix's top eigenvalue can only
    be <= the full matrix's).
    """
    AtB = B @ A                      # (T, n) -- one matmul, replaces per-frame A.T@b
    maskf = mask.astype(np.float64)
    X = np.zeros_like(AtB)
    Y = X
    t_k = 1.0
    for _ in range(max_iter):
        grad = Y @ AtA - AtB         # (T, n) -- one matmul per iteration, ALL frames at once
        X_new = np.maximum(Y - step * grad, 0.0) * maskf
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        Y = X_new + ((t_k - 1.0) / t_next) * (X_new - X)
        X = X_new
        t_k = t_next
    return X


# ── the public entry point ───────────────────────────────────────────────

class NNLSChromaNumpy:
    """Stateful holder for the two one-time-computed matrices (kernel, dict)
    so repeated calls (e.g. across songs in one process) don't rebuild them.
    """

    def __init__(self):
        self.K = build_logfreq_kernel()
        self.D = build_dictionary()
        self.hw = _hamming_half_octave()
        self.sin_bps = np.sin(2 * np.pi * np.arange(N_BPS) / N_BPS)
        self.cos_bps = np.cos(2 * np.pi * np.arange(N_BPS) / N_BPS)
        # batched-FISTA precomputation (P0-T2b): AtA is fixed (same dict for
        # every frame), so its top eigenvalue (-> Lipschitz step) is computed
        # once here, not per song.
        self.AtA = self.D.T @ self.D                       # (84, 84)
        L = float(np.linalg.eigvalsh(self.AtA)[-1])
        self.fista_step = 1.0 / L if L > 0 else 1.0
        self.bass_win_oc = BASSWINDOW.reshape(N_OCTAVE, 12)     # octave-major, matches D's column order
        self.treble_win_oc = TREBLEWINDOW.reshape(N_OCTAVE, 12)

    def extract_bothchroma(self, y: np.ndarray, sr: int = SR, method: str = "fista",
                            max_iter: int = 250):
        """y: float32/float64 mono audio at `sr` (must be SR=44100, same
        assumption `nnls_features.extract_bothchroma` makes via librosa.load
        (sr=SR)). Returns (arr (T,24) float32, times (T,) float64), index 0
        = A, bass half first then treble -- same convention as the plugin.

        `method="fista"` (default, P0-T2b): batched accelerated projected
        gradient over ALL frames at once, ~15-20x faster than the per-frame
        active-set solve (default `max_iter=250` -- parity plateaus there,
        see docs/portage_mobile/p0_t2b_nnls_vitesse.md). `method="lawson"`:
        the original per-frame Lawson-Hanson active-set NNLS (P0-T2, kept
        for reference/parity auditing) -- exact to 1e-10 per frame but a
        Python loop, slow.
        """
        if sr != SR:
            raise ValueError(f"nnls_chroma numpy port expects sr={SR}, got {sr}")
        if method not in ("fista", "lawson"):
            raise ValueError(f"method must be 'fista' or 'lawson', got {method!r}")
        y = np.asarray(y, dtype=np.float64)

        n = len(y)
        # match vamp/frames.py: frames start at 0, step STEP_SIZE, continue
        # while start < n (last frame zero-padded on the right).
        starts = np.arange(0, max(n, 1), STEP_SIZE)
        starts = starts[starts < n] if n > 0 else starts[:0]
        T = len(starts)
        if T == 0:
            return np.zeros((0, 24), np.float32), np.zeros((0,), np.float64)

        win = _hann_periodic(BLOCK_SIZE)

        pad_needed = starts[-1] + BLOCK_SIZE - n
        y_padded = np.concatenate([y, np.zeros(max(0, pad_needed), dtype=np.float64)]) if pad_needed > 0 else y

        silence_thresh = BLOCK_SIZE * 2.0 / 16384.0  # == 2.0 for BLOCK_SIZE=16384

        if ROLLON > 0:
            # rollon>0 is coded-but-never-exercised by nnls_features.py (see
            # module docstring); keep the exact per-frame reference path for
            # that rare case rather than vectorising an untested branch.
            nm_all = np.empty((T, N_NOTE), dtype=np.float64)
            for t, s0 in enumerate(starts):
                frame = y_padded[s0:s0 + BLOCK_SIZE]
                windowed = frame * win
                spec = np.fft.rfft(windowed)[:N_FFT]
                mag = np.abs(spec)
                mag = np.minimum(mag, BLOCK_SIZE)
                energysum = (mag ** 2).sum()
                cum = 0.0
                for ib in range(2, N_FFT):
                    cum += mag[ib] ** 2
                    if cum < energysum * ROLLON / 100:
                        mag[ib - 2] = 0.0
                    else:
                        break
                if mag.max(initial=0.0) < silence_thresh:
                    mag[:] = 0.0
                nm_all[t] = self.K @ mag
        else:
            # vectorised over all T frames at once (P0-T2b): sliding_window_view
            # gives every BLOCK_SIZE-length frame without copies, then window
            # + FFT + log-freq projection are each ONE batched call instead of
            # a Python loop of T calls. Identical arithmetic to the per-frame
            # loop above, just batched -- not an approximation.
            frames = sliding_window_view(y_padded, BLOCK_SIZE)[::STEP_SIZE][:T]  # (T, BLOCK_SIZE)
            windowed = frames * win[None, :]
            spec = np.fft.rfft(windowed, axis=1)[:, :N_FFT]
            mag = np.abs(spec)
            mag = np.minimum(mag, BLOCK_SIZE)
            silent = mag.max(axis=1) < silence_thresh
            mag[silent] = 0.0
            nm_all = mag @ self.K.T   # (T, N_FFT) @ (N_FFT, N_NOTE) -> (T, N_NOTE)

        # ── global tuning estimate (tuningmode=0 path only) ───────────────
        tone_starts = np.arange(0, round(N_NOTE * 0.62 / N_BPS) * N_BPS + 1, N_BPS)
        # sum over all "tone" positions and all frames, per BPS offset
        gathered = nm_all[:, tone_starts[:, None] + np.arange(N_BPS)[None, :]]  # (T, n_tones, 3)
        mean_tunings = gathered.sum(axis=1).mean(axis=0)  # (3,)
        real = float((mean_tunings * self.cos_bps).sum())
        imag = float((mean_tunings * self.sin_bps).sum())
        normalised_tuning = np.arctan2(imag, real) / (2 * np.pi)
        int_shift = int(np.floor(normalised_tuning * 3))
        float_shift = normalised_tuning * 3 - int_shift

        # ── tune + whiten (vectorised over all T frames at once) ─────────
        f2 = np.zeros((T, N_NOTE), dtype=np.float64)
        k = np.arange(2, N_NOTE - 3)
        f2[:, k] = (nm_all[:, k + int_shift] * (1 - float_shift) +
                    nm_all[:, k + int_shift + 1] * float_shift)
        # (edges already 0)

        running_mean = _special_convolution(f2, self.hw)
        dev2 = (f2 - running_mean) ** 2
        running_var = _special_convolution(dev2, self.hw)
        running_std = np.sqrt(np.maximum(running_var, 0.0))

        diff = f2 - running_mean
        whitened = np.where(diff > 0, diff / np.power(np.maximum(running_std, 1e-30), WHITENING), 0.0)
        f2w = np.where(running_std > 0, whitened, f2)  # runningstd==0 -> unchanged (matches C++ no-op branch)

        # ── NNLS + fold into bass/treble chroma ────────────────────────────
        centers = N_BPS // 2 + 2 + N_BPS * np.arange(N_SEMITONE_NOTES)  # iNote centers, 84 of them
        centers = centers[centers < N_NOTE - N_BPS // 2]

        if method == "lawson":
            bass = np.zeros((T, 12), dtype=np.float64)
            treble = np.zeros((T, 12), dtype=np.float64)
            for t in range(T):
                b = f2w[t]
                if not (b > 0).any():
                    continue
                currval = b[centers - 1] + b[centers] + b[centers + 1]
                sig = np.where(currval > 0)[0]
                if sig.size == 0:
                    continue
                Ad = self.D[:, sig]
                x = nnls_lh(Ad, b)
                semitone = np.zeros(N_SEMITONE_NOTES, dtype=np.float64)
                semitone[sig] = x
                treble[t] = np.array([semitone[i::12].dot(TREBLEWINDOW[i::12]) for i in range(12)])
                bass[t] = np.array([semitone[i::12].dot(BASSWINDOW[i::12]) for i in range(12)])
        else:
            # batched FISTA (P0-T2b): same "signifIndex" restriction as the
            # lawson path, but as a per-frame column MASK on the shared
            # dictionary (see nnls_batch_fista docstring) so the whole T-frame
            # batch is solved with a handful of (T,84)/(84,84) matmuls.
            currval_all = f2w[:, centers - 1] + f2w[:, centers] + f2w[:, centers + 1]  # (T, 84)
            mask = currval_all > 0
            X = nnls_batch_fista(self.D, self.AtA, f2w, mask, max_iter, self.fista_step)  # (T, 84)
            Xr = X.reshape(T, N_OCTAVE, 12)  # octave-major columns, see __init__ comment
            bass = np.einsum('toc,oc->tc', Xr, self.bass_win_oc)
            treble = np.einsum('toc,oc->tc', Xr, self.treble_win_oc)

        arr = np.concatenate([bass, treble], axis=1).astype(np.float32)  # (T,24) bass|treble, index0=A
        times = starts.astype(np.float64) / SR
        return arr, times


_singleton: NNLSChromaNumpy | None = None


def extract_bothchroma_numpy(y: np.ndarray, sr: int = SR):
    """Drop-in, numpy-only alternative to
    `harmonia.nnls_features.extract_bothchroma`'s `vamp.collect(...,
    output="bothchroma")` call -- same (arr, times) contract. Lazily builds
    (and caches in-process) the audio-independent kernel/dictionary
    matrices on first call.
    """
    global _singleton
    if _singleton is None:
        _singleton = NNLSChromaNumpy()
    return _singleton.extract_bothchroma(y, sr)
