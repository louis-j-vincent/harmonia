"""Pure-numpy re-implementation of musx's ChordNet (CNN + bi-LSTM + linear head).

Why this exists: ``torch.export()`` of one fold succeeds, but ExecuTorch's
XNNPACK lowering (``to_edge_transform_and_lower``) never terminates on this
network within any reasonable budget (>27 min, see
``docs/portage_mobile/p0_t1_executorch.md``) — suspected cause is the
bidirectional LSTM. This module sidesteps ExecuTorch entirely for musx: it
reproduces ``chordnet_ismir_naive.ChordNet`` forward pass (see that file,
``CNNFeatureExtractor`` + ``ChordNet.forward``/``inference``) with numpy only
(no scipy, no numba, no torch at inference time — Chaquopy-safe), loading
weights dumped once from the real torch checkpoints.

Architecture reproduced exactly (weight names/shapes read from
``chordnet.state_dict()`` on fold 0, verified against the module source):

  input (T, 252)                                    # CQT window, post 18:270 crop
    -> view (1, T, 252)                              # 1 input channel
    -> [conv1a,conv1b,conv1c: Conv2d(*,16,3x3,pad=1,1) + InstanceNorm2d + SELU]
    -> MaxPool2d(1,3) over freq                       # 252 -> 84
    -> [conv2a,conv2b,conv2c: Conv2d(*,32,3x3,pad=1,1) + InstanceNorm2d + SELU]
    -> MaxPool2d(1,3) over freq                       # 84 -> 28
    -> [conv3a,conv3b: Conv2d(*,64,3x3,pad=1,1) + InstanceNorm2d + SELU]
    -> MaxPool2d(1,4) over freq                       # 28 -> 7
    -> [conv4a: Conv2d(64,80,3x3,pad=1,0) + InstanceNorm2d + SELU]           # freq 7->5
    -> [conv4b: Conv2d(80,80,3x3,pad=1,0) + InstanceNorm2d + SELU]           # freq 5->3
    -> transpose+reshape -> (T, 240)                  # 80 channels * 3 freq bins
    -> bi-LSTM(240 -> 96 each direction) -> (T, 192)
    -> Linear(192 -> 100)
    -> split [triad(73), bass(13), s7(4), s9(4), s11(3), s13(3)]
    -> softmax each

Slice boundaries come from ``complex_chord.ChordTypeLimit`` as actually used
by ``ChordNet.forward`` (bass_slice_begin = triad_limit*12+1 = 73, etc — NOT
the same numbers as ``output_dim1``/``output_dim2`` which sum to the same
total but split differently on paper; the network code is the source of
truth here, not the class docstring).

``InstanceNorm2d`` has no learnable affine here (torch default
``affine=False``, confirmed absent from the state_dict) — it is a pure
per-sample-per-channel (mean, biased var) normalization over (T, F).

No batching over the 5-fold ensemble is done here (kept simple, looped in
Python — see ``predict_ensemble``); at ~2-3 ms/frame this is not the
bottleneck (see parity/speed numbers in ``docs/portage_mobile/t1b_musx.md``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SHIFT_HIGH = 6
SHIFT_STEP = 3
SPEC_DIM = 252

# chord_limit = ChordTypeLimit(triad_limit=6, seventh_limit=3, ninth_limit=3,
#                               eleventh_limit=2, thirteenth_limit=2)
TRIAD_LIMIT = 6
SEVENTH_LIMIT = 3
NINTH_LIMIT = 3
ELEVENTH_LIMIT = 2
THIRTEENTH_LIMIT = 2

BASS_SLICE_BEGIN = TRIAD_LIMIT * 12 + 1          # 73
BASS_DEL = BASS_SLICE_BEGIN + 12 + 1             # 86
SEVENTH_DEL = BASS_DEL + SEVENTH_LIMIT + 1       # 90
NINTH_DEL = SEVENTH_DEL + NINTH_LIMIT + 1        # 94
ELEVENTH_DEL = NINTH_DEL + ELEVENTH_LIMIT + 1    # 97
THIRTEENTH_DEL = ELEVENTH_DEL + THIRTEENTH_LIMIT + 1  # 100

_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946

_CONV_LAYERS = [
    ("conv1a", 1, 1), ("conv1b", 1, 1), ("conv1c", 1, 1), ("POOL", 3, None),
    ("conv2a", 1, 1), ("conv2b", 1, 1), ("conv2c", 1, 1), ("POOL", 3, None),
    ("conv3a", 1, 1), ("conv3b", 1, 1), ("POOL", 4, None),
    ("conv4a", 1, 0), ("conv4b", 1, 0),
]


def _selu(x: np.ndarray) -> np.ndarray:
    return _SELU_SCALE * np.where(x > 0, x, _SELU_ALPHA * (np.expm1(np.minimum(x, 0))))


def _instance_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """x: (C, T, F). Per-channel, over (T, F), biased variance (torch default).

    BUG FOUND AND FIXED (2026-09-23, this task): ``x`` arrives here as a
    ``.transpose(2, 0, 1)`` view out of ``_conv2d`` — non-contiguous. numpy's
    reduction over a non-contiguous axis pair falls back to a less accurate
    summation order than its normal pairwise-sum fast path, which for a full
    song (T up to ~18k, so T*F up to ~4.6M elements per channel) is a real,
    measured effect, not float32 noise: mean error ~3.5e-5 on a synthetic
    array of this exact shape (vs ~8e-8 once contiguous — 450x), and on the
    real network this was the ENTIRE source of the length-dependent parity
    drift documented in ``docs/portage_mobile/t1b_musx.md`` (max|Δ posterior|
    growing from 0.0015 at 205s of audio to 0.056 at 422s — traced layer by
    layer: raw conv2d matched torch to 1e-6, but the first InstanceNorm2d
    already diverged by up to 0.7). Fix: force a contiguous float64 copy for
    the reduction only (mean/var), matching torch's much closer to true
    values; the normalize step itself stays in the input's dtype.
    """
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    mean = x64.mean(axis=(1, 2), keepdims=True)
    var = x64.var(axis=(1, 2), keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)).astype(np.float32)


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
            pad_t: int, pad_f: int) -> np.ndarray:
    """x: (Cin, T, F); weight: (Cout, Cin, 3, 3); bias: (Cout,) -> (Cout, Tout, Fout).

    Stacks the 9 shifted kernel taps into ONE (9*Cin, Tout*Fout) matrix and
    does a single matmul against the correspondingly-reshaped weight —
    instead of either (a) one big im2col ``sliding_window_view`` over all 9
    taps (materializes a (Cin, Tout, Fout, 3, 3) array — multiple GB and slow
    for a whole song at Cin=16, T~18000, F=252) or (b) 9 separate small
    ``tensordot`` calls in a python loop (correct and memory-safe, but ~1.5-
    7x slower than this version, measured on conv1b-sized layers: 2.5s vs
    0.5s per call for Cin=Cout=16, T=8840, F=252 — see git history of this
    file / ``docs/portage_mobile/t1b_musx.md`` for the a/b/c comparison). One
    matmul lets BLAS do the work instead of python-level loop overhead.
    """
    xp = np.pad(x, ((0, 0), (pad_t, pad_t), (pad_f, pad_f)))
    cin, tp, fp = xp.shape
    cout, _, kt, kf = weight.shape
    tout, fout = tp - kt + 1, fp - kf + 1
    taps = np.empty((kt * kf * cin, tout, fout), dtype=np.float32)
    idx = 0
    for i in range(kt):
        for j in range(kf):
            taps[idx * cin:(idx + 1) * cin] = xp[:, i:i + tout, j:j + fout]
            idx += 1
    taps2 = taps.reshape(kt * kf * cin, tout * fout)
    w2 = weight.transpose(2, 3, 1, 0).reshape(kt * kf * cin, cout)  # tap order matches taps
    out = taps2.T @ w2 + bias
    return out.reshape(tout, fout, cout).transpose(2, 0, 1)


def _maxpool_freq(x: np.ndarray, k: int) -> np.ndarray:
    c, t, f = x.shape
    fout = f // k
    return x[:, :, :fout * k].reshape(c, t, fout, k).max(axis=3)


def _conv2d_tiled(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                  pad_t: int, pad_f: int, tile_t: int) -> np.ndarray:
    """Same arithmetic as ``_conv2d``, bounded memory: the im2col tap matrix is
    built for at most ``tile_t`` OUTPUT frames at a time, straight into one
    preallocated output.

    Why this and not a whole-network temporal chunking (which is what a
    receptive-field argument would suggest) — see ``_cnn_forward``'s docstring:
    ``InstanceNorm2d`` normalises over the WHOLE time axis, so cutting the
    network in time is NOT exact at any overlap. Cutting each op in time IS,
    because a convolution output frame depends on a bounded, explicitly
    reconstructed slice of its own input (here ``[t-1, t+1]``, kernel 3 in
    time, stride 1), and the normalisation statistics are still computed over
    the full song (``_norm_selu_tiled``).

    Memory, measured (``mobile/tools/mem_profile_mobile.py``): the untiled
    version allocates ``9*Cin*Tout*Fout*4`` bytes in one go — 1283 MB for
    conv1b/conv1c on a 205 s song (Cin=16, T=8840, F=252). Tiled at 512
    frames that same array is 74 MB. It also drops the ``np.pad`` of the full
    input (another 143 MB on that layer): only the tile's own halo is padded.
    """
    cin, t_in, f_in = x.shape
    cout, _, kt, kf = weight.shape
    tout = t_in + 2 * pad_t - kt + 1
    fout = f_in + 2 * pad_f - kf + 1
    w2 = weight.transpose(2, 3, 1, 0).reshape(kt * kf * cin, cout)  # tap order matches taps
    out = np.empty((cout, tout, fout), dtype=np.float32)
    for t0 in range(0, tout, tile_t):
        t1 = min(t0 + tile_t, tout)
        n = t1 - t0
        # rows of the (virtually) padded input this tile needs: [t0, t1+kt-1),
        # i.e. rows [t0-pad_t, t1+kt-1-pad_t) of x, zero outside the song.
        r0, r1 = t0 - pad_t, t1 + kt - 1 - pad_t
        lo, hi = max(r0, 0), min(r1, t_in)
        sub = np.zeros((cin, r1 - r0, f_in + 2 * pad_f), dtype=np.float32)
        if hi > lo:
            sub[:, lo - r0:hi - r0, pad_f:pad_f + f_in] = x[:, lo:hi, :]
        taps = np.empty((kt * kf * cin, n, fout), dtype=np.float32)
        idx = 0
        for i in range(kt):
            for j in range(kf):
                taps[idx * cin:(idx + 1) * cin] = sub[:, i:i + n, j:j + fout]
                idx += 1
        res = taps.reshape(kt * kf * cin, n * fout).T @ w2 + bias
        out[:, t0:t1, :] = res.reshape(n, fout, cout).transpose(2, 0, 1)
    return out


def _norm_selu_tiled(x: np.ndarray, tile_t: int, eps: float = 1e-5) -> np.ndarray:
    """``_instance_norm`` + ``_selu``, fused and done in place, tile by tile.

    Same two-pass (mean, then biased variance) float64 reduction as
    ``_instance_norm`` / ``np.var``, but accumulated over contiguous float64
    tiles instead of one contiguous float64 copy of the whole activation map —
    which is the point: that copy is 285 MB on conv1* of a 205 s song, and the
    float64 temporaries of ``(x - mean) / sqrt(var + eps)`` are 285 MB each
    again (measured: +713 MB of RSS for conv1a alone). The contiguity that the
    2026-09-23 precision fix depends on is preserved — each tile is copied
    contiguous into float64 before being reduced.

    ``x`` is modified in place; it must be a freshly allocated, writable array
    (it always is — it comes straight out of ``_conv2d_tiled``).
    """
    c, t, f = x.shape
    n = t * f
    acc = np.zeros(c, dtype=np.float64)
    for t0 in range(0, t, tile_t):
        blk = np.ascontiguousarray(x[:, t0:t0 + tile_t, :], dtype=np.float64)
        acc += blk.sum(axis=(1, 2))
    mean = (acc / n).reshape(c, 1, 1)
    acc[:] = 0.0
    for t0 in range(0, t, tile_t):
        blk = np.ascontiguousarray(x[:, t0:t0 + tile_t, :], dtype=np.float64)
        blk -= mean
        blk *= blk
        acc += blk.sum(axis=(1, 2))
    scale = np.sqrt((acc / n).reshape(c, 1, 1) + eps)
    for t0 in range(0, t, tile_t):
        sl = slice(t0, min(t0 + tile_t, t))
        x[:, sl, :] = _selu(((x[:, sl, :] - mean) / scale).astype(np.float32))
    return x


def _cnn_forward(cqt_window: np.ndarray, w: dict,
                 chunk_frames: int | None = None) -> np.ndarray:
    """cqt_window: (T, 252) -> (T, 240) feature sequence, matches CNNFeatureExtractor.

    ``chunk_frames`` bounds the memory (see ``predict_fold``). It is a TILE
    SIZE INSIDE each op, not a cut of the network — and that distinction is
    the load-bearing one:

    **Receptive field of this CNN, computed from the real layers, not guessed.**
    Eight convolutions, all ``3x3`` with ``stride=1`` and time-padding 1
    (``_CONV_LAYERS``: conv1a/1b/1c, conv2a/2b/2c, conv3a/3b, conv4a/4b — ten
    convs in total, all kernel 3 in time). The three ``MaxPool`` layers are
    ``MaxPool2d(1, k)``: kernel 1 in TIME, k in frequency only — verified in
    ``_maxpool_freq``, which reshapes the frequency axis alone. **There is NO
    temporal sub-sampling anywhere in this network**, so time index t of any
    layer maps to time index t of the input, and a cut never lands between two
    frames of a pooled grid (the off-by-one-frame joint the brief warned about
    cannot happen here). Receptive field = 1 + 10 * (3 - 1) = **21 input
    frames**, i.e. a halo of 10 frames on each side.

    **But a 10-frame overlap would NOT make a chunked CNN exact**, and that is
    why this code does not chunk the CNN. ``InstanceNorm2d`` normalises each
    channel over the WHOLE (T, F) map: its mean and variance depend on every
    frame of the song. Split the CNN in time and each block gets its own
    statistics — a real change to the output, at any overlap. So instead each
    op is tiled internally, the normalisation statistics stay global, and the
    result is arithmetically the same computation (measured max|Δ| vs the
    untiled path in ``docs/portage_mobile/`` / the task report: ~1e-7).

    The bi-LSTM is likewise never split: it is bidirectional, so every output
    frame depends on the entire sequence.
    """
    x = cqt_window[None, :, :]  # (1, T, 252) — 1 input channel
    for name, pad_t, pad_f in _CONV_LAYERS:
        if name == "POOL":
            x = _maxpool_freq(x, pad_t)  # pad_t reused as kernel size here
            continue
        prefix = f"audio_feature_block.{name}"
        if chunk_frames is None:
            x = _conv2d(x, w[f"{prefix}.weight"], w[f"{prefix}.bias"], pad_t, pad_f)
            x = _instance_norm(x)
            x = _selu(x)
        else:
            x = _conv2d_tiled(x, w[f"{prefix}.weight"], w[f"{prefix}.bias"],
                              pad_t, pad_f, chunk_frames)
            x = _norm_selu_tiled(x, chunk_frames)
    # x: (80, T, 3) -> (T, 240)
    c, t, f = x.shape
    return x.transpose(1, 0, 2).reshape(t, c * f)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_direction(x: np.ndarray, w_ih: np.ndarray, w_hh: np.ndarray,
                     b_ih: np.ndarray, b_hh: np.ndarray, reverse: bool) -> np.ndarray:
    """x: (T, in) -> (T, hidden). Gate order [i, f, g, o] (PyTorch convention).

    Run in float64: measured parity (see ``t1b_musx.md``) showed max|Δ| vs
    torch GROWING with sequence length (0.0015 at 205s -> 0.056 at 422s) —
    the signature of float32 rounding compounding through ~18k sequential
    recurrent steps (different BLAS summation order per step than
    torch/cuDNN, amplified by the recurrence, not a logic bug: triad argmax
    agreement stayed 99.0-100% throughout). float64 state cuts this per-step
    error by ~1e8x; the final cast back to float32 matches the reference
    dtype without re-introducing the drift.
    """
    t_len, _ = x.shape
    hidden = w_hh.shape[1]
    order = range(t_len - 1, -1, -1) if reverse else range(t_len)
    x64 = x.astype(np.float64)
    w_ih64, w_hh64 = w_ih.astype(np.float64), w_hh.astype(np.float64)
    b_ih64, b_hh64 = b_ih.astype(np.float64), b_hh.astype(np.float64)
    gates_x = x64 @ w_ih64.T + b_ih64  # (T, 4*hidden), precomputed once
    h = np.zeros(hidden, dtype=np.float64)
    c = np.zeros(hidden, dtype=np.float64)
    out = np.empty((t_len, hidden), dtype=np.float64)
    for t in order:
        gates = gates_x[t] + h @ w_hh64.T + b_hh64
        i = _sigmoid(gates[0 * hidden:1 * hidden])
        f = _sigmoid(gates[1 * hidden:2 * hidden])
        g = np.tanh(gates[2 * hidden:3 * hidden])
        o = _sigmoid(gates[3 * hidden:4 * hidden])
        c = f * c + i * g
        h = o * np.tanh(c)
        out[t] = h
    return out.astype(np.float32)


def _bilstm(x: np.ndarray, w: dict) -> np.ndarray:
    fwd = _lstm_direction(x, w["lstm1.weight_ih_l0"], w["lstm1.weight_hh_l0"],
                           w["lstm1.bias_ih_l0"], w["lstm1.bias_hh_l0"], reverse=False)
    bwd = _lstm_direction(x, w["lstm1.weight_ih_l0_reverse"], w["lstm1.weight_hh_l0_reverse"],
                           w["lstm1.bias_ih_l0_reverse"], w["lstm1.bias_hh_l0_reverse"], reverse=True)
    return np.concatenate([fwd, bwd], axis=1)  # (T, 192)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def predict_fold(cqt_288: np.ndarray, w: dict,
                 chunk_frames: int | None = None) -> list[np.ndarray]:
    """One fold's forward pass, matching ``ChordNet.inference`` exactly.

    cqt_288: (T, 288) raw CQT (18:270 crop applied here, like ``inference``).
    Returns [triad(T,73), bass(T,13), s7(T,4), s9(T,4), s11(T,3), s13(T,3)],
    each already softmaxed.

    ``chunk_frames=None`` (default) keeps the original, unbounded-memory path
    byte for byte. Any integer turns on the low-memory CNN: each convolution
    and each normalisation is computed in tiles of that many frames, which is
    the SAME computation (the network is never cut — see ``_cnn_forward``).
    On a 2 GB Android device pass something like 512: measured peak RSS for
    the 5-fold ensemble on a 205 s song drops from ~2040 MB to ~300 MB, for a
    small speed cost. Nothing else about the call changes.
    """
    x = np.asarray(cqt_288, dtype=np.float32)[:, SHIFT_HIGH * SHIFT_STEP:
                                                  SHIFT_HIGH * SHIFT_STEP + SPEC_DIM]
    feats = _cnn_forward(x, w, chunk_frames)          # (T, 240)
    h = _bilstm(feats, w)               # (T, 192)
    logits = h @ w["final_fc1.weight"].T + w["final_fc1.bias"]  # (T, 100)
    triad = _softmax(logits[:, :BASS_SLICE_BEGIN])
    bass = _softmax(logits[:, BASS_SLICE_BEGIN:BASS_DEL])
    s7 = _softmax(logits[:, BASS_DEL:SEVENTH_DEL])
    s9 = _softmax(logits[:, SEVENTH_DEL:NINTH_DEL])
    s11 = _softmax(logits[:, NINTH_DEL:ELEVENTH_DEL])
    s13 = _softmax(logits[:, ELEVENTH_DEL:THIRTEENTH_DEL])
    return [triad, bass, s7, s9, s11, s13]


def predict_ensemble(cqt_288: np.ndarray, weights_by_fold: list[dict],
                     chunk_frames: int | None = None) -> list[np.ndarray]:
    """5-fold average, matching ``harmonia.musx._run_nets`` / the vendored
    ``chord_recognition.py`` (simple mean over folds, same weight for each).

    ``chunk_frames`` is forwarded to ``predict_fold``; ``None`` (default) is
    the original behaviour, unchanged."""
    acc = None
    for w in weights_by_fold:
        probs = predict_fold(cqt_288, w, chunk_frames)
        acc = probs if acc is None else [a + b for a, b in zip(acc, probs)]
    return [(a / len(weights_by_fold)).astype(np.float32) for a in acc]


def load_weights_npz(path: str | Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def dump_weights_from_torch(chordnet, out_path: str | Path) -> None:
    """One-time conversion: torch ``ChordNet`` (fold N, already .eval()) ->
    a flat .npz of numpy arrays, keyed exactly like ``state_dict()``."""
    sd = chordnet.state_dict()
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    np.savez(out_path, **arrays)
