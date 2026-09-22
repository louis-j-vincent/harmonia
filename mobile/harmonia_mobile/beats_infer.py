"""Beat This! (final0) end-to-end on ExecuTorch: chunking, .pte inference,
aggregation, and the "minimal" postprocessor -- all pure numpy + the
ExecuTorch Python runtime, no torch import needed at call time (torch is
only used by the export script, once, offline).

Reproduces three things read from `beat_this` 1.1.0
(`~/harmonia/.venv-mobile/lib/python3.12/site-packages/beat_this/`), each
verified against the torch original in `mobile/tools/parity_beats.py`:

1. `inference.split_piece` / `split_predict_aggregate` -- the model only
   accepts fixed 1500-frame (30 s) chunks (this is a hard limit of the
   EXPORTED graph, confirmed in P0-T1: dynamic-time export specializes past
   1500 anyway), so a full song is split into overlapping 1500-frame
   windows with a 6-frame border, run one at a time, and stitched back with
   `overlap_mode="keep_first"`.
2. the `final0` network itself, via the ExecuTorch `.pte` (see
   `mobile/tools/export_beat_this.py`).
3. `model.postprocessor.Postprocessor(type="minimal")` -- max-pool peak
   picking, dedup, downbeat-snapped-to-nearest-beat. `dbn=False` in
   `harmonia/beats.py` means this is the ONLY postprocessor the live
   pipeline ever uses; the "dbn" branch (madmom) is not ported.

`beats_from_audio` is the one entry point the rest of the mobile pipeline
should call.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .beats_frontend import log_mel_spectrogram, FPS

CHUNK_SIZE = 1500
BORDER_SIZE = 6
DEFAULT_PTE = Path(__file__).resolve().parents[1] / "models" / "beat_this_final0_fixed1500.pte"

_runtime_method = None  # loaded once per process


def _load_method(pte_path: Path | str = DEFAULT_PTE):
    global _runtime_method
    if _runtime_method is None:
        from executorch.runtime import Runtime
        rt = Runtime.get()
        program = rt.load_program(str(pte_path))
        _runtime_method = program.load_method("forward")
    return _runtime_method


def _run_pte(spect_chunk: np.ndarray, pte_path: Path | str = DEFAULT_PTE) -> tuple[np.ndarray, np.ndarray]:
    """spect_chunk: (1500, 128) float32 -> (beat_logits, downbeat_logits), each (1500,)."""
    import torch  # ExecuTorch's python runtime speaks torch.Tensor in/out; no autograd/graph work happens here
    method = _load_method(pte_path)
    x = torch.tensor(spect_chunk[None, :, :], dtype=torch.float32)  # (1, 1500, 128)
    out = method.execute((x,))
    beat = out[0].numpy()[0]       # (1500,)
    downbeat = out[1].numpy()[0]   # (1500,)
    return beat, downbeat


def split_piece(spect: np.ndarray, chunk_size: int = CHUNK_SIZE, border_size: int = BORDER_SIZE,
                 avoid_short_end: bool = True) -> tuple[list[np.ndarray], np.ndarray]:
    """Numpy port of `beat_this.inference.split_piece`. `spect`: (T, 128)."""
    n = len(spect)
    starts = np.arange(-border_size, n - border_size, chunk_size - 2 * border_size)
    if avoid_short_end and n > chunk_size - 2 * border_size:
        starts[-1] = n - (chunk_size - border_size)
    chunks = []
    for start in starts:
        lo = max(start, 0)
        hi = min(start + chunk_size, n)
        left = max(0, -start)
        right = max(0, min(border_size, start + chunk_size - n))
        piece = spect[lo:hi]
        if left or right:
            piece = np.pad(piece, ((left, right), (0, 0)), mode="constant")
        chunks.append(piece)
    return chunks, starts


def aggregate_prediction(pred_chunks: list[tuple[np.ndarray, np.ndarray]], starts: np.ndarray,
                          full_size: int, chunk_size: int, border_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Numpy port of `beat_this.inference.aggregate_prediction`, `overlap_mode="keep_first"`
    only (the only mode `Audio2Frames.spect2frames` ever uses)."""
    beat_out = np.full(full_size, -1000.0, dtype=np.float32)
    downbeat_out = np.full(full_size, -1000.0, dtype=np.float32)
    # keep_first: process chunks in REVERSE order so earlier excerpts overwrite later ones
    for start, (beat, downbeat) in reversed(list(zip(starts, pred_chunks))):
        if border_size > 0:
            beat = beat[border_size:-border_size]
            downbeat = downbeat[border_size:-border_size]
        lo = start + border_size
        hi = start + chunk_size - border_size
        beat_out[lo:hi] = beat
        downbeat_out[lo:hi] = downbeat
    return beat_out, downbeat_out


def _maxpool1d_same(x: np.ndarray, kernel: int = 7, pad: int = 3) -> np.ndarray:
    """np equivalent of `F.max_pool1d(x, kernel, stride=1, padding=pad)` on a 1D array,
    padding with -inf (matches torch's pad-then-pool, never picks a padding slot as max
    unless the whole window is -inf, which cannot happen here since real values are >= -1000)."""
    padded = np.pad(x, (pad, pad), mode="constant", constant_values=-np.inf)
    n = len(x)
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)  # (n, kernel)
    return windows.max(axis=1)


def _deduplicate_peaks(peaks: np.ndarray, width: int = 1) -> np.ndarray:
    """Exact numpy port of `beat_this.model.postprocessor.deduplicate_peaks`:
    merge runs of frame indices within `width` of each other into their mean."""
    if len(peaks) == 0:
        return np.array([], dtype=np.float64)
    result = []
    p = float(peaks[0])
    c = 1
    for p2 in peaks[1:]:
        p2 = float(p2)
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)


def postprocess_minimal(beat_logits: np.ndarray, downbeat_logits: np.ndarray,
                         fps: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """Numpy port of `Postprocessor(type="minimal")`: peak-pick (+/-70ms window == 7
    frames @ 50fps), threshold logit>0 (prob>0.5), dedup adjacent frames, convert to
    seconds, snap each downbeat time to its nearest beat time, dedup downbeats."""
    beat_pooled = _maxpool1d_same(beat_logits)
    downbeat_pooled = _maxpool1d_same(downbeat_logits)
    beat_peak = (beat_logits == beat_pooled) & (beat_logits > 0)
    downbeat_peak = (downbeat_logits == downbeat_pooled) & (downbeat_logits > 0)

    beat_frames = np.nonzero(beat_peak)[0]
    downbeat_frames = np.nonzero(downbeat_peak)[0]
    beat_frames = _deduplicate_peaks(beat_frames, width=1)
    downbeat_frames = _deduplicate_peaks(downbeat_frames, width=1)

    beat_time = beat_frames / fps
    downbeat_time = downbeat_frames / fps

    if len(beat_time) > 0:
        for i, d_time in enumerate(downbeat_time):
            idx = np.argmin(np.abs(beat_time - d_time))
            downbeat_time[i] = beat_time[idx]
    downbeat_time = np.unique(downbeat_time)
    return beat_time, downbeat_time


def frames_from_spect(spect: np.ndarray, pte_path: Path | str = DEFAULT_PTE) -> tuple[np.ndarray, np.ndarray]:
    """(T, 128) log-mel spectrogram -> (beat_logits, downbeat_logits), each (T,) float32.
    Numpy port of `Spect2Frames.spect2frames` (chunk_size=1500, border_size=6,
    overlap_mode="keep_first" -- the only settings `Audio2Beats` ever uses)."""
    chunks, starts = split_piece(spect, CHUNK_SIZE, BORDER_SIZE, avoid_short_end=True)
    pred_chunks = [_run_pte(c, pte_path) for c in chunks]
    beat_logits, downbeat_logits = aggregate_prediction(pred_chunks, starts, len(spect), CHUNK_SIZE, BORDER_SIZE)
    return beat_logits, downbeat_logits


def beats_from_audio(y22050: np.ndarray, pte_path: Path | str = DEFAULT_PTE) -> tuple[np.ndarray, np.ndarray]:
    """The single entry point: mono float32 waveform @ 22050 Hz -> (beats, downbeats),
    each an array of times in seconds. Matches `beat_this.inference.File2Beats(
    device="cpu", dbn=False)(audio_path)`'s return order and values (see
    `mobile/tools/parity_beats.py`)."""
    spect = log_mel_spectrogram(y22050, sample_rate=22050)
    beat_logits, downbeat_logits = frames_from_spect(spect, pte_path)
    beats, downbeats = postprocess_minimal(beat_logits, downbeat_logits)
    return beats, downbeats
