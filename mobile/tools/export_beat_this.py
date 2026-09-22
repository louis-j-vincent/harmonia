"""Export Beat This! (small0) to ExecuTorch .pte, and compare to torch reference.

Run with the MOBILE venv (~/harmonia/.venv-mobile), never .venv:
    ~/harmonia/.venv-mobile/bin/python mobile/tools/export_beat_this.py

Two things are exported separately, because torch.stft/mel-filterbank are NOT
guaranteed to lower to XNNPACK (see report):
  1. the LogMelSpect front-end alone (waveform -> log-mel), to see whether it
     even survives torch.export + to_edge on its own.
  2. the BeatThis network alone (log-mel spect -> beat/downbeat logits), which
     is the actual "network" of the task.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from beat_this.inference import load_model, load_audio
from beat_this.preprocessing import LogMelSpect

AUDIO = Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav")
OUT_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 1500  # beat_this's fixed training/inference chunk length in frames (see inference.py split_predict_aggregate)


def export_network(model: torch.nn.Module, example: torch.Tensor, name: str,
                    dynamic_time: bool, arg_name: str = "x", time_dim: int = 0):
    """torch.export -> to_edge -> XNNPACK partition -> .pte. Returns (pte_path, edge_program) or (None, exception)."""
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    dynamic_shapes = None
    if dynamic_time:
        max_dim = 10_000_000 if arg_name == "x" else 1500
        min_dim = 1024 if arg_name == "x" else 8
        t = torch.export.Dim("time", min=min_dim, max=max_dim)
        dynamic_shapes = {arg_name: {time_dim: t}}

    t0 = time.time()
    try:
        exported = torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes)
    except Exception as e:
        return None, f"torch.export failed: {type(e).__name__}: {e}"
    t1 = time.time()
    try:
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        pte_program = edge.to_executorch()
    except Exception as e:
        return None, f"to_edge/XNNPACK failed: {type(e).__name__}: {e}"
    t2 = time.time()
    pte_path = OUT_DIR / f"{name}.pte"
    with open(pte_path, "wb") as f:
        pte_program.write_to_file(f)
    print(f"[{name}] export {t1-t0:.1f}s, to_edge+lower {t2-t1:.1f}s, size {pte_path.stat().st_size/1e6:.2f} MB")
    return pte_path, edge


def main():
    print("== loading audio ==")
    signal, sr = load_audio(str(AUDIO))
    print("sr", sr, "len", len(signal), "dur_s", len(signal) / sr)

    print("== building torch reference: waveform -> log-mel -> model ==")
    spect_mod = LogMelSpect(device="cpu")
    sig_t = torch.tensor(np.asarray(signal, dtype=np.float32))
    if sr != 22050:
        import soxr
        sig_np = soxr.resample(np.asarray(signal), in_rate=sr, out_rate=22050)
        sig_t = torch.tensor(sig_np, dtype=torch.float32)
    with torch.no_grad():
        ref_spect = spect_mod(sig_t)  # (T, 128)
    print("ref_spect", ref_spect.shape, ref_spect.dtype)

    model = load_model("small0", device="cpu")
    model.eval()

    # ---- 1) try exporting the LogMelSpect front-end alone ----
    print("\n== attempt: export LogMelSpect (waveform -> log-mel) ==")
    example_wave = sig_t[:22050 * 10]  # 10s excerpt, keep export cheap
    pte, info = export_network(spect_mod, example_wave, "beat_this_logmel", dynamic_time=True,
                                arg_name="x", time_dim=0)
    if pte is None:
        print("LOGMEL EXPORT FAILED:", info)
    else:
        print("LOGMEL EXPORT OK ->", pte)

    # ---- 2) export the BeatThis network on a FIXED chunk size (1500 frames) ----
    print("\n== attempt: export BeatThis network (fixed 1500-frame chunk, static shape) ==")
    example_chunk = ref_spect[:CHUNK].unsqueeze(0).contiguous()  # (1, 1500, 128)
    if example_chunk.shape[1] < CHUNK:
        pad = CHUNK - example_chunk.shape[1]
        example_chunk = torch.nn.functional.pad(example_chunk, (0, 0, 0, pad))

    class BeatThisWrapper(torch.nn.Module):
        """model(x) returns a dict {'beat':..., 'downbeat':...} -> tuple for export."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, spect):
            out = self.m(spect)
            return out["beat"], out["downbeat"]

    wrapped = BeatThisWrapper(model).eval()
    with torch.no_grad():
        ref_beat, ref_downbeat = wrapped(example_chunk)
    print("ref_beat", ref_beat.shape, "ref_downbeat", ref_downbeat.shape)

    pte2, info2 = export_network(wrapped, example_chunk, "beat_this_small0_fixed1500", dynamic_time=False,
                                  arg_name="spect", time_dim=1)
    if pte2 is None:
        print("NETWORK (fixed-shape) EXPORT FAILED:", info2)
    else:
        print("NETWORK (fixed-shape) EXPORT OK ->", pte2)

    # ---- 3) if fixed-shape worked, try dynamic time dimension (real audio isn't always 1500 frames) ----
    if pte2 is not None:
        print("\n== attempt: export BeatThis network with DYNAMIC time dim ==")
        pte3, info3 = export_network(wrapped, example_chunk, "beat_this_small0_dynamic", dynamic_time=True,
                                      arg_name="spect", time_dim=1)
        if pte3 is None:
            print("NETWORK (dynamic-shape) EXPORT FAILED:", info3)
        else:
            print("NETWORK (dynamic-shape) EXPORT OK ->", pte3)

    # save torch references + example input for the parity script
    np.savez(
        OUT_DIR / "beat_this_reference.npz",
        example_chunk=example_chunk.numpy(),
        ref_beat=ref_beat.detach().numpy(),
        ref_downbeat=ref_downbeat.detach().numpy(),
        example_wave=example_wave.numpy(),
        ref_spect=ref_spect.numpy(),
    )
    print("\nsaved reference tensors ->", OUT_DIR / "beat_this_reference.npz")


if __name__ == "__main__":
    main()
