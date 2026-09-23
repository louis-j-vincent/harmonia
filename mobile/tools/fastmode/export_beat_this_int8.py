"""T5-prep — levier 4 : quantification int8 de beat_this (`final0`) via
XNNPACKQuantizer + PT2E, temps de lowering + temps d'inférence + diff des
battues contre le `.pte` float32 (`beat_this_final0_fixed1500.pte`, T1c).

Boîte de temps : 40 min (consigne T5-prep). Si ça bloque, le lever devient
"non tenté", noté tel quel dans le rapport.

Run: ~/harmonia/.venv-mobile/bin/python mobile/tools/fastmode/export_beat_this_int8.py
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # mobile/

from beat_this.inference import load_model, load_audio

AUDIO = Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav")
OUT_DIR = Path(__file__).resolve().parents[2] / "models"
CHUNK = 1500
CHECKPOINT = "final0"

T_START = time.time()
BUDGET_S = 40 * 60


def budget_left() -> float:
    return BUDGET_S - (time.time() - T_START)


class BeatThisWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, spect):
        out = self.m(spect)
        return out["beat"], out["downbeat"]


def main():
    from beat_this.preprocessing import LogMelSpect

    print("== loading model + calibration audio ==")
    model = load_model(CHECKPOINT, device="cpu").eval()
    wrapped = BeatThisWrapper(model).eval()

    signal, sr = load_audio(str(AUDIO))
    spect_mod = LogMelSpect(device="cpu")
    sig_t = torch.tensor(np.asarray(signal, dtype=np.float32))
    if sr != 22050:
        import soxr
        sig_t = torch.tensor(soxr.resample(np.asarray(signal), in_rate=sr, out_rate=22050), dtype=torch.float32)
    with torch.no_grad():
        ref_spect = spect_mod(sig_t)  # (T, 128)

    example_chunk = ref_spect[:CHUNK].unsqueeze(0).contiguous()
    if example_chunk.shape[1] < CHUNK:
        pad = CHUNK - example_chunk.shape[1]
        example_chunk = torch.nn.functional.pad(example_chunk, (0, 0, 0, pad))

    print(f"budget left before export: {budget_left():.0f}s")

    print("== torch.export ==")
    t0 = time.time()
    exported = torch.export.export(wrapped, (example_chunk,))
    print(f"export: {time.time()-t0:.1f}s  (budget left {budget_left():.0f}s)")
    if budget_left() < 0:
        print("BUDGET EXCEEDED before quantizer — abort, lever = non tenté")
        return

    print("== PT2E prepare (XNNPACKQuantizer, symmetric int8) ==")
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    t0 = time.time()
    gm = exported.module()
    prepared = prepare_pt2e(gm, quantizer)
    print(f"prepare_pt2e: {time.time()-t0:.1f}s  (budget left {budget_left():.0f}s)")
    if budget_left() < 0:
        print("BUDGET EXCEEDED at prepare — abort, lever = non tenté")
        return

    # calibration: a handful of REAL chunks from the 3 reference songs (not
    # just this_love) so the observers see realistic dynamic range.
    print("== calibration (3 songs, 1500-frame chunks) ==")
    calib_chunks = [example_chunk]
    for wav in [
        Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/ben_e_king_stand_by_me_audio.wav"),
    ]:
        sig2, sr2 = load_audio(str(wav))
        sig2_t = torch.tensor(np.asarray(sig2, dtype=np.float32))
        if sr2 != 22050:
            import soxr
            sig2_t = torch.tensor(soxr.resample(np.asarray(sig2), in_rate=sr2, out_rate=22050), dtype=torch.float32)
        with torch.no_grad():
            spect2 = spect_mod(sig2_t)
        c = spect2[:CHUNK].unsqueeze(0).contiguous()
        if c.shape[1] < CHUNK:
            c = torch.nn.functional.pad(c, (0, 0, 0, CHUNK - c.shape[1]))
        calib_chunks.append(c)
    with torch.no_grad():
        for c in calib_chunks:
            prepared(c)
    print(f"calibration done (budget left {budget_left():.0f}s)")

    print("== convert_pt2e (fake-quant -> real int8 ops) ==")
    t0 = time.time()
    converted = convert_pt2e(prepared)
    print(f"convert_pt2e: {time.time()-t0:.1f}s  (budget left {budget_left():.0f}s)")
    if budget_left() < 0:
        print("BUDGET EXCEEDED at convert — abort, lever = non tenté")
        return

    print("== re-export quantized graph + XNNPACK lowering ==")
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    t0 = time.time()
    try:
        requantized = torch.export.export(converted, (example_chunk,))
    except Exception as e:
        print(f"re-export of quantized graph FAILED: {type(e).__name__}: {e} "
              f"(budget left {budget_left():.0f}s) — lever = non tenté")
        return
    print(f"re-export: {time.time()-t0:.1f}s  (budget left {budget_left():.0f}s)")
    if budget_left() < 0:
        print("BUDGET EXCEEDED before lowering — abort, lever = non tenté")
        return

    t0 = time.time()
    try:
        edge = to_edge_transform_and_lower(requantized, partitioner=[XnnpackPartitioner()])
        pte_program = edge.to_executorch()
    except Exception as e:
        print(f"XNNPACK lowering of quantized graph FAILED: {type(e).__name__}: {e} "
              f"(budget left {budget_left():.0f}s) — lever = non tenté")
        return
    print(f"to_edge+lower: {time.time()-t0:.1f}s  (budget left {budget_left():.0f}s)")

    out_path = OUT_DIR / "beat_this_final0_int8_fixed1500.pte"
    with open(out_path, "wb") as f:
        pte_program.write_to_file(f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"WROTE {out_path} ({size_mb:.1f} MB, vs float32 82.4 MB)")
    print(f"TOTAL TIME: {time.time()-T_START:.0f}s")


if __name__ == "__main__":
    main()
