"""Inference speed of the musx fold-0 .pte (portable kernels, no XNNPACK),
ExecuTorch python runtime, 1 thread, chunked like the real network needs
(static export window = 500 frames = 500*512/22050 = 11.61 s of audio).

Reports per-chunk latency and the extrapolated cost for 4 minutes of audio,
for ONE fold; the real pipeline averages 5 folds (see musx.py::_run_nets),
so multiply by 5 for the ensemble (or run 5 exports — not done here, see
report for why).

Run with the MOBILE venv:
    ~/harmonia/.venv-mobile/bin/python mobile/tools/speed_musx_executorch.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

MODELS = Path(__file__).resolve().parents[1] / "models"
CQT_REF = Path(__file__).resolve().parent / "cqt_ref" / "maroon_5_this_love.npz"
WINDOW_FRAMES = 500


def main():
    from executorch.runtime import Runtime
    rt = Runtime.get()
    program = rt.load_program(str(MODELS / "musx_fold0_portable.pte"))
    method = program.load_method("forward")

    z = np.load(CQT_REF)
    cqt = z["cqt"].astype(np.float32)
    window = cqt[:WINDOW_FRAMES]
    t_input = (torch.tensor(window),)

    # warmup
    for _ in range(2):
        method.execute(t_input)

    n_runs = 5
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        out = method.execute(t_input)
        times.append(time.time() - t0)
    times = np.array(times)
    print(f"per-chunk ({WINDOW_FRAMES} frames = {WINDOW_FRAMES*512/22050:.2f}s audio) latency: "
          f"{times.mean()*1000:.1f} ms +/- {times.std()*1000:.1f} ms (min {times.min()*1000:.1f}, "
          f"max {times.max()*1000:.1f}), n={n_runs}")

    chunk_audio_s = WINDOW_FRAMES * 512 / 22050
    n_chunks_4min = int(np.ceil(240.0 / chunk_audio_s))
    total_1fold = times.mean() * n_chunks_4min
    total_5fold = total_1fold * 5
    print(f"4 min of audio = {n_chunks_4min} chunks of {chunk_audio_s:.2f}s")
    print(f"  -> 1 fold:  {total_1fold:.2f} s")
    print(f"  -> 5 folds (ensemble, naive x5): {total_5fold:.2f} s")


if __name__ == "__main__":
    main()
