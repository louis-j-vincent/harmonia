"""Parity: .pte (ExecuTorch python runtime) vs torch reference, on real tensors
saved by export_beat_this.py / export_musx.py. Run with the MOBILE venv.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"


def run_pte(pte_path: Path, inputs: tuple[np.ndarray, ...]):
    from executorch.runtime import Runtime
    import torch

    rt = Runtime.get()
    program = rt.load_program(str(pte_path))
    method = program.load_method("forward")
    t_inputs = tuple(torch.tensor(x) for x in inputs)
    t0 = time.time()
    out = method.execute(t_inputs)
    t1 = time.time()
    return [o.numpy() if hasattr(o, "numpy") else np.asarray(o) for o in out], (t1 - t0)


def beat_this_parity():
    print("\n===== beat_this parity =====")
    ref = np.load(MODELS / "beat_this_reference.npz")
    example_chunk = ref["example_chunk"]  # (1, 1500, 128)
    ref_beat = ref["ref_beat"]
    ref_downbeat = ref["ref_downbeat"]

    for name in ["beat_this_small0_fixed1500", "beat_this_small0_dynamic"]:
        pte_path = MODELS / f"{name}.pte"
        if not pte_path.exists():
            print(f"[{name}] .pte missing, skip")
            continue
        outs, dt = run_pte(pte_path, (example_chunk,))
        pte_beat, pte_downbeat = outs
        d_beat = np.abs(pte_beat - ref_beat).max()
        d_downbeat = np.abs(pte_downbeat - ref_downbeat).max()
        print(f"[{name}] inference {dt*1000:.1f} ms | max|Δ beat logits|={d_beat:.3e} "
              f"max|Δ downbeat logits|={d_downbeat:.3e}")

        # decode to beat TIMES the same way beat_this's Postprocessor("minimal") does,
        # and compare times directly (what actually reaches the pipeline).
        from beat_this.model.postprocessor import Postprocessor
        import torch
        post = Postprocessor(type="minimal")
        bt_ref, dt_ref = post(torch.tensor(ref_beat), torch.tensor(ref_downbeat))
        bt_pte, dt_pte = post(torch.tensor(pte_beat), torch.tensor(pte_downbeat))
        bt_ref, bt_pte = np.asarray(bt_ref[0]), np.asarray(bt_pte[0])
        n = min(len(bt_ref), len(bt_pte))
        if len(bt_ref) != len(bt_pte):
            print(f"    !! different beat COUNT: torch={len(bt_ref)} pte={len(bt_pte)}")
        if n > 0:
            time_diff_ms = np.abs(bt_ref[:n] - bt_pte[:n]) * 1000
            print(f"    beat times: n_common={n}, max|Δt|={time_diff_ms.max():.2f} ms, "
                  f"mean|Δt|={time_diff_ms.mean():.2f} ms")
        else:
            print("    no beats decoded on this chunk (silence/short excerpt) — inconclusive")


def musx_parity():
    print("\n===== musx parity =====")
    ref_path = MODELS / "musx_reference.npz"
    if not ref_path.exists():
        print("musx_reference.npz missing (export_musx.py didn't get far enough) — skip")
        return
    ref = np.load(ref_path)
    cqt = ref["cqt_window"]
    ref_triad = ref["ref_triad"]

    pte_path = MODELS / "musx_fold0.pte"
    if not pte_path.exists():
        print("musx_fold0.pte missing — skip")
        return
    outs, dt = run_pte(pte_path, (cqt,))
    pte_triad = outs[0]
    d = np.abs(pte_triad - ref_triad).max()
    argmax_ref = ref_triad.argmax(axis=-1)
    argmax_pte = pte_triad.argmax(axis=-1)
    agree = (argmax_ref == argmax_pte).mean()
    print(f"[musx_fold0] inference {dt*1000:.1f} ms | max|Δ triad posterior|={d:.3e} | "
          f"argmax-per-frame agreement={agree*100:.1f}%")


if __name__ == "__main__":
    beat_this_parity()
    musx_parity()
