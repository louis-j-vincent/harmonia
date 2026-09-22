"""Export ONE musx fold (ChordNet, fold 0) to ExecuTorch .pte, compare to torch.

The CQT front-end (librosa.core.hybrid_cqt, third_party/musx_ismir2019/extractors/cqt.py)
is NOT torch — it can never be captured by torch.export. This script therefore
takes a PRE-COMPUTED CQT (from harmonia's own disk cache, same extractor, same
song) as the network's input, exactly like ChordNet.inference(cqt) does in
harmonia/musx.py::_run_nets. That is the real boundary: audio -> CQT is numpy/
librosa work that has to be redone for Android; CQT -> chord posteriors is the
network this script exports.

Run with the MOBILE venv:
    ~/harmonia/.venv-mobile/bin/python mobile/tools/export_musx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/.claude/worktrees/mobile")
MUSX_DIR = REPO / "third_party" / "musx_ismir2019"
OUT_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-computed CQT cache (real production cache, same extractor as CQTV2) --
# avoids needing librosa/numba in the mobile venv just to get an input tensor.
CQT_CACHE = Path("/Users/vincente/harmonia/data/cache/musx_cqt/maroon_5_this_love.npz")

SHIFT_HIGH = 6
SHIFT_STEP = 3
SPEC_DIM = 252
FOLD = 0
MODEL_NAME = f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{FOLD}.best"

# window of frames to export/parity-check on (keep the graph + comparison cheap)
WINDOW_FRAMES = 500


class ChordNetInference(torch.nn.Module):
    """Reproduces chordnet_ismir_naive.ChordNet.inference() as a single forward(),
    so torch.export sees one static call graph (no python-level slicing/softmax
    happening OUTSIDE the traced graph, matching mobile execution exactly).
    Only the triad output is returned (6-way -> parity focuses on 'argmax per
    frame' of the triad plane, as asked); the other 5 heads export the same way.
    """

    def __init__(self, chordnet: torch.nn.Module):
        super().__init__()
        self.net = chordnet

    def forward(self, cqt_288: torch.Tensor) -> torch.Tensor:
        # cqt_288: (seq_length, 288) -> center 252-bin crop, exactly as ChordNet.inference
        x = cqt_288[:, SHIFT_HIGH * SHIFT_STEP: SHIFT_HIGH * SHIFT_STEP + SPEC_DIM]
        seq_length = x.shape[0]
        x = x.view((1, seq_length, SPEC_DIM))
        triad_logits = self.net(x)[0]
        return F.softmax(triad_logits, dim=1)


class InMusxDir:
    def __enter__(self):
        self._cwd = Path.cwd()
        import os
        os.chdir(MUSX_DIR)
        if str(MUSX_DIR) not in sys.path:
            sys.path.insert(0, str(MUSX_DIR))
        return self

    def __exit__(self, *exc):
        import os
        os.chdir(self._cwd)
        return False


def load_fold(fold: int):
    with InMusxDir():
        from mir.nn.train import NetworkInterface
        from chordnet_ismir_naive import ChordNet
        ni = NetworkInterface(ChordNet(None), MODEL_NAME, load_checkpoint=False)
        print("finalized (weights loaded from .sdict)?", ni.finalized)
        return ni.net.eval()


def export_network(model, example, name):
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    t0 = time.time()
    try:
        exported = torch.export.export(model, (example,))
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
    print(f"[{name}] export {t1-t0:.1f}s, to_edge+lower {t2-t1:.1f}s, "
          f"size {pte_path.stat().st_size/1e6:.2f} MB")
    return pte_path, edge


def main():
    print("== loading cached CQT ==", CQT_CACHE)
    z = np.load(CQT_CACHE)
    cqt = z["cqt"] if "cqt" in z.files else z[z.files[0]]
    print("cqt shape", cqt.shape, cqt.dtype)  # expect (n_frame, 288)

    window = cqt[:WINDOW_FRAMES].astype(np.float32)
    cqt_t = torch.tensor(window)

    print("== loading fold", FOLD, "==")
    chordnet = load_fold(FOLD)
    wrapper = ChordNetInference(chordnet).eval()

    with torch.no_grad():
        ref_triad = wrapper(cqt_t)
    print("ref_triad", ref_triad.shape)

    print("\n== attempt: export ChordNetInference (fold 0, fixed", WINDOW_FRAMES, "frames) ==")
    pte, info = export_network(wrapper, cqt_t, f"musx_fold{FOLD}")
    if pte is None:
        print("MUSX EXPORT FAILED:", info)
    else:
        print("MUSX EXPORT OK ->", pte)

    np.savez(OUT_DIR / "musx_reference.npz", cqt_window=window, ref_triad=ref_triad.detach().numpy())
    print("saved reference ->", OUT_DIR / "musx_reference.npz")


if __name__ == "__main__":
    main()
