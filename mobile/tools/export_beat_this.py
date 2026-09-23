"""Export Beat This! (final0, the REAL checkpoint) to ExecuTorch .pte, and
compare to torch reference.

Run with the MOBILE venv (~/harmonia/.venv-mobile), never .venv:
    ~/harmonia/.venv-mobile/bin/python mobile/tools/export_beat_this.py

2026-09-23 (T1c) CORRECTION: T1 exported "small0" (9 MB). But the live
pipeline (`harmonia/beats.py:46`, `File2Beats(device="cpu", dbn=False)`)
never names a checkpoint, so `beat_this.inference.load_model` defaults to
"final0" (transformer_dim=512 vs small0's 128; 20.25M params vs 2.10M;
~81 MB .ckpt) -- verified by reading `load_model`'s signature
(`checkpoint_path: str | None = "final0"`). The phone must render the SAME
beats as the Mac, so THIS is the checkpoint that ships; small0 stays a
possible "fast mode" for later, Louis's call (CLAUDE.md rule 6).

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

import sys

from beat_this.inference import load_model, load_audio
from beat_this.preprocessing import LogMelSpect

AUDIO = Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav")
OUT_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = "final0"  # the ONLY checkpoint harmonia/beats.py actually loads (load_model's default)
CHUNK = 1500  # beat_this's fixed training/inference chunk length in frames (see inference.py split_predict_aggregate)


def export_network(model: torch.nn.Module, example: torch.Tensor, name: str,
                    dynamic_time: bool, arg_name: str = "x", time_dim: int = 0,
                    backend: str = "xnnpack"):
    """torch.export -> to_edge -> (XNNPACK partition) -> .pte.

    `backend="portable"` saute le partitionneur : le graphe reste en
    opérateurs portables, interprétés noeud par noeud par le runtime. C'est
    PLUS LENT (T1b l'a mesuré sur musx) et ce n'est PAS ce qui tournera sur
    le téléphone de Louis.

    Pourquoi ce mode existe quand même (P2d, 2026-09-23) : l'ÉMULATEUR ne
    peut pas exécuter les noyaux XNNPACK. Son CPU invité annonce `sme2`
    (l'hyperviseur d'Apple laisse passer les bits de la puce hôte) mais
    l'instruction SME `rdsvl`, que XNNPACK exécute une fois à
    l'initialisation pour lire la longueur de vecteur en streaming, y lève un
    SIGILL. C'est une limite de l'émulateur, pas du portage — mais sans ce
    mode, AUCUN chart ne peut être produit sur l'émulateur, et la porte de
    la phase 2 ne pourrait pas être franchie.

    Returns (pte_path, edge_program) or (None, exception).
    """
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
        if backend == "portable":
            # PAS `to_edge_transform_and_lower(..., partitioner=[])` : avec une
            # liste vide, cette fonction casse plus loin (« expected
            # compiled_fn to be GraphModule »). Le chemin sans délégué est
            # `to_edge` tout court.
            from executorch.exir import to_edge
            edge = to_edge(exported)
        else:
            edge = to_edge_transform_and_lower(exported,
                                               partitioner=[XnnpackPartitioner()])
        pte_program = edge.to_executorch()
    except Exception as e:
        return None, f"to_edge/{backend} failed: {type(e).__name__}: {e}"
    t2 = time.time()
    pte_path = OUT_DIR / f"{name}.pte"
    with open(pte_path, "wb") as f:
        pte_program.write_to_file(f)
    print(f"[{name}] export {t1-t0:.1f}s, to_edge+lower {t2-t1:.1f}s, size {pte_path.stat().st_size/1e6:.2f} MB")
    return pte_path, edge


def main():
    # --portable : n'exporte QUE le réseau en forme fixe, sans XNNPACK (voir
    # la docstring d'export_network). Les étapes « mel » et « forme
    # dynamique » ne servent pas ce cas et coûteraient des minutes.
    if "--portable" in sys.argv:
        return main_portable()
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

    model = load_model(CHECKPOINT, device="cpu")
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

    pte2, info2 = export_network(wrapped, example_chunk, "beat_this_final0_fixed1500", dynamic_time=False,
                                  arg_name="spect", time_dim=1)
    if pte2 is None:
        print("NETWORK (fixed-shape) EXPORT FAILED:", info2)
    else:
        print("NETWORK (fixed-shape) EXPORT OK ->", pte2)

    # ---- 3) if fixed-shape worked, try dynamic time dimension (real audio isn't always 1500 frames) ----
    if pte2 is not None:
        print("\n== attempt: export BeatThis network with DYNAMIC time dim ==")
        pte3, info3 = export_network(wrapped, example_chunk, "beat_this_final0_dynamic", dynamic_time=True,
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




def main_portable():
    """Le MÊME réseau, la MÊME forme fixe (1, 1500, 128), sans partitionneur.

    Sert uniquement à faire tourner l'émulateur (voir `export_network`). Le
    `.pte` produit s'appelle `beat_this_final0_fixed1500_portable.pte` — le
    nom dit ce qu'il est, pour qu'on ne l'embarque jamais par erreur en
    croyant mesurer la vitesse du téléphone.
    """
    print("== export PORTABLE (sans XNNPACK) du réseau final0, forme fixe ==")
    # MÊME PRÉAMBULE QUE `main`, volontairement. Un export minimal
    # (`torch.export.export(wrapper, (torch.zeros(1,1500,128),))`) échoue dans
    # ce venv avec « expected compiled_fn to be GraphModule », alors que le
    # chemin de `main` passe — la décomposition de torch 2.14 dépend donc de
    # quelque chose que fait le préambule (passe avant sous `no_grad`, exemple
    # réel plutôt que des zéros). Plutôt que de deviner lequel, on refait le
    # même chemin : c'est lui qui est validé.
    signal, sr = load_audio(str(AUDIO))
    spect_mod = LogMelSpect(device="cpu")
    sig_t = torch.tensor(np.asarray(signal, dtype=np.float32))
    if sr != 22050:
        import soxr
        sig_t = torch.tensor(soxr.resample(np.asarray(signal), in_rate=sr,
                                           out_rate=22050), dtype=torch.float32)
    with torch.no_grad():
        ref_spect = spect_mod(sig_t)
    model = load_model(CHECKPOINT, device="cpu")
    model.eval()

    class BeatThisWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, spect):
            out = self.m(spect)
            return out["beat"], out["downbeat"]

    wrapped = BeatThisWrapper(model).eval()
    example_chunk = ref_spect[:CHUNK].unsqueeze(0).contiguous()
    with torch.no_grad():
        wrapped(example_chunk)
    pte, info = export_network(wrapped, example_chunk,
                               "beat_this_final0_fixed1500_portable",
                               dynamic_time=False, arg_name="spect",
                               time_dim=1, backend="portable")
    if pte is None:
        print("PORTABLE EXPORT FAILED:", info)
        return 1
    print("PORTABLE EXPORT OK ->", pte)
    return 0


if __name__ == "__main__":
    main()
