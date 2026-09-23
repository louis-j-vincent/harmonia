"""Diagnostic rapide : le diff int8 vs float32 de `parity_int8.py` compare les
battues PAR INDICE (a[0] vs b[0], a[1] vs b[1], …) — si le compte diffère
(369 vs 322 sur This Love), UN SEUL pic en trop au début décale tout le
reste et fabrique un "écart max" énorme qui ne dit rien sur la qualité
réelle du modèle int8. Ce script refait la comparaison par PLUS PROCHE
VOISIN (comme un vrai score de battues) pour trancher : le modèle int8
est-il vraiment cassé, ou est-ce un artefact de comparaison ?
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "mobile"))

import harmonia_mobile.beats_infer as beats_infer_mod
from harmonia_mobile.beats_infer import beats_from_audio

MODELS = Path(__file__).resolve().parents[2] / "models"
FLOAT_PTE = MODELS / "beat_this_final0_fixed1500.pte"
INT8_PTE = MODELS / "beat_this_final0_int8_fixed1500.pte"
WAV = Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav")


def nearest_diff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return None
    d = np.abs(a[:, None] - b[None, :]).min(axis=1)
    return d


def main():
    import soundfile as sf
    import soxr
    y, sr = sf.read(str(WAV), dtype="float32")
    if sr != 22050:
        y = soxr.resample(y, sr, 22050)
    beats_infer_mod._runtime_method = None
    beats_f, db_f = beats_from_audio(y, pte_path=FLOAT_PTE)
    beats_infer_mod._runtime_method = None
    beats_i, db_i = beats_from_audio(y, pte_path=INT8_PTE)

    print(f"float: {len(beats_f)} beats, {len(db_f)} downbeats")
    print(f"int8:  {len(beats_i)} beats, {len(db_i)} downbeats")
    d = nearest_diff(beats_f, beats_i) * 1000
    print(f"nearest-neighbor |Δt| beats (ms): max={d.max():.1f} median={np.median(d):.1f} "
          f"mean={d.mean():.1f} n>50ms={int((d>50).sum())}/{len(d)}")
    # first 10 beat times, both models, to eyeball where they diverge
    print("first 15 float beats: ", [round(x, 2) for x in beats_f[:15]])
    print("first 15 int8 beats:  ", [round(x, 2) for x in beats_i[:15]])
    print("last 10 float beats: ", [round(x, 2) for x in beats_f[-10:]])
    print("last 10 int8 beats:  ", [round(x, 2) for x in beats_i[-10:]])


if __name__ == "__main__":
    main()
