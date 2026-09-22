"""Generate NNLS-Chroma VAMP-plugin references (Mac only, real `vamp.collect`)
for the P0-T2 parity check against `harmonia_mobile.nnls_chroma` (numpy port).

Run with the worktree's `.venv` (has `vamp` + the native plugin installed) --
NOT the mobile venv, and installs nothing. Saves one .npz per song under
mobile/tools/nnls_ref/ (arr, times, sr, step, duration) -- kept as a single
.npz per song (arr+times+metadata together) rather than loose .npy files,
same data, one file to keep track of.

    .venv/bin/python mobile/tools/gen_nnls_ref.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import librosa
import vamp

SR = 44100
AUDIO_DIR = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
OUT_DIR = Path(__file__).resolve().parent / "nnls_ref"

SONGS = {
    "maroon_5_this_love": AUDIO_DIR / "maroon_5_this_love.m4a",
    "ben_e_king_stand_by_me_audio": AUDIO_DIR / "ben_e_king_stand_by_me_audio.m4a",
    "autumn_leaves": AUDIO_DIR / "autumn_leaves.m4a",  # jazz standard, the 3rd song
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in SONGS.items():
        t0 = time.time()
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        t1 = time.time()
        out = vamp.collect(y.astype(np.float32), SR, "nnls-chroma:nnls-chroma",
                            output="bothchroma")
        step, arr = out["matrix"]
        arr = np.asarray(arr, np.float32)
        times = np.arange(arr.shape[0]) * float(step)
        t2 = time.time()
        np.savez(OUT_DIR / f"{name}.npz", arr=arr, times=times, sr=SR,
                  step=float(step), duration=len(y) / SR)
        print(f"{name}: load {t1-t0:.2f}s, vamp {t2-t1:.2f}s, arr {arr.shape}, "
              f"step {float(step):.6f}s, dur {len(y)/SR:.1f}s")


if __name__ == "__main__":
    main()
