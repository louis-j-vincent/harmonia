"""T5-prep — levier 4 : battues du `.pte` int8 (`beat_this_final0_int8_fixed1500.pte`)
vs le `.pte` float32 de référence (T1c, identique à torch), 3 morceaux, temps
+ écart max |Δt| + compte de battues.

Run: ~/harmonia/.venv-mobile/bin/python mobile/tools/fastmode/parity_int8.py OUT.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "mobile"))

import harmonia_mobile.beats_infer as beats_infer_mod
from harmonia_mobile.beats_infer import beats_from_audio


def _reset_pte_cache():
    # `_load_method` caches `_runtime_method` globally, ignoring `pte_path`
    # on subsequent calls (loaded once per process) — fine for the live
    # pipeline (one model, one process) but wrong here where we swap
    # float32/int8 in the SAME process. Reset between calls.
    beats_infer_mod._runtime_method = None

MODELS = Path(__file__).resolve().parents[2] / "models"
FLOAT_PTE = MODELS / "beat_this_final0_fixed1500.pte"
INT8_PTE = MODELS / "beat_this_final0_int8_fixed1500.pte"

SONGS = {
    "maroon_5_this_love": Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav"),
    "ben_e_king_stand_by_me_audio": Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/ben_e_king_stand_by_me_audio.wav"),
    "autumn_leaves": Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio/autumn_leaves.m4a"),
}


def load_wav_22050(path: Path) -> np.ndarray:
    import soundfile as sf
    if path.suffix.lower() == ".wav":
        y, sr = sf.read(str(path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr == 22050:
            return y
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/a.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                         "-ac", "1", "-ar", "22050", out], check=True)
        y, sr = sf.read(out, dtype="float32")
        assert sr == 22050
        return y


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("parity_int8.json")
    if not INT8_PTE.exists():
        out_path.write_text(json.dumps({"error": "int8 .pte absent — levier 4 non tenté"}))
        print("int8 .pte absent")
        return
    results = {}
    for stem, path in SONGS.items():
        y = load_wav_22050(path)
        _reset_pte_cache()
        t0 = time.perf_counter()
        beats_f, downbeats_f = beats_from_audio(y, pte_path=FLOAT_PTE)
        t_float = time.perf_counter() - t0
        _reset_pte_cache()
        t0 = time.perf_counter()
        beats_i, downbeats_i = beats_from_audio(y, pte_path=INT8_PTE)
        t_int8 = time.perf_counter() - t0

        def compare(a, b):
            n = min(len(a), len(b))
            if n == 0:
                return {"n_a": len(a), "n_b": len(b), "max_abs_diff_ms": None}
            d = np.abs(np.asarray(a[:n]) - np.asarray(b[:n])) * 1000
            return {"n_a": len(a), "n_b": len(b), "max_abs_diff_ms": float(d.max()),
                    "mean_abs_diff_ms": float(d.mean())}

        results[stem] = {
            "duration_s": len(y) / 22050.0,
            "t_float_s": round(t_float, 3),
            "t_int8_s": round(t_int8, 3),
            "beats": compare(beats_f, beats_i),
            "downbeats": compare(downbeats_f, downbeats_i),
        }
        print(f"{stem}: float {t_float:.2f}s int8 {t_int8:.2f}s "
              f"beats {results[stem]['beats']} downbeats {results[stem]['downbeats']}", flush=True)

    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
