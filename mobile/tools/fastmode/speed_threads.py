"""T5-prep — levier 1 : parallélisme (ne change AUCUN chart, seulement le temps).

Trois mesures, venv MOBILE (numpy + `.pte`), 1 morceau (this_love, 205 s,
CQT déjà dumpé par P2b `mobile/tools/cqt_ref/`, wav déjà utilisé par T1c) :

1. musx numpy (OpenBLAS, via `threadpoolctl.threadpool_limits`) à 1/2/4 threads.
2. beat_this `.pte` (XNNPACK, via `executorch.extension.pybindings.
   portable_lib._unsafe_reset_threadpool`) à 1/2/4 threads.
3. les deux EN MÊME TEMPS (deux threads Python, 1 thread lib chacun) contre
   la somme séquentielle 1-thread — pour voir si le GIL est relâché des
   deux côtés (BLAS et XNNPACK) sur CE binding, pas seulement en théorie.

Run: ~/harmonia/.venv-mobile/bin/python mobile/tools/fastmode/speed_threads.py OUT.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soxr
import threadpoolctl

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "mobile"))

from harmonia_mobile import musx_net  # noqa: E402
from harmonia_mobile.beats_infer import DEFAULT_PTE, beats_from_audio  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CQT_REF = Path(__file__).resolve().parents[2] / "tools" / "cqt_ref" / "maroon_5_this_love.npz"
WAV = Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav")


def load_wav_22050(path: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/a.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                         "-ac", "1", "-ar", "22050", out], check=True)
        import soundfile as sf
        y, sr = sf.read(out, dtype="float32")
        assert sr == 22050
        return y


def set_pte_threads(n: int):
    from executorch.extension.pybindings import portable_lib as pl
    pl._unsafe_reset_threadpool(n)


def time_pte(y: np.ndarray, n_threads: int) -> float:
    set_pte_threads(n_threads)
    t0 = time.perf_counter()
    beats_from_audio(y)
    return time.perf_counter() - t0


def time_musx(cqt288: np.ndarray, weights, n_threads: int) -> float:
    t0 = time.perf_counter()
    with threadpoolctl.threadpool_limits(limits=n_threads, user_api="blas"):
        musx_net.predict_ensemble(cqt288, weights)
    return time.perf_counter() - t0


def main():
    out_path = Path(sys.argv[1])
    print("loading audio + CQT reference...")
    y = load_wav_22050(WAV)
    z = np.load(CQT_REF)
    cqt288 = z["cqt"].astype(np.float32)
    weights = [musx_net.load_weights_npz(MODELS_DIR / f"musx_fold{i}_weights.npz") for i in range(5)]
    duration_s = len(y) / 22050.0
    results = {"song": "maroon_5_this_love", "duration_s": duration_s}

    # 1 ── musx numpy (OpenBLAS) thread scan
    results["musx_threads"] = {}
    for n in (1, 2, 4):
        # warm-up (first BLAS call sometimes pays init cost)
        with threadpoolctl.threadpool_limits(limits=n, user_api="blas"):
            musx_net.predict_ensemble(cqt288, weights[:1])
        t = time_musx(cqt288, weights, n)
        results["musx_threads"][n] = round(t, 3)
        print(f"musx numpy, {n} thread(s): {t:.2f}s (5 folds, {duration_s:.0f}s audio)")

    # 2 ── beat_this .pte (XNNPACK) thread scan
    results["pte_threads"] = {}
    for n in (1, 2, 4):
        set_pte_threads(n)
        beats_from_audio(y)  # warm-up
        t = time_pte(y, n)
        results["pte_threads"][n] = round(t, 3)
        print(f".pte XNNPACK, {n} thread(s): {t:.2f}s")

    # 3 ── sequential (1 thread each) vs true parallel (2 python threads)
    set_pte_threads(1)
    t0 = time.perf_counter()
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        musx_net.predict_ensemble(cqt288, weights)
    t_musx_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    beats_from_audio(y)
    t_pte_seq = time.perf_counter() - t0
    seq_total = t_musx_seq + t_pte_seq

    out = {}
    def _musx_job():
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            t0 = time.perf_counter()
            musx_net.predict_ensemble(cqt288, weights)
            out["musx"] = time.perf_counter() - t0

    def _pte_job():
        set_pte_threads(1)
        t0 = time.perf_counter()
        beats_from_audio(y)
        out["pte"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    th1 = threading.Thread(target=_musx_job)
    th2 = threading.Thread(target=_pte_job)
    th1.start(); th2.start()
    th1.join(); th2.join()
    parallel_wall = time.perf_counter() - t0

    results["sequential_1thread_each"] = {
        "musx_s": round(t_musx_seq, 3), "pte_s": round(t_pte_seq, 3),
        "total_s": round(seq_total, 3),
    }
    results["parallel_2_python_threads_1lib_each"] = {
        "musx_s": round(out.get("musx", -1), 3), "pte_s": round(out.get("pte", -1), 3),
        "wall_s": round(parallel_wall, 3),
        "speedup_vs_sequential": round(seq_total / parallel_wall, 3) if parallel_wall else None,
    }
    print(f"sequential total: {seq_total:.2f}s ; parallel wall: {parallel_wall:.2f}s "
          f"(speedup {seq_total/parallel_wall:.2f}x)")

    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
