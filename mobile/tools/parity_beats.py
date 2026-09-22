"""P0-T1c: end-to-end parity + speed, torch `File2Beats` (final0, dbn=False)
vs the numpy+.pte mobile pipeline (`harmonia_mobile.beats_infer.beats_from_audio`),
on the 3 reference songs.

Run with the MOBILE venv:
    ~/harmonia/.venv-mobile/bin/python mobile/tools/parity_beats.py

Audio is loaded EXACTLY as `harmonia/beats.py::track()` loads it for the live
pipeline: try `beat_this.preprocessing.load_audio` first (torchaudio ->
soundfile -> madmom), and on failure (m4a: no torchcodec, libsndfile has no
AAC, madmom is py3.12-broken -- the same wall documented in beats.py) fall
back to an ffmpeg transcode to a temp mono 22050 Hz wav and retry. The
resulting signal is resampled to mono/22050 Hz ONCE (matching
`Audio2Frames.signal2spect`) and that SAME array is fed to both the torch
reference and the numpy/.pte path, so the comparison isolates the
front-end+network+postprocessor, not two different audio decodes.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
torch.set_num_threads(1)  # single-thread, matches T5's eventual phone comparison basis

from beat_this.inference import Audio2Beats
from beat_this.preprocessing import load_audio
import soxr

from harmonia_mobile.beats_frontend import log_mel_spectrogram
from harmonia_mobile.beats_infer import frames_from_spect, postprocess_minimal, DEFAULT_PTE

SONGS = {
    "maroon_5_this_love": Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav"),
    "ben_e_king_stand_by_me_audio": Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/ben_e_king_stand_by_me_audio.wav"),
    "autumn_leaves": Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio/autumn_leaves.m4a"),
}


def load_signal_like_beats_py(path: Path) -> np.ndarray:
    """Same two-step load as `harmonia.beats.track()`: `load_audio` first,
    ffmpeg transcode + retry on failure. Returns mono float32 @ 22050 Hz."""
    try:
        signal, sr = load_audio(str(path))
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            wav = f"{td}/a.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "22050", wav],
                check=True, timeout=300)
            signal, sr = load_audio(wav)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(1)
    if sr != 22050:
        signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
    return signal.astype(np.float32)


def torch_reference(signal: np.ndarray):
    a2b = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    t0 = time.perf_counter()
    beats, downbeats = a2b(signal, 22050)
    dt = time.perf_counter() - t0
    return np.asarray(beats), np.asarray(downbeats), dt


def mobile_pipeline(signal: np.ndarray, pte_path=DEFAULT_PTE):
    t0 = time.perf_counter()
    spect = log_mel_spectrogram(signal, sample_rate=22050)
    t1 = time.perf_counter()
    beat_logits, downbeat_logits = frames_from_spect(spect, pte_path)
    t2 = time.perf_counter()
    beats, downbeats = postprocess_minimal(beat_logits, downbeat_logits)
    t3 = time.perf_counter()
    timing = {"frontend_s": t1 - t0, "network_s": t2 - t1, "postproc_s": t3 - t2, "total_s": t3 - t0}
    return beats, downbeats, timing


def compare(name: str, ref_b, ref_d, mine_b, mine_d) -> dict:
    def _cmp(a, b, label):
        n = min(len(a), len(b))
        d_ms = np.abs(a[:n] - b[:n]) * 1000 if n else np.array([0.0])
        return {
            "n_ref": len(a), "n_mine": len(b), "n_common": n,
            "count_match": len(a) == len(b),
            "max_dt_ms": float(d_ms.max()) if n else float("nan"),
            "mean_dt_ms": float(d_ms.mean()) if n else float("nan"),
        }
    return {"beats": _cmp(ref_b, mine_b, "beats"), "downbeats": _cmp(ref_d, mine_d, "downbeats")}


def main():
    if not Path(DEFAULT_PTE).exists():
        print(f"!! {DEFAULT_PTE} missing -- run mobile/tools/export_beat_this.py first")
        sys.exit(1)

    print(f"{'song':32s} {'n_beat(ref/mine)':18s} {'max|dt| ms (beat)':18s} "
          f"{'n_db(ref/mine)':16s} {'max|dt| ms (db)':16s}")
    all_results = {}
    all_timing = {}
    for name, path in SONGS.items():
        signal = load_signal_like_beats_py(path)
        dur_s = len(signal) / 22050
        ref_b, ref_d, ref_dt = torch_reference(signal)
        mine_b, mine_d, timing = mobile_pipeline(signal)
        cmp = compare(name, ref_b, ref_d, mine_b, mine_d)
        all_results[name] = {"duration_s": dur_s, "torch_total_s": ref_dt, **cmp}
        all_timing[name] = timing
        n_beat_str = f"{cmp['beats']['n_ref']}/{cmp['beats']['n_mine']}"
        n_db_str = f"{cmp['downbeats']['n_ref']}/{cmp['downbeats']['n_mine']}"
        print(f"{name:32s} {n_beat_str:18s} "
              f"{cmp['beats']['max_dt_ms']:<18.3f} "
              f"{n_db_str:16s} "
              f"{cmp['downbeats']['max_dt_ms']:<16.3f}")

    print("\n=== speed (1 thread, per stage) ===")
    print(f"{'song':32s} {'dur_s':8s} {'frontend_s':11s} {'network_s':10s} {'postproc_s':11s} {'total_s':8s} {'torch_total_s':13s}")
    for name, t in all_timing.items():
        dur = all_results[name]["duration_s"]
        print(f"{name:32s} {dur:<8.1f} {t['frontend_s']:<11.3f} {t['network_s']:<10.3f} "
              f"{t['postproc_s']:<11.3f} {t['total_s']:<8.3f} {all_results[name]['torch_total_s']:<13.3f}")

    import json
    out = {"results": all_results, "timing": all_timing}
    out_path = ROOT / "models" / "parity_beats_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
