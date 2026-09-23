"""Qui fait déborder le chroma NNLS ? — un tableau à la fois.

Rejoue `NNLSChromaNumpy.extract_bothchroma` étape par étape (mêmes opérations,
même ordre) en remettant à zéro le pic `tracemalloc` entre chaque étape, et
affiche pour chaque tableau intermédiaire sa FORME, son DTYPE et sa TAILLE.

Un sous-processus par morceau : `ru_maxrss` est un watermark monotone.

    ~/harmonia/.venv-mobile/bin/python mobile/tools/nnlsmem_profile.py
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AUDIO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def mb(a) -> float:
    return round(a.nbytes / 1e6, 1)


def child(song: str) -> None:
    sys.path.insert(0, str(REPO))
    from mobile.harmonia_mobile import backends_mobile as bm
    from mobile.harmonia_mobile import nnls_chroma as nc
    from numpy.lib.stride_tricks import sliding_window_view

    y32 = bm.audio_ffmpeg(AUDIO / f"{song}.m4a", 44100)
    S = nc.NNLSChromaNumpy()          # noyau + dictionnaire, hors mesure
    steps = []
    tracemalloc.start()

    def mark(nom, **arrays):
        _, peak = tracemalloc.get_traced_memory()
        steps.append(dict(etape=nom, pic_mb=round(peak / 1e6, 1),
                          tableaux={k: dict(forme=list(v.shape), dtype=str(v.dtype), mo=mb(v))
                                    for k, v in arrays.items()}))
        tracemalloc.reset_peak()

    y = np.asarray(y32, dtype=np.float64)
    mark("y float64", y=y)

    n = len(y)
    starts = np.arange(0, max(n, 1), nc.STEP_SIZE)
    starts = starts[starts < n]
    T = len(starts)
    win = nc._hann_periodic(nc.BLOCK_SIZE)
    pad = starts[-1] + nc.BLOCK_SIZE - n
    y_padded = np.concatenate([y, np.zeros(max(0, pad), dtype=np.float64)]) if pad > 0 else y
    frames = sliding_window_view(y_padded, nc.BLOCK_SIZE)[::nc.STEP_SIZE][:T]
    mark("frames (vue, 0 copie)", frames=frames)

    windowed = frames * win[None, :]
    mark("windowed = frames*win", windowed=windowed)

    spec = np.fft.rfft(windowed, axis=1)[:, :nc.N_FFT]
    mark("spec = rfft(windowed)", spec=spec)
    del windowed

    mag = np.abs(spec)
    mark("mag = |spec|", mag=mag)
    del spec

    mag = np.minimum(mag, nc.BLOCK_SIZE)
    silent = mag.max(axis=1) < 2.0
    mag[silent] = 0.0
    nm_all = mag @ S.K.T
    mark("nm_all = mag @ K.T", nm_all=nm_all)
    del mag, frames, y_padded, y

    tone_starts = np.arange(0, round(nc.N_NOTE * 0.62 / nc.N_BPS) * nc.N_BPS + 1, nc.N_BPS)
    gathered = nm_all[:, tone_starts[:, None] + np.arange(nc.N_BPS)[None, :]]
    mean_tunings = gathered.sum(axis=1).mean(axis=0)
    mark("accordage global (gathered)", gathered=gathered)
    del gathered
    real = float((mean_tunings * S.cos_bps).sum())
    imag = float((mean_tunings * S.sin_bps).sum())
    nt = np.arctan2(imag, real) / (2 * np.pi)
    int_shift = int(np.floor(nt * 3)); float_shift = nt * 3 - int_shift

    f2 = np.zeros((T, nc.N_NOTE), dtype=np.float64)
    k = np.arange(2, nc.N_NOTE - 3)
    f2[:, k] = nm_all[:, k + int_shift] * (1 - float_shift) + nm_all[:, k + int_shift + 1] * float_shift
    del nm_all
    running_mean = nc._special_convolution(f2, S.hw)
    dev2 = (f2 - running_mean) ** 2
    running_var = nc._special_convolution(dev2, S.hw)
    running_std = np.sqrt(np.maximum(running_var, 0.0))
    diff = f2 - running_mean
    whitened = np.where(diff > 0, diff / np.power(np.maximum(running_std, 1e-30), 1.0), 0.0)
    f2w = np.where(running_std > 0, whitened, f2)
    mark("accordage + blanchiment", f2w=f2w)
    del f2, running_mean, dev2, running_var, running_std, diff, whitened

    centers = nc.N_BPS // 2 + 2 + nc.N_BPS * np.arange(nc.N_SEMITONE_NOTES)
    centers = centers[centers < nc.N_NOTE - nc.N_BPS // 2]
    currval_all = f2w[:, centers - 1] + f2w[:, centers] + f2w[:, centers + 1]
    mask = currval_all > 0
    X = nc.nnls_batch_fista(S.D, S.AtA, f2w, mask, 250, S.fista_step)
    mark("FISTA (250 it.)", X=X, mask=mask)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("__JSON__" + json.dumps(dict(
        song=song, T=T, duree_s=round(n / 44100, 1), etapes=steps,
        pic_rss_mb=round(rss_mb(), 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        child(a.child); return
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    out = {}
    for song in SONGS:
        r = subprocess.run([sys.executable, __file__, "--child", song],
                           capture_output=True, text=True, env=env, cwd=str(REPO))
        line = [l for l in r.stdout.splitlines() if l.startswith("__JSON__")]
        if not line:
            print("ECHEC", song, r.stdout[-1500:], r.stderr[-2500:]); continue
        d = json.loads(line[0][len("__JSON__"):])
        out[song] = d
        print(f"\n=== {song}  ({d['duree_s']}s, T={d['T']} trames)  RSS max {d['pic_rss_mb']} Mo")
        for s in d["etapes"]:
            det = "  ".join(f"{k}{v['forme']}{v['dtype']}={v['mo']}Mo" for k, v in s["tableaux"].items())
            print(f"  {s['etape']:<28} pic_etape={s['pic_mb']:>7.1f} Mo   {det}")
    p = Path(__file__).resolve().parent / "parity_out" / "nnlsmem_profile.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("\n->", p)


if __name__ == "__main__":
    main()
