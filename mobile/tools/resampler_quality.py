"""P2a-bis — banc d'essai numpy du rééchantillonneur Kaiser-sinc polyphase
d'`AudioDecoder.kt` (`designKaiserSincFilter`/`polyphaseResample`), isolé de
tout décodage AAC : l'entrée est le décodage natif du Mac lui-même (même
source que le décodeur Android verrait après démuxage), donc AUCUNE
différence de décodeur possible — seul le rééchantillonneur est sous test.
Compare contre `librosa.resample(res_type="soxr_hq")`, la référence retenue
par le contrat de décodage (p0_t3).

Équivalence vérifiée avant ce script (voir docs/portage_mobile/p2a_bis_resampler.md,
section 1) : `scipy.signal.resample_poly(x, up, down, window=h/up)` reproduit
EXACTEMENT (diff ~1e-16) une traduction ligne-à-ligne du
`polyphaseResample`/`designKaiserSincFilter` Kotlin sur un signal jouet — donc
la mesure ci-dessous porte sur la MÊME formule de filtre que le Kotlin, pas
une approximation scipy.

Usage: .venv/bin/python -m mobile.tools.resampler_quality
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import resample_poly

AUDIO_DIR = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")

# (stem, native_sr attendu, sr cible) — les deux paires de la bibliothèque
# (contrat p0_t3 : cibles 22050 battues/musx, 44100 nnls-chroma).
CASES = [
    ("maroon_5_this_love", 48000, 22050),
    ("ben_e_king_stand_by_me_audio", 48000, 22050),
    ("maroon_5_she_will_be_loved_official_music_video", 48000, 22050),
    ("08DjMT-qR9g", 44100, 22050),
    ("maroon_5_this_love", 48000, 44100),
]


def bessel_i0(x: float) -> float:
    s = 1.0
    t = 1.0
    k = 1
    while t > s * 1e-12:
        t *= (x / (2 * k)) ** 2
        s += t
        k += 1
    return s


def design_kaiser_sinc_filter(up: int, taps_per_phase: int, cutoff: float, beta: float) -> np.ndarray:
    """Même formule que AudioDecoder.kt::designKaiserSincFilter : sinc x
    fenêtre de Kaiser, longueur 2*taps_per_phase*up+1, gain unité en DC."""
    half_len = taps_per_phase * up
    n = np.arange(-half_len, half_len + 1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(n == 0, 2.0 * cutoff, np.sin(2.0 * np.pi * cutoff * n) / (np.pi * n))
    r = n / half_len
    inside = np.clip(1.0 - r * r, 0.0, None)
    i0beta = bessel_i0(beta)
    w = np.array([bessel_i0(beta * np.sqrt(v)) for v in inside]) / i0beta
    h = sinc * w
    h *= up / h.sum()
    return h


def resample(x: np.ndarray, sr_in: int, sr_out: int, taps_per_phase: int, beta: float,
             cutoff_scale: float = 1.0) -> tuple[np.ndarray, float]:
    g = np.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    cutoff = cutoff_scale * 0.5 / max(up, down)
    t0 = time.time()
    h = design_kaiser_sinc_filter(up, taps_per_phase, cutoff, beta)
    y = resample_poly(x, up, down, window=h / up)
    dt = time.time() - t0
    return y, dt


def compare(y: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    n = min(len(y), len(ref))
    d = np.abs(y[:n] - ref[:n])
    return float(d.max()), float(d.mean())


def main() -> None:
    print(f"{'stem':45s} {'sr_in':>6} {'sr_out':>6} {'taps/ph':>7} {'beta':>6} {'cutoff%':>7} "
          f"{'maxdiff':>10} {'meandiff':>10} {'filter_s':>9}")
    configs = [
        (64, 8.6, 1.0),
        (128, 8.6, 1.0),
        (256, 8.6, 1.0),
        (128, 12.26, 1.0),
        (256, 12.26, 1.0),
        (256, 12.26, 0.93),
        (512, 12.26, 0.93),
    ]
    for stem, native_sr, target_sr in CASES:
        path = AUDIO_DIR / f"{stem}.m4a"
        native, sr = librosa.load(str(path), sr=None, mono=True)
        assert sr == native_sr, (stem, sr, native_sr)
        ref = librosa.resample(native, orig_sr=native_sr, target_sr=target_sr, res_type="soxr_hq")
        for taps, beta, cscale in configs:
            y, dt = resample(native, native_sr, target_sr, taps, beta, cscale)
            maxd, meand = compare(y, ref)
            print(f"{stem:45s} {native_sr:6d} {target_sr:6d} {taps:7d} {beta:6.2f} {cscale*100:6.1f}% "
                  f"{maxd:10.6f} {meand:10.6f} {dt:9.2f}")


if __name__ == "__main__":
    main()
