"""P2d — compare les battues calculées SUR L'ÉMULATEUR à celles du Mac.

Trois chemins comparés, pour séparer ce qui vient du DÉCODEUR de ce qui
vient du CALCUL :

1. `--mac-ref`   : PCM du Mac (celui que `beats.py` reçoit aujourd'hui,
                   transcodage ffmpeg, figé par `dump_reference_pcm.py`)
                   + réseau `.pte` XNNPACK + runtime python d'ExecuTorch.
                   C'est la référence.
2. `--device`    : le JSON rendu par `GET /api/debug/beats` sur le téléphone
                   (PCM MediaCodec + `.pte` portable + runtime natif JNI).
3. `--mac-devpcm`: PCM RAPATRIÉ DU TÉLÉPHONE, rejoué par le chemin Mac.
                   La différence (1)→(3) est imputable au DÉCODEUR seul ;
                   la différence (3)→(2) au calcul sur ARM.

Sans (3), un écart de battues ne se laisse pas attribuer, et c'est justement
ce que P2c avait montré : sur Stand By Me, deux décodages du MÊME fichier au
MÊME taux donnent 345 battues contre 348.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PCM = REPO / "mobile" / "tools" / "parity_out" / "pcm"


def mac_beats(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from mobile.harmonia_mobile.beats_infer import beats_from_audio
    return beats_from_audio(np.asarray(y, dtype=np.float32))


def _pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Le compte, puis l'écart. Si les comptes diffèrent on ne fabrique PAS un
    appariement plausible : on dit de combien ils diffèrent et on apparie
    chaque battue de a à la plus proche de b, en le disant."""
    out = {"n": [int(len(a)), int(len(b))], "meme_compte": len(a) == len(b)}
    if len(a) == len(b) and len(a):
        out["max_dt_ms"] = round(1000 * float(np.max(np.abs(a - b))), 3)
        out["mediane_dt_ms"] = round(1000 * float(np.median(np.abs(a - b))), 3)
    elif len(a) and len(b):
        d = np.abs(a[:, None] - b[None, :]).min(axis=1)
        out["max_dt_ms_plus_proche"] = round(1000 * float(d.max()), 3)
        out["mediane_dt_ms_plus_proche"] = round(1000 * float(np.median(d)), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--device-json", required=True)
    ap.add_argument("--device-pcm", default=None,
                    help=".npy du PCM 22050 rapatrié du téléphone (optionnel)")
    a = ap.parse_args()

    dev = json.loads(Path(a.device_json).read_text())
    dev_beats = np.asarray(dev["beats"], dtype=np.float64)
    dev_down = np.asarray(dev["downbeats"], dtype=np.float64)

    y_mac = np.load(PCM / f"{a.song}__22050beats.npy")
    mb, md = mac_beats(y_mac)

    res = {"song": a.song,
           "n_samples": {"mac": int(y_mac.size), "device": int(dev["n_samples"])},
           "battues_mac_vs_device": _pair(mb, dev_beats),
           "temps_forts_mac_vs_device": _pair(md, dev_down)}

    if a.device_pcm:
        y_dev = np.load(a.device_pcm)
        n = min(len(y_mac), len(y_dev))
        res["pcm"] = {"max_abs_diff": float(np.max(np.abs(
            y_mac[:n].astype(np.float64) - y_dev[:n].astype(np.float64)))),
            "delta_n_samples": int(len(y_dev) - len(y_mac))}
        b2, d2 = mac_beats(y_dev)
        res["battues_mac_avec_pcm_device"] = _pair(mb, b2)
        res["battues_pcmdevice_mac_vs_device_arm"] = _pair(b2, dev_beats)

    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
