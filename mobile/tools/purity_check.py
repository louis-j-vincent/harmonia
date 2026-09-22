"""La preuve « ça tournera sous Chaquopy » : les quatre briques numpy du
chemin mobile, exécutées pour de vrai, puis `sys.modules` inspecté.

À lancer dans `~/harmonia/.venv-mobile` (où torch, librosa, scipy et soxr
SONT installés — c'est bien le point : la preuve est qu'ils ne sont pas
IMPORTÉS, pas qu'ils sont absents du disque).

    ~/harmonia/.venv-mobile/bin/python mobile/tools/purity_check.py

CE QUE CETTE PREUVE NE COUVRE PAS, et il faut le dire : la brique `beats`.
Elle a besoin du runtime ExecuTorch, et le paquet pip `executorch` importe
`torch` sur le Mac (vérifié : `import executorch.runtime` met `torch` dans
`sys.modules`). Sur Android le runtime d'ExecuTorch est une bibliothèque
native chargée par JNI, sans torch — c'est la raison d'être d'ExecuTorch.
Donc : sur Mac cette preuve porte sur CQT + musx + NNLS + find_peaks ; la
pureté de `beats` sur téléphone est un fait de packaging Android, à revérifier
au premier APK qui l'embarque (T5), pas ici.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FORBIDDEN = ("torch", "librosa", "vamp", "scipy", "soxr", "numba")
STEM = "ben_e_king_stand_by_me_audio"
PCM = REPO / "mobile" / "tools" / "parity_out" / "pcm"


def main() -> int:
    from mobile.harmonia_mobile.backends_mobile import _find_peaks_numpy
    from mobile.harmonia_mobile.cqt_musx import song_cqt_288
    from mobile.harmonia_mobile.musx_net import load_weights_npz, predict_ensemble
    from mobile.harmonia_mobile.nnls_chroma import extract_bothchroma_numpy

    y22 = np.load(PCM / f"{STEM}__22050.npy")[: 22050 * 20]
    y44 = np.load(PCM / f"{STEM}__44100.npy")[: 44100 * 20]

    cqt = song_cqt_288(y22, sr=22050)
    w = [load_weights_npz(REPO / "mobile" / "models" / f"musx_fold{i}_weights.npz")
         for i in range(5)]
    probs = predict_ensemble(cqt[:200], w)
    arr, times = extract_bothchroma_numpy(y44, 44100)
    pk = _find_peaks_numpy(np.abs(np.sin(np.arange(500) / 7.0)),
                           height=0.3, distance=4)

    print(f"cqt {cqt.shape} · musx {[p.shape for p in probs]} · "
          f"nnls {arr.shape} · pics {len(pk)}")
    leaked = sorted(m for m in FORBIDDEN if m in sys.modules)
    print(f"modules interdits importés : {leaked or 'AUCUN'}")
    print(f"({len(sys.modules)} modules au total)")
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
