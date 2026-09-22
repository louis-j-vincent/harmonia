"""`_find_peaks_numpy` == `scipy.signal.find_peaks` ? (à lancer sous `.venv`)

Le port numpy remplace scipy dans `harmonia/sections/similarity.py::_peaks`.
Cette fonction n'est PAS sur le chemin critique du chart
(`docs/portage_mobile/p0_t3_deps_natives.md` : `folding.py` n'appelle que
`halfbar_features`), donc la parité du chart ne la teste pas — d'où ce test
direct, sinon le port ne serait vérifié par rien.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mobile.harmonia_mobile.backends_mobile import _find_peaks_numpy  # noqa: E402

rng = np.random.default_rng(0)
bad = 0
for k in range(300):
    n = int(rng.integers(20, 4000))
    kind = k % 3
    if kind == 0:
        x = rng.random(n)
    elif kind == 1:                      # avec des PLATEAUX (le cas piège)
        x = np.round(rng.random(n) * 4) / 4
    else:                                # une courbe lisse + bruit
        x = np.convolve(rng.random(n), np.ones(9) / 9, "same")
    thr = float(rng.random())
    dist = int(rng.integers(1, 12))
    a, _ = find_peaks(x, height=thr, distance=dist)
    b = _find_peaks_numpy(x, height=thr, distance=dist)
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        bad += 1
        if bad <= 3:
            print("DIFF", k, n, thr, dist, a[:10], b[:10])
print(f"{300 - bad}/300 identiques")
raise SystemExit(0 if bad == 0 else 1)
