"""Le bundle par morceau : tout ce dont la recherche de sections a besoin, en cache.

`voice_with_hard` recalcule à chaque appel la séparation demucs, le pyin, les
matrices — 20 à 50 s par morceau. Un balayage de seuil en fait des centaines
d'appels. On fige donc une bonne fois (S, M, mute, start, pics, V, grille) dans
`scratchpad/order_cache/<stem>.pkl`, et les expériences ne manipulent plus que
la RECHERCHE elle-même, qui coûte des millisecondes.

    .venv/bin/python scratchpad/order_bundle.py        # construit les 12
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

CACHE = HERE / "scratchpad" / "order_cache"
CACHE.mkdir(exist_ok=True)


def build(stem: str) -> dict:
    from ssm_zoo import AUDIO
    from peak_profile import fused_profile
    import harmonia_min.voice_sections as VS
    from harmonia_min import harmonic_sections as HS
    VA, VM, MS, _B8 = VS._scripts()

    P, kept, rest, lines, n, extra = fused_profile(stem)
    m = len(P)
    grid = extra["grid"]
    triad = extra["triad"]

    V = HS.harmonic_vectors(triad, grid)
    S = V @ V.T
    voc = VA.separate_vocals(AUDIO / f"{stem}.m4a")
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)
    start = VS.sung_start(notes, grid, mute)

    def to_bar(c):
        return int(round(c * n / max(1, m)))

    hard = [h for h in sorted({to_bar(c) for c in kept}) if 0 < h < n]
    rest_bars = [h for h in sorted({to_bar(c) for c in rest}) if 0 < h < n]
    prof_bar = np.array([float(P[min(m - 1, int(round(b * m / max(1, n)))) ]) for b in range(n)])

    return {"stem": stem, "n": n, "grid": list(grid), "S": S, "M": M,
            "mute": np.asarray(mute, bool), "start": int(start), "hard": hard,
            "rest": rest_bars, "V": V, "P": P, "prof_bar": prof_bar,
            "half_mute": extra.get("mute")}


def get(stem: str, rebuild=False) -> dict:
    p = CACHE / f"{stem}.pkl"
    if p.exists() and not rebuild:
        with p.open("rb") as f:
            return pickle.load(f)
    b = build(stem)
    with p.open("wb") as f:
        pickle.dump(b, f)
    return b


def main():
    from ssm_zoo import SONGS, AUDIO
    stems = sys.argv[1:] or [s for s, _ in SONGS]
    for s in stems:
        if not (AUDIO / f"{s}.m4a").exists():
            print(f"  ?? pas d'audio {s}")
            continue
        b = get(s, rebuild="--rebuild" in sys.argv)
        print(f"  ok {s}: n={b['n']} start={b['start']} pics={b['hard']}")


if __name__ == "__main__":
    main()
