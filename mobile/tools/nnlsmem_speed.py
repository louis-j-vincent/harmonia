"""Pic mémoire + vitesse du chroma NNLS, avec et sans tuilage — un PROCESSUS par variante.

Deux raisons de séparer : `ru_maxrss` est un watermark monotone (il ne
redescend jamais), et la variante sans tuilage alloue ~1,7 Go, ce qui met la
machine sous pression et ralentit ensuite ce qui tourne après elle.

`pic_alloc_mb` (tracemalloc, domaine numpy inclus) = somme des tableaux
VIVANTS au pire instant — le chiffre à comparer au budget du téléphone.
`pic_rss_mb` = ce que l'OS a dû fournir (surestime, l'allocateur macOS garde
ses gros blocs).

    ~/harmonia/.venv-mobile/bin/python mobile/tools/nnlsmem_speed.py
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]
VARIANTES = ["none", "128", "256", "512"]


def child(song: str, tag: str) -> None:
    sys.path.insert(0, str(REPO))
    from mobile.harmonia_mobile import backends_mobile as bm
    from mobile.harmonia_mobile import nnls_chroma as nc
    y = bm.audio_ffmpeg(AUDIO / f"{song}.m4a", 44100)
    S = nc.NNLSChromaNumpy()          # noyau + dictionnaire : hors mesure
    ch = None if tag == "none" else int(tag)
    tracemalloc.start()
    t0 = time.perf_counter()
    arr, _ = S.extract_bothchroma(y, 44100, chunk_frames=ch)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("__JSON__" + json.dumps(dict(
        t=dt, T=int(arr.shape[0]), duree_s=len(y) / 44100,
        pic_alloc_mb=round(peak / 1e6, 1),
        pic_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1))))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3]); return
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    rows: dict = {}
    for song in SONGS:
        for tag in VARIANTES:
            r = subprocess.run([sys.executable, __file__, "--child", song, tag],
                               capture_output=True, text=True, env=env, cwd=str(REPO))
            line = [l for l in r.stdout.splitlines() if l.startswith("__JSON__")]
            if not line:
                print("ECHEC", song, tag, r.stderr[-1500:]); continue
            d = json.loads(line[0][len("__JSON__"):])
            d["s_par_4min"] = round(d["t"] * 240 / d["duree_s"], 2)
            rows.setdefault(song, {})[tag] = d
            print(f"{song:<48} chunk={tag:<5} {d['t']:>6.2f}s ({d['s_par_4min']}s/4min)  "
                  f"pic_vivant={d['pic_alloc_mb']:>7.1f} Mo  rss={d['pic_rss_mb']:.0f} Mo",
                  flush=True)
    out = Path(__file__).resolve().parent / "parity_out" / "nnlsmem_speed.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
