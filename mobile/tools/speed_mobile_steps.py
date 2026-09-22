"""Vitesse du chemin mobile sur le Mac, étape par étape, EN SÉRIE.

Pourquoi un script séparé de `parity_pipeline.py` : le pipeline lance les
battues et musx dans deux threads (`analyze_steps`, ThreadPoolExecutor), donc
les temps par fente qu'il mesure sont des durées MURALES sous contention —
utiles pour le total, trompeuses par étape. Ici tout est en série.

    OMP_NUM_THREADS=1 ... ~/harmonia/.venv-mobile/bin/python \
        mobile/tools/speed_mobile_steps.py

Le runtime XNNPACK garde son propre pool de threads, que les variables
d'environnement ne contraignent pas (déjà noté dans `t1c_beat_this_complet.md`)
— la ligne « battues » n'est donc pas un vrai 1-thread, et c'est dit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
AUDIO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]


def main() -> None:
    from mobile.harmonia_mobile import backends_mobile as bm
    bm.install(audio="ffmpeg")          # décodage réel, pas le dump figé
    rows = {}
    for stem in SONGS:
        p = AUDIO / f"{stem}.m4a"
        r = {}
        t = time.perf_counter(); y22 = bm._load(p, 22050)
        r["decodage_22050"] = round(time.perf_counter() - t, 2)
        t = time.perf_counter(); y44 = bm._load(p, 44100)
        r["decodage_44100"] = round(time.perf_counter() - t, 2)
        r["duree_s"] = round(len(y22) / 22050, 1)

        from mobile.harmonia_mobile.beats_infer import beats_from_audio
        t = time.perf_counter(); beats_from_audio(y22)
        r["battues"] = round(time.perf_counter() - t, 2)

        from mobile.harmonia_mobile.cqt_musx import song_cqt_288
        t = time.perf_counter(); cqt = song_cqt_288(y22, sr=22050)
        r["cqt"] = round(time.perf_counter() - t, 2)
        t = time.perf_counter(); bm.musx_posteriors_numpy(cqt)
        r["musx_reseaux"] = round(time.perf_counter() - t, 2)

        from mobile.harmonia_mobile.nnls_chroma import extract_bothchroma_numpy
        t = time.perf_counter(); extract_bothchroma_numpy(y44, 44100)
        r["nnls"] = round(time.perf_counter() - t, 2)
        rows[stem] = r
        print(stem, json.dumps(r), flush=True)
    (REPO / "mobile" / "tools" / "parity_out" / "speed_mobile.json").write_text(
        json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
