"""Coût en vitesse du découpage, mesuré dans des processus SÉPARÉS.

Mesurer les deux variantes dans le même processus fausse le résultat : la
variante sans découpage alloue ~2 Go et met la machine sous pression mémoire,
ce qui ralentit ensuite celle qui suit. Un processus par variante.

    ~/harmonia/.venv-mobile/bin/python mobile/tools/mem_speed_chunk.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]


def child(song: str, chunk: int | None) -> None:
    sys.path.insert(0, str(REPO / "mobile"))
    from harmonia_mobile import musx_net
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mem_chunk_parity import get_cqt
    cqt = get_cqt(song)
    w = [musx_net.load_weights_npz(REPO / "mobile" / "models" / f"musx_fold{i}_weights.npz")
         for i in range(5)]
    kw = {} if chunk is None else {"chunk_frames": chunk}
    t0 = time.perf_counter()
    musx_net.predict_ensemble(cqt, w, **kw)
    dt = time.perf_counter() - t0
    import resource
    print("__JSON__" + json.dumps(dict(
        t=dt, T=int(cqt.shape[0]), duree_s=cqt.shape[0] * 512 / 22050,
        pic_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], None if sys.argv[3] == "none" else int(sys.argv[3]))
        return
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    rows = {}
    for song in SONGS:
        for tag in ("none", "256"):
            out = subprocess.run([sys.executable, __file__, "--child", song, tag],
                                 capture_output=True, text=True, env=env, cwd=str(REPO))
            line = [l for l in out.stdout.splitlines() if l.startswith("__JSON__")]
            if not line:
                print("ECHEC", song, tag, out.stderr[-1500:]); continue
            d = json.loads(line[0][len("__JSON__"):])
            d["s_par_4min"] = round(d["t"] * 240 / d["duree_s"], 1)
            rows.setdefault(song, {})[tag] = d
            print(f"{song:<48} chunk={tag:<5} {d['t']:>7.1f}s  "
                  f"({d['s_par_4min']}s/4min)  pic={d['pic_mb']:.0f} Mo", flush=True)
    out = Path(__file__).resolve().parent / "parity_out" / "mem_speed_chunk.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
