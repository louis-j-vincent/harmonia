"""Le découpage en tuiles de `musx_net` change-t-il le résultat ? (et à quel prix)

Référence = `predict_ensemble(cqt, w)` SANS découpage (le chemin déjà validé à
2e-6 contre torch, `docs/portage_mobile/t1b_musx.md`). Comparé à
`predict_ensemble(cqt, w, chunk_frames=N)` : écart max sur les 6 têtes, accord
d'argmax sur les 6 têtes, et temps.

    ~/harmonia/.venv-mobile/bin/python mobile/tools/mem_chunk_parity.py [--chunks 256,512,1024]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mobile"))
from harmonia_mobile import musx_net  # noqa: E402

CQT_REF = Path(__file__).resolve().parent / "cqt_ref"
MODELS = Path(__file__).resolve().parents[1] / "models"
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]
HEADS = ("triad", "bass", "s7", "s9", "s11", "s13")


def get_cqt(song: str) -> np.ndarray:
    p = CQT_REF / f"{song}.npz"
    if p.exists():
        with np.load(p) as z:
            return z["cqt"].astype(np.float32)
    c = Path(os.environ.get("TMPDIR", "/tmp")) / f"mem_cqt_{song}.npy"
    if c.exists():
        return np.load(c).astype(np.float32)
    raise SystemExit(f"pas de CQT pour {song} ({p} ni {c})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="512")
    a = ap.parse_args()
    chunks = [int(x) for x in a.chunks.split(",")]

    w = [musx_net.load_weights_npz(MODELS / f"musx_fold{i}_weights.npz") for i in range(5)]
    rows = []
    for song in SONGS:
        cqt = get_cqt(song)
        dur = cqt.shape[0] * 512 / 22050
        t0 = time.perf_counter()
        ref = musx_net.predict_ensemble(cqt, w)
        t_ref = time.perf_counter() - t0
        ref_am = [r.argmax(axis=1) for r in ref]
        for ch in chunks:
            t0 = time.perf_counter()
            got = musx_net.predict_ensemble(cqt, w, chunk_frames=ch)
            t_ch = time.perf_counter() - t0
            per_head = {n: float(np.abs(x - y).max()) for n, x, y in zip(HEADS, ref, got)}
            am = {n: float((x == y.argmax(axis=1)).mean())
                  for n, x, y in zip(HEADS, ref_am, got)}
            r = dict(song=song, duree_s=round(dur, 1), T=int(cqt.shape[0]), chunk=ch,
                     ecart_max=max(per_head.values()), par_tete=per_head,
                     argmax=am, argmax_min=min(am.values()),
                     t_sans=round(t_ref, 2), t_avec=round(t_ch, 2),
                     s_par_4min_sans=round(t_ref * 240 / dur, 1),
                     s_par_4min_avec=round(t_ch * 240 / dur, 1))
            rows.append(r)
            print(f"{song:<48} chunk={ch:<6} ecart_max={r['ecart_max']:.3e} "
                  f"argmax_min={r['argmax_min']*100:.4f}%  "
                  f"{r['s_par_4min_sans']}s -> {r['s_par_4min_avec']}s /4min", flush=True)
        del ref, got
    out = Path(__file__).resolve().parent / "parity_out" / "mem_chunk_parity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print("\nseuil: ecart_max <= 1e-5 et argmax == 100% ->",
          "OK" if all(r["ecart_max"] <= 1e-5 and r["argmax_min"] == 1.0 for r in rows) else "ECHEC")
    print("->", out)


if __name__ == "__main__":
    main()
