"""Le tuilage du chroma NNLS change-t-il la sortie ? (basse ET aigu)

Référence = `extract_bothchroma_numpy(y, sr)` SANS tuilage (le chemin validé
contre le vrai plugin VAMP à 3e-4, docs/portage_mobile/p0_t2_nnls_numpy.md).
Comparé à `chunk_frames=N` : écart max sur les 12 colonnes de BASSE, sur les
12 colonnes d'AIGU, et accord de l'argmax de la basse (ce que
`_write_sounding_bass` consomme réellement).

    ~/harmonia/.venv-mobile/bin/python mobile/tools/nnlsmem_parity.py [--chunks 128,256,512]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AUDIO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
sys.path.insert(0, str(REPO))
from mobile.harmonia_mobile import backends_mobile as bm  # noqa: E402
from mobile.harmonia_mobile.nnls_chroma import extract_bothchroma_numpy  # noqa: E402

SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]
SEUIL = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="128,256,512")
    a = ap.parse_args()
    chunks = [int(x) for x in a.chunks.split(",")]

    rows = []
    for song in SONGS:
        y = bm.audio_ffmpeg(AUDIO / f"{song}.m4a", 44100)
        dur = len(y) / 44100
        t0 = time.perf_counter()
        ref, times = extract_bothchroma_numpy(y, 44100)
        t_ref = time.perf_counter() - t0
        ref_bass_am = ref[:, :12].argmax(axis=1)
        for ch in chunks:
            t0 = time.perf_counter()
            got, times2 = extract_bothchroma_numpy(y, 44100, chunk_frames=ch)
            t_ch = time.perf_counter() - t0
            d_bass = float(np.abs(ref[:, :12] - got[:, :12]).max())
            d_treb = float(np.abs(ref[:, 12:] - got[:, 12:]).max())
            am = float((ref_bass_am == got[:, :12].argmax(axis=1)).mean())
            r = dict(song=song, duree_s=round(dur, 1), T=int(ref.shape[0]), chunk=ch,
                     ecart_basse=d_bass, ecart_aigu=d_treb,
                     ampl_max=float(np.abs(ref).max()),
                     temps_identiques=bool(np.array_equal(times, times2)),
                     argmax_basse=am,
                     t_sans=round(t_ref, 2), t_avec=round(t_ch, 2),
                     s_par_4min_sans=round(t_ref * 240 / dur, 2),
                     s_par_4min_avec=round(t_ch * 240 / dur, 2))
            rows.append(r)
            print(f"{song:<48} chunk={ch:<5} basse={d_bass:.3e} aigu={d_treb:.3e} "
                  f"argmax={am*100:.4f}%  {r['s_par_4min_sans']}s -> {r['s_par_4min_avec']}s /4min",
                  flush=True)
        del ref, got
    out = Path(__file__).resolve().parent / "parity_out" / "nnlsmem_parity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    ok = all(max(r["ecart_basse"], r["ecart_aigu"]) <= SEUIL and r["argmax_basse"] == 1.0
             and r["temps_identiques"] for r in rows)
    print(f"\nseuil: ecart <= {SEUIL} sur basse ET aigu, argmax basse 100% ->",
          "OK" if ok else "ECHEC")
    print("->", out)


if __name__ == "__main__":
    main()
