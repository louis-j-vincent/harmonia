"""Qui fait dévier les battues : le RÉSEAU porté, ou le DÉCODEUR qui le nourrit ?

`ben_e_king_stand_by_me_audio` sort 347 battues par le chemin Mac et 345 par
le chemin mobile, alors que T1c avait mesuré 345/345 à 0,000 ms entre torch et
`.pte` — en nourrissant les DEUX avec la même séquence d'échantillons. La
seule chose qui a changé ici, c'est la source du PCM. On le vérifie au lieu
de le supposer (CLAUDE.md règle 6).

À lancer sous `~/harmonia/.venv-mobile` (il faut executorch pour le `.pte`).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
AUDIO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
PCM = REPO / "mobile" / "tools" / "parity_out" / "pcm"
STEM = "ben_e_king_stand_by_me_audio"


def main() -> None:
    from mobile.harmonia_mobile.beats_infer import beats_from_audio
    p = AUDIO / f"{STEM}.m4a"

    # A. le PCM que `librosa.load(sr=22050)` rend (ce que le chemin mobile a lu)
    y_lib = np.load(PCM / f"{STEM}__22050.npy")
    # B. le PCM que ffmpeg rend — c'est CE fichier-là que `beat_this` reçoit
    #    sur le Mac : `harmonia.beats._beatthis_track` retombe sur
    #    `ffmpeg -ac 1 -ar 22050 a.wav` parce qu'aucun lecteur de beat_this ne
    #    sait ouvrir un .m4a sur cette machine.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1",
                          "-ar", "22050", "-f", "f32le", "-"],
                         capture_output=True).stdout
    y_ff = np.frombuffer(raw, dtype=np.float32).copy()

    n = min(len(y_lib), len(y_ff))
    print(f"n_samples  librosa={len(y_lib)}  ffmpeg={len(y_ff)}  Δ={len(y_ff)-len(y_lib)}")
    print(f"max|Δ| sur les {n} premiers échantillons : "
          f"{float(np.abs(y_lib[:n] - y_ff[:n]).max()):.6f}")
    print(f"  (début) max|Δ| sur les 22050 premiers : "
          f"{float(np.abs(y_lib[:22050] - y_ff[:22050]).max()):.6f}")

    for nom, y in (("librosa (chemin mobile)", y_lib), ("ffmpeg (chemin Mac)", y_ff)):
        b, d = beats_from_audio(np.asarray(y, np.float32))
        print(f"{nom:26s} -> {len(b)} battues, {len(d)} temps forts, "
              f"1re battue {b[0]:.4f}s")


if __name__ == "__main__":
    main()
