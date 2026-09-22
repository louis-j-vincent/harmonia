"""Fige le DÉCODAGE de référence du Mac, une fois, pour que la parité P2c
mesure les cinq briques de calcul et PAS le décodeur.

Pourquoi ce fichier existe : les deux venvs n'ont pas la même librosa
(`.venv` = 0.11.0 avec repli `audioread`/CoreAudio, `~/harmonia/.venv-mobile`
= 1.0.0, qui a SUPPRIMÉ ce repli et ne sait donc pas ouvrir un `.m4a`).
Décoder « avec librosa » des deux côtés n'aurait donc PAS été le même
décodage — exactement le genre d'écart silencieux que la règle n°1 de
CLAUDE.md dit d'aller vérifier avant de mesurer quoi que ce soit.

Ce script tourne sous `.venv` (le venv de prod) et écrit, par morceau, le
PCM que CHAQUE site d'appel du pipeline obtient AUJOURD'HUI :

    <stem>__22050.npy      librosa.load(sr=22050, mono=True)  musx
    <stem>__22050beats.npy ffmpeg -ac 1 -ar 22050            Beat This!
    <stem>__44100.npy      librosa.load(sr=44100, mono=True)  chroma NNLS
    <stem>__8000w90.npy    ffmpeg -t 90 -ar 8000 -f f32le    debut.enveloppe

Les battues ont leur PROPRE fichier, et ce n'est pas un détail : sur ce Mac
aucun lecteur de `beat_this` ne sait ouvrir un `.m4a`, donc
`harmonia.beats._beatthis_track` retombe TOUJOURS sur un transcodage
`ffmpeg -ac 1 -ar 22050`. Le réseau de battues ne voit donc pas le même
signal que musx. Mesuré sur Stand By Me : 448 échantillons d'écart et
max|Δ| = 0,387 entre les deux décodages — assez pour changer le nombre de
battues (348 contre 345, `mobile/tools/diag_beats_decoder.py`).

Le 8 kHz vient de ffmpeg et pas de librosa PARCE QUE le pipeline du Mac
appelle bien ffmpeg à cet endroit-là (`harmonia/debut.py::enveloppe`) — on
fige ce que chaque site fait, pas un décodeur uniforme qu'aucun site n'emploie.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
AUDIO_DIR = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
OUT = REPO / "mobile" / "tools" / "parity_out" / "pcm"
SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]


def main() -> None:
    import librosa
    OUT.mkdir(parents=True, exist_ok=True)
    print("librosa", librosa.__version__)
    for stem in SONGS:
        p = AUDIO_DIR / f"{stem}.m4a"
        for sr in (22050, 44100):
            y, _ = librosa.load(str(p), sr=sr, mono=True)
            np.save(OUT / f"{stem}__{sr}.npy", np.asarray(y, np.float32))
            print(stem, sr, y.shape)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p),
             "-ac", "1", "-ar", "22050", "-f", "f32le", "-"],
            capture_output=True).stdout
        yb = np.frombuffer(raw, dtype=np.float32).copy()
        np.save(OUT / f"{stem}__22050beats.npy", yb)
        print(stem, "22050 (ffmpeg, battues)", yb.shape)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p), "-t", "90",
             "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
            capture_output=True).stdout
        y8 = np.frombuffer(raw, dtype=np.float32).copy()
        np.save(OUT / f"{stem}__8000w90.npy", y8)
        print(stem, "8000/90s", y8.shape)


if __name__ == "__main__":
    main()
