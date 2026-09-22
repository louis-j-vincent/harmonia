"""Pont Python vers le décodeur audio Java (P2a, 2026-09-23) — PAS dans
`harmonia/`. Appelé par la route de debug (`mobile_debug.py`) et, plus tard,
par les stages du pipeline (`beats.py`, `musx.py`, `nnls_features.py`) une
fois qu'elles sont adaptées pour tourner sur téléphone (hors périmètre P2a).

`AudioDecoder.kt` (`fr.harmonia.app.AudioDecoder`, `@JvmStatic`) fait TOUT le
travail (démuxage MediaExtractor, décodage MediaCodec, downmix mono,
retrait du délai d'amorçage AAC, rééchantillonnage FIR polyphase) — ce
fichier ne fait que traverser la frontière Chaquopy et convertir le
`float[]` Java en `np.ndarray`.

Ce que ce fichier ne fait PAS : choisir le SR cible (l'appelant le fait,
comme `librosa.load(..., sr=...)` sur le Mac) ; mettre en cache quoi que ce
soit (contrairement à `harmonia.cache`, hors périmètre ici).
"""
from __future__ import annotations

import numpy as np
from java import jclass

_AudioDecoder = None


def _decoder():
    global _AudioDecoder
    if _AudioDecoder is None:
        _AudioDecoder = jclass("fr.harmonia.app.AudioDecoder")
    return _AudioDecoder


def load_audio(path: str, sr: int) -> tuple[np.ndarray, int]:
    """Équivalent mobile de `librosa.load(path, sr=sr, mono=True)`.

    Rend (float32 mono ndarray, sr). Le SR de sortie est TOUJOURS `sr`
    (AudioDecoder rééchantillonne si besoin) — jamais le SR natif du
    fichier, contrairement à `librosa.load(sr=None)`.
    """
    samples = _decoder().decode(path, int(sr))
    # Chaquopy expose les tableaux primitifs Java via le protocole buffer
    # (zero-copy) depuis la 9.x — `.copy()` pour posséder la mémoire avant
    # que le GC Java ne récupère le tableau source. Repli défensif (liste
    # Python) si jamais le buffer échoue sur une version de Chaquopy où ce
    # n'est pas exposé — non exercé si le buffer marche, gardé pour ne
    # jamais planter silencieusement la route de debug.
    try:
        arr = np.frombuffer(samples, dtype=np.float32).copy()
    except (TypeError, ValueError):
        arr = np.array(list(samples), dtype=np.float32)
    return arr, int(sr)


def duration_seconds(path: str) -> float:
    """Équivalent mobile de `pipeline.duration_seconds` (ffprobe sur le Mac) :
    lit `MediaFormat.KEY_DURATION` via `MediaExtractor`, pas de sous-processus."""
    return float(_decoder().durationSeconds(path))
