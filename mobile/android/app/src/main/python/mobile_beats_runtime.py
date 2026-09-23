"""Pont Python vers le runtime ExecuTorch NATIF d'Android (P2d, 2026-09-23) —
PAS un fichier de `harmonia/`, et pas non plus de `mobile/harmonia_mobile/`
(qui doit rester importable sur le Mac, où `java` n'existe pas).

CE QU'IL REMPLACE, ET POURQUOI. `mobile/harmonia_mobile/beats_infer.py`
exécute le `.pte` de Beat This! par `executorch.runtime` — un paquet pip qui
tire `torch`, et qui n'existe pas sur Android. Sur téléphone, le runtime
ExecuTorch est `libexecutorch.so` (AAR `org.pytorch:executorch-android`),
piloté par JNI depuis `BeatsRuntime.kt`. Ce fichier ne fait que traverser la
frontière Chaquopy.

CE QU'IL NE REMPLACE PAS — et c'est le point : le mel-spectrogramme, le
découpage en blocs de 1500 trames, l'agrégation et le post-traitement
restent le MÊME code numpy que sur le Mac. Un seul endroit change, l'appel
au réseau, donc un écart de battues entre l'émulateur et le Mac ne peut
venir que de deux choses : le décodeur audio, ou l'arithmétique du réseau
sur ARM. C'est exactement la séparation que P2d doit mesurer.

Usage (une fois, depuis `mobile_entry.start_server`) :

    import mobile_beats_runtime
    mobile_beats_runtime.install(pte_path)   # AVANT d'importer le pipeline
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("harmonia.mobile.beats")

_BeatsRuntime = None
_loaded_path: str | None = None


def _runtime():
    global _BeatsRuntime
    if _BeatsRuntime is None:
        from java import jclass
        _BeatsRuntime = jclass("fr.harmonia.app.BeatsRuntime")
    return _BeatsRuntime


def num_threads() -> int:
    """Les threads XNNPACK réellement demandés (réglés sur le nombre de
    coeurs par `BeatsRuntime.load`). Sert au rapport de vitesse."""
    try:
        return int(_runtime().numThreads())
    except Exception:
        return 0


def run_chunk(spect_chunk: np.ndarray, pte_path=None):
    """UN bloc de mel dedans, les deux têtes de logits dehors.

    Signature identique à `beats_infer._run_pte` (c'est tout l'intérêt de la
    fente) : `(1500, 128) float32 -> ((1500,), (1500,))`.
    """
    x = np.ascontiguousarray(spect_chunk, dtype=np.float32)
    n_frames = x.shape[0]
    shape = [1, int(x.shape[0]), int(x.shape[1])]
    flat = _runtime().forward(x.reshape(-1), shape)
    out = np.frombuffer(flat, dtype=np.float32) if not isinstance(flat, np.ndarray) else flat
    out = np.asarray(out, dtype=np.float32)
    # On NE DEVINE PAS le découpage. Le graphe a deux sorties de `n_frames`
    # chacune ; si la longueur totale n'est pas exactement 2 x n_frames, le
    # modèle embarqué n'est pas celui qu'on croit — mieux vaut une exception
    # ici qu'un chart entier construit sur des logits mal recoupés (règle
    # n°1 de CLAUDE.md : les bugs de calibration silencieux rendent des
    # chiffres plausibles).
    if out.size != 2 * n_frames:
        raise RuntimeError(
            f"sortie du .pte de taille {out.size}, attendu {2 * n_frames} "
            f"(2 têtes x {n_frames} trames) — modèle inattendu")
    return out[:n_frames].copy(), out[n_frames:].copy()


def install(pte_path: str, threads: int = 0) -> int:
    """Charge le `.pte` et branche `run_chunk` dans `beats_infer`.

    Rend le nombre de threads XNNPACK retenu. Lève si le `.pte` est absent
    ou si le runtime natif refuse de le charger — jamais de repli silencieux
    vers le runtime python (qui n'existe de toute façon pas ici).
    """
    global _loaded_path
    if not os.path.exists(pte_path):
        raise FileNotFoundError(f"pte absent: {pte_path}")
    n = threads if threads > 0 else (os.cpu_count() or 1)
    if not _runtime().load(str(pte_path), int(n)):
        raise RuntimeError(f"BeatsRuntime.load a échoué sur {pte_path}")
    _loaded_path = str(pte_path)

    from mobile.harmonia_mobile import beats_infer
    beats_infer.set_chunk_runner(run_chunk)
    got = num_threads()
    log.info("runtime ExecuTorch natif branché (%s, %d threads)", pte_path, got)
    return got
