"""Route Flask de debug pour vérifier la parité du décodeur Android contre le
Mac (P2a, 2026-09-23) — PAS dans `harmonia/`, jamais montée en dehors de ce
point d'entrée mobile. Enregistrée par `mobile_entry.start_server` juste
après `create_app()`.

`GET /api/debug/decode?file=<stem>&sr=<sr>&n=<N>` décode `<stem>.m4a` (cherché
dans `HARMONIA_AUDIO_DIR`, donc les morceaux de seed une fois copiés par
`MainActivity.seedStateOnce`) au taux `sr` via `mobile_audio.load_audio`,
et rend assez d'informations pour comparer bit-pour-bit contre une référence
Mac produite avec les mêmes fonctions que `harmonia/pipeline.py` :
n_samples, sr, les N premiers échantillons, le RMS du signal entier, et le
sha256 des octets float32 bruts (comparaison rapide sans retransmettre tout
le tableau pour les fichiers longs).

`offset` (optionnel, défaut 0) : décale la fenêtre de `N` échantillons
rendue — PAS dans le contrat P2a d'origine (qui ne prévoyait que le début du
fichier), ajouté pour vérifier qu'il n'y a pas de DÉRIVE au milieu du
morceau (une erreur d'indexation dans le rééchantillonneur polyphase
pourrait accumuler une erreur que les 2000 premiers échantillons, souvent du
silence, ne révéleraient jamais).
"""
from __future__ import annotations

import hashlib

import numpy as np
from flask import Blueprint, jsonify, request

import mobile_audio
from harmonia.settings import SETTINGS

bp = Blueprint("mobile_debug", __name__)


@bp.get("/api/debug/decode")
def decode():
    stem = request.args.get("file", "")
    sr = int(request.args.get("sr", "22050"))
    n = int(request.args.get("n", "2000"))
    offset = int(request.args.get("offset", "0"))
    if not stem:
        return jsonify({"error": "missing ?file=<stem>"}), 400

    path = SETTINGS.audio_dir / f"{stem}.m4a"
    if not path.exists():
        return jsonify({"error": f"not found: {path}"}), 404

    arr, out_sr = mobile_audio.load_audio(str(path), sr)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64)))) if arr.size else 0.0

    return jsonify({
        "file": stem,
        "sr": out_sr,
        "n_samples": int(arr.size),
        "first_n": [round(float(v), 8) for v in arr[offset:offset + n]],
        "rms": rms,
        "sha256_of_float32_bytes": digest,
    })
