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


@bp.get("/api/debug/beats")
def beats():
    """Beat This! SEUL, de bout en bout, sans le reste du pipeline (P2d).

    Pourquoi une route à part alors que l'analyse complète exerce déjà les
    battues : elle ne les exerce qu'APRÈS le réseau musx, qui est l'étage
    gourmand en mémoire. Tant qu'il n'est pas découpé, l'analyse meurt avant
    d'arriver aux battues, et le runtime ExecuTorch natif — l'objet même de
    P2d — ne serait mesuré par rien. Cette route l'isole : décodage
    MediaCodec -> mel numpy -> `.pte` par JNI -> post-traitement numpy.

    Rend les temps de chaque moitié (décodage / réseau+post) et TOUTES les
    battues, pour que le Mac puisse les comparer une à une plutôt que de se
    contenter d'un compte.
    """
    import time
    stem = request.args.get("file", "")
    if not stem:
        return jsonify({"error": "missing ?file=<stem>"}), 400
    path = SETTINGS.audio_dir / f"{stem}.m4a"
    if not path.exists():
        return jsonify({"error": f"not found: {path}"}), 404

    from mobile.harmonia_mobile import backends_mobile as bm
    from mobile.harmonia_mobile.beats_infer import beats_from_audio
    import mobile_beats_runtime

    t0 = time.time()
    y = bm._load(str(path), 22050)
    t_decode = time.time() - t0
    t1 = time.time()
    beats_, downbeats = beats_from_audio(y, bm._pte_path)
    t_net = time.time() - t1
    return jsonify({
        "file": stem,
        "n_samples": int(y.size),
        "seconds_decode": round(t_decode, 3),
        "seconds_beats": round(t_net, 3),
        "xnnpack_threads": mobile_beats_runtime.num_threads(),
        "n_beats": int(len(beats_)),
        "n_downbeats": int(len(downbeats)),
        "beats": [round(float(t), 6) for t in beats_],
        "downbeats": [round(float(t), 6) for t in downbeats],
    })


@bp.get("/api/debug/stages")
def stages():
    """Les étages de calcul SAUF les battues, sur le morceau ENTIER (P2d).

    Raison d'être : sur cet émulateur, Beat This! tue le process (SIGILL dans
    l'initialisation SME de XNNPACK — voir le rapport P2d), et comme les
    battues tournent dans un thread de `analyze_steps`, c'est toute l'analyse
    qui meurt. Sans cette route, le coût et la mémoire des quatre autres
    briques sur un vrai morceau ne seraient mesurés nulle part sur device.

    Ce qu'elle ne fait PAS : produire un chart (elle ne passe pas par
    `analyze_steps`), ni mesurer les battues.
    """
    import time
    import resource
    stem = request.args.get("file", "")
    chunk = request.args.get("chunk")
    path = SETTINGS.audio_dir / f"{stem}.m4a"
    if not stem or not path.exists():
        return jsonify({"error": f"not found: {stem}"}), 404

    from mobile.harmonia_mobile import backends_mobile as bm
    if chunk is not None:
        bm._musx_chunk_frames = None if chunk in ("", "none") else int(chunk)

    def rss_mb():
        # ru_maxrss est en OCTETS sur macOS, en KILO-octets sous Linux
        # (donc Android). Une confusion d'unité ici donnerait un « pic
        # mémoire » faux d'un facteur 1024 sans que rien ne le signale.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    out = {"file": stem, "musx_chunk_frames": bm._musx_chunk_frames}
    t = time.time(); y22 = bm._load(str(path), 22050)
    out["decodage_22050_s"] = round(time.time() - t, 2)
    out["n_samples_22050"] = int(y22.size)
    out["pic_rss_mo_apres_decodage"] = rss_mb()

    from mobile.harmonia_mobile.cqt_musx import song_cqt_288
    t = time.time(); cqt = song_cqt_288(y22, sr=22050)
    out["cqt_s"] = round(time.time() - t, 2)
    out["cqt_shape"] = list(cqt.shape)
    out["pic_rss_mo_apres_cqt"] = rss_mb()

    t = time.time(); probs = bm.musx_posteriors_numpy(cqt)
    out["reseau_musx_s"] = round(time.time() - t, 2)
    out["musx_shapes"] = [list(p.shape) for p in probs]
    out["pic_rss_mo_apres_musx"] = rss_mb()
    del probs, cqt

    t = time.time(); arr, times = bm.nnls_bothchroma_numpy(str(path))
    out["chroma_nnls_s"] = round(time.time() - t, 2)
    out["nnls_shape"] = list(np.asarray(arr).shape)
    out["pic_rss_mo_fin"] = rss_mb()
    return jsonify(out)


@bp.get("/api/debug/dump_pcm")
def dump_pcm():
    """Écrit le PCM décodé par MediaCodec dans un `.npy` du cache de l'app,
    pour qu'on le rapatrie (`adb pull`) et qu'on le rejoue SUR LE MAC (P2d).

    C'est la seule façon d'attribuer un écart de battues : le Mac rejoue son
    propre chemin de calcul avec le signal DU TÉLÉPHONE, et ce qui bouge
    alors est du décodeur, pas de l'arithmétique ARM. `/api/debug/decode`
    (P2a) ne rend que les N premiers échantillons et un sha256 — assez pour
    détecter un décalage, pas pour rejouer le calcul.
    """
    import os
    stem = request.args.get("file", "")
    sr = int(request.args.get("sr", "22050"))
    path = SETTINGS.audio_dir / f"{stem}.m4a"
    if not stem or not path.exists():
        return jsonify({"error": f"not found: {stem}"}), 404
    from mobile.harmonia_mobile import backends_mobile as bm
    y = bm._load(str(path), sr)
    out = os.path.join(os.environ.get("TMPDIR", "/data/local/tmp"),
                       f"{stem}__{sr}_device.npy")
    np.save(out, np.asarray(y, dtype=np.float32))
    return jsonify({"path": out, "n_samples": int(y.size), "sr": sr,
                    "bytes": os.path.getsize(out)})
