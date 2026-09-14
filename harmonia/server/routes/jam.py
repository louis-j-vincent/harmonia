"""Jam Mode : le tampon micro grandit, la boucle se cherche dessus.

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14). Le moteur est
`harmonia_min.jam.JamSession` (Beat This! + musx, jamais librosa — voir sa
docstring) ; `_ffmpeg` vient de `analyze.py`, seule fabrique de transcodage.

CE QUI DÉBLOQUE LE MICRO : voir `app.py` (contexte sûr, Tailscale HTTPS).

Ce que ce module ne fait PAS : transformer une boucle trouvée en chart de
bibliothèque — signalé dans `/api/jam/stop`, pas construit.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from harmonia.server.routes.analyze import _ffmpeg

log = logging.getLogger("harmonia.server.routes.jam")

bp = Blueprint("jam", __name__)

_jam_sessions: dict = {}
_jam_lock = threading.Lock()


@bp.post("/api/jam/start")
def api_jam_start():
    from harmonia_min.jam import JamSession
    sid = f"jam_{int(time.time() * 1000)}"
    with _jam_lock:
        _jam_sessions[sid] = JamSession(sr=44100)
    log.info("jam %s: session ouverte", sid)
    return jsonify({"session_id": sid})


def _jam_decode(sid: str, sess) -> None:
    """Une passe de décodage, dans son thread. Une seule à la fois par session.

    POURQUOI EN FOND et pas dans la réponse au morceau : la passe redécode TOUT
    le tampon (voir jam.py), donc elle s'allonge avec le jam — mesuré autour de
    13 s sur 80 s de tampon, quand les morceaux arrivent toutes les 8 s. Les
    enchaîner dans la requête ferait grossir la file sans fin. Ici le morceau
    est simplement ajouté au tampon et repart avec le dernier état connu ; les
    passes qui tombent pendant qu'une autre tourne sont SAUTÉES, pas empilées —
    le tampon les contient déjà, la passe suivante les verra.
    """
    tmp = Path(tempfile.mkdtemp(prefix="harmonia_jam_"))
    try:
        t0 = time.time()
        sess.update(tmp / "buf.wav")
        log.info("jam %s: passe sur %.0f s de tampon en %.1f s (boucle: %s)",
                 sid, sess.elapsed_s(), time.time() - t0,
                 "oui" if sess.cur else "pas encore")
    except Exception:
        # Jamais silencieux, jamais fatal : un morceau bruité ne doit pas tuer
        # la session, mais la trace doit exister.
        log.exception("jam %s: passe de décodage échouée", sid)
    finally:
        sess.busy = False
        shutil.rmtree(tmp, ignore_errors=True)


@bp.post("/api/jam/chunk")
def api_jam_chunk():
    """Un morceau de micro (~8 s) → il rejoint le tampon, et on rend l'état.

    L'état rendu est celui de la DERNIÈRE passe terminée (`decoding` dit si une
    autre tourne). C'est le « near-live, pas live-live » assumé : quelques
    secondes de retard contre des accords qu'on peut lire.
    """
    sid = request.form.get("session_id") or ""
    with _jam_lock:
        sess = _jam_sessions.get(sid)
    if sess is None:
        return jsonify({"error": "Session de jam inconnue ou expirée"}), 404
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "Aucun audio reçu"}), 400

    tmp = Path(tempfile.mkdtemp(prefix="harmonia_chunk_"))
    try:
        raw = tmp / "chunk.upload"
        f.save(raw)
        wav = tmp / "chunk.wav"
        if not _ffmpeg(raw, wav, ["-ac", "1", "-ar", str(sess.sr)], timeout=30):
            return jsonify({"error": "Morceau illisible"}), 400
        import soundfile as sf
        y, _sr = sf.read(wav)
        sess.append(y.mean(1) if getattr(y, "ndim", 1) > 1 else y)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not sess.busy:
        sess.busy = True
        threading.Thread(target=_jam_decode, args=(sid, sess),
                         daemon=True).start()
    return jsonify({**sess.state(), "decoding": bool(sess.busy)})


@bp.post("/api/jam/stop")
def api_jam_stop():
    """Fin de session : le tampon est libéré. Rien n'est encore enregistré dans
    la bibliothèque — transformer la boucle trouvée en chart normal est
    l'étape d'après, signalée ici et pas construite."""
    sid = (request.get_json(silent=True) or {}).get("session_id") or ""
    with _jam_lock:
        _jam_sessions.pop(sid, None)
    return jsonify({"ok": True})


# Ce que ce module ne fait PAS : l'enregistrement "un coup" (voir
# `analyze.api_record_analyze`) — Jam garde un tampon vivant entre plusieurs
# morceaux, Enregistrer en fait un seul fichier fini.
