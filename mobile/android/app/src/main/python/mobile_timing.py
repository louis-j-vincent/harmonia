"""Chronométrage par brique sur le téléphone (P2d, 2026-09-23) — PAS dans
`harmonia/`, et sans effet sur ce qui est calculé.

POURQUOI PAS LES `report(...)` DU PIPELINE. `analyze_steps` publie six jalons
(`report(1..6)`), que `/api/job/<id>` ressert : ils suffisent à dire « les
battues sont finies », pas à dire « le décodage a coûté 4 s et le réseau
musx 90 ». Or sur téléphone c'est exactement la répartition qui décide si
la porte T5 (4 min de musique en moins de 2 min de calcul) passe ou non. Et
deux étages tournent dans des THREADS qui se recouvrent (battues et musx) :
des jalons muraux ne se soustraient pas.

CE QUE ÇA FAIT : enveloppe les six fonctions de `backends_mobile` d'un
chronomètre, écrit une ligne dans logcat à chaque fin d'appel, et cumule par
brique. `install()` doit être appelé APRÈS `backends_mobile.install()` (il
enveloppe les fonctions déjà posées dans le registre) et rend les cumuls par
`GET /api/debug/timings`.

CE QUE ÇA NE FAIT PAS : mesurer la mémoire (c'est `adb shell dumpsys
meminfo` depuis le Mac, hors process) ; distinguer le temps CPU du temps
mural (les briques numpy peuvent multi-threader via BLAS).
"""
from __future__ import annotations

import logging
import threading
import time

from flask import Blueprint, jsonify

log = logging.getLogger("harmonia.mobile.timing")

bp = Blueprint("mobile_timing", __name__)

_lock = threading.Lock()
_totals: dict[str, dict] = {}


def _record(name: str, dt: float, note: str = "") -> None:
    with _lock:
        e = _totals.setdefault(name, {"calls": 0, "seconds": 0.0})
        e["calls"] += 1
        e["seconds"] = round(e["seconds"] + dt, 3)
    log.info("brique %-18s %7.2f s %s", name, dt, note)


def _wrap(name, fn, note_of=None):
    def inner(*a, **kw):
        t0 = time.time()
        try:
            out = fn(*a, **kw)
        finally:
            dt = time.time() - t0
        _record(name, dt, note_of(a, out) if note_of else "")
        return out
    inner.__name__ = getattr(fn, "__name__", name)
    inner.__doc__ = getattr(fn, "__doc__", None)
    return inner


def reset() -> None:
    with _lock:
        _totals.clear()


def totals() -> dict:
    with _lock:
        return {k: dict(v) for k, v in _totals.items()}


def install() -> None:
    """Enveloppe les briques DÉJÀ installées dans `harmonia.backends`.

    On ré-installe les mêmes fentes avec les versions chronométrées : le
    registre ne garde qu'une implémentation par fente, donc pas de double
    comptage possible. Le décodage est compté à part du reste parce que,
    sur téléphone, c'est la seule brique dont le coût dépend du conteneur
    audio et pas de la longueur du calcul.
    """
    from harmonia.backends import BACKENDS
    from mobile.harmonia_mobile import backends_mobile as bm

    def _audio_note(args, out):
        sr = args[1] if len(args) > 1 else "?"
        return f"sr={sr} n={getattr(out, 'size', '?')}"

    def _beats_note(args, out):
        try:
            return f"{len(out[0])} battues, {len(out[1])} temps forts"
        except Exception:
            return ""

    def _shape_note(args, out):
        return f"shape={getattr(out, 'shape', '?')}"

    BACKENDS.install(
        load_audio=_wrap("decodage", bm._load, _audio_note),
        duration_seconds=_wrap("duree", bm.duration_seconds),
        slice_to_wav=_wrap("slice_to_wav", bm.slice_to_wav),
        beats=_wrap("battues_pte", bm.beats_pte, _beats_note),
        musx_features=_wrap("cqt_musx", bm.musx_features_numpy, _shape_note),
        musx_posteriors=_wrap("reseau_musx", bm.musx_posteriors_numpy),
        nnls_bothchroma=_wrap("chroma_nnls", bm.nnls_bothchroma_numpy),
        find_peaks=bm._find_peaks_numpy,   # microseconde, pas chronométré
    )
    log.info("chronomètres posés sur les briques mobiles")


@bp.get("/api/debug/timings")
def api_timings():
    """Les cumuls par brique depuis le dernier `?reset=1`."""
    from flask import request
    if request.args.get("reset"):
        reset()
    return jsonify(totals())
