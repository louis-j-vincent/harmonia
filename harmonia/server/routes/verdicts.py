"""Les arbitrages de Louis sur une page de diagnostic, remontés tout seuls.

Louis, 2026-09-16 : « c'est completement con ton truc faut que je réponde a
toutes pour copier jvais pas répondre à 200 trucs, et fais en sorte que les
réponses te remontent automatiquement si tu peux ».

Jusqu'ici chaque page d'arbitrage (`/plots/*.html`, `/reports/*.html`) gardait
ses réponses dans le `localStorage` du téléphone, et il fallait un bouton
« copier » puis un collage dans le fil. C'est-à-dire : du travail pour lui, et
des réponses perdues dès qu'il changeait d'appareil.

Ici les pages POSTent au fil de l'eau et la session lit le fichier. Le format
est volontairement OUVERT — `{"<id de ligne>": <ce que la page veut>}` — parce
que chaque page pose une question différente et qu'un schéma figé obligerait à
toucher au serveur à chaque nouvelle page. Ce que le serveur garantit, c'est le
NOM de fichier et la taille, rien de plus.

Où ça atterrit : `state/human/verdicts/<nom>.json`, donc dans la moitié SUIVIE
PAR GIT de `state/` — ces réponses sont de la vérité terrain faite à la main,
au même titre que `state/human/bass_verdicts.json`, et un arbitrage perdu est
irrécupérable (incident du 2026-08-12).

Dernier écrit gagne, comme `chart-meta` et `folders` : la page envoie toujours
son document entier, pas un delta.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.routes.verdicts")

bp = Blueprint("verdicts", __name__)

DIR = SETTINGS.repo / "state" / "human" / "verdicts"
#: un nom de page, pas un chemin — ni `..`, ni séparateur, ni point.
_NOM = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
#: de quoi tenir plusieurs centaines d'arbitrages ; au-delà c'est une erreur
#: d'appel, pas un usage.
MAX_OCTETS = 512 * 1024


def _path(nom: str):
    return DIR / f"{nom}.json" if _NOM.match(nom or "") else None


@bp.get("/api/verdicts/<nom>")
def lire(nom):
    p = _path(nom)
    if p is None:
        return jsonify({"error": "bad name"}), 400
    if not p.exists():
        return jsonify({"nom": nom, "reponses": {}, "n": 0})
    try:
        return jsonify(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        log.warning("verdicts: lecture de %s impossible : %s", nom, exc)
        return jsonify({"error": "unreadable"}), 500


@bp.post("/api/verdicts/<nom>")
def ecrire(nom):
    p = _path(nom)
    if p is None:
        return jsonify({"error": "bad name"}), 400
    doc = request.get_json(silent=True)
    if not isinstance(doc, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    reponses = doc.get("reponses")
    if not isinstance(reponses, dict):
        return jsonify({"error": "reponses must be an object"}), 400
    corps = {
        "nom": nom,
        "page": str(doc.get("page") or "")[:300],
        "maj": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(reponses),
        "reponses": reponses,
        # ce que la page veut garder à côté des réponses (l'énoncé de chaque
        # ligne, par exemple) : on le stocke sans le lire.
        "contexte": doc.get("contexte") or {},
    }
    brut = json.dumps(corps, ensure_ascii=False, indent=1)
    if len(brut.encode("utf-8")) > MAX_OCTETS:
        return jsonify({"error": "too large"}), 413
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        # écriture atomique : un arbitrage à moitié écrit serait pire que pas
        # d'arbitrage du tout, et `state/human` n'est pas régénérable.
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(brut, encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:
        log.warning("verdicts: écriture de %s impossible : %s", nom, exc)
        return jsonify({"error": "could not persist"}), 500
    log.info("verdicts %s : %d réponse(s)", nom, len(reponses))
    return jsonify({"ok": True, "n": len(reponses), "maj": corps["maj"]})
