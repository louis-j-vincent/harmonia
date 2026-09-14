"""iReal Pro (recherche + export) et la recherche de tablatures Ultimate Guitar.

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14). Les moteurs
sont ceux de l'ancienne app, dans `harmonia.integrations.*` — importés tels
quels, jamais forkés (deux exportateurs donneraient deux charts iReal
différents pour le même morceau).

Ce que ce module ne fait PAS : importer une tablature en chart — une tab n'a
ni mesures ni temps, ça demande un alignement audio qui n'est pas construit
(voir la route dropée `/api/tab-import`, plan décision 3).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from harmonia.server.jobs import CHARTS_DIR

log = logging.getLogger("harmonia.server.routes.irealb")

bp = Blueprint("irealb", __name__)


@bp.post("/api/irealb-search")
def irealb_search():
    """Chercher un morceau dans la communauté iReal Pro.

    Le moteur est celui de l'ancienne app (`harmonia.irealb_fetcher`), appelé
    tel quel : deux sources (les ~2200 standards, puis le forum), fusionnées
    et dédoublonnées. Rien n'est réécrit ici — une seconde recherche iReal
    aurait été une seconde vérité pour la même question.
    """
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400
    query = f"{title} {(data.get('artist') or '').strip()}".strip()
    try:
        from harmonia.integrations.irealb_fetcher import search_community
        return jsonify({"results": search_community(query)})
    except Exception as exc:                              # noqa: BLE001
        log.exception("irealb-search failed")
        return jsonify({"error": str(exc)}), 500


@bp.get("/api/irealb-export/<file>")
def irealb_export(file):
    """Un chart → une URL `irealb://` qu'iReal Pro ouvre directement.

    L'exportateur est celui de l'ancienne app
    (`harmonia.irealb_export.chart_model_to_irealb_url`). Il attend le
    ChartModel de `harmonia.output.chart_model` ; le nôtre en diffère sur un
    seul point — un silence s'y écrit `nc: true` là où l'autre écrit
    `q: "N"`. On PROJETTE donc notre modèle sur le sien plutôt que de forker
    l'exportateur : deux exportateurs, ce serait deux charts iReal
    différents pour le même morceau.
    """
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    m = json.loads(p.read_text(encoding="utf-8"))
    projete = {
        "title": m.get("title") or "Untitled",
        "key": m.get("key") or {"tonic": 0, "mode": "major"},
        "sections": [
            {"label": s.get("label") or "A", "reps": s.get("reps", 1),
             "bars": [[{"root": c.get("root", 0),
                        "q": "N" if c.get("nc") else (c.get("q") or "")}
                       for c in bar]
                      for bar in (s.get("bars") or [])]}
            for s in (m.get("sections") or [])
        ],
    }
    if not projete["sections"]:
        return jsonify({"error": "ce chart n'a pas de sections à exporter"}), 400
    try:
        from harmonia.integrations.irealb_export import chart_model_to_irealb_url
        return jsonify({"url": chart_model_to_irealb_url(projete)})
    except Exception as exc:                              # noqa: BLE001
        log.exception("irealb export failed for %s", file)
        return jsonify({"error": str(exc)}), 500


# ── tablatures (delta2 §9): façade HTTP sur harmonia.integrations.tab_fetcher ─
# La recherche est un vrai passage sur Ultimate Guitar (curl_cffi); si la
# dépendance ou le réseau manquent, on renvoie une liste vide — l'onglet reste
# visible et l'état « rien trouvé » du shell fait le travail (choix delta2).
# L'IMPORT (`/api/tab-import`) est dropé (plan décision 3) : une tablature n'a
# ni mesures ni temps, en faire un chart demande l'alignement audio, pas construit.

@bp.post("/api/tab-search")
def tab_search():
    body = request.get_json(silent=True) or {}
    q = (body.get("q") or "").strip()
    if not q:
        return jsonify({"results": []})
    try:
        from harmonia.integrations.tab_fetcher import search_tabs
        found = search_tabs(q, tab_types=("Chords",), max_results=12)
    except Exception:  # noqa: BLE001 — curl_cffi absent / UG down: liste vide
        log.warning("tab-search failed for %r", q, exc_info=True)
        return jsonify({"results": []})
    return jsonify({"results": [{
        "id": r.id,
        "title": r.song_name,
        "artist": r.artist_name,
        "kind": (r.tab_type or "").lower() == "chords" and "chords" or r.tab_type,
        "rating": round(r.rating, 1) if r.rating else None,
        "url": r.tab_url,
    } for r in found]})


# Ce que ce module ne fait PAS : la recherche YouTube/locale (voir
# `analyze.py`) — deux moteurs de recherche différents pour deux sources
# différentes.
