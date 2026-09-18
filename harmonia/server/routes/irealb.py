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
import re
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from harmonia.server.jobs import CHARTS_DIR
from harmonia.settings import SETTINGS

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


@bp.post("/api/irealb-search-stream")
def irealb_search_stream():
    """La même recherche, rendue AU FUR ET À MESURE (Louis, 2026-09-18).

    Une ligne JSON par lot (NDJSON) : d'abord les ~2200 standards, prêts en
    0,4 s, puis un lot par fil du forum, sur 4 à 6 s. En un seul bloc,
    l'écran restait vide six secondes alors que le premier résultat existait
    depuis la première demi-seconde.

    Le corps de la requête est lu AVANT le générateur : une fois la réponse
    commencée, le contexte de requête de Flask n'existe plus.
    """
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400
    query = f"{title} {(data.get('artist') or '').strip()}".strip()

    def lignes():
        from harmonia.integrations.irealb_fetcher import search_community_stream
        try:
            for source, lot in search_community_stream(query):
                yield json.dumps({"source": source, "results": lot},
                                 ensure_ascii=False) + "\n"
        except Exception as exc:                          # noqa: BLE001
            log.exception("irealb-search-stream failed")
            yield json.dumps({"error": str(exc)}) + "\n"
        yield json.dumps({"done": True}) + "\n"

    # `no-transform` + `X-Accel-Buffering` : sans eux un proxy peut garder la
    # réponse entière avant de la rendre, et le flux ne sert plus à rien.
    return Response(lignes(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache, no-transform",
                             "X-Accel-Buffering": "no"})


@bp.post("/api/irealb-import")
def irealb_import():
    """Une URL `irealb://` → un chart de la bibliothèque, ouvert tout de suite.

    C'est le geste qui manquait : la recherche iReal trouvait des morceaux
    depuis les sprints 11-14, et aucun ne pouvait s'ouvrir — le client
    appelait un `importIrealChart` qui n'existait nulle part, et la carte de
    résultat retombait sur le chemin YouTube (`watch?v=undefined`).

    Un chart importé est écrit comme n'importe quel autre chart, dans
    `CHARTS_DIR`, avec `audio_url` vide : c'est la seule chose qui le
    distingue pour le reste de l'app (`hasAudio` dans `/api/library`,
    `S.audioUrl` dans le shell). Rien n'est analysé, rien n'est déduit — la
    grille est celle que le musicien a écrite, c'est tout l'intérêt.

    Le compositeur d'iReal atterrit dans `chart_meta.json` comme « artiste »,
    le sidecar que Louis édite déjà à la main ; il n'entre jamais dans le
    chart lui-même (même règle que l'autosave de `jobs.py`).
    """
    data = request.get_json(silent=True) or {}
    url = (data.get("irealb_url") or "").strip()
    if not url:
        return jsonify({"error": "no irealb:// url provided"}), 400
    try:
        from harmonia.integrations.irealb_import import url_to_chart
        model = url_to_chart(url)
    except Exception as exc:                              # noqa: BLE001
        log.exception("irealb import failed")
        return jsonify({"error": f"grille iReal illisible : {exc}"}), 400

    # Deux imports du même titre ne s'écrasent pas : le second prend un
    # suffixe. Écraser silencieusement coûterait les annotations posées sur
    # le premier (`state/human/annotations/<stem>.json` est indexé par stem).
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = model["file"]
    if (CHARTS_DIR / f"{stem}.json").exists():
        n = 2
        while (CHARTS_DIR / f"{stem}_{n}.json").exists():
            n += 1
        stem = f"{stem}_{n}"
        model["file"] = stem
    (CHARTS_DIR / f"{stem}.json").write_text(
        json.dumps(model, ensure_ascii=False), encoding="utf-8")

    artist = (data.get("composer") or model["meta"].get("composer") or "").strip()
    if artist:
        try:
            from harmonia.server.jobs import META_PATH, _load_chart_meta
            meta = _load_chart_meta()
            meta[stem] = {"artist": artist[:120], "title": model["title"][:200]}
            META_PATH.parent.mkdir(parents=True, exist_ok=True)
            META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        except OSError as exc:                            # noqa: BLE001
            log.warning("chart_meta non écrit pour %s: %s", stem, exc)

    log.info("iReal importé : %s → %s (%d mesures, %d sections)",
             model["title"], stem, model["nBars"], len(model["sections"]))
    return jsonify({"file": stem, "title": model["title"],
                    "nBars": model["nBars"],
                    "sections": len(model["sections"])})


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
# L'IMPORT a été dropé au refactor pour une bonne raison — « une tablature n'a
# ni mesures ni temps, en faire un chart demande l'alignement audio, pas
# construit ». L'alignement existe depuis le 2026-09-18
# (`tab_align.poser_tout`), donc la raison est tombée et l'import est revenu,
# avec son jumeau : le tab comme DEUXIÈME AVIS sur nos propres accords.

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


DOUTES_DIR = SETTINGS.cache_dir / "tab_doutes"


def _requete(stem: str, chart: dict) -> str:
    """De quoi chercher le tab : le titre du chart, et l'artiste s'il est connu.

    Le titre d'un chart venu de YouTube porte souvent « official video » ou
    « remastered » ; on les enlève, sinon la recherche Ultimate Guitar ne
    trouve rien.
    """
    titre = (chart.get("title") or stem or "").strip()
    try:
        from harmonia.server.jobs import _load_chart_meta
        meta = (_load_chart_meta() or {}).get(stem) or {}
    except Exception:                                     # noqa: BLE001
        meta = {}
    titre = re.sub(r"\b(official|lyric|music|audio|video|hd|remaster(ed)?|"
                   r"live|4k|mv|topic)\b", " ", titre, flags=re.I)
    titre = re.sub(r"[\[\(][^\]\)]*[\]\)]", " ", titre)
    titre = re.sub(r"\s+", " ", titre).strip(" -–—_")
    artiste = (meta.get("artist") or "").strip()
    return f"{artiste} {titre}".strip() if artiste else titre


def _chart_ou_404(file: str):
    p = CHARTS_DIR / f"{Path(str(file)).stem}.json"
    if not p.exists():
        return None, (jsonify({"error": "chart inconnu"}), 404)
    return json.loads(p.read_text(encoding="utf-8")), None


@bp.get("/api/tab-doutes")
def tab_doutes_lire():
    """Les doutes déjà calculés pour ce morceau, sans rien relancer.

    Le chart s'ouvre des dizaines de fois par séance ; aller chercher un tab
    sur Ultimate Guitar à chaque ouverture serait lent et impoli. Le calcul se
    demande explicitement (POST), la lecture est gratuite.
    """
    stem = Path(str(request.args.get("file") or "")).stem
    if not stem:
        return jsonify({"doutes": [], "etat": "sans objet"})
    f = DOUTES_DIR / f"{stem}.json"
    if not f.exists():
        return jsonify({"doutes": [], "etat": "jamais demandé"})
    try:
        return jsonify(json.loads(f.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        log.warning("doutes illisibles pour %s", stem)
        return jsonify({"doutes": [], "etat": "illisible"})


@bp.post("/api/tab-doutes")
def tab_doutes_calculer():
    """Va chercher un tab et marque les accords sur lesquels il nous contredit.

    Ce n'est PAS « le tab a raison ». Nos charts battent les tabs UG
    (Louis, 2026-08-05) ; le tab est un deuxième avis, et un désaccord dit où
    écouter, pas quoi écrire. Un accord que Louis a confirmé n'est jamais
    marqué : ce qu'il a validé est la vérité terrain.
    """
    body = request.get_json(silent=True) or {}
    stem = Path(str(body.get("file") or "")).stem
    chart, erreur = _chart_ou_404(stem)
    if erreur:
        return erreur
    requete = (body.get("q") or "").strip() or _requete(stem, chart)
    try:
        from harmonia.integrations import tab_chart as TC
        pose = TC.poser(chart, requete, url=(body.get("url") or "").strip() or None)
    except Exception as exc:                              # noqa: BLE001
        log.exception("tab-doutes a échoué pour %s", stem)
        return jsonify({"doutes": [], "etat": f"échec : {exc}"}), 200
    if pose is None:
        out = {"doutes": [], "etat": "aucun tab utilisable", "requete": requete}
    else:
        d = TC.doutes(chart, pose)
        out = {"doutes": d, "etat": "ok", "requete": requete,
               "tab": pose["tab"], "transposition": pose["decalage"],
               "forme": pose["forme"]["cout"],
               "ecrits": sum(len(b) for s in (chart.get("sections") or [])
                             for b in (s.get("bars") or []))}
    try:
        DOUTES_DIR.mkdir(parents=True, exist_ok=True)
        (DOUTES_DIR / f"{stem}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:                                # noqa: BLE001
        log.warning("doutes non écrits pour %s : %s", stem, exc)
    log.info("tab-doutes %s : %d douteux sur %d écrits (%s)",
             stem, len(out["doutes"]), out.get("ecrits", 0), out["etat"])
    return jsonify(out)


@bp.post("/api/tab-import")
def tab_import():
    """Un tab posé sur l'audio d'un morceau déjà analysé → un NOUVEAU chart.

    Il ne remplace jamais l'existant : il s'écrit sous `tab_<stem>`, s'ouvre à
    côté, et Louis compare. Écraser lui coûterait ses annotations, qui sont
    indexées par stem (`state/human/annotations/<stem>.json`).

    Tout ce qui vient de l'audio — grille, temps, tonalité, départ — est repris
    du chart source. Seuls les ACCORDS et les SECTIONS changent de source.
    """
    body = request.get_json(silent=True) or {}
    stem = Path(str(body.get("file") or "")).stem
    source, erreur = _chart_ou_404(stem)
    if erreur:
        return erreur
    requete = (body.get("q") or "").strip() or _requete(stem, source)
    try:
        from harmonia.integrations import tab_chart as TC
        pose = TC.poser(source, requete,
                        url=(body.get("url") or "").strip() or None)
    except Exception as exc:                              # noqa: BLE001
        log.exception("tab-import a échoué pour %s", stem)
        return jsonify({"error": f"tab illisible : {exc}"}), 400
    if pose is None:
        return jsonify({"error": "aucun tab utilisable pour ce morceau"}), 404

    cible = f"tab_{stem}"
    if (CHARTS_DIR / f"{cible}.json").exists():
        n = 2
        while (CHARTS_DIR / f"{cible}_{n}.json").exists():
            n += 1
        cible = f"{cible}_{n}"
    titre = f"{source.get('title') or stem} · tab"
    model = TC.chart(source, pose, titre=titre, stem=cible)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    (CHARTS_DIR / f"{cible}.json").write_text(
        json.dumps(model, ensure_ascii=False), encoding="utf-8")
    log.info("tab importé : %s → %s (%d mesures, %d sections, forme %s)",
             requete, cible, model["nBars"], len(model["sections"]),
             model["meta"]["occam"]["forme"])
    return jsonify({"file": cible, "title": titre, "nBars": model["nBars"],
                    "sections": len(model["sections"]),
                    "tab": pose["tab"], "transposition": pose["decalage"],
                    "occam": model["meta"]["occam"]})


# Ce que ce module ne fait PAS : la recherche YouTube/locale (voir
# `analyze.py`) — deux moteurs de recherche différents pour deux sources
# différentes.
