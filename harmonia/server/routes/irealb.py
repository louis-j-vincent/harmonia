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

import html
import json
import logging
import re
import time
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


@bp.get("/tab-paroles/<file>")
def tab_paroles(file):
    """La feuille « accords sur les mots » de CE morceau, en HTML.

    Louis, 2026-09-23, depuis son iPhone : « donne-moi un lien html ouvrable ».
    C'est donc SON serveur qui fabrique la page, à partir du tab que sa propre
    machine va chercher — la même source que `/api/tab-doutes`, qui tourne déjà.

    Ce que la page pose, ce sont NOS accords, aux colonnes où le tabbeur a mis
    les siens, avec les siens en petit quand ils diffèrent. La question qu'elle
    sert à trancher est unique : est-ce que nos accords tombent sur la bonne
    syllabe ? L'accord avec le tab n'est pas le critère (Louis, 2026-08-05 :
    nos charts battent les tabs).

    Rien n'est mis en cache ici : la page se regarde, elle ne se collectionne
    pas. Elle vit sur le serveur personnel de Louis (Tailscale), au même titre
    que `/soudure/<file>` et `/ssm/<file>`.
    """
    from harmonia.integrations import tab_paroles as TP
    from harmonia.integrations.tab_chart import poser
    from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs

    chart, err = _chart_ou_404(file)
    if err:
        return err
    stem = Path(chart.get("audio_url") or "").stem
    requete = (request.args.get("q") or "").strip() or _requete(stem, chart)

    def _dire(msg: str, code: int = 200):
        return Response(
            '<!doctype html><meta charset=utf-8><meta name=viewport '
            'content="width=device-width,initial-scale=1">'
            '<div style="font:16px/1.5 -apple-system,system-ui,sans-serif;'
            'max-width:26rem;margin:22vh auto;padding:0 1.5rem;color:#1c1c1c">'
            f'<p>{msg}</p><p style="color:#9a9287;font-size:13px">'
            f'Recherche : « {requete} »</p></div>',
            status=code, mimetype="text/html")

    try:
        pose = poser(chart, requete)
    except Exception:                                        # noqa: BLE001
        # Jamais muet : une page blanche ferait croire à un morceau sans tab.
        log.exception("tab-paroles : la pose a échoué pour %s", stem)
        return _dire("La recherche de tab a échoué — voir le journal.", 502)
    if pose is None:
        return _dire("Aucun tab utilisable pour ce morceau.", 404)

    res = search_tabs(requete, tab_types=("Chords",), max_results=6)
    if not res:
        return _dire("Aucun tab trouvé.", 404)
    brut = fetch_tab_chords(res[0]).raw_content

    # NOS INSTANTS D'ABORD (2026-09-24). L'ancienne feuille allait chercher
    # notre accord à l'instant où le TABBEUR avait posé le sien : un accord
    # qu'il a mis un temps trop tôt nous faisait écrire le mauvais. Quand
    # l'alignement des mots existe — et Louis l'a validé à l'oreille — on pose
    # chaque accord devant le mot qui sonne à SON propre t0, et le tab ne sert
    # plus que de texte.
    from harmonia import cache
    from harmonia.integrations import paroles_align as PA
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    align, source = None, "colonnes du tab"
    try:
        garde = cache.load_json("paroles", audio)
        if garde and garde.get("align"):
            align = garde["align"]
    except Exception:                                        # noqa: BLE001
        log.exception("tab-paroles : cache d'alignement illisible pour %s", stem)

    # CETTE ROUTE CALCULE SON PROPRE ALIGNEMENT (2026-09-24). Elle se contentait
    # de LIRE un cache que seule `/karaoke/` remplissait : vider ce cache la
    # faisait retomber sur les colonnes du tabbeur, silencieusement du point de
    # vue de l'écran (le journal le disait, pas la page). Un couplage caché
    # entre deux routes est exactement le genre de repli muet que ce projet
    # refuse. Elle paie donc l'alignement elle-même si personne ne l'a fait.
    if align is None:
        voix = SETTINGS.data_cache / "stems" / "htdemucs" / stem / "vocals.wav"
        if voix.exists():
            try:
                mots_al = PA.mots_a_aligner(PA.lignes_de_paroles(brut))
                if mots_al:
                    align = PA.aligner(voix, mots_al)
                    cache.save_json("paroles", audio, {
                        "requete": requete, "align": align,
                        "tab": {"titre": res[0].song_name,
                                "artiste": res[0].artist_name,
                                "url": res[0].tab_url}})
                    log.info("tab-paroles %s : alignement calculé ici "
                             "(%d mots)", stem, len(align))
            except Exception:                                # noqa: BLE001
                log.exception("tab-paroles : alignement impossible pour %s",
                              stem)

    # LE DÉCOUPAGE VIENT DE NOTRE GRILLE (2026-09-24), plus des lignes du tab :
    # une ligne = deux mesures, et l'accord de la barre ouvre la ligne. Voir
    # `tab_paroles.feuille_barres` pour le cas qui l'a imposé.
    feuille, par_barres = [], False
    if align:
        # la structure humaine du tab — strophes et noms de sections — voyage
        # avec le texte, sinon la feuille est une suite de lignes indistinctes
        feuille = TP.feuille_barres(
            chart, align, structure=PA.lignes_de_paroles(brut))
        source, par_barres = "mesures", bool(feuille)
    if align and not feuille:
        feuille = TP.feuille_calee(chart, brut, align)
        source = "nos instants"
    if not feuille:
        # Repli ASSUMÉ et DIT : sans voix isolée il n'y a pas d'alignement, et
        # les colonnes du tabbeur restent la seule information de placement.
        feuille = TP.feuille(chart, pose, brut)
        source = "colonnes du tab"
    if not feuille:
        return _dire("Les colonnes du tab et ses accords ne concordent pas : "
                     "rien n'a été posé.", 409)
    n_acc = (sum(len(w["accords"]) for l in feuille for w in l["mots"])
             if (par_barres or par_mots)
             else sum(len(l["accords"]) for l in feuille))
    log.info("tab-paroles %s : %d ligne(s), %d accord(s), placement par %s, "
             "transposition %+d", stem, len(feuille), n_acc, source,
             int(pose.get("decalage") or 0))
    if par_barres:
        n_mots = sum(len(l["mots"]) for l in feuille)
        n_acc = sum(len(w["accords"]) for l in feuille for w in l["mots"])
        return Response(
            TP.page_barres(chart.get("title") or stem, feuille, pose["tab"],
                           f"{len(feuille)} ligne(s) de 2 mesures · "
                           f"{n_mots} mot(s) · {n_acc} changement(s)"),
            mimetype="text/html")
    if par_mots:
        # Le rendu par MOTS : l'accord est empilé sur son mot par le
        # navigateur, donc aucune police ne peut les séparer (voir
        # `tab_paroles.page_mots`).
        n_mots = sum(len(l["mots"]) for l in feuille)
        return Response(
            TP.page_mots(chart.get("title") or stem, feuille, pose["tab"],
                         f"{len(feuille)} ligne(s) · {n_mots} mot(s) · "
                         f"{n_acc} accord(s)"),
            mimetype="text/html")
    return Response(
        TP.page(chart.get("title") or stem, feuille, pose["tab"],
                int(pose.get("decalage") or 0)),
        mimetype="text/html")


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


@bp.get("/karaoke/<file>")
def karaoke(file):
    """Juger NOTRE calage dans le temps À L'OREILLE, sur ce morceau.

    Louis, 2026-09-23 : « bien calée, faut pas qu'il y ait de décalage ». Un
    chiffre ne répond pas à ça. Cette page joue l'audio et fait défiler deux
    choses sous une ligne fixe : nos accords, et UN TIC PAR MOT aligné. Si le
    calage est juste, le tic passe la ligne au moment où la syllabe sonne ; si
    ça dérive, ça s'entend et ça se voit au même instant.

    ELLE ÉCRIT LE MOT SOUS SON TIC depuis le 2026-09-24 (Louis : « c'est juste
    pour debug »). Le tic seul ne dit que l'INSTANT ; avec le mot on juge la
    SYLLABE, qui est la vraie question — un tic peut tomber pile sur une
    attaque et concerner le mot d'à côté. Le texte vient du tab que sa machine
    est allée chercher et ne quitte pas son serveur.

    DEUX VOIES D'ACCORDS depuis le 2026-09-24 : les nôtres, et ceux du tab
    ramenés sur notre horloge par `tab_chart.poser`. Les superposer montre
    OÙ on diverge, là où un taux d'accord global ne dit pas si on se trompe
    partout un peu ou beaucoup au même endroit.

    L'alignement coûte ~9 s de calcul plus le chargement du modèle : il est mis
    en cache par `harmonia.cache` sous la clé `<stem>__<taille>`, avec la
    requête qui l'a produit — rouvrir la page est gratuit, changer de tab
    (`?q=`) recalcule.
    """
    from harmonia import cache
    from harmonia.integrations import paroles_align as PA
    from harmonia.integrations import tab_chart as TC

    chart_p = CHARTS_DIR / f"{Path(str(file)).stem}.json"
    requete = ""

    def _dire(msg: str, code: int = 200):
        # Jamais muet : une page blanche ferait croire à un morceau muet.
        return Response(
            '<!doctype html><meta charset=utf-8><meta name=viewport '
            'content="width=device-width,initial-scale=1">'
            '<div style="font:16px/1.5 -apple-system,system-ui,sans-serif;'
            'max-width:26rem;margin:22vh auto;padding:0 1.5rem;color:#1c1c1c">'
            f'<p>{msg}</p><p style="color:#9a9287;font-size:13px">'
            f'{html.escape(str(file))}'
            + (f' · recherche « {html.escape(requete)} »' if requete else '')
            + '</p></div>', status=code, mimetype="text/html")

    if not chart_p.exists():
        return _dire("Chart inconnu.", 404)
    chart = json.loads(chart_p.read_text(encoding="utf-8"))
    stem = Path(chart.get("audio_url") or "").stem
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    if not stem or not audio.exists():
        return _dire("Ce chart n'a pas d'audio — il n'y a rien à caler.", 404)
    voix = PA.voix_de(stem)
    if not voix.exists():
        # 60 morceaux sur la bibliothèque ont une voix séparée ; le dire
        # plutôt que d'échouer sur un fichier absent.
        return _dire("Pas de voix séparée pour ce morceau : il manque "
                     f"<code>{html.escape(str(voix))}</code>. "
                     "L'alignement se fait sur la voix seule, jamais sur le "
                     "mixage — la batterie et la basse sont du bruit pour un "
                     "modèle de parole.", 404)

    requete = (request.args.get("q") or "").strip() or _requete(stem, chart)
    seuil = float(request.args.get("seuil") or PA.SCORE_DOUTEUX)
    silence = float(request.args.get("silence") or 3.0)

    garde = cache.load_json("paroles", audio)
    if not garde or garde.get("requete") != requete:
        from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs
        try:
            res = search_tabs(requete, tab_types=("Chords",), max_results=6)
        except Exception:                                     # noqa: BLE001
            log.exception("karaoke : la recherche de tab a échoué pour %s", stem)
            return _dire("La recherche de tab a échoué — voir le journal.", 502)
        if not res:
            return _dire("Aucun tab trouvé : sans texte il n'y a rien à "
                         "aligner.", 404)
        try:
            brut = fetch_tab_chords(res[0]).raw_content
        except Exception:                                     # noqa: BLE001
            log.exception("karaoke : tab illisible pour %s", stem)
            return _dire("Le tab trouvé est illisible — voir le journal.", 502)
        mots = PA.mots_a_aligner(PA.lignes_de_paroles(brut))
        if not mots:
            return _dire("Ce tab n'a que des accords, pas de paroles : rien "
                         "à placer.", 409)
        t0 = time.time()
        try:
            align = PA.aligner(voix, mots)
        except Exception as exc:                              # noqa: BLE001
            log.exception("karaoke : l'alignement a échoué pour %s", stem)
            return _dire(f"L'alignement a échoué : {html.escape(str(exc))}",
                         500)
        if not align:
            return _dire("L'alignement n'a placé aucun mot — le texte et "
                         "l'audio ne correspondent pas.", 409)
        garde = {"requete": requete, "align": align,
                 "tab": {"titre": res[0].song_name, "artiste": res[0].artist_name,
                         "url": res[0].tab_url, "note": round(res[0].rating or 0, 1)},
                 "secondes": round(time.time() - t0, 1)}
        cache.save_json("paroles", audio, garde)
        log.info("karaoke %s : %d mots alignés en %.1f s (tab %s)",
                 stem, len(align), garde["secondes"], garde["tab"]["url"])

    pistes = PA.piste(PA.avec_trous(garde["align"], seuil=seuil,
                                    silence_mini=silence))
    accords = PA.accords_du_chart(chart)

    # LES ACCORDS DU TAB, deuxième voie. `poser` coûte une requête réseau plus
    # un alignement : on le range dans le MÊME cache que les mots, et un cache
    # écrit avant l'existence de cette voie se complète tout seul à la
    # première ouverture, sans être invalidé.
    if "accords_tab" not in garde:
        try:
            pose = TC.poser(chart, garde.get("requete") or requete)
            garde["accords_tab"] = PA.accords_du_tab(pose) if pose else []
        except Exception:                                     # noqa: BLE001
            # Jamais muet, et jamais bloquant : sans la voie du tab la page
            # garde tout son sens, c'est la nôtre qu'on vient écouter.
            log.exception("karaoke : accords du tab indisponibles pour %s", stem)
            garde["accords_tab"] = []
        cache.save_json("paroles", audio, garde)
    accords_tab = garde.get("accords_tab") or []
    fin = max([a["t1"] for a in accords] + [t["t1"] for t in pistes["ticks"]]
              + [1.0])
    r = PA.resume(pistes["ticks"], seuil=seuil)
    tab = garde.get("tab") or {}
    diff = sum(1 for g in accords_tab
               if (n := next((a["nom"] for a in accords
                              if a["t0"] <= g["t0"] < a["t1"]), None))
               and n != g["nom"])
    sous = (f'{r["n"]} mots placés · score médian {r["median"]:.2f} · '
            f'{100 * r["part_douteux"]:.0f}% sous {seuil:.2f} · plus grand '
            f'trou {r["plus_grand_trou"]:.1f} s · '
            f'accords : {len(accords)} à nous, {len(accords_tab)} au tab, '
            f'{diff} en désaccord · '
            f'texte : <a href="{html.escape(tab.get("url") or "")}">'
            f'{html.escape(tab.get("titre") or "?")}</a>')
    return Response(
        PA.page(chart.get("title") or stem, sous,
                {"audio": f"/audio/{stem}.m4a", "duree": round(fin + 4, 2),
                 "accords": accords, "accords_tab": accords_tab,
                 "ticks": pistes["ticks"],
                 "silences": pistes["silences"]}),
        mimetype="text/html")
