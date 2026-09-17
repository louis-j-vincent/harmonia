"""L'outil sections du chart, Soudure, /ssm, l'algo des phrases à 4 mots, et
les sections annotées à la main — tout ce qui découpe un morceau en blocs.

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14). `CHARTS_DIR`
vient de `jobs.py` (une seule définition, voir sa docstring) ; `SECTIONS_DIR`
et `SECTIONS_DRAFT_DIR` viennent de `SETTINGS.sections_dir` /
`SETTINGS.sections_draft_dir` (sprint 15 : `state/human/`, suivi par git —
c'est la vérité terrain des sections écrite à la main).

Ce que ce module ne fait PAS : détecter des sections tout seul (le moteur de
prod est `harmonia.sections`, sprint 9) ; savoir empiler des accords (voir
`harmonia.refold`/`harmonia.soudure.sections_pour_chart`, importés tels
quels — portés depuis `harmonia_min` au sprint 22).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify, make_response, request

from harmonia.server.jobs import CHARTS_DIR
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.routes.sections")

bp = Blueprint("sections", __name__)

AUDIO_DIR = SETTINGS.audio_dir
SECTIONS_DIR = SETTINGS.sections_dir
#: Les gestes de l'outil du chart (brouillons) — séparés des 18
#: annotations faites à la main, qui sont la vérité terrain du projet.
SECTIONS_DRAFT_DIR = SETTINGS.sections_draft_dir

#: Le dernier morceau calculé pour la page Soudure (clé: fichier + mtime).
_SOUDURE_CACHE: dict = {}


def _safe_stem(stem: str) -> str:
    return "".join(c for c in stem if c.isalnum() or c in "._-")[:120]


@bp.get("/soudure/<file>")
def soudure(file):
    """Le jeu de soudure sur UNE chanson de la bibliothèque (Louis,
    2026-08-13 : « mets le moi comme une option sur chaque chanson »).

    La page est `docs/plots/soudure.html`, autonome et inchangée : on lui pose
    simplement son `window.SONG` devant, comme le fait déjà
    `scripts/soudure_pages.py` pour les morceaux du banc. Un seul fichier, une
    seule page — pas de copie du moteur ici.
    """
    from harmonia.soudure import song_du_chart
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    # La recherche de coutures coûte jusqu'à 2,9 s sur le plus long morceau
    # (213 mesures) : on garde le résultat tant que le chart n'a pas bougé,
    # sinon chaque aller-retour depuis le chart le fait repayer.
    cle = (p.name, p.stat().st_mtime_ns)
    song = _SOUDURE_CACHE.get(cle)
    if song is None:
        chart = json.loads(p.read_text(encoding="utf-8"))
        song = song_du_chart(chart, audio_dir=AUDIO_DIR)
        if song:
            _SOUDURE_CACHE.clear()           # un seul morceau à la fois suffit
            _SOUDURE_CACHE[cle] = song
    if not song:
        return ("<!doctype html><meta charset=utf-8><div style=\"font:16px "
                "-apple-system,system-ui,sans-serif;max-width:26rem;"
                "margin:22vh auto;padding:0 1.5rem;color:#1c1c1c\">"
                "<p>Ce chart n’a pas assez de mesures pour faire une bande.</p>"
                f"<a href=\"/?open={file}\" style=\"color:#8a2b2b\">"
                "Retour au chart</a></div>"), 404
    gabarit = (SETTINGS.repo / "docs" / "plots" / "soudure.html").read_text(encoding="utf-8")
    tete = ("<script>window.SONG = "
            + json.dumps(song, ensure_ascii=False, separators=(",", ":"))
            + ";</script>\n")
    page = gabarit.replace("<body>", "<body>\n" + tete, 1)
    page = page.replace("<title>Soudure</title>",
                        f"<title>Soudure — {song['titre']}</title>", 1)
    resp = make_response(page)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    # Le gabarit change quand on corrige la page : pas de cache, sinon Safari
    # ressert une version périmée (déjà payé le 2026-08-13 sur le son).
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@bp.get("/ssm/<file>")
def page_ssm(file):
    """La matrice SSM du morceau, cliquable, avec tête de lecture.

    Louis, 2026-08-16 : « je veux une matrice ssm avec playhead cliquable »,
    puis « branche-le moi en direct sur chaque chanson ».

    Le moteur est `harmonia/ssm_page.py` — le MÊME que celui qui écrit les
    pages statiques de `docs/plots/ssm_*.html` (`scripts/ssm_playhead.py`).
    Une seule fabrique : deux copies de la page divergeraient, et c'est celle
    du serveur qu'il ouvrira depuis le chart.

    Coût : la grille sort du chart et les postérieurs musx sont en cache
    disque (clé = stem), donc ~1 s. `base_url` reste vide — la page est servie
    par ce serveur, ses liens relatifs tombent déjà sur les bonnes routes.
    """
    from harmonia.ssm_page import page_html
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    chart = json.loads(p.read_text(encoding="utf-8"))
    html = page_html(chart, audio_dir=AUDIO_DIR)
    if html is None:
        # Pas de 404 muet : un chart trop court ou sans audio sur disque est
        # une raison compréhensible, et il faut une porte de sortie.
        return ("<!doctype html><meta charset=utf-8><div style=\"font:16px "
                "-apple-system,system-ui,sans-serif;max-width:26rem;"
                "margin:22vh auto;padding:0 1.5rem;color:#1c1c1c\">"
                "<p>Pas de matrice pour ce morceau : il faut au moins quatre "
                "mesures et son audio sur le disque.</p>"
                f"<a href=\"/?open={Path(file).stem}\" style=\"color:#8a2b2b\">"
                "Retour au chart</a></div>"), 404
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    # Même règle que /soudure : la page change quand on la corrige, et Safari
    # resservirait une version périmée (déjà payé le 2026-08-13 sur le son).
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@bp.post("/api/phrases4")
def phrases4():
    """« Appliquer » : re-inférer les sections à partir des soudures de Louis.

    Louis, 2026-08-14 : « quand on a soudé des sections, il faut un bouton
    appliquer qui re-infère les sections via l'algo des phrases à 4 mots, mais
    avec nos sections déjà fixées ».

    Le corps porte l'état de la bande — les jetons TELS QUE Louis les a soudés,
    et les bornes en mesures — et rien d'autre : la route est sans mémoire,
    donc la page de recherche sous /plots s'en sert aussi bien que l'app.

    L'algorithme est celui de `scripts/quatre_mots.py`, importé de
    `harmonia.phrases4` — pas une seconde version. Ses soudures à lui
    entrent comme point de DÉPART : l'agglomération les prolonge, elle ne peut
    pas les défaire.
    """
    from harmonia.phrases4 import phrases
    d = request.get_json(silent=True) or {}
    jetons = d.get("jetons") or []
    bornes = d.get("bornes") or []
    if not jetons or len(bornes) < 2:
        return jsonify({"error": "jetons et bornes requis"}), 400
    try:
        depart = [(int(j[0]), int(j[1]), str(j[2])) for j in jetons]
        bornes = [int(b) for b in bornes]
    except (TypeError, ValueError, IndexError):
        return jsonify({"error": "jetons mal formés"}), 400
    if depart[0][0] != 0 or depart[-1][1] != len(bornes) - 1:
        return jsonify({"error": "les jetons ne couvrent pas la chanson"}), 400
    mot = "".join(t[2] for t in depart)
    secs, info = phrases(mot, depart=depart)
    return jsonify({"cible": info["cible"], "couts": info["couts"],
                    "sections": [
        {"label": s["label"], "prime": bool(s["prime"]),
         "reste": bool(s["queue"]),
         "mesure_debut": bornes[s["j0"]] + 1, "mesure_fin": bornes[s["j1"]],
         "j0": s["j0"], "j1": s["j1"], "type": s["type"]}
        for s in secs]})


@bp.post("/api/soudure/valider/<file>")
def soudure_valider(file):
    """Écrire le découpage de la Soudure DANS le chart, puis y renvoyer.

    Louis, 2026-08-14 : « quand la soudure est appliquée, on est redirigé sur
    le chart avec la nouvelle structure ».

    C'est la seule route de l'outil qui TOUCHE aux données. Deux précautions :
    le chart d'avant est copié dans `state/cache/charts.bak_soudure/` (une
    session concurrente peut travailler sur ces fichiers, et un découpage se
    regrette — cette copie est une sécurité de charts régénérables, pas de
    l'état humain : elle vit sous `cache_dir`, pas `human_dir`), et on refuse
    d'écrire un découpage qui ne couvre pas le morceau — un chart à trous
    serait pire que l'ancien.
    """
    from harmonia.soudure import sections_pour_chart
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    secs = (request.get_json(silent=True) or {}).get("sections") or []
    if not secs:
        return jsonify({"error": "aucune section"}), 400
    chart = json.loads(p.read_text(encoding="utf-8"))
    n = chart.get("nBars") or 0
    try:
        couvert = set()
        for s in secs:
            couvert |= set(range(int(s["mesure_debut"]) - 1, int(s["mesure_fin"])))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "sections mal formées"}), 400
    manque = sorted(set(range(n)) - couvert)
    if manque:
        return jsonify({"error": f"{len(manque)} mesures ne sont dans aucune "
                                 f"section (la première est la {manque[0] + 1})"}), 400
    # LES ACCORDS EMPILÉS SONT UNE CONSÉQUENCE DES SECTIONS, donc on les refait
    # avec CELLES-CI (Louis, 2026-08-17 : « lorsqu'on renomme les sections, on
    # retourne sur le brut et donc pas les accords renommés, qui eux sont la
    # CONSÉQUENCE des sections »). On repart du chart brut — la vérité terrain,
    # que le repli d'avant n'a pas touchée — et on ré-empile selon le nouveau
    # découpage. Garder l'ancien empilement ferait dire à sa structure ce qu'a
    # dit la précédente ; le jeter lui rendrait un chart moins bon qu'avant.
    from harmonia.refold import refold
    bars, rap = refold(chart, secs, AUDIO_DIR)
    neuves = sections_pour_chart(chart, secs, bars=bars,
                                 fold_report=rap.get("rapport"))
    if not neuves:
        return jsonify({"error": "aucune section utilisable"}), 400
    try:
        bak = SETTINGS.cache_dir / "charts.bak_soudure"
        bak.mkdir(parents=True, exist_ok=True)
        (bak / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        chart["sections"] = neuves
        # Le repli d'origine décrivait les ANCIENNES sections : le garder
        # ferait lire au chart une carte qui ne correspond plus au terrain.
        # Celui de `refold` décrit CE découpage-ci, il a le droit d'y rester.
        chart["fold"] = rap.get("rapport") or {}
        chart["form"] = None
        p.write_text(json.dumps(chart, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("soudure valider %s: %s", file, exc)
        return jsonify({"error": "écriture impossible"}), 500
    _SOUDURE_CACHE.clear()
    log.info("soudure: %s réécrit avec %d sections, empilement %s (copie "
             "dans %s)", p.name, len(neuves),
             (f"{rap.get('n_reecrites')} mesures" if rap.get("ok")
              else f"REFUSÉ ({rap.get('raison')})"), bak)
    return jsonify({"ok": True, "sections": len(neuves),
                    # dit toujours ce que l'empilement a fait, y compris rien :
                    # un repli muet ferait croire le chart amélioré quand il
                    # n'est que brut.
                    "empile": bool(rap.get("ok")),
                    "mesures_empilees": rap.get("n_reecrites", 0),
                    "empile_raison": rap.get("raison"),
                    "url": "/?open=" + Path(file).stem})


#: Ressemblance minimale pour oser donner à un bloc une lettre de LOUIS plutôt
#: qu'une lettre neuve. Mesuré le 2026-09-16 sur ses 16 découpages validés
#: (108 blocs à nommer) :
#:
#:   seuil   blocs nommés   dont justes   gestes épargnés   noms faux
#:   aucun        108           54 %            58             50
#:   0,75          88           65 %            57             31
#:   0,85          72           74 %            53             19
#:
#: Le critère n'est pas la précision seule : dans son usage, un bloc laissé
#: sans nom coûte LE MÊME geste de correction qu'un bloc mal nommé (il faut le
#: renommer dans les deux cas). Ce qu'on maximise, c'est donc le nombre de
#: blocs justes — et 0,75 en épargne presque autant que l'absence de seuil
#: (57 contre 58) avec deux fois moins de noms faux (31 contre 50).
RESSEMBLE_MIN = 0.75

#: Les sections qui, par définition, ne se rejouent PAS ailleurs dans le
#: morceau — donc jamais proposées comme gabarit de ressemblance.
#:
#: Louis, 2026-09-16 : « déjà une intro ne se rejoue pas plus tard ».
#: Vérifié sur ses 20 découpages validés : « intro » est unique dans les 15
#: morceaux qui en ont une, « outro » dans les 8 — zéro répétition, jamais.
#: C'est ce qui réparait Chain of Fools, où son « intro » d'UNE mesure
#: ressemblait à tout et raflait les dix blocs du morceau (92 % → 0 %).
#:
#: « bridge » (répétée dans 1 morceau sur 5) et « queue » (2 sur 4) ne sont PAS
#: dans cette liste : elles sont le plus souvent uniques, mais pas toujours, et
#: une règle dure s'y tromperait.
JAMAIS_REJOUEES = {"intro", "outro"}


def _plus_proche(ressemblance, b0: int, longueur: int) -> str | None:
    """La lettre de Louis dont la plage ressemble le plus à ce bloc, ou None.

    `ressemblance` = (SSM mesure×mesure, {lettre: (début, longueur)}). La
    comparaison est la diagonale bloc-à-bloc de la SSM (`_diag`), la même que
    l'outil de sections emploie pour chercher une reprise.
    """
    S, gabarits = ressemblance
    if longueur <= 0:
        return None
    from harmonia.sections.similarity import _diag
    # ESSAYÉ ET RETIRÉ le même jour : départager les ressemblances proches par
    # la LONGUEUR du gabarit (un bloc de 8 mesures est un A de 8 plutôt qu'un B
    # de 10). Motivé par Chain of Fools, où toutes les lettres sont fausses
    # alors que la structure est juste — mais mesuré, ça ne le réparait pas
    # (les deux scores n'y sont pas à égalité) et ça coûtait Easy On Me
    # (100 % → 90 % de paires bien groupées). Un réglage qui ne gagne rien ne
    # reste pas.
    best, score = None, RESSEMBLE_MIN
    for lab, (t0, tl) in gabarits.items():
        sc = _diag(S, t0, b0, max(1, min(longueur, tl)))
        if sc > score:
            best, score = lab, sc
    return best


@bp.post("/api/sections/inferer/<file>")
def sections_inferer(file):
    """Ce que Louis a surligné + ce que l'algo des quatre mots en déduit.

    Corps : {humain: [{label, mesure_debut, mesure_fin}]} — ses coups de
    surligneur, en mesures 1-indexées, fin incluse.

    LA RÈGLE DE NOMMAGE, qui est tout l'intérêt : ses sections à lui sont
    figées et gardent SON nom ; celles que l'algorithme trouve avec le MÊME
    contenu prennent aussi son nom — c'est le « infère les sections
    similaires » de sa demande. Le reste reçoit des lettres neuves, choisies
    parmi celles qu'il n'a pas déjà utilisées, pour qu'un B de l'algo ne
    puisse jamais être confondu avec un B de sa main.
    """
    from harmonia.phrases4 import LETTERS, phrases
    from harmonia.soudure import mot_sur_traits, song_du_chart, traits_propres
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    chart = json.loads(p.read_text(encoding="utf-8"))
    song = song_du_chart(chart, audio_dir=AUDIO_DIR)
    if not song:
        return jsonify({"error": "chart trop court"}), 400
    mot, bornes = song["mot"], song["jetons"]
    humain = (request.get_json(silent=True) or {}).get("humain") or []

    # SES TRAITS SONT LES BORNES DE LA GRILLE — ils ne sont plus arrondis sur
    # elle (Louis, 2026-09-17 : « du moment qu'un humain annote une section, il
    # n'y a pas à le corriger, c'est LA vérité terrain, et c'est lui qui
    # définit où commence la chanson »).
    #
    # Avant, la grille de bi-mesures était posée de deux en deux depuis la
    # mesure 1, sans lui, puis ses traits y étaient quantifiés : sur ses 18
    # découpages annotés, **52 débuts de section sur 191 reculaient d'une
    # mesure et 27 traits sur 191 étaient jetés en silence** parce que leur
    # jeton de départ était déjà pris par le trait précédent. Un `A` tracé
    # mesure 10 ressortait mesure 9. C'est l'explication du symptôme noté dans
    # `docs/known_issues.md` — « un découpage parfait noté 0 %, décalé d'un
    # cran » : la machine trouvait bien ses blocs, c'est la quantification de
    # SES traits qui les décalait.
    #
    # Les morceaux à phase paire n'en voyaient rien, d'où un bug qui paraissait
    # capricieux : le décalage ne dépendait que de la parité de son trait.
    gardes, perdus = traits_propres(humain, len(bornes) - 1)
    if gardes:
        refait = mot_sur_traits(chart, [(b0, b1) for b0, b1, _ in gardes],
                                audio_dir=AUDIO_DIR)
        if refait:
            bornes, mot, _source = refait
    if perdus:
        # Jamais muet : l'ancien code les faisait disparaître sans trace.
        log.warning("sections %s: %d trait(s) écarté(s) — %s", file, len(perdus),
                    "; ".join(f"{t.get('label')}: {t['raison']}" for t in perdus))
    index = {b: j for j, b in enumerate(bornes)}
    fixes = []                       # (j0, j1, label) triés, sans chevauchement
    for b0, b1, lab in gardes:
        j0, j1 = index.get(b0), index.get(b1 + 1)
        if j0 is None or j1 is None:
            # Ne peut arriver que si la grille n'a pas été refaite (chart trop
            # court pour `mot_sur_traits`) : on le dit plutôt que d'arrondir.
            perdus.append({"label": lab, "raison": "hors de la grille"})
            continue
        fixes.append((j0, j1 - 1, lab))

    depart, j = [], 0
    for j0, j1, _lab in fixes:
        while j < j0:
            depart.append((j, j + 1, mot[j]))
            j += 1
        depart.append((j0, j1 + 1, mot[j0:j1 + 1]))
        j = j1 + 1
    while j < len(mot):
        depart.append((j, j + 1, mot[j]))
        j += 1

    # SES TRAITS SONT DES MURS. Sans le gel, `merges4` reprend l'agglomération
    # à partir d'eux et continue de les souder ENTRE EUX : sur Let It Be, son
    # couplet (4 mots) et son refrain (2 mots) tenaient ensemble sous cible=6,
    # fusionnaient, et le bloc soudé — qui ne correspondait plus à aucune de
    # ses sections — repartait sous une lettre de la machine. **15 sections
    # envoyées, 1 seule rendue sous son nom** (2026-08-17). Prolonger une
    # soudure est la règle de l'outil Soudure (/api/phrases4, inchangée) ;
    # ici un trait est une section entière, elle se garde telle quelle.
    secs, info = phrases(mot, depart=depart,
                         geles={(j0, j1 + 1) for j0, j1, _lab in fixes})
    # le contenu de chacune de ses sections -> son nom
    par_contenu = {mot[j0:j1 + 1]: lab for j0, j1, lab in fixes}
    # Les plages qu'il a VRAIMENT tracées, pour les distinguer à l'écran de
    # celles où l'algorithme a propagé son nom. C'est toute la différence entre
    # « c'est moi qui l'ai dit » et « la machine a suivi », et c'est ce qui
    # rend l'outil relisible : sans ça il ne saurait plus ce qu'il a affirmé.
    tracees = {(j0, j1) for j0, j1, _ in fixes}
    siens = set(par_contenu.values())
    libres = [c for c in LETTERS if c not in siens]
    renom, k = {}, 0
    out = []
    # SES BRIQUES SERVENT AUSSI À NOMMER CE QUI NE LEUR EST PAS IDENTIQUE
    # (Louis, 2026-09-16 : « les sections suivantes devraient automatiquement
    # être complétées en cherchant le même pattern plusieurs fois dans la
    # chanson via les matrices ssm », puis « on lui ajoute l'info de quelles
    # sont les vraies briques des sections »).
    #
    # Le nommage ci-dessus ne propage que sur une égalité EXACTE du mot : un
    # refrain dont un seul jeton diffère repartait sous une lettre de machine.
    # En second recours seulement, on compare le bloc aux plages que Louis a
    # tracées, par la SSM chord-tone (`sections.similarity`, la même que la
    # détection) : s'il ressemble assez à l'une d'elles, il prend SA lettre.
    #
    # MESURÉ sur ses 16 découpages validés, en simulant son geste (il marque la
    # 1re occurrence de chaque lettre, puis valide) : l'accord lettre-par-mesure
    # sur ce qu'il n'a PAS marqué passe de 36 % à 55 %. Le seuil vient du genou
    # de la courbe précision/couverture (0,85 : 74 % des blocs nommés sont
    # justes, contre 54 % sans seuil ; au-delà la précision plafonne). Une
    # lettre FAUSSE est pire qu'une lettre neuve — elle a l'air d'une
    # affirmation de sa part — donc on préfère la précision à la couverture.
    #
    # CE QUE ÇA NE RÉSOUT PAS : la vérité terrain est son propre découpage, que
    # lui-même dit imparfait ; ces 55 % mesurent l'accord avec lui, pas la
    # justesse musicale. Et la SSM est HARMONIQUE : deux sections qui tournent
    # sur la même boucle (couplet/refrain de soul ou de funk) restent
    # indiscernables ici — c'est la limite prouvée ce jour-là, la voie mélodie
    # ayant été supprimée au refactor (voir `section_tool.substrates`).
    ressemblance = None
    if par_contenu:
        try:
            from harmonia import musx as _musx
            from harmonia.sections.similarity import ssm
            stem_a = Path(chart.get("audio_url") or "").stem
            audio_a = AUDIO_DIR / f"{stem_a}.m4a"
            if stem_a and audio_a.exists():
                ressemblance = (
                    ssm(_musx.frame_posteriors(audio_a)[0], chart["barGrid"]),
                    {lab: (bornes[j0], bornes[j1 + 1] - bornes[j0])
                     for j0, j1, lab in fixes
                     if lab.strip().lower() not in JAMAIS_REJOUEES})
        except Exception:                                    # noqa: BLE001
            # Jamais muet : sans ressemblance on retombe sur les lettres
            # neuves, ce qui est l'ancien comportement — mais on veut savoir.
            log.exception("nommage par ressemblance indisponible")
            ressemblance = None
    for s in secs:
        t = s["type"]
        if t in par_contenu:
            lab = par_contenu[t]
            source = "humain" if (s["j0"], s["j1"] - 1) in tracees else "propage"
        elif ressemblance and (proche := _plus_proche(
                ressemblance, bornes[s["j0"]],
                bornes[s["j1"]] - bornes[s["j0"]])):
            lab, source = proche, "ressemble"
        else:
            if s["label"] not in renom:
                renom[s["label"]] = libres[k % len(libres)] if libres else s["label"]
                k += 1
            lab, source = renom[s["label"]], "algo"
        out.append({"label": lab + ("′" if s["prime"] else ""),
                    "mesure_debut": bornes[s["j0"]] + 1,
                    "mesure_fin": bornes[s["j1"]],
                    "source": source, "reste": bool(s["queue"]), "type": t})
    return jsonify({"sections": out, "cible": info["cible"],
                    "ecartes": [{"label": t.get("label"), "raison": t["raison"]}
                                for t in perdus]})


# La page /sections/<file> a vécu une heure le 2026-08-15 : Louis voulait
# annoter SUR le chart de l'app, pas sur une page à part (« mets moi le MÊME
# chart que d'habitude, la même interface »). L'outil est revenu dans le
# chart ; seule la route d'inférence ci-dessus lui survit, et c'est elle qu'il
# appelle.


@bp.post("/api/section-repeats/<file>")
def api_section_repeats(file):
    """« Je viens de passer le doigt sur ces mesures : où ça se rejoue ? »

    Corps {b0, b1, claimed:[mesures déjà prises par les lettres validées],
    thr?, melody?} → la réponse de `section_tool.find_repeats`.

    Le stem de l'audio est relu dans le chart plutôt que reçu du client :
    c'est la même règle que /api/bar1, et ça évite qu'une page fabrique un
    chemin. Les postérieures musx sont en cache disque (clé = stem), donc
    l'appel tient largement dans un geste — la mélodie, elle, coûterait une
    séparation de voix et reste sur demande explicite.
    """
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    body = request.get_json(silent=True) or {}
    if body.get("b0") is None or body.get("b1") is None:
        return jsonify({"error": "b0/b1 required"}), 400
    # Valider AVANT d'appeler le moteur : un type faux y levait une exception
    # rendue en 500 (page HTML), et le shell fait `r.json()` dessus — il
    # cassait sans rien afficher.
    try:
        b0 = int(body["b0"])
        b1 = int(body["b1"])
    except (TypeError, ValueError, OverflowError):
        return jsonify({"error": "b0/b1 doivent être des entiers"}), 400
    thr = body.get("thr")
    if thr is not None:
        try:
            thr = float(thr)
        except (TypeError, ValueError):
            return jsonify({"error": "thr doit être un nombre"}), 400
        if not 0.0 <= thr <= 1.0:
            return jsonify({"error": "thr doit être entre 0 et 1"}), 400
    claimed = body.get("claimed") or []
    if not isinstance(claimed, list):
        return jsonify({"error": "claimed doit être une liste de mesures"}), 400
    try:
        claimed = [int(x) for x in claimed]
    except (TypeError, ValueError):
        return jsonify({"error": "claimed doit contenir des entiers"}), 400
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(model.get("audio_url") or "").stem or \
        Path(file).stem.removeprefix("min_")
    audio = AUDIO_DIR / f"{stem}.m4a"
    try:
        from harmonia import musx as _musx
        from harmonia import section_tool as st
        triad = _musx.frame_posteriors(audio)[0]
        out = st.find_repeats(
            model["barGrid"], triad, b0, b1,
            audio=audio if body.get("melody") else None,
            melody=bool(body.get("melody")),
            thr=thr, claimed_bars=claimed)
    except Exception as exc:  # noqa: BLE001 — l'UI affiche l'erreur
        log.exception("section-repeats failed for %s", file)
        return jsonify({"error": f"repeat search failed: {exc}"}), 500
    out["file"] = Path(file).stem
    out["stem"] = stem
    out["n_bars"] = len(model["barGrid"]) - 1
    return jsonify(out)


def _sections_known(stem: str, model: dict) -> dict:
    """Ce qu'on sait déjà des sections de ce morceau, par ordre de confiance.

    Louis, 2026-08-09 : « même quand les sections sont écrites, on devrait
    pouvoir les modifier dans le même outil, et il devrait aussi être
    présent pour les chansons déjà annotées. » L'outil doit donc OUVRIR sur
    l'existant, jamais sur une page blanche — sinon modifier une annotation
    veut dire la refaire.

    Ordre : la vérité écrite à la main d'abord (`state/human/sections/`), puis le
    brouillon en cours (`sections_draft/`), puis, à défaut, ce que le
    détecteur a trouvé et que le chart affiche. Le dernier est le seul qui
    ne vient pas de lui : il est étiqueté comme tel pour que l'UI le dise.
    """
    # Le PLUS RÉCENT des deux gagne, pas la vérité par principe : l'outil
    # sauvegarde un brouillon à chaque geste, et donner systématiquement la
    # priorité au fichier validé faisait disparaître tout le travail de
    # correction dès qu'on refermait l'outil sans enregistrer (audit
    # 2026-08-10). La réponse dit toujours d'où ça vient.
    found = []
    for d, src in ((SECTIONS_DIR, "truth"), (SECTIONS_DRAFT_DIR, "draft")):
        f = d / f"{stem}.json"
        if not f.exists():
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        secs = [x for x in (doc.get("sections") or [])
                if isinstance(x, dict) and "b0" in x and "b1" in x]
        if secs:
            found.append((f.stat().st_mtime, src, secs,
                          bool(doc.get("validated"))))
    if found:
        found.sort(reverse=True)                      # le plus récent d'abord
        _, src, secs, validated = found[0]
        return {"source": src, "sections": secs, "validated": validated}
    # à défaut : les sections du chart lui-même, une entrée par passage
    out = []
    for s in model.get("sections", []):
        for rng in s.get("barRanges", []) or []:
            if len(rng) == 2:
                out.append({"label": s.get("label") or "?",
                            "b0": int(rng[0]), "b1": int(rng[1])})
    out.sort(key=lambda s: s["b0"])
    return {"source": "chart", "sections": out, "validated": False}


@bp.post("/api/sections/<stem>")
def save_sections(stem):
    stem = _safe_stem(stem)
    if not stem:
        return jsonify({"error": "bad stem"}), 400
    doc = request.get_json(silent=True) or {}
    secs = doc.get("sections", [])
    if not isinstance(secs, list) or not all(isinstance(x, dict) for x in secs):
        # Il écrivait le fichier PUIS plantait en comptant : la vérité terrain
        # se retrouvait invalide sur disque avec un 500 côté client.
        return jsonify({"error": "sections doit être une liste d'objets"}), 400
    try:
        SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        (SECTIONS_DIR / f"{stem}.json").write_text(
            json.dumps({"stem": stem,
                        "n": doc.get("n"),
                        "validated": bool(doc.get("validated")),
                        "sections": doc.get("sections", [])},
                       ensure_ascii=False, indent=1))
    except (OSError, TypeError, ValueError) as exc:
        log.warning("sections save failed for %s: %s", stem, exc)
        return jsonify({"error": "could not persist sections"}), 500
    log.info("sections %s: %d", stem, len(doc.get("sections", [])))
    return jsonify({"ok": True, "stem": stem,
                    "count": len(doc.get("sections", []))})


@bp.get("/api/sections/<stem>")
def get_sections(stem):
    p = SECTIONS_DIR / f"{_safe_stem(stem)}.json"
    if not p.exists():
        return jsonify({"stem": stem, "sections": []})
    try:
        return jsonify(json.loads(p.read_text()))
    except (OSError, ValueError):
        return jsonify({"stem": stem, "sections": []})


@bp.get("/api/sections")
def all_sections():
    """Tout d'un coup — c'est ce que lisent les scripts d'analyse."""
    out = {}
    for p in sorted(SECTIONS_DIR.glob("*.json")) if SECTIONS_DIR.exists() else []:
        try:
            out[p.stem] = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return jsonify(out)


@bp.get("/api/section-marks/<file>")
def api_section_marks_get(file):
    """Ce que l'outil doit afficher à l'ouverture (voir `_sections_known`)."""
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = _safe_stem(Path(model.get("audio_url") or "").stem or
                      Path(file).stem.removeprefix("min_"))
    known = _sections_known(stem, model)
    known.update({"stem": stem, "n": len(model["barGrid"]) - 1})
    return jsonify(known)


@bp.post("/api/section-marks/<file>")
def api_section_marks(file):
    """Les marques validées → le morceau écrit par sections.

    Corps {marks:[{label, occurrences:[{b0,b1}]}], validated?} → la liste
    de sections, ET son écriture dans `state/human/sections/<stem>.json`, le
    fichier que Louis remplit déjà à la main dans /reports/annotate.html.
    Même schéma, même endpoint de lecture, mêmes scripts de mesure en aval :
    l'outil du chart et la page d'annotation écrivent au même endroit,
    sinon deux vérités terrain divergentes coexisteraient.
    """
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = _safe_stem(Path(model.get("audio_url") or "").stem or
                      Path(file).stem.removeprefix("min_"))
    n_bars = len(model["barGrid"]) - 1
    body = request.get_json(silent=True)
    if body is None or not isinstance(body, dict):
        # Un corps illisible valait « aucune marque » : la sauvegarde
        # automatique remplaçait alors le brouillon en cours par une liste
        # vide, donc effaçait le travail (audit 2026-08-10).
        return jsonify({"error": "corps JSON invalide"}), 400
    marks = body.get("marks")
    if marks is not None and not isinstance(marks, list):
        return jsonify({"error": "marks doit être une liste"}), 400
    from harmonia import section_tool as st
    sections = st.sections_from_marks(marks or [], n_bars)
    keep = [{"label": s["label"], "b0": s["b0"], "b1": s["b1"]}
            for s in sections if not s.get("pending")]
    # UN BROUILLON N'ÉCRASE PAS UNE VÉRITÉ TERRAIN. `state/human/sections/` porte
    # les 18 annotations faites à la main par Louis — la seule référence de
    # sections du projet, ce que lisent section_bench et section_metric.
    # L'outil sauvegarde à chaque geste ; sans séparation, le premier essai
    # remplaçait ses 7 sections de Bein Green par 23 cellules de 2 mesures
    # (constaté en test). Les gestes vont donc dans `sections_draft/`, et
    # seul un `validated: true` explicite touche la vraie annotation — en
    # gardant d'abord une copie `.bak` de ce qui était là.
    validated = bool(body.get("validated"))
    target_dir = SECTIONS_DIR if validated else SECTIONS_DRAFT_DIR
    doc = {"stem": stem, "n": n_bars, "validated": validated,
           "sections": keep}
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / f"{stem}.json"
        bak = target_dir / f"{stem}.json.bak"
        if validated and dest.exists() and not bak.exists():
            # PREMIÈRE sauvegarde seulement : le .bak doit garder l'annotation
            # d'ORIGINE. L'écraser à chaque fois faisait qu'un deuxième
            # enregistrement détruisait définitivement le travail à la main.
            bak.write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    except (OSError, TypeError, ValueError) as exc:
        log.warning("section marks save failed for %s: %s", stem, exc)
        return jsonify({"error": "could not persist sections"}), 500
    log.info("outil sections %s: %d marque(s) → %d section(s) écrite(s) dans "
             "%s", stem, len(marks or []), len(keep), target_dir.name)
    return jsonify({"ok": True, "stem": stem, "n": n_bars,
                    "sections": sections, "written": len(keep),
                    "draft": not validated})


# Ce que ce module ne fait PAS : servir le ChartModel lui-même (voir
# `library.py`) ; parler à yt-dlp ou à la recherche (voir
# `analyze.py`/`youtube.py`).
