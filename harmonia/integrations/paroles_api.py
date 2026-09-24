"""Les paroles par une API SOUS LICENCE, pas par un scraper.

Louis, 2026-09-24, a demandé d'aller chercher les paroles « directement
ailleurs sur internet de manière automatique ». Aspirer un site de paroles est
un outil de reproduction : refusé. Il a choisi l'autre voie — une API sous
licence, avec SA clé — et c'est celle-ci.

CE QU'UNE CLÉ ACHÈTE RÉELLEMENT, à lire avant d'en prendre une :

  * **Musixmatch**, palier développeur gratuit : l'API rend un EXTRAIT
    d'environ 30 % du texte, suivi d'une mention de copyright, et les
    conditions interdisent l'usage commercial. Le texte complet demande un
    contrat commercial. Pour notre usage — poser des accords sur les paroles
    d'un morceau qu'on travaille — 30 % ne suffit pas à couvrir un morceau.
  * **LyricFind**, **Musixmatch commercial** : texte complet, mais pas de clé
    en libre-service — il faut passer par leur équipe commerciale.

Autrement dit : cette intégration est prête, et ce qu'elle rendra dépend
entièrement du contrat derrière la clé. Sans clé, rien ne change — le texte
continue de venir du tab que l'app récupère déjà.

LA CLÉ NE VIT PAS DANS LE CODE. Elle est lue par `harmonia.settings`, le seul
endroit du projet qui touche `os.environ`, sous le nom `HARMONIA_PAROLES_CLE`.
Rien n'est écrit sur disque à part le cache de paroles habituel, qui reste sur
la machine de Louis.
"""
from __future__ import annotations

import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

#: le fournisseur par défaut ; `FOURNISSEURS` dit ce qui est câblé
DEFAUT = "musixmatch"
DELAI = 12          # secondes avant d'abandonner l'appel réseau


def _musixmatch(titre: str, artiste: str, cle: str) -> dict | None:
    """(texte, complet, source) — l'API Musixmatch, ou None si elle ne rend rien.

    `complet` est faux quand l'API a marqué l'extrait comme tronqué : c'est le
    cas du palier gratuit, et l'appelant DOIT le dire à l'écran plutôt que de
    laisser croire à des paroles entières.
    """
    import json
    base = "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get"
    q = urllib.parse.urlencode({"q_track": titre, "q_artist": artiste,
                                "apikey": cle})
    try:
        with urllib.request.urlopen(f"{base}?{q}", timeout=DELAI) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:                                        # noqa: BLE001
        # Jamais muet : sans ça, une clé invalide ressemblerait à « ce morceau
        # n'a pas de paroles ».
        logger.exception("paroles_api : appel Musixmatch impossible")
        return None
    corps = ((d.get("message") or {}).get("body") or {}).get("lyrics") or {}
    texte = corps.get("lyrics_body") or ""
    if not texte.strip():
        logger.info("paroles_api : Musixmatch ne rend rien pour %r / %r",
                    titre, artiste)
        return None
    tronque = bool(corps.get("lyrics_copyright")) and "..." in texte
    return {"texte": texte, "complet": not tronque, "source": "musixmatch",
            "copyright": corps.get("lyrics_copyright") or ""}


FOURNISSEURS = {"musixmatch": _musixmatch}


def paroles(titre: str, artiste: str = "", *, fournisseur: str = "",
            cle: str = "") -> dict | None:
    """Le texte d'un morceau par l'API configurée, ou None.

    Rend `{texte, complet, source, copyright}`. `complet` faux veut dire que
    l'API a tronqué — voir la docstring du module, c'est le cas du palier
    gratuit de Musixmatch.

    Sans clé, rend None sans rien tenter : l'appelant garde alors le texte du
    tab, et le journal le dit.
    """
    from harmonia.settings import SETTINGS
    cle = cle or getattr(SETTINGS, "paroles_cle", "") or ""
    if not cle:
        logger.info("paroles_api : pas de clé (HARMONIA_PAROLES_CLE), on garde "
                    "le texte du tab")
        return None
    nom = (fournisseur or getattr(SETTINGS, "paroles_source", "")
           or DEFAUT).lower()
    f = FOURNISSEURS.get(nom)
    if f is None:
        logger.warning("paroles_api : fournisseur %r inconnu (connus : %s)",
                       nom, ", ".join(sorted(FOURNISSEURS)))
        return None
    r = f(titre, artiste, cle)
    if r and not r["complet"]:
        logger.warning("paroles_api : %s a rendu un EXTRAIT tronqué pour "
                       "%r — le palier gratuit ne donne pas le texte entier",
                       nom, titre)
    return r


def en_lignes(texte: str) -> list[dict]:
    """Le texte brut d'une API → la MÊME forme que `lignes_de_paroles`.

    `[{ligne, texte, section, strophe}]`. Tout l'aval — alignement forcé,
    découpage par mesures, rendu — travaille déjà sur cette forme : l'adapter
    ici évite d'avoir deux chemins à maintenir selon la provenance du texte.

    Une ligne vide ouvre une strophe, comme dans un tab. Il n'y a PAS de nom de
    section (une API rend du texte, pas la structure d'un tabbeur) : la feuille
    perdra donc ses titres « Verse » / « Chorus » si le texte vient de là. À
    compenser plus tard par nos propres sections, qui elles sont mesurées.
    """
    out, saut = [], False
    for i, ligne in enumerate((texte or "").splitlines()):
        propre = ligne.strip()
        if not propre:
            saut = True
            continue
        # la mention de copyright que les API ajoutent en pied n'est pas chantée
        if propre.startswith("***") or propre.lower().startswith("..."):
            continue
        out.append({"ligne": i, "texte": propre, "section": "",
                    "strophe": saut})
        saut = False
    return out
