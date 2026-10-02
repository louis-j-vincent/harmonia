"""Les paroles que Louis écrit lui-même — la source la plus sûre qu'on ait.

POURQUOI C'EST LA PREMIÈRE SOURCE, et pas la dernière. Les trois autres
échouent chacune à leur manière :

  * le TAB d'Ultimate Guitar : il n'existe pas pour tous les morceaux (Bora
    Bora de Henri Salvador, Bein Green… — quatre de la bibliothèque), et quand
    il existe son texte est approximatif ;
  * une API SOUS LICENCE : son palier gratuit ne rend qu'environ 30 % du texte
    (voir `paroles_api`), et le texte entier demande un contrat ;
  * aspirer un site de paroles : refusé, c'est un outil de reproduction.

Son texte à lui n'a aucun de ces défauts : il est complet, il est juste, il
reste sur sa machine, et il n'appartient à personne d'autre de décider s'il
peut l'écrire.

CE QUE ÇA DÉBLOQUE : un morceau SANS AUCUN TAB devient lisible comme un tab.
On n'a jamais eu besoin du tab pour les accords — ceux-là sont les nôtres,
mesurés sur l'audio. Le tab ne servait qu'à fournir le TEXTE. Dès qu'il vient
d'ailleurs, la feuille se passe entièrement d'Ultimate Guitar.

OÙ ÇA VIT : `state/human/paroles/<stem>.txt`, à côté des sections et des
annotations faites main — donc SUIVI PAR GIT. C'est délibéré : un travail à la
main perdu dans un dossier ignoré, c'est déjà arrivé une fois (2026-08-12), et
cette séparation est le correctif.

LE FORMAT EST CELUI D'UN TEXTE ORDINAIRE : une ligne par ligne chantée, une
ligne vide entre deux strophes, et un `[Refrain]` ou `[Couplet 1]` seul sur sa
ligne pour nommer une section. Exactement ce qu'on lit déjà dans un tab, donc
rien de nouveau à apprendre — et on peut y coller le texte d'un tab puis le
corriger.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")


def chemin(stem: str):
    """Le fichier de paroles de ce morceau (qu'il existe ou non)."""
    from harmonia.settings import SETTINGS
    return SETTINGS.human_dir / "paroles" / f"{stem}.txt"


def existe(stem: str) -> bool:
    try:
        return chemin(stem).is_file()
    except Exception:                                        # noqa: BLE001
        return False


def lire(stem: str) -> str:
    """Le texte brut tel qu'il a été écrit, ou "" s'il n'y en a pas."""
    p = chemin(stem)
    try:
        return p.read_text(encoding="utf-8") if p.is_file() else ""
    except Exception:                                        # noqa: BLE001
        logger.exception("paroles_maison : %s illisible", p)
        return ""


def ecrire(stem: str, texte: str) -> int:
    """Enregistre le texte et rend le nombre de lignes chantées retenues."""
    p = chemin(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text((texte or "").strip() + "\n", encoding="utf-8")
    n = len(en_lignes(texte))
    logger.info("paroles_maison : %s écrit (%d ligne(s) chantée(s))", p, n)
    return n


def en_lignes(texte: str) -> list[dict]:
    """Le texte → la MÊME forme que `paroles_align.lignes_de_paroles`.

    `[{ligne, texte, section, strophe}]`. Tout l'aval — alignement forcé,
    découpage par mesures, rendu — travaille déjà sur cette forme ; l'adapter
    ici évite d'avoir deux chemins selon la provenance du texte.

    Une ligne vide ouvre une strophe, un `[...]` nomme la section qui suit.
    """
    out: list[dict] = []
    section, saut = "", False
    for i, ligne in enumerate((texte or "").splitlines()):
        propre = ligne.strip()
        if not propre:
            saut = True
            continue
        m = _SECTION.match(propre)
        if m:
            section = m.group(1).strip()
            saut = True
            continue
        out.append({"ligne": i, "texte": propre, "section": section,
                    "strophe": saut})
        saut = False
    return out
