"""Nos boucles d'accords contre celles d'un tab bien noté.

POURQUOI CETTE FORME DE COMPARAISON. Louis, 2026-09-24 : « les tabs sont faits
par des humains flemmards donc l'alignement des accords c'est de l'à-peu-près ».
Un tab ne sait donc PAS quand on change d'accord — mais il sait très bien
LESQUELS, et dans quel ordre. On ne compare donc aucun instant : seulement la
suite d'accords de chaque section, réduite à sa boucle.

C'est robuste à un tabbeur pressé par construction, et ça a fait ses preuves :
sur Stand By Me le tab confirme nos deux sections à 100 %, et sur This Love il
a trouvé un fa MAJEUR que nous écrivions mineur — confirmé à l'oreille par
Louis, puis corrigé dans `folding` (règle « l'unanimité directe bat la moyenne
indirecte »).

LA COMPARAISON EST CANONIQUE, jamais textuelle. On écrit « F#- » là où un tab
écrit « F#m » : comparer des chaînes ferait sortir 60 % de faux désaccords sur
un morceau où tout concorde. On compare des triplets (fondamentale, qualité,
basse), après transposition du tab dans notre tonalité.

CE QUE ÇA NE DIT PAS : où placer les changements, ni si notre DÉCOUPAGE en
sections est bon. Seulement quels accords vivent dans chaque section.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from harmonia.integrations.tab_align import lire_accord, nom_q5
from harmonia.integrations.tab_chart import lire_du_chart, poser
from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs

logger = logging.getLogger(__name__)

_SEC = re.compile(r"^\s*\[([^\]]+)\]\s*$")

#: en dessous de cette note, un tab n'est pas une référence d'accords
#: (mémoire projet, 2026-07-30 : « UG tab >4.7★ = trustworthy chords »)
NOTE_MINI = 4.7


def _canon(t, dec: int = 0):
    if t is None:
        return None
    r, q, b = t
    return ((r + dec) % 12, q, None if b is None else (b + dec) % 12)


def joli(c) -> str:
    return nom_q5(c[0], c[1], c[2]) if c else "?"


def _sans_repetitions(seq: list) -> list:
    out: list = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def boucle(seq: list) -> list:
    """La plus courte période qui explique la suite, sinon la suite entière."""
    n = len(seq)
    for p in range(1, n // 2 + 1):
        if n % p == 0 and all(seq[i] == seq[i % p] for i in range(n)):
            return seq[:p]
    return seq


def boucles_du_tab(brut: str, dec: int) -> dict:
    """{section du tab: boucle d'accords}, transposée dans notre tonalité."""
    out: dict = {}
    cour = None
    for ligne in (brut or "").splitlines():
        m = _SEC.match(re.sub(r"\[/?(ch|tab)\]", "", ligne))
        if m:
            cour = m.group(1).strip()
            out.setdefault(cour, [])
            continue
        if cour is None or "[ch]" not in ligne:
            continue
        for tok in re.findall(r"\[ch\](.*?)\[/ch\]", ligne):
            a = lire_accord(tok.strip())
            if a is None:
                continue
            out[cour].append(_canon((a["root"], a["q5"], a.get("bass")), dec))
    return {k: boucle(_sans_repetitions(v)) for k, v in out.items() if v}


def _normal(x: str) -> str:
    """Un nom réduit à ses lettres et chiffres, pour comparer des graphies."""
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def meme_morceau(notre_titre: str, notre_artiste: str, t) -> bool:
    """Ce tab parle-t-il bien de NOTRE morceau ?

    SANS CE GARDE-FOU, L'OUTIL MENT. Le 2026-09-30, faute de l'avoir écrit, la
    recherche par TITRE SEUL a rendu :

        « Bora Bora » (Henri Salvador)  ->  « Electric Love » de BØRNS
        « Bein Green »                  ->  « Tired Of Being Alone » d'Al Green

    et les « désaccords d'accords » qui en sortaient comparaient donc deux
    chansons sans rapport. Louis l'a vu tout de suite ; l'outil, non.

    On exige que l'artiste concorde quand on le connaît, et à défaut que le
    titre concorde vraiment (pas une sous-chaîne lointaine).
    """
    tt, ta = _normal(t.song_name), _normal(t.artist_name)
    nt, na = _normal(notre_titre), _normal(notre_artiste)
    if na:
        if not (na in ta or ta in na):
            return False
    if not nt:
        return False
    return nt in tt or tt in nt


def comparer(chart: dict, requete: str, artiste: str = "") -> dict | None:
    """Nos sections contre celles du meilleur tab trouvé pour `requete`.

    Rend `{note, tab, sections: [{label, reps, nous, tab, manque, enplus}]}`,
    ou None si aucun tab exploitable. `manque` = ce que le tab a et pas nous ;
    `enplus` = l'inverse. Les deux vides = les boucles concordent.
    """
    # l'ARTISTE fait partie de la requête : sans lui, UG rend une autre chanson
    res = search_tabs(requete, artiste, tab_types=("Chords",), max_results=8)
    res = [x for x in res if meme_morceau(requete, artiste, x)]
    if not res:
        logger.info("tab_boucles : aucun tab qui soit bien %r de %r",
                    requete, artiste or "?")
        return None
    pose = poser(chart, requete)
    if pose is None:
        return None
    t = res[0]
    brut = fetch_tab_chords(t).raw_content
    leurs = boucles_du_tab(brut, int(pose.get("decalage") or 0))
    if not leurs:
        return None

    out = []
    for sec in chart.get("sections") or []:
        nos = boucle(_sans_repetitions(
            [_canon(lire_du_chart(c))
             for b in (sec.get("bars") or []) for c in b
             if lire_du_chart(c)]))
        if not nos:
            continue
        best, sc, bw = None, -1.0, []
        for k, w in leurs.items():
            s = len(set(nos) & set(w)) / max(1, len(set(nos) | set(w)))
            if s > sc:
                best, sc, bw = k, s, w
        out.append({"label": sec.get("label"), "reps": sec.get("reps"),
                    "tab_section": best, "score": round(sc, 3),
                    "nous": [joli(x) for x in nos],
                    "tab": [joli(x) for x in bw],
                    "manque": [joli(x) for x in bw if x not in nos],
                    "enplus": [joli(x) for x in nos if x not in bw]})
    return {"note": float(getattr(t, "rating", 0) or 0),
            "votes": int(getattr(t, "votes", 0) or 0),
            "tab": {"titre": t.song_name, "artiste": t.artist_name,
                    "url": t.tab_url},
            "decalage": int(pose.get("decalage") or 0),
            "sections": out}


# ── CE QUE CETTE COMPARAISON VAUT, ET CE QU'ELLE NE VAUT PAS ────────────────
#
# Balayage 2026-09-28 sur les 22 morceaux à sections faites main, tous avec un
# tab entre 4,73★ et 4,91★. Trois filtres successifs, et seul le dernier tient :
#
#   1. « sections dont la boucle concorde » — INUTILISABLE. Mon appariement
#      colle notre section à la SIENNE par ressemblance ; quand le tabbeur
#      regroupe couplet et refrain sous un seul en-tête, sa boucle contient des
#      accords des deux. Fallin' montrait les mêmes trois accords « manquants »
#      dans TOUTES ses sections : signature d'un défaut d'appariement, pas de
#      22 erreurs.
#
#   2. « désaccords de tierce par section » — 25 cas, encore trompeur. Sur
#      Goodbye Yellow Brick Road on écrit DÉJÀ Bb, Bb- et Bb-7 dans des mesures
#      différentes ; réduire la section à une boucle n'en retient qu'un, d'où un
#      faux désaccord. On ne peut pas comparer mesure à mesure non plus : les
#      positions d'un tab sont « de l'à-peu-près » (Louis, 2026-09-24).
#
#   3. « tierce que le tab emploie et qu'on n'écrit NULLE PART du morceau,
#      alors qu'on écrit systématiquement l'autre sur cette fondamentale » —
#      3 cas seulement, et c'est LE bon filtre. Il ne dépend d'aucune position
#      ni d'aucun appariement de sections.
#
# VALIDATION : This Love ressort de ce filtre, et c'est le seul cas que Louis
# ait confirmé à l'oreille (fa majeur, pas mineur). Le filtre retient donc le
# cas connu-vrai et écarte 22 des 25 comme non comparables.
#
# LES DEUX AUTRES CANDIDATS, à arbitrer à l'oreille :
#   Bora Bora   4,89★   la : nous toujours MAJEUR, le tab dit mineur
#   Bein Green  4,77★   fa : nous toujours MAJEUR, le tab dit mineur
