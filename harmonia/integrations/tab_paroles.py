"""Le texte d'un tab, avec NOS accords posés dessus.

Louis, 2026-09-23 : « je veux que tu puisses me créer des tabs, donc m'extraire
les paroles des chansons, puis y coller les accords qu'on a, comme sur guitar
tabs ».

VÉRIFICATION DE PRÉMISSE, et c'est tout ce que ce module est pour l'instant.
Avant d'installer un modèle de transcription pour lire les paroles dans
l'audio, on regarde si la DISPOSITION se lit — des accords au-dessus des mots,
à la bonne syllabe. Le texte vient donc du tab qu'on récupère déjà
(`tab_fetcher`), pas de l'audio : c'est la moitié gratuite du problème, et
elle suffit à trancher si la vue vaut la peine.

CE QU'ON RÉUTILISE, ET QUI EXISTAIT DÉJÀ :

  * `tab_fetcher.fetch_tab_chords` rend `raw_content`, le document complet avec
    son balisage `[ch]…[/ch]` — les lignes de texte y sont, on les jetait ;
  * `tab_align.sequence_du_tab` rend la suite des accords AVEC leur `pos`, le
    décalage en caractères dans ce document — c'est ce `pos` qui fait le pont
    entre la grille du tab et notre ligne du temps ;
  * `tab_chart.poser` aligne cette suite sur NOTRE audio et rend, pour chaque
    accord du tab, la fenêtre de temps qu'il occupe chez nous.

Il ne manquait donc que la traduction « position dans le texte → notre accord
à cet instant », et c'est ce fichier.

L'HYPOTHÈSE PORTEUSE, testée dans `tests/test_tab_paroles.py` plutôt que
supposée : `tab_chart.poser` travaille sur la suite COMPRESSÉE
(`tab_align.compresser` fond les répétitions immédiates), donc l'indice `i` de
ses segments n'est PAS l'indice du jeton dans le document. On reconstruit la
correspondance en rejouant la même règle de fusion, et le test vérifie qu'elle
retombe exactement sur ce que `compresser` produit. C'est le motif d'erreur
nº 1 du CLAUDE.md : une erreur d'indice ici donnerait des accords plausibles et
faux, sans rien casser visiblement.

CE QUE ÇA NE RÉSOUT PAS :
  * le texte reste celui du tab — donc seulement les morceaux qui en ont un, et
    ses fautes de frappe et ses lignes manquantes avec ;
  * la COLONNE d'un accord reste celle que le tabbeur a tapée. On sait à quelle
    syllabe il l'a mis, pas à quelle syllabe il FAUDRAIT le mettre : notre
    alignement corrige le NOM de l'accord et son instant, pas sa place dans la
    ligne. Un tab dont les colonnes sont fausses restera faux ici ;
  * rien n'est extrait de l'audio. La lecture des paroles dans le son est
    l'étape d'après, et c'est elle qui portera le vrai risque (la
    reconnaissance vocale sur du chant est bien moins bonne que sur de la
    parole).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_CH = re.compile(r"\[ch\](.*?)\[/ch\]", re.S)
#: le balisage qu'on retire pour retrouver la ligne telle qu'elle s'affiche
_BALISE = re.compile(r"\[/?(?:ch|tab)\]")


def jetons_du_tab(brut: str) -> list[dict]:
    """Chaque accord ÉCRIT dans le document, avec l'endroit où il est écrit.

    `[{ligne, colonne, texte, ordre}]` — `colonne` est la colonne dans la
    ligne DÉBALISÉE (celle que l'œil voit), pas dans le document brut, sinon
    l'accord se décalerait de la longueur de `[ch]` à chaque fois.

    `ordre` est le rang du jeton dans la suite non compressée : c'est la clé
    qui rejoint `tab_align.sequence_du_tab`.
    """
    out, ordre = [], 0
    for i, ligne in enumerate((brut or "").splitlines()):
        propre, col, j = [], 0, 0
        for m in _CH.finditer(ligne):
            avant = _BALISE.sub("", ligne[j:m.start()])
            propre.append(avant)
            col += len(avant)
            txt = m.group(1).strip()
            out.append({"ligne": i, "colonne": col, "texte": txt,
                        "ordre": ordre})
            ordre += 1
            propre.append(txt)
            col += len(txt)
            j = m.end()
        # le reste de la ligne ne porte aucun accord, rien à enregistrer
    return out


def indices_compresses(seq: list[dict]) -> list[int]:
    """Pour chaque entrée de `seq` (non compressée), l'indice qu'elle prend
    dans `tab_align.compresser(seq)`.

    On rejoue la règle de fusion de `compresser` — répétition IMMÉDIATE du même
    (root, q5, basse) — au lieu de la deviner. Le test vérifie que le compte
    retombe sur `len(compresser(seq))`.
    """
    out, dernier, k = [], None, -1
    for a in seq:
        cle = (a.get("root"), a.get("q5"), a.get("bass"))
        if dernier is None or cle != dernier:
            k += 1
            dernier = cle
        out.append(k)
    return out


def feuille_calee(chart: dict, brut: str, align: list[dict]) -> list[dict]:
    """La feuille, NOS accords posés sur NOS instants — plus les colonnes du tab.

    LE DÉFAUT QUE ÇA CORRIGE (Louis, 2026-09-24 : « le guitar tabs que tu fais
    fait des accords pas bons, tu mets bien les mêmes accords que dans le
    karaoké ? »). Oui, les mêmes — mais `feuille` les lisait à la MAUVAISE
    HORLOGE. Elle allait chercher notre accord à l'instant où le TABBEUR avait
    posé le sien : si lui l'a mis un temps trop tôt, on lit notre accord à ce
    moment-là et on écrit le mauvais. Le karaoké, lui, prend nos accords à NOS
    instants — d'où « ceux-là ont l'air bons ».

    Ici on ne consulte plus le tab que pour son TEXTE. Chaque accord à nous est
    posé devant le mot qui sonne à son propre `t0`, d'après l'alignement forcé
    (`paroles_align.aligner`) que Louis a validé à l'oreille le 2026-09-24.

    CE QUE ÇA NE RÉSOUT PAS : un accord qui tombe dans un trou de l'alignement
    (passage instrumental, ligne que le tab n'a pas écrite) n'a aucun mot où
    s'accrocher — il est rattaché au mot suivant, et l'indiquer reste à faire.
    """
    from harmonia.integrations.paroles_align import lignes_muettes
    from harmonia.integrations.tab_chart import texte_du_chart

    notres = [c for c in ((chart.get("prompter") or {}).get("chords") or [])
              if float(c.get("t1") or 0) > float(c.get("t0") or 0)]
    if not align or not notres:
        return []

    # CE QUI N'A PAS ÉTÉ CHANTÉ SORT (Louis, 2026-09-24). L'audio le dit tout
    # seul : une ligne d'en-tête ou de crédits ne correspond à rien dans le
    # chant, donc son score s'effondre. Pas besoin d'une source extérieure.
    muettes = lignes_muettes(align)
    align = [m for m in align if m["ligne"] not in muettes]
    if not align:
        return []
    if muettes:
        logger.info("tab_paroles : %d ligne(s) écartée(s), rien n'y est "
                    "chanté (rangs %s)", len(muettes),
                    ", ".join(str(x) for x in sorted(muettes)[:8]))

    lignes = (brut or "").splitlines()
    par_ligne: dict[int, list] = {}
    for m in align:
        par_ligne.setdefault(m["ligne"], []).append(m)

    # chaque accord va au mot qui sonne à son t0 ; à défaut, au mot suivant
    mots = sorted(align, key=lambda m: m["t0"])
    poses: dict[int, list] = {}
    for c in notres:
        t = float(c["t0"])
        cible = next((m for m in mots if m["t0"] <= t < m["t1"]), None)
        if cible is None:
            cible = next((m for m in mots if m["t0"] >= t), None)
        if cible is None:
            continue
        poses.setdefault(cible["ligne"], []).append((cible, c))

    out = []
    for i in sorted(par_ligne):
        texte = _BALISE.sub("", lignes[i]).rstrip() if i < len(lignes) else ""
        accords = []
        for mot, c in poses.get(i, []):
            accords.append({"colonne": int(mot.get("debut_car") or 0),
                            "notre": texte_du_chart(c), "leur": "",
                            "t": round(float(c["t0"]), 2)})
        if not accords:
            continue
        out.append({"texte": texte, "accords": accords})
    return out


def feuille(chart: dict, pose: dict, brut: str) -> list[dict]:
    """Le tab réécrit : ses lignes de texte, NOS accords au-dessus.

    `[{texte, accords: [{colonne, notre, leur, leur_ecrit, t}]}]`, une entrée
    par ligne de parole qui porte au moins un accord. `leur` est l'accord du
    tab RAMENÉ DANS NOTRE TONALITÉ (voir le décalage plus bas), `leur_ecrit`
    est ce que le tab écrit vraiment. Les deux sont gardés à côté du nôtre pour
    que le désaccord se VOIE au lieu d'être écrasé — c'est la même loi que la
    marque de doute (`tab_chart.doutes`) : on ne dit pas qui a raison, on
    montre les deux.
    """
    from harmonia.integrations.tab_align import nom_q5, sequence_du_tab
    from harmonia.integrations.tab_chart import texte_du_chart

    seq = sequence_du_tab(brut)
    vers_compresse = indices_compresses(seq)
    jetons = jetons_du_tab(brut)
    if len(jetons) != len(seq):
        # Jamais muet : les deux lectures du même document doivent compter le
        # même nombre d'accords. Si elles divergent, `lire_accord` en a rejeté
        # (un « N.C. », un mot pris pour un accord) et les colonnes ne
        # correspondent plus — on préfère ne rien rendre qu'un décalage.
        logger.warning("tab_paroles : %d jeton(s) écrit(s) pour %d accord(s) "
                       "lu(s) — on ne pose rien", len(jetons), len(seq))
        return []

    # notre accord à un instant donné, depuis le chart brut
    notres = [c for c in ((chart.get("prompter") or {}).get("chords") or [])]

    def notre_a(t: float) -> str:
        for c in notres:
            if float(c.get("t0", 0)) <= t < float(c.get("t1", 0)):
                return texte_du_chart(c)
        return "—"

    # pour chaque indice compressé, l'instant où il commence chez nous
    debut = {}
    for s in pose.get("segments") or []:
        debut.setdefault(int(s["i"]), float(s["t0"]))

    lignes = (brut or "").splitlines()
    par_ligne: dict[int, list] = {}
    # LE TAB N'EST PAS FORCÉMENT DANS NOTRE TONALITÉ. `poser` rend le décalage
    # qu'il a fallu pour le poser sur notre audio ; comparer son accord ÉCRIT au
    # nôtre sans l'appliquer, c'est comparer deux tonalités — mesuré le
    # 2026-09-23, ça donnait 2 % d'accord sur This Love (décalage +3) là où la
    # vraie valeur est bien plus haute. Motif d'erreur nº 1 du CLAUDE.md : des
    # chiffres plausibles et faux. On rend donc SON accord dans NOTRE tonalité,
    # et on garde le texte d'origine à côté pour pouvoir relire le tab.
    dec = int(pose.get("decalage") or 0)
    for jet in jetons:
        k = vers_compresse[jet["ordre"]]
        t = debut.get(k)
        a = seq[jet["ordre"]]
        b = a.get("bass")
        leur = nom_q5((a["root"] + dec) % 12, a["q5"],
                      None if b is None else (b + dec) % 12)
        par_ligne.setdefault(jet["ligne"], []).append({
            "colonne": jet["colonne"], "leur": leur, "leur_ecrit": jet["texte"],
            "notre": notre_a(t) if t is not None else "—",
            "t": None if t is None else round(t, 2)})

    out = []
    for i, accords in sorted(par_ligne.items()):
        # la ligne de PAROLES est celle qui suit la ligne d'accords
        texte = ""
        for j in (i + 1, i + 2):
            if j < len(lignes):
                cand = _BALISE.sub("", lignes[j])
                if cand.strip() and "[ch]" not in lignes[j]:
                    texte = cand.rstrip()
                    break
        out.append({"texte": texte, "accords": accords})
    return out


# ── le rendu, une seule définition ──────────────────────────────
# Partagé par `tools.tab_paroles_page` (en ligne de commande) et par la
# route `/tab-paroles/<file>` de l'app : Louis lit depuis son iPhone, donc
# c'est SON serveur qui fabrique la page, à partir du tab qu'il a récupéré.
import html  # noqa: E402

CSS = """
:root{--pap:#f6f3ec;--enc:#1c1c1c;--nous:#8c2f2f;--eux:#9a9287;--lig:#e2dcd0}
@media(prefers-color-scheme:dark){:root{--pap:#16150f;--enc:#ece7dc;
  --nous:#e08a7a;--eux:#7d766b;--lig:#2b2820}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--pap);color:var(--enc);
  font:16px/1.5 -apple-system,system-ui,sans-serif;max-width:52rem;margin:0 auto}
h1{font:600 22px/1.25 Georgia,serif;margin:0 0 2px}
.sous{color:var(--eux);font-size:13px;margin:0 0 18px}
.bloc{margin:0 0 26px;border-top:1px solid var(--lig);padding-top:14px}
pre{margin:0;font:14px/1.15 ui-monospace,SFMono-Regular,Menlo,monospace;
  white-space:pre;overflow-x:auto;-webkit-overflow-scrolling:touch}
.acc{color:var(--nous);font-weight:700}
.eux{color:var(--eux);font-weight:400;font-size:11px}
.mots{color:var(--enc)}
a{color:var(--nous)}
"""


def ligne_accords(accords, largeur: int, montrer_eux: bool) -> str:
    """La ligne d'accords, NOS noms posés aux colonnes du tab.

    Deux accords qui se chevauchent après substitution (le nôtre est plus long
    que le sien) sont décalés d'un cran plutôt qu'écrasés : perdre un accord en
    silence serait pire que décaler la colonne d'un caractère.
    """
    buf = []
    col = 0
    for a in sorted(accords, key=lambda x: x["colonne"]):
        cible = max(a["colonne"], col + 1 if col else 0)
        buf.append(" " * max(0, cible - col))
        col = max(col, cible)
        nom = a["notre"]
        morceau = f'<span class="acc">{html.escape(nom)}</span>'
        if montrer_eux and a["leur"] != nom:
            morceau += f'<span class="eux">{html.escape(a["leur"])}</span>'
            col += len(nom) + len(a["leur"])
        else:
            col += len(nom)
        buf.append(morceau)
    return "".join(buf)


def page(titre: str, feuille: list[dict], tab: dict, decalage: int,
         montrer_eux: bool = True) -> str:
    blocs = []
    for l in feuille:
        blocs.append('<pre>' + ligne_accords(l["accords"], 0, montrer_eux)
                     + "\n" + f'<span class="mots">{html.escape(l["texte"])}</span>'
                     + "</pre>")
    note = (f'tab source : <a href="{html.escape(tab.get("url") or "")}">'
            f'{html.escape(tab.get("titre") or "?")} — '
            f'{html.escape(tab.get("artiste") or "?")}</a> '
            f'({tab.get("note")}★, {tab.get("votes")} votes)'
            f' · transposé de {decalage:+d} demi-ton(s) pour tomber sur notre audio')
    return (f'<!doctype html><html lang="fr"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(titre)}</title><style>{CSS}</style>'
            f'<h1>{html.escape(titre)}</h1><p class="sous">{note}<br>'
            f'En rouge, <b>nos</b> accords. En gris, ceux du tab quand ils '
            f'diffèrent. La question : tombent-ils sur la bonne syllabe ?</p>'
            f'<div class="bloc">{"".join(blocs)}</div></html>')


