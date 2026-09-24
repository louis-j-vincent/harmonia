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

#: DISTANCE MAXIMALE ENTRE UN ACCORD ET LE DÉBUT DE SON MOT, en secondes.
#: Deux pannes, une seule règle. (1) Un accord qui tombe dans un passage
#: instrumental s'entassait sur le premier mot d'après. (2) L'alignement forcé
#: produit parfois un mot ANORMALEMENT LONG : mesuré sur Stand By Me le
#: 2026-09-24, un mot de 26,4 s contre une durée médiane de 0,22 s — il
#: avalait à lui seul 18 accords, 360 px de large, d'où le débordement
#: horizontal sur téléphone. Exiger que l'accord soit proche du DÉBUT du mot,
#: et pas seulement dans sa fenêtre, traite les deux.
DELAI_MOT = 1.5

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

    # LES ACCORDS UNIFORMISÉS D'ABORD (Louis, 2026-09-24). Le repliement a déjà
    # fait dire la même chose aux occurrences d'une même section ; on lit donc
    # les accords ÉCRITS. Sans sections repliées, `accords_replies` ne rend
    # rien et on retombe sur le décodage à plat — repli DIT, pas silencieux.
    replies = accords_replies(chart)
    if replies:
        notres = [{**r["accord"], "t0": r["t0"], "t1": r["t1"]} for r in replies]
        logger.info("tab_paroles : %d accord(s) lus sur les sections repliées",
                    len(notres))
    else:
        notres = [c for c in ((chart.get("prompter") or {}).get("chords") or [])
                  if float(c.get("t1") or 0) > float(c.get("t0") or 0)]
        logger.info("tab_paroles : pas de section repliée, on lit le décodage "
                    "à plat (%d accord(s))", len(notres))
    notres = [c for c in notres
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

    # CHAQUE ACCORD VA AU MOT QUI SONNE À SON t0 — et à personne si aucun mot
    # ne sonne près de là. Le repli « à défaut, le mot suivant » n'avait pas de
    # garde-fou : tous les accords d'un passage instrumental s'entassaient sur
    # le premier mot d'après. Mesuré le 2026-09-24 : UN SEUL mot portait 18
    # accords sur Stand By Me, 360 px de large, d'où le débordement horizontal
    # sur téléphone. Un accord qui n'a pas de parole sous lui n'est pas écrit
    # ici — il reste dans le chart, c'est cette feuille-ci qui ne le montre pas.
    # UNE MESURE DE CE MORCEAU-CI, lue sur ses battues plutôt que supposée :
    # 2,0 s sur Stand By Me, 2,5 s sur This Love. C'est l'unité du seuil de
    # rappel ci-dessous.
    import statistics as st
    _t = sorted(float(x) for x in (chart.get("beatTimes") or []))
    _pas = (st.median([b - a for a, b in zip(_t, _t[1:])])
            if len(_t) > 3 else 0.5)
    une_mesure = _pas * int(chart.get("bpb") or 4)

    mots = sorted(align, key=lambda m: m["t0"])
    poses: dict[int, list] = {}
    perdus = 0
    # le premier mot de chaque ligne, pour la règle de bascule ci-dessous
    premiers = set()
    for m in mots:
        if m["ligne"] not in {x["ligne"] for x in mots if x["t0"] < m["t0"]}:
            premiers.add(m["rang"])

    for c in notres:
        t = float(c["t0"])
        cible = next((m for m in mots
                      if m["t0"] <= t < m["t1"] and t - m["t0"] <= DELAI_MOT),
                     None)
        if cible is None:
            suivant = next((m for m in mots if m["t0"] >= t), None)
            if suivant is not None and suivant["t0"] - t <= DELAI_MOT:
                cible = suivant
                # UN ACCORD QUI TOMBE JUSTE AVANT UN DÉBUT DE LIGNE S'ÉCRIT
                # SUR LA LIGNE D'AVANT (Louis, 2026-09-24) — c'est ainsi qu'un
                # tab est vraiment écrit : le changement de fin de phrase se
                # note au-dessus de la fin de cette phrase. Mesuré, l'écart au
                # dernier mot d'avant :
                #   Stand By Me  0,4…1,4 s puis 2,7 et 5,3  (mesure 2,0 s)
                #   This Love    0,1…1,6 s                  (mesure 2,5 s)
                # Une mesure sépare les cas normaux des deux vrais trous.
                if suivant["rang"] in premiers:
                    avant = [m for m in mots if m["t1"] <= t]
                    if avant and t - avant[-1]["t1"] <= une_mesure:
                        cible = avant[-1]
        if cible is None:
            perdus += 1
            continue
        poses.setdefault(cible["ligne"], []).append((cible, c))
    if perdus:
        logger.info("tab_paroles : %d accord(s) sans parole dessous, non "
                    "écrits sur la feuille", perdus)

    # UN ACCORD ACCROCHÉ À SON MOT, pas à une colonne (Louis, 2026-09-24 :
    # « sur le html version téléphone les accords ne sont pas alignés au-dessus
    # de la bonne parole »). L'ancien rendu comptait des CARACTÈRES dans un
    # <pre> : n'importe quelle différence typographique entre la ligne
    # d'accords et la ligne de mots — graisse, repli de police, réglage de
    # taille de texte de Safari iOS — décale tout ce qui suit. Mesuré : 0 px
    # d'écart sous Chromium à 390 comme à 1280 px, et pourtant faux sur son
    # iPhone. Plutôt que de chercher LA cause, on retire la dépendance : le
    # mot porte son accord, le navigateur les empile, et aucune police ne peut
    # plus les séparer. Ça règle aussi le défilement horizontal, qui n'a rien
    # à faire sur un téléphone.
    par_mot: dict = {}
    for i, liste in poses.items():
        for mot, c in liste:
            par_mot.setdefault(mot["rang"], []).append(c)

    def sonnant_a(t: float):
        """L'accord qui SONNE à cet instant — pas celui qui y commence."""
        for c in notres:
            if float(c["t0"]) <= t < float(c["t1"]):
                return c
        return None

    out = []
    for i in sorted(par_ligne):
        mots = []
        for m in sorted(par_ligne[i], key=lambda x: x.get("debut_car") or 0):
            accs = par_mot.get(m["rang"]) or []
            mots.append({"mot": m["mot"],
                         "accords": [texte_du_chart(c) for c in accs],
                         # « les accords qui ne tombent pas sur les temps
                         # mets-les en petit » : une syncope réelle ou une
                         # frontière mal placée, dans les deux cas plus discret
                         "petits": [petit_si(c) for c in accs],
                         "t": round(float(m["t0"]), 2),
                         "douteux": bool(m.get("score") is not None
                                         and m["score"] < 0.30)})
        if not any(w["accords"] for w in mots):
            continue
        # L'ACCORD EN DÉBUT DE LIGNE, ET QUAND IL NE SERT À RIEN.
        #
        # Le rappel existe pour une seule raison : une ligne qui commence AU
        # MILIEU d'un accord ne disait pas lequel, il fallait remonter la ligne
        # d'avant. Mais si un VRAI changement arrive juste derrière, le rappel
        # devient du bruit — on lit deux accords là où il n'y a qu'un
        # changement (Louis, 2026-09-24 : « n'écris pas la répétition de
        # l'accord précédent sur une nouvelle ligne s'il y a un autre accord
        # juste après »).
        #
        # LE SEUIL EST UNE MESURE, et il est mesuré, pas choisi. Délais entre
        # le début de ligne et le premier vrai accord, sur les deux morceaux :
        #
        #     Stand By Me   1,4 · 1,4 · 1,4 · 1,4 · 1,5   puis 3,1   (mesure 2,0 s)
        #     This Love     0,2 … 1,7 (neuf valeurs)      puis 4,3   (mesure 2,5 s)
        #
        # Les deux populations sautent au même endroit, et une mesure passe
        # entre elles sur les deux morceaux. Ne survivent que les rappels où
        # l'on resterait vraiment dans le flou.
        if mots and not mots[0]["accords"]:
            suivant = next((w for w in mots[1:] if w["accords"]), None)
            trop_tot = (suivant is not None
                        and suivant["t"] - mots[0]["t"] <= une_mesure)
            c = None if trop_tot else sonnant_a(mots[0]["t"])
            if c is not None:
                mots[0]["accords"] = [texte_du_chart(c)]
                mots[0]["petits"] = [False]      # un rappel n'est pas un changement
                mots[0]["rappel"] = True
        out.append({"mots": mots})

    # UN ACCORD RÉPÉTÉ DEUX FOIS DE SUITE NE S'ÉCRIT QU'UNE FOIS (Louis,
    # 2026-09-24). C'est la convention de tous les tabs : on note les
    # CHANGEMENTS, pas l'accord à chaque fois qu'il pourrait être rejoué. La
    # suppression se fait en ORDRE DE LECTURE et enjambe les sauts de ligne —
    # un accord identique à celui qui finissait la ligne d'avant est une
    # répétition, même s'il ouvre la ligne suivante. C'est d'ailleurs ce qui
    # rend le rappel de début de ligne cohérent : il ne survit que lorsqu'il
    # dit quelque chose de neuf.
    precedent = None
    n_doublons = 0
    for l in out:
        for w in l["mots"]:
            gardes, petits = [], []
            pet = w.get("petits") or [False] * len(w["accords"])
            for k, nom in enumerate(w["accords"]):
                if nom == precedent:
                    n_doublons += 1
                    continue
                gardes.append(nom)
                petits.append(pet[k] if k < len(pet) else False)
                precedent = nom
            w["accords"], w["petits"] = gardes, petits
            if not gardes:
                w.pop("rappel", None)
    if n_doublons:
        logger.info("tab_paroles : %d répétition(s) d'accord non réécrite(s)",
                    n_doublons)
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




# ── le rendu par MOTS : l'accord est empilé sur son mot ─────────────────────
# Remplace le comptage de caractères dans un <pre>. Le navigateur empile, donc
# aucune police, graisse ni réglage de taille de texte ne peut plus décaler un
# accord de son mot. Et la ligne se replie au lieu de défiler horizontalement.
CSS_MOTS = """
:root{--pap:#f6f3ec;--enc:#1c1c1c;--nous:#8c2f2f;--dout:#b5651d;
  --eux:#9a9287;--lig:#e2dcd0}
@media(prefers-color-scheme:dark){:root{--pap:#16150f;--enc:#ece7dc;
  --nous:#e08a7a;--dout:#d89a5b;--eux:#7d766b;--lig:#2b2820}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0 auto;padding:16px;max-width:46rem;background:var(--pap);
  color:var(--enc);font:16px/1.5 -apple-system,system-ui,sans-serif}
h1{font:600 21px/1.25 Georgia,serif;margin:0 0 2px}
.sous{color:var(--eux);font-size:13px;margin:0 0 20px}
.ligne{margin:0 0 16px}
.u{display:inline-block;position:relative;padding-top:1.45em;
  vertical-align:bottom}
.u i{position:absolute;left:0;top:0;font:700 13px/1.45 -apple-system,
  system-ui,sans-serif;color:var(--nous);font-style:normal;white-space:nowrap}
.u b{font:400 17px/1.35 Georgia,serif;white-space:nowrap}
.u i.rap{opacity:.45;font-weight:500}
.u i em{font-style:normal}
.u i em+em{margin-left:.35em}
.u i em.pt{font-size:.78em;opacity:.75;font-weight:500}
a{color:var(--nous)}
"""


def page_mots(titre: str, feuille: list[dict], tab: dict,
              sous_extra: str = "") -> str:
    """La feuille, un mot par élément, son accord empilé dessus.

    `feuille` : ce que rend `feuille_calee` — `[{mots: [{mot, accords, ...}]}]`.
    """
    blocs = []
    for l in feuille:
        u = []
        for w in l["mots"]:
            pet = w.get("petits") or [False] * len(w["accords"])
            acc = "".join(
                f'<em class="{"pt" if (pet[k] if k < len(pet) else False) else ""}">'
                f'{html.escape(a)}</em>'
                for k, a in enumerate(w["accords"]))
            u.append(f'<span class="u{" dout" if w["douteux"] else ""}">'
                     f'<i class="{"rap" if w.get("rappel") else ""}">{acc}</i>'
                     f'<b>{html.escape(w["mot"])}</b></span>')
        # une VRAIE espace entre les unités : la ligne doit se lire comme une
        # phrase, pas comme une liste de mots espacés
        blocs.append(f'<div class="ligne">{" ".join(u)}</div>')
    note = (f'texte : <a href="{html.escape(tab.get("url") or "")}">'
            f'{html.escape(tab.get("titre") or "?")}</a>' if tab else "")
    return (f'<!doctype html><html lang="fr"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(titre)}</title><style>{CSS_MOTS}</style>'
            f'<h1>{html.escape(titre)}</h1>'
            f'<p class="sous">{sous_extra}{" · " if sous_extra and note else ""}'
            f'{note}<br>Nos accords, posés sur le mot qui sonne à leur instant. '
            f'Un mot en gris est un mot dont le calage est douteux.</p>'
            f'{"".join(blocs)}</html>')


# ── les accords UNIFORMISÉS par le repliement ───────────────────────────────
def accords_replies(chart: dict) -> list[dict]:
    """Les accords ÉCRITS du chart, étalés sur toutes les occurrences.

    Louis, 2026-09-24 : « sers-toi du repliement des sous-sections pour
    détecter les parties similaires et uniformiser les accords ».

    Le repliement a DÉJÀ fait ce travail et on ne le refait pas : une section
    jouée six fois porte huit barres écrites UNE SEULE FOIS (`section["bars"]`)
    plus une fenêtre de temps par barre et par occurrence
    (`section["barSpans"][barre][occurrence]`). Lire l'accord là, c'est lire
    la version uniformisée ; lire `prompter.chords`, c'est lire le décodage à
    plat, qui peut dire autre chose à la troisième reprise qu'à la première.

    Rend `[{t0, t1, accord}]` trié — l'accord écrit, projeté sur chaque
    occurrence. Une barre qui porte plusieurs accords voit sa fenêtre
    découpée dans les MÊMES proportions que dans la barre écrite.

    CE QUE ÇA NE RÉSOUT PAS : si `SETTINGS.sections == "aucune"` et que Louis
    n'a rien annoté, le chart n'a qu'une section « A » d'une occurrence — il
    n'y a alors rien à uniformiser, et cette fonction rend l'équivalent du
    décodage à plat. L'uniformisation vient de SES sections, pas d'ici.
    """
    import statistics as st
    out = []
    for sec in chart.get("sections") or []:
        bars = sec.get("bars") or []
        spans = sec.get("barSpans") or []
        # LA DURÉE TYPIQUE DE CETTE SECTION-CI. Comparer à une durée globale
        # écraserait une section entière dont tous les accords sont courts
        # (This Love, section B : 2 temps partout, et tous structurels).
        durees = [float(c.get("t1") or 0) - float(c.get("t0") or 0)
                  for b in bars for c in b
                  if float(c.get("t1") or 0) > float(c.get("t0") or 0)]
        typique = st.median(durees) if durees else 0.0
        for k, barre in enumerate(bars):
            if k >= len(spans) or not barre:
                continue
            # positions relatives des accords DANS la barre écrite
            t0b = min(float(c.get("t0") or 0) for c in barre)
            t1b = max(float(c.get("t1") or 0) for c in barre)
            duree = max(1e-6, t1b - t0b)
            for fen in spans[k] or []:
                if not fen or len(fen) < 2:
                    continue
                a, b = float(fen[0]), float(fen[1])
                L = max(1e-6, b - a)
                for c in barre:
                    d0 = (float(c.get("t0") or 0) - t0b) / duree
                    d1 = (float(c.get("t1") or 0) - t0b) / duree
                    dc = float(c.get("t1") or 0) - float(c.get("t0") or 0)
                    petit = bool(typique and dc < PART_TRANSITION * typique)
                    out.append({"t0": a + d0 * L, "t1": a + min(1.0, d1) * L,
                                "accord": {**c, "petit": petit}})
    return sorted(out, key=lambda x: x["t0"])


#: un accord plus court que cette fraction de la durée TYPIQUE DE SA SECTION
#: est un accord de passage — voir `petit_si`
PART_TRANSITION = 0.60


def petit_si(accord: dict) -> bool:
    """Cet accord doit-il s'écrire en petit ?

    LA DEMANDE ET POURQUOI ELLE A CHANGÉ. Louis, 2026-09-24 : « les accords qui
    ne tombent pas sur les temps mets-les en petit ». Mesuré avant d'écrire :
    la condition n'arrive JAMAIS.

        Stand By Me   décodage à plat : 1 accord sur 38 · replié : 0 sur 127
        This Love     décodage à plat : 0 sur 119    · replié : 0 sur 113
        écart au temps le plus proche, accords repliés : 21 ms au pire

    Le mécanisme est structurel, pas réglable : `SETTINGS.quarter_bar` fait
    poser les changements d'accord SUR la grille des temps (tout temps peut en
    porter un, rien entre deux temps). Un accord hors des temps ne peut pas
    être produit. Et lire les accords repliés l'exclut doublement, puisque
    leurs fenêtres viennent de la grille de mesures.

    CE QU'ON ÉCRIT EN PETIT À LA PLACE. Louis, 2026-09-24, en donnant le vrai
    critère : « sur la première partie il y a clairement 4 accords, et les
    autres sont des accords de transition ». C'est la DURÉE, et elle sort toute
    seule des accords repliés de This Love :

        section A   G/B · C- · F-7   4 temps chacun   ← la charpente
                    Dø7 · Ab-7       2 temps          ← les transitions
        section B   C- · F- · Bb · Eb  2 temps chacun ← tous structurels !

    D'OÙ LA RÈGLE RELATIVE, et c'est tout l'enjeu : en section B, TOUS les
    accords font 2 temps. Un seuil absolu à « moins de 4 temps » y écraserait
    la section entière. On compare donc chaque accord à la durée TYPIQUE DE SA
    PROPRE SECTION (la médiane), et on écrit en petit ce qui est nettement plus
    court. Sur les trois sections mesurées, ça marque exactement les accords
    que Louis appelle des transitions, et rien d'autre.

    `accord["petit"]` est posé par `accords_replies`, qui seul connaît la
    section d'appartenance ; cette fonction ne fait que le relire.

    CE QUE ÇA NE DIT PAS : qu'il est FAUX, ni qu'il ne faut pas le jouer. Un
    accord de passage est un vrai accord — il est écrit plus discrètement
    parce qu'il ne porte pas la section.
    """
    return bool(accord.get("petit"))


def hors_des_temps(chart: dict, marge: float = 0.25) -> callable:
    """Rend `f(t) -> bool` : cet instant tombe-t-il À CÔTÉ d'un temps ?

    GARDÉE MAIS PLUS UTILISÉE PAR LE RENDU (2026-09-24) : mesurée sur deux
    morceaux, elle rend faux partout — voir `petit_si` pour le pourquoi et
    pour ce qui la remplace. On la garde parce qu'elle redeviendra vraie le
    jour où le décodage saura placer un accord entre deux temps.

    `marge` est la fraction d'un temps au-delà de laquelle on considère que ça
    tombe à côté. Le temps de référence est la MÉDIANE des intervalles du
    morceau, pas un tempo supposé.
    """
    temps = sorted(float(t) for t in (chart.get("beatTimes") or []))
    if len(temps) < 4:
        return lambda t: False
    import statistics as st
    pas = st.median([b - a for a, b in zip(temps, temps[1:])]) or 0.5
    seuil = marge * pas

    def f(t: float) -> bool:
        import bisect
        i = bisect.bisect_left(temps, t)
        proches = [temps[j] for j in (i - 1, i) if 0 <= j < len(temps)]
        return bool(proches) and min(abs(t - x) for x in proches) > seuil

    return f


# ── la feuille bâtie sur LA GRILLE DE MESURES ──────────────────────────────
#: combien de mesures par ligne écrite
MESURES_PAR_LIGNE = 2


def feuille_barres(chart: dict, align: list[dict], *,
                   structure: list[dict] | None = None,
                   par_ligne: int = 0) -> list[dict]:
    """La feuille : le PHRASÉ du tab, calé sur NOS mesures.

    Louis, 2026-09-24 : « il faut imaginer que chaque ligne correspond à une ou
    deux barres, et donc on y met les paroles associées... la ligne devrait
    commencer par le C- ».

    DEUX ESSAIS, ET LE SECOND EST LE BON. J'ai d'abord découpé par tranches
    FIXES de deux mesures, en jetant les lignes du tab. Mesuré ensuite, ce que
    couvre réellement une ligne de tab :

        This Love     médiane 2 mesures   (2 ×16, 4 ×5, 3 ×4, 1 ×4, 7 ×2)
        Stand By Me   médiane 4 mesures   (4 ×8, 3 ×4, 5 ×2, 2 ×2, …)

    Une tranche fixe de deux mesures coupe donc chaque phrase de Stand By Me en
    deux, et la longueur varie AUSSI à l'intérieur d'un même morceau. Le phrasé
    n'est pas une constante : il est déjà écrit dans le tab, qui est le seul
    endroit où quelqu'un l'a noté à la main.

    ON GARDE DONC SES LIGNES POUR GROUPER, et on ne prend de notre grille que
    ce qu'elle sait vraiment : OÙ SONT LES BARRES. Chaque ligne est rattachée à
    la mesure où tombe son premier mot, et l'accord qui sonne sur cette barre
    ouvre la ligne — toujours écrit, même s'il se répète. C'est ce qui corrige
    le refrain de This Love : son premier mot tombe 0,60 s APRÈS la barre 17,
    donc le C- de cette mesure n'apparaissait nulle part ; il ouvre maintenant
    la ligne par construction.

    `par_ligne` est ignoré, gardé pour ne pas casser les appelants : le
    regroupement ne se décide plus en nombre de mesures.

    Rend `[{mesure, t0, tete, tete_petit, mots}]`.

    CE QUE ÇA NE RÉSOUT PAS : le phrasé reste celui du tabbeur, avec ses
    coupures maladroites là où il en a fait. On corrige l'ACCORD de tête et le
    numéro de mesure, pas son découpage des phrases.
    """
    import bisect

    from harmonia.integrations.tab_chart import texte_du_chart

    grid = [float(x) for x in (chart.get("barGrid") or [])]
    if len(grid) < 3 or not align:
        return []
    muettes = lignes_muettes_de(align)
    par: dict = {}
    for m in align:
        if m["ligne"] in muettes:
            continue
        par.setdefault(m["ligne"], []).append(m)
    if not par:
        return []
    rep = accords_replies(chart)
    if not rep:
        return []
    import statistics as st
    _t = sorted(float(x) for x in (chart.get("beatTimes") or []))
    _pas = (st.median([b - a for a, b in zip(_t, _t[1:])])
            if len(_t) > 3 else 0.5)
    une_mesure = _pas * int(chart.get("bpb") or 4)

    def sonnant(t: float):
        for r in rep:
            if r["t0"] <= t < r["t1"]:
                return r["accord"]
        return None

    # LA STRUCTURE HUMAINE, telle que le tabbeur l'a écrite : à quelle section
    # appartient chaque ligne, et laquelle ouvre une strophe. Sans elle la
    # feuille est une suite de lignes indifférenciées — Louis, 2026-09-24 :
    # « ça ressemble à rien, les retours à la ligne devraient correspondre à
    # des strophes ».
    infos = {x["ligne"]: x for x in (structure or [])}

    # LA MESURE DE CHAQUE LIGNE, calculée d'AVANCE : il faut connaître celle de
    # la ligne SUIVANTE pour savoir où la ligne courante s'arrête.
    ordre = sorted(par)
    barre = {}
    for lg in ordre:
        a = min(x["t0"] for x in par[lg])
        b = max(0, min(bisect.bisect_right(grid, a) - 1, len(grid) - 2))
        # L'ANACROUSE APPARTIENT À LA MESURE QUI SUIT. Un mot qui démarre juste
        # AVANT une barre est une levée : le rattacher à la mesure d'avant
        # raccourcit sa ligne et gonfle la suivante. Mesuré sur le refrain de
        # This Love : deux lignes ancrées à une mesure d'écart, l'une recevant
        # deux accords et l'autre six, alors que le refrain est une boucle
        # régulière de quatre. On avance donc d'une mesure quand le mot est à
        # moins d'un temps de la barre suivante.
        if b + 2 < len(grid) and (grid[b + 1] - a) < _pas:
            b += 1
        barre[lg] = b

    out, precedent = [], None
    for idx, lg in enumerate(ordre):
        ms = sorted(par[lg], key=lambda x: x["t0"])
        b = barre[lg]
        # LA TÊTE : l'accord qui sonne SUR LA BARRE de cette ligne, pas à son
        # premier mot — c'est toute la différence, le mot arrivant souvent
        # après le premier temps.
        tete_c = sonnant(grid[b])
        tete = texte_du_chart(tete_c) if tete_c is not None else ""
        precedent = tete or precedent

        # UNE LIGNE S'ARRÊTE OÙ COMMENCE LA SUIVANTE. Je bornais par « dernier
        # mot + 1,5 s », ce qui débordait sur la ligne d'après : les mêmes
        # accords étaient écrits DEUX FOIS, et sur le refrain de This Love —
        # une boucle de quatre accords — ça donnait une bouillie. Mesuré :
        # la ligne de la mesure 17 courait jusqu'à 46,92 s alors que celle de
        # la mesure 18 commençait à 44,00 s. Chaque accord appartient
        # désormais à une seule ligne.
        b_suiv = barre[ordre[idx + 1]] if idx + 1 < len(ordre) else len(grid) - 1
        fin = grid[max(b_suiv, b + 1)] if b_suiv > b else ms[-1]["t1"]

        # CHAQUE ACCORD VA À UN SEUL MOT, ET C'EST LE SENS QUI COMPTE.
        # Je faisais l'inverse : chaque mot ramassait TOUS les accords de sa
        # fenêtre (sa durée plus 1,5 s). À ce tempo — un accord toutes les deux
        # battues sur le refrain de This Love — un mot en ramassait deux ou
        # trois, et ils s'affichaient collés au-dessus de LUI au lieu d'être
        # répartis sur les mots qu'ils accompagnent. C'est ce que Louis voyait
        # sur son iPhone : « Bb C- F- » entassés en début de ligne.
        #
        # On parcourt donc les ACCORDS, et chacun choisit son mot : celui qui
        # sonne à son instant, sinon le plus proche dans la ligne.
        items = [{"mot": m["mot"], "accords": [], "petits": [],
                  "t": round(float(m["t0"]), 2),
                  "douteux": bool(m.get("score") is not None
                                  and m["score"] < 0.30)}
                 for m in ms]
        for r in rep:
            t = r["t0"]
            if not (grid[b] < t < fin):
                continue                         # hors de cette ligne
            nom = texte_du_chart(r["accord"])
            if nom == precedent:
                continue                         # pas deux fois de suite
            k = next((i for i, m in enumerate(ms) if m["t0"] <= t < m["t1"]),
                     None)
            if k is None:
                # Pas de mot à cet instant : on n'accepte le plus proche que
                # s'il est VRAIMENT proche. Sans ce garde-fou, tous les accords
                # d'un trou instrumental se rabattaient sur le même mot — on
                # voyait cinq accords collés au-dessus d'une seule syllabe.
                k = min(range(len(ms)), key=lambda i: abs(ms[i]["t0"] - t))
                if abs(ms[k]["t0"] - t) > une_mesure / 2:
                    continue
            if len(items[k]["accords"]) >= 2:
                # Un mot ne porte pas trois accords : au-delà, c'est que
                # l'alignement ou la grille dit n'importe quoi ici.
                continue
            items[k]["accords"].append(nom)
            items[k]["petits"].append(bool(r["accord"].get("petit")))
            precedent = nom
        info = infos.get(lg) or {}
        out.append({"mesure": b + 1, "t0": round(grid[b], 2), "tete": tete,
                    "tete_petit": bool(tete_c is not None
                                       and tete_c.get("petit")),
                    "section": info.get("section") or "",
                    "strophe": bool(info.get("strophe")),
                    "mots": items})
    return out


def lignes_muettes_de(align: list[dict]) -> set:
    """Passe-plat vers `paroles_align.lignes_muettes` — une seule définition."""
    from harmonia.integrations.paroles_align import lignes_muettes
    return lignes_muettes(align)


CSS_BARRES = CSS_MOTS + """
.bar{display:flex;gap:8px;align-items:flex-start;margin:0 0 6px}
.bar.str{margin-top:20px}
.num{flex:0 0 1.9em;text-align:right;font:500 10px/1 ui-monospace,Menlo,
  monospace;color:var(--lig);padding-top:1.6em}
.corps{flex:1 1 auto;min-width:0}
.sect{font:600 12px/1 -apple-system,system-ui,sans-serif;color:var(--eux);
  letter-spacing:.08em;text-transform:uppercase;margin:26px 0 8px;
  padding-bottom:4px;border-bottom:1px solid var(--lig)}
.sect:first-of-type{margin-top:6px}

"""


def page_barres(titre: str, feuille: list[dict], tab: dict,
                sous_extra: str = "") -> str:
    """La feuille par mesures — une ou deux barres par ligne, l'accord en tête.

    `feuille` : ce que rend `feuille_barres`.
    """
    blocs, section = [], None
    for l in feuille:
        if l.get("tete") and l["mots"]:
            p0 = l["mots"][0]
            p0["accords"] = [l["tete"], *p0.get("accords", [])]
            p0["petits"] = [bool(l.get("tete_petit")),
                            *(p0.get("petits") or [])]
        if l.get("section") and l["section"] != section:
            section = l["section"]
            blocs.append(f'<div class="sect">{html.escape(section)}</div>')
        u = []
        for w in l["mots"]:
            pet = w.get("petits") or [False] * len(w["accords"])
            acc = "".join(
                f'<em class="{"pt" if (pet[k] if k < len(pet) else False) else ""}">'
                f'{html.escape(a)}</em>'
                for k, a in enumerate(w["accords"]))
            u.append(f'<span class="u{" dout" if w["douteux"] else ""}">'
                     f'<i>{acc}</i><b>{html.escape(w["mot"])}</b></span>')
        # LA TÊTE SE FOND DANS LES ACCORDS DU PREMIER MOT. En faire une
        # unité à part obligeait à lui réserver une largeur invisible, qui
        # décalait le premier mot vers la droite. Placée devant les accords du
        # premier mot, elle occupe la même rangée sans rien pousser — et c'est
        # exactement la disposition d'un tab : l'accord de la barre, puis les
        # changements, au-dessus des mots.
        # une strophe respire : c'est le tabbeur qui a mis la ligne vide
        blocs.append(
            f'<div class="bar{" str" if l.get("strophe") else ""}">'
            f'<div class="num">{l["mesure"]}</div>'
            f'<div class="corps">'
            f'<div class="ligne">{" ".join(u)}</div></div></div>')
    note = (f'texte : <a href="{html.escape(tab.get("url") or "")}">'
            f'{html.escape(tab.get("titre") or "?")}</a>' if tab else "")
    return (f'<!doctype html><html lang="fr"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(titre)}</title><style>{CSS_BARRES}</style>'
            f'<h1>{html.escape(titre)}</h1>'
            f'<p class="sous">{sous_extra}{" · " if sous_extra and note else ""}'
            f'{note}<br>Une ligne = deux mesures. L\'accord en tête est celui '
            f'qui sonne sur la barre ; il est toujours écrit, même s\'il se '
            f'répète. Les accords de passage sont en petit.</p>'
            f'{"".join(blocs)}</html>')
