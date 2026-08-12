"""harmonia_min/section_tool.py — le moteur de l'outil sections du chart brut.

Louis, 2026-08-09 : « un petit outil sections, où on clique sur une section,
on sélectionne A B C intro verse bridge, on passe le doigt le long des
barres correspondantes, puis on valide. Dès qu'on valide, l'outil repère
automatiquement les autres repeats de la section (si on a sélectionné 16
barres, on peut partir du principe que ce sont deux répétitions de 8
mesures, ou 4 de 4, et chercher les répétitions à cette granularité-là dans
le morceau) → output : morceau fait par sections. »

Ce module est la moitié SERVEUR : il ne connaît ni le doigt ni les boutons.
Il reçoit un intervalle de mesures choisi à la main et rend les autres
occurrences de la même musique.

IL N'INVENTE PAS DE DÉTECTEUR. Le détecteur de prod (`voice_sections`) sait
déjà répondre à « où ce bloc se rejoue-t-il ? » — c'est même tout ce qu'il
fait : `_slide` glisse un bloc sur le morceau, `block_score` fusionne
harmonie et chant en normalisant par l'ancre, `_peaks` garde les vrais pics
sur la grille de 2. Réglé contre les dix-sept morceaux annotés par Louis
(score 0,782). La seule chose qui change ici : **l'ancre n'est plus un
curseur qui balaie le morceau, c'est le doigt de l'utilisateur.** Écrire un
second scorer de similarité aurait créé deux vérités divergentes pour la
même question — l'outil aurait pu contredire le chart qu'il annote.

TROIS AJOUTS, chacun pour une raison précise.

1. LA GRANULARITÉ SE DEVINE. Une sélection de L mesures peut être une
   cellule de L, L/2, L/4 ou L/8 répétée. On teste ces découpages sur la
   sélection ELLE-MÊME (ses sous-blocs se ressemblent-ils ?) et on retient
   le plus petit qui tient. C'est la demande de Louis, et c'est aussi
   « under-fold, never over-fold » pris par l'autre bout : on ne découpe
   pas en 4 ce qui ne se répète qu'en 8, mais si ça se répète en 4 on le
   dit — sinon les reprises de 4 mesures restent invisibles.
   MESURÉ (18 morceaux annotés, 100 occurrences à retrouver) : chercher
   avec la cellule ou avec la sélection entière donne le MÊME rappel
   (81 % contre 80 %, bruit 1016 contre 948 mesures). La lecture littérale
   de la demande ne coûte donc rien — elle est appliquée telle quelle.
   Attention au piège de mesure qui a failli la faire rejeter : compter
   une occurrence annotée comme retrouvée seulement si UNE trouvaille la
   couvre à moitié déclarait 0 % pour les cellules (deux cellules de 4
   collées couvrent une occurrence de 8, chacune à 50 % pile). La bonne
   question est la couverture par l'UNION des trouvailles.

2. LA TRANSPOSITION EST CHERCHÉE. Une reprise modulée est la même section ;
   la matrice harmonique ordinaire y est structurellement aveugle (mesuré
   sur Sunny, qui monte d'un demi-ton par reprise : 0,10–0,16 sans
   rotation, 0,93–0,97 avec). En prod la rotation ne vit que dans
   `merge_letters`, APRÈS coup ; ici elle doit être dans la recherche,
   parce que l'utilisateur désigne une occurrence et attend les autres.

3. LE CHANT EST OPTIONNEL. `block_score` veut deux voies (harmonie +
   mélodie) ; la mélodie coûte une séparation de voix (demucs) qui n'est
   PAS forcément en cache quand le chart brut vient d'apparaître — et
   l'outil doit répondre au doigt, tout de suite. Sans mélodie on passe la
   voie harmonique dans les deux entrées : `block_score` retombe alors
   exactement sur l'harmonie seule. La réponse dit laquelle a servi.

4. LES LETTRES DÉJÀ VALIDÉES SONT RETIRÉES DU JEU (`claimed_bars`).
   L'utilisateur étiquette A, valide, puis B… Passer au moteur ce qui est
   déjà pris fait 86 % de rappel contre 82 %, et 411 mesures hors-lettre
   contre 711 — mesuré sur les mêmes 18 morceaux. Nuance qui décide de
   tout : on ne retire QUE ce que l'utilisateur a validé, jamais ce que
   l'outil a proposé. Une première version verrouillait les trouvailles
   brutes de A ; ses erreurs mangeaient les mesures de B et de C, et le
   rappel tombait à 53 %.

CE QUE CE MODULE NE FAIT PAS (règle #4) : il ne devine aucune frontière que
l'utilisateur n'a pas donnée, il ne nomme rien, il ne touche pas aux
accords, et il ne comble pas les trous entre sections — les mesures que
personne n'a réclamées restent explicitement en attente. Il ne DÉTECTE pas
non plus la mesure insérée qui fait basculer la parité d'un morceau : il la
contourne ancre par ancre, comme la prod (`voice_sections._pass`).

CE QUI RESTE À LA MAIN, mesuré : 14 % des occurrences ne sont pas
retrouvées, et il reste ~12 mesures hors-lettre par lettre à
désélectionner. L'outil réduit le travail, il ne le supprime pas.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Les découpages testés sur la sélection (décision n°1). Jamais moins de
#: 2 mesures : la grille de l'annotation est la double-barre (UNIT=2).
SPLITS = (1, 2, 4, 8)
MIN_CELL_BARS = 2
#: Cohérence interne minimale pour retenir un découpage. 0,88 = le régime
#: des vraies boucles mesuré par le repli (PERIOD_MIN_SCORE 0,80 sur un
#: autre substrat ; les boucles réelles y scorent 0,86–0,94).
TILE_MIN = 0.88
#: Ce qu'une reprise TRANSPOSÉE doit dépasser, en plus du seuil normal.
#: Avec douze rotations à essayer le maximum monte tout seul : une exigence
#: molle ferait passer n'importe quoi pour une modulation (même raison, et
#: même valeur, que MERGE_ROT dans voice_sections).
ROT_BONUS = 0.06
#: Le seuil de l'OUTIL, plus haut que celui du détecteur automatique
#: (THR8=0,66 / THR4=0,70). Mesuré sur les 18 morceaux annotés, 100
#: occurrences à retrouver — balayage complet :
#:   0,66 → 81 % retrouvées, 1044 mesures hors-lettre
#:   0,72 → 81 %,  851
#:   0,78 → 82 %,  711     ← retenu : meilleur rappel, un tiers de bruit en moins
#:   0,84 → 78 %,  587
#:   0,90 → 67 %,  471
#: Pourquoi plus haut qu'en prod : le détecteur avance de gauche à droite et
#: chaque ancre VERROUILLE ce qu'elle prend, ce qui protège les suivantes.
#: L'outil, lui, part d'une sélection isolée sur un morceau entier libre —
#: sans ce verrou, un seuil de 0,66 ramasse tout ce qui boucle.
TOOL_THR = 0.78
#: Garde-fou d'inondation : au-delà de ce rapport entre le nombre de
#: reprises trouvées avec la petite cellule et avec la sélection entière, la
#: cellule n'identifie plus la section — elle a trouvé la boucle du morceau.
#: Constaté au rendu : un geste de 4 mesures sur Bein Green devenait une
#: cellule de 2 et rendait 21 reprises. Mesuré sur les 18 morceaux, le
#: garde-fou coûte 1 point de rappel (80 % contre 81 %).
FLOOD_RATIO = 3.0


def _vs():
    from harmonia_min import voice_sections as VS
    return VS


def substrates(grid, triad, audio=None, melody: bool = False):
    """Les matrices de similarité mesure×mesure, comme la prod les fait.

    Renvoie (S, V, M, mute, channels) : S = harmonie (n,n), V = vecteurs de
    hauteurs (n,12) pour la rotation, M/mute = chant (ou None), et le nom
    des voies effectivement disponibles.
    """
    from harmonia_min import harmonic_sections as HS
    V = HS.harmonic_vectors(triad, grid)
    S = V @ V.T
    if not melody or audio is None:
        return S, V, None, None, "harmonie"
    VS = _vs()
    VA, VM, MS, _B8 = VS._scripts()
    voc = VA.separate_vocals(Path(audio))
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, len(grid) - 1)
    return S, V, M, mute, "harmonie+chant"


def _cell_agreement(S, b0, L, p) -> float:
    """Cohérence interne : les L/p cellules de la sélection se
    ressemblent-elles ? Moyenne des comparaisons bloc-à-bloc, sur le
    substrat harmonique (le même que la fusion de lettres en prod)."""
    VS = _vs()
    n_cells = L // p
    if n_cells < 2:
        return 1.0                      # une seule cellule : rien à trahir
    vals = [VS._diag(S, b0 + i * p, b0 + j * p, p)
            for i in range(n_cells) for j in range(i + 1, n_cells)]
    return float(np.mean(vals)) if vals else 1.0


def choose_granularity(S, b0, b1) -> dict:
    """Décision n°1 : le plus petit découpage de la sélection qui se tient."""
    L = b1 - b0 + 1
    tried = []
    for s in SPLITS:
        if s > 1 and L % s:
            continue
        p = L // s
        if p < MIN_CELL_BARS:
            continue
        sc = _cell_agreement(S, b0, L, p)
        tried.append({"cell_bars": p, "agreement": round(sc, 3),
                      "ok": bool(sc >= TILE_MIN)})
    ok = [t for t in tried if t["ok"] and t["cell_bars"] < L]
    pick = min(ok, key=lambda t: t["cell_bars"]) if ok else \
        {"cell_bars": L, "agreement": 1.0}
    return {"cell_bars": pick["cell_bars"], "n_inner": L // pick["cell_bars"],
            "agreement": pick["agreement"], "tried": tried}


def _rotated_S(V, k):
    """La matrice harmonie quand on transpose la RECHERCHE de k demi-tons."""
    return V @ np.roll(V, k, axis=1).T


def find_repeats(grid, triad, b0: int, b1: int, *, audio=None,
                 melody: bool = False, transpose: bool = True,
                 thr: float | None = None,
                 force_cell: int | None = None,
                 claimed_bars=()) -> dict:
    """Les autres occurrences de la sélection [b0, b1] dans le morceau.

    Renvoie un dict prêt à sérialiser :
      {cell_bars, n_inner, agreement, tried, thr, channels, parity,
       occurrences: [{b0, b1, score, rot, source}],
       curve: [...]}   # le score en chaque mesure, pour un curseur d'UI
    `source` : "selection" (ce que le doigt a désigné), "inner" (les
    répétitions internes à la sélection), "found" (ce que l'outil a
    trouvé ailleurs), "found-transposed" (idem, modulé).
    """
    VS = _vs()
    S, V, M, mute, channels = substrates(grid, triad, audio, melody)
    n = len(grid) - 1
    b0 = max(0, min(int(b0), n - 1))
    b1 = max(b0, min(int(b1), n - 1))
    if b1 == b0:
        # Une seule mesure : on ÉLARGIT la sélection à deux (vers l'avant, ou
        # vers l'arrière en fin de morceau) au lieu de chercher avec un bloc
        # plus grand qu'elle. Un bloc de 1 n'identifie aucune section, et un
        # bloc plus LONG que la sélection casse le scorer : `_slide` ne
        # calcule rien en b0, `block_score` divise alors par 1e-6 et rend des
        # scores à 1,5 million avec des reprises hors du morceau (constaté à
        # l'audit sur la dernière mesure de Bein Green).
        if b1 + 1 < n:
            b1 += 1
        elif b0 > 0:
            b0 -= 1
    L = b1 - b0 + 1
    g = choose_granularity(S, b0, b1)
    if force_cell:                          # bras d'expérience / garde-fou
        p = max(MIN_CELL_BARS, min(int(force_cell), L))
        g = {**g, "cell_bars": p, "n_inner": max(1, L // p)}
    base_thr = float(thr) if thr is not None else TOOL_THR
    Mm = M if M is not None else S          # pas de chant → harmonie seule

    base_claimed = np.zeros(n, bool)
    for b in claimed_bars:                 # ce que les lettres déjà validées
        if 0 <= int(b) < n:                # occupent — l'utilisateur étiquette
            base_claimed[int(b)] = True    # A, puis B, puis C…

    def _search(p):
        """Une recherche complète avec la cellule de p mesures."""
        n_in = max(1, L // p)
        cm = VS._slide(Mm, b0, p, n)
        ch = VS._slide(S, b0, p, n)
        sc = VS.block_score(cm, ch, b0, n, mute=mute, block=p)
        claimed = base_claimed.copy()
        for c in range(n_in):              # la sélection est déjà réclamée
            claimed[b0 + c * p:min(n, b0 + (c + 1) * p)] = True
        occ = [{"b0": b0 + c * p, "b1": b0 + (c + 1) * p - 1, "score": 1.0,
                "rot": 0, "source": "selection" if c == 0 else "inner"}
               for c in range(n_in)]
        parity = "paire"
        hits = VS._peaks(sc, b0, n, base_thr, p, claimed, par=0)
        if not hits:
            # Le repli de parité impaire (règle de prod) : une chanson qui
            # gagne une mesure en route bascule sa seconde moitié sur l'autre
            # parité. Autorisé seulement quand la voie paire est VIDE — donc
            # là où il n'y a rien à casser — et il faut payer ODD_BONUS.
            hits = VS._peaks(sc, b0, n, base_thr + VS.ODD_BONUS, p, claimed,
                             par=1)
            if hits:
                parity = "impaire"
        for h in hits:
            claimed[h:min(n, h + p)] = True
            occ.append({"b0": h, "b1": min(n - 1, h + p - 1),
                        "score": round(float(sc[h]), 3), "rot": 0,
                        "source": "found"})
        # La transposition ne s'applique qu'aux cellules d'au moins 4 mesures :
        # une cellule de 2 mesures tournée colle partout (Bein Green : 10
        # « reprises transposées » sur un motif de 2 mesures, toutes fausses).
        if transpose and p >= 4:
            for k in range(1, 12):
                Sk = _rotated_S(V, k)
                chk = VS._slide(Sk, b0, p, n)
                # `block_score` moyenne DEUX voies. Sans chant, la seconde est
                # l'harmonie elle-même : lui passer la version NON tournée
                # faisait voter la moitié de l'évidence contre la reprise
                # modulée qu'on cherche. Avec chant, la mélodie est invariante
                # par transposition à ce niveau, donc elle reste telle quelle.
                cmk = chk if M is None else cm
                sck = VS.block_score(cmk, chk, b0, n, mute=mute, block=p)
                for h in VS._peaks(sck, b0, n, base_thr + ROT_BONUS, p,
                                   claimed):
                    claimed[h:min(n, h + p)] = True
                    occ.append({"b0": h, "b1": min(n - 1, h + p - 1),
                                "score": round(float(sck[h]), 3), "rot": k,
                                "source": "found-transposed"})
        occ.sort(key=lambda o: o["b0"])
        n_found = sum(1 for o in occ if o["source"].startswith("found"))
        return {"cell_bars": p, "n_inner": n_in, "occurrences": occ,
                "parity": parity, "n_found": n_found,
                "curve": [round(float(x), 3) for x in sc]}

    p = min(L, max(MIN_CELL_BARS, g["cell_bars"]))   # jamais plus que la sélection
    # Une sélection d'UNE mesure : le bloc de recherche est quand même de
    # deux. Un bloc d'une mesure ne peut identifier aucune section — il
    # matche tout ce qui contient le même accord, et le garde-fou
    # d'inondation ne peut pas s'armer (il compare la cellule à la sélection
    # entière, ici identiques). La grille d'annotation est de toute façon la
    # double-barre (UNIT=2 dans la page de Louis).
    res = _search(p)
    fallback = None
    if p < L:
        # GARDE-FOU DE L'INONDATION. Découper la sélection est ce que Louis a
        # demandé, et sur les chiffres c'est neutre (81 % contre 80 % de
        # rappel). À l'usage, non : sur Bein Green, un geste de 4 mesures
        # devient une cellule de 2 qui trouve VINGT-ET-UNE reprises — le
        # morceau entier s'allume et l'outil est inutilisable. Une cellule
        # qui trouve beaucoup plus que la sélection entière n'identifie plus
        # la SECTION, elle a trouvé la boucle harmonique du morceau. Dans ce
        # cas on garde la sélection entière et on le DIT.
        whole = _search(L)
        if res["n_found"] > FLOOD_RATIO * max(whole["n_found"], 1):
            fallback = {"cell_bars": p, "n_found": res["n_found"]}
            res = whole
            p = L

    logger.info("outil sections: sélection %d-%d (L=%d) → cellule %d mesures "
                "(cohérence %.2f), seuil %.2f, parité %s, %d occurrences [%s]"
                "%s", b0, b1, L, p, g["agreement"], base_thr, res["parity"],
                len(res["occurrences"]), channels,
                f" — repli anti-inondation depuis {fallback['cell_bars']} "
                f"mesures ({fallback['n_found']} reprises)" if fallback else "")
    return {"cell_bars": res["cell_bars"], "n_inner": res["n_inner"],
            "agreement": g["agreement"], "tried": g["tried"],
            "thr": round(base_thr, 3), "channels": channels,
            "parity": res["parity"], "flood_fallback": fallback,
            "occurrences": res["occurrences"], "curve": res["curve"]}


def sections_from_marks(marks: list[dict], n_bars: int) -> list[dict]:
    """Les marques validées → la liste de sections du morceau.

    `marks` = [{label, occurrences:[{b0,b1,…}]}] dans l'ordre de validation.
    Sortie au format EXACT des annotations de Louis
    (`state/sections/<stem>.json` : [{label, b0, b1}], indices de mesure,
    bornes incluses) — c'est le même fichier, le même endpoint, et les
    scripts de mesure (`section_bench`, `section_metric`) le lisent déjà.

    Deux règles :
      * une marque plus récente gagne sur une plus ancienne en cas de
        recouvrement (la dernière intention de l'utilisateur fait foi) ;
      * les mesures qu'aucune marque ne couvre restent des trous EXPLICITES
        (`label: "?"`, `pending: true`), jamais rattachées d'office à la
        section voisine — Louis lui-même laisse des mesures sans section
        (She Will Be Loved, mesure 33), et un outil qui comblerait tout seul
        lui ferait signer une frontière qu'il n'a pas vue.
    Deux occurrences adjacentes d'une même lettre restent DEUX entrées :
    les fusionner effacerait la reprise, qui est justement l'objet
    (`section_metric` pose la même règle).
    """
    owner: list[str | None] = [None] * max(0, n_bars)
    clean = []                             # (b0, b1, label) valides seulement
    dropped = 0
    for m in marks if isinstance(marks, list) else []:
        if not isinstance(m, dict):
            dropped += 1
            continue
        lab = m.get("label")
        occ = m.get("occurrences")
        if not isinstance(lab, str) or not lab.strip() or lab.strip() == "?" \
                or not isinstance(occ, list):
            dropped += 1
            continue
        for o in occ:
            try:
                a, z = int(o["b0"]), int(o["b1"])
            except (TypeError, ValueError, KeyError, IndexError):
                dropped += 1
                continue
            a, z = max(0, a), min(n_bars - 1, z)
            if a > z:
                dropped += 1
                continue
            clean.append((a, z, lab.strip()))
    if dropped:
        logger.warning("sections_from_marks: %d marque(s) ignorée(s) "
                       "(hors bornes, label vide ou format faux)", dropped)
    for a, z, lab in clean:
        for b in range(a, z + 1):
            owner[b] = lab
    # Les débuts d'occurrence qui POSSÈDENT encore leur mesure. Sans ce
    # filtre, une occurrence entièrement recouverte par une marque plus
    # récente laissait quand même sa frontière : « B 1–16 » ressortait en
    # deux entrées B adjacentes, ce qui veut dire UNE REPRISE dans ce format.
    starts = {a for a, z, lab in clean if 0 <= a < n_bars and owner[a] == lab}
    out, b = [], 0
    while b < n_bars:
        lab = owner[b]
        e = b
        while (e + 1 < n_bars and owner[e + 1] == lab
               and (e + 1) not in starts):
            e += 1
        entry = {"label": lab or "?", "b0": b, "b1": e}
        if lab is None:
            entry["pending"] = True
        out.append(entry)
        b = e + 1
    return out
