"""Deux façons de découper un morceau à partir des briques de Louis.

CE MODULE EST EN PRODUCTION depuis le 2026-09-17. Il est né comme banc
d'essai — Louis : « n'essaye pas de fix, c'est moi qui dirai comment fix »,
puis « fais-moi une simulation des 2 stratégies pour que je puisse les tester
interactivement » — et il a été acté le soir même : « notre brique +
SongFormer comme méthode principale, c'est acté, tu me le mets en prod ».

`sections_inferer` (la route qu'appelle « Valider les sections ») lit
`occurrences_de_tous` + `remplir_par_songformer`. `remplir_par_algo` n'est
plus qu'un point de comparaison, gardé pour la route `/api/sections/simuler/`
et la page `tools/chart_simule.py`, où Louis peut confronter les deux.

LE POINT DE DÉPART COMMUN, et c'est la découverte de la journée. Aujourd'hui
`sections_inferer` jette le trait de Louis dans une agglomération générique
qui redécoupe tout le morceau. Lui veut l'inverse :

    « Une fois que j'ai annoté une section, ça devient une brique, et la
    première chose à faire c'est de trouver d'autres occurrences de cette
    section. »

`cherche_brique` fait exactement ça, et c'est mesuré : sur Don't Want My Love
avec sa brique A (mesures 7-14), elle rend 7-14, 15-22 et 34-41 — les trois
occurrences que SongFormer trouve, et exactement celles que la production
rate (elle propose 15-22, 29-36, 49-56). Le résultat ne bouge pas entre 0,85
et 0,95 de seuil.

LES DEUX STRATÉGIES ne diffèrent QUE par ce qu'elles mettent dans les trous
— et c'est SongFormer qui a été retenu :

  * `remplir_par_algo` — l'agglomération des quatre mots, mais confinée aux
    trous, toutes les occurrences gelées ;
  * `remplir_par_songformer` — ce que le modèle avait trouvé, au lieu de le
    jeter. C'est la seule des deux qui sache écrire « intro » et « outro » :
    les lettres de l'autre ne peuvent venir que de Louis ou de l'alphabet —
    sauf la zone de tête, « intro » depuis le 2026-09-30.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("harmonia.sections.simulation")

#: plancher absolu : sous ce score, ce n'est jamais la même brique
PLANCHER_REPRISE = 0.85
#: combien d'écarts-types au-dessus de la médiane DU MORCEAU une reprise doit
#: se détacher — le seuil est relatif, parce que la distribution ne l'est pas
Z_REPRISE = 1.5
#: on ne dépasse jamais ça : au-delà, même la brique elle-même serait recalée
PLAFOND_REPRISE = 0.999
#: ces rôles ne se rejouent pas — Louis, 2026-09-16 : « une intro ne se rejoue
#: pas plus tard »
JAMAIS_REJOUEES = {"intro", "outro", "silence"}
#: l'alphabet des lettres neuves, d'où l'on retire celles de Louis
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: CE QU'UNE REPRISE TRANSPOSÉE DOIT PAYER EN PLUS. Repris tel quel de
#: `section_tool.ROT_BONUS` (2026-09-19) : une brique tournée ressemble à plus
#: de choses qu'une brique droite, donc elle doit se détacher davantage pour
#: qu'on la croie.
BONUS_TRANSPOSE = 0.06
#: EN DESSOUS DE QUATRE MESURES, ON NE CHERCHE PAS DE TRANSPOSITION. Mesuré
#: dans `section_tool` sur Bein Green : un motif de 2 mesures tourné colle
#: PARTOUT — dix « reprises transposées », toutes fausses. Un motif court n'a
#: pas assez de matière pour que sa transposition veuille dire quelque chose.
MIN_MESURES_TRANSPOSE = 4
#: UNE SECTION, C'EST UNE MINI-BOUCLE D'AU MOINS 4 MESURES jouée au moins
#: 2 fois, modulo la dernière (Louis, 2026-08-16). La boucle cherchée dans une
#: brique ne descend jamais sous ça, et une brique de moins de 2 × 4 mesures ne
#: peut pas MONTRER une boucle : elle garde l'ancienne loi (voir plus bas).
MIN_BOUCLE = 4
#: RÈGLE 3 (2026-09-30) — un départ est une reprise quand son score a franchi
#: au moins cette FRACTION du chemin entre le fond local de la courbe et une
#: reprise parfaite (1,0). Mesuré sur les 20 découpages validés de Louis : les
#: vraies reprises à 0,68–1,0 (Stand By Me, la plus basse : 0,70–0,77), le
#: palier va de 0,5 à 0,6 sans perdre une mesure juste, 0,65 commence à en
#: perdre. Ce qui passe au-dessus et qui est faux, ce sont des jumeaux
#: harmoniques (mêmes accords ailleurs), qu'aucun seuil ne sépare.
PROEMINENCE_MIN = 0.6
#: un fond à moins de ça d'une reprise parfaite n'est plus un fond : le passage
#: ENTIER ressemble à la brique (vamp, Chain of Fools, tout à 0,99). Sans ce
#: plancher, (c − fond)/(1 − fond) divise du bruit par presque rien.
ECART_FOND_MIN = 0.05
#: RÈGLE 4 (2026-09-30 soir, voir `reference_du_morceau`) : un trou entre
#: deux hauteurs de pics consécutives coupe s'il dépasse TROU_K fois la
#: variabilité des reprises du morceau.
TROU_K = 2.0
#: la plus petite variabilité admise : en dessous, deux pics à 0,9998 et
#: 0,9999 « feraient un trou » (The Ronettes C) — c'est du bruit de moyenne.
BRUIT_PIC = 0.01
#: RÈGLE 5 (2026-09-30 nuit, voir `maxima_locaux`) : un départ doit dépasser
#: ses voisins à moins d'une boucle. Interrupteur pour la page de comparaison
#: v3/v4 (`tools/occurrences_brique_v4.py`) — pas un réglage, toujours vrai.
REGLE5 = True
#: RÈGLE 5b (Louis, 2026-09-30, sur la page v4 : « Oui pour la 5b très
#: bien ») : deux départs voisins sont à égalité tant que leur écart reste
#: sous la TOLÉRANCE DU MORCEAU de la règle 4 (0,04–0,07 mesuré), plus sous
#: `BRUIT_PIC`. Interrupteur pour rejouer v4 dans la page v5 — toujours vrai.
EGALITE_5B = True
#: ... « mais alors il faut trancher par rapport à la cohérence globale du
#: morceau, en cherchant d'autres sections et en gardant ces 2 candidats
#: potentiels sans en acter un » : voir `resoudre_suspens`. Interrupteur pour
#: la page v5 (faux = l'ancien choix, le plus de boucles puis le score).
SUSPENS = True
#: OPTION (c) (Louis, 2026-09-30, après la page v5 : Sunny A perdait 18–25
#: au profit d'un jumeau de C) : une autre section ne mange une zone tenue par
#: un calage en suspens que si son score y dépasse celui du calage de PLUS que
#: la tolérance du morceau — celle qui a défini l'égalité (5b) ; pas de
#: nouvelle constante. Voir `occurrences_de_tous`. Interrupteur pour la page
#: v5 (faux = v5 : toute reprise sûre mange) — toujours vrai.
MANGE_NETTEMENT = True
#: au-delà de ce nombre d'alternatives dans un groupe (un long plateau), on
#: n'énumère plus : le groupe est tranché tout de suite au score, comme avant.
SUSPENS_MAX_ALT = 16


def _chevauche(a: int, b: int, pris) -> bool:
    return any(not (b < x or a > y) for x, y in pris)


def _niveau_brique(S, b0: int, L: int) -> float:
    """Ce que « la même mesure » veut dire DANS CETTE BRIQUE : à mi-chemin
    entre 1 (identique) et la ressemblance médiane de ses mesures entre elles.
    Relatif à la brique, pas une constante : sur une brique où tout se
    ressemble il monte, sur une brique contrastée il descend."""
    import numpy as np
    vals = [S[b0 + i, b0 + j] for i in range(L) for j in range(i + 1, L)]
    return (1.0 + float(np.median(vals))) / 2.0 if vals else 1.0


def _periode(S, b0: int, L: int, tau: float, lo: int, hi: int):
    """(l, T) — le plus petit décalage l de [lo, hi] auquel la brique se répète
    elle-même, ou (None, None). Les mesures doivent coller à leur partenaire
    l mesures plus loin (≥ tau) depuis le DÉBUT de la brique, sans trou, sur
    une suite de T mesures ; il faut T ≥ 2 l (au moins deux passages) et ce qui
    reste après la suite ne dépasse pas un passage — la dernière fois peut
    avoir une autre fin (« modulo la dernière »)."""
    for l in range(lo, hi + 1):
        r = 0
        while r < L - l and S[b0 + r, b0 + r + l] >= tau:
            r += 1
        T = l + r
        if T >= 2 * l and L - T <= l:
            return l, T
    return None, None


def boucle_minimale(S, b0: int, b1: int) -> dict:
    """RÈGLE 2 de Louis (2026-09-30) : « Quand je donne une section, il faut
    regarder si elle-même est une répétition d'une boucle minimale [...] C'est
    la boucle minimale qui définit la section qui devra ensuite être
    cherchée. »

    Rend {"boucle": Lb, "suite": T, "sous_periode": p0, "tau": ...}. `Lb` est
    le plus petit décalage de 4 mesures ou plus (jamais moins : une section
    fait au moins 4 mesures) auquel la brique se rejoue elle-même depuis son
    début — voir `_periode` —, ou la longueur de la brique si elle ne se
    répète pas. `T` = combien de mesures la suite de boucles couvre ; ce qui
    reste au-delà est la FIN de la brique (autre fin, ou passage incomplet).

    `p0` est la vraie période harmonique DANS la boucle, qui peut descendre
    sous 4 (deux accords qui alternent : p0 = 2 dans une boucle de 4). Un
    départ décalé d'un multiple de p0 ressemble à la boucle aussi bien que le
    bon : l'harmonie ne peut pas choisir la phase, et ces décalages ne doivent
    pas servir de fond (voir `_fond`).

    CE QUE ÇA NE RÉSOUT PAS : une brique dont la boucle interne varie à chaque
    passage (autre accord en mesure 3 au 2e tour) n'est pas vue comme une
    boucle — il faut que chaque mesure colle, sauf dans le dernier passage. Une
    brique qui est un vamp (un seul accord, tout se ressemble) donne une boucle
    de 4 et une sous-période de 1 : la phase est alors indécidable, et la
    recherche s'abstient (aucun fond).
    """
    L = b1 - b0 + 1
    tau = _niveau_brique(S, b0, L)
    Lb, T = L, L
    if L >= 2 * MIN_BOUCLE:
        l, t = _periode(S, b0, L, tau, MIN_BOUCLE, L // 2)
        if l is not None:
            Lb, T = l, t
    p0, _ = _periode(S, b0, min(L, 2 * Lb), tau, 1, Lb) if Lb < L else (None, None)
    if p0 is None or Lb % p0:
        p0 = Lb
    return {"boucle": Lb, "suite": T, "sous_periode": p0, "tau": tau}


def _fond(c, p: int, Lb: int, p0: int, interdit) -> float | None:
    """RÈGLE 3 : le FOND local du départ p — la médiane de la courbe aux
    départs décalés de 1 à Lb−1 mesures de chaque côté (ce que la brique donne
    quand elle est mal calée sur une vraie reprise). On n'y met ni les voisins
    à ±Lb (ce sont peut-être d'autres reprises, qui ne sont pas du fond), ni
    les multiples de la sous-période p0 (ils collent aussi bien que p), ni —
    RÈGLE 1 — les départs dont la fenêtre chevauche la brique (ses décalages
    sur elle-même : ni des pics, ni du fond)."""
    import numpy as np
    vals = [c[q] for d in range(1, Lb) if d % p0 for q in (p - d, p + d)
            if 0 <= q < len(c) and not interdit(q)]
    return float(np.median(vals)) if vals else None


def reference_du_morceau(pics: list[dict]) -> dict | None:
    """RÈGLE 4 de Louis (2026-09-30 soir, après la page v2 : « beaucoup
    beaucoup beaucoup mieux », seul Let It Be est rejeté — le B, vraiment
    différent, sort comme un A) :

        « Prends le score du pic le + grand trouvé en dehors de la section
        déjà identifiée — ça nous donne une idée de la variabilité
        inter-chanson. Ensuite utilise cet indice de variabilité + la taille
        de tous les autres pics pour décider d'un score qui permet de
        différencier le A du B dans Let It Be, tout en gardant Stand By Me. »

    Parmi les départs qui passent la règle 3, R = le plus haut (hors zone
    donnée, par construction). On trie les autres par hauteur et on cherche le
    PLUS GRAND TROU entre deux hauteurs consécutives. La tolérance du morceau
    est `TROU_K × max(1 − R, dispersion du groupe du haut, BRUIT_PIC)` : 1 − R dit à quel
    point même la meilleure reprise de CE morceau est loin d'une copie
    parfaite (Stand By Me 0,057 : bruité ; Let It Be 0,000 : exact), la
    dispersion du haut dit à quel point les reprises du groupe de R varient
    entre elles. On coupe — tout ce qui est sous le trou est refusé (`ok =
    False`, `sous_reference = True`) — si le trou dépasse la tolérance ET si
    le groupe du bas est PLUS NOMBREUX que celui du haut (« la taille de tous
    les autres pics ») : c'est alors le niveau du reste du morceau (des
    jumeaux partout), pas quelques reprises à fin différente. Sans cette
    condition, Every Breath You Take perdait ses deux A à autre fin (0,919 et
    0,902 contre 0,998 : exactement les hauteurs du refrain de Let It Be),
    Don't Know Why, This Love et Fjufjv4rH0s une reprise chacun.

    Mesuré : Let It Be A — R 1,000, groupe du haut 0,994–1,000 (6 pics),
    trou 0,066 jusqu'à 0,928, tolérance 0,02, 15 pics en dessous → refrain
    refusé (24 mesures en trop → 0). Stand By Me A — R 0,943, plus grand trou
    0,020, tolérance 2 × 0,057 = 0,114 → rien ne bouge. Porte ET avec la règle 3 : la règle 3 juge un départ contre son
    fond local (elle écarte les décalages), la règle 4 juge les pics qui ont
    passé contre les autres pics (elle écarte les jumeaux moins ressemblants
    que les vraies reprises). Aucune des deux ne fait le travail de l'autre :
    le refrain de Let It Be a une proéminence de 0,63–0,73.

    CE QUE ÇA NE RÉSOUT PAS : un jumeau AUSSI ressemblant que les vraies
    reprises (She Will Be Loved, la même grille presque partout) ne laisse
    aucun trou. Si la meilleure trouvaille hors zone est elle-même un jumeau
    (aucune vraie reprise), elle devient la référence. Et vu depuis une
    section, un jumeau à 0,91 d'un morceau exact est refusé même quand Louis
    dit « même section » (Stand By Me vu depuis C : les A sont à 0,91 contre
    0,99, exactement la situation du refrain de Let It Be)."""
    oks = sorted((x for x in pics if x["ok"]), key=lambda x: -x["c"])
    if len(oks) < 2:
        return None
    h = [x["c"] for x in oks]
    gaps = [h[i] - h[i + 1] for i in range(len(h) - 1)]
    i = max(range(len(gaps)), key=lambda k: gaps[k])
    tol = TROU_K * max(1.0 - h[0], h[0] - h[i], BRUIT_PIC)
    n_haut, n_bas = i + 1, len(h) - i - 1
    ref = {"R": round(h[0], 4), "trou": round(gaps[i], 4), "bas_du_haut": round(h[i], 4),
           "tolerance": round(tol, 4), "n_haut": n_haut, "n_bas": n_bas,
           "coupe": bool(gaps[i] > tol and n_bas > n_haut)}
    for x in pics:
        x["sous_reference"] = False
    if ref["coupe"]:
        for x in oks[i + 1:]:
            x["ok"] = False
            x["sous_reference"] = True
    return ref


def maxima_locaux(c, Lb: int, candidats, tol: float | None = None) -> set[int]:
    """RÈGLE 5 de Louis (2026-09-30 nuit, après `docs/plots/the_walk_decalage.html`) :

        « si on hésite entre un départ et celui de ses deux voisins, on prend
        celui qui est plus grand que ses voisins d'avant et d'après
        (généralise au cas où on aurait des boucles plus grandes que 2) ».

    Rend les départs de `candidats` dont le score est le maximum de la courbe
    `c` parmi les AUTRES candidats à ±(Lb − 1) mesures — une suppression des
    non-maxima dans une fenêtre d'une boucle. Deux candidats à moins d'une
    boucle l'un de l'autre sont deux calages rivaux de la même reprise : seul
    le plus haut est un départ.

    LES VOISINS SONT DES CANDIDATS, pas toute la courbe : « on hésite entre »
    deux départs seulement s'ils ont tous deux passé la règle 3 (plancher +
    proéminence). Un plateau à 0,95 qui ne se détache pas de son fond n'est pas
    une reprise rivale : il n'efface pas un pic isolé à 0,88 juste à côté.

    ÉGALITÉ : un voisin ne bat p que s'il le dépasse de plus de `BRUIT_PIC`
    (0,01, le bruit de moyenne déjà admis par la règle 4). À moins que ça, les
    deux restent des départs et c'est la couverture qui choisit. Pas « le plus
    tôt gagne » : sur un plateau de sous-période (The Walk, A–Bm : 29, 31,
    33… tous à 0,995–1,0), la règle stricte ne garderait qu'UN départ, au
    hasard du bruit, et effacerait la moitié des vraies reprises.

    RÈGLE 1 : un départ dont la fenêtre chevauche la zone donnée n'est jamais
    candidat, donc jamais voisin — ses scores sont la brique contre elle-même,
    ils n'effacent pas une reprise posée juste après la brique.

    CE QUE ÇA NE RÉSOUT PAS : c'est la courbe qui décide, pas l'oreille. Si le
    calage décalé est plus haut que le bon (un jumeau qui déborde mieux),
    c'est lui qui reste. Sur un plateau (écarts ≤ 0,01), la phase reste
    choisie par la couverture : la règle ne tranche qu'entre des calages
    nettement différents. Et elle ne crée aucune reprise : elle ne fait
    qu'en retirer.

    RÈGLE 5b (2026-09-30) : `tol` remplace `BRUIT_PIC` comme seuil
    d'égalité — `cherche_sur_matrices` y passe la tolérance du morceau de la
    règle 4. À 0,01, Lazy Song A (bon départ 0,963, décalé de 2 à 0,991) et
    She Will Be Loved A (0,977 contre 0,994) perdaient des mesures justes :
    la règle tranchait sur du bruit. Les deux calages restent alors, et
    c'est `resoudre_suspens` qui tranche."""
    tol = BRUIT_PIC if tol is None else tol
    cands = sorted(candidats)
    out = set()
    for p in cands:
        voisins = [c[q] for q in cands if q != p and abs(q - p) < Lb]
        if not voisins or c[p] >= max(voisins) - tol:
            out.add(p)
    return out


def _selections(comp: list[int], Lb: int, cap: int) -> list[list[int]]:
    """Toutes les façons de caser le PLUS de boucles sans recouvrement parmi
    les départs `comp` (triés), au plus `cap` + 1 (au-delà : trop pour
    énumérer, l'appelant tranche au score)."""
    n = len(comp)
    nb = [0] * (n + 1)                  # nb[i] = le plus de boucles dans comp[i:]
    for i in range(n - 1, -1, -1):
        j = next((k for k in range(i + 1, n) if comp[k] >= comp[i] + Lb), n)
        nb[i] = max(nb[i + 1], 1 + nb[j])
    out: list[list[int]] = []

    def rec(i, cur, reste):
        if len(out) > cap:
            return
        if reste == 0:
            out.append(list(cur))
            return
        if i >= n or nb[i] < reste:
            return
        j = next((k for k in range(i + 1, n) if comp[k] >= comp[i] + Lb), n)
        if 1 + nb[j] >= reste:
            cur.append(comp[i])
            rec(j, cur, reste - 1)
            cur.pop()
        rec(i + 1, cur, reste)
    rec(0, [], nb[0])
    return out


def _termes(alt, placees, label, n) -> dict:
    """Ce qui juge une alternative, une fois les autres sections posées."""
    def touche(o, p):
        return not (o[1] < p["m0"] or p["m1"] < o[0])
    viv = [o for o in alt if not any(touche(o, p) for p in placees)]
    meme = [(p["m0"], p["m1"]) for p in placees if p["label"] == label] \
        + [(o[0], o[1]) for o in viv]
    fins, debuts = {z for _a, z in meme}, {a for a, _z in meme}
    contacts = len({o[0] for o in viv if o[0] - 1 in fins}
                   | {o[1] + 1 for o in viv if o[1] + 1 in debuts})
    tous = [(p["m0"], p["m1"]) for p in placees] + [(o[0], o[1]) for o in viv]
    trous = set()
    for o in viv:
        avant = max((z for _a, z in tous if z < o[0]), default=-1)
        apres = min((a for a, _z in tous if a > o[1]), default=n)
        for g0, g1 in ((avant + 1, o[0] - 1), (o[1] + 1, apres - 1)):
            if 1 <= g1 - g0 + 1 < MIN_BOUCLE:
                trous.add((g0, g1))
    return {"vivantes": viv, "mangees": len(alt) - len(viv), "contacts": contacts,
            "fragments": len(trous), "score": round(sum(o[2] for o in viv), 4)}


def resoudre_suspens(groupes: list[dict], placees: list[dict], n: int) -> list[tuple]:
    """Trancher les CANDIDATS EN SUSPENS par la cohérence globale du morceau.

    Louis, 2026-09-30, sur la page v4 : « Oui pour la 5b très bien, mais
    alors il faut trancher par rapport à la cohérence globale du morceau, en
    cherchant d'autres sections et en gardant ces 2 candidats potentiels sans
    en acter un. Puis on voit ce qui reste quand les autres sections ont
    mangé ce qu'elles ont à manger. »

    `groupes` : chacun `{"label", "boucle", "alternatives": [[(m0, m1, score,
    rot), ...], ...]}` — des calages rivaux d'une même reprise, exclusifs,
    qui casent le même nombre de boucles (voir `cherche_sur_matrices`).
    `placees` : tout ce qui est déjà posé, `{"m0", "m1", "label"}` — les
    traits de Louis et les reprises sûres de TOUTES ses briques. La liste est
    complétée au fur et à mesure (un groupe tranché compte pour les suivants).

    L'ordre des critères, chacun ne départageant que les ex æquo du
    précédent :
      1. `mangee` — les boucles de l'alternative déjà prises par une autre
         section tombent ; la plus complète reste ;
      2. `phase` — le plus de contacts avec les autres occurrences de la même
         brique (une boucle qui commence juste après une autre, ou finit
         juste avant) : même phase que ce qui est confirmé ;
      3. `fragments` — le moins de trous orphelins (1 à 3 mesures libres)
         laissés autour de ses boucles ;
      4. `score` — seulement si tout le reste est à égalité.
    Les groupes que le 1er critère tranche seul passent d'abord ; les
    autres dans l'ordre des mesures. Chaque groupe reçoit `choix` (indice),
    `raison` et `termes` (un dict par alternative), pour la page.

    Rend `[(label, occurrence), ...]` — les boucles retenues, à poser.

    CE QUE ÇA NE RÉSOUT PAS : les « autres sections » sont celles que Louis a
    tracées et leurs reprises sûres — pas les blocs que l'algo des quatre mots
    nommera ensuite dans les trous (il a besoin des occurrences pour
    travailler). Le critère `fragments` n'en est qu'une approximation : il
    évite de laisser à l'algo des bouts trop courts pour être une section.
    Seuls des calages qui casent AUTANT de boucles sont mis en suspens ; un
    calage à 2 boucles contre 1 est acté par la couverture, même si une autre
    section mange ensuite l'une de ses 2 boucles.
    """
    placees = list(placees)
    restants = sorted(groupes, key=lambda g: min(o[0] for a in g["alternatives"] for o in a))
    out: list[tuple] = []
    crit = (("mangee", lambda t: len(t["vivantes"])), ("phase", lambda t: t["contacts"]),
            ("fragments", lambda t: -t["fragments"]), ("score", lambda t: t["score"]))
    while restants:
        evals = [(g, [_termes(a, placees, g["label"], n) for a in g["alternatives"]])
                 for g in restants]

        def tranche_seul(ts):
            m = max(len(t["vivantes"]) for t in ts)
            return sum(1 for t in ts if len(t["vivantes"]) == m) == 1
        g, ts = next(((g, ts) for g, ts in evals if tranche_seul(ts)), evals[0])
        restants.remove(g)
        # chaque critère ne regarde que les ex æquo du précédent ; la raison
        # est celui qui laisse une seule alternative
        idx, raison = list(range(len(ts))), "score"
        m = max(len(t["vivantes"]) for t in ts)
        hauts = [k for k in idx if len(ts[k]["vivantes"]) == m]
        if m == 0:
            # tout est mangé : rien à poser, la zone appartient aux autres
            idx, raison = hauts, "tout_mange"
        elif len({tuple(ts[k]["vivantes"]) for k in hauts}) == 1 and \
                any(t["mangees"] for t in ts):
            # ce qui les distinguait a été mangé : il reste les mêmes boucles
            idx, raison = hauts, "mangee"
        else:
            for nom, f in crit:
                m = max(f(ts[k]) for k in idx)
                idx = [k for k in idx if f(ts[k]) == m]
                if len(idx) == 1:
                    raison = nom
                    break
        k = idx[0]
        g["choix"], g["raison"] = k, raison
        g["termes"] = [{x: t[x] for x in ("mangees", "contacts", "fragments", "score")}
                       | {"vivantes": [(o[0], o[1]) for o in t["vivantes"]]} for t in ts]
        for o in ts[k]["vivantes"]:
            placees.append({"m0": o[0], "m1": o[1], "label": g["label"]})
            out.append((g["label"], o))
    return out


def _passages(p: int, Lb: int, T: int, L: int) -> list[tuple[int, int]]:
    """La brique posée en p, découpée comme elle l'est elle-même : T // Lb
    boucles, puis sa fin (s'il en reste)."""
    k = T // Lb
    out = [(p + j * Lb, p + (j + 1) * Lb - 1) for j in range(k)]
    if k * Lb < L:
        out.append((p + k * Lb, p + L - 1))
    return out


def cherche_sur_matrices(S, V, b0: int, b1: int) -> tuple[list[tuple], dict]:
    """Le cœur de `cherche_brique`, sur des matrices déjà calculées (la page
    de diagnostic l'appelle directement). Rend (reprises, détail) ; chaque
    reprise est `(m0, m1, score, rot)`, triées, la brique (ou ses passages)
    comprise. `détail` porte la courbe, le fond et chaque pic jugé — tout ce
    qu'il faut pour le dessiner. Voir `cherche_brique` pour la loi."""
    import numpy as np

    from harmonia.sections import similarity as VS
    n = S.shape[0]
    L = b1 - b0 + 1
    b = boucle_minimale(S, b0, b1)
    Lb, T, p0 = b["boucle"], b["suite"], b["sous_periode"]
    detail = dict(b, mode="relatif", L=L, seuil=None, mediane=None, sigma=None,
                  pics=[], courbe=[])

    # RÈGLE 2 : la brique devient ses propres passages. Le premier vaut 1,0 ;
    # les suivants, leur ressemblance au premier.
    out = []
    for a, z in _passages(b0, Lb, T, L):
        m = z - a + 1
        sc = 1.0 if a == b0 else float(np.mean([S[b0 + i, a + i] for i in range(m)]))
        out.append((a, z, sc, 0))
    queue = (out[-1][0], out[-1][1]) if T // Lb * Lb < L else None
    pris = [(b0, b1)]

    def interdit(q, lg):
        return not (q + lg - 1 < b0 or q > b1)          # RÈGLE 1

    c = np.asarray(VS._slide(S, b0, Lb, n))[:max(1, n - Lb + 1)]
    detail["courbe"] = [round(float(x), 4) for x in c]

    if L < 2 * MIN_BOUCLE:
        # ANCIENNE LOI pour une brique trop courte pour montrer une boucle :
        # seuil du morceau médiane + Z σ, borné — mais calculé HORS de la zone
        # donnée (règle 1). Mesuré : la règle 3 sur une brique de 4 mesures
        # dont les accords alternent (Let It Be « bridge », Lazy Song C) la
        # recopiait sur tout le morceau (+52 et +48 mesures).
        detail["mode"] = "ancien"
        hors = np.array([c[q] for q in range(len(c)) if not interdit(q, Lb)])
        if len(hors) < 3:
            detail["raison"] = "trop_court"
            return sorted(set(out)), detail
        seuil = min(PLAFOND_REPRISE, max(PLANCHER_REPRISE,
                                         float(np.median(hors) + Z_REPRISE * hors.std())))
        detail.update(seuil=round(seuil, 3), mediane=round(float(np.median(hors)), 3),
                      sigma=round(float(hors.std()), 3))

        def moissonne(scores, seuil_k, rot):
            for q in sorted(range(len(scores)), key=lambda x: -scores[x]):
                if scores[q] < seuil_k:
                    break
                if _chevauche(q, q + L - 1, pris):
                    continue
                pris.append((q, q + L - 1))
                out.append((q, q + L - 1, float(scores[q]), rot))
        moissonne(c, seuil, 0)
        if L >= MIN_MESURES_TRANSPOSE:
            for k in range(1, 12):
                Sk = V @ np.roll(V, k, axis=1).T
                moissonne(np.asarray(VS._slide(Sk, b0, L, n))[:len(c)],
                          min(PLAFOND_REPRISE, seuil + BONUS_TRANSPOSE), k)
        return sorted(set(out)), detail

    # RÈGLE 3 : chaque départ jugé contre SON fond local.
    for q in range(len(c)):
        if interdit(q, Lb):
            continue
        fond = _fond(c, q, Lb, p0, lambda x: interdit(x, Lb))
        if fond is None:
            continue
        rho = (float(c[q]) - fond) / max(1.0 - fond, ECART_FOND_MIN)
        ok = bool(c[q] >= PLANCHER_REPRISE and rho >= PROEMINENCE_MIN)
        detail["pics"].append({"p": q, "c": round(float(c[q]), 4),
                               "fond": round(fond, 4), "rho": round(rho, 3),
                               "ok": ok, "pris": False})
    detail["reference"] = reference_du_morceau(detail["pics"])
    # RÈGLE 5 : seuls les maxima locaux (à ±Lb−1) restent candidats. APRÈS la
    # règle 4 (qui compte donc encore les calages décalés parmi « tous les
    # autres pics », comme en v3 : ne pas déplacer Let It Be), AVANT le choix.
    tol_egal = BRUIT_PIC
    if EGALITE_5B and detail["reference"]:
        tol_egal = detail["reference"]["tolerance"]            # RÈGLE 5b
    detail["egalite"] = round(tol_egal, 4)
    if REGLE5:
        passe3 = {x["p"] for x in detail["pics"] if x["ok"] or x.get("sous_reference")}
        maxi = maxima_locaux(c, Lb, passe3, tol=tol_egal)
        for x in detail["pics"]:
            x["maximum_local"] = x["p"] in maxi
            if x["ok"] and not x["maximum_local"]:
                x["ok"] = False
                x["pas_maximum"] = True
    cand = {x["p"]: x for x in detail["pics"] if x["ok"]}
    detail["suspens"] = []
    N = len(c)

    def pose(i):
        cand[i]["pris"] = True
        pris.append((i, i + Lb - 1))
        out.append((i, i + Lb - 1, cand[i]["c"], 0))

    if SUSPENS:
        # Parmi les départs qui passent, le plus de reprises sans
        # recouvrement — groupe par groupe (des départs à moins d'une boucle
        # l'un de l'autre, de proche en proche). Un groupe qui n'a qu'UNE
        # façon de caser le plus de boucles est posé ; s'il en a plusieurs,
        # ce sont des calages rivaux à égalité : EN SUSPENS, aucun n'est
        # acté ici (voir `resoudre_suspens`).
        comps: list[list[int]] = []
        for p in sorted(cand):
            if comps and p - comps[-1][-1] < Lb:
                comps[-1].append(p)
            else:
                comps.append([p])
        for comp in comps:
            sels = _selections(comp, Lb, SUSPENS_MAX_ALT)
            if len(sels) > SUSPENS_MAX_ALT:
                sels = [max(sels, key=lambda s: sum(cand[p]["c"] for p in s))]
            if len(sels) == 1:
                for p in sels[0]:
                    pose(p)
            else:
                detail["suspens"].append({"label": None, "boucle": Lb, "departs": sels,
                                          "tolerance": detail["egalite"],
                                          "alternatives": [[(p, p + Lb - 1, cand[p]["c"], 0)
                                                            for p in s] for s in sels]})
    else:
        # l'ancien choix (v4) : le plus de reprises, à égalité les mieux notées
        best = [0.0] * (N + Lb + 1)
        prend = [False] * (N + Lb + 1)
        for i in range(N - 1, -1, -1):
            best[i] = best[i + 1]
            if i in cand and best[i + Lb] + 1.0 + cand[i]["c"] > best[i]:
                best[i] = best[i + Lb] + 1.0 + cand[i]["c"]
                prend[i] = True
        i = 0
        while i < N:
            if prend[i]:
                pose(i)
                i += Lb
            else:
                i += 1

    # LA FIN DE LA BRIQUE : juste après une suite de boucles retrouvée
    # ailleurs, les mesures qui suivent peuvent rejouer la fin de la brique
    # (San Francisco Street, mes. 54-57). On la prend si elle ressemble plus à
    # la fin de la brique qu'à la boucle, et au moins au plancher.
    def fins(occs, pris_ici) -> list[tuple]:
        if queue is None:
            return []
        t0, t1 = queue
        t = t1 - t0 + 1
        debuts = {a for a, _z, _s, _r in occs}
        neuves = []
        for a, z, _s, _r in sorted(o for o in occs if o[0] > b1 or o[1] < b0):
            q = z + 1
            if q in debuts or q + t - 1 >= n or _chevauche(q, q + t - 1, pris_ici):
                continue
            st = float(np.mean([S[t0 + i, q + i] for i in range(t)]))
            sl = float(np.mean([S[b0 + i, q + i] for i in range(t)]))
            if st >= PLANCHER_REPRISE and st > sl:
                pris_ici.append((q, q + t - 1))
                neuves.append((q, q + t - 1, st, 0))
                debuts.add(q)
        return neuves
    out.extend(fins(out, pris))
    # chaque alternative en suspens reçoit SA fin ; puis toutes leurs mesures
    # sont réservées : une transposée ne s'installe pas dans une zone en suspens
    for g in detail["suspens"]:
        for alt in g["alternatives"]:
            alt.extend(fins(alt, list(pris)))
        for alt in g["alternatives"]:
            pris.extend((o[0], o[1]) for o in alt)

    # LES TRANSPOSÉES : la brique ENTIÈRE tournée (plus de matière qu'une
    # boucle de 4, qui tournée colle trop souvent), même règle 3, et elle doit
    # battre la droite au même endroit de `BONUS_TRANSPOSE`. Une trouvaille est
    # redécoupée en passages comme la brique.
    c_brique = np.asarray(VS._slide(S, b0, L, n))[:max(1, n - L + 1)]
    for k in range(1, 12):
        Sk = V @ np.roll(V, k, axis=1).T
        ck = np.asarray(VS._slide(Sk, b0, L, n))[:len(c_brique)]
        for q in sorted(range(len(ck)), key=lambda x: -ck[x]):
            if ck[q] < PLANCHER_REPRISE + BONUS_TRANSPOSE:
                break
            if _chevauche(q, q + L - 1, pris) or ck[q] < c_brique[q] + BONUS_TRANSPOSE:
                continue
            fond = _fond(ck, q, L, L, lambda x: interdit(x, L))
            if fond is None or (ck[q] - fond) / max(1.0 - fond, ECART_FOND_MIN) < PROEMINENCE_MIN:
                continue
            pris.append((q, q + L - 1))
            for a, z in _passages(q, Lb, T, L):
                out.append((a, z, float(ck[q]), k))

    # EN SUSPENS, tranché ICI avec ce que la brique sait seule (ses propres
    # occurrences sûres) : `out` reste complet pour qui appelle la recherche
    # d'une brique isolée. `occurrences_de_tous` retire ces choix et retranche
    # une fois TOUTES les briques posées (le vrai « cohérence globale »).
    if detail["suspens"]:
        placees = [{"m0": a, "m1": z, "label": None} for a, z, _s, _r in out]
        out.extend(o for _l, o in resoudre_suspens(detail["suspens"], placees, n))
    return sorted(set(out)), detail


def cherche_brique(chart: dict, b0: int, b1: int, audio_dir,
                   label: str = "") -> tuple[list[tuple], dict]:
    """(reprises, diagnostic) pour la brique [b0, b1] — mesures 0-indexées,
    fin incluse.

    Chaque reprise est `(m0, m1, score, rot)`, `rot` étant le nombre de
    demi-tons dont la brique a été TRANSPOSÉE pour être reconnue (0 = telle
    quelle). Le diagnostic dit pourquoi on n'a rien trouvé quand c'est le cas —
    voir `RAISONS` en fin de docstring.

    On fait glisser la brique le long de la matrice de ressemblance chord-tone
    (`sections.similarity`, la même que l'app utilise déjà pour répondre « où
    ce bloc se rejoue-t-il ? ») et on retient les pics sans recouvrement. La
    brique elle-même est TOUJOURS rendue — entière, ou découpée en ses
    passages (règle 2 ci-dessous).

    LES TROIS RÈGLES DE LOUIS (2026-09-30), après la page
    `docs/plots/occurrences_brique_2026-09-30.html` — « le détecteur avec pics
    est bon », il lui faut trois changements. Le cœur est
    `cherche_sur_matrices` ; la page v2 le montre terme par terme
    (`docs/plots/occurrences_brique_v2_2026-09-30.html`).

      1. LA ZONE DONNÉE N'EST PAS UN PIC. Un départ dont la fenêtre CHEVAUCHE
         la brique (même d'une mesure) n'est jamais une reprise, et il n'entre
         pas non plus dans le fond ou le seuil contre lequel on juge les
         autres : ses scores sont la brique comparée à elle-même, décalée.
      2. LA BOUCLE MINIMALE (`boucle_minimale`). Une brique d'au moins 8
         mesures qui est k fois une boucle plus courte (≥ 4 mesures, la
         dernière fois pouvant finir autrement) est découpée en ses passages,
         et c'est la BOUCLE qu'on cherche ailleurs. Sa fin (le dernier passage
         différent) est reprise juste après une suite de boucles retrouvée.
         San Francisco Street : 24 = 5 × 4 + une fin ; la reprise de 16
         mesures, ratée à longueur fixe, est retrouvée en entier.
      3. LA HAUTEUR DU PIC PAR RAPPORT AU RESTE. Un départ est une reprise si
         son score a parcouru au moins `PROEMINENCE_MIN` du chemin entre son
         FOND LOCAL (la médiane de la courbe aux départs décalés de 1 à Lb−1
         mesures, voir `_fond`) et 1,0 — et reste au-dessus du plancher 0,85.
         Plus de seuil global ni de plafond 0,999 (qui recalait Let It Be,
         The Walk, Chain of Fools à 0,99). Stand By Me : ses reprises à 0,90
         sur un fond à 0,65 passent ; un 0,97 sur un plateau à 0,95 non.
         Parmi les départs qui passent : le plus de reprises sans
         recouvrement, à égalité les mieux notées.
      4. LA RÉFÉRENCE DU MORCEAU (`reference_du_morceau`) : un groupe de pics
         nettement sous la meilleure reprise, et plus nombreux qu'elle, est
         refusé (Let It Be : le refrain).
      5. UN DÉPART DOIT DÉPASSER SES VOISINS (`maxima_locaux`, 2026-09-30
         nuit). Parmi les départs qui passent les règles 3–4, n'est candidat
         que celui qui est le plus haut de tous ceux qui ont passé la règle 3
         à ±(Lb − 1) mesures (égalité à 0,01 près : les deux restent). Le
         choix « le plus de reprises » ne choisit donc plus qu'entre des
         maxima locaux. The Walk : en 25–56, 7 boucles décalées de 2 (dont
         les deux de bord à 0,91, sur le C♯m du B) battaient 6 boucles en
         phase à 0,996 ; 27 perd contre 29, 51 contre 49 — elles ne sont plus
         candidates. Posée APRÈS la règle 4, qui compte donc encore les
         calages décalés parmi « tous les autres pics » (inchangée). Ne touche
         ni les passages de la brique (toujours rendus), ni la recherche de sa
         fin (posée juste après une suite de boucles retrouvée : elle suit la
         phase des boucles), ni les transposées (glouton par hauteur sur la
         brique entière), ni l'ancienne loi des briques < 8 mesures (glouton
         par hauteur, qui prend déjà le plus haut d'abord).
      5b. L'ÉGALITÉ = LA TOLÉRANCE DU MORCEAU (2026-09-30, « Oui pour la 5b »),
         et deux calages rivaux qui casent autant de boucles restent EN
         SUSPENS (`info["suspens"]`) : tranchés ici avec ce que la brique sait
         seule, retranchés par `occurrences_de_tous` une fois toutes les
         briques posées (`resoudre_suspens` : mangée, phase, fragments, score).

    Une brique de MOINS DE 8 MESURES ne peut pas montrer une boucle de 4 jouée
    deux fois : elle garde l'ancienne loi (seuil médiane + 1,5 σ du morceau,
    ci-dessous), calculée hors de la zone donnée. La règle 3 sur une brique de
    4 mesures dont les accords alternent la recopiait partout.

    CE QUE LES TROIS RÈGLES NE RÉSOLVENT PAS : les JUMEAUX HARMONIQUES — une
    autre section qui joue les mêmes accords sort comme une reprise (Let It Be :
    le refrain partage la 2e moitié du couplet ; She Will Be Loved : la même
    grille presque tout du long ; les intros qui jouent la boucle). Découper
    en boucles de 4 les rend PLUS fréquents : une boucle de 4 est moins
    spécifique qu'une brique de 8. Un VAMP (Chain of Fools, tout à 0,99)
    n'a pas de fond : on s'abstient. La PHASE d'une boucle à sous-période
    (deux accords qui alternent) est choisie par la couverture, pas par
    l'oreille. Et une reprise plus LONGUE que la brique garde ses mesures en
    trop libres.

    L'ANCIENNE LOI — LE SEUIL RELATIF AU MORCEAU (2026-09-17, gardée pour les
    briques de moins de 8 mesures), et c'était tout l'enjeu. Un seuil absolu de
    0,90 marchait sur Don't Want My Love et cassait Cry Me A River : Louis,
    2026-09-17, « j'ai annoté, l'outil n'a complètement pas respecté ce que
    j'avais écrit ». Mesuré, les deux distributions n'ont rien à voir :

        Don't Want My Love   médiane 0,692 · 1,000 0,996 0,992 0,990 puis
                             CHUTE de 0,135 à 0,855
        Cry Me A River       médiane 0,896 · 1,000 0,963 0,963 0,961 0,952…
                             chutes de 0,000 à 0,009, AUCUNE falaise

    Sur le second, 38 départs sur 80 passaient 0,90 : sa brique était donc
    recopiée sur la moitié du morceau. Le morceau est une ballade dont toutes
    les mesures se ressemblent harmoniquement — le canal ne porte pas
    l'information, et un seuil fixe ne peut pas le savoir.

    D'où : `médiane + 1,5 écart-type`, borné par `PLANCHER_REPRISE` et
    `PLAFOND_REPRISE`. Un morceau SANS CONTRASTE s'abstient tout seul, ce qui
    est la bonne réponse — la brique de Louis reste posée là où il l'a mise et
    le remplissage des trous garde le reste. C'est la même loi que le projet
    applique déjà ailleurs (`soudure._otsu`, un seuil par morceau ; mémoire
    « per-song threshold, and a channel with no contrast abstains »).

    LES REPRISES TRANSPOSÉES (2026-09-19). Le dernier refrain monté d'un ton
    était le trou écrit de cette fonction : la matrice compare des vecteurs de
    classes de hauteur ABSOLUES, donc une reprise modulée n'y ressemble à
    rien. On refait donc glisser la brique sur les onze matrices tournées
    (`V @ roll(V, k).T`, comme `section_tool._rotated_S`), et une reprise
    trouvée là porte son `rot`. Deux garde-fous, tous deux repris de
    `section_tool` où ils ont été payés :

      * `BONUS_TRANSPOSE` — une brique tournée ressemble à plus de choses
        qu'une brique droite, donc elle doit se détacher DAVANTAGE ;
      * `MIN_MESURES_TRANSPOSE` — sous quatre mesures on ne cherche pas du
        tout. Sur Bein Green, un motif de 2 mesures tourné donnait dix
        « reprises transposées », toutes fausses.

    Les reprises DROITES se servent les premières, à tous les coups : une
    reprise à l'identique vaut toujours mieux qu'une reprise modulée au même
    endroit.

    UNE INTRO NE SE REJOUE PAS (Louis, 2026-09-16). Un rôle de
    `JAMAIS_REJOUEES` ne cherche aucune reprise — sur Cry Me A River, « intro »
    était recopié SEPT fois. Cette règle existait dans l'ancienne route et
    avait été perdue en la réécrivant.

    CE QUE ÇA NE RÉSOUT PAS : sur un morceau sans contraste on ne trouve
    RIEN, même quand la section se rejoue vraiment — on préfère rater une
    reprise (le remplissage tient alors le trou) que d'écrire son nom sur la
    moitié du morceau. Le diagnostic le DIT au lieu de rendre une liste vide
    silencieuse, mais il ne le répare pas. Et une reprise qui change de mode
    (le refrain passé en mineur) n'est toujours pas retrouvée : une rotation
    déplace les douze classes ensemble, elle ne change pas les tierces.

    RAISONS possibles dans le diagnostic : `"role"` (une intro ne se rejoue
    pas), `"pas_d_audio"`, `"trop_court"`, `"sans_contraste"` (rien ne dépasse
    le seuil du morceau), ou `None` quand des reprises ont été trouvées.
    """
    import numpy as np

    from harmonia import musx as _musx
    from harmonia.sections import similarity as VS
    seule = [(b0, b1, 1.0, 0)]
    info = {"label": label, "m0": b0, "m1": b1, "n_reprises": 0,
            "seuil": None, "mediane": None, "sigma": None, "raison": None}

    if (label or "").strip().lower() in JAMAIS_REJOUEES:
        info["raison"] = "role"
        return seule, info

    grid = chart.get("barGrid") or []
    n = len(grid) - 1
    L = b1 - b0 + 1
    stem = Path(chart.get("audio_url") or "").stem
    audio = Path(audio_dir) / f"{stem}.m4a"
    if n < 2 or L < 1 or not stem or not audio.exists():
        # Sans audio il n'y a pas de matrice, donc aucune reprise à chercher.
        # Ce n'est pas un repli muet — on le DIT — et ça ne trahit pas son
        # geste : ça se contente de ne rien ajouter.
        logger.info("cherche_brique : pas d'audio pour %s, la brique reste "
                    "seule", stem or "?")
        info["raison"] = "pas_d_audio"
        return seule, info

    triad = _musx.frame_posteriors(audio)[0]
    V = VS.harmonic_vectors(triad, grid)
    S = VS.ssm(triad, grid)
    if n - L + 1 < 3:
        info["raison"] = "trop_court"
        return seule, info
    out, d = cherche_sur_matrices(np.asarray(S), np.asarray(V), b0, b1)
    info.update(seuil=d["seuil"], mediane=d["mediane"], sigma=d["sigma"],
                boucle=d["boucle"], sous_periode=d["sous_periode"],
                mode=d["mode"], suspens=d.get("suspens") or [])
    if d.get("raison"):
        info["raison"] = d["raison"]
        return seule, info
    info["n_reprises"] = sum(1 for o in out if o[1] < b0 or o[0] > b1)
    if info["n_reprises"] == 0:
        info["raison"] = "sans_contraste"
    tournees = sum(1 for o in out if o[3])
    logger.info("cherche_brique %s [%d-%d] « %s » : boucle %d/%d mesures, "
                "loi %s → %d occurrence(s) dont %d hors de la brique, %d "
                "transposée(s)", stem, b0 + 1, b1 + 1, label, d["boucle"], L,
                d["mode"], len(out), info["n_reprises"], tournees)
    return out, info


def occurrences_de_tous(chart: dict, traits: list[dict],
                        audio_dir) -> tuple[list[dict], list[dict], list[dict]]:
    """(occurrences, traits écartés, diagnostics) — chaque trait de Louis, plus
    ses reprises, sans que deux briques se marchent dessus.

    Les traits sont traités dans l'ordre où il les a posés : les reprises
    d'une brique ne peuvent pas recouvrir un trait déjà placé, ni les reprises
    d'une brique précédente. Le premier arrivé garde sa place, ce qui est la
    seule règle qui ne trahisse pas son intention.

    OPTION (c) (Louis, 2026-09-30, `MANGE_NETTEMENT`) : toutes les briques
    sont cherchées AVANT de poser quoi que ce soit, et une reprise sûre ne
    prend la zone d'un calage EN SUSPENS d'une autre brique que si son score
    y dépasse celui de chaque calage recouvert de plus que la tolérance du
    groupe (la tolérance du morceau de la règle 4, celle de la 5b). Sinon
    elle est retenue, et revient seulement si le suspens est tranché
    ailleurs. Les traits de Louis gagnent toujours. Le sort d'une zone en
    suspens ne dépend donc pas de l'ordre des briques (Sunny A : 8 → 16
    mesures justes). CE QUE ÇA NE RÉSOUT PAS : un jumeau EN SUSPENS plus haut
    qu'une vraie reprise la bat (X-yIEMduRXk : C prend 27–35 dans le 2e A) ;
    entre deux reprises SÛRES rivales, le premier trait dans les mesures
    garde la priorité.

    Chaque occurrence porte `rot` : les demi-tons dont la brique a été
    transposée pour être reconnue là (0 = telle quelle).

    Le troisième retour est la liste des diagnostics de `cherche_brique`, un
    par trait gardé — DE QUOI DIRE À L'ÉCRAN pourquoi une brique n'a pas
    trouvé de reprise, au lieu de laisser croire à une panne. Louis,
    2026-09-19 : un morceau sans contraste doit se DIRE, pas se taire.
    """
    from harmonia.soudure import traits_propres
    n = len((chart.get("barGrid") or [])) - 1
    gardes, ecartes = traits_propres(traits, n)
    pris: list[tuple] = []
    out: list[dict] = []
    diagnostics: list[dict] = []
    en_suspens: list[dict] = []
    resultats = []
    for i, (b0, b1, label) in enumerate(gardes):
        reprises, info = cherche_brique(chart, b0, b1, audio_dir, label=label)
        diagnostics.append(info)
        # LES CANDIDATS EN SUSPENS (règle 5b, 2026-09-30) : `cherche_brique`
        # les a tranchés avec ce que la brique sait seule ; on retire ce
        # choix, et on les tranche plus bas, une fois TOUTES les briques
        # posées (« on voit ce qui reste quand les autres sections ont mangé
        # ce qu'elles ont à manger »).
        for g in (info or {}).get("suspens") or []:
            g = dict(g, label=label, brique=i, choix_seul=g.get("choix"),
                     raison_seul=g.get("raison"))
            en_suspens.append(g)
            alts = {o for alt in g["alternatives"] for o in alt}
            reprises = [o for o in reprises if tuple(o) not in alts]
        resultats.append((b0, b1, label, reprises))

    def touche(x, y, c, d):
        return not (y < c or d < x)

    # OPTION (c) (Louis, 2026-09-30) : les zones tenues par un calage en
    # suspens, TOUTES connues avant qu'aucune reprise sûre ne soit posée —
    # c'est ce qui rend la règle indépendante de l'ordre des briques. Un
    # calage qui recouvre un trait de Louis ne tient rien : le trait le
    # mangera de toute façon.
    tenues = [(o[0], o[1], o[2], g.get("tolerance", BRUIT_PIC), g["brique"])
              for g in en_suspens for alt in g["alternatives"] for o in alt
              if not any(touche(o[0], o[1], c, d) for c, d, _l in gardes)] \
        if MANGE_NETTEMENT else []
    retenues: list[tuple] = []
    for i, (b0, b1, label, reprises) in enumerate(resultats):
        # UN TRAIT DE LOUIS EST SOUVERAIN, AVANT OU APRÈS (2026-09-30, The
        # Walk) : les traits sont traités dans l'ordre des mesures, donc une
        # reprise de A pouvait recouvrir le trait B posé plus loin — et tous
        # les passages de B étaient ensuite refusés, B disparaissait. Une
        # reprise ne recouvre aucun AUTRE trait ; ses propres passages, si.
        autres = [(c, d) for c, d, _l in gardes if (c, d) != (b0, b1)]
        for x, y, sc, rot in reprises:
            if any(touche(x, y, c, d) for c, d in pris + autres):
                continue
            trace = b0 <= x and y <= b1
            # (c) : une reprise sûre ne prend la zone d'un calage en suspens
            # d'une AUTRE brique que si elle y est NETTEMENT plus haute que
            # chacun des calages qu'elle recouvre. Sinon elle est retenue, et
            # ne revient que si le suspens est finalement tranché ailleurs.
            if not trace and any(touche(x, y, t0, t1) and sc <= ts + tol
                                 for t0, t1, ts, tol, j in tenues if j != i):
                retenues.append((i, x, y, sc, rot, label))
                continue
            pris.append((x, y))
            out.append({"m0": x, "m1": y, "label": label,
                        "score": round(sc, 3), "rot": int(rot),
                        # ses passages (règle 2 : sa brique découpée en
                        # boucles) restent SES traits
                        "trace": trace})
    if en_suspens:
        placees = [{"m0": o["m0"], "m1": o["m1"], "label": o["label"]} for o in out] \
            + [{"m0": c, "m1": d, "label": lab} for c, d, lab in gardes]
        for label, (x, y, sc, rot) in resoudre_suspens(en_suspens, placees, n):
            if any(touche(x, y, c, d) for c, d in pris):
                continue
            pris.append((x, y))
            # `suspens` : une PROPOSITION tranchée par la cohérence du
            # morceau, jamais une contrainte — la route la rend « ressemble »,
            # et elle est donc retranchée à chaque « Valider ».
            out.append({"m0": x, "m1": y, "label": label, "score": round(sc, 3),
                        "rot": int(rot), "trace": False, "suspens": True})
        for info in diagnostics:
            info["suspens_global"] = [g for g in en_suspens if g["label"] == info.get("label")]
    # (c) : les reprises retenues dont la zone est restée libre (le suspens
    # a choisi un autre calage) reviennent — dans le même ordre qu'au premier
    # passage (celui des traits dans les mesures, pas celui où Louis les a
    # tracés) : entre deux reprises SÛRES rivales, rien ne change. Au
    # score, Sunny donnait 27–34 à un jumeau de D (0,959) plutôt qu'au B
    # juste (0,935) que v5 posait.
    for _i, x, y, sc, rot, label in sorted(retenues):
        if any(touche(x, y, c, d) for c, d in pris):
            continue
        pris.append((x, y))
        out.append({"m0": x, "m1": y, "label": label, "score": round(sc, 3),
                    "rot": int(rot), "trace": False})
    return sorted(out, key=lambda o: (o["m0"], o["m1"], o["label"])), ecartes, diagnostics


def remplir_par_algo(chart: dict, occ: list[dict], audio_dir) -> list[dict]:
    """Les trous, remplis par l'algorithme des quatre mots.

    Toutes les occurrences entrent comme unités GELÉES — pas seulement celles
    que Louis a tracées. L'agglomération ne travaille donc que dans les trous
    et ne peut plus donner le nom d'une brique à un bloc qu'elle a fabriqué
    elle-même.

    L'INTRO DE TÊTE (2026-09-30) : la zone libre avant la première section
    posée sort en UN bloc « intro » (source "algo" — recalculé à chaque
    Valider, jamais figé en contrainte), N.C. ou pas : une tête muette est
    une intro que chart.js cache en lecture, une tête à accords s'imprime.

    CE QUE ÇA NE RÉSOUT PAS : l'algorithme ne sait produire que des lettres ;
    seule la tête reçoit un nom de rôle. Un trou au milieu garde une lettre
    (même s'il est muet), la queue n'est jamais « outro ». Si Louis a posé sa
    propre « intro » plus loin que la mesure 0, ce qui la précède garde une
    lettre (pas de seconde intro). Une tête à la fois muette ET à accords
    reste UNE intro imprimée avec ses N.C. de tête (la pipeline, elle, la
    coupe en deux, `pipeline._one_section`). Si l'algo avait donné à un bloc
    de tête la même lettre qu'un bloc plus loin, ce dernier garde sa lettre.
    """
    from harmonia.phrases4 import grouper_restes, merges4, nommer, phrases
    from harmonia.soudure import mot_sur_traits

    traits = [(o["m0"], o["m1"]) for o in occ]
    bornes, mot, _src = mot_sur_traits(chart, traits, audio_dir=audio_dir)
    idx = {b: j for j, b in enumerate(bornes)}
    fixes, par_jeton = [], {}
    for o in occ:
        j0, j1 = idx.get(o["m0"]), idx.get(o["m1"] + 1)
        if j0 is None or j1 is None:
            continue
        fixes.append((j0, j1 - 1))
        par_jeton[(j0, j1)] = o["label"]
    depart, j = [], 0
    for j0, j1 in fixes:
        while j < j0:
            depart.append((j, j + 1, mot[j])); j += 1
        depart.append((j0, j1 + 1, mot[j0:j1 + 1])); j = j1 + 1
    while j < len(mot):
        depart.append((j, j + 1, mot[j])); j += 1
    geles = {(j0, j1 + 1) for j0, j1 in fixes}

    _secs, info = phrases(mot, depart=depart, geles=geles)
    steps = merges4(mot, cible=info["cible"], depart=depart, geles=geles)
    blocs = grouper_restes(nommer(steps[-1]["jetons"], info["cible"],
                                  geles=geles), mot, info["cible"])
    siennes = {L.upper() for L in par_jeton.values()}
    libres = [c for c in ALPHABET if c not in siennes]
    # L'INTRO DE TÊTE (Louis, 2026-09-30 : « quand il manque une section avant
    # un A, c'est une intro, pas un C »). Tout ce que l'algo a découpé AVANT
    # la première section posée devient UN bloc « intro » — un seul, même si
    # l'algo y voyait deux blocs, sinon deux intros de même longueur seraient
    # repliées ensemble par `sections_pour_chart` (qui groupe sur (label,
    # longueur)). Il ne consomme aucune lettre : le premier bloc neuf d'après
    # prend toujours la première lettre libre. Pas de règle si rien n'est
    # posé (pas de « première section »), ni si Louis a déjà posé une intro.
    tete = min((o["m0"] for o in occ), default=0)
    if any(str(o["label"]).lower() == "intro" for o in occ):
        tete = 0
    renom, k, out = {}, 0, []
    for b in blocs:
        sien = par_jeton.get((b["j0"], b["j1"]))
        if sien is None and bornes[b["j1"]] - 1 < tete:
            if out and out[-1]["label"] == "intro":
                out[-1]["m1"] = bornes[b["j1"]] - 1
            else:
                out.append({"m0": bornes[b["j0"]], "m1": bornes[b["j1"]] - 1,
                            "label": "intro", "source": "algo", "mot": b["type"]})
            continue
        if sien is not None:
            nom, source = sien, "brique"
        else:
            if b["label"] not in renom:
                renom[b["label"]] = libres[k % len(libres)] if libres else b["label"]
                k += 1
            nom = renom[b["label"]] + ("′" if b["prime"] else "")
            source = "algo"
        out.append({"m0": bornes[b["j0"]], "m1": bornes[b["j1"]] - 1,
                    "label": nom, "source": source, "mot": b["type"]})
    return out


def remplir_par_songformer(chart: dict, occ: list[dict],
                           auto: list[dict]) -> list[dict]:
    """Les trous, remplis par ce que le modèle avait trouvé.

    Les mesures couvertes par une brique de Louis gardent SON nom, et deux
    occurrences voisines de la même brique restent deux blocs — le morceau les
    joue deux fois. Partout ailleurs on garde les SEGMENTS de SongFormer tels
    qu'il les a posés, rognés autour des briques.

    On ne repasse PAS par un tableau « une étiquette par mesure » : ça fondait
    deux segments voisins portant la même lettre en un seul bloc, et un
    morceau qui joue A puis A ressortait avec un A de seize mesures. Les
    frontières du modèle sont une information, pas un effet de bord.

    `auto` est la liste des segments du modèle, chacun avec ses `barRanges` —
    la réponse que la production jetait avant le 2026-09-17.
    """
    n = len((chart.get("barGrid") or [])) - 1
    pris = [False] * n
    out = []
    for o in occ:
        a, b = max(0, o["m0"]), min(n - 1, o["m1"])
        if b < a:
            continue
        out.append({"m0": a, "m1": b, "label": o["label"], "source": "brique",
                    "mot": ""})
        for i in range(a, b + 1):
            pris[i] = True
    for sec in auto or []:
        lab = sec.get("label") or "?"
        for a, b in (sec.get("barRanges") or []):
            i = max(0, a)
            fin = min(n - 1, b)
            while i <= fin:
                if pris[i]:
                    i += 1
                    continue
                j = i
                while j + 1 <= fin and not pris[j + 1]:
                    j += 1
                out.append({"m0": i, "m1": j, "label": lab,
                            "source": "songformer", "mot": ""})
                i = j + 1
    return sorted(out, key=lambda b: b["m0"])
