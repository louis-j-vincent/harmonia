"""harmonia_min/soudure.py — le mot d'un chart, pour le jeu de soudure.

Louis, 2026-08-13 : « mets le moi comme une option sur chaque chanson dans le
chart, car c'est vraiment une interface hyper pratique. »

La page (`docs/plots/soudure.html`) attend une chanson sous forme de bande de
jetons : une lettre par bi-mesure, deux jetons de même lettre étant le même
endroit du morceau. Ce module fabrique cette bande à partir du chart que l'app
sert déjà — donc sans audio, sans analyse, et surtout **sans second détecteur
de similarité**.

C'est la règle que `section_tool.py` a posée et qui vaut ici mot pour mot :
écrire un second scorer de similarité créerait deux vérités divergentes pour la
même question, et l'outil pourrait contredire le chart qu'il annote. On ne
compare donc rien : deux bi-mesures portent la même lettre si et seulement si
elles jouent **exactement la même basse**, mesure par mesure. C'est de
l'égalité, pas de la ressemblance ; il n'y a ni seuil, ni réglage, ni modèle.

POURQUOI LA BASSE, ET PAR MESURE. Le mot du projet est passé aux accords à la
BASSE le 2026-08-12 (commit c3802b1) ; on suit. Le grain est la mesure, pas le
temps, et c'est mesuré : sur les 45 charts servis, la part de bi-mesures qui
appartiennent à une signature vue au moins deux fois est de

    basse par mesure   médiane 94 %   min 60 %    <- retenu
    basse par temps    médiane 62 %   min 13 %
    accord par temps   médiane 49 %   min  5 %

Au grain du temps, un morceau sur cinq n'a presque rien à souder — le jeu
n'aurait pas de matière. Au grain de la mesure, il en a partout.

CE QUE ÇA NE RÉSOUT PAS. L'égalité stricte SOUS-GROUPE : deux passages qui se
ressemblent sans être identiques reçoivent deux lettres différentes, et la
soudure « partout à la fois » ne les attrapera pas ensemble. C'est le sens de
la préférence de Louis (« under-fold, never over-fold ») et le jeu permet de
les souder à la main, mais il faut le savoir : ce mot-ci est plus bavard que
celui de `vote_fill.fill`, qui lui groupe par similarité de SSM.

Et sur un morceau à un seul accord (Chain of Fools), la basse ne dit rien : le
mot devient une seule lettre répétée. La page reste jouable, mais elle n'a plus
de structure à montrer — c'est honnête, ce n'est pas utile.
"""
from __future__ import annotations

import itertools
import logging

log = logging.getLogger(__name__)

# 62 symboles, pas 26. Le maximum observé sur les 45 charts est 29 bi-mesures
# distinctes (H D3Vffhvs4, 106 mesures) : à 26 lettres, trois morceaux voyaient
# des bi-mesures DIFFÉRENTES recevoir la même lettre, donc se faire souder
# ensemble comme si c'était le même endroit. La page ne fait qu'identifier des
# symboles, elle ne les lit pas — le jeu de caractères peut être large.
LETTRES = ("abcdefghijklmnopqrstuvwxyz"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "0123456789")


def _basse_par_mesure(chart: dict) -> list[tuple]:
    """Une signature par mesure : (classe de hauteur de la basse, silence).

    La basse d'une mesure est celle de l'accord qui y dure le plus longtemps.
    `bass` vaut -1 quand le chart n'a pas d'inversion : la fondamentale est
    alors la basse qui sonne, ce qui est la définition retenue par le projet
    (`corpus_schema.sounding_bass_pc`).
    """
    grid = chart.get("barGrid") or []
    accords = (chart.get("prompter") or {}).get("chords") or []
    out = []
    for b in range(len(grid) - 1):
        t0, t1 = grid[b], grid[b + 1]
        duree: dict = {}
        for c in accords:
            chevauche = min(t1, c.get("t1", 0.0)) - max(t0, c.get("t0", 0.0))
            if chevauche <= 0:
                continue
            bass = c.get("bass", -1)
            if bass is None or bass < 0:
                bass = c.get("root", -1)
            cle = (int(bass), bool(c.get("nc")))
            duree[cle] = duree.get(cle, 0.0) + chevauche
        out.append(max(duree, key=duree.get) if duree else (-1, True))
    return out


def _mesure1(chart: dict) -> int:
    """L'index de la mesure que Louis a marquée « mesure 1 », 0 s'il n'a rien
    marqué.

    `chart["bar1"]` est sa marque EN SECONDES — la seule forme qui survive à
    un recalcul (voir `pipeline._model`). La grille de mesures a été rebâtie
    autour d'elle, donc une borne de `barGrid` tombe dessus : on la retrouve
    en cherchant la plus proche, comme la pipeline recale la marque sur le
    temps le plus proche.
    """
    grid = chart.get("barGrid") or []
    n = len(grid) - 1
    try:
        t = float(chart["bar1"])
    except (KeyError, TypeError, ValueError):
        return 0
    if n < 2:
        return 0
    return min(range(n), key=lambda b: abs(float(grid[b]) - t))


def _depart_valide(depart: int, n_mesures: int) -> int:
    """La marque, ramenée à ce qui laisse encore une bande derrière elle.

    Une marque posée à trois mesures de la fin donnerait une bande de deux
    plaques : on préfère le treillis d'origine à une bande qui ne montre plus
    rien. `4` est le minimum que `grille_et_mot` exige déjà.
    """
    try:
        depart = int(depart)
    except (TypeError, ValueError):
        return 0
    return depart if 0 < depart <= n_mesures - 4 else 0


def _tete(depart: int) -> list[int]:
    """Les bornes des jetons AVANT la mesure 1 — l'intro, à son propre grain.

    Elle garde la bi-mesure, et c'est le PREMIER jeton (donc le plus loin de
    la structure) qui absorbe la mesure orpheline d'une intro de longueur
    impaire. L'inverse — l'orpheline juste avant la marque — redécalerait
    d'une mesure tout ce que la marque vient de caler.
    """
    if depart <= 0:
        return []
    if depart % 2 == 0:
        return list(range(0, depart, 2))
    return [0] + list(range(3, depart, 2))


def _jetons(n_mesures: int, depart: int = 0) -> list[int]:
    """Les bornes des bi-mesures. La mesure orpheline d'un morceau impair est
    rattachée au dernier jeton plutôt que jetée — sinon la fin du morceau
    disparaît de la bande sans que rien ne le dise.

    `depart` (2026-08-15) : la mesure 1 de Louis. Le treillis part de LÀ, pas
    de la barre 0 — sinon une plaque sur deux enjambe la frontière qu'il vient
    de poser, et aucune section ne peut commencer sur sa mesure 1.
    """
    if n_mesures < 2:
        return [0, n_mesures]
    depart = _depart_valide(depart, n_mesures)
    bornes = _tete(depart) + list(range(depart, n_mesures - 1, 2))
    bornes.append(n_mesures)
    return bornes


def mot_du_chart(chart: dict) -> dict | None:
    """{mot, jetons, n_mesures, temps_par_mesure, t0} — ou None si le chart
    n'a pas de quoi faire une bande."""
    grid = chart.get("barGrid") or []
    n = len(grid) - 1
    if n < 2:
        return None
    sig = _basse_par_mesure(chart)
    bornes = _jetons(n, _mesure1(chart))

    vus: dict = {}
    lettres = []
    for j in range(len(bornes) - 1):
        cle = tuple(sig[m] for m in range(bornes[j], bornes[j + 1]))
        if cle not in vus:
            if len(vus) < len(LETTRES):
                vus[cle] = LETTRES[len(vus)]
            else:
                # Au-delà de 62, deux bi-mesures DIFFÉRENTES se diraient
                # identiques et se feraient souder ensemble. Jamais vu (29 au
                # maximum), mais si ça arrive on le dit plutôt que de mentir.
                vus[cle] = LETTRES[-1]
                log.warning("soudure: plus de %d bi-mesures distinctes — les "
                            "dernières partagent un symbole", len(LETTRES))
        lettres.append(vus[cle])

    return {
        "n_mesures": n,
        "temps_par_mesure": [round(grid[i + 1] - grid[i], 4) for i in range(n)],
        "mot": "".join(lettres),
        "jetons": bornes,
        "t0": round(grid[0], 4),
    }


def _otsu(v, lo=0.30, hi=0.999, n=200, defaut=0.90) -> float:
    """Le seuil qui sépare le mieux la distribution du morceau en deux paquets.

    Repris tel quel de `scripts/vote_fill.otsu`, avec sa raison d'être :
    mesuré le 2026-08-12, la médiane des ressemblances entre bi-mesures vaut
    0,33 sur Sunny, 0,90 sur Let It Be et 0,99 sur Blue Lights. Un seuil fixe
    tombe donc au milieu de la distribution d'un morceau et au ras d'un autre.
    Otsu lit la FORME de la distribution, pas son niveau.
    """
    import numpy as np
    v = np.asarray(v, float)
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 8:
        return defaut
    best, seuil = -1.0, defaut
    for t in np.linspace(v.min() + 1e-6, v.max() - 1e-6, n):
        a, c = v[v <= t], v[v > t]
        if a.size < 2 or c.size < 2:
            continue
        w = a.size * c.size * (a.mean() - c.mean()) ** 2
        if w > best:
            best, seuil = w, float(t)
    return seuil


def _matrice_bimesures(S, bornes):
    """B[j,k] normalisée : à quel point la bi-mesure j ressemble à la k."""
    import numpy as np
    from harmonia_min import voice_sections as VS
    J = len(bornes) - 1
    B = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            L = min(bornes[j + 1] - bornes[j], bornes[k + 1] - bornes[k])
            B[j, k] = VS._diag(S, bornes[j], bornes[k], L)
    d = np.sqrt(np.clip(np.diag(B), 1e-9, None))
    return B / np.outer(d, d)


def _grille(n: int, coutures: dict) -> list[int]:
    """Les bornes des jetons, avec des coutures à des endroits donnés.

    `coutures` : {mesure de départ du jeton -> sa largeur, 1 ou 3}. Partout
    ailleurs un jeton fait deux mesures. Un jeton de largeur impaire fait
    basculer la parité de tout ce qui suit — c'est lui, la mesure en trop ou
    la mesure manquante.

    On applique une couture au premier jeton qui ATTEINT OU DÉPASSE sa
    position. Le faire seulement sur une égalité exacte était un piège :
    la recherche travaille sur des positions quelconques, la grille avance de
    deux en deux, et une couture demandée à une mesure impaire n'était jamais
    posée — la grille sortait silencieusement différente de celle qu'on venait
    de noter (constaté sur Lost Without U : deux coutures trouvées, une seule
    appliquée, et 66 % au lieu de 100 %).
    """
    cs = sorted(coutures.items())
    bornes, p, i = [], 0, 0
    while p < n - 1:
        bornes.append(p)
        while i < len(cs) and cs[i][0] < p:
            i += 1                                   # couture déjà dépassée
        if i < len(cs) and p >= cs[i][0] - 1:
            p += cs[i][1]
            i += 1
        else:
            p += 2
    bornes.append(n)
    return bornes


def _coutures_de(bornes: list[int]) -> list[int]:
    """Les jetons qui ne font pas deux mesures — hors le dernier, qui fait
    trois par construction dans un morceau de longueur impaire."""
    return [j for j in range(len(bornes) - 2)
            if (bornes[j + 1] - bornes[j]) != 2]


#: Ce qu'une couture doit faire GAGNER pour être retenue : cinq points de
#: jumelles exprimables. C'est le « il doit se payer » de
#: `vote_fill.phases()`. Sans ce péage, une boucle de quatre mesures se laisse
#: décaler d'une mesure sans rien perdre, et on décale au hasard.
MARGE_COUTURE = 0.05

#: À partir d'où deux passages de quatre mesures sont « manifestement les
#: mêmes ». Haut exprès : le critère ne doit s'appuyer que sur des reprises
#: que personne ne discute.
SEUIL_JUMELLE = 0.95

#: Combien de positions entrent dans la recherche exhaustive. 24 tient
#: largement (les triplets font 2024 combinaisons x 8 largeurs), et le champ
#: est trie par ce qu'une coupure y rapporte, donc les vraies coutures y sont.
CANDIDATS = 24

#: Au plus trois coutures CHERCHÉES. La grille construite peut porter un jeton
#: impair de plus, quand la fin du morceau n'en laisse pas le choix — c'est le
#: cas d'un chart sur 46 (Lghgu8Gqz9U). La grille reste valide : largeurs 1, 2
#: ou 3, couverture complète, vérifié sur les 46.
#: Au plus trois coutures. Une par mesure en trop; au-delà on n'explique plus
#: un accident de notation, on plie la grille jusqu'a ce que tout colle.
MAX_COUTURES = 3


def _boucle(S, n: int) -> int | None:
    """La période de la boucle du morceau, en mesures — `abababab` rend 2.

    Louis, 2026-08-14 : « il faut donc un algo d'identification de
    patterns/boucles, ici c'est du abababababab par exemple ». On la lit sur
    la matrice : pour chaque décalage p, la ressemblance moyenne d'une mesure
    avec celle qui est p plus loin. La boucle est le p qui tient le mieux.
    On la cherche entre 2 et 16 mesures — en deçà c'est la mesure elle-même,
    au-delà c'est une section, pas une boucle.
    """
    import numpy as np
    if n < 12:
        return None
    scores = {}
    for p in range(2, min(17, n // 3)):
        d = np.array([S[b, b + p] for b in range(n - p)])
        scores[p] = float(np.mean(d))
    if not scores:
        return None
    return max(scores, key=scores.get)


def _merite_du_trou(S, n: int, p: int):
    """Pour chaque mesure, à quel point elle MÉRITE d'être le trou.

    Louis : « le second critère, celui qui mérite d'être le trou, est
    exactement le bon outil ». Une mesure insérée est celle qui ne ressemble
    PAS à ce que la boucle annonce à sa place : on la note par son désaccord
    avec la mesure d'une période avant et d'une période après. Une mesure bien
    dans la boucle a deux voisines de période qui lui ressemblent ; la mesure
    en trop, non — c'est elle qui décale tout ce qui suit.
    """
    import numpy as np
    m = np.zeros(n)
    for b in range(n):
        vus = []
        if b - p >= 0:
            vus.append(S[b, b - p])
        if b + p < n:
            vus.append(S[b, b + p])
        m[b] = 1.0 - (float(np.mean(vus)) if vus else 1.0)
    return m


def _jumelles(S, n: int):
    """(P, Q, W) — les paires de mesures manifestement jumelles, et leur poids.

    On compare des blocs de QUATRE mesures, pas de deux : à deux mesures, dans
    un morceau en boucle, tout ressemble à tout et le critère n'arbitre plus
    rien. On exige aussi huit mesures d'écart, pour ne pas compter comme
    « reprise » le simple fait qu'une boucle se répète immédiatement.
    """
    import numpy as np
    L = 4
    m = n - L
    if m < 10:
        return np.array([]), np.array([]), np.array([])
    D = np.zeros((m, m))
    for i in range(L):                       # moyenne des 4 diagonales
        D += S[i:i + m, i:i + m]
    D /= L
    idx = np.triu_indices(m, k=8)
    ok = D[idx] > SEUIL_JUMELLE
    return idx[0][ok], idx[1][ok], D[idx][ok]


def _mot_depuis(B) -> str:
    """Les lettres, depuis la matrice de ressemblance entre bi-mesures."""
    import numpy as np
    J = B.shape[0]
    thr = _otsu(B[~np.eye(J, dtype=bool)])
    used, lab, k = set(), [-1] * J, 0
    for j in sorted(range(J), key=lambda j: -(B[j] >= thr).sum()):
        if j in used:
            continue
        mem = [j]
        for q in range(J):
            if q in used or q == j:
                continue
            if float(np.mean([B[q, x] for x in mem])) >= thr:
                mem.append(q)
        for q in mem:
            lab[q] = k
        used.update(mem)
        k += 1
    ren, k = {}, 0
    for j in range(J):
        if lab[j] not in ren:
            ren[lab[j]] = k
            k += 1
    return "".join(LETTRES[ren[lab[j]] % len(LETTRES)] for j in range(J))


def _meilleure_grille(S, n: int) -> tuple[list[int], dict]:
    """La grille de bi-mesures qui fait que le plus de bi-mesures retrouvent
    leur jumelle — rigide par défaut, avec une couture si elle la mérite.

    LE PROBLÈME. Louis, 2026-08-14, sur Lost Without U : « il y a un trou
    d'une mesure, et donc ça décale toutes les bi-barres qui sont pourtant les
    mêmes ». Mesuré sur ce morceau (89 mesures, nombre impair) : 44 mesures
    sur 80 ont leur reprise la plus nette à une distance IMPAIRE — mesure 5 ↔
    26, mesure 20 ↔ 41, à 0,97 et 0,99 de ressemblance. Une grille figée sur
    (0,1)(2,3)(4,5)… ne franchit jamais un écart impair : la même musique
    reçoit deux lettres selon qu'elle tombe avant ou après le trou, et le
    compagnon propose alors des paires qui n'ont pas de sens.

    LA RÈGLE EST CELLE DE `scripts/vote_fill.phases()`, que Louis a fait
    écrire le 2026-08-12 sur She Will Be Loved (« la barre en extra à la
    mesure 33 fait qu'on a un décalage sur la parité »). Ses deux leçons,
    payées là-bas, valent ici mot pour mot :

      * PAS de ±1 partout — essayé, ça DÉTRUIT la structure : dans une boucle
        de 4 mesures, une fenêtre décalée d'une mesure ressemble encore à
        tout, et Blue Lights perdait ses B (`aaaaaabc` → `aaaaaaaa`) ;
      * le décalage doit être LOCAL et il doit SE PAYER.

    LA MONNAIE, ET UN ESSAI RATÉ. La recherche compte des voix de pics, l'app
    n'en a pas. Premier essai : la LONGUEUR DE DESCRIPTION du mot, la mesure
    que Louis a choisie le 2026-08-13 pour l'arbitrage voisin du groupage.
    Mesuré, elle ne marche pas ici : elle pose une couture sur **28 charts sur
    45** — un trou d'une mesure n'existe pas dans deux tiers des morceaux — et
    elle CASSE Be My Baby (100 % → 0 % de jumelles bien nommées). Un mot un peu
    plus court n'est pas un mot plus juste.

    Ce qui marche est plus bête et plus direct : on note une grille sur le
    SYMPTÔME. Deux mesures manifestement jumelles ne peuvent porter la même
    lettre que si elles occupent la même position dans leur jeton ; on compte
    donc la part des jumelles que la grille sait exprimer. Une couture qui
    recolle des jumelles séparées par le trou monte ; une couture posée au
    hasard casse des jumelles déjà alignées et descend. Résultat sur les 46
    charts : 12 coutures, aucune régression.
    """
    import numpy as np
    P, Q, W = _jumelles(S, n)
    if not len(P):
        return _jetons(n), {"couture": None, "jumelles": 0}

    def note(bornes):
        """La part des jumelles que cette grille sait EXPRIMER.

        Deux mesures jumelles ne peuvent porter la même lettre que si elles
        occupent la même position dans leur jeton. C'est tout le critère : on
        ne demande pas à la grille d'être courte à décrire, on lui demande de
        ne pas rendre invisibles des reprises qui existent.
        """
        if len(bornes) < 4:
            return -1.0
        pos = np.zeros(n, dtype=int)
        for j in range(len(bornes) - 1):
            for b in range(bornes[j], min(bornes[j + 1], n)):
                pos[b] = b - bornes[j]
        return float(np.sum(W * (pos[P] == pos[Q])) / np.sum(W))

    # LES COUTURES SE CHERCHENT ENSEMBLE, PAS UNE PAR UNE. Louis, 2026-08-14 :
    # « sur Lost Without U il y a 2 trous dans la chanson je crois, et les deux
    # sont des petits trous ». Il a raison, et la pose gloutonne ne pouvait pas
    # les voir : mesuré sur ce morceau, la meilleure couture SEULE est à la
    # mesure 54 (48 % → 66 %), alors que la meilleure PAIRE est (20, 53) et
    # atteint 100 %. La paire optimale ne contient pas le meilleur élément
    # seul — deux trous qui ne servent qu'ensemble, chacun ne payant rien tout
    # seul. Un glouton s'arrête donc à 66 % en croyant avoir fini.
    #
    # LA BONNE REPRÉSENTATION. Une couture ne fait qu'une chose : inverser la
    # parité relative des paires qu'elle ENJAMBE. Une paire (p, q) est donc
    # exprimable si le nombre de coutures dans ]p, q] a la même parité que
    # l'écart q − p. Tout le problème tient dans ce booléen, ce qui permet de
    # noter un jeu de coutures sans jamais reconstruire la grille : on peut
    # alors essayer TOUTES les paires de positions, ce qui est indispensable
    # puisque les bonnes ne se voient pas une par une.
    base = note(_grille(n, {}))
    sw = float(np.sum(W))
    enjambe = (P[None, :] < np.arange(n)[:, None]) & \
              (np.arange(n)[:, None] <= Q[None, :])       # (position, paire)
    besoin = ((Q - P) % 2).astype(bool)

    def score(cuts):
        x = np.zeros(len(P), bool)
        for c in cuts:
            x ^= enjambe[c]
        return float(np.sum(W[x == besoin]) / sw)

    # LE SECOND CRITÈRE, ET POURQUOI IL FAUT UN SECOND CRITÈRE. Le score de
    # parité est PLAT sur une fenêtre : mesuré sur Lost Without U, cinq paires
    # de positions différentes atteignent toutes 100 % — la première couture
    # peut tomber en 21, 22 ou 23, la seconde en 54, 55 ou 56. Le critère dit
    # QU'IL Y A un trou dans cette fenêtre, jamais LEQUEL, et prendre le
    # premier venu posait la mesure avalée à côté du vrai trou : Louis
    # l'entendait (« sur robin thicke on détecte mal les trous d'une mesure »).
    #
    # Sa réponse, 2026-08-14 : « une fois que tu as identifié un pattern, tu
    # vois le premier endroit qui casse cette boucle […] le second critère,
    # celui qui mérite d'être le trou, est exactement le bon outil ». On
    # départage donc les ex æquo par le MÉRITE : la mesure avalée doit être
    # celle qui ne ressemble pas à ce que la boucle annonce à sa place.
    periode = _boucle(S, n)
    merite = _merite_du_trou(S, n, periode) if periode else np.zeros(n)

    def merite_de(cuts):
        return float(sum(merite[c] for c in cuts if 0 <= c < n))

    ok = np.arange(4, max(5, n - 5))
    meilleur, coutures, mer = base, (), 0.0

    def retiens(cand, v, k):
        nonlocal meilleur, coutures, mer
        v -= k * MARGE_COUTURE
        m = merite_de(cand)
        if v > meilleur + 1e-9 or (abs(v - meilleur) <= 1e-9 and m > mer):
            meilleur, coutures, mer = v, cand, m

    for c in ok:
        retiens((int(c),), score((c,)), 1)
    for i, c1 in enumerate(ok):
        for c2 in ok[i + 1:]:
            retiens((int(c1), int(c2)), score((c1, c2)), 2)
    # une troisième, en extension du meilleur couple (le cas à trois trous est
    # rare : on ne paie pas n³ pour lui)
    if len(coutures) == 2:
        for c3 in ok:
            if c3 in coutures:
                continue
            cand = tuple(sorted(coutures + (int(c3),)))
            retiens(cand, score(cand), 3)

    # La LARGEUR (une mesure avalée, ou trois) ne change pas la parité — elle
    # change seulement l'alignement interne du jeton de couture. On la choisit
    # après coup, sur la vraie grille.
    meilleures, courant = {}, -1.0
    for largeurs in itertools.product((1, 3), repeat=len(coutures)):
        cand = dict(zip(coutures, largeurs))
        v = note(_grille(n, cand))
        if v > courant:
            meilleures, courant = cand, v
    coutures = meilleures
    bornes_finales = _grille(n, coutures)
    info = {"jumelles": int(len(P)), "part_rigide": round(base, 3),
            "part_retenue": round(courant, 3),
            "coutures": [(bornes_finales[j], bornes_finales[j + 1] - bornes_finales[j])
                         for j in _coutures_de(bornes_finales)]}
    if coutures:
        log.info("soudure: %d couture(s) %s, jumelles exprimables %.0f%% -> %.0f%%",
                 len(coutures), sorted(coutures.items()), base * 100, courant * 100)
    return _grille(n, coutures), info


def grille_et_mot(grid, triad, depart: int = 0):
    """(bornes, mot, info) — la grille de jetons ET les lettres qui vont avec.

    Toujours pas de second détecteur : la matrice est celle de
    `section_tool.substrates` — les vecteurs chord-tone de `harmonic_sections`
    sur les postérieures musx, c'est-à-dire le substrat sur lequel l'app
    répond déjà « où ce bloc se rejoue-t-il ? ». La comparaison bloc-à-bloc
    est `voice_sections._diag`, celle de la prod. Ne sont nouveaux ici que le
    grain (la bi-mesure), la couture, et le fait d'en tirer des lettres.

    Le groupage est en LIEN MOYEN, pas complet — Louis, 2026-08-12, sur
    Grenade : en lien complet une seule paire ratée sur six empêche le groupe
    et ses quatre couplets sortaient en `aaaa`, `babb`, `bfbd`, `gfge`.

    Différence assumée avec le mot de recherche (`vote_fill.bibar_word`) :
    celui-ci lit la BASSE, celui-là l'harmonie complète. Même règle de
    groupage, même seuil, substrat différent.

    `depart` (2026-08-15) : la mesure 1 de Louis. La recherche de coutures
    tourne alors sur le morceau À PARTIR d'elle, et l'intro reçoit sa propre
    tête de bande — deux raisons, pas une : le treillis se cale sur sa
    frontière (sans quoi une plaque sur deux l'enjambe et aucune section ne
    peut y commencer), et l'intro cesse de peser dans une recherche de trous
    qui ne parle que de la forme.
    """
    from harmonia_min import section_tool as st
    n = len(grid) - 1
    if n < 4:
        return None, None, {}
    S, _V, _M, _mute, _ch = st.substrates(grid, triad)
    depart = _depart_valide(depart, n)
    bornes, info = _meilleure_grille(S[depart:, depart:], n - depart)
    if depart:
        bornes = _tete(depart) + [depart + b for b in bornes]
        # `info["coutures"]` sert à la page pour DIRE où sont les trous : en
        # coordonnées du morceau, comme les bornes, sinon elle en montre à
        # côté (elles sont comparées à `bornes` dans `song_du_chart`).
        info = {**info, "coutures": [(depart + c, w)
                                     for c, w in info.get("coutures", [])]}
    if len(bornes) < 3:
        return None, None, info
    return bornes, _mot_depuis(_matrice_bimesures(S, bornes)), info


#: sous ce quotient de la mesure, un accord qui déborde n'est qu'un éclat de
#: frontière et pas un accord de la mesure (même seuil que chart_retour).
_ECLAT = 0.10


def _meta_par_temps(chart: dict) -> list:
    """[(t0, t1, accord)] — les accords ÉCRITS des sections, avec leur temps.

    Ils portent ce que la liste à plat n'a pas : `sug` (les candidats du
    modèle, ce que l'éditeur d'annotation montre), `n` (combien de fois le
    morceau joue cet accord), `colour`, `confirmed`. On les retrouve par le
    temps, jamais par l'index — c'est tout le sujet de accords_par_mesure.
    """
    out = []
    for s in chart.get("sections") or []:
        for bar in s.get("bars") or []:
            for c in bar or []:
                try:
                    out.append((float(c["t0"]), float(c["t1"]), c))
                except (KeyError, TypeError, ValueError):
                    continue
    return out


def accords_par_mesure(chart: dict) -> list:
    """Le contenu de CHAQUE mesure, pris dans la liste À PLAT et par le TEMPS.

    Louis, 2026-08-17, sur Easy On Me : « les barres 25 et 26 sont détectées
    comme A- et Bb, mais affichées dans le chart de la modification des
    sections comme Bb et F, c'est quoi ce bug ????? ». Il avait raison :
    A- et Bb sont bien ce que le modèle a décodé à 81,6 s et 85,0 s ; Bb et F
    sont les accords des mesures **43 et 44**.

    LA CAUSE. La version d'avant reconstruisait chaque mesure depuis le motif
    ÉCRIT de sa section, en le pavant modulo sa longueur :

        par_mesure[b] = bars[(b - b0) % len(bars)]

    Un pavage n'est légitime que pour un repli INTERNE — une section de huit
    mesures qui écrit quatre mesures jouées deux fois. Il ment dès qu'une
    section replie des passages de LONGUEURS DIFFÉRENTES : sur Easy On Me la
    section D couvrait [24,25] (deux mesures), [42,49] (huit) et [62,62] (une)
    avec un seul motif de huit mesures écrit d'après le passage long. Les
    mesures 24 et 25 recevaient donc les deux premières mesures de ce
    motif — les accords des mesures 42 et 43 — et le vrai contenu du passage
    court n'existait plus nulle part dans les sections.

    C'est la règle « under-fold, never over-fold » prise en défaut en amont :
    deux passages qui ne durent pas pareil ne sont pas le même passage. On ne
    la répare pas ici — on cesse d'en dépendre. `prompter.chords` est la liste
    à plat du décodage, un accord par empan de temps réel, et elle n'a jamais
    été repliée : c'est la seule source qui sait encore ce que jouent les
    mesures 25 et 26.

    Les métadonnées (`sug`, `n`, `colour`, `confirmed`) sont recollées depuis
    l'accord écrit qui occupe LE MÊME temps, quand il y en a un — sinon la
    réécriture d'un chart faisait disparaître les candidats du modèle de son
    éditeur d'annotation. Là où le repli avait tout perdu (le passage court),
    on rend l'accord sans sa métadonnée plutôt qu'une métadonnée d'ailleurs.

    Repli sur l'ancien pavage si le chart n'a pas de liste à plat.
    """
    grid = chart.get("barGrid") or []
    n = chart.get("nBars") or (len(grid) - 1)
    plat = (chart.get("prompter") or {}).get("chords") or []
    if not plat or len(grid) < 2:
        return _bars_par_mesure_pave(chart)
    bpb = int(chart.get("bpb") or 4)
    meta = _meta_par_temps(chart)
    out = []
    for b in range(n):
        t0, t1 = float(grid[b]), float(grid[b + 1])
        duree = max(t1 - t0, 1e-6)
        cases = []
        for c in plat:
            try:
                a, z = max(t0, float(c["t0"])), min(t1, float(c["t1"]))
            except (KeyError, TypeError, ValueError):
                continue
            if (z - a) / duree < _ECLAT:
                continue
            neuf = {"root": int(c.get("root") or 0), "q": c.get("q") or "",
                    "bass": int(c.get("bass", -1)), "nc": bool(c.get("nc")),
                    # « carry » : l'accord a commencé AVANT cette mesure, la
                    # vue ne le réécrit pas.
                    "carry": float(c["t0"]) < t0 - 1e-3,
                    "beat": max(0, min(bpb - 1,
                                       int(round((a - t0) / duree * bpb)))),
                    "bar": b, "c": float(c.get("c") or 0.0),
                    "t0": a, "t1": z}
            src = _meta_du_temps(meta, neuf)
            for k in ("sug", "n", "colour", "confirmed", "flag", "inflect"):
                if src is not None and k in src:
                    neuf[k] = src[k]
            cases.append(neuf)
        out.append(cases)
    return out


def _meta_du_temps(meta: list, accord: dict):
    """L'accord écrit qui occupe le même temps ET dit le même accord."""
    best, best_ov = None, 0.0
    for t0, t1, c in meta:
        ov = min(t1, accord["t1"]) - max(t0, accord["t0"])
        if ov <= 0:
            continue
        if (int(c.get("root") or 0) != accord["root"]
                or (c.get("q") or "") != accord["q"]):
            continue
        if ov > best_ov:
            best, best_ov = c, ov
    # la moitié de l'accord, sinon c'est un voisin qui déborde
    return best if best_ov >= 0.5 * max(accord["t1"] - accord["t0"], 1e-6) else None


def _bars_par_mesure_pave(chart: dict) -> list:
    """L'ancien pavage — gardé pour un chart sans liste à plat UNIQUEMENT.

    Attention au repli INTERNE : une section peut couvrir huit mesures et ne
    porter que quatre `bars`, son motif se répétant deux fois dans la plage.
    Ignorer ça laissait des mesures vides sur 26 charts sur 46 ; avec le
    pavage, zéro. Ce que le pavage ne sait PAS faire : un repli de passages
    de longueurs différentes (voir accords_par_mesure).
    """
    n = chart.get("nBars") or (len(chart.get("barGrid") or []) - 1)
    par_mesure: dict = {}
    for s in chart.get("sections") or []:
        bars = s.get("bars") or []
        if not bars:
            continue
        for b0, b1 in s.get("barRanges") or []:
            for b in range(b0, min(b1, n - 1) + 1):
                par_mesure[b] = bars[(b - b0) % len(bars)]
    return [par_mesure.get(b, []) for b in range(n)]


def _bars_par_mesure(chart: dict) -> list:
    """Alias historique — voir accords_par_mesure."""
    return accords_par_mesure(chart)


def sections_pour_chart(chart: dict, secs: list[dict],
                        bars: list | None = None) -> list[dict]:
    """Les sections du chart, refaites à partir du découpage de la Soudure.

    `secs` : [{label, mesure_debut, mesure_fin}] en mesures 1-indexées, fin
    incluse — la forme que la page exporte.

    `bars` : les mesures déjà construites, quand l'appelant les a re-dérivées
    du nouveau découpage (`refold.refold`). Sans lui, on prend le chart brut
    tel quel — correct, mais sans l'empilement des répétitions, qui est une
    CONSÉQUENCE des sections et doit donc être refait avec celles-ci
    (Louis, 2026-08-17 ; voir le module refold).

    On rend la MÊME forme que la pipeline (`pipeline.py`, montage des
    sections) : id, label, tag, reps, spans, barRanges, bars, barSpans. Les
    occurrences d'une même lettre sont repliées ensemble **seulement si elles
    ont la même longueur** — c'est la règle de Louis, « under-fold, never
    over-fold » : une section se réécrit à la longueur qu'elle joue vraiment,
    et deux occurrences qui ne durent pas pareil restent deux entrées.
    """
    grid = chart.get("barGrid") or []
    n = chart.get("nBars") or (len(grid) - 1)
    bars = accords_par_mesure(chart) if bars is None else bars
    groupes: dict = {}
    for s in secs:
        b0 = max(0, int(s["mesure_debut"]) - 1)
        b1 = min(n - 1, int(s["mesure_fin"]) - 1)
        if b1 < b0:
            continue
        groupes.setdefault((str(s["label"]), b1 - b0), []).append((b0, b1))

    out, vus = [], {}
    for (lab, _lg), occ in sorted(groupes.items(), key=lambda kv: kv[1][0][0]):
        occ.sort()
        vus[lab] = vus.get(lab, 0) + 1
        b0, b1 = occ[0]
        out.append({
            "id": "L" + lab + ("" if vus[lab] == 1 else str(vus[lab])),
            "label": lab, "tag": "", "reps": len(occ),
            "spans": [[grid[a], grid[min(b + 1, n)]] for a, b in occ],
            "barRanges": [[a, b] for a, b in occ],
            "bars": bars[b0:b1 + 1],
            "barSpans": [[[grid[b], grid[min(b + 1, n)]]]
                         for b in range(b0, b1 + 1)],
        })
    out.sort(key=lambda s: s["barRanges"][0][0])
    return out


def song_du_chart(chart: dict, audio_dir=None) -> dict | None:
    """La chanson complète attendue par la page, audio et retour compris.

    Le mot vient de la RESSEMBLANCE harmonique quand on peut la calculer, et
    de l'égalité stricte des basses sinon. Mesuré sur 14 charts (les autres
    donnent la même image) :

        mot                soudures   1re paire   jetons restants à la fin
        basse (égalité)        7         ×6              47 %
        harmonie (SSM)         6        ×12              19 %

    Autant de soudures, mais des paires deux fois plus fréquentes et un
    morceau qui se replie deux fois plus loin — l'égalité stricte sous-groupe,
    comme prévu. Le repli est donc un vrai repli, pas un choix : il est nommé
    dans `mot_source` et la page l'affiche, pour qu'on ne juge jamais un
    découpage sans savoir de quel mot il sort.
    """
    base = mot_du_chart(chart)
    if not base:
        return None
    base["mot_source"] = "basse"
    try:
        from pathlib import Path
        from harmonia_min import musx as _musx
        stem = Path(chart.get("audio_url") or "").stem
        audio = (audio_dir or Path("docs/audio")) / f"{stem}.m4a"
        if stem and audio.exists():
            bornes, m, info = grille_et_mot(chart["barGrid"],
                                            _musx.frame_posteriors(audio)[0],
                                            depart=_mesure1(chart))
            if m:
                base["mot"] = m
                base["jetons"] = bornes
                base["mot_source"] = "harmonie"
                # La bande DIT ses coutures : un jeton d'une ou trois mesures
                # là où le morceau a une mesure en trop ou en moins. Sans ça,
                # une plaque plus étroite que ses voisines passe pour un bug.
                # On lit les DÉCISIONS, pas les longueurs : le dernier jeton
                # d'un morceau de longueur impaire fait trois mesures par
                # construction (la mesure orpheline y est rattachée), et le
                # compter comme couture faisait passer 22 charts sur 46 pour
                # troués alors qu'ils sont 12.
                mesures_cousues = {c for c, _ in info.get("coutures", [])}
                base["coutures"] = [j for j in range(len(bornes) - 1)
                                    if bornes[j] in mesures_cousues]
    except Exception:                                    # noqa: BLE001
        log.exception("soudure: mot par ressemblance indisponible, "
                      "repli sur la basse")

    # La mesure 1 de Louis, 1-indexée comme tout ce que la page affiche : le
    # treillis part de là, et une page qui le sait peut le dire.
    base["mesure1"] = _mesure1(chart) + 1
    base["titre"] = chart.get("title") or chart.get("file") or "Sans titre"
    base["audio_url"] = chart.get("audio_url") or None
    base["file"] = chart.get("file") or None
    if chart.get("file"):
        base["retour"] = {"href": "/?open=" + chart["file"], "label": "le chart"}
    return base
