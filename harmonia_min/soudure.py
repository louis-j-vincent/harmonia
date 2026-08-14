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


def _jetons(n_mesures: int) -> list[int]:
    """Les bornes des bi-mesures. La mesure orpheline d'un morceau impair est
    rattachée au dernier jeton plutôt que jetée — sinon la fin du morceau
    disparaît de la bande sans que rien ne le dise."""
    if n_mesures < 2:
        return [0, n_mesures]
    bornes = list(range(0, n_mesures - 1, 2))
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
    bornes = _jetons(n)

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

    ok = np.arange(4, max(5, n - 5))
    meilleur, coutures = base, ()
    s1 = np.array([score((c,)) for c in ok])
    j = int(np.argmax(s1))
    if s1[j] - MARGE_COUTURE > meilleur:
        meilleur, coutures = s1[j] - MARGE_COUTURE, (int(ok[j]),)
    # toutes les PAIRES, vectorisées par première coupure
    for i, c1 in enumerate(ok):
        v = np.array([score((c1, c2)) for c2 in ok[i + 1:]])
        if not len(v):
            continue
        k = int(np.argmax(v))
        if v[k] - 2 * MARGE_COUTURE > meilleur:
            meilleur = v[k] - 2 * MARGE_COUTURE
            coutures = (int(c1), int(ok[i + 1 + k]))
    # une troisième, en extension du meilleur couple (le cas à trois trous est
    # rare : on ne paie pas n³ pour lui)
    if len(coutures) == 2:
        for c3 in ok:
            if c3 in coutures:
                continue
            v = score(tuple(sorted(coutures + (int(c3),))))
            if v - 3 * MARGE_COUTURE > meilleur:
                meilleur = v - 3 * MARGE_COUTURE
                coutures = tuple(sorted(coutures + (int(c3),)))

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


def grille_et_mot(grid, triad):
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
    """
    from harmonia_min import section_tool as st
    n = len(grid) - 1
    if n < 4:
        return None, None, {}
    S, _V, _M, _mute, _ch = st.substrates(grid, triad)
    bornes, info = _meilleure_grille(S, n)
    if len(bornes) < 3:
        return None, None, info
    return bornes, _mot_depuis(_matrice_bimesures(S, bornes)), info


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
                                            _musx.frame_posteriors(audio)[0])
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

    base["titre"] = chart.get("title") or chart.get("file") or "Sans titre"
    base["audio_url"] = chart.get("audio_url") or None
    if chart.get("file"):
        base["retour"] = {"href": "/?open=" + chart["file"], "label": "le chart"}
    return base
