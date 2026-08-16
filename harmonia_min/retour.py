"""harmonia_min/retour.py — l'algo du retour : où la première barre revient-elle ?

LA RÈGLE EST DE LOUIS (2026-08-16), recopiée mot pour mot :

  « en commençant à la 1ère barre de musx, on teste toutes les barres suivantes
    jusqu'à ce qu'on trouve une ressemblance forte avec la 1ère barre. Quand
    c'est le cas, on va créer un mot correspondant aux barres 0 jusqu'à la barre
    juste avant la barre ressemblante. Si ce mot fait + de 6 barres et qu'il se
    répète juste après, modulo les 2 dernières barres (e.g. le mot fait 8 barres
    et les 8 barres suivantes sont le même mot, possiblement avec les 2
    dernières barres qui changent) alors on définit ça comme une première
    section, on regarde toutes les répétitions de cette section et ça nous fait
    une première section dans notre vocabulaire. Puis on recommence la même
    chose avec la première barre du reste du chart qui n'a pas encore été
    affecté à une section. »

Ce que ça dit musicalement : une section, c'est une période qui **revient sur
elle-même**. On ne cherche donc pas une frontière (où ça change), on cherche un
retour (où ça recommence), et on demande à ce retour de se confirmer une
deuxième fois avant d'y croire. La tolérance sur les deux dernières barres,
c'est la cadence : `abac` et `abad` sont la même période, ouverte puis fermée —
la même règle que `phrases4.nommer` applique aux bi-mesures.

TROIS CHOSES QUE LA RÈGLE NE DIT PAS, et que ce module tranche. Elles sont
marquées `# CHOIX` dans le code et ressortent telles quelles dans la trace,
pour qu'on discute des trois, pas de l'algorithme entier :

  CHOIX 1 — « ressemblance forte » = au-dessus du seuil d'Otsu sur les cases
    hors-diagonale de la SSM du morceau. Un seuil qui lit la FORME de la
    distribution du morceau, pas une constante absolue — et surtout pas un
    quantile fixe : mesuré ici même, le quantile 0,90 vaut 0,986 sur This Love
    et 0,9996 sur Let It Be, où il ne laisse plus passer aucun retour. Otsu
    rend 0,67 et 0,76 sur les deux. C'est le même remède que `soudure._otsu`,
    pour la même maladie (les cosinus de vecteurs positifs s'écrasent contre 1).
  CHOIX 2 — si le premier retour fort donne un mot qui échoue (trop court, ou
    qui ne se répète pas), on n'abandonne pas le départ : on essaie le retour
    fort suivant. La règle littérale s'arrête au premier ; la trace note à
    chaque fois `litteral: true/false` sur le candidat retenu, donc on voit
    exactement ce que ce rattrapage a changé.
  CHOIX 3 — si AUCUN retour ne marche depuis cette barre de départ, on avance
    d'une barre et on recommence. Les barres ainsi enjambées restent sans
    section (le « reste » : intros, ponts, outros).

CE QUE CET ALGO NE RÉSOUT PAS (règle #4 du projet) :
  * il ne nomme rien — les sections sortent A, B, C… dans l'ordre de
    découverte, pas couplet/refrain ;
  * il ne coupe pas une section trop longue. Si couplet et refrain font 8
    barres chacun et se suivent toujours dans le même ordre, le mot de 16
    barres se répète, et l'algo rend UNE section de 16 ;
  * il ne récupère pas le « reste ». Une intro qui ne revient jamais reste
    sans étiquette, c'est voulu, mais rien ici ne la nomme ;
  * la ressemblance est harmonique seulement (substrat accords/notes de la
    SSM de prod). Deux sections aux mêmes accords et au chant différent sont
    la même chose pour lui.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: CHOIX 1. Le seuil de « ressemblance forte » se lit par Otsu sur les cases
#: hors-diagonale. Aucun réglage à choisir : c'est la distribution du morceau
#: qui le pose.

#: « + de 6 barres ». Un mot de 6 barres ou moins n'est pas une section.
LONGUEUR_MIN = 7

#: « modulo les 2 dernières barres » — les deux dernières barres du mot sont
#: exemptées de la comparaison, c'est la cadence qui a le droit de changer.
QUEUE_LIBRE = 2

#: Un retour à une barre d'écart est un effet de bord de la SSM (une barre
#: ressemble à sa voisine parce que l'accord déborde), pas un retour. Même
#: garde-fou que `harmonic_sections.LAG_MIN`.
ECART_MIN = 2

#: CHOIX 6 (Louis, 2026-08-16 : « quand il y a plusieurs candidats, privilégie
#: les multiples de 4 barres »). Entre tous les retours qui passent le test, on
#: prend le plus court dont la longueur est un multiple de 4 — la carrure de la
#: musique populaire. Sans cette règle on prenait le PREMIER, et sur Stand By Me
#: c'était un mot de 7 mesures : une longueur qui n'existe pas dans ce morceau,
#: qui décalait tout le découpage, et qui était en plus trop courte pour qu'une
#: boucle de 4 s'y referme. Repli sur le premier qui passe si aucun n'est un
#: multiple de 4.
CARRURE = 4

#: « un pattern de répétition interne d'au moins 4 barres » — en dessous, une
#: boucle de deux mesures n'est plus un motif de section, c'est un balancement
#: d'accords (I-V I-V), et le prendre pour modèle ferait de tout le morceau une
#: seule section.
PERIODE_MIN = 4

#: CHOIX 8 (Louis, 2026-08-16) : « tu ne peux pas avoir une boucle de 4 solo qui
#: traîne — tu peux filer tes 4 à la suite, mais pour définir une répétition de
#: section plus tard dans le morceau il faut minimum de 8 barres ».
#:
#: Une occurrence isolée de quatre mesures n'est pas un retour de la section,
#: c'est une coïncidence de boucle : dans une musique construite sur quatre
#: accords, quatre mesures se ressemblent partout. Ce qui prouve qu'une section
#: revient, c'est qu'elle revient ASSEZ LONGTEMPS. Les occurrences collées les
#: unes aux autres s'additionnent — deux boucles de 4 côte à côte font bien un
#: retour de 8 — mais un bloc de 4 tout seul entre deux trous est jeté.
REPETITION_MIN = 8

LETTRES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── le substrat ─────────────────────────────────────────────────────────────

def basse_par_mesure(bass: np.ndarray, grid) -> np.ndarray:
    """(n_mesures, 12) — la basse que musx entend dans chaque mesure, normée.

    `bass` est le plan de basse de musx (13 colonnes : N puis les 12 classes de
    hauteur). La colonne N est jetée : « pas de basse » n'est pas une hauteur, et
    la garder ferait ressembler entre elles toutes les mesures silencieuses.
    """
    from harmonia_min import musx as _musx
    out = []
    for b in range(len(grid) - 1):
        a = int(grid[b] / _musx.FRAME_DT)
        z = max(a + 1, int(grid[b + 1] / _musx.FRAME_DT))
        seg = bass[a:min(z, len(bass))]
        out.append(seg.mean(0) if len(seg) else np.zeros(bass.shape[1]))
    V = np.asarray(out, dtype=float)[:, 1:]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def ssm_mesures(chart: dict, audio_dir=None, substrat: str = "accords"
                ) -> np.ndarray | None:
    """S[i,j] — à quel point la mesure i ressemble à la mesure j.

    `substrat` :
      "accords"        — LE DÉFAUT, la matrice de la prod
                         (`harmonic_sections.ssm`, vecteurs de notes d'accord).
                         Louis, 2026-07-30, catégorique : jamais de SSM
                         fondamentale-seule, un Bb doit être plus proche d'un Gm
                         que d'un F.
      "basse"          — la basse que musx entend, et rien d'autre.
      "accords*basse"  — le produit : deux mesures se ressemblent si elles ont
                         les mêmes notes d'accord ET la même basse.

    Les deux derniers sont là POUR REGARDER (Louis, 2026-08-16 : « sur Let It Be
    on chope mal les différences harmoniques […] je me demande si utiliser la
    matrice SSM de la basse ne pourrait pas aider »), pas pour décider : le
    défaut ne change pas tant que Louis n'a pas tranché sur les distances que
    `docs/plots/retour_basse_vs_accords.html` lui met sous les yeux.
    """
    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as _musx

    grid = chart.get("barGrid") or []
    if len(grid) < 5:
        return None
    stem = Path(chart.get("audio_url") or "").stem
    audio = Path(audio_dir or "docs/audio") / f"{stem}.m4a"
    if not stem or not audio.exists():
        log.warning("retour: pas d'audio pour %s", stem)
        return None
    probs = _musx.frame_posteriors(audio)
    if substrat == "basse":
        V = basse_par_mesure(probs[1], grid)
        return V @ V.T
    S = np.asarray(HS.ssm(probs[0], grid), dtype=float)
    if substrat == "accords":
        return S
    if substrat != "accords*basse":
        raise ValueError(f"retour: substrat inconnu {substrat!r}")
    V = basse_par_mesure(probs[1], grid)
    return S * (V @ V.T)


def hors_diagonale(S: np.ndarray) -> np.ndarray:
    """Les cases d'écart ≥ `ECART_MIN` — la seule distribution dont on tire un
    seuil. La diagonale et ses voisines valent ~1 par construction et
    tireraient n'importe quel seuil vers le haut (même raisonnement que
    `harmonic_sections.off_diagonal`)."""
    i = np.arange(len(S))
    return S[np.abs(i[:, None] - i[None, :]) >= ECART_MIN]


def seuil_fort(S: np.ndarray) -> float:
    """CHOIX 1 : le seuil de « ressemblance forte », posé par le morceau."""
    from harmonia_min.soudure import _otsu
    hors = hors_diagonale(S)
    return _otsu(hors) if hors.size else 1.0


# ── la comparaison de deux mots ─────────────────────────────────────────────

def exemption(L: int) -> int:
    """Combien de mesures de fin sont exemptées sur une suite de `L` mesures.

    LA RÈGLE UNIQUE (Louis, 2026-08-16, après deux allers-retours) : « on ne va
    faire qu'une règle pour l'instant — lors d'une suite consécutive d'au moins
    8 barres, exemption sur la dernière barre, et c'est tout. »

    Donc : une seule mesure, la dernière, et seulement si la suite atteint
    `REPETITION_MIN`. En dessous, rien n'est exempté — quatre mesures dont on
    en pardonne une, c'est 25 % de la preuve jetée, et c'est exactement ce qui
    avait fait mordre le A de This Love sur le pont (mesures 50–57).

    L'HISTORIQUE, pour qu'on ne le refasse pas :
      v1  zéro sur une boucle, deux sur un mot — la règle littérale de départ ;
      v2  une mesure sur quatre, deux sur huit — demandée, puis rejetée à
          l'oreille le jour même (« le A ne prend que 50 à 53 ») ;
      v3  zéro partout sur les boucles ;
      v4  CELLE-CI — une seule règle, au niveau de la SUITE et non du bloc.
          C'est le bon niveau : sur This Love le B est une suite de 8 mesures
          faite de deux boucles de 4, et la mesure qui change est la dernière
          de la SUITE (m.23, 43, 63, 71, 79), jamais celle du premier bloc.

    Ce que l'exemption ne dispense PAS de faire : noter la mesure exemptée.
    `_occurrences` la garde dans `exemptees`, et toute mesure qui diverge —
    exemptée ou non — reste dans `variante` pour le repliement du chart.
    """
    return 1 if L >= REPETITION_MIN else 0


def _compare(S, a: int, b: int, L: int, queue: int | None = None) -> dict:
    """Le mot en `a` contre le mot en `b`, ses `queue` dernières mesures exemptées.

    Rend {sims, queue_sims, moyenne, n_compare}. Les mesures exemptées ne
    comptent PAS dans la moyenne, mais leur ressemblance est calculée quand
    même et rendue dans `queue_sims` : c'est ce qui dit, plus tard, si cette
    occurrence-là est une variante à écrire sur le chart ou une simple redite.
    Sans ça l'exemption serait un oubli, pas une tolérance.
    """
    queue = exemption(L) if queue is None else queue
    k = max(1, L - queue)
    tous = [float(S[a + t, b + t]) for t in range(L)]
    return {"sims": tous[:k], "queue_sims": tous[k:],
            "moyenne": float(np.mean(tous[:k])), "n_compare": k}


def periode_interne(S, depart: int, L: int, seuil: float) -> dict:
    """Le mot boucle-t-il sur lui-même ? — la règle de Louis du 2026-08-16 :

      « lorsqu'on a identifié la 1ère section, il faut regarder si elle a
        elle-même un pattern de répétition interne d'au moins 4 barres. Si
        c'est le cas, c'est lui qu'on prend comme template pour détecter les
        similarités avec les barres suivantes (ça permet de prendre en compte
        le cas des chansons qui ont une partie du A qui boucle en intro avant
        que la chanson commence). »

    Ce que ça corrige : une intro qui répète en boucle quatre mesures du A fait
    un mot de huit mesures qui n'est PAS le A — c'est le A vu deux fois. Prendre
    la boucle comme modèle, et non le mot entier, rend le motif reconnaissable
    partout où il joue, intro comprise.

    Rend {periodes, retenue} : `periodes` liste chaque période testée avec son
    détail (c'est ce que la page affiche), `retenue` est la plus courte qui
    passe, ou None. La plus COURTE : si un mot de 16 boucle à 4, il boucle
    aussi à 8, et c'est 4 le motif.
    """
    out = []
    for p in range(PERIODE_MIN, L // 2 + 1):
        # Le mot contre lui-même décalé de p. On compare sur toute la longueur
        # disponible, sans exempter de queue : ici on ne cherche pas une
        # cadence, on cherche une boucle.
        sims = [float(S[depart + t, depart + t + p]) for t in range(L - p)]
        moy = float(np.mean(sims))
        out.append({"p": p, "sims": sims, "moyenne": moy, "passe": moy >= seuil})
    retenue = next((c for c in out if c["passe"]), None)
    return {"periodes": out, "retenue": retenue}


def _contigu(restants: list[int], i: int, L: int) -> bool:
    """Les L mesures qui commencent en `i` DANS LA CHANSON RECOUSUE sont-elles
    encore d'un seul tenant dans la vraie chanson ?

    Le recousu sert à rendre deux passages voisins comparables par-dessus une
    section déjà retirée ; il ne donne pas le droit d'inventer une section qui
    sauterait un trou. Une occurrence doit rester un morceau de musique
    continu — sinon on ne peut pas l'écrire sur un chart.
    """
    return (i + L <= len(restants)
            and restants[i + L - 1] - restants[i] == L - 1)


def _occurrences(P, restants: list[int], modele: int, L: int, seuil: float,
                 queue: int | None = None, modele_b0: int = 0) -> dict:
    """Toutes les répétitions du mot `modele` dans la chanson recousue.

    `P` est la SSM restreinte aux mesures encore libres, `restants` la table qui
    ramène un index recousu à sa vraie mesure. Balayage de toutes les positions,
    puis choix glouton de gauche à droite — deux occurrences ne peuvent pas se
    chevaucher, et la première trouvée gagne.

    CHOIX 5 (Louis, 2026-08-16 : « pourquoi on vire les mesures 50–53 ? elles
    n'ont pas passé le seuil pour valider que c'est un A ») — RECONNAÎTRE une
    occurrence est plus exigeant que REPÉRER un retour, et ça ne peut pas être
    le même nombre. Le seuil de retour est le plus lâche de l'algorithme : il
    sert à dire « tiens, ça se ressemble, regardons », pas à trancher. Réutilisé
    tel quel pour accepter une occurrence, il laissait passer This Love 50–53 à
    0,726 pour un seuil de 0,672, au milieu d'un pont — alors que les vraies
    occurrences du A sortent toutes entre 0,91 et 1,00.

    Le seuil d'occurrence se lit donc sur les scores de CE modèle-là : Otsu
    entre le tas des fenêtres qui ne ressemblent pas et le petit tas de celles
    qui sont vraiment le motif. Jamais en dessous du seuil de retour.
    """
    from harmonia_min.soudure import _otsu
    cand = []
    for j in range(0, len(restants) - L + 1):
        if not _contigu(restants, j, L):
            continue
        cand.append({"b0": restants[j], "b1": restants[j + L - 1], "j": j,
                     **_compare(P, modele, j, L, queue)})
    v = np.array([c["moyenne"] for c in cand])
    seuil_occ = max(seuil, _otsu(v, defaut=seuil)) if v.size else seuil

    retenu, fin = [], -1
    for o in cand:
        if o["moyenne"] >= seuil_occ and o["j"] > fin:
            # CE QUI DEVRA ÊTRE ÉCRIT SUR LE CHART (Louis, 2026-08-16 :
            # « attention à noter quelque part ces exemptions, car lors du
            # repliement du chart il faudra les noter sur le chart »).
            # Une mesure exemptée n'a pas été comparée ; si en plus elle ne
            # ressemble PAS au modèle, alors cette occurrence-là joue autre
            # chose à cet endroit, et le repliement doit l'écrire au lieu de
            # recopier le modèle. `exemptees` porte les deux cas, `variante`
            # ne garde que les mesures qui divergent vraiment.
            o["exemptees"], o["variante"] = [], []   # posés par suite, plus bas
            retenu.append(o)
            fin = o["j"] + L - 1
    # CHOIX 8 : on regroupe les occurrences COLLÉES, et un groupe qui n'atteint
    # pas `REPETITION_MIN` mesures n'est pas un retour de la section.
    groupes, courant = [], []
    for o in retenu:
        if courant and o["b0"] == courant[-1]["b1"] + 1:
            courant.append(o)
        else:
            if courant:
                groupes.append(courant)
            courant = [o]
    if courant:
        groupes.append(courant)
    gardes, solos, suites = [], [], []
    for g in groupes:
        long = g[-1]["b1"] - g[0]["b0"] + 1
        if long < REPETITION_MIN:
            solos.extend(g)
            continue
        gardes.extend(g)

        # L'EXEMPTION, AU NIVEAU DE LA SUITE (`exemption`, la règle unique de
        # Louis). La suite fait au moins REPETITION_MIN mesures : sa DERNIÈRE
        # mesure est exemptée, et c'est tout. C'est le bon niveau — sur This
        # Love le B est une suite de 8 faite de deux boucles de 4, et la mesure
        # qui change est la dernière de la SUITE (m.23, 43, 63, 71, 79), jamais
        # celle du premier bloc. Exempter bloc par bloc pardonnait deux mesures
        # sur huit au lieu d'une, et le A mordait sur le pont.
        bars = [{"mesure": o["b0"] + t, "mesure_modele": modele_b0 + t, "sim": v}
                for o in g for t, v in enumerate(o["sims"] + o["queue_sims"])]
        ex = bars[-1] if exemption(long) else None
        notes = bars[:-1] if ex else bars
        suites.append({
            "b0": g[0]["b0"], "b1": g[-1]["b1"], "n": long,
            "score": float(np.mean([x["sim"] for x in notes])) if notes else 1.0,
            "exemptee": ex,
            "variante": [x for x in notes if x["sim"] < seuil_occ],
            "occurrences": g})

    # Chaque occurrence reporte ce que SA suite a décidé : la page et le
    # repliement lisent l'occurrence, pas la suite.
    for su in suites:
        for o in su["occurrences"]:
            o["suite"] = (su["b0"], su["b1"])
            o["exemptees"] = ([su["exemptee"]] if su["exemptee"]
                              and su["exemptee"]["mesure"] <= o["b1"]
                              and su["exemptee"]["mesure"] >= o["b0"] else [])
            o["variante"] = [x for x in su["variante"]
                             if o["b0"] <= x["mesure"] <= o["b1"]]

    ecarte = [o for o in cand
              if seuil <= o["moyenne"] < seuil_occ]
    return {"occurrences": gardes, "seuil_occ": float(seuil_occ),
            "ecartees": ecarte, "solos": solos, "suites": suites,
            "n_fenetres": len(cand), "queue": queue}


# ── l'algorithme ────────────────────────────────────────────────────────────

def sections(S: np.ndarray, seuil: float | None = None,
             max_etapes: int = 40) -> dict:
    """{sections, etapes, seuil} — l'algo du retour, avec toute sa trace.

    `etapes` contient TOUT ce qui a été essayé, y compris les candidats
    rejetés et pourquoi : c'est la sortie que la démo affiche, et c'est elle
    qui sert à régler l'algo (règle du projet : illustrer, pas quantifier).
    """
    n = len(S)
    seuil = seuil_fort(S) if seuil is None else float(seuil)
    libre = [True] * n
    trouvees: list[dict] = []
    etapes: list[dict] = []
    pointeur = 0

    for _ in range(max_etapes):
        # LA CHANSON RECOUSUE (Louis, 2026-08-16) : « une fois qu'on a enlevé
        # les A, les mesures 16 à 23 sont directement suivies des mesures 36 à
        # 43, et il y a donc bien une répétition qui se trouve facilement ».
        # Toute l'étape se joue donc dans ces coordonnées-là : `restants` est la
        # chanson privée des sections déjà trouvées, `P` sa matrice, et deux
        # passages séparés par une section retirée y sont VOISINS. Sans ça, le
        # mot butait sur la première mesure prise et la répétition qui le
        # valide n'avait littéralement pas la place d'exister.
        restants = [b for b in range(n) if libre[b]]
        if len(restants) < 2 * LONGUEUR_MIN:
            break
        dep = next((i for i, b in enumerate(restants) if b >= pointeur), None)
        if dep is None or dep > len(restants) - 2 * LONGUEUR_MIN:
            break
        P = S[np.ix_(restants, restants)]
        depart = restants[dep]

        etape: dict = {"depart": depart, "restants": list(restants),
                       "coutures": [i for i in range(len(restants) - 1)
                                    if restants[i + 1] != restants[i] + 1],
                       "candidats": [], "retenu": None, "action": None,
                       "section": None}
        premier_fort_vu = False

        # On évalue TOUS les retours avant d'en choisir un : le choix se fait
        # ensuite sur l'ensemble (CHOIX 6, la carrure), et la page peut montrer
        # ce que chaque candidat écarté aurait donné.
        for k in range(dep + ECART_MIN, len(restants)):
            sim = float(P[dep, k])
            fort = sim >= seuil
            L = k - dep
            cand = {"barre": restants[k], "j": k, "sim": sim, "fort": fort,
                    "litteral": fort and not premier_fort_vu,
                    "L": L, "verdict": None, "repet": None, "passe": False}
            if not fort:
                cand["verdict"] = "pas un retour"
                etape["candidats"].append(cand)
                continue
            premier_fort_vu = True

            mot_entier = _contigu(restants, dep, L)
            repet_entiere = _contigu(restants, k, L)
            if repet_entiere:
                cand["repet"] = _compare(P, dep, k, L)

            if L < LONGUEUR_MIN:
                cand["verdict"] = f"mot de {L} mesures — il en faut plus de 6"
            elif not mot_entier:
                cand["verdict"] = "le mot enjamberait une section déjà retirée"
            elif not repet_entiere:
                cand["verdict"] = "pas la place pour la répétition qui suit"
            elif cand["repet"]["moyenne"] < seuil:
                cand["verdict"] = (f"ne se répète pas juste après "
                                   f"(moyenne {cand['repet']['moyenne']:.3f} "
                                   f"< {seuil:.3f})")
            else:
                cand["passe"] = True
            etape["candidats"].append(cand)

        # CHOIX 6 : parmi les retours qui passent, le plus court dont la
        # longueur est un multiple de 4. À défaut, le premier qui passe.
        passants = [c for c in etape["candidats"] if c["passe"]]
        carrures = [c for c in passants if c["L"] % CARRURE == 0]
        etape["retenu"] = (carrures or passants)[0] if passants else None
        etape["premier_passant"] = passants[0] if passants else None
        for c in passants:
            c["verdict"] = ("retenu" if c is etape["retenu"]
                            else "aurait marché aussi")
            c["hors_carrure"] = c["L"] % CARRURE != 0

        if etape["retenu"] is None:
            # CHOIX 3 : rien depuis cette mesure — on avance d'une, elle reste
            # sans section.
            etape["action"] = "avance"
            pointeur = depart + 1
            etapes.append(etape)
            continue

        L = etape["retenu"]["L"]

        # La règle de la boucle interne : si le mot se répète lui-même à une
        # période d'au moins PERIODE_MIN mesures, c'est CETTE boucle le modèle,
        # pas le mot entier.
        # Le mot est d'un seul tenant (on l'a exigé), donc le chercher dans la
        # vraie chanson ou dans la recousue revient au même ici.
        boucle = periode_interne(S, depart, L, seuil)
        etape["boucle"] = boucle
        modele_L = boucle["retenue"]["p"] if boucle["retenue"] else L
        queue = exemption(modele_L)                       # la règle unique

        rec = _occurrences(P, restants, dep, modele_L, seuil, queue,
                           modele_b0=depart)
        occ = rec["occurrences"]
        if not occ:
            # CHOIX 8 a tout jeté : le « retour » n'était qu'une boucle de
            # quatre isolée, même à l'endroit où on l'a trouvée. Ce n'est pas
            # une section. On avance, comme quand aucun retour ne marche —
            # sinon aucune mesure n'est prise, le pointeur ne bouge pas, et on
            # tourne jusqu'à `max_etapes` en empilant des sections vides (vu
            # sur Sunny : 36 sections, et Yesterday : 37).
            etape["action"] = "avance"
            etape["solos"] = rec["solos"]
            pointeur = depart + 1
            etapes.append(etape)
            continue
        label = LETTRES[len(trouvees) % len(LETTRES)]
        for o in occ:
            for b in range(o["b0"], o["b1"] + 1):
                libre[b] = False
        sec = {"label": label, "L": modele_L, "mot": L, "modele": depart,
               "boucle": boucle["retenue"]["p"] if boucle["retenue"] else None,
               "occurrences": occ, "seuil_occ": rec["seuil_occ"],
               "ecartees": rec["ecartees"], "n_fenetres": rec["n_fenetres"],
               "queue": rec["queue"], "solos": rec["solos"],
               "suites": rec["suites"]}
        trouvees.append(sec)
        etape["action"] = "section"
        etape["section"] = sec
        etapes.append(etape)

    reste = [b for b in range(n) if libre[b]]
    return {"sections": trouvees, "etapes": etapes, "seuil": seuil,
            "n_mesures": n, "reste": reste}


def par_mesure(res: dict) -> list[str | None]:
    """La lettre de chaque mesure, None pour le reste — la vue « à plat »."""
    out: list[str | None] = [None] * res["n_mesures"]
    for sec in res["sections"]:
        for o in sec["occurrences"]:
            for b in range(o["b0"], o["b1"] + 1):
                out[b] = sec["label"]
    return out
