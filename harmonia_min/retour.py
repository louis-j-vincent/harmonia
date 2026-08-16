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

#: « un pattern de répétition interne d'au moins 4 barres » — en dessous, une
#: boucle de deux mesures n'est plus un motif de section, c'est un balancement
#: d'accords (I-V I-V), et le prendre pour modèle ferait de tout le morceau une
#: seule section.
PERIODE_MIN = 4

LETTRES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── le substrat ─────────────────────────────────────────────────────────────

def ssm_mesures(chart: dict, audio_dir=None) -> np.ndarray | None:
    """S[i,j] — la ressemblance harmonique entre la mesure i et la mesure j.

    C'est LA matrice de la prod (`harmonic_sections.ssm`, vecteurs de notes
    d'accord), au grain de la mesure. Louis, 2026-07-30, catégorique : jamais
    de SSM fondamentale-seule, un Bb doit être plus proche d'un Gm que d'un F.
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
    triad = _musx.frame_posteriors(audio)[0]
    return np.asarray(HS.ssm(triad, grid), dtype=float)


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

def _compare(S, a: int, b: int, L: int, queue: int = QUEUE_LIBRE) -> dict:
    """Le mot en `a` contre le mot en `b`, ses `queue` dernières mesures exemptées.

    Rend {sims, moyenne, n_compare}. La queue n'est pas comparée du tout :
    c'est ça, le « modulo les 2 dernières ».

    CHOIX 4 — `queue` tombe à 0 quand le modèle est une BOUCLE interne. La
    tolérance de Louis parle de la cadence d'une phrase de huit mesures ; sur
    une boucle de quatre, exempter deux mesures reviendrait à ne comparer que
    la moitié du motif, et n'importe quoi passerait.
    """
    k = max(1, L - queue)
    sims = [float(S[a + t, b + t]) for t in range(k)]
    return {"sims": sims, "moyenne": float(np.mean(sims)), "n_compare": k}


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


def _occurrences(S, modele: int, L: int, libre: list[bool], seuil: float,
                 queue: int = QUEUE_LIBRE) -> list[dict]:
    """Toutes les répétitions du mot `modele` (L mesures) dans les mesures libres.

    Balayage de toutes les positions, puis choix glouton de gauche à droite —
    deux occurrences ne peuvent pas se chevaucher, et la première trouvée gagne.
    """
    n = len(S)
    trouve = []
    for j in range(0, n - L + 1):
        if not all(libre[j:j + L]):
            continue
        c = _compare(S, modele, j, L, queue)
        if c["moyenne"] >= seuil:
            trouve.append({"b0": j, "b1": j + L - 1, **c})
    retenu, fin = [], -1
    for o in trouve:
        if o["b0"] > fin:
            retenu.append(o)
            fin = o["b1"]
    return retenu


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
        depart = next((b for b in range(pointeur, n) if libre[b]), None)
        if depart is None or depart > n - 2 * LONGUEUR_MIN:
            break

        # Jusqu'où le mot a le droit de courir : la prochaine mesure DÉJÀ prise.
        limite = next((b for b in range(depart + 1, n) if not libre[b]), n)

        etape: dict = {"depart": depart, "limite": limite, "candidats": [],
                       "retenu": None, "action": None, "section": None}
        premier_fort_vu = False

        # On évalue TOUS les retours, y compris ceux d'après le retenu : c'est
        # le contrefactuel (« et si on avait pris celui-là ? »), la seule façon
        # de voir sur la page ce que le choix du PREMIER retour a coûté.
        for k in range(depart + ECART_MIN, limite + 1):
            sim = float(S[depart, k]) if k < n else float("nan")
            fort = k < n and sim >= seuil
            cand = {"barre": k, "sim": sim, "fort": fort,
                    "litteral": fort and not premier_fort_vu,
                    "apres_coup": etape["retenu"] is not None,
                    "L": k - depart, "verdict": None, "repet": None,
                    "passe": False}
            if not fort:
                cand["verdict"] = "pas un retour"
                etape["candidats"].append(cand)
                continue
            premier_fort_vu = True
            L = k - depart

            if k + L <= n and all(libre[k:k + L]):
                cand["repet"] = _compare(S, depart, k, L)

            if L < LONGUEUR_MIN:
                cand["verdict"] = f"mot de {L} mesures — il en faut plus de 6"
            elif k + L > limite or not all(libre[k:k + L]):
                cand["verdict"] = "pas la place pour la répétition qui suit"
            elif cand["repet"]["moyenne"] < seuil:
                cand["verdict"] = (f"ne se répète pas juste après "
                                   f"(moyenne {cand['repet']['moyenne']:.3f} "
                                   f"< {seuil:.3f})")
            else:
                cand["passe"] = True
                cand["verdict"] = "retenu" if etape["retenu"] is None \
                    else "aurait marché aussi"
                if etape["retenu"] is None:
                    etape["retenu"] = cand
            etape["candidats"].append(cand)

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
        boucle = periode_interne(S, depart, L, seuil)
        etape["boucle"] = boucle
        modele_L = boucle["retenue"]["p"] if boucle["retenue"] else L
        queue = 0 if boucle["retenue"] else QUEUE_LIBRE   # CHOIX 4

        occ = _occurrences(S, depart, modele_L, libre, seuil, queue)
        label = LETTRES[len(trouvees) % len(LETTRES)]
        for o in occ:
            for b in range(o["b0"], o["b1"] + 1):
                libre[b] = False
        sec = {"label": label, "L": modele_L, "mot": L, "modele": depart,
               "boucle": boucle["retenue"]["p"] if boucle["retenue"] else None,
               "occurrences": occ}
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
