"""Aligner la grille d'un tab Ultimate Guitar sur ce que musx entend.

Louis, 2026-09-18 : « détecte les raw priors donnés par musx, puis trouve la
bonne transposition de tona pour l'accorder avec guitar tabs, et matche les
accords de guitar tabs DANS L'ORDRE en maximisant la log-vraisemblance des
accords de guitar tabs étant donné les priors musx inférés. Ça va te demander
d'établir une notion de distance MUSICALE entre accords. Il faut aussi un prior
sur la grille de guitar tabs pour inférer à quelle fréquence tel accord
apparaît. »

L'IDÉE, en une phrase : le tab dit QUOI et DANS QUEL ORDRE, l'audio dit QUAND.
Ni l'un ni l'autre ne suffit. Le tab n'a aucune notion de temps — il pose un
accord au-dessus d'une syllabe — et musx n'a aucune notion de forme : il entend
chaque instant sans savoir que le morceau répète une grille de quatre accords.

CE QUE CE MODULE N'EST PAS. Ce n'est pas « le tab a raison ». Louis, 2026-08-05 :
« harmonia's charts beat UG tabs ; use tabs for section STRUCTURE only, never
as chord ground truth ». Ici le tab n'est pas une vérité mais une SÉQUENCE
CANDIDATE : on cherche l'alignement qui rend l'audio le plus vraisemblable
sachant cette séquence. Un tab qui se trompe fait chuter la vraisemblance, et
ça se voit.

CE QUE ÇA NE RÉSOUT PAS, écrit d'avance pour ne pas le découvrir en route :
un tab qui omet un passage (solo, pont instrumental) laisse un trou que
l'alignement devra combler en étirant ses voisins ; un tab en capo déclare une
tonalité de manche, pas la tonalité qui sonne, d'où la recherche de
transposition ; et un tab qui note `Bdim7` là où musx n'a que cinq familles
(maj/min/dom/hdim/dim) perd sa septième — la distance musicale ci-dessous
absorbe cet écart au lieu de le refuser.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

import numpy as np

logger = logging.getLogger("harmonia.integrations.tab_align")

#: les cinq familles de `span_rescore.QUAL5`, dans le même ordre
QUAL5 = ["maj", "min", "dom", "hdim", "dim"]
#: les notes de chaque famille, en demi-tons depuis la fondamentale
NOTES_Q5 = {0: (0, 4, 7), 1: (0, 3, 7), 2: (0, 4, 7, 10),
            3: (0, 3, 6, 10), 4: (0, 3, 6)}
NOMS = "C C# D Eb E F F# G Ab A Bb B".split()
#: ce que coûte, par case, un écart à la cadence attendue du tab.
#: Mesuré sur les deux morceaux du POC : l'accord avec le top-1 de musx
#: vaut 79 % et 93 % sur tout le plateau 0,4–0,8, puis chute (Grenade
#: tombe à 44 % à 2,0, l'alignement traversant le tab au pas de course).
#: On se place au milieu du plateau.
RAIDEUR = 0.6
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# ── lire un accord de tab ───────────────────────────────────────────────────

def pitch_class(txt: str) -> int | None:
    """« Bb », « F# », « Ab » → sa classe de hauteur, ou None."""
    if not txt:
        return None
    t = txt.strip()
    if not t or t[0].upper() not in _PC:
        return None
    pc = _PC[t[0].upper()]
    for c in t[1:]:
        if c in "#♯":
            pc += 1
        elif c in "b♭":
            pc -= 1
        else:
            break
    return pc % 12


#: l'ordre compte : on teste les suffixes longs d'abord, sinon « m » attrape
#: le « m » de « maj7 » et un accord majeur devient mineur.
_FAMILLES = (
    ("dim7", 4), ("dim", 4), ("°", 4), ("o7", 4),
    ("m7b5", 3), ("ø", 3), ("half", 3),
    ("maj7", 0), ("maj9", 0), ("maj", 0), ("M7", 0), ("Δ", 0),
    ("m7", 1), ("m9", 1), ("m11", 1), ("m6", 1), ("min", 1), ("m", 1), ("-", 1),
    ("13", 2), ("11", 2), ("9", 2), ("7", 2),
)


def lire_accord(txt: str) -> dict | None:
    """« E7/G# » → {root: 4, q5: 2, bass: 8, texte: "E7/G#"}, ou None.

    Les familles sont ramenées aux cinq de musx : un `maj7` est un majeur, un
    `m9` un mineur, un `13` une dominante. On ne perd donc pas l'accord, on
    perd sa couleur — et c'est assumé, parce que c'est tout ce que les
    postérieures savent dire. `sus`, `add`, `no3` et les parenthèses sont
    ignorés pour la même raison.

    `N.C.` et les marqueurs de rythme (« x2 », « |ate; ») rendent None : ce ne
    sont pas des accords, et les garder décalerait toute la séquence.
    """
    if not txt:
        return None
    t = re.sub(r"[()\s]", "", str(txt)).replace("♭", "b").replace("♯", "#")
    if not t or t.upper() in ("N.C.", "NC", "X", "TACET"):
        return None
    bass = None
    if "/" in t:
        t, _, b = t.partition("/")
        bass = pitch_class(b)
    root = pitch_class(t)
    if root is None:
        return None
    # ce qui reste après la fondamentale et ses altérations
    reste = t[1:].lstrip("#b♯♭")
    reste = re.sub(r"(sus[24]?|add\d+|no\d+|\*)", "", reste, flags=re.I)
    q5 = 0
    for suffixe, q in _FAMILLES:
        if reste.startswith(suffixe):
            q5 = q
            break
    return {"root": root, "q5": q5, "bass": bass,
            "texte": str(txt).strip()}


def nom_q5(root: int, q5: int, bass: int | None = None) -> str:
    s = NOMS[root % 12] + {0: "", 1: "m", 2: "7", 3: "ø", 4: "°"}[q5]
    if bass is not None and bass % 12 != root % 12:
        s += "/" + NOMS[bass % 12]
    return s


# ── la séquence du tab, dans l'ordre ────────────────────────────────────────

_CH = re.compile(r"\[ch\](.*?)\[/ch\]", re.S)
_SECTION = re.compile(r"^\s*\[([^\]]{2,40})\]\s*$", re.M)


def sequence_du_tab(brut: str) -> list[dict]:
    """La suite des accords du tab, DANS L'ORDRE, avec leur section.

    On relit le balisage `[ch]…[/ch]` du document plutôt que la liste
    dédupliquée de `TabChords.chords` : c'est la SUITE qui porte la forme, et
    c'est elle qu'on aligne. Deux `Dm` consécutifs restent deux entrées — le
    tab les a écrits deux fois parce que l'accord se rejoue.

    Chaque entrée porte la section du tab (« Verse 1 », « Chorus »…) telle
    qu'elle est écrite : ça ne sert pas à l'alignement, mais c'est ce qui
    permettra à Louis de lire le résultat à côté du tab d'origine.
    """
    bornes = [(m.start(), m.group(1).strip()) for m in _SECTION.finditer(brut or "")]
    out = []
    for m in _CH.finditer(brut or ""):
        acc = lire_accord(m.group(1))
        if acc is None:
            continue
        section = None
        for pos, nom in bornes:
            if pos < m.start():
                section = nom
            else:
                break
        out.append({**acc, "section": section, "pos": m.start()})
    return out


def compresser(seq: list[dict]) -> list[dict]:
    """Les répétitions immédiates fondues en une entrée qui porte son compte.

    Un tab écrit l'accord au-dessus de CHAQUE ligne de parole qu'il couvre :
    `Dm` trois fois de suite, c'est un seul Dm tenu trois lignes, pas trois
    changements. Les garder séparés ferait croire à trois cases de grille et
    l'alignement étirerait le morceau.
    """
    out = []
    for a in seq:
        if out and out[-1]["root"] == a["root"] and out[-1]["q5"] == a["q5"] \
                and out[-1]["bass"] == a["bass"]:
            out[-1]["repetitions"] += 1
            continue
        out.append({**a, "repetitions": 1})
    return out


def prior_de_grille(seq: list[dict]) -> dict:
    """À quelle fréquence chaque accord du tab apparaît — le prior de grille.

    Louis : « il faut aussi un prior sur la grille de guitar tabs pour inférer
    à quelle fréquence tel accord apparaît ». Rendu : pour chaque (root, q5),
    sa part des apparitions, et la durée moyenne qu'il occupe (en nombre de
    lignes du tab qu'il couvre).

    À quoi ça sert : l'alignement doit décider combien de mesures donner à
    chaque case. Un accord que le tab écrit une fois sur trois et tient deux
    lignes n'a pas la même espérance de durée qu'un accord de passage.
    """
    n = sum(a.get("repetitions", 1) for a in seq) or 1
    part: Counter = Counter()
    duree: dict = {}
    for a in seq:
        cle = (a["root"], a["q5"])
        part[cle] += a.get("repetitions", 1)
        duree.setdefault(cle, []).append(a.get("repetitions", 1))
    return {"part": {k: v / n for k, v in part.items()},
            "duree_moyenne": {k: float(np.mean(v)) for k, v in duree.items()},
            "n_cases": len(seq), "n_apparitions": n}


# ── la distance MUSICALE entre deux accords ─────────────────────────────────

def notes_de(root: int, q5: int) -> frozenset:
    """Les classes de hauteur d'un accord (root, famille)."""
    return frozenset((root + s) % 12 for s in NOTES_Q5.get(q5, (0, 4, 7)))


def proximite(a: tuple, b: tuple, poids_basse: float = 0.15) -> float:
    """À quel point deux accords sonnent pareil, de 0 à 1.

    `a` et `b` sont `(root, q5)` ou `(root, q5, basse)`.

    LA MESURE : le recouvrement des NOTES (Jaccard sur les classes de
    hauteur), plus un bonus si la basse est la même. Pas la distance entre
    fondamentales : deux accords à un demi-ton l'un de l'autre n'ont rien en
    commun, alors qu'une tierce d'écart en partage deux notes sur trois.

    VÉRIFIÉ contre la règle que Louis a posée en majuscules (2026-07-30, « any
    SSM must be the chord-tone one where Bb is closer to Gm than to F ») :

        Bb↔Gm  {10,2,5} ∩ {7,10,2} = 2 sur 4 → 0,50
        Bb↔F   {10,2,5} ∩ {5,9,0}  = 1 sur 5 → 0,20

    C'est la même famille de mesure que `sections.similarity` utilise déjà
    pour dire où un bloc se rejoue — une seule notion de « ça sonne pareil »
    dans le projet, pas deux.

    CE QUE ÇA NE DIT PAS : le RÔLE. Un G7 et un Db7 sont interchangeables dans
    une cadence (substitution tritonique) et n'ont ici que deux notes
    communes ; à l'inverse Am et C en partagent deux sans jouer le même rôle.
    La distance de rôle est une autre question, ouverte depuis le 2026-09-15.
    """
    na, nb = notes_de(a[0], a[1]), notes_de(b[0], b[1])
    inter = len(na & nb)
    union = len(na | nb) or 1
    s = inter / union
    ba = a[2] if len(a) > 2 and a[2] is not None else a[0]
    bb = b[2] if len(b) > 2 and b[2] is not None else b[0]
    if ba % 12 == bb % 12:
        s = min(1.0, s + poids_basse)
    return s


def matrice_proximite(poids_basse: float = 0.0) -> np.ndarray:
    """(60, 60) — la proximité entre tous les candidats de `span_rescore`.

    L'index est celui de `span_rescore.idx_of` : `root * 5 + famille`. On la
    calcule une fois, elle ne dépend d'aucun morceau.
    """
    M = np.zeros((60, 60))
    for i in range(60):
        ri, qi = divmod(i, 5)
        for j in range(60):
            rj, qj = divmod(j, 5)
            M[i, j] = proximite((ri, qi), (rj, qj), poids_basse)
    return M


_PROX = None


def prox() -> np.ndarray:
    global _PROX
    if _PROX is None:
        _PROX = matrice_proximite()
    return _PROX


# ── ce que musx entend, mesure par mesure ───────────────────────────────────

def priors_musx(audio, grille) -> np.ndarray:
    """(n_mesures, 60) — la postérieure de musx sur chaque mesure.

    Les 60 candidats sont ceux de `span_rescore` : 12 fondamentales × 5
    familles, index `root * 5 + famille`. On passe par `pool_span_musx` +
    `acoustic_logp_musx`, c'est-à-dire la LOI DE POOLING du projet, celle que
    l'éditeur d'accords utilise déjà — pas une seconde façon de lire les
    trames.
    """
    from harmonia import musx as _musx
    from harmonia.span_rescore import acoustic_logp_musx, pool_span_musx
    probs = _musx.frame_posteriors(audio)
    n = len(grille) - 1
    out = np.zeros((n, 60))
    for b in range(n):
        span = (float(grille[b]), float(grille[b + 1]))
        out[b] = np.exp(acoustic_logp_musx(*pool_span_musx(probs, [span]))[0][0])
    return out


def vraisemblance(p_mesure: np.ndarray, case: tuple, plancher=1e-4) -> float:
    """log P(cet accord de tab | ce que musx entend sur cette mesure).

    PAS une égalité stricte : la masse que musx pose sur un accord PROCHE
    compte, à hauteur de sa proximité. Sans ça, un tab qui écrit `Cmaj7` là où
    musx entend `C` serait jugé faux, et un tab juste à la couleur près serait
    rejeté d'un bloc.

    Le plancher empêche un −∞ : une case que musx contredit franchement doit
    coûter cher, pas rendre l'alignement entier impossible.
    """
    i = (case[0] % 12) * 5 + case[1]
    return float(np.log(max(plancher, float(p_mesure @ prox()[i]))))


def meilleure_transposition(seq: list[dict], P: np.ndarray) -> tuple[int, list]:
    """De combien de demi-tons transposer le tab pour qu'il colle à l'audio.

    Louis : « trouve la bonne transposition de tona pour l'accorder avec
    guitar tabs ». C'est un vrai besoin, pas une précaution : le tab de This
    Love est écrit en Am (capo, ou transposé pour la guitare) alors que le
    disque est en Do mineur — trois demi-tons d'écart.

    On note chaque décalage par la vraisemblance de la GRILLE ENTIÈRE contre
    la postérieure moyenne du morceau, pondérée par le prior de grille : un
    accord qui occupe 30 % du tab pèse 30 % du score. C'est beaucoup moins
    cher qu'un alignement complet par décalage, et ça suffit — l'écart entre
    le bon décalage et les autres se compte en dizaines de nats.

    Rend `(décalage, les 12 scores triés)`.
    """
    moy = P.mean(0)
    pr = prior_de_grille(seq)["part"]
    scores = []
    for s in range(12):
        tot = 0.0
        for (root, q5), part in pr.items():
            i = ((root + s) % 12) * 5 + q5
            tot += part * float(np.log(max(1e-4, float(moy @ prox()[i]))))
        scores.append((tot, s))
    scores.sort(reverse=True)
    return scores[0][1], scores


def aligner(seq: list[dict], P: np.ndarray, decalage: int = 0,
            saut_max: int = 3, raideur: float = RAIDEUR) -> list[int]:
    """Quelle case du tab joue sur chaque mesure — dans l'ORDRE, sans retour.

    C'est la demande de Louis : « matche les accords de guitar tabs DANS
    L'ORDRE en maximisant la log-vraisemblance ». Donc une programmation
    dynamique monotone : la mesure b joue la case i, la mesure b+1 joue i ou
    plus loin. Jamais avant — un tab ne revient pas en arrière.

    ON PEUT AVANCER DE PLUSIEURS CASES DANS UNE MESURE, et c'est le point.
    Une première version n'autorisait que « rester » ou « avancer d'une », et
    elle décrochait à la mesure 13 sur This Love : le tab y écrit 115 cases
    pour 80 mesures, donc une case dure en moyenne 0,7 mesure — l'alignement
    était forcé d'avancer à chaque mesure, il perdait la phase du cycle de
    quatre accords et ne la retrouvait jamais. Un tab pose souvent deux
    accords dans la même mesure ; le modèle doit pouvoir le dire.

    LA DURÉE VIENT DU PRIOR DE GRILLE, et c'est un vrai prior. Une mesure
    consomme `k` cases ; on attend `r = cases / mesures` en moyenne, donc on
    paie `raideur × log Poisson(k ; r)`. C'est ce que Louis appelait « un
    prior sur la grille pour inférer à quelle fréquence tel accord apparaît » :
    un accord que le tab tient sur trois lignes pèse trois fois plus dans `r`.

    POURQUOI PAS `|k - r|`, la première version. Avec `k` entier, une valeur
    absolue est LINÉAIRE entre rester et avancer : l'écart de coût vaut
    `raideur × (1 - 2r)` quel que soit `r ≥ 1`, c'est-à-dire un pouce constant
    sur la balance dont le SIGNE bascule à `r = 0,5` — pas une cadence. Elle
    classait aussi un double saut moins cher qu'un surplace. Le Poisson se
    normalise, pique en `k ≈ r`, et punit la course comme le surplace.
    Mesuré sur le POC : This Love inchangé (78,7 % d'accord avec musx),
    Grenade 93,0 → 95,0 %, log-vraisemblance -0,243 → -0,231.

    CE QUE ÇA NE RÉSOUT PAS, deux choses.
    1. Un tab qui saute un passage instrumental : rien ici ne peut insérer une
       case que le tab n'a pas écrite — l'alignement étirera ses voisines, et
       ça se voit sur la page de contrôle.
    2. Un tab BEAUCOUP plus court que le morceau (`r < 0,5`, p. ex. 40 cases
       pour 100 mesures) : le prior dit alors qu'il faut rester, et si l'audio
       n'a aucun contraste l'alignement peut ne jamais consommer la grille.
       Ce n'est pas un bug — c'est le MAP du modèle — mais ça se surveille par
       la ligne « cases utilisées » de la page. Sur les deux morceaux du POC
       `r` vaut 1,44 et 1,05, et le chemin atteint bien la dernière case.
    """
    t = table(seq, P, decalage, saut_max, raideur)
    if t is None:
        return []
    best, prov = t["best"], t["prov"]
    n = best.shape[0]
    i = int(np.argmax(best[n - 1]))
    chemin = [0] * n
    for b in range(n - 1, -1, -1):
        chemin[b] = i
        if b:
            i = max(0, i - int(prov[b, i]))
    return chemin


def table(seq: list[dict], P: np.ndarray, decalage: int = 0,
          saut_max: int = 3, raideur: float = RAIDEUR) -> dict | None:
    """La table de programmation dynamique, et tout ce qui a servi à la remplir.

    `aligner` en lit le chemin ; `detail_mesure` en lit le raisonnement. Une
    seule implémentation, donc la page d'explication ne peut pas diverger de
    ce que le code fait vraiment — une erreur que ce projet a déjà payée
    (« vérifier ce qu'une chose FAIT avant d'expliquer pourquoi ça marche »).

    Rend `None` si la table n'a pas de sens (aucune mesure, ou aucune case).

      best[b, i]  le meilleur score cumulé d'un chemin qui finit sur la case i
                  à la mesure b ;
      prov[b, i]  de combien de cases on a avancé pour y arriver ;
      E[b, i]     la log-vraisemblance de la case i sur la mesure b ;
      cout[k]     le prix a priori d'avancer de k cases dans une mesure ;
      r           la cadence attendue, en cases par mesure.
    """
    n, m = P.shape[0], len(seq)
    if n == 0 or m == 0:
        return None
    poids = np.array([max(1, a.get("repetitions", 1)) for a in seq], dtype=float)
    # la cadence attendue : combien de cases une mesure consomme en moyenne
    r = float(poids.sum()) / max(1, n)
    saut = min(saut_max, max(1, int(np.ceil(r)) + 1))
    # log P(k cases consommées) sous un Poisson de moyenne r, pesé par
    # `raideur`. Une loi normalisée, pas un V arbitraire : voir `aligner`.
    cout = np.array([raideur * (k * math.log(max(1e-9, r)) - r
                                - math.lgamma(k + 1))
                     for k in range(saut + 1)])

    E = np.array([[vraisemblance(P[b], ((seq[i]["root"] + decalage) % 12,
                                        seq[i]["q5"]))
                   for i in range(m)] for b in range(n)])
    NEG = -1e9
    best = np.full((n, m), NEG)
    prov = np.zeros((n, m), dtype=np.int8)
    best[0, 0] = E[0, 0]
    for b in range(1, n):
        for i in range(m):
            meilleur, dk = NEG, 0
            for k in range(0, saut + 1):
                j = i - k
                if j < 0 or best[b - 1, j] <= NEG / 2:
                    continue
                v = best[b - 1, j] + cout[k]
                if v > meilleur:
                    meilleur, dk = v, k
            if meilleur > NEG / 2:
                best[b, i], prov[b, i] = meilleur + E[b, i], dk
    return {"best": best, "prov": prov, "E": E, "cout": cout, "r": r,
            "saut": saut, "NEG": NEG}


def detail_mesure(seq: list[dict], P: np.ndarray, b: int, decalage: int = 0,
                  combien: int = 4, **kw) -> dict:
    """Pourquoi la mesure `b` a reçu la case qu'elle a reçue.

    Rend les cases que l'alignement pouvait choisir à cette mesure, chacune
    avec ses DEUX termes — ce que l'audio en pense (`lv`) et ce que la grille
    en pense (`prior`) — et leur somme. C'est la page d'explication de Louis
    (2026-09-18 : « montre-moi exactement comment tu fais matcher le prior et
    la logproba, j'aimerais comprendre tout le chemin »).

    Les deux termes sont des LOGS, dans la même unité (des nats), donc ils
    s'additionnent. C'est tout le truc : pas de poids à régler entre eux,
    juste une probabilité jointe qu'on lit en logarithme.
    """
    t = table(seq, P, decalage, **kw)
    chemin = aligner(seq, P, decalage, **kw)
    if t is None:
        return {}
    best, prov, E, cout, NEG = (t["best"], t["prov"], t["E"], t["cout"], t["NEG"])
    retenue = chemin[b]
    precedente = chemin[b - 1] if b else 0
    cands = []
    for k in range(len(cout)):
        i = precedente + k
        if i >= len(seq) or (b and best[b - 1, precedente] <= NEG / 2):
            continue
        c = seq[i]
        cands.append({
            "case": i, "saut": k,
            "accord": nom_q5((c["root"] + decalage) % 12, c["q5"]),
            "section": c.get("section"),
            "lv": float(E[b, i]), "prior": float(cout[k]) if b else 0.0,
            "total": float(E[b, i] + (cout[k] if b else 0.0)),
            "retenue": i == retenue})
    return {"mesure": b, "case_precedente": precedente, "case_retenue": retenue,
            "candidats": cands, "r": t["r"]}


def part_des_temoins(p_mesure: np.ndarray, case: tuple,
                     combien: int = 4) -> list[tuple]:
    """Qui, dans ce que musx entend, soutient cette case — et de combien.

    La vraisemblance d'une case est `log Σ_c P(c) × proximité(c, case)`. Cette
    somme se décompose : chaque accord que musx envisage y verse sa
    probabilité, escomptée par sa distance musicale à la case. On rend les
    plus gros versements, pour que la somme se lise au lieu d'être crue.
    """
    M = prox()[case[0] * 5 + case[1]]
    parts = [(int(c), float(p_mesure[c]), float(M[c]), float(p_mesure[c] * M[c]))
             for c in range(60) if p_mesure[c] * M[c] > 1e-4]
    parts.sort(key=lambda x: -x[3])
    return parts[:combien]


# ── le tab entier, posé sur le morceau entier ───────────────────────────────

def priors_musx_temps(audio, temps) -> np.ndarray:
    """Comme `priors_musx`, mais une ligne par TEMPS et non par mesure."""
    return priors_musx(audio, temps)


def _log_duree(L: np.ndarray, d: float) -> np.ndarray:
    """Le prix d'avoir tenu un accord `L` temps quand on en attendait `d`.

    Poisson, comme la cadence : `log P(L ; d)`. C'est ici que le tab parle de
    DURÉE — un accord écrit au-dessus de trois lignes de paroles attend trois
    fois plus de temps qu'un accord écrit au-dessus d'une seule.
    """
    d = max(1e-6, float(d))
    return L * math.log(d) - d - _lgamma(L)


_LG = None


def _lgamma(L: np.ndarray) -> np.ndarray:
    global _LG
    n = int(np.max(L)) + 2
    if _LG is None or len(_LG) < n:
        _LG = np.array([math.lgamma(k + 1) for k in range(max(n, 512))])
    return _LG[L]


#: ce que coûte un changement d'accord selon sa place dans la mesure, en
#: nats. Zéro sur le premier temps, un peu sur le troisième (une demi-mesure
#: est une place normale), beaucoup sur les temps faibles. Mesuré : voir
#: `poser_tout`.
TEMPS_FORT = 0.4


def _cout_depart(n: int, bpb: int, poids: float, origine: int = 0) -> np.ndarray:
    """Le prix de faire COMMENCER un accord sur chaque temps du morceau.

    `origine` est l'indice du temps qui porte la PREMIÈRE barre de mesure. Il
    n'est pas toujours 0 : This Love a une levée d'un temps, donc ses premiers
    temps de mesure sont les indices 1, 5, 9… Compter la phase depuis le temps
    zéro y donnait 5 % de changements « sur le temps fort » contre 63 % sur
    Grenade, et la conclusion — « ce morceau ne tombe pas sur la grille » —
    était fausse : c'était la mesure qui l'était. Avec la bonne origine les
    deux morceaux se comportent pareil, 62 % et 63 % des changements sur le
    premier temps, 31 % et 30 % au milieu de la mesure.
    """
    if bpb < 2 or poids <= 0:
        return np.zeros(n + 1)
    pos = (np.arange(n + 1) - int(origine)) % bpb
    c = np.full(n + 1, -poids)                    # temps faible
    c[pos == 0] = 0.0                             # premier temps
    if bpb % 2 == 0:
        c[pos == bpb // 2] = -poids * 0.25        # milieu de mesure
    return c


def termes(seq: list[dict], P: np.ndarray, decalage: int = 0,
           raideur: float = RAIDEUR, bpb: int = 0,
           temps_fort: float = TEMPS_FORT, origine: int = 0) -> dict | None:
    """Les trois termes qui décident, sans la programmation dynamique.

    `poser_tout` les additionne ; la page d'explication les lit un par un.
    Une seule source, donc la page ne peut pas raconter autre chose que ce
    que le code fait.

      E[t, i]      ce que l'audio pense de l'accord i sur le temps t ;
      cum[t, i]    leur somme cumulée, pour lire un segment d'un coup ;
      attendue[i]  la durée que le tab annonce pour l'accord i, en temps ;
      depart[t]    ce que coûte un changement d'accord sur le temps t.
    """
    n, m = P.shape[0], len(seq)
    if n == 0 or m == 0 or m > n:
        return None
    poids = np.array([max(1, a.get("repetitions", 1)) for a in seq], dtype=float)
    E = np.array([[vraisemblance(P[t], ((seq[i]["root"] + decalage) % 12,
                                        seq[i]["q5"]))
                   for i in range(m)] for t in range(n)])
    return {"E": E, "cum": np.vstack([np.zeros(m), np.cumsum(E, axis=0)]),
            "attendue": poids * (n / poids.sum()),
            "depart": _cout_depart(n, bpb, temps_fort, origine),
            "raideur": raideur}


def termes_du_segment(t: dict, i: int, debut: int, fin: int) -> dict:
    """Ce que coûte de faire jouer l'accord `i` du temps `debut` au temps `fin`.

    Les trois termes sont des LOGS, dans la même unité, donc ils s'ajoutent.
    C'est tout le modèle, en une ligne.
    """
    audio = float(t["cum"][fin, i] - t["cum"][debut, i])
    duree = float(_log_duree(np.array([fin - debut]),
                             t["attendue"][i])[0]) * t["raideur"]
    dep = float(t["depart"][debut])
    return {"audio": audio, "duree": duree, "depart": dep,
            "total": audio + duree + dep, "temps": fin - debut}


def poser_tout(seq: list[dict], P: np.ndarray, decalage: int = 0,
               raideur: float = RAIDEUR, bpb: int = 0,
               temps_fort: float = TEMPS_FORT,
               origine: int = 0) -> list[int] | None:
    """Pose TOUS les accords du tab sur TOUT le morceau, dans l'ordre.

    Louis, 2026-09-18, en corrigeant le modèle : « on matche bien toute la
    longueur des accords consécutifs à toute la longueur du chart, et ensuite
    la seule question qui permet de maximiser la log-proba totale c'est
    comment je pose mes accords du tab à l'intérieur, sans jamais en changer
    l'ordre (un accord toujours après un autre) ».

    C'est une contrainte plus forte que ce que faisait `aligner`, et elle est
    meilleure sur trois points.

    1. LES DEUX BOUTS SONT ANCRÉS. `aligner` ancrait le début (case 0 sur la
       première mesure) et laissait la fin flotter : rien n'obligeait le
       dernier accord du tab à tomber à la fin du morceau.
    2. AUCUN ACCORD N'EST JETÉ. `aligner` sautait les cases qu'il n'arrivait
       pas à caser — 39 sur 115 sur This Love, parce qu'une mesure ne peut
       porter qu'un accord. Ici l'unité est le TEMPS : 322 temps pour 115
       accords, il y a la place pour tous, et deux accords peuvent partager
       une mesure comme sur un vrai chart.
    3. LE PRIOR REDEVIENT UTILE. Les deux bouts étant fixés et chaque case
       recevant au moins un temps, le nombre total d'avancées ne dépend plus
       du chemin : un prior sur la CADENCE serait devenu une constante, donc
       inerte. Le prior porte donc sur la DURÉE de chaque accord, tirée de ce
       que le tab écrit — trois lignes de paroles sous un même accord valent
       trois fois une.

    Rend la case jouée sur chaque temps, ou `None` si le tab a plus d'accords
    que le morceau n'a de temps (auquel cas la contrainte est infaisable et
    on le DIT, on ne rogne pas en silence).
    """
    t = termes(seq, P, decalage, raideur, bpb, temps_fort, origine)
    if t is None:
        return None
    n, m = P.shape[0], len(seq)
    attendue, cum, depart = t["attendue"], t["cum"], t["depart"]

    NEG = -1e18
    # best[i, b] : cases 0..i posées, la case i FINIT juste avant le temps b
    best = np.full((m, n + 1), NEG)
    prov = np.zeros((m, n + 1), dtype=np.int32)
    L = np.arange(n + 1)
    # la première case part forcément du temps 0
    b0 = np.arange(1, n + 1)
    best[0, 1:] = cum[1:, 0] - cum[0, 0] + _log_duree(b0, attendue[0]) * raideur
    for i in range(1, m):
        # b doit laisser au moins un temps à chaque case restante
        for b in range(i + 1, n - (m - 1 - i) + 1):
            s = np.arange(i, b)                     # les fins possibles de i-1
            v = (best[i - 1, s] + (cum[b, i] - cum[s, i])
                 + _log_duree(b - s, attendue[i]) * raideur + depart[s])
            j = int(np.argmax(v))
            best[i, b], prov[i, b] = v[j], s[j]
    if best[m - 1, n] <= NEG / 2:
        return None
    # retour en arrière depuis « la dernière case finit à la fin du morceau »
    chemin = [0] * n
    b = n
    for i in range(m - 1, -1, -1):
        s = int(prov[i, b]) if i else 0
        for t in range(s, b):
            chemin[t] = i
        b = s
    return chemin
