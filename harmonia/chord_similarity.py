"""Une mesure de ressemblance entre deux accords, s(a, b | contexte) dans [0, 1].

Louis, 2026-10-01 : « il faut une mesure de similarité / de distance
reproductible et généralisable [...] Pense aussi à prendre en compte
l'harmonie du contexte. » Au lieu d'empiler des équivalences écrites à la
main (famille par temps, Bb/C ≡ C7sus4, couleurs ignorées), UNE fonction.

ÉTAT : candidates en comparaison (page `docs/plots/similarite_accords.html`,
note `docs/similarite_accords.md`). RIEN ici n'est branché dans le repli :
Louis arbitre d'abord sur la page. Le point de branchement prévu est
`harmonia.roles.meme_lecture(a, b, contexte)`.

Les candidates (toutes symétriques, toutes invariantes par transposition) :

  notes        cosinus des notes sonnées (basse comprise, comptée comme une
               note de plus si étrangère) — la ressemblance chord-tone du
               projet : B♭ est plus près de G- que de F.
  notes_basse  la même, la basse pesant W_BASSE fois (la « fondamentale =
               basse sonnée » du projet, CLAUDE.md règle 3).
  tiv          cosinus des Tonal Interval Vectors (DFT des notes, poids
               TIV.lib) — un espace TONAL : il rapproche les accords voisins
               par quintes.
  musx         appris : à quel point musx hésite entre les deux au même
               instant, sur 310 morceaux (table `assets/chord_similarity_musx.json`,
               refaite par `tools/chord_similarity_musx.py`).
  fonction     contexte : la fonction tonale (T / PD / D) dans la tonalité
               locale, lue sur le degré diatonique le plus proche ; 1 si
               mêmes fonctions, Jaccard sinon. Sans tonalité : None.
  notes7       `notes` sans les couleurs (9, 11, 13) — LA RECOMMANDATION.
  notes+fonction, notes+musx   moyennes 50/50.

CE QUE LA RECOMMANDATION NE RÉSOUT PAS (détail : docs/similarite_accords.md) :
  * le rythme harmonique : « D-7 | G7 » contre « D-7 | D-7 » (Lost Without U)
    est un décalage d'un accord, pas deux accords voisins — la mesure le voit
    comme D-7 contre G7 (0,50) ;
  * la tierce ne pèse pas plus qu'une autre note : tout couple de quatre
    notes qui en partagent trois vaut 0,75 (G7 ≈ Gm7 autant que D-7 ≈
    G7sus4) ;
  * le contexte tonal n'entre pas : la mesure sert à comparer deux passes
    d'une MÊME lettre, où tonalité et voisins sont déjà les mêmes ;
  * un accord de passage en plus (« A♭ » contre « A♭ G ») reste une
    différence : c'est la tolérance de structure de `folding.meme_moment`,
    pas une affaire de ressemblance d'accords.

Conventions d'accord : le dict du chart, {root, q, bass, nc}.
"""
from __future__ import annotations

import cmath
import json
import math
from functools import lru_cache
from pathlib import Path

from harmonia.roles import pcs

#: Version de la DÉFINITION (à incrémenter si une constante ci-dessous bouge).
VERSION = 1
#: Poids de la basse dans `notes_basse` (1 = une note comme une autre).
W_BASSE = 2.0
#: Poids des coefficients DFT 1..6 du TIV (TIV.lib, Ramires et al. 2020).
TIV_POIDS = (2.0, 11.0, 17.0, 16.0, 19.0, 7.0)
#: En dessous de ce cosinus avec TOUS les degrés diatoniques, un accord n'a
#: pas de fonction dans la tonalité (chromatique) : `fonction` rend None.
SEUIL_DIATONIQUE = 0.6

_ASSET = Path(__file__).resolve().parent / "assets" / "chord_similarity_musx.json"


# ── vecteurs de notes ────────────────────────────────────────────────────────

def notes_jouees(c: dict, couleurs: bool = True) -> frozenset[int]:
    """Les notes sonnées (`roles.pcs`, basse comprise). `couleurs=False`
    retire les EXTENSIONS au-dessus de la septième, lues depuis la
    fondamentale écrite : la 9 et la 11 d'un accord qui a une tierce, la 9
    et la 13 d'un accord qui a une septième mineure (C9 → C7, D-11 → D-7,
    G13 → G7, C69 → C6, C9sus4 → C7sus4). Les sus (pas de tierce), la 6 (pas de septième), les
    altérations (b9, #11…) et la basse restent."""
    out = pcs(c)
    if couleurs:
        return out
    r = int(c["root"]) % 12
    rel = {(p - r) % 12 for p in out}
    drop = set()
    if 3 in rel or 4 in rel:
        drop |= {2, 5}
    if 10 in rel:
        drop |= {2, 9}            # la 9 d'une 7sus4 aussi (9sus4 → 7sus4)
    b = c.get("bass", -1)
    garde = set() if b is None or b < 0 else {int(b) % 12}
    return frozenset(p for p in out if (p - r) % 12 not in drop or p in garde)


def vecteur_notes(c: dict, w_basse: float = 1.0, couleurs: bool = True) -> list[float]:
    """12 poids : 1 par note sonnée (`notes_jouees`) ; la basse pèse
    `w_basse` (la fondamentale si pas de basse écrite)."""
    v = [0.0] * 12
    if c.get("nc"):
        return v
    for p in notes_jouees(c, couleurs):
        v[p] = 1.0
    b = c.get("bass", -1)
    b = int(c["root"]) % 12 if b is None or b < 0 else int(b) % 12
    v[b] = w_basse
    return v


def _cos(u, v) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return 1.0 if nu == nv else 0.0
    return sum(x * y for x, y in zip(u, v)) / (nu * nv)


def _nc_ou(a: dict, b: dict):
    """Deux N.C. se valent (1), un N.C. ne vaut aucun accord (0), sinon None."""
    na, nb = bool(a.get("nc")), bool(b.get("nc"))
    if na or nb:
        return 1.0 if na and nb else 0.0
    return None


def notes(a: dict, b: dict, contexte=None) -> float:
    r = _nc_ou(a, b)
    return r if r is not None else _cos(vecteur_notes(a), vecteur_notes(b))


def notes7(a: dict, b: dict, contexte=None) -> float:
    """`notes` sans les couleurs (9, 11, 13 : voir `notes_jouees`)."""
    r = _nc_ou(a, b)
    return r if r is not None else _cos(vecteur_notes(a, couleurs=False),
                                        vecteur_notes(b, couleurs=False))


def notes_basse(a: dict, b: dict, contexte=None) -> float:
    r = _nc_ou(a, b)
    return r if r is not None else _cos(vecteur_notes(a, W_BASSE),
                                        vecteur_notes(b, W_BASSE))


# ── TIV ─────────────────────────────────────────────────────────────────────

def _tiv(v: list[float]) -> list[complex]:
    s = sum(v) or 1.0
    return [w * sum(v[n] * cmath.exp(-2j * math.pi * k * n / 12) for n in range(12)) / s
            for k, w in zip(range(1, 7), TIV_POIDS)]


def tiv(a: dict, b: dict, contexte=None) -> float:
    """Cosinus réel des TIV, ramené de [-1, 1] à [0, 1]."""
    r = _nc_ou(a, b)
    if r is not None:
        return r
    ta, tb = _tiv(vecteur_notes(a)), _tiv(vecteur_notes(b))
    num = sum((x * y.conjugate()).real for x, y in zip(ta, tb))
    na = math.sqrt(sum(abs(x) ** 2 for x in ta))
    nb = math.sqrt(sum(abs(x) ** 2 for x in tb))
    return 0.5 * (1.0 + num / (na * nb))


# ── musx (appris) ───────────────────────────────────────────────────────────

#: musx qualité → (intervalles depuis la fondamentale, intervalle de basse).
MUSX_PCS = {
    "maj": ((0, 4, 7), 0), "min": ((0, 3, 7), 0), "7": ((0, 4, 7, 10), 0),
    "maj7": ((0, 4, 7, 11), 0), "min7": ((0, 3, 7, 10), 0), "dim": ((0, 3, 6), 0),
    "dim7": ((0, 3, 6, 9), 0), "hdim7": ((0, 3, 6, 10), 0), "aug": ((0, 4, 8), 0),
    "sus2": ((0, 2, 7), 0), "sus4": ((0, 5, 7), 0), "sus4(b7)": ((0, 5, 7, 10), 0),
    "9": ((0, 4, 7, 10, 2), 0), "min9": ((0, 3, 7, 10, 2), 0),
    "maj9": ((0, 4, 7, 11, 2), 0), "11": ((0, 4, 7, 10, 2, 5), 0),
    "13": ((0, 4, 7, 10, 2, 5, 9), 0),
    "maj(6)": ((0, 4, 7, 9), 0), "min(6)": ((0, 3, 7, 9), 0),
    "maj/3": ((0, 4, 7), 4), "maj/5": ((0, 4, 7), 7), "maj/b7": ((0, 4, 7, 10), 10),
    "maj/2": ((0, 4, 7, 2), 2), "min/b3": ((0, 3, 7), 3), "min/5": ((0, 3, 7), 7),
    "min/b7": ((0, 3, 7, 10), 10), "min/2": ((0, 3, 7, 2), 2),
}


@lru_cache(maxsize=1)
def table_musx() -> dict:
    t = json.loads(_ASSET.read_text())
    return {"Q": t["qualities"], "S": t["S"], "version": t["version"],
            "diagnostics": t["diagnostics"]}


@lru_cache(maxsize=4096)
def _vers_musx(root: int, q: str, bass: int) -> tuple[int, int]:
    """L'étiquette musx la plus proche (fondamentale, index de qualité) :
    même basse d'abord, puis Jaccard des notes, puis la queue la plus courte.
    Déterministe ; exacte pour toute queue que musx sait écrire."""
    c = {"root": root, "q": q, "bass": bass}
    notes_c = pcs(c)
    basse_c = root if bass < 0 else bass
    Q = table_musx()["Q"]
    best, key = None, None
    for qi, name in enumerate(Q):
        rel, biv = MUSX_PCS[name]
        for r in range(12):
            n = frozenset((r + x) % 12 for x in rel)
            jac = len(n & notes_c) / len(n | notes_c)
            k = ((r + biv) % 12 == basse_c, jac, -len(rel), -qi)
            if key is None or k > key:
                best, key = (r, qi), k
    return best


def _basse(c: dict) -> int:
    b = c.get("bass", -1)
    return -1 if b is None or b < 0 else int(b) % 12


def musx(a: dict, b: dict, contexte=None) -> float:
    r = _nc_ou(a, b)
    if r is not None:
        return r
    ra, qa = _vers_musx(int(a["root"]) % 12, a.get("q") or "", _basse(a))
    rb, qb = _vers_musx(int(b["root"]) % 12, b.get("q") or "", _basse(b))
    return float(table_musx()["S"][qa][qb][(rb - ra) % 12])


def nom_musx(c: dict) -> str:
    noms = "C Db D Eb E F Gb G Ab A Bb B".split()
    r, qi = _vers_musx(int(c["root"]) % 12, c.get("q") or "", _basse(c))
    return f"{noms[r]}:{table_musx()['Q'][qi]}"


# ── fonction tonale (contexte) ──────────────────────────────────────────────

#: Les sept accords de septième diatoniques (intervalles depuis la tonique)
#: et leur fonction. Mineur = mineur harmonique pour V et vii (la sensible).
_DEGRES = {
    "major": [((0, 4, 7, 11), "T"), ((2, 5, 9, 0), "PD"), ((4, 7, 11, 2), "T"),
              ((5, 9, 0, 4), "PD"), ((7, 11, 2, 5), "D"), ((9, 0, 4, 7), "T"),
              ((11, 2, 5, 9), "D")],
    "minor": [((0, 3, 7, 10), "T"), ((2, 5, 8, 0), "PD"), ((3, 7, 10, 2), "T"),
              ((5, 8, 0, 3), "PD"), ((7, 11, 2, 5), "D"), ((8, 0, 3, 7), "PD"),
              ((11, 2, 5, 8), "D")],
}


def fonctions(c: dict, cle: tuple[int, str]) -> frozenset[str] | None:
    """Les fonctions du (ou des, à égalité) degré(s) diatonique(s) le(s) plus
    proche(s) en notes ; None si aucun n'atteint SEUIL_DIATONIQUE."""
    if c.get("nc"):
        return None
    t, mode = int(cle[0]) % 12, cle[1] if cle[1] in _DEGRES else "major"
    v = vecteur_notes(c)
    scores = []
    for rel, f in _DEGRES[mode]:
        d = [0.0] * 12
        for x in rel:
            d[(t + x) % 12] = 1.0
        scores.append((_cos(v, d), f))
    m = max(s for s, _ in scores)
    if m < SEUIL_DIATONIQUE:
        return None
    return frozenset(f for s, f in scores if s >= m - 1e-9)


def fonction(a: dict, b: dict, contexte=None) -> float | None:
    """Jaccard des fonctions dans la tonalité `contexte["cle"]` ; None sans
    tonalité ou si l'un des deux est chromatique (pas de fonction)."""
    r = _nc_ou(a, b)
    if r is not None:
        return r
    cle = (contexte or {}).get("cle")
    if cle is None:
        return None
    fa, fb = fonctions(a, cle), fonctions(b, cle)
    if fa is None or fb is None:
        return None
    return len(fa & fb) / len(fa | fb)


def notes_fonction(a: dict, b: dict, contexte=None) -> float:
    n, f = notes(a, b), fonction(a, b, contexte)
    return n if f is None else 0.5 * n + 0.5 * f


def notes_musx(a: dict, b: dict, contexte=None) -> float:
    return 0.5 * notes(a, b) + 0.5 * musx(a, b)


# ── v2 : la tierce pèse (Louis, 2026-10-01) ───────────────────────────────
# « G7 différent de Gm7, ce sont des familles différentes, il faut faire peser
# la tierce. Un sus4 est ambigu entre mineur et majeur par essence, donc
# proche des deux, car la quarte ne différencie pas. »

#: Une tierce contraire (majeure contre mineure, même fondamentale) multiplie
#: la ressemblance par ce facteur : G7/Gm7 0,75 → 0,30, C/Cm 0,67 → 0,27 —
#: sous le seuil de tous les crans sauf « tout ».
PENALITE_TIERCE = 0.4
#: Deux fondamentales différentes qui ont CHACUNE une tierce sont deux
#: accords définis, deux familles : × ce facteur (D^7 / F#-7 0,75 → 0,60,
#: C^7 / A-7 → 0,60, F7 / D-7 → 0,60) — sauf si l'un est CONTENU dans
#: l'autre (une voix sans fondamentale, une basse ajoutée : Fm/A♭ ⊂ Dø7,
#: B° ⊂ G7, F ⊂ D-7). Une sus n'a pas de tierce : D-7 / G7sus4 reste 0,75.
PENALITE_DEUX_TIERCES = 0.8
#: La « case de la tierce » lue depuis la fondamentale : 2 (sus2), 3, 4
#: (tierces), 5 (sus4). Masquée quand un seul des deux accords a une tierce.
_CASE_TIERCE = (2, 3, 4, 5)


def _fondamentale_et_tierce(c: dict) -> tuple[int, str | None]:
    """(fondamentale sonnante, "M" / "m" / None) — la fondamentale de
    `roles.accord_sonnant` (B♭/C se lit C7sus4 : pas de tierce)."""
    from harmonia.roles import accord_sonnant
    r, q = accord_sonnant(c)
    rel = {(p - r) % 12 for p in notes_jouees({**c, "root": r, "q": q,
                                                "bass": c.get("bass", -1)}, False)}
    if 4 in rel:
        return r, "M"
    if 3 in rel:
        return r, "m"
    return r, None


def _cle(c: dict) -> tuple:
    b = c.get("bass", -1)
    return (bool(c.get("nc")), int(c.get("root", 0)) % 12, c.get("q") or "",
            -1 if b is None or b < 0 else int(b) % 12)


@lru_cache(maxsize=65536)
def _v2(ka: tuple, kb: tuple) -> float:
    a = {"nc": ka[0], "root": ka[1], "q": ka[2], "bass": ka[3]}
    b = {"nc": kb[0], "root": kb[1], "q": kb[2], "bass": kb[3]}
    r = _nc_ou(a, b)
    if r is not None:
        return r
    va, vb = vecteur_notes(a, couleurs=False), vecteur_notes(b, couleurs=False)
    (ra, ta), (rb, tb) = _fondamentale_et_tierce(a), _fondamentale_et_tierce(b)
    if ra == rb and (ta is None) != (tb is None):
        # une sus contre un accord à tierce, même fondamentale : la quarte
        # ne différencie pas — on compare sans la case de la tierce
        for iv in _CASE_TIERCE:
            va[(ra + iv) % 12] = vb[(ra + iv) % 12] = 0.0
    s = _cos(va, vb)
    if ta is not None and tb is not None:
        if ra == rb and ta != tb:
            s *= PENALITE_TIERCE
        elif ra != rb:
            na = frozenset(i for i, x in enumerate(va) if x)
            nb = frozenset(i for i, x in enumerate(vb) if x)
            if not (na <= nb or nb <= na):
                s *= PENALITE_DEUX_TIERCES
    return s


def v2(a: dict, b: dict, contexte=None) -> float:
    """`notes7` + la tierce (fondamentale = celle qui SONNE,
    `roles.accord_sonnant`) :
      * même fondamentale, tierces contraires → × PENALITE_TIERCE (G7/Gm7) ;
      * même fondamentale, une seule a une tierce (une sus) → comparées sans
        la case de la tierce, 2 à 5 demi-tons (G7sus4 ≡ G7 ≡ Gm7) ;
      * fondamentales différentes, toutes deux avec une tierce, aucune
        contenue dans l'autre → × PENALITE_DEUX_TIERCES (C^7/A-7, F7/D-7) ;
      * sinon `notes7` tel quel (D-7/G7sus4 0,75, B♭/C/C7sus4 0,75)."""
    return _v2(_cle(a), _cle(b))


CANDIDATES = {
    "notes": notes, "notes7": notes7, "notes_basse": notes_basse, "tiv": tiv, "musx": musx,
    "fonction": fonction, "notes+fonction": notes_fonction, "notes+musx": notes_musx,
    "v2": v2,
}


#: LA RECOMMANDATION (docs/similarite_accords.md) : `notes7`.
RECOMMANDEE = "v2"

#: Proposition de crans pour le curseur « Section merging », lus sur la
#: recommandation TEMPS PAR TEMPS (une mesure vaut son pire temps). Chaque
#: seuil tombe au MILIEU d'un trou de la distribution des valeurs de `notes7` (qui
#: ne prend que quelques valeurs : 1, .89, .87, .80, .77, .75 | .67 | .60,
#: .58, .52, .50 | .45 … sur 23 qualités usuelles sans basse étrangère),
#: donc aucun de ces accords n'est à cheval sur un seuil :
#:   0  mêmes notes exactement (`notes` = 1 : C7 ≠ C9, D-7 = F6)
#:   1  ≥ 0.71 : couleurs, D-7 ≈ G7sus4, B♭/C ≈ C7sus4, G7 ≈ Gm7, C^7 ≈ A-7
#:   2  ≥ 0.63 : + relatif mineur (C ≈ A-), tierce changée sur triade (A- ≈ A)
#:   3  ≥ 0.47 : + triton (G7 ≈ D♭7), ii ≈ V (D-7 ≈ G7), F- ≈ F7
#:   4  tout
CRANS = {1: 0.71, 2: 0.63, 3: 0.47}


def cran(a: dict, b: dict, contexte: dict | None = None) -> int:
    """Le plus petit cran du curseur où `a` et `b` comptent pour le même accord."""
    if notes(a, b) >= 1.0 - 1e-9:
        return 0
    s = notes7(a, b)
    for c in sorted(CRANS):
        if s >= CRANS[c] - 1e-9:
            return c
    return 4


#: LE CURSEUR « Section merging » branché (2026-10-01, Louis : « c'est au cas
#: par cas, c'est là que le levier de granularité […] est utile ») : cran →
#: (seuil de `v2` pour « même accord », décalage de rythme toléré en temps
#: dans une mesure). Monotone : plus le cran monte, plus le seuil descend et
#: plus le décalage permis grandit. Les crans 0 et 4 gardent EN PLUS leurs
#: effets d'écriture propres dans `folding` (0 : pas de boucle, les fins de
#: couleur ; 4 : un bloc ×N).
CRANS_V2 = {0: (0.72, 0), 1: (0.72, 0), 2: (0.63, 1), 3: (0.38, 2), 4: (0.38, 2)}
CRAN_DEFAUT = 1


def cran_v2(a: dict, b: dict) -> int:
    """Le plus petit cran du curseur où `roles.meme_lecture(grain="famille")`
    dit « même accord » (0 si mêmes notes exactement, 4 = jamais avant
    « tout »). Sans le décalage de rythme, qui se juge sur une mesure."""
    if notes(a, b) >= 1.0 - 1e-9:
        return 0
    s = v2(a, b)
    for n in (1, 2, 3):
        if s >= CRANS_V2[n][0] - 1e-9:
            return n
    return 4


def similarite(a: dict, b: dict, contexte: dict | None = None,
               mesure: str = RECOMMANDEE) -> float | None:
    """s(a, b | contexte) dans [0, 1], 1 = le même accord. `mesure` choisit
    la candidate (voir CANDIDATES) ; par défaut la recommandation."""
    return CANDIDATES[mesure](a, b, contexte)
