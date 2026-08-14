"""Les ANCRES : une reprise claire du chant fixe deux sections, on comble ensuite.

    .venv/bin/python scripts/ancres.py --dump          # les 12, en texte
    .venv/bin/python scripts/ancres.py                 # -> docs/plots/ancres.html

Louis, 2026-08-13, en regardant la matrice de match section×section :

  « Dans la matrice où tu as encadré en rouge le meilleur match hors lui-même, on
    voit une succession diagonale de matchs : ça correspond clairement à une
    reprise avec décalage. Lorsqu'on trouve une diagonale il faut absolument
    l'exploiter. Donc : grâce à la voix, si on trouve une répétition claire, on
    s'en sert pour fixer deux sections, puis on remplit les trous entre ces deux
    répétitions. »

L'OBJET QUI PORTE LES DIAGONALES. On fait glisser une fenêtre de `W` mesures et on
note, pour chaque couple de positions (p, q), le score de blocs de la page
`match_bloc` : le carré diagonal de p contre le bloc croisé p×q. Ça donne une
carte `R[p,q]` dont les DIAGONALES sont les reprises —

    `R[p, p+δ]` élevé sur une plage continue de p
        = le morceau se rejoue tel quel δ mesures plus loin, sur toute la plage.

C'est exactement la succession de matchs qu'il a vue. On lit donc `R` par
DÉCALAGE, ce qui est la matrice temps-décalage de RefraiD retrouvée à partir de
son critère à lui.

DEUX LECTURES D'UNE PLAGE, et la seconde est gratuite :

  * décalage ≥ longueur  ->  **une paire** : le segment, et sa reprise δ plus loin ;
  * décalage < longueur  ->  **une période** : le segment se rejoue tous les δ
    mesures, donc autant d'occurrences que la plage en contient. C'est ce qui
    attrape les trois B enchaînés de la fin de This Love sans rien coder de plus.

CE QU'ON EN FAIT, dans l'ordre — son algorithme, mot pour mot :

  1. on relève toutes les plages, à tous les décalages ;
  2. chacune propose un segment ; on recalcule son score à sa VRAIE longueur
     contre tout le morceau, ce qui donne **toutes** ses occurrences (une famille,
     pas une paire) ;
  3. on pose **la famille la plus payante d'abord** — au sens du coût épistémique
     `occ·l − l − occ`, celui-là même qui choisit déjà le lien de groupage ;
  4. ce qui reste libre est un **trou**, et repart au découpage par mots de
     bi-mesures (`mots4`), qui n'a plus le droit de traverser une ancre.

POURQUOI LE COÛT ET NON « LA PLUS NOMBREUSE D'ABORD ». `mots4.placer` classe par
nombre d'occurrences, ce qui est juste là-bas parce que toutes ses phrases font
quatre mots. Ici les longueurs diffèrent, et un segment de 4 mesures se retrouve
MÉCANIQUEMENT plus souvent qu'un segment de 16 : sur This Love ce classement
posait treize blocs de 4 mesures et n'ancrait plus rien.

POURQUOI LES BORDS SONT CALÉS SUR LA GRILLE DE BI-MESURES. À `W = 2` la plage de
This Love au décalage 20 court des mesures 8 à 28, alors que sa frontière à lui
est en 9 : une fenêtre à cheval sur une mesure MUETTE et une mesure chantée
ressemble à la mesure chantée seule, donc elle marque encore. L'ambiguïté est
d'exactement `W − 1` mesures et la voix ne peut pas la lever — une section qui
commence par un silence ou une anacrouse n'a rien à dire là-dessus. On cale donc
chaque bord sur la grille de bi-mesures (celle de `vote_fill`, calée sur la
parité de la première mesure chantée), qui est déjà l'unité de tout le reste.
La voix décide QUELLES frontières existent ; la grille décide de la mesure exacte.

CE QUE ÇA NE RÉSOUT PAS. La voix est muette là où personne ne chante : une intro
ou un pont instrumental ne peut pas être ancré, il ressort en trou. Et le critère
demande un alignement au temps près, donc une mesure insérée (Grenade, Blue
Lights) coupe une diagonale en deux au lieu d'en laisser une.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from voix_variantes import cases, _ssm, f_hauteur          # noqa: E402
from licks import notes_de                                 # noqa: E402
from vote_fill import otsu, fill, votes as vf_votes         # noqa: E402
import order_bundle                                        # noqa: E402

RES = 4           # 4 cases par mesure : le temps
W = 2             # la fenêtre glissante qui propose, en mesures
MIN = 8           # une ancre fait au moins 8 mesures (4 bi-mesures : la regle deja
                  # en place dans vote_fill.MIN_BI). A 4, la boucle de 4 mesures
                  # de Stand By Me sort 21 fois et rafle tout le classement.
MARGE = 0.0       # écart minimal au deuxième meilleur match, en unités de seuil
PHASE = True      # recaler la phase des ancres sur les pics de changement
SEUIL_MIN = 0.50  # plancher : sous ça, « ça se ressemble » ne veut plus rien dire
_CACHE: dict[str, tuple] = {}


# ── la matrice de voix et le critère de blocs ────────────────────────────────

def voix(stem):
    """(S au temps, n) — la matrice hauteur réelle · temps, celle de match_bloc."""
    if stem in _CACHE:
        return _CACHE[stem]
    b = order_bundle.get(stem)
    n, grid = b["n"], np.asarray(b["grid"])
    S = np.nan_to_num(_ssm(f_hauteur(notes_de(stem), cases(grid, n, RES))))
    _CACHE[stem] = (S, n)
    return S, n


def profil(S, p, l, n, qs=None):
    """Le score de blocs du segment [p, p+l) contre les positions `qs`.

    `qs` vaut par défaut toutes les mesures ; lui passer la grille de bi-mesures
    divise le travail par deux et interdit les départs hors grille.
    """
    a0, L = p * RES, l * RES
    X = S[a0:a0 + L, a0:a0 + L]
    nx = float(np.linalg.norm(X))
    qs = list(range(n - l + 1)) if qs is None else [q for q in qs if q + l <= n]
    if nx <= 1e-9 or not qs:
        return np.zeros(len(qs)), qs
    Y = np.stack([S[a0:a0 + L, q * RES:q * RES + L] for q in qs])
    ny = np.sqrt(np.einsum("kij,kij->k", Y, Y))
    out = np.einsum("ij,kij->k", X, Y) / np.clip(nx * ny, 1e-9, None)
    return np.where(ny > 1e-9, out, 0.0), qs


def carte(S, n, w=W):
    """R[p,q] : le critère à fenêtre fixe. Ses DIAGONALES sont les reprises."""
    P = n - w + 1
    R = np.zeros((P, P))
    for p in range(P):
        R[p, :] = profil(S, p, w, n)[0][:P]
    return R


def score_span(V, a0, a1, b0, b1):
    """Ton critère entre deux plages de mesures — longueur commune = la plus courte.

    Retourne `nan` quand l'une des deux plages est MUETTE : la voix n'a alors
    pas d'avis, et un zéro se lirait à tort comme « ces deux sections sont
    différentes ». C'est le cas des intros et des ponts instrumentaux.
    """
    l = min(a1 - a0, b1 - b0)
    if l <= 0:
        return float("nan")
    L = l * RES
    X = V[a0 * RES:a0 * RES + L, a0 * RES:a0 * RES + L]
    Y = V[a0 * RES:a0 * RES + L, b0 * RES:b0 * RES + L]
    nx, ny = float(np.linalg.norm(X)), float(np.linalg.norm(Y))
    if nx <= 1e-9 or ny <= 1e-9:
        return float("nan")
    return float((X * Y).sum()) / (nx * ny)


def dist_voix(stem, spans):
    """La distance de voix entre plages, RAMENÉE À SON SEUIL : < 1 = même section.

    C'est la forme sous laquelle `mots4` peut mélanger deux critères d'échelles
    différentes (la voix et les lettres de basse) : chacun est exprimé en
    multiples de son propre seuil, donc « 1 » veut dire la même chose des deux
    côtés. `nan` = la voix se tait, c'est aux lettres de trancher.

    Le seuil est par morceau (Otsu), comme partout ailleurs ici : la ressemblance
    médiane d'un morceau va de 0,33 à 0,99, aucune constante ne marche.
    """
    V, _n = voix(stem)
    m = len(spans)
    Sc = np.full((m, m), np.nan)
    for i, (a0, a1) in enumerate(spans):
        for j, (b0, b1) in enumerate(spans):
            Sc[i, j] = 1.0 if i == j else score_span(V, a0, a1, b0, b1)
    hors = Sc[~np.eye(m, dtype=bool)]
    hors = hors[np.isfinite(hors)]
    th = max(SEUIL_MIN, float(otsu(hors))) if hors.size else SEUIL_MIN
    return (1.0 - Sc) / max(1e-6, 1.0 - th), th


def regrouper(stem, spans):
    """Les sections POSÉES, regroupées par la voix. (familles, muettes, seuil)

    C'est le seul endroit où le critère a été validé : sur des plages ALIGNÉES,
    il donne la bonne lettre à 89 % des sections annotées (12 morceaux, contrôle
    du 2026-08-14). Lâché sur toutes les fenêtres glissantes il s'écroule — il
    est lisse, donc deux fenêtres également MAL calées se ressemblent aussi.

    Une section muette (aucun chant) ne rejoint aucune famille : elle garde
    l'étiquette que lui donnent les lettres de basse.
    """
    DV, th = dist_voix(stem, spans)
    m = len(spans)
    muet = [not np.isfinite(DV[i, [j for j in range(m) if j != i]]).any()
            for i in range(m)]
    # LE PLUS PROCHE, PAS LE PREMIER SOUS LE SEUIL. Le seuil en lien moyen
    # (première version) donnait 64 % d'étiquettes contre 71 % aux lettres de
    # basse : il suffit d'une section mal calée pour aspirer tout un groupe.
    # L'argmax est la règle du contrôle qui vaut 89 %, et il ne rattache une
    # section que si son meilleur match la dépasse d'une MARGE (`MARGE`) —
    # sinon la voix hésite, et une voix qui hésite ne renomme rien.
    lab = [-1] * m
    for i in range(m):
        if muet[i]:
            continue
        avant = [j for j in range(i) if not muet[j] and np.isfinite(DV[i, j])]
        if not avant:
            continue
        j = min(avant, key=lambda j: DV[i, j])
        autres = [DV[i, k] for k in avant if k != j]
        marge = (min(autres) - DV[i, j]) if autres else 1.0
        if DV[i, j] <= 1.0 and marge >= MARGE:
            lab[i] = lab[j] if lab[j] >= 0 else j
    return lab, muet, th


def gain(occ: int, l: int) -> float:
    """Ce que la famille FAIT ÉCONOMISER, en mesures — le coût épistémique.

    Sans elle, ses `occ` occurrences de `l` mesures s'écrivent en entier
    (`occ × l`) ; avec elle, on écrit le motif une fois puis un renvoi par
    occurrence. C'est la même longueur de description que `mots4.cout`.
    """
    return occ * l - l - occ


def plages(R, seuil, w=W, mini=MIN):
    """Les plages continues au-dessus du seuil, décalage par décalage.

    [(départ, longueur, décalage, score moyen)]. La longueur est celle du
    segment COUVERT : une plage de m fenêtres de w mesures couvre m + w − 1
    mesures.
    """
    P = R.shape[0]
    out = []
    for d in range(mini, P):
        p = 0
        while p < P - d:
            if R[p, p + d] < seuil:
                p += 1
                continue
            q = p
            while q + 1 < P - d and R[q + 1, q + 1 + d] >= seuil:
                q += 1
            lg = (q - p) + w
            if lg >= mini:
                out.append((p, lg, d,
                            float(np.mean([R[k, k + d] for k in range(p, q + 1)]))))
            p = q + 1
    return out


def caler(b, x0, n):
    """La mesure `b` ramenée sur la grille de bi-mesures la plus proche."""
    cand = [v for v in x0 if 0 <= v <= n]
    return int(min(cand, key=lambda v: (abs(v - b), v))) if cand else int(b)


def occurrences(pr, qs, p, l, seuil):
    """Les positions où [p, p+l) se rejoue, sans chevauchement, la meilleure d'abord."""
    pris: list[int] = []
    for k in sorted(range(len(qs)), key=lambda k: -pr[k]):
        q = qs[k]
        if pr[k] < seuil and q != p:
            break
        if any(abs(q - r) < l for r in pris):
            continue
        pris.append(q)
    return sorted(pris)


# ── l'algorithme ─────────────────────────────────────────────────────────────

def ancres(stem, seuil=None, verbose=False):
    """Les ancres posées, la plus payante d'abord, et les trous qui restent.

      `pose`  [{b0, b1, tour, occ, score, l}]   les ancres, dans l'ordre posé
      `trous` [(b0, b1)]                        ce qui reste libre
      `R`, `seuil`, `plages`, `x0`              de quoi tracer la démonstration
    """
    S, n = voix(stem)
    b = order_bundle.get(stem)
    x0 = sorted(set(list(fill(b, stem)["x0"]) + [n]))
    grille = [v for v in x0 if v < n]

    # 1. LES BORDS SONT AUX EXTRÉMITÉS DE LA DIAGONALE, pas dans le niveau du
    #    score. C'est la leçon de la figure de `ancres_page` : décaler UNE des
    #    deux copies de deux mesures fait tomber le score à 45 % — le critère
    #    est net. Mais décaler LES DEUX du même nombre ne change rien : le long
    #    d'une diagonale, toutes les fenêtres marchent aussi bien. Chercher le
    #    meilleur segment par score ne peut donc pas trouver le bord ; il faut
    #    lire là où la diagonale COMMENCE et où elle S'ARRÊTE.
    R = carte(S, n)
    off = np.array([R[p, q] for p in range(R.shape[0]) for q in range(R.shape[0])
                    if abs(p - q) >= MIN])
    if seuil is None:
        seuil = max(SEUIL_MIN, float(otsu(off))) if off.size else SEUIL_MIN

    cands = {}
    for (p, lg, d, _s) in plages(R, seuil):
        # la plage de fenêtres p…p+lg−W couvre les mesures p…p+lg−1, à une
        # mesure près de chaque côté (une fenêtre à cheval sur une mesure MUETTE
        # et une mesure chantée ressemble à la mesure chantée seule). On lève
        # l'ambiguïté avec le critère lui-même, qui est net à cette échelle.
        best = None
        for a in (p, p + 1):
            for b in (p + lg - 1, p + lg):
                lon = min(b - a, d) if d < b - a else b - a   # d < longueur = période
                if lon < MIN or a + d + lon > n or a < 0:
                    continue
                v = score_span(S, a, a + lon, a + d, a + d + lon)
                if np.isfinite(v) and (best is None or v > best[0]):
                    best = (v, a, lon)
        if best is None:
            continue
        _v, a, lon = best
        if (a, lon) in cands:
            continue
        pr, qs = profil(S, a, lon, n)
        occ = occurrences(pr, qs, a, lon, seuil)
        if len(occ) >= 2:
            idx = {q: k for k, q in enumerate(qs)}
            cands[(a, lon)] = {"b0": a, "l": lon, "occ": occ,
                               "gain": gain(len(occ), lon),
                               "score": float(np.mean([pr[idx[q]] for q in occ]))}

    # 2. LA PHASE VIENT DES VOIX, PAS DE LA VOIX. Une zone périodique n'a pas de
    #    phase pour le critère : décaler TOUTES les copies du même nombre de
    #    mesures laisse le score identique — c'est mathématique, pas un défaut de
    #    réglage. (C'est l'erreur-type #4 du CLAUDE.md, retrouvée telle quelle :
    #    `score_periods` détectait la longueur de période et jamais sa phase.)
    #    Sans ce recalage, Bein' Green, Every Breath You Take et The Walk sortent
    #    tous DEUX MESURES trop tôt, systématiquement.
    #    On essaie donc chaque phase possible et on garde celle dont les
    #    frontières sont les mieux soutenues par les pics de changement.
    pics = {p["bar"]: p["votes"] for p in vf_votes(stem)}

    def soutien(dep, l, occ):
        return sum(pics.get(q + k, 0) for q in occ for k in (0, l))

    for f in (list(cands.values()) if PHASE else []):
        l = f["l"]
        best = max(((soutien(f["b0"] + ph, l, [q + ph for q in f["occ"]]), -abs(ph), ph)
                    for ph in range(-l + 2, l - 1, 2)
                    if f["b0"] + ph >= 0 and max(f["occ"]) + ph + l <= n),
                   default=None)
        if best and best[2]:
            f["b0"] += best[2]
            f["occ"] = [q + best[2] for q in f["occ"]]
            f["phase"] = best[2]

    # 3. la plus PAYANTE d'abord (pas la plus nombreuse : voir le docstring).
    fam = sorted(cands.values(), key=lambda f: (-f["gain"], -f["score"], f["b0"]))

    pris: list = [None] * n
    pose, tour = [], 0
    for f in fam:
        mises = [q for q in f["occ"]
                 if q + f["l"] <= n
                 and all(pris[k] is None for k in range(q, q + f["l"]))]
        if len(mises) < 2:               # une ancre seule n'ancre rien
            continue
        for q in mises:
            for k in range(q, q + f["l"]):
                pris[k] = tour
            pose.append({"b0": q, "b1": q + f["l"] - 1, "tour": tour,
                         "occ": len(mises), "score": f["score"], "l": f["l"]})
        if verbose:
            print(f"    tour {tour}: {f['l']} mes. × {len(mises)} -> "
                  f"{[q + 1 for q in mises]}  score {f['score']:.2f}  "
                  f"gain {f['gain']:.0f}")
        tour += 1

    trous, i = [], 0
    while i < n:
        if pris[i] is None:
            j = i
            while j + 1 < n and pris[j + 1] is None:
                j += 1
            trous.append((i, j))
            i = j + 1
        else:
            i += 1
    pose.sort(key=lambda s: s["b0"])
    return {"pose": pose, "trous": trous, "seuil": seuil, "fam": fam,
            "n": n, "S": S, "x0": x0, "grille": grille}


def coupes(stem, seuil=None) -> list[int]:
    """Les frontières que les ancres imposent — la PREMIÈRE INTENTION de découpe.

    C'est ce que `mots4.sections` reçoit : des mesures qu'aucune phrase n'a le
    droit de traverser. Les trous entre ancres restent au découpage par mots.
    """
    a = ancres(stem, seuil)
    c = {s["b0"] for s in a["pose"]} | {s["b1"] + 1 for s in a["pose"]}
    return sorted(x for x in c if 0 < x < a["n"])


def main():
    from ssm_zoo import SONGS, AUDIO, gt_sections
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        list(SONGS) if not args else [(s, s) for s in args])
    tot = [0, 0, 0]
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        a = ancres(stem, verbose=True)
        gt = gt_sections(stem)
        print(f"\n{title}  ({a['n']} mes., seuil {a['seuil']:.2f}, "
              f"{len(a['plages'])} plages)")
        print("  ancres " + " ".join(f"{s['b0']+1}-{s['b1']+1}" for s in a["pose"]))
        print("  trous  " + " ".join(f"{t0+1}-{t1+1}" for t0, t1 in a["trous"]))
        if gt:
            g = {s["b0"] for s in gt["sections"][1:]}
            c = set(coupes(stem))
            bons = len(c & g)
            print("  toi    " + " ".join(f"{s['label']}{s['b0']+1}-{s['b1']+1}"
                                         for s in gt["sections"]))
            print(f"  coupes {sorted(x+1 for x in c)} — {bons}/{len(c)} justes, "
                  f"{bons}/{len(g)} des tiennes")
            tot[0] += bons; tot[1] += len(c); tot[2] += len(g)
    if tot[1]:
        print(f"\nTOTAL  précision {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.0%}  ·  "
              f"rappel {tot[0]}/{tot[2]} = {tot[0]/tot[2]:.0%}")


if __name__ == "__main__":
    main()
