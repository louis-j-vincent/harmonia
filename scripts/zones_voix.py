"""Les zones de reprise du chant : la voix donne l'étendue, les pics donnent la phase.

    .venv/bin/python scripts/zones_voix.py            # les 12, en texte
    .venv/bin/python scripts/zones_page.py            # -> docs/plots/zones_voix.html

Louis, 2026-08-13 : « on voit une succession diagonale de matchs, ça correspond
clairement à une reprise avec décalage — lorsqu'on trouve une diagonale il faut
absolument l'exploiter. Grâce à la voix, si on trouve une répétition claire, on
s'en sert pour fixer deux sections, puis on remplit les trous. »
Puis, le 2026-08-14, sur le partage des rôles : « top, tu me fais ça alors ».

L'OBJET. Une fenêtre de `W` mesures glisse sur le morceau ; `R[p,q]` porte le
critère de blocs de `match_bloc` (le carré diagonal de p contre le bloc croisé
p×q). Une DIAGONALE de `R` est une reprise : `R[p, p+δ]` élevé sur une plage
continue de p veut dire « le morceau se rejoue tel quel δ mesures plus loin, sur
toute cette plage ». C'est la matrice temps-décalage de RefraiD, retrouvée à
partir de son critère à lui.

LE PARTAGE DES RÔLES, qui est tout le fichier :

    la VOIX donne le DÉCALAGE et l'ÉTENDUE.   Les PICS donnent la PHASE.

Pourquoi il faut deux sources, et ce n'est pas un choix de confort : décaler UNE
des deux copies de deux mesures fait tomber le score à 45 % du vrai (le critère
est net), mais décaler **les deux ensemble** ne change RIEN. Le long d'une
diagonale toutes les fenêtres marchent aussi bien. Une diagonale ne peut donc pas
dire où la section commence — c'est l'erreur-type #4 du CLAUDE.md retrouvée telle
quelle, `score_periods` détectait la période et jamais sa phase. Sans recalage,
Bein' Green, Every Breath You Take et The Walk sortent tous deux mesures trop tôt.

CE QUE ÇA CHANGE, ET C'EST LE VRAI GAIN. Une zone se rejoue à l'identique : une
coupe trouvée à l'offset `c` de la première copie vaut pour TOUTES les copies. On
note donc chaque offset par le total des voix qu'il récolte **sur l'ensemble des
copies**. Un pic faible mais confirmé trois fois bat un pic isolé fort — ce
qu'aucune lecture locale du profil de changement ne peut faire. Une décision,
appliquée n fois.

CE QUE ÇA NE RÉSOUT PAS. La voix se tait là où personne ne chante : une intro ou
un pont instrumental ne fait partie d'aucune zone, il ressort en trou et repart
au découpage par mots de bi-mesures. Et le critère demande un alignement au temps
près, donc une mesure insérée (Grenade, Blue Lights) coupe une diagonale en deux.
Sunny module d'un demi-ton : la hauteur réelle n'étant pas invariante par
transposition, ses reprises d'après la modulation sont invisibles ici (il faudrait
la variante « intervalles » de `voix_variantes`).
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

from voix_variantes import cases, _ssm, f_hauteur           # noqa: E402
from licks import notes_de                                  # noqa: E402
from vote_fill import otsu, votes as vf_votes               # noqa: E402
import order_bundle                                         # noqa: E402

RES = 4           # 4 cases par mesure : le temps
W = 2             # la fenêtre glissante qui propose une diagonale, en mesures
MIN = 8           # une section fait au moins 8 mesures (4 bi-mesures, la règle
                  # déjà en place dans `vote_fill.MIN_BI`)
SEUIL_MIN = 0.50  # plancher : sous ça, « ça se ressemble » ne veut plus rien dire
_CACHE: dict[str, tuple] = {}
LETTRES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
    """Le score de blocs du segment [p, p+l) contre les positions `qs`."""
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
    """Le critère entre deux plages de mesures — longueur commune = la plus courte.

    `nan` quand l'une des deux est MUETTE : la voix n'a alors pas d'avis, et un
    zéro se lirait à tort comme « ces deux sections sont différentes ».
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


def plages(R, seuil, w=W, mini=MIN):
    """Les plages continues au-dessus du seuil, décalage par décalage.

    [(départ, longueur couverte, décalage, score moyen)].
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


# ── les pics : la phase ──────────────────────────────────────────────────────

def pics_de(stem) -> dict[int, int]:
    """{mesure: nombre de signaux qui proposent une frontière ici}."""
    return {p["bar"]: p["votes"] for p in vf_votes(stem)}


def caler_zone(p, lg, d, n, pics):
    """Les bords de la zone, à une mesure près, tranchés par les pics.

    La fenêtre de `W` mesures laisse `W − 1` mesures d'ambiguïté sur chaque bord :
    une fenêtre à cheval sur une mesure MUETTE et une mesure chantée ressemble à
    la mesure chantée seule, donc elle marque encore. La voix ne peut pas lever
    ça — les pics, si.
    """
    best = None
    for a in (p - 1, p, p + 1):
        for b in (p + lg - 1, p + lg, p + lg + 1):
            if b - a < MIN or a < 0 or b + d > n:
                continue
            v = sum(pics.get(x, 0) for x in (a, b, a + d, b + d))
            if best is None or v > best[0]:
                best = (v, a, b)
    return (best[1], best[2]) if best else (p, p + lg)


def couper_zone(unite, copies, pics, mini=MIN):
    """Les coupes INTERNES d'une zone, communes à toutes ses copies.

    Une zone se rejoue à l'identique : une coupe à l'offset `c` de la première
    copie vaut pour toutes. On note chaque offset par le total des voix qu'il
    récolte **sur l'ensemble des copies** — un pic faible confirmé trois fois bat
    un pic isolé fort — puis on prend les meilleurs tant qu'ils tiennent `mini`.
    """
    offs = []
    for c in range(mini, unite - mini + 1):
        v = sum(pics.get(q + c, 0) for q in copies)
        if v > 0:
            offs.append((v, c))
    pris: list[int] = []
    for v, c in sorted(offs, key=lambda x: (-x[0], x[1])):
        if all(abs(c - k) >= mini for k in pris) and min(c, unite - c) >= mini:
            pris.append(c)
    return sorted(pris)


# ── l'algorithme ─────────────────────────────────────────────────────────────

def zones(stem, seuil=None, verbose=False):
    """Les zones de reprise, leurs coupes internes, et les trous.

      `zones`    [{t0, t1, d, unite, copies, score}]
      `sections` [{b0, b1, fam, zone}]   les copies découpées, lettre partagée
      `trous`    [(b0, b1)]              ce que la voix ne couvre pas
    """
    S, n = voix(stem)
    R = carte(S, n)
    off = np.array([R[p, q] for p in range(R.shape[0]) for q in range(R.shape[0])
                    if abs(p - q) >= MIN])
    if seuil is None:
        seuil = max(SEUIL_MIN, float(otsu(off))) if off.size else SEUIL_MIN
    pl = plages(R, seuil)
    pics = pics_de(stem)

    libre = [True] * n
    Z, secs = [], []
    while True:
        # LA RÉPÉTITION LA PLUS CLAIRE D'ABORD : la plus étendue, pondérée par sa
        # qualité. C'est sa phrase — « si on trouve une répétition claire, on
        # s'en sert pour fixer deux sections » —, et l'étendue est le bon poids
        # parce que c'est la seule chose que la diagonale mesure sans ambiguïté.
        cand = None
        for (p, lg, d, s) in pl:
            t0, t1 = caler_zone(p, lg, d, n, pics)
            L = t1 - t0
            if L < MIN or t1 + d > n:
                continue
            # DEUX LECTURES D'UNE DIAGONALE, et il faut les distinguer :
            #   décalage < longueur -> PÉRIODE : ça se rejoue tous les d, donc
            #     autant de copies CONTIGUËS que la plage en contient ;
            #   décalage ≥ longueur -> PAIRE : deux copies seulement, séparées
            #     par du morceau qui n'appartient pas à la zone. Les traiter
            #     comme une période invente des copies inexistantes — c'est ce
            #     qui donnait « unité 8 × 6 » à Bein' Green sur un décalage 40.
            if d < L:
                unite = d
                copies = [t0 + k * d for k in range((L + d) // d)]
            else:
                unite = L
                copies = [t0, t0 + d]
            if unite < MIN or len(copies) < 2 or copies[-1] + unite > n:
                continue
            if not all(libre[k] for q in copies
                       for k in range(q, min(n, q + unite))):
                continue
            poids = len(copies) * unite * s
            if cand is None or poids > cand[0]:
                cand = (poids, t0, d, unite, copies, s)
        if cand is None:
            break
        _w, t0, d, unite, copies, s = cand
        cuts = couper_zone(unite, copies, pics)
        bornes = [0] + cuts + [unite]
        for k, q in enumerate(copies):
            for i in range(len(bornes) - 1):
                secs.append({"b0": q + bornes[i], "b1": q + bornes[i + 1] - 1,
                             "fam": i, "zone": len(Z), "copie": k})
            for x in range(q, min(n, q + unite)):
                libre[x] = False
        Z.append({"t0": t0, "t1": copies[-1] + unite, "d": d, "unite": unite,
                  "copies": copies, "cuts": cuts, "score": s})
        if verbose:
            print(f"    zone {[q+1 for q in copies]} · décalage {d} · "
                  f"unité {unite} · coupes internes {cuts} · score {s:.2f}")

    trous, i = [], 0
    while i < n:
        if libre[i]:
            j = i
            while j + 1 < n and libre[j + 1]:
                j += 1
            trous.append((i, j))
            i = j + 1
        else:
            i += 1
    secs.sort(key=lambda x: x["b0"])
    for x in secs:
        x["label"] = LETTRES[(x["zone"] * 4 + x["fam"]) % 26]
        x["nu"] = x["label"]
    return {"zones": Z, "sections": secs, "trous": trous, "seuil": seuil,
            "n": n, "S": S, "pics": pics}


def coupes(stem, seuil=None) -> list[int]:
    """Les frontières que les zones imposent — la première intention de découpe."""
    a = zones(stem, seuil)
    c = {s["b0"] for s in a["sections"]} | {s["b1"] + 1 for s in a["sections"]}
    return sorted(x for x in c if 0 < x < a["n"])


# ── le juge : ce que le critère fait le mieux ────────────────────────────────

def dist_voix(stem, spans):
    """La distance de voix entre plages, RAMENÉE À SON SEUIL : < 1 = même section.

    Sous cette forme, `mots4` peut mélanger deux critères d'échelles différentes
    (la voix et les lettres de basse) : chacun est en multiples de son propre
    seuil. `nan` = la voix se tait, c'est aux lettres de trancher.
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

    Le seul régime où le critère est validé : sur des plages ALIGNÉES il donne la
    bonne lettre à 89 % des sections annotées. Lâché sur toutes les fenêtres
    glissantes il s'écroule — mesuré, voir `docs/known_issues.md`.
    """
    DV, th = dist_voix(stem, spans)
    m = len(spans)
    muet = [not np.isfinite(DV[i, [j for j in range(m) if j != i]]).any()
            for i in range(m)]
    lab = [-1] * m
    for i in range(m):
        if muet[i]:
            continue
        avant = [j for j in range(i) if not muet[j] and np.isfinite(DV[i, j])]
        if not avant:
            continue
        j = min(avant, key=lambda j: DV[i, j])
        if DV[i, j] <= 1.0:
            lab[i] = lab[j] if lab[j] >= 0 else j
    return lab, muet, th


def main():
    from ssm_zoo import SONGS, AUDIO, gt_sections
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or list(SONGS)
    tot = [0, 0, 0]
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        a = zones(stem, verbose="-v" in sys.argv)
        gt = gt_sections(stem)["sections"]
        c = set(coupes(stem))
        g = {s["b0"] for s in gt[1:]}
        bons = len(c & g)
        tot[0] += bons; tot[1] += len(c); tot[2] += len(g)
        print(f"\n{title}  ({a['n']} mes., seuil {a['seuil']:.2f}, "
              f"{len(a['zones'])} zones)")
        for z in a["zones"]:
            print(f"   zone mes.{z['t0']+1}-{z['t1']} · décalage {z['d']} · "
                  f"unité {z['unite']} × {len(z['copies'])} · coupes {z['cuts']}")
        print("  nous " + " ".join(f"{s['label']}{s['b0']+1}-{s['b1']+1}"
                                   for s in a["sections"]))
        print("  toi  " + " ".join(f"{s['label']}{s['b0']+1}-{s['b1']+1}"
                                   for s in gt))
        print(f"  -> {bons}/{len(c)} justes, {bons}/{len(g)} des tiennes")
    if tot[1]:
        print(f"\nTOTAL  précision {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.0%}  ·  "
              f"rappel {tot[0]}/{tot[2]} = {tot[0]/tot[2]:.0%}")


if __name__ == "__main__":
    main()
