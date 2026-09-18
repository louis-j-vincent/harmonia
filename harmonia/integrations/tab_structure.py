"""La forme du morceau, déduite de la grille posée — au rasoir d'Occam.

Louis, 2026-09-18 : « avec cette vérité-là en tête et la règle qu'un pattern
en général se répète, tu peux déduire la structure finale de la grille →
rasoir d'Occam : c'est l'explication la plus simple, et donc l'écriture la
plus minimale en termes de nombres de sections différentes, qui explique le
morceau. »

POURQUOI C'EST FAISABLE ICI ET PAS AILLEURS. Les détecteurs de section du
projet travaillent sur de l'audio : ils comparent des ressemblances floues et
doivent choisir un seuil. Une grille venue d'un tab est SYMBOLIQUE et exacte —
deux mesures sont identiques ou elles ne le sont pas. La question redevient
celle d'un compresseur : quelle est la plus courte façon d'écrire cette suite
de mesures ?

LE COÛT, dans l'ordre lexicographique :

  1. le nombre de MESURES qu'il faut écrire en tout — la somme des longueurs
     des sections distinctes ;
  2. le nombre de sections différentes ;
  3. le nombre de sections posées — à écriture égale, moins de morceaux.

L'ORDRE COMPTE, et le mettre à l'envers ne marche pas. Compter d'abord les
sections DIFFÉRENTES semble être ce que demande « le moins de sections
différentes », mais ça ne récompense jamais la répétition : la façon la moins
chère d'avoir peu de lettres est d'en faire de très longues. Essayé, mesuré —
This Love sortait « A B C C » avec des blocs de 32 mesures, ce qui n'est pas
une forme, c'est un découpage en tranches. C'est la LONGUEUR ÉCRITE qui fait
payer la répétition : un refrain de 8 mesures coûte 8 la première fois et
zéro les suivantes. Le peu de lettres en découle, il ne se demande pas.

LA LOI DU RETOUR est respectée : une section fait au moins `MINI` mesures,
et seule la DERNIÈRE a le droit d'être plus courte (Louis, 2026-08-16 — une
section est une mini-boucle d'au moins 4 mesures, l'exemption porte sur la
suite, jamais sur le bloc).

CE QUE ÇA NE RÉSOUT PAS, et il faut le dire.

  * L'ÉGALITÉ EST EXACTE. Un couplet dont la dernière mesure change compte
    pour une section neuve. C'est volontaire au premier jet — mieux vaut voir
    deux lettres et décider de les fondre que fondre en douce deux choses
    différentes (« under-fold, never over-fold », 2026-07-30). `variantes()`
    dit ensuite quelles lettres ne diffèrent que par leur fin.
  * LE FAISCEAU EST FINI. Une reprise n'est gratuite que si la même section a
    déjà été posée, donc le coût dépend du chemin ; on garde les 24 meilleurs
    par position. Ce n'est pas une garantie d'optimalité, c'est un compromis
    mesuré.
  * ON NE NOMME PAS. « A », « B », « C » dans l'ordre d'apparition. Dire
    lequel est le refrain demande autre chose que de la répétition.
"""
from __future__ import annotations

#: la loi du retour : une section fait au moins quatre mesures
MINI = 4
#: au-delà, ce n'est plus une section mais un morceau de morceau
MAXI = 32

LETTRES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def mots_par_mesure(segments: list[dict], grille: list[float],
                    temps: list[float]) -> list[tuple]:
    """Une mesure → les accords qui y SONNENT, temps par temps, dédoublonnés.

    Le premier jet prenait les accords qui COMMENCENT dans la mesure. C'était
    faux, et ça cassait exactement les répétitions qu'on cherche : la même
    boucle de quatre accords sortait `· | Cm | Fm | D°` la première fois et
    `G7/B | Cm | Fm | D°` la deuxième, selon qu'un accord avait commencé une
    fraction de temps avant la barre ou après. Deux mesures qui sonnent pareil
    doivent s'écrire pareil — ce qui fait l'identité d'une mesure, c'est ce
    qu'on y entend, pas ce qui s'y déclenche.
    """
    out = []
    for b in range(len(grille) - 1):
        a0, a1 = grille[b], grille[b + 1]
        dedans = []
        for t in temps:
            if not (a0 - 1e-6 <= t < a1 - 1e-6):
                continue
            g = next((x for x in segments
                      if x["t0"] - 1e-6 <= t < x["t1"] - 1e-6), None)
            if g is not None and (not dedans or dedans[-1] != g["texte"]):
                dedans.append(g["texte"])
        out.append(tuple(dedans))
    return out


def forme(mots: list[tuple], mini: int = MINI, maxi: int = MAXI,
          faisceau: int = 24) -> dict:
    """La plus petite écriture qui explique la suite de mesures.

    Rend `{"sections": [(debut, fin, lettre)], "motifs": {lettre: [mesures]},
    "cout": (n_mesures_ecrites, n_lettres, n_sections)}`.
    """
    n = len(mots)
    if n == 0:
        return {"sections": [], "motifs": {}, "cout": (0, 0, 0)}

    # `dp[i]` garde les meilleurs chemins qui expliquent les mesures 0..i.
    # Un FAISCEAU et pas un seul chemin : une reprise n'est gratuite que si la
    # même section a DÉJÀ ÉTÉ POSÉE, donc le coût dépend du chemin, et le
    # meilleur chemin à la mesure 20 n'est pas toujours celui qui permettra la
    # meilleure reprise à la mesure 60.
    #
    # Essayé et rejeté : rendre une reprise gratuite dès que le bloc est
    # APPARU plus tôt, sans qu'il ait été posé (le choix de LZ). La table
    # devient exacte et l'écriture tombe de 72 à 32 mesures sur This Love —
    # mais la forme sort dix lettres pour quatre payées, parce qu'un bloc
    # devient gratuit en citant un passage à cheval sur deux sections. Aucun
    # musicien n'écrit ça. Ce qu'on compte ici, c'est le CHART qu'on écrirait.
    dp: list[list[tuple]] = [[] for _ in range(n + 1)]
    dp[0] = [((0, 0, 0), ())]
    for i in range(1, n + 1):
        cands = []
        # la dernière section a le droit d'être plus courte : l'exemption
        # porte sur la SUITE, jamais sur le bloc
        bas = 1 if i == n else mini
        for L in range(bas, min(maxi, i) + 1):
            j = i - L
            bloc = tuple(mots[j:i])
            for (tw, nd, ns), segs in dp[j]:
                deja = any(tuple(mots[a:b]) == bloc for a, b in segs)
                cout = ((tw, nd, ns + 1) if deja
                        else (tw + L, nd + 1, ns + 1))
                cands.append((cout, segs + ((j, i),)))
        cands.sort(key=lambda x: x[0])
        vus, garde = set(), []
        for cout, segs in cands:
            # deux chemins de même coût et de même dernier bloc mènent au même
            # avenir : on n'en garde qu'un
            cle = (cout, segs[-1])
            if cle in vus:
                continue
            vus.add(cle)
            garde.append((cout, segs))
            if len(garde) >= faisceau:
                break
        dp[i] = garde
    if not dp[n]:
        return {"sections": [], "motifs": {}, "cout": (0, 0, 0)}

    cout, segs = dp[n][0]
    noms: dict[tuple, str] = {}
    sections = []
    for a, b in segs:
        bloc = tuple(mots[a:b])
        if bloc not in noms:
            noms[bloc] = LETTRES[len(noms) % len(LETTRES)]
        sections.append((a, b, noms[bloc]))
    motifs = {lettre: list(bloc) for bloc, lettre in noms.items()}
    return {"sections": sections, "motifs": motifs, "cout": cout}


def mot(sections: list[tuple]) -> str:
    """La forme en une ligne : « A A B A A B C B »."""
    return " ".join(l for _, _, l in sections)


def variantes(motifs: dict) -> list[tuple]:
    """Les lettres qui ne diffèrent que par leur FIN — candidates à fondre.

    L'égalité est exacte, donc un couplet dont la dernière mesure change sort
    en deux lettres. Plutôt que de fondre en douce (« under-fold, never
    over-fold »), on le SIGNALE : à Louis de dire si c'est la même section.
    """
    out = []
    lettres = sorted(motifs)
    for i, x in enumerate(lettres):
        for y in lettres[i + 1:]:
            a, b = motifs[x], motifs[y]
            if len(a) != len(b):
                continue
            diff = [k for k in range(len(a)) if a[k] != b[k]]
            if diff and len(diff) <= max(1, len(a) // 4) \
                    and min(diff) >= len(a) - max(1, len(a) // 4):
                out.append((x, y, len(diff)))
    return out
