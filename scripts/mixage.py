"""Le mélange : chaque ingrédient entre à l'étage où il est compétent.

    .venv/bin/python scripts/mixage.py --bench          # contre la ligne de base
    .venv/bin/python scripts/mixage.py --sweep          # le réglage, en LOSO
    .venv/bin/python scripts/mixage.py <stem>           # un morceau, en détail

Louis, 2026-08-14 : « chaque outil (fills de batterie, cadences…) rajoute une
information, toutes seules elles sont incomplètes, ensemble elles forment un
prior solide. Maintenant qu'on a tous ces ingrédients, trouve comment les
mixer. »

LE PIÈGE À NE PAS FAIRE : moyenner les courbes. Ce ne sont pas sept détecteurs
de la même chose — chacun répond à une question différente, et deux d'entre eux
disent déjà la même chose (les fills SONT le damier de batterie, mesuré : 50 %
contre 51 %). Les moyenner, c'est voter deux fois avec le même bulletin.

CHAQUE INGRÉDIENT, ET LA SEULE QUESTION À LAQUELLE IL RÉPOND :

  ancres dures      « ici, c'est sûr »            92 %, mais 1,4 par morceau
  les 39 critères   « ici, ça change un peu »     un coût continu, partout
  cadences          la PHASE de la grille         88 % de leurs pics sur une phase
  grille relative   l'ESPACEMENT admissible       82 % (multiple de 4)
  zones_voix        l'ÉTENDUE d'une reprise       42 % seule
  LLM               la FORME (qui rejoue qui)     9/10 sur This Love

D'où un DÉCOUPAGE SOUS CONTRAINTES, résolu une fois par morceau en programmation
dynamique, et non un vote :

    coût = Σ frontières [1 − profil(b)]        ← les 39 critères, terme de données
         + Σ segments  −log p(longueur)        ← la grille, apprise en LOSO
         + λ · (nombre de frontières)          ← ce qui empêche de couper partout
    sous contrainte : aucun segment ne contient une ancre dure en son intérieur.

ÉTAT D'AVANCEMENT — l'ordre validé avec Louis, chaque étape mesurée contre la
ligne de base (`scripts/bench_sections.py` : précision 71 %, rappel 72 %) :

  [1] ancres dures + vote des 39 + prior de longueur      ← CE FICHIER, aujourd'hui
  [2] + la phase venue des cadences
  [3] + les étendues de zones_voix
  [4] + la forme du LLM (la seule étape qui coûte de l'argent)

Règle d'arrêt posée d'avance : une étape qui ne bouge pas la justesse en
leave-one-song-out est RETIRÉE, pas gardée parce qu'elle a du sens.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

LMIN, LMAX = 2, 32       # longueurs de section admissibles, en mesures
LISSE = 1.0              # lissage du prior de longueur (Laplace)


# ── le terme de données : les 39 critères ramenés à une courbe ──────────────

def profil(stem):
    """[0,1] par mesure — « à quel point les signaux disent que ça change ici ».

    Une voix par SIGNAL (le max de ses critères), puis la moyenne des sept. Pas
    la moyenne des 39 : sinon l'harmonie, qui a six variantes, pèserait six fois
    plus que la batterie qui en a une — c'est la règle déjà appliquée au vote
    des ancres, et c'est le même piège de double comptage que les fills.
    """
    import ancres
    d = ancres.courbes(stem)
    n = d["n"]
    par_sig: dict[str, np.ndarray] = {}
    for sig, _nom, _g, res, c in d["c"]:
        c = np.nan_to_num(np.asarray(c, float))
        o = np.argsort(np.argsort(c))
        r = o / max(1, len(c) - 1)                 # rangs : les échelles diffèrent
        bar = np.zeros(n)
        for b in range(n):                          # ramener à la mesure par le max
            bar[b] = r[b * res:(b + 1) * res].max() if res > 1 else r[b]
        par_sig[sig] = np.maximum(par_sig.get(sig, np.zeros(n)), bar)
    P = np.mean(np.stack(list(par_sig.values())), axis=0)
    P[0] = 0.0
    return P, n


# ── le prior de longueur, appris (jamais posé à la main) ────────────────────

def prior_longueur(exclure=(), lisse=LISSE):
    """-log p(longueur de section), estimé sur les annotations SAUF `exclure`.

    C'est la forme correcte de la « grille de 4 » : vérifié sur les 20
    annotations, « une section commence sur un multiple de 4 » est faux (44 %)
    mais « l'écart entre deux frontières est un multiple de 4 » est vrai (82 %).
    Le prior est donc sur la LONGUEUR, pas sur la position — et il est appris,
    pas écrit, pour qu'on puisse le sortir en leave-one-song-out.
    """
    import json
    P = HERE / "harmonia_min" / "state" / "sections"
    cnt = Counter()
    for f in sorted(P.glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("validated") or f.stem in exclure:
            continue
        b = sorted({s["b0"] for s in d["sections"]} | {d["n"]})
        for x, y in zip(b, b[1:]):
            if LMIN <= y - x <= LMAX:
                cnt[y - x] += 1
    tot = sum(cnt.values()) + lisse * (LMAX - LMIN + 1)
    return {L: -np.log((cnt[L] + lisse) / tot) for L in range(LMIN, LMAX + 1)}


# ── le découpage sous contraintes ───────────────────────────────────────────

def decouper(stem, lam=1.0, poids_donnees=4.0, exclure_soi=True, dures=None):
    """[frontières] — la programmation dynamique de l'étage 1.

    `lam` est le prix d'une frontière : c'est lui qui décide COMBIEN on en pose,
    et c'est le seul réglage libre. `poids_donnees` met le terme des 39 critères
    à l'échelle du prior de longueur (qui est en nats).
    """
    import ancres
    P, n = profil(stem)
    if dures is None:
        A, _ph, _m, _n, _V = ancres.ancres(stem)
        dures = sorted(b for b, _v, _q, _d, dure in A if dure)
    lp = prior_longueur(exclure=(stem,) if exclure_soi else ())

    INF = float("inf")
    dp = np.full(n + 1, INF)
    prev = np.zeros(n + 1, int)
    dp[0] = 0.0
    for j in range(LMIN, n + 1):
        # aucune ancre dure ne doit se retrouver À L'INTÉRIEUR d'un segment
        for i in range(max(0, j - LMAX), j - LMIN + 1):
            if dp[i] == INF:
                continue
            if any(i < a < j for a in dures):
                continue
            c = dp[i] + lp[j - i]
            if j < n:                                   # la fin n'est pas une frontière
                c += lam + poids_donnees * (1.0 - P[j])
            if c < dp[j]:
                dp[j] = c
                prev[j] = i
    if dp[n] == INF:
        return sorted(dures)
    out, j = [], n
    while j > 0:
        j = prev[j]
        if j > 0:
            out.append(j)
    return sorted(out)


# ── mesure, avec la métrique EXACTE de la ligne de base ─────────────────────

def bench(lam=1.0, poids=4.0, verbeux=True):
    """Précision / rappel des frontières, sur les 12 morceaux du banc.

    Même métrique que `scripts/bench_sections.py` : une frontière est juste si
    elle tombe à la mesure EXACTE. Comparer à autre chose serait tricher.
    """
    from ssm_zoo import SONGS, AUDIO, gt_sections
    bons = nous = toi = 0
    lignes = []
    for stem, titre in SONGS:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        B = set(decouper(stem, lam=lam, poids_donnees=poids))
        G = {s["b0"] for s in gt_sections(stem)["sections"] if s["b0"] > 0}
        b = len(B & G)
        bons += b; nous += len(B); toi += len(G)
        lignes.append(f"{titre:24s} {b:2d}/{len(B):2d} justes · {b:2d}/{len(G):2d} siennes")
    if verbeux:
        print("\n".join(lignes))
        print(f"\n{'TOTAL':24s} précision {bons}/{nous} = {bons / max(1, nous):.0%}"
              f" · rappel {bons}/{toi} = {bons / max(1, toi):.0%}"
              f"   [ligne de base : 71 % / 72 %]")
    return bons / max(1, nous), bons / max(1, toi), nous


def sweep():
    """Le réglage de `lam`, le seul libre — et il est choisi en LOSO."""
    print(f"{'lam':>5}{'poids':>7} | {'précision':>10}{'rappel':>8}{'F1':>7}{'frontières':>12}")
    for poids in (2.0, 4.0, 6.0):
        for lam in (0.0, 0.5, 1.0, 1.5, 2.0):
            p, r, nb = bench(lam=lam, poids=poids, verbeux=False)
            f1 = 2 * p * r / max(1e-9, p + r)
            print(f"{lam:>5.1f}{poids:>7.1f} | {p:>9.0%}{r:>8.0%}{f1:>7.0%}{nb:>12}")


if __name__ == "__main__":
    if "--etape1" in sys.argv:
        etape1()
    elif "--sweep" in sys.argv:
        sweep()
    elif "--bench" in sys.argv:
        bench()
    else:
        for s in [a for a in sys.argv[1:] if not a.startswith("--")]:
            print(s, decouper(s))


# ── ÉTAPE 1, VERSION CORRIGÉE : mots4 inchangé + les ancres en contrainte ───

def une(stem, avec_ancres=True, voix=False):
    """Un morceau : le découpage de la prod, avec ou sans la contrainte d'ancres.

    C'est `bench_sections.une()` mot pour mot, à un paramètre près — sinon la
    comparaison ne mesurerait pas ce qu'on croit.
    """
    import numpy as np
    import mots4, order_bundle, ancres as AN
    from vote_fill import fill
    from ssm_zoo import gt_sections
    b = order_bundle.get(stem)
    n = b["n"]
    Sv = np.nan_to_num(np.asarray(b["M"], float))
    dures = ()
    if avec_ancres:
        A, _ph, _m, _n, _V = AN.ancres(stem)
        dures = tuple(x for x, _v, _q, _d, dure in A if dure)
    essais = {}
    for lien in ("complet", "moyen"):
        R = fill(b, stem, lien=lien)
        se, _d = mots4.sections(R["mot"], R["x0"], n, R["sim"], b["start"],
                                cuts=set(R["cuts"]), Sv=Sv,
                                stem=stem if voix else None, ancres=dures)
        essais[lien] = (mots4.cout(se), se)
    lien = min(essais, key=lambda k: (essais[k][0], k != "complet"))
    nous = essais[lien][1]
    toi = gt_sections(stem)["sections"]
    nb = {s["b0"] for s in nous if s["b0"] > 0}
    tb = {s["b0"] for s in toi if s["b0"] > 0}
    return {"n": n, "nous": nous, "toi": toi, "bons": len(nb & tb),
            "nous_n": len(nb), "toi_n": len(tb), "dures": dures, "lien": lien}


def etape1():
    """La mesure qui décide : `mots4` seul contre `mots4` + ancres dures."""
    from ssm_zoo import SONGS, AUDIO
    import bench_sections as BS
    print(f"{'morceau':<24}{'sans ancres':>14}{'avec':>10}{'ancres dures':>16}")
    T = {False: [0, 0, 0], True: [0, 0, 0]}
    eti = {False: [], True: []}
    for stem, titre in SONGS:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        r0, r1 = une(stem, False), une(stem, True)
        for k, r in ((False, r0), (True, r1)):
            T[k][0] += r["bons"]; T[k][1] += r["nous_n"]; T[k][2] += r["toi_n"]
            eti[k].append(BS.etiquettes(r["nous"], r["toi"], r["n"]))
        d = r1["bons"] - r0["bons"]
        fl = "  +" if d > 0 else ("  −" if d < 0 else "   ")
        print(f"{titre:<24}{r0['bons']:>6}/{r0['nous_n']:<2}{r1['bons']:>8}/{r1['nous_n']:<2}"
              f"{fl}{str(list(r1['dures'])):>16}")
    import numpy as np
    for k, nom in ((False, "mots4 seul"), (True, "mots4 + ancres")):
        b, nn, tn = T[k]
        print(f"{nom:<24} précision {b}/{nn} = {b / max(1, nn):.0%} · "
              f"rappel {b}/{tn} = {b / max(1, tn):.0%} · "
              f"étiquettes {np.mean(eti[k]):.0%}")
