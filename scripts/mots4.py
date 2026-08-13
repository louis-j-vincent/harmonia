"""Les mots de quatre lettres : on les scanne, on les compte, on les compare.

    .venv/bin/python scripts/mots4.py [<stem> ...]
        -> docs/plots/mots4.html + une page par morceau (audio + tête de lecture)

Louis, 2026-08-12, deux messages :

  « Sur la matrice de transition on tient le bon bout mais il faut aller au bout
    des choses. Bein' Green, la phrase `aaabcaabcabadcabcaeadcfghi` : les mots
    optimaux sont `aa abca abca badc abca eadc fghi` (parce que b et e sont très
    similaires — ce sont en fait les mêmes mots) → ça minimise les mots de 4
    lettres possibles, avec après intro et outro. […] Si je mets la matrice de
    transition au carré, j'ai les transitions de niveau 2 ? […] J'aimerais savoir
    quelle opération mathématique me permet de représenter dans une matrice tous
    les mots de 4 lettres qui existent — sinon on peut juste scanner toutes les
    phrases de 4 mots, les rassembler et compter ceux où il y en a le +. »

  « Il faudrait aussi avoir une notion de distance entre chaque lettre (mot) afin
    qu'on puisse établir la distance harmonique entre 2 phrases de 4 mots. »

LA RÉPONSE MATHÉMATIQUE, D'ABORD. **Non, `T²` ne donne pas les mots de longueur
3.** `T²[x,z] = Σ_y T[x,y]·T[y,z]` **somme sur la lettre du milieu**, donc elle
l'oublie : on obtient « combien de chemins de longueur 2 mènent de x à z », jamais
lesquels. C'est vrai aussi en probabilités — `P²` est la loi à deux pas, une
marginale. L'objet exact des mots de 4 lettres est un **tenseur d'ordre 4**
`N[w,x,y,z]`, ou, sous forme de matrice, le **graphe de de Bruijn** : les nœuds
sont les triplets, les arêtes les quadruplets, soit une matrice K³×K³. Aucune
puissance de la matrice K×K ne peut le contenir : elle n'a pas assez de place.

Donc : **sa deuxième idée est la bonne**, on scanne les 4-grammes et on compte.
C'est linéaire en la longueur du morceau, instantané.

CE QUE FAIT CE FICHIER, dans l'ordre.

  1. **La distance entre lettres.** `D[x,y] = 1 − ressemblance moyenne` entre
     toutes les bi-mesures de la lettre x et toutes celles de la lettre y, prise
     dans la matrice de basse. C'est sa demande : deux lettres ne sont plus
     « égales ou différentes », elles sont à une distance. Sur Bein' Green, `b`
     et `e` en sont à 0,1 l'une de l'autre — ce sont bien presque le même mot.
  2. **La distance entre deux phrases** de quatre mots : la moyenne des distances
     lettre à lettre, position par position. `badc` et `eadc` ne diffèrent qu'en
     première position, et de peu : leur distance est ~0,03.
  3. **Le découpage.** On essaie les quatre décalages possibles et on garde celui
     qui **minimise le nombre de phrases DISTINCTES**, distinctes au sens de la
     distance ci-dessus et non de l'égalité stricte. C'est mot pour mot son
     critère (« ça minimise les mots de 4 lettres possibles »), et sur Bein' Green
     il retrouve exactement son découpage : `aa | abca abca badc abca eadc | fghi`.

LE CRITÈRE DE VOIX, GARDÉ SOUS LE COUDE (sa demande du même jour). Le score de
chaque section — la moyenne du bloc diagonal de la matrice de voix — est affiché
sur la page mais ne décide rien. Il départage juste sur les deux cas qu'il a
nommés (This Love 2e A : 0,55 pour le sien contre 0,37 pour le nôtre ; Let It Be
1er A : 0,41 contre 0,14) mais **branché comme déplaceur de frontières il
dégrade**, dans les trois variantes essayées — voir `quatre_mots.caler_voix`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections     # noqa: E402
from peak_profile import page                              # noqa: E402
from hard_prior_sections import colourmap                  # noqa: E402
from change_curves import load_curves, draw_curves         # noqa: E402
from vote_fill import (fill, otsu, fig2b64_fixed, SYMCOL,  # noqa: E402
                       PLOT_L, PLOT_R, INK, HARD)
from quatre_mots import bloc_voix                          # noqa: E402
import order_bundle                                        # noqa: E402
import order_lab as OL                                     # noqa: E402

OUTDIR = HERE / "docs" / "plots"
L4 = 4             # la longueur d'une phrase, en mots de deux mesures
RATIO = 1.5        # deux phrases sont « les mêmes » jusqu'à 1,5 fois ce que vaut
                   # « la même lettre » dans ce morceau (voir `decoupe`)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def dist_lettres(word: str, B):
    """(lettres, S̄, D) — la ressemblance moyenne entre lettres, et sa distance.

    `S̄[x,y]` est la moyenne de toutes les ressemblances entre une bi-mesure de x
    et une bi-mesure de y ; `D = 1 − S̄`. La diagonale est la fidélité d'une
    lettre à elle-même (vide pour une lettre qui n'arrive qu'une fois : il n'y a
    aucune paire à moyenner, et on la met alors à 1).
    """
    lets = sorted(set(word))
    pos = {c: [i for i, ch in enumerate(word) if ch == c] for c in lets}
    K = len(lets)
    S = np.full((K, K), np.nan)
    for i, x in enumerate(lets):
        for j, y in enumerate(lets):
            v = [B[p, q] for p in pos[x] for q in pos[y] if p != q]
            if v:
                S[i, j] = float(np.mean(v))
    D = 1.0 - S
    # une lettre qui n'arrive qu'une fois n'a pas de distance à elle-même : la
    # laisser à 0 en ferait la lettre la plus fidèle du morceau, ce qui est faux.
    interne = np.diag(D).copy()
    moy = float(np.nanmean(interne)) if np.isfinite(interne).any() else 0.2
    for i in range(K):
        if not np.isfinite(D[i, i]):
            D[i, i] = moy
    D = np.where(np.isfinite(D), D, moy)
    return lets, S, D


def dist_phrases(p: str, q: str, lets, D) -> float:
    """La distance harmonique entre deux phrases : moyenne position par position."""
    idx = {c: i for i, c in enumerate(lets)}
    return float(np.mean([D[idx[a], idx[b]] for a, b in zip(p, q)]))


def grouper(phrases, lets, D, seuil):
    """Les phrases groupées par distance (lien moyen). [(indice de groupe)]"""
    lab = [-1] * len(phrases)
    groupes: list[list[int]] = []
    for i, p in enumerate(phrases):
        mis = None
        for g, mem in enumerate(groupes):
            d = np.mean([dist_phrases(p, phrases[m], lets, D) for m in mem])
            if d <= seuil:
                mis = g; break
        if mis is None:
            groupes.append([i]); mis = len(groupes) - 1
        else:
            groupes[mis].append(i)
        lab[i] = mis
    return lab, groupes


def familles(word, B, L=L4):
    """Toutes les phrases de L mots, à TOUTES les positions, groupées.

    On ne découpe rien encore : on glisse une fenêtre de L mots sur le morceau
    entier et on range les fenêtres par ressemblance. Une famille est donc « ce
    motif de quatre mots, partout où il apparaît », y compris à des positions qui
    se chevauchent — c'est le placement qui tranchera.
    """
    lets, S, D = dist_lettres(word, B)
    interne = float(np.mean([D[i, i] for i in range(len(lets))]))
    seuil = RATIO * max(interne, 1e-3)
    pos = list(range(0, len(word) - L + 1))
    ph = [word[i:i + L] for i in pos]
    lab, groupes = grouper(ph, lets, D, seuil)
    fam = []
    for g, mem in enumerate(groupes):
        membres = [ph[m] for m in mem]
        # LA RÉFÉRENCE D'UNE FAMILLE EST SA FORME LA PLUS FRÉQUENTE, pas son
        # premier membre. Sur Every Breath You Take la famille contient `abdb`
        # une fois et `abcb` trois fois : prendre le premier venu donnait `abdb`
        # comme référence, et la règle du prime (« elles ne diffèrent que par la
        # dernière lettre ») ne se déclenchait jamais contre `abca`, puisque
        # `abdb` en diffère aussi par l'avant-dernière.
        ref = max(set(membres), key=lambda t: (membres.count(t), -membres.index(t)))
        fam.append({"type": ref, "occ": [pos[m] for m in mem], "membres": membres})
    return fam, lets, S, D, seuil


def nommer_primes(types, lets, D, tol=0.5):
    """{type: lettre} — A et A′ quand deux sections ne diffèrent que par la fin.

    Louis, 2026-08-12 : « lorsque 2 sections ne divergent que par la dernière
    barre, appelle-les A et A′ par exemple ; fais-en une règle implémentée
    partout. »

    Deux sections portent la même lettre, la seconde marquée d'un prime, quand
    elles ont la MÊME longueur, le même début (toutes les lettres sauf la
    dernière, à la distance près) et une dernière lettre différente. C'est
    l'ouvert et le clos d'une même période : la première et la deuxième cadence.
    Sans cette règle, la deuxième prend une lettre neuve et le morceau paraît
    avoir deux fois plus de sections qu'il n'en a.

    `tol` est en unités de distance entre lettres : le début doit correspondre
    à mieux que la moitié de la distance interne moyenne du morceau — sinon ce
    n'est pas la même section avec une autre fin, c'est une autre section.
    """
    idx = {c: i for i, c in enumerate(lets)}
    interne = float(np.mean([D[i, i] for i in range(len(lets))]))
    seuil = tol * max(interne, 1e-3)
    noms: dict[str, str] = {}
    bases: list[tuple[str, str]] = []          # (type de référence, lettre)
    k = 0
    for t in types:
        mis = None
        for u, lab in bases:
            if len(u) != len(t) or len(t) < 2:
                continue
            # une lettre comparée à ELLE-MÊME vaut 0, pas sa distance interne :
            # sinon deux têtes identiques (`abc` contre `abc`) sortent à 0,08 et
            # dépassent le seuil — c'est ce qui empêchait la règle de se
            # déclencher sur Every Breath You Take.
            tete = [0.0 if a == b else D[idx[a], idx[b]]
                    for a, b in zip(u[:-1], t[:-1])]
            # la dernière lettre doit seulement être AUTRE : si elle était
            # proche, les deux phrases seraient déjà la même famille. Exiger en
            # plus qu'elle soit LOIN (ce que faisait la première version) empêche
            # la règle de se déclencher sur Every Breath You Take, où `abca` et
            # `abcb` ne diffèrent que par une lettre voisine — le cas même que
            # Louis a nommé.
            if tete and float(np.mean(tete)) <= seuil and u[-1] != t[-1]:
                mis = lab + "′"
                break
        if mis is None:
            mis = LETTERS[k % len(LETTERS)]; k += 1
            bases.append((t, mis))
        noms[t] = mis
    return noms


def placer(word, x0, B, Sv=None, L=L4, cuts=()):
    """LA PHRASE LA PLUS FRÉQUENTE D'ABORD, partout où elle est.

    Louis, 2026-08-12, sur This Love : « je ne comprends pas pourquoi on ne crée
    pas tous les B avant les C : c'est exactement pour ça qu'on rate un B et
    qu'on décale le A et le C. L'ordre des choses devrait être : on fusionne la
    phrase de 4 avec le plus d'occurrences, puis la deuxième phrase de 4 avec le
    plus d'occurrences, en cas d'égalité la matrice SSM de la voix tranche. »

    Il a raison, et c'est un défaut de MÉTHODE, pas de réglage : la version
    précédente choisissait un décalage puis pavait de gauche à droite. Un pavage
    de gauche à droite décide de la place d'un B en fonction de ce qui le
    précède, jamais en fonction de ses propres reprises — donc il en rate un, et
    tout ce qui suit se décale.

    Ici : on classe les familles par nombre d'occurrences, on pose la plus
    fréquente à toutes ses positions libres, puis la suivante, etc. Deux familles
    à égalité sont départagées par la VOIX — la moyenne du bloc diagonal de la
    matrice de voix sur leurs occurrences —, ce qui est exactement l'emploi qu'il
    lui a assigné : arbitrer une hésitation, jamais décider seule.
    """
    fam, lets, S, D, seuil = familles(word, B, L)

    def voix_fam(f):
        if Sv is None:
            return 0.0
        v = [bloc_voix(Sv, x0[p], x0[min(p + L, len(x0) - 1)] - 1) for p in f["occ"]]
        return float(np.mean(v)) if v else 0.0

    for f in fam:
        f["voix"] = voix_fam(f)
    # le plus d'occurrences d'abord ; à égalité, la voix tranche ; puis la
    # position, pour rester déterministe.
    fam.sort(key=lambda f: (-len(f["occ"]), -f["voix"], f["occ"][0]))

    pris = [None] * len(word)
    placees = []
    for tour, f in enumerate(fam):
        # DE GAUCHE À DROITE à l'intérieur d'une famille : ses occurrences sont
        # régulièrement espacées, les poser dans l'ordre garde la grille. Les
        # poser par score de voix décroissant part d'une occurrence quelconque et
        # décale toutes les suivantes d'un ou deux mots — Blue Lights s'y
        # fragmentait en `bca` / `abc`. La voix sert à ordonner les FAMILLES,
        # pas les occurrences d'une même famille.
        occ = sorted(f["occ"])
        for p in occ:
            if any(pris[q] is not None for q in range(p, min(p + L, len(word)))):
                continue
            for q in range(p, min(p + L, len(word))):
                pris[q] = len(placees)
            placees.append({"j0": p, "j1": min(p + L, len(word)), "fam": f,
                            "mot": word[p:min(p + L, len(word))],
                            "tour": tour})
    # NE PAS trier `placees` : `pris` contient leurs indices d'insertion.

    # ce qui n'est entré dans aucune phrase forme des blocs (Louis, même jour)
    out, i = [], 0
    while i < len(word):
        if pris[i] is not None:
            k = pris[i]
            out.append(placees[k]); i = placees[k]["j1"]
        else:
            j = i
            while j + 1 < len(word) and pris[j + 1] is None:
                j += 1
            out.append({"j0": i, "j1": j + 1, "fam": None, "tour": -1,
                        "mot": word[i:j + 1], "type": word[i:j + 1]})
            i = j + 1

    types = []
    for s in out:
        t = s["fam"]["type"] if s["fam"] else s["type"]
        if t not in types:
            types.append(t)
    noms = nommer_primes(types, lets, D)
    for s in out:
        t = s["fam"]["type"] if s["fam"] else s["type"]
        s["type"] = t
        s["label"] = noms[t]
        s["reste"] = s["fam"] is None
        # LE PRIME SE JUGE PAR OCCURRENCE, contre la référence de sa famille.
        # Louis : « lorsque 2 sections ne divergent que par la dernière barre,
        # appelle-les A et A′ ». La première version comparait les TYPES entre
        # eux et ne se déclenchait jamais : deux phrases qui ne diffèrent que par
        # leur dernière lettre sont justement celles que le groupage réunit en
        # premier (leur distance vaut le quart d'une distance de lettre), donc
        # elles ne survivent pas comme deux types. Elles survivent comme deux
        # OCCURRENCES d'une même famille — et c'est là qu'on les distingue.
        if s["fam"] and s.get("mot") and s["mot"] != t and len(s["mot"]) == len(t) \
                and s["mot"][:-1] == t[:-1] and s["mot"][-1] != t[-1]:
            s["label"] = s["label"] + "′"
    return out, {"fam": fam, "lets": lets, "S": S, "D": D, "seuil": seuil,
                 "types": types}


def sections(word, x0, n, B, start_bar=0, L=L4, cuts=(), Sv=None):
    """Les sections en mesures."""
    pl, d = placer(word, x0, B, Sv, L, cuts)
    out = []
    for s in pl:
        out.append({"b0": x0[s["j0"]], "b1": x0[s["j1"]] - 1,
                    "label": s["label"] + (" (reste)" if s["reste"] else ""),
                    "nu": s["label"].rstrip("′"), "type": s["type"],
                    "mot": s.get("mot", s["type"]), "tour": s.get("tour", -1)})
    for s in out:
        if s["b1"] < start_bar:
            s["label"], s["nu"] = "intro", "intro"
    d["k"] = len(d["types"]); d["off"] = pl[0]["j0"]; d["pics"] = 0
    d["phrases"] = [s["type"] for s in pl]
    d["queue"] = 0
    return out, d


def matrice_png(d, word) -> str:
    """La distance entre lettres, et les phrases groupées."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("h", ["#1d4d69", "#9fc0d4", "#faf6ec"])
    lets, D = d["lets"], d["D"]
    K = len(lets)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 5.2), facecolor="#fffdf6",
        gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.22})

    ax1.imshow(D, cmap=cmap, vmin=0, vmax=max(0.5, float(D.max())),
               interpolation="nearest")
    for i in range(K):
        for j in range(K):
            ax1.text(j, i, f"{D[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                     fontsize=7.5 if K > 9 else 8.5,
                     color="#fffdf6" if D[i, j] < 0.35 * D.max() else "#4a4438")
    lab = [f"{c}" for c in lets]
    ax1.set_xticks(range(K)); ax1.set_yticks(range(K))
    ax1.set_xticklabels(lab, fontsize=9); ax1.set_yticklabels(lab, fontsize=9)
    ax1.tick_params(length=0, colors="#4a4438")
    for i, t in enumerate(ax1.get_xticklabels() + ax1.get_yticklabels()):
        t.set_bbox(dict(facecolor=SYMCOL[(ord(lets[i % K]) - 97) % len(SYMCOL)],
                        edgecolor="none", boxstyle="round,pad=0.22"))
    ax1.set_title("distance entre lettres  (0 = le même mot)", fontsize=10.5,
                  color="#4a4438", pad=8)

    T = d["types"]
    KT = len(T)
    M = np.zeros((KT, KT))
    for i, p in enumerate(T):
        for j, q in enumerate(T):
            m = min(len(p), len(q))
            M[i, j] = dist_phrases(p[:m], q[:m], d["lets"], d["D"])
    ax2.imshow(M, cmap=cmap, vmin=0, vmax=max(0.5, float(M.max())),
               interpolation="nearest")
    for i in range(KT):
        for j in range(KT):
            ax2.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                     fontsize=7 if KT > 9 else 8.5,
                     color="#fffdf6" if M[i, j] < 0.35 * max(M.max(), 1e-6)
                     else "#4a4438")
    ax2.set_xticks(range(KT)); ax2.set_yticks(range(KT))
    lab2 = [f"{LETTERS[i % 26]}·{t}" for i, t in enumerate(T)]
    ax2.set_xticklabels(lab2, fontsize=7.5, rotation=90, family="monospace")
    ax2.set_yticklabels(lab2, fontsize=7.5, family="monospace")
    ax2.tick_params(length=0, colors="#4a4438")
    ax2.set_title(f"distance entre les phrases retenues  (seuil {d['seuil']:.2f})",
                  fontsize=10.5, color="#4a4438", pad=8)
    for ax in (ax1, ax2):
        for sp in ax.spines.values():
            sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.06)
    return fig2b64_fixed(fig)


def cout(secs) -> float:
    """Le coût épistémique d'un découpage : ce qu'il faut écrire pour le dire.

        dictionnaire : la somme des longueurs des types DISTINCTS
      + partition    : un renvoi par section
      + restes       : leur longueur ENTIÈRE, car ils ne s'expliquent par rien.

    C'est la même longueur de description que dans `quatre_mots.cout`, et elle
    sert ici à une question que Louis n'a pas eu à trancher lui-même : **quel
    lien de groupage ce morceau veut-il ?** Un morceau dont les couplets sont
    rejoués (Grenade) paie cher en lien complet — ses quatre A y prennent quatre
    entrées de dictionnaire au lieu d'une. Un morceau bâti sur des boucles
    copiées (Bein' Green, The Walk) paie cher en lien moyen — le moyen y fusionne
    des sections voisines, donc gonfle les restes. Le moins cher gagne.
    """
    types = {s["type"] for s in secs if "reste" not in s["label"]}
    d = sum(len(t) for t in types)
    p = len(secs)
    r = sum(1 for s in secs if "reste" in s["label"])
    long_restes = sum((s["b1"] - s["b0"] + 1) // 2 for s in secs
                      if "reste" in s["label"])
    return d + p + long_restes


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    Sv = np.nan_to_num(np.asarray(b["M"], float))
    essais = {}
    for lien in ("complet", "moyen"):
        Ri = fill(b, stem, lien=lien)
        si, di = sections(Ri["mot"], Ri["x0"], n, Ri["sim"], b["start"],
                          cuts=set(Ri["cuts"]), Sv=Sv)
        essais[lien] = (cout(si), Ri, si, di)
    # LE MORCEAU CHOISIT SON LIEN, par coût épistémique. Le nôtre n'entre plus
    # dans la boucle : c'est la longueur de description qui tranche.
    lien = min(essais, key=lambda k: (essais[k][0], k != "complet"))
    _c, R, secs, d = essais[lien]
    word, x0, B = R["mot"], R["x0"], R["sim"]
    # LES DEUX LIENS CÔTE À CÔTE (demande de Louis, 2026-08-12). Le lien complet
    # exige qu'une bi-mesure ressemble à TOUS les membres de son groupe, le lien
    # moyen à leur moyenne. Grenade a besoin du moyen (son couplet est rejoué
    # différemment, 17 % de ses paires sont sous le seuil) ; The Walk et Every
    # Breath You Take préfèrent le complet. C'est à l'oreille de trancher.
    autre = "moyen" if lien == "complet" else "complet"
    _ca, Rc, secs_c, _dc = essais[autre]
    Sv = np.nan_to_num(np.asarray(b["M"], float))
    C = load_curves(stem, n)
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    today = OL.new_sections(b)[0]

    strips = [(f"lien {lien.upper()}\n(choisi)", secs), (f"lien {autre}", secs_c),
              ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))
    # LA CHRONOLOGIE : un tour par famille posée, dans l'ordre où on les pose.
    # Louis : « montre-moi toutes les étapes qu'il y a eues chronologiquement
    # pour le choix des sections ».
    tours = sorted({s["tour"] for s in secs if s["tour"] >= 0})
    rows = 2 + len(tours) + len(strips)
    fig, axs = plt.subplots(
        rows, 1, figsize=(13.0, 3.0 + 0.34 * len(tours) + 0.62 * len(strips)),
        facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [5.6, 0.95] + [0.7] * len(tours)
                     + [1.25] * len(strips), "hspace": 0.0})

    def deco(ax, lab, colour=INK, last=False):
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9,
                      color=colour)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        for j in R["cuts"]:
            ax.axvline(x0[j], color=HARD, lw=1.2, alpha=0.8)
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.6, zorder=7)
        if last:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])

    deco(axs[0], "profils de\nchangement", INK)
    draw_curves(axs[0], C, n)

    ax = axs[1]
    for j, ch in enumerate(word):
        ax.add_patch(plt.Rectangle((x0[j], 0.15), x0[j + 1] - x0[j], 0.7,
                                   facecolor=SYMCOL[(ord(ch) - 97) % len(SYMCOL)],
                                   edgecolor="#fffdf6", lw=0.9))
        if n <= 120:
            ax.text((x0[j] + x0[j + 1]) / 2, 0.5, ch, ha="center", va="center",
                    fontsize=7.5, color="#4a4438")
    deco(ax, "le mot", "#4a4438")

    cm = colourmap()
    for k, (ax, t) in enumerate(zip(axs[2:2 + len(tours)], tours)):
        for s in secs:
            if s["tour"] < 0 or s["tour"] > t:
                continue
            neuf = s["tour"] == t
            ax.add_patch(plt.Rectangle(
                (s["b0"], 0.14), s["b1"] - s["b0"] + 1, 0.72,
                facecolor=cm(s["nu"]), alpha=1.0 if neuf else 0.25,
                edgecolor="#fffdf6", lw=0.9,
                hatch="///" if (neuf and "′" in s["label"]) else None))
            if neuf and s["b1"] - s["b0"] + 1 >= max(4, n * 0.04):
                ax.text((s["b0"] + s["b1"] + 1) / 2, 0.5, s["label"], ha="center",
                        va="center", fontsize=7.5, color=INK)
        prem = next(s for s in secs if s["tour"] == t)
        nb = sum(1 for s in secs if s["tour"] == t)
        deco(ax, f"{k + 1}. {prem['type']} × {nb}", "#6f6858")

    for k, (ax, (lab, ss)) in enumerate(zip(axs[2 + len(tours):], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle(
                (s["b0"], 0.08), w, 0.84, facecolor=cm(s.get("nu", s["label"])),
                edgecolor="#fffdf6", lw=1.1,
                hatch="///" if "′" in str(s["label"]) else None))
            if w >= max(4, n * 0.04):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK)
        deco(ax, lab, INK, last=(k == len(strips) - 1))

    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.99, bottom=0.08)
    img = fig2b64_fixed(fig)
    mat = matrice_png(d, word)

    btns = "".join(
        f'<button class=blk data-p="[{s["b0"]},{s["b1"] + 1}]">{s["label"]}'
        f'<small>{s["type"]} · voix {bloc_voix(Sv, s["b0"], s["b1"]):.2f}</small>'
        "</button>" for s in secs)

    body = f"""<section><h2>{title} <span class=sub>{n} mesures ·
{d["k"]} phrases distinctes · {len(d["phrases"])} sections</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>les phrases</span>{btns}</div>
<div class=votes><b>Le mot ({lien}, choisi — coût {essais[lien][0]:.0f}) —</b> <code>{word}</code><br>
<b>Le mot ({autre} — coût {essais[autre][0]:.0f}) —</b> <code>{Rc["mot"]}</code><br>
<b>Le découpage —</b> {" ".join(d["phrases"])}</div>
<img src="data:image/png;base64,{mat}" alt="distances">
<div class=votes><b>À gauche</b>, la distance entre lettres : 0 = c'est le même
mot. <b>À droite</b>, chaque phrase de quatre mots avec sa distance à toutes les
autres ; deux phrases sous le seuil sont la même section. Le <b>voix</b> sur
chaque bouton est le critère mis sous le coude : la cohérence du chant à
l'intérieur de la section. Il ne décide rien ici.</div>
</section>
<audio id=au preload=metadata playsinline src="../audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — les mots de quatre lettres", body, back=True,
                lede_html=LEDE, back_href="mots4.html",
                back_label="tous les morceaux")


LEDE = """<div class=lede><b>La réponse à ta question maths d'abord.</b>
Non, <code>T²</code> ne donne pas les mots de longueur 3 :
<code>T²[x,z] = Σ_y T[x,y]·T[y,z]</code> <b>somme sur la lettre du milieu</b>,
donc elle l'oublie — tu obtiens « combien de chemins mènent de x à z », jamais
lesquels. Vrai aussi en probabilités : <code>P²</code> est la loi à deux pas, une
marginale. L'objet exact des mots de 4 lettres est un <b>tenseur d'ordre 4</b>,
ou en matrice le <b>graphe de de Bruijn</b> (nœuds = triplets, arêtes =
quadruplets, K³×K³). Aucune puissance d'une matrice K×K ne peut le contenir : elle
n'a pas la place. <b>Donc ta deuxième idée est la bonne</b> — on scanne les
4-grammes et on compte, c'est instantané.<br><br>
<b>Et ta distance entre lettres.</b> Deux lettres ne sont plus égales ou
différentes : elles sont à une distance (1 − leur ressemblance moyenne dans la
matrice de basse). La distance entre deux phrases de quatre mots est la moyenne
position par position. <code>badc</code> et <code>eadc</code> ne diffèrent qu'en
première position, et de peu.<br><br>
On essaie les quatre décalages possibles et on garde celui qui <b>minimise le
nombre de phrases distinctes</b> — ton critère, littéralement.</div>"""


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    if "--dump" in sys.argv:
        for stem, title in todo:
            b = order_bundle.get(stem); R = fill(b, stem)
            Sv = np.nan_to_num(np.asarray(b["M"], float))
            secs, d = sections(R["mot"], R["x0"], b["n"], R["sim"], b["start"],
                               cuts=set(R["cuts"]), Sv=Sv)
            lets, D = d["lets"], d["D"]
            proches = [(f"{x}~{y}", f"{D[i, j]:.2f}")
                       for i, x in enumerate(lets) for j, y in enumerate(lets)
                       if i < j and D[i, j] < 0.25]
            print(f"\n{title}  mot {R['mot']}")
            print(f"  décalage {d['off']} · {d['k']} phrases distinctes : "
                  f"{' '.join(d['phrases'])}")
            print(f"  lettres proches : {proches or '-'}")
            gt = gt_sections(stem)
            print("  nous " + " ".join(f"{s['label']}{s['b0']+1}-{s['b1']+1}"
                                       for s in secs))
            if gt:
                print("  toi  " + " ".join(f"{s['label']}{s['b0']+1}-{s['b1']+1}"
                                           for s in gt["sections"]))
        return
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"m4_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="m4_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "mots4.html").write_text(page(
        "Les mots de quatre lettres",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede_html=LEDE, back_href="mots4.html"))
    print(f"wrote docs/plots/mots4.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
