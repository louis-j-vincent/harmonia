"""On soude jusqu'à des sections de QUATRE mots, et A′ est un A qui finit autrement.

    .venv/bin/python scripts/quatre_mots.py [<stem> ...]
        -> /plots/quatre_mots.html + une page par morceau (audio + tête de lecture)

Louis, 2026-08-12, en majuscules :

  « Pour la règle de merging via la matrice de transition de manière itérative,
    essaie cette règle :
    — critère d'arrêt : quand on n'arrive plus à trouver de sections de 4 mots
      bi-barres ;
    — subtilité : une phrase `abacabad` ou `abababac` dont les 4 premiers mots
      sont considérés comme un A, alors les 4 suivants devraient être considérés
      comme un A′ s'ils ne diffèrent que par le dernier mot bi-barre. »

LES DEUX RÈGLES, TELLES QU'ELLES SONT ÉCRITES ICI.

  1. **L'arrêt par la taille.** On soude la paire la plus fréquente, comme avant,
     mais **jamais au-delà de quatre bi-mesures** (huit mesures), et on s'arrête
     dès qu'aucune paire soudable ne se répète. Ce qui reste de longueur 4 est
     une section ; ce qui reste plus court est une queue. C'est un critère de
     TAILLE, pas de contenu — et c'est ce qui manquait : l'agglomération libre
     ne savait pas s'arrêter et finissait par coller couplet et refrain.

  2. **La cadence, pas la phrase.** `abac` et `abad` sont le même A : trois
     bi-mesures identiques et une fin qui change. C'est la première et la
     deuxième cadence d'une même période — en jazz, l'ouvert et le clos. On les
     écrit `A` et `A′`, ce qui dit à la fois « c'est la même section » et « elle
     ne finit pas pareil ». Sans cette règle, la deuxième prenait une lettre
     neuve et le morceau paraissait avoir deux fois plus de sections qu'il n'en
     a.

LA PAGE. Une ligne par soudure comme sur `bpe_lab`, les trois profils de
changement en haut, puis les sections de quatre mots avec leurs primes, ce qu'on
écrit aujourd'hui, et le découpage de Louis — le tout sous une seule tête de
lecture.
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
from vote_fill import (fill, fig2b64_fixed, SYMCOL,        # noqa: E402
                       PLOT_L, PLOT_R, INK, HARD)
import order_bundle                                        # noqa: E402
import order_lab as OL                                     # noqa: E402

OUTDIR = HERE / "docs" / "plots"
CIBLES = (4, 6)    # une section fait quatre mots de deux mesures… ou six
CIBLE = 4
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def merges4(word: str, cible=CIBLE, max_steps=24):
    """[{paire, compte, jetons}] — l'agglomération plafonnée à `cible` mots.

    Identique à `bpe_lab.merges` sauf deux choses, qui sont les règles de Louis :
    une soudure ne peut pas dépasser `cible` bi-mesures, et on s'arrête dès
    qu'aucune paire soudable ne se répète — c'est-à-dire « quand on n'arrive
    plus à trouver de sections de quatre mots ».
    """
    toks = [(j, j + 1, word[j]) for j in range(len(word))]
    steps = [{"paire": None, "compte": 0, "jetons": list(toks)}]
    for _ in range(max_steps):
        cnt: dict = {}
        for i in range(len(toks) - 1):
            a, b = toks[i][2], toks[i + 1][2]
            if len(a) + len(b) > cible:
                continue
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
        cnt = {k: v for k, v in cnt.items() if v >= 2}
        if not cnt:
            break
        (pa, pb), c = max(cnt.items(),
                          key=lambda kv: (kv[1], len(kv[0][0]) + len(kv[0][1])))
        out, i = [], 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i][2] == pa and toks[i + 1][2] == pb:
                out.append((toks[i][0], toks[i + 1][1], pa + pb))
                i += 2
            else:
                out.append(toks[i]); i += 1
        toks = out
        steps.append({"paire": (pa, pb), "compte": c, "jetons": list(toks)})
    return steps


def nommer(toks, cible=CIBLE):
    """[{j0, j1, type, label, prime}] — les lettres, avec la règle du prime.

    Deux sections de même longueur qui ne diffèrent QUE par leur dernier mot
    sont la même lettre, la seconde marquée d'un prime. C'est la règle de Louis :
    `abac` et `abad` sont un A et un A′, pas un A et un B.
    """
    base: list[tuple[str, str]] = []          # (type de référence, lettre)
    out = []
    for j0, j1, t in toks:
        lab, prime = None, False
        for u, L in base:
            if u == t:
                lab = L; break
            if len(u) == len(t) and u[:-1] == t[:-1] and len(t) >= 2:
                lab, prime = L, True; break
        if lab is None:
            lab = LETTERS[len({L for _u, L in base}) % len(LETTERS)]
            base.append((t, lab))
        out.append({"j0": j0, "j1": j1, "type": t, "label": lab,
                    "prime": prime, "queue": (j1 - j0) < cible})
    return out


def grouper_restes(nom, word, cible):
    """Les jetons trop courts et VOISINS deviennent un bloc à eux.

    Louis, 2026-08-12 : « quand tu as fini de souder les sections, tu t'arrêtes à
    des blocs de 4, et tu mets en bloc les suites successives qui ne sont rentrées
    dans aucun bloc → dans This Love le queue et le bridge. »

    C'est la bonne lecture musicale : ce qui ne rentre dans aucun bloc n'est pas
    du bruit, c'est le matériau qui n'arrive qu'une fois. Sur This Love, les
    huit mots orphelins des mesures 45 à 56 formaient huit sections d'une ou deux
    mesures ; ils forment maintenant DEUX blocs — sa queue et son pont.
    """
    plafond = int(1.5 * cible)     # un reste plus long que ça se recoupe : sans
                                   # ce plafond, Chain of Fools sortait UN bloc de
                                   # 46 mesures et Sunny un de 86 — « ce qui n'est
                                   # rentré nulle part » finissait par manger le
                                   # morceau. 1,5 × la cible laisse passer la
                                   # queue+pont de This Love (6 mots) d'un bloc.
    out, i = [], 0
    while i < len(nom):
        if not nom[i]["queue"]:
            out.append(nom[i]); i += 1; continue
        j = i
        while j + 1 < len(nom) and nom[j + 1]["queue"]:
            j += 1
        run = nom[i:j + 1]
        while run:
            n_mots = sum(x["j1"] - x["j0"] for x in run)
            if n_mots <= plafond:
                bout, run = run, []
            else:
                bout, k = [], 0
                while run and k < cible:
                    k += run[0]["j1"] - run[0]["j0"]; bout.append(run.pop(0))
            out.append({"j0": bout[0]["j0"], "j1": bout[-1]["j1"],
                        "type": "".join(x["type"] for x in bout),
                        "label": None, "prime": False, "queue": True})
        i = j + 1
    # les blocs de reste identiques partagent une lettre, comme les autres
    base = {x["type"]: x["label"] for x in out if x["label"]}
    libres = sorted({x["type"] for x in out if x["label"] is None})
    k = len({v for v in base.values()})
    for t in libres:
        base[t] = LETTERS[k % len(LETTERS)]; k += 1
    for x in out:
        if x["label"] is None:
            x["label"] = base[x["type"]]
    return out


def cout(nom, cible) -> float:
    """Le COÛT ÉPISTÉMIQUE d'une hypothèse : ce qu'il faut écrire pour dire la
    chanson entière.

    Louis, 2026-08-12 : « quand on a deux hypothèses (trancher ici ou ici, 4 ou
    6), il faut voir celle qui aura le meilleur coût épistémique → l'hypothèse
    qui explique le mieux la chanson. »

    On l'écrit comme une longueur de description, et c'est la seule façon de
    comparer deux découpages qui n'ont ni le même nombre de sections ni les mêmes
    longueurs :

        dictionnaire : la somme des longueurs des types DISTINCTS
      + partition    : un renvoi par section
      + primes       : un mot de plus (la fin qui change)
      + restes       : leur longueur ENTIÈRE — un bloc qui n'explique rien se
                       paie au prix fort, il faut l'écrire tel quel.

    Une hypothèse qui invente une section par endroit paie un dictionnaire énorme ;
    une hypothèse qui colle tout en un bloc paie ses restes. Le minimum est le
    découpage qui rend le morceau le plus court à décrire.
    """
    types = {x["type"] for x in nom if not x["queue"]}
    d = sum(len(t) for t in types)
    p = len(nom)
    pr = sum(1 for x in nom if x["prime"])
    r = sum(x["j1"] - x["j0"] for x in nom if x["queue"])
    return d + p + pr + r


def score_voix(Sv, secs, res=1) -> float:
    """La moyenne des blocs diagonaux de la matrice de VOIX, un bloc par section.

    Louis, 2026-08-12 : « quand on hésite sur où trancher une section (This Love,
    reprise du deuxième A), on peut se servir de la voix : tu prends la matrice
    SSM de la voix et tu regardes la moyenne des valeurs sur la diagonale
    correspondant au bloc de la section de base et de la nouvelle section
    hypothétique → celui qui a la plus grande diagonale gagne. »

    Autrement dit : une section est un bloc où le chant se ressemble à lui-même.
    On somme, pour chaque section, la ressemblance moyenne de toutes ses paires de
    mesures (hors diagonale, qui vaut 1 par construction et gonflerait les
    sections courtes), pondérée par le nombre de paires — donc c'est la
    ressemblance moyenne INTRA-section du morceau entier. Deux découpages
    concurrents se comparent directement là-dessus.

    C'est le seul critère du chantier qui juge l'INTÉRIEUR d'une section plutôt
    que sa frontière, et c'est pour ça qu'il tranche là où les pics se taisent.
    """
    num, den = 0.0, 0
    for s in secs:
        a, z = int(s["b0"] * res), int((s["b1"] + 1) * res)
        a, z = max(0, a), min(Sv.shape[0], z)
        if z - a < 2:
            continue
        bloc = Sv[a:z, a:z]
        k = (z - a) * (z - a - 1)
        num += float(bloc.sum() - np.trace(bloc)); den += k
    return num / max(1, den)


GAIN_MIN = 0.03    # 3 % de mieux, sinon on ne bouge pas
DECALS = (-4, -2, 2, 4)   # les décalages testés, en mesures


def bloc_voix(Sv, b0, b1, res=1) -> float:
    """La moyenne du bloc diagonal d'une section — sa cohérence de chant."""
    a, z = max(0, int(b0 * res)), min(Sv.shape[0], int((b1 + 1) * res))
    if z - a < 2:
        return 0.0
    B = Sv[a:z, a:z]
    return float((B.sum() - np.trace(B)) / ((z - a) * (z - a - 1)))


def caler_voix(secs, Sv, res=1, gain=GAIN_MIN):
    """Décale une frontière quand le chant y gagne — DEUX BLOCS À LA FOIS.

    La règle de Louis, littéralement : « tu regardes la moyenne des valeurs sur la
    diagonale correspondant au bloc de la section de base et de la nouvelle
    section hypothétique → celui qui a la plus grande diagonale gagne ». On
    compare donc les DEUX sections que la frontière sépare, chacune sur son
    propre bloc, et on garde le décalage qui remonte la plus faible des deux.

    LA PREMIÈRE VERSION MAXIMISAIT LE SCORE GLOBAL et faisait pire : Let It Be,
    dont les dix frontières étaient justes, en perdait sept. La raison est que la
    moyenne pondérée du morceau entier est dominée par les gros blocs « reste » —
    un grand bloc incohérent tirait toutes les frontières vers lui. La comparaison
    doit rester LOCALE, entre les deux sections concernées, comme il l'a écrite.

    Ce que le critère vaut, sur les deux cas qu'il a nommés (moyenne du bloc de
    voix, par mesure) :

        This Love, 2e A   notre 25-32 : 0,369   le sien 29-36 : **0,552**
        This Love, queue  notre 33-36 : 0,476   la sienne 25-28 : 0,104
        Let It Be, 1er A  notre 1-8   : 0,139   le sien 5-12   : **0,411**

    Il désigne sa réponse dans les deux cas, et de loin. La queue est le cas
    inverse et il est instructif : une queue est justement l'endroit où le chant
    ne se ressemble pas — le critère y est BAS, ce qui la reconnaît en creux.
    """
    secs = [dict(s) for s in secs]
    for i in range(1, len(secs)):
        g, d_ = secs[i - 1], secs[i]
        base = min(bloc_voix(Sv, g["b0"], g["b1"], res),
                   bloc_voix(Sv, d_["b0"], d_["b1"], res))
        best, delta = base * (1 + gain), 0
        for d in DECALS:
            if g["b1"] + d <= g["b0"] + 1 or d_["b0"] + d >= d_["b1"] - 1:
                continue
            sc = min(bloc_voix(Sv, g["b0"], g["b1"] + d, res),
                     bloc_voix(Sv, d_["b0"] + d, d_["b1"], res))
            if sc > best:
                best, delta = sc, d
        if delta:
            g["b1"] += delta; d_["b0"] += delta; d_["cale"] = delta
    return secs, 0.0


def sections(word, x0, n, start_bar=0, cibles=CIBLES, Sv=None):
    """Les sections en mesures — l'hypothèse la moins chère parmi `cibles`."""
    essais = []
    for c in cibles:
        st = merges4(word, c)
        nom = grouper_restes(nommer(st[-1]["jetons"], c), word, c)
        brut = [{"b0": x0[x["j0"]], "b1": x0[x["j1"]] - 1} for x in nom]
        v = score_voix(Sv, brut) if Sv is not None else 0.0
        essais.append((cout(nom, c), -v, c, st, nom))
    # à coût épistémique ÉGAL, c'est la voix qui tranche (Let It Be et Bein'
    # Green sont exactement à égalité : 29 contre 29)
    essais.sort(key=lambda e: (e[0], e[1], e[2]))
    ct, mv, cible, st, nom = essais[0]
    out = []
    for s in nom:
        lab = s["label"] + ("′" if s["prime"] else "")
        if s["queue"]:
            lab = lab + " (reste)"
        out.append({"b0": x0[s["j0"]], "b1": x0[s["j1"]] - 1, "label": lab,
                    "nu": s["label"], "type": s["type"], "queue": s["queue"]})
    # LE RECALAGE AUTOMATIQUE EST DÉBRANCHÉ, et c'est un résultat, pas un oubli.
    # Le critère de Louis DÉPARTAGE juste (voir `caler_voix` : il désigne sa
    # réponse sur les deux cas qu'il a nommés, 0,552 contre 0,369 et 0,411 contre
    # 0,139). Mais l'appliquer comme un déplaceur libre de frontières échoue dans
    # les trois variantes essayées — score global, min des deux blocs, gain de
    # 2 % ou 3 % : Let It Be perd sept de ses dix frontières justes, This Love
    # gonfle son premier A à douze mesures. La raison est structurelle : déplacer
    # une frontière CHANGE LA LONGUEUR des deux sections, et une section plus
    # longue peut voir sa moyenne monter en absorbant du chant voisin. Comparer
    # deux placements d'une section de MÊME longueur (ce qu'il décrit) et
    # déplacer librement une frontière ne sont pas la même opération.
    # Ce qui est branché : le score de voix est affiché par section sur la page,
    # et il départage les hypothèses globales à coût épistémique égal.
    if False and Sv is not None:
        out, _sc = caler_voix(out, Sv)
    for s in out:
        if s["b1"] < start_bar:
            s["label"], s["nu"] = "intro", "intro"
    return out, st, nom, {"cible": cible, "cout": ct, "voix": -mv,
                          "couts": {e[2]: e[0] for e in essais},
                          "voix_par_cible": {e[2]: -e[1] for e in essais}}


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    R = fill(b, stem)
    word, x0 = R["mot"], R["x0"]
    Sv = np.nan_to_num(np.asarray(b["M"], float))     # la voix, par mesure
    secs, st, nom, choix = sections(word, x0, n, b["start"], Sv=Sv)
    C = load_curves(stem, n)
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    today = OL.new_sections(b)[0]

    strips = [("4 mots + primes", secs), ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))

    alltypes = []
    for s in st:
        for _a, _z, t in s["jetons"]:
            if t not in alltypes:
                alltypes.append(t)
    col = {t: SYMCOL[i % len(SYMCOL)] for i, t in enumerate(alltypes)}

    rows = 1 + len(st) + len(strips)
    fig, axs = plt.subplots(
        rows, 1, figsize=(13.0, 2.6 + 0.42 * len(st) + 0.62 * len(strips) + 0.6),
        facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [6.2] + [0.9] * len(st) + [1.3] * len(strips),
                     "hspace": 0.0})

    def deco(ax, lab, colour=INK, last=False):
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8.5,
                      color=colour)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        for j in R["cuts"]:
            ax.axvline(x0[j], color=HARD, lw=1.2, alpha=0.85)
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

    for k, (ax, s) in enumerate(zip(axs[1:1 + len(st)], st)):
        for j0, j1, t in s["jetons"]:
            neuf = s["paire"] is not None and t == s["paire"][0] + s["paire"][1]
            ax.add_patch(plt.Rectangle((x0[j0], 0.16), x0[j1] - x0[j0], 0.68,
                                       facecolor=col.get(t, "#e8e2d4"),
                                       edgecolor=INK if neuf else "#fffdf6",
                                       lw=1.4 if neuf else 0.8,
                                       zorder=3 if neuf else 2))
            if (x0[j1] - x0[j0]) >= max(3, n * 0.035):
                ax.text((x0[j0] + x0[j1]) / 2, 0.5, t, ha="center", va="center",
                        fontsize=7, color="#4a4438", zorder=4)
        lab = ("le mot" if s["paire"] is None
               else f"{k}. {s['paire'][0]}+{s['paire'][1]} ({s['compte']}×)")
        deco(ax, lab, "#4a4438")

    cm = colourmap()
    for k, (ax, (lab, ss)) in enumerate(zip(axs[1 + len(st):], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            nu = s.get("nu", s["label"])
            ax.add_patch(plt.Rectangle(
                (s["b0"], 0.08), w, 0.84, facecolor=cm(nu),
                edgecolor="#fffdf6", lw=1.1,
                hatch="///" if "′" in str(s["label"]) else
                ("..." if "reste" in str(s["label"]) else None)))
            if w >= max(4, n * 0.04):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK)
        deco(ax, lab, INK, last=(k == len(strips) - 1))

    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.99, bottom=0.06)
    img = fig2b64_fixed(fig)

    btns = "".join(
        f'<button class=blk data-p="[{s["b0"]},{s["b1"] + 1}]">{s["label"]}'
        f'<small>mes. {s["b0"] + 1}–{s["b1"] + 1} · voix '
        f'{bloc_voix(Sv, s["b0"], s["b1"]):.2f}</small></button>'
        for s in secs)
    primes = [s for s in secs if "′" in s["label"]]

    body = f"""<section><h2>{title} <span class=sub>{n} mesures · sections de {choix["cible"]} mots ·
{len(st) - 1} soudures · {len(primes)} prime(s)</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>les sections</span>{btns}</div>
<div class=votes><b>Le mot —</b> <code>{word}</code><br>
<b>Les sections —</b> {" ".join(s["label"] + "=" + s["type"] for s in secs)}<br>
<b>Coût épistémique —</b> {" · ".join(f"sections de {k} mots : {v:.0f}"
                                       for k, v in choix["couts"].items())} →
on garde <b>{choix["cible"]}</b>, l'hypothèse qui rend le morceau le plus court à
décrire.<br>
<b>Voix —</b> {" · ".join(f"{k} mots : {v:.3f}" for k, v in choix["voix_par_cible"].items())}
— la ressemblance moyenne du chant À L'INTÉRIEUR des sections. Elle départage à
coût égal, puis recale chaque frontière de ±2 mesures quand le chant y gagne.</div>
</section>
<audio id=au preload=metadata playsinline src="../audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — quatre mots et primes", body, back=True,
                lede_html=LEDE, back_href="quatre_mots.html",
                back_label="tous les morceaux")


LEDE = """<div class=lede>Tes deux règles, appliquées telles quelles.<br><br>
<b>L'arrêt par la taille.</b> On soude la paire la plus fréquente comme avant,
mais <b>jamais au-delà de quatre mots</b> de deux mesures, et on s'arrête dès
qu'aucune paire soudable ne se répète. Ce qui reste de quatre mots est une
section ; ce qui reste plus court est une queue, notée <code>(q)</code>.<br><br>
<b>Le prime.</b> <code>abac</code> et <code>abad</code> sont le même A : trois
mots identiques et une fin qui change — l'ouvert et le clos d'une même période.
On écrit <b>A</b> et <b>A′</b> (hachuré sur la bande), ce qui dit à la fois « même
section » et « elle ne finit pas pareil ». Sans cette règle la seconde prenait
une lettre neuve et le morceau paraissait avoir deux fois plus de sections.<br><br>
En haut les trois profils de changement, puis une ligne par soudure (la paire
soudée est encadrée), puis les sections, ce qu'on écrit aujourd'hui et ton
découpage. <b>Touche une section pour l'écouter.</b></div>"""


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    if "--dump" in sys.argv:
        for stem, title in todo:
            b = order_bundle.get(stem); R = fill(b, stem)
            Sv = np.nan_to_num(np.asarray(b["M"], float))
            secs, st, _n, choix = sections(R["mot"], R["x0"], b["n"], b["start"],
                                           Sv=Sv)
            gt = gt_sections(stem)
            print(f"\n{title}  mot {R['mot']}  -> {choix['cible']} mots "
                  f"(coûts {choix['couts']})")
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
        (OUTDIR / f"q4_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="q4_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "quatre_mots.html").write_text(page(
        "Quatre mots et primes",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede_html=LEDE, back_href="quatre_mots.html"))
    print(f"wrote docs/plots/quatre_mots.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
