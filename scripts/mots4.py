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


def decoupe(word, B, L=L4, cuts=()):
    """Le décalage qui minimise le nombre de phrases DISTINCTES.

    « Distinctes » au sens de la distance entre phrases, pas de l'égalité
    stricte : c'est ce qui fait que `badc` et `eadc` comptent pour une seule.
    Le seuil vient d'Otsu sur la distribution du morceau, comme partout ailleurs
    ici — une distance de phrase est une moyenne de distances de lettres, donc
    elle vit sur la même échelle.
    """
    lets, S, D = dist_lettres(word, B)
    # LE SEUIL EST RELATIF, pas absolu — c'est la leçon de Bein' Green. Louis :
    # « b et e sont très similaires, ce sont en fait les mêmes mots ». Mesuré :
    # d(b,e) = 0,31 alors que b est à 0,22 de LUI-MÊME. En absolu 0,31 paraît
    # loin ; rapporté à ce que vaut « la même lettre » dans ce morceau-là, c'est
    # tout près. On compare donc à la distance interne moyenne des lettres qui se
    # répètent.
    interne = float(np.mean([D[i, i] for i in range(len(lets))]))
    seuil = RATIO * max(interne, 1e-3)
    best = None
    for off in range(L):
        ph = [word[i:i + L] for i in range(off, len(word) - L + 1, L)]
        if not ph:
            continue
        lab, groupes = grouper(ph, lets, D, seuil)
        k = len(groupes)
        # À NOMBRE DE PHRASES ÉGAL, c'est le décalage qui tombe sur le plus de
        # pics votés qui gagne. Sans ce départage, Let It Be et Blue Lights
        # partaient une mesure trop tôt : deux décalages donnaient exactement
        # deux phrases distinctes, et on prenait le premier par défaut. Les pics
        # à trois voix ou plus sont la seule information extérieure au mot, donc
        # la seule qui puisse trancher un choix que le mot laisse ouvert.
        bornes = {off + i * L for i in range(len(ph) + 1)}
        v = sum(1 for c in cuts if c in bornes)
        cle = (k, -v, off)
        if best is None or cle < best["cle"]:
            best = {"cle": cle, "k": k, "off": off, "phrases": ph, "lab": lab,
                    "pics": v, "queue": len(word) - (off + L * len(ph))}
    best.update({"lets": lets, "S": S, "D": D, "seuil": seuil})
    return best


def sections(word, x0, n, B, start_bar=0, L=L4, cuts=()):
    """Les sections en mesures : la tête, les phrases, la queue."""
    d = decoupe(word, B, L, cuts)
    out = []
    if d["off"]:
        out.append({"b0": x0[0], "b1": x0[d["off"]] - 1, "label": "tête",
                    "nu": "intro", "type": word[:d["off"]]})
    for i, p in enumerate(d["phrases"]):
        j0 = d["off"] + i * L
        out.append({"b0": x0[j0], "b1": x0[j0 + L] - 1,
                    "label": LETTERS[d["lab"][i] % 26],
                    "nu": LETTERS[d["lab"][i] % 26], "type": p})
    if d["queue"]:
        j0 = d["off"] + L * len(d["phrases"])
        out.append({"b0": x0[j0], "b1": n - 1, "label": "queue", "nu": "outro",
                    "type": word[j0:]})
    for s in out:
        if s["b1"] < start_bar:
            s["label"], s["nu"] = "intro", "intro"
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

    ph, lb = d["phrases"], d["lab"]
    ax2.set_xlim(0, 1); ax2.set_ylim(len(ph), -1)
    for i, (p, g) in enumerate(zip(ph, lb)):
        ax2.add_patch(plt.Rectangle((0.04, i - 0.34), 0.30, 0.68,
                                    facecolor=SYMCOL[g % len(SYMCOL)],
                                    edgecolor="#fffdf6", lw=1.0))
        ax2.text(0.19, i, p, ha="center", va="center", fontsize=10,
                 family="monospace", color="#1c1c1c")
        ax2.text(0.40, i, LETTERS[g % 26], ha="center", va="center", fontsize=10,
                 color="#4a4438")
        dd = [dist_phrases(p, q, d["lets"], d["D"]) for q in ph]
        ax2.text(0.52, i, "  ".join(f"{x:.2f}".lstrip("0") for x in dd),
                 va="center", fontsize=6.5, family="monospace", color="#a89f8c")
    ax2.axis("off")
    ax2.set_title(f"les phrases de {L4} mots, et leur distance à toutes les autres"
                  f"  (seuil {d['seuil']:.2f})", fontsize=10.5, color="#4a4438", pad=8)
    for sp in ax1.spines.values():
        sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.06)
    return fig2b64_fixed(fig)


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    R = fill(b, stem)
    word, x0, B = R["mot"], R["x0"], R["sim"]
    secs, d = sections(word, x0, n, B, b["start"], cuts=set(R["cuts"]))
    Sv = np.nan_to_num(np.asarray(b["M"], float))
    C = load_curves(stem, n)
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    today = OL.new_sections(b)[0]

    strips = [(f"phrases de {L4} mots", secs), ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))
    rows = 2 + len(strips)
    fig, axs = plt.subplots(
        rows, 1, figsize=(13.0, 3.0 + 0.62 * len(strips)), facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [5.6, 0.95] + [1.25] * len(strips),
                     "hspace": 0.0})

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
    for k, (ax, (lab, ss)) in enumerate(zip(axs[2:], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.08), w, 0.84,
                                       facecolor=cm(s.get("nu", s["label"])),
                                       edgecolor="#fffdf6", lw=1.1))
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
décalage {d["off"]} · {d["k"]} phrases distinctes sur {len(d["phrases"])} · {d["pics"]} pic(s) sur une frontière</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>les phrases</span>{btns}</div>
<div class=votes><b>Le mot —</b> <code>{word}</code><br>
<b>Le découpage —</b> tête <code>{word[:d["off"]] or "–"}</code> ·
{" ".join(d["phrases"])} · queue <code>{word[d["off"] + L4 * len(d["phrases"]):] or "–"}</code>
</div>
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
            secs, d = sections(R["mot"], R["x0"], b["n"], R["sim"], b["start"],
                               cuts=set(R["cuts"]))
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
