"""On soude la transition la plus fréquente, on recommence : le morceau se
construit tout seul, de deux mesures à seize.

    .venv/bin/python scripts/bpe_lab.py [<stem> ...]
        -> /plots/bpe_lab.html + une page par morceau (audio + tête de lecture)

Louis, 2026-08-12 :

  « Je pense qu'on peut faire beaucoup avec la matrice de transition de
    bi-mesure — si on prend le 1er chiffre le + significatif, ça nous donne une
    transition de 2 barres vers 2 autres barres, on en fait une entité, et on
    rajoute cette entrée à la matrice ; montre ce que ça donnerait si on suit ce
    processus itérativement. »

C'est mot pour mot l'algorithme BPE (byte-pair encoding), celui qui fabrique le
vocabulaire des modèles de langue : compter les paires, souder la plus fréquente,
recompter. Appliqué ici au mot de deux mesures, il construit le morceau par
agglomération — 2 mesures, puis 4, puis 8, puis 16 — **sans qu'on ait à décider
d'une échelle**. C'est précisément ce qui manquait : depuis le début de ce
chantier l'harmonie donne la PÉRIODE et jamais l'ÉCHELLE, et ici l'échelle sort
du comptage.

CE QUE LA PAGE MONTRE. Une ligne par soudure, sur l'axe des mesures et sous une
seule tête de lecture. À chaque ligne, la paire qu'on vient de souder est
encadrée ; le reste garde sa couleur. On lit donc l'ordre dans lequel le morceau
s'agglomère, et où ça s'arrête. En dessous, les entités finales, ce qu'on écrit
aujourd'hui, et le découpage de Louis.

CE QUI EST INTÉRESSANT À REGARDER, et ce n'est pas mesuré :

  * les soudures des premiers tours sont les balancements de 4 mesures
    (`a`+`b`) — la boucle du morceau, pas ses sections ;
  * les tours du milieu font les 8 mesures ;
  * les derniers tours soudent des entités DIFFÉRENTES entre elles
    (couplet+refrain) : c'est le moment où l'agglomération dépasse la section et
    commence à écrire la forme. C'est là qu'il faut savoir s'arrêter, et le
    critère d'arrêt n'est pas écrit — les traits noirs (pics à ≥ 3 voix) sont
    dessinés pour qu'on voie si une soudure les enjambe.

L'ARRÊT, tel qu'il est aujourd'hui : on s'arrête quand plus aucune paire ne se
répète (compte < 2), ou quand la soudure ferait une entité de plus de 32 mesures.
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
from vote_fill import (fill, fig2b64_fixed, SYMCOL, MIN_VOTES,  # noqa: E402
                       PLOT_L, PLOT_R, INK, HARD)
import order_bundle                                        # noqa: E402
import order_lab as OL                                     # noqa: E402

OUTDIR = HERE / "docs" / "plots"
MAX_BI = 16        # 32 mesures : au-delà, une « entité » n'est plus une section
MAX_STEPS = 16     # au-delà, la page devient illisible


def merges(word: str, max_bi=MAX_BI, max_steps=MAX_STEPS):
    """[{paire, compte, jetons}] — l'agglomération, tour par tour.

    Un jeton est `(j0, j1, type)` en bi-mesures ; le `type` est la suite de
    lettres minuscules qu'il recouvre, donc deux jetons de même type sont deux
    endroits où le morceau rejoue exactement la même chose.

    La paire choisie est la plus FRÉQUENTE ; à égalité, la plus courte, puis la
    plus à gauche. On soude toutes ses occurrences d'un coup, de gauche à droite
    et sans chevauchement — sinon `aaa` se mangerait lui-même.
    """
    toks = [(j, j + 1, word[j]) for j in range(len(word))]
    steps = [{"paire": None, "compte": 0, "jetons": list(toks)}]
    for _ in range(max_steps):
        cnt: dict = {}
        for i in range(len(toks) - 1):
            a, b = toks[i][2], toks[i + 1][2]
            if (toks[i + 1][1] - toks[i][0]) > max_bi:
                continue
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
        if not cnt:
            break
        best = max(cnt.items(), key=lambda kv: (kv[1], -len(kv[0][0]) - len(kv[0][1])))
        (pa, pb), c = best
        if c < 2:
            break
        out, i = [], 0
        while i < len(toks):
            if (i + 1 < len(toks) and toks[i][2] == pa and toks[i + 1][2] == pb):
                out.append((toks[i][0], toks[i + 1][1], pa + pb))
                i += 2
            else:
                out.append(toks[i]); i += 1
        toks = out
        steps.append({"paire": (pa, pb), "compte": c, "jetons": list(toks)})
    return steps


def transition(toks):
    """(types, matrice) — qui suit qui, sur les jetons courants."""
    types = sorted({t[2] for t in toks}, key=lambda s: (len(s), s))
    idx = {t: i for i, t in enumerate(types)}
    T = np.zeros((len(types), len(types)))
    for x, y in zip(toks, toks[1:]):
        T[idx[x[2]], idx[y[2]]] += 1
    return types, T


def _colour_of(types):
    """Une couleur par type de jeton, stable d'un tour à l'autre."""
    return {t: SYMCOL[i % len(SYMCOL)] for i, t in enumerate(types)}


def matrices_png(steps) -> str:
    """La matrice de transition au départ et à l'arrivée, côte à côte."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("h", ["#faf6ec", "#9fc0d4", "#1d4d69"])
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.4), facecolor="#fffdf6",
                            gridspec_kw={"wspace": 0.22})
    for ax, st, lab in ((axs[0], steps[0], "au départ — les bi-mesures"),
                        (axs[1], steps[-1], f"après {len(steps) - 1} soudures")):
        types, T = transition(st["jetons"])
        short = [t if len(t) <= 6 else t[:5] + "…" for t in types]
        ax.imshow(T, cmap=cmap, vmin=0, vmax=max(1.0, T.max()), interpolation="nearest")
        for i in range(len(types)):
            for j in range(len(types)):
                if T[i, j]:
                    ax.text(j, i, f"{int(T[i, j])}", ha="center", va="center",
                            fontsize=7.5 if len(types) > 9 else 8.5,
                            color="#fffdf6" if T[i, j] > 0.55 * T.max() else "#4a4438")
        ax.set_xticks(range(len(types))); ax.set_yticks(range(len(types)))
        ax.set_xticklabels(short, fontsize=7.5, rotation=90)
        ax.set_yticklabels(short, fontsize=7.5)
        ax.tick_params(length=0, colors="#4a4438")
        ax.set_title(lab, fontsize=10.5, color="#4a4438", pad=8)
        ax.set_xlabel("suivie de…", fontsize=9, color="#8a8371")
        for sp in ax.spines.values():
            sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.16)
    return fig2b64_fixed(fig)


def traverse(tok, cuts):
    """Cette soudure enjambe-t-elle un pic voté ?"""
    return any(tok[0] < j < tok[1] for j in cuts)


def arret(steps, cuts):
    """Le dernier tour AVANT la première soudure qui enjambe un pic voté.

    Le critère d'arrêt, et c'est là que les voix reviennent. L'agglomération ne
    sait pas s'arrêter toute seule : après avoir fabriqué le bon vocabulaire
    (`abab` et `cccd` sur This Love, `aaaaaabc` sur Blue Lights) elle continue et
    soude le couplet au refrain. Un pic voté par au moins trois matrices est
    précisément l'endroit où le morceau dit « nouvelle section » — une entité qui
    l'enjambe est donc allée trop loin. Sur Blue Lights ça coupe après le tour 5
    (les cinq entités de 16 mesures), sur This Love après le tour 5 aussi.

    Ce n'est PAS validé à l'oreille : c'est le critère le plus simple qui utilise
    ce qu'on a déjà, et la page dessine les soudures fautives en rouge pour qu'on
    puisse le contredire.
    """
    for k, s in enumerate(steps):
        if s["paire"] is None:
            continue
        neuf = s["paire"][0] + s["paire"][1]
        if any(t == neuf and traverse((j0, j1), cuts)
               for j0, j1, t in s["jetons"]):
            return k - 1
    return len(steps) - 1


def entites(steps, a, n, k=-1):
    """Les jetons d'un tour, en mesures, avec une lettre par type."""
    toks = steps[k]["jetons"]
    ren, k, out = {}, 0, []
    for j0, j1, t in toks:
        if t not in ren:
            ren[t] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k % 26]; k += 1
        out.append({"b0": a + 2 * j0, "b1": a + 2 * j1 - 1, "label": ren[t],
                    "type": t})
    return out


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    R = fill(b, stem)
    word, a, J = R["mot"], R["ancre"], R["J"]
    st = merges(word)
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    kstop = arret(st, R["cuts"])
    ents = entites(st, a, n, kstop)
    today = OL.new_sections(b)[0]

    strips = [(f"les entités\n(arrêt tour {kstop})", ents),
              ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))

    # une couleur par type, attribuée sur le vocabulaire FINAL puis héritée par
    # les tours précédents : un jeton garde sa couleur quand il grossit.
    alltypes = []
    for s in st:
        for _j0, _j1, t in s["jetons"]:
            if t not in alltypes:
                alltypes.append(t)
    col = _colour_of(alltypes)

    rows = len(st) + len(strips)
    fig, axs = plt.subplots(
        rows, 1, figsize=(13.0, 0.42 * len(st) + 0.62 * len(strips) + 0.6),
        facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [0.9] * len(st) + [1.3] * len(strips),
                     "hspace": 0.0})

    def deco(ax, lab, colour=INK, last=False):
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8.5,
                      color=colour)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        for j in R["cuts"]:
            ax.axvline(a + 2 * j, color=HARD, lw=1.2, alpha=0.85)
        if last:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])

    for k, (ax, s) in enumerate(zip(axs, st)):
        for j0, j1, t in s["jetons"]:
            neuf = s["paire"] is not None and t == s["paire"][0] + s["paire"][1]
            mal = neuf and traverse((j0, j1), R["cuts"])
            ax.add_patch(plt.Rectangle((a + 2 * j0, 0.16), 2 * (j1 - j0), 0.68,
                                       facecolor=col.get(t, "#e8e2d4"),
                                       edgecolor=GT_LINE if mal else
                                       (INK if neuf else "#fffdf6"),
                                       lw=2.0 if mal else (1.4 if neuf else 0.8),
                                       zorder=4 if mal else (3 if neuf else 2)))
            if (j1 - j0) * 2 >= max(3, n * 0.035):
                ax.text(a + 2 * (j0 + j1) / 2, 0.5,
                        t if len(t) <= 8 else f"{2 * (j1 - j0)} mes.",
                        ha="center", va="center", fontsize=7, color="#4a4438",
                        zorder=4)
        lab = ("le mot" if s["paire"] is None
               else f"{k}. {s['paire'][0]}+{s['paire'][1]} ({s['compte']}×)")
        if k == kstop:
            lab += "   ← arrêt"
        deco(ax, lab, INK if k == kstop else "#4a4438")

    cm = colourmap()
    for k, (ax, (lab, ss)) in enumerate(zip(axs[len(st):], strips)):
        for s_ in ss:
            w = s_["b1"] - s_["b0"] + 1
            ax.add_patch(plt.Rectangle((s_["b0"], 0.08), w, 0.84,
                                       facecolor=cm(s_["label"]),
                                       edgecolor="#fffdf6", lw=1.1))
            if w >= max(4, n * 0.04):
                ax.text(s_["b0"] + w / 2, 0.5, str(s_["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK)
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.55)
        deco(ax, lab, INK, last=(k == len(strips) - 1))

    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.99, bottom=0.06)
    img = fig2b64_fixed(fig)
    mats = matrices_png(st)

    tours = "".join(
        f'<button class=blk data-p="[{a + 2 * s["jetons"][0][0]},'
        f'{a + 2 * s["jetons"][0][1]}]">{k}<small>{s["paire"][0]}+{s["paire"][1]}'
        f' · {s["compte"]}×</small></button>'
        for k, s in enumerate(st) if s["paire"])
    ent = "".join(
        f'<button class=blk data-p="[{e["b0"]},{e["b1"] + 1}]">{e["label"]}'
        f'<small>mes. {e["b0"] + 1}–{e["b1"] + 1} · {e["b1"] - e["b0"] + 1} mes.'
        "</small></button>" for e in ents)

    body = f"""<section><h2>{title} <span class=sub>{n} mesures · {J} bi-mesures
· {len(st) - 1} soudures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>les soudures</span>{tours}</div>
<div class=lane><span class=lab>les entités</span>{ent}</div>
<img src="data:image/png;base64,{mats}" alt="matrices de transition">
<div class=votes>La matrice de transition <b>au départ</b> (bi-mesures) et
<b>à l'arrivée</b> (entités soudées). C'est le même comptage à chaque tour : on
prend la case la plus forte, on soude, on recompte.</div>
</section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — agglomération des bi-mesures", body, back=True,
                lede_html=LEDE, back_href="bpe_lab.html",
                back_label="tous les morceaux")


LEDE = f"""<div class=lede>Ton idée, appliquée telle quelle : on prend la
<b>case la plus forte</b> de la matrice de transition, on <b>soude</b> ces deux
bi-mesures en une entité, on la rajoute à la matrice, et on recommence.<br><br>
Une ligne par soudure. La paire qu'on vient de souder est <b>encadrée en
noir</b> ; chaque couleur est un type d'entité, donc deux rectangles de même
couleur sont deux endroits où le morceau rejoue exactement la même chose. Les
traits noirs verticaux sont les pics à {MIN_VOTES} voix ou plus — ils ne
contraignent rien ici, ils sont dessinés pour qu'on voie si une soudure les
enjambe.<br><br>
On s'arrête quand plus aucune paire ne se répète, ou quand la soudure ferait une
entité de plus de 32 mesures. <b>Le vrai critère d'arrêt n'est pas écrit</b> :
les derniers tours soudent des entités différentes entre elles (couplet+refrain),
et c'est là que l'agglomération dépasse la section.<br><br>
<b>Touche une entité pour l'écouter.</b></div>"""


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"bpe_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="bpe_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "bpe_lab.html").write_text(page(
        "Agglomération des bi-mesures",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede_html=LEDE, back_href="bpe_lab.html"))
    print(f"wrote docs/plots/bpe_lab.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
