"""Basse+harmonie, voix, timbre — les trois profils de changement superposés.

    .venv/bin/python scripts/three_profiles.py [<stem> ...]
        -> /plots/three_profiles.html + une page par morceau (audio + playhead)

Louis, 2026-08-12 : « superpose-moi les profils de changement des matrices SSM
basse + harmonie, voix et timbre pour voir comment on peut différencier les
sections. »

POURQUOI CES TROIS-LÀ, et pas les sept. Elles ne disent pas la même chose, et
c'est tout l'intérêt de les mettre sur le même axe :

  * **basse + harmonie** — ce qui est JOUÉ. C'est la seule des trois qui a le
    droit de trancher (règle de Louis, 2026-08-12 : « le timbre seul ne devrait
    jamais servir à trancher des sections »). Ses monts larges et pseudo-
    symétriques sont, eux, des marqueurs sûrs : 21 monts sur les douze morceaux,
    21 contiennent une de ses frontières.
  * **voix** — QUI chante, et surtout QUAND personne ne chante. Le couloir doré
    en fond marque les demi-mesures muettes : c'est l'indicateur solo / pont /
    intro, et il ne dépend d'aucun pic.
  * **timbre** — QUI joue (MFCC). Sur la pop moderne c'est souvent la plus
    lisible des trois (This Love : contraste 0,81 contre 0,18 pour la basse),
    parce qu'une section s'y annonce par l'arrivée d'un instrument. Mais elle
    bouge aussi pour un simple effet de production, d'où son statut : elle
    FABRIQUE des hypothèses, elle ne les valide pas.

CE QU'IL FAUT REGARDER SUR LA PAGE. Chaque courbe porte ses propres pics (petits
triangles de sa couleur). Trois cas se lisent d'un coup d'œil :

  1. les trois pointent au même endroit → frontière franche, aucun doute ;
  2. le timbre seul pointe → changement d'arrangement à l'intérieur d'une
     section (un instrument entre) — à ne PAS couper ;
  3. la voix s'éteint sur plusieurs mesures → pont ou solo, même si les deux
     autres courbes sont plates.

En dessous, les entités de l'agglomération (`bpe_lab`), ce qu'on écrit
aujourd'hui, et le découpage de Louis, sur le même axe.
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
from change_curves import (load_curves, draw_curves, pics_bar,   # noqa: E402
                           COL, MUET, TROIS)
from hard_prior_sections import colourmap                  # noqa: E402
from vote_fill import fill, fig2b64_fixed, PLOT_L, PLOT_R, INK, HARD, MIN_VOTES  # noqa: E402
import bpe_lab as BP                                       # noqa: E402
import order_bundle                                        # noqa: E402
import order_lab as OL                                     # noqa: E402

OUTDIR = HERE / "docs" / "plots"


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    C = load_curves(stem, n)
    m, hb, lines = C["m"], C["hb"], None
    R = fill(b, stem)
    st = BP.merges(R["mot"])
    kstop = BP.arret(st, R["cuts"])
    ents = BP.entites(st, R["x0"], n, kstop)
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    today = OL.new_sections(b)[0]
    mute = C["mute"]

    strips = [("les entités", ents), ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))
    rows = 2 + len(strips)                      # courbes · voix · bandes
    fig, axs = plt.subplots(
        rows, 1, figsize=(13.0, 3.4 + 0.62 * len(strips)), facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [5.2, 0.75] + [1.15] * len(strips),
                     "hspace": 0.0})
    x = C["x"]

    def deco(ax, lab, colour=INK, last=False, peaks=True):
        ax.set_xlim(0, n); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9.5,
                      color=colour)
        for s in ax.spines.values():
            s.set_visible(False)
        if peaks:
            for j in R["cuts"]:
                ax.axvline(R["x0"][j], color=HARD, lw=1.2, alpha=0.55,
                           zorder=1)
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.9, alpha=0.8, zorder=5)
        if last:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])

    # 1 ── les trois courbes, superposées
    draw_curves(axs[0], C, n)
    deco(axs[0], "profils de\nchangement", INK)

    # 2 ── le couloir voix, en dur : muet ou non
    ax = axs[1]
    ax.fill_between(x, mute.astype(float), color="#c07a1e", lw=0, alpha=0.85,
                    step="mid")
    ax.set_ylim(0, 1.05)
    deco(ax, "personne\nne chante", "#8a6d1f")

    # 3 ── les bandes
    cm = colourmap()
    for k, (ax, (lab, ss)) in enumerate(zip(axs[2:], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.08), w, 0.84,
                                       facecolor=cm(s["label"]),
                                       edgecolor="#fffdf6", lw=1.1, zorder=2))
            if w >= max(4, n * 0.04):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK, zorder=4)
        ax.set_ylim(0, 1)
        deco(ax, lab, INK, last=(k == len(strips) - 1))

    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.99, bottom=0.07)
    img = fig2b64_fixed(fig)

    poids = " · ".join(f"{nom} {C['lignes'].get(nom, {}).get('poids', 0):.0%}"
                       for nom in TROIS)
    ent = "".join(
        f'<button class=blk data-p="[{e["b0"]},{e["b1"] + 1}]">{e["label"]}'
        f'<small>mes. {e["b0"] + 1}–{e["b1"] + 1}</small></button>' for e in ents)

    body = f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>les entités</span>{ent}</div>
<div class=votes><b>Poids de ces trois matrices dans la fusion —</b> {poids}.
Le poids est le contraste : « si je coupe à ses pics, est-ce que ça sépare
vraiment ses blocs ». Une matrice à 0 % ne dit rien sur ce morceau.</div>
</section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — trois profils superposés", body, back=True,
                lede_html=LEDE, back_href="three_profiles.html",
                back_label="tous les morceaux")


LEDE = f"""<div class=lede>Les trois profils de changement sur le même axe :
<b style="color:#2f7dbd">basse + harmonie</b> (ce qui est joué),
<b style="color:#c07a1e">voix</b> (qui chante) et
<b style="color:#8155c6">timbre</b> (qui joue). Les petits triangles sont les
pics propres à chaque courbe. Le <b>fond doré</b> marque les passages où
personne ne chante — l'indicateur pont / solo / intro. Traits rouges : tes
frontières. Traits noirs fins : les pics à {MIN_VOTES} voix ou plus.<br><br>
<b>Trois cas se lisent d'un coup d'œil.</b> (1) Les trois pointent au même
endroit → frontière franche. (2) Le <b style="color:#8155c6">timbre</b> pointe
seul → un instrument entre à l'intérieur d'une section : à ne PAS couper — c'est
ta règle, le timbre fabrique des hypothèses, il ne tranche pas. (3) La voix
s'éteint sur plusieurs mesures → pont ou solo, même si les deux autres courbes
sont plates.<br><br>
En dessous, les entités de l'agglomération, ce qu'on écrit aujourd'hui, et ton
découpage. <b>Touche une entité pour l'écouter.</b></div>"""


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"trois_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="trois_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "three_profiles.html").write_text(page(
        "Trois profils superposés",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede_html=LEDE, back_href="three_profiles.html"))
    print(f"wrote docs/plots/three_profiles.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
