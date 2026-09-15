"""Le vocabulaire trouvé, et le pavage qu'il produit — à regarder et à écouter.

    .venv/bin/python scripts/vocab_lab.py  ->  /plots/vocab_lab.html + une page par morceau

Louis, 2026-08-12 : « on n'a pas besoin de les séparer. Quand on identifie les A,
B et C, on remplit au bon endroit […] les pics servent juste à caler quelques
frontières dont on est sûrs afin de trancher entre plusieurs hypothèses de
remplissage », puis « arrête de mesurer, donne-moi juste les démos en visuel ».

CE QUE LA PAGE MONTRE, tout sur l'axe des mesures et sous une seule tête de
lecture :

  * **un couloir par motif** trouvé dans le morceau — chaque rectangle est une
    occurrence, et la longueur du motif est écrite dessus. C'est le
    « vocabulaire » : ce que le morceau rejoue, avant qu'on ait décidé quoi que
    ce soit sur les sections.
  * **le pavage** que la programmation dynamique en tire, avec les pics durs qui
    interdisent qu'une pose les enjambe.
  * **ce qu'on écrit aujourd'hui** (la recherche gauche→droite) et **ton
    découpage**, pour comparer.

Les couleurs des motifs et des sections sont indépendantes : un motif n'est pas
une section tant que le pavage ne l'a pas posé.
"""
from __future__ import annotations

import base64
import io
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

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections   # noqa: E402
from peak_profile import page                            # noqa: E402
from hard_prior_sections import colourmap                # noqa: E402
import order_bundle                                      # noqa: E402
import order_lab as OL                                   # noqa: E402
import vocab_tiling as VT                                # noqa: E402

OUTDIR = HERE / "docs" / "plots"
PLOT_L, PLOT_R = 0.155, 0.995
INK = "#1c1c1c"
HARD = "#0d2437"
MOTIF = ["#a8c8dc", "#e0c9a6", "#c4d8bf", "#dcc0c8", "#cdc6e0", "#e6d9a8"]
NVOC = 6            # combien de motifs on dessine


def fig2b64_fixed(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=112)      # marges fixes : la tête de lecture
    plt.close(fig)
    try:
        from PIL import Image
        b.seek(0)
        im = Image.open(b).convert("RGB").quantize(colors=128, method=2)
        s = io.BytesIO(); im.save(s, format="PNG", optimize=True)
        if s.tell() < b.getbuffer().nbytes:
            b = s
    except Exception:
        pass
    return base64.b64encode(b.getvalue()).decode()


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid, hard = b["n"], b["grid"], b["hard"]
    voc = VT.vocabulary(b)[:NVOC]
    pave = VT.sections(b) or []
    today = OL.new_sections(b)[0]
    gt = gt_sections(stem)
    strips = [("le pavage", pave), ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))

    rows = len(voc) + len(strips)
    fig, axs = plt.subplots(rows, 1, figsize=(13.0, 0.46 * len(voc) + 0.62 * len(strips) + 0.6),
                            facecolor="#fffdf6",
                            gridspec_kw={"height_ratios": [0.8] * len(voc) + [1.15] * len(strips),
                                         "hspace": 0.0})
    axs = np.atleast_1d(axs)

    def deco(ax, lab, colour=INK, last=False):
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9.5,
                      color=colour)
        for s in ax.spines.values():
            s.set_visible(False)
        for h in hard:
            ax.axvline(h, color=HARD, lw=1.5, alpha=0.9)
        if last:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])

    for i, (ax, m) in enumerate(zip(axs, voc)):
        col = MOTIF[i % len(MOTIF)]
        for p in m["occ"]:
            ax.add_patch(plt.Rectangle((p, 0.18), m["L"], 0.64, facecolor=col,
                                       edgecolor="#fffdf6", lw=1.0))
        deco(ax, f"motif {i+1} · {m['L']} mes. × {len(m['occ'])}", "#4a4438")

    cm = colourmap()
    for j, (ax, (lab, ss)) in enumerate(zip(axs[len(voc):], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.08), w, 0.84,
                                       facecolor=cm(s["label"]),
                                       edgecolor="#fffdf6", lw=1.1))
            if w >= max(2, n * 0.035):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK)
        if gt:
            for g in [x["b0"] for x in gt["sections"][1:]]:
                ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.55)
        deco(ax, lab, INK, last=(j == len(strips) - 1))

    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.985, bottom=0.10)
    img = fig2b64_fixed(fig)

    btns = "".join(
        f'<button class=blk data-p="[{s["b0"]},{s["b1"]+1}]">{s["label"]}'
        f'<small>mes. {s["b0"]+1}–{s["b1"]+1}</small></button>' for s in pave)
    mot = "".join(
        f'<button class=blk data-p="[{m["occ"][0]},{m["occ"][0]+m["L"]}]">'
        f'motif {i+1}<small>{m["L"]} mes. × {len(m["occ"])}</small></button>'
        for i, m in enumerate(voc))

    body = f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>les motifs</span>{mot}</div>
<div class=lane><span class=lab>le pavage</span>{btns}</div>
</section>
<audio id=au preload=metadata playsinline src="../audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — vocabulaire et pavage", body, back=True,
                lede_html=LEDE, back_href="vocab_lab.html",
                back_label="tous les morceaux")


LEDE = """<div class=lede>D'abord le <b>vocabulaire</b> : un couloir par motif que
le morceau rejoue, chaque rectangle est une occurrence, la longueur est écrite à
gauche. Rien n'est encore décidé sur les sections — c'est juste ce qui se
répète.<br><br>
Puis <b>le pavage</b> : on remplit la grille de mesures avec ces motifs, et les
<b>traits noirs</b> (les pics durs) interdisent qu'une pose les enjambe. Deux
sections identiques collées ne sont plus « une section à séparer » : ce sont deux
poses du même motif.<br><br>
En dessous, <b>ce qu'on écrit aujourd'hui</b> avec la recherche gauche→droite, et
<b>ton découpage</b> (traits rouges fins). <b>Touche un motif ou une section pour
l'écouter.</b><br><br>
<b>État honnête</b> : le pavage n'est pas encore au niveau de ce qu'on écrit
aujourd'hui — regarde surtout les couloirs de motifs, c'est là que se joue la
suite.</div>"""


def main():
    stems = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in stems] or (
        [(s, s) for s in stems] if stems else list(SONGS))
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"vocab_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="vocab_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "vocab_lab.html").write_text(page(
        "Vocabulaire et pavage",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede=False, back_href="vocab_lab.html"))
    print(f"wrote docs/plots/vocab_lab.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
