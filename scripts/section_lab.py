"""Tout sur le même axe : les pics de chaque matrice, le profil fusionné, les
sections — et un bouton play.

    .venv/bin/python scripts/section_lab.py [<stem> ...]
        -> /plots/section_lab.html + une page par morceau

Louis, 2026-08-11 :

  « Je veux que tu mettes tout ensemble. Je veux voir à la fois les pics et la
    détection de sections sur le même axe x, parce qu'il faut que je puisse
    comprendre les deux. Et il faut un bouton play pour que je puisse jouer. Je
    veux les pics pour chaque catégorie d'ailleurs — les pics harmoniques, de
    voix, de rythme, tout ça. Le pic fusion avec cette détection. Et les sections
    sur le même axe x. »

CE QU'IL Y A SUR LA PAGE, de haut en bas, tout à la MÊME échelle de mesures :

  1. le **profil fusionné** (Σ contraste · nouveauté) et ses pics retenus ;
  2. **une bande par matrice** — accords, basse, harmonie, voix, rythme, timbre —
     avec ses propres pics et son poids dans la fusion ;
  3. la bande **voix muette** : où personne ne chante (solo / pont / intro) ;
  4. les **découpages en sections** : la prod contrainte par les pics, la prod
     seule, et le tien.

Les traits noirs verticaux traversent TOUT : ce sont les pics durs. Les traits
rouges sont tes frontières. Une seule tête de lecture court sur l'ensemble, donc
un pic, la matrice qui l'a produit et la section qui commence là se lisent sur la
même verticale — c'était le point de la demande.

LA CONVENTION DES MESURES est celle de `pattern_lanes` : la mesure i occupe
[i, i+1], une frontière se pose en `i`, une valeur qui appartient à la mesure i
se pose en `i + 0,5`. Les courbes sont en demi-mesures, donc la case c se pose en
`(c + 0,5) / 2` et un pic (une frontière) en `c / 2`.

NE PAS METTRE `bbox_inches="tight"` : les marges du tracé sont fixes
(`PLOT_L`/`PLOT_R`) parce que la tête de lecture est positionnée en pourcentage
de la largeur de l'image.
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

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections       # noqa: E402
from peak_profile import fused_profile, page                 # noqa: E402
from hard_prior_sections import (sections_from_peaks, score,  # noqa: E402
                                 colourmap)

OUTDIR = HERE / "docs" / "plots"
PLOT_L, PLOT_R = 0.135, 0.995      # marges FIXES — la tête de lecture en dépend
INK = "#1c1c1c"
HARD = "#0d2437"


def fig2b64_fixed(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=112)      # surtout PAS bbox_inches="tight"
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
    P, kept, rest, lines, n, extra = fused_profile(stem)
    r = sections_from_peaks(stem)
    m = len(P)
    grid, hard = r["grid"], r["hard"]
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []

    votants = sorted((d for d in lines if d["nom"] != "fusion"),
                     key=lambda d: -d["poids"])
    strips = [("prod + pics durs", r["sections"]), ("la prod seule", r["prod"])]
    if gt:
        strips.append(("toi", gt["sections"]))

    nrow = 1 + len(votants) + 1 + len(strips)
    ratios = [2.8] + [1.15] * len(votants) + [0.5] + [1.2] * len(strips)
    H = 0.34 * sum(ratios) + 0.75
    fig, axs = plt.subplots(nrow, 1, figsize=(13.2, H), facecolor="#fffdf6",
                            gridspec_kw={"height_ratios": ratios, "hspace": 0.0})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - 0.30 / H,
                        bottom=0.42 / H)

    xcurve = (np.arange(m) + 0.5) / 2.0        # une valeur de demi-mesure
    def xcut(c):                               # une frontière de demi-mesure
        return c / 2.0

    def deco(ax, lab, colour=INK, last=False):
        ax.set_xlim(0, n)
        ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10,
                      color=colour)
        for s in ax.spines.values():
            s.set_color("#e0d7c2")
        if last:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8.5, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])

    def verticals(ax, red=True):
        for h in hard:
            ax.axvline(h, color=HARD, lw=1.7, alpha=0.95)
        if red:
            for b in gtb:
                ax.axvline(b, color=GT_LINE, lw=0.8, alpha=0.75)

    # 1 ── le profil fusionné
    ax = axs[0]
    ax.fill_between(xcurve, P, color="#3d7fa6", lw=0)
    verticals(ax)
    # Retenu = triangle NOIR PLEIN + une barre qui traverse la page. Suivant =
    # petit triangle CREUX. Louis, 2026-08-12 : il a lu des « suivants » gris
    # comme des retenus sur Blue Lights, parce qu'un aplat gris et un aplat noir
    # se ressemblent à cette taille. Ce n'est pas une nuance de gris qu'il faut,
    # c'est une forme différente.
    for c in kept:
        ax.plot([xcut(c)], [1.1], marker="v", ms=10, color=HARD, clip_on=False)
    for c in rest:
        ax.plot([xcut(c)], [1.1], marker="v", ms=7, markerfacecolor="none",
                markeredgecolor="#9aa3a9", markeredgewidth=1.1, clip_on=False)
    ax.set_ylim(0, 1.2)
    deco(ax, "PROFIL\nfusionné")

    # 2 ── une bande par matrice, avec ses propres pics
    for ax, d in zip(axs[1:1 + len(votants)], votants):
        pale = d["poids"] < 0.05
        col = "#8fb6cc" if pale else "#3d7fa6"
        ax.fill_between(xcurve, np.nan_to_num(d["nov"]), color=col, lw=0)
        verticals(ax)
        for c in d["cuts"]:
            ax.plot([xcut(c)], [0.98], marker="v", ms=6,
                    color="#aab4bb" if pale else "#1b4a6b", clip_on=False)
        ax.set_ylim(0, 1.05)
        deco(ax, f"{d['nom']}  {d['poids']:.0%}", "#9aa3a9" if pale else INK)

    # 3 ── la voix : où personne ne chante
    ax = axs[1 + len(votants)]
    mu = np.asarray(r["mute"], bool)[:m] if r["mute"] is not None else np.zeros(m, bool)
    ax.fill_between(xcurve, mu.astype(float), color="#c9a227", lw=0, step="mid")
    verticals(ax)
    ax.set_ylim(0, 1.05)
    deco(ax, "voix : muette", "#8a6d1f")

    # 4 ── les découpages, même axe
    cm = colourmap()
    for i, (ax, (lab, secs)) in enumerate(zip(axs[2 + len(votants):], strips)):
        for s in secs:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.06), w, 0.88,
                                       facecolor=cm(s["label"]),
                                       edgecolor="#fffdf6", lw=1.2))
            if w >= max(2, n * 0.035):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=9, color=INK)
        verticals(ax)
        ax.set_ylim(0, 1)
        deco(ax, lab, INK, last=(i == len(strips) - 1))
    img = fig2b64_fixed(fig)

    # les boutons d'écoute : les sections proposées, puis les pics
    sec_btns = "".join(
        f'<button class=blk data-p="[{s["b0"]},{s["b1"]+1}]">{s["label"]}'
        f'<small>mes. {s["b0"]+1}–{s["b1"]+1}</small></button>'
        for s in r["sections"])
    peak_btns = "".join(
        f'<button class=blk data-p="[{max(0, h-2)},{min(n, h+2)}]">mes. {h+1}'
        f'<small>{"juste" if any(abs(h-g) <= 1 for g in gtb) else "à juger"}</small>'
        "</button>" for h in hard)

    poids = " · ".join(f"{d['nom']} {d['poids']:.0%}" for d in votants
                       if d["poids"] >= 0.01)
    if gt:
        a = score(r["sections"], gt["sections"], n)
        b = score(r["prod"], gt["sections"], n)
        cmp_ = (f"contre ton découpage : <b>prod + pics {a:.3f}</b> · "
                f"prod seule {b:.3f} — la contrainte "
                f"<b>{'gagne' if a > b + 0.005 else ('perd' if a < b - 0.005 else 'fait pareil')}</b> ici")
    else:
        cmp_ = "pas d'annotation pour ce morceau"

    body = f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>les pics durs</span>{peak_btns or '<span class=hint>aucun</span>'}</div>
<div class=lane><span class=lab>les sections</span>{sec_btns}</div>
<div class=verdict><b>Poids dans la fusion —</b> {poids}<br>{cmp_}</div>
</section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — pics et sections, même axe", body, back=True,
                lede_html=LEDE, back_href="section_lab.html",
                back_label="tous les morceaux")


LEDE = """<div class=lede>Tout est sur le <b>même axe de mesures</b>, du profil
de pics jusqu'aux sections, et une seule tête de lecture court sur l'ensemble :
un pic, la matrice qui l'a produit et la section qui commence là se lisent sur la
même verticale.<br><br>
De haut en bas : le <b>profil fusionné</b> (chaque matrice pondérée par son
contraste) et ses pics retenus (triangles noirs) ; <b>une bande par matrice</b>
avec ses propres pics et son poids ; la bande <b>voix muette</b> (jaune = personne
ne chante — solo, pont, intro) ; puis les <b>découpages</b> : la recherche de la
prod contrainte par les pics, la prod seule, et le tien.<br><br>
Les <b>traits noirs</b> traversent tout : ce sont les pics durs, qu'aucune
section n'a le droit de franchir. Les <b>traits rouges</b> sont tes frontières —
elles n'entrent dans aucun calcul.<br><br>
<b>Touche le graphique pour te déplacer</b>, ou un bouton pour écouter un pic
(deux mesures autour) ou une section entière.</div>"""


def main():
    stems = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in stems] or (
        [(s, s) for s in stems] if stems else list(SONGS))
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            print(f"  ?? pas d'audio pour {stem}")
            continue
        (OUTDIR / f"lab_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="lab_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "section_lab.html").write_text(page(
        "Pics et sections, même axe",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede=False, back_href="section_lab.html"))
    print(f"wrote docs/plots/section_lab.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
