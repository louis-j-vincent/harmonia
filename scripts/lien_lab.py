"""Le lien de groupage, choisi par le morceau — les douze arbitrages sur une page.

    .venv/bin/python scripts/lien_lab.py   ->  docs/plots/lien_lab.html
    open docs/plots/lien_lab.html          (local, sans serveur)

Louis, 2026-08-12 : « show me illustration ».

CE QU'ON REGARDE. Une bi-mesure rejoint un groupe de bi-mesures semblables selon
deux règles possibles :

  * **lien COMPLET** — elle doit ressembler à TOUS les membres du groupe ;
  * **lien MOYEN** — elle doit ressembler à leur moyenne.

Ce n'est pas un réglage, c'est une question musicale. Un refrain copié-collé
supporte le lien complet ; un couplet rejoué à chaque fois ne le supporte pas —
sur Grenade, ses quatre A sont à 0,86 de ressemblance interne avec un minimum à
0,60, soit **17 % de paires sous le seuil**, et une seule paire ratée sur six
suffit à casser le groupe en lien complet.

QUI TRANCHE : le morceau, par longueur de description.

    coût = Σ longueurs des types distincts + 1 par section
           + la longueur ENTIÈRE de chaque bloc « reste »

La bande retenue est celle du coût le plus bas, encadrée. Sur les douze
morceaux : Grenade et Sunny prennent le moyen, les dix autres le complet — soit
exactement l'arbitrage qu'on faisait à la main avant d'avoir ce critère.
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

from ssm_zoo import SONGS, GT_LINE, gt_sections, fig2b64   # noqa: E402
from hard_prior_sections import colourmap                  # noqa: E402
from vote_fill import fill, INK                            # noqa: E402
import mots4                                               # noqa: E402
import order_bundle                                        # noqa: E402

OUT = HERE / "docs" / "plots" / "lien_lab.html"


def song_png(stem: str):
    b = order_bundle.get(stem)
    n = b["n"]
    Sv = np.nan_to_num(np.asarray(b["M"], float))
    res = {}
    for lien in ("complet", "moyen"):
        R = fill(b, stem, lien=lien)
        se, _d = mots4.sections(R["mot"], R["x0"], n, R["sim"], b["start"],
                                cuts=set(R["cuts"]), Sv=Sv)
        res[lien] = (mots4.cout(se), se, R["mot"])
    choisi = min(res, key=lambda k: (res[k][0], k != "complet"))
    gt = gt_sections(stem)
    bandes = [(f"complet · {res['complet'][0]:.0f}", res["complet"][1],
               choisi == "complet"),
              (f"moyen · {res['moyen'][0]:.0f}", res["moyen"][1],
               choisi == "moyen")]
    if gt:
        bandes.append(("toi", gt["sections"], False))
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []

    fig, axs = plt.subplots(len(bandes), 1, figsize=(12.6, 0.52 * len(bandes) + 0.35),
                            facecolor="#fffdf6", gridspec_kw={"hspace": 0.12})
    cm = colourmap()
    for k, (ax, (lab, ss, pris)) in enumerate(zip(np.atleast_1d(axs), bandes)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle(
                (s["b0"], 0.08), w, 0.84, facecolor=cm(s.get("nu", s["label"])),
                edgecolor="#fffdf6", lw=1.0,
                hatch="///" if "′" in str(s["label"]) else None))
            if w >= max(5, n * 0.05):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8, color=INK)
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.6)
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9,
                      color=INK if pris else "#a89f8c",
                      fontweight="bold" if pris else "normal")
        for sp in ax.spines.values():
            sp.set_visible(pris)
            sp.set_color("#0d2437")
        if k == len(bandes) - 1:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=7.5, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])
    fig.subplots_adjust(left=0.13, right=0.995, top=0.97, bottom=0.16)
    return fig2b64(fig), choisi, res


def main():
    body = []
    for stem, title in SONGS:
        img, choisi, res = song_png(stem)
        ecart = abs(res["complet"][0] - res["moyen"][0])
        note = ("les deux se valent — on garde le complet" if ecart == 0
                else f"le {choisi} coûte {ecart:.0f} de moins")
        body.append(
            f'<section><h2>{title} <span class=sub>lien {choisi} · {note}</span></h2>'
            f'<img src="data:image/png;base64,{img}" alt="{title}"></section>')
        print(f"  ok {title} -> {choisi}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Le lien de groupage, choisi par le morceau</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1280px;margin:0 auto;padding:22px 14px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:980px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:12px 14px;margin-bottom:12px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block}}
</style></head><body><div class=wrap>
<h1>Le lien de groupage, choisi par le morceau</h1>
<div class=lede>Une bi-mesure rejoint un groupe de bi-mesures semblables selon
deux règles possibles : en <b>lien complet</b> elle doit ressembler à TOUS les
membres du groupe, en <b>lien moyen</b> à leur moyenne.<br><br>
Ce n'est pas un réglage, c'est une question musicale. Un refrain copié-collé
supporte le lien complet ; <b>un couplet rejoué à chaque fois ne le supporte
pas</b> — sur Grenade tes quatre A sont à 0,86 de ressemblance interne avec un
minimum à 0,60, soit 17 % de paires sous le seuil, et une seule paire ratée sur
six suffit à casser le groupe.<br><br>
<b>C'est le morceau qui tranche</b>, par longueur de description : la somme des
longueurs des types distincts, plus un renvoi par section, plus la longueur
entière de chaque bloc « reste » (il ne s'explique par rien). La bande retenue
est encadrée. Traits rouges : tes frontières.<br><br>
Sur les douze morceaux, <b>Grenade et Sunny prennent le moyen, les dix autres le
complet</b> — exactement l'arbitrage qu'on faisait à la main avant d'avoir ce
critère.</div>
{''.join(body)}</div></body></html>""")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
