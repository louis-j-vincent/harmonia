"""La matrice SSM de la VOIX, en grand, pour les douze morceaux — page locale.

    .venv/bin/python scripts/voix_ssm.py       ->  docs/plots/voix_ssm.html
    open docs/plots/voix_ssm.html              (aucun serveur, aucun réseau)

Louis, 2026-08-12 : « montre-moi matrice SSM voix !!!! » — et, juste avant :
« j'ai une connexion pourrie, mets-les-moi en local sur le MacBook ».

Cette page n'a ni audio ni tête de lecture : rien que les matrices, donc elle
s'ouvre en `file://` sans serveur et pèse le minimum.

TROIS MATRICES PAR MORCEAU, mêmes axes de mesures, ses frontières en rouge :

  * **voix par demi-mesure** — celle du zoo. Mouchetée, presque vide : une
    demi-mesure de chant tient deux ou trois notes, son profil de douze hauteurs
    est trop maigre pour que deux cases se ressemblent autrement que par hasard.
    C'est elle qui donnait « aucune répétition », et c'était un artefact.
  * **voix par mesure** — la même chaîne (demucs → pyin → notes) agrégée sur une
    mesure entière, celle de la prod. Lisse, et ses blocs se voient à l'œil.
  * **basse par mesure** — celle qui fabrique le mot des bi-mesures, pour
    comparer ce que chacune sait séparer.
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

from ssm_zoo import SONGS, GT_LINE, gt_sections, fig2b64    # noqa: E402
from peak_profile import fused_profile                      # noqa: E402
import order_bundle                                         # noqa: E402

OUT = HERE / "docs" / "plots" / "voix_ssm.html"
INK = "#1c1c1c"


def song_png(stem: str) -> tuple[str, int]:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("h", ["#faf6ec", "#9fc0d4", "#1d4d69"])
    b = order_bundle.get(stem)
    n = b["n"]
    _P, _k, _r, lines, _n, _e = fused_profile(stem)
    Szoo = np.nan_to_num(next(d["S"] for d in lines if d["nom"] == "voix"))
    Sbas = np.nan_to_num(next(d["S"] for d in lines if d["nom"] == "basse"))
    M = np.nan_to_num(np.asarray(b["M"], float))
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.7), facecolor="#fffdf6",
                            gridspec_kw={"wspace": 0.13})
    for ax, S, lab in ((axs[0], Szoo, "VOIX — par demi-mesure (mouchetée)"),
                       (axs[1], M, "VOIX — par mesure (prod) ← celle qui compte"),
                       (axs[2], Sbas, "basse — par demi-mesure")):
        ax.imshow(S, cmap=cmap, vmin=0, vmax=1, extent=[0, n, n, 0],
                  interpolation="nearest")
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.8)
            ax.axhline(g, color=GT_LINE, lw=0.7, alpha=0.8)
        step = 16 if n <= 120 else 32
        ax.set_xticks(np.arange(0, n + 1, step)); ax.set_yticks(np.arange(0, n + 1, step))
        ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7.5)
        ax.set_yticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7.5)
        ax.tick_params(length=2, colors="#8a8371")
        ax.set_title(lab, fontsize=10, color="#4a4438", pad=6)
        for sp in ax.spines.values():
            sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.9, bottom=0.08)
    return fig2b64(fig), n


def main():
    stems = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in stems] or list(SONGS)
    body = []
    for stem, title in todo:
        img, n = song_png(stem)
        body.append(f'<section><h2>{title} <span class=sub>{n} mesures</span></h2>'
                    f'<img src="data:image/png;base64,{img}" alt="{title}"></section>')
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Matrices SSM de la voix</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1320px;margin:0 auto;padding:22px 14px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:980px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:12px 14px;margin-bottom:12px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block}}
</style></head><body><div class=wrap>
<h1>Les matrices SSM de la voix</h1>
<div class=lede>Trois matrices par morceau, mêmes axes de mesures, tes frontières
en rouge.<br><br>
<b>À gauche, la voix par DEMI-mesure</b> — celle que j'utilisais. Mouchetée,
presque vide : une demi-mesure de chant ne tient que deux ou trois notes, son
profil de douze hauteurs est trop maigre pour que deux cases se ressemblent
autrement que par hasard. C'est elle qui donnait « aucune répétition », et
c'était un artefact de résolution, pas une propriété du chant.<br><br>
<b>Au milieu, la voix par MESURE</b> — la même chaîne (demucs → pyin → notes)
agrégée sur une mesure entière, celle de la prod. Lisse, et ses blocs se voient à
l'œil : le pont de This Love (mesures 49-56) saute aux yeux.<br><br>
<b>À droite, la basse</b>, celle qui fabrique aujourd'hui le mot de bi-mesures —
pour comparer ce que chacune sait séparer.<br><br>
Page locale : aucun serveur, aucun réseau, aucun son.</div>
{''.join(body)}</div></body></html>""")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
