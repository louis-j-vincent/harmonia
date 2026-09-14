"""The distributions the quantiles are taken from — to choose a threshold by eye.

« Affiche-moi les histogrammes des distributions que tu utilises pour computer
les quantiles, on va se servir de ça pour choisir un seuil. »

Three quantiles are taken from ONE distribution: the off-diagonal values of the
harmonic SSM (|lag| ≥ 2) of that song.

    PHASE_QUANTILE = 0.90   a bar is "strong" at a lag  → starts a run
    (run search)     0.80   continuation level, hysteresis
    TILE_QUANTILE  = 0.95   the shipped sections.py tiling threshold

Plotted with the values our own detections actually take, so the bulk and the
real matches can be compared: BLUE = every off-diagonal value, RED = the
bar-to-bar values of the matches the dictionary accepted (its entries'
occurrences, read along the diagonal). Where those two separate is where a
threshold belongs.

    python scripts/threshold_histograms.py  ->  /reports/threshold_histograms.html
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
from ssm_rows_plot import fig2b64                        # noqa: E402
import harmonic_method as HM                             # noqa: E402
from dictionary_harmonic import bar_grid, build          # noqa: E402

OUT = HERE / "harmonia_min/state/reports/threshold_histograms.html"
INK = "#1c1c1c"
QS = [(0.80, "#2a6fb0", "q80 — continuation (hystérésis)"),
      (0.90, "#8a2b2b", "q90 — démarrage d'une série"),
      (0.95, "#1f8a5b", "q95 — seuil de tuilage livré")]
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why"),
         ("bobby_hebb_sunny_official_audio", "Sunny (grille OK, module)")]


def song_html(stem, title):
    cap = bar_grid(stem)
    grid = cap["grid"]
    n = len(grid) - 1
    S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
    i = np.arange(n)
    off = S[np.abs(i[:, None] - i[None, :]) >= HM.LAG_MIN]

    # the values our own accepted matches actually take, read bar-to-bar
    entries, _ = build(S, n)
    matched = []
    for e in entries:
        L = e["L"]
        for o in e["occ"]:
            matched += [float(S[e["b0"] + k, o + k]) for k in range(L)
                        if e["b0"] + k < n and o + k < n]
    matched = np.array(matched) if matched else np.array([])

    fig, axs = plt.subplots(1, 2, figsize=(13, 3.4))
    for ax, logy in ((axs[0], False), (axs[1], True)):
        ax.hist(off, bins=80, color="#9db4cc", label="toutes les valeurs hors diagonale")
        if len(matched):
            ax.hist(matched, bins=80, color="#8a2b2b", alpha=.85,
                    label="les correspondances retenues (lecture diagonale)")
        for q, col, lab in QS:
            v = float(np.quantile(off, q))
            ax.axvline(v, color=col, lw=1.6)
            ax.text(v, ax.get_ylim()[1] * (.95 if q == .8 else .8 if q == .9 else .65),
                    f" {lab.split(' — ')[0]} = {v:.3f}", color=col, fontsize=8)
        if logy:
            ax.set_yscale("log")
            ax.set_title("échelle log — on voit la queue", fontsize=9, loc="left")
        else:
            ax.set_title("échelle linéaire", fontsize=9, loc="left")
        ax.set_xlabel("similarité harmonique", fontsize=8.5)
    axs[0].legend(fontsize=8)
    h = fig2b64(fig)

    rows = "".join(
        f"<tr><td style='color:{col}'><b>{lab}</b></td>"
        f"<td>{float(np.quantile(off, q)):.4f}</td>"
        f"<td>{float((off >= np.quantile(off, q)).mean()):.1%} de la matrice</td>"
        f"<td>{'' if not len(matched) else f'{float((matched >= np.quantile(off, q)).mean()):.1%} des correspondances'}</td>"
        f"</tr>" for q, col, lab in QS)
    return f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<img src="data:image/png;base64,{h}">
<table><tr><th>quantile</th><th>valeur</th><th>ce qu'il garde de la matrice</th>
<th>ce qu'il garde des vraies correspondances</th></tr>{rows}</table></section>"""


def main():
    body = ""
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            continue
        body += song_html(stem, title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les distributions derrière les quantiles</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1300px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{max-width:100%;border-radius:8px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
</style></head><body><div class=wrap>
<h1>Les distributions derrière les quantiles</h1>
<div class=lede>Les trois seuils sortent tous de la <b>même</b> distribution :
les valeurs hors diagonale de la matrice harmonique du morceau (|décalage| ≥ 2).
En bleu toutes ces valeurs, en rouge celles que prennent les correspondances que
le dictionnaire a retenues, lues mesure à mesure. Là où les deux se séparent est
l'endroit où un seuil a un sens. L'échelle log à droite montre la queue, où tout
se joue — sur ces morceaux le q90 vaut 0,986 à 0,990, donc l'écart entre « même
mesure » et « mesure différente » est inférieur à un pour cent.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
