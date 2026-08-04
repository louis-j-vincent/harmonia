"""Build /reports/sections_v2.html — Louis's binary bi-bar SSM, seen and scored.

Shows, for This Love and Don't Know Why: the binary bi-bar matrix as an image,
the runs of 1s it finds on the sub-diagonals, the segmentation that falls out,
and the side-by-side with what `harmonia_min/sections.py` produces today.
Plus the Billboard numbers from `scripts/bibar_final.py`.

Captures the pipeline's REAL grid/bars/segments by spying on `detect_sections`
during a live `analyze()` (same trick as `scripts/sections_explainer.py`;
deepcopy because the fold stage later mutates those bar dicts).

    python scripts/bibar_report.py
"""
from __future__ import annotations

import base64
import copy
import io
import json
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
from bibar_binary_ssm import (bibar, bin_ssm, binarise, cont_ssm,  # noqa: E402
                              density, lag_runs, run_edge_score)
from harmonia_min import sections as hs                          # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")
OUT = HERE / "harmonia_min/state/reports/sections_v2.html"
SONGS = [("maroon_5_this_love", "Maroon 5 — This Love"),
         ("norah_jones_don_t_know_why", "Norah Jones — Don't Know Why")]
INK, GRN, ACC, BLU = "#1c1c1c", "#1f8a5b", "#8a2b2b", "#2a6fb0"
BEST = dict(mode="topk", kb=2, kt=4, join="concat", tau=0.8)


def fig2b64(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def capture(stem, title):
    from harmonia_min import pipeline as _pl
    cap, real = {}, hs.detect_sections

    def spy(grid, arr, times, bars=None):
        out = real(grid, arr, times, bars)
        cap.update(grid=grid, arr=np.asarray(arr), times=times,
                   bars=copy.deepcopy(bars), segs=out)
        return out

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title=title,
                    file_key=f"min_{stem}", audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = real
    return cap


def bar_chroma(grid, arr, times):
    nb = len(grid) - 1
    C = np.zeros((nb, 24))
    for b in range(nb):
        sel = (times >= grid[b]) & (times < grid[b + 1])
        C[b] = arr[sel].mean(0) if sel.any() else \
            arr[int(np.argmin(np.abs(times - 0.5 * (grid[b] + grid[b + 1]))))]
    return C


def song_block(stem, title):
    cap = capture(stem, title)
    grid, arr, times = cap["grid"], cap["arr"], cap["times"]
    segs, nb = cap["segs"], len(cap["grid"]) - 1
    Cb = bar_chroma(grid, arr, times)
    Bx = bibar(binarise(Cb, BEST["mode"], BEST["kb"], BEST["kt"]), BEST["join"])
    M = bin_ssm(Bx, BEST["tau"])
    Mc = cont_ssm(Cb, BEST["join"], 0.93)
    runs = lag_runs(M, min_run=6)
    sc = run_edge_score(M, "edgesum")
    scc = run_edge_score(Mc, "edgesum")

    # 1. the binary matrix, next to the continuous one it is a binarisation of
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    axs[0].imshow(M, cmap="Greys", origin="lower", interpolation="nearest")
    axs[0].set_title(f"binaire — densité hors-bande {density(M)*100:.0f}%",
                     fontsize=9, loc="left")
    axs[1].imshow(Mc, cmap="Greys", origin="lower", interpolation="nearest")
    axs[1].set_title(f"continu seuillé à 0,93 — {density(Mc)*100:.0f}%",
                     fontsize=9, loc="left")
    for a in axs:
        a.set_xlabel("bi-mesure", fontsize=8)
        for s in segs:
            a.axvline(s["b0"], color=ACC, lw=.7, alpha=.6)
    im_ssm = fig2b64(fig)

    # 2. the runs on the sub-diagonals, drawn as (start bar, lag)
    fig, ax = plt.subplots(figsize=(11, 3.0))
    for (b0, ln, lag) in runs:
        ax.plot([b0, b0 + ln], [lag, lag], color=BLU, lw=2.2, alpha=.75,
                solid_capstyle="butt")
        ax.plot([b0], [lag], "o", color=ACC, ms=3.5)
    for s in segs:
        ax.axvline(s["b0"], color=ACC, lw=.8, ls="--", alpha=.7)
    ax.set_xlim(0, nb)
    ax.set_xlabel("mesure", fontsize=8)
    ax.set_ylabel("décalage (lag)", fontsize=8)
    ax.set_title(f"{len(runs)} séries de 1 (≥6 bi-mesures) · point rouge = "
                 "début de série · tirets = sections actuelles",
                 fontsize=8.5, loc="left")
    im_runs = fig2b64(fig)

    # 3. the score, and the segmentation that falls out of it
    thr = sc.mean() + sc.std()
    cuts = [i for i in range(2, len(sc) - 2)
            if sc[i] == sc[max(0, i - 2):i + 3].max() and sc[i] >= thr]
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.plot(sc, color=INK, lw=1, label="binaire")
    ax.plot(scc * sc.std() / max(scc.std(), 1e-9), color=BLU, lw=.9, alpha=.6,
            label="continu (rééchelonné)")
    ax.axhline(thr, color="#999", ls=":", lw=1)
    for c in cuts:
        ax.axvline(c, color=GRN, lw=1.3, alpha=.8)
    for s in segs:
        ax.axvline(s["b0"], color=ACC, lw=1, ls="--", alpha=.7)
    ax.legend(fontsize=7)
    ax.set_xlabel("mesure", fontsize=8)
    ax.set_title("score « une répétition COMMENCE ici » = somme sur les lags "
                 "des séries qui démarrent à cette mesure", fontsize=8.5,
                 loc="left")
    im_sc = fig2b64(fig)

    # 4. side by side
    fig, ax = plt.subplots(figsize=(11, 1.5))
    pal = ["#1f8a5b", "#8a2b2b", "#2a6fb0", "#b07a2a", "#6a4c93", "#2a8a8a"]
    for i, s in enumerate(segs):
        ax.add_patch(plt.Rectangle((s["b0"], 1.1), s["b1"] - s["b0"] + 1, .8,
                                   color=pal[(ord(s["label"]) - 65) % len(pal)],
                                   alpha=.75))
        ax.text(s["b0"] + .4, 1.35, s["label"], fontsize=8, color="w")
    edges = [0] + cuts + [nb]
    for a, b in zip(edges, edges[1:]):
        ax.add_patch(plt.Rectangle((a, .1), b - a, .8, fill=False, ec=INK, lw=1.4))
        ax.text(a + .4, .35, f"{a}", fontsize=7, color=INK)
    ax.set_xlim(0, nb)
    ax.set_ylim(0, 2.1)
    ax.set_yticks([.5, 1.5])
    ax.set_yticklabels(["bi-barres\nbinaires", "sections.py\n(actuel)"],
                       fontsize=7)
    ax.set_xlabel("mesure", fontsize=8)
    im_cmp = fig2b64(fig)

    cur = " · ".join(f"{s['label']}&nbsp;{s['b0']}–{s['b1']}" for s in segs)
    return f"""
<section><h2>{title}</h2>
<div class=num>{nb} mesures &nbsp;·&nbsp; {len(Bx)} bi-mesures &nbsp;·&nbsp;
{len(runs)} séries de 1 &nbsp;·&nbsp; densité {density(M)*100:.0f}%</div>

<h3>1 · La matrice de uns et de zéros</h3>
<p>Chaque bi-mesure (2 mesures, pas glissant d'une mesure) devient un vecteur
<b>binaire</b> : les {BEST['kb']} cases de basse et les {BEST['kt']} cases
d'aigu les plus fortes passent à 1, tout le reste à 0. On compare deux
bi-mesures par leur <b>produit scalaire binaire</b> normalisé, et on met un 1
quand il dépasse {BEST['tau']}. À droite, la même lecture sur le chroma
<b>continu</b> — c'est la comparaison qui décide si binariser sert.</p>
<img src="data:image/png;base64,{im_ssm}">

<h3>2 · Les séries de 1 dans les sous-diagonales</h3>
<p>Une sous-diagonale de décalage <i>l</i>, c'est la question « la mesure
<i>i</i> ressemble-t-elle à la mesure <i>i−l</i> ? ». Une <b>série de 1</b>
veut dire « ce passage rejoue ce qui s'est passé <i>l</i> mesures plus tôt ».
Chaque trait bleu est une série ; le point rouge est son <b>début</b>.</p>
<img src="data:image/png;base64,{im_runs}">

<h3>3 · Le score, et les coupes qui en tombent</h3>
<img src="data:image/png;base64,{im_sc}">

<h3>4 · Côte à côte avec ce qu'on produit aujourd'hui</h3>
<img src="data:image/png;base64,{im_cmp}">
<p class=cap>actuel : {cur}<br>bi-barres binaires : coupes à
{', '.join(str(c) for c in cuts) or '—'}</p>
</section>"""


def numbers_block():
    p = CACHE / "bibar_final.json"
    if not p.exists():
        return "<section><h2>Chiffres</h2><p>pas encore mesurés</p></section>"
    d = json.load(open(p))
    rows = "".join(
        f"<tr{' class=hi' if 'LOUIS' in r['name'] or 'CONTINUOUS' in r['name'] else ''}>"
        f"<td>{r['name'].replace('  ...','&nbsp;&nbsp;…').replace('    ','&nbsp;&nbsp;&nbsp;&nbsp;')}</td>"
        f"<td><b>{r['exact']:.1f}%</b></td><td>{r['pm1']:.1f}%</td>"
        f"<td class=g>{r['biased']:.1f}%</td></tr>" for r in d["rows"])
    au = d.get("auc", {})
    tr = d.get("transpose", {})
    trr = "".join(f"<tr><td>{k}</td><td>{v['P']:.3f}</td><td>{v['R']:.3f}</td>"
                  f"<td><b>{v['F']:.3f}</b></td></tr>" for k, v in tr.items())
    bd = d.get("boundary", {})
    bdr = "".join(f"<tr><td>±{k} mesure</td><td>{v['binary']:.3f}</td>"
                  f"<td>{v['novelty']:.3f}</td></tr>" for k, v in bd.items())
    return f"""
<section><h2>Les chiffres — {d['n_tracks']} morceaux Billboard,
{d['n_starts']} débuts de section annotés</h2>

<div class=warn><b>Le point le plus important de cette page.</b> La mesure
utilisée jusqu'ici départageait les égalités <b>en préférant le plus petit
décalage</b>. Or un indice qui donne <b>le même score partout</b> obtient alors
100 % — y compris l'indice « ne bouge jamais » (1<sup>re</sup> ligne). C'est
exactement le cas de la correspondance de chaînes d'accords : à l'intérieur
d'une section répétée, les 8 mesures collent aussi bien à chaque décalage. Son
<b>75,2 %</b> annoncé hier était donc surtout l'arbitrage, pas l'indice : à
égalité tirée au sort il tombe à {d['rows'][2]['exact']:.1f} %, à peine
au-dessus du hasard ({d['rows'][0]['exact']:.1f} %).</div>

<table><tr><th>indice</th><th>bonne mesure</th><th>à ±1</th>
<th class=g>ancien arbitrage</th></tr>{rows}</table>
<p class=cap>« bonne mesure » = le maximum de l'indice tombe exactement sur la
mesure annotée, fenêtre ±4 mesures, égalités tirées au sort.</p>

<h3>L'hypothèse de Louis, testée directement</h3>
<p>Hypothèse : les reprises diffèrent par l'<b>arrangement</b> (mêmes accords,
densité et voicings différents), donc un « ces notes sont-elles présentes ? »
binaire séparerait mieux les paires même-section que le cosinus continu.
Mesuré en AUC sur les paires de sections annotées :</p>
<div class=num>binaire {au.get('binary', 0):.3f} &nbsp;·&nbsp;
continu {au.get('continuous', 0):.3f}</div>

<h3>Invariance par transposition (Sunny)</h3>
<table><tr><th>similarité</th><th>précision</th><th>rappel</th><th>F</th></tr>
{trr}</table>

<h3>Détection de frontières, sans savoir où chercher</h3>
<table><tr><th>tolérance</th><th>séries binaires</th><th>pics de nouveauté</th></tr>
{bdr}</table>
</section>"""


def main():
    blocks = "".join(song_block(s, t) for s, t in SONGS)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Sections v2 — la SSM binaire de bi-mesures</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:920px;margin:0 auto;padding:22px 16px 60px}}
h1{{font:italic 600 26px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 15px system-ui;margin:0 0 10px;color:{ACC}}}
h3{{font:700 13px system-ui;margin:18px 0 6px;color:{INK}}}
img{{max-width:100%;border-radius:8px;margin:8px 0}}
table{{border-collapse:collapse;font-size:12.5px;margin:10px 0;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:5px 9px;text-align:left}}
th{{background:#f7f3e9;font-weight:700}}
tr.hi td{{background:#eef7f1}}
td.g,th.g{{color:#a09880}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:12px}}
.num{{background:{INK};color:#f4eee2;padding:7px 12px;border-radius:8px;font:600 13px system-ui;display:inline-block;margin:8px 0}}
.cap{{font-size:11.5px;color:#8a8371}}
.warn{{background:#fdf3f0;border-left:3px solid {ACC};padding:10px 13px;border-radius:0 8px 8px 0;font-size:13.5px}}
.verdict{{background:#f7f3e9;border-left:3px solid {GRN};padding:10px 13px;border-radius:0 8px 8px 0;font-size:14px}}
</style></head><body><div class=wrap>
<h1>La SSM binaire de bi-mesures</h1>
<div class=lede>Ton idée du 5 août, implémentée et mesurée : unité = la
bi-mesure, vecteur binaire, produit scalaire binaire seuillé, lecture des
séries de 1 dans les sous-diagonales.</div>

<section><h2>Le verdict en quatre lignes</h2>
<div class=verdict>
<p><b>1. Ta lecture des séries de 1 dans les sous-diagonales marche</b>, et
c'est la vraie trouvaille. Elle place <b>36,8 %</b> des débuts de section sur
la bonne mesure, contre <b>27,6 %</b> pour la courbe de nouveauté qu'on utilise
aujourd'hui. Additionnée à la nouveauté : <b>40,2 %</b>.</p>
<p><b>2. Mais binariser <i>coûte</i> 6 points</b> : exactement le même calcul
sur le chroma continu fait 36,8 %, la version binaire 30,8 %. Ton hypothèse
« les reprises diffèrent par l'arrangement, donc un binaire pardonne » est
<b>testée et non confirmée</b> (AUC 0,839 binaire contre 0,874 continu).
L'unité bi-mesure et la lecture des sous-diagonales sont les bonnes idées ;
le seuillage à 0/1 jette de l'information utile.</p>
<p><b>3. En revanche ça ne remplace pas nos coupes.</b> Sans savoir où
chercher, les séries donnent de moins bonnes frontières que nos pics
(F 0,161 contre 0,194). C'est un indice de <i>placement</i>, pas un détecteur.</p>
<p><b>4. Le 75,2 % d'hier n'existe pas</b> — artefact de la façon dont on
départageait les égalités. C'est le résultat le plus important de la session,
détaillé juste en dessous.</p>
</div></section>
{numbers_block()}
{blocks}
</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
