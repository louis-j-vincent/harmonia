"""Le dictionnaire de motifs — Louis's design, 2026-08-05.

  1. « des pics clairs qui doivent correspondre à un quartile des valeurs de la
     matrice SSM » -> at what QUANTILE of its own off-diagonal distribution do
     a song's visible row peaks sit?
  2. « en cumulant les demi-barres adjacentes on trouve un signal robuste »
     -> half-bar vs bar vs summed-adjacent-half-bar.
  3. « le premier bloc de répétition ... est la première entrée d'un
     dictionnaire de motifs. Pour trouver les autres occurrences on fait
     glisser ce motif sur l'AXE DES X — pas sur la diagonale — et on fait le
     produit scalaire avec le carré sur lequel il tombe. »

    python scripts/pattern_dict_report.py  ->  /reports/pattern_dict.html
"""
from __future__ import annotations

import base64
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
from pattern_dict_core import (CACHE, SONGS, all_peaks, grain_signals,  # noqa: E402
                               offdiag, peak_contrast, q_of, slide_dot,
                               slide_peaks, song_data)

OUT = HERE / "harmonia_min/state/reports/pattern_dict.html"
INK, ACC, GRN, BLU, SAND = "#1c1c1c", "#8a2b2b", "#1f8a5b", "#2a6fb0", "#8a8371"
Q_MOTIF = 0.95           # the adaptive threshold this page argues for
PROM = 0.25


def fig2b64(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


# ── (3) the first dictionary entry ──────────────────────────────────────────
def first_motif_block(S, q=Q_MOTIF, min_lag=2, maxL=32):
    """« le premier motif de répétition qui démarre sur le temps 1 d'une
    mesure ». On the BAR SSM every index IS a downbeat, so "beat 1" is free;
    what must be chosen is where it starts and how long it is.

    Scan bars in order; for each, the shortest L whose whole BLOCK repeats —
    mean of S[b+i, b+L+i] over i < L above the song's own q-quantile. A block
    diagonal, not a single row peak: one high cell is a coincidence, L high
    cells in a row is a repeat.
    """
    n = len(S)
    thr = float(np.quantile(offdiag(S, min_lag), q))
    for b in range(n):
        for L in range(min_lag, min(maxL, (n - b) // 2) + 1):
            sc = float(np.mean([S[b + i, b + L + i] for i in range(L)]))
            if sc >= thr:
                return dict(b0=b, L=L, score=sc, thr=thr)
    return None


def song_fig(S, segs, mot, ds, f, pk, n, f_raw=None):
    """SSM on top, the sliding dot product directly below it, SAME x axis —
    which is the whole point of "sliding along X, not along the diagonal"."""
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[3.1, 1.25, .34],
                                             hspace=.07))
    ax = axs[0]
    ax.imshow(S, cmap="magma", vmin=max(0.0, np.percentile(S, 5)), vmax=1.0,
              origin="upper", extent=(-.5, n - .5, n - .5, -.5),
              interpolation="nearest", aspect="auto")
    b0, L = mot["b0"], mot["L"]
    ax.axhspan(b0 - .5, b0 + L - .5, color=GRN, alpha=.22)
    for p in pk:
        ax.add_patch(plt.Rectangle((p["d"] - .5, b0 - .5), L, L, fill=False,
                                   ec="#5ad1a0", lw=1.1))
    ax.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False,
                               ec="#ffffff", lw=1.8))
    ax.set_ylabel("mesure (ligne)", fontsize=9)

    ax = axs[1]
    if f_raw is not None:
        lo, hi = float(f_raw.min()), float(f_raw.max())
        ax.plot(ds, (f_raw - lo) / max(hi - lo, 1e-9) * (f.max() - f.min())
                + f.min(), lw=.9, color=BLU, alpha=.5,
                label="produit scalaire brut (recalé)")
    ax.plot(ds, f, lw=1.3, color=INK, label="centré = corrélation de blocs")
    ax.plot([p["d"] for p in pk], [p["val"] for p in pk], "o", ms=4.5,
            color=GRN, zorder=5)
    ax.axvline(b0, color=ACC, lw=1.4)
    ax.set_ylabel("produit scalaire\nmotif × carré", fontsize=8)
    ax.legend(fontsize=7, loc="lower right", framealpha=.85)
    ax.grid(alpha=.18, lw=.6)

    ax = axs[2]
    pal = ["#8a2b2b", "#2a6fb0", "#1f8a5b", "#b8860b", "#6a4c93", "#c05a2b",
           "#3d8b8b", "#a03060"]
    letters = sorted({s["label"] for s in segs})
    for s in segs:
        c = pal[letters.index(s["label"]) % len(pal)]
        ax.axvspan(s["b0"] - .5, s["b1"] + .5, color=c, alpha=.55)
        if s["b1"] - s["b0"] >= 2:
            ax.text((s["b0"] + s["b1"]) / 2, .5, s["label"], ha="center",
                    va="center", fontsize=8, color="w", fontweight="bold")
    ax.set_yticks([])
    ax.set_ylabel("sections\nactuelles", fontsize=7.5, rotation=0,
                  ha="right", va="center")
    ax.set_xlabel("mesure (colonne) — décalage du motif sur l'axe des X",
                  fontsize=9)
    ax.set_xlim(-.5, n - .5)
    return fig2b64(fig)


def main():
    rows_q, rows_g, body = [], [], ""
    for stem, title, good in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            print(f"  (no audio: {stem})")
            continue
        d = song_data(stem)
        S, n, segs = d["Sb"], d["n"], d["segs"]
        bg = offdiag(S, 2)

        # ── (1) at what quantile do the visible peaks sit? ───────────────────
        ps = all_peaks(S, 2, PROM)
        pr = np.array([p["prom"] for p in ps])
        val = np.array([p["val"] for p in ps])
        clear = pr >= np.quantile(pr, 0.75)      # "the ones a human sees"
        qs = np.array([q_of(bg, v) for v in val])
        rows_q.append(dict(
            title=title, good=good, n=n, npk=len(ps),
            q_all=float(np.median(qs)), q_clear=float(np.median(qs[clear])),
            q25=float(np.percentile(qs[clear], 25)),
            q75=float(np.percentile(qs[clear], 75)),
            q_of_080=q_of(bg, 0.80),
            cos_at_95=float(np.quantile(bg, 0.95))))

        # ── (2) which grain is cleanest? ─────────────────────────────────────
        G = grain_signals(d["Sh"], n)
        g = {}
        for name in ("half", "sum2", "bar"):
            M, u = G[name]
            g[name] = peak_contrast(M, 4 if u == 0.5 else 2, PROM)
        rows_g.append(dict(title=title, good=good, **{
            k: g[k] for k in ("half", "sum2", "bar")}))

        # ── (3) the dictionary ───────────────────────────────────────────────
        mot = first_motif_block(S)
        ds, f = slide_dot(S, mot["b0"], mot["L"], center=True)
        _, f_raw = slide_dot(S, mot["b0"], mot["L"], center=False)
        pk = slide_peaks(ds, f, mot["b0"], min_sep=max(2, mot["L"] // 2),
                         prom_frac=PROM)
        img = song_fig(S, segs, mot, ds, f, pk, n, f_raw)

        starts = [s["b0"] for s in segs]
        occ = sorted({mot["b0"]} | {p["d"] for p in pk})
        hit = sum(any(abs(o - s) <= 1 for s in starts) for o in occ)
        cov = sum(any(abs(o - s) <= 1 for o in occ) for s in starts)
        fq = [q_of(f, p["val"]) for p in pk]

        badge = "" if good else ('<span class=warn>grille de mesures '
                                 'DOUTEUSE — ne pas en tirer de règle</span>')
        body += f"""<section><h2>{title} {badge}</h2>
<div class=num>{n} mesures · motif = mesures <b>{mot['b0']}–{mot['b0']+mot['L']-1}</b>
(L = {mot['L']}) · score du bloc {mot['score']:.3f} ≥ seuil adaptatif
{mot['thr']:.3f} (95<sup>e</sup> centile du morceau)</div>
<img src="data:image/png;base64,{img}">
<p class=cap><b>En haut</b> : la SSM au niveau mesure. La bande verte = les
lignes du motif ; le carré blanc = le motif lui-même ; les carrés verts = là où
le motif retombe quand on le fait glisser sur l'axe des X.
<b>Au milieu</b> : le produit scalaire entre le motif et le carré sur lequel il
tombe, en fonction du décalage — même axe X que la matrice, donc chaque pic est
lisible à la verticale. Noir = version centrée (on retire à chaque carré sa
moyenne avant le produit) ; bleu pâle = le produit brut, recalé pour être
comparable. <b>En bas</b> : les sections que
<code>sections.py</code> écrit aujourd'hui.</p>
<table><tr><th></th><th>valeur</th></tr>
<tr><td>occurrences trouvées</td><td>{', '.join(map(str, occ))}</td></tr>
<tr><td>débuts de section actuels</td><td>{', '.join(map(str, starts))}</td></tr>
<tr><td>occurrences tombant sur un début de section (±1)</td>
<td><b>{hit}/{len(occ)}</b></td></tr>
<tr><td>débuts de section retrouvés par une occurrence (±1)</td>
<td><b>{cov}/{len(starts)}</b></td></tr>
<tr><td>centile médian des pics du produit scalaire</td>
<td>{np.median(fq)*100:.0f}<sup>e</sup></td></tr></table></section>"""
        print(f"  ok {title}  motif {mot['b0']}-{mot['b0']+mot['L']-1} "
              f"L={mot['L']}  {len(occ)} occurrences")

    # ── the two summary tables ──────────────────────────────────────────────
    def cls(r):
        return "" if r["good"] else " class=sus"
    tq = "".join(
        f"<tr{cls(r)}><td>{r['title']}{'' if r['good'] else ' ⚠'}</td>"
        f"<td>{r['n']}</td>"
        f"<td><b>{r['q_clear']*100:.1f}</b></td>"
        f"<td>{r['q25']*100:.0f}–{r['q75']*100:.0f}</td>"
        f"<td>{r['q_all']*100:.1f}</td>"
        f"<td>{r['cos_at_95']:.3f}</td>"
        f"<td><b>{r['q_of_080']*100:.0f}</b></td></tr>" for r in rows_q)
    tg = "".join(
        f"<tr{cls(r)}><td>{r['title']}{'' if r['good'] else ' ⚠'}</td>"
        + "".join(f"<td>{r[k]['z_height']:.2f}</td><td>{r[k]['z_prom']:.2f}</td>"
                  f"<td>{r[k]['per_row']:.0f}</td>"
                  for k in ("half", "sum2", "bar")) + "</tr>"
        for r in rows_g)

    bb = json.load(open(CACHE / "pattern_quantile.json"))
    boot = json.load(open(CACHE / "pq_boot.json"))

    def bbrow(k, lab, sec):
        r = bb[sec][k]
        kn = f" ({r['knob']})" if "knob" in r else ""
        dn = (f"{r['dens'][0]*100:.1f} ± {r['dens'][1]*100:.1f}"
              if "dens" in r else "—")
        return (f"<tr><td>{lab}{kn}</td><td><b>{r['exact']:.1f}%</b></td>"
                f"<td>{r['pm1']:.1f}%</td><td>{r['song_med']:.1f}%</td>"
                f"<td>{r['song_p25']:.0f}–{r['song_p75']:.0f}%</td>"
                f"<td>{r['song_below_chance']:.0f}%</td><td>{dn}</td></tr>")

    tbb = ""
    for sec, lab in (("standalone", "cue seul"), ("fused", "fusionné")):
        tbb += (f"<tr class=hd><td colspan=7>{lab} — 114 morceaux tenus à "
                f"l'écart, {bb[sec]['never_move']['n_starts']} départs</td></tr>")
        tbb += bbrow("never_move", "témoin : ne jamais bouger", sec)
        tbb += bbrow("fixed", "TILE_MIN = 0,80 <i>(ce qui tourne)</i>", sec)
        tbb += bbrow("quantile", "(A) centile par morceau", sec)
        tbb += bbrow("peaks", "(B) détection de pics directe", sec)

    head = f"""<section><h2>La réponse courte</h2>
<p><b>Oui, les pics clairs tombent tous au même endroit de la distribution du
morceau : le 95<sup>e</sup>–96<sup>e</sup> centile.</b> Sur les trois morceaux
dont la grille de mesures est bonne, la médiane est 95,7 / 95,8 / 95,8 %. Le
seuil fixe 0,80, lui, tombe au 81<sup>e</sup> centile sur This Love et au
93<sup>e</sup> sur Sunny — il ne désigne pas la même chose d'un morceau à
l'autre. <b>Mais remplacer 0,80 par un centile ne fait pas gagner de points</b>
sur le banc Billboard : c'est un match nul (+0,4 pp en fusion, −1,3 pp seul,
intervalles de confiance à cheval sur zéro). Ce que le centile achète, ce n'est
pas de la précision, c'est de la <b>stabilité</b> : la densité de la matrice
passe de 26 % ± 17 à 9,9 % ± 0,4.</p>
<p>Et la <b>détection de pics directe</b> (famille B, sans aucun seuil) perd
nettement : 29,8 % contre 37,6 %. Trop de pics par ligne — elle garde 25 % de
la matrice au lieu de 10 %.</p></section>

<section><h2>1. À quel centile tombent les pics ? — par morceau</h2>
<table><tr><th>morceau</th><th>mesures</th><th>centile des pics CLAIRS</th>
<th>écart interquartile</th><th>tous pics</th><th>cosinus au 95<sup>e</sup></th>
<th>centile où tombe 0,80</th></tr>{tq}</table>
<p class=cap>« Pic clair » = maximum local de la ligne, à |décalage| ≥ 2, dont la
proéminence est dans le quart supérieur du morceau. Aucun seuil absolu n'entre
dans cette définition — c'est justement ce qu'on cherche à mesurer.
La dernière colonne est le problème : le même 0,80 vaut le 43<sup>e</sup>
centile sur Billie Jean et le 95<sup>e</sup> sur Georgia.
⚠ = grille de mesures cassée (3,87 et 3,06 temps par mesure), gardée hors de
tout réglage.</p></section>

<section><h2>2. Quel grain donne le signal le plus propre ?</h2>
<p>Louis : « en cumulant les demi-barres adjacentes on trouve un signal
robuste ». Mesuré : hauteur du pic et proéminence, en écarts-types du fond de
la matrice — donc comparable entre grains.</p>
<table><tr><th rowspan=2>morceau</th><th colspan=3>demi-mesure</th>
<th colspan=3>demi-mesures cumulées</th><th colspan=3>mesure</th></tr>
<tr><th>haut.</th><th>proém.</th><th>pics/ligne</th>
<th>haut.</th><th>proém.</th><th>pics/ligne</th>
<th>haut.</th><th>proém.</th><th>pics/ligne</th></tr>{tg}</table>
<p class=cap><b>Louis a raison, et c'est le cumul qui gagne, pas la mesure.</b>
Cumuler deux demi-mesures adjacentes bat la demi-mesure seule sur les cinq
morceaux (hauteur et proéminence), et bat le grain mesure sur quatre sur cinq
en proéminence — tout en gardant la résolution demi-mesure, que le grain mesure
perd. Le gain est modeste (+0,15 à +0,20 σ) mais il va toujours dans le même
sens.</p></section>

<section><h2>3. Le centile bat-il 0,80 ? — 285 morceaux Billboard</h2>
<p>Protocole : les morceaux sont coupés en deux par MORCEAU (171 pour régler,
114 tenus à l'écart, graine fixe) ; chaque famille choisit son réglage sur la
moitié « réglage » uniquement ; les deux sont ensuite mesurées sur la moitié
tenue à l'écart. Égalités tranchées AU HASARD.</p>
<table><tr><th>règle</th><th>mesure exacte</th><th>±1</th>
<th>médiane par morceau</th><th>quartiles</th><th>morceaux ≤ hasard</th>
<th>densité de la matrice</th></tr>{tbb}</table>
<p class=cap>Réglages choisis sur la moitié « réglage » : centile q = 0,90 ;
proéminence = 0,50. Bootstrap par morceau sur la moitié tenue à l'écart :
centile − fixe = <b>{boot['STANDALONE'][0]:+.1f} à {boot['STANDALONE'][1]:+.1f} pp</b>
(cue seul) et <b>{boot['FUSED'][0]:+.1f} à {boot['FUSED'][1]:+.1f} pp</b>
(fusionné) — zéro est dans les deux intervalles. <b>Match nul.</b>
La colonne qui bouge vraiment est la dernière : le seuil fixe laisse passer
26 % de la matrice en moyenne mais avec un écart-type de 17 points ; le centile
en laisse passer 9,9 % ± 0,4 sur tous les morceaux.</p></section>

<section><h2>4. Le dictionnaire de motifs</h2>
<p>Le premier motif est le premier bloc de mesures qui se répète : on balaie les
mesures dans l'ordre, et pour chacune la plus petite longueur L dont
<i>tout le bloc</i> revient (moyenne de la diagonale décalée au-dessus du
95<sup>e</sup> centile du morceau — le seuil adaptatif du §1, pas une
constante). Puis on fait glisser ce carré <b>sur l'axe des X</b>, lignes
figées, et on prend le produit scalaire avec le carré sur lequel il tombe.</p>
<p class=cap>Lien avec la matrice temps-décalage de RefraiD (Goto 2006) : ce
n'est pas la même chose, et il vaut mieux le dire. RefraiD lit une LIGNE de la
matrice temps-décalage, c'est-à-dire un seul décalage à la fois, et cherche des
segments horizontaux. Ici on glisse un CARRÉ : à chaque décalage la comparaison
porte sur les L² relations internes du motif, pas sur L cellules d'une
diagonale. C'est plus proche d'une corrélation de matrices de Gram que d'une
lecture de diagonale, et c'est ce qui la rend insensible à un décalage de
tonalité ou de timbre qui affecterait tout le bloc de la même façon.</p>
</section>"""

    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Le dictionnaire de motifs</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1000px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:{SAND};font-size:13px;margin-bottom:22px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 16px system-ui;margin:0 0 10px;color:{ACC}}}
img{{max-width:100%;border-radius:8px;margin-top:6px}}
.cap{{font-size:12px;color:{SAND};margin:6px 0 0}}
.num{{background:{INK};color:#f4eee2;padding:6px 11px;border-radius:8px;font:600 12.5px system-ui;display:inline-block}}
.warn{{background:#b8860b;color:#fff;font:700 10px system-ui;letter-spacing:.04em;
      padding:3px 7px;border-radius:5px;margin-left:8px;vertical-align:2px}}
table{{border-collapse:collapse;font-size:12px;margin-top:12px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
th{{background:#f7f3e9}}
tr.sus td{{color:#9a8f77;font-style:italic}}
tr.hd td{{background:{INK};color:#f4eee2;font-weight:700}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class=wrap>
<h1>Le dictionnaire de motifs</h1>
<div class=lede>Où tombent les pics de la SSM dans la distribution de leur propre
morceau, et ce qu'on trouve en faisant glisser le premier motif répété le long
de l'axe des X.</div>
{head}{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")
    json.dump(dict(quantiles=rows_q, grains=rows_g),
              open(CACHE / "pattern_dict_songs.json", "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
