"""Onze matrices de voix, côte à côte, pour trois morceaux — à l'œil, sans chiffre.

    .venv/bin/python scripts/voix_variantes.py [<stem> ...]
        -> docs/plots/voix_variantes.html   (local, sans serveur)

Louis, 2026-08-12 : « propose-moi aussi d'autres matrices voix, une dizaine de
variations pour les 3 chansons qu'on a, parce que la distance des notes à
granularité demi-mesure ou quart de mesure peut aussi être intéressante, mais tu
ne peux rien quantifier, il faut que je les voie et que je décide visuellement. »

AUCUN CHIFFRE SUR CETTE PAGE. Onze matrices par morceau, mêmes axes de mesures,
ses frontières en rouge. C'est lui qui tranche.

LES TROIS AXES QU'ON FAIT VARIER, et pourquoi chacun compte :

  * **la granularité** — mesure, demi-mesure, temps. Un lick tient sur un temps
    ou deux ; une section sur une mesure. Ce n'est pas la même matrice.
  * **ce qu'on garde de la note** — sa classe (mod 12), sa hauteur réelle
    (l'octave compte : un refrain se chante souvent plus haut), son intervalle
    avec la précédente (invariant par transposition), son attaque seule (le
    rythme), son contour (monte/descend).
  * **comment on la compte** — au seul instant où elle COMMENCE (ce que fait le
    projet aujourd'hui, et qui rend muettes toutes les cases d'une note tenue),
    ou ÉTALÉE sur toutes les cases qu'elle traverse.

La variante 1 est celle du projet aujourd'hui : c'est le point de comparaison.
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

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections, fig2b64   # noqa: E402
from vote_fill import INK                                          # noqa: E402
from licks import notes_de, distance                               # noqa: E402
import order_bundle                                                # noqa: E402

OUT = HERE / "docs" / "plots" / "voix_variantes.html"
TROIS = ["maroon_5_this_love", "jorja_smith_blue_lights_a_colors_show", "bein_green"]


def cases(grid, n, res):
    """Les bornes des cases : res = 1 (mesure), 2 (demi-mesure), 4 (temps)."""
    t = []
    for b in range(n):
        t0, t1 = grid[b], grid[b + 1]
        t += [t0 + (t1 - t0) * k / res for k in range(res)]
    t.append(grid[n])
    return np.asarray(t)


def _ssm(F):
    """Cosinus des lignes normalisées, les lignes vides mises à zéro."""
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    vide = (nrm[:, 0] <= 1e-9)
    V = F / np.clip(nrm, 1e-9, None)
    S = np.clip(V @ V.T, 0, 1)
    S[vide, :] = 0.0
    S[:, vide] = 0.0
    return S


def f_classe(notes, bords, etale=False):
    """12 demi-tons pondérés par la durée."""
    K = len(bords) - 1
    F = np.zeros((K, 12))
    for t0, d, m in notes:
        if etale:
            i0 = int(np.searchsorted(bords, t0) - 1)
            i1 = int(np.searchsorted(bords, t0 + d) - 1)
            for i in range(max(0, i0), min(K, i1 + 1)):
                a = max(t0, bords[i]); z = min(t0 + d, bords[i + 1])
                if z > a:
                    F[i, int(m) % 12] += (z - a)
        else:
            i = int(np.searchsorted(bords, t0) - 1)
            if 0 <= i < K:
                F[i, int(m) % 12] += d
    return F


def f_hauteur(notes, bords, etale=True):
    """La hauteur RÉELLE, sur trois octaves — l'octave n'est plus jetée."""
    K = len(bords) - 1
    ms = [int(m) for _t, _d, m in notes]
    lo = min(ms) if ms else 48
    F = np.zeros((K, 37))
    for t0, d, m in notes:
        p = int(m) - lo
        if not (0 <= p < 37):
            continue
        i0 = int(np.searchsorted(bords, t0) - 1)
        i1 = int(np.searchsorted(bords, t0 + d) - 1) if etale else i0
        for i in range(max(0, i0), min(K, i1 + 1)):
            F[i, p] += 1.0
    return F


def f_intervalles(notes, bords):
    """L'histogramme des INTERVALLES — invariant par transposition."""
    K = len(bords) - 1
    F = np.zeros((K, 25))
    for (t0, _d, m0), (_t1, _d1, m1) in zip(notes, notes[1:]):
        i = int(np.searchsorted(bords, t0) - 1)
        v = int(round(m1 - m0)) + 12
        if 0 <= i < K and 0 <= v < 25:
            F[i, v] += 1.0
    return F


def f_rythme(notes, bords, sous=4):
    """Les positions d'ATTAQUE dans la case — la figure rythmique seule."""
    K = len(bords) - 1
    F = np.zeros((K, sous))
    for t0, _d, _m in notes:
        i = int(np.searchsorted(bords, t0) - 1)
        if 0 <= i < K:
            a, z = bords[i], bords[i + 1]
            F[i, min(sous - 1, int(sous * (t0 - a) / max(1e-9, z - a)))] += 1.0
    return F


def f_contour(notes, bords):
    """Monte / reste / descend, plus le registre moyen de la case."""
    K = len(bords) - 1
    F = np.zeros((K, 4))
    for (t0, _d, m0), (_t1, _d1, m1) in zip(notes, notes[1:]):
        i = int(np.searchsorted(bords, t0) - 1)
        if 0 <= i < K:
            F[i, 0 if m1 > m0 + 0.5 else (2 if m1 < m0 - 0.5 else 1)] += 1.0
            F[i, 3] += (m0 - 60) / 12.0
    return F


def f_squelette(notes, bords):
    """La note la plus LONGUE de la case, en classe — la mélodie squelette."""
    K = len(bords) - 1
    F = np.zeros((K, 12))
    best = {}
    for t0, d, m in notes:
        i = int(np.searchsorted(bords, t0) - 1)
        if 0 <= i < K and d > best.get(i, (0, 0))[0]:
            best[i] = (d, int(m) % 12)
    for i, (_d, c) in best.items():
        F[i, c] = 1.0
    return F


def ssm_sequence(notes, bords, fen=2):
    """La distance d'ALIGNEMENT des licks, ramenée à la case.

    Chaque case porte la séquence `(intervalle, écart)` des notes de la fenêtre
    qui commence là ; deux cases se comparent par la distance d'édition. C'est la
    seule variante qui garde l'ORDRE des notes.
    """
    K = len(bords) - 1
    idx = np.searchsorted(bords, [nt[0] for nt in notes]) - 1
    seqs = []
    for i in range(K):
        mem = [(k, notes[k]) for k, j in enumerate(idx) if i <= j < i + fen]
        seq = [(int(round(b[2] - a[2])), float(b[0] - a[0]))
               for (_ka, a), (_kb, b) in zip(mem, mem[1:])]
        seqs.append(seq)
    S = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            if not seqs[i] or not seqs[j]:
                v = 0.0
            else:
                v = max(0.0, 1.0 - distance(seqs[i], seqs[j]))
            S[i, j] = S[j, i] = v
    return S


def variantes(stem):
    """[(titre, gloss, S, res)] — les onze matrices."""
    b = order_bundle.get(stem)
    n, grid = b["n"], np.asarray(b["grid"])
    notes = notes_de(stem)
    B1, B2, B4 = (cases(grid, n, r) for r in (1, 2, 4))
    V = []
    V.append(("1 · classes, mesure", "CE QU'ON UTILISE AUJOURD'HUI : 12 demi-tons "
              "pondérés par la durée, comptés là où la note commence",
              _ssm(f_classe(notes, B1)), 1))
    V.append(("2 · classes, demi-mesure", "la même, deux fois plus fine",
              _ssm(f_classe(notes, B2)), 2))
    V.append(("3 · classes, temps", "la même, au temps — la granularité d'un lick",
              _ssm(f_classe(notes, B4)), 4))
    V.append(("4 · classes étalées, demi-mesure",
              "une note tenue compte dans TOUTES les cases qu'elle traverse, plus "
              "seulement dans celle où elle commence",
              _ssm(f_classe(notes, B2, etale=True)), 2))
    V.append(("5 · hauteur réelle, demi-mesure",
              "l'octave n'est plus jetée : un refrain chanté plus haut se "
              "distingue d'un couplet sur les mêmes notes",
              _ssm(f_hauteur(notes, B2)), 2))
    V.append(("6 · hauteur réelle, temps", "la même, au temps",
              _ssm(f_hauteur(notes, B4)), 4))
    V.append(("7 · intervalles, demi-mesure",
              "l'histogramme des intervalles — invariant par transposition, donc "
              "le seul candidat pour Sunny qui module",
              _ssm(f_intervalles(notes, B2)), 2))
    V.append(("8 · rythme seul, demi-mesure",
              "où tombent les ATTAQUES dans la case, sans aucune hauteur — la "
              "figure rythmique du chant",
              _ssm(f_rythme(notes, B2)), 2))
    V.append(("9 · contour, demi-mesure",
              "monte / reste / descend, plus le registre moyen",
              _ssm(f_contour(notes, B2)), 2))
    V.append(("10 · squelette, demi-mesure",
              "la note la plus longue de la case, et rien d'autre",
              _ssm(f_squelette(notes, B2)), 2))
    V.append(("11 · séquence alignée, demi-mesure",
              "la SEULE qui garde l'ORDRE des notes : distance d'édition sur "
              "(intervalle, écart d'attaque), comme la page des licks",
              ssm_sequence(notes, B2), 2))
    return V, n, b


def song_png(stem):
    V, n, b = variantes(stem)
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("h", ["#faf6ec", "#9fc0d4", "#1d4d69"])
    cols = 4
    rows = int(np.ceil(len(V) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(13.2, 3.4 * rows),
                            facecolor="#fffdf6",
                            gridspec_kw={"wspace": 0.14, "hspace": 0.28})
    axs = np.atleast_1d(axs).ravel()
    for ax, (titre, _g, S, _r) in zip(axs, V):
        S = np.nan_to_num(S)
        ax.imshow(S, cmap=cmap, vmin=0, vmax=1, extent=[0, n, n, 0],
                  interpolation="nearest", aspect="auto")
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.6, alpha=0.6)
            ax.axhline(g, color=GT_LINE, lw=0.6, alpha=0.6)
        ax.set_title(titre, fontsize=9, color="#4a4438", pad=5)
        step = 16 if n <= 120 else 32
        ax.set_xticks(np.arange(0, n + 1, step)); ax.set_yticks(np.arange(0, n + 1, step))
        ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7)
        ax.set_yticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7)
        ax.tick_params(length=2, colors="#8a8371")
        for sp in ax.spines.values():
            sp.set_color("#e0d7c2")
    for ax in axs[len(V):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.995, top=0.955, bottom=0.03)
    return fig2b64(fig), V, n


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    stems = args or TROIS
    titres = dict(SONGS)
    body = []
    for stem in stems:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        img, V, n = song_png(stem)
        glos = "".join(f"<li><b>{t}</b> — {g}</li>" for t, g, _s, _r in V)
        body.append(
            f'<section><h2>{titres.get(stem, stem)} <span class=sub>{n} mesures'
            f'</span></h2><img src="data:image/png;base64,{img}" alt="{stem}">'
            f'<ul class=gl>{glos}</ul></section>')
        print(f"  ok {titres.get(stem, stem)}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Onze matrices de voix</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1340px;margin:0 auto;padding:22px 14px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:980px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:12px 14px;margin-bottom:12px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block}}
ul.gl{{margin:10px 0 0;padding-left:18px;font-size:12.5px;color:#6f6857;columns:2;column-gap:26px}}
ul.gl li{{margin-bottom:4px;break-inside:avoid}} ul.gl b{{color:{INK}}}
</style></head><body><div class=wrap>
<h1>Onze matrices de voix</h1>
<div class=lede>Trois morceaux, onze façons de mesurer la ressemblance du chant,
mêmes axes de mesures, tes frontières en rouge. <b>Aucun chiffre</b> — c'est à
l'œil.<br><br>
Trois choses varient : <b>la granularité</b> (mesure, demi-mesure, temps),
<b>ce qu'on garde de la note</b> (sa classe, sa hauteur réelle, son intervalle,
son attaque, son contour) et <b>comment on la compte</b> (au seul instant où elle
commence — ce que fait le projet aujourd'hui, ce qui rend muettes toutes les
cases d'une note tenue — ou étalée sur toute sa durée).<br><br>
La <b>n°1 est celle qu'on utilise aujourd'hui</b> : c'est le point de
comparaison. Ce qu'on cherche à l'œil : des <b>carrés</b> nets sur la diagonale
(les sections se tiennent) et des <b>blocs hors diagonale</b> aux bons endroits
(les reprises), sans que tout devienne uniformément sombre — une matrice où tout
se ressemble ne dit rien.</div>
{''.join(body)}</div></body></html>""")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
