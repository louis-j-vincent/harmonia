"""Les couloirs de motifs : la matrice, puis un couloir par motif, puis le vainqueur par mesure.

    python scripts/pattern_lanes.py [<stem> ...]   ->  /reports/pattern_lanes.html

Louis, 2026-08-05 : « Pour chaque musique, tu me plot la matrice, et en dessous
(sharex=True) chaque ligne qui correspond à une section, avec les recouvrements
possibles qu'ils occupent, représentés comme un surlignage de barre, donc
l'opacité est fonction de la valeur du pic. Ça permet de voir visuellement les
alignements de chaque répétition, et que je puisse voir quelles règles faire
pour gérer au mieux la détection de sections. Chaque pattern est une ligne
différente et une couleur différente. Enfin la dernière ligne, c'est le
découpage par barre, avec à chaque barre en surlignant la couleur du pattern qui
a le plus haut score pour cette barre. »

Ce que dessine chaque couloir : le motif glissé sur TOUTE la chanson
(`harmonic_sections.slide`, la lecture en diagonale), et à chaque maximum local
de cette courbe un rectangle qui occupe exactement les mesures que le motif
occuperait s'il commençait là. L'opacité est le score. Aucun seuil n'est
appliqué : les placements refusés par la règle de pic sont dessinés comme les
autres, en pâle — c'est tout l'intérêt de la vue.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.signal import find_peaks      # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from ssm_rows_plot import fig2b64                                  # noqa: E402
from harmonia_min import sections as hs, musx as mx                # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402

INK = "#1c1c1c"
COLS = ["#b3261e", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed", "#0f766e",
        "#be123c", "#0369a1", "#a16207", "#4338ca"]
ALPHA_LO, ALPHA_HI = 0.80, 1.00   # l'échelle d'opacité, annoncée sur la page
ALPHA_K = 3.2                     # …et sa courbure. Louis, 2026-08-05 : « je la
                                  # veux exponentielle, de manière à ce que je
                                  # puisse différencier un 0,99 d'un 0,92 ; elle
                                  # doit aller de 0,80 à 1 »
SOLID = 0.95                      # Louis : au-dessus, un placement est SOLIDE —
                                  # il verrouille ses mesures pour les motifs
                                  # suivants, qui n'ont plus le droit d'y jouer
DEFAULT = ["bruno_mars_grenade_official_music_video"]


def alpha_of(s):
    """score -> opacité, EXPONENTIELLE sur [0,80 ; 1,00].

    Tout l'intérêt d'un score de similarité est dans son dernier dixième : 0,92
    et 0,99 ne veulent pas dire la même chose musicalement, alors que 0,4 et 0,6
    veulent dire « rien à voir » tous les deux. Une échelle linéaire les rend
    indiscernables. Ici 0,80 → invisible, 0,92 → 0,28, 0,95 → 0,45, 0,99 → 0,84.
    """
    t = float(np.clip((s - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0, 1))
    return 0.05 + 0.93 * (np.expm1(ALPHA_K * t) / np.expm1(ALPHA_K))


def refine(grid, u):
    """Découpe chaque mesure en `u` cases égales. u=1 : mesure ; u=2 : demi-mesure."""
    if u == 1:
        return list(map(float, grid))
    out = []
    for a, b in zip(grid[:-1], grid[1:]):
        out += [float(a) + (float(b) - float(a)) * j / u for j in range(u)]
    return out + [float(grid[-1])]


def load(stem, u=1):
    """Rejoue le pipeline pour récupérer la grille de mesures + la matrice."""
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    c = {}

    def spy(g, a, t, bars=None, **k):
        c.update(grid=g, bars=copy.deepcopy(bars))
        return real(g, a, t, bars, **k)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x", file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    grid = refine(c["grid"], u)
    V = HS.harmonic_vectors(mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0], grid)
    return V @ V.T, len(grid) - 1, grid


# Marges FIXES : le curseur de lecture est un div posé sur l'image, donc la
# mesure b doit tomber à une position connue en pourcentage de la largeur. Un
# `bbox_inches="tight"` rognerait de façon imprévisible et décalerait le curseur.
PLOT_L, PLOT_R = 0.118, 0.995


# ── LE COMPTAGE DES MESURES, UNE FOIS POUR TOUTES ───────────────────────────
# Louis, 2026-08-07 : « petit décalage entre les pics et les blocs, un que tu
# commences mesure 0, l'autre mesure 1 — est-ce que le comptage des mesures est
# bien uniforme dans notre code ? »
#
# Il ne l'était pas. Trois conventions cohabitaient pour poser un trait de
# frontière : `b0`, `b0 + .5` et `b0 - .5`, et une courbe posée en `c + L/2`.
#
# LA CONVENTION, valable partout où l'axe des x est en mesures (imshow avec
# `extent=(0, n, 0, n)`, rectangles `Rectangle((b0, y), longueur, h)`) :
#
#     la mesure d'indice i occupe l'intervalle [i, i+1]
#     une FRONTIÈRE (le début de la mesure i)        ->  x = i        = edge(i)
#     une VALEUR qui appartient à la mesure i        ->  x = i + 0.5  = mid(i)
#
# Et pour l'affichage humain, la mesure i s'écrit « mesure i+1 » : l'indice 0
# est la première mesure. Le curseur de lecture suit la même règle — la
# fraction de largeur `f/n` place le trait au DÉBUT de la mesure f.
#
# Une série indexée par mesure se pose donc en `mid(np.arange(n))`, jamais en
# `arange(n)` tout court, et surtout jamais décalée d'une demi-longueur de bloc :
# c'est ce `c + L/2` qui décalait les pics de deux mesures sur les blocs de
# quatre dans `mini_threshold`.

def edge(i):
    """Frontière : le DÉBUT de la mesure i, en coordonnées du tracé."""
    return i


def mid(i):
    """Le milieu de la mesure i — là où se pose une valeur qui lui appartient."""
    return i + 0.5


# ── LES COULEURS DE SEUIL, PARTAGÉES ────────────────────────────────────────
# Louis, 2026-08-07 : « il faut que le jeu de couleurs des pics/seuils soit
# cohérent avec les couleurs des bi-barres avec décile 0.9 ». Le décile 90 était
# vert sur `bibar_voice` et violet sur `voice_first` — même seuil, deux couleurs,
# donc impossible de lire les deux pages ensemble. Un seuil = une couleur, ici.
THR_COLS = {"absolu": "#8a8371", "q90": "#1f8a5b", "q97": "#b3261e"}


def fig2b64_fixed(fig):
    import base64
    import io
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110)     # surtout PAS bbox_inches="tight"
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def blab(i, u):
    """index de case → étiquette de mesure : 12 en pleine mesure, 12½ en demie."""
    return f"{i // u + 1}" + ("½" if u == 2 and i % 2 else "")


def lock_pass(S, n, cells):
    """LE VERROU, calculé avant tout dessin. Louis, 2026-08-05 :

      « Si un bloc est à 95+, alors TOUT le bloc disparaît : si le bloc est de
        2 mesures, les 2 mesures disparaissent pour la détection du bloc
        suivant ; s'il est de 4 mesures, ce sont les 4 mesures. »

    Aucune exception, pas même pour l'ancre d'un motif : un motif qui voudrait
    s'ancrer dans des mesures déjà verrouillées **n'existe pas**. C'est la
    correction du 2026-08-05 — j'avais d'abord exempté l'ancre, ce qui laissait
    The Walk garder deux motifs posés en plein milieu de matière déjà reconnue
    à 0,99.

    Les placements d'un même motif se verrouillent aussi entre eux, par score
    décroissant : sinon un motif de 8 mesures sur une boucle de 2 mesures se
    dessine trente fois, à cheval sur lui-même.

    Retourne, par motif : None s'il est supprimé, sinon
    {"keep": [(départ, score)], "cut": [départs retirés]}.
    """
    solid = np.zeros(n, bool)
    out = []
    for e in cells:
        L, curve = e["L"], e["curve"]
        if solid[e["b0"]:min(n, e["b0"] + L)].any():
            out.append(None)                       # ancre déjà prise → supprimé
            continue
        pk, _ = find_peaks(curve)
        starts = [c for c in sorted(set(pk.tolist()) | {e["b0"]} | set(e["occ"]))
                  if c < len(curve)
                  and (float(curve[c]) >= ALPHA_LO or c in e["occ"] or c == e["b0"])]
        keep, cut, local = [], [], solid.copy()
        for c0 in sorted(starts, key=lambda c: -float(curve[c])):
            if local[c0:min(n, c0 + L)].any():
                cut.append(c0)
                continue
            s = float(curve[c0])
            keep.append((c0, s))
            if s >= SOLID:
                local[c0:min(n, c0 + L)] = True
        solid = local
        out.append({"keep": sorted(keep), "cut": sorted(cut)})
    return out, solid


def lanes_figure(S, n, cells, secs, u=1):
    """La figure demandée : matrice, un couloir par motif, vainqueur par mesure."""
    lanes, solid_all = lock_pass(S, n, cells)
    live = [i for i, l in enumerate(lanes) if l is not None]
    k = len(live)
    heights = [5.6] + [0.72] * k + [0.30, 0.95]      # 0.30 = respiration
    H = 5.6 + 0.72 * k + 2.2
    fig, axs = plt.subplots(
        len(heights), 1, sharex=True, figsize=(12.6, H),
        gridspec_kw={"height_ratios": heights, "hspace": 0.16})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - 0.34 / H,
                        bottom=0.60 / H)

    # ── la matrice ─────────────────────────────────────────────────────────
    ax = axs[0]
    ax.imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
              vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
              aspect="auto", interpolation="nearest")
    ax.set_ylabel("mesure", fontsize=8)
    yt = list(range(0, n + 1, 4 * u * max(1, round(n / u / 40))))
    ax.set_yticks(yt)
    ax.set_yticklabels([str(t // u) for t in yt])
    ax.tick_params(labelsize=7)
    cell = "1 mesure" if u == 1 else "1 demi-mesure"
    ax.set_title(f"matrice de similarité harmonique (1 case = {cell} × {cell})",
                 fontsize=8.5, color="#6f6858", pad=4)

    # ── un couloir par motif SURVIVANT ─────────────────────────────────────
    per_bar_best = np.zeros(n)          # meilleur score couvrant chaque mesure
    per_bar_who = np.full(n, -1, int)   # …et le motif qui le réalise
    dropped = [0] * len(cells)          # placements retirés parce que déjà pris
    for row, idx in enumerate(live):
        e, lane = cells[idx], lanes[idx]
        ax, col, L = axs[1 + row], COLS[idx % len(COLS)], e["L"]
        dropped[idx] = len(lane["cut"])
        for c0 in lane["cut"]:          # ce que le verrou a coupé, en creux
            ax.add_patch(plt.Rectangle(
                (c0, .12), L, .76, facecolor="none", edgecolor="#c9c1ab",
                lw=.8, ls=(0, (2, 2)), zorder=2))
        for c0, s in lane["keep"]:
            kept = c0 in e["occ"] or c0 == e["b0"]
            ax.add_patch(plt.Rectangle(
                (c0, .12), L, .76, facecolor=col, alpha=alpha_of(s),
                edgecolor=(INK if kept else "none"), lw=1.15, zorder=3 if kept else 2))
            if s >= SOLID:
                ax.add_patch(plt.Rectangle(
                    (c0, .12), L, .76, facecolor="none", edgecolor=INK,
                    lw=1.6, zorder=5))
            if s >= 0.90 or kept:
                ax.text(c0 + L / 2, .5, f"{s:.2f}", ha="center", va="center",
                        fontsize=5.4, rotation=90 if L <= 2 else 0,
                        color="#fff" if alpha_of(s) > .55 else INK, zorder=6)
            for b in range(c0, min(n, c0 + L)):
                if s > per_bar_best[b]:
                    per_bar_best[b], per_bar_who[b] = s, idx
        ax.axvspan(e["b0"], e["b0"] + L, color=col, alpha=.10, zorder=1)
        tag = "trou" if e.get("from_gap") else f"mes. {blab(e['b0'], u)}"
        ax.set_ylabel(f"motif {idx+1}\n{L/u:g} mes. · {tag}", fontsize=6.6, rotation=0,
                      ha="right", va="center", color=col)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#ddd5c0")

    # ── l'échelle d'opacité, dessinée pour qu'on puisse la lire sur la figure ─
    ax = axs[1 + k]
    ax.set_ylim(0, 1)
    ax.set_xlim(0, n)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks([])
    for j, v in enumerate((0.80, 0.85, 0.90, 0.95, 0.98, 1.00)):
        x = n * (0.012 + 0.030 * j)
        ax.add_patch(plt.Rectangle((x, .25), n * 0.026, .58, facecolor=INK,
                                   alpha=alpha_of(v), edgecolor="none"))
        ax.text(x + n * 0.013, .05, f"{v:.2f}", ha="center", va="bottom",
                fontsize=5.6, color="#8a8371")
    ax.text(n * 0.205, .5, "opacité = score de similarité (échelle exponentielle)",
            ha="left", va="center", fontsize=6.4, color="#8a8371")

    # ── la dernière ligne : le vainqueur par mesure ────────────────────────
    ax = axs[-1]
    for b in range(n):
        w = per_bar_who[b]
        if w < 0:
            continue
        ax.add_patch(plt.Rectangle((b, .34), 1, .62, facecolor=COLS[w % len(COLS)],
                                   alpha=alpha_of(per_bar_best[b]), edgecolor="none"))
    # …et, en dessous, ce que le pipeline écrit aujourd'hui (référence)
    seen = {}
    for s in secs:
        seen.setdefault(s["letter"], "#8a8371")
        ax.add_patch(plt.Rectangle((s["b0"], .02), s["b1"] - s["b0"] + 1, .26,
                                   facecolor="#efe8d6", edgecolor="#b9b09a", lw=.7))
        ax.text((s["b0"] + s["b1"] + 1) / 2, .15, s["letter"], ha="center",
                va="center", fontsize=6.4, color="#6f6858", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("vainqueur\npar mesure\n· sections\nactuelles", fontsize=6.6,
                  rotation=0, ha="right", va="center", color=INK)
    ax.set_xlim(0, n)
    ticks = list(range(0, n + 1, 4 * u))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t // u) for t in ticks])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("mesure", fontsize=8)
    return fig2b64_fixed(fig), {"dropped": dropped, "solid": solid_all,
                                "lanes": lanes, "live": live}


def _btn(b0, b1, col, label, sub=""):
    """Un bouton qui joue exactement les mesures b0..b1 (0-indexé) et s'arrête."""
    return (f"<button class=blk style='border-color:{col};color:{col}' "
            f"data-p='[{b0},{b1 + 1},\"{col}\"]'>{label}"
            + (f"<small>{sub}</small>" if sub else "") + "</button>")


def song_html(stem, u=1):
    S, n, grid = load(stem, u)
    cells, _ = HS.build_cells(S, n)
    secs = HS.sections_from(S, n, cells)
    img, info = lanes_figure(S, n, cells, secs, u)
    live, lanes = info["live"], info["lanes"]
    stats = {"n": n, "cells": len(live), "killed": len(cells) - len(live),
             "letters": len({s["letter"] for s in secs}),
             "solid": float(info["solid"].mean()), "dropped": sum(info["dropped"])}

    rows = ""
    for i, e in enumerate(cells):
        d = info["dropped"][i]
        dead = lanes[i] is None
        occ = ("—" if dead else
               " ".join(blab(c, u) for c, _ in lanes[i]["keep"]))
        rows += (f"<tr class='{'out' if dead else ''}'>"
                 f"<td><span class=dot style='background:"
                 f"{'#c9c1ab' if dead else COLS[i%len(COLS)]}'></span>"
                 f"motif {i+1}</td><td>{e['L']/u:g}</td><td>{blab(e['b0'], u)}</td>"
                 f"<td>{occ}</td><td>{'trou' if e.get('from_gap') else 'ancré'}</td>"
                 f"<td>{'<b>motif supprimé</b> — son ancre est déjà verrouillée'
                        if dead else (d or '—')}</td></tr>")

    # les boutons : d'abord les sections écrites, puis les placements de chaque
    # motif — c'est là qu'on entend si deux occurrences sont vraiment la même
    # musique ou si le verrou a rapproché deux choses différentes
    sec_btns = "".join(_btn(s["b0"], s["b1"], "#6f6858",
                            f"▶ {s['letter']}", f"mes. {blab(s['b0'], u)}–{blab(s['b1'], u)}")
                       for s in secs)
    lane_btns = ""
    for i in live:
        e, col = cells[i], COLS[i % len(COLS)]
        pl = [c for c, _ in lanes[i]["keep"]]
        lane_btns += (
            f"<div class=lane><span class=lab style='color:{col}'>motif {i+1} · "
            f"{e['L']/u:g} mes.</span>"
            + "".join(_btn(o, min(n - 1, o + e["L"] - 1), col, f"▶ {blab(o, u)}")
                      for o in pl)
            + "</div>")

    wsec = " ".join(f"{s['letter']}[{blab(s['b0'], u)}-{blab(s['b1'], u)}]" for s in secs)
    return f"""<section><h2>{stem.replace('_',' ').title()}
<span class=sub>{n // u} mesures{' (lues en demi-mesures)' if u == 2 else ''} ·
{len(live)} motifs{f' (+{len(cells)-len(live)} supprimés par le verrou)' if len(cells) > len(live) else ''} ·
{info['solid'].mean():.0%} des mesures verrouillées à ≥ {SOLID:.2f} ·
{sum(info['dropped'])} placements retirés</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>sections écrites</span>{sec_btns}</div>
{lane_btns}
<table><tr><th>motif</th><th>longueur</th><th>ancre</th>
<th>placements retenus (mesures, 1-indexé)</th><th>origine</th>
<th>retirés (déjà solides)</th></tr>{rows}</table>
<p class=verdict>sections écrites aujourd'hui : {wsec}</p></section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U={u};</script>""", stats


def page(title, body, back=False, lede=True, half=False):
    LEDE = f"""<div class=lede>En haut la matrice. Puis <b>un couloir par motif</b>,
même axe des mesures : chaque rectangle est un placement possible de ce motif — il
occupe exactement les mesures que le motif couvrirait s'il commençait là, et son
<b>opacité est son score</b>, sur une échelle <b>exponentielle de {ALPHA_LO:.2f} à
{ALPHA_HI:.2f}</b> — tout se joue dans le dernier dixième, donc 0,92 et 0,99 ne
doivent pas se ressembler (0,92 → 28 %, 0,95 → 45 %, 0,99 → 84 %). En dessous de
{ALPHA_LO:.2f} un placement ne dit rien et n'est pas dessiné.
Aucun autre seuil : les placements <b>refusés</b> par la règle de pic sont là,
en pâle et sans contour ; ceux qui sont <b>retenus</b> ont un contour noir. La
bande pâle du fond marque l'ancre du motif.<br><br>
<b>Le verrou à {SOLID:.2f}</b> — dès qu'un placement dépasse ce score, il est tenu
pour solide (cadre noir épais) et <b>tout son bloc disparaît</b> : ses mesures,
toutes ses mesures, sont retirées de la recherche qui suit. Un motif de 2 mesures
en retire 2, un motif de 8 en retire 8. <b>Sans exception</b> : un motif dont
l'ancre tombe dans des mesures déjà verrouillées <b>n'existe pas</b> — son
couloir n'est pas dessiné et le tableau le note « motif supprimé ». Les
placements retirés restent visibles en pointillé gris et à vide, pour qu'on voie
ce que le verrou a coupé.<br><br>
Dernière ligne : <b>pour chaque mesure, la couleur du motif dont le placement a
le plus haut score sur cette mesure</b>, et juste en dessous, en gris, les
sections que le pipeline écrit aujourd'hui.</div>"""
    if half:
        LEDE += f"""<div class=lede style="background:#f7f3e9;border-radius:8px;
padding:9px 11px"><b>Lecture en demi-mesures.</b> La matrice compare des
demi-mesures, pas des mesures : elle est deux fois plus fine dans chaque
direction, et la recherche de motifs travaille dans la même unité. Conséquence à
garder en tête — les bornes de la recherche comptent en cases, donc elles
signifient ici <b>1 à 8 mesures</b> (au lieu de 2 à 16) et une section peut
descendre à <b>une mesure</b>. Les étiquettes restent en mesures ; « 12½ » est la
deuxième moitié de la mesure 12.</div>"""
    return f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{title}</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
tr.out td{{color:#a89f8c;background:#f4efe3}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.verdict{{font-size:13px;background:#f7f3e9;border-radius:8px;padding:9px 11px;margin:10px 0 0}}
a{{color:#8a2b2b}}
.idx{{border-collapse:collapse;font-size:13px;width:100%;background:#fffdf6}}
.idx td,.idx th{{border:1px solid #e5dcc6;padding:7px 9px}}
.idx th{{background:#f7f3e9;font-size:11px;text-align:left}}
.idx td a{{display:block;font-weight:600;text-decoration:none}}
.back{{display:inline-block;margin-bottom:12px;font-size:13px;text-decoration:none}}
.plot{{position:relative;margin-bottom:8px}}
.plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;align-items:center;gap:10px;margin:0 0 10px}}
.pp{{width:42px;height:42px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:15px;cursor:pointer;flex:none}}
.pos{{font:600 12.5px ui-monospace,monospace}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
.lane{{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:0 0 6px}}
.lab{{font:600 11px system-ui;color:#8a8371;width:104px;flex:none}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:7px;
  padding:5px 8px;cursor:pointer;font:600 12px system-ui;color:#6f6858}}
button.blk small{{display:block;font:500 9.5px ui-monospace,monospace;
  color:#a89f8c;margin-top:1px}}
button.blk.on{{color:#fff !important}} button.blk.on small{{color:#f3ece0}}
</style></head><body><div class=wrap>
{'<a class=back href="pattern_lanes.html">← toutes les chansons</a>' if back else ''}
<h1>{title}</h1>
{LEDE if lede else ''}
{body}</div>
<script>
(function(){{
  const au=document.getElementById("au"); if(!au) return;
  const G=window.GRID, n=G.length-1, L0=window.PLOT[0], W=window.PLOT[1]-L0;
  const plot=document.querySelector(".plot"), cur=plot.querySelector(".cur"),
        hit=plot.querySelector(".hit"), pp=document.querySelector(".pp"),
        pos=document.querySelector(".pos");
  hit.style.left=(L0*100)+"%"; hit.style.width=(W*100)+"%";
  // temps -> mesure (flottante) et retour ; la grille n'est PAS régulière,
  // donc on interpole entre deux frontières de mesure, jamais en tempo fixe
  const t2b=t=>{{ if(t<=G[0])return 0; if(t>=G[n])return n;
    let lo=0,hi=n; while(hi-lo>1){{const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}}
    return lo+(t-G[lo])/(G[lo+1]-G[lo]); }};
  const b2t=f=>{{ const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    return G[i]+(f-i)*(G[i+1]-G[i]); }};
  const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
  let stopAt=null, raf=null, onBtn=null;
  function draw(){{
    const f=t2b(au.currentTime);
    cur.style.display="block";
    cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
    pos.textContent="mes. "+(Math.floor(f/window.U)+1)+" · "+fmt(au.currentTime);
  }}
  function clear(){{ if(onBtn){{ onBtn.classList.remove("on");
    onBtn.style.background="#f7f3e9"; onBtn=null; }} }}
  function tick(){{ draw();
    if(stopAt!=null && au.currentTime>=stopAt){{ au.pause(); stopAt=null; clear(); }}
    if(!au.paused) raf=requestAnimationFrame(tick); }}
  function go(t0,t1,btn){{
    clear(); stopAt=t1;
    if(btn){{ onBtn=btn; btn.classList.add("on");
      btn.style.background=btn.style.borderColor; }}
    const seek=()=>{{ try{{ au.currentTime=t0; }}catch(e){{}} draw(); }};
    if(au.readyState>=1) seek();
    else au.addEventListener("loadedmetadata",seek,{{once:true}});
    au.play().catch(()=>{{ clear(); }});
  }}
  au.addEventListener("play",()=>{{ pp.textContent="❚❚"; tick(); }});
  au.addEventListener("pause",()=>{{ pp.textContent="▶";
    cancelAnimationFrame(raf); draw(); }});
  pp.onclick=()=>{{ if(au.paused){{ stopAt=null; clear(); au.play().catch(()=>{{}}); }}
                    else au.pause(); }};
  hit.onclick=e=>{{ const r=hit.getBoundingClientRect();
    go(b2t(n*(e.clientX-r.left)/r.width), null, null); }};
  document.querySelectorAll("[data-p]").forEach(b=>{{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(G[d[0]], G[Math.min(n,d[1])], b); }});
}})();
</script></body></html>"""


def main():
    # --demi : la matrice et toute la recherche de motifs en DEMI-MESURES.
    # Louis, 2026-08-05 : « sur Yesterday fais-moi la même matrice SSM mais
    # granularité 1/2 barre plutôt qu'une barre. »  Les constantes de
    # `harmonic_sections` comptent en CASES, pas en mesures : en demi-mesure,
    # LAG_MIN/LAG_MAX = 2/16 veulent donc dire 1 à 8 mesures au lieu de 2 à 16,
    # et MIN_SECTION_BARS = 2 veut dire une mesure. La page le dit.
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    u = 2 if "--demi" in sys.argv[1:] else 1
    suffix = "_demibarre" if u == 2 else ""
    stems = argv or sorted(p.stem for p in (HERE / "docs/audio").glob("*.m4a"))
    OUT = HERE / "harmonia_min/state/reports"
    done, failed = [], []
    for i, st in enumerate(stems, 1):
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            body, stats = song_html(st, u)
        except Exception as exc:                       # grille refusée, etc.
            failed.append((st, f"{type(exc).__name__}: {exc}"))
            print(f"  !! {i}/{len(stems)} {st} — {type(exc).__name__}: {exc}")
            continue
        f = OUT / f"pattern_lanes_{st}{suffix}.html"
        f.write_text(page(st.replace("_", " ").title()
                          + (" — demi-mesures" if u == 2 else ""), body,
                          back=True, half=(u == 2)))
        done.append((st, stats, f.stat().st_size // 1024))
        print(f"  ok {i}/{len(stems)} {st}  ({f.stat().st_size // 1024} KB)")

    rows = ""
    for st, s, kb in sorted(done, key=lambda d: -d[1]["solid"]):
        rows += (f"<tr><td><a href='pattern_lanes_{st}{suffix}.html'>"
                 f"{st.replace('_',' ').title()}</a></td>"
                 f"<td>{s['n']}</td><td>{s['cells']}</td><td>{s['letters']}</td>"
                 f"<td>{s['solid']:.0%}</td><td>{s['dropped']}</td></tr>")
    for st, why in failed:
        rows += (f"<tr><td style='color:#a89f8c'>{st.replace('_',' ').title()}</td>"
                 f"<td colspan=5 style='color:#a89f8c'>{why}</td></tr>")
    idx = (f"<p class=lede><b>{len(done)} chansons</b>, une page chacune — trop lourd "
           f"en une seule. Triées par part des mesures verrouillées à ≥ {SOLID:.2f} : "
           f"en haut celles dont la structure est franche, en bas celles où rien ne "
           f"se répète assez pour verrouiller quoi que ce soit."
           + (f" {len(failed)} refusées en amont (grille de mesures)." if failed else "")
           + "</p>"
           f"<table class=idx><tr><th>chanson</th><th>mesures</th><th>motifs</th>"
           f"<th>lettres</th><th>verrouillé</th><th>placements retirés</th></tr>"
           f"{rows}</table>")
    out = OUT / f"pattern_lanes{suffix}.html"
    out.write_text(page("Couloirs de motifs" + (" — demi-mesures" if u == 2 else ""),
                        idx, lede=False))
    print(f"\nwrote {out.relative_to(HERE)} — {len(done)} ok, {len(failed)} refusées")


if __name__ == "__main__":
    main()
