"""Le critère de Louis : comparer un bloc DIAGONAL à un bloc CROISÉ, pas à pas.

    .venv/bin/python scripts/match_bloc.py [<stem>]
        -> docs/plots/match_bloc.html    (local, sans serveur)

Louis, 2026-08-13 :

  * matrice **hauteur réelle · temps** (la n°6 de la page des onze variantes) ;
  * pour comparer la section A à la position B, on prend le carré de A SUR la
    diagonale, le bloc A×B HORS diagonale, on normalise chacun par sa norme, et
    on fait leur produit scalaire. « + ils se superposent, + on a un bon score. »

CE QUE ÇA MESURE, EN UNE PHRASE. Le carré diagonal de A est *le portrait
intérieur de A* : quel temps de A ressemble à quel autre temps de A. Le bloc
croisé A×B dit quel temps de A ressemble à quel temps de B. Si B rejoue A, le
second reproduit le premier — même diagonale pleine, mêmes motifs autour. Le
produit scalaire normalisé (un cosinus entre deux blocs vus comme des vecteurs
de L² nombres) mesure exactement cette superposition.

CE QUE ÇA NE MESURE PAS. Ce n'est pas « B ressemble à A en moyenne » : c'est
« B ressemble à A **de la même façon que A se ressemble à lui-même** ». La page
trace les deux courbes pour qu'on voie la différence. Et le critère est
ASYMÉTRIQUE — score(A→B) prend le portrait de A pour référence, score(B→A)
prend celui de B ; la matrice de l'étape 5 le montre.

Le bloc diagonal a toujours 1 sur sa diagonale (un temps se ressemble à
lui-même). Le bloc croisé n'a une diagonale pleine que si B est aligné temps à
temps sur A — c'est ce qui fait la sélectivité du critère, et aussi ce qui le
rend sensible au décalage d'un seul temps.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap        # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from ssm_zoo import SONGS, GT_LINE, gt_sections              # noqa: E402
from vote_fill import INK                                    # noqa: E402
from voix_variantes import cases, _ssm, f_hauteur            # noqa: E402
from licks import notes_de                                   # noqa: E402
import order_bundle                                          # noqa: E402

OUT = HERE / "docs" / "plots" / "match_bloc.html"
STEM = "maroon_5_this_love"
RES = 4                                   # 4 cases par mesure = le temps
PLOT_L, PLOT_R = 0.075, 0.995             # marges fixes : la tête de lecture s'y cale

CM = LinearSegmentedColormap.from_list("h", ["#faf6ec", "#9fc0d4", "#1d4d69"])
CM_P = LinearSegmentedColormap.from_list("p", ["#fffdf6", "#e8c98a", "#a2521a"])
BLEU, OR_, GRIS = "#2f7dbd", "#c07a1e", "#a89f8c"


# ── le critère ───────────────────────────────────────────────────────────────

def blocs(S, a0, L, b0):
    """(carré diagonal de A, bloc croisé A×B) — deux carrés L×L."""
    return S[a0:a0 + L, a0:a0 + L], S[a0:a0 + L, b0:b0 + L]


def score(S, a0, L, b0):
    """Le cosinus des deux blocs, plus tous les termes pour l'affichage."""
    X, Y = blocs(S, a0, L, b0)
    nx, ny = float(np.linalg.norm(X)), float(np.linalg.norm(Y))
    d = float((X * Y).sum())
    return {"X": X, "Y": Y, "nx": nx, "ny": ny, "dot": d,
            "cos": d / (nx * ny) if nx > 1e-9 and ny > 1e-9 else 0.0,
            "moy": float(Y.mean())}


def balayage(S, a0, L):
    """Le critère à chaque position de départ possible du bloc comparé."""
    K = S.shape[0]
    cos = np.zeros(K - L + 1)
    moy = np.zeros(K - L + 1)
    for b0 in range(K - L + 1):
        r = score(S, a0, L, b0)
        cos[b0], moy[b0] = r["cos"], r["moy"]
    return cos, moy


# ── la page ──────────────────────────────────────────────────────────────────

def png(fig, tight=True):
    import base64
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=112, facecolor="#fffdf6",
                **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def cadre(ax, i0, j0, L, col, lab, res=RES, lw=2.0):
    ax.add_patch(plt.Rectangle((j0 / res, i0 / res), L / res, L / res,
                               fill=False, edgecolor=col, lw=lw, zorder=5))
    ax.text(j0 / res + L / (2 * res), i0 / res - 0.8, lab, ha="center",
            va="bottom", fontsize=8.5, color=col, fontweight="bold", zorder=6)


def fig_matrice(S, n, gtb, A, cands):
    fig, ax = plt.subplots(figsize=(7.6, 7.6), facecolor="#fffdf6")
    ax.imshow(S, cmap=CM, vmin=0, vmax=1, extent=[0, n, n, 0],
              interpolation="nearest", aspect="auto")
    for g in gtb:
        ax.axvline(g, color=GT_LINE, lw=0.5, alpha=0.45)
        ax.axhline(g, color=GT_LINE, lw=0.5, alpha=0.45)
    a0, L = A
    cadre(ax, a0, a0, L, "#0d2437", "A × A")
    for (b0, col, lab) in cands:
        cadre(ax, a0, b0, L, col, lab)
    ax.set_xticks(np.arange(0, n + 1, 8)); ax.set_yticks(np.arange(0, n + 1, 8))
    ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, 8)], fontsize=7.5)
    ax.set_yticklabels([str(i + 1) for i in np.arange(0, n + 1, 8)], fontsize=7.5)
    ax.tick_params(length=2, colors="#8a8371")
    for sp in ax.spines.values():
        sp.set_color("#e0d7c2")
    return png(fig)


def fig_trois(rs, labs, cols, L):
    """Le carré diagonal, puis chaque bloc croisé, à la même échelle."""
    fig, axs = plt.subplots(1, len(rs) + 1, figsize=(3.5 * (len(rs) + 1), 3.9),
                            facecolor="#fffdf6",
                            gridspec_kw={"wspace": 0.16})
    tous = [rs[0]["X"]] + [r["Y"] for r in rs]
    noms = ["A × A  (la référence)"] + labs
    coul = ["#0d2437"] + cols
    for ax, Z, nm, c in zip(axs, tous, noms, coul):
        ax.imshow(Z, cmap=CM, vmin=0, vmax=1, interpolation="nearest",
                  aspect="auto")
        ax.set_title(nm, fontsize=9.5, color=c, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(1.6)
        ax.set_xlabel(f"‖·‖ = {np.linalg.norm(Z):.1f}", fontsize=8.5,
                      color="#6f6857")
    fig.suptitle(f"les blocs extraits — {L}×{L} temps", fontsize=9,
                 color="#8a8371", y=1.03)
    return png(fig)


def fig_produits(rs, labs, cols):
    """Le produit terme à terme des blocs NORMALISÉS : la somme EST le score."""
    fig, axs = plt.subplots(1, len(rs), figsize=(4.3 * len(rs), 4.5),
                            facecolor="#fffdf6", gridspec_kw={"wspace": 0.14})
    axs = np.atleast_1d(axs)
    P = [(r["X"] / max(r["nx"], 1e-9)) * (r["Y"] / max(r["ny"], 1e-9)) for r in rs]
    vmax = max(float(p.max()) for p in P) if P else 1.0
    for ax, p, r, nm, c in zip(axs, P, rs, labs, cols):
        ax.imshow(p, cmap=CM_P, vmin=0, vmax=vmax, interpolation="nearest",
                  aspect="auto")
        ax.set_title(f"A×A  ⊙  {nm}", fontsize=9.5, color=c, fontweight="bold",
                     pad=6)
        ax.set_xlabel(f"somme des cases  =  {r['cos']:.3f}", fontsize=10,
                      color=c, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(1.6)
    fig.suptitle("chaque case = le produit des deux cases normalisées ; leur "
                 "somme est le produit scalaire", fontsize=9, color="#8a8371",
                 y=1.02)
    return png(fig)


def fig_balayage(cos, moy, n, L, gtb, sect, a0, cands):
    x = np.arange(len(cos)) / RES
    fig, ax = plt.subplots(figsize=(12.6, 3.5), facecolor="#fffdf6")
    ax.plot(x, moy, color=GRIS, lw=1.2, label="pour comparer : la moyenne du "
            "bloc croisé (le critère naïf)")
    ax.plot(x, cos, color=BLEU, lw=1.8, label="ton critère : cos(A×A , A×B)")
    for g in gtb:
        ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.55)
    ax.axvline(a0 / RES, color="#0d2437", lw=1.4, ls=(0, (4, 2)))
    ax.annotate("A elle-même", (a0 / RES, 1.0), textcoords="offset points",
                xytext=(6, -2), ha="left", va="top", fontsize=8.5,
                color="#0d2437", fontweight="bold")
    for (b0, col, lab) in cands:
        ax.plot([b0 / RES], [cos[b0]], "o", ms=8, color=col, zorder=5)
        ax.annotate(f"{lab}\n{cos[b0]:.3f}", (b0 / RES, cos[b0]),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=8.5, color=col, fontweight="bold", zorder=6)
    tr = ax.get_xaxis_transform()
    for s in sect:
        ax.text((s["b0"] + s["b1"] + 1) / 2, -0.20, str(s["label"]),
                transform=tr, ha="center", va="top", fontsize=8.5,
                color="#8a8371", clip_on=False)
    ax.set_xlim(0, n); ax.set_ylim(0, 1.06)
    ax.set_xticks(np.arange(0, n + 1, 8))
    ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, 8)], fontsize=7.5)
    ax.set_yticks([0, 0.5, 1]); ax.tick_params(length=2, colors="#8a8371")
    ax.set_ylabel("score", fontsize=9, color="#6f6857")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5,
              frameon=False, handlelength=1.6)
    for sp in ax.spines.values():
        sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.88, bottom=0.22)
    return png(fig, tight=False)


def fig_croix(S, sect, gtb):
    """Le critère entre TOUTES ses sections — longueur commune = la plus courte."""
    N = len(sect)
    M = np.zeros((N, N))
    for i, si in enumerate(sect):
        for j, sj in enumerate(sect):
            L = RES * min(si["b1"] - si["b0"] + 1, sj["b1"] - sj["b0"] + 1)
            L = min(L, S.shape[0] - RES * si["b0"], S.shape[0] - RES * sj["b0"])
            if L <= 1:
                continue
            M[i, j] = score(S, RES * si["b0"], L, RES * sj["b0"])["cos"]
    lab = [f"{s['label']} {s['b0']+1}" for s in sect]
    fig, ax = plt.subplots(figsize=(0.72 * N + 2.6, 0.62 * N + 2.4),
                           facecolor="#fffdf6")
    ax.imshow(M, cmap=CM, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="#fffdf6" if M[i, j] > 0.62 else "#4a4438")
        # le meilleur match HORS soi-même : encadré, c'est le verdict de la ligne
        hors = [(M[i, j], j) for j in range(N) if j != i]
        if hors and max(hors)[0] > 0.05:
            _v, j = max(hors)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#b4472c", lw=2.0, zorder=5))
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(lab, fontsize=7.5, rotation=45, ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.tick_params(length=2, colors="#6f6857")
    ax.set_ylabel("la section de RÉFÉRENCE (son carré diagonal)", fontsize=8.5,
                  color="#6f6857")
    for sp in ax.spines.values():
        sp.set_color("#e0d7c2")
    return png(fig), M, lab


def main():
    stem = (sys.argv[1:] or [STEM])[0]
    titre = dict(SONGS).get(stem, stem)
    b = order_bundle.get(stem)
    n, grid = b["n"], np.asarray(b["grid"])
    bords = cases(grid, n, RES)
    S = np.nan_to_num(_ssm(f_hauteur(notes_de(stem), bords)))
    gt = gt_sections(stem)
    sect = gt["sections"] if gt else []
    gtb = [s["b0"] for s in sect[1:]]

    # A = la première section chantée ; les deux comparaisons : sa reprise, et
    # la section suivante (le contre-exemple).
    ref = next((s for s in sect if s["label"] not in ("intro", "queue", "outro")),
               sect[0])
    a0, L = RES * ref["b0"], RES * (ref["b1"] - ref["b0"] + 1)
    rep = next((s for s in sect if s["label"] == ref["label"] and s["b0"] > ref["b0"]),
               None)
    aut = next((s for s in sect if s["label"] != ref["label"] and s["b0"] > ref["b0"]
                and s["label"] not in ("queue", "outro")), None)
    cands = [(RES * rep["b0"], BLEU, f"A × {rep['label']}({rep['b0']+1})"),
             (RES * aut["b0"], OR_, f"A × {aut['label']}({aut['b0']+1})")]
    rs = [score(S, a0, L, c[0]) for c in cands]
    labs = [c[2] for c in cands]
    cols = [c[1] for c in cands]

    cos, moy = balayage(S, a0, L)
    im_m = fig_matrice(S, n, gtb, (a0, L), cands)
    im_b = fig_trois(rs, labs, cols, L)
    im_p = fig_produits(rs, labs, cols)
    im_s = fig_balayage(cos, moy, n, L, gtb, sect, a0, cands)
    im_c, Mx, lab = fig_croix(S, sect, gtb)

    def ligne(r, lb, c):
        return (f"<div class=calc><b style='color:{c}'>{lb}</b> &nbsp; "
                f"⟨A×A , bloc⟩ = <b>{r['dot']:.1f}</b> &nbsp;÷&nbsp; "
                f"(‖A×A‖ = {r['nx']:.1f} × ‖bloc‖ = {r['ny']:.1f} "
                f"= {r['nx']*r['ny']:.1f}) &nbsp;=&nbsp; "
                f"<b style='color:{c};font-size:16px'>{r['cos']:.3f}</b></div>")

    pics = [i for i in range(1, len(cos) - 1)
            if cos[i] >= cos[i - 1] and cos[i] > cos[i + 1] and cos[i] > 0.80]
    lus = ", ".join(f"mes. {p/RES+1:.2f}".rstrip("0").rstrip(".") for p in pics[:14])
    g = [round(t, 3) for t in grid]

    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Bloc diagonal contre bloc croisé</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.6 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1280px;margin:0 auto;padding:22px 14px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:1000px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 16px;margin-bottom:12px}}
h2{{font:700 17px system-ui;margin:0 0 4px;color:#8a2b2b}}
h2 span{{font:500 12px system-ui;color:#8a8371}}
p{{font-size:13.5px;color:#6f6857;margin:2px 0 10px;max-width:1000px}}
p b{{color:{INK}}}
img{{width:100%;border-radius:8px;display:block}}
img.mid{{max-width:640px;margin:0 auto}}
.calc{{font:500 13px ui-monospace,SFMono-Regular,monospace;background:#f7f3e9;
  border:1px solid #eae2cd;border-radius:8px;padding:7px 11px;margin:6px 0}}
.plot{{position:relative}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;align-items:center;gap:10px;margin:8px 0 0}}
.pp{{width:38px;height:38px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:14px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
audio{{display:none}}
</style></head><body><div class=wrap>
<h1>Ton critère, étape par étape — {titre}</h1>
<div class=lede>Matrice <b>hauteur réelle · au temps</b> (la n°6 des onze
variantes). Pour comparer la section <b>A</b> à une position <b>B</b> : on prend
le carré de A <b>sur</b> la diagonale, le bloc A×B <b>hors</b> diagonale, on
normalise chacun par sa norme et on fait leur produit scalaire.<br><br>
En une phrase : le carré diagonal est <b>le portrait intérieur de A</b> — quel
temps de A ressemble à quel autre temps de A. Le bloc croisé dit quel temps de A
ressemble à quel temps de B. <b>Si B rejoue A, le second reproduit le
premier.</b> Ce n'est donc pas « B ressemble à A en moyenne », c'est « B
ressemble à A de la même façon que A se ressemble à lui-même » — la courbe grise
de l'étape 4 est le critère naïf, pour voir la différence.<br><br>
Ici <b>A = {ref['label']} (mes. {ref['b0']+1}–{ref['b1']+1})</b>, {L} temps de
côté. Traits rouges : tes frontières.</div>

<section><h2>1 · où sont les blocs <span>la matrice entière</span></h2>
<p>Le carré <b>noir</b> sur la diagonale, c'est A contre lui-même : la référence.
Les deux carrés de couleur sont sur la <b>même bande de lignes</b> (les temps de
A) mais à d'autres colonnes — les deux endroits qu'on veut comparer à A.</p>
<img class=mid src="data:image/png;base64,{im_m}" alt="matrice"></section>

<section><h2>2 · les blocs extraits <span>{L}×{L} temps chacun</span></h2>
<p>Le carré diagonal a toujours <b>une diagonale pleine</b> (un temps se
ressemble à lui-même). Un bloc croisé n'en a une que si les deux passages sont
<b>alignés temps à temps</b> — c'est ce qui fait la sélectivité du critère.
Sous chaque bloc, sa norme.</p>
<img src="data:image/png;base64,{im_b}" alt="blocs"></section>

<section><h2>3 · le produit scalaire <span>terme à terme, puis la somme</span></h2>
<p>On divise chaque bloc par sa norme, on multiplie <b>case par case</b>, et on
somme. Les cases sombres sont celles qui rapportent : là où les deux blocs sont
forts <b>au même endroit</b>. Une case forte dans un seul des deux ne rapporte
rien.</p>
<img src="data:image/png;base64,{im_p}" alt="produits">
{ligne(rs[0], labs[0], cols[0])}{ligne(rs[1], labs[1], cols[1])}</section>

<section data-grid='{g}'><h2>4 · le même calcul à toutes les positions
<span>on fait glisser le bloc comparé, temps par temps</span></h2>
<p>Une seule référence (A), une valeur par position de départ possible. Les pics
devraient tomber sur les reprises de A. La courbe <b style="color:{GRIS}">grise</b>
est la moyenne du bloc croisé — le critère naïf : elle monte partout où le chant
est simplement dense, sans distinguer <i>quoi</i> se répète.</p>
<div class=plot><img src="data:image/png;base64,{im_s}" alt="balayage">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button>
<span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche la courbe pour te déplacer dans le morceau</span></div>
<audio preload=metadata playsinline src="../audio/{stem}.m4a"></audio>
<div class=calc>pics au-dessus de 0,80 : {lus or "aucun"}</div></section>

<section><h2>5 · entre toutes tes sections
<span>ligne = la référence, colonne = la position comparée</span></h2>
<p>Longueur commune = la plus courte des deux. <b>La matrice n'est pas
symétrique</b> : la ligne A prend le portrait de A pour référence, la ligne B
celui de B — le critère demande « est-ce que B rejoue A », pas « A et B se
ressemblent-ils ». La diagonale vaut 1 par construction. <b>Encadré en rouge :
le meilleur match de chaque ligne hors elle-même</b> — c'est le verdict.
Les zéros de l'intro ne sont pas un bug : personne n'y chante, la matrice de
voix y est vide.</p>
<img class=mid src="data:image/png;base64,{im_c}" alt="croix"></section>
</div>
<script>
document.querySelectorAll("section[data-grid]").forEach(function(sec){{
  var G = JSON.parse(sec.dataset.grid), n = G.length - 1;
  var L0 = {PLOT_L}, W = {PLOT_R} - {PLOT_L};
  var au = sec.querySelector("audio"), cur = sec.querySelector(".cur"),
      hit = sec.querySelector(".hit"), pp = sec.querySelector(".pp"),
      pos = sec.querySelector(".pos"), raf = null;
  hit.style.left = (L0*100)+"%"; hit.style.width = (W*100)+"%";
  function t2b(t){{ if(t<=G[0])return 0; if(t>=G[n])return n;
    var lo=0,hi=n; while(hi-lo>1){{var m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}}
    return lo+(t-G[lo])/(G[lo+1]-G[lo]); }}
  function b2t(f){{ var i=Math.max(0,Math.min(n-1,Math.floor(f)));
    return G[i]+(f-i)*(G[i+1]-G[i]); }}
  function fmt(s){{ return Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0"); }}
  function draw(){{ var f=t2b(au.currentTime); cur.style.display="block";
    cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
    pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime); }}
  function tick(){{ draw(); if(!au.paused) raf=requestAnimationFrame(tick); }}
  au.addEventListener("play", function(){{ pp.textContent="❚❚"; tick(); }});
  au.addEventListener("pause", function(){{ pp.textContent="▶";
    cancelAnimationFrame(raf); draw(); }});
  pp.onclick=function(){{ au.paused ? au.play().catch(function(){{}}) : au.pause(); }};
  hit.onclick=function(e){{ var r=hit.getBoundingClientRect();
    var t=b2t(n*(e.clientX-r.left)/r.width);
    var seek=function(){{ try{{ au.currentTime=t; }}catch(err){{}} draw(); }};
    if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{{once:true}});
    au.play().catch(function(){{}}); }};
}});
</script></body></html>""")
    print(f"  A = {ref['label']} mes.{ref['b0']+1}-{ref['b1']+1}  ({L} temps)")
    for r, lb in zip(rs, labs):
        print(f"    {lb:24s} cos={r['cos']:.3f}   moy={r['moy']:.3f}")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
