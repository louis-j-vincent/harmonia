"""Les diagonales de la voix : ce qu'elles disent, et ce qu'elles ne disent pas.

    .venv/bin/python scripts/zones_page.py   ->  docs/plots/zones_voix.html
    open docs/plots/zones_voix.html               (local, sans serveur)

Louis, 2026-08-13 : « on voit une succession diagonale de matchs, ça correspond
clairement à une reprise avec décalage — lorsqu'on trouve une diagonale il faut
absolument l'exploiter. »

Il a raison sur l'objet : les diagonales sont là, et elles sont franches. La page
les montre, morceau par morceau, avec ce qu'un détecteur en tire et ce que ça
vaut contre ses frontières à lui.

CE QUE LA MESURE DIT : le critère est un excellent JUGE et un mauvais CHERCHEUR.
Si on lui donne ses sections, il donne la bonne lettre à 89 % d'entre elles ;
lâché pour trouver les frontières lui-même il plafonne à 42 % de justesse.

POURQUOI — et ce n'est PAS parce qu'il serait flou. La deuxième figure le montre :
décaler UNE des deux copies de deux mesures fait tomber le score à ~45 % du vrai.
Le critère est net.

La vraie raison est ailleurs, et elle est structurelle. **Décaler LES DEUX copies
du même nombre de mesures ne change rien du tout** : le long d'une diagonale,
toutes les fenêtres marchent aussi bien. Une diagonale dit donc parfaitement
QUEL décalage et SUR QUELLE ÉTENDUE ça se rejoue, et ne dit RIEN de la phase —
où la section commence à l'intérieur de la zone qui se répète. C'est
l'erreur-type #4 du projet, retrouvée telle quelle : `score_periods` détectait la
longueur de période et jamais sa phase. Sans recalage, Bein' Green, Every Breath
You Take et The Walk sortent tous DEUX MESURES trop tôt, systématiquement.

D'où le partage des rôles : **la voix donne le décalage et l'étendue, les pics de
changement donnent la phase.**
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

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections       # noqa: E402
from hard_prior_sections import colourmap                    # noqa: E402
from vote_fill import fill, INK                              # noqa: E402
import zones_voix as AN                                          # noqa: E402
import bench_sections as BS                                  # noqa: E402
import mots4                                                 # noqa: E402
import order_bundle                                          # noqa: E402

OUT = HERE / "docs" / "plots" / "zones_voix.html"
PLOT_L, PLOT_R = 0.10, 0.995
CM = LinearSegmentedColormap.from_list("h", ["#faf6ec", "#9fc0d4", "#1d4d69"])


def png(fig, tight=True):
    import base64
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=104, facecolor="#fffdf6",
                **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    try:
        from PIL import Image
        buf.seek(0)
        im = Image.open(buf).convert("RGB").quantize(colors=128, method=2)
        small = BytesIO(); im.save(small, format="PNG", optimize=True)
        if small.tell() < buf.getbuffer().nbytes:
            buf = small
    except Exception:
        pass
    return base64.b64encode(buf.getvalue()).decode()


def fig_carte(stem, a, gtb, n):
    """La carte des matchs : ses DIAGONALES sont les reprises."""
    S, _n = AN.voix(stem)
    R = AN.carte(S, n)
    fig, ax = plt.subplots(figsize=(5.4, 5.4), facecolor="#fffdf6")
    ax.imshow(R, cmap=CM, vmin=0, vmax=1, extent=[0, n, n, 0],
              interpolation="nearest", aspect="auto")
    for g in gtb:
        ax.axvline(g, color=GT_LINE, lw=0.55, alpha=0.5)
        ax.axhline(g, color=GT_LINE, lw=0.55, alpha=0.5)
    step = 8 if n <= 90 else 16
    ax.set_xticks(np.arange(0, n + 1, step)); ax.set_yticks(np.arange(0, n + 1, step))
    ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7)
    ax.set_yticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7)
    ax.tick_params(length=2, colors="#8a8371")
    for sp in ax.spines.values():
        sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.99, bottom=0.06)
    return png(fig)


def fig_lisse(stem, n, gt):
    """LE CRITÈRE EST NET : le score quand on décale UNE des deux copies.

    Pour chacune de ses sections qui a un jumeau, le score du jumeau décalé de
    −6 à +6 mesures. Le pic à 0 est franc — à ±2 mesures il ne reste que ~45 %.
    C'est la figure qui interdit d'expliquer l'échec du détecteur par un critère
    flou : il ne l'est pas. Ce qui manque est la PHASE, et elle est invisible
    ici par construction, puisque décaler les DEUX copies ensemble ne bouge rien.
    """
    S, _n = AN.voix(stem)
    dec = np.arange(-6, 7, 2)
    lignes = []
    for i, si in enumerate(gt):
        jum = [sj for j, sj in enumerate(gt)
               if j != i and sj["label"] == si["label"] and sj["b0"] != si["b0"]]
        if not jum:
            continue
        sj = jum[0]
        v = [AN.score_span(S, si["b0"], si["b1"] + 1, sj["b0"] + d, sj["b1"] + 1 + d)
             for d in dec]
        if np.isfinite(v).all() and max(v) > 0:
            lignes.append((si["label"], np.asarray(v)))
    fig, ax = plt.subplots(figsize=(5.4, 2.5), facecolor="#fffdf6")
    for lab, v in lignes:
        ax.plot(dec, v / max(v.max(), 1e-9), lw=1.3, alpha=0.75, marker="o", ms=3)
    if lignes:
        M = np.mean([v / max(v.max(), 1e-9) for _l, v in lignes], axis=0)
        ax.plot(dec, M, lw=2.6, color="#0d2437", label="moyenne", zorder=5)
        for d, y in zip(dec, M):
            if d in (-4, -2, 2, 4):
                ax.annotate(f"{y:.0%}", (d, y), textcoords="offset points",
                            xytext=(0, -13), ha="center", fontsize=7.5,
                            color="#0d2437")
    ax.axvline(0, color=GT_LINE, lw=0.8, alpha=0.6)
    ax.set_xticks(dec); ax.set_xlabel("décalage du jumeau, en mesures", fontsize=8.5,
                                      color="#6f6857")
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "", "le vrai"], fontsize=7.5)
    ax.tick_params(length=2, colors="#8a8371", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.24)
    return png(fig), (float(np.mean([v[dec == 2] / np.clip(v[dec == 0], 1e-9, None)
                                     for _l, v in lignes])) if lignes else 0.0)


def fig_bandes(a, nous, gt, n, grid):
    cm = colourmap()
    bandes = [("zones de reprise\n(voix + pics)",
               [{"b0": s["b0"], "b1": s["b1"], "label": s["label"],
                 "nu": s["label"]} for s in a["sections"]]),
              ("ce qu'on écrit", nous), ("toi", gt)]
    gtb = [s["b0"] for s in gt[1:]]
    fig, axs = plt.subplots(3, 1, figsize=(12.2, 1.95), facecolor="#fffdf6",
                            gridspec_kw={"hspace": 0.14})
    for k, (ax, (lab, ss)) in enumerate(zip(axs, bandes)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle(
                (s["b0"], 0.08), w, 0.84, facecolor=cm(s.get("nu", s["label"])),
                edgecolor="#fffdf6", lw=1.0))
            if w >= max(5, n * 0.05) and str(s["label"]):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8, color=INK)
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.6)
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8.5,
                      color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 2:
            step = 8 if n <= 90 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=7.5, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.97, bottom=0.24)
    return png(fig, tight=False)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or list(SONGS)
    body, tot = [], [0, 0, 0]
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        b = order_bundle.get(stem)
        n, grid = b["n"], b["grid"]
        gt = gt_sections(stem)["sections"]
        gtb = [s["b0"] for s in gt[1:]]
        a = AN.zones(stem)
        r = BS.une(stem)
        c = {s["b0"] for s in a["sections"]} | {s["b1"] + 1 for s in a["sections"]}
        c = {x for x in c if 0 < x < n}
        g = set(gtb)
        tot[0] += len(c & g); tot[1] += len(c); tot[2] += len(g)
        im_c = fig_carte(stem, a, gtb, n)
        im_l, _pente = fig_lisse(stem, n, gt)
        im_b = fig_bandes(a, r["nous"], gt, n, grid)
        gl = [round(t, 3) for t in grid]
        body.append(
            f'<section data-grid=\'{gl}\'><h2>{title} '
            f'<span class=sub>{n} mesures · les ancres trouvent '
            f'{len(c & g)} de tes {len(g)} frontières, et en inventent '
            f'{len(c) - len(c & g)}</span></h2>'
            f'<div class=duo><img class=sq src="data:image/png;base64,{im_c}">'
            f'<img class=sq src="data:image/png;base64,{im_l}"></div>'
            f'<div class=plot><img src="data:image/png;base64,{im_b}">'
            '<div class=cur></div><div class=hit></div></div>'
            '<div class=bar><button class=pp>▶</button>'
            '<span class=pos>mes. 1 · 0:00</span>'
            '<span class=hint>touche les bandes pour te déplacer</span></div>'
            f'<audio preload=metadata playsinline src="../audio/{stem}.m4a"></audio>'
            '</section>')
        print(f"  ok {title}  {len(c & g)}/{len(c)}")
    prec = tot[0] / max(1, tot[1]); rapp = tot[0] / max(1, tot[2])
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les diagonales de la voix</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.6 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1280px;margin:0 auto;padding:22px 14px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:1000px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:12px 14px;margin-bottom:12px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.duo{{display:flex;gap:12px;align-items:flex-start;margin-bottom:8px;flex-wrap:wrap}}
.duo .sq{{flex:1 1 320px;min-width:280px;max-width:520px;border-radius:8px}}
img{{width:100%;display:block;border-radius:8px}}
.plot{{position:relative}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;align-items:center;gap:10px;margin:6px 0 0}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
audio{{display:none}}
</style></head><body><div class=wrap>
<h1>Les diagonales de la voix</h1>
<div class=lede>Tu as raison sur l'objet : <b>les diagonales sont là, et elles
sont franches</b>. À gauche de chaque morceau, la carte des matchs — chaque case
(p, q) porte ton critère entre les deux endroits ; <b>une diagonale = le morceau
se rejoue tel quel un certain nombre de mesures plus loin</b>. Traits rouges :
tes frontières.<br><br>
<b>Ce que la mesure dit.</b> Le critère est un excellent JUGE et un mauvais
CHERCHEUR. Si on lui donne tes sections, il donne la bonne lettre à <b>89 %</b>
d'entre elles. Lâché pour trouver les frontières tout seul, il en pose
{tot[0]}/{tot[1]} de justes — <b>{prec:.0%}</b> — et retrouve {rapp:.0%} des
tiennes.<br><br>
<b>Pourquoi — et ce n'est pas parce qu'il serait flou.</b> La figure de droite
montre l'inverse : décaler <b>une</b> des deux copies de deux mesures fait tomber
le score à ~45 % du vrai. Le critère est net.<br><br>
La vraie raison est structurelle : <b>décaler LES DEUX copies du même nombre de
mesures ne change rien du tout</b>. Le long d'une diagonale, toutes les fenêtres
marchent aussi bien. Une diagonale dit donc parfaitement <b>quel décalage</b> et
<b>sur quelle étendue</b> ça se rejoue — et rien de <b>la phase</b>, c'est-à-dire
où la section commence à l'intérieur de la zone qui se répète. Sans recalage,
Bein' Green, Every Breath You Take et The Walk sortent tous <b>deux mesures trop
tôt</b>, systématiquement.<br><br>
<b>D'où le partage des rôles</b> : la voix donne le décalage et l'étendue, les
pics de changement donnent la phase. C'est exactement l'erreur-type n°4 du
projet, retrouvée telle quelle — <code>score_periods</code> détectait la longueur
de période et jamais sa phase.<br><br>
Rien n'est branché en production : les trois branchements essayés
(la voix groupe les fenêtres / la voix renomme les sections posées, par seuil
puis par plus-proche) font tous baisser le résultat, les chiffres sont dans
<code>known_issues.md</code>.</div>
{''.join(body)}</div>
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
  au.addEventListener("play", function(){{
    document.querySelectorAll("audio").forEach(function(x){{ if(x!==au) x.pause(); }});
    pp.textContent="❚❚"; tick(); }});
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
    print(f"\nancres : précision {tot[0]}/{tot[1]} = {prec:.0%} · "
          f"rappel {tot[0]}/{tot[2]} = {rapp:.0%}")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
