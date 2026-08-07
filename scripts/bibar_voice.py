"""Le découpage en double-mesures, et ce que la voix en pense. Démo, aucun algo.

    python scripts/bibar_voice.py [<stem> ...]  ->  /reports/bibar_voice.html

Louis, 2026-08-07 :

  « Fais-moi une démo — aucun algo à implémenter pour l'instant, je veux juste
    le visuel — du découpage en double barres des chansons comme on fait
    jusqu'à maintenant, une ligne par différente section de 2 barres, et avec la
    voix tu vas venir détecter chaque section et glisser pour voir lesquelles
    sont le + en accord ou pas. Tu me représentes ça via un striage différent sur
    chaque ligne de bi-barres : hachurés pour un premier groupe de cohérence à la
    voix, plein pour le second, dashed hachuré pour le troisième… »

CE QUE MONTRE LA PAGE. Deux lectures du même morceau, superposées sur les mêmes
mesures.

  * LA COULEUR, c'est l'HARMONIE. Une ligne par double-mesure distincte : on
    prend toutes les fenêtres de deux mesures sur la grille, on regroupe celles
    qui se ressemblent harmoniquement, et chaque groupe devient une ligne avec
    tous ses emplacements. C'est le découpage habituel, inchangé.

  * LE REMPLISSAGE, c'est la VOIX. À l'intérieur d'une ligne, on glisse le bloc
    chanté de chaque emplacement contre les autres : ceux qui se répondent
    partagent un remplissage. Plein pour le groupe le plus nombreux, puis
    hachuré, puis croisé, puis pointillé.

CE QUE ÇA REND LISIBLE. Une ligne entièrement pleine, c'est une double-mesure où
harmonie et voix disent la même chose partout — solide. Une ligne bariolée, c'est
un endroit où les mêmes accords portent des chants différents : soit deux
sections distinctes qui partagent leur harmonie, soit un couplet dont les paroles
changent. C'est exactement l'arbitrage que fait l'oreille, et que ni l'une ni
l'autre des deux matrices ne peut faire seule.

Rien n'est décidé ici, rien n'alimente le pipeline. C'est un visuel pour
regarder.
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
from pattern_lanes import (load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R,  # noqa: E402
                           edge, mid, THR_COLS)
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import channels as CN                                                     # noqa: E402

CELL = 2           # la double-mesure
MAX_LANES = 12     # au-delà on ne lit plus rien ; ce qui est coupé est annoncé
DEFAULT = B8.DEFAULT

# plein d'abord — le groupe le plus nombreux est celui qu'on lit sans effort
FILLS = [None, "///", "xxx", "...", "\\\\\\", "+++", "ooo", "||"]
FILL_NAMES = ["plein", "hachuré", "croisé", "pointillé", "hachuré inverse",
              "croisillon", "pastillé", "barré"]


def link(items, same):
    """Regroupement proche-en-proche. Renvoie la liste des groupes d'indices."""
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if same(items[i], items[j]):
                parent[find(i)] = find(j)
    out = {}
    for i in range(len(items)):
        out.setdefault(find(i), []).append(i)
    return sorted(out.values(), key=lambda g: (-len(g), g[0]))


def bibar_lanes(S, M, n, start, thr_h, thr_m, cell=CELL):
    """Les double-mesures distinctes, AVEC confiscation et sans solitaires.

    Louis, 2026-08-07 : « on confisque comme dans mini_sections, sinon il y en a
    trop et on s'y perd. Les solitaires n'ont pas droit à une ligne. »

    La première version partitionnait tout le morceau : chaque fenêtre de deux
    mesures atterrissait dans un groupe, y compris celles qui ne ressemblent à
    rien, et rien n'était retiré du jeu. D'où trois à quatre fois plus de lignes
    qu'avant, illisible.

    Ici on reprend la mécanique de `mini_sections`, verrouillée à deux mesures :
    on avance sur la grille, on cherche les reprises de la fenêtre courante, et
    **si elle en a on prend toutes ses places et on les retire du jeu**. Une
    fenêtre sans reprise n'est pas jetée pour autant — elle n'ouvre juste pas de
    ligne, et elle reste disponible pour être la reprise d'une fenêtre plus
    loin.

    Le remplissage, lui, ne change pas : dans chaque ligne on regroupe les
    emplacements dont les chants se répondent.
    """
    claimed = np.zeros(n, bool)
    lanes, cursor = [], start % cell
    while cursor + cell <= n:
        if claimed[cursor:cursor + cell].any():
            cursor += cell
            continue
        occ = [c for c in range(start % cell, n - cell + 1, cell)
               if abs(c - cursor) >= cell and not claimed[c:c + cell].any()
               and HS.diag_match(S, cursor, c, cell) >= thr_h]
        if not occ:                       # solitaire : pas de ligne
            cursor += cell
            continue
        places = [cursor] + occ
        for c in places:
            claimed[c:c + cell] = True
        vgroups = link(places,
                       lambda a, b: float(np.mean([M[a + k, b + k]
                                                   for k in range(cell)])) >= thr_m)
        lanes.append({"places": places,
                      "vgroups": [[places[i] for i in vg] for vg in vgroups]})
        cursor += cell
    return sorted(lanes, key=lambda L: (-len(L["places"]), L["places"][0]))


MAX_PROFILES = 8


def profile(S, n, b0, start, cell=CELL):
    """La double-mesure `b0` glissée sur tout le morceau : ses pics sont ses reprises.

    Louis, 2026-08-07 : « je veux voir le plot des seuils, pour que je puisse
    voir les pics tout ça ». Les bandes ne montrent que le résultat, pas la
    matière : ici on voit la courbe, les pics, et où chaque seuil vient couper.
    """
    xs = [c for c in range(start % cell, n - cell + 1, cell)]
    return xs, [HS.diag_match(S, b0, c, cell) for c in xs]


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(grid, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)

    vh = CN.Voice("harmonie", S, n, CELL, CN.FLOOR_H)
    vm = CN.Voice("chant", M, n, CELL, CN.FLOOR_M, mute=mute)
    # LES SEUILS À ARBITRER. Louis, 2026-08-07 : « je n'ai pas arbitré sur le
    # seuil strict ni rien, montre-moi des démos du seuil pour que j'arbitre ».
    # Trois réglages du MÊME critère, côte à côte, sur les mêmes mesures : celui
    # d'aujourd'hui, et deux crans du seuil propre au morceau.
    vals = CN.pair_values(S, n, CELL)
    setups = [("0.90 — absolu, celui d'aujourd'hui", 0.90)]
    for q in (0.90, 0.97):
        setups.append((f"décile {int(q*100)} du morceau", float(np.quantile(vals, q))))
    runs = []
    for name, thr in setups:
        L = bibar_lanes(S, M, n, vstart, thr, vm.thr)
        runs.append({"name": name, "thr": thr, "cut": max(0, len(L) - MAX_LANES),
                     "lanes": L[:MAX_LANES]})

    # les profils : une courbe par ancre retenue au décile 90, avec les trois
    # seuils posés dessus — c'est là qu'on VOIT ce que chaque seuil coupe
    anchors = [L["places"][0] for L in runs[1]["lanes"]][:MAX_PROFILES]
    profs = [{"b0": b0, "xy": profile(S, n, b0, vstart)} for b0 in anchors]

    heights = [2.0, 2.0, 0.26] + [0.70] * max(1, len(profs))
    for r in runs:
        heights += [0.26] + [0.40] * max(1, len(r["lanes"]))
    Hh = sum(heights) + 1.6
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .30})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.62 / Hh)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.4); axs[0].tick_params(labelsize=6.2)
    axs[0].set_title("HARMONIE — elle donne la COULEUR des lignes", fontsize=8,
                     color="#2a6fb0", loc="left", pad=3)
    axs[1].imshow(M, origin="lower", extent=(0, n, 0, n), cmap="Purples",
                  vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axs[1].set_ylabel("mesure", fontsize=7.4); axs[1].tick_params(labelsize=6.2)
    axs[1].set_title("CHANT — il donne le REMPLISSAGE", fontsize=8,
                     color="#7c3aed", loc="left", pad=3)
    axs[2].axis("off")
    axs[2].set_title("LES PICS ET LES SEUILS — chaque double-mesure glissée sur "
                     "le morceau. gris = 0.90 absolu · vert = décile 90 · "
                     "rouge = décile 97", fontsize=8.4, color="#8a2b2b",
                     loc="left", pad=1)
    k = 3
    if not profs:
        axs[k].axis("off"); k += 1
    for i, pr in enumerate(profs):
        ax, col = axs[k], COLS[i % len(COLS)]
        k += 1
        xs, ys = pr["xy"]
        ax.fill_between([mid(x) for x in xs], ys, color=col, alpha=.22, lw=0)
        ax.plot([mid(x) for x in xs], ys, color=col, lw=1.2)
        for (nm, thr), c2 in zip([(r["name"], r["thr"]) for r in runs],
                                 (THR_COLS["absolu"], THR_COLS["q90"],
                                  THR_COLS["q97"])):
            ax.axhline(thr, color=c2, lw=1.0, ls=(0, (3, 2)))
        ax.axvline(edge(pr["b0"]), color="#111", lw=1.4)
        ax.set_xlim(0, n); ax.set_ylim(0, 1.06); ax.set_yticks([0, .9])
        ax.tick_params(labelsize=5.4)
        ax.set_ylabel(f"mes. {pr['b0']+1}", fontsize=6, rotation=0, ha="right",
                      va="center", color=col)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for r in runs:
        axs[k].axis("off")
        axs[k].set_title(
            f"{r['name']}  —  seuil {r['thr']:.3f}  —  {len(r['lanes'])} ligne(s)"
            + (f", {r['cut']} de plus non montrée(s)" if r["cut"] else ""),
            fontsize=8.4, color="#8a2b2b", loc="left", pad=1)
        k += 1
        if not r["lanes"]:
            axs[k].axis("off"); k += 1
        for i, L in enumerate(r["lanes"]):
            ax, col = axs[k], COLS[i % len(COLS)]
            k += 1
            for gi, grp in enumerate(L["vgroups"]):
                h = FILLS[gi % len(FILLS)]
                for c in grp:
                    ax.add_patch(plt.Rectangle(
                        (edge(c), .14), CELL, .72,
                        facecolor=col if h is None else "none",
                        edgecolor=INK if h is None else col,
                        hatch=h, lw=.8 if h is None else 1.0,
                        alpha=1.0 if h is None else .95))
            ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
            ax.set_ylabel(f"×{len(L['places'])} · {len(L['vgroups'])} voix",
                          fontsize=6, rotation=0, ha="right", va="center", color=col)
            for sp in ax.spines.values():
                sp.set_color("#e5dcc6")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.2)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    rows = ""
    for r in runs:
        rows += (f"<tr class=hd><td colspan=4><b>{r['name']}</b> — seuil "
                 f"{r['thr']:.3f} — {len(r['lanes'])} ligne(s)</td></tr>")
        for i, L in enumerate(r["lanes"]):
            parts = " · ".join(
                f"<i>{FILL_NAMES[gi % len(FILL_NAMES)]}</i> : "
                + ", ".join(str(c + 1) for c in grp)
                for gi, grp in enumerate(L["vgroups"]))
            rows += (f"<tr><td>ligne {i+1}</td><td>{len(L['places'])}</td>"
                     f"<td>{len(L['vgroups'])}</td><td>{parts}</td></tr>")
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{L['places'][0]},{L['places'][0]+CELL}]'>"
        f"L{i+1}<small>{L['places'][0]+1}</small></button>"
        for i, L in enumerate(runs[1]["lanes"]))
    note = " · ".join(f"{r['name'].split(' —')[0]} {len(r['lanes'])} lignes" for r in runs)
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
départ mes. {vstart+1} · {note} · seuil chant {vm.thr:.2f}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>clique dans le graphe pour écouter</span>{btns}</div>
<table><tr><th>ligne</th><th>emplacements</th><th>groupes de voix</th>
<th>qui va avec qui (mesure)</th></tr>{rows}</table></section>"""


def main():
    stems = sys.argv[1:] or DEFAULT
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            body += song(st)
            print(f"  ok {st}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    out = HERE / "harmonia_min/state/reports/bibar_voice.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Double-mesures : la couleur c'est l'harmonie, le remplissage c'est la voix</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}} td i{{color:#8a2b2b;font-style:normal;font-weight:700}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:0 0 6px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:88px}}
.hint{{font:500 11px system-ui;color:#a89f8c;margin-right:6px}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 7px;cursor:pointer;font:700 11.5px ui-monospace,monospace}}
button.blk small{{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#8a2b2b !important}}
</style></head><body><div class=wrap>
<h1>Double-mesures : la couleur c'est l'harmonie, le remplissage c'est la voix</h1>
<div class=lede>Deux lectures du même morceau, sur les mêmes mesures.<br><br>
<b>La couleur, c'est l'harmonie.</b> Une ligne par double-mesure distincte : on
prend toutes les fenêtres de deux mesures sur la grille, on regroupe celles qui
se ressemblent, et chaque groupe devient une ligne avec tous ses emplacements.
C'est le découpage habituel, inchangé.<br><br>
<b>Le remplissage, c'est la voix.</b> À l'intérieur d'une ligne, on glisse le
bloc chanté de chaque emplacement contre les autres. Ceux qui se répondent
partagent un remplissage : <b>plein</b> pour le groupe le plus nombreux, puis
<b>hachuré</b>, puis <b>croisé</b>, puis <b>pointillé</b>.<br><br>
Une ligne <b>entièrement pleine</b> : harmonie et voix disent la même chose
partout, c'est solide. Une ligne <b>bariolée</b> : les mêmes accords portent des
chants différents — soit deux sections distinctes qui partagent leur harmonie,
soit un couplet dont les paroles changent. C'est l'arbitrage que fait l'oreille,
et qu'aucune des deux matrices ne peut faire seule.<br><br>
Rien n'est décidé ici et rien n'alimente le pipeline : c'est un visuel pour
regarder.</div>
{body}</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
const L0=%L0%, W=%W%;
let stopAt=null,onBtn=null,live=null,raf=null;
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
function clr(){{ if(onBtn){{onBtn.classList.remove("on");onBtn=null;}} }}
function draw(){{
  if(!live) return;
  const G=live.G, n=G.length-1;
  let t=au.currentTime, f;
  if(t<=G[0]) f=0; else if(t>=G[n]) f=n; else {{
    let lo=0,hi=n; while(hi-lo>1){{const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}}
    f=lo+(t-G[lo])/(G[lo+1]-G[lo]); }}
  live.cur.style.display="block";
  live.cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
  live.pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime);
}}
function tick(){{ draw();
  if(stopAt!=null&&au.currentTime>=stopAt){{au.pause();stopAt=null;clr();}}
  if(!au.paused) raf=requestAnimationFrame(tick); }}
au.addEventListener("play",()=>{{ if(live) live.pp.textContent="❚❚"; tick(); }});
au.addEventListener("pause",()=>{{ if(live) live.pp.textContent="▶";
  cancelAnimationFrame(raf); draw(); }});
function go(sec,t0,t1,btn){{
  if(live && live.sec!==sec){{ live.pp.textContent="▶"; live.cur.style.display="none"; }}
  live=sec._p; clr(); stopAt=t1;
  if(btn){{onBtn=btn;btn.classList.add("on");}}
  if(au.getAttribute("src")!==sec.dataset.audio){{
    au.setAttribute("src",sec.dataset.audio);au.load();}}
  const seek=()=>{{try{{au.currentTime=t0;}}catch(e){{}} draw();}};
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{{once:true}});
  au.play().catch(()=>clr());
}}
document.querySelectorAll("section[data-grid]").forEach(sec=>{{
  const G=JSON.parse(sec.dataset.grid), n=G.length-1;
  const hit=sec.querySelector(".hit");
  sec._p={{G:G,pos:sec.querySelector(".pos"),cur:sec.querySelector(".cur"),
          pp:sec.querySelector(".pp"),sec:sec}};
  hit.style.left=(L0*100)+"%"; hit.style.width=(W*100)+"%";
  hit.onclick=e=>{{ const r=hit.getBoundingClientRect();
    const f=n*(e.clientX-r.left)/r.width;
    const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    go(sec, G[i]+(f-i)*(G[i+1]-G[i]), null, null); }};
  sec.querySelector(".pp").onclick=()=>{{
    if(au.paused||live!==sec._p) go(sec,G[0],null,null); else au.pause(); }};
  sec.querySelectorAll("[data-p]").forEach(b=>{{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(sec,G[d[0]],G[Math.min(n,d[1])],b); }});
}});
</script></body></html>""".replace("%L0%", str(PLOT_L)).replace("%W%", str(round(PLOT_R - PLOT_L, 6))))
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
