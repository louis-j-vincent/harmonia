"""La matrice binaire des mini-sections — même symbole ou pas, rien d'autre.

    python scripts/binary_ssm.py [<stem> ...]  ->  /reports/binary_ssm.html

Louis, 2026-08-05 : « tu peux faire la matrice SSM binaire qui vient des
différentes mini-sections qu'on a identifiées ? »

Trois matrices par morceau, sur le même axe :

  1. l'HARMONIQUE, continue, celle d'où tout vient ;
  2. la BINAIRE PAR MESURE : 1 si les deux mesures portent la même mini-section,
     0 sinon. C'est la même information, mais après décision — tout le gris a
     disparu ;
  3. la BINAIRE PAR PASSAGE : la chaîne contre elle-même, un carré par couple de
     mini-sections. C'est la plus petite des trois, et la seule où une
     répétition de SUCCESSION se voit directement : deux passages successifs qui
     se répètent plus loin forment un segment de diagonale. Ces segments sont
     entourés.

L'intérêt de la troisième : la fusion par paires ne voit que les voisins
immédiats, et la carte « qui suit qui » ne voit que les couples. Une diagonale
de longueur 3 dans cette matrice est une succession de trois mini-sections qui
revient — exactement ce que ni l'une ni l'autre n'attrape.
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
from pattern_lanes import load, fig2b64_fixed, COLS, INK        # noqa: E402
import hypo_sizes as HY                                          # noqa: E402

ACC = "#b3261e"
DEFAULT = ["bein_green", "norah_jones_don_t_know_why", "maroon_5_this_love",
           "let_it_be_remastered_2009", "mayer_hawthorne_the_walk",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]


def diag_runs(syms, holes, min_len=2):
    """Les segments de diagonale : une succession de ≥ min_len passages qui
    revient telle quelle ailleurs. Retourne [(i, j, longueur)] avec i < j."""
    out, m = [], len(syms)
    for lag in range(1, m):
        i, run = 0, 0
        while i + lag < m:
            same = (syms[i] == syms[i + lag] and not holes[i] and not holes[i + lag])
            run = run + 1 if same else 0
            if not same and run == 0 and i + lag + 1 >= m:
                break
            if run >= min_len and (i + lag + 1 >= m
                                   or i + 1 >= m - lag
                                   or syms[i + 1] != syms[i + 1 + lag]
                                   or holes[i + 1]):
                out.append((i - run + 1, i - run + 1 + lag, run))
            i += 1
    return out


def figure(S, n, chain):
    syms = [c["sym"] for c in chain]
    holes = [c["hole"] for c in chain]
    m = len(chain)

    # binaire par mesure
    owner = np.full(n, -1)
    names = {}
    for c in chain:
        if c["hole"]:
            continue
        names.setdefault(c["sym"], len(names))
        owner[c["b0"]:c["b1"] + 1] = names[c["sym"]]
    B = ((owner[:, None] == owner[None, :]) & (owner[:, None] >= 0)).astype(float)

    # binaire par passage
    C = np.array([[1.0 if (syms[i] == syms[j] and not holes[i] and not holes[j])
                   else 0.0 for j in range(m)] for i in range(m)])

    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.5),
                            gridspec_kw={"width_ratios": [1, 1, 1.12]})
    fig.subplots_adjust(left=.05, right=.99, top=.88, bottom=.13, wspace=.22)

    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_title("harmonique (continue)", fontsize=8.5, color="#2a6fb0", loc="left")
    axs[0].set_xlabel("mesure", fontsize=7.5)

    axs[1].imshow(B, origin="lower", extent=(0, n, 0, n), cmap="Greys",
                  vmin=0, vmax=1.4, aspect="auto", interpolation="nearest")
    axs[1].set_title("binaire, par mesure", fontsize=8.5, color=INK, loc="left")
    axs[1].set_xlabel("mesure", fontsize=7.5)

    ax = axs[2]
    ax.imshow(C, origin="lower", extent=(0, m, 0, m), cmap="Greys",
              vmin=0, vmax=1.4, aspect="auto", interpolation="nearest")
    runs = diag_runs(syms, holes)
    for i, j, L in runs:
        if L < 2:
            continue
        ax.plot([i, i + L], [j, j + L], color=ACC, lw=2.2, solid_capstyle="butt")
        ax.plot([j, j + L], [i, i + L], color=ACC, lw=2.2, solid_capstyle="butt")
    ax.set_xticks(np.arange(m) + .5)
    ax.set_xticklabels(["·" if h else s for s, h in zip(syms, holes)],
                       fontsize=5.6, rotation=90)
    ax.set_yticks(np.arange(m) + .5)
    ax.set_yticklabels(["·" if h else s for s, h in zip(syms, holes)], fontsize=5.6)
    ax.tick_params(length=0)
    ax.set_title(f"binaire, par passage — {len([r for r in runs if r[2] >= 2])} "
                 "successions répétées (en rouge)", fontsize=8.5, color=ACC, loc="left")
    for a in axs[:2]:
        a.tick_params(labelsize=6.6)
    axs[0].set_ylabel("mesure", fontsize=7.5)
    return fig2b64_fixed(fig), runs, syms, holes


def song(stem):
    S, n, grid = load(stem)
    cells = HY.build_hypo(S, n)
    ch0 = HY.chain_of(cells, n)
    ch4, _ = HY.merge_two_to_four(ch0)
    img, runs, syms, holes = figure(S, n, ch4)
    seen, rows = set(), ""
    for i, j, L in sorted(runs, key=lambda r: (-r[2], r[0])):
        pat = tuple(syms[i:i + L])
        if pat in seen:
            continue
        seen.add(pat)
        hits = [k for k in range(len(syms) - L + 1)
                if tuple(syms[k:k + L]) == pat and not any(holes[k:k + L])]
        rows += (f"<tr><td><b>{' '.join(pat)}</b></td><td>{L}</td>"
                 f"<td>{len(hits)}</td>"
                 f"<td>mes. {' · '.join(str(ch4[k]['b0']+1) for k in hits)}</td></tr>")
    chain = " ".join("·" if c["hole"] else c["sym"] for c in ch4)
    gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
    btns = "".join(
        f"<button class=blk style='border-color:{COLS[(ord(c['sym'][0])-ord('a'))%len(COLS)] if not c['hole'] else '#8a8371'}' "
        f"data-p='[{c['b0']},{c['b1']+1}]'>{'·' if c['hole'] else c['sym']}"
        f"<small>{c['b0']+1}</small></button>" for c in ch4)
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
{len(ch4)} passages</span></h2>
<img src="data:image/png;base64,{img}">
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span></div>
<div class=lane><span class=lab>écouter</span>{btns}</div>
<p class=verdict><b>la chaîne</b> : <span class=chain>{chain}</span></p>
<table><tr><th>succession</th><th>longueur</th><th>reprises</th>
<th>commence en</th></tr>{rows or '<tr><td colspan=4>aucune</td></tr>'}</table>
</section>"""


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
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    out = HERE / "harmonia_min/state/reports/binary_ssm.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>La matrice binaire des mini-sections</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
.chain{{font:600 13px ui-monospace,monospace;color:#8a2b2b;letter-spacing:.06em}}
.verdict{{font-size:12.5px;background:#f7f3e9;border-radius:8px;padding:9px 11px;margin:8px 0}}
.bar{{display:flex;align-items:center;gap:10px;margin:0 0 8px}}
.pp{{width:38px;height:38px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12.5px ui-monospace,monospace}}
.lane{{display:flex;flex-wrap:wrap;gap:3px;align-items:center;margin:0 0 6px}}
.lab{{font:600 11px system-ui;color:#8a8371;width:66px;flex:none}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 6px;cursor:pointer;font:700 11.5px ui-monospace,monospace}}
button.blk small{{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#8a2b2b !important}}
</style></head><body><div class=wrap>
<h1>La matrice binaire des mini-sections</h1>
<div class=lede>Trois matrices par morceau. À gauche <b>l'harmonique</b>, continue,
celle d'où tout vient. Au milieu <b>la binaire par mesure</b> : noir si les deux
mesures portent la même mini-section, blanc sinon — la même information, mais
après décision, tout le gris a disparu. À droite <b>la binaire par passage</b> :
la chaîne contre elle-même, un carré par couple de mini-sections.<br><br>
C'est la troisième qui sert. Une <b>répétition de succession</b> y est un segment
de diagonale — deux, trois, quatre passages qui reviennent dans le même ordre.
Ils sont tracés en rouge, et listés sous la figure. Ni la fusion par paires ni la
carte « qui suit qui » ne les voient : elles ne regardent que les voisins
immédiats.</div>
{body}</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
let stopAt=null,onBtn=null,live=null;
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
function clr(){{ if(onBtn){{onBtn.classList.remove("on");onBtn=null;}} }}
function tick(){{ if(live) live.pos.textContent=fmt(au.currentTime);
  if(stopAt!=null&&au.currentTime>=stopAt){{au.pause();stopAt=null;clr();}}
  if(!au.paused) requestAnimationFrame(tick); }}
au.addEventListener("play",tick);
function go(sec,t0,t1,btn){{
  live=sec._p; clr(); stopAt=t1;
  if(btn){{onBtn=btn;btn.classList.add("on");}}
  if(au.getAttribute("src")!==sec.dataset.audio){{
    au.setAttribute("src",sec.dataset.audio);au.load();}}
  const seek=()=>{{try{{au.currentTime=t0;}}catch(e){{}}}};
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{{once:true}});
  au.play().catch(()=>clr());
}}
document.querySelectorAll("section[data-grid]").forEach(sec=>{{
  const G=JSON.parse(sec.dataset.grid), n=G.length-1;
  sec._p={{G:G,pos:sec.querySelector(".pos")}};
  sec.querySelector(".pp").onclick=()=>{{ if(au.paused) go(sec,G[0],null,null);
                                          else au.pause(); }};
  sec.querySelectorAll("[data-p]").forEach(b=>{{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(sec,G[d[0]],G[Math.min(n,d[1])],b); }});
}});
</script></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
