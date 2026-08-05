"""Cinq façons de trouver les frontières, dont les deux standards du domaine.

    python scripts/boundary_variants.py [<stem> ...]
      -> /reports/boundary_variants.html

Louis, 2026-08-06 : « bon la règle n'est pas bonne, regarde en ligne ce qui se
fait et tu essayes différentes variantes stp. »

Ce qui se fait, d'après la synthèse de référence du domaine (Nieto, Mysore,
Wang, Smith, Schlüter, Grill, Pons, « Audio-Based Music Structure Analysis:
Current Trends, Open Challenges, and Applications », TISMIR 2020,
https://transactions.ismir.net/articles/10.5334/tismir.54) : les méthodes de
frontière se rangent en deux familles, et les deux algorithmes cités comme
références sont implémentés ici.

  1. HOMOGÉNÉITÉ — « ici la musique arrête de se ressembler à elle-même ».
     C'est le NOYAU EN DAMIER de Foote (2000) : un damier glissé le long de la
     diagonale de la matrice ; il vaut beaucoup quand un bloc homogène se
     termine et qu'un autre commence. Le classique absolu du domaine.

  2. RÉPÉTITION — « ici, ce qui se répétait cesse de se répéter ». Ce sont les
     STRUCTURAL FEATURES de Serrà, Müller, Grosche et Arcos (2014), donnés par
     la synthèse comme l'état de l'art non supervisé sur les frontières. On
     range la matrice en LAGS (chaque ligne = un décalage), on lisse, et la
     frontière est là où cette colonne de lags change brusquement : à cet
     endroit, l'ensemble des endroits où l'on se répète bascule.

  3–5. Les nôtres, pour comparer sur le même axe : la variété des suites
     (Harris 1955, sur la chaîne de mini-sections), la même avec la règle de
     Louis (première section sur un multiple de 4 mesures), et la règle
     purement métrique (frontière tous les 8 mesures) comme témoin.

Toutes produisent des frontières, puis les lettres sont attribuées de la même
façon pour toutes — deux sections partagent une lettre si leur diagonale
harmonique atteint 0,95 sur leur longueur commune. Seules les FRONTIÈRES sont
comparées ; l'étiquetage est neutralisé.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.ndimage import gaussian_filter1d   # noqa: E402
from scipy.signal import find_peaks           # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402
import spectral_sections as SP                                            # noqa: E402
import successor_cut as SC                                                # noqa: E402

TRUTH = {"norah_jones_don_t_know_why": 2, "maroon_5_this_love": 3,
         "mayer_hawthorne_the_walk": 2}
DEFAULT = ["norah_jones_don_t_know_why", "maroon_5_this_love",
           "mayer_hawthorne_the_walk", "bein_green",
           "let_it_be_remastered_2009",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]


# ── 1. Foote : le noyau en damier ──────────────────────────────────────────
def foote(S, M=8):
    """Novelty de Foote (2000). M = demi-taille du damier, en mesures."""
    g = np.exp(-.5 * ((np.arange(-M, M) + .5) / (M / 2.0)) ** 2)
    K = np.outer(np.sign(np.arange(-M, M) + .5), np.sign(np.arange(-M, M) + .5))
    K = K * np.outer(g, g)
    n = len(S)
    c = np.zeros(n)
    for i in range(n):
        a, b = i - M, i + M
        if a < 0 or b > n:
            continue
        c[i] = float((S[a:b, a:b] * K).sum())
    return np.clip(c, 0, None)


# ── 2. Serrà : les structural features ─────────────────────────────────────
def structure_features(S, kappa=2, sigma=2.0):
    """Serrà, Müller, Grosche & Arcos (2014).

    On empile les diagonales de la matrice : la ligne `l` de la matrice de lags
    dit « à quel point la mesure t ressemble à la mesure t − l ». Une colonne de
    cette matrice est donc le PROFIL DE RÉPÉTITION de l'instant t : à quelles
    distances il se répète. On lisse le long du temps, puis la nouveauté est la
    distance entre le profil un peu avant et le profil un peu après.

    L'idée qui la distingue de Foote : Foote demande « la musique change-t-elle
    ici ? », Serrà demande « ce qui se répétait change-t-il ici ? ». Un refrain
    qui revient à l'identique ne déclenche pas Foote au milieu, mais bascule
    bien le profil de répétition à ses bords.
    """
    n = len(S)
    K = max(4, n // 2)
    L = np.zeros((K, n))
    for l in range(K):
        for t in range(n):
            L[l, t] = S[t, (t - l) % n]
    L = gaussian_filter1d(L, sigma, axis=1, mode="nearest")
    c = np.zeros(n)
    for t in range(n):
        a, b = max(0, t - kappa), min(n - 1, t + kappa)
        c[t] = float(np.linalg.norm(L[:, b] - L[:, a]))
    return np.clip(c, 0, None)


def peaks_of(c, n, min_gap=4, frac=0.25):
    """Les pics : au moins `min_gap` mesures d'écart, au-dessus d'une fraction du max."""
    if c.max() <= 0:
        return []
    idx, _ = find_peaks(c, distance=min_gap, height=frac * c.max())
    return [int(i) for i in idx if 0 < i < n]


# ── les frontières deviennent des sections, étiquetage commun ──────────────
def sections_from_bounds(S, n, bounds, same=0.95, min_bars=4):
    b = sorted({0} | {x for x in bounds if 0 < x < n} | {n})
    spans = [(b[i], b[i + 1] - 1) for i in range(len(b) - 1)]
    spans = [s for s in spans if s[1] >= s[0]]
    merged = []                              # une section trop courte est absorbée
    for s in spans:
        if merged and s[1] - s[0] + 1 < min_bars:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    parent = list(range(len(merged)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            L = min(merged[i][1] - merged[i][0], merged[j][1] - merged[j][0]) + 1
            if L >= 2 and HS.diag_match(S, merged[i][0], merged[j][0], L) >= same:
                parent[find(i)] = find(j)
    letters, out = {}, []
    for i, (a, z) in enumerate(merged):
        r = find(i)
        if r not in letters:
            letters[r] = chr(ord("A") + len(letters))
        out.append({"b0": a, "b1": z, "letter": letters[r]})
    return out


def strip(ax, secs, n, label, gt=None):
    seen = {}
    for s in secs:
        seen.setdefault(s["letter"], COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   color=seen[s["letter"]]))
        if s["b1"] - s["b0"] >= 1:
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                    va="center", color="#fff", fontsize=7.2, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.2)
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    k = len({s["letter"] for s in secs})
    col = INK if gt is None else ("#1f8a5b" if k == gt else "#8a2b2b")
    ax.set_ylabel(f"{label}\n{k} lettres", fontsize=6.8, rotation=0,
                  ha="right", va="center", color=col)


def song(stem):
    S, n, grid = load(stem)
    c0, _ = SP.core_range(S)
    cells = HY.build_hypo(S, n, unit=2, start=c0)
    chain = HY.merge_two_to_four(HY.chain_of(cells, n))[0]
    syms = [c["sym"] if not c["hole"] else "·" for c in chain]
    var, _ = SC.variety(syms)

    nf, ns = foote(S), structure_features(S)
    bf, bs = peaks_of(nf, n), peaks_of(ns, n)
    spans = SC.cut_on_rise(syms, var)
    bv = [chain[a]["b0"] for a, _ in spans][1:]
    spans4 = SC.first_on_multiple(chain, spans, start=c0)
    bv4 = [chain[a]["b0"] for a, _ in spans4][1:]
    b8 = list(range(c0, n, 8))[1:]

    variants = [
        ("Foote — damier", bf, nf),
        ("Serrà — répétition", bs, ns),
        ("variété des suites", bv, None),
        ("variété + 1re × 4", bv4, None),
        ("tous les 8 mesures", b8, None),
    ]
    gt = TRUTH.get(stem)
    outs = [(name, sections_from_bounds(S, n, b)) for name, b, _ in variants]

    heights = [3.0, 0.85, 0.85] + [0.55] * len(outs) + [0.55]
    H = sum(heights) + 1.3
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .24})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .32 / H, bottom=.58 / H)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)
    for ax, c, b, name, col in ((axs[1], nf, bf, "Foote", "#8a2b2b"),
                                (axs[2], ns, bs, "Serrà", "#2a6fb0")):
        y = c / (c.max() or 1)
        ax.fill_between(np.arange(n) + .5, y, color=col, alpha=.28)
        ax.plot(np.arange(n) + .5, y, color=col, lw=1.1)
        for x in b:
            ax.axvline(x, color=col, lw=1.0, ls=(0, (2, 2)))
        ax.set_ylim(0, 1.1); ax.set_yticks([])
        ax.set_ylabel(name, fontsize=6.8, rotation=0, ha="right", va="center", color=col)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for i, (name, secs) in enumerate(outs):
        strip(axs[3 + i], secs, n, name, gt)
    old_cells, _ = HS.build_cells(S, n)
    old = HS.sections_from(S, n, old_cells)
    strip(axs[-1], old, n, "règle actuelle")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fmt = lambda x: " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in x)
    rows = "".join(
        f"<tr class='{'ok' if gt and len({s['letter'] for s in secs}) == gt else ''}'>"
        f"<td><b>{name}</b></td><td>{len({s['letter'] for s in secs})}</td>"
        f"<td>{len(secs)}</td><td class=f>{fmt(secs)}</td></tr>"
        for name, secs in outs)
    rows += (f"<tr class=old><td>règle actuelle</td>"
             f"<td>{len({s['letter'] for s in old})}</td><td>{len(old)}</td>"
             f"<td class=f>{fmt(old)}</td></tr>")
    gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
    btns = "".join(f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
                   f"<small>{s['b0']+1}</small></button>" for s in outs[1][1])
    gtxt = f"<span class=gt>vérité connue : {gt} sections</span>" if gt else ""
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures</span></h2>{gtxt}
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique · les boutons jouent la variante Serrà</span>
{btns}</div>
<table><tr><th>variante</th><th>lettres</th><th>sections</th><th>découpage</th></tr>
{rows}</table></section>"""


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
    out = HERE / "harmonia_min/state/reports/boundary_variants.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Cinq façons de trouver les frontières</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} a{{color:#8a2b2b}} ol{{margin:8px 0 0;padding-left:20px}}
ol li{{margin-bottom:6px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
.gt{{display:inline-block;font:700 12px system-ui;color:#0f5132;background:#e4f0e8;
  border-radius:6px;padding:2px 8px;margin-bottom:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}}
td.f{{font:500 11px ui-monospace,monospace}}
tr.ok td{{background:#e4f0e8}} tr.old td{{color:#8a8371}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:0 0 6px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:88px}}
.hint{{font:500 11px system-ui;color:#a89f8c;margin-right:6px}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 7px;cursor:pointer;font:700 11.5px ui-monospace,monospace}}
button.blk small{{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#8a2b2b !important}}
</style></head><body><div class=wrap>
<h1>Cinq façons de trouver les frontières</h1>
<div class=lede>D'après la synthèse de référence du domaine
(<a href="https://transactions.ismir.net/articles/10.5334/tismir.54">Nieto et
al., TISMIR 2020</a>), les méthodes de frontière se rangent en deux familles, et
je n'en utilisais aucune des deux.
<ol>
<li><b>Homogénéité — le damier de Foote (2000).</b> Un damier glissé le long de
la diagonale de la matrice ; il vaut beaucoup quand un bloc homogène se termine
et qu'un autre commence. « Ici la musique arrête de se ressembler à
elle-même. »</li>
<li><b>Répétition — les <i>structural features</i> de Serrà et al. (2014)</b>,
données par cette synthèse comme l'état de l'art non supervisé. On range la
matrice en <b>décalages</b> : une colonne dit, pour un instant donné, <b>à
quelles distances il se répète</b>. La frontière est là où ce profil bascule.
« Ici, ce qui se répétait cesse de se répéter. » La différence avec Foote
compte : un refrain qui revient à l'identique ne réveille pas Foote en son
milieu, mais fait bien basculer le profil de répétition à ses bords.</li>
<li><b>La variété des suites</b> — la nôtre d'hier, sur la chaîne de
mini-sections.</li>
<li>La même <b>avec ta règle</b> (première section sur un multiple de 4).</li>
<li><b>Tous les 8 mesures</b>, comme témoin : si une méthode ne bat pas la
grille aveugle, elle ne sert à rien.</li>
</ol>
<br>Les deux courbes de nouveauté sont dessinées sous la matrice, avec leurs
pics. <b>Seules les frontières changent d'une variante à l'autre</b> :
l'étiquetage est le même pour toutes (deux sections partagent une lettre si leur
diagonale harmonique atteint 0,95), pour qu'on compare des découpages et non des
nommages. Les morceaux dont tu as donné la vérité l'affichent en vert, et la
ligne qui la retrouve est surlignée.</div>
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
