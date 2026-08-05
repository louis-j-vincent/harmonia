"""Chercher la lentille qui retrouve This Love (3) et Don't Know Why (2).

    python scripts/fusion_lens.py  ->  /reports/fusion_lens.html

Louis, 2026-08-06 : « Foote and Serrà have good instincts, use those to guide our
fusion, + the interesting insights we had. I'll let you try multiple things and
see with which lens you can recover This Love and Don't Know Why. »

CE QUE LA PAGE PRÉCÉDENTE A APPRIS, et qui oriente tout ici : la grille aveugle
de 8 mesures faisait aussi bien que Foote, que Serrà et que nous. Autrement dit
**le problème n'est pas de trouver les frontières, c'est de regrouper les
morceaux obtenus sous le bon nombre de lettres.** On balaie donc les deux
étages séparément, et on regarde lequel décide vraiment.

ÉTAGE 1 — d'où viennent les frontières :
    Foote (homogénéité), Serrà (répétition), leur FUSION (moyenne des deux
    courbes normalisées — l'idée de Louis : les deux ont de bons instincts, l'un
    voit le changement de matière, l'autre le changement de ce qui se répète),
    la variété des suites sur notre chaîne, et la grille aveugle de 8 mesures.

ÉTAGE 2 — comment deux morceaux deviennent la même lettre :
    * DIAGONALE — mesure contre mesure sur la longueur commune. Sensible au
      moindre décalage de coupe.
    * DIAGONALE GLISSÉE — la même en essayant ±2 mesures de décalage et en
      gardant le meilleur. Deux couplets coupés un peu différemment se
      retrouvent.
    * BLOC — la moyenne du bloc entier, donc l'ORDRE des accords ne compte plus,
      seulement le vocabulaire. C'est la lentille la plus tolérante : elle dit
      « ces deux passages sont faits de la même matière harmonique ».
    et un seuil balayé de 0,75 à 0,95.

Les trois vérités de Louis (Norah 2, This Love 3, The Walk 2) sont le juge.
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
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402
import spectral_sections as SP                                            # noqa: E402
import successor_cut as SC                                                # noqa: E402
import boundary_variants as BV                                            # noqa: E402

TRUTH = {"norah_jones_don_t_know_why": 2, "maroon_5_this_love": 3,
         "mayer_hawthorne_the_walk": 2}
OTHERS = ["bein_green", "let_it_be_remastered_2009",
          "bruno_mars_grenade_official_music_video",
          "maroon_5_she_will_be_loved_official_music_video"]
LENSES = ["diagonale", "diagonale glissée", "bloc"]
THRS = [0.75, 0.80, 0.85, 0.90, 0.95]
MIN_BARS = 4


def sim(S, a, b, lens, shift=2):
    """La ressemblance de deux segments, selon la lentille choisie."""
    (a0, a1), (b0, b1) = a, b
    La, Lb = a1 - a0 + 1, b1 - b0 + 1
    L = min(La, Lb)
    if L < 2:
        return 0.0
    if lens == "diagonale":
        return HS.diag_match(S, a0, b0, L)
    if lens == "diagonale glissée":
        best = 0.0
        for d in range(-shift, shift + 1):
            if b0 + d < 0 or b0 + d + L > len(S):
                continue
            best = max(best, HS.diag_match(S, a0, b0 + d, L))
        return best
    blk = S[a0:a1 + 1, b0:b1 + 1]
    return float(blk.mean()) if blk.size else 0.0


def group(S, spans, lens, thr):
    """Union-find sur les segments : même lettre dès que la lentille dit oui."""
    parent = list(range(len(spans)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if sim(S, spans[i], spans[j], lens) >= thr:
                parent[find(i)] = find(j)
    letters, out = {}, []
    for i, (a, z) in enumerate(spans):
        r = find(i)
        if r not in letters:
            letters[r] = chr(ord("A") + len(letters))
        out.append({"b0": a, "b1": z, "letter": letters[r]})
    return out


def spans_from(bounds, n, min_bars=MIN_BARS):
    b = sorted({0} | {x for x in bounds if 0 < x < n} | {n})
    sp = [(b[i], b[i + 1] - 1) for i in range(len(b) - 1) if b[i + 1] > b[i]]
    out = []
    for s in sp:
        if out and s[1] - s[0] + 1 < min_bars:
            out[-1] = (out[-1][0], s[1])
        else:
            out.append(s)
    return out


def boundaries(stem):
    """Les cinq sources de frontières, dont la fusion Foote+Serrà."""
    S, n, grid = load(stem)
    c0, _ = SP.core_range(S)
    cells = HY.build_hypo(S, n, unit=2, start=c0)
    chain = HY.merge_two_to_four(HY.chain_of(cells, n))[0]
    syms = [c["sym"] if not c["hole"] else "·" for c in chain]
    var, _ = SC.variety(syms)
    nf, ns = BV.foote(S), BV.structure_features(S)
    fus = (nf / (nf.max() or 1) + ns / (ns.max() or 1)) / 2
    return S, n, grid, {
        "Foote": BV.peaks_of(nf, n),
        "Serrà": BV.peaks_of(ns, n),
        "fusion F+S": BV.peaks_of(fus, n),
        "variété": [chain[a]["b0"] for a, _ in SC.cut_on_rise(syms, var)][1:],
        "grille 8": list(range(c0, n, 8))[1:],
    }, fus


def main():
    stems = list(TRUTH) + OTHERS
    data = {}
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            continue
        try:
            data[st] = boundaries(st)
            print(f"  ok {st}")
        except Exception as exc:
            print(f"  !! {st} — {type(exc).__name__}: {exc}")

    src_names = list(next(iter(data.values()))[3])
    judged = [s for s in TRUTH if s in data]
    results = {}
    for src in src_names:
        for lens in LENSES:
            for thr in THRS:
                counts = {}
                for st in judged:
                    S, n, _, B, _ = data[st]
                    secs = group(S, spans_from(B[src], n), lens, thr)
                    counts[st] = len({x["letter"] for x in secs})
                hits = sum(1 for st in judged if counts[st] == TRUTH[st])
                err = sum(abs(counts[st] - TRUTH[st]) for st in judged)
                results[(src, lens, thr)] = (hits, -err, counts)
    best = max(results, key=lambda k: results[k][:2])
    print(f"\n  meilleure lentille : {best} -> {results[best][2]} "
          f"({results[best][0]}/{len(judged)})")

    head = "".join(f"<th>{st.replace('_',' ').title()[:14]}<br>"
                   f"<span class=gt>{TRUTH[st]}</span></th>" for st in judged)
    rows = ""
    for cfg in sorted(results, key=lambda k: (-results[k][0], -results[k][1])):
        h, e, c = results[cfg]
        tds = "".join(f"<td class='{'ok' if c[st] == TRUTH[st] else 'no'}'>{c[st]}</td>"
                      for st in judged)
        rows += (f"<tr class='{'win' if cfg == best else ''}'><td>{cfg[0]}</td>"
                 f"<td>{cfg[1]}</td><td>{cfg[2]:.2f}</td>{tds}"
                 f"<td>{h}/{len(judged)}</td></tr>")

    body = ""
    for st, (S, n, grid, B, fus) in data.items():
        src, lens, thr = best
        rows_fig = []
        for name in src_names:
            secs = group(S, spans_from(B[name], n), lens, thr)
            rows_fig.append((name, secs, name == src))
        heights = [3.0, 0.75] + [0.55] * len(rows_fig)
        H = sum(heights) + 1.3
        fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                                gridspec_kw={"height_ratios": heights, "hspace": .24})
        fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .32 / H, bottom=.58 / H)
        axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                      vmin=float(np.percentile(S, 5)),
                      vmax=float(np.percentile(S, 99)),
                      aspect="auto", interpolation="nearest")
        axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)
        y = fus / (fus.max() or 1)
        axs[1].fill_between(np.arange(n) + .5, y, color="#7c3aed", alpha=.28)
        axs[1].plot(np.arange(n) + .5, y, color="#7c3aed", lw=1.1)
        axs[1].set_ylim(0, 1.1); axs[1].set_yticks([])
        axs[1].set_ylabel("fusion\nFoote+Serrà", fontsize=6.6, rotation=0,
                          ha="right", va="center", color="#7c3aed")
        for sp in ("top", "right", "left"):
            axs[1].spines[sp].set_visible(False)
        gt = TRUTH.get(st)
        for i, (name, secs, win) in enumerate(rows_fig):
            BV.strip(axs[2 + i], secs, n, name + (" ★" if win else ""), gt)
        axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
        axs[-1].set_xlabel("mesure", fontsize=8)
        img = fig2b64_fixed(fig)
        secs = group(S, spans_from(B[src], n), lens, thr)
        fmt = " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in secs)
        gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
        btns = "".join(f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>"
                       f"{s['letter']}<small>{s['b0']+1}</small></button>"
                       for s in secs)
        gtxt = f"<span class=gt>vérité : {gt} sections</span>" if gt else ""
        body += f"""<section data-grid='{gridjs}' data-audio="/audio/{st}.m4a">
<h2>{st.replace('_',' ').title()}<span class=sub>{n} mesures</span></h2>{gtxt}
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>les boutons jouent la lentille retenue</span>{btns}</div>
<p class=verdict>{fmt}</p></section>"""

    out = HERE / "harmonia_min/state/reports/fusion_lens.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>La lentille</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} ul{{margin:8px 0 0;padding-left:20px}} li{{margin-bottom:5px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
.gt{{display:inline-block;font:700 12px system-ui;color:#0f5132;background:#e4f0e8;
  border-radius:6px;padding:2px 8px;margin-bottom:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12px;width:100%;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 7px;text-align:left}}
th{{background:#f7f3e9;font-size:10.5px}}
td.ok{{background:#e4f0e8;color:#0f5132;font-weight:700}}
td.no{{color:#a89f8c}} tr.win td{{background:#e4f0e8;font-weight:700}}
.verdict{{font:500 11.5px ui-monospace,monospace;background:#f7f3e9;
  border-radius:8px;padding:9px 11px;margin:8px 0 0}}
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
<h1>La lentille</h1>
<div class=lede>La page précédente a appris une chose qui oriente tout :
<b>la grille aveugle de 8 mesures faisait aussi bien que Foote, que Serrà et que
nous</b>. Le problème n'est donc pas de trouver les frontières — c'est de
regrouper les morceaux obtenus sous le bon nombre de lettres. On balaie ici les
deux étages séparément pour voir lequel décide vraiment.<br><br>
<b>D'où viennent les frontières</b> : Foote, Serrà, leur <b>fusion</b> (la
moyenne des deux courbes normalisées — l'un voit le changement de matière,
l'autre le changement de ce qui se répète), la variété des suites sur notre
chaîne, et la grille aveugle.<br><br>
<b>Comment deux morceaux deviennent la même lettre</b> — c'est ça, la lentille :
<ul>
<li><b>diagonale</b> — mesure contre mesure. Sensible au moindre décalage de
coupe.</li>
<li><b>diagonale glissée</b> — la même en essayant ±2 mesures et en gardant le
meilleur : deux couplets coupés différemment se retrouvent.</li>
<li><b>bloc</b> — la moyenne du bloc entier, donc <b>l'ordre des accords ne
compte plus</b>, seulement la matière. La plus tolérante.</li>
</ul>
avec un seuil balayé de 0,75 à 0,95.<br><br>
Juge : tes trois vérités — Norah 2, This Love 3, The Walk 2.</div>
<section><h2>Le balayage</h2>
<table><tr><th>frontières</th><th>lentille</th><th>seuil</th>{head}
<th>vérités</th></tr>{rows}</table></section>
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
