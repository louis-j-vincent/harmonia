"""Couper là où la suite devient imprévisible — la matrice « qui suit qui », exploitée.

    python scripts/successor_cut.py [<stem> ...]  ->  /reports/successor_cut.html

Louis, 2026-08-06 : « voyons si tu exploites la matrice qui suit qui. »

Jusqu'ici on ne s'en servait que dans un seul cas, le plus facile : quand une
mini-section est TOUJOURS suivie de la même, on colle. C'est le cas où la
matrice est certaine. Mais elle dit bien plus que ça : elle dit, à chaque
endroit, **combien de suites différentes sont possibles**. C'est ce nombre qui
porte l'information de frontière.

L'idée, en une phrase : **une frontière de section est un endroit où le nombre
de suites possibles augmente brusquement.** Au milieu d'une section, la suite
est forcée — après le premier accord du refrain vient toujours le deuxième. À la
fin de la section, la suite s'ouvre : le refrain peut être suivi d'un couplet,
d'un pont, d'une coda. Le morceau devient imprévisible exactement là où il
change de partie.

C'est une idée ancienne et solide, empruntée à la linguistique : Zellig Harris,
« From Phoneme to Morpheme » (Language, 1955), découpe les mots d'une langue
inconnue en comptant, lettre après lettre, combien de continuations différentes
existent — les frontières de mots sont les pics de ce compte. Ici les lettres
sont nos mini-sections et les mots sont les sections.

**Le contexte est ce qui rend la règle utilisable.** Compté sur une seule
mini-section, `a` est suivi de `b` puis de `c` puis de `c` sur Norah : trois
suites, on ne peut rien conclure. Compté sur les DEUX dernières, `a c` n'est
suivi que d'une seule chose. On regarde donc le contexte le plus long qui
apparaisse au moins deux fois dans le morceau, et on lui demande sa variété.

Rien d'autre n'est utilisé : ni la matrice harmonique, ni le rythme, ni un
seuil. Seulement la chaîne des mini-sections de 2/4 mesures — le socle que Louis
a validé — et le comptage des suites.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
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

MAX_CTX = 4       # jusqu'où on remonte pour trouver un contexte qui se répète
DEFAULT = ["norah_jones_don_t_know_why", "maroon_5_this_love",
           "mayer_hawthorne_the_walk", "bein_green",
           "let_it_be_remastered_2009",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]


def variety(syms, max_ctx=MAX_CTX):
    """Pour chaque position, combien de suites différentes sont possibles.

    On prend le contexte le PLUS LONG qui apparaisse au moins deux fois dans le
    morceau — un contexte unique ne peut rien apprendre, il n'a qu'une suite par
    construction et ferait croire à une certitude. Retourne (variété, longueur
    du contexte utilisé).
    """
    m = len(syms)
    nxt = [defaultdict(set) for _ in range(max_ctx + 1)]
    cnt = [Counter() for _ in range(max_ctx + 1)]
    for L in range(1, max_ctx + 1):
        for i in range(m - L):
            ctx = tuple(syms[i:i + L])
            cnt[L][ctx] += 1
            nxt[L][ctx].add(syms[i + L])
    var, used = np.zeros(m), np.zeros(m, int)
    for i in range(m):
        for L in range(min(max_ctx, i + 1), 0, -1):
            ctx = tuple(syms[i - L + 1:i + 1])
            if cnt[L][ctx] >= 2:
                var[i], used[i] = len(nxt[L][ctx]), L
                break
        else:
            var[i], used[i] = len(nxt[1][(syms[i],)]) or 1, 1
    return var, used


def cut_on_rise(syms, var):
    """La règle de Harris : on coupe après un pic de variété.

    « après ce passage, le morceau peut partir dans plus de directions
    qu'avant » — c'est là que la section s'achève. Une simple montée suffit,
    aucun seuil : var[i] > var[i-1].
    """
    m = len(syms)
    cuts = [0]
    for i in range(1, m - 1):
        if var[i] > var[i - 1] and var[i] >= var[i + 1]:
            cuts.append(i + 1)
    cuts.append(m)
    return [(a, b - 1) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


FIRST_MULT = 4     # en MESURES


def first_on_multiple(chain, spans, mult=FIRST_MULT, start=0):
    """La première section ne se termine que sur un multiple de 4 mesures.

    Louis, 2026-08-06 : « la première section ne change que sur un multiple de
    4 temps. »  Lecture retenue : 4 MESURES, pas 4 noires — tout notre découpage
    est déjà calé sur la mesure, donc « multiple de 4 temps » au sens strict
    serait déjà vrai partout et ne dirait rien. Si tu voulais dire les noires,
    dis-le et je change une ligne.

    Pourquoi la PREMIÈRE en particulier : c'est elle qui fixe la phase de tout
    le reste. Une première section de 13 mesures décale les frontières de tout
    ce qui suit, et une erreur de deux mesures au début se paie jusqu'à la fin
    du morceau. Les suivantes héritent d'un départ juste.

    On déplace donc la première coupe sur la frontière de passage la plus proche
    qui tombe sur un multiple de `mult` mesures depuis le début du cœur. Si
    aucune ne convient, on laisse la coupe où elle est plutôt que d'inventer une
    frontière au milieu d'un passage.
    """
    if len(spans) < 2:
        return spans
    a, b = spans[0]
    want = [i for i in range(len(chain) - 1)
            if (chain[i]["b1"] + 1 - start) % mult == 0]
    if not want:
        return spans
    j = min(want, key=lambda i: (abs(i - b), i))
    if j == b:
        return spans
    # la section suivante DOIT reprendre juste après, sinon on ouvre un trou
    # dans la couverture — c'est le défaut de la première version.
    out, first = [(a, j)], True
    for x, y in spans[1:]:
        if y <= j:
            continue
        out.append((j + 1 if first else x, y))
        first = False
    return out


SAME = 0.95     # deux sections sont la même musique à partir d'ici


def sections_of(chain, spans, S=None, same=SAME):
    """Les segments deviennent des sections, et les LETTRES viennent de la matrice.

    Nommer une section par sa suite exacte de symboles ne marche pas : deux
    couplets qui diffèrent d'un seul passage reçoivent alors deux lettres, et
    Norah sortait A B C D E là où il y a deux sections. On compare donc les
    sections entre elles sur la matrice harmonique, en diagonale et sur leur
    longueur commune — c'est la même mesure que partout ailleurs, et elle
    tolère qu'une reprise soit plus longue ou plus courte que l'originale.
    """
    segs = []
    for a, b in spans:
        hole = all(chain[i]["hole"] for i in range(a, b + 1))
        segs.append({"b0": chain[a]["b0"], "b1": chain[b]["b1"], "hole": hole,
                     "key": tuple(c["sym"] for c in chain[a:b + 1])})
    parent = list(range(len(segs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if segs[i]["hole"] or segs[j]["hole"]:
                continue
            if segs[i]["key"] == segs[j]["key"]:
                parent[find(i)] = find(j)
                continue
            if S is None:
                continue
            L = min(segs[i]["b1"] - segs[i]["b0"], segs[j]["b1"] - segs[j]["b0"]) + 1
            if L >= 2 and HS.diag_match(S, segs[i]["b0"], segs[j]["b0"], L) >= same:
                parent[find(i)] = find(j)
    letters, out = {}, []
    for i, sg in enumerate(segs):
        if sg["hole"]:
            out.append({"b0": sg["b0"], "b1": sg["b1"], "letter": "·"})
            continue
        r = find(i)
        if r not in letters:
            letters[r] = chr(ord("A") + len(letters))
        out.append({"b0": sg["b0"], "b1": sg["b1"], "letter": letters[r]})
    return out, letters


def succ_grid(chain):
    """La matrice « qui suit qui » telle qu'elle est vraiment utilisée."""
    syms = sorted({c["sym"] for c in chain if not c["hole"]})
    idx = {s: i for i, s in enumerate(syms)}
    cols = syms + ["·", "fin"]
    M = np.zeros((len(syms), len(cols)))
    for i, c in enumerate(chain):
        if c["hole"]:
            continue
        nx = chain[i + 1] if i + 1 < len(chain) else None
        j = (len(cols) - 1 if nx is None
             else (len(cols) - 2 if nx["hole"] else idx[nx["sym"]]))
        M[idx[c["sym"]], j] += 1
    return M, syms, cols


def strip(ax, secs, n, label, win=False):
    seen = {}
    for s in secs:
        if s["letter"] == "·":
            ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                       facecolor="#e0d8c4"))
            continue
        seen.setdefault(s["letter"], COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   color=seen[s["letter"]]))
        if s["b1"] - s["b0"] >= 1:
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                    va="center", color="#fff", fontsize=7.2, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.2)
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel(label, fontsize=6.8, rotation=0, ha="right", va="center",
                  color="#1f8a5b" if win else INK)


def figure(S, n, chain, var, used, secs, old, alt, raw):
    heights = [3.2, 1.15, 0.60, 0.55, 0.55, 0.55]
    H = sum(heights) + 1.3
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .24})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .32 / H, bottom=.58 / H)

    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)

    # la variété, en escalier sur les mesures
    ax = axs[1]
    for i, c in enumerate(chain):
        ax.add_patch(plt.Rectangle((c["b0"], 0), c["b1"] - c["b0"] + 1, var[i],
                                   facecolor="#2a6fb0", alpha=.30,
                                   edgecolor="#2a6fb0", lw=.8))
        ax.text((c["b0"] + c["b1"] + 1) / 2, var[i] + .12,
                f"{int(var[i])}", ha="center", fontsize=5.6, color="#2a6fb0")
        ax.text((c["b0"] + c["b1"] + 1) / 2, -.5,
                "·" if c["hole"] else c["sym"], ha="center", va="top",
                fontsize=5.6, color="#8a8371")
    ax.set_ylim(-1.1, max(2.2, var.max() + .8))
    ax.set_yticks(range(0, int(var.max()) + 1))
    ax.tick_params(labelsize=6)
    ax.set_ylabel("suites\npossibles", fontsize=6.8, rotation=0, ha="right",
                  va="center", color="#2a6fb0")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    strip(axs[2], secs, n, f"COUPE\naux pics\n{len({s['letter'] for s in secs})} lettres",
          win=True)
    strip(axs[3], raw, n, "sans la règle\ndu multiple de 4")
    strip(axs[4], alt, n, "« toujours\nsuivi de »")
    strip(axs[5], old, n, "règle\nactuelle")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    return fig2b64_fixed(fig)


def song(stem):
    S, n, grid = load(stem)
    c0, _ = SP.core_range(S)
    cells = HY.build_hypo(S, n, unit=2, start=c0)
    chain = HY.merge_two_to_four(HY.chain_of(cells, n))[0]
    syms = [c["sym"] if not c["hole"] else "·" for c in chain]
    var, used = variety(syms)
    spans = cut_on_rise(syms, var)
    raw, _ = sections_of(chain, spans, S)          # avant la règle du multiple
    spans = first_on_multiple(chain, spans, start=c0)
    secs, letters = sections_of(chain, spans, S)

    alt_chain, _ = HY.merge_always([dict(c) for c in chain])
    alt, _ = HY.sections_of(alt_chain)
    old_cells, _ = HS.build_cells(S, n)
    old = HS.sections_from(S, n, old_cells)
    img = figure(S, n, chain, var, used, secs, old, alt, raw)

    M, rs, cs = succ_grid(chain)
    head = "".join(f"<th>{c}</th>" for c in cs)
    rows = ""
    for i, r in enumerate(rs):
        tot = M[i].sum()
        tds = "".join(
            f"<td class='{'one' if (M[i, j] == tot and tot >= 2) else ''}'>"
            f"{int(M[i, j]) or ''}</td>" for j in range(len(cs)))
        rows += (f"<tr><td><b>{r}</b></td>{tds}"
                 f"<td>{int((M[i] > 0).sum())}</td></tr>")

    fmt = lambda x: " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in x)
    gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
    btns = "".join(f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
                   f"<small>{s['b0']+1}</small></button>" for s in secs)
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
{len(chain)} passages · {len(letters)} lettres</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique · les boutons jouent la coupe aux pics</span>
{btns}</div>
<table class=succ><tr><th>suivi de →</th>{head}<th>suites<br>possibles</th></tr>
{rows}</table>
<p class=verdict><b>coupe aux pics + 1<sup>re</sup> section sur un multiple de 4</b>
— {len(letters)} lettres : {fmt(secs)}<br>
<b>sans la règle du multiple</b> : {fmt(raw)}<br>
<b>« toujours suivi de »</b> : {fmt(alt)}<br>
<b>règle actuelle</b> : {fmt(old)}</p></section>"""


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
    out = HERE / "harmonia_min/state/reports/successor_cut.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Couper là où ça devient imprévisible</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} a{{color:#8a2b2b}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:auto;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 7px;text-align:center}}
th{{background:#f7f3e9;font-size:11px}}
td.one{{background:#e4f0e8;color:#0f5132;font-weight:700}}
.verdict{{font:500 11.5px ui-monospace,monospace;background:#f7f3e9;
  border-radius:8px;padding:9px 11px;margin:10px 0 0;line-height:1.8}}
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
<h1>Couper là où ça devient imprévisible</h1>
<div class=lede>On n'utilisait la matrice « qui suit qui » que dans son cas le
plus facile : quand une mini-section est <b>toujours</b> suivie de la même, on
colle. Mais elle dit bien plus — elle dit à chaque endroit <b>combien de suites
différentes sont possibles</b>.<br><br>
<b>Une frontière de section est un endroit où ce nombre augmente.</b> Au milieu
d'une section la suite est forcée ; à la fin elle s'ouvre — un refrain peut être
suivi d'un couplet, d'un pont ou d'une coda. Le morceau devient imprévisible
exactement là où il change de partie.<br><br>
C'est une idée ancienne et solide, empruntée à la linguistique : Zellig Harris,
<i>From Phoneme to Morpheme</i> (1955), découpe les mots d'une langue inconnue
en comptant, lettre après lettre, le nombre de continuations différentes — les
frontières de mots sont les pics de ce compte. Ici les lettres sont nos
mini-sections et les mots sont les sections.<br><br>
<b>Le contexte est ce qui rend la règle utilisable.</b> Sur une seule
mini-section, le <code>a</code> de Norah est suivi de <code>b</code> puis de
<code>c</code> puis de <code>c</code> : trois suites, on ne conclut rien. Sur les
deux dernières, <code>a c</code> n'est suivi que d'une seule chose. On prend donc
le contexte le plus long qui apparaisse au moins deux fois dans le morceau.
<br><br>
Rien d'autre n'est utilisé : ni la matrice harmonique, ni le rythme, ni aucun
seuil. Seulement la chaîne des mini-sections de 2/4 mesures — le socle — et le
comptage des suites. La courbe bleue est ce compte, passage par passage.</div>
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
