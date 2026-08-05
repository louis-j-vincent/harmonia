"""Quatre façons de fabriquer les sections finales, à partir des mêmes outils.

    python scripts/final_sections.py [<stem> ...]  ->  /reports/final_sections.html

Louis, 2026-08-05 : « grâce à ces outils, propose-moi des façons de créer les
sections finales, illustre-moi comme tu as fait avant avec les couleurs par
section finale, + les plots avec une barre par section de départ. »

Les outils déjà là : les mini-sections de 2 mesures (`hypo_sizes.build_hypo`),
leur regroupement en 4 (`merge_two_to_four`), la carte « qui suit qui »
(`succ_map`) et les successions répétées lues sur la matrice binaire par passage
(`binary_ssm.diag_runs`). Les quatre méthodes ci-dessous ne changent rien à ces
outils — elles ne diffèrent que par la question posée à la chaîne.

  A — LA PLUS LONGUE D'ABORD. On prend la plus longue succession qui revient,
      on en fait une section partout où elle apparaît, on l'enlève, on
      recommence sur ce qui reste. Une section est donc une phrase longue et
      répétée ; les restes finissent en sections d'un seul passage.

  B — CELLE QUI COUVRE LE PLUS D'ABORD. Même chose, mais on choisit par
      longueur × nombre de reprises. Une succession courte qui revient six fois
      passe avant une longue qui revient deux fois.

  C — QUI SUIT QUI. On ne coupe que là où c'est imprévisible : deux passages
      restent ensemble tant que le premier est TOUJOURS suivi du second. Une
      frontière est un endroit où la suite bifurque.

  D — LE MOINS DE SECTIONS, CHACUNE DEVANT REVENIR. On cherche le découpage en
      le moins de morceaux possible, sous la contrainte que chaque morceau soit
      une succession qui revient au moins deux fois (un passage isolé reste
      permis, sinon rien n'est faisable). Programmation dynamique sur la chaîne,
      donc pas de choix glouton du tout.

Aucune n'est déclarée gagnante ici. Les quatre sont dessinées l'une sous l'autre
sur le même axe, en couleurs par section finale, avec au-dessus une barre par
mini-section de départ.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402

DEFAULT = ["bein_green", "norah_jones_don_t_know_why", "maroon_5_this_love",
           "let_it_be_remastered_2009", "mayer_hawthorne_the_walk",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]


def occurrences(syms, holes, pat):
    """Où la succession `pat` revient, sans chevauchement ni trou."""
    L, out, j = len(pat), [], 0
    while j <= len(syms) - L:
        if list(syms[j:j + L]) == list(pat) and not any(holes[j:j + L]):
            out.append(j)
            j += L
        else:
            j += 1
    return out


def all_patterns(syms, holes, max_len=None):
    """Toutes les successions qui reviennent ≥ 2 fois -> {pat: [départs]}."""
    m = len(syms)
    max_len = max_len or m // 2
    out = {}
    for L in range(1, max_len + 1):
        for i in range(m - L + 1):
            if any(holes[i:i + L]):
                continue
            pat = tuple(syms[i:i + L])
            if pat in out:
                continue
            hits = occurrences(syms, holes, pat)
            if len(hits) >= 2:
                out[pat] = hits
    return out


def cut_greedy(chain, key):
    """A et B : on prend la meilleure succession selon `key`, on la fige, on
    recommence sur ce qui reste. `key(pat, hits) -> score`."""
    syms = [c["sym"] for c in chain]
    holes = [c["hole"] for c in chain]
    taken = [False] * len(chain)
    out = []
    while True:
        free_syms = [s if not taken[i] else None for i, s in enumerate(syms)]
        free_holes = [h or taken[i] for i, h in enumerate(holes)]
        pats = all_patterns([s or "\x00" for s in free_syms], free_holes)
        if not pats:
            break
        pat, hits = max(pats.items(), key=lambda kv: key(kv[0], kv[1]))
        if len(pat) < 1:
            break
        for h in hits:
            out.append((h, h + len(pat) - 1, "".join(pat)))
            for k in range(h, h + len(pat)):
                taken[k] = True
        if all(taken):
            break
    for i, c in enumerate(chain):          # ce qui reste : un passage = une section
        if not taken[i]:
            out.append((i, i, "·" if c["hole"] else c["sym"]))
    return sorted(out)


def cut_follows(chain):
    """C : on ne coupe qu'aux bifurcations de la carte « qui suit qui »."""
    merged, _ = HY.merge_always([dict(c) for c in chain])
    out, i = [], 0
    for seg in merged:                     # remonter aux indices de la chaîne
        L = 1
        j = i
        while j + 1 < len(chain) and chain[j]["b1"] < seg["b1"]:
            j += 1
            L += 1
        out.append((i, j, "·" if seg["hole"] else seg["sym"]))
        i = j + 1
    return out


def cut_fewest(chain):
    """D : le moins de morceaux, chacun devant revenir. Programmation dynamique."""
    syms = [c["sym"] for c in chain]
    holes = [c["hole"] for c in chain]
    m = len(chain)
    pats = all_patterns(syms, holes)
    starts = {}
    for pat, hits in pats.items():
        for h in hits:
            starts.setdefault(h, []).append(pat)
    INF = float("inf")
    dp = [INF] * (m + 1)
    back = [None] * (m + 1)
    dp[0] = 0
    for i in range(m):
        if dp[i] == INF:
            continue
        for pat in starts.get(i, []):
            j = i + len(pat)
            if dp[i] + 1 < dp[j]:
                dp[j] = dp[i] + 1
                back[j] = (i, "".join(pat))
        j = i + 1                          # toujours permis : un passage seul
        if dp[i] + 1 < dp[j]:
            dp[j] = dp[i] + 1
            back[j] = (i, "·" if holes[i] else syms[i])
    out, j = [], m
    while j > 0:
        i, name = back[j]
        out.append((i, j - 1, name))
        j = i
    return sorted(out)


def to_sections(chain, cuts):
    """[(i, j, nom)] sur la chaîne -> sections en mesures + lettres."""
    letters, out = {}, []
    for i, j, name in cuts:
        if name == "·":
            out.append({"b0": chain[i]["b0"], "b1": chain[j]["b1"], "letter": "·"})
            continue
        if name not in letters:
            letters[name] = chr(ord("A") + len(letters))
        out.append({"b0": chain[i]["b0"], "b1": chain[j]["b1"],
                    "letter": letters[name], "name": name})
    return out, letters


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
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=6.8, rotation=0, ha="right", va="center",
                  color="#1f8a5b" if win else INK)


def figure(S, n, chain, results, old):
    syms = sorted({c["sym"] for c in chain if not c["hole"]})
    k = len(syms)
    heights = [3.6] + [0.40] * k + [0.30] + [0.58] * (len(results) + 1)
    H = 3.6 + .40 * k + 0.30 + .58 * (len(results) + 1) + 1.1
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .20})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .28 / H, bottom=.50 / H)

    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5)
    axs[0].tick_params(labelsize=6.4)

    # une barre par mini-section de départ
    for i, sym in enumerate(syms):
        ax, col = axs[1 + i], COLS[i % len(COLS)]
        for c in chain:
            if c["sym"] == sym and not c["hole"]:
                ax.add_patch(plt.Rectangle((c["b0"], .16), c["b1"] - c["b0"] + 1,
                                           .68, facecolor=col, edgecolor=INK, lw=.7))
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(sym, fontsize=7, rotation=0, ha="right", va="center", color=col)
        for sp in ax.spines.values():
            sp.set_color("#e5dcc6")
    axs[1 + k].axis("off")

    for j, (name, secs, letters) in enumerate(results):
        strip(axs[2 + k + j], secs, n, f"{name}\n{len(letters)} lettres")
    strip(axs[-1], old, n, "règle actuelle")
    axs[-1].set_xticks(range(0, n + 1, 4))
    axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    return fig2b64_fixed(fig)


def song(stem):
    S, n, grid = load(stem)
    cells = HY.build_hypo(S, n)
    ch = HY.merge_two_to_four(HY.chain_of(cells, n))[0]
    old_cells, _ = HS.build_cells(S, n)
    old = HS.sections_from(S, n, old_cells)

    methods = [
        ("A — la plus longue", cut_greedy(ch, lambda p, h: (len(p), len(h)))),
        ("B — la plus couvrante", cut_greedy(ch, lambda p, h: (len(p) * len(h), len(p)))),
        ("C — qui suit qui", cut_follows(ch)),
        ("D — le moins de sections", cut_fewest(ch)),
    ]
    results = []
    for name, cuts in methods:
        secs, letters = to_sections(ch, cuts)
        results.append((name, secs, letters))
    img = figure(S, n, ch, results, old)

    fmt = lambda secs: " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in secs)
    rows = "".join(
        f"<tr><td><b>{name}</b></td><td>{len(letters)}</td><td>{len(secs)}</td>"
        f"<td class=f>{fmt(secs)}</td></tr>" for name, secs, letters in results)
    rows += (f"<tr class=old><td>règle actuelle</td>"
             f"<td>{len({s['letter'] for s in old})}</td><td>{len(old)}</td>"
             f"<td class=f>{fmt(old)}</td></tr>")
    gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
        f"<small>{s['b0']+1}</small></button>" for s in results[3][1])
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
{len(ch)} passages</span></h2>
<img src="data:image/png;base64,{img}">
<div class=bar><button class=pp>▶</button><span class=pos>0:00</span>
<span class=lab>écouter le découpage D</span>{btns}</div>
<table><tr><th>méthode</th><th>lettres</th><th>sections</th>
<th>découpage</th></tr>{rows}</table></section>"""


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
            import traceback; traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    out = HERE / "harmonia_min/state/reports/final_sections.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Quatre façons de faire les sections</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} ol{{margin:8px 0 0;padding-left:20px}} ol li{{margin-bottom:6px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}}
td.f{{font:500 11px ui-monospace,monospace}}
tr.old td{{color:#8a8371}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:0 0 10px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:38px}}
.lab{{font:600 11px system-ui;color:#8a8371;margin-right:4px}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 7px;cursor:pointer;font:700 11.5px ui-monospace,monospace}}
button.blk small{{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#8a2b2b !important}}
</style></head><body><div class=wrap>
<h1>Quatre façons de faire les sections</h1>
<div class=lede>Mêmes outils pour les quatre : les mini-sections de 2 mesures,
regroupées en 4 quand l'une suit toujours l'autre. Seule change la question
posée à la chaîne.
<ol>
<li><b>A — la plus longue d'abord.</b> On prend la plus longue succession qui
revient, elle devient une section partout où elle apparaît, on l'enlève, on
recommence.</li>
<li><b>B — la plus couvrante d'abord.</b> Pareil, mais on choisit par longueur ×
nombre de reprises : une courte qui revient six fois passe avant une longue qui
revient deux fois.</li>
<li><b>C — qui suit qui.</b> On ne coupe qu'aux bifurcations : deux passages
restent ensemble tant que le premier est toujours suivi du second.</li>
<li><b>D — le moins de sections, chacune devant revenir.</b> Le découpage en le
moins de morceaux possible, chaque morceau devant être une succession qui
revient au moins deux fois. Calculé exactement, sans choix glouton.</li>
</ol>
<br>En haut de chaque morceau, <b>une barre par mini-section de départ</b> : où
chacune joue. En dessous, les quatre découpages en couleurs, puis la règle
actuelle. Les boutons jouent le découpage D.</div>
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
