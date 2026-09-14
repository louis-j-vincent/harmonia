"""Où A et A′ divergent — et pourquoi ça arbitre le découpage.

    python scripts/align_diff.py [<stem> ...]  ->  /reports/align_diff.html

Louis, 2026-08-06 : « Aussi autre chose pour arbitrer : les différences
d'alignement entre les A et A′, B et B′. »

L'IDÉE, ET POURQUOI ELLE TRANCHE. Deux passages qui portent la même lettre ne
sont jamais identiques. Ce qui compte n'est pas COMBIEN ils diffèrent, c'est
**où**.

  * Une section et sa variante s'accordent au début et divergent **à la fin** :
    c'est une vraie reprise avec une autre sortie — turnaround, montée, dernière
    mesure changée. L'alignement est bon.
  * Elles divergent **au milieu, ou dès le début** : ce ne sont pas les mêmes
    huit mesures posées au même endroit. L'alignement est faux, et le pavage
    qui les a appariées est à jeter, même s'il affiche une belle moyenne.

Une moyenne de ressemblance ne fait pas cette différence : 0,90 de moyenne peut
être « tout est bon sauf les deux dernières mesures » ou « c'est mou partout ».
On regarde donc le PROFIL mesure par mesure de chaque paire, et on note à quel
point le désaccord est tardif.

LA MESURE. Pour une paire, on lit l'accord mesure contre mesure le long de la
diagonale, puis on calcule le barycentre du désaccord : 0 = tout le désaccord
est sur la première mesure, 1 = tout est sur la dernière. Une paire sans
désaccord vaut 1 — rien à reprocher. Le score d'un pavage est la moyenne sur
toutes ses paires, et il ne remplace pas le coût d'écriture : il le départage.
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
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import blocks_flex as BF                                                  # noqa: E402
import melody_ssm as MS                                                   # noqa: E402

TRUTH_START = MS.TRUTH_START
DEFAULT = B8.DEFAULT


def pair_profile(S, a, b, L):
    """L'accord mesure contre mesure entre deux passages alignés."""
    L = min(L, len(S) - max(a, b))
    return np.array([S[a + i, b + i] for i in range(max(0, L))])


def lateness(prof):
    """0 = le désaccord est tout au début, 1 = tout à la fin. Pas de désaccord → 1."""
    d = np.clip(1.0 - prof, 0, None)
    if d.sum() <= 1e-9 or len(prof) < 2:
        return 1.0
    w = np.arange(len(prof)) / (len(prof) - 1)
    return float((d * w).sum() / d.sum())


def pairs_of(secs):
    """Les paires à comparer : chaque passage contre le PREMIER de sa lettre."""
    first, out = {}, []
    for s in secs:
        if s["kind"] != "bloc":
            continue
        base = s["letter"].rstrip("′")
        if base not in first:
            first[base] = s
            continue
        out.append((first[base], s, base))
    return out


def score_tiling(S, secs):
    ps = pairs_of(secs)
    if not ps:
        return 0.0, []
    rows = []
    for a, b, base in ps:
        L = min(a["b1"] - a["b0"], b["b1"] - b["b0"]) + 1
        prof = pair_profile(S, a["b0"], b["b0"], L)
        rows.append({"base": base, "a": a, "b": b, "prof": prof,
                     "late": lateness(prof), "mean": float(prof.mean())})
    return float(np.mean([r["late"] for r in rows])), rows


def song(stem):
    S, n, grid = load(stem)
    H = BF.head_matrix(S, n)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    b_sing = VA.bar_of(grid, onset)
    vstart = MS.voice_start(b_sing, n)

    cands = sorted({c for c in range(0, min(n - 8, 14), 2)} | ({vstart} if vstart
                                                               is not None else set()))
    hyps = []
    for c in cands:
        b = BF.best_tiling(S, H, n, c)
        if not b:
            continue
        secs = BF.to_sections(n, c, b[1], b[2], b[3], b[4])
        late, rows = score_tiling(S, secs)
        hyps.append({"start": c, "secs": secs, "cost": b[7], "late": late,
                     "rows": rows})
    by_cost = min(hyps, key=lambda h: h["cost"])
    # LA PLACE DU CRITÈRE. Seul, il se trompe : il choisit la mesure 7 sur les
    # deux morceaux verrouillés, alors que la vérité est 5. Il n'est pas fait
    # pour choisir un départ — un départ tardif rogne les paires et flatte
    # mécaniquement leur profil.
    #
    # En DÉPARTAGE, il est décisif. Sur Bein Green les départs 3 et 5 coûtent
    # exactement pareil, 26 mesures écrites, et rien d'autre ne les sépare :
    # l'alignement les sépare de 27 % contre 70 %. Le départ 3 fait diverger A
    # et A′ dès le DÉBUT — ce ne sont pas les mêmes huit mesures. Le départ 5
    # les fait diverger à la fin, comme une vraie reprise.
    best_cost = min(h["cost"] for h in hyps)
    by_both = max((h for h in hyps if h["cost"] == best_cost),
                  key=lambda h: round(h["late"], 3))
    # …et le départ, lui, vient de la voix : c'est la règle que Louis a validée.
    by_voice = next((h for h in hyps if h["start"] == vstart), None)
    by_late = by_both

    # la figure : le profil de chaque paire du pavage retenu par l'alignement
    rows = by_late["rows"]
    heights = [2.4] + [0.42] * len(rows) + [0.30, 0.55, 0.55]
    Hh = sum(heights) + 1.4
    fig, axs = plt.subplots(len(heights), 1, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .30})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.60 / Hh)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_xlim(0, n); axs[0].set_ylabel("mesure", fontsize=7.5)
    axs[0].tick_params(labelsize=6.4)
    axs[0].set_title(f"pavage retenu par l'alignement — départ mesure "
                     f"{by_late['start']+1}", fontsize=8.4, color="#8a2b2b",
                     loc="left", pad=3)

    for i, r in enumerate(rows):
        ax = axs[1 + i]
        p = r["prof"]
        ax.imshow(p[None, :], aspect="auto", cmap="RdYlGn", vmin=.5, vmax=1,
                  extent=(0, len(p), 0, 1), interpolation="nearest")
        for k, v in enumerate(p):
            ax.text(k + .5, .5, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                    color="#111")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{r['base']} : mes. {r['a']['b0']+1} vs {r['b']['b0']+1}\n"
                      f"désaccord à {r['late']:.0%} de la fin",
                      fontsize=6.2, rotation=0, ha="right", va="center",
                      color="#1f8a5b" if r["late"] > .6 else "#8a2b2b")
    axs[1 + len(rows)].axis("off")
    B8.strip(axs[-2], by_late["secs"], n,
             f"ALIGNEMENT\ndépart {by_late['start']+1} · {by_late['late']:.0%}")
    B8.strip(axs[-1], by_cost["secs"], n,
             f"coût d'écriture\ndépart {by_cost['start']+1} · coût {by_cost['cost']}")
    for a in (axs[-2], axs[-1]):
        a.set_xlim(0, n)
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    tr = TRUTH_START.get(stem)
    verdict = ""
    if tr is not None:
        ok = (by_both["start"] + 1) == tr
        okv = by_voice is not None and (by_voice["start"] + 1) == tr
        verdict = (f"<span class='gt {'ok' if ok else 'no'}'>vérité : mesure {tr} — "
                   f"coût seul {by_cost['start']+1}, coût + alignement "
                   f"{by_both['start']+1} {'✓' if ok else '✗'}</span> "
                   f"<span class='gt {'ok' if okv else 'no'}'>la voix dit "
                   f"{by_voice['start']+1 if by_voice else '—'} "
                   f"{'✓' if okv else '✗'}</span>")
    fmt = lambda x: " ".join(
        ("intro" if s["kind"] == "intro" else "?" if s["kind"] == "reste"
         else s["letter"]) + f"[{s['b0']+1}-{s['b1']+1}]" for s in x)
    trs = "".join(
        f"<tr class='{'win' if h is by_both else ''}'><td>mes. {h['start']+1}</td>"
        f"<td>{h['late']:.0%}</td><td>{h['cost']}</td>"
        f"<td class=f>{fmt(h['secs'])}</td></tr>" for h in hyps)
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{r['a']['b0']},{r['a']['b1']+1}]'>{r['base']}"
        f"<small>{r['a']['b0']+1}</small></button>"
        f"<button class=blk data-p='[{r['b']['b0']},{r['b']['b1']+1}]'>{r['base']}′"
        f"<small>{r['b']['b0']+1}</small></button>" for r in rows[:6])
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
{len(rows)} paires</span></h2>{verdict}
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>écoute une paire l'une après l'autre</span>{btns}</div>
<table><tr><th>départ</th><th>désaccord tardif</th><th>coût</th>
<th>découpage</th></tr>{trs}</table></section>"""


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
    out = HERE / "harmonia_min/state/reports/align_diff.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Où A et A′ divergent</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
.gt{{display:inline-block;font:700 12px system-ui;border-radius:6px;padding:2px 8px;margin-bottom:8px}}
.gt.ok{{color:#0f5132;background:#e4f0e8}} .gt.no{{color:#8a2b2b;background:#f7e4e4}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}} td.f{{font:500 11px ui-monospace,monospace}}
tr.win td{{background:#e4f0e8;font-weight:700}}
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
<h1>Où A et A′ divergent</h1>
<div class=lede>Deux passages qui portent la même lettre ne sont jamais
identiques. Ce qui compte n'est pas <b>combien</b> ils diffèrent, c'est
<b>où</b>.<br><br>
S'ils s'accordent au début et divergent <b>à la fin</b>, c'est une vraie reprise
avec une autre sortie — turnaround, montée, dernière mesure changée : l'alignement
est bon. S'ils divergent <b>au milieu ou dès le début</b>, ce ne sont pas les
mêmes huit mesures posées au même endroit, et le pavage qui les a appariées est à
jeter, même s'il affiche une belle moyenne.<br><br>
Une moyenne ne fait pas cette différence : 0,90 peut vouloir dire « tout est bon
sauf les deux dernières mesures » comme « c'est mou partout ». On lit donc le
<b>profil mesure par mesure</b> de chaque paire — une bande par paire, vert =
d'accord, rouge = pas d'accord — et on mesure à quel point le désaccord est
tardif : 0 % = tout est au début, 100 % = tout est à la fin.<br><br>
Ce critère ne remplace pas le coût d'écriture, il le <b>départage</b>. Les deux
découpages sont dessinés en bas, et les boutons jouent chaque paire l'une après
l'autre.</div>
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
