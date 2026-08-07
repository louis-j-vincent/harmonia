"""Ta vérité contre ce qu'on détecte aujourd'hui. Rien d'autre.

    python scripts/truth_vs_us.py  ->  /reports/truth_vs_us.html

Louis, 2026-08-07 : « montre-moi notre détection contre tous les ground truths
que j'ai annotés ».

Deux bandes par morceau, sur les mêmes mesures : la sienne au-dessus, la nôtre
en dessous, et les deux jouables. Ce qu'on montre est EXACTEMENT ce que le
serveur produit en `HARMONIA_SECTIONS=voice` — même code, même seuils, même
règle d'intro — et non une variante de laboratoire. C'est le seul moyen de
regarder un écart et de savoir qu'on regarde le vrai.

LES CHIFFRES, et pourquoi ce sont ceux-là. Louis a prévenu que les noms et les
regroupements sont à son appréciation : « des fois je vais différencier un A d'un
B alors que d'autres considéreront que c'est la même chose, des fois on va merger
un B et un C ». Un seul score d'accord punirait donc du désaccord de goût comme
si c'était une erreur. On rend ses deux exigences séparément :

  FRONTIÈRES PARTAGÉES — la part de nos sections dont les deux bords tombent sur
  des frontières à lui. Couper au milieu d'une de ses sections est une vraie
  faute, même bien nommée.

  CORRESPONDANCE — existe-t-il une fonction constante de ses sections vers les
  nôtres (A→A, B+C→B) ? Renommer ne coûte rien, sur-découper non plus tant que
  le découpage se répète à l'identique. C'est sa formulation, implémentée telle
  quelle dans `score_eval.correspondence`.

Et l'INTRO à part, parce qu'elle décale tout le reste quand elle est fausse.
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
from pattern_lanes import fig2b64_fixed, INK, PLOT_L, PLOT_R                # noqa: E402
import blocks8 as B8                                                       # noqa: E402
import score_eval as SE                                                    # noqa: E402
from harmonia_min import musx as mx, voice_sections as VS                  # noqa: E402
from harmonia_min import sections as hs, pipeline as _pl                   # noqa: E402


def grid_of(path):
    """La grille de mesures que le pipeline utilise vraiment."""
    c = {}
    real = hs.detect_sections

    def spy(g, a, t, bars=None, **k):
        c.update(grid=g)
        return real(g, a, t, bars, **k)

    hs.detect_sections = spy
    try:
        _pl.analyze(path, title="x", file_key="x", audio_url="")
    finally:
        hs.detect_sections = real
    return c["grid"]


def song(stem, T):
    p = HERE / f"docs/audio/{stem}.m4a"
    grid = grid_of(p)
    n = len(grid) - 1
    ours = VS.detect_sections(grid, mx.frame_posteriors(p)[0], None, p)

    his = [{"b0": s["b0"], "b1": min(n - 1, s["b1"]), "kind": "bloc",
            "letter": s["label"], "rep": 2}
           for s in T[stem]["sections"] if s["b1"] >= s["b0"] and s["b0"] < n]
    mine = [{"b0": s["b0"], "b1": s["b1"], "kind": "bloc",
             "letter": str(s["label"]), "rep": 2} for s in ours]

    ref = SE.truth_labels(T[stem]["sections"], n)
    pred = np.full(n, -1)
    for i, s in enumerate(ours):
        if s["label"] != "intro":
            pred[s["b0"]:s["b1"] + 1] = i
    edges, corr = SE.correspondence(pred, ref)
    pr, rc = SE.bounds_pr(SE.bounds(pred), SE.bounds(ref))
    hi = next((s["b1"] + 1 for s in T[stem]["sections"]
               if s["label"] == "intro"), 0)
    mi = next((s["b1"] + 1 for s in ours if s["label"] == "intro"), 0)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12.6, 2.4),
                            gridspec_kw={"height_ratios": [1, 1], "hspace": .45})
    Hh = 2.4
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .30 / Hh, bottom=.55 / Hh)
    B8.strip(axs[0], his, n, "TOI")
    B8.strip(axs[1], mine, n, "NOUS")
    axs[1].set_xticks(range(0, n + 1, 4)); axs[1].tick_params(labelsize=6.2)
    axs[1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fmt = lambda x: " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in x)
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
        f"<small>{s['b0']+1}</small></button>" for s in his)
    ok = "ok" if mi == hi else "no"
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>tes sections</span>{btns}</div>
<table><tr><th>intro</th><th>frontières justes</th><th>frontières trouvées</th>
<th>correspondance</th></tr>
<tr><td class={ok}>{mi} mes. · toi {hi}</td><td>{pr:.2f}</td><td>{rc:.2f}</td>
<td>{corr:.2f}</td></tr></table>
<p class=verdict><b>toi</b> &nbsp;: {fmt(his)}<br><b>nous</b> : {fmt(mine)}</p>
</section>""", (mi == hi, pr, rc, corr, edges)


def main():
    T = SE.truth()
    stems = sys.argv[1:] or [k for k, v in T.items() if v.get("sections")]
    body, acc = "", []
    for st in sorted(stems):
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable"); continue
        try:
            html, m = song(st, T)
            body += html; acc.append(m)
            print(f"  ok {st}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    if acc:
        a = np.array([[float(x) for x in m] for m in acc])
        tot = (f"<p class=tot><b>{int(a[:,0].sum())}/{len(acc)}</b> intros justes"
               f" · frontières justes <b>{a[:,1].mean():.2f}</b>"
               f" · trouvées <b>{a[:,2].mean():.2f}</b>"
               f" · correspondance <b>{a[:,3].mean():.2f}</b></p>")
        print(f"\n  intros {int(a[:,0].sum())}/{len(acc)} · justes {a[:,1].mean():.2f}"
              f" · trouvees {a[:,2].mean():.2f} · corresp {a[:,3].mean():.2f}")
    else:
        tot = ""
    out = HERE / "harmonia_min/state/reports/truth_vs_us.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Ta vérité contre nous</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:8px}}
.lede b{{color:{INK}}}
.tot{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:10px;padding:9px 12px;
  font:600 13.5px system-ui;margin:0 0 18px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
td.ok{{background:#e4f0e8}} td.no{{background:#f7dede}}
.verdict{{font:500 11.5px ui-monospace,monospace;background:#f7f3e9;
  border-radius:8px;padding:9px 11px;margin:10px 0 0;line-height:1.9}}
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
<h1>Ta vérité contre nous</h1>
<div class=lede>Ta bande au-dessus, la nôtre en dessous, sur les mêmes mesures.
C'est <b>exactement</b> ce que le serveur produit en <code>voice</code> — même
code, mêmes seuils, même règle d'intro — pas une variante de laboratoire.<br><br>
<b>frontières justes</b> : parmi celles qu'on pose, combien tombent chez toi.
<b>frontières trouvées</b> : parmi les tiennes, combien on retrouve.
<b>correspondance</b> : existe-t-il une fonction constante de tes sections vers
les nôtres (A→A, B+C→B) — renommer ne coûte rien, sur-découper non plus tant que
c'est régulier. L'<b>intro</b> est comptée à part parce qu'une intro fausse
décale tout le reste.</div>
{tot}{body}</div>
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
