"""Où sont nos mesures, au son. Une règle graduée qu'on peut écouter.

    python scripts/bar_ruler.py <stem> [...]  ->  /reports/bars_<stem>.html

Louis, 2026-08-07 : « Come Away With Me, montre-moi, je ne sais pas où est ta
mesure 1 ».

C'est un manque réel de toutes les pages précédentes : elles discutent de la
mesure 9 contre la mesure 10 sans jamais permettre de vérifier que notre mesure 9
est bien celle qu'il entend. Tant que la grille n'est pas audible, tout désaccord
sur une frontière est indécidable — on ne sait pas si on parle du même endroit.

La page donne donc trois choses, et rien de plus : l'énergie d'attaque avec une
barre par mesure et son numéro, un bouton par mesure qui joue deux mesures à
partir de là, et le découpage en sections dessous. Un clic dans le graphe
déplace la lecture.
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
from pattern_lanes import fig2b64_fixed, INK, PLOT_L, PLOT_R               # noqa: E402
import blocks8 as B8                                                      # noqa: E402
from harmonia_min import musx as mx, voice_sections as VS                 # noqa: E402
from harmonia_min import sections as hs, pipeline as _pl, beats as B      # noqa: E402


def song(stem):
    p = HERE / f"docs/audio/{stem}.m4a"
    c = {}
    real = hs.detect_sections

    def spy(g, a, t, bars=None, **k):
        c.update(grid=g)
        return real(g, a, t, bars, **k)

    hs.detect_sections = spy
    try:
        _pl.analyze(p, title="x", file_key="x", audio_url="")
    finally:
        hs.detect_sections = real
    grid = [float(x) for x in c["grid"]]
    n = len(grid) - 1
    d = B.track(p)
    beats = list(map(float, d["beats"]))

    import librosa
    y, sr = librosa.load(str(p), sr=22050, mono=True)
    o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    t = librosa.frames_to_time(np.arange(len(o)), sr=sr, hop_length=512)
    o = o / max(1e-9, float(o.max()))

    secs = VS.detect_sections(grid, mx.frame_posteriors(p)[0], None, p)

    # l'axe est en MESURES pour que le curseur du navigateur retombe juste
    def to_bar(x):
        return float(np.interp(x, grid, np.arange(n + 1)))

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12.6, 3.2),
                            gridspec_kw={"height_ratios": [2.2, 0.7], "hspace": .38})
    Hh = 3.2
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.62 / Hh)
    ax = axs[0]
    ax.fill_between([to_bar(x) for x in t], o, color="#8a8371", alpha=.45, lw=0)
    for k, bt in enumerate(beats):          # les temps, en gris pâle
        ax.axvline(to_bar(bt), color="#d8cfb4", lw=.6, zorder=0)
    for b in range(n + 1):                  # les MESURES, en noir
        ax.axvline(b, color="#111", lw=1.1 if b % 4 == 0 else .55,
                   alpha=.9 if b % 4 == 0 else .45)
        if b % 4 == 0 and b < n:
            ax.text(b + .08, 1.03, str(b + 1), fontsize=6.6, color="#111",
                    va="bottom", fontweight="bold")
    ax.set_xlim(0, n); ax.set_ylim(0, 1.14); ax.set_yticks([])
    ax.set_ylabel("attaques", fontsize=7, rotation=0, ha="right", va="center")
    ax.set_title("une barre noire par mesure · les traits pâles sont les temps",
                 fontsize=8, color="#8a2b2b", loc="left", pad=12)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    strip = [{"b0": s["b0"], "b1": s["b1"], "kind": "bloc",
              "letter": str(s["label"]), "rep": 2} for s in secs]
    B8.strip(axs[1], strip, n, "sections")
    axs[1].set_xticks(range(0, n + 1, 4)); axs[1].tick_params(labelsize=6.2)
    axs[1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fm = lambda x: f"{int(x // 60)}:{x % 60:05.2f}"
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{b},{min(n, b + 2)}]'>{b + 1}"
        f"<small>{fm(grid[b])}</small></button>" for b in range(n))
    rows = "".join(
        f"<tr><td>{s['label']}</td><td>mes. {s['b0']+1}–{s['b1']+1}</td>"
        f"<td>{fm(grid[s['b0']])} → {fm(grid[min(n, s['b1']+1)])}</td></tr>"
        for s in secs)
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
mesure 1 à <b>{fm(grid[0])}</b> · une mesure dure {np.median(np.diff(grid)):.2f} s</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>chaque bouton joue 2 mesures</span></div>
<div class=grid>{btns}</div>
<table><tr><th>section</th><th>mesures</th><th>temps</th></tr>{rows}</table>
</section>"""


def main():
    stems = sys.argv[1:] or ["norah_jones_come_away_with_me"]
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable"); continue
        try:
            body += song(st); print(f"  ok {st}")
        except Exception as exc:
            import traceback
            traceback.print_exc(); print(f"  !! {st} — {exc}")
    name = "bars_" + (stems[0] if len(stems) == 1 else "corpus")
    out = HERE / f"harmonia_min/state/reports/{name}.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Où sont nos mesures</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:16px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
.grid{{display:flex;flex-wrap:wrap;gap:3px;margin:8px 0}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:10px}}
th,td{{border:1px solid #e5dcc6;padding:3px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#b3261e;opacity:.9;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.6)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 4px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 13px ui-monospace,monospace;min-width:110px}}
.hint{{font:500 11px system-ui;color:#a89f8c}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:5px;
  padding:2px 5px;cursor:pointer;font:700 11px ui-monospace,monospace;min-width:34px}}
button.blk small{{display:block;font:500 7.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#b3261e !important}}
</style></head><body><div class=wrap>
<h1>Où sont nos mesures</h1>
<div class=lede>Une barre noire par mesure, épaisse toutes les quatre et
numérotée ; les traits pâles sont les temps. <b>Chaque bouton joue deux mesures</b>
à partir de là, et un clic dans le graphe déplace la lecture. Tant que la grille
n'est pas audible, un désaccord sur une frontière est indécidable — on ne sait
pas si on parle du même endroit.</div>
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
