"""À quoi ressemble le score si on y met aussi l'harmonie ? Démo, rien de branché.

    python scripts/score_demo.py [<stem> ...]  ->  /reports/score_demo.html

Louis, 2026-08-07 :

  « Sur This Love le score n'est pas suffisant pour recouvrir les sections, il
    faudrait sûrement aussi rajouter le même score (height + ½ sharpness) de
    l'harmonie pour créer le score. Enfin le score devrait peut-être être relatif
    au pic de référence (celui du début de la section initiale) afin de pouvoir
    faire une règle généralisable, mais je ne suis pas sûr de cette dernière
    règle. Tu me fais une démo ? Je veux voir à quoi ce score ressemble. »

QUATRE VARIANTS, MÊME MACHINERIE, sur les mêmes blocs :

  1. **voix**            hauteur + ½ finesse sur le glissement chanté. C'est ce
                         qui tourne aujourd'hui dans `voice_first`.
  2. **harmonie**        exactement la même formule, sur le glissement
                         harmonique. Montrée seule pour qu'on voie ce qu'elle
                         apporte et ce qu'elle abîme.
  3. **voix + harmonie** la moyenne des deux. Une reprise vraie devrait piquer
                         sur les deux voies ; une boucle d'accords qui tourne
                         sans que la section recommence ne pique que sur une.
  4. **normalisé**       la même, divisée par le score de l'ancre elle-même — le
                         « pic de référence » dont Louis doute. L'idée est de
                         rendre les morceaux comparables : Norah vit vers 1.5 et
                         The Walk vers 0.6, donc aucun seuil global ne marche
                         aujourd'hui. Le doute est légitime et la page est faite
                         pour le trancher à l'œil : si la normalisation marche,
                         les quatre morceaux doivent tomber dans la même plage.

Chaque variante garde ses pics avec la règle inchangée — le plus fort se sert le
premier, rien à moins de huit mesures d'un pic déjà pris — et sa bande de
sections est dessinée en dessous. On voit donc la courbe ET sa conséquence.

Rien n'alimente le pipeline : `voice_first` continue de tourner sur la voix
seule tant que Louis n'a pas tranché.
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
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import melody_check as MC                                                 # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import channels as CN                                                     # noqa: E402
import voice_first as VF                                                  # noqa: E402

BLOCK, UNIT = VF.BLOCK, VF.UNIT
DEFAULT = B8.DEFAULT
VARIANTS = [("voix (actuel)", "#7c3aed"), ("harmonie", "#2a6fb0"),
            ("voix + harmonie", "#b3261e"), ("normalisé par l'ancre", "#1f8a5b")]


def score_series(cur, n, unit=UNIT):
    """La formule de Louis appliquée partout : hauteur + ½ finesse."""
    return np.array([VF.score_at(cur, p, n, unit) for p in range(n)])


def four_scores(cur_m, cur_h, b0, n):
    """Les quatre variantes, sur le même bloc."""
    sm, sh = score_series(cur_m, n), score_series(cur_h, n)
    mix = (sm + sh) / 2.0
    ref = max(1e-6, float(mix[b0]))       # le pic de référence : l'ancre elle-même
    return [sm, sh, mix, mix / ref]


def keep(series, b0, n, thr, claimed=None, block=BLOCK, unit=UNIT):
    """La règle de sélection, inchangée : le plus fort d'abord, rien à moins de 8."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(series, height=thr, distance=unit)
    cand = [int(p) for p in idx
            if (p - b0) % unit == 0 and abs(p - b0) >= block and p + block <= n
            and (claimed is None or not claimed[p:p + block].any())]
    out = []
    for p in sorted(cand, key=lambda p: -series[p]):
        if any(abs(p - q) < block for q in out):
            continue
        out.append(p)
    return sorted(out)


def run(S, M, n, vstart, vi, max_blocks=2):
    """Deux passes avec la variante `vi`, comme `voice_first`."""
    claimed = np.zeros(n, bool)
    runs, cursor = [], vstart
    while cursor + BLOCK <= n and len(runs) < max_blocks:
        if claimed[cursor:cursor + BLOCK].any():
            cursor += UNIT
            continue
        cur_m = MC.slide_on(M, cursor, BLOCK, n)
        cur_h = MC.slide_on(S, cursor, BLOCK, n)
        series = four_scores(cur_m, cur_h, cursor, n)[vi]
        # seuil : le décile 90 des scores de CE bloc, la seule échelle qui se
        # transporte d'un morceau à l'autre tant que rien n'est tranché
        thr = float(np.quantile(series, 0.90))
        occ = keep(series, cursor, n, thr, claimed)
        runs.append({"b0": cursor, "occ": occ, "series": series, "thr": thr,
                     "cur_m": cur_m, "cur_h": cur_h})
        for c in [cursor] + occ:
            claimed[c:min(n, c + BLOCK)] = True
        cursor += BLOCK
    return runs


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(grid, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)

    allruns = [run(S, M, n, vstart, vi) for vi in range(len(VARIANTS))]
    nb = max(len(r) for r in allruns)

    heights = [1.9] + [0.95] * nb + [0.26] + [0.52] * len(VARIANTS)
    Hh = sum(heights) + 1.6
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .30})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.62 / Hh)
    axs[0].imshow(M, origin="lower", extent=(0, n, 0, n), cmap="Purples",
                  vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.4); axs[0].tick_params(labelsize=6.2)
    axs[0].set_title("CHANT", fontsize=8, color="#7c3aed", loc="left", pad=3)

    ref = allruns[0]
    for bi in range(nb):
        ax = axs[1 + bi]
        b0 = ref[bi]["b0"] if bi < len(ref) else None
        for vi, (nm, col) in enumerate(VARIANTS):
            rs = allruns[vi]
            if bi >= len(rs):
                continue
            r = rs[bi]
            y = r["series"]
            top = max(1e-6, float(np.max(y)))
            ax.plot(mid(np.arange(n)), y / top, color=col, lw=1.15, alpha=.9)
            for p in r["occ"]:
                ax.plot([mid(p)], [y[p] / top], "o", ms=5, color=col)
        if b0 is not None:
            ax.axvline(edge(b0), color="#111", lw=1.5)
            ax.text(edge(b0) + .6, 1.02, f" bloc mes. {b0+1}", fontsize=6,
                    va="top", color="#111")
        ax.set_ylim(0, 1.12); ax.set_yticks([])
        ax.set_ylabel(f"bloc {bi+1}\n(chaque courbe à son échelle)", fontsize=6,
                      rotation=0, ha="right", va="center", color="#4a4436")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axs[1 + nb].axis("off")
    axs[1 + nb].set_title("ce que chaque variante DÉCOUPE", fontsize=8.4,
                          color="#8a2b2b", loc="left", pad=1)
    for vi, (nm, col) in enumerate(VARIANTS):
        secs = VF.sections_of(allruns[vi], n, vstart)
        B8.strip(axs[2 + nb + vi], secs, n, nm)
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.2)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fmt = lambda x: " ".join(
        ("intro" if s["kind"] == "intro" else s["letter"])
        + f"[{s['b0']+1}-{s['b1']+1}]" for s in x)
    thrs = lambda rs: ", ".join(f"{r['thr']:.2f}" for r in rs)
    rows = ""
    for vi, (nm, col) in enumerate(VARIANTS):
        rs = allruns[vi]
        span = " · ".join(
            f"bloc {r['b0']+1} → {', '.join(str(p+1) for p in r['occ']) or '—'}"
            for r in rs)
        vals = [f"{r['series'][p]:.2f}" for r in rs for p in r["occ"]]
        rows += (f"<tr><td><b style='color:{col}'>{nm}</b></td>"
                 f"<td>{span}</td>"
                 f"<td>{', '.join(vals) or '—'}</td>"
                 f"<td>{thrs(rs)}</td></tr>")
    lines = "<br>".join(
        f"<b style='color:{col}'>{nm}</b> : {fmt(VF.sections_of(allruns[vi], n, vstart))}"
        for vi, (nm, col) in enumerate(VARIANTS))
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
départ mes. {vstart+1}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>clique dans le graphe pour écouter</span></div>
<table><tr><th>variante</th><th>reprises trouvées</th><th>scores retenus</th>
<th>seuil (décile 90 du bloc)</th></tr>{rows}</table>
<p class=verdict>{lines}</p></section>"""


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
    out = HERE / "harmonia_min/state/reports/score_demo.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Le score, avec l'harmonie dedans</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} ol{{margin:8px 0 0;padding-left:20px}} ol li{{margin-bottom:5px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
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
</style></head><body><div class=wrap>
<h1>Le score, avec l'harmonie dedans</h1>
<div class=lede>Quatre variantes du même score, sur les mêmes blocs.
<ol>
<li><b style="color:#7c3aed">voix</b> — hauteur + ½ finesse sur le glissement
chanté. C'est ce qui tourne aujourd'hui.</li>
<li><b style="color:#2a6fb0">harmonie</b> — exactement la même formule sur le
glissement harmonique, montrée seule pour voir ce qu'elle apporte et ce qu'elle
abîme.</li>
<li><b style="color:#b3261e">voix + harmonie</b> — la moyenne. Une vraie reprise
devrait piquer sur les deux ; une boucle d'accords qui tourne sans que la section
recommence ne pique que sur une.</li>
<li><b style="color:#1f8a5b">normalisé par l'ancre</b> — la même, divisée par le
score du bloc de référence. C'est l'idée dont tu doutais : elle vise à rendre les
morceaux comparables, puisque Norah vit vers 1.5 et The Walk vers 0.6 et
qu'aucun seuil global ne marche aujourd'hui. <b>Le test est simple : si elle
marche, les quatre morceaux doivent tomber dans la même plage.</b></li>
</ol>
<br>Chaque courbe est tracée <b>à sa propre échelle</b> pour qu'on compare les
formes, pas les niveaux ; les niveaux sont dans le tableau. Les pastilles sont
les reprises que chaque variante retient, avec la règle inchangée — le plus fort
se sert le premier, rien à moins de huit mesures d'un pic déjà pris. En bas,
<b>ce que chaque variante découpe</b>.<br><br>
Rien n'est branché : <code>voice_first</code> continue sur la voix seule tant que
tu n'as pas tranché.</div>
{body}</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
const L0=%L0%, W=%W%;
let stopAt=null,live=null,raf=null;
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
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
  if(stopAt!=null&&au.currentTime>=stopAt){{au.pause();stopAt=null;}}
  if(!au.paused) raf=requestAnimationFrame(tick); }}
au.addEventListener("play",()=>{{ if(live) live.pp.textContent="❚❚"; tick(); }});
au.addEventListener("pause",()=>{{ if(live) live.pp.textContent="▶";
  cancelAnimationFrame(raf); draw(); }});
function go(sec,t0,t1){{
  if(live && live.sec!==sec){{ live.pp.textContent="▶"; live.cur.style.display="none"; }}
  live=sec._p; stopAt=t1;
  if(au.getAttribute("src")!==sec.dataset.audio){{
    au.setAttribute("src",sec.dataset.audio);au.load();}}
  const seek=()=>{{try{{au.currentTime=t0;}}catch(e){{}} draw();}};
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{{once:true}});
  au.play().catch(()=>{{}});
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
    go(sec, G[i]+(f-i)*(G[i+1]-G[i]), null); }};
  sec.querySelector(".pp").onclick=()=>{{
    if(au.paused||live!==sec._p) go(sec,G[0],null); else au.pause(); }};
}});
</script></body></html>""".replace("%L0%", str(PLOT_L)).replace("%W%", str(round(PLOT_R - PLOT_L, 6))))
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
