"""Le chant confirme-t-il l'hypothèse ? On glisse le bloc de chant, on regarde où il retombe.

    python scripts/melody_check.py [<stem> ...]  ->  /reports/melody_check.html

Louis, 2026-08-06 :

  « On peut exploiter la matrice SSM de voix de la manière suivante : une fois
    qu'on a une hypothèse sur tel bloc qui se répète, on regarde si c'est
    cohérent sur les patterns de chant. Quel est le score si je fais glisser le
    pattern de bloc de chant — pas la matrice diagonale — sur l'axe x : est-ce
    que si je prends le bloc correspondant au premier A sur la matrice SSM de
    chant, je retrouve bien un alignement réel quand j'arrive au deuxième
    bloc ? »

C'est un TEST, pas un détecteur, et c'est ce qui le rend utile. L'hypothèse vient
de l'harmonie ; le chant sert de témoin indépendant. On prend le premier A tel
que l'harmonie l'a découpé, on le glisse mesure par mesure sur toute la chanson
en ne regardant QUE ce que la voix y chante, et on lit la courbe :

  * un pic franc pile sur le deuxième A  ->  les deux couplets se chantent
    pareil, l'hypothèse tient sur une voie que l'harmonie n'a pas fournie ;
  * un pic ailleurs, décalé de deux mesures  ->  l'alignement est faux, et le
    décalage dit de combien ;
  * pas de pic du tout  ->  le chant ne dit rien ici, souvent parce que les
    paroles changent d'un couplet à l'autre. C'est une réponse aussi : le
    témoin se récuse, il ne contredit pas.

Le score d'une hypothèse est la part de ses reprises attendues où la courbe de
chant fait effectivement un pic (à une mesure près).
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
import vocal_melody as VM                                                 # noqa: E402

TOL = 1            # « pile dessus » = à une mesure près
DEFAULT = B8.DEFAULT


def slide_on(M, a0, L, n):
    """Le bloc [a0, a0+L) glissé sur l'axe des mesures. Pas la diagonale : le BLOC."""
    out = np.zeros(n)
    for c in range(0, n - L + 1):
        vals = [M[a0 + i, c + i] for i in range(L)
                if a0 + i < n and c + i < n]
        out[c] = float(np.mean(vals)) if vals else 0.0
    return out


def clear_elsewhere(cur, thr, want, tol):
    """Des pics francs ailleurs qu'à l'endroit attendu — la seule vraie objection."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(cur, height=thr, distance=4)
    return [int(i) for i in idx if abs(i - want) > tol]


def check(M, secs, n, tol=TOL):
    """Pour chaque lettre : la courbe de chant du premier bloc, et où elle pique."""
    first, groups = {}, {}
    for s in secs:
        if s["kind"] != "bloc":
            continue
        base = s["letter"].rstrip("′")
        groups.setdefault(base, []).append(s)
        first.setdefault(base, s)
    out = []
    for base, members in groups.items():
        if len(members) < 2:
            continue
        a = first[base]
        L = a["b1"] - a["b0"] + 1
        cur = slide_on(M, a["b0"], L, n)
        if cur.max() <= 0:
            out.append({"base": base, "cur": cur, "hits": [], "want": [],
                        "mute": True})
            continue
        want = [s["b0"] for s in members if s is not a]
        hits = []
        for w in want:
            lo, hi = max(0, w - tol), min(n, w + tol + 1)
            loc = int(np.argmax(cur[lo:hi])) + lo
            near = cur[loc]
            # LA VOIX NE PEUT QU'AJOUTER, JAMAIS RETRANCHER. Louis,
            # 2026-08-06 : « on peut avoir la même section mais chantée
            # différemment ; le chant est un indicateur, un pic clair sert à
            # contester ou suggérer un début, mais un pic pas clair ne devrait
            # pas réfuter l'hypothèse qu'on recommence une section ».
            #
            # Trois verdicts, donc, et pas deux. Un pic franc à l'endroit
            # attendu CONFIRME. Pas de pic du tout ne dit RIEN — les paroles
            # changent d'un couplet à l'autre, c'est la règle plutôt que
            # l'exception, et la section recommence quand même. Seul un pic
            # franc AILLEURS est une objection, et encore : elle porte sur le
            # placement, pas sur l'existence de la reprise.
            thr = max(0.55, float(np.median(cur[cur > 0])) + .12)
            ok = near >= thr
            elsewhere = [p for p in clear_elsewhere(cur, thr, w, tol)]
            hits.append({"want": w, "at": loc, "val": float(near),
                         "ok": bool(ok),
                         "verdict": "confirmé" if ok else
                                    ("objection" if elsewhere else "sans avis")})
        out.append({"base": base, "cur": cur, "hits": hits, "want": want,
                    "mute": False})
    return out


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    b_sing = VA.bar_of(grid, onset)
    vstart = MS.voice_start(b_sing, n) or 0

    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)

    H = BF.head_matrix(S, n)
    b = BF.best_tiling(S, H, n, vstart)
    secs = BF.to_sections(n, vstart, b[1], b[2], b[3], b[4])
    res = check(M, secs, n)

    tested = [h for r in res for h in r["hits"]]
    okn = sum(1 for h in tested if h["ok"])
    # « sans avis » ne compte ni pour ni contre : on ne le met pas au
    # dénominateur, sinon l'absence de preuve deviendrait une preuve contraire.
    voiced = [h for h in tested if h["verdict"] != "sans avis"]
    mum = len(tested) - len(voiced)

    heights = [2.2, 2.2] + [0.85] * len(res) + [0.55]
    Hh = sum(heights) + 1.4
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .28})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.60 / Hh)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.4); axs[0].tick_params(labelsize=6.2)
    axs[0].set_title("HARMONIE — d'où vient l'hypothèse", fontsize=8, color="#2a6fb0",
                     loc="left", pad=3)
    axs[1].imshow(M, origin="lower", extent=(0, n, 0, n), cmap="Purples",
                  vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axs[1].set_ylabel("mesure", fontsize=7.4); axs[1].tick_params(labelsize=6.2)
    axs[1].set_title("CHANT — le témoin", fontsize=8, color="#7c3aed", loc="left", pad=3)

    for i, r in enumerate(res):
        ax = axs[2 + i]
        col = COLS[i % len(COLS)]
        ax.fill_between(np.arange(n) + .5, r["cur"], color=col, alpha=.30, lw=0)
        ax.plot(np.arange(n) + .5, r["cur"], color=col, lw=1.1)
        a0 = [s for s in secs if s["kind"] == "bloc"
              and s["letter"].rstrip("′") == r["base"]][0]["b0"]
        ax.axvline(a0 + .5, color="#111", lw=1.4)
        ax.text(a0 + 1, 1.02, "  le bloc de départ", fontsize=6, va="top", color="#111")
        for h in r["hits"]:
            ax.axvline(h["want"] + .5, color="#1f8a5b" if h["ok"] else "#8a2b2b",
                       lw=1.4, ls=(0, (3, 2)))
            ax.plot([h["at"] + .5], [h["val"]], "o", ms=4,
                    color="#1f8a5b" if h["ok"] else "#8a2b2b")
            ax.text(h["want"] + .8, .06, f"{h['val']:.2f}", fontsize=5.8,
                    color="#1f8a5b" if h["ok"] else "#8a2b2b")
        ax.set_ylim(0, 1.12); ax.set_yticks([])
        ok = sum(1 for h in r["hits"] if h["ok"])
        ax.set_ylabel(f"{r['base']} glissé\n{ok}/{len(r['hits'])} confirmées",
                      fontsize=6.4, rotation=0, ha="right", va="center", color=col)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    B8.strip(axs[-1], secs, n, f"hypothèse\ndépart mes. {vstart+1}")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.2)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    rows = ""
    for r in res:
        if r["mute"]:
            rows += (f"<tr><td><b>{r['base']}</b></td><td colspan=3>"
                     "aucun chant dans ce bloc — le témoin se récuse</td></tr>")
            continue
        for h in r["hits"]:
            d = h["at"] - h["want"]
            rows += (f"<tr class='{'ok' if h['ok'] else 'no'}'><td><b>{r['base']}</b></td>"
                     f"<td>mesure {h['want']+1}</td><td>{h['val']:.2f}</td>"
                     f"<td>{h['verdict']}"
                     f"{f' · pic décalé de {d:+d} mesure(s)' if d and h['ok'] else ''}"
                     f"</td></tr>")
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
        f"<small>{s['b0']+1}</small></button>" for s in secs if s["kind"] == "bloc")
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
départ mes. {vstart+1} (règle de la voix) ·
<b>{okn}/{len(voiced) or 1} confirmées</b> · {mum} sans avis</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>écoute deux blocs de même lettre à la suite</span>{btns}</div>
<table><tr><th>lettre</th><th>reprise attendue</th><th>score du chant</th>
<th>verdict</th></tr>{rows}</table></section>"""


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
    out = HERE / "harmonia_min/state/reports/melody_check.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Le chant confirme-t-il ?</title><style>
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
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
tr.ok td{{background:#e4f0e8}} tr.no td{{background:#faf4e6}}
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
<h1>Le chant confirme-t-il ?</h1>
<div class=lede>L'hypothèse vient de l'harmonie ; le chant sert de <b>témoin
indépendant</b>. On prend le premier A tel que l'harmonie l'a découpé, on le
<b>glisse mesure par mesure</b> sur toute la chanson en ne regardant que ce que
la voix y chante, et on lit la courbe.<br><br>
Un <b>pic franc pile sur le deuxième A</b> : les deux couplets se chantent
pareil, l'hypothèse tient sur une voie que l'harmonie n'a pas fournie. Un pic
<b>décalé</b> : l'alignement est faux, et le décalage dit de combien. <b>Pas de
pic du tout</b> : le chant ne dit rien ici, souvent parce que les paroles
changent d'un couplet à l'autre — le témoin se récuse, il ne contredit pas.
<br><br>
Le trait noir est le bloc de départ, les traits pointillés les reprises
attendues : <b>vert</b> quand le chant confirme, <b>rouge</b> quand il ne dit
rien. Les boutons jouent deux blocs de même lettre à la suite.</div>
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
