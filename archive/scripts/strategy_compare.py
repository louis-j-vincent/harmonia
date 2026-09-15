"""Chaque stratégie, ses sections, et ta vérité juste au-dessus.

    python scripts/strategy_compare.py [<stem> ...]  ->  /reports/strategies.html

Louis, 2026-08-07 :

  « Oui il faut aussi un terme qui paie la quantité, mais tient compte de la
    qualité aussi — montre-moi en HTML les sections détectées par chaque
    stratégie, que je voie si tes métriques de décision sont les bonnes. »

La demande derrière la demande est la bonne : mes deux chiffres — accord des
frontières, accord des lettres — sont eux-mêmes des choix, et rien ne dit qu'ils
mesurent ce que son oreille appelle un bon découpage. Ils sont donc affichés à
côté du dessin, pas à la place.

LES CRITÈRES DE SEUIL COMPARÉS. Chacun choisit son propre seuil, par morceau, en
maximisant sa clé ; on voit ensuite ce que ça découpe.

  moyenne          la règle de Louis. Mesurée fausse le 2026-08-07 : quand le
                   seuil monte de 0.30 à 0.86 la moyenne monte de 0.67 à 0.94
                   pendant que les reprises tombent de 6.9 à 2.1. Récompense
                   d'expliquer moins.
  somme            l'inverse exact : Σ des scores retenus. Paie la quantité,
                   mais un pic médiocre rapporte toujours quelque chose, donc
                   elle tire le seuil vers le bas.
  moyenne × part   la moyenne, multipliée par la part du morceau expliquée.
                   C'est le compromis que Louis demande : un pic de plus ne
                   paie que s'il ne dégrade pas trop les autres.
  seuil fixe       0.66 pour les blocs de 8, 0.54 pour ceux de 4 — les valeurs
                   qui maximisent l'accord des frontières sur tout le corpus.
                   Témoin : si un critère adaptatif ne fait pas mieux qu'un
                   nombre en dur, il ne sert à rien.

ET LA STRATÉGIE HYBRIDE, celle que Louis avait décrite avant qu'on la mesure :
« les blocs de 8 ANCRENT la chanson, et après il nous reste juste à compléter les
trous ». Les 8 mesures placent mieux les frontières (0.79 contre 0.76) mais ne
couvrent que 90 % du morceau ; les 4 mesures couvrent 95 %. On ancre donc en 8,
puis on repasse en 4 sur ce qui reste.
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
from pattern_lanes import (load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R)  # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import score_eval as SE                                                   # noqa: E402

GRID = np.round(np.arange(0.30, 0.94, 0.02), 2)


def runs_to_sections(runs, n, vstart):
    """Les blocs verrouillés deviennent des sections contiguës, trous compris."""
    owner = np.full(n, -1)
    for i, r in enumerate(runs):
        for c in [r["b0"]] + r["occ"]:
            owner[c:min(n, c + r["block"])] = i
    out, b = [], 0
    if vstart > 0:
        out.append({"b0": 0, "b1": vstart - 1, "kind": "intro", "letter": "intro"})
        b = vstart
    while b < n:
        z = b
        while z + 1 < n and owner[z + 1] == owner[b]:
            z += 1
        i = owner[b]
        out.append({"b0": b, "b1": z, "rep": 2 if i >= 0 else 1,
                    "kind": "bloc" if i >= 0 else "reste",
                    "letter": chr(ord("A") + i) if i >= 0 else "?"})
        b = z + 1
    ren, k = {}, 0
    for s in out:
        if s["kind"] == "bloc" and s["letter"] not in ren:
            ren[s["letter"]] = chr(ord("A") + k); k += 1
    for s in out:
        if s["kind"] == "bloc":
            s["letter"] = ren[s["letter"]]
    return out


def tag(runs, block):
    return [{**r, "block": block} for r in runs]


def hybride(S, M, n, vstart, t8, t4):
    """Ancrer en 8, combler en 4 — l'architecture que Louis avait décrite."""
    r8, claimed = SE.anchor_runs(S, M, n, vstart, t8, 20, 8)
    runs = tag(r8, 8)
    cursor = vstart
    from scipy.signal import find_peaks
    import melody_check as MC
    import voice_first as VF
    while cursor + 4 <= n:
        if claimed[cursor:cursor + 4].any():
            cursor += 2
            continue
        cur_m = MC.slide_on(M, cursor, 4, n)
        cur_h = MC.slide_on(S, cursor, 4, n)
        sc = VF.block_score(cur_m, cur_h, cursor, n)
        idx, _ = find_peaks(sc, height=t4, distance=2)
        cand = [int(p) for p in idx
                if (p - cursor) % 2 == 0 and abs(p - cursor) >= 4
                and p + 4 <= n and not claimed[p:p + 4].any()]
        occ = []
        for p in sorted(cand, key=lambda p: -sc[p]):
            if any(abs(p - q) < 4 for q in occ):
                continue
            occ.append(p)
        if occ:
            runs.append({"b0": cursor, "occ": sorted(occ), "block": 4,
                         "scores": [float(sc[p]) for p in occ]})
            for c in [cursor] + occ:
                claimed[c:min(n, c + 4)] = True
        cursor += 4
    return runs


def pick(S, M, n, vstart, block, key):
    """Le seuil que ce critère choisit, sur CE morceau."""
    best = None
    for thr in GRID:
        runs, _ = SE.anchor_runs(S, M, n, vstart, float(thr), 20, block)
        sc = [s for r in runs for s in r["scores"]]
        if not sc:
            continue
        lab = SE.labels_of(runs, n, vstart, block)
        part = float(np.mean(lab[vstart:] >= 0)) if n > vstart else 0.0
        v = {"moyenne": float(np.mean(sc)),
             "somme": float(np.sum(sc)),
             "moyenne × part": float(np.mean(sc)) * part}[key]
        if best is None or v > best[0]:
            best = (v, float(thr), tag(runs, block))
    return best if best else (0.0, GRID[0], [])


def song(stem, T):
    S, n, g = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(g, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, g, n)

    ref = SE.truth_labels(T[stem]["sections"], n)
    rb = SE.bounds(ref)

    strats = []
    for key in ("moyenne", "somme", "moyenne × part"):
        v, thr, runs = pick(S, M, n, vstart, 8, key)
        strats.append((f"bloc 8 · {key} (seuil {thr:.2f})", runs, 8))
    r, _ = SE.anchor_runs(S, M, n, vstart, 0.66, 20, 8)
    strats.append(("bloc 8 · seuil fixe 0.66", tag(r, 8), 8))
    # LE BLOC 4, PLUS EXIGEANT. Louis, 2026-08-07 : « et bloc 4 avec un seuil
    # plus haut ? les blocs de 4 sont plus intéressants je pense ». Les quatre
    # métriques lui donnent raison sur un point précis : à 0.54 le bloc 4
    # retrouve 85 % de ses frontières, le meilleur score du tableau — mais il en
    # pose beaucoup qu'il n'a pas. Monter le seuil devrait échanger l'un contre
    # l'autre ; à quel prix, c'est ce qu'on regarde.
    for t in (0.54, 0.62, 0.70, 0.78):
        r, _ = SE.anchor_runs(S, M, n, vstart, t, 20, 4)
        strats.append((f"bloc 4 · seuil fixe {t:.2f}", tag(r, 4), 4))
    strats.append(("HYBRIDE · ancrer en 8, combler en 4",
                   hybride(S, M, n, vstart, 0.66, 0.54), 8))

    truth_secs = [{"b0": s["b0"], "b1": s["b1"], "kind": "bloc",
                   "letter": s["label"], "rep": 2}
                  for s in T[stem]["sections"] if s["b1"] >= s["b0"]]

    heights = [0.60, 0.20] + [0.55] * len(strats)
    Hh = sum(heights) + 1.5
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .34})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.62 / Hh)
    B8.strip(axs[0], truth_secs, n, "TA VÉRITÉ")
    axs[0].set_title("la référence", fontsize=8, color="#1f8a5b", loc="left", pad=3)
    axs[1].axis("off")
    rows = ""
    for i, (name, runs, block) in enumerate(strats):
        secs = runs_to_sections(runs, n, vstart)
        pred = np.full(n, -1)
        for j, r in enumerate(runs):
            for c in [r["b0"]] + r["occ"]:
                pred[c:min(n, c + r["block"])] = j
        pred[:vstart] = -1
        pr, rc = SE.bounds_pr(SE.bounds(pred), rb)
        over, under = SE.entropies(pred, ref)
        cov = float(np.mean(pred[vstart:] >= 0)) if n > vstart else 0.0
        B8.strip(axs[2 + i], secs, n, name)
        rows += (f"<tr><td>{name}</td><td>{sum(len(r['occ']) for r in runs)}</td>"
                 f"<td>{pr:.2f}</td><td><b>{rc:.2f}</b></td>"
                 f"<td>{over:.2f}</td><td>{under:.2f}</td>"
                 f"<td>{cov:.0%}</td></tr>")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.2)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    gridjs = "[" + ",".join(f"{x:.3f}" for x in g) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
        f"<small>{s['b0']+1}</small></button>" for s in truth_secs)
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
départ mes. {vstart+1}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>tes sections</span>{btns}</div>
<table><tr><th>stratégie</th><th>rep.</th><th>frontières<br>justes</th>
<th>frontières<br>trouvées</th><th>sur-<br>découpage</th>
<th>sous-<br>découpage</th><th>couv.</th></tr>{rows}</table></section>"""


def main():
    T = SE.truth()
    stems = sys.argv[1:] or [k for k, v in T.items() if v.get("sections")]
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable"); continue
        try:
            body += song(st, T)
            print(f"  ok {st}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    out = HERE / "harmonia_min/state/reports/strategies.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Chaque stratégie, contre ta vérité</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} ul{{margin:8px 0 0;padding-left:20px}} ul li{{margin-bottom:4px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
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
<h1>Chaque stratégie, contre ta vérité</h1>
<div class=lede><b>Ta vérité est la bande du haut.</b> En dessous, ce que chaque
stratégie découpe, sur les mêmes mesures. Les chiffres sont à côté du dessin et
non à sa place : ce sont <b>mes</b> choix de mesure, et c'est justement ce que tu
dois pouvoir contester.
<ul>
<li><b>moyenne</b> — ta règle. Elle choisit un seuil trop haut : la moyenne des
survivants s'améliore mécaniquement quand on en garde moins.</li>
<li><b>somme</b> — l'inverse exact. Paie la quantité, mais un pic médiocre
rapporte quand même, donc elle tire le seuil vers le bas.</li>
<li><b>moyenne × part</b> — le compromis que tu demandes : la moyenne multipliée
par la part du morceau expliquée. Un pic de plus ne paie que s'il ne dégrade pas
trop les autres.</li>
<li><b>seuil fixe</b> — un nombre en dur, comme témoin. Si un critère adaptatif
ne bat pas un nombre en dur, il ne sert à rien.</li>
<li><b>HYBRIDE</b> — ta phrase, mesurée : les blocs de 8 placent mieux les
frontières (0.79 contre 0.76) mais couvrent 90 % du morceau contre 95 % pour les
blocs de 4. Donc on ancre en 8, on comble en 4.</li>
</ul>
<br><b>Les chiffres tiennent compte de ton avertissement</b> — « des fois je vais
différencier un A d'un B alors que d'autres considéreront que c'est la même
chose, des fois on va merger un B et un C ». Un seul chiffre d'accord punit
pareil « tu as coupé là où je n'aurais pas coupé » et « tu as collé ce que
j'aurais séparé », alors que ce ne sont pas la même faute et que l'une des deux
est souvent une question de goût. Donc quatre nombres, pas deux :<br>
<b>frontières justes</b> = parmi celles qu'on pose, combien tombent chez toi (on
n'invente pas). <b>frontières trouvées</b> = parmi les tiennes, combien on
retrouve (on n'en rate pas). Fusionner deux sections fait chuter la seconde sans
toucher la première.<br>
<b>sur-découpage</b> 1.00 = on n'a rien séparé de plus que toi ; il baisse si on
distingue un B et un C que tu avais laissés ensemble. <b>sous-découpage</b> 1.00
= on n'a rien collé que tu séparais. Ces deux-là ne regardent que l'information
partagée, <b>pas les noms</b> : un découpage identique au tien avec d'autres
lettres garde 1.00 partout.<br><br>
Les boutons jouent <b>tes</b> sections.</div>
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
