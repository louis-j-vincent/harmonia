"""La page : l'intro cherchée dans la voix, un descripteur à la fois.

    python scripts/intro_vocal.py     ->  state/reports/intro_vocal.html

Sert le tableau de `intro_vocal_eval`, la matrice SSM vocale de chaque morceau,
les courbes des descripteurs les plus prometteurs, et de quoi ÉCOUTER : la
frontière de Louis, celle de la règle livrée, celle de chaque critère.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from pattern_lanes import fig2b64_fixed, INK, PLOT_L, PLOT_R, edge, mid  # noqa: E402
import intro_vocal as IV                                          # noqa: E402
import intro_vocal_eval as EV                                     # noqa: E402
import intro_vocal_push as PU                                     # noqa: E402

# les courbes montrées : les meilleures mesurées, plus celles que Louis a
# nommées explicitement (la ressemblance au reste, le timbre, le vibrato)
SHOW = ["voiced", "rms", "mfcc_delta", "vibrato", "sim_pitch", "sim_timbre"]
C_TRUTH, C_SHIP, C_CRIT = "#8a2b2b", "#1f8a5b", "#7c3aed"


def figure(st, d, preds):
    n = d["n"]
    F = d["F"]
    M = F["_M_pitch"]
    hs = [2.9] + [0.85] * len(SHOW)
    H = sum(hs) + .9
    fig, axs = plt.subplots(len(hs), 1, figsize=(12.6, H), sharex=True,
                            gridspec_kw={"height_ratios": hs})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .28 / H,
                        bottom=.62 / H, hspace=.13)

    def marks(ax, y0=None):
        """Les trois traits, toujours dans le même ordre de dessin : la vérité
        de Louis en dessous (large), la règle livrée par-dessus, le critère en
        dernier — sinon un critère juste EFFACE la vérité qu'il retrouve."""
        ax.axvline(edge(d["truth"]), color=C_TRUTH, lw=2.6, alpha=.9)
        ax.axvline(edge(d["shipped"]), color=C_SHIP, lw=2.2, ls=(0, (5, 3)))

    ax = axs[0]
    ax.imshow(M, origin="lower", extent=(0, n, 0, n), aspect="auto",
              cmap="magma", vmin=0, vmax=max(.3, float(np.nanpercentile(M, 97))))
    ax.text(.006, .96, "matrice SSM du chant", transform=ax.transAxes,
            fontsize=9, color="w", va="top", weight="bold")
    ax.set_xlim(0, n)
    marks(ax)
    ax.axhline(edge(d["truth"]), color=C_TRUTH, lw=1.4, alpha=.6)
    ax.tick_params(labelsize=8)

    x = mid(np.arange(n))
    for ax, name in zip(axs[1:], SHOW):
        y = np.asarray(F[name], float)
        lo = float(np.nanmin(y))
        ax.plot(x, y, color="#2a2a2a", lw=1.1)
        ax.fill_between(x, lo, y, color="#2a2a2a", alpha=.07)
        ax.set_xlim(0, n)
        ax.set_yticks([])
        marks(ax)
        k = preds.get(name)
        if k is not None:
            ax.axvline(edge(k), color=C_CRIT, lw=1.5, ls=(0, (2, 1.6)))
        ax.text(.006, .93, IV.FR[name], transform=ax.transAxes, fontsize=8,
                va="top", color="#4a4436",
                bbox=dict(fc="#fffdf6", ec="none", pad=1.2, alpha=.75))
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axs[-1].set_xlabel("mesure", fontsize=9)
    axs[-1].tick_params(labelsize=8)
    return fig2b64_fixed(fig)


def song_html(st, d, P):
    preds = {c: P[c][st][0] for c in SHOW if c in P}
    img = figure(st, d, preds)
    g = d["grid"]
    n = d["n"]
    tr, k0 = d["truth"], d["shipped"]
    gridjs = "[" + ",".join(f"{v:.3f}" for v in g) + "]"

    def btn(a, b, lab, col):
        a, b = max(0, min(n, a)), max(0, min(n, b))
        if b <= a:
            return ""
        return (f"<button class=blk style='border-color:{col}' "
                f"data-p='[{a},{b}]'>{lab}<small>mes. {a+1}–{b}</small></button>")

    b = btn(0, tr, "intro de Louis", C_TRUTH) if tr else \
        "<span class=hint>pas d'intro chez Louis</span>"
    b += btn(tr, tr + 4, "début de chanson", C_TRUTH)
    if k0 != tr:
        b += btn(min(k0, tr), max(k0, tr), "les mesures en litige", C_SHIP)

    rows = "".join(
        f"<tr class={'ok' if preds[c] == tr else 'no'}><td>{IV.FR[c]}</td>"
        f"<td>mes. {preds[c]+1}</td>"
        f"<td>{preds[c]-tr:+d}</td></tr>" for c in SHOW if c in preds)
    r = PU.ratio(d, "pitch")
    rtxt = ("bloc trop muet, pas d'avis" if not np.isfinite(r)
            else f"{r:.3f} — {'POUSSE' if r < .50 else 'ne pousse pas'}")
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{st}.m4a">
<h2>{st.replace('_', ' ').title()}<span class=sub>{n} mesures ·
Louis : intro de {tr} mesure(s) · règle livrée : {k0} ·
rapport de reprise {rtxt}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
{b}</div>
<table><tr><th>descripteur</th><th>sa frontière</th><th>écart</th></tr>
{rows}</table></section>"""


def table_html(rows, D, wi):
    tr = "".join(
        f"<tr class={'ok' if r['exact'] >= 6 else ''}>"
        f"<td>{IV.FR.get(r['key'].rstrip('⁻'), r['key'])}"
        f"{' <i>(sens inversé)</i>' if r['key'].endswith('⁻') else ''}"
        f"{' <i>(seuil ' + r['key'].split('@')[1] + ')</i>' if '@' in r['key'] else ''}"
        f"</td><td><b>{r['exact']}</b>/9</td><td>{r['tol1']}/9</td>"
        f"<td>{r['err']:.1f}</td><td>{r['auc']:.2f}</td></tr>"
        for r in rows)
    return f"""<table class=big><tr><th>descripteur vocal, SEUL</th>
<th>frontière exacte</th><th>à ±1 mesure</th><th>erreur moyenne</th>
<th>présence</th></tr>
<tr class=hd><td>la règle LIVRÉE (1re mesure chantée + levée)</td>
<td><b>{sum(D[s]['shipped'] == D[s]['truth'] for s in wi)}</b>/9</td>
<td>{sum(abs(D[s]['shipped'] - D[s]['truth']) <= 1 for s in wi)}/9</td>
<td>{np.mean([abs(D[s]['shipped'] - D[s]['truth']) for s in wi]):.1f}</td>
<td>1.00</td></tr>
<tr class=hd><td>« toujours 4 mesures », sans rien écouter</td>
<td><b>{sum(D[s]['truth'] == 4 for s in wi)}</b>/9</td>
<td>{sum(abs(4 - D[s]['truth']) <= 1 for s in wi)}/9</td>
<td>{np.mean([abs(4 - D[s]['truth']) for s in wi]):.1f}</td><td>0.50</td></tr>
{tr}</table>"""


def main(stems=None):
    D = IV.collect(stems or None)
    P = EV.predictions(D)
    rows, wi = EV.score(D, P)
    body = "".join(song_html(st, D[st], P) for st in sorted(D))
    R = {s: PU.ratio(D[s], "pitch") for s in sorted(D)}
    push = "".join(
        f"<tr class={'ok' if D[s]['truth'] != D[s]['shipped'] else ''}>"
        f"<td>{s.replace('_', ' ')[:34]}</td><td>{D[s]['shipped']}</td>"
        f"<td>{D[s]['truth']}</td>"
        f"<td>{'—' if not np.isfinite(R[s]) else f'{R[s]:.3f}'}</td></tr>"
        for s in sorted(D, key=lambda s: (np.inf if not np.isfinite(R[s]) else R[s])))

    out = HERE / "harmonia_min/state/reports/intro_vocal.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("%TABLE%", table_html(rows, D, wi))
                   .replace("%PUSH%", push)
                   .replace("%BODY%", body)
                   .replace("%L0%", str(PLOT_L))
                   .replace("%W%", str(round(PLOT_R - PLOT_L, 6))))
    print(f"écrit {out.relative_to(HERE)} ({out.stat().st_size // 1024} Ko)")


PAGE = """<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>L'intro, cherchée dans la voix</title><style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:#2f2a20}
.wrap{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}
h1{font:italic 600 25px Georgia,serif;margin:0 0 3px}
.lede{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:16px}
.lede b{color:#2f2a20}
section{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}
h2{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}
h3{font:700 15px system-ui;margin:16px 0 6px;color:#2f2a20}
.sub{font:500 12px system-ui;color:#8a8371;margin-left:8px;font-weight:500}
img{width:100%;border-radius:8px;display:block}
table{border-collapse:collapse;font-size:12.5px;margin-top:8px}
table.big{width:100%}
th,td{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}
th{background:#f7f3e9;font-size:11px}
tr.ok td{background:#e4f0e8} tr.no td{background:#faf4e6;color:#8a8371}
tr.hd td{background:#f2ece0;font-weight:700}
.plot{position:relative;margin-bottom:8px} .plot img{margin:0}
.cur{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}
.hit{position:absolute;top:0;bottom:0;cursor:crosshair}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:0 0 6px}
.pp{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}
.pos{font:600 12px ui-monospace,monospace;min-width:88px}
.hint{font:500 11px system-ui;color:#a89f8c;margin-right:6px}
button.blk{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 8px;cursor:pointer;font:700 11.5px system-ui;color:#4a4436}
button.blk small{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}
button.blk.on{color:#fff !important;background:#8a2b2b !important}
.warn{background:#faf0e0;border-left:3px solid #b3261e;border-radius:6px;
  padding:9px 12px;font-size:13px;margin:10px 0}
.key{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:5px}
</style></head><body><div class=wrap>
<h1>L'intro, cherchée dans la voix</h1>
<div class=lede>« Je pense que les indices sont dans le vocal. Des fois, entre
l'intro et le début de la chanson, c'est la même harmonie, mais c'est le vocal
qui change. » — Louis, 2026-08-07. Cette page prend l'idée au mot : une
trentaine de descripteurs pris sur la <b>piste vocale seule</b> (demucs), mesure
par mesure, et pour chacun <b>tout seul</b>, combien de morceaux sur dix il
place la frontière au bon endroit.<br><br>
<span class=key style="background:#8a2b2b"></span><b>rouge plein</b> : la
frontière de Louis. &nbsp;
<span class=key style="background:#1f8a5b"></span><b>vert tireté</b> : la règle
livrée aujourd'hui (1<sup>re</sup> mesure chantée, + 1 si la note arrive après
40 % de la mesure). &nbsp;
<span class=key style="background:#7c3aed"></span><b>violet pointillé</b> : ce
que dit le descripteur de la ligne.</div>

<section><h2>Le tableau, un critère à la fois</h2>
<div class=lede>Chaque descripteur devient une frontière par la même règle, et
une seule : la <b>marche</b> la plus nette dans les seize premières mesures —
l'écart entre la moyenne des huit mesures d'avant et celle des huit mesures
d'après, divisé par l'écart-type. Pas de réglage par descripteur, sinon le
tableau compare des réglages et non des descripteurs.<br><br>
<b>Le plancher est haut, et il faut le dire avant de lire une seule ligne.</b>
La vérité de Louis vaut 1, 4, 1, 0, 4, 4, 8, 8, 4, 2 mesures : répondre
« toujours 4 » sans rien écouter fait déjà 4/10. Et la règle livrée fait 9/10.
Un descripteur à 5/9 n'apporte donc rien du tout.<br><br>
La colonne <b>présence</b> est l'AUC de « ce morceau a-t-il une vraie intro
(≥ 4 mesures) ou pas (≤ 2) ». 0,50 = pile ou face.</div>
%TABLE%
<div class=warn><b>Verdict.</b> Le meilleur descripteur vocal seul fait
<b>6/9</b> — deux points sous la règle livrée. Aucun ne la bat, et les huit
premiers du tableau mesurent tous la même chose : <b>la voix entre</b> (énergie,
présence, vitesse du timbre). L'idée de Louis sur la ressemblance au reste du
morceau existe bien dans les données, mais comme <i>arbitre</i>, pas comme
détecteur — voir la section suivante.</div></section>

<section><h2>« Des fois il n'y a pas d'intro »</h2>
<div class=lede>Sa contrainte dure. Trois choses mesurées, dans l'ordre.<br><br>
<b>1. La règle livrée y répond déjà, gratuitement.</b> « Pas d'intro » veut dire
« on chante dès la mesure 1 » : la règle rend alors 0 et aucune section d'intro
n'est écrite. Sur les dix morceaux elle a raison <b>10 fois sur 10</b> sur la
seule question présence/absence.<br><br>
<b>2. Aucun descripteur vocal ne fait mieux, et on ne peut pas le savoir.</b>
Le corpus annoté contient <b>un seul</b> morceau sans intro (Grenade). Une
détection de présence validée sur un exemple négatif n'est pas validée. Le
substitut mesurable — séparer les intros longues (≥ 4 mesures, 6 morceaux) des
quasi-nulles (≤ 2 mesures, 4 morceaux) — donne au mieux une AUC de 0,96
(ressemblance de timbre aux 6 meilleures, sens inversé). <b>Test de
permutation</b> : sur une seule hypothèse p = 0,010, mais ce 0,96 est le
meilleur de 78 détecteurs, donc p corrigé ≤ <b>0,74</b>. C'est du bruit de
sélection, rien d'autre.<br><br>
<b>3. Le vrai danger n'est pas de rater une intro, c'est d'en inventer une.</b>
Tous les détecteurs à « marche » du tableau sont construits pour rendre une
mesure ≥ 1 : aucun ne peut dire « pas d'intro ». Les brancher tels quels
casserait Grenade à coup sûr. C'est la raison principale pour laquelle rien de
ce tableau n'est branché.</div></section>

<section><h2>L'idée de Louis, là où elle marche : arbitrer, pas détecter</h2>
<div class=lede>La règle livrée se trompe sur <b>un seul</b> morceau, The Walk,
où le chanteur chante quatre mesures d'intro. Sa phrase s'applique exactement
là : ces quatre mesures chantées <b>ne se reproduisent jamais</b>, alors que les
quatre suivantes sont un vrai couplet.<br><br>
On mesure donc le <b>rapport de reprise</b> : la ressemblance moyenne du bloc
contesté au reste du morceau, divisée par celle du bloc suivant. En dessous de 1,
le bloc contesté appartient moins à la chanson que celui d'après.<br><br>
Ce n'est pas ce qui avait été essayé et écarté. Là, on <i>glissait</i> le bloc et
on regardait ses pics — « aucun contraste sur The Walk, tous les blocs à
0,33–0,40 ». Un pic de glissement exige une reprise <b>alignée</b> ; la
ressemblance moyenne n'exige rien.</div>
<table class=big><tr><th>morceau</th><th>règle livrée</th><th>Louis</th>
<th>rapport de reprise</th></tr>%PUSH%</table>
<div class=warn><b>Ce qui est vrai, et ce qui ne l'est pas.</b><br>
Vrai : un seuil à 0,50 corrige The Walk et <b>ne touche à rien d'autre</b> —
10/10 sur le corpus annoté. Sur les <b>31 morceaux</b> du disque où le rapport
est calculable, The Walk est le <b>minimum absolu</b> (0,451) et le seul en
dessous de 0,50 ; si le rapport n'avait aucun rapport avec « c'est encore
l'intro », tomber au rang 1 sur 31 arrive une fois sur 31 (p ≈ 0,03).<br>
Pas vrai : que ce soit démontré. En <b>validation croisée</b> (le seuil réglé
sans le morceau testé) on retombe à <b>9/10</b>, exactement la règle livrée —
parce que The Walk est le seul exemple positif, donc les neuf autres ne peuvent
pas apprendre le seuil. Et la marge est mince : juste au-dessus de 0,451 il y a
un peloton à 0,531 / 0,569 / 0,572 / 0,582 dont on ignore la vérité. Le seuil
n'est <b>pas</b> posé dans un trou, contrairement au 0,40 de la levée
(0,31 → 0,48).<br>
<b>Donc : rien n'est branché.</b> Il faut un deuxième morceau annoté à intro
chantée pour trancher.</div></section>

<h3>Morceau par morceau</h3>
%BODY%
</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
const L0=%L0%, W=%W%;
let stopAt=null,onBtn=null,live=null,raf=null;
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
function clr(){ if(onBtn){onBtn.classList.remove("on");onBtn=null;} }
function draw(){
  if(!live) return;
  const G=live.G, n=G.length-1;
  let t=au.currentTime, f;
  if(t<=G[0]) f=0; else if(t>=G[n]) f=n; else {
    let lo=0,hi=n; while(hi-lo>1){const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}
    f=lo+(t-G[lo])/(G[lo+1]-G[lo]); }
  live.cur.style.display="block";
  live.cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
  live.pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime);
}
function tick(){ draw();
  if(stopAt!=null&&au.currentTime>=stopAt){au.pause();stopAt=null;clr();}
  if(!au.paused) raf=requestAnimationFrame(tick); }
au.addEventListener("play",()=>{ if(live) live.pp.textContent="❚❚"; tick(); });
au.addEventListener("pause",()=>{ if(live) live.pp.textContent="▶";
  cancelAnimationFrame(raf); draw(); });
function go(sec,t0,t1,btn){
  if(live && live.sec!==sec){ live.pp.textContent="▶"; live.cur.style.display="none"; }
  live=sec._p; clr(); stopAt=t1;
  if(btn){onBtn=btn;btn.classList.add("on");}
  if(au.getAttribute("src")!==sec.dataset.audio){
    au.setAttribute("src",sec.dataset.audio);au.load();}
  const seek=()=>{try{au.currentTime=t0;}catch(e){} draw();};
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{once:true});
  au.play().catch(()=>clr());
}
document.querySelectorAll("section[data-grid]").forEach(sec=>{
  const G=JSON.parse(sec.dataset.grid), n=G.length-1;
  const hit=sec.querySelector(".hit");
  sec._p={G:G,pos:sec.querySelector(".pos"),cur:sec.querySelector(".cur"),
          pp:sec.querySelector(".pp"),sec:sec};
  hit.style.left=(L0*100)+"%"; hit.style.width=(W*100)+"%";
  hit.onclick=e=>{ const r=hit.getBoundingClientRect();
    const f=n*(e.clientX-r.left)/r.width;
    const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    go(sec, G[i]+(f-i)*(G[i+1]-G[i]), null, null); };
  sec.querySelector(".pp").onclick=()=>{
    if(au.paused||live!==sec._p) go(sec,G[0],null,null); else au.pause(); };
  sec.querySelectorAll("[data-p]").forEach(b=>{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(sec,G[d[0]],G[Math.min(n,d[1])],b); });
});
</script></body></html>"""


if __name__ == "__main__":
    main(sys.argv[1:])
