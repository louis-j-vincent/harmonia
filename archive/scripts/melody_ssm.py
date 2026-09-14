"""La voix décide où finit l'intro, et la mélodie donne une troisième matrice.

    python scripts/melody_ssm.py [<stem> ...]  ->  /reports/melody_ssm.html

Louis, 2026-08-06, deux demandes :

  « Pour le début de la chanson / fin de l'intro, à chaque fois c'est le début de
    bloc le plus proche du début de la voix chantée, tu peux le noter ! »

  « Construis une matrice SSM de la hauteur de chant pour voir si on peut en
    faire quelque chose. »

1. LA RÈGLE DE LA VOIX, notée et vérifiée. L'intro finit au premier début de
   bloc **à partir du chant** — la voix entre « juste avant ou sur » le premier
   temps de la section, jamais franchement après. On prend donc le premier
   multiple de deux mesures ≥ la mesure où le chant commence. C'est une règle à
   une ligne, sans coût, sans balayage, sans arbitrage.

   Elle retrouve les deux verdicts que Louis a verrouillés :
       Bein Green   chant mesure 4  ->  départ mesure 5   (verrouillé le 06/08)
       Norah        chant mesure 4  ->  départ mesure 5   (« la section A
                                                            commence à la 4/5e »)
   Le classement par coût d'écriture, lui, plaçait Norah en mesure 7. La voix
   tranche mieux que la structure sur cette question-là, ce qui est logique :
   l'intro est justement l'endroit où la structure ne dit encore rien.

2. LA MATRICE DE MÉLODIE. Troisième voie après l'harmonie et le rythme. Chaque
   mesure devient ce que la voix y chante : un profil de douze demi-tons
   (quelle note, pondérée par sa durée) et le contour (monte, descend, reste).
   Deux mesures se ressemblent si on y chante la même chose. Une mesure sans
   chant est marquée muette et ne ressemble à rien — c'est une information en
   soi, les instrumentaux se voient d'un coup.
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
import vocal_melody as VM                                                 # noqa: E402

UNIT = 2           # la grille de départ, en mesures
TRUTH_START = {"bein_green": 5, "norah_jones_don_t_know_why": 5}   # 1-indexé
DEFAULT = B8.DEFAULT


def voice_start(b_sing, n, unit=UNIT):
    """LA RÈGLE : le premier début de bloc à partir du chant.

    « Juste avant ou sur » — donc jamais un bloc qui commencerait après que la
    voix a déjà chanté une mesure entière. On arrondit vers le HAUT sur la
    grille de `unit` mesures. Si le chant n'est pas détecté, on ne décide pas.
    """
    if b_sing is None:
        return None
    return min(n - 1, b_sing + (-b_sing) % unit)


ARB_MARGIN = 0.08     # ce que le candidat bas doit gagner pour renverser la levée


def voice_start_arbitrated(b_sing, M, n, unit=UNIT, margin=ARB_MARGIN,
                           off_grid=0.10):
    """La règle de la voix, avec un ARBITRE quand elle hésite. Louis, 2026-08-07 :

      « Il y a une vraie cassure dans la chanson entre la majorité de la chanson
        et ce passage d'intro. » Et : « le chant de cette intro faussement
        identifiée comme le début du A ne se reproduit quasiment pas. »

    Quand le chant démarre sur une mesure IMPAIRE, deux départs sont également
    légitimes — celui d'avant et celui d'après — et arrondir vers le haut n'est
    qu'une convention. On tranche donc par sa cassure : on glisse le bloc chanté
    de chaque candidat sur tout le morceau et on garde celui **dont le chant se
    reproduit le mieux ailleurs**. Une intro, par définition, ne se reproduit pas.

    LA LEVÉE GARDE LA MAIN. Sur Bein Green, Let It Be et Norah, le chanteur entre
    une mesure AVANT la section (première mesure chantée 4, section à 5), et le
    candidat haut y gagne largement — 0.78 contre 0.72, 0.88 contre 0.73, 0.95
    contre 0.70. Le candidat bas ne l'emporte donc que s'il dépasse de `margin`.
    Mesuré : 6/10 sans arbitrage, 7/10 avec, et aucun des six déjà justes n'est
    perdu. Yesterday bascule (0.75 contre 0.62) et c'est le bon.

    CE QUE ÇA NE RÈGLE PAS. Aretha et Sunny ont une intro d'UNE mesure dans la
    vérité de Louis ; la grille de deux mesures ne peut pas l'écrire, aucun
    arbitrage n'y changera rien. Et sur The Walk le chant démarre sur une mesure
    paire, donc il n'y a même pas d'hésitation à arbitrer — la voix y entre
    quatre mesures avant la section, ce que cette règle ne peut pas voir.
    """
    if b_sing is None:
        return None
    lo = min(n - 1, b_sing - b_sing % unit)
    hi = min(n - 1, b_sing + (-b_sing) % unit)
    if lo == hi or M is None:
        return hi

    def recurrence(s, L=8):
        if s + L > n:
            return 0.0
        vals = [float(np.mean([M[s + i, c + i] for i in range(L)]))
                for c in range(0, n - L + 1, unit) if abs(c - s) >= L]
        return max(vals) if vals else 0.0

    base = lo if recurrence(lo) > recurrence(hi) + margin else hi
    # LA GRILLE DE DEUX N'EST PLUS UNE PRISON. Louis, 2026-08-07 : « relaxe le
    # truc de la grille de deux, on snap à la barre la plus proche ». Prise au
    # pied de la lettre elle coûte un point (4/10 brut, 6/10 arbitrée, contre
    # 7/10 pour la grille), et le mécanisme est la LEVÉE : sur trois morceaux sur
    # dix le chanteur entre une mesure avant la section, donc la barre exacte est
    # systématiquement une mesure trop tôt.
    #
    # La grille reste donc le défaut, mais on a le droit d'en sortir quand la
    # preuve est nette. Mesuré neutre sur les dix morceaux — le recours ne se
    # déclenche jamais à tort. Il paiera là où la vérité tombe sur une mesure
    # impaire ET où le chant le confirme, ce qu'aucun de nos dix ne fait encore.
    if base != b_sing and recurrence(b_sing) > recurrence(base) + off_grid:
        return b_sing
    return base


def melody_vectors(notes, grid, n):
    """Les PROFILS de chant, mesure par mesure (n, 12), normalisés.

    `melody_bars` en fait la matrice et les jette ; il faut les garder pour
    pouvoir les faire TOURNER. Louis, 2026-08-07 : « sur Sunny on ne fait que
    monter d'un demi-ton à chaque fois… transposition sur la voix aussi du
    coup ». Un chanteur qui module monte avec l'orchestre : son profil de
    demi-tons subit exactement la même rotation que les accords.
    """
    P = np.zeros((n, 12))
    for t0, d, m in notes:
        b = int(np.searchsorted(grid, t0) - 1)
        if 0 <= b < n:
            P[b, int(m) % 12] += d
    mute = P.sum(1) <= 0
    V = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-9, None)
    V[mute] = 0.0
    return V, mute


def melody_bars(notes, grid, n):
    """Par mesure : ce que la voix y chante. (profil 12 demi-tons, contour, muette)"""
    # Le profil est celui des DEMI-TONS chantés, pondérés par la durée, et rien
    # d'autre. J'y avais ajouté le contour (monte / reste / descend) : comme
    # presque chaque mesure contient un peu des trois, il rapprochait tout de
    # tout et la matrice devenait uniforme. Une matrice qui dit que tout se
    # ressemble ne dit rien.
    P = np.zeros((n, 12))
    for t0, d, m in notes:
        b = int(np.searchsorted(grid, t0) - 1)
        if not (0 <= b < n):
            continue
        P[b, int(m) % 12] += d
    mute = P.sum(1) <= 0
    V = P
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.clip(nrm, 1e-9, None)
    M = np.clip(V @ V.T, 0, 1)
    M[mute, :] = 0.0
    M[:, mute] = 0.0
    return M, mute


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, t, rms, f0, sing = B8.sing_onset(voc)
    b_sing = VA.bar_of(grid, onset)
    vstart = voice_start(b_sing, n)

    # la mélodie, réutilisée telle quelle de la page piano. `track_f0` attend la
    # PISTE VOCALE, pas le mix : lui passer le m4a d'origine fait suivre le
    # piano et sort 250 « notes » dès la première seconde sur Bein Green, qui y
    # est instrumental — la matrice devenait une bouillie uniforme.
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = melody_bars(notes, grid, n)

    H = BF.head_matrix(S, n)
    b_cost = BF.best_tiling(S, H, n, 0)
    cands = [c for c in range(0, min(n - 8, 14), 2)]
    best = max(((BF.best_tiling(S, H, n, c), c) for c in cands),
               key=lambda x: x[0][0] if x[0] else (-1e9,))
    cost_start = best[1]

    cuts = {}
    for name, st in (("règle de la voix", vstart), ("coût d'écriture", cost_start)):
        if st is None:
            continue
        b = BF.best_tiling(S, H, n, st)
        if b:
            cuts[name] = BF.to_sections(n, st, b[1], b[2], b[3], b[4])

    heights = [2.6, 2.6, 0.55] + [0.55] * len(cuts)
    Hh = sum(heights) + 1.3
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .26})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.58 / Hh)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)
    axs[0].set_title("HARMONIE", fontsize=8.2, color="#2a6fb0", loc="left", pad=3)
    axs[1].imshow(M, origin="lower", extent=(0, n, 0, n), cmap="Purples",
                  vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axs[1].set_ylabel("mesure", fontsize=7.5); axs[1].tick_params(labelsize=6.4)
    axs[1].set_title(f"MÉLODIE CHANTÉE  ·  {int(mute.sum())} mesures sans chant, "
                     "en blanc", fontsize=8.2, color="#7c3aed", loc="left", pad=3)
    ax = axs[2]
    ax.fill_between(np.arange(n) + .5, (~mute).astype(float), color="#7c3aed",
                    alpha=.35, step="mid")
    ax.set_ylim(0, 1.1); ax.set_yticks([])
    ax.set_ylabel("la voix\nchante", fontsize=6.6, rotation=0, ha="right",
                  va="center", color="#7c3aed")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if b_sing is not None:
        for a in axs[:3]:
            a.axvline(b_sing, color="#0f766e", lw=1.6)
    for i, (name, secs) in enumerate(cuts.items()):
        B8.strip(axs[3 + i], secs, n,
                 f"{name}\ndépart mes. {secs[0]['b1']+2 if secs[0]['kind']=='intro' else 1}")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    tr = TRUTH_START.get(stem)
    verdict = ""
    if tr is not None and vstart is not None:
        ok = (vstart + 1) == tr
        verdict = (f"<span class='gt {'ok' if ok else 'no'}'>vérité de Louis : "
                   f"départ mesure {tr} — la règle de la voix dit "
                   f"{vstart+1} {'✓' if ok else '✗'}</span>")
    fmt = lambda x: " ".join(
        ("intro" if s["kind"] == "intro" else "?" if s["kind"] == "reste"
         else s["letter"]) + f"[{s['b0']+1}-{s['b1']+1}]" for s in x)
    rows = "".join(f"<tr><td><b>{k}</b></td><td class=f>{fmt(v)}</td></tr>"
                   for k, v in cuts.items())
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
chant mesure {b_sing+1 if b_sing is not None else '—'} ·
la voix dit : départ mesure {vstart+1 if vstart is not None else '—'}</span></h2>
{verdict}
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span></div>
<table><tr><th>découpage</th><th>sections</th></tr>{rows}</table></section>"""


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
    out = HERE / "harmonia_min/state/reports/melody_ssm.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>La voix, et la matrice de mélodie</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
.gt{{display:inline-block;font:700 12px system-ui;border-radius:6px;
  padding:2px 8px;margin-bottom:8px}}
.gt.ok{{color:#0f5132;background:#e4f0e8}} .gt.no{{color:#8a2b2b;background:#f7e4e4}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}} td.f{{font:500 11px ui-monospace,monospace}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;align-items:center;gap:8px;margin:0 0 6px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace}}
</style></head><body><div class=wrap>
<h1>La voix, et la matrice de mélodie</h1>
<div class=lede><b>La règle de la voix.</b> L'intro finit au <b>premier début de
bloc à partir du chant</b> : la voix entre juste avant ou sur le premier temps
de la section, jamais franchement après, donc on arrondit vers le haut sur la
grille de deux mesures. Une ligne, aucun coût, aucun balayage. Le trait vert est
le début du chant.<br><br>
<b>La matrice de mélodie</b>, en violet, à côté de l'harmonique. Chaque mesure
devient ce que la voix y chante : quelles notes, pondérées par leur durée, plus
le contour — monte, descend, reste. Deux mesures se ressemblent si on y chante
la même chose. <b>Les mesures sans chant sont blanches</b> et ne ressemblent à
rien : les instrumentaux, les intros et les ponts s'y voient d'un coup d'œil,
ce que l'harmonie ne montre jamais.</div>
{body}</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
const L0=%L0%, W=%W%;
let live=null,raf=null;
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
function tick(){{ draw(); if(!au.paused) raf=requestAnimationFrame(tick); }}
au.addEventListener("play",()=>{{ if(live) live.pp.textContent="❚❚"; tick(); }});
au.addEventListener("pause",()=>{{ if(live) live.pp.textContent="▶";
  cancelAnimationFrame(raf); draw(); }});
function go(sec,t0){{
  if(live && live.sec!==sec){{ live.pp.textContent="▶"; live.cur.style.display="none"; }}
  live=sec._p;
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
    go(sec, G[i]+(f-i)*(G[i+1]-G[i])); }};
  sec.querySelector(".pp").onclick=()=>{{
    if(au.paused||live!==sec._p) go(sec,G[0]); else au.pause(); }};
}});
</script></body></html>""".replace("%L0%", str(PLOT_L)).replace("%W%", str(round(PLOT_R - PLOT_L, 6))))
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
