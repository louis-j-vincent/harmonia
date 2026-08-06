"""Blocs de 8, avec la queue de 2 mesures en option — et une règle générale pour la placer.

    python scripts/blocks_flex.py [<stem> ...]  ->  /reports/blocks_flex.html

Louis, 2026-08-06, sur Don't Know Why :

  « La section A commence à la 4e/5e mesure, donc ça a été attrapé par une
    hypothèse. Maintenant la partie B commence à la mesure 14/15 — donc il faut
    bien choper ça, c'est les 2 mesures en plus qui finissent Norah, et il
    faudrait une règle générale qui les chope. »

LA RÈGLE GÉNÉRALE. Une section n'est pas « huit mesures », c'est **huit mesures,
plus éventuellement deux**. Le pavage ne fixe donc plus une longueur : à chaque
bloc, on a le CHOIX entre 8 et 10, et c'est la structure du morceau qui tranche.

Comment elle tranche : parmi tous les pavages possibles, on garde celui qui
**recouvre le plus de chanson avec le moins de blocs distincts**, exactement le
critère d'hier. Une queue de deux mesures ne s'ajoute donc que si elle FAIT
GAGNER — si, en décalant tout ce qui suit de deux mesures, les blocs suivants se
mettent à coïncider. Sur Norah, un A de 10 mesures fait tomber le B sur la
mesure 15 et tout le reste s'aligne ; un A de 8 le fait tomber sur la 13 et
plus rien ne coïncide. La règle n'a pas besoin de connaître Norah.

Ce qui est balayé : le départ (l'intro, guidée par le chant) × le sous-ensemble
de blocs qui prennent leur queue de deux mesures. On limite à trois queues par
morceau — au-delà, ce n'est plus une section de huit mesures avec une extension,
c'est autre chose, et il faudrait le dire autrement.

L'astérisque d'hier reste : deux blocs sont le même quand leurs SIX premières
mesures coïncident, les dernières sont libres.
"""
from __future__ import annotations

import itertools
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

BLOCK, TAIL = 8, 2
HEAD = 6           # les six premières mesures décident de l'identité
SAME = 0.90
MAX_TAILS = 3      # au-delà, ce n'est plus « huit plus deux »
DEFAULT = B8.DEFAULT


def head_matrix(S, n, head=HEAD):
    """H[p, q] = à quel point les `head` mesures en p et en q sont la même chose."""
    H = np.zeros((n, n))
    for p in range(n - head + 1):
        for q in range(p, n - head + 1):
            v = HS.diag_match(S, p, q, head)
            H[p, q] = H[q, p] = v
    return H


def evaluate(H, n, starts, lens):
    """Groupe les blocs par leur tête, et note le pavage.

    **Une queue sur le DERNIER bloc ne compte pas.** Louis, 2026-08-06, a
    verrouillé Bein Green au départ mesure 5. Les deux meilleurs pavages y
    étaient :

        départ 3   I2 A8 A8 B8 A8 B8 A′10     96 %, 1 queue
        départ 5   I4 A8 A8 B8 A8 B8 A8       92 %, 0 queue   <- le bon

    Le départ 3 ne gagne ses quatre points qu'en raccourcissant l'intro de deux
    mesures, et sa queue ne sert qu'à ravaler les deux mesures que ce
    raccourcissement laisse à la fin. Elle ne fait rien pour la musique.

    D'où la règle, qui n'est pas un réglage : **une queue de deux mesures ne se
    justifie que si elle DÉCALE ce qui suit**. Sur Norah, le A de dix mesures
    fait tomber tout le reste du morceau en place — elle est méritée. Sur le
    dernier bloc d'un morceau il n'y a rien après à décaler : la queue ne peut
    donc rien mériter, et ses mesures ne sont pas comptées comme expliquées.
    """
    k = len(starts)
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(k):
        for j in range(i + 1, k):
            if H[starts[i], starts[j]] >= SAME:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    covered = 0
    for i in range(k):
        if len(groups[find(i)]) < 2:
            continue
        L = lens[i]
        if L > BLOCK and i == k - 1:      # queue finale : rien à décaler après
            L = BLOCK
        covered += L
    distinct = sum(1 for g in groups.values() if len(g) >= 2)
    # LE COÛT D'ÉCRITURE : combien de mesures il faut écrire pour rendre le
    # morceau. Chaque bloc distinct s'écrit une fois, à sa longueur ; l'intro et
    # les restes s'écrivent en entier puisque rien ne les explique ; et chaque
    # passage coûte une ligne de plus sur la grille. Minimiser ça, c'est
    # exactement « le meilleur recouvrement avec des blocs minimaux » — les deux
    # moitiés de la phrase de Louis dans un seul nombre, au lieu de deux clés
    # dont l'une écrase l'autre.
    # LE DÉBUT ET LA FIN NE DOIVENT PAS RAPPORTER DE POINTS. Louis,
    # 2026-08-06 : « on ne devrait pas gagner de points de couverture sur le
    # début ou la fin ». Ce qui donnait ces points, c'était le RATIO de
    # couverture (mesures expliquées / mesures du morceau) : une hypothèse qui
    # commence tôt a une intro courte, donc un dénominateur plus favorable, et
    # gagne sans rien expliquer de plus. Le ratio ne décide donc plus rien — il
    # n'est plus qu'affiché, et calculé sur le CORPS seul, entre le premier et
    # le dernier bloc.
    #
    # Ce qui décide est le coût d'écriture, où l'intro et le reste comptent
    # leurs mesures comme tout le monde : ce sont des mesures qu'il faudra bien
    # écrire puisque rien ne les explique. Deux variantes essayées et fausses,
    # notées pour ne pas y revenir :
    #   * intro ET reste retirés du coût — le pavage le moins cher devient celui
    #     qui explique le moins, deux blocs et vingt-deux mesures abandonnées ;
    #   * intro seule retirée — y jeter douze mesures devient gratuit, et un
    #     départ mesure 13 gagne parce qu'il a une ligne de passage en moins.
    body = starts[-1] + lens[-1] - starts[0]
    written = starts[0] + (n - (starts[-1] + lens[-1]))  # intro + reste final
    for g in groups.values():
        written += max(lens[i] for i in g)               # une fois par bloc distinct
    cost = written + len(starts)                         # une ligne par passage
    return (covered / body if body else 0.0), distinct, parent, groups, cost


def tilings(n, start, block=BLOCK, tail=TAIL, max_tails=MAX_TAILS):
    """Tous les pavages : chaque bloc prend, ou non, sa queue de deux mesures."""
    room = n - start
    kmax = room // block
    out = []
    for k in range(max(1, kmax - 1), kmax + 1):     # le pavage doit aller au bout
        base = k * block
        if base > room:
            continue
        extra = (room - base) // tail
        for t in range(0, min(max_tails, k, extra) + 1):
            for combo in itertools.combinations(range(k), t):
                lens = [block + (tail if i in combo else 0) for i in range(k)]
                if sum(lens) > room:
                    continue
                starts, p = [], start
                for L in lens:
                    starts.append(p)
                    p += L
                out.append((starts, lens))
    return out


def best_tiling(S, H, n, start):
    best = None
    for starts, lens in tilings(n, start):
        cov, dist, parent, groups, cost = evaluate(H, n, starts, lens)
        # Le coût d'écriture d'abord, puis le moins de queues, puis LE MOINS DE
        # RESTE À LA FIN, puis la plus courte intro.
        #
        # La clé du reste final est celle qui verrouille Bein Green sur le
        # départ mesure 5, comme Louis l'a tranché à l'oreille. Aux départs 3 et
        # 5, le pavage est le MÊME (A8 A8 B8 A8 B8 A8) et coûte le même prix,
        # 26 mesures écrites : le départ 3 le pose deux mesures plus tôt et
        # laisse donc deux mesures orphelines à la fin, le départ 5 tombe pile.
        # Un morceau finit où il finit ; un pavage qui s'arrête deux mesures
        # avant la fin laisse une queue que rien n'explique, alors que les deux
        # mesures de plus au début sont franchement une intro.
        rest = n - (starts[-1] + lens[-1])
        score = (-cost, -sum(1 for L in lens if L > BLOCK), -rest, -start)
        if best is None or score > best[0]:
            best = (score, starts, lens, parent, groups, cov, dist, cost)
    return best


def to_sections(n, start, starts, lens, parent, groups):
    """Les blocs deviennent des sections.

    **Une queue est TOUJOURS à la fin d'une section, jamais au début.** Louis,
    2026-08-06. C'est vrai par construction — un bloc qui prend sa queue
    s'allonge vers l'AVANT, `lens[i] = 8 + 2`, donc les deux mesures
    supplémentaires sont ses deux dernières et le bloc suivant démarre deux
    mesures plus loin. Rien dans le modèle ne peut coller deux mesures au début
    d'une section.

    Mais « vrai par construction » est exactement le genre d'affirmation qui
    cesse d'être vraie sans prévenir, alors la fonction le VÉRIFIE : chaque
    section commence là où la précédente s'arrête, et une section marquée ′ fait
    bien 10 mesures. Si un jour ce n'est plus le cas, ça lève ici au lieu de
    produire un chart faux en silence.
    """
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    letters, out = {}, []
    if start > 0:
        out.append({"b0": 0, "b1": start - 1, "kind": "intro", "letter": "intro"})
    for i, (s, L) in enumerate(zip(starts, lens)):
        r = find(i)
        if r not in letters:
            letters[r] = chr(ord("A") + len(letters))
        out.append({"b0": s, "b1": min(n - 1, s + L - 1), "kind": "bloc",
                    "letter": letters[r] + ("′" if L > BLOCK else ""),
                    "rep": len(groups[r]), "queue": L > BLOCK})
    end = starts[-1] + lens[-1]
    if end < n:
        out.append({"b0": end, "b1": n - 1, "kind": "reste", "letter": "?"})
    for a, b in zip(out, out[1:]):          # aucune section ne saute ni ne recouvre
        assert b["b0"] == a["b1"] + 1, f"trou/chevauchement entre {a} et {b}"
    for sec in out:
        if sec["kind"] == "bloc" and sec.get("queue"):
            assert sec["b1"] - sec["b0"] + 1 == BLOCK + TAIL, \
                f"queue mal placée : {sec}"
    return out


def song(stem):
    S, n, grid = load(stem)
    H = head_matrix(S, n)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, t, rms, f0, sing = B8.sing_onset(voc)
    b_sing = VA.bar_of(grid, onset)

    cands = sorted({s for s in range(0, min(n - BLOCK, 14), 2)}
                   | ({b_sing - (b_sing % 2) + d for d in (-4, -2, 0, 2)}
                      if b_sing is not None else set()))
    cands = [c for c in cands if 0 <= c <= n - BLOCK]

    hyps = []
    for c in cands:
        b = best_tiling(S, H, n, c)
        if not b:
            continue
        score, starts, lens, parent, groups, cov, dist, cost = b
        secs = to_sections(n, c, starts, lens, parent, groups)
        rigid = B8.tile(S, n, c)
        hyps.append({"start": c, "score": score, "cov": cov, "dist": dist,
                     "cost": cost,
                     "secs": secs, "lens": lens,
                     "tails": sum(1 for L in lens if L > BLOCK),
                     "rigid": rigid})
    hyps.sort(key=lambda h: h["score"], reverse=True)
    best = hyps[0]
    shown = hyps[:5]

    heights = [2.4, 0.7, 0.7] + [0.55] * len(shown) + [0.55]
    Hh = sum(heights) + 1.3
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .24})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .32 / Hh, bottom=.58 / Hh)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)
    g = np.asarray(grid, float)
    tb = np.interp(t, g, np.arange(len(g)))
    axs[1].fill_between(tb, rms / (rms.max() or 1), color="#8a2b2b", alpha=.30, lw=0)
    axs[1].set_ylim(0, 1.05); axs[1].set_yticks([])
    axs[1].set_ylabel("voix", fontsize=6.6, rotation=0, ha="right", va="center",
                      color="#8a2b2b")
    axs[2].plot(tb, f0, ".", ms=1.0, color="#8a8371")
    axs[2].plot(tb[sing], f0[sing], ".", ms=1.7, color="#0f766e")
    axs[2].set_yscale("log"); axs[2].set_yticks([])
    axs[2].set_ylabel("chant", fontsize=6.6, rotation=0, ha="right", va="center",
                      color="#0f766e")
    for a in (axs[1], axs[2]):
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        if b_sing is not None:
            a.axvline(b_sing, color="#0f766e", lw=1.6)
    for i, h in enumerate(shown):
        B8.strip(axs[3 + i], h["secs"], n,
                 f"départ {h['start']+1}\n{h['cov']:.0%} · {h['dist']} blocs"
                 + (f" · {h['tails']} queue(s)" if h["tails"] else ""),
                 win=(h is best))
        axs[3 + i].axvline(h["start"], color="#111", lw=1.6)
    B8.strip(axs[-1], best["rigid"]["secs"], n,
             f"rigide 8\n{best['rigid']['couverture']:.0%}")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    def fmt(secs):
        return " ".join(
            ("intro" if s["kind"] == "intro" else
             "?" if s["kind"] == "reste" else s["letter"])
            + f"[{s['b0']+1}-{s['b1']+1}]" for s in secs)
    rows = "".join(
        f"<tr class='{'win' if h is best else ''}'><td>mes. {h['start']+1}</td>"
        f"<td>{h['cov']:.0%}</td><td>{h['dist']}</td><td>{h['tails']}</td>"
        f"<td class=f>{fmt(h['secs'])}</td></tr>" for h in hyps)
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>"
        f"{'intro' if s['kind']=='intro' else ('?' if s['kind']=='reste' else s['letter'])}"
        f"<small>{s['b0']+1}</small></button>" for s in best["secs"])
    lens_txt = " + ".join(str(L) for L in best["lens"])
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
chant mes. {b_sing+1 if b_sing is not None else '—'} ·
retenue : départ mes. {best['start']+1}, blocs {lens_txt}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>les boutons jouent les blocs retenus</span>{btns}</div>
<table><tr><th>départ</th><th>couverture</th><th>blocs</th><th>queues</th>
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
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    out = HERE / "harmonia_min/state/reports/blocks_flex.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Huit mesures, plus deux</title><style>
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
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}} td.f{{font:500 11px ui-monospace,monospace}}
tr.win td{{background:#e4f0e8;font-weight:700}}
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
<h1>Huit mesures, plus deux</h1>
<div class=lede>Une section n'est pas « huit mesures » : c'est <b>huit mesures,
plus éventuellement deux</b>. Le pavage ne fixe donc plus une longueur — à chaque
bloc il y a le choix entre 8 et 10, et c'est la structure du morceau qui
tranche.<br><br>
<b>Comment elle tranche</b> : parmi tous les pavages possibles, on garde celui
qui recouvre le plus de chanson avec le moins de blocs distincts. Une queue de
deux mesures ne s'ajoute donc <b>que si elle fait gagner</b> — si, en décalant
tout ce qui suit de deux mesures, les blocs suivants se mettent à coïncider. Sur
Norah, un A de dix mesures fait tomber le B sur la mesure 15 et tout s'aligne ;
un A de huit le fait tomber sur la treize et plus rien ne coïncide. La règle n'a
pas besoin de connaître Norah.<br><br>
Les blocs qui prennent leur queue sont notés <b>A′</b>. On en autorise trois au
plus par morceau : au-delà, ce n'est plus une section de huit avec une
extension, c'est autre chose, et il faudrait le dire autrement. L'astérisque
reste : deux blocs sont le même quand leurs <b>six premières mesures</b>
coïncident. La dernière bande de chaque figure est le pavage <b>rigide</b> de la
page précédente, pour voir ce que la queue apporte.</div>
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
