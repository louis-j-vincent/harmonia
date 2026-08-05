"""Arbitrer les frontières avec le rythme, jugé sur les morceaux dont on connaît la vérité.

    python scripts/rhythm_arbitrage.py [<stem> ...]
      -> /reports/rhythm_arbitrage.html

Louis, 2026-08-05 : « go avec binaire oui, mais ce n'est toujours pas la bonne
technique. Je te laisse regarder si on peut arbitrer avec la SSM rythmique, je
te laisse tester tout ça. Tu connais la vérité sur Norah Jones et Maroon 5 This
Love, prends-les et sers-t'en pour arbitrer les choix sur le reste. »

LA VÉRITÉ, donnée par lui plus tôt dans la journée :

    Norah Jones, Don't Know Why      2 sections
    Maroon 5, This Love              3 sections  (son B et son D sont la même)
    Mayer Hawthorne, The Walk        2 sections  (A = Bm7, puis C#m7 Bm7)

Ces trois nombres sont les seuls juges de cette page. Tout le reste — quelle
matrice, combien de groupes, faut-il recaler les frontières sur le rythme — est
choisi par eux, puis appliqué tel quel aux autres morceaux. C'est la seule
manière honnête de trancher entre des réglages qui se valent tous à l'œil.

CE QUI EST BALAYÉ :

  * la matrice  : par mesure continue / par passage binaire / par passage continue
  * les groupes : k = 2 à 6
  * le rythme   : frontières laissées telles quelles, ou RECALÉES sur le pic de
                  nouveauté rythmique le plus proche (± 2 mesures)

Le recalage rythmique est volontairement modeste, comme l'a recommandé la revue
de littérature : il ne CRÉE jamais de frontière, il ne fait que déplacer une
frontière déjà décidée par l'harmonie vers l'endroit où la batterie change. La
nouveauté rythmique est la courbe de Foote (Foote 2000, "Automatic audio
segmentation using a measure of audio novelty",
https://doi.org/10.1109/ICME.2000.869637) lue sur la SSM rythmique agrégée à
deux mesures — l'échelle que la première étude a trouvée utilisable.
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
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402
import spectral_sections as SP                                            # noqa: E402
from rhythm_vs_harmony import bar_patches                                 # noqa: E402

TRUTH = {"norah_jones_don_t_know_why": 2,
         "maroon_5_this_love": 3,
         "mayer_hawthorne_the_walk": 2}
MATRICES = ["par mesure", "passage binaire", "passage continu"]
KS = [2, 3, 4, 5, 6]
SNAP_BARS = 2          # de combien de mesures on s'autorise à bouger une frontière
FOOTE_M = 8            # la largeur du noyau, en mesures
OTHERS = ["bein_green", "let_it_be_remastered_2009",
          "bruno_mars_grenade_official_music_video",
          "maroon_5_she_will_be_loved_official_music_video",
          "the_police_every_breath_you_take_official_music_video"]


# ── le rythme ───────────────────────────────────────────────────────────────
def rhythm_novelty(stem, grid, n, agg=2, M=FOOTE_M):
    """Courbe de nouveauté rythmique, par mesure. 0 si le morceau n'a pas de batterie."""
    P, how = bar_patches(stem, grid)
    # agrégation à `agg` mesures, CONCATÉNÉES (pas moyennées) — c'est l'échelle
    # que la première étude a trouvée lisible ; à une mesure c'est du bruit
    m = n // agg
    V = np.array([P[i * agg:(i + 1) * agg].reshape(-1) for i in range(m)])
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    R = np.clip(V @ V.T, 0, 1)
    k = max(2, M // agg)
    K = np.outer(np.sign(np.arange(-k, k) + .5), np.sign(np.arange(-k, k) + .5))
    g = np.exp(-.5 * (np.arange(-k, k) + .5) ** 2 / (k / 2.0) ** 2)
    K = K * np.outer(g, g)
    nov = np.zeros(m)
    for i in range(m):
        a, b = i - k, i + k
        if a < 0 or b > m:
            continue
        nov[i] = float((R[a:b, a:b] * K).sum())
    nov = np.clip(nov, 0, None)
    if nov.max() > 0:
        nov = nov / nov.max()
    out = np.zeros(n)                       # remis à l'échelle de la mesure
    for i in range(m):
        out[i * agg:(i + 1) * agg] = nov[i]
    return out, how, R


def snap(secs, nov, n, tol=SNAP_BARS):
    """Recale chaque frontière sur le pic de nouveauté rythmique le plus proche.

    Ne crée jamais de frontière, n'en supprime jamais : elle bouge d'au plus
    `tol` mesures, et seulement s'il y a un pic strictement meilleur là-bas.
    """
    if nov.max() <= 0:
        return secs
    peaks = [b for b in range(1, n - 1)
             if nov[b] >= nov[b - 1] and nov[b] >= nov[b + 1] and nov[b] > 0]
    if not peaks:
        return secs
    out = [dict(s) for s in secs]
    for i in range(1, len(out)):
        b = out[i]["b0"]
        near = [p for p in peaks if abs(p - b) <= tol]
        if not near:
            continue
        p = max(near, key=lambda q: (nov[q], -abs(q - b)))
        if p == b or p <= out[i - 1]["b0"] or p >= out[i]["b1"]:
            continue
        out[i]["b0"] = p
        out[i - 1]["b1"] = p - 1
    return [s for s in out if s["b1"] >= s["b0"]]


# ── les configurations ──────────────────────────────────────────────────────
def cuts_for(stem):
    """Toutes les configurations d'un morceau -> {(matrice, k, recalage): sections}."""
    S, n, grid = load(stem)
    c0, c1 = SP.core_range(S)
    cells = HY.build_hypo(S, n, unit=2, start=c0)
    chain = HY.merge_two_to_four(HY.chain_of(cells, n))[0]
    B, C = SP.passage_matrices(S, chain)
    nov, how, _ = rhythm_novelty(stem, grid, n)

    sources = {
        "par mesure": (SP.affinity(S[c0:c1 + 1, c0:c1 + 1]), None, c0),
        "passage binaire": (SP.affinity(B), chain, 0),
        "passage continu": (SP.affinity(C), chain, 0),
    }
    out = {}
    for name, (A, ch, off) in sources.items():
        V, _ = SP.embed(A)
        for k in KS:
            lab = SP.cluster(V, k)
            if ch is None:
                secs, _ = SP.sections_of(lab, len(A))
                secs = [{**x, "b0": x["b0"] + off, "b1": x["b1"] + off} for x in secs]
            else:
                secs, _ = SP.sections_of(SP.spread_to_bars(lab, ch, n), n)
            out[(name, k, False)] = secs
            out[(name, k, True)] = snap(secs, nov, n)
    return S, n, grid, nov, how, out


def n_letters(secs):
    return len({s["letter"] for s in secs})


# ── la figure ───────────────────────────────────────────────────────────────
def strip(ax, secs, n, label, win=False):
    seen = {}
    for s in secs:
        seen.setdefault(s["letter"], COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   color=seen[s["letter"]]))
        if s["b1"] - s["b0"] >= 1:
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                    va="center", color="#fff", fontsize=7.2, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.2)
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel(label, fontsize=6.8, rotation=0, ha="right", va="center",
                  color="#1f8a5b" if win else INK)


def figure(S, n, nov, rows, old):
    heights = [3.2, 0.9] + [0.55] * len(rows) + [0.55]
    H = sum(heights) + 1.2
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .22})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .30 / H, bottom=.55 / H)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)

    axs[1].fill_between(np.arange(n) + .5, nov, color="#0f766e", alpha=.35)
    axs[1].plot(np.arange(n) + .5, nov, color="#0f766e", lw=1.1)
    axs[1].set_ylim(0, 1.05); axs[1].set_yticks([])
    axs[1].set_ylabel("nouveauté\nrythmique", fontsize=6.8, rotation=0,
                      ha="right", va="center", color="#0f766e")
    for sp in ("top", "right", "left"):
        axs[1].spines[sp].set_visible(False)

    for i, (lab, secs, win) in enumerate(rows):
        strip(axs[2 + i], secs, n, lab, win)
    strip(axs[-1], old, n, "règle\nactuelle")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    return fig2b64_fixed(fig)


def main():
    stems = sys.argv[1:] or (list(TRUTH) + OTHERS)
    data = {}
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            data[st] = cuts_for(st)
            print(f"  ok {st}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {st} — {type(exc).__name__}: {exc}")

    # ── l'arbitrage : quelles configurations respectent la vérité ? ─────────
    judged = [s for s in TRUTH if s in data]
    scores = {}
    for name in MATRICES:
        for k in KS:
            for sn in (False, True):
                hits = sum(1 for st in judged
                           if n_letters(data[st][5][(name, k, sn)]) == TRUTH[st])
                err = sum(abs(n_letters(data[st][5][(name, k, sn)]) - TRUTH[st])
                          for st in judged)
                scores[(name, k, sn)] = (hits, -err)
    best = max(scores, key=scores.get)
    print(f"\n  vérité respectée par : {best}  {scores[best]}")

    # CE QUE L'ARBITRAGE MONTRE VRAIMENT, et ce n'est pas ce que j'espérais :
    # le nombre de lettres suit k presque mécaniquement (k=2 -> 2 lettres,
    # k=3 -> 3), donc la vérité ne départage QUE k, pas la matrice. Et aucun k
    # ne convient aux trois : Norah veut 2, This Love 3, The Walk 2. k ne peut
    # donc pas être une constante. On fixe donc k par morceau là où on connaît
    # la vérité, et la question qui reste — la seule que ces trois morceaux
    # peuvent trancher — devient : le recalage rythmique met-il les frontières
    # au bon endroit ?
    MAT = "passage continu"
    kd = {st: TRUTH.get(st, 3) for st in data}

    # tableau d'arbitrage
    head = "".join(f"<th>{st.replace('_',' ').title()[:16]}<br>"
                   f"<span class=gt>vérité {TRUTH[st]}</span></th>" for st in judged)
    arb = ""
    for cfg, sc in sorted(scores.items(), key=lambda kv: (-kv[1][0], -kv[1][1])):
        name, k, sn = cfg
        cells = "".join(
            f"<td class='{'ok' if n_letters(data[st][5][cfg]) == TRUTH[st] else 'no'}'>"
            f"{n_letters(data[st][5][cfg])}</td>" for st in judged)
        arb += (f"<tr class='{'win' if cfg == best else ''}'>"
                f"<td>{name}</td><td>k = {k}</td>"
                f"<td>{'oui' if sn else 'non'}</td>{cells}"
                f"<td>{sc[0]}/{len(judged)}</td></tr>")

    body = ""
    for st, (S, n, grid, nov, how, cuts) in data.items():
        old_cells, _ = HS.build_cells(S, n)
        old = HS.sections_from(S, n, old_cells)
        rows = []
        for sn in (False, True):
            cfg = (MAT, kd[st], sn)
            rows.append((("RECALÉ sur le rythme" if sn else "harmonie seule")
                         + f"\n{n_letters(cuts[cfg])} lettres", cuts[cfg], sn))
        img = figure(S, n, nov, rows, old)
        fmt = lambda x: " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in x)
        gt = (f"<span class=gt>vérité connue : {TRUTH[st]} sections</span>"
              if st in TRUTH else "")
        gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
        moved = sum(1 for a, b in zip(cuts[(MAT, kd[st], False)],
                                      cuts[(MAT, kd[st], True)])
                    if a["b0"] != b["b0"])
        btns = "".join(f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
                       f"<small>{s['b0']+1}</small></button>"
                       for s in cuts[(MAT, kd[st], True)])
        body += f"""<section data-grid='{gridjs}' data-audio="/audio/{st}.m4a">
<h2>{st.replace('_',' ').title()}<span class=sub>{n} mesures · k = {kd[st]} ·
{moved} frontière(s) déplacée(s) par le rythme · {how}</span></h2>
{gt}
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique · les boutons jouent le découpage recalé</span>
{btns}</div>
<table><tr><th>découpage</th><th>lettres</th><th>sections</th></tr>
<tr><td class=f>{fmt(cuts[(MAT, kd[st], False)])}</td>
<td>{n_letters(cuts[(MAT, kd[st], False)])}</td>
<td>{len(cuts[(MAT, kd[st], False)])}</td></tr>
<tr class=win><td class=f>{fmt(cuts[(MAT, kd[st], True)])}</td>
<td>{n_letters(cuts[(MAT, kd[st], True)])}</td>
<td>{len(cuts[(MAT, kd[st], True)])}</td></tr>
<tr><td class=f>{fmt(old)}</td><td>{n_letters(old)}</td><td>{len(old)}</td></tr>
</table></section>"""

    out = HERE / "harmonia_min/state/reports/rhythm_arbitrage.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Arbitrer avec le rythme</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} a{{color:#8a2b2b}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 6px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
.gt{{display:inline-block;font:700 12px system-ui;color:#0f5132;background:#e4f0e8;
  border-radius:6px;padding:2px 8px;margin-bottom:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}}
td.f{{font:500 11px ui-monospace,monospace}}
td.ok{{background:#e4f0e8;color:#0f5132;font-weight:700}}
td.no{{color:#a89f8c}} tr.win td{{background:#e4f0e8;font-weight:700}}
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
<h1>Arbitrer avec le rythme</h1>
<div class=lede>Trois morceaux ont une vérité connue, que tu as donnée :
<b>Norah 2 sections, This Love 3, The Walk 2</b>. Ce sont les seuls juges de
cette page. Tout le reste — quelle matrice, combien de groupes, faut-il recaler
les frontières sur la batterie — est choisi par eux, puis appliqué tel quel aux
autres morceaux.<br><br>
<b>Le recalage rythmique</b> est modeste par construction : il ne crée jamais de
frontière et n'en supprime jamais. Il déplace une frontière déjà décidée par
l'harmonie d'au plus deux mesures, vers le pic de <b>nouveauté rythmique</b> le
plus proche. Cette courbe est le noyau de Foote (Foote 2000) lu sur la SSM
rythmique agrégée à deux mesures — l'échelle que la première étude avait trouvée
lisible. Elle est dessinée en vert sous chaque matrice.<br><br>
Pour chaque morceau : la matrice, la nouveauté rythmique, puis le découpage
harmonique seul et le même <b>recalé</b>, puis la règle actuelle.</div>
<section><h2>L'arbitrage — et ce qu'il révèle</h2>
<p class=sub><b>Le comptage de lettres ne départage que k, pas la matrice.</b>
Regarde le tableau : à k = 2 les trois matrices donnent 2 lettres, à k = 3 elles
en donnent 3. Le nombre de groupes demandé devient le nombre de lettres, presque
mécaniquement. Et surtout, <b>aucun k ne convient aux trois morceaux</b> : Norah
veut 2, This Love 3, The Walk 2. k ne peut donc pas être une constante du
pipeline.<br><br>
J'ai aussi testé si k se lisait dans le spectre lui-même (le plus grand écart
entre valeurs propres, la méthode classique) : il tombe juste une fois sur
trois, quelle que soit la matrice. Ça ne marche pas non plus.<br><br>
Donc k est fixé ici <b>par morceau</b> à la vérité quand on la connaît, à 3
sinon, la matrice est la <b>passage continu</b> (la plus propre à l'œil sur la
page précédente), et la question qui reste — la seule que ces trois morceaux
peuvent réellement trancher — est : <b>le recalage rythmique met-il les
frontières au bon endroit ?</b> Il déplace 2 à 4 frontières par morceau, donc il
n'est pas cosmétique. C'est à l'oreille.</p>
<table><tr><th>matrice</th><th>groupes</th><th>recalé sur<br>le rythme</th>
{head}<th>vérités<br>respectées</th></tr>{arb}</table></section>
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
