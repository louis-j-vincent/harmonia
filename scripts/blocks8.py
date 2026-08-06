"""Hypothèses de blocs de 8 mesures — celle qui recouvre le mieux avec le moins de blocs.

    python scripts/blocks8.py [<stem> ...]  ->  /reports/blocks8.html

Louis, 2026-08-06 :

  « Dès qu'il y a une variation de hauteur, ça veut dire que la personne chante.
    Si la personne rappe, c'est monotone. Donc tu peux mieux détecter le début.
    Ça commence à chanter dès que ça commence à chanter, tout simplement.

    Maintenant tu vas créer des hypothèses de BLOCS DE HUIT. Une section, pour
    l'instant, c'est un bloc de huit mesures. Tu me crées des hypothèses : « oui
    ça c'est bien un bloc de huit parce qu'il se répète » — et j'y mets un
    astérisque important : **les deux dernières mesures peuvent changer**.

    Une hypothèse, c'est : voilà un premier bloc de huit, voilà un deuxième, et
    ce qu'il faut voir c'est ce qui reste au milieu. Qu'est-ce qui se répète,
    qu'est-ce qui ne se répète pas, où sont les résidus ? **L'hypothèse
    gagnante, c'est celle qui recouvre le mieux la chanson avec le moins de
    blocs.**

    Sur Norah il y a une intro, puis la chanson commence, et il y a deux mesures
    en plus qui se répètent à la fin de la section : le trou qu'on a, ce sont
    ces deux mesures-là. Sur Bein Green, le premier A que tu détectes avant
    qu'il y ait une variation, c'est une intro — et ça dépend de l'hypothèse de
    départ. »

CE QUE FAIT LE MODÈLE

1. **Le chant, autrement.** Le critère précédent cherchait une hauteur TENUE ;
   c'était le mauvais sens. Chanter, c'est BOUGER entre des notes ; rapper ou
   parler sur un ton, c'est rester plat. On garde donc les instants voisés et
   assez forts, et on déclare « ça chante » dès que la hauteur parcourt au
   moins deux demi-tons dans la seconde qui suit.

2. **Une hypothèse = un décalage de départ.** Pour chaque départ possible (les
   mesures paires autour du chant, plus quelques témoins), on pave la chanson
   en blocs de 8 mesures à partir de là. Ce qui précède le premier bloc est
   l'INTRO, ce qui dépasse à la fin est la CODA.

3. **Deux blocs sont le même** quand leurs **six premières mesures** coïncident.
   Les deux dernières sont libres — c'est l'astérisque : une section revient
   avec une fin différente (turnaround, montée, variation) sans cesser d'être
   la même section.

4. **On note l'hypothèse** par ce qu'elle explique : la part de la chanson
   couverte par des blocs qui reviennent au moins deux fois, puis, à couverture
   égale, le plus petit nombre de blocs distincts. C'est exactement « le
   meilleur recouvrement avec des blocs minimaux ».

5. **Les résidus sont nommés.** Un reste de 2 ou 4 mesures collé à un bloc qui
   se répète n'est pas une section : c'est la queue de cette section, écrite
   comme telle. Un reste plus long qui ne ressemble à rien est un vrai trou.
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

BLOCK = 8          # une section, pour l'instant, c'est huit mesures
FREE_TAIL = 2      # …dont les deux dernières sont libres
SAME = 0.90        # deux blocs sont le même à partir d'ici
TAIL_MAX = 4       # un reste d'au plus tant de mesures est une queue, pas une section
DEFAULT = ["norah_jones_don_t_know_why", "bein_green", "maroon_5_this_love",
           "mayer_hawthorne_the_walk", "let_it_be_remastered_2009",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]


# ── 1. le chant : la hauteur BOUGE ─────────────────────────────────────────
def sing_onset(vocal_path, sr=22050, span_cents=200, look=1.0, min_run=0.3):
    """Le premier chant. Chanter, c'est bouger entre des notes.

    Louis, 2026-08-06 : « dès qu'il y a une variation de hauteur, la personne
    chante ; si elle rappe, c'est monotone ». Le critère précédent demandait une
    hauteur TENUE et attendait donc la première note longue du morceau — Let It
    Be ne « chantait » qu'à 165 s. On demande maintenant l'inverse : voisé,
    assez fort, et la hauteur parcourt au moins `span_cents` (deux demi-tons)
    dans la seconde qui suit. Une intro instrumentale n'a pas de voix ; un rap
    est voisé mais plat ; un chant bouge tout de suite.
    """
    import librosa
    y, sr = librosa.load(str(vocal_path), sr=sr, mono=True)
    hop = 256
    f0, voiced, _ = librosa.pyin(y, sr=sr, hop_length=hop,
                                 fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C6"))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    f0, voiced = f0[:len(rms)], voiced[:len(rms)]
    cents = 1200 * np.log2(np.where(np.isfinite(f0), f0, 1.0) / 55.0)
    loud = rms > max(1e-4, .12 * float(np.percentile(rms, 95)))
    ok = voiced & loud & np.isfinite(cents)
    w = max(4, int(round(look * sr / hop)))
    moves = np.zeros(len(rms), bool)
    for i in range(len(rms)):
        seg = cents[i:i + w][ok[i:i + w]]
        moves[i] = len(seg) >= w // 3 and (seg.max() - seg.min()) >= span_cents
    sing = ok & moves
    need = int(round(min_run * sr / hop))
    run, onset = 0, None
    for i, s in enumerate(sing):
        run = run + 1 if s else 0
        if run >= need:
            onset = float(t[i - run + 1])
            break
    return onset, t, rms, np.where(voiced, f0, np.nan), sing


# ── 2. les blocs de 8 et leurs hypothèses ──────────────────────────────────
def block_same(S, a, b, block=BLOCK, free=FREE_TAIL):
    """Deux blocs sont le même si leurs (block − free) premières mesures collent."""
    L = block - free
    if a + L > len(S) or b + L > len(S):
        return 0.0
    return HS.diag_match(S, a, b, L)


def tile(S, n, start, block=BLOCK, same=SAME):
    """Une hypothèse : pave à partir de `start`, groupe les blocs, nomme les restes."""
    starts = list(range(start, n - block + 1, block))
    if not starts:
        return None
    parent = list(range(len(starts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            if block_same(S, starts[i], starts[j]) >= same:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(starts)):
        groups.setdefault(find(i), []).append(i)

    letters, out = {}, []
    if start > 0:
        out.append({"b0": 0, "b1": start - 1, "kind": "intro", "letter": "intro"})
    for i, s in enumerate(starts):
        r = find(i)
        if r not in letters:
            letters[r] = chr(ord("A") + len(letters))
        out.append({"b0": s, "b1": s + block - 1, "kind": "bloc",
                    "letter": letters[r], "rep": len(groups[r])})
    tail0 = starts[-1] + block
    if tail0 < n:
        out.append({"b0": tail0, "b1": n - 1, "kind": "reste", "letter": "?"})

    # un reste court collé à un bloc qui revient est la QUEUE de ce bloc
    for i, s in enumerate(out):
        if s["kind"] == "reste" and s["b1"] - s["b0"] + 1 <= TAIL_MAX and i > 0:
            prev = out[i - 1]
            if prev["kind"] == "bloc" and prev.get("rep", 1) >= 2:
                s["kind"] = "queue"
                s["letter"] = prev["letter"] + "′"

    covered = sum(s["b1"] - s["b0"] + 1 for s in out
                  if s["kind"] in ("bloc", "queue") and
                  (s["kind"] == "queue" or s.get("rep", 1) >= 2))
    distinct = len({s["letter"] for s in out if s["kind"] == "bloc"
                    and s.get("rep", 1) >= 2})
    return {"secs": out, "couverture": covered / n, "blocs": distinct,
            "start": start,
            # à couverture et nombre de blocs égaux, on préfère la PLUS COURTE
            # intro : ce qui est jeté au début n'est expliqué par rien, donc en
            # jeter moins est strictement mieux. Sur Norah, départ mes. 7 et
            # départ mes. 15 sont à égalité parfaite (79 %, 2 blocs) et seule
            # cette troisième clé les départage.
            "score": (round(covered / n, 3), -distinct, -start)}


def strip(ax, secs, n, label, win=False):
    seen = {}
    for s in secs:
        if s["kind"] == "intro":
            ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                       facecolor="#cfc7b2", hatch="///",
                                       edgecolor="#a89f8c", lw=.6))
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, "intro", ha="center",
                    va="center", fontsize=6, color="#4a4438")
            continue
        if s["kind"] == "reste":
            ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                       facecolor="#efe8d6", edgecolor="#b9b09a"))
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, "?", ha="center",
                    va="center", fontsize=6.5, color="#8a8371")
            continue
        if s["kind"] == "queue":
            base = seen.get(s["letter"][0], "#8a8371")
            ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                       facecolor=base, alpha=.45,
                                       edgecolor="#fff", lw=1.0))
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                    va="center", fontsize=6, color="#fff")
            continue
        solo = s.get("rep", 1) < 2
        seen.setdefault(s["letter"], COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   facecolor=seen[s["letter"]],
                                   alpha=.35 if solo else 1.0))
        ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                va="center", color="#fff" if not solo else "#4a4438",
                fontsize=7.4, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.2)
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel(label, fontsize=6.6, rotation=0, ha="right", va="center",
                  color="#1f8a5b" if win else INK)


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, t, rms, f0, sing = sing_onset(voc)
    b_sing = VA.bar_of(grid, onset)

    cands = sorted({s for s in range(0, min(n - BLOCK, 16), 2)}
                   | ({b_sing - (b_sing % 2) + d for d in (-4, -2, 0, 2)}
                      if b_sing is not None else set()))
    cands = [c for c in cands if 0 <= c <= n - BLOCK]
    hyps = [h for h in (tile(S, n, c) for c in cands) if h]
    hyps.sort(key=lambda h: h["score"], reverse=True)
    best = hyps[0] if hyps else None
    shown = hyps[:6]

    heights = [2.6, 0.75, 0.75] + [0.55] * len(shown)
    H = sum(heights) + 1.3
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .24})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .32 / H, bottom=.58 / H)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.5); axs[0].tick_params(labelsize=6.4)
    g = np.asarray(grid, float)
    tb = np.interp(t, g, np.arange(len(g)))
    axs[1].fill_between(tb, rms / (rms.max() or 1), color="#8a2b2b", alpha=.30, lw=0)
    axs[1].set_ylim(0, 1.05); axs[1].set_yticks([])
    axs[1].set_ylabel("voix", fontsize=6.8, rotation=0, ha="right", va="center",
                      color="#8a2b2b")
    axs[2].plot(tb, f0, ".", ms=1.1, color="#8a8371")
    axs[2].plot(tb[sing], f0[sing], ".", ms=1.8, color="#0f766e")
    axs[2].set_yscale("log"); axs[2].set_yticks([])
    axs[2].set_ylabel("hauteur\nvert = chant", fontsize=6.6, rotation=0,
                      ha="right", va="center", color="#0f766e")
    for a in (axs[1], axs[2]):
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        if b_sing is not None:
            a.axvline(b_sing, color="#0f766e", lw=1.6)
    if b_sing is not None:
        axs[1].text(b_sing, 1.02, f"  chant : mes. {b_sing+1} ({onset:.1f} s)",
                    fontsize=7, color="#0f766e", va="top")
    for i, h in enumerate(shown):
        strip(axs[3 + i], h["secs"], n,
              f"départ mes. {h['start']+1}\n{h['couverture']:.0%} · {h['blocs']} blocs",
              win=(h is best))
        axs[3 + i].axvline(h["start"], color="#111", lw=1.6)
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    def fmt(h):
        return " ".join(
            ("intro" if s["kind"] == "intro" else
             "?" if s["kind"] == "reste" else s["letter"])
            + f"[{s['b0']+1}-{s['b1']+1}]" for s in h["secs"])
    rows = "".join(
        f"<tr class='{'win' if h is best else ''}'><td>mes. {h['start']+1}</td>"
        f"<td>{h['couverture']:.0%}</td><td>{h['blocs']}</td>"
        f"<td class=f>{fmt(h)}</td></tr>" for h in hyps)
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>"
        f"{'intro' if s['kind']=='intro' else ('?' if s['kind']=='reste' else s['letter'])}"
        f"<small>{s['b0']+1}</small></button>" for s in best["secs"]) if best else ""
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
chant {f'mes. {b_sing+1} ({onset:.1f} s)' if onset else 'non détecté'} ·
retenue : départ mes. {best['start']+1 if best else '—'}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>les boutons jouent les blocs de l'hypothèse retenue</span>{btns}</div>
<table><tr><th>départ</th><th>couverture</th><th>blocs distincts</th>
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
    out = HERE / "harmonia_min/state/reports/blocks8.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Blocs de huit</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} ol{{margin:8px 0 0;padding-left:20px}} ol li{{margin-bottom:6px}}
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
<h1>Blocs de huit</h1>
<div class=lede>Une section, pour l'instant, <b>c'est un bloc de huit mesures</b>.
Une hypothèse, c'est un endroit où le premier bloc commence ; tout le reste en
découle.
<ol>
<li><b>Le chant, dans le bon sens.</b> Le critère d'hier cherchait une hauteur
tenue — c'était l'inverse de ce qu'il fallait, et il attendait la première note
longue du morceau. Chanter, c'est <b>bouger entre des notes</b> ; rapper ou
parler sur un ton, c'est rester plat. On déclare donc « ça chante » dès que la
voix est là, assez forte, et que la hauteur parcourt deux demi-tons dans la
seconde qui suit.</li>
<li><b>Le pavage.</b> À partir du départ supposé, la chanson est pavée en blocs
de 8. Ce qui précède est l'<b>intro</b> (hachuré), ce qui dépasse à la fin est un
reste.</li>
<li><b>L'astérisque.</b> Deux blocs sont le même dès que leurs <b>six premières
mesures</b> coïncident : les deux dernières sont libres. Une section revient
avec une fin différente — turnaround, montée, variation — sans cesser d'être la
même section.</li>
<li><b>Le classement.</b> L'hypothèse gagnante est celle qui <b>couvre le plus
de chanson avec le moins de blocs distincts</b>. Un bloc qui ne revient jamais
est dessiné en pâle : il ne compte pas comme expliqué.</li>
<li><b>Les résidus sont nommés.</b> Un reste de 2 à 4 mesures collé à un bloc
qui revient n'est pas une section : c'est la <b>queue</b> de cette section, notée
A′. C'est le cas de Norah — les deux mesures en trop à la fin.</li>
</ol></div>
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
