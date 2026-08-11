"""Le profil de pics FUSIONNÉ, pondéré par la clarté de chaque matrice — à écouter.

    .venv/bin/python scripts/peak_profile.py [<stem> ...]
        -> /plots/peak_profile.html  (index)  +  /plots/peak_<stem>.html  (par morceau)

Louis, 2026-08-10 :

  « Ce qu'il faudrait, ce serait de faire une fusion des profils de pics des
    matrices pondérées par la "significativité" de chaque matrice (à quel point
    ces pics sont "clairs") -> ça nous donne un profil de pics, on peut
    sélectionner ses pics les + clairs pour définir les débuts de nos sections.
    Ensuite, on commence nos recherches de sections en partant de ces pics, avec
    la règle claire qu'une section ne peut jamais traverser un pic. […] La
    matrice de voix nous donne un indicateur des sections de solo/bridge. »

  « Donne-moi des visualisations de ces profils de pics détectés avec le
    playhead dès que tu peux afin que je puisse juger de leur utilité. »

CE QUE CETTE PAGE EST : la visualisation demandée, une page par morceau, avec
l'audio et la tête de lecture. Elle s'arrête au PROFIL et aux PICS. La règle
« une section ne traverse jamais un pic » et la recherche de sections qui en
part ne sont PAS écrites ici — il faut d'abord juger les pics à l'oreille.

LA FUSION, exactement :

    profil(x) = Σ_matrices  poids_m · nouveauté_m(x)     avec  Σ poids = 1
    poids_m   = max(contraste_m, 0)

Le poids est le CONTRASTE de `ssm_clarity` — « si je coupe aux pics de cette
matrice, est-ce que ça sépare vraiment ses blocs ». C'est lui, et pas la netteté,
parce que mesuré sur les douze morceaux annotés c'est le contraste qui classe
comme l'œil (This Love : timbre 0,81 contre basse 0,18, alors que la netteté les
donnait à égalité). Une matrice à contraste négatif — ses coupures rapprochent
plus qu'elles ne séparent — pèse zéro, elle ne vote pas contre.

`fusion` (la moyenne des rangs des sept matrices) est EXCLUE de la pondération :
elle est déjà faite de toutes les autres, la compter reviendrait à voter deux
fois. Elle reste affichée comme une ligne parmi les autres, pour comparaison.

LES PICS RETENUS sortent de la même règle du coude que dans `ssm_clarity`
(`peak_report`) : le nombre de pics n'est pas fixé, c'est l'endroit où l'écart
entre un pic et le suivant est le plus grand. La page montre AUSSI les suivants,
en pâle, pour que Louis puisse décider lui-même où couper la liste.

LE COULOIR VOIX : les demi-mesures où personne ne chante. C'est l'indicateur
solo/bridge qu'il demande — un passage instrumental long est un candidat
section à part entière, indépendamment de tout pic.
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

from ssm_zoo import (SONGS, AUDIO, GT_LINE, _novelty,        # noqa: E402
                     fig2b64, gt_sections, substrates)
from ssm_clarity import peak_report, block_contrast          # noqa: E402

OUTDIR = HERE / "docs" / "plots"
PLOT_L, PLOT_R = 0.085, 0.995      # marges du tracé, pour caler la tête de lecture
INK = "#1c1c1c"
SHOW_EXTRA = 6                     # pics au-delà du coude, montrés en pâle


def fused_profile(stem: str):
    """(profil, pics_retenus, pics_suivants, lignes, n, extra) — tout en cases
    de demi-mesure."""
    subs, n, _T, extra = substrates(stem)
    lines = []
    for nm, gloss, S in subs:
        nov = _novelty(S)
        r = peak_report(nov)
        lines.append({"nom": nm, "gloss": gloss, "nov": nov, "cuts": r["cuts"],
                      "S": S, "contraste": block_contrast(S, r["cuts"])})

    votants = [d for d in lines if d["nom"] != "fusion"]
    w = np.array([max(d["contraste"], 0.0) for d in votants])
    if w.sum() <= 0:                       # aucune matrice ne sépare quoi que ce soit
        w = np.ones(len(votants))
    w = w / w.sum()
    for d, wi in zip(votants, w):
        d["poids"] = float(wi)
    for d in lines:
        d.setdefault("poids", 0.0)

    P = np.zeros(len(lines[0]["nov"]))
    for d, wi in zip(votants, w):
        P += wi * np.nan_to_num(d["nov"])
    P = P / max(1e-9, P.max())

    r = peak_report(P)
    kept = r["cuts"]
    from scipy.signal import find_peaks
    idx, _ = find_peaks(np.nan_to_num(P), distance=8)
    rest = sorted((int(i) for i in idx if int(i) not in kept),
                  key=lambda i: -P[i])[:SHOW_EXTRA]
    return P, kept, rest, lines, n, extra


# ── la page d'un morceau ────────────────────────────────────────────────────

def song_page(stem: str, title: str) -> str:
    P, kept, rest, lines, n, extra = fused_profile(stem)
    m = len(P)
    grid = extra["grid"]
    mute = extra["mute"]
    gt = gt_sections(stem)
    gtb = [sg["b0"] for sg in gt["sections"][1:]] if gt else []

    votants = [d for d in lines if d["nom"] != "fusion"]
    votants.sort(key=lambda d: -d["poids"])
    rows = 1 + len(votants) + 1                       # profil + votants + voix
    H = 1.35 + 0.62 * (rows - 1)
    fig, axs = plt.subplots(rows, 1, figsize=(12.6, H), facecolor="#fffdf6",
                            gridspec_kw={"height_ratios": [2.4] + [1] * (rows - 1),
                                         "hspace": 0.0})

    def deco(ax, lab, colour=INK):
        ax.set_xlim(0, m - 1)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10,
                      color=colour)
        for s in ax.spines.values():
            s.set_color("#e0d7c2")

    # 1 ── le profil fusionné
    ax = axs[0]
    ax.fill_between(np.arange(m), P, color="#3d7fa6", lw=0)
    for b in gtb:
        ax.axvline(b * m / max(1, n), color=GT_LINE, lw=0.9, alpha=0.8)
    for c in kept:
        ax.axvline(c, color="#0d2437", lw=1.6)
        ax.plot([c], [1.1], marker="v", ms=9, color="#0d2437", clip_on=False)
    for c in rest:
        ax.plot([c], [1.1], marker="v", ms=7, color="#b9c4cc", clip_on=False)
    ax.set_ylim(0, 1.2)
    deco(ax, "PROFIL\nfusionné")

    # 2 ── qui a voté, et avec quel poids
    for ax, d in zip(axs[1:1 + len(votants)], votants):
        pale = d["poids"] < 0.05
        ax.fill_between(np.arange(m), np.nan_to_num(d["nov"]),
                        color="#8fb6cc" if pale else "#3d7fa6", lw=0, alpha=0.95)
        for b in gtb:
            ax.axvline(b * m / max(1, n), color=GT_LINE, lw=0.6, alpha=0.55)
        ax.set_ylim(0, 1.05)
        deco(ax, f"{d['nom']}  {d['poids']:.0%}", "#9aa3a9" if pale else INK)

    # 3 ── le couloir voix : où personne ne chante (solo / bridge / intro)
    ax = axs[-1]
    mu = np.asarray(mute, bool)[:m] if mute is not None else np.zeros(m, bool)
    ax.fill_between(np.arange(m), mu.astype(float), color="#c9a227", lw=0, alpha=0.9)
    for b in gtb:
        ax.axvline(b * m / max(1, n), color=GT_LINE, lw=0.6, alpha=0.55)
    ax.set_ylim(0, 1.05)
    deco(ax, "voix : muette", "#8a6d1f")
    img = fig2b64(fig)

    def to_bar(c):
        return int(round(c * n / max(1, m)))

    kept_bars = [to_bar(c) for c in kept]
    rest_bars = [to_bar(c) for c in rest]
    hit = sum(1 for b in kept_bars if any(abs(b - g) <= 1 for g in gtb))
    found = sum(1 for g in gtb if any(abs(b - g) <= 1 for b in kept_bars))

    btns = "".join(
        f'<button class=blk data-p="[{max(0, b - 1)},{min(n, b + 3)}]">mes. {b + 1}'
        f'<small>{"juste" if any(abs(b - g) <= 1 for g in gtb) else "à toi de dire"}'
        "</small></button>" for b in kept_bars)
    extra_btns = "".join(
        f'<button class=blk data-p="[{max(0, b - 1)},{min(n, b + 3)}]">mes. {b + 1}</button>'
        for b in rest_bars)

    weights = " · ".join(f"{d['nom']} {d['poids']:.0%}" for d in votants
                         if d["poids"] >= 0.01)
    verif = (f"{hit}/{len(kept_bars)} des pics retenus tombent sur une de tes "
             f"frontières (± 1 mesure), et ils en retrouvent {found} sur "
             f"{len(gtb)}" if gtb else "pas d'annotation pour ce morceau")

    body = f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>pics retenus</span>{btns}</div>
<div class=lane><span class=lab>les suivants</span>{extra_btns or '<span class=hint>aucun</span>'}</div>
<div class=verdict><b>Poids de la fusion —</b> {weights}<br>
<b>Vérification (elle n'entre pas dans le calcul) —</b> {verif}.</div>
</section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=2;</script>"""
    return page(f"{title} — profil de pics", body, back=True)


LEDE = """<div class=lede>En haut le <b>profil de pics fusionné</b> : la somme
des courbes de nouveauté des matrices, chacune pondérée par son <b>contraste</b>
— « si je coupe à ses pics, est-ce que ça sépare vraiment ses blocs ». Les
triangles noirs sont les pics retenus (le nombre n'est pas fixé : c'est là où
l'écart entre un pic et le suivant est le plus grand), les gris sont les
suivants, montrés pour que tu voies où la liste s'arrête. Les traits rouges sont
tes frontières — elles ne servent à rien dans le calcul, elles sont là pour
juger.<br><br>
En dessous, <b>qui a voté et pour combien</b>. Une matrice à moins de 5 % est
dessinée en pâle : elle ne pèse rien sur ce morceau. Tout en bas le
<b>couloir de la voix</b> : en jaune les demi-mesures où personne ne chante —
l'indicateur de solo / pont / intro.<br><br>
<b>Touche le graphique pour aller écouter</b>, ou touche un pic pour entendre les
deux mesures autour.</div>"""


def page(title, body, back=False, lede=True, lede_html=None,
         back_href="peak_profile.html", back_label="tous les morceaux"):
    """La coquille commune : mise en page, audio, tête de lecture.

    `lede_html` remplace le chapeau quand une autre page réutilise la coquille
    (`hard_prior_sections.py`) — une seule tête de lecture dans le dépôt, pas
    deux copies qui divergeront.
    """
    LEDE_ = lede_html if lede_html is not None else LEDE
    return _page(title, body, back, lede, LEDE_, back_href, back_label)


def _page(title, body, back, lede, LEDE, back_href, back_label):
    return f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{title}</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1120px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
.plot{{position:relative;margin-bottom:8px}}
.plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;align-items:center;gap:10px;margin:0 0 10px}}
.pp{{width:42px;height:42px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:15px;cursor:pointer;flex:none}}
.pos{{font:600 12.5px ui-monospace,monospace}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
.lane{{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:0 0 6px}}
.lab{{font:600 11px system-ui;color:#8a8371;width:96px;flex:none}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:7px;
  padding:5px 8px;cursor:pointer;font:600 12px system-ui;color:#6f6858}}
button.blk small{{display:block;font:500 9.5px ui-monospace,monospace;
  color:#a89f8c;margin-top:1px}}
button.blk.on{{background:#0d2437;color:#fff}} button.blk.on small{{color:#cbd8e0}}
.verdict{{font-size:13px;background:#f7f3e9;border-radius:8px;padding:9px 11px;margin:10px 0 0}}
a{{color:#8a2b2b}}
.back{{display:inline-block;margin-bottom:12px;font-size:13px;text-decoration:none}}
.idx{{border-collapse:collapse;font-size:13.5px;width:100%}}
.idx td{{border-bottom:1px solid #e5dcc6;padding:8px 9px}}
.idx td a{{font-weight:600;text-decoration:none}}
</style></head><body><div class=wrap>
{f'<a class=back href="{back_href}">← {back_label}</a>' if back else ''}
<h1>{title}</h1>
{LEDE if lede else ''}
{body}</div>
<script>
(function(){{
  const au=document.getElementById("au"); if(!au) return;
  const G=window.GRID, n=G.length-1, L0=window.PLOT[0], W=window.PLOT[1]-L0;
  const plot=document.querySelector(".plot"), cur=plot.querySelector(".cur"),
        hit=plot.querySelector(".hit"), pp=document.querySelector(".pp"),
        pos=document.querySelector(".pos");
  hit.style.left=(L0*100)+"%"; hit.style.width=(W*100)+"%";
  const t2b=t=>{{ if(t<=G[0])return 0; if(t>=G[n])return n;
    let lo=0,hi=n; while(hi-lo>1){{const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}}
    return lo+(t-G[lo])/(G[lo+1]-G[lo]); }};
  const b2t=f=>{{ const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    return G[i]+(f-i)*(G[i+1]-G[i]); }};
  const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
  let stopAt=null, raf=null, onBtn=null;
  function draw(){{
    const f=t2b(au.currentTime);
    cur.style.display="block";
    cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
    pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime);
  }}
  function clear(){{ if(onBtn){{ onBtn.classList.remove("on"); onBtn=null; }} }}
  function tick(){{ draw();
    if(stopAt!=null && au.currentTime>=stopAt){{ au.pause(); stopAt=null; clear(); }}
    if(!au.paused) raf=requestAnimationFrame(tick); }}
  function go(t0,t1,btn){{
    clear(); stopAt=t1;
    if(btn){{ onBtn=btn; btn.classList.add("on"); }}
    const seek=()=>{{ try{{ au.currentTime=t0; }}catch(e){{}} draw(); }};
    if(au.readyState>=1) seek();
    else au.addEventListener("loadedmetadata",seek,{{once:true}});
    au.play().catch(()=>{{ clear(); }});
  }}
  au.addEventListener("play",()=>{{ pp.textContent="❚❚"; tick(); }});
  au.addEventListener("pause",()=>{{ pp.textContent="▶";
    cancelAnimationFrame(raf); draw(); }});
  pp.onclick=()=>{{ if(au.paused){{ stopAt=null; clear(); au.play().catch(()=>{{}}); }}
                    else au.pause(); }};
  hit.onclick=e=>{{ const r=hit.getBoundingClientRect();
    go(b2t(n*(e.clientX-r.left)/r.width), null, null); }};
  document.querySelectorAll("[data-p]").forEach(b=>{{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(G[d[0]], G[Math.min(n,d[1])], b); }});
}})();
</script></body></html>"""


def main():
    stems = sys.argv[1:] or [s for s, _ in SONGS]
    todo = [(s, t) for s, t in SONGS if s in stems] or [(s, s) for s in stems]
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            print(f"  ?? pas d'audio pour {stem}")
            continue
        (OUTDIR / f"peak_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="peak_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "peak_profile.html").write_text(page(
        "Profils de pics fusionnés",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede=True))
    print(f"wrote docs/plots/peak_profile.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
