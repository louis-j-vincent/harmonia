"""La réduction de la matrice : clustering spectral sur similarité + transition.

    python scripts/spectral_sections.py [<stem> ...]
      -> /reports/spectral_sections.html

Louis, 2026-08-05 : « le challenge quand on a nos matrices SSM faites depuis les
parties, c'est de trouver la réduction matricielle la plus simple qui explique
notre répartition en groupes… c'est un problème de clustering en fait, pour
lequel on a déjà la matrice de similarité / transition. Est-ce qu'il y a quelque
chose à faire avec ça ? »

Oui, et c'est exactement la méthode de référence du domaine. McFee & Ellis,
« Analyzing song structure with spectral clustering », ISMIR 2014
(https://brianmcfee.net/papers/ismir2014_spectral.pdf) : on fabrique un graphe
dont les sommets sont les mesures et où deux mesures sont reliées de deux façons
— parce qu'elles se ressemblent (la SSM, ta matrice de similarité) et parce
qu'elles se suivent (le chemin du temps, ta matrice de transition). On prend le
laplacien de ce graphe, ses premiers vecteurs propres, et on regroupe dedans.

**Pourquoi c'est « la réduction la plus simple ».** Les vecteurs propres du
laplacien sont, littéralement, les façons les moins coûteuses de couper le
graphe : le premier sépare le morceau en deux blocs en traversant le moins de
ressemblance possible, le deuxième raffine, etc. Prendre les k premiers et
regrouper dedans, c'est demander « quelle partition en k groupes coûte le moins
cher en ressemblance brisée ». Il n'y a pas de seuil, pas de motif, pas de
chaîne de symboles : juste la matrice et le nombre de groupes.

**Et le nombre de groupes est le seul réglage.** La page dessine k = 2 à 8 l'un
sous l'autre. Aucun n'est déclaré gagnant : c'est une échelle, comme un zoom, et
c'est à l'oreille de dire lequel est la bonne granularité pour ce morceau.

Ce que ça apporte par rapport à tout ce qu'on a fait aujourd'hui : la chaîne de
cellules décide d'abord d'un alphabet, puis raisonne sur des symboles ; ici rien
n'est décidé avant, la matrice entière est utilisée d'un coup, et les frontières
tombent où elles tombent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from sklearn.cluster import KMeans       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from pattern_lanes import load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R  # noqa: E402
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402

KS = [2, 3, 4, 5]
MU = 0.35         # le poids de la ressemblance face au fil du temps. Plus bas
                  # que 1/2 : à 0,5 le regroupement saute d'un groupe à l'autre
                  # toutes les deux mesures sur Bein Green — une section doit
                  # d'abord être CONTINUE, la ressemblance vient ensuite.
SMOOTH = 5        # lissage médian des étiquettes, en mesures — même raison :
                  # une mesure isolée d'un autre groupe au milieu d'une section
                  # est un accident, pas une frontière.
N_VEC = 10        # combien de vecteurs propres on garde au maximum
DEFAULT = ["bein_green", "norah_jones_don_t_know_why", "maroon_5_this_love",
           "let_it_be_remastered_2009", "mayer_hawthorne_the_walk",
           "bruno_mars_grenade_official_music_video",
           "maroon_5_she_will_be_loved_official_music_video"]


CORE_MATCH = 0.90    # « cette mesure participe à une répétition »
CORE_GAP = 4         # …avec quelque chose d'assez loin pour ne pas être elle-même


def core_range(S, match=CORE_MATCH, gap=CORE_GAP):
    """Le CŒUR du morceau : on jette l'intro et la coda.

    Louis, 2026-08-05 : « il faudrait ignorer le début et la fin, car ils nous
    induisent en erreur quant à où est le début du morceau, et nous font
    utiliser des sections / patterns de répétition qui ne sont pas forcément les
    bons. »

    Pas de fraction arbitraire : une mesure entre dans le cœur dès qu'elle
    **participe à une répétition** — il existe une autre mesure, à au moins
    `gap` mesures d'elle, qui lui ressemble à `match`. On rogne alors la suite
    initiale et la suite finale de mesures qui n'en font partie pour personne.
    Une intro qui ne revient jamais, une coda qui s'éteint, une queue de silence
    : dehors. Tout ce qui revient, même une seule fois, reste.

    Ce que ça change concrètement : l'ancrage. Toutes nos recherches de motifs
    partent de la première mesure libre ; si cette mesure est une intro de 3
    mesures qui ne revient nulle part, la phase de tous les motifs suivants est
    décalée et on cherche des périodes à partir du mauvais point de départ.
    """
    n = len(S)
    reach = np.array([max([S[b, j] for j in range(n) if abs(j - b) >= gap] or [0.0])
                      for b in range(n)])
    b0 = 0
    while b0 < n - 1 and reach[b0] < match:
        b0 += 1
    b1 = n - 1
    while b1 > b0 and reach[b1] < match:
        b1 -= 1
    return b0, b1


# ── trois matrices possibles, et le décalage d'intro ────────────────────────
def passage_matrices(S, chain):
    """La chaîne de passages contre elle-même, en binaire et en continu.

    Louis, 2026-08-05 : « tu as pris quelle SSM pour en extraire les vecteurs
    propres ? je pense que la plus intéressante est la binaire par passage. »

    Il a raison sur le fond et à moitié tort sur la forme, alors la page montre
    les trois.

    * PAR MESURE, CONTINUE — ce que j'avais pris. Elle voit tout, y compris ce
      que l'alphabet de cellules a raté, mais ses frontières peuvent tomber
      n'importe où, y compris au milieu d'un passage.
    * PAR PASSAGE, BINAIRE — sa proposition. Petite, nette, et surtout ses
      frontières ne peuvent tomber QU'ENTRE deux passages, ce qui règle d'un
      coup le décalage d'une mesure. Son défaut : si deux passages sont la même
      musique et ont reçu des symboles différents, elle dit 0 et aucun
      regroupement ne le rattrapera.
    * PAR PASSAGE, CONTINUE — le mélange : la résolution du passage (donc pas
      de frontière au milieu) mais la valeur mesurée sur la matrice
      harmonique, pas sur l'égalité des symboles. Deux passages étiquetés
      différemment mais qui sonnent pareil se retrouvent.
    """
    m = len(chain)
    B = np.zeros((m, m))
    C = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            a, b = chain[i], chain[j]
            B[i, j] = float(a["sym"] == b["sym"] and not a["hole"] and not b["hole"])
            L = min(a["b1"] - a["b0"], b["b1"] - b["b0"]) + 1
            C[i, j] = HS.diag_match(S, a["b0"], b["b0"], L)
    return B, C


def spread_to_bars(labels, chain, n):
    """Les étiquettes par passage redeviennent des étiquettes par mesure."""
    out = np.full(n, -1)
    for lab, c in zip(labels, chain):
        out[c["b0"]:c["b1"] + 1] = lab
    last = 0
    for b in range(n):                     # les mesures hors chaîne suivent
        if out[b] < 0:
            out[b] = last
        else:
            last = out[b]
    return out


def affinity(S, mu=MU):
    """Le graphe : on se ressemble (la SSM) OU on se suit (le fil du temps).

    La SSM seule ne sait rien du temps — deux mesures identiques à 40 mesures
    d'écart lui sont aussi proches que deux mesures voisines. Le fil du temps
    seul ne sait rien des reprises. Le mélange des deux est ce qui fait qu'une
    section est à la fois « continue » et « qui revient ».
    """
    n = len(S)
    R = np.clip(S.copy(), 0, 1)
    np.fill_diagonal(R, 0)
    # on ne garde, par mesure, que ses meilleures ressemblances : un graphe
    # complet est une bouillie où tout est un peu relié à tout
    k = max(2, int(round(np.sqrt(n))))
    thr = np.partition(R, -k, axis=1)[:, -k][:, None]
    R = np.where(R >= thr, R, 0.0)
    R = np.maximum(R, R.T)
    P = np.zeros_like(R)                 # le fil du temps
    idx = np.arange(n - 1)
    P[idx, idx + 1] = P[idx + 1, idx] = 1.0
    return mu * R + (1 - mu) * P


def embed(A, n_vec=N_VEC):
    """Les vecteurs propres du laplacien normalisé — la réduction elle-même."""
    d = A.sum(1)
    d[d == 0] = 1e-9
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(len(A)) - (A * Dm[:, None]) * Dm[None, :]
    w, V = np.linalg.eigh(L)
    V = V[:, np.argsort(w)][:, :n_vec]
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.clip(nrm, 1e-9, None), np.sort(w)[:n_vec]


def cluster(V, k, smooth=SMOOTH):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(V[:, :k])
    if smooth > 1:                       # médiane glissante sur les étiquettes
        out = lab.copy()
        h = smooth // 2
        for i in range(len(lab)):
            win = lab[max(0, i - h):i + h + 1]
            vals, cnt = np.unique(win, return_counts=True)
            out[i] = vals[np.argmax(cnt)]
        lab = out
    return lab


def sections_of(lab, n, min_bars=2):
    """Les étiquettes par mesure deviennent des sections contiguës."""
    out, b = [], 0
    while b < n:
        e = b
        while e + 1 < n and lab[e + 1] == lab[b]:
            e += 1
        out.append({"b0": b, "b1": e, "cl": int(lab[b])})
        b = e + 1
    # une section d'une mesure est absorbée par sa voisine de gauche
    merged = []
    for s in out:
        if merged and s["b1"] - s["b0"] + 1 < min_bars:
            merged[-1]["b1"] = s["b1"]
        else:
            merged.append(dict(s))
    letters, res = {}, []
    for s in merged:
        if s["cl"] not in letters:
            letters[s["cl"]] = chr(ord("A") + len(letters))
        res.append({"b0": s["b0"], "b1": s["b1"], "letter": letters[s["cl"]]})
    return res, letters


def strip(ax, secs, n, label):
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
    ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", color=INK)


def figure(S, A, V, n, cuts, old, core):
    rows = 2 + len(cuts) + 1
    heights = [3.4, 1.5] + [0.52] * len(cuts) + [0.52]
    H = sum(heights) + 1.2
    fig, axs = plt.subplots(rows, 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .22})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .30 / H, bottom=.55 / H)

    c0, c1 = core
    axs[0].imshow(A, origin="lower", extent=(c0, c1 + 1, c0, c1 + 1), cmap="magma",
                  aspect="auto", interpolation="nearest",
                  vmin=0, vmax=float(np.percentile(A[A > 0], 95)) if (A > 0).any() else 1)
    axs[0].set_ylabel("mesure", fontsize=7.5)
    axs[0].tick_params(labelsize=6.4)
    axs[0].set_xlim(0, n); axs[0].set_ylim(0, n)
    for a, b in ((0, c0), (c1 + 1, n)):
        if b > a:
            axs[0].axvspan(a, b, color="#d8d0bc", zorder=3)
            axs[0].axhspan(a, b, color="#d8d0bc", zorder=3)
    axs[0].set_title("le graphe : se ressemblent (SSM) + se suivent (fil du temps)"
                     f"  ·  gris = intro/coda ignorées (mes. 1–{c0} et {c1+2}–{n})"
                     if (c0 > 0 or c1 + 1 < n) else
                     "le graphe : se ressemblent (SSM) + se suivent (fil du temps)",
                     fontsize=8.2, color="#6f6858", loc="left", pad=3)

    E = V[:, 1:9].T
    axs[1].imshow(E, aspect="auto", origin="upper", extent=(c0, c1 + 1, 8, 0),
                  cmap="RdBu_r", vmin=-np.abs(E).max(), vmax=np.abs(E).max(),
                  interpolation="nearest")
    axs[1].set_xlim(0, n)
    axs[1].set_yticks([]); axs[1].set_ylabel("vecteurs\npropres", fontsize=7,
                                             rotation=0, ha="right", va="center")

    for i, (k, secs) in enumerate(cuts):
        strip(axs[2 + i], secs, n, k)
        for a, b in ((0, c0), (c1 + 1, n)):
            if b > a:
                axs[2 + i].add_patch(plt.Rectangle((a, 0), b - a, 1,
                                                   facecolor="#d8d0bc", zorder=4))
    strip(axs[-1], old, n, "règle\nactuelle")
    axs[-1].set_xticks(range(0, n + 1, 4))
    axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    return fig2b64_fixed(fig)


def song(stem):
    S, n, grid = load(stem)
    c0, c1 = core_range(S)                     # on jette l'intro et la coda

    # L'ANCRAGE. Louis : « l'intro est souvent répétée, mais il ne faut pas
    # qu'elle soit comptée comme une mesure de trop qui décale tout. » La grille
    # de 2 mesures part donc du CŒUR, pas de la mesure 1 : une intro de longueur
    # impaire ne décale plus la phase de tout ce qui suit.
    cells = HY.build_hypo(S, n, unit=2) if c0 % 2 == 0 else \
        HY.build_hypo(S, n, unit=2, start=c0)
    chain = HY.merge_two_to_four(HY.chain_of(cells, n))[0]

    Sc = S[c0:c1 + 1, c0:c1 + 1]
    B, C = passage_matrices(S, chain)
    runs = [
        ("par mesure, continue", affinity(Sc), None),
        ("par passage, binaire", affinity(B), chain),
        ("par passage, continue", affinity(C), chain),
    ]
    cuts = []
    for name, A, ch in runs:
        V, _ = embed(A)
        for k in KS:
            lab = cluster(V, k)
            if ch is None:
                secs, _ = sections_of(lab, len(A))
                secs = [{**x, "b0": x["b0"] + c0, "b1": x["b1"] + c0} for x in secs]
            else:
                secs, _ = sections_of(spread_to_bars(lab, ch, n), n)
            cuts.append((f"{name}\nk = {k}", secs))
    old_cells, _ = HS.build_cells(S, n)
    old = HS.sections_from(S, n, old_cells)
    A0 = runs[0][1]
    V0, _ = embed(A0)
    img = figure(S, A0, V0, n, cuts, old, (c0, c1))

    fmt = lambda secs: " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in secs)
    rows = "".join(f"<tr><td><b>{k.replace(chr(10), ' · ')}</b></td>"
                   f"<td>{len(secs)}</td>"
                   f"<td class=f>{fmt(secs)}</td></tr>" for k, secs in cuts)
    rows += (f"<tr class=old><td>règle actuelle</td><td>{len(old)}</td>"
             f"<td class=f>{fmt(old)}</td></tr>")
    gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
    k4 = cuts[len(KS) * 2 + 2][1]
    btns = "".join(f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
                   f"<small>{s['b0']+1}</small></button>" for s in k4)
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
cœur = {c0+1}–{c1+1}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer · les boutons jouent k = 4</span>
{btns}</div>
<table><tr><th>groupes</th><th>sections</th><th>découpage</th></tr>{rows}</table>
</section>"""


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
    out = HERE / "harmonia_min/state/reports/spectral_sections.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Réduire la matrice</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} a{{color:#8a2b2b}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371;margin-left:8px}}
img{{width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11px}}
td.f{{font:500 11px ui-monospace,monospace}} tr.old td{{color:#8a8371}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:0 0 10px}}
.pp{{width:36px;height:36px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:13px;cursor:pointer;flex:none}}
.pos{{font:600 12px ui-monospace,monospace;min-width:88px}}
.hint{{font:500 11px system-ui;color:#a89f8c;margin-right:6px}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:6px;
  padding:3px 7px;cursor:pointer;font:700 11.5px ui-monospace,monospace}}
button.blk small{{display:block;font:500 8.5px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{color:#fff !important;background:#8a2b2b !important}}
</style></head><body><div class=wrap>
<h1>Réduire la matrice</h1>
<div class=lede>Ta question : quelle est la <b>réduction la plus simple de la
matrice</b> qui explique le découpage en groupes ? C'est un problème de
clustering, et il existe une réponse standard qui utilise exactement les deux
matrices qu'on a déjà.<br><br>
On construit un graphe dont les sommets sont les mesures. Deux mesures y sont
reliées de deux façons : <b>parce qu'elles se ressemblent</b> (la SSM) et
<b>parce qu'elles se suivent</b> (le fil du temps). On prend le laplacien de ce
graphe et ses premiers <b>vecteurs propres</b>.<br><br>
<b>Pourquoi c'est « le plus simple ».</b> Les vecteurs propres du laplacien
sont, littéralement, les façons les moins coûteuses de couper le graphe : le
premier sépare le morceau en deux en brisant le moins de ressemblance possible,
le deuxième raffine, et ainsi de suite. Demander une partition en k groupes,
c'est demander celle qui casse le moins de ressemblance. Aucun seuil, aucun
motif, aucune chaîne de symboles — juste la matrice et le nombre de groupes.
<br><br>
C'est la méthode de McFee &amp; Ellis, <i>Analyzing song structure with spectral
clustering</i>, ISMIR 2014
(<a href="https://brianmcfee.net/papers/ismir2014_spectral.pdf">papier</a>).
<br><br>
<b>Le nombre de groupes est le seul réglage</b>, et il n'est pas choisi ici : la
page dessine k = 2 à 8 l'un sous l'autre. C'est un zoom, pas un verdict. La
bande « vecteurs propres » au-dessus montre la réduction elle-même — chaque
ligne est une façon de couper, et les blocs de couleur y sont déjà visibles
avant tout regroupement.</div>
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
