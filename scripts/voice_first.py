"""Chercher les reprises au lieu de les supposer : le bloc glissé sur la voix.

    python scripts/voice_first.py [<stem> ...]  ->  /reports/voice_first.html

Louis, 2026-08-07 :

  « Quand on a créé nos blocs de 8, comment tu trouves les suivants ? Avec la
    matrice SSM de la voix glissée ? Car je pense que c'est ça la bonne
    stratégie : une fois qu'on a chopé les VRAIS pics, ça nous donne toutes les
    répétitions du 1er bloc, et ça nous fait déjà une grosse partie de la
    structure qui est bloquée. Ensuite on cherche le deuxième bloc de 8 pareil,
    en trouvant toujours les autres grâce à la matrice SSM voix glissée. »

CE QUE FAISAIT `melody_check`, ET QUI N'EST PAS ÇA. Les blocs y sont posés sur
une grille rigide — `vstart`, +8, +16… — donc leurs positions ne sont jamais
cherchées, elles sont imposées ; on les regroupe ensuite en comparant leurs six
premières mesures sur la matrice HARMONIQUE ; et la voix glissée n'arrive
qu'après, comme témoin, pour déplacer une frontière. Elle ne trouve rien.

CE QUE FAIT CETTE PAGE. L'ordre est inversé.

  1. Le premier bloc part de la fin de l'intro (la règle de la voix : le premier
     début de bloc à partir du chant). Huit mesures.
  2. On le GLISSE sur la matrice de chant, sur tout le morceau, et on garde ses
     vrais pics. Chaque pic est une reprise du bloc — on les verrouille toutes
     d'un coup, elles portent la même lettre.
  3. On reprend au premier endroit libre, huit mesures, et on recommence.
  4. Ce qui n'a jamais été réclamé reste `?`.

« VRAI pic » veut dire trois choses à la fois, et les trois sont nécessaires :
c'est un maximum local, il dépasse le seuil PROPRE AU MORCEAU (le neuvième
décile des paires de chant de ce morceau — voir `channels.py`, un 0.62 absolu ne
veut rien dire d'un morceau à l'autre), et il tombe sur la grille de deux
mesures. Sans la dernière condition on récupère des reprises décalées d'une
mesure, qui ne peuvent pas être des débuts de section.

CE QUE LA PAGE MONTRE POUR ARBITRER. Sous chaque courbe de chant, la MÊME courbe
calculée sur l'harmonie. Là où les deux piquent au même endroit, la reprise est
solide ; là où seule la voix pique, c'est elle qui apporte l'information ; là où
seule l'harmonie pique, c'est probablement la boucle d'accords qui tourne sans
que la section recommence. C'est l'arbitrage que Louis fait à l'oreille.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.signal import find_peaks      # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))
from pattern_lanes import (load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R,  # noqa: E402
                           edge, mid)
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import melody_check as MC                                                 # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import channels as CN                                                     # noqa: E402

BLOCK = 8          # « nos blocs de 8 »
UNIT = 2           # la grille : une section ne commence pas sur une mesure impaire
DEFAULT = B8.DEFAULT


SHARP_W = 0.5      # ce que pèse la finesse d'un pic à côté de sa hauteur


def sharpness(cur, p, n, unit=UNIT):
    """À quel point le pic est POINTU : la dérivée seconde, changée de signe.

    Un plateau large à 0.70 et une pointe à 0.68 ne disent pas la même chose. Le
    plateau veut dire « ça se ressemble vaguement sur douze mesures », la pointe
    veut dire « ça recommence ICI ». Mesurée sur le pas de deux mesures, puisque
    c'est la grille sur laquelle une section a le droit de commencer.
    """
    return 2 * cur[p] - cur[max(0, p - unit)] - cur[min(n - 1, p + unit)]


def true_peaks(cur, b0, n, thr, unit=UNIT, block=BLOCK, claimed=None,
               sharp_w=SHARP_W):
    """Les VRAIS pics de la courbe glissée : les reprises du bloc.

    Trois conditions d'éligibilité, aucune décorative :
      * maximum local — sinon un plateau donne dix « reprises » collées ;
      * au-dessus du seuil du morceau, pas d'un nombre absolu ;
      * sur la grille de deux mesures, sinon on propose des débuts de section
        décalés d'une mesure, ce qu'aucune section ne fait.

    Puis LE PLUS FORT SE SERT LE PREMIER, et c'est le correctif du 2026-08-07.
    Louis : « sur Norah le 3e A reprend mesure 31, le pic violet le montre
    clairement, pourtant c'en est un autre qui est détecté ». Mesuré :

        mesure 27   hauteur 0.554   finesse +0.70      <- retenu à tort
        mesure 31   hauteur 0.847   finesse +1.22      <- le bon, sur les DEUX

    Le seuil n'y était pour rien (0.546, la mesure 27 le franchit vraiment). Le
    coupable était le filtre de voisinage : je refusais tout pic à moins de huit
    mesures d'un pic déjà pris, **en parcourant la courbe de gauche à droite**,
    donc le premier tuait le meilleur. Le même défaut que le balayage
    gauche-droite des mini-sections, une couche plus bas.

    On classe donc les candidats par score et on sert dans cet ordre. Le score
    est celui que propose Louis — « c'est à la fois la hauteur des pics et leur
    sharpness qui nous dit le bon endroit » : hauteur + la moitié de la finesse.
    Sur Norah les deux composantes désignent la mesure 31, donc leur somme aussi ;
    le poids ne fait pencher aucune balance ici, il départage les cas où la
    hauteur seule hésite.
    """
    idx, _ = find_peaks(cur, height=thr, distance=unit)
    cand = []
    for p in idx:
        p = int(p)
        if (p - b0) % unit:
            continue
        if abs(p - b0) < block or p + block > n:
            continue
        if claimed is not None and claimed[p:p + block].any():
            continue
        cand.append((cur[p] + sharp_w * max(0.0, sharpness(cur, p, n, unit)), p))
    out = []
    for _, p in sorted(cand, key=lambda t: -t[0]):
        if any(abs(p - q) < block for q in out):
            continue
        out.append(p)
    return sorted(out)


def passes(S, M, n, vstart, thr_m, thr_h, block=BLOCK, unit=UNIT, max_blocks=2):
    """La boucle de Louis : un bloc, ses reprises par la voix, on verrouille, au suivant.

    Louis, 2026-08-07 : « une fois qu'on a trouvé un bloc de 8, on utilise le
    glissage sur matrice SSM de voix pour trouver les occurrences suivantes de ce
    bloc. Puis on refait ça avec le deuxième bloc à trouver. Pour l'instant on ne
    s'embête pas avec les queues : fais-moi déjà la détection des 2 différents
    blocs de 8 et on les ancre dans la chanson. »

    D'où `max_blocks=2` : on s'arrête après le deuxième. Ce qui reste n'est pas
    une erreur, c'est ce qu'on n'a pas encore expliqué.
    """
    claimed = np.zeros(n, bool)
    runs, cursor = [], vstart
    while cursor + block <= n and len(runs) < max_blocks:
        if claimed[cursor:cursor + block].any():
            cursor += unit
            continue
        cur_m = MC.slide_on(M, cursor, block, n)
        cur_h = MC.slide_on(S, cursor, block, n)
        occ = true_peaks(cur_m, cursor, n, thr_m, claimed=claimed)
        # L'HARMONIE EN SECOURS, PAS EN CONCURRENCE. Sur un passage instrumental
        # la voix n'a rien à dire — courbe plate, zéro pic — et le bloc
        # resterait seul alors qu'il revient franchement. On ne lui demande son
        # avis que dans ce cas-là : quand le chant n'a produit aucun pic.
        source = "chant"
        if not occ and cur_m.max() < thr_m:
            occ = true_peaks(cur_h, cursor, n, thr_h, claimed=claimed)
            source = "harmonie (le chant est muet ici)"
        runs.append({"b0": cursor, "occ": occ, "cur_m": cur_m, "cur_h": cur_h,
                     "source": source})
        for c in [cursor] + occ:
            claimed[c:min(n, c + block)] = True
        cursor += block
    return runs, claimed


def sections_of(runs, n, vstart, block=BLOCK):
    """Les blocs verrouillés deviennent des sections contiguës, sans trou.

    Tout ce qui n'a jamais été réclamé devient un `?` : c'est une réponse, pas un
    échec. Un morceau a des ponts et des solos qui ne reviennent nulle part.
    """
    owner = np.full(n, -1)
    for i, r in enumerate(runs):
        for c in [r["b0"]] + r["occ"]:
            owner[c:min(n, c + block)] = i
    out, b = [], 0
    if vstart > 0:
        out.append({"b0": 0, "b1": vstart - 1, "kind": "intro", "letter": "intro"})
        b = vstart
    while b < n:
        z = b
        while z + 1 < n and owner[z + 1] == owner[b]:
            z += 1
        i = owner[b]
        out.append({"b0": b, "b1": z, "rep": 1 + len(runs[i]["occ"]) if i >= 0 else 1,
                    "kind": "bloc" if i >= 0 else "reste",
                    "letter": chr(ord("A") + i) if i >= 0 else "?"})
        b = z + 1
    # les lettres se suivent dans l'ordre d'apparition, pas dans l'ordre des passes
    order, ren = {}, {}
    for s in out:
        if s["kind"] == "bloc" and s["letter"] not in ren:
            ren[s["letter"]] = chr(ord("A") + len(order))
            order[s["letter"]] = True
    for s in out:
        if s["kind"] == "bloc":
            s["letter"] = ren[s["letter"]]
    for a, c in zip(out, out[1:]):
        assert c["b0"] == a["b1"] + 1, f"trou/chevauchement : {a} -> {c}"
    return out


TAIL_MAX = 4       # au-delà, un trou n'est plus une queue : c'est une section


def fill_holes(secs, S, M, n, vm, vh, tail_max=TAIL_MAX):
    """L'ÉTAPE SUIVANTE, mise de côté par Louis le 2026-08-07 (« pour l'instant
    on ne s'embête pas avec les queues »). Écrite, pas branchée.

    Les blocs de 8 ANCRENT ; les trous se déduisent après. Louis, 2026-08-07 :

      « On a le droit à des sections autres que 8 mesures, c'est juste que les
        sections de 8 mesures ANCRENT la chanson, et après il nous reste juste à
        compléter les trous, pour déduire ce qu'est un bridge, ce qu'est une
        queue de section… »

    Un trou n'est pas un échec, c'est une question à trancher, et il n'y a que
    trois réponses possibles :

      * **une queue** — court (≤ 4 mesures) et collé derrière une section
        ancrée. Il la rallonge : le 8 devient 10, et la section prend un ′.
        C'est la règle que Louis a posée le 2026-08-06 — « une queue c'est
        toujours à la fin d'une section, jamais au début » — donc un trou court
        se rattache TOUJOURS à ce qui le précède, jamais à ce qui le suit.
      * **une section qui revient** — long, et il ressemble à un autre trou
        ailleurs dans le morceau. Les deux prennent la même lettre, à leur vraie
        longueur, qui n'a aucune raison de faire huit.
      * **un pont** — long et unique. Il garde sa place et sa lettre à lui ; un
        morceau a le droit d'avoir un passage qui n'arrive qu'une fois.

    Le trou de tête reste l'intro, celui de queue reste la fin.
    """
    out = [dict(s) for s in secs]
    # 1. les trous COURTS rejoignent la section qui les précède
    fused = []
    for s in out:
        if (s["kind"] == "reste" and fused and fused[-1]["kind"] == "bloc"
                and s["b1"] - s["b0"] + 1 <= tail_max):
            fused[-1]["b1"] = s["b1"]
            fused[-1]["queue"] = True
            if not fused[-1]["letter"].endswith("′"):
                fused[-1]["letter"] += "′"
            continue
        fused.append(s)
    # 2. les trous LONGS sont des sections : se ressemblent-ils entre eux ?
    holes = [i for i, s in enumerate(fused) if s["kind"] == "reste"]
    parent = {i: i for i in holes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ai in range(len(holes)):
        for bi in range(ai + 1, len(holes)):
            a, b = fused[holes[ai]], fused[holes[bi]]
            La = a["b1"] - a["b0"] + 1
            Lb = b["b1"] - b["b0"] + 1
            if min(La, Lb) < 2 or max(La, Lb) > 1.5 * min(La, Lb):
                continue
            L = min(La, Lb)
            same = (vm.says(a["b0"], b["b0"], L) if vm.alive else None)
            if same is None:
                same = vh.says(a["b0"], b["b0"], L)
            if same:
                parent[find(holes[ai])] = find(holes[bi])
    used = {s["letter"].rstrip("′") for s in fused if s["kind"] == "bloc"}
    nxt = chr(ord("A") + len(used))
    groups, names = {}, {}
    for i in holes:
        groups.setdefault(find(i), []).append(i)
    for root, mem in groups.items():
        if len(mem) >= 2:
            while nxt in used:
                nxt = chr(ord(nxt) + 1)
            names[root] = nxt
            used.add(nxt)
            nxt = chr(ord(nxt) + 1)
    for i in holes:
        r = find(i)
        if r in names:
            fused[i].update(kind="bloc", letter=names[r], rep=len(groups[r]))
        else:
            # long, et il ne revient nulle part : c'est un pont
            first = fused[0] is fused[i]
            last = fused[-1] is fused[i]
            fused[i]["letter"] = "intro" if first else ("fin" if last else "pont")
            fused[i]["rep"] = 1
    for a, c in zip(fused, fused[1:]):
        assert c["b0"] == a["b1"] + 1, f"trou/chevauchement : {a} -> {c}"
    return fused


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(grid, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)

    vm = CN.Voice("chant", M, n, BLOCK, CN.FLOOR_M, mute=mute)
    vh = CN.Voice("harmonie", S, n, BLOCK, CN.FLOOR_H)
    runs, claimed = passes(S, M, n, vstart, vm.thr, vh.thr)
    secs = sections_of(runs, n, vstart)

    heights = [2.2] + [0.92] * max(1, len(runs)) + [0.55]
    Hh = sum(heights) + 1.5
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .30})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.62 / Hh)
    axs[0].imshow(M, origin="lower", extent=(0, n, 0, n), cmap="Purples",
                  vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.4); axs[0].tick_params(labelsize=6.2)
    axs[0].set_title("CHANT — c'est elle qui cherche les reprises maintenant",
                     fontsize=8, color="#7c3aed", loc="left", pad=3)
    if not runs:
        axs[1].axis("off")
    for i, r in enumerate(runs):
        ax, col = axs[1 + i], COLS[i % len(COLS)]
        x = mid(np.arange(n))
        ax.fill_between(x, r["cur_m"], color="#7c3aed", alpha=.22, lw=0)
        ax.plot(x, r["cur_m"], color="#7c3aed", lw=1.3)
        ax.plot(x, r["cur_h"], color="#2a6fb0", lw=.9, alpha=.65)
        ax.axhline(vm.thr, color="#7c3aed", lw=.9, ls=(0, (1, 2)))
        ax.axvline(edge(r["b0"]), color="#111", lw=1.5)
        ax.text(edge(r["b0"]) + .6, 1.0, " le bloc", fontsize=6, va="top", color="#111")
        for p in r["occ"]:
            ax.add_patch(plt.Rectangle((edge(p), 0), BLOCK, 1.12, facecolor=col,
                                       alpha=.16, lw=0))
            ax.axvline(edge(p), color=col, lw=1.7)
            ax.plot([mid(p)], [r["cur_m"][p]], "o", ms=5, color=col)
            ax.text(edge(p) + .6, .10, f"{r['cur_m'][p]:.2f}", fontsize=6, color=col)
        ax.set_ylim(0, 1.12); ax.set_yticks([])
        ax.set_ylabel(f"bloc mes. {r['b0']+1}\n{len(r['occ'])} reprise(s)",
                      fontsize=6.4, rotation=0, ha="right", va="center", color=col)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    B8.strip(axs[-1], secs, n, f"ANCRES\ndépart mes. {vstart+1}")
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.2)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fmt = lambda x: " ".join(
        ("intro" if s["kind"] == "intro" else
         "?" if s["kind"] == "reste" and s["letter"] == "?" else s["letter"])
        + f"[{s['b0']+1}-{s['b1']+1}]" for s in x)
    rows = "".join(
        f"<tr><td><b>mes. {r['b0']+1}</b></td>"
        f"<td>{', '.join(str(p+1) for p in r['occ']) or '—'}</td>"
        f"<td>{len(r['occ'])}</td><td>{r['source']}</td></tr>" for r in runs)
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
        f"<small>{s['b0']+1}</small></button>" for s in secs if s["kind"] == "bloc")
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
départ mes. {vstart+1} · {len(runs)} bloc(s) · seuil chant {vm.thr:.2f}
(propre au morceau)</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>écoute deux blocs de même lettre</span>{btns}</div>
<table><tr><th>bloc</th><th>reprises trouvées (mesure)</th><th>combien</th>
<th>trouvées par</th></tr>{rows}</table>
<p class=verdict>{fmt(secs)}</p></section>"""


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
    out = HERE / "harmonia_min/state/reports/voice_first.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Le bloc glissé sur la voix</title><style>
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
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
.verdict{{font:500 11.5px ui-monospace,monospace;background:#f7f3e9;
  border-radius:8px;padding:9px 11px;margin:10px 0 0;line-height:1.8}}
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
<h1>Le bloc glissé sur la voix</h1>
<div class=lede>Jusqu'ici les blocs étaient <b>posés sur une grille rigide</b> —
départ, +8, +16 — puis regroupés sur l'harmonie ; la voix n'arrivait qu'après,
comme témoin. Ici l'ordre est inversé : <b>c'est la voix qui cherche</b>.
<ol>
<li>Le premier bloc part de la fin de l'intro. Huit mesures.</li>
<li>On le <b>glisse sur la matrice de chant</b> et on garde ses vrais pics :
chacun est une reprise du bloc. On les verrouille toutes d'un coup.</li>
<li>On reprend au premier endroit libre, huit mesures, et on recommence.</li>
<li>Ce qui n'est jamais réclamé reste <b>?</b> — un pont, un solo, ça arrive.</li>
</ol>
<br>Un <b>vrai pic</b> est un maximum local, au-dessus du seuil <b>propre au
morceau</b> (un 0.62 absolu ne veut rien dire d'un morceau à l'autre), et
<b>sur la grille de deux mesures</b> : une section ne commence pas sur une
mesure impaire.<br><br>
La courbe <b>violette</b> est le chant, la <b>bleue</b> la même chose calculée
sur l'harmonie — elle n'est là que pour arbitrer. Les deux piquent ensemble :
reprise solide. Seule la voix pique : c'est elle qui apporte l'information.
Seule l'harmonie pique : c'est souvent la boucle d'accords qui tourne sans que
la section recommence.</div>
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
