"""Le seuil des mini-sections : absolu, ou propre au morceau ? La démo pour trancher.

    python scripts/mini_threshold.py [<stem> ...]  ->  /reports/mini_threshold.html

Louis, 2026-08-07 : « montre-moi la démo des seuils relatifs pour qu'on décide
ensemble ».

CE QUI EST EN JEU. Une mini-section existe quand un bout de deux, quatre… mesures
« revient » ailleurs dans le morceau. « Revient » veut dire : similarité ≥ 0.90.
Ce 0.90 est le même pour tous les morceaux, et voici ce qu'il vaut réellement —
la part de TOUTES les paires de deux mesures d'un morceau qui le franchissent :

    The Walk    62 %        Let It Be   52 %        This Love   22 %
    Norah       28 %        Grenade     17 %        Bein Green   9 %

Sur The Walk, « ce bout revient ailleurs » est vrai de presque n'importe quel
bout : le morceau tourne sur un accord. Le critère ne trie plus rien, il dit oui.
C'est pour ça que la première cellule de Let It Be sort avec vingt-deux
occurrences — elle « correspond » à la moitié du morceau.

LES TROIS VARIANTES MONTRÉES.

  actuel     max(harmonie, chant) ≥ 0.90, le même nombre pour tous.
  relatif    chaque voie contre SON neuvième décile dans CE morceau, à CETTE
             longueur, avec 0.90 en plancher. Sur les morceaux où l'harmonie
             sépare bien, le décile tombe sous 0.90 et rien ne bouge ; sur The
             Walk il monte à 0.998. Le OU est conservé : Louis a posé qu'une
             reprise peut être harmonique OU chantée.
  priorité   même seuils, mais l'harmonie décide quand elle discrimine et le
             chant ne prend le relais que là où elle est morte. C'est la règle
             déjà retenue pour NOMMER les sections (`channels.py`) ; la question
             ouverte est de savoir si elle vaut aussi pour les TROUVER.

Rien n'est modifié dans `challenge.py` tant que Louis n'a pas tranché : les
mini-sections sont la fondation qu'il a verrouillée le 2026-08-05.
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
from pattern_lanes import (load, fig2b64_fixed, COLS, INK, PLOT_L, PLOT_R,  # noqa: E402
                           edge, mid)
import harmonia_min.harmonic_sections as HS                               # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import channels as CN                                                     # noqa: E402
import challenge as CH                                                    # noqa: E402
import hypo_sizes as HY                                                   # noqa: E402

DEFAULT = B8.DEFAULT
LENGTHS = CH.LENGTHS
UNIT = CH.UNIT


def scales(X, n, lengths, floor):
    """Seuil et contraste de cette voie, longueur par longueur."""
    out = {}
    for L in lengths:
        if n - L + 1 <= 0:
            continue
        v = CN.pair_values(X, n, L)
        if v.size == 0:
            continue
        out[L] = {"thr": max(floor, float(np.quantile(v, CN.Q))),
                  "contrast": CN.contrast(v),
                  "share90": float(np.mean(v >= 0.90))}
    return out


def cells_with(S, M, n, start, accept, lengths=LENGTHS, unit=UNIT):
    """`challenge.mini_sections`, mais le « ça revient » est un paramètre.

    Structure identique — mêmes ancres, même grille de deux mesures, même
    « la plus petite longueur qui tient l'emporte » — pour que la seule
    différence mesurable entre les variantes soit le critère lui-même.
    """
    cells, claimed, cursor = [], np.zeros(n, bool), start
    while cursor < n:
        if claimed[cursor] or (cursor - start) % unit:
            cursor += 1
            continue
        b0, taken = cursor, None
        for L in lengths:
            if b0 + L > n:
                break
            for c in range(start % unit, n - L + 1, unit):
                if abs(c - b0) < L or claimed[c:c + L].any():
                    continue
                if accept(b0, c, L):
                    taken = L
                    break
            if taken:
                break
        if taken is None:
            cursor += unit
            continue
        L, occ = taken, []
        for c in range(start % unit, n - L + 1, unit):
            if claimed[c:c + L].any() or any(abs(c - o) < L for o in occ + [b0]):
                continue
            if accept(b0, c, L):
                occ.append(c)
        cells.append({"b0": b0, "L": L, "occ": sorted(occ)})
        for c in [b0] + occ:
            claimed[c:min(n, c + L)] = True
        cursor = b0 + unit
    return cells


def cells_greedy(S, M, n, start, accept, seed=4, lengths=LENGTHS, unit=UNIT):
    """**Le bloc de quatre qui revient le plus, d'abord.** Louis, 2026-08-07.

    Le balayage habituel part de la gauche et prend la première ancre qui
    trouve un écho — donc une correspondance accidentelle en début de morceau
    verrouille la grille pour tout le reste, et la vraie section, celle qui
    revient six fois, doit se contenter de ce qui traîne. L'ordre de lecture
    décide de la structure, ce qui n'a aucune raison musicale.

    Ici on inverse : on compte, pour chaque bloc de quatre mesures possible,
    combien de fois il revient ailleurs, et **le plus répété se sert le
    premier**. Il prend toutes ses places, on recompte sur ce qui reste, et on
    recommence tant qu'un bloc de quatre revient. Ce qui reste après passe par
    le balayage habituel, qui n'a plus que les miettes à trier.
    """
    claimed = np.zeros(n, bool)
    cells = []
    while True:
        best = None
        for b0 in range(start % unit, n - seed + 1, unit):
            if claimed[b0:b0 + seed].any():
                continue
            occ = []
            for c in range(start % unit, n - seed + 1, unit):
                if claimed[c:c + seed].any() or abs(c - b0) < seed:
                    continue
                if any(abs(c - o) < seed for o in occ):
                    continue
                if accept(b0, c, seed):
                    occ.append(c)
            if occ and (best is None or len(occ) > len(best[1])):
                best = (b0, occ)
        if best is None:
            break
        b0, occ = best
        cells.append({"b0": b0, "L": seed, "occ": sorted(occ)})
        for c in [b0] + occ:
            claimed[c:min(n, c + seed)] = True
    # ce qui n'a pas été pris repasse par le balayage gauche-droite habituel
    cursor = start
    while cursor < n:
        if claimed[cursor] or (cursor - start) % unit:
            cursor += 1
            continue
        b0, taken = cursor, None
        for L in lengths:
            if b0 + L > n or claimed[b0:b0 + L].any():
                break
            for c in range(start % unit, n - L + 1, unit):
                if abs(c - b0) < L or claimed[c:c + L].any():
                    continue
                if accept(b0, c, L):
                    taken = L
                    break
            if taken:
                break
        if taken is None:
            cursor += unit
            continue
        L, occ = taken, []
        for c in range(start % unit, n - L + 1, unit):
            if claimed[c:c + L].any() or any(abs(c - o) < L for o in occ + [b0]):
                continue
            if accept(b0, c, L):
                occ.append(c)
        cells.append({"b0": b0, "L": L, "occ": sorted(occ)})
        for c in [b0] + occ:
            claimed[c:min(n, c + L)] = True
        cursor = b0 + unit
    return sorted(cells, key=lambda e: e["b0"])


def variants(S, M, n, hs, ms):
    """Les trois critères, comme trois fonctions `accept`."""
    def absolute(b0, c, L):
        return max(HS.diag_match(S, b0, c, L),
                   float(np.mean([M[b0 + k, c + k] for k in range(L)]))) >= 0.90

    def relative(b0, c, L):
        h = hs.get(L)
        m = ms.get(L)
        ok_h = h is not None and HS.diag_match(S, b0, c, L) >= h["thr"]
        ok_m = m is not None and float(np.mean([M[b0 + k, c + k]
                                                for k in range(L)])) >= m["thr"]
        return ok_h or ok_m

    def priority(b0, c, L):
        h, m = hs.get(L), ms.get(L)
        if h is not None and h["contrast"] >= CN.DEAD:
            return HS.diag_match(S, b0, c, L) >= h["thr"]
        if m is not None:
            return float(np.mean([M[b0 + k, c + k] for k in range(L)])) >= m["thr"]
        return False

    return [("actuel · 0.90 pour tous", absolute),
            ("relatif · le décile du morceau", relative),
            ("priorité · l'harmonie d'abord", priority)]


def assemble(cells, n):
    """Les cellules deviennent des sections nommées, par les fusions déjà en place."""
    ch = HY.chain_of([{"L": e["L"], "b0": e["b0"], "occ": e["occ"]} for e in cells], n)
    for fn in (HY.merge_two_to_four, HY.merge_always, HY.merge_runs):
        ch, _ = fn(ch)
    secs, _ = HY.sections_of(ch)
    return [{**x, "kind": "bloc" if x["letter"] != "·" else "reste", "rep": 2}
            for x in secs]


def profile(S, M, n, b0, L, unit=UNIT):
    """Le profil des pics : ce que l'ancre `b0` trouve partout ailleurs.

    C'est LE plot qui rend la décision visible. Pour chaque position `c` du
    morceau, à quel point le bout [b0, b0+L) y ressemble — donc les pics sont
    les reprises. Poser le 0.90 absolu et le seuil du morceau comme deux lignes
    horizontales sur cette courbe montre directement ce que chacun laisse
    passer : sur Let It Be la ligne 0.90 rase le sol et tout devient un pic ;
    sur Bein Green elle passe entre les pics et le fond, elle fait son travail.
    """
    cs = list(range(0, n - L + 1, unit))
    h = np.array([HS.diag_match(S, b0, c, L) for c in cs])
    m = np.array([float(np.mean([M[b0 + k, c + k] for k in range(L)])) for c in cs])
    return np.array(cs, float), h, m


def song(stem):
    S, n, grid = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(grid, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)

    hs = scales(S, n, LENGTHS, CN.FLOOR_H)
    ms = scales(M, n, LENGTHS, CN.FLOOR_M)
    vv_ = variants(S, M, n, hs, ms)
    runs = []
    for name, acc in vv_:
        cells = cells_with(S, M, n, vstart, acc)
        runs.append({"name": name, "cells": cells, "secs": assemble(cells, n)})
    # l'idée de Louis, sur les deux critères qui comptent : commencer par le
    # bloc de quatre le plus répété au lieu de balayer de gauche à droite
    for name, acc in (vv_[0], vv_[1]):
        cells = cells_greedy(S, M, n, vstart, acc)
        runs.append({"name": f"le 4-mesures le + répété d'abord · {name.split(' · ')[0]}",
                     "cells": cells, "secs": assemble(cells, n)})

    # Les ancres dont on montre le profil : celles de la variante « relatif »,
    # complétées par celles de l'actuelle, sur la grille de deux mesures.
    anchors = []
    for r in runs:
        for e in r["cells"]:
            if not any(a["b0"] == e["b0"] for a in anchors):
                anchors.append({"b0": e["b0"], "L": e["L"]})
    anchors = sorted(anchors, key=lambda a: a["b0"])[:8]
    profs = [{**a, "xy": profile(S, M, n, a["b0"], a["L"])} for a in anchors]

    heights = [2.2, 0.30] + [0.72] * max(1, len(profs))
    for r in runs:
        heights += [0.30] + [0.38] * max(1, len(r["cells"])) + [0.50]
    Hh = sum(heights) + 1.5
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, Hh),
                            gridspec_kw={"height_ratios": heights, "hspace": .30})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .34 / Hh, bottom=.62 / Hh)
    axs[0].imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
                  aspect="auto", interpolation="nearest")
    axs[0].set_ylabel("mesure", fontsize=7.4); axs[0].tick_params(labelsize=6.2)
    axs[0].set_title("HARMONIE", fontsize=8, color="#2a6fb0", loc="left", pad=3)

    axs[1].axis("off")
    axs[1].set_title("LES PICS — chaque ancre glissée sur le morceau. "
                     "rouge = 0.90 absolu · bleu = seuil harmonie du morceau · "
                     "violet = seuil chant",
                     fontsize=8.4, color="#8a2b2b", loc="left", pad=1)
    k = 2
    if not profs:
        axs[k].axis("off"); k += 1
    for p in profs:
        ax = axs[k]
        cs, h, m = p["xy"]
        th = hs.get(p["L"], {}).get("thr")
        tm = ms.get(p["L"], {}).get("thr")
        ax.plot(mid(cs), h, color="#2a6fb0", lw=1.2, label="harmonie")
        ax.plot(mid(cs), m, color="#7c3aed", lw=1.0, alpha=.85)
        ax.axhline(0.90, color="#b3261e", lw=1.3, ls=(0, (4, 2)))
        if th:
            ax.axhline(th, color="#2a6fb0", lw=1.0, ls=(0, (1, 2)))
        if tm:
            ax.axhline(tm, color="#7c3aed", lw=1.0, ls=(0, (1, 2)))
        ax.axvline(edge(p["b0"]), color="#111", lw=1.3)
        # ce que chaque règle retient : au-dessus du 0.90 rouge, au-dessus du bleu
        for c, v in zip(cs, h):
            if abs(c - p["b0"]) < p["L"]:
                continue
            if v >= 0.90:
                ax.plot([mid(c)], [v], "o", ms=3.4, color="#b3261e", alpha=.8)
            if th and v >= th:
                ax.plot([mid(c)], [v], "o", ms=6, mfc="none",
                        mec="#2a6fb0", mew=1.3)
        ax.set_xlim(0, n); ax.set_ylim(0, 1.06); ax.set_yticks([0, .9])
        ax.tick_params(labelsize=5.4)
        ax.set_ylabel(f"ancre mes. {p['b0']+1}\n{p['L']} mes.", fontsize=6,
                      rotation=0, ha="right", va="center", color="#4a4436")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        k += 1
    for r in runs:
        axs[k].axis("off")
        axs[k].set_title(f"{r['name']}  —  {len(r['cells'])} mini-sections",
                         fontsize=8.4, color="#8a2b2b", loc="left", pad=1)
        k += 1
        if not r["cells"]:
            axs[k].axis("off"); k += 1
        for i, e in enumerate(r["cells"]):
            ax, col = axs[k], COLS[i % len(COLS)]
            for c in [e["b0"]] + e["occ"]:
                ax.add_patch(plt.Rectangle((c, .16), e["L"], .68, facecolor=col,
                                           edgecolor=INK, lw=.7))
            ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
            ax.set_ylabel(f"{e['L']} mes. · ×{1+len(e['occ'])}", fontsize=6,
                          rotation=0, ha="right", va="center", color=col)
            for sp in ax.spines.values():
                sp.set_color("#e5dcc6")
            k += 1
        B8.strip(axs[k], r["secs"], n, "sections")
        k += 1
    axs[-1].set_xticks(range(0, n + 1, 4)); axs[-1].tick_params(labelsize=6.2)
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64_fixed(fig)

    fmt = lambda x: " ".join(
        ("?" if s["kind"] == "reste" else s["letter"]) + f"[{s['b0']+1}-{s['b1']+1}]"
        for s in x)
    srow = "".join(
        f"<tr><td>{L}</td><td>{hs[L]['share90']*100:.0f} %</td>"
        f"<td>{hs[L]['thr']:.3f}</td><td>{ms[L]['thr']:.3f}</td></tr>"
        for L in sorted(hs) if L in ms)
    rrow = "".join(
        f"<tr><td>{r['name']}</td><td>{len(r['cells'])}</td>"
        f"<td>{', '.join(str(e['L']) for e in r['cells']) or '—'}</td>"
        f"<td>{max((1+len(e['occ']) for e in r['cells']), default=0)}</td>"
        f"<td>{len({s['letter'] for s in r['secs'] if s['kind']=='bloc'})}</td></tr>"
        for r in runs)
    lines = "<br>".join(f"<b>{r['name']}</b> : {fmt(r['secs'])}" for r in runs)
    gridjs = "[" + ",".join(f"{x:.3f}" for x in grid) + "]"
    btns = "".join(
        f"<button class=blk data-p='[{s['b0']},{s['b1']+1}]'>{s['letter']}"
        f"<small>{s['b0']+1}</small></button>"
        for s in runs[1]["secs"] if s["kind"] == "bloc")
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures ·
harmonie {'MUETTE' if hs.get(8, {}).get('contrast', 1) < CN.DEAD else 'utile'}
(contraste {hs.get(8, {}).get('contrast', float('nan')):.3f})</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>sections de la variante « relatif »</span>{btns}</div>
<div class=cols>
<table><tr><th>longueur</th><th>paires ≥0.90</th><th>seuil harmonie</th>
<th>seuil chant</th></tr>{srow}</table>
<table><tr><th>variante</th><th>mini-sections</th><th>longueurs</th>
<th>occ. max</th><th>lettres</th></tr>{rrow}</table></div>
<p class=verdict>{lines}</p></section>"""


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
    out = HERE / "harmonia_min/state/reports/mini_threshold.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Le seuil, absolu ou propre au morceau ?</title><style>
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
.cols{{display:flex;flex-wrap:wrap;gap:14px;margin-top:8px}}
table{{border-collapse:collapse;font-size:12.5px}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
.verdict{{font:500 11.5px ui-monospace,monospace;background:#f7f3e9;
  border-radius:8px;padding:9px 11px;margin:10px 0 0;line-height:1.9}}
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
<h1>Le seuil, absolu ou propre au morceau ?</h1>
<div class=lede>Une mini-section existe quand un bout de deux ou quatre mesures
<b>revient ailleurs</b>. « Revient » veut dire : similarité ≥ 0.90 — le même
nombre pour tous les morceaux. Voici ce qu'il vaut vraiment, en part de
<b>toutes</b> les paires de deux mesures qui le franchissent :<br><br>
The Walk <b>62 %</b> · Let It Be <b>52 %</b> · Norah 28 % · This Love 22 % ·
Grenade 17 % · Bein Green <b>9 %</b><br><br>
Sur The Walk, « ce bout revient ailleurs » est vrai de presque n'importe quel
bout : le morceau tourne sur un accord, le critère ne trie plus rien. C'est
pourquoi la première cellule de Let It Be sort avec vingt-deux occurrences —
elle « correspond » à la moitié du morceau.<br><br>
<b>Les courbes du haut sont la décision elle-même.</b> Chaque ancre est glissée
sur tout le morceau : les <b>pics</b> sont les endroits où elle revient. La ligne
<b>rouge</b> est le 0.90 absolu d'aujourd'hui, la <b>bleue pointillée</b> le
seuil de l'harmonie dans ce morceau, la <b>violette</b> celui du chant. Les
points rouges sont ce que la règle actuelle retient, les cercles bleus ce que la
règle relative retient. Là où la ligne rouge rase le sol, tout devient un pic.
<br><br>
Trois variantes, même machinerie, seul le critère change.
<b>actuel</b> : 0.90 pour tous. <b>relatif</b> : chaque voie contre son propre
neuvième décile dans ce morceau, à cette longueur, plancher 0.90 — donc rien ne
bouge là où l'harmonie sépare déjà bien. <b>priorité</b> : mêmes seuils, mais
l'harmonie décide seule quand elle discrimine, le chant ne parle que là où elle
est morte.<br><br>
Les boutons jouent les sections de la variante « relatif ». Rien n'est encore
changé dans le pipeline.</div>
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
