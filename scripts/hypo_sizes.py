"""DÉMO — trois hypothèses de taille (2, 4, 8) et la répétition n'a plus à être immédiate.

    python scripts/hypo_sizes.py [<stem> ...]  ->  /reports/hypo_sizes.html

Louis, 2026-08-05, en regardant Bein Green :

  « Des fois on a une grosse section qui prend le dessus et empêche une petite
    section d'exister, rendant un mauvais découpage. Ce qu'il aurait dû se
    passer, c'est que la grosse section B devrait être découpée en plus petites
    sous-sections qui la composent, même si celles-ci n'ont pas de répétitions
    directes. Plutôt que de chercher le pattern de répétition minimal via notre
    technique, ce serait, pour chaque bloc de début qu'on rencontre, émettre
    plusieurs hypothèses dans l'ordre : le pattern est de taille 2, sinon 4,
    sinon 8. Et surtout IMPORTANT on relaxe l'hypothèse que le pattern doit se
    répéter à la suite — s'il se répète plus tard c'est déjà bien. »

Plus sa règle de grille du même jour : « la granularité minimale est de deux
mesures, on redéfinit un grid » — donc une ancre, une longueur et un placement
ne tombent que sur la grille de 2 mesures.

CE QUI CHANGE, EXACTEMENT. La règle d'aujourd'hui demande à un bloc de revenir
IMMÉDIATEMENT : à l'ancre b, elle compare les d mesures qui commencent en b avec
les d mesures qui commencent en b+d. Un motif qui joue ici et revient trente
mesures plus loin marque zéro. C'est ce qui laisse une grosse section avaler
tout le morceau : la seule distance qui « revient tout de suite » est la grande.

La nouvelle règle glisse le bloc sur TOUT le morceau et se contente d'une
répétition, où qu'elle soit. Et elle essaie 2 avant 4 avant 8, donc la petite
gagne dès qu'elle existe.
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
from pattern_lanes import (load, fig2b64_fixed, alpha_of, PLOT_L, PLOT_R,
                           COLS, INK, SOLID)                       # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402

SIZES = (2, 4, 8)     # les hypothèses, dans l'ordre
UNIT = 2              # la grille : rien ne commence sur une mesure impaire
MATCH = 0.95          # « ça se répète » = le verrou qu'il a déjà validé
MAX_CELLS = 12
DEFAULT = ["bein_green", "maroon_5_she_will_be_loved_official_music_video",
           "mayer_hawthorne_the_walk", "bruno_mars_grenade_official_music_video",
           "maroon_5_this_love", "norah_jones_don_t_know_why",
           "let_it_be_remastered_2009"]


def build_hypo(S, n, *, sizes=SIZES, unit=UNIT, match=MATCH, max_cells=MAX_CELLS):
    """L'étage 1 de Louis, version 2026-08-05 soir."""
    cells, claimed, cursor = [], np.zeros(n, bool), 0
    while cursor < n and len(cells) < max_cells:
        if claimed[cursor] or cursor % unit:
            cursor += 1
            continue
        b0, chosen = cursor, None
        for L in sizes:                       # 2, puis 4, puis 8
            if b0 + L > n or claimed[b0:b0 + L].any():
                continue
            curve = HS.slide(S, L, b0)
            cand = [c for c in range(0, len(curve), unit)
                    if abs(c - b0) >= L                       # pas de recouvrement
                    and float(curve[c]) >= match
                    and not claimed[c:min(n, c + L)].any()]
            if cand:                          # ça se répète QUELQUE PART : ça suffit
                chosen = (L, curve, cand)
                break
        if chosen is None:
            cursor += unit
            continue
        L, curve, cand = chosen
        occ, taken = [], claimed.copy()
        taken[b0:min(n, b0 + L)] = True
        for c in sorted(cand, key=lambda c: -float(curve[c])):
            if taken[c:min(n, c + L)].any():
                continue
            occ.append(c)
            taken[c:min(n, c + L)] = True
        occ.sort()
        cells.append({"L": L, "b0": b0, "curve": curve, "occ": occ,
                      "lag": L, "run": L})
        for s in [b0] + occ:
            claimed[s:min(n, s + L)] = True
        cursor = b0 + unit

    # les trous : même passe qu'aujourd'hui, mais calée sur la grille de 2
    for g0, g1 in HS._free_runs(claimed, n):
        g0 -= g0 % unit
        L = g1 - g0 + 1
        L += (-L) % unit
        if L < min(sizes) or g0 + L > n:
            continue
        curve = HS.slide(S, L, g0)
        occ, taken = [], claimed.copy()
        for c in sorted(range(0, len(curve), unit), key=lambda c: -float(curve[c])):
            if c == g0 or float(curve[c]) < match or taken[c:min(n, c + L)].any():
                continue
            occ.append(c)
            taken[c:min(n, c + L)] = True
        if not occ:
            continue
        occ.sort()
        cells.append({"L": L, "b0": g0, "curve": curve, "occ": occ,
                      "lag": L, "run": L, "from_gap": True})
        for s in [g0] + occ:
            claimed[s:min(n, s + L)] = True
    return cells


def chain_of(cells, n):
    """La suite de passages : [{sym, b0, b1}], les trous compris, dans l'ordre."""
    owner = [None] * n
    for i, e in enumerate(cells):
        for c in [e["b0"]] + e["occ"]:
            for b in range(c, min(n, c + e["L"])):
                owner[b] = chr(ord("a") + i)
    out, b, hole = [], 0, 0
    while b < n:
        z = b
        while z + 1 < n and owner[z + 1] == owner[b]:
            z += 1
        if owner[b] is None:                 # un trou n'est pas un motif : il
            hole += 1                        # ne fusionne avec rien et ne
            out.append({"sym": f"?{hole}", "b0": b, "b1": z, "hole": True})
        else:
            out.append({"sym": owner[b], "b0": b, "b1": z, "hole": False})
        b = z + 1
    return out


def merge_always(chain, min_pairs=2, both_ways=False):
    """Louis, 2026-08-05 : « on va fusionner 2 sections si elles se suivent toujours ».

    Pris au pied de la lettre — UN SEUL SENS : X et Y fusionnent dès que X est
    toujours suivi de Y. J'avais d'abord exigé les deux sens (X toujours suivi de
    Y **et** Y toujours précédé de X), par prudence ; mesuré sur les sept
    morceaux, sa formulation gagne et la mienne perd. Bein Green :

        brut      a b c d a b c d e f e g h b c d e f e g h b c d
        deux sens a bcd a bcd e f e gh bcd e f e gh bcd      5 lettres
        UN SENS   abcd abcd efeghbcd efeghbcd                2 lettres

    Deux lettres, c'est la forme du morceau : deux A de dix mesures puis deux B
    de seize. Les deux sens s'arrêtent trop tôt parce que `b` arrive aussi après
    `h`, ce qui bloque `a b` — alors que `a` est bel et bien toujours suivi de
    `b`. Sur les six autres morceaux les deux versions donnent le même nombre de
    lettres, donc rien n'est perdu ailleurs.

    Un `X` en toute fin de morceau n'a pas de voisin : on ne lui en tient pas
    rigueur. On répète jusqu'au point fixe, donc `a b`, puis `ab c`, puis `abc d`.
    `both_ways=True` garde l'ancienne version stricte, pour comparaison.
    """
    chain = [dict(c) for c in chain]
    log = []
    while True:
        nxt, prv = {}, {}
        for i, c in enumerate(chain):
            if c["hole"]:
                continue
            a = chain[i + 1]["sym"] if i + 1 < len(chain) else None
            p = chain[i - 1]["sym"] if i > 0 else None
            nxt.setdefault(c["sym"], []).append(a)
            prv.setdefault(c["sym"], []).append(p)
        best = None
        for x, outs in nxt.items():
            seen = [o for o in outs if o is not None]
            if len(seen) < min_pairs or len(set(seen)) != 1:
                continue
            y = seen[0]
            if y == x or y.startswith("?"):
                continue
            if both_ways:
                ins = [p for p in prv.get(y, []) if p is not None]
                if len(ins) < min_pairs or set(ins) != {x}:
                    continue
            best = (x, y)
            break
        if best is None:
            return chain, log
        x, y = best
        log.append(f"{x} + {y}")
        out, i = [], 0
        while i < len(chain):
            if (chain[i]["sym"] == x and i + 1 < len(chain)
                    and chain[i + 1]["sym"] == y):
                out.append({"sym": x + y, "b0": chain[i]["b0"],
                            "b1": chain[i + 1]["b1"], "hole": False})
                i += 2
            else:
                out.append(chain[i])
                i += 1
        chain = out


def merge_runs(chain, min_len=2, min_reps=2):
    """Les SUCCESSIONS qui reviennent — Louis, 2026-08-05 soir :

      « Il faut repérer des patterns de succession de ces mini-sections. »

    La fusion par paires (« X toujours suivi de Y ») attrape les chaînes, pas
    les refrains : sur Norah la suite est `a b a c a c a c a b d e d …`, aucune
    cellule n'est toujours suivie de la même, donc rien ne fusionne — alors que
    `a c` revient quatre fois, à l'œil.

    Ici on cherche donc, dans la suite, la SOUS-SUITE qui explique le plus de
    passages : on essaie toutes les sous-suites d'au moins `min_len` symboles
    qui reviennent au moins `min_reps` fois sans se chevaucher, on garde celle
    qui couvre le plus de passages (longueur × nombre de reprises), on la
    remplace par un seul symbole, et on recommence. Les trous ne peuvent jamais
    entrer dans une sous-suite : ils ne sont pas de la musique reconnue.

    C'est la même idée qu'un compresseur qui remplace un motif répété par un
    nom : la structure qui reste est celle qui a le mieux résisté.
    """
    chain = [dict(c) for c in chain]
    log = []
    while True:
        syms = [c["sym"] for c in chain]
        holes = [c["hole"] for c in chain]
        best = None
        for L in range(len(syms) // 2, min_len - 1, -1):
            for i in range(len(syms) - L + 1):
                if any(holes[i:i + L]):
                    continue
                pat = syms[i:i + L]
                hits, j = [], 0
                while j <= len(syms) - L:
                    if syms[j:j + L] == pat and not any(holes[j:j + L]):
                        hits.append(j)
                        j += L
                    else:
                        j += 1
                if len(hits) >= min_reps:
                    cover = L * len(hits)
                    if best is None or cover > best[0]:
                        best = (cover, L, tuple(pat), hits)
            if best:
                break                      # la plus longue d'abord
        if best is None:
            return chain, log
        _, L, pat, hits = best
        name = "".join(pat)
        log.append(f"{' '.join(pat)}  ×{len(hits)}")
        out, i, hits = [], 0, set(hits)
        while i < len(chain):
            if i in hits:
                out.append({"sym": name, "b0": chain[i]["b0"],
                            "b1": chain[i + L - 1]["b1"], "hole": False})
                i += L
            else:
                out.append(chain[i])
                i += 1
        chain = out


def sections_of(chain):
    """La chaîne devient des sections : une lettre par symbole distinct."""
    letters, out = {}, []
    for c in chain:
        if c["hole"]:
            out.append({"b0": c["b0"], "b1": c["b1"], "letter": "·"})
            continue
        if c["sym"] not in letters:
            letters[c["sym"]] = chr(ord("A") + len(letters))
        out.append({"b0": c["b0"], "b1": c["b1"], "letter": letters[c["sym"]]})
    return out, letters


def strip(ax, secs, n, label):
    seen = {}
    for s in secs:
        seen.setdefault(s["letter"], COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   color=seen[s["letter"]]))
        if s["b1"] - s["b0"] >= 1:
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                    va="center", color="#fff", fontsize=7, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.1)
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", color=INK)


def figure(S, n, cells, secs_run, secs_fus, secs_new, secs_old):
    k = len(cells)
    heights = [4.6] + [0.62] * k + [0.70, 0.62, 0.55, 0.55]
    H = 4.6 + .62 * k + 3.3
    fig, axs = plt.subplots(len(heights), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": heights, "hspace": .22})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .30 / H, bottom=.55 / H)

    ax = axs[0]
    ax.imshow(S, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
              vmin=float(np.percentile(S, 5)), vmax=float(np.percentile(S, 99)),
              aspect="auto", interpolation="nearest")
    ax.set_ylabel("mesure", fontsize=8)
    ax.tick_params(labelsize=6.6)

    for i, e in enumerate(cells):
        ax, col, L = axs[1 + i], COLS[i % len(COLS)], e["L"]
        for c in [e["b0"]] + e["occ"]:
            s = float(e["curve"][c]) if c < len(e["curve"]) else 1.0
            ax.add_patch(plt.Rectangle((c, .14), L, .72, facecolor=col,
                                       alpha=alpha_of(s), edgecolor=INK, lw=1.0))
            ax.text(c + L / 2, .5, f"{s:.2f}", ha="center", va="center",
                    fontsize=5.4, rotation=90 if L <= 2 else 0,
                    color="#fff" if alpha_of(s) > .55 else INK)
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(f"motif {i+1}\n{L} mes."
                      + ("\n(trou)" if e.get("from_gap") else ""),
                      fontsize=6.4, rotation=0, ha="right", va="center", color=col)
        for sp in ax.spines.values():
            sp.set_color("#ddd5c0")

    strip(axs[-4], secs_run, n, "SUCCESSIONS\nrépétées")
    strip(axs[-3], secs_fus, n, "paires\n« toujours suivi »")
    strip(axs[-2], secs_new, n, "aucune\nfusion")
    strip(axs[-1], secs_old, n, "règle\nactuelle")
    axs[-1].set_xlim(0, n)
    axs[-1].set_xticks(range(0, n + 1, 4))
    axs[-1].tick_params(labelsize=6.4)
    axs[-1].set_xlabel("mesure", fontsize=8)
    return fig2b64_fixed(fig)


def song(stem):
    S, n, grid = load(stem)
    old_cells, _ = HS.build_cells(S, n)
    old = HS.sections_from(S, n, old_cells)
    cells = build_hypo(S, n)
    new = HS.sections_from(S, n, cells)
    ch0 = chain_of(cells, n)
    ch1, log = merge_always(ch0)                       # un seul sens : sa règle
    ch2, log2 = merge_always(ch0, both_ways=True)      # les deux sens, plus stricte
    ch3, log3 = merge_runs(ch1)                        # puis les successions
    fus, letters = sections_of(ch1)
    run, letters3 = sections_of(ch3)
    _, letters2 = sections_of(ch2)
    img = figure(S, n, cells, run, fus, new, old)

    def fmt(secs):
        return " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in secs)

    # la SUITE DE CELLULES, mesure par mesure — c'est la matière de l'étage 2 :
    # les sections sont les répétitions de SÉQUENCES là-dedans, pas des cellules
    owner = ["·"] * n
    for i, e in enumerate(cells):
        for c in [e["b0"]] + e["occ"]:
            for b in range(c, min(n, c + e["L"])):
                owner[b] = chr(ord("a") + i)
    seq, b = [], 0
    while b < n:
        L = 1
        while b + L < n and owner[b + L] == owner[b] and L < 8:
            L += 1
        seq.append(owner[b])
        b += L
    chain = " ".join(seq)
    chain2 = " ".join(("·" if c["hole"] else c["sym"]) for c in ch1)
    chain3 = " ".join(("·" if c["hole"] else c["sym"]) for c in ch2)
    chain4 = " ".join(("·" if c["hole"] else c["sym"]) for c in ch3)

    rows = "".join(
        f"<tr><td><span class=dot style='background:{COLS[i%len(COLS)]}'></span>"
        f"motif {i+1}</td><td>{e['L']}</td><td>{e['b0']+1}</td>"
        f"<td>{' '.join(str(o+1) for o in [e['b0']] + e['occ'])}</td>"
        f"<td>{'trou' if e.get('from_gap') else 'hypothèse ' + str(e['L'])}</td></tr>"
        for i, e in enumerate(cells))
    def btns(items):
        out = ""
        for c in items:
            col = "#8a8371" if c["hole"] else COLS[
                (ord(c["sym"][0]) - ord("a")) % len(COLS)]
            out += (f"<button class=blk style='border-color:{col};color:{col}' "
                    f"data-p='[{c['b0']},{c['b1']+1}]'>"
                    f"{'·' if c['hole'] else c['sym']}"
                    f"<small>{c['b0']+1}–{c['b1']+1}</small></button>")
        return out

    mini_btns, run_btns = btns(ch0), btns(ch3)
    gridjs = "[" + ",".join(f"{t:.3f}" for t in grid) + "]"
    return f"""<section data-grid='{gridjs}' data-audio="/audio/{stem}.m4a">
<h2>{stem.replace('_',' ').title()}<span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<table><tr><th>motif</th><th>taille</th><th>ancre</th><th>placements</th>
<th>venu de</th></tr>{rows}</table>
<div class=lane><span class=lab>mini-sections</span>{mini_btns}</div>
<div class=lane><span class=lab>après successions</span>{run_btns}</div>
<p class=verdict><b>la suite de cellules</b> :
<span class=chain>{chain}</span><br>
<b>1 — paires « toujours suivi de »</b>{' : ' + ', '.join(log) if log else ' : aucune'}
<span class=chain>{chain2}</span><br>
<b>2 — successions répétées</b>{' : ' + ' | '.join(log3) if log3 else ' : aucune'}
<span class=chain>{chain4}</span><br>
<span class=sub>variante stricte (les deux sens exigés) — {len(letters2)} lettres :
{chain3}</span><br><br>
<b>SUCCESSIONS</b> — {len(letters3)} lettres : {fmt(run)}<br>
<b>paires seules</b> — {len(letters)} lettres : {fmt(fus)}<br>
<b>aucune fusion</b> — {len({s['letter'] for s in new})} lettres : {fmt(new)}<br>
<b>règle actuelle</b> — {len({s['letter'] for s in old})} lettres :
{fmt(old)}</p></section>""", (stem, len({s['letter'] for s in old}),
                              len({s['letter'] for s in new}), len(letters),
                              len(log))


def main():
    stems = sys.argv[1:] or DEFAULT
    body, summ = "", []
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            html, row = song(st)
        except Exception as exc:
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
            continue
        body += html
        summ.append(row)
        print(f"  ok {st}  {row[1]} -> {row[2]} -> {row[3]} lettres "
              f"({row[4]} fusions)")
    head = "".join(f"<tr><td>{s[0].replace('_',' ').title()}</td>"
                   f"<td>{s[1]}</td><td>{s[2]}</td><td class=win>{s[3]}</td>"
                   f"<td>{s[4]}</td></tr>" for s in summ)
    out = HERE / "harmonia_min/state/reports/hypo_sizes.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>2, sinon 4, sinon 8</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
th.win,td.win{{background:#e4f0e8;font-weight:700}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.chain{{font:600 13px ui-monospace,monospace;color:#8a2b2b;letter-spacing:.06em}}
.verdict{{font-size:12.5px;background:#f7f3e9;border-radius:8px;padding:9px 11px;margin:10px 0 0;line-height:1.7}}
.plot{{position:relative;margin-bottom:8px}} .plot img{{margin:0}}
.cur{{position:absolute;top:0;bottom:0;width:2px;background:#111;opacity:.8;
  display:none;pointer-events:none;box-shadow:0 0 0 1px rgba(255,255,255,.55)}}
.hit{{position:absolute;top:0;bottom:0;cursor:crosshair}}
.bar{{display:flex;align-items:center;gap:10px;margin:0 0 10px}}
.pp{{width:40px;height:40px;border-radius:50%;border:1px solid #d8cfb4;
  background:#f7f3e9;font-size:14px;cursor:pointer;flex:none}}
.pos{{font:600 12.5px ui-monospace,monospace}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
.lane{{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:0 0 6px}}
.lab{{font:600 11px system-ui;color:#8a8371;width:112px;flex:none}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:7px;
  padding:4px 7px;cursor:pointer;font:700 12px ui-monospace,monospace}}
button.blk small{{display:block;font:500 9px ui-monospace,monospace;
  color:#a89f8c;margin-top:1px}}
button.blk.on{{color:#fff !important}} button.blk.on small{{color:#f3ece0}}
</style></head><body><div class=wrap>
<h1>2, sinon 4, sinon 8</h1>
<div class=lede>Pour chaque bloc de début rencontré, trois hypothèses dans
l'ordre : <b>le motif fait 2 mesures</b>, sinon 4, sinon 8. La première qui tient
gagne, donc la petite passe avant la grande.<br><br>
Et le changement qui compte : <b>le motif n'a plus à revenir tout de suite</b>.
Aujourd'hui on compare les d mesures qui commencent ici avec les d mesures
suivantes — un motif qui revient trente mesures plus loin marque zéro, et c'est
exactement ce qui laisse une grosse section avaler le morceau. Ici on glisse le
bloc sur toute la chanson et une seule répétition, n'importe où, suffit.<br><br>
Avec ta règle de grille : rien ne commence sur une mesure impaire, tout est calé
sur la grille de deux mesures.<br><br>
Sous chaque morceau, les deux découpages : la nouvelle règle et celle
d'aujourd'hui, sur le même axe.</div>
<section><h2>Vue d'ensemble</h2>
<table><tr><th>morceau</th><th>lettres<br>aujourd'hui</th><th>lettres<br>2/4/8 seul</th>
<th class=win>lettres<br>2/4/8 + fusion</th><th>fusions<br>faites</th></tr>
{head}</table></section>
{body}</div>
<audio id=au preload=metadata playsinline></audio>
<script>
const au=document.getElementById("au");
let stopAt=null, raf=null, onBtn=null, live=null;
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
function clear(){{ if(onBtn){{ onBtn.classList.remove("on");
  onBtn.style.background="#f7f3e9"; onBtn=null; }} }}
function draw(){{
  if(!live) return;
  const G=live.G, n=G.length-1, L0=%L0%, W=%W%;
  let t=au.currentTime, f;
  if(t<=G[0]) f=0; else if(t>=G[n]) f=n; else {{
    let lo=0,hi=n; while(hi-lo>1){{const m=(lo+hi)>>1; G[m]<=t?lo=m:hi=m;}}
    f=lo+(t-G[lo])/(G[lo+1]-G[lo]); }}
  live.cur.style.display="block";
  live.cur.style.left="calc("+((L0+W*f/n)*100)+"% - 1px)";
  live.pos.textContent="mes. "+(Math.floor(f)+1)+" · "+fmt(au.currentTime);
}}
function tick(){{ draw();
  if(stopAt!=null && au.currentTime>=stopAt){{ au.pause(); stopAt=null; clear(); }}
  if(!au.paused) raf=requestAnimationFrame(tick); }}
au.addEventListener("play",()=>{{ if(live) live.pp.textContent="❚❚"; tick(); }});
au.addEventListener("pause",()=>{{ if(live) live.pp.textContent="▶";
  cancelAnimationFrame(raf); draw(); }});
function go(sec,t0,t1,btn){{
  if(live && live.sec!==sec){{ live.pp.textContent="▶"; live.cur.style.display="none"; }}
  live=sec._p; clear(); stopAt=t1;
  if(btn){{ onBtn=btn; btn.classList.add("on"); btn.style.background=btn.style.borderColor; }}
  if(au.getAttribute("src")!==sec.dataset.audio){{
    au.setAttribute("src",sec.dataset.audio); au.load(); }}
  const seek=()=>{{ try{{ au.currentTime=t0; }}catch(e){{}} draw(); }};
  if(au.readyState>=1) seek(); else au.addEventListener("loadedmetadata",seek,{{once:true}});
  au.play().catch(()=>clear());
}}
document.querySelectorAll("section[data-grid]").forEach(sec=>{{
  const G=JSON.parse(sec.dataset.grid), n=G.length-1;
  const cur=sec.querySelector(".cur"), hit=sec.querySelector(".hit"),
        pp=sec.querySelector(".pp"), pos=sec.querySelector(".pos");
  sec._p={{G:G,cur:cur,pos:pos,pp:pp,sec:sec}};
  hit.style.left=(%L0%*100)+"%"; hit.style.width=(%W%*100)+"%";
  hit.onclick=e=>{{ const r=hit.getBoundingClientRect();
    const f=n*(e.clientX-r.left)/r.width;
    const i=Math.max(0,Math.min(n-1,Math.floor(f)));
    go(sec, G[i]+(f-i)*(G[i+1]-G[i]), null, null); }};
  pp.onclick=()=>{{ if(au.paused||live!==sec._p) go(sec, G[0], null, null);
                    else au.pause(); }};
  sec.querySelectorAll("[data-p]").forEach(b=>{{
    const d=JSON.parse(b.dataset.p);
    b.onclick=()=>go(sec, G[d[0]], G[Math.min(n,d[1])], b); }});
}});
</script></body></html>""".replace("%L0%", str(PLOT_L)).replace("%W%", str(round(PLOT_R - PLOT_L, 6))))
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
