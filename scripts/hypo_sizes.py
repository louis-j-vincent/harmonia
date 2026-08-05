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


def figure(S, n, cells, secs_new, secs_old):
    k = len(cells)
    heights = [4.6] + [0.62] * k + [0.55, 0.55]
    H = 4.6 + .62 * k + 2.0
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

    strip(axs[-2], secs_new, n, "NOUVELLE\nrègle")
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
    img = figure(S, n, cells, new, old)

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

    rows = "".join(
        f"<tr><td><span class=dot style='background:{COLS[i%len(COLS)]}'></span>"
        f"motif {i+1}</td><td>{e['L']}</td><td>{e['b0']+1}</td>"
        f"<td>{' '.join(str(o+1) for o in [e['b0']] + e['occ'])}</td>"
        f"<td>{'trou' if e.get('from_gap') else 'hypothèse ' + str(e['L'])}</td></tr>"
        for i, e in enumerate(cells))
    return f"""<section><h2>{stem.replace('_',' ').title()}
<span class=sub>{n} mesures</span></h2>
<img src="data:image/png;base64,{img}">
<table><tr><th>motif</th><th>taille</th><th>ancre</th><th>placements</th>
<th>venu de</th></tr>{rows}</table>
<p class=verdict><b>la suite de cellules</b> (une lettre par passage) :
<span class=chain>{chain}</span><br><br>
<b>nouvelle règle</b> — {len({s['letter'] for s in new})} lettres :
{fmt(new)}<br><b>règle actuelle</b> — {len({s['letter'] for s in old})} lettres :
{fmt(old)}</p></section>""", (stem, len({s['letter'] for s in old}),
                              len({s['letter'] for s in new}),
                              [e["L"] for e in old_cells], [e["L"] for e in cells])


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
        print(f"  ok {st}  {row[1]}→{row[2]} lettres")
    head = "".join(f"<tr><td>{s[0].replace('_',' ').title()}</td>"
                   f"<td>{s[1]}</td><td>{s[2]}</td>"
                   f"<td>{s[3]}</td><td>{s[4]}</td></tr>" for s in summ)
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
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.chain{{font:600 13px ui-monospace,monospace;color:#8a2b2b;letter-spacing:.06em}}
.verdict{{font-size:12.5px;background:#f7f3e9;border-radius:8px;padding:9px 11px;margin:10px 0 0;line-height:1.7}}
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
<table><tr><th>morceau</th><th>lettres<br>aujourd'hui</th><th>lettres<br>nouvelle</th>
<th>tailles de motifs aujourd'hui</th><th>tailles de motifs nouvelle</th></tr>
{head}</table></section>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
