"""The four motif-search criteria, seen rather than tabulated.

« Montre-moi des plots, je ne fais PAS confiance à tes chiffres et tu ne devrais
pas non plus. »

Fair. One of the numbers in the table I gave him was measured with the SQUARE
while the others were on the DIAGONAL — re-measured properly, the "mean" row
goes from "4 entries, 72/80 bars" to "1 entry, 32/80". The table was comparing
two changes at once. Everything here runs on the diagonal, so the only thing
that differs between columns is the criterion.

Per song, per criterion: the SSM with every dictionary entry boxed in its own
colour, and the coverage strip underneath. What each criterion does to the song
is then a picture, not a claim.

    python scripts/criteria_visual.py   ->  /reports/criteria_visual.html
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
from ssm_rows_plot import fig2b64                              # noqa: E402
import harmonic_method as HM                                   # noqa: E402
from dictionary_harmonic import bar_grid, build, ENTRY_COLS    # noqa: E402

OUT = HERE / "harmonia_min/state/reports/criteria_visual.html"
INK = "#1c1c1c"
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why")]
CRITERIA = [
    ("mean", "moyenne sur le décalage",
     "l'ancien critère : le décalage dont la similarité moyenne est la plus forte"),
    ("total", "évidence totale au-dessus d'un seuil",
     "somme de ce qui dépasse le seuil — favorise les décalages courts, qui ont plus de paires"),
    ("run", "plus longue série, à tous les tours",
     "la plus longue suite consécutive de mesures fortes, dès le premier tour"),
    ("hybrid", "moyenne au tour 1, série ensuite",
     "ce qui tourne actuellement"),
]


def panel(S, n, segs, crit):
    entries, _ = build(S, n, criterion=crit)
    owner = -np.ones(n, int)
    for ei, e in enumerate(entries):
        owner[e["b0"]:min(n, e["b0"] + e["L"])] = ei
        for o in e["occ"]:
            owner[o:min(n, o + e["L"])] = ei

    fig, axs = plt.subplots(2, 1, figsize=(4.6, 5.4),
                            gridspec_kw={"height_ratios": [5, 1]})
    axs[0].imshow(S, cmap="RdYlBu_r", origin="lower",
                  vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    for ei, e in enumerate(entries):
        col = ENTRY_COLS[ei % len(ENTRY_COLS)]
        L = e["L"]
        for b in sorted(set([e["b0"]] + e["occ"])):
            axs[0].add_patch(plt.Rectangle((b - .5, e["b0"] - .5), L, L,
                                           fill=False, ec=col, lw=1.5))
    axs[0].set_xticks([]); axs[0].set_yticks([])
    for b in range(n):
        c = ENTRY_COLS[owner[b] % len(ENTRY_COLS)] if owner[b] >= 0 else "#e5dcc6"
        axs[1].add_patch(plt.Rectangle((b, 0), 1, 1, color=c))
    for sg in segs:
        axs[1].axvline(sg["b0"], color=INK, lw=.9)
    axs[1].set_xlim(0, n); axs[1].set_ylim(0, 1); axs[1].set_yticks([])
    axs[1].set_xlabel("mesure", fontsize=7)
    img = fig2b64(fig)
    return img, entries, int((owner >= 0).sum())


def main():
    body = ""
    for stem, title in SONGS:
        cap = bar_grid(stem)
        grid, segs = cap["grid"], cap["segs"]
        n = len(grid) - 1
        S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
        cells = ""
        for crit, nm, note in CRITERIA:
            img, entries, cov = panel(S, n, segs, crit)
            mot = ", ".join(str(e["L"]) for e in entries) or "—"
            hi = ' style="border:2px solid #8a2b2b"' if crit == "hybrid" else ""
            cells += (f'<div class=cell{hi}><b>{nm}</b>'
                      f'<div class=note>{note}</div>'
                      f'<img src="data:image/png;base64,{img}">'
                      f'<div class=meta>{len(entries)} entrée(s) · motifs de '
                      f'{mot} mesures · {cov}/{n} mesures couvertes</div></div>')
        body += (f'<section><h2>{title} <span class=sub>{n} mesures</span></h2>'
                 f'<div class=row>{cells}</div></section>')
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les quatre critères, en images</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1400px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 12px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.row{{display:flex;gap:12px;flex-wrap:wrap}}
.cell{{flex:1;min-width:260px;background:#f7f3e9;border-radius:10px;padding:9px 10px}}
.cell b{{font:700 13px system-ui}}
.note{{font-size:11.5px;color:#8a8371;margin:2px 0 6px;min-height:30px}}
.meta{{font:600 11.5px system-ui;color:#2a6fb0;margin-top:4px}}
img{{max-width:100%;border-radius:6px}}
</style></head><body><div class=wrap>
<h1>Les quatre critères de recherche de motif, en images</h1>
<div class=lede>Même morceau, même matrice, même règle de pics, même lecture
diagonale — <b>seul le critère de recherche change d'une colonne à l'autre</b>.
En haut la SSM avec chaque entrée du dictionnaire encadrée dans sa couleur, en
bas la bande de couverture (gris = aucune entrée ne le revendique, traits noirs =
les sections actuelles). Le cadre rouge est ce qui tourne.
<br><br><b>Correction :</b> dans le tableau que j'avais envoyé, la ligne
« moyenne » avait été mesurée avec le CARRÉ et les autres avec la diagonale — la
comparaison mélangeait deux changements. Ici tout est à la diagonale.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
