"""Plusieurs structures possibles, et les règles pour trancher entre elles.

    python scripts/structure_hypotheses.py [<stem> ...]
      -> /reports/structure_hypotheses.html

Louis, 2026-08-05 : « des fois on loupe des répétitions. Est-ce qu'on peut
relaxer la règle de recherche de pics pour avoir juste qu'on est au-dessus de
0,9 × le plus haut pic ? Ça va nous faire des faux positifs, mais la subtilité
c'est que lorsqu'on cherche les motifs suivants on comprend quand même tous les
motifs existants, et on s'arrête quand tout est couvert avec la complétion des
trous. Et ensuite on va avoir plusieurs hypothèses de structure et il faudra
trouver la plus cohérente. Montre-moi une démo de ça avec les règles pour
trancher. »

**Les hypothèses.** La règle de pic a deux critères séparables : la marge sur le
fond local, et le plancher (une fraction d'un pic de référence). On fait varier
les deux — plancher 0,80 / 0,85 / 0,90 / 0,95, marge active ou non, plancher
mesuré sur le pic initial ou sur le plus haut pic. Chaque combinaison donne un
dictionnaire, donc une structure. Tout le reste est identique : même matrice,
même étage 1, même étage 2, même complétion des trous.

**Les règles pour trancher.** Elles sont volontairement peu nombreuses et
chacune se lit en une phrase. Aucune n'est un réglage : ce sont des mesures sur
la structure produite.

  1. FIDÉLITÉ — pour chaque occurrence acceptée, à quel point elle ressemble
     vraiment à son motif, mesure contre mesure. C'est le garde-fou contre les
     faux positifs que la relaxation invite. En dessous d'un plancher, une
     hypothèse est écartée, quoi qu'elle vaille ailleurs.
  2. COUVERTURE — la part des mesures qu'une cellule revendique. Ce qui reste
     n'est expliqué par rien.
  3. COÛT D'ÉCRITURE — le nombre de mesures qu'il faut écrire pour rendre le
     morceau : une fois chaque cellule distincte, plus une ligne par section.
     C'est le critère d'Occam, rendu littéral : la meilleure structure est
     celle qui explique le plus de musique avec le moins d'écriture.
  4. RÉGULARITÉ — la part des sections dont la longueur est un multiple de 4
     mesures. Une structure de chanson qui tombe juste est plus crédible.

Le verdict combine les trois dernières APRÈS le filtre de fidélité, et la page
montre chaque valeur pour que le classement puisse être contesté à l'œil.
"""
from __future__ import annotations

import copy
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
from ssm_rows_plot import fig2b64                                  # noqa: E402
from harmonia_min import sections as hs, musx as mx                # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402

INK = "#1c1c1c"
COLS = ["#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed", "#0f766e",
        "#be123c", "#0369a1", "#a16207", "#4338ca"]
NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()
FIDELITY_FLOOR = 0.90     # en dessous, l'hypothèse est écartée : ses
                          # occurrences ne ressemblent pas assez à leur motif
DEFAULT = ["bruno_mars_grenade_official_music_video", "maroon_5_this_love",
           "norah_jones_don_t_know_why", "let_it_be_remastered_2009",
           "mayer_hawthorne_the_walk"]

VARIANTS = [
    ("règle actuelle", dict(peak_frac=None, peak_margin=True, against_max=False)),
    ("0,95 du pic initial, sans marge", dict(peak_frac=.95, peak_margin=False)),
    ("0,90 du pic initial, sans marge", dict(peak_frac=.90, peak_margin=False)),
    ("0,85 du pic initial, sans marge", dict(peak_frac=.85, peak_margin=False)),
    ("0,80 du pic initial, sans marge", dict(peak_frac=.80, peak_margin=False)),
    ("0,90 du PLUS HAUT pic, sans marge",
     dict(peak_frac=.90, peak_margin=False, against_max=True)),
    ("0,85 du PLUS HAUT pic, sans marge",
     dict(peak_frac=.85, peak_margin=False, against_max=True)),
    ("0,90 du plus haut pic, marge gardée",
     dict(peak_frac=.90, peak_margin=True, against_max=True)),
]


def load(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    c = {}

    def spy(g, a, t, bars=None, **k):
        c.update(grid=g, bars=copy.deepcopy(bars))
        return real(g, a, t, bars, **k)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x", file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    grid = c["grid"]
    V = HS.harmonic_vectors(mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0], grid)
    return V @ V.T, len(grid) - 1, c["bars"]


def evaluate(S, n, cells, secs):
    """Les quatre mesures. Rien ici n'est réglable — ce sont des constats."""
    fid = [HS.diag_match(S, e["b0"], o, e["L"]) for e in cells for o in e["occ"]]
    cov = np.zeros(n, bool)
    for e in cells:
        for s in sorted(set([e["b0"]]) | set(e["occ"])):
            cov[s:min(n, s + e["L"])] = True
    letters = {s["letter"] for s in secs}
    written = sum(next(x["b1"] - x["b0"] + 1 for x in secs if x["letter"] == L)
                  for L in letters)
    regular = np.mean([(s["b1"] - s["b0"] + 1) % 4 == 0 for s in secs]) if secs else 0.0
    return {
        "fidelite": float(np.mean(fid)) if fid else 0.0,
        "couverture": float(cov.mean()),
        "cout": int(written + len(secs)),
        "regularite": float(regular),
        "lettres": len(letters), "sections": len(secs), "cellules": len(cells),
    }


def strip(ax, secs, n, label, win=False):
    seen = {}
    for s in secs:
        seen.setdefault(s["letter"], COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   color=seen[s["letter"]]))
        if s["b1"] - s["b0"] >= 1:
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                    va="center", color="#fff", fontsize=7.5, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.1)
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(("★ " if win else "") + label, fontsize=7.4, rotation=0,
                  ha="right", va="center",
                  color="#1f8a5b" if win else "#8a8371")


def song_html(stem):
    S, n, bars = load(stem)
    hyps = []
    for name, kw in VARIANTS:
        cells, _ = HS.build_cells(S, n, **kw)
        secs = HS.sections_from(S, n, cells)
        m = evaluate(S, n, cells, secs)
        hyps.append({"name": name, "cells": cells, "secs": secs, **m})

    # 1 — le filtre de fidélité, puis le classement
    kept = [h for h in hyps if h["fidelite"] >= FIDELITY_FLOOR]
    pool = kept or hyps
    # rang moyen sur les trois critères restants (coût bas, couverture haute,
    # régularité haute) — un rang, pas une somme pondérée, pour qu'aucun
    # critère n'écrase les autres par son échelle
    def ranks(key, reverse):
        order = sorted(pool, key=lambda h: h[key], reverse=reverse)
        return {id(h): i for i, h in enumerate(order)}
    r1, r2, r3 = ranks("cout", False), ranks("couverture", True), ranks("regularite", True)
    for h in pool:
        h["rang"] = (r1[id(h)] + r2[id(h)] + r3[id(h)]) / 3
    win = min(pool, key=lambda h: h["rang"])

    fig, axs = plt.subplots(len(hyps), 1, figsize=(12.4, .42 * len(hyps) + .7),
                            gridspec_kw={"hspace": .85})
    for ax, h in zip(np.atleast_1d(axs), hyps):
        strip(ax, h["secs"], n, h["name"], win=(h is win))
    np.atleast_1d(axs)[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64(fig)

    rows = ""
    for h in hyps:
        out = h["fidelite"] < FIDELITY_FLOOR
        cls = "win" if h is win else ("out" if out else "")
        rows += (f"<tr class='{cls}'><td>{h['name']}</td>"
                 f"<td class='{'bad' if out else 'good'}'>{h['fidelite']:.3f}</td>"
                 f"<td>{h['couverture']:.0%}</td><td>{h['cout']}</td>"
                 f"<td>{h['regularite']:.0%}</td><td>{h['lettres']}</td>"
                 f"<td>{h['cellules']}</td>"
                 f"<td>{'écartée — fidélité' if out else f'{h[chr(114)+chr(97)+chr(110)+chr(103)]:.1f}'}</td></tr>")

    wsec = " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in win["secs"])
    return f"""<section><h2>{stem.replace('_',' ').title()}
<span class=sub>{n} mesures · {len(hyps)} hypothèses</span></h2>
<img src="data:image/png;base64,{img}">
<table><tr><th>hypothèse</th><th>fidélité</th><th>couverture</th>
<th>coût d'écriture</th><th>régularité</th><th>lettres</th><th>cellules</th>
<th>rang moyen</th></tr>{rows}</table>
<p class=verdict><b>Retenue : {win['name']}</b><br>{wsec}</p>
</section>"""


def main():
    stems = sys.argv[1:] or DEFAULT
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        body += song_html(st)
        print(f"  ok {st}")
    out = HERE / "harmonia_min/state/reports/structure_hypotheses.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Plusieurs structures, et comment trancher</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1050px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:20px}}
.lede b{{color:{INK}}}
ol{{margin:8px 0 0;padding-left:20px}} ol li{{margin-bottom:5px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{max-width:100%;border-radius:8px;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
tr.win td{{background:#e4f0e8;font-weight:700}}
tr.out td{{color:#a89f8c}}
td.good{{color:#0f5132}} td.bad{{color:#8a2b2b}}
.verdict{{font-size:13px;background:#f7f3e9;border-radius:8px;padding:9px 11px;margin:10px 0 0}}
</style></head><body><div class=wrap>
<h1>Plusieurs structures, et comment trancher</h1>
<div class=lede>La règle de pic a deux critères séparables : la marge sur le fond
local, et le plancher exprimé en fraction d'un pic de référence. On les fait
varier — c'est ta relaxation — et <b>chaque combinaison donne une structure
différente</b>. Tout le reste est identique : même matrice, même étage 1, même
étage 2, même complétion des trous.
<br><br><b>Les règles pour trancher</b>, dans l'ordre où elles s'appliquent :
<ol>
<li><b>fidélité</b> — à quel point chaque occurrence acceptée ressemble
vraiment à son motif, mesure contre mesure. C'est le garde-fou contre les faux
positifs qu'invite la relaxation. Sous {FIDELITY_FLOOR:.2f}, l'hypothèse est
écartée quoi qu'elle vaille ailleurs.</li>
<li><b>couverture</b> — la part des mesures qu'une cellule revendique. Le reste
n'est expliqué par rien.</li>
<li><b>coût d'écriture</b> — combien de mesures il faut écrire pour rendre le
morceau : une fois chaque cellule distincte, plus une ligne par section. C'est
Occam rendu littéral, et c'est le critère qui pénalise vraiment les faux
positifs : ils fabriquent des lettres qu'il faut écrire.</li>
<li><b>régularité</b> — la part des sections dont la longueur est un multiple
de 4 mesures.</li>
</ol>
Le verdict est le <b>rang moyen</b> sur les trois derniers, après le filtre de
fidélité — un rang et pas une somme pondérée, pour qu'aucun critère n'écrase les
autres par son échelle. Toutes les valeurs sont dans le tableau : le classement
peut être contesté à l'œil.</div>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
