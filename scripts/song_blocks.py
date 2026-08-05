"""Block detection on ONE song, end to end — for when a chart looks wrong.

    python scripts/song_blocks.py <stem> [<stem> ...]
      -> /reports/song_blocks_<stem>.html

Written 2026-08-05 for Louis's report on Mayer Hawthorne's *The Walk* (« on ne
détecte qu'une section alors qu'il y en a une deuxième, mais elle est très
proche harmoniquement de la première »). Generalised on purpose: any song whose
sections look wrong gets the same five pictures, in the order that isolates the
cause.

  1. the SSM with the sections drawn on it — where the boundaries landed;
  2. the DISTRIBUTION of the matrix — how much of this song is "the same" at
     all. A vamp has a median near 1: no threshold can work there and the
     picture says so immediately;
  3. every dictionary entry: its sliding curve, its motif, its occurrences;
  4. which entry owns which bar;
  5. **section against section, read bar to bar** — the one that answers "why
     do these two get different letters / the same letter". 1.00 means the two
     passages are literally the same harmony bar for bar.

Then the chords of each letter, so the letters can be checked by reading.

Everything imports `harmonia_min.harmonic_sections` — the code the app runs.
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
import harmonic_method as HM                                       # noqa: E402
from harmonia_min import sections as hs                            # noqa: E402
from harmonia_min.harmonic_sections import (                       # noqa: E402
    build_dictionary, sections_from, off_diagonal, PHASE_QUANTILE,
    CONT_QUANTILE)

INK = "#1c1c1c"
COLS = ["#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed", "#0f766e",
        "#be123c", "#0369a1"]
NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()


def capture(stem):
    """Run the real pipeline, keep the grid and the per-bar chords it built."""
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None, **kw):
        cap.update(grid=grid, bars=copy.deepcopy(bars))
        return real(grid, arr, times, bars, **kw)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title=stem,
                    file_key="x", audio_url="")
    finally:
        hs.detect_sections = real
    return cap


def bar_text(bar):
    if not bar:
        return "—"
    out = []
    for c in bar:
        if c.get("nc"):
            out.append("N.C.")
        else:
            out.append(NAMES[c["root"]] + c["q"] +
                       ("(t)" if c.get("carry") else ""))
    return " ".join(out)


def song_html(stem, title):
    cap = capture(stem)
    grid, bars = cap["grid"], cap["bars"]
    n = len(grid) - 1
    S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
    entries, _ = build_dictionary(S, n)
    secs = sections_from(S, n, entries)
    letters = sorted({s["letter"] for s in secs})
    col_of = {L: COLS[i % len(COLS)] for i, L in enumerate(letters)}

    # 1 — the SSM with the sections on it, letters underneath
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 8.0),
                            gridspec_kw={"height_ratios": [8, 1]})
    axs[0].imshow(S, cmap="RdYlBu_r", origin="lower",
                  vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    for s in secs:
        axs[0].axvline(s["b0"] - .5, color=INK, lw=1.0)
        axs[0].axhline(s["b0"] - .5, color=INK, lw=1.0)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_title("la matrice harmonique, avec les frontières trouvées",
                     fontsize=9, loc="left")
    for s in secs:
        axs[1].add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                       color=col_of[s["letter"]]))
        axs[1].text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                    va="center", color="#fff", fontsize=10, fontweight="bold")
    axs[1].set_xlim(0, n); axs[1].set_ylim(0, 1); axs[1].set_yticks([])
    axs[1].set_xlabel("mesure", fontsize=8)
    p_ssm = fig2b64(fig)

    # 2 — the distribution: how much of this song is "the same"
    off = off_diagonal(S)
    q80, q90 = float(np.quantile(off, CONT_QUANTILE)), float(np.quantile(off, PHASE_QUANTILE))
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.hist(off, bins=90, color="#9db4cc")
    for v, c, lab in ((float(np.median(off)), "#8a8371", "médiane"),
                      (q80, "#2a6fb0", "q80"), (q90, "#8a2b2b", "q90")):
        ax.axvline(v, color=c, lw=1.6)
        ax.text(v, ax.get_ylim()[1] * .9, f" {lab} = {v:.3f}", color=c, fontsize=8)
    ax.set_xlabel("similarité harmonique entre deux mesures du morceau", fontsize=8.5)
    p_hist = fig2b64(fig)

    # 3 — the dictionary entries
    panels = []
    for ei, e in enumerate(entries):
        col = COLS[ei % len(COLS)]
        L, b0, curve = e["L"], e["b0"], e["curve"]
        fig, axx = plt.subplots(1, 2, figsize=(13, 3.4),
                                gridspec_kw={"width_ratios": [2.1, 1]})
        axx[0].plot(np.arange(len(curve)), curve, lw=1.3, color=INK)
        axx[0].axhline(HM.INITIAL_PEAK_FRAC * curve[b0], color=col, ls="--", lw=1)
        axx[0].axvline(b0, color=col, lw=1.8)
        axx[0].plot(e["occ"], curve[e["occ"]], "o", color=col, ms=8,
                    mfc="none", mew=1.9)
        axx[0].set_xlabel("mesure où l'on pose le motif", fontsize=8)
        axx[0].set_title(f"entrée {ei+1} — motif de {L} mesures pris mesure "
                         f"{b0+1}, {len(e['occ'])} occurrences",
                         fontsize=9, loc="left", color=col)
        axx[1].imshow(S, cmap="RdYlBu_r", origin="lower",
                      vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
        axx[1].add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False,
                                       ec=INK, lw=2.2))
        for o in e["occ"]:
            axx[1].add_patch(plt.Rectangle((o - .5, b0 - .5), L, L, fill=False,
                                           ec=col, lw=1.7, ls="--"))
        axx[1].set_xticks([]); axx[1].set_yticks([])
        panels.append(fig2b64(fig))

    # 4 — who owns which bar
    owner = -np.ones(n, int)
    for ei, e in enumerate(entries):
        for b in sorted(set([e["b0"]] + list(e["occ"]))):
            owner[b:min(n, b + e["L"])] = ei
    fig, ax = plt.subplots(figsize=(13, 1.2))
    for b in range(n):
        c = COLS[owner[b] % len(COLS)] if owner[b] >= 0 else "#e5dcc6"
        ax.add_patch(plt.Rectangle((b, 0), 1, 1, color=c))
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("mesure — couleur = entrée du dictionnaire, "
                  "gris = revendiqué par personne", fontsize=8)
    p_own = fig2b64(fig)

    # 5 — section against section, bar to bar. THE picture.
    k = len(secs)
    D = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            L = min(secs[a]["b1"] - secs[a]["b0"] + 1,
                    secs[b]["b1"] - secs[b]["b0"] + 1)
            D[a, b] = HM.diag_match(S, secs[a]["b0"], secs[b]["b0"], L)
    fig, ax = plt.subplots(figsize=(0.62 * k + 2.4, 0.62 * k + 2.0))
    ax.imshow(D, cmap="RdYlBu_r", vmin=0, vmax=1)
    for a in range(k):
        for b in range(k):
            ax.text(b, a, f"{D[a, b]:.2f}", ha="center", va="center",
                    fontsize=7.4, color="#fff" if D[a, b] > .72 or D[a, b] < .18 else INK)
    lab = [f"{s['letter']} {s['b0']+1}" for s in secs]
    ax.set_xticks(range(k)); ax.set_xticklabels(lab, fontsize=7.5, rotation=90)
    ax.set_yticks(range(k)); ax.set_yticklabels(lab, fontsize=7.5)
    for i, s in enumerate(secs):
        ax.get_xticklabels()[i].set_color(col_of[s["letter"]])
        ax.get_yticklabels()[i].set_color(col_of[s["letter"]])
    p_pair = fig2b64(fig)

    # the chords of each letter, read from the first occurrence
    first = {}
    for s in secs:
        first.setdefault(s["letter"], s)
    chord_rows = ""
    for L in letters:
        s = first[L]
        spans = [f"{x['b0']+1}–{x['b1']+1}" for x in secs if x["letter"] == L]
        txt = " | ".join(bar_text(bars[b]) for b in range(s["b0"], s["b1"] + 1))
        chord_rows += (f"<tr><td style='color:{col_of[L]}'><b>{L}</b></td>"
                       f"<td>{', '.join(spans)}</td><td class=ch>{txt}</td></tr>")

    ent_rows = "".join(
        f"<tr><td style='color:{COLS[i % len(COLS)]}'><b>entrée {i+1}</b></td>"
        f"<td>{e['L']} mesures</td><td>mesure {e['b0']+1}</td>"
        f"<td>{', '.join(str(o+1) for o in e['occ'])}</td></tr>"
        for i, e in enumerate(entries))

    return f"""<section><h2>{title} <span class=sub>{n} mesures ·
{len(entries)} entrée(s) · {len(secs)} sections · {len(letters)} lettres</span></h2>

<h3>1 — où sont tombées les frontières</h3>
<img src="data:image/png;base64,{p_ssm}">

<h3>2 — combien de ce morceau est « pareil »</h3>
<img src="data:image/png;base64,{p_hist}">
<p class=cap>Toutes les paires de mesures distantes d'au moins deux mesures.
Médiane {float(np.median(off)):.3f}, q80 {q80:.3f}, q90 {q90:.3f}. Quand la
médiane est déjà haute, le morceau ne change quasiment pas d'harmonie et
<b>aucun seuil ne peut séparer ses sections</b> — la cause est là, pas dans la
règle de détection.</p>

<h3>3 — chaque entrée du dictionnaire</h3>
{''.join(f'<img src="data:image/png;base64,{g}">' for g in panels)}
<table><tr><th></th><th>motif</th><th>pris en</th><th>occurrences (mesures)</th></tr>
{ent_rows}</table>

<h3>4 — qui possède quelle mesure</h3>
<img src="data:image/png;base64,{p_own}">

<h3>5 — chaque section contre chaque section, lue mesure à mesure</h3>
<img src="data:image/png;base64,{p_pair}">
<p class=cap>1,00 = les deux passages ont exactement la même harmonie, mesure
par mesure. C'est ici qu'on voit si deux lettres différentes désignent en fait
la même musique — ou l'inverse.</p>

<h3>Ce que chaque lettre joue</h3>
<table><tr><th>lettre</th><th>mesures</th><th>accords</th></tr>{chord_rows}</table>
<p class=cap>(t) = accord tenu, réécrit parce qu'il sonne encore.</p>
</section>"""


def main():
    stems = sys.argv[1:] or ["mayer_hawthorne_the_walk"]
    body = ""
    for stem in stems:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            print(f"  !! docs/audio/{stem}.m4a introuvable")
            continue
        body += song_html(stem, stem.replace("_", " ").title())
        print(f"  ok {stem}")
    out = HERE / ("harmonia_min/state/reports/song_blocks_"
                  + "_".join(s[:28] for s in stems) + ".html")
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Détection de blocs — {', '.join(stems)}</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 12px;color:#8a2b2b}}
h3{{font:700 11px system-ui;margin:20px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{max-width:100%;border-radius:8px;margin-bottom:4px}}
.cap{{font-size:12px;color:#8a8371;margin:2px 0 10px;max-width:900px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:6px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9}}
.ch{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}
</style></head><body><div class=wrap>
<h1>Détection de blocs, morceau par morceau</h1>
<div class=lede>La méthode telle qu'elle tourne dans l'app : matrice
d'harmonie entre mesures, dictionnaire de motifs répétés, puis sections. Les
cinq images vont de « où sont les frontières » à « pourquoi ces deux passages
ont la même lettre, ou pas ».</div>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
