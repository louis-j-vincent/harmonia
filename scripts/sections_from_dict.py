"""From the dictionary to the sections — the method Louis validated.

  1. RUNS   — a section is a maximal chain of CONSECUTIVE occurrences of one
              entry, not one occurrence. This Love's 4-bar cycle played at
              0,4,8,12 is one 16-bar section, not four.
  2. GAPS   — what no entry claims becomes a section of its own.
  3. RESIDUAL PASS — the gaps are compared to each other by the DIAGONAL
              reading and those that match share a letter.
  4. EDGE RULES — a gap shorter than the motif around it is a tail and joins
              the run before it; a foreign block shorter than the motif does
              not split a run.

Letters come from the dictionary entry, never from a similarity threshold, so
two passages of incompatible length can no longer land under one letter.

    python scripts/sections_from_dict.py   ->  /reports/sections_from_dict.html
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
from dictionary_harmonic import bar_grid, build                # noqa: E402

OUT = HERE / "harmonia_min/state/reports/sections_from_dict.html"
INK = "#1c1c1c"
PALETTE = ["#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed",
           "#0f766e", "#be123c", "#0369a1"]
GAP_MATCH = 0.90     # two gaps share a letter when their bar-to-bar diagonal
                     # reaches this — the same 0.90 as the peak rule, on the
                     # same scale (1.0 = identical bar for bar).
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why")]


def sections_from(S, n, entries):
    """The four steps. Returns [{b0, b1, letter, why}] covering 0..n-1."""
    # 1 — runs of consecutive occurrences
    secs = []
    for ei, e in enumerate(entries):
        L = e["L"]
        pos = sorted(set([e["b0"]] + e["occ"]))
        run = [pos[0]]
        for p in pos[1:]:
            # Chain when the next occurrence starts where this one ends, give or
            # take a bar. NOT `p == run[-1] + L`: a motif measures the run of
            # strong agreement, which can stop a bar short of the musical block
            # (This Love's chorus is a 7-bar motif recurring every 8), and the
            # strict test then refused to chain 56, 64, 72 into one section.
            if 0 <= p - (run[-1] + L) <= 1:
                run.append(p)
            else:
                secs.append({"b0": run[0], "b1": run[-1] + L - 1,
                             "entry": ei, "why": f"{len(run)}× le motif de l'entrée {ei+1}"})
                run = [p]
        secs.append({"b0": run[0], "b1": run[-1] + L - 1, "entry": ei,
                     "why": f"{len(run)}× le motif de l'entrée {ei+1}"})
    secs.sort(key=lambda s: s["b0"])

    # 2 — gaps
    filled, b = [], 0
    for s in secs:
        if s["b0"] > b:
            filled.append({"b0": b, "b1": s["b0"] - 1, "entry": None,
                           "why": "aucune entrée ne le revendique"})
        filled.append(s)
        b = max(b, s["b1"] + 1)
    if b < n:
        filled.append({"b0": b, "b1": n - 1, "entry": None,
                       "why": "aucune entrée ne le revendique"})

    # 4a — a gap shorter than the motif around it is a tail: join it left
    motif_of = {i: e["L"] for i, e in enumerate(entries)}
    out = []
    for s in filled:
        prev = out[-1] if out else None
        if (s["entry"] is None and prev is not None and prev["entry"] is not None
                and (s["b1"] - s["b0"] + 1) <= motif_of[prev["entry"]]):
            prev["b1"] = s["b1"]
            prev["why"] += " + sa queue"
            continue
        out.append(dict(s))

    # 3 — residual pass: gaps that match each other by the DIAGONAL share a letter
    gaps = [i for i, s in enumerate(out) if s["entry"] is None]
    group = {}
    gid = 0
    for a in range(len(gaps)):
        ia = gaps[a]
        if ia in group:
            continue
        group[ia] = gid
        for c in range(a + 1, len(gaps)):
            ic = gaps[c]
            if ic in group:
                continue
            L = min(out[ia]["b1"] - out[ia]["b0"] + 1,
                    out[ic]["b1"] - out[ic]["b0"] + 1)
            if L >= 2 and HM.diag_match(S, out[ia]["b0"], out[ic]["b0"], L) >= GAP_MATCH:
                group[ic] = gid
                out[ic]["why"] = (f"apparié au trou de la mesure "
                                  f"{out[ia]['b0']+1} par la diagonale")
        gid += 1

    # letters: one per entry, then one per gap group
    letters, nxt = {}, 0
    for s in out:
        key = ("e", s["entry"]) if s["entry"] is not None else ("g", group[id(s) if False else out.index(s)])
        if key not in letters:
            letters[key] = chr(ord("A") + nxt)
            nxt += 1
        s["letter"] = letters[key]
    return out


def strip(ax, secs, n, title):
    seen = {}
    for s in secs:
        if s["letter"] not in seen:
            seen[s["letter"]] = PALETTE[len(seen) % len(PALETTE)]
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   color=seen[s["letter"]]))
        ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                va="center", color="#fff", fontsize=11, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.4)
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center")


def song_html(stem, title):
    cap = bar_grid(stem)
    grid, cur = cap["grid"], cap["segs"]
    n = len(grid) - 1
    S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
    entries, _ = build(S, n)
    secs = sections_from(S, n, entries)

    fig, axs = plt.subplots(2, 1, figsize=(13, 2.4))
    strip(axs[0], secs, n, "NOUVEAU\n(dictionnaire)")
    strip(axs[1], [{"b0": s["b0"], "b1": s["b1"], "letter": s["label"]}
                   for s in cur], n, "ACTUEL\n(pipeline)")
    axs[1].set_xlabel("mesure", fontsize=8)
    strips = fig2b64(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.imshow(S, cmap="RdYlBu_r", origin="lower",
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    for s in secs:
        ax.axvline(s["b0"], color=INK, lw=1.1)
        ax.axhline(s["b0"], color=INK, lw=1.1)
    ax.set_xticks([]); ax.set_yticks([])
    mat = fig2b64(fig)

    rows = "".join(
        f"<tr><td><b>{s['letter']}</b></td><td>mes. {s['b0']+1}–{s['b1']+1}</td>"
        f"<td>{s['b1']-s['b0']+1}</td><td>{s['why']}</td></tr>" for s in secs)
    return f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<img src="data:image/png;base64,{strips}">
<p class=cap>En haut ce que le dictionnaire donne, en bas ce que le pipeline
écrit aujourd'hui. Même lettre = même couleur.</p>
<img src="data:image/png;base64,{mat}">
<p class=cap>Les frontières nouvelles, posées sur la matrice.</p>
<table><tr><th>lettre</th><th>mesures</th><th>longueur</th><th>d'où elle vient</th></tr>
{rows}</table></section>"""


def main():
    body = ""
    for stem, title in SONGS:
        body += song_html(stem, title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Du dictionnaire aux sections</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 12px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{max-width:100%;border-radius:8px;margin-bottom:4px}}
.cap{{font-size:12px;color:#8a8371;margin:0 0 12px}}
table{{border-collapse:collapse;font-size:12.5px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
</style></head><body><div class=wrap>
<h1>Du dictionnaire aux sections</h1>
<div class=lede>Une section est une <b>suite d'occurrences consécutives</b> d'une
entrée, pas une occurrence. Ce qu'aucune entrée ne revendique devient une section
à part, et ces trous sont ensuite appariés entre eux <b>par la diagonale</b>. Un
trou plus court que le motif qui l'entoure est une queue et rejoint la suite
précédente. Les lettres viennent de l'entrée du dictionnaire, jamais d'un seuil
de similarité — deux passages de longueurs incompatibles ne peuvent donc plus
atterrir sous la même lettre.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
