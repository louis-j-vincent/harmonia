"""The threshold choice, judged by a positive set our own detector never touched.

`/reports/threshold_histograms.html` (the previous page) plotted, in red, the
bar-to-bar values of the matches OUR OWN dictionary (`dictionary_harmonic.build`)
accepted. That is partly circular: those matches were selected using the very
quantiles (q80/q90/q95) the page exists to help choose, so the red curve is
partly a consequence of the threshold, not an independent judge of it.

This page replaces the red with a positive set built from PUBLISHED song
structure — a trusted Ultimate Guitar tab (rating ≥ 4.7★, project trust order:
iReal Pro > UG ≥4.7★ > other tabs > model output), never our detector — and
turns "where do the two curves separate" into an actual measured separation
(ROC / AUC) instead of an eyeballed histogram gap.

    GREEN = bar-to-bar harmonic similarity `S[a+k, b+k]` between two
            DIFFERENT occurrences of the SAME published section, at the SAME
            position k inside it (e.g. verse 1 bar 3 vs verse 2 bar 3).
            These are true positives — the published structure says this
            pair of bars is the same music, played twice.
    GREY  = `S[i, j]` between two bars the published structure puts in
            DIFFERENT sections (any position). True negatives.

How the published structure gets onto OUR bar grid: `scratchpad/ug_align.py`
already aligns a UG tab to our audio by ASR lyric anchors (phase 2) — every
UG chord/section event gets a real t0/t1 in OUR seconds. The `SECTIONS`
literal below is those event section-start times, transcribed by hand from
that alignment (see the `SOURCES` dict for the exact run: URL, rating,
votes, anchor count/quality). A bar's label = whichever published section
covers its CENTRE time; run-length-encoding that per-bar label sequence
gives the section OCCURRENCES (a repeated section name after a different
name in between = a new occurrence). Bars before the tab's first event or
after its last (usually an instrumental intro UG doesn't transcribe) are
UNCOVERED and dropped from the study, never guessed.

    python scripts/threshold_labelled.py  ->  /reports/threshold_labelled.html
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
from ssm_rows_plot import fig2b64                        # noqa: E402
import harmonic_method as HM                              # noqa: E402
from dictionary_harmonic import bar_grid                  # noqa: E402

OUT = HERE / "harmonia_min/state/reports/threshold_labelled.html"
INK = "#1c1c1c"
GREEN, GREY, BLUE_BG = "#1f8a5b", "#8a8371", "#c9c2ae"
QS = [(0.80, "#2a6fb0", "q80"), (0.85, "#c58a2e", "q85"),
      (0.90, "#8a2b2b", "q90"), (0.95, "#7c3aed", "q95")]
SEC_COLOR = {"Intro": "#8a8371", "Verse": "#2a6fb0", "Chorus": "#8a2b2b", "Bridge": "#c58a2e"}

# ── PUBLISHED STRUCTURE — edit this to override, one line per section ──────
# (section name, start time in OUR audio, seconds). A section runs until the
# next line's start time; the last one runs to the end of the bar grid.
# Times = section-start events from a lyric-ASR-anchored alignment of the UG
# tab to our audio (`scratchpad/ug_align.py --phase2`), read off directly —
# NOT from our own section/chord detector. See SOURCES below for the tab
# used and how confident that alignment is.
SECTIONS: dict[str, list[tuple[str, float]]] = {
    "maroon_5_this_love": [
        ("Intro",    2.94),
        ("Verse",   21.44),
        ("Chorus",  41.94),
        ("Verse",   71.94),
        ("Chorus",  92.44),
        ("Bridge", 122.44),
        ("Chorus", 142.94),
    ],
    "norah_jones_don_t_know_why": [
        ("Verse",   10.64),
        ("Chorus",  59.64),
        ("Verse",   79.14),
        ("Chorus", 102.14),
        ("Verse",  136.14),
    ],
}

SOURCES = {
    "maroon_5_this_love": dict(
        url="https://tabs.ultimate-guitar.com/tab/maroon-5/this-love-chords-786697",
        rating=4.86, votes=2412,
        detail="alignement par ancrage lyrique (phase 2, transcription ASR whisper) : "
               "30 ancrages lyriques, similarité ligne-à-ligne 0,63–1,00 ; accord "
               "d'accord racine avec notre grille 95,8 % ; verdict du test de "
               "contraste DTW harmonique « determinate ».",
    ),
    "norah_jones_don_t_know_why": dict(
        url="https://tabs.ultimate-guitar.com/tab/norah-jones/dont-know-why-chords-839620",
        rating=4.86, votes=893,
        detail="alignement par ancrage lyrique (phase 2, transcription ASR whisper) : "
               "21 ancrages lyriques sur toute la durée du morceau, similarité "
               "0,67–1,00 ; accord d'accord racine avec notre grille 82,3 % ; le "
               "test de contraste DTW harmonique SEUL lit « harmony-underdetermined » "
               "(z=1,15) — ce sont les ancrages lyriques qui portent cet alignement, "
               "pas le coût harmonique seul, et c'est écrit ici pour ne pas le cacher.",
    ),
}
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why")]


# ── published structure → per-bar labels → occurrences ─────────────────────

def section_bounds(stem: str, grid: np.ndarray) -> list[tuple[str, float, float]]:
    secs = SECTIONS[stem]
    return [(name, t0, (secs[i + 1][1] if i + 1 < len(secs) else float(grid[-1])))
            for i, (name, t0) in enumerate(secs)]


def bar_labels(stem: str, grid: np.ndarray) -> list[str | None]:
    """One label per bar — whichever published section covers its CENTRE time.
    None = not covered by the tab (usually an un-transcribed instrumental
    intro); such bars are dropped from every comparison below."""
    n = len(grid) - 1
    bounds = section_bounds(stem, grid)
    labels: list[str | None] = [None] * n
    for b in range(n):
        c = 0.5 * (grid[b] + grid[b + 1])
        for name, t0, t1 in bounds:
            if t0 <= c < t1:
                labels[b] = name
                break
    return labels


def occurrences(labels: list[str | None]) -> list[tuple[str, int, int]]:
    """Run-length-encode the per-bar labels into (name, start_bar, end_bar)
    blocks. The SAME name appearing again after a different name in between
    is a NEW occurrence — this is what lets "verse 1 bar 3 vs verse 2 bar 3"
    mean something."""
    occ, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] is None:
            i += 1
            continue
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        occ.append((labels[i], i, j))
        i = j
    return occ


def green_grey(S: np.ndarray, labels: list[str | None],
               occ: list[tuple[str, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(labels)
    green: list[float] = []
    for a in range(len(occ)):
        for b in range(a + 1, len(occ)):
            na, sa, ea = occ[a]
            nb, sb, eb = occ[b]
            if na != nb:
                continue
            L = min(ea - sa, eb - sb)
            for k in range(L):
                i, j = sa + k, sb + k
                if abs(i - j) >= HM.LAG_MIN:
                    green.append(float(S[i, j]))
    grey: list[float] = []
    for i in range(n):
        for j in range(i + HM.LAG_MIN, n):
            if labels[i] is not None and labels[j] is not None and labels[i] != labels[j]:
                grey.append(float(S[i, j]))
    return np.array(green), np.array(grey)


# ── figures ──────────────────────────────────────────────────────────────

def strip_fig(grid: np.ndarray, labels: list[str | None],
              occ: list[tuple[str, int, int]], segs) -> "plt.Figure":
    n = len(grid) - 1
    fig, ax = plt.subplots(figsize=(13, 2.1))
    seen: dict[str, int] = {}
    for name, s, e in occ:
        seen[name] = seen.get(name, 0) + 1
        col = SEC_COLOR.get(name, "#555")
        ax.add_patch(plt.Rectangle((s, 0.5), e - s, 0.42, color=col, alpha=.88))
        ax.text((s + e) / 2, 0.71, f"{name} #{seen[name]}", ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
    for b in range(n):
        if labels[b] is None:
            ax.add_patch(plt.Rectangle((b, 0.5), 1, 0.42, fill=False, hatch="////",
                                       ec="#b9b09a", lw=0))
    for sg in segs:
        ax.axvline(sg["b0"], color=INK, lw=1.1, ymin=0.04, ymax=0.46)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.71])
    ax.set_yticklabels(["sections ACTUELLES\n(traits noirs, pour info)",
                        "structure PUBLIÉE\n(vérité indépendante)"], fontsize=7.5)
    ax.set_xticks(range(0, n + 1, 4))
    ax.set_xlabel("mesure — sur notre grille", fontsize=8.5)
    return fig


def hist_fig(off: np.ndarray, green: np.ndarray, grey: np.ndarray) -> "plt.Figure":
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, logy in ((axs[0], False), (axs[1], True)):
        ax.hist(off, bins=80, color=BLUE_BG, alpha=.55, density=True,
                label="contexte — toutes les valeurs hors diagonale")
        ax.hist(grey, bins=50, color=GREY, alpha=.8, density=True,
                label="GRIS — sections DIFFÉRENTES (vrais négatifs)")
        ax.hist(green, bins=34, color=GREEN, alpha=.8, density=True,
                label="VERT — même section, même position (vrais positifs)")
        for q, col, lab in QS:
            v = float(np.quantile(off, q))
            ax.axvline(v, color=col, lw=1.6)
        # labels as a stacked legend, not inline text — q90/q95 sit within
        # 0.01 of each other on these songs and inline labels collide.
        for k, (q, col, lab) in enumerate(QS):
            v = float(np.quantile(off, q))
            ax.text(0.985, 0.95 - 0.075 * k, f"{lab} = {v:.3f}", color=col,
                    fontsize=8, ha="right", va="top", transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=.7, edgecolor="none", pad=1.2))
        if logy:
            ax.set_yscale("log")
            ax.set_title("échelle log — la queue", fontsize=9, loc="left")
        else:
            ax.set_title("échelle linéaire", fontsize=9, loc="left")
        ax.set_xlabel("similarité harmonique bord-à-bord (bar-to-bar)", fontsize=8.5)
        ax.set_ylabel("densité", fontsize=8.5)
    axs[0].legend(fontsize=7.3, loc="upper left")
    return fig


def roc_curve(green: np.ndarray, grey: np.ndarray):
    vals = np.concatenate([green, grey])
    lab = np.concatenate([np.ones(len(green)), np.zeros(len(grey))])
    order = np.argsort(-vals)
    lab_s = lab[order]
    tpr = np.concatenate([[0], np.cumsum(lab_s) / max(len(green), 1), [1]])
    fpr = np.concatenate([[0], np.cumsum(1 - lab_s) / max(len(grey), 1), [1]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def roc_fig(off: np.ndarray, green: np.ndarray, grey: np.ndarray):
    fpr, tpr, auc = roc_curve(green, grey)
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, 1], [0, 1], "--", color="#b9b09a", lw=1.2, label="hasard (AUC 0,5)")
    ax.plot(fpr, tpr, color=INK, lw=2)
    for k, (q, col, lab) in enumerate(QS):
        v = float(np.quantile(off, q))
        tp = float((green >= v).mean())
        fp = float((grey >= v).mean())
        ax.plot(fp, tp, "o", color=col, ms=7, mec="white", mew=1)
        # stagger the offset — these four points cluster together on both
        # songs (q90/q95 are within 0.01 of each other by construction)
        dx, dy = (6, 6) if k % 2 == 0 else (6, -11)
        ax.annotate(lab, (fp, tp), fontsize=7.5, color=col,
                    xytext=(dx, dy), textcoords="offset points")
    ax.set_xlabel("% des non-répétitions acceptées à tort", fontsize=8.5)
    ax.set_ylabel("% des vraies répétitions gardées", fontsize=8.5)
    ax.set_title(f"courbe ROC — AUC {auc:.3f}", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    return fig, auc


def youden_optimum(off: np.ndarray, green: np.ndarray, grey: np.ndarray):
    """The best a SINGLE threshold can do on this song — not a shipped
    quantile, a diagnostic: how good is the best possible fixed cut."""
    cand = np.unique(np.concatenate([green, grey]))
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in cand:
        tpr = float((green >= t).mean())
        fpr = float((grey >= t).mean())
        j = tpr - fpr
        if j > best[0]:
            best = (j, t, tpr, fpr)
    j, t, tpr, fpr = best
    pct = float((off < t).mean()) * 100
    return {"t": t, "pctile": pct, "keep": tpr * 100, "accept": fpr * 100, "j": j}


# ── per-song page ────────────────────────────────────────────────────────

def song_html(stem: str, title: str) -> str:
    cap = bar_grid(stem)
    grid, segs = cap["grid"], cap["segs"]
    n = len(grid) - 1
    S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
    labels = bar_labels(stem, grid)
    occ = occurrences(labels)
    green, grey = green_grey(S, labels, occ)
    i_all = np.arange(n)
    off = S[np.abs(i_all[:, None] - i_all[None, :]) >= HM.LAG_MIN]
    n_uncovered = sum(l is None for l in labels)

    src = SOURCES[stem]
    strip = fig2b64(strip_fig(grid, labels, occ, segs))
    hist = fig2b64(hist_fig(off, green, grey))
    roc_fig_obj, auc = roc_fig(off, green, grey)
    roc = fig2b64(roc_fig_obj)
    yo = youden_optimum(off, green, grey)

    table_rows = ""
    for q, col, lab in QS:
        v = float(np.quantile(off, q))
        keep = float((green >= v).mean()) * 100 if len(green) else float("nan")
        acc = float((grey >= v).mean()) * 100 if len(grey) else float("nan")
        table_rows += (f"<tr><td style='color:{col}'><b>{lab}</b></td><td>{v:.4f}</td>"
                       f"<td>{keep:.1f}%</td><td>{acc:.1f}%</td></tr>")
    table_rows += (f"<tr class=opt><td><b>meilleur seuil possible</b></td>"
                   f"<td>{yo['t']:.4f} <span class=cap>(≈ {yo['pctile']:.0f}<sup>e</sup> "
                   f"centile hors-diagonale)</span></td>"
                   f"<td>{yo['keep']:.1f}%</td><td>{yo['accept']:.1f}%</td></tr>")

    # one-sentence, data-driven recommendation
    if auc >= 0.85 and yo["accept"] <= 15:
        reco = (f"<b>Recommandation :</b> un seuil unique sépare bien ce morceau "
               f"(AUC {auc:.2f}) — le meilleur compromis mesuré garde "
               f"{yo['keep']:.0f}&nbsp;% des vraies répétitions en acceptant à tort "
               f"seulement {yo['accept']:.0f}&nbsp;% des non-répétitions, autour du "
               f"{yo['pctile']:.0f}<sup>e</sup> centile — proche de q85.")
    elif auc >= 0.70:
        reco = (f"<b>Recommandation :</b> la séparation est réelle mais imparfaite "
               f"(AUC {auc:.2f}) — même le MEILLEUR seuil mesuré ne fait que "
               f"{yo['j']:.2f} de marge : pour garder {yo['keep']:.0f}&nbsp;% des "
               f"vraies répétitions il faut accepter à tort {yo['accept']:.0f}&nbsp;% "
               f"des non-répétitions. Tous les niveaux livrés (q80–q95, ≥ "
               f"{float(np.quantile(off, 0.80)):.2f}) sont trop stricts pour ce "
               f"morceau — ils gardent moins de la moitié des vraies répétitions.")
    else:
        reco = (f"<b>Recommandation :</b> les deux distributions se chevauchent trop "
               f"pour qu'un seuil les sépare (AUC {auc:.2f}, à peine mieux que le "
               f"hasard) — c'est un résultat négatif valable, pas une chose à cacher.")

    return f"""<section><h2>{title} <span class=sub>{n} mesures ·
{len(green)} paires vertes · {len(grey)} paires grises ·
{n_uncovered} mesure(s) non couverte(s) par le tab</span></h2>

<h3>1 — la bande d'alignement (à vérifier à l'œil avant tout le reste)</h3>
<p class=cap>Source : <a href="{src['url']}">{src['url']}</a> —
Ultimate Guitar, <b>{src['rating']:.2f}★ / {src['votes']} votes</b>. {src['detail']}</p>
<img src="data:image/png;base64,{strip}">

<h3>2 — les deux distributions</h3>
<img src="data:image/png;base64,{hist}">

<h3>3 — la séparation, mesurée</h3>
<div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap">
<img src="data:image/png;base64,{roc}" style="max-width:380px">
<table><tr><th>seuil candidat</th><th>valeur brute</th>
<th>% vraies répétitions gardées</th><th>% non-répétitions acceptées à tort</th></tr>
{table_rows}</table>
</div>

<h3>4 — verdict</h3>
<p class=reco>{reco}</p>
</section>"""


def main() -> None:
    body = ""
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            print(f"  skip {title} (no audio)")
            continue
        body += song_html(stem, title)
        print(f"  ok {title}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Le seuil face à une vérité indépendante</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1300px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:920px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:18px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
h3{{font:700 11px system-ui;margin:16px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{max-width:100%;border-radius:8px}}
.cap{{font-size:11.5px;color:#8a8371;max-width:900px}}
.reco{{font-size:14px;background:#f7f3e9;border-radius:8px;padding:10px 12px;max-width:900px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
tr.opt td{{background:#f0ead9;font-weight:600}}
a{{color:#2a6fb0}}
</style></head><body><div class=wrap>
<h1>Le seuil face à une vérité indépendante</h1>
<div class=lede>La page précédente (<a href="/reports/threshold_histograms.html">
threshold_histograms</a>) comparait tout hors-diagonale (bleu) aux correspondances
que <b>notre propre dictionnaire</b> avait retenues (rouge) — un juge partiellement
circulaire, puisque ces correspondances sortent des seuils mêmes qu'on cherche à
choisir. Ici le positif ne vient plus du détecteur : c'est la structure PUBLIÉE du
morceau (un tab Ultimate Guitar ≥4,7★, jamais notre pipeline), alignée sur notre
audio par ancrage lyrique. VERT = deux passages que la structure publiée dit être
la même section à la même position (ex. couplet 1 mesure 3 contre couplet 2 mesure
3) — une vraie répétition. GRIS = deux passages de sections différentes — une
vraie absence de répétition. Une courbe ROC répond à la question qu'un histogramme
ne peut que suggérer : à quel point ces deux ensembles se séparent, pour de vrai.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
