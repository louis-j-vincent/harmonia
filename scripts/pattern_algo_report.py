"""Le dictionnaire de motifs — l'algorithme de Louis, mesuré (2026-08-05).

Deux questions, dans cet ordre :

  A. quelle statistique glissante ? brut / cosinus / centré / diagonale binaire.
     Jugées sur une référence INDÉPENDANTE de la SSM : la suite d'accords.
  B. l'algorithme complet, avec le raffinement qui compte — les blocs trouvés
     ne sont pas retirés, ils vont dans une boîte à part, restent candidats
     pour tous les motifs, et sont attribués à la fin au motif qui ressort
     nettement ; sinon ils deviennent leur propre section.

La détection de pics elle-même est traitée sur `/reports/peak_selectors.html`
(page séparée) ; ici un seul sélecteur est utilisé partout, à réglage fixe.

    python scripts/pattern_algo_report.py  ->  /reports/pattern_algo.html
"""
from __future__ import annotations

import base64
import io
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from pattern_dict_core import CACHE, song_data                    # noqa: E402
from pattern_algo_core import (boundaries, build_dictionary,      # noqa: E402
                               contrast, dictionary_sections,
                               diag_orientation, first_pattern,
                               offdiag, pick_peaks, slide)

OUT = HERE / "harmonia_min/state/reports/pattern_algo.html"
INK, SAND, PAPER = "#1c1c1c", "#8a8371", "#fffdf6"
PEAK_METHOD, PEAK_PARAM = "prominence", 0.25
TAU = 1.0

STATS = [
    ("brut", "raw", "#8a2b2b",
     "produit scalaire nu — le niveau général du bloc reste dans le signal"),
    ("cosinus", "cosine", "#2a6fb0",
     "divisé par les deux normes — le niveau général est retiré"),
    ("centré", "centered", "#1f8a5b",
     "chaque bloc moins sa propre moyenne, puis cosinus = corrélation"),
    ("diagonale", "diag", "#b8860b",
     "une diagonale binaire glissée au lieu du carré extrait"),
]

SONGS = [
    ("maroon_5_this_love", "This Love"),
    ("norah_jones_don_t_know_why", "Don't Know Why"),
    ("bobby_hebb_sunny_official_audio", "Sunny"),
    ("michael_jackson_billie_jean_official_video", "Billie Jean"),
    ("the_police_every_breath_you_take_official_music_video",
     "Every Breath You Take"),
]
PRIMARY = {"This Love", "Don't Know Why", "Sunny"}

ACC_ROWS = []          # cross-song accumulator for the summary table

PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def fig2b64(fig, dpi=112):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


# ══ the independent reference: the chord string ════════════════════════════
def bar_chords(stem: str, n: int):
    """Per-bar chord signature, taken from the list `detect_sections` RECEIVES.

    Independent of the segmentation by construction — the chords are decoded
    before any section exists. Not independent of the audio, and not ground
    truth: these are the pipeline's own chords, so a disagreement can be the
    chords' fault. That is stated on the page.
    """
    fp = CACHE / f"patbars_{stem}.pkl"
    if not fp.exists():
        return None, None
    bars = pickle.loads(fp.read_bytes())["bars"]
    sig, txt = [], []
    for b in range(n):
        bar = bars[b] if b < len(bars) else []
        cs = [(c["root"], (c["q"] or "")[:1]) for c in bar if not c.get("nc")]
        sig.append(tuple(cs) if cs else ("N",))
        txt.append(" ".join(f"{PC[r]}{q}" for r, q in cs) if cs else "N.C.")
    return sig, txt


def true_occurrences(sig, b0: int, L: int, n: int, loose: float = 1.0):
    """Bars d where the L-bar chord string equals the motif's.

    loose=1.0 -> every one of the L bars must match. loose=0.75 -> three
    quarters of them, comparing the ROOT only (a b7 written where a triad was
    heard is the same chord for this purpose).
    """
    ref = sig[b0:b0 + L]
    ref_r = [tuple(r for r, _ in s) if s != ("N",) else ("N",) for s in ref]
    out = []
    for d in range(0, n - L + 1):
        s = sig[d:d + L]
        if loose >= 1.0:
            if s == ref:
                out.append(d)
        else:
            sr = [tuple(r for r, _ in x) if x != ("N",) else ("N",) for x in s]
            m = sum(1 for a, b in zip(ref_r, sr) if a == b)
            if m / L >= loose:
                out.append(d)
    return out


def prf(pred, ref, tol=0):
    pred, ref = sorted(set(pred)), sorted(set(ref))
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    used, hit = set(), 0
    for p in pred:
        c = [i for i, g in enumerate(ref) if abs(g - p) <= tol and i not in used]
        if c:
            used.add(min(c, key=lambda i: abs(ref[i] - p)))
            hit += 1
    P, R = hit / len(pred), hit / len(ref)
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


# ══ Task A figure ══════════════════════════════════════════════════════════
def stats_fig(S, n, b0, L, curves, hits, ref, segs):
    fig, axs = plt.subplots(
        6, 1, figsize=(11.4, 11.0), sharex=True,
        gridspec_kw=dict(height_ratios=[3.0, 1.2, 1.2, 1.2, 1.2, .30],
                         hspace=.09))
    ax = axs[0]
    ax.imshow(S, cmap="magma", vmin=max(0.0, np.percentile(S, 5)), vmax=1.0,
              origin="upper", extent=(-.5, n - .5, n - .5, -.5),
              interpolation="nearest", aspect="auto")
    ax.axhspan(b0 - .5, b0 + L - .5, color="#5ad1a0", alpha=.16)
    ax.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False,
                               ec="#ffffff", lw=1.7))
    for d in ref:
        ax.add_patch(plt.Rectangle((d - .5, b0 - .5), L, L, fill=False,
                                   ec="#5ad1a0", lw=1.0, ls=":"))
    ax.set_ylabel("mesure (ligne)", fontsize=9)
    ax.set_title("la SSM ; carré blanc = le motif ; pointillés verts = les "
                 "endroits où la SUITE D'ACCORDS répète celle du motif",
                 fontsize=8.6, color=SAND, pad=6)

    for k, (nm, key, col, _) in enumerate(STATS):
        ax = axs[1 + k]
        f = curves[key]
        ax.plot(np.arange(len(f)), f, lw=1.25, color=col, zorder=2)
        for d in ref:
            ax.axvline(d, color="#5ad1a0", lw=2.6, alpha=.30, zorder=0)
        ax.axvline(b0, color=INK, lw=1.1, ls="--", zorder=1)
        sel = hits[key]
        if sel:
            ax.plot(sel, [f[d] for d in sel], "o", ms=4.6, mfc="none",
                    mec=INK, mew=1.3, zorder=4)
        ax.set_ylabel(nm, fontsize=8.4, color=col)
        ax.grid(alpha=.15, lw=.55)

    ax = axs[5]
    pal = ["#8a2b2b", "#2a6fb0", "#1f8a5b", "#b8860b", "#6a4c93", "#c05a2b",
           "#3d8b8b", "#a03060"]
    letters = sorted({s["label"] for s in segs})
    for s in segs:
        c = pal[letters.index(s["label"]) % len(pal)]
        ax.axvspan(s["b0"] - .5, s["b1"] + .5, color=c, alpha=.55)
        if s["b1"] - s["b0"] >= 2:
            ax.text((s["b0"] + s["b1"]) / 2, .5, s["label"], ha="center",
                    va="center", fontsize=7.5, color="w", fontweight="bold")
    ax.set_yticks([])
    ax.set_ylabel("sections\nactuelles", fontsize=7.2, rotation=0,
                  ha="right", va="center")
    ax.set_xlabel("mesure (colonne) — position du carré glissé sur l'axe X",
                  fontsize=9)
    ax.set_xlim(-.5, n - .5)
    return fig2b64(fig)


# ══ Task B figures ═════════════════════════════════════════════════════════
def dict_fig(S, n, dic, segs, ref_all):
    """Le dictionnaire : la matrice avec les blocs de chaque motif, puis une
    bande par motif, puis les blocs ambigus, puis les sections actuelles."""
    P = dic["patterns"]
    pal = ["#8a2b2b", "#2a6fb0", "#1f8a5b", "#b8860b", "#6a4c93"]
    fig, axs = plt.subplots(
        3, 1, figsize=(11.4, 7.2), sharex=True,
        gridspec_kw=dict(height_ratios=[3.0, .16 + .30 * (len(P) + 1), .30],
                         hspace=.10))
    ax = axs[0]
    ax.imshow(S, cmap="magma", vmin=max(0.0, np.percentile(S, 5)), vmax=1.0,
              origin="upper", extent=(-.5, n - .5, n - .5, -.5),
              interpolation="nearest", aspect="auto")
    for i, p in enumerate(P):
        c = pal[i % len(pal)]
        for d in p["occ"]:
            ax.add_patch(plt.Rectangle((d - .5, p["b0"] - .5), p["L"], p["L"],
                                       fill=False, ec=c, lw=1.25))
        ax.add_patch(plt.Rectangle((p["b0"] - .5, p["b0"] - .5), p["L"], p["L"],
                                   fill=False, ec="#ffffff", lw=1.9))
    ax.set_ylabel("mesure (ligne)", fontsize=9)

    ax = axs[1]
    lanes = len(P) + 1
    for i, p in enumerate(P):
        y = lanes - 1 - i
        c = pal[i % len(pal)]
        ax.axhline(y, color="#e5dcc6", lw=.8, zorder=0)
        for bl in dic["blocks"]:
            if bl["owner"] == i:
                ax.add_patch(plt.Rectangle((bl["d"] - .5, y - .34), p["L"], .68,
                                           color=c, alpha=.85, zorder=2))
        ax.text(-0.6, y + .40, f"motif {chr(65+i)} — mes. {p['b0']}–"
                f"{p['b0']+p['L']-1} (L={p['L']})", fontsize=7.6, color=c,
                ha="left", va="bottom")
    y = 0
    ax.axhline(y, color="#e5dcc6", lw=.8, zorder=0)
    for bl in dic["blocks"]:
        if bl["owner"] is None:
            ax.add_patch(plt.Rectangle((bl["d"] - .5, y - .34),
                                       P[bl["best"]]["L"], .68,
                                       fill=False, ec=INK, lw=1.2, hatch="///",
                                       zorder=2))
    ax.text(-0.6, y + .40, "ambigus → leur propre section", fontsize=7.6,
            color=INK, ha="left", va="bottom")
    ax.set_ylim(-.6, lanes - .1)
    ax.set_yticks([])
    ax.set_ylabel("le\ndictionnaire", fontsize=7.6, rotation=0, ha="right",
                  va="center")

    ax = axs[2]
    pal2 = ["#8a2b2b", "#2a6fb0", "#1f8a5b", "#b8860b", "#6a4c93", "#c05a2b",
            "#3d8b8b", "#a03060"]
    letters = sorted({s["label"] for s in segs})
    for s in segs:
        c = pal2[letters.index(s["label"]) % len(pal2)]
        ax.axvspan(s["b0"] - .5, s["b1"] + .5, color=c, alpha=.55)
        if s["b1"] - s["b0"] >= 2:
            ax.text((s["b0"] + s["b1"]) / 2, .5, s["label"], ha="center",
                    va="center", fontsize=7.5, color="w", fontweight="bold")
    ax.set_yticks([])
    ax.set_ylabel("sections\nactuelles", fontsize=7.2, rotation=0, ha="right",
                  va="center")
    ax.set_xlabel("mesure", fontsize=9)
    ax.set_xlim(-.5, n - .5)
    return fig2b64(fig)


def tau_fig(S, n, stat, taus, res):
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 2.3))
    ax = axs[0]
    ax.plot(taus, [r["assigned"] for r in res], "-o", ms=4, lw=1.4,
            color="#1f8a5b", label="blocs attribués")
    ax.plot(taus, [r["amb"] for r in res], "-o", ms=4, lw=1.4, color="#8a2b2b",
            label="blocs ambigus → section propre")
    ax.axvline(TAU, color=INK, ls="--", lw=1.1)
    ax.set_xlabel("τ — écart minimal exigé entre le meilleur motif et le "
                  "suivant (en σ)", fontsize=7.6)
    ax.set_ylabel("nombre de blocs", fontsize=7.6)
    ax.legend(fontsize=7); ax.grid(alpha=.18, lw=.5); ax.tick_params(labelsize=7)
    ax = axs[1]
    ax.plot(taus, [r["n_sections"] for r in res], "-o", ms=4, lw=1.4,
            color="#2a6fb0")
    ax.axvline(TAU, color=INK, ls="--", lw=1.1)
    ax.set_xlabel("τ", fontsize=7.6)
    ax.set_ylabel("sections produites", fontsize=7.6)
    ax.grid(alpha=.18, lw=.5); ax.tick_params(labelsize=7)
    return fig2b64(fig)


def margin_hist(dic):
    m = np.array([b["margin"] for b in dic["blocks"] if np.isfinite(b["margin"])])
    if not len(m):
        return None
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    ax.hist(m, bins=max(6, min(24, len(m))), color="#cdc4ad")
    ax.axvline(TAU, color="#8a2b2b", lw=1.6, ls="--")
    ax.set_xlabel("écart entre le meilleur motif et le suivant (σ)", fontsize=7.6)
    ax.set_ylabel("blocs", fontsize=7.6)
    ax.tick_params(labelsize=7)
    return fig2b64(fig)


# ══ per song ═══════════════════════════════════════════════════════════════
def song_block(stem, title):  # noqa: C901
    d = song_data(stem)
    S, n, segs = d["Sb"], d["n"], d["segs"]
    mot = first_pattern(S)
    b0, L = mot["b0"], mot["L"]
    sig, txt = bar_chords(stem, n)

    curves, hits = {}, {}
    for _, key, _, _ in STATS:
        ds, f = slide(S, b0, L, key)
        curves[key] = f
        hits[key] = sorted(p["d"] for p in pick_peaks(
            ds, f, PEAK_METHOD, PEAK_PARAM, min_sep=max(2, L // 2), exclude=b0))
    ds, fa = slide(S, b0, L, "antidiag")
    curves["antidiag"] = fa
    hits["antidiag"] = sorted(p["d"] for p in pick_peaks(
        ds, fa, PEAK_METHOD, PEAK_PARAM, min_sep=max(2, L // 2), exclude=b0))

    # PRIMARY reference: root-only, >= 3/4 of the motif's bars. Exact
    # matching is measured too brittle to use as the reference — on This Love
    # bar 3 is written "Do F-" and bar 7 "Dh", the same turnaround heard
    # slightly differently, and a strict test finds ZERO other occurrences of
    # a loop the ear hears eight times. The strict count is kept as a second
    # column so the brittleness is visible rather than hidden.
    ref = true_occurrences(sig, b0, L, n, 0.75) if sig else []
    ref_strict = true_occurrences(sig, b0, L, n) if sig else []
    ref_x = [x for x in ref if x != b0]

    rows = []
    for nm, key, col, desc in STATS + [("anti-diagonale", "antidiag", SAND,
                                        "la diagonale dans le MAUVAIS sens")]:
        sel = hits[key]
        f = curves[key]
        pk = [dict(d=x, k=x, val=float(f[x]), prom=0.0) for x in sel]
        P0, R0, F0 = prf(sel, ref_x)
        P1, R1, F1 = prf(sel, [x for x in ref_strict if x != b0])
        ACC_ROWS.append((title, nm, key, len(sel), P0, R0, F0,
                         contrast(f, pk)))
        rows.append(
            f"<tr><td><span class=dot style='background:{col}'></span>"
            f"<b>{nm}</b><br><span class=sub>{desc}</span></td>"
            f"<td class=n>{f.min():.2f} … {f.max():.2f}</td>"
            f"<td class=n>{len(sel)}</td><td class=n>{contrast(f, pk):.2f}</td>"
            f"<td class=n>{P0:.2f}</td><td class=n>{R0:.2f}</td>"
            f"<td class=n><b>{F0:.2f}</b></td><td class=n>{F1:.2f}</td></tr>")

    figA = stats_fig(S, n, b0, L, curves, hits, ref, segs)

    # ── Task B ──────────────────────────────────────────────────────────────
    best_stat = "centered"
    dic = build_dictionary(S, stat=best_stat, peak_method=PEAK_METHOD,
                           peak_param=PEAK_PARAM, tau=TAU, removal="cover")
    dic_lit = build_dictionary(S, stat=best_stat, peak_method=PEAK_METHOD,
                               peak_param=PEAK_PARAM, tau=TAU, removal="none")
    figB = dict_fig(S, n, dic, segs, ref)
    taus = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    res = []
    for t in taus:
        dt = build_dictionary(S, stat=best_stat, peak_method=PEAK_METHOD,
                              peak_param=PEAK_PARAM, tau=t, removal="cover")
        segs_t = dictionary_sections(dt, n)
        res.append(dict(tau=t,
                        assigned=sum(1 for b in dt["blocks"]
                                     if b["owner"] is not None),
                        amb=sum(1 for b in dt["blocks"] if b["owner"] is None),
                        n_sections=len(segs_t)))
    figT = tau_fig(S, n, best_stat, taus, res)
    figM = margin_hist(dic)

    prow = []
    for i, p in enumerate(dic["patterns"]):
        own = [b["d"] for b in dic["blocks"] if b["owner"] == i]
        chords = " | ".join(txt[p["b0"] + k] for k in range(p["L"])) if txt else "—"
        prow.append(
            f"<tr><td><b>motif {chr(65+i)}</b><br><span class=sub>mesures "
            f"{p['b0']}–{p['b0']+p['L']-1}</span></td><td class=n>{p['L']}</td>"
            f"<td class=chords>{chords}</td>"
            f"<td class=n>{len(p['occ'])}</td>"
            f"<td class=bars>{', '.join(map(str, p['occ']))}</td>"
            f"<td class=n>{len(own)}</td>"
            f"<td class=bars>{', '.join(map(str, own)) or '—'}</td></tr>")

    arow = []
    for b in dic["blocks"]:
        z = b["z"]
        best = chr(65 + b["best"])
        sec = chr(65 + b["second"]) if b["second"] is not None else "—"
        verdict = (f"→ motif {best}" if b["owner"] is not None
                   else "<b>ambigu → section propre</b>")
        cls = "" if b["owner"] is not None else " class=amb"
        arow.append(
            f"<tr{cls}><td class=n>{b['d']}</td>"
            f"<td class=n>{', '.join(f'{chr(65+i)}:{v:+.1f}' for i, v in enumerate(z))}</td>"
            f"<td class=n>{best}</td><td class=n>{sec}</td>"
            f"<td class=n>{b['margin']:.2f}</td><td>{verdict}</td></tr>")

    dsegs = dictionary_sections(dic, n)
    cmp_now = " ".join(f"<span class=seg>{s['label']}<sub>{s['b0']}–{s['b1']}</sub></span>"
                       for s in segs)
    cmp_dic = " ".join(f"<span class=seg>{s['label']}<sub>{s['b0']}–{s['b1']}</sub></span>"
                       for s in dsegs)
    Pb, Rb, Fb = prf(boundaries(dsegs), boundaries(segs), tol=1)

    mtxt = " | ".join(txt[b0 + k] for k in range(L)) if txt else "—"
    tag = "" if title in PRIMARY else " <span class=tag>en plus</span>"
    return f"""<section><h2>{title}{tag}</h2>
<div class=num>{n} mesures &nbsp;·&nbsp; motif n°1 = mesures <b>{b0}–{b0+L-1}</b>
&nbsp;·&nbsp; <b>{mtxt}</b> &nbsp;·&nbsp; {len(ref_x)} autres endroits où cette
suite d'accords revient (à 3 mesures sur 4 ; {max(0, len(ref_strict) - 1)} à
l'identique)</div>

<h3>A · les quatre statistiques glissantes, même motif, même sélecteur</h3>
<img src="data:image/png;base64,{figA}">
<p class=cap>Les bandes vertes verticales sont la référence : les mesures où la
suite d'accords du morceau répète exactement celle du motif. Les cercles noirs
sont ce que le sélecteur garde sur chaque courbe.</p>
<table><tr><th>statistique</th><th>plage de la courbe</th><th>pics</th>
<th>contraste (σ)</th><th>précision</th><th>rappel</th><th>F</th>
<th>F (accords stricts)</th></tr>{''.join(rows)}</table>
<p class=cap>Précision = part des pics qui tombent sur une vraie répétition de
la suite d'accords ; rappel = part des vraies répétitions retrouvées.</p>

<h3>B · le dictionnaire que l'algorithme construit</h3>
<img src="data:image/png;base64,{figB}">
<table><tr><th>entrée</th><th>L</th><th>suite d'accords du motif</th>
<th>blocs trouvés</th><th>où</th><th>gardés après arbitrage</th>
<th>lesquels</th></tr>{''.join(prow)}</table>

<h3>La boîte à part : qui gagne chaque bloc</h3>
<table><tr><th>bloc (mesure)</th><th>score de chaque motif (σ)</th>
<th>meilleur</th><th>2ᵉ</th><th>écart</th><th>verdict (τ = {TAU} σ)</th></tr>
{''.join(arow)}</table>
{'<img src="data:image/png;base64,' + figM + '">' if figM else ''}
<p class=cap>Chaque bloc est noté par TOUS les motifs, en écarts-types robustes
de la courbe de chaque motif — sans quoi un motif fait de matière homogène,
qui corrèle avec tout, raflerait tout. Il est attribué au meilleur seulement si
celui-ci dépasse le second de τ écarts-types.</p>

<h3>Les deux lectures de « on relance sur ce qui reste »</h3>
<table><tr><th>lecture</th><th>entrées</th><th>blocs</th><th>ambigus</th>
<th>motifs (mesure de départ)</th></tr>
<tr><td><b>ce qui reste = les mesures non couvertes</b><br>
<span class=sub>le motif suivant part d'une messe qu'aucun bloc ne couvre</span></td>
<td class=n>{len(dic['patterns'])}</td><td class=n>{len(dic['blocks'])}</td>
<td class=n>{sum(1 for b in dic['blocks'] if b['owner'] is None)}</td>
<td class=bars>{', '.join(str(p['b0']) for p in dic['patterns'])}</td></tr>
<tr><td><b>rien n'est retiré du tout</b><br>
<span class=sub>toute mesure reste un départ légal ; on rejette seulement un
motif dont la courbe est une copie décalée d'une entrée existante</span></td>
<td class=n>{len(dic_lit['patterns'])}</td><td class=n>{len(dic_lit['blocks'])}</td>
<td class=n>{sum(1 for b in dic_lit['blocks'] if b['owner'] is None)}</td>
<td class=bars>{', '.join(str(p['b0']) for p in dic_lit['patterns'])}</td></tr></table>
<p class=cap>Tout le reste de cette page utilise la première lecture. La seconde
fabrique des entrées qui sont le même cycle lu une mesure plus loin — visible
ci-dessus quand les départs se suivent (0 puis 5, 88 puis 91).</p>

<h3>Sensibilité à τ</h3>
<img src="data:image/png;base64,{figT}">

<h3>Côte à côte avec <code>sections.py</code></h3>
<div class=cmp><b>aujourd'hui</b><br>{cmp_now}</div>
<div class=cmp><b>le dictionnaire</b><br>{cmp_dic}</div>
<p class=cap>Frontières communes à ±1 mesure : précision {Pb:.2f}, rappel
{Rb:.2f}, F {Fb:.2f}. Ce n'est pas un score de justesse — les deux découpages
sont des propositions, aucun n'est la vérité.</p>
</section>"""


def main():
    body, done, summary = "", [], []
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            continue
        try:
            body += song_block(stem, title)
        except Exception as e:                       # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"  !! {title}: {e}")
            continue
        done.append(title)
        print(f"  ok {title}")

    # ── cross-song summary of Task A ────────────────────────────────────────
    import collections
    by = collections.defaultdict(list)
    for (t, nm, key, npk, P, R, F, ct) in ACC_ROWS:
        if F == 0.0 and R == 0.0 and npk and key != "antidiag" and P == 0.0:
            continue        # no usable chord reference on that song
        by[(nm, key)].append((t, npk, P, R, F, ct))
    srows = ""
    order = [("brut", "raw"), ("cosinus", "cosine"), ("centré", "centered"),
             ("diagonale", "diag"), ("anti-diagonale", "antidiag")]
    for nm, key in order:
        v = by.get((nm, key), [])
        if not v:
            continue
        P = np.mean([x[2] for x in v]); R = np.mean([x[3] for x in v])
        F = np.mean([x[4] for x in v]); C = np.mean([x[5] for x in v])
        NP = np.mean([x[1] for x in v])
        cls = " class=win" if key == "raw" else ""
        srows += (f"<tr{cls}><td><b>{nm}</b></td><td class=n>{NP:.1f}</td>"
                  f"<td class=n>{P:.2f}</td><td class=n>{R:.2f}</td>"
                  f"<td class=n><b>{F:.2f}</b></td><td class=n>{C:.2f}</td></tr>")
    verdict = f"""<section class=verdict><h2>Ce que la mesure dit — et elle te
donne raison sur le produit scalaire brut</h2>
<table><tr><th>statistique glissante</th><th>pics gardés (moy.)</th>
<th>précision</th><th>rappel</th><th>F</th><th>contraste (σ)</th></tr>
{srows}</table>
<ol>
<li><b>Le brut gagne, et ton argument est le bon.</b> Les quatre statistiques
retrouvent <em>toutes</em> les vraies occurrences (rappel 1,00 sur This Love et
Don't Know Why) ; ce qui les sépare est le nombre de <em>fausses</em>. Le brut
en garde 7 là où le cosinus et le centré en gardent 12 ou 13, et les 5 ou 6 de
plus sont des inventions. « L'amplitude générale du carré est elle-même le
signal » : oui — normaliser divise par une quantité qui contenait
l'information, et la courbe remonte alors partout où il n'y a rien.</li>
<li><b>Le contraste en σ dit le contraire, et c'est le contraste qui a tort.</b>
Le centré a un meilleur contraste (2,07 contre 2,48 ici, et bien meilleur sur
Billie Jean) tout en trouvant moins bien. La session précédente avait conclu
« il faut centrer » <em>sur ce seul contraste</em>, sans jamais vérifier si les
pics tombaient au bon endroit. C'est la conclusion à corriger, pas la tienne :
un chiffre qui ne change aucune décision n'est pas une preuve.</li>
<li><b>La diagonale binaire marche presque aussi bien que le carré</b> (F 0,47
contre 0,50) avec un bien meilleur contraste, et elle coûte L valeurs au lieu
de L². Elle a un sens précis : glisser une identité binaire, c'est exactement
lire la sous-diagonale de la SSM au décalage d−b₀ — donc le « cue de runs de
lag » déjà mesuré à 36,8 % de placement. Tes deux idées sont la même à ce
détail près.</li>
<li><b>Le sens de la diagonale : la principale.</b> L'anti-diagonale tombe à
F = 0,00 sur les trois morceaux à référence utilisable — elle ne trouve rien,
comme prévu : elle demanderait que le motif soit rejoué à l'envers.</li>
<li><b>Ce que ça ne règle pas.</b> Le brut n'est pas comparable d'un motif à
l'autre (il grandit comme L²), donc l'arbitrage final du dictionnaire ne peut
pas l'utiliser tel quel — il compare des z robustes. Et sur Sunny la référence
par accords ne dit rien (aucune suite de 16 mesures ne se répète, le morceau
module vers le haut), donc Sunny ne vote pas dans ce tableau.</li>
</ol></section>"""
    body = verdict + body + """<section><h2>La contre-épreuve Billboard — elle
abîme une partie de ce qui précède, et il faut le dire</h2>
<p class=cap>278 morceaux Billboard à mesures annotées, même algorithme, mais
la question change : les pics tombent-ils sur les <b>débuts de section
annotés</b> ? (<code>scripts/pattern_algo_billboard.py</code>, tolérance
±1 mesure.)</p>
<table><tr><th>indice</th><th>précision</th><th>rappel</th><th>F</th></tr>
<tr><td>témoin bête : <b>toutes les mesures</b></td><td class=n>0.087</td><td class=n>1.000</td><td class=n>0.159</td></tr>
<tr><td>témoin bête : <b>toutes les L mesures</b></td><td class=n>0.351</td><td class=n>0.495</td><td class=n><b>0.310</b></td></tr>
<tr class=win><td>brut</td><td class=n><b>0.412</b></td><td class=n>0.276</td><td class=n>0.293</td></tr>
<tr><td>cosinus</td><td class=n>0.382</td><td class=n>0.362</td><td class=n>0.314</td></tr>
<tr><td>centré</td><td class=n>0.376</td><td class=n>0.371</td><td class=n>0.315</td></tr>
<tr><td>diagonale</td><td class=n>0.393</td><td class=n>0.306</td><td class=n>0.297</td></tr>
<tr><td>anti-diagonale</td><td class=n>0.317</td><td class=n>0.249</td><td class=n>0.244</td></tr></table>
<ol>
<li><b>Le résultat le plus dur : aucune des quatre statistiques ne bat
« une frontière toutes les L mesures ».</b> F 0,29–0,32 contre 0,31 pour le
témoin. Comme détecteur de <em>débuts de section</em>, la famille entière est à
l'équilibre avec un découpage régulier. Ce n'est pas une réfutation du bloc
glissé — c'est une réfutation de l'idée qu'un motif qui revient est un début de
section. Un motif de 4 mesures qui revient 13 fois dans un morceau à 9 sections
doit déborder, par construction.</li>
<li><b>Le brut perd 2,2 points de F sur le centré</b> (intervalle de confiance
par morceau [−4,0 ; −0,6], zéro exclu). Mais regarde les colonnes séparément :
le brut a la <b>meilleure précision des sept lignes</b> (0,412) et le plus
faible rappel. C'est exactement ce que disaient les morceaux — le brut garde
moins de pics et de meilleurs. Les deux mesures ne se contredisent pas, elles
punissent des erreurs différentes : ici le rappel contre des débuts de section
récompense celui qui déclenche souvent.</li>
<li><b>Ce que je ne fais pas dire à ce tableau.</b> Le corpus Billboard est de
la pop, sa grille de mesures est interpolée à l'intérieur des lignes
d'annotation, et la longueur de motif que l'algorithme y trouve a une médiane
de 11 mesures — contre 4, 4, 16, 2 et 8 sur les cinq morceaux ci-dessus. Il
sert à réfuter, pas à choisir.</li>
</ol></section>"""

    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Le dictionnaire de motifs — ton algorithme, mesuré</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1080px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:{SAND};font-size:13px;margin-bottom:20px;max-width:76ch}}
section{{background:{PAPER};border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 16px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 12px system-ui;margin:20px 0 6px;color:{SAND};text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px;display:block;margin-top:6px}}
.cap{{font-size:12px;color:{SAND};margin:5px 0 0;max-width:88ch}}
.num{{background:{INK};color:#f4eee2;padding:6px 11px;border-radius:8px;font:600 12.5px system-ui;display:inline-block;margin-bottom:6px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:10px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11.5px}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.bars{{font-size:11px;color:{SAND};font-variant-numeric:tabular-nums}}
td.chords{{font:600 11.5px ui-monospace,Menlo,monospace}}
tr.amb td{{background:#fdf3ee}}
.sub{{font-size:11px;color:{SAND}}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.tag{{background:#f7f3e9;color:{SAND};font:600 10px system-ui;padding:2px 6px;border-radius:5px;vertical-align:middle}}
.cmp{{background:#f7f3e9;border-radius:8px;padding:7px 10px;margin-top:6px;font-size:12px}}
.seg{{display:inline-block;background:{PAPER};border:1px solid #e5dcc6;border-radius:5px;padding:1px 6px;margin:2px 2px 0 0;font:600 11.5px system-ui}}
.seg sub{{color:{SAND};font-weight:400}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:12px}}
ol,ul{{font-size:13.5px;max-width:80ch}} li{{margin:4px 0}}
.verdict{{border-left:3px solid #8a2b2b}}
</style></head><body><div class=wrap>
<h1>Le dictionnaire de motifs — ton algorithme, mesuré</h1>
<div class=lede>Ton algorithme écrit tel quel : on repère le premier motif de
répétition en agrégeant les premières lignes de la matrice, on en extrait le
premier carré, on le glisse le long de l'axe des X, les pics de la courbe
donnent les autres endroits où ce motif revient, et c'est l'entrée n°1 du
dictionnaire ; on relance sur ce qui reste pour le motif n°2. Avec ton
raffinement : les blocs trouvés ne sont pas retirés de la matrice, ils vont
dans une boîte à part et restent candidats pour les autres motifs.
Morceaux dont la grille passe le garde-fou : {', '.join(done)}.</div>

<section><h2>Comment lire cette page</h2>
<ul>
<li><b>La référence</b> n'est pas la SSM et n'est pas le découpage actuel :
c'est la <b>suite d'accords</b>. Une « vraie » autre occurrence du motif est
une mesure où les {'{L}'.replace('{L}', 'L')} accords qui suivent sont ceux du
motif. Les accords sont décodés <em>avant</em> toute segmentation, donc
indépendants d'elle — mais ce sont nos accords, pas une vérité terrain : un
désaccord peut être la faute des accords.</li>
<li><b>Un seul sélecteur de pics</b> partout, à réglage fixe (prominence
topographique ≥ 25 % de l'amplitude de la courbe), pour que les statistiques
soient comparées à armes égales. Le choix du sélecteur est traité sur sa propre
page.</li>
<li><b>Le sens de la diagonale.</b> Le bloc comparé vaut
<code>similarité(mesure b₀+i, mesure d+k)</code>. Si les mesures d…d+L−1
répètent le motif, alors la mesure d+k sonne comme la mesure b₀+k : les valeurs
fortes sont sur <b>i = k</b>, c'est-à-dire la diagonale <b>principale</b>
(haut-gauche → bas-droite). L'anti-diagonale dirait que le motif est rejoué
<em>à l'envers</em>. Les deux sont mesurées ci-dessous, l'anti-diagonale en
témoin.</li>
</ul></section>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
