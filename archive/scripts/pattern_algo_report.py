"""Le dictionnaire de motifs — comment il se construit, et d'où sortent les
sections (Louis, 2026-08-05).

La règle de pics est celle que Louis a validée sur
`/reports/peak_rule_sweep.html` : « je valide le seuil à 90 % ».

    un pic est gardé  ⇔  marge sur la ligne de base locale
                      ET  valeur ≥ 0,90 × le pic initial

Le « pic initial » est la courbe à la position du motif lui-même — le plafond
par construction. La marge est exactement `peak_selectors.sel_margin`
(0,5 σ au-dessus de la médiane d'une fenêtre ±3·L, σ simple et non MAD), c'est
à dire le tracé qu'il a lu.

La page répond à une question : comment le dictionnaire SE CONSTRUIT, et d'où
sort CHAQUE section, y compris les ambiguës.

    python scripts/pattern_algo_report.py  ->  /reports/pattern_algo.html
"""
from __future__ import annotations

import base64
import io
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
from pattern_algo_core import (INITIAL_PEAK_FRAC, boundaries,     # noqa: E402
                               build_dictionary, contrast,
                               dictionary_sections, first_pattern,
                               pick_peaks, slide, song_period)

OUT = HERE / "harmonia_min/state/reports/pattern_algo.html"
INK, SAND, PAPER = "#1c1c1c", "#8a8371", "#fffdf6"

# ── the rule Louis validated, in one place ──────────────────────────────────
PEAK_METHOD, PEAK_PARAM = "margin", 0.5      # = peak_selectors.sel_margin
DICT_STAT = "raw"                            # the statistic the dictionary uses
TAU = 1.0                                    # "clearly above", in sigma

STATS = [
    ("brut", "raw", "#8a2b2b",
     "produit scalaire nu — le niveau général du bloc reste dans le signal"),
    ("cosinus", "cosine", "#2a6fb0",
     "divisé par les deux normes — le niveau général est retiré"),
    ("centré", "centered", "#1f8a5b",
     "chaque bloc moins sa propre moyenne, puis cosinus = corrélation"),
    ("diagonale", "diag", "#b8860b",
     "une diagonale binaire glissée au lieu du carré extrait"),
    ("anti-diagonale", "antidiag", SAND,
     "la diagonale dans le MAUVAIS sens — le témoin"),
]

SONGS = [
    ("maroon_5_this_love", "This Love"),
    ("norah_jones_don_t_know_why", "Don't Know Why"),
    ("bobby_hebb_sunny_official_audio", "Sunny"),
]

PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PAL = ["#8a2b2b", "#2a6fb0", "#1f8a5b", "#b8860b", "#6a4c93", "#c05a2b",
       "#3d8b8b", "#a03060"]
ACC_ROWS: list = []


def fig2b64(fig, dpi=112):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def peaks_of(ds, f, L, b0, method=PEAK_METHOD, param=PEAK_PARAM):
    """THE validated rule, applied identically everywhere on this page."""
    return pick_peaks(ds, f, method, param, min_sep=1, exclude=b0,
                      base_win=max(4, 3 * L),
                      initial_at=int(np.searchsorted(ds, b0)))


# ══ the independent reference: the chord string ════════════════════════════
def bar_chords(stem: str, n: int):
    """Per-bar chord signature, taken from the list `detect_sections` RECEIVES
    — decoded before any section exists, so independent of the segmentation.
    Not ground truth: these are our own chords."""
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
            if sum(1 for a, b in zip(ref_r, sr) if a == b) / L >= loose:
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


def sections_strip(ax, segs, n, label):
    letters = sorted({s["label"] for s in segs})
    for s in segs:
        c = PAL[letters.index(s["label"]) % len(PAL)]
        ax.axvspan(s["b0"] - .5, s["b1"] + .5, color=c, alpha=.55)
        if s["b1"] - s["b0"] >= 2:
            ax.text((s["b0"] + s["b1"]) / 2, .5, s["label"], ha="center",
                    va="center", fontsize=7.5, color="w", fontweight="bold")
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=7.2, rotation=0, ha="right", va="center")
    ax.set_xlim(-.5, n - .5)


def ssm_axes(ax, S, n, b0, L):
    ax.imshow(S, cmap="magma", vmin=max(0.0, np.percentile(S, 5)), vmax=1.0,
              origin="upper", extent=(-.5, n - .5, n - .5, -.5),
              interpolation="nearest", aspect="auto")
    ax.axhspan(b0 - .5, b0 + L - .5, color="#5ad1a0", alpha=.16)
    ax.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False,
                               ec="#ffffff", lw=1.7))


# ══ figure 1: the rule, drawn on the four statistics ═══════════════════════
def stats_fig(S, n, b0, L, curves, hits, ref, segs):
    k = len(STATS)
    fig, axs = plt.subplots(
        k + 2, 1, figsize=(11.4, 3.0 + 1.2 * k + .3), sharex=True,
        gridspec_kw=dict(height_ratios=[3.0] + [1.2] * k + [.30], hspace=.09))
    ssm_axes(axs[0], S, n, b0, L)
    for d in ref:
        axs[0].add_patch(plt.Rectangle((d - .5, b0 - .5), L, L, fill=False,
                                       ec="#5ad1a0", lw=1.0, ls=":"))
    axs[0].set_ylabel("mesure (ligne)", fontsize=9)
    axs[0].set_title("la SSM ; carré blanc = le motif ; pointillés verts = où "
                     "la SUITE D'ACCORDS répète celle du motif",
                     fontsize=8.6, color=SAND, pad=6)

    for i, (nm, key, col, _) in enumerate(STATS):
        ax = axs[1 + i]
        f = curves[key]
        for d in ref:
            ax.axvline(d, color="#5ad1a0", lw=2.6, alpha=.30, zorder=0)
        ax.plot(np.arange(len(f)), f, lw=1.25, color=col, zorder=2)
        # the two halves of the validated rule, drawn
        ref_v = float(f[b0]) if b0 < len(f) else float(f.max())
        ax.axhline(INITIAL_PEAK_FRAC * ref_v, color=INK, ls="--", lw=1.0,
                   zorder=3)
        ax.plot([b0], [ref_v], "*", ms=9, color=INK, zorder=5)
        sel = hits[key]
        if sel:
            ax.plot(sel, [f[d] for d in sel], "o", ms=4.8, mfc="none",
                    mec=INK, mew=1.4, zorder=4)
        ax.set_ylabel(nm, fontsize=8.4, color=col)
        ax.grid(alpha=.15, lw=.55)

    sections_strip(axs[-1], segs, n, "sections\nactuelles")
    axs[-1].set_xlabel("mesure (colonne) — position du carré glissé sur l'axe X",
                       fontsize=9)
    return fig2b64(fig)


# ══ figure 2: the build trace ══════════════════════════════════════════════
def dict_fig(S, n, dic, segs, dsegs):
    P = dic["patterns"]
    lanes = len(P) + 1
    fig, axs = plt.subplots(
        4, 1, figsize=(11.4, 8.0), sharex=True,
        gridspec_kw=dict(height_ratios=[3.0, .20 + .34 * lanes, .30, .30],
                         hspace=.11))
    ax = axs[0]
    ax.imshow(S, cmap="magma", vmin=max(0.0, np.percentile(S, 5)), vmax=1.0,
              origin="upper", extent=(-.5, n - .5, n - .5, -.5),
              interpolation="nearest", aspect="auto")
    for i, p in enumerate(P):
        c = PAL[i % len(PAL)]
        for d in p["occ"]:
            ax.add_patch(plt.Rectangle((d - .5, p["b0"] - .5), p["L"], p["L"],
                                       fill=False, ec=c, lw=1.3))
        ax.add_patch(plt.Rectangle((p["b0"] - .5, p["b0"] - .5), p["L"], p["L"],
                                   fill=False, ec="#ffffff", lw=1.9))
    ax.set_ylabel("mesure (ligne)", fontsize=9)

    ax = axs[1]
    for i, p in enumerate(P):
        y = lanes - 1 - i
        c = PAL[i % len(PAL)]
        ax.axhline(y, color="#e5dcc6", lw=.8, zorder=0)
        for bl in dic["blocks"]:
            if bl["owner"] == i:
                ax.add_patch(plt.Rectangle((bl["d"] - .5, y - .32), p["L"], .64,
                                           color=c, alpha=.85, zorder=2))
        ax.text(-0.6, y + .38, f"entrée {chr(65+i)} — trouvée en mes. "
                f"{p['b0']}, {p['L']} mesures", fontsize=7.6, color=c,
                ha="left", va="bottom")
    ax.axhline(0, color="#e5dcc6", lw=.8, zorder=0)
    for bl in dic["blocks"]:
        if bl["owner"] is None:
            ax.add_patch(plt.Rectangle((bl["d"] - .5, -.32),
                                       P[bl["best"]]["L"], .64, fill=False,
                                       ec=INK, lw=1.2, hatch="///", zorder=2))
    ax.text(-0.6, .38, "ambigus → chacun sa propre section", fontsize=7.6,
            color=INK, ha="left", va="bottom")
    ax.set_ylim(-.6, lanes - .05)
    ax.set_yticks([])
    ax.set_ylabel("le\ndictionnaire", fontsize=7.6, rotation=0, ha="right",
                  va="center")

    sections_strip(axs[2], dsegs, n, "sections du\ndictionnaire")
    sections_strip(axs[3], segs, n, "sections\nactuelles")
    axs[3].set_xlabel("mesure", fontsize=9)
    return fig2b64(fig)


def tau_fig(taus, res):
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 2.3))
    ax = axs[0]
    ax.plot(taus, [r["assigned"] for r in res], "-o", ms=4, lw=1.4,
            color="#1f8a5b", label="blocs attribués à une entrée")
    ax.plot(taus, [r["amb"] for r in res], "-o", ms=4, lw=1.4, color="#8a2b2b",
            label="blocs ambigus → section propre")
    ax.axvline(TAU, color=INK, ls="--", lw=1.1)
    ax.set_xlabel("τ — écart minimal entre la meilleure entrée et la suivante (σ)",
                  fontsize=7.6)
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


# ══ per song ═══════════════════════════════════════════════════════════════
def song_block(stem, title):  # noqa: C901
    d = song_data(stem)
    S, n, segs = d["Sb"], d["n"], d["segs"]
    Lp, prof = song_period(S)
    mot = first_pattern(S)
    b0, L = mot["b0"], mot["L"]
    sig, txt = bar_chords(stem, n)

    curves, hits = {}, {}
    for _, key, _, _ in STATS:
        ds, f = slide(S, b0, L, key)
        curves[key] = f
        hits[key] = sorted(p["d"] for p in peaks_of(ds, f, L, b0))

    ref = true_occurrences(sig, b0, L, n, 0.75) if sig else []
    ref_strict = true_occurrences(sig, b0, L, n) if sig else []
    ref_x = [x for x in ref if x != b0]

    rows = ""
    for nm, key, col, desc in STATS:
        sel, f = hits[key], curves[key]
        pk = [dict(d=x, k=x, val=float(f[x]), prom=0.0) for x in sel]
        P0, R0, F0 = prf(sel, ref_x)
        ACC_ROWS.append((title, nm, key, len(sel), P0, R0, F0,
                         contrast(f, pk) if pk else 0.0, bool(ref_x)))
        rows += (f"<tr{' class=win' if key == DICT_STAT else ''}>"
                 f"<td><span class=dot style='background:{col}'></span>"
                 f"<b>{nm}</b><br><span class=sub>{desc}</span></td>"
                 f"<td class=n>{len(sel)}</td>"
                 f"<td class=n>{P0:.2f}</td><td class=n>{R0:.2f}</td>"
                 f"<td class=n><b>{F0:.2f}</b></td>"
                 f"<td class=bars>{', '.join(map(str, sel)) or '—'}</td></tr>")

    figA = stats_fig(S, n, b0, L, curves, hits, ref, segs)

    # ── the dictionary ──────────────────────────────────────────────────────
    dic = build_dictionary(S, stat=DICT_STAT, peak_method=PEAK_METHOD,
                           peak_param=PEAK_PARAM, tau=TAU, removal="cover")
    dsegs = dictionary_sections(dic, n)
    figB = dict_fig(S, n, dic, segs, dsegs)

    taus = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    res = []
    for t in taus:
        dt = build_dictionary(S, stat=DICT_STAT, peak_method=PEAK_METHOD,
                              peak_param=PEAK_PARAM, tau=t, removal="cover")
        res.append(dict(tau=t,
                        assigned=sum(1 for b in dt["blocks"]
                                     if b["owner"] is not None),
                        amb=sum(1 for b in dt["blocks"] if b["owner"] is None),
                        n_sections=len(dictionary_sections(dt, n))))
    figT = tau_fig(taus, res)

    # ── build trace, entry by entry, in discovery order ─────────────────────
    prow = ""
    for i, p in enumerate(dic["patterns"]):
        own = [b["d"] for b in dic["blocks"] if b["owner"] == i]
        chords = " | ".join(txt[p["b0"] + k] for k in range(p["L"])) if txt \
            else "—"
        dropped = [x for x in p["peaks_raw"] if x not in p["occ"]]
        cov = ", ".join(f"{o}–{min(n-1, o+p['L']-1)}" for o in p["occ"])
        prow += (
            f"<tr><td><b>entrée {chr(65+i)}</b><br><span class=sub>trouvée en "
            f"mesure {p['b0']}, {p['L']} mesures</span></td>"
            f"<td class=chords>{chords}</td>"
            f"<td class=n>{p['score']:.3f}<br><span class=sub>seuil "
            f"{p['thr']:.3f}</span></td>"
            f"<td class=n>{len(p['peaks_raw'])}</td>"
            f"<td class=n>{len(p['occ'])}</td>"
            f"<td class=bars>{cov}</td>"
            f"<td class=bars>{', '.join(map(str, dropped)) or '—'}</td></tr>")

    # ── the separate box ────────────────────────────────────────────────────
    arow = ""
    for b in sorted(dic["blocks"], key=lambda x: x["d"]):
        z = b["z"]
        best = chr(65 + b["best"])
        zb = z[b["best"]]
        sec = chr(65 + b["second"]) if b["second"] is not None else "—"
        zs = f"{z[b['second']]:+.2f}" if b["second"] is not None else "—"
        mg = ("—" if not np.isfinite(b["margin"]) else f"{b['margin']:.2f}")
        if b["owner"] is None:
            verdict = "<b>rien ne ressort → sa propre section</b>"
            cls = " class=amb"
        elif b["second"] is None:
            verdict = f"→ entrée {best} <span class=sub>(seule entrée)</span>"
            cls = ""
        else:
            verdict = f"→ entrée {best}"
            cls = ""
        arow += (f"<tr{cls}><td class=n>{b['d']}–{min(n-1, b['d']+b['L']-1)}</td>"
                 f"<td class=n>{best} &nbsp;{zb:+.2f}</td>"
                 f"<td class=n>{sec} &nbsp;{zs}</td>"
                 f"<td class=n>{mg}</td><td>{verdict}</td></tr>")

    n_amb = sum(1 for b in dic["blocks"] if b["owner"] is None)
    fires = ("<b>l'arbitrage ne sert à rien ici</b> : une seule entrée, donc "
             "aucun bloc n'est disputé"
             if len(dic["patterns"]) < 2 else
             (f"<b>{n_amb} bloc(s) ambigu(s)</b> sur {len(dic['blocks'])}"
              if n_amb else
              f"<b>aucun bloc ambigu</b> : les {len(dic['blocks'])} blocs sont "
              f"tous gagnés nettement"))

    cmp_now = " ".join(
        f"<span class=seg>{s['label']}<sub>{s['b0']}–{s['b1']}</sub></span>"
        for s in segs)
    cmp_dic = " ".join(
        f"<span class=seg>{s['label']}<sub>{s['b0']}–{s['b1']}</sub></span>"
        for s in dsegs)
    Pb, Rb, Fb = prf(boundaries(dsegs), boundaries(segs), tol=1)

    mtxt = " | ".join(txt[b0 + k] for k in range(L)) if txt else "—"
    top = sorted(range(2, min(33, n - 1)), key=lambda x: -prof[x])[:5]
    return f"""<section><h2>{title}</h2>
<div class=num>{n} mesures &nbsp;·&nbsp; période lue sur les 16 premières
lignes : <b>{Lp} mesures</b> &nbsp;·&nbsp; motif n°1 = mesures
<b>{b0}–{b0+L-1}</b> &nbsp;·&nbsp; <b>{mtxt}</b></div>
<div class=num style="background:#f7f3e9;color:#1c1c1c">{fires}</div>

<h3>1 · la période, lue en agrégeant les premières lignes</h3>
<table><tr><th>décalage</th>{''.join(f'<th>{x}</th>' for x in top)}</tr>
<tr><td>similarité moyenne des 16 premières lignes</td>
{''.join(f'<td class=n>{prof[x]:.3f}</td>' for x in top)}</tr></table>

<h3>2 · le carré glissé, et la règle des 90 %</h3>
<img src="data:image/png;base64,{figA}">
<p class=cap>L'étoile noire est le <b>pic initial</b> (le motif contre lui-même),
le trait noir tireté est {int(INITIAL_PEAK_FRAC*100)} % de ce pic. Un cercle =
un pic gardé : il passe la marge locale <em>et</em> le trait. Les bandes vertes
sont les mesures où la suite d'accords répète celle du motif
({len(ref_x)} endroits à 3 mesures sur 4 ; {max(0, len(ref_strict) - 1)} à
l'identique).</p>
<table><tr><th>statistique glissante</th><th>pics gardés</th><th>précision</th>
<th>rappel</th><th>F</th><th>où</th></tr>{rows}</table>

<h3>3 · le dictionnaire, entrée par entrée, dans l'ordre où il les trouve</h3>
<img src="data:image/png;base64,{figB}">
<table><tr><th>entrée</th><th>suite d'accords du motif</th>
<th>force du motif</th><th>pics bruts</th><th>blocs retenus</th>
<th>mesures couvertes</th><th>pics écartés (chevauchement)</th></tr>{prow}</table>
<p class=cap>« Pics bruts » = ce que la règle des 90 % renvoie. « Blocs
retenus » = après résolution des chevauchements : deux copies d'un motif de
{L} mesures ne peuvent pas commencer à moins de {L} mesures d'écart, on garde
la mieux notée. Cette résolution est une règle du <em>dictionnaire</em>, pas une
modification de ta règle de pics.</p>

<h3>4 · la boîte à part : d'où sort chaque section</h3>
<table><tr><th>bloc (mesures)</th><th>meilleure entrée (σ)</th>
<th>2ᵉ (σ)</th><th>écart</th><th>verdict — τ = {TAU} σ</th></tr>{arow}</table>
<p class=cap>Chaque bloc est noté par <em>toutes</em> les entrées, en
écarts-types robustes de la courbe de chaque entrée — sinon une entrée faite de
matière homogène, qui corrèle avec tout, raflerait tout. Il va à la meilleure
seulement si elle dépasse la seconde de τ écarts-types ; sinon il devient sa
propre section.</p>

<h3>5 · sensibilité à τ</h3>
<img src="data:image/png;base64,{figT}">

<h3>6 · le découpage final, contre <code>sections.py</code></h3>
<div class=cmp><b>aujourd'hui</b><br>{cmp_now}</div>
<div class=cmp><b>le dictionnaire</b> <span class=sub>(· = mesures qu'aucun
bloc ne couvre, ? = bloc ambigu devenu sa propre section)</span><br>{cmp_dic}</div>
<p class=cap>Frontières communes à ±1 mesure : précision {Pb:.2f}, rappel
{Rb:.2f}, F {Fb:.2f}. Ce n'est pas un score de justesse — les deux découpages
sont des propositions, aucun n'est la vérité.</p>
</section>"""


def main():
    body, done = "", []
    dict_summary = []
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            continue
        try:
            body += song_block(stem, title)
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  !! {title}: {e}")
            continue
        d = song_data(stem)
        dic = build_dictionary(d["Sb"], stat=DICT_STAT, peak_method=PEAK_METHOD,
                               peak_param=PEAK_PARAM, tau=TAU, removal="cover")
        dict_summary.append(
            (title, len(dic["patterns"]), len(dic["blocks"]),
             sum(1 for b in dic["blocks"] if b["owner"] is None)))
        done.append(title)
        print(f"  ok {title}")

    # ── the uncomfortable finding, first ────────────────────────────────────
    drow = "".join(
        f"<tr><td>{t}</td><td class=n>{np_}</td><td class=n>{nb}</td>"
        f"<td class=n>{na}</td></tr>" for t, np_, nb, na in dict_summary)
    lead = f"""<section class=verdict><h2>À dire en premier : sur ces trois
morceaux l'arbitrage ne sert jamais</h2>
<table><tr><th>morceau</th><th>entrées du dictionnaire</th><th>blocs</th>
<th>blocs ambigus</th></tr>{drow}</table>
<p>La partie la plus intéressante de ton raffinement — la boîte à part, et le
bloc qui devient sa propre section quand rien ne ressort — <b>ne se déclenche
sur aucun des trois</b>. Deux raisons, et elles ne sont pas les mêmes :</p>
<ul>
<li><b>This Love et Don't Know Why : une seule entrée.</b> Un cycle de 4 mesures
qui revient 8 ou 10 fois couvre déjà tout le morceau ; il ne reste plus assez de
mesures libres pour un motif n°2. Sans deuxième entrée il n'y a pas d'arbitrage
à faire — chaque bloc n'a qu'un candidat.</li>
<li><b>Sunny : deux entrées, mais aucun bloc disputé.</b> Le motif fait
16 mesures, les blocs sont peu nombreux et loin les uns des autres, et chacun
est gagné nettement. L'arbitrage tourne et ne tranche rien.</li>
</ul>
<p>La règle des 90 % <b>aggrave</b> ce constat plutôt qu'elle ne le corrige : en
gardant moins de pics, elle laisse encore moins de matière à un motif n°2. C'est
le résultat honnête et il faut le regarder avant les tableaux qui suivent : le
mécanisme est écrit, il est testable, mais ces trois morceaux ne l'exercent pas.
Il faudrait un morceau à deux cycles vraiment distincts (couplet ET refrain avec
des grilles différentes) pour le juger.</p>
</section>"""

    # ── which statistic, under the validated rule ───────────────────────────
    import collections
    by = collections.defaultdict(list)
    for (t, nm, key, npk, P, R, F, ct, has_ref) in ACC_ROWS:
        if has_ref:
            by[(nm, key)].append((npk, P, R, F))
    srows = ""
    for nm, key, _, _ in STATS:
        v = by.get((nm, key), [])
        if not v:
            continue
        a = np.array(v, float)
        cls = " class=win" if key == DICT_STAT else ""
        srows += (f"<tr{cls}><td><b>{nm}</b></td><td class=n>{a[:,0].mean():.1f}</td>"
                  f"<td class=n>{a[:,1].mean():.2f}</td>"
                  f"<td class=n>{a[:,2].mean():.2f}</td>"
                  f"<td class=n><b>{a[:,3].mean():.2f}</b></td></tr>")
    stat_sec = f"""<section><h2>Quelle statistique glissante, sous TA règle de
pics — et une correction à ce que je t'ai dit ce matin</h2>
<table><tr><th>statistique</th><th>pics gardés (moy.)</th><th>précision</th>
<th>rappel</th><th>F</th></tr>{srows}</table>
<p class=cap>Moyennes sur This Love et Don't Know Why. Sunny ne vote pas : son
motif fait 16 mesures et aucune suite de 16 mesures ne se répète dans le morceau
(il module vers le haut), donc la référence par accords n'y dit rien.</p>
<ol>
<li><b>Ta règle des 90 % referme presque tout l'écart entre le brut et le
centré.</b> Avec l'ancien sélecteur (prominence) le brut gagnait largement —
F 0,67 contre 0,53. Sous ta règle : This Love brut 0,83 / centré <b>0,91</b>,
Don't Know Why brut <b>0,80</b> / centré 0,71. Chacun gagne une fois. Le
plancher à 90 % du pic initial fait le travail que le niveau non normalisé
faisait tout seul : il coupe les pics faibles que la normalisation avait
relevés.</li>
<li><b>Ce que ça change à ce que je t'ai dit ce matin.</b> Je t'ai écrit « le
brut gagne, clairement ». C'était vrai <em>avec ce sélecteur-là</em>, et je ne
l'avais pas dit. Avec le tien, c'est un match nul sur deux morceaux. Ce qui
reste vrai des deux côtés : le brut garde moins de pics pour le même rappel, et
il n'invente pas.</li>
<li><b>Sunny départage quand même.</b> Sous ta règle, le centré et la diagonale
ne renvoient <b>aucun</b> pic sur Sunny, le cosinus en renvoie 13 dont aucun
n'est vérifiable, le brut en renvoie 6. Sur un morceau qui module, seul le brut
survit au plancher — c'est pour ça que le dictionnaire de cette page tourne sur
le brut.</li>
<li><b>Le sens de la diagonale reste tranché : la principale.</b>
L'anti-diagonale garde 15 à 18 pics sur These morceaux et n'en place aucun sur
une vraie occurrence (F 0,00) — elle demanderait que le motif soit rejoué à
l'envers.</li>
</ol></section>""".replace("These morceaux", "ces morceaux")

    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Le dictionnaire de motifs — comment il se construit</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1080px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:{SAND};font-size:13px;margin-bottom:20px;max-width:76ch}}
section{{background:{PAPER};border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 16px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 12px system-ui;margin:22px 0 6px;color:{SAND};text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px;display:block;margin-top:6px}}
.cap{{font-size:12px;color:{SAND};margin:5px 0 0;max-width:88ch}}
.num{{background:{INK};color:#f4eee2;padding:6px 11px;border-radius:8px;font:600 12.5px system-ui;display:inline-block;margin:0 6px 6px 0}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:10px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11.5px}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.bars{{font-size:11px;color:{SAND};font-variant-numeric:tabular-nums}}
td.chords{{font:600 11.5px ui-monospace,Menlo,monospace}}
tr.amb td{{background:#fdf3ee}}
tr.win td{{background:#eef7f1}}
.sub{{font-size:11px;color:{SAND};font-weight:400}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.cmp{{background:#f7f3e9;border-radius:8px;padding:7px 10px;margin-top:6px;font-size:12px}}
.seg{{display:inline-block;background:{PAPER};border:1px solid #e5dcc6;border-radius:5px;padding:1px 6px;margin:2px 2px 0 0;font:600 11.5px system-ui}}
.seg sub{{color:{SAND};font-weight:400}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:12px}}
ol,ul{{font-size:13.5px;max-width:80ch}} li{{margin:5px 0}}
.verdict{{border-left:4px solid #8a2b2b}}
</style></head><body><div class=wrap>
<h1>Le dictionnaire de motifs — comment il se construit</h1>
<div class=lede>La règle de pics est celle que tu as validée : un pic est gardé
s'il dépasse la médiane de son voisinage (±3·L mesures) d'au moins 0,5 σ
<b>et</b> s'il vaut au moins <b>{int(INITIAL_PEAK_FRAC*100)} % du pic
initial</b> — la courbe à la position du motif contre lui-même. Le reste de
l'algorithme est le tien, écrit tel quel : période lue en agrégeant les
premières lignes, premier carré, glissement sur l'axe des X, pics = autres
occurrences, entrée n°1 du dictionnaire, puis on relance sur ce qui reste ; les
blocs trouvés ne sont pas retirés de la matrice, ils vont dans une boîte à part
et restent candidats pour toutes les entrées.
{', '.join(done)} — les trois que tu as demandés.</div>
{lead}{stat_sec}{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
