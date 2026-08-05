"""The whole peak study again, on the chord-tone SSM Louis chose.

« On utilise accords → notes distribution totale. Remontre-moi l'étude des
détections de pics avec cette matrice, tu me refais tous les plots. »

The substrate is now: musx's chord posterior per bar, projected onto the 12
pitch classes through the chord-tone matrix, full distribution (no top-k). The
dot product is therefore harmonic overlap — B♭ major and G minor share two notes.

Everything the earlier pages showed on the chroma SSM is redrawn here on that
matrix: the SSM itself, the sliding-block curve, the four peak selectors side by
side, and the margin + %-of-initial-peak sweep. Peak COUNTS are printed because
they describe what a selector does; no invented score appears anywhere.

    python scripts/peaks_on_harmonic.py   ->  /reports/peaks_on_harmonic.html
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
from harmonia_min import sections as hs, musx as mx           # noqa: E402
from ssm_rows_plot import fig2b64                             # noqa: E402
from ssm_harmonic import chord_tone_matrix, capture, unit     # noqa: E402
from peak_selectors import (sel_prominence, sel_quantile,     # noqa: E402
                            sel_margin, sel_topn)
from pattern_slide import off_diag, dominant_lag              # noqa: E402

OUT = HERE / "harmonia_min/state/reports/peaks_on_harmonic.html"
INK = "#1c1c1c"
DT = 23.22e-3
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why"),
         ("bobby_hebb_sunny_official_audio", "Sunny")]
SELECTORS = [("proéminence (1σ)", "#8a2b2b", "o", sel_prominence),
             ("quantile q90 de la courbe", "#1f8a5b", "s", sel_quantile),
             ("marge sur le fond local", "#2a6fb0", "^", sel_margin),
             ("top-N espacés de L", "#c58a2e", "D", sel_topn)]
FRACS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
FCOLS = ["#8a2b2b", "#c58a2e", "#1f8a5b", "#2a6fb0", "#7c3aed", "#0f766e"]
CHOSEN = 0.90


def harmonic_ssm(stem):
    """Chord posteriors → chord tones → 12-d per bar → SSM. Plus the sections
    the shipped detector currently draws, for the green reference lines."""
    cap = capture(stem)
    grid = cap["grid"]
    n = len(grid) - 1
    triad = mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0]
    M = chord_tone_matrix(triad.shape[1])
    P = []
    for b in range(n):
        a, z = int(grid[b] / DT), max(int(grid[b] / DT) + 1, int(grid[b + 1] / DT))
        seg = triad[a:min(z, len(triad))]
        P.append(seg.mean(0) if len(seg) else np.zeros(triad.shape[1]))
    V = unit(np.array(P) @ M)
    from harmonia_min import sections as _hs
    segs = _hs.detect_sections(grid, np.asarray(cap["arr"]), cap["times"], None)
    return V @ V.T, n, segs


def slide(S, n):
    flat = off_diag(S, n)
    L, _ = dominant_lag(S, n)
    thr = float(np.quantile(flat, 0.90))
    b0 = next((b for b in range(n - L) if S[b, b + L] >= thr), 0)
    P = S[b0:b0 + L, b0:b0 + L]
    raw = np.array([float((P * S[b0:b0 + L, c:c + L]).sum()) / (L * L)
                    for c in range(0, n - L + 1)])
    return raw, L, b0


def ssm_panel(ax, S, b0, L, boxes, col):
    ax.imshow(S, cmap="RdYlBu_r", origin="lower",
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    ax.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False, ec=INK, lw=2.2))
    for c in boxes:
        ax.add_patch(plt.Rectangle((c - .5, b0 - .5), L, L, fill=False,
                                   ec=col, lw=1.6, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])


def song_html(stem, title):
    S, n, segs = harmonic_ssm(stem)
    raw, L, b0 = slide(S, n)
    x = np.arange(len(raw))
    ref = float(raw[b0])

    # 0 — the matrix itself
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.imshow(S, cmap="RdYlBu_r", origin="lower",
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    for sg in segs:
        ax.axvline(sg["b0"], color="#1f8a5b", lw=.7, alpha=.5)
        ax.axhline(sg["b0"], color="#1f8a5b", lw=.7, alpha=.5)
    ax.set_xlabel("mesure", fontsize=8); ax.set_ylabel("mesure", fontsize=8)
    mat = fig2b64(fig)

    # 0b — PHASE: which period, and where a cycle starts
    lags = list(range(2, min(17, n - 2) + 1))
    prof = [float(np.mean([S[b, b + Lx] for b in range(n - Lx)])) for Lx in lags]
    K = min(16, n)
    agg = []
    for d in range(-min(K, n - 1), n - 1):
        vals = [S[b, b + d] for b in range(K) if 0 <= b + d < n]
        agg.append(float(np.mean(vals)) if vals else np.nan)
    aggx = np.arange(-min(K, n - 1), n - 1)
    phase = [float(np.mean([S[b, b + L] for b in range(n - L) if b % L == ph]))
             for ph in range(L)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 2.9))
    axs[0].bar(lags, prof, color=["#8a2b2b" if Lx == L else "#b9b09a" for Lx in lags])
    axs[0].set_title(f"similarité moyenne par décalage — max à {L}", fontsize=8.5, loc="left")
    axs[0].set_xlabel("décalage (mesures)", fontsize=8)
    axs[1].plot(aggx, agg, lw=1.2, color=INK)
    for m_ in range(1, 1 + int((n - 1) / L)):
        axs[1].axvline(m_ * L, color="#1f8a5b", lw=.8, alpha=.6)
    axs[1].axvline(0, color="#8a2b2b", lw=1.2)
    axs[1].set_title(f"agrégat des {K} premières lignes — verts = multiples de {L}",
                     fontsize=8.5, loc="left")
    axs[1].set_xlabel("décalage (mesures)", fontsize=8)
    axs[2].bar(range(L), phase, color=["#8a2b2b" if p_ == int(np.argmax(phase)) else "#b9b09a"
                                       for p_ in range(L)])
    axs[2].set_title(f"phase : quelle position modulo {L} tuile le mieux "
                     f"(→ {int(np.argmax(phase))})", fontsize=8.5, loc="left")
    axs[2].set_xlabel(f"position dans le cycle de {L}", fontsize=8)
    phase_fig = fig2b64(fig)

    # 1 — the four selectors on the curve
    picks = [f(raw, L) for _, _, _, f in SELECTORS]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(x, raw, lw=1.4, color=INK, zorder=3)
    for i, ((nm, col, mk, _), pk) in enumerate(zip(SELECTORS, picks)):
        ax.plot(x[pk], raw[pk] * (1 + .045 * (i + 1)), mk, color=col, ms=7,
                mew=1.4, mfc="none", label=f"{nm} — {len(pk)} pics", zorder=4)
    for sg in segs:
        ax.axvline(sg["b0"], color="#1f8a5b", lw=.8, alpha=.35, zorder=1)
    ax.axvline(b0, color="#8a2b2b", lw=1.6, zorder=2)
    ax.set_xlabel("décalage sur l'axe x, en mesures", fontsize=9)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    curve = fig2b64(fig)

    fig, axs = plt.subplots(1, 4, figsize=(4.0 * 4, 4.2))
    for a_, (nm, col, _, _), pk in zip(axs, SELECTORS, picks):
        ssm_panel(a_, S, b0, L, pk, col)
        a_.set_title(f"{nm}\n{len(pk)} pics", fontsize=8, color=col, loc="left")
    sel_mats = fig2b64(fig)

    # 2 — the validated rule swept
    base = sel_margin(raw, L, margin=0.5)
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(x, raw, lw=1.4, color=INK, zorder=3)
    ax.axvline(b0, color="#8a2b2b", lw=1.6, zorder=2)
    for sg in segs:
        ax.axvline(sg["b0"], color="#1f8a5b", lw=.8, alpha=.30, zorder=1)
    for f_, c in zip(FRACS, FCOLS):
        ax.axhline(f_ * ref, color=c, ls="--", lw=.9, alpha=.75)
        ax.text(len(raw) * 1.004, f_ * ref, f"{int(f_*100)}%", color=c,
                fontsize=7, va="center")
    for c_ in base:
        surv = max([f_ for f_ in FRACS if raw[c_] >= f_ * ref], default=None)
        if surv is None:
            ax.plot(c_, raw[c_], "x", color="#b9b09a", ms=7, zorder=4)
        else:
            ax.plot(c_, raw[c_], "o", color=FCOLS[FRACS.index(surv)], ms=8,
                    mfc="none", mew=1.8, zorder=5)
    ax.set_xlabel("décalage (mesures) · cercle = retenu par la marge, couleur = "
                  "seuil le plus exigeant franchi · × = rejeté", fontsize=9)
    sweep = fig2b64(fig)

    fig, axs = plt.subplots(1, len(FRACS), figsize=(3.1 * len(FRACS), 3.4))
    counts = []
    for a_, f_, c in zip(axs, FRACS, FCOLS):
        pk = np.array([k for k in base if raw[k] >= f_ * ref], int)
        counts.append(len(pk))
        ssm_panel(a_, S, b0, L, pk, c)
        a_.set_title(f"≥ {int(f_*100)}% — {len(pk)} pics"
                     + (" ←" if abs(f_ - CHOSEN) < 1e-9 else ""),
                     fontsize=8.5, color=c, loc="left")
    sweep_mats = fig2b64(fig)

    rows = "".join(
        f"<tr><td style='color:{c}'><b>≥ {int(f_*100)}%"
        f"{' ← retenu' if abs(f_-CHOSEN)<1e-9 else ''}</b></td><td>{k}</td>"
        f"<td>{sorted(int(g) for g in np.diff(sorted([q for q in base if raw[q] >= f_*ref]))) if k > 1 else '—'}</td></tr>"
        for f_, c, k in zip(FRACS, FCOLS, counts))

    return f"""<section><h2>{title} <span class=sub>{n} mesures · motif de {L}
mesures pris à la mesure {b0}</span></h2>
<h3>La matrice</h3><img src="data:image/png;base64,{mat}">
<p class=cap>Traits verts = les frontières que le détecteur actuel pose.</p>
<h3>Phase : quelle période, et où commence un cycle</h3>
<img src="data:image/png;base64,{phase_fig}">
<p class=cap>À gauche, la similarité moyenne à chaque décalage — le maximum
donne la période. Au milieu, l'agrégat des premières lignes que tu décrivais,
avec les multiples de la période en vert. À droite, laquelle des {L} positions
du cycle tuile le mieux : c'est la phase.</p>
<h3>Les quatre sélecteurs sur la courbe du bloc glissé</h3>
<img src="data:image/png;base64,{curve}">
<img src="data:image/png;base64,{sel_mats}">
<h3>La règle validée : marge locale + X % du pic initial</h3>
<img src="data:image/png;base64,{sweep}">
<img src="data:image/png;base64,{sweep_mats}">
<table><tr><th>seuil</th><th>pics</th><th>écarts entre pics</th></tr>{rows}</table>
</section>"""


def main():
    body = ""
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            continue
        body += song_html(stem, title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Détection de pics — matrice harmonique</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1400px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
h3{{font:700 11px system-ui;margin:18px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px;margin-bottom:6px}}
.cap{{font-size:12px;color:#8a8371;margin:0 0 4px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
</style></head><body><div class=wrap>
<h1>Détection de pics — sur la matrice harmonique</h1>
<div class=lede>Substrat : le postérieur d'accords musx par mesure, projeté sur
les 12 notes par la matrice des notes d'accords, <b>distribution complète</b>.
Le produit scalaire est donc du recouvrement harmonique. Toute l'étude des pics
est redessinée dessus : la matrice, la courbe du bloc glissé, les quatre
sélecteurs, et le balayage de ta règle validée.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
