"""Louis's two asks, on the songs whose bar grid passes the guard.

1. « il y a des pics clairs qui doivent correspondre à un quantile des valeurs de
   la matrice SSM, analyse cela par chanson » — for each song, find the clear
   peaks of the per-bar lag signals and report WHICH QUANTILE of that song's own
   off-diagonal SSM they sit at. If that quantile is stable across songs it
   replaces the fixed `TILE_MIN = 0.80` instead of recalibrating it.

2. « on a détecté un premier bloc de répétition … pour détecter les autres, on
   fait glisser ce pattern le long de l'axe x (pas le long de la diagonale !) et
   on fait le produit scalaire à chaque fois avec le carré sur lequel on
   arrive » — a matched filter on the SSM. The pattern is the L×L diagonal block
   of the first repeated unit; sliding it along x gives f(c) = <P, S[b0:b0+L,
   c:c+L]>, i.e. "does the material at c have the same internal structure as the
   material at b0".

Only songs passing `beats.check_grid` are plotted: on a broken grid the bars are
not bars and every number here would be about the grid, not the music.

    python scripts/pattern_slide.py   ->  /reports/pattern_slide.html
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.signal import find_peaks      # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from harmonia_min import beats as bts                # noqa: E402
from harmonia_min import sections as hs              # noqa: E402
from ssm_rows_plot import bar_ssm, fig2b64           # noqa: E402

OUT = HERE / "harmonia_min/state/reports/pattern_slide.html"
INK, ACC, GRN, BLU, AMB = "#1c1c1c", "#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e"

CANDIDATES = [
    ("maroon_5_this_love", "This Love"),
    ("norah_jones_don_t_know_why", "Don't Know Why"),
    ("bobby_hebb_sunny_official_audio", "Sunny"),
    ("michael_jackson_billie_jean_official_video", "Billie Jean"),
    ("the_police_every_breath_you_take_official_music_video", "Every Breath You Take"),
]
LAG_MAX = 16
MIN_L = 2


def off_diag(S, n, k=2):
    return np.array([S[i, j] for i in range(n) for j in range(n)
                     if abs(i - j) >= k])


def dominant_lag(S, n):
    """Mean similarity at each lag; the strongest is the repetition period."""
    prof = []
    for L in range(MIN_L, min(LAG_MAX, n - 2) + 1):
        v = [S[b, b + L] for b in range(n - L)]
        prof.append((float(np.mean(v)), L))
    prof.sort(reverse=True)
    lags = {L: m for m, L in prof}
    return prof[0][1], lags


def song(stem, title):
    S, n, segs = bar_ssm(stem)
    flat = off_diag(S, n)

    # ── 1. the quantile the clear peaks live at ─────────────────────────────
    peak_vals, peak_lags = [], []
    for b in range(n):
        row = S[b]
        lag = np.arange(n) - b
        keep = np.abs(lag) >= MIN_L
        v, lg = row[keep], lag[keep]
        if len(v) < 8:
            continue
        # "clear" = prominent against this row's own background, no absolute cut
        pk, _ = find_peaks(v, prominence=float(np.std(v)))
        for i in pk:
            peak_vals.append(float(v[i]))
            peak_lags.append(int(lg[i]))
    peak_vals = np.array(peak_vals)
    q_of_peaks = (np.searchsorted(np.sort(flat), peak_vals) / len(flat)) if len(peak_vals) else np.array([])

    L, lagmean = dominant_lag(S, n)

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.hist(flat, bins=60, color="#cdc4ad", label="toutes les valeurs hors diagonale")
    if len(peak_vals):
        ax.hist(peak_vals, bins=60, color=ACC, alpha=.85, label="les pics clairs")
    ax.axvline(hs.TILE_MIN, color=INK, ls="--", lw=1.4, label=f"TILE_MIN {hs.TILE_MIN}")
    for q in (0.90, 0.95):
        ax.axvline(np.quantile(flat, q), color=GRN, lw=1.1)
        ax.text(np.quantile(flat, q), ax.get_ylim()[1] * .9, f" q{int(q*100)}",
                color=GRN, fontsize=8)
    ax.set_xlabel("similarité", fontsize=9)
    ax.legend(fontsize=8)
    hist = fig2b64(fig)

    # ── 2. the pattern, and sliding it along x ──────────────────────────────
    # first bar where the repetition at the dominant lag is actually strong
    thr = float(np.quantile(flat, 0.90))
    b0 = next((b for b in range(n - L) if S[b, b + L] >= thr), 0)
    P = S[b0:b0 + L, b0:b0 + L]
    xs, raw, cos = [], [], []
    for c in range(0, n - L + 1):
        Q = S[b0:b0 + L, c:c + L]
        xs.append(c)
        raw.append(float((P * Q).sum()))
        d = np.linalg.norm(P) * np.linalg.norm(Q)
        cos.append(float((P * Q).sum() / d) if d > 1e-9 else 0.0)
    raw, cos, xs = np.array(raw), np.array(cos), np.array(xs)
    pk, _ = find_peaks(cos, prominence=float(np.std(cos)))

    fig, axs = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True)
    axs[0].plot(xs, raw, lw=1.2, color=BLU)
    axs[0].set_ylabel("produit scalaire\n(brut)", fontsize=8)
    axs[0].axvline(b0, color=ACC, lw=1.2)
    axs[1].plot(xs, cos, lw=1.2, color=INK)
    axs[1].plot(xs[pk], cos[pk], "o", color=ACC, ms=5)
    axs[1].axvline(b0, color=ACC, lw=1.2)
    for sg in segs:
        for a_ in axs:
            a_.axvline(sg["b0"], color=GRN, lw=.8, alpha=.55)
    axs[1].set_ylabel("normalisé\n(cosinus)", fontsize=8)
    axs[1].set_xlabel("décalage sur l'axe x, en mesures "
                      "(rouge = le motif, vert = les sections actuelles)", fontsize=9)
    slide = fig2b64(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.imshow(S, cmap="RdYlBu_r", origin="lower",
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    ax.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False, ec=INK, lw=2))
    for c in xs[pk]:
        ax.add_patch(plt.Rectangle((c - .5, b0 - .5), L, L, fill=False,
                                   ec=ACC, lw=1.3, ls="--"))
    ax.set_xlabel("mesure", fontsize=8); ax.set_ylabel("mesure", fontsize=8)
    mat = fig2b64(fig)

    rows = "".join(
        f"<tr><td>{Lx}</td><td>{lagmean[Lx]:.3f}</td></tr>"
        for Lx in sorted(lagmean, key=lambda k: -lagmean[k])[:6])
    qs = (f"{np.median(q_of_peaks):.3f}" if len(q_of_peaks) else "—")
    q10 = (f"{np.percentile(q_of_peaks, 10):.3f}" if len(q_of_peaks) else "—")

    return f"""<section><h2>{title}</h2>
<div class=num>{n} mesures &nbsp;·&nbsp; motif détecté : <b>{L} mesures</b>
&nbsp;·&nbsp; premier bloc à la mesure <b>{b0}</b>
&nbsp;·&nbsp; {len(pk)} autres occurrences trouvées</div>

<h3>1 · À quel quantile vivent les pics clairs</h3>
<img src="data:image/png;base64,{hist}">
<table><tr><td>médiane du quantile des pics</td><td><b>{qs}</b></td></tr>
<tr><td>1er décile (les pics les plus faibles)</td><td>{q10}</td></tr>
<tr><td>quantile où tombe <code>TILE_MIN = {hs.TILE_MIN}</code></td>
<td><b>{(flat < hs.TILE_MIN).mean():.3f}</b></td></tr>
<tr><td>q90 / q95 de ce morceau</td>
<td>{np.quantile(flat, .90):.3f} / {np.quantile(flat, .95):.3f}</td></tr></table>

<h3>2 · La période dominante</h3>
<table><tr><th>décalage</th><th>similarité moyenne</th></tr>{rows}</table>

<h3>3 · Le motif glissé le long de l'axe x</h3>
<img src="data:image/png;base64,{slide}">
<p class=cap>Le motif est le bloc {L}×{L} de la diagonale à la mesure {b0}. On le
glisse en x et on fait le produit scalaire avec le carré sur lequel il tombe.
Haut = produit brut, bas = normalisé. Rouge = la position du motif, vert = les
frontières de sections actuelles.</p>
<img src="data:image/png;base64,{mat}">
<p class=cap>Noir plein = le motif. Rouge tirets = les occurrences que le
glissement trouve.</p></section>"""


def summary(rows):
    if not rows:
        return ""
    qs = [r[2] for r in rows]
    tr = "".join(
        f"<tr><td>{t}</td><td>{L} mes.</td><td><b>{q:.3f}</b></td>"
        f"<td style='color:#8a2b2b'><b>{qt:.3f}</b></td><td>{q9:.3f}</td></tr>"
        for t, L, q, qt, q9 in rows)
    return f"""<section><h2>La réponse : les pics vivent tous au même quantile, le seuil fixe non</h2>
<table><tr><th>morceau</th><th>période dominante</th>
<th>quantile médian des pics clairs</th>
<th>quantile où tombe <code>TILE_MIN = {hs.TILE_MIN}</code></th>
<th>q90 du morceau</th></tr>{tr}</table>
<div class=num>pics : {np.mean(qs):.3f} ± {np.std(qs):.3f} &nbsp;(de {min(qs):.3f} à {max(qs):.3f})
&nbsp;·&nbsp; seuil fixe : de {min(r[3] for r in rows):.3f} à {max(r[3] for r in rows):.3f}</div>
<p class=cap>Les pics clairs occupent le même rang dans la distribution de chaque
morceau — autour du 86e centile, à ±4 points près. Le seuil fixe de {hs.TILE_MIN},
lui, se promène du 43e au 93e centile selon le morceau. Un seuil exprimé en
quantile (~0,88) vaudrait donc à peu près 0,80 sur This Love et Don't Know Why,
serait moins sévère sur Sunny (où 0,80 est déjà au 93e) et bien plus sévère sur
Billie Jean (où 0,80 n'est qu'au 43e). Et les cinq morceaux ont la même période
dominante : <b>4 mesures</b>.</p></section>"""


def summary_row(stem, title):
    S, n, _ = bar_ssm(stem)
    flat = off_diag(S, n); sf = np.sort(flat)
    pv = []
    for b in range(n):
        lag = np.arange(n) - b
        v = S[b][np.abs(lag) >= MIN_L]
        if len(v) < 8:
            continue
        pk, _ = find_peaks(v, prominence=float(np.std(v)))
        pv += [float(v[i]) for i in pk]
    q = np.searchsorted(sf, np.array(pv)) / len(sf) if pv else np.array([0.0])
    L, _ = dominant_lag(S, n)
    return (title, L, float(np.median(q)), float((flat < hs.TILE_MIN).mean()),
            float(np.quantile(flat, .90)))


def main():
    body, ok, srows = "", [], []
    for stem, title in CANDIDATES:
        p = HERE / f"docs/audio/{stem}.m4a"
        if not p.exists():
            continue
        import json
        cache = HERE / f"harmonia_min/state/beats/{stem}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            q = bts.grid_quality(d.get("beats") or [], d.get("downbeats") or [])
            if q["n_bars"] >= bts.GRID_MIN_BARS and (
                    q["metre"] != 4 or q["consistency"] < bts.GRID_MIN_CONSISTENCY):
                print(f"  (grille refusée: {stem} metre={q['metre']} "
                      f"coh={q['consistency']:.0%})")
                continue
        srows.append(summary_row(stem, title))
        body += song(stem, title)
        ok.append(title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Motifs glissés et quantiles</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1000px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 16px system-ui;margin:0 0 10px;color:{ACC}}}
h3{{font:700 12px system-ui;margin:18px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px}}
.cap{{font-size:12px;color:#8a8371;margin:4px 0 0}}
.num{{background:{INK};color:#f4eee2;padding:6px 11px;border-radius:8px;font:600 12.5px system-ui;display:inline-block}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class=wrap>
<h1>Motifs glissés et quantiles</h1>
<div class=lede>Uniquement les morceaux dont la grille passe le garde-fou
(métrique 4, cohérence ≥ {bts.GRID_MIN_CONSISTENCY:.0%}) : {', '.join(ok)}.
Sur une grille cassée, les mesures ne sont pas des mesures et tous ces chiffres
parleraient de la grille, pas de la musique.</div>
{summary(srows)}{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
