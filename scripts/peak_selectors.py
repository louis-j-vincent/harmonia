"""Peak selectors on the sliding-block curve, side by side — Louis decides.

« En prio je veux l'analyse de la détection de pics pour le block glissé — je
veux des plots + la matrice SSM et je déduirai moi ce qui est le mieux. »

So this page ARGUES NOTHING. Four selectors are run on the same curve, drawn on
the same axes with different markers, and every occurrence each one finds is
boxed on the SSM in that selector's colour. What each costs is tabulated. The
reading is his.

The curve is the RAW sliding dot product — his preference, and his reasoning is
sound: sliding along x with the rows held fixed, the block's overall level IS
the harmonic correlation with the base square, so normalising divides out the
signal. The normalised and mean-centred curves are drawn faintly underneath for
comparison, not used for the selection.

Only songs whose bar grid passes `beats.check_grid` — on a broken grid the bars
are not bars and every peak here would be about the grid.

    python scripts/peak_selectors.py   ->  /reports/peak_selectors.html
"""
from __future__ import annotations

import json
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
from harmonia_min import beats as bts                       # noqa: E402
from ssm_rows_plot import bar_ssm, fig2b64                  # noqa: E402
from pattern_slide import off_diag, dominant_lag, MIN_L     # noqa: E402

OUT = HERE / "harmonia_min/state/reports/peak_selectors.html"
INK = "#1c1c1c"

SONGS = [
    ("maroon_5_this_love", "This Love"),
    ("norah_jones_don_t_know_why", "Don't Know Why"),
    ("bobby_hebb_sunny_official_audio", "Sunny"),
    ("michael_jackson_billie_jean_official_video", "Billie Jean"),
    ("the_police_every_breath_you_take_official_music_video", "Every Breath You Take"),
]

# name, colour, marker, function(curve) -> indices
SELECTORS = [
    ("proéminence (1σ)", "#8a2b2b", "o"),
    ("quantile q90 de la courbe", "#1f8a5b", "s"),
    ("marge sur le fond local", "#2a6fb0", "^"),
    ("top-N espacés de L", "#c58a2e", "D"),
]


def sel_prominence(y, L):
    pk, _ = find_peaks(y, prominence=float(np.std(y)))
    return pk


def sel_quantile(y, L):
    pk, _ = find_peaks(y, height=float(np.quantile(y, 0.90)))
    return pk


def sel_margin(y, L, w=None, margin=0.5):
    """A peak must stand `margin`·σ above the MEDIAN of its own neighbourhood —
    local, so a slow drift in the curve cannot create or hide peaks."""
    w = w or max(4, 3 * L)
    pk, _ = find_peaks(y)
    keep = []
    for i in pk:
        a, b = max(0, i - w), min(len(y), i + w + 1)
        base = float(np.median(y[a:b]))
        if y[i] - base >= margin * float(np.std(y)):
            keep.append(i)
    return np.array(keep, int)


def sel_topn(y, L, n=None):
    """Greedy: take the highest, forbid anything within L bars, repeat."""
    n = n or max(2, len(y) // max(1, 2 * L))
    y2, out = y.copy(), []
    for _ in range(n):
        i = int(np.argmax(y2))
        if not np.isfinite(y2[i]):
            break
        out.append(i)
        y2[max(0, i - L):i + L + 1] = -np.inf
    return np.array(sorted(out), int)


FUNCS = [sel_prominence, sel_quantile, sel_margin, sel_topn]


def curves(S, n):
    """raw / cosine / centred sliding dot product of the first repeated block."""
    flat = off_diag(S, n)
    L, _ = dominant_lag(S, n)
    thr = float(np.quantile(flat, 0.90))
    b0 = next((b for b in range(n - L) if S[b, b + L] >= thr), 0)
    P = S[b0:b0 + L, b0:b0 + L]
    Pc = P - P.mean()
    raw, cos, cen = [], [], []
    for c in range(0, n - L + 1):
        Q = S[b0:b0 + L, c:c + L]
        raw.append(float((P * Q).sum()) / (L * L))
        d = np.linalg.norm(P) * np.linalg.norm(Q)
        cos.append(float((P * Q).sum() / d) if d > 1e-9 else 0.0)
        Qc = Q - Q.mean()
        dd = np.linalg.norm(Pc) * np.linalg.norm(Qc)
        cen.append(float((Pc * Qc).sum() / dd) if dd > 1e-9 else 0.0)
    return (np.array(raw), np.array(cos), np.array(cen), L, b0)


def song(stem, title):
    S, n, segs = bar_ssm(stem)
    raw, cos, cen, L, b0 = curves(S, n)
    x = np.arange(len(raw))
    picks = [f(raw, L) for f in FUNCS]

    # ── the four selectors on one curve ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(x, raw, lw=1.4, color=INK, label="produit scalaire BRUT (la courbe jugée)", zorder=3)
    for i, ((nm, col, mk), pk) in enumerate(zip(SELECTORS, picks)):
        off = 1 + 0.045 * (i + 1)
        ax.plot(x[pk], raw[pk] * off, mk, color=col, ms=7, mew=1.4,
                mfc="none", label=f"{nm} — {len(pk)} pics", zorder=4)
    for sg in segs:
        ax.axvline(sg["b0"], color="#1f8a5b", lw=.8, alpha=.35, zorder=1)
    ax.axvline(b0, color="#8a2b2b", lw=1.6, zorder=2)
    ax.set_xlabel("décalage sur l'axe x, en mesures  "
                  "(trait rouge = le motif de référence, traits verts pâles = sections actuelles)",
                  fontsize=9)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    main = fig2b64(fig)

    # ── the other two statistics, for reference only ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 2.2))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    ax.plot(x, z(raw), lw=1.3, color=INK, label="brut")
    ax.plot(x, z(cos), lw=1.0, color="#8a8371", alpha=.9, label="cosinus normalisé")
    ax.plot(x, z(cen), lw=1.0, color="#2a6fb0", alpha=.9, label="centré par bloc")
    ax.set_ylabel("z-score", fontsize=8)
    ax.set_xlabel("décalage (mesures)", fontsize=9)
    ax.legend(fontsize=8, ncol=3)
    alts = fig2b64(fig)

    # ── one SSM per selector, its finds boxed ───────────────────────────────
    fig, axs = plt.subplots(1, len(SELECTORS), figsize=(4.0 * len(SELECTORS), 4.2))
    for a_, (nm, col, _mk), pk in zip(axs, SELECTORS, picks):
        a_.imshow(S, cmap="RdYlBu_r", origin="lower",
                  vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
        a_.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False,
                                   ec=INK, lw=2.2))
        for c in pk:
            a_.add_patch(plt.Rectangle((c - .5, b0 - .5), L, L, fill=False,
                                       ec=col, lw=1.6, ls="--"))
        a_.set_title(f"{nm}\n{len(pk)} pics", fontsize=8, color=col, loc="left")
        a_.set_xticks([]); a_.set_yticks([])
    mats = fig2b64(fig)

    starts = {s["b0"] for s in segs}
    rows = ""
    for (nm, col, _), pk in zip(SELECTORS, picks):
        hits = sum(1 for c in pk if any(abs(c - s) <= 1 for s in starts))
        cov = sum(1 for s in starts if any(abs(c - s) <= 1 for c in pk))
        gaps = np.diff(sorted(pk)) if len(pk) > 1 else np.array([])
        rows += (f"<tr><td style='color:{col}'><b>{nm}</b></td><td>{len(pk)}</td>"
                 f"<td>{hits}/{len(pk) if len(pk) else 1}</td>"
                 f"<td>{cov}/{len(starts)}</td>"
                 f"<td>{(str(sorted(set(int(g) for g in gaps)))[:34] if len(gaps) else '—')}</td></tr>")

    return f"""<section><h2>{title}</h2>
<div class=num>{n} mesures &nbsp;·&nbsp; motif de <b>{L} mesures</b> pris à la
mesure <b>{b0}</b> &nbsp;·&nbsp; {len(segs)} sections aujourd'hui</div>
<img src="data:image/png;base64,{main}">
<h3>Les quatre sélecteurs sur la SSM</h3>
<img src="data:image/png;base64,{mats}">
<p class=cap>Noir plein = le motif de référence. Pointillés = ce que chaque
sélecteur trouve, dans sa couleur.</p>
<h3>Ce que chacun coûte</h3>
<table><tr><th>sélecteur</th><th>pics</th><th>dont sur un début de section</th>
<th>débuts couverts</th><th>écarts entre pics</th></tr>{rows}</table>
<h3>Les deux autres statistiques, pour référence</h3>
<img src="data:image/png;base64,{alts}">
<p class=cap>Normalisées en z-score pour être superposables. La sélection
ci-dessus est faite sur la courbe brute uniquement.</p>
</section>"""


def main():
    body, done = "", []
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            continue
        cache = HERE / f"harmonia_min/state/beats/{stem}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            q = bts.grid_quality(d.get("beats") or [], d.get("downbeats") or [])
            if q["n_bars"] >= bts.GRID_MIN_BARS and (
                    q["metre"] != 4 or q["consistency"] < bts.GRID_MIN_CONSISTENCY):
                print(f"  (grille refusée: {stem})")
                continue
        body += song(stem, title)
        done.append(title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Détection de pics sur le bloc glissé</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1200px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:820px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 16px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 12px system-ui;margin:18px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px}}
.cap{{font-size:12px;color:#8a8371;margin:4px 0 0}}
.num{{background:{INK};color:#f4eee2;padding:6px 11px;border-radius:8px;font:600 12.5px system-ui;display:inline-block;margin-bottom:10px}}
table{{border-collapse:collapse;font-size:12.5px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
</style></head><body><div class=wrap>
<h1>Détection de pics sur le bloc glissé</h1>
<div class=lede>Quatre sélecteurs, la même courbe, les mêmes axes — et pour
chacun, ce qu'il trouve encadré sur la SSM dans sa couleur. Aucune
recommandation : les quatre sont montrés à égalité, la lecture est la tienne.
La courbe jugée est le produit scalaire <b>brut</b>. Morceaux dont la grille
passe le garde-fou : {', '.join(done)}.</div>
{body}
<section><h3>Ce que fait chaque sélecteur</h3>
<ul>
<li><b>Proéminence (1σ)</b> — un maximum local dont la hauteur au-dessus de son
col dépasse un écart-type de la courbe. Purement topographique, aucun niveau
absolu.</li>
<li><b>Quantile q90</b> — un maximum local dont la valeur est dans les 10 %
supérieurs de la courbe. Global : sensible à un décalage lent de la courbe.</li>
<li><b>Marge sur le fond local</b> — un maximum local qui dépasse d'au moins
0,5 σ la <i>médiane de son voisinage</i> (±3 longueurs de motif). Local, donc
insensible à une dérive.</li>
<li><b>Top-N espacés</b> — on prend le plus haut, on interdit tout ce qui est à
moins d'une longueur de motif, on recommence. Garantit l'espacement, impose un
nombre.</li>
</ul></section>
</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
