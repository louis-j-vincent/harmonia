"""Each SSM row as a 1-D signal, rolled so lag 0 is the bar itself.

Louis, 2026-08-05: « Ce seuil est un problème. Pour chaque barre, extrais la
ligne de la matrice SSM, affiche-la comme un signal 1D, plot les 10 premiers
pour chaque chanson, et fais rouler chaque ligne pour que le temps 0 corresponde
au temps de la barre. »

The point of the view: `tiling_runs` asks one question per bar — "is
cos(b, b±P) >= TILE_MIN for P in 2,4,8?" — which is a fixed horizontal line
drawn across these signals. Plotting the signals shows what that line actually
cuts, and whether the peaks it is meant to catch stand out at all.

The x axis is the true lag (j - b), NOT `np.roll`: a circular roll would wrap
the start of the song onto the end and invent similarity that isn't there. Each
row therefore spans -b .. n-1-b, all aligned on 0.

    python scripts/ssm_rows_plot.py   ->  /reports/ssm_rows.html
"""
from __future__ import annotations

import base64
import copy
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
from harmonia_min import sections as hs             # noqa: E402
from harmonia_min.folding import _bar_vecs          # noqa: E402

OUT = HERE / "harmonia_min/state/reports/ssm_rows.html"
N_ROWS = 10
INK, ACC, GRN, BLU = "#1c1c1c", "#8a2b2b", "#1f8a5b", "#2a6fb0"

SONGS = [
    ("maroon_5_this_love", "This Love"),
    ("norah_jones_don_t_know_why", "Don't Know Why"),
    ("bobby_hebb_sunny_official_audio", "Sunny"),
    ("michael_jackson_billie_jean_official_video", "Billie Jean"),
    ("ray_charles_georgia_on_my_mind_official_video", "Georgia On My Mind"),
]


def fig2b64(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def bar_ssm(stem):
    """Bar-level SSM, exactly the matrix `tiling_runs` reads."""
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None, **_kw):
        out = real(grid, arr, times, bars, **_kw)
        cap.update(grid=grid, arr=arr, times=times, segs=copy.deepcopy(out))
        return out

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x",
                    file_key="x", audio_url="")
    finally:
        hs.detect_sections = real
    F = hs.halfbar_features(cap["grid"], np.asarray(cap["arr"]), cap["times"])
    n = len(cap["grid"]) - 1
    Vb = _bar_vecs(F, n)
    return Vb @ Vb.T, n, cap["segs"]


def song_html(stem, title):
    S, n, segs = bar_ssm(stem)
    rows = min(N_ROWS, n)

    # ── (a) the 10 rows overlaid on one lag axis ────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 3.4))
    acc = {}
    for b in range(rows):
        lag = np.arange(n) - b
        ax.plot(lag, S[b], lw=.9, alpha=.55, color=BLU)
        for L, v in zip(lag, S[b]):
            acc.setdefault(int(L), []).append(float(v))
    ks = sorted(acc)
    ax.plot(ks, [np.mean(acc[k]) for k in ks], lw=2.2, color=INK,
            label=f"moyenne des {rows} lignes")
    ax.axhline(hs.TILE_MIN, color=ACC, ls="--", lw=1.4,
               label=f"TILE_MIN = {hs.TILE_MIN}")
    for P in (2, 4, 8):
        ax.axvline(P, color=GRN, lw=.9, alpha=.65)
        ax.axvline(-P, color=GRN, lw=.9, alpha=.65)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlabel("décalage en mesures (0 = la mesure elle-même)", fontsize=9)
    ax.set_ylabel("similarité", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    over = fig2b64(fig)

    # ── (b) the same 10, one per panel ──────────────────────────────────────
    fig, axs = plt.subplots(rows, 1, figsize=(11, 1.05 * rows), sharex=False)
    for b in range(rows):
        a = axs[b] if rows > 1 else axs
        lag = np.arange(n) - b
        a.plot(lag, S[b], lw=1.0, color=INK)
        a.fill_between(lag, hs.TILE_MIN, S[b], where=S[b] >= hs.TILE_MIN,
                       color=GRN, alpha=.35, interpolate=True)
        a.axhline(hs.TILE_MIN, color=ACC, ls="--", lw=1)
        for P in (2, 4, 8):
            a.axvline(P, color=GRN, lw=.8, alpha=.6)
            a.axvline(-P, color=GRN, lw=.8, alpha=.6)
        a.axvline(0, color=INK, lw=1)
        a.set_ylim(0, 1.02)
        a.set_yticks([hs.TILE_MIN])
        a.set_yticklabels([str(hs.TILE_MIN)], fontsize=6)
        a.set_ylabel(f"m.{b}", fontsize=7, rotation=0, ha="right", va="center")
        if b < rows - 1:
            a.set_xticklabels([])
    (axs[-1] if rows > 1 else axs).set_xlabel(
        "décalage en mesures (0 = la mesure elle-même)", fontsize=9)
    grid = fig2b64(fig)

    # ── what the fixed threshold does on these rows ─────────────────────────
    lines = []
    for b in range(rows):
        hits = [P for P in (2, 4, 8)
                if (b + P < n and S[b, b + P] >= hs.TILE_MIN)
                or (b - P >= 0 and S[b, b - P] >= hs.TILE_MIN)]
        off = np.array([S[b, j] for j in range(n) if abs(j - b) >= 2])
        best = float(off.max()) if len(off) else 0.0
        arg = int(np.argmax(off)) if len(off) else 0
        js = [j for j in range(n) if abs(j - b) >= 2]
        lines.append(
            f"<tr><td>{b}</td><td>{'P' + ', P'.join(map(str, hits)) if hits else '<i>aucune</i>'}</td>"
            f"<td>{best:.3f}</td><td>{(js[arg] - b) if js else 0:+d}</td>"
            f"<td>{float(np.median(off)) if len(off) else 0:.3f}</td></tr>")

    return f"""<section><h2>{title}</h2>
<div class=num>{n} mesures &nbsp;·&nbsp; {len(segs)} sections détectées</div>
<h3>Les {rows} premières lignes, superposées</h3>
<img src="data:image/png;base64,{over}">
<p class=cap>Chaque courbe bleue est une ligne de la SSM, décalée pour que 0 soit
sa propre mesure. Noir = la moyenne. Rouge tirets = le seuil
<code>TILE_MIN = {hs.TILE_MIN}</code>. Verts = les décalages 2, 4 et 8, les seuls
que <code>tiling_runs</code> interroge.</p>
<h3>Les mêmes, une par ligne</h3>
<img src="data:image/png;base64,{grid}">
<p class=cap>Le vert plein est ce qui dépasse le seuil.</p>
<table><tr><th>mesure</th><th>périodes retenues</th><th>meilleure similarité
(hors ±1)</th><th>à quel décalage</th><th>médiane de la ligne</th></tr>
{''.join(lines)}</table></section>"""


def main():
    body = ""
    stats = []
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            print(f"  (pas d'audio: {stem})")
            continue
        S, n, _ = bar_ssm(stem)
        v = np.array([S[b, j] for b in range(n) for j in range(n)
                      if abs(j - b) >= 2])
        stats.append((title, float(np.median(v)), float(np.percentile(v, 90)),
                      float((v >= hs.TILE_MIN).mean() * 100)))
        body += song_html(stem, title)
        print(f"  ok {title}")

    srows = "".join(
        f"<tr><td>{t}</td><td>{m:.3f}</td><td>{p9:.3f}</td>"
        f"<td><b>{pc:.1f}%</b></td></tr>" for t, m, p9, pc in stats)
    body = f"""<section><h2>Le seuil fixe ne veut pas dire la même chose d'un morceau à l'autre</h2>
<p>Pour chaque morceau, la distribution des similarités hors diagonale (|décalage| ≥ 2) :</p>
<table><tr><th>morceau</th><th>médiane</th><th>9e décile</th>
<th>part de la matrice ≥ {hs.TILE_MIN}</th></tr>{srows}</table>
<p class=cap>Un seuil fixe à {hs.TILE_MIN} tombe donc au ~45e centile sur Billie Jean
et au-delà du 90e sur Georgia. Sur Billie Jean la <b>médiane</b> dépasse le seuil :
plus d'une paire de mesures sur deux « tuile », d'où ses 12 runs. Sur Georgia et
Sunny, 5 à 7 % de la matrice le franchit et les runs ne se forment presque pas.
Le seuil ne mesure pas « est-ce que ça se répète », il mesure à quel point le
morceau est harmoniquement homogène.</p></section>""" + body
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les lignes de la SSM, mesure par mesure</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1000px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 16px system-ui;margin:0 0 10px;color:{ACC}}}
h3{{font:700 12px system-ui;margin:16px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px}}
.cap{{font-size:12px;color:#8a8371;margin:4px 0 0}}
.num{{background:{INK};color:#f4eee2;padding:6px 11px;border-radius:8px;font:600 12.5px system-ui;display:inline-block}}
table{{border-collapse:collapse;font-size:12px;margin-top:12px}}
th,td{{border:1px solid #e5dcc6;padding:3px 9px;text-align:left}}
th{{background:#f7f3e9}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class=wrap>
<h1>Les lignes de la SSM, mesure par mesure</h1>
<div class=lede>Chaque ligne de la matrice de similarité au niveau MESURE, tracée
comme un signal 1-D et décalée pour que 0 soit sa propre mesure. C'est exactement
ce que <code>tiling_runs</code> lit — sauf qu'il n'en regarde que trois points
(décalages 2, 4, 8) et les compare à une barre horizontale fixe.
Axe = vrai décalage, pas <code>np.roll</code> : un roulement circulaire
ramènerait le début du morceau sur sa fin et inventerait de la similarité.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
