"""Every candidate substrate, as a PICTURE. No metric leads this page.

Louis, 2026-08-05: « avant de mesurer quoi que ce soit, montre-moi les matrices
SSM qui correspondent aux idées que je te propose à chaque fois. Je veux voir
visuellement s'il y a une matrice qui vraiment me départage. Si ça départage
visuellement, tout le reste c'est donné. »

So: the same three songs, every substrate, side by side, big. Two rows per song —
the first on a COMMON absolute scale 0..1 so the background levels are comparable
between substrates, the second stretched per-matrix (5th–99th percentile) so the
structure inside each is readable even when its background is near zero. A number
under each says what its actual range is, so the stretch cannot mislead.

    python scripts/ssm_substrates.py   ->  /reports/ssm_substrates.html
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
from harmonia_min import sections as hs, musx as mx      # noqa: E402
from ssm_rows_plot import fig2b64                        # noqa: E402

OUT = HERE / "harmonia_min/state/reports/ssm_substrates.html"
SONGS = [
    ("maroon_5_this_love", "This Love"),
    ("norah_jones_don_t_know_why", "Don't Know Why"),
    ("bobby_hebb_sunny_official_audio", "Sunny"),
]
DT = 23.22e-3


def capture(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None):
        out = real(grid, arr, times, bars)
        cap.update(grid=grid, arr=arr, times=times)
        return out

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x", file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    return cap


def unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def substrates(stem):
    """[(name, note, bar-level SSM)] — every idea on the table, same grid."""
    cap = capture(stem)
    grid = cap["grid"]
    n = len(grid) - 1
    F = hs.halfbar_features(grid, np.asarray(cap["arr"]), cap["times"])
    bar = lambda X: unit(np.array([X[2 * b:2 * b + 2].mean(0) for b in range(n)]))

    out = []
    Vc = bar(F)
    out.append(("chroma 24-d — CE QUI TOURNE",
                "moyenne des cosinus basse et aigu, à poids égal", Vc @ Vc.T))

    # bass half / treble half on their own — the shipped matrix is their mean
    Vb = bar(np.concatenate([F[:, :12], np.zeros_like(F[:, 12:])], 1))
    Vt = bar(np.concatenate([np.zeros_like(F[:, :12]), F[:, 12:]], 1))
    out.append(("chroma — BASSE seule", "la moitié grave du vecteur", Vb @ Vb.T))
    out.append(("chroma — AIGU seul", "la moitié aiguë", Vt @ Vt.T))

    # top-3 chroma bins per half
    Ft = F.copy()
    for h in (slice(0, 12), slice(12, 24)):
        X = Ft[:, h].copy()
        for i in range(len(X)):
            X[i, np.argsort(X[i])[:-3]] = 0
        Ft[:, h] = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9) / np.sqrt(2)
    V3 = bar(Ft)
    out.append(("chroma — TOP-3 cases par moitié",
                "on ne garde que les 3 plus fortes de chaque moitié", V3 @ V3.T))

    # chord-posterior substrates
    triad = mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0]
    P = []
    for b in range(n):
        a, z = int(grid[b] / DT), max(int(grid[b] / DT) + 1, int(grid[b + 1] / DT))
        seg = triad[a:min(z, len(triad))]
        P.append(seg.mean(0) if len(seg) else np.zeros(triad.shape[1]))
    P = np.array(P)
    for k, nm, note in [
            (None, "ACCORDS — distribution complète",
             "le postérieur musx moyenné sur la mesure, 73 classes"),
            (3, "ACCORDS — TON IDÉE : top-3 pondérés",
             "les 3 accords les plus probables, pondérés par leurs probas"),
            (1, "ACCORDS — top-1 (étiquette dure)",
             "l'argmax seul — ce que la v1 faisait, et qui l'avait tuée")]:
        X = P.copy()
        if k:
            for i in range(len(X)):
                X[i, np.argsort(X[i])[:-k]] = 0
        X = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
        Vk = unit(X)
        out.append((nm, note, Vk @ Vk.T))
    return out, n


def song_html(stem, title):
    subs, n = substrates(stem)
    k = len(subs)

    def grid_fig(stretch):
        fig, axs = plt.subplots(1, k, figsize=(4.1 * k, 4.5))
        for a_, (nm, _note, S) in zip(axs, subs):
            if stretch:
                lo, hi = np.percentile(S, 5), np.percentile(S, 99)
            else:
                lo, hi = 0.0, 1.0
            a_.imshow(S, cmap="RdYlBu_r", origin="lower", vmin=lo,
                      vmax=max(hi, lo + 1e-6))
            a_.set_title(nm, fontsize=8.5, loc="left",
                         color="#8a2b2b" if "TON IDÉE" in nm else "#1c1c1c")
            a_.set_xticks([]); a_.set_yticks([])
        return fig2b64(fig)

    common, stretched = grid_fig(False), grid_fig(True)
    i = np.arange(n)
    mask = np.abs(i[:, None] - i[None, :]) >= 2
    cells = "".join(
        f"<div class=cell><b>{nm}</b><br><span class=note>{note}</span>"
        f"<br><span class=rng>hors diagonale : médiane "
        f"{np.median(S[mask]):.3f} · q90 {np.quantile(S[mask], .9):.3f}</span></div>"
        for nm, note, S in subs)

    return f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<h3>Échelle commune 0–1 — les niveaux de fond sont comparables entre matrices</h3>
<img src="data:image/png;base64,{common}">
<h3>Échelle étirée par matrice — la structure interne de chacune</h3>
<img src="data:image/png;base64,{stretched}">
<div class=cells>{cells}</div></section>"""


def main():
    body = ""
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            continue
        body += song_html(stem, title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les substrats, en images</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:#1c1c1c}}
.wrap{{max-width:1500px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:880px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 12px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
h3{{font:700 11px system-ui;margin:16px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px}}
.cells{{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}}
.cell{{flex:1;min-width:150px;font-size:11.5px;background:#f7f3e9;border-radius:8px;padding:7px 9px}}
.note{{color:#8a8371}}
.rng{{color:#2a6fb0;font-weight:600}}
</style></head><body><div class=wrap>
<h1>Les substrats de la SSM, en images</h1>
<div class=lede>Sept façons de construire la matrice, les mêmes trois morceaux,
la même grille de mesures. <b>Aucun chiffre ne commande cette page</b> — la
première rangée est à échelle commune 0–1 pour que les niveaux de fond soient
comparables entre substrats, la seconde est étirée par matrice pour que la
structure interne de chacune reste lisible même quand son fond est à zéro. Les
médianes sous chaque colonne sont là pour que l'étirement ne puisse pas
tromper.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
