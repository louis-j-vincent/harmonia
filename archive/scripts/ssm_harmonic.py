"""Harmonic similarity substrates — matrices only, no numbers.

Louis, 2026-08-05, correcting me twice:

  « Le produit scalaire que j'avais défini pour les accords, c'est une SIMILARITÉ
    HARMONIQUE — si♭ est plus proche de sol mineur que de la♭. »
  « Tu me montres juste les matrices, tu ne me fais pas de score. Les scores que
    tu montres ça ne sert à rien. Et je ne sais même pas ce que c'est la
    salience. »

Both fair. My previous chord substrate used musx's 73 chord CLASSES as
coordinates, so every chord was orthogonal to every other — B♭ was exactly as far
from Gm as from A♭. That is the opposite of what he asked for. The fix is to
represent each chord by its CHORD TONES: B♭ = {B♭,D,F} and Gm = {G,B♭,D} share
two notes, A♭ = {A♭,C,E♭} shares none.

So every substrate here lives in the 12 pitch classes, where the dot product IS
harmonic overlap. And the page prints no metric of any kind.

    python scripts/ssm_harmonic.py   ->  /reports/ssm_harmonic.html
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

OUT = HERE / "harmonia_min/state/reports/ssm_harmonic.html"
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why"),
         ("bobby_hebb_sunny_official_audio", "Sunny")]
DT = 23.22e-3

# musx triad plane: col 0 = N; col i>=1 is root (i-1)%12, type (i-1)//12+1 in
# {maj, min, sus4, sus2, dim, aug}. Their chord tones, as semitone offsets:
TONES = {1: (0, 4, 7), 2: (0, 3, 7), 3: (0, 5, 7),
         4: (0, 2, 7), 5: (0, 3, 6), 6: (0, 4, 8)}


def chord_tone_matrix(n_cols: int) -> np.ndarray:
    """(n_cols, 12) — each chord class as its pitch-class indicator."""
    M = np.zeros((n_cols, 12))
    for i in range(1, n_cols):
        root, typ = (i - 1) % 12, (i - 1) // 12 + 1
        for s in TONES.get(typ, (0, 4, 7)):
            M[i, (root + s) % 12] = 1.0
    return M


def capture(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None, **_kw):
        out = real(grid, arr, times, bars, **_kw)
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


def topk_rows(X, k):
    Y = X.copy()
    for i in range(len(Y)):
        Y[i, np.argsort(Y[i])[:-k]] = 0
    return Y


def substrates(stem):
    cap = capture(stem)
    grid = cap["grid"]
    n = len(grid) - 1
    arr = np.asarray(cap["arr"])
    F = hs.halfbar_features(grid, arr, cap["times"])
    perbar = lambda X: np.array([X[2 * b:2 * b + 2].mean(0) for b in range(n)])

    # chroma per bar, treble and bass halves kept separately
    Cb = perbar(F[:, :12])          # bass 12
    Ct = perbar(F[:, 12:])          # treble 12
    Call = Cb + Ct                  # the 12 pitch classes, both registers

    # chord posteriors per bar, projected onto their CHORD TONES
    triad = mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0]
    P = []
    for b in range(n):
        a, z = int(grid[b] / DT), max(int(grid[b] / DT) + 1, int(grid[b + 1] / DT))
        seg = triad[a:min(z, len(triad))]
        P.append(seg.mean(0) if len(seg) else np.zeros(triad.shape[1]))
    P = np.array(P)
    M = chord_tone_matrix(P.shape[1])

    out = []
    add = lambda nm, note, V: out.append((nm, note, unit(V) @ unit(V).T))

    add("CE QUI TOURNE — chroma 24-d",
        "les deux registres séparés, pas de notion d'accord", perbar(F))
    add("notes : chroma 12 classes",
        "basse et aigu repliés sur les 12 notes", Call)
    add("notes : TOP-3 des notes",
        "les 3 pitch classes les plus fortes de la mesure", topk_rows(Call, 3))
    add("notes : TOP-3 binaire",
        "les mêmes 3 notes, toutes à 1 — recouvrement pur", (topk_rows(Call, 3) > 0).astype(float))
    add("ACCORDS → NOTES, distribution complète",
        "chaque accord remplacé par ses notes, pondéré par sa proba", P @ M)
    add("ACCORDS → NOTES, TON IDÉE : top-3 pondérés",
        "les 3 accords les plus probables, chacun par ses notes", topk_rows(P, 3) @ M)
    add("ACCORDS → NOTES, top-1",
        "l'accord le plus probable seul, par ses notes", topk_rows(P, 1) @ M)
    return out, n


def song_html(stem, title):
    subs, n = substrates(stem)
    k = len(subs)

    def fig(stretch):
        f, axs = plt.subplots(1, k, figsize=(4.2 * k, 4.6))
        for a_, (nm, _t, S) in zip(axs, subs):
            lo, hi = ((np.percentile(S, 5), np.percentile(S, 99)) if stretch
                      else (0.0, 1.0))
            a_.imshow(S, cmap="RdYlBu_r", origin="lower", vmin=lo,
                      vmax=max(hi, lo + 1e-6))
            a_.set_title(nm, fontsize=8.5, loc="left",
                         color="#8a2b2b" if "TON IDÉE" in nm else "#1c1c1c")
            a_.set_xticks([]); a_.set_yticks([])
        return fig2b64(f)

    cells = "".join(f"<div class=cell><b>{nm}</b><br><span class=note>{t}</span></div>"
                    for nm, t, _ in subs)
    return f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<h3>Échelle commune 0–1</h3><img src="data:image/png;base64,{fig(False)}">
<h3>Échelle étirée par matrice</h3><img src="data:image/png;base64,{fig(True)}">
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
<title>Similarité harmonique — les matrices</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:#1c1c1c}}
.wrap{{max-width:1500px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 12px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
h3{{font:700 11px system-ui;margin:16px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px}}
.cells{{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}}
.cell{{flex:1;min-width:150px;font-size:11.5px;background:#f7f3e9;border-radius:8px;padding:7px 9px}}
.note{{color:#8a8371}}
</style></head><body><div class=wrap>
<h1>Similarité harmonique — les matrices</h1>
<div class=lede>Tout vit désormais dans les <b>12 classes de hauteur</b>, donc le
produit scalaire <b>est</b> le recouvrement harmonique : si♭ majeur {{si♭ ré fa}}
et sol mineur {{sol si♭ ré}} partagent deux notes, la♭ majeur {{la♭ do mi♭}} n'en
partage aucune. Ma version précédente utilisait les 73 <i>classes d'accord</i>
comme coordonnées, où tous les accords sont orthogonaux entre eux — si♭ y était
aussi loin de sol mineur que de la♭. C'était l'inverse de ce qui était demandé.
<b>Aucun chiffre sur cette page.</b></div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
