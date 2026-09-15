"""La matrice rythmique à côté de la matrice harmonique — pour REGARDER.

    python scripts/rhythm_vs_harmony.py [<stem> ...]
      -> /reports/rhythm_vs_harmony.html

Louis, 2026-08-05 : « Je veux que tu me montres les matrices SSM rythme en plus
des matrices SSM qu'on a déjà, pour voir si je peux en faire quelque chose. »

Aucun chiffre n'est utilisé ici pour trancher quoi que ce soit. La page montre,
pour chaque morceau et sur le même axe de mesures :

  1. la batterie décomposée — 3 bandes (grosse caisse / caisse claire-médium /
     cymbales) × 16 doubles-croches par mesure, dessinée comme un rouleau ;
  2. la matrice HARMONIQUE, celle qu'on utilise déjà ;
  3. la matrice RYTHMIQUE au cosinus, mesure contre mesure ;
  4. la même en noyau RBF, dont la largeur vient de l'étalement des distances de
     la chanson elle-même ;
  5. les sections écrites aujourd'hui, en dessous, pour se repérer.

Ce qui change par rapport à `scratchpad/rhythm_ssm.py` (2026-07-30, resté sans
signal sur This Love) : la comparaison est par MESURE et non par demi-mesure, et
le noyau RBF est proposé à côté du cosinus. Ce sont les deux différences que la
littérature de segmentation pointe ; elles sont mises côte à côte pour être
jugées à l'œil, pas par un score.
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
sys.path.insert(0, str(HERE / "scratchpad"))
from pattern_lanes import load, fig2b64_fixed, PLOT_L, PLOT_R      # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402
from rhythm_ssm import separate_drums, _drum_onset_bands, _hpss_percussive  # noqa: E402

INK, ACC = "#1c1c1c", "#b3261e"
STEPS = 16          # doubles-croches par mesure
BANDS = ["grosse caisse", "claire / médium", "cymbales"]
DEFAULT = ["maroon_5_she_will_be_loved_official_music_video",
           "maroon_5_this_love",
           "the_police_every_breath_you_take_official_music_video",
           "mayer_hawthorne_the_walk",
           "bruno_mars_grenade_official_music_video"]


def bar_patches(stem, grid):
    """(n_mesures, 3, 16) — la batterie échantillonnée en doubles-croches."""
    import librosa
    src = separate_drums(HERE / f"docs/audio/{stem}.m4a")
    if src is not None:
        y, sr = librosa.load(str(src), sr=22050, mono=True)
        how = "stem de batterie (demucs htdemucs)"
    else:
        y, sr = _hpss_percussive(HERE / f"docs/audio/{stem}.m4a")
        how = "HPSS percussif — PAS un vrai stem de batterie, plus faible"
    env, times = _drum_onset_bands(y, sr)
    out = np.zeros((len(grid) - 1, env.shape[0], STEPS), np.float32)
    for b, (t0, t1) in enumerate(zip(grid[:-1], grid[1:])):
        ts = t0 + (np.arange(STEPS) + .5) / STEPS * (t1 - t0)
        for k in range(env.shape[0]):
            out[b, k] = np.interp(ts, times, env[k])
    return out, how


def matrices(P):
    """Cosinus et RBF sur les motifs de mesure aplatis."""
    V = P.reshape(len(P), -1).astype(np.float64)
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    C = np.clip(V @ V.T, 0, 1)
    D = np.sqrt(np.clip(2 - 2 * C, 0, None))          # distance euclidienne
    iu = np.triu_indices(len(V), 1)
    # la largeur du noyau vient du morceau : le premier quartile des distances,
    # donc « ce qui compte comme proche ICI ». L'écart-type, essayé d'abord,
    # donne une matrice uniformément bleue sur She Will Be Loved.
    scale = float(np.percentile(D[iu], 25)) or 1.0
    R = np.exp(-(D ** 2) / (2 * scale ** 2))
    return C, R


def figure(stem, S_harm, C, R, P, secs, n, how):
    H = 6.9
    fig = plt.figure(figsize=(12.6, H))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.85], hspace=.34, wspace=.15)
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - .30 / H, bottom=.50 / H)

    # le rouleau de batterie
    ax = fig.add_subplot(gs[0, :])
    roll = P.transpose(1, 2, 0).reshape(-1, len(P))   # (3*16, n_mesures)
    ax.imshow(roll ** .55, aspect="auto", origin="upper", cmap="magma",
              extent=(0, n, 3 * STEPS, 0), vmin=0,
              vmax=float(np.percentile(roll, 99) ** .55) or 1.0,
              interpolation="nearest")
    for k in (STEPS, 2 * STEPS):
        ax.axhline(k, color="#fff", lw=1.1)
    ax.set_yticks([STEPS * (i + .5) for i in range(3)])
    ax.set_yticklabels(BANDS, fontsize=6.6)
    ax.set_xticks(range(0, n + 1, 8))
    ax.tick_params(labelsize=6.6)
    ax.set_title(f"la batterie, décomposée — 3 bandes × 16 doubles-croches par mesure  ·  {how}",
                 fontsize=8.2, color="#6f6858", pad=4, loc="left")

    for j, (M, name) in enumerate(((S_harm, "HARMONIE (ce qu'on utilise)"),
                                   (C, "RYTHME — cosinus"),
                                   (R, "RYTHME — noyau RBF"))):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(M, origin="lower", extent=(0, n, 0, n), cmap="RdYlBu_r",
                  vmin=float(np.percentile(M, 5)), vmax=float(np.percentile(M, 99)),
                  aspect="auto", interpolation="nearest")
        for s in secs:
            ax.axvline(s["b0"], color="#111", lw=.5, alpha=.45)
            ax.axhline(s["b0"], color="#111", lw=.5, alpha=.45)
        ax.set_title(name, fontsize=8.2, color=ACC if j else "#2a6fb0", loc="left")
        ax.tick_params(labelsize=6.4)
        ax.set_xlabel("mesure", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("mesure", fontsize=7.5)
    return fig2b64_fixed(fig)


def song(stem):
    S, n, grid = load(stem)
    cells, _ = HS.build_cells(S, n)
    secs = HS.sections_from(S, n, cells)
    P, how = bar_patches(stem, grid)
    C, R = matrices(P)
    img = figure(stem, S, C, R, P, secs, n, how)
    wsec = " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in secs)
    return f"""<section><h2>{stem.replace('_',' ').title()}
<span class=sub>{n} mesures</span></h2>
<img src="data:image/png;base64,{img}">
<p class=sub>Les traits fins sur les matrices sont les frontières de sections
écrites aujourd'hui : {wsec}</p></section>"""


def main():
    stems = sys.argv[1:] or DEFAULT
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            body += song(st)
            print(f"  ok {st}")
        except Exception as exc:
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
    out = HERE / "harmonia_min/state/reports/rhythm_vs_harmony.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Rythme et harmonie, côte à côte</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:{ACC}}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
</style></head><body><div class=wrap>
<h1>Rythme et harmonie, côte à côte</h1>
<div class=lede>En haut, <b>la batterie décomposée</b> : trois bandes — grosse
caisse, caisse claire et médium, cymbales — échantillonnées en seize
doubles-croches par mesure. C'est la matière première ; chaque colonne est une
mesure de groove.<br><br>
En dessous, trois matrices sur le même axe de mesures : celle qu'on utilise
aujourd'hui (<b>harmonie</b>), puis la <b>rythmique au cosinus</b>, puis la
<b>rythmique en noyau RBF</b> — le noyau écarte les moyennes ressemblances et
garde les franches, sa largeur venant de l'étalement des distances du morceau
lui-même. Les traits fins sont les frontières de sections écrites aujourd'hui.
<br><br>
Rien n'est tranché ici. Deux différences avec la tentative du 30 juillet, qui
n'avait rien donné : on compare des <b>mesures entières</b> et non des
demi-mesures, et le noyau est proposé à côté du cosinus.</div>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
