"""Les trois profils de changement — basse+harmonie, voix, timbre — partagés.

Louis, 2026-08-12 : « je veux que tu m'affiches TOUJOURS les profils de pics
issus des matrices SSM pour timbre, voix, basse + harmonique. C'est d'eux dont on
va se servir pour décider de critères d'arrêt pour notre merging. »

Ce module ne fait que ça : charger les trois courbes et savoir les dessiner sur
un axe de mesures. Il existe pour que `three_profiles.py` (leur page dédiée) et
`bpe_lab.py` (l'agglomération, où les critères d'arrêt se décideront) partagent
exactement le même tracé — un import croisé entre les deux aurait été circulaire.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from peak_profile import fused_profile          # noqa: E402

# palette validée par dataviz/scripts/validate_palette.js : bande de clarté,
# plancher de chroma, séparation daltonienne et contraste — les cinq PASS.
COL = {"basse + harmonie": "#2f7dbd", "voix": "#c07a1e", "timbre": "#8155c6"}
MUET = "#f2e3c4"       # le fond doré : personne ne chante
TROIS = list(COL)


def load_curves(stem: str, n: int) -> dict:
    """Les trois courbes de changement + le couloir muet, prêtes à dessiner.

    Louis, 2026-08-12 : « je veux que tu m'affiches TOUJOURS les profils de pics
    issus des matrices SSM pour timbre, voix, basse + harmonique. C'est d'eux
    dont on va se servir pour décider de critères d'arrêt pour notre merging. »
    Elles sont donc en haut de toutes les pages de sections, pas seulement de la
    leur — et `pics_bar()` donne leurs pics en mesures pour qu'un critère d'arrêt
    puisse s'y référer.
    """
    P, _kept, _rest, lines, _n, extra = fused_profile(stem)
    m = len(P); hb = max(1, m // n)
    out = {"m": m, "hb": hb, "x": (np.arange(m) + 0.5) / hb, "lignes": {}}
    for nom in TROIS:
        d = next((z for z in lines if z["nom"] == nom), None)
        if d is None:
            continue
        out["lignes"][nom] = {"nov": np.nan_to_num(d["nov"]), "cuts": d["cuts"],
                              "poids": d.get("poids", 0.0)}
    mu = extra.get("mute")
    out["mute"] = (np.asarray(mu, bool)[:m] if mu is not None
                   else np.zeros(m, bool))
    return out


def pics_bar(C: dict, nom: str) -> list[int]:
    """Les pics d'une courbe, en mesures — ce sur quoi un critère d'arrêt mord."""
    hb = C["hb"]
    return sorted({int(round((c + 0.5) / hb)) for c in C["lignes"][nom]["cuts"]})


def draw_curves(ax, C, n, legend=True, triangles=True):
    """Les trois courbes superposées, fond doré compris. Axe des mesures."""
    m, hb, x = C["m"], C["hb"], C["x"]
    mute = C["mute"]
    i = 0
    while i < m:                                  # le fond : personne ne chante
        if mute[i]:
            j = i
            while j + 1 < m and mute[j + 1]:
                j += 1
            if (j - i + 1) / hb >= 1.5:
                ax.axvspan(i / hb, (j + 1) / hb, color=MUET, lw=0, zorder=0)
            i = j + 1
        else:
            i += 1
    for k, nom in enumerate(TROIS):
        d = C["lignes"].get(nom)
        if d is None:
            continue
        ax.plot(x, d["nov"], color=COL[nom], lw=2.0, label=nom, zorder=3, alpha=0.92)
        if triangles:
            for c in d["cuts"]:
                ax.plot([(c + 0.5) / hb], [1.06 + 0.05 * k], marker="v", ms=7,
                        color=COL[nom], clip_on=False, zorder=6)
    ax.set_ylim(0, 1.45 if legend else 1.2)
    if legend:
        ax.legend(loc="upper left", ncol=3, fontsize=9.5, frameon=False,
                  bbox_to_anchor=(0.0, 1.0), handlelength=1.6)


