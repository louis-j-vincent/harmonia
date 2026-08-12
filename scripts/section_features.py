"""Les features par mesure, pour le modèle de sections. Une seule vérité.

    .venv/bin/python scripts/section_features.py        # (re)construit le cache

Louis, 2026-08-12 : « ok go pour le modèle ML ».

Ce module ne modélise rien. Il transforme les ingrédients qu'on a déjà — les
sept matrices SSM, la voix, les accords — en un tableau `(n_mesures, n_features)`
par morceau, plus le vecteur d'étiquettes « cette mesure est-elle un début de
section ». Tout ce qui suit (criblage linéaire, puis modèle séquentiel) lit
CE fichier et rien d'autre, pour qu'il n'y ait jamais deux définitions des
features en circulation.

LES FAMILLES DE FEATURES, et pourquoi chacune :

  nouveauté ×4 largeurs   La nouveauté en damier de chaque matrice, aux noyaux
                          2, 4, 8 et 16 mesures. Aujourd'hui on n'en calcule
                          qu'une, à 8 mesures, et ce choix est arbitraire : une
                          frontière de couplet et une fin de pont ne se lisent
                          pas à la même échelle. Le modèle choisira.
  similarité à distance   `S[b, b±L]` pour L = 2, 4, 8, 16 mesures. C'est
                          l'information de PÉRIODE — « cette mesure se rejoue
                          huit plus loin » — que la nouveauté ne porte pas.
  voix                    muette / début de silence / fin de silence. L'entrée
                          du chant et les instrumentaux.
  accords                 changement d'accord sur la mesure, combien, et si le
                          rythme de changement est le même qu'à la mesure
                          précédente (l'idée du prior de répétition).
  position                b mod 2/4/8, position relative, distance aux bouts.
  contexte du morceau     le contraste de chaque matrice et son étalement hors
                          diagonale, diffusés sur toutes les mesures. C'est ce
                          qui permet à un modèle d'apprendre « sur CE morceau,
                          écoute le timbre » — la version apprenable de la
                          pondération à la main de `peak_profile`.

L'ÉTIQUETTE est `1` sur la mesure de début de chaque section annotée (sauf la
mesure 0, qui est un début par construction et n'apprend rien).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from ssm_zoo import SONGS, gt_sections                      # noqa: E402
from peak_profile import fused_profile                      # noqa: E402
from harmonia_min.sections import _novelty as prod_novelty  # noqa: E402

CACHE = HERE / "data" / "cache" / "section_feats.npz"
KERNELS_BARS = (2, 4, 8, 16)       # largeurs de noyau, en mesures
LAGS_BARS = (2, 4, 8, 16)          # distances de similarité, en mesures
SUBSTRATES = ["accords", "basse", "harmonie", "basse + harmonie",
              "voix", "rythme", "timbre"]


def _nov(S: np.ndarray, kw_hb: int) -> np.ndarray:
    """Nouveauté en damier, bords masqués et normalisée — celle de la prod."""
    nv = np.clip(prod_novelty(S, kw_hb), 0, None)
    m = len(nv)
    out = np.zeros(m)
    lo, hi = kw_hb, m - kw_hb
    if hi > lo:
        out[lo:hi] = nv[lo:hi] / max(1e-9, nv[lo:hi].max())
    return out


def song_features(stem: str) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """(X (n_bars, F), y (n_bars,), noms, n_bars)."""
    P, kept, rest, lines, n, extra = fused_profile(stem)
    m = len(P)
    hb = m // n                                    # cases par mesure (2)
    by = {d["nom"]: d for d in lines}
    cols, names = [], []

    def add(v, name):
        cols.append(np.asarray(v, float).reshape(n)); names.append(name)

    # à la MESURE : on prend la case de demi-mesure qui ouvre la mesure
    def to_bar(x):
        return np.asarray(x)[::hb][:n]

    for nm in SUBSTRATES:
        S = by[nm]["S"]
        for kb in KERNELS_BARS:
            add(to_bar(_nov(S, kb * hb)), f"nov[{nm}]@{kb}")
        for L in LAGS_BARS:
            lag = L * hb
            fwd = np.array([S[i, i + lag] if i + lag < m else 0.0
                            for i in range(m)])
            bwd = np.array([S[i, i - lag] if i - lag >= 0 else 0.0
                            for i in range(m)])
            add(to_bar(fwd), f"lag+{L}[{nm}]")
            add(to_bar(bwd), f"lag-{L}[{nm}]")

    # la voix
    mute = np.asarray(extra["mute"], bool) if extra.get("mute") is not None \
        else np.zeros(m, bool)
    mb = to_bar(mute).astype(float)
    add(mb, "voix.muette")
    add(np.r_[0, np.diff(mb) > 0].astype(float), "voix.debut_silence")
    add(np.r_[0, np.diff(mb) < 0].astype(float), "voix.fin_silence")

    # les accords : le rythme de changement, mesure par mesure
    bars = extra.get("bars") or []
    nch = np.zeros(n)
    sig = [()] * n
    for b in range(min(n, len(bars))):
        ev = [e for e in (bars[b] or []) if not e.get("carry")]
        nch[b] = len(ev)
        sig[b] = tuple(sorted(e.get("beat", 0) for e in ev))
    add((nch > 0).astype(float), "acc.change")
    add(nch, "acc.n_changes")
    add([1.0 if b > 0 and sig[b] == sig[b - 1] else 0.0 for b in range(n)],
        "acc.meme_rythme")

    # la position dans le morceau
    idx = np.arange(n)
    for k in (2, 4, 8):
        add((idx % k == 0).astype(float), f"pos.mod{k}")
    add(idx / max(1, n - 1), "pos.rel")
    add(np.minimum(idx, 16) / 16, "pos.depuis_debut")
    add(np.minimum(n - 1 - idx, 16) / 16, "pos.avant_fin")

    # le contexte du morceau, diffusé (c'est lui qui permet la porte apprise)
    i = np.arange(m)
    far = np.abs(i[:, None] - i[None, :]) >= 4
    for nm in SUBSTRATES:
        add(np.full(n, by[nm]["contraste"]), f"ctx.contraste[{nm}]")
        add(np.full(n, float(by[nm]["S"][far].std())), f"ctx.etalement[{nm}]")

    X = np.stack(cols, axis=1)
    gt = gt_sections(stem)
    y = np.zeros(n)
    if gt:
        for s in gt["sections"][1:]:
            if 0 < s["b0"] < n:
                y[s["b0"]] = 1.0
    return X, y, names, n


def build(stems=None) -> dict:
    stems = stems or [s for s, _ in SONGS]
    out = {}
    names = None
    for stem in stems:
        X, y, names, n = song_features(stem)
        out[f"X:{stem}"] = X.astype(np.float32)
        out[f"y:{stem}"] = y.astype(np.float32)
        print(f"  ok {stem[:34]:34s} {X.shape[0]:3d} mesures × {X.shape[1]} features "
              f"· {int(y.sum())} frontières")
    out["names"] = np.array(names)
    out["stems"] = np.array(stems)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"wrote {CACHE.relative_to(HERE)}")
    return out


def load() -> dict:
    if not CACHE.exists():
        return build()
    return dict(np.load(CACHE, allow_pickle=False))


if __name__ == "__main__":
    build(sys.argv[1:] or None)
