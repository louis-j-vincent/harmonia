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


def _shape_feats(x: np.ndarray, n: int) -> dict:
    """Les formes que Louis lit à l'œil, rendues mesurables. `x` est la nouveauté
    d'un substrat À L'ÉCHELLE DE LA MESURE.

    Deux familles, et elles viennent toutes les deux de lui :

    * le MONT (2026-08-12) — « les pics allongés pseudo-symétriques sont TOUJOURS
      marqueurs d'une section ». Mesuré : 21 monts sur les douze morceaux, 21
      contiennent une frontière validée.
    * la FORME (sa précision du même jour) — « montée douce, aval abrupt, vallée
      au milieu, puis le miroir ». Mesurée à l'échelle de la mesure : douze
      occurrences, et le sommet du lobe GAUCHE est à ±2 mesures d'une vraie
      frontière **douze fois sur douze**. Mieux : l'écart est CONSTANT par
      morceau (+2 partout sur Blue Lights, +3 sur Stand By Me, +1 sur The Walk),
      donc la forme repère la frontière à un décalage près propre au morceau.

    On rend des distances plutôt que des indicatrices : un modèle linéaire ne
    peut rien faire d'un drapeau qui vaut 1 sur trois mesures du morceau.
    """
    from scipy.signal import find_peaks
    m = len(x)
    out = {}

    # asymétrie locale : longueur de la montée qui finit ici moins celle de la
    # descente qui commence ici. C'est « monte doucement, tombe vite » en un
    # nombre.
    rise = np.zeros(m); fall = np.zeros(m)
    for i in range(1, m):
        rise[i] = rise[i-1] + 1 if x[i] >= x[i-1] else 0
    for i in range(m - 2, -1, -1):
        fall[i] = fall[i+1] + 1 if x[i] >= x[i+1] else 0
    out["asym"] = (rise - fall) / max(1.0, m ** 0.5)

    # le MONT : au-dessus de la médiane, large d'au moins 6 mesures
    pos = x[x > 0]
    thr = float(np.quantile(pos, 0.5)) if pos.size else 0.0
    tops, inm = [], np.zeros(m)
    i = 0
    while i < m:
        if x[i] >= thr:
            j = i
            while j + 1 < m and x[j + 1] >= thr:
                j += 1
            if j - i + 1 >= 6:
                inm[i:j+1] = 1.0
                tops.append(i + int(np.argmax(x[i:j+1])))
            i = j + 1
        else:
            i += 1
    out["in_mont"] = inm
    out["d_mont"] = _dist_to(tops, m)

    # la FORME : deux lobes miroirs séparés par une vallée creuse
    pk, _ = find_peaks(x, prominence=0.08)
    vals, lefts = [], []
    for a, b in zip(pk, pk[1:]):
        if b - a > 20:
            continue
        v = a + int(np.argmin(x[a:b+1]))
        if x[v] > 0.55 * min(x[a], x[b]):
            continue
        s_ = a
        while s_ > 0 and x[s_-1] <= x[s_]:
            s_ -= 1
        e_ = b
        while e_ + 1 < m and x[e_+1] <= x[e_]:
            e_ += 1
        if (a - s_) < 1.4 * max(1, v - a):
            continue
        if (e_ - b) < 1.4 * max(1, b - v):
            continue
        vals.append(v); lefts.append(a)
    out["d_vallee"] = _dist_to(vals, m)
    out["d_lobe_g"] = _dist_to(lefts, m)
    return out


def _dist_to(idx, m, cap=8):
    """Distance (bornée) à l'élément le plus proche d'une liste de positions."""
    if not idx:
        return np.full(m, 1.0)
    a = np.asarray(idx)
    d = np.abs(np.arange(m)[:, None] - a[None, :]).min(1)
    return np.minimum(d, cap) / cap


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

    # les FORMES de Louis, sur chaque substrat (mont, vallée, asymétrie)
    for nm in SUBSTRATES:
        xb = to_bar(_nov(by[nm]["S"], 8 * hb))
        for k, v in _shape_feats(np.asarray(xb, float), n).items():
            add(v, f"forme.{k}[{nm}]")

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


def sung_start_bar(stem: str, grid, n: int) -> int:
    """La mesure où le chant commence — la règle de la prod, réutilisée ailleurs.

    `voice_sections` s'en sert pour finir l'intro. On s'en sert ici pour donner
    sa PHASE à la grille métrique : la forme commence là où le chant commence, et
    tout est un multiple de 4 mesures à partir de là. Mesuré (±0 mesure, budget
    1/8, couverture 12 %) : la phase du chant rend 60 % de rappel médian contre
    48 % pour la phase choisie par le score du modèle, et l'oracle est à 61 %.
    Elle répare exactement les trois morceaux où la phase du modèle était fausse
    — Sunny 0 -> 73 %, Stand By Me 0 -> 60 %, Chain of Fools 0 -> 22 % — et
    n'abîme aucun des autres.
    """
    import harmonia_min.voice_sections as VS
    VA, VM, MS, _ = VS._scripts()
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    _M, mute = MS.melody_bars(notes, grid, n)
    return int(VS.sung_start(notes, grid, mute))


def build(stems=None) -> dict:
    stems = stems or [s for s, _ in SONGS]
    out = {}
    names = None
    for stem in stems:
        X, y, names, n = song_features(stem)
        out[f"X:{stem}"] = X.astype(np.float32)
        out[f"y:{stem}"] = y.astype(np.float32)
        _P, _k, _r, _l, _n, extra = fused_profile(stem)
        out[f"sung:{stem}"] = np.array([sung_start_bar(stem, extra["grid"], n)])
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
