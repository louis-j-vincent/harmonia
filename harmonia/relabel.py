"""λ1 t+b — re-nommer chaque accord par sa différence aux deux accords voisins.

Louis, 2026-10-12 : « on utilise le processeur standard pour inférer les
changements d'accord ; sur chaque accord inféré, on utilise le λ1 t+b pour
inférer l'accord à cet endroit, en faisant le différentiel avec les 2 accords
voisins (pas les temps, les accords) ». Et en mode annotation : un accord qu'on
ajoute sur une case vide reçoit son nom par la même règle.

UNE implémentation, deux appelants :

  * `relabel_segments` — la pipeline (`pipeline.analyze_steps`, juste après
    `musx.redecode`) et le gabarit empilé (`folding._decode_template`). Les
    frontières du Viterbi ne bougent JAMAIS : seul le nom change.
  * `label_for_span` — la route `/api/chord-candidates/<file>` de l'écran
    Annotate, pour un créneau qui n'existait pas à la cuisson.

La règle, par plan musx (triade 73, basse 13 ; 7e/9e/11e/13e INTOUCHÉES) :

    p      = moyenne des trames musx sur l'empan de l'accord
             (même mise en commun que `span_rescore.pool_span_musx` :
             trames [round(t0/FRAME_DT), round(t1/FRAME_DT)), un empan plus
             court qu'une trame lit la trame la plus proche)
    S      = (p − p_avant) + (p − p_après)
    q      = norm( max(0, p + λ·S) + ε ),   ε = 1e-3 (le décodeur prend des log)
    nom    = argmax des log-observations du décodeur vendu sur q
             (`XHMMDecoder.get_chord_tag_obs`, même vocabulaire
             `SETTINGS.chord_dict`, même produit triade × basse × 7e × 9e ×
             11e × 13e — c'est exactement ce que le Viterbi maximise sur un
             empan aux frontières fixées, à la mise en commun près).
    garde  = un accord du Viterbi ne change de nom que si le nom sur q
             DIFFÈRE du nom sur p seul (λ = 0) : à λ = 0 la règle rend le
             décodage tel quel, et tout changement est dû aux voisines.

Choix (à arbitrer par Louis, voir docs/known_issues.md 2026-10-12) :

  * BORDS — une seule voisine : S = p − p_voisine (le terme manquant vaut 0,
    « pas d'information »), PAS 2(p − p_voisine) : l'essai par demi-mesure
    comptait la voisine deux fois et la 1re/dernière case sortait comme
    artefact récurrent. Aucune voisine : S = 0, q = p.
  * N.C. — un segment « N » n'est jamais renommé (le décodeur a dit « pas
    d'accord », le delta n'a pas à en inventer un), et une voisine « N »
    compte comme ABSENTE (règle des bords) : contraster un accord avec du
    silence ferait monter tout le plan. Symétriquement, un accord n'est
    jamais renommé en N.C. (on ne retire pas un accord que le Viterbi a posé).

LES CHAMPS QUI DÉPENDENT DU NOM, dans le chart : `c` (confiance) est relu
sur le NOUVEAU nom par `musx.label_confidence` (`bars.layout_bars`), donc un
accord révélé par le contraste sort avec sa vraie part — souvent plus basse
— et la couleur de doute qui va avec. `sug`, `casc` et `sugBass` ne changent
PAS de définition : ce sont le classement du modèle sur la moyenne p de
l'empan (le compas montre ce que musx entend, pas ce que le contraste
choisit) ; le nom écrit peut donc ne pas être `sug[0]`. `musx_suggestions`
garde la queue écrite quand la case (racine, famille) écrite est classée.
La couleur harmonique et la tonalité (`analyze_harmony`) se recalculent sur
les noms écrits, donc PEUVENT bouger (Cry Me A River, ton C♯m → G♯m).
`relabel_gabarit` (settings) : la règle n'est PAS appliquée au gabarit
empilé d'une lettre repliée tant que Louis n'a pas arbitré son effet.

CE QUE ÇA NE RÉSOUT PAS : (1) une note commune avec une voisine est punie
(D♭6 → D♭ après B♭7 dans l'essai) ; (2) les bords n'ont qu'un côté de
contraste ; (3) ça ne crée ni ne retire aucune frontière — un accord que le
Viterbi n'a pas découpé reste invisible (c'est l'écran Annotate qui le
rend visible : on coupe, `label_for_span` le nomme) ; (4) p est une MOYENNE
de trames, le Viterbi somme des log : la garde ci-dessus empêche ce seul
écart de renommer un accord, mais un nom né du contraste reste lu sur la
moyenne.
"""
from __future__ import annotations

import functools
import logging

import numpy as np

from harmonia import musx as _musx
from harmonia.settings import SETTINGS

logger = logging.getLogger(__name__)

#: plans touchés par le delta : triade (0) et basse (1). Les extensions
#: (7e, 9e, 11e, 13e) gardent p : leur classe « aucune » est constante, S ≈ 0
#: et le bruit gagnerait (essai S⁺ du 2026-10-12 : 9e/11e/13e inventées).
DELTA_PLANES = (0, 1)
EPS = 1e-3


@functools.lru_cache(maxsize=4)
def _hmm(chord_dict: str):
    """Le décodeur vendu (vocabulaire seul ; ses pénalités ne servent pas ici)."""
    with _musx._InMusxDir():
        return _musx._decoder(_musx.DEFAULT_PENALTY, chord_dict=chord_dict)


def frame_span(t0: float, t1: float, n_frame: int) -> tuple[int, int]:
    """[f0, f1) — les trames musx d'un empan, convention `pool_span_musx`.

    La trame f couvre [f·FRAME_DT, (f+1)·FRAME_DT) ; `musx._tags_to_lab` écrit
    les frontières à des multiples exacts de FRAME_DT, donc un segment du
    Viterbi retombe sur ses trames exactes. Empan vide (plus court qu'une
    trame, ou hors des postérieures) : la trame la plus proche du milieu.
    """
    f0 = max(0, int(round(t0 / _musx.FRAME_DT)))
    f1 = min(n_frame, int(round(t1 / _musx.FRAME_DT)))
    if f1 <= f0:
        j = int(np.clip(round(0.5 * (t0 + t1) / _musx.FRAME_DT), 0, max(n_frame - 1, 0)))
        return j, j + 1
    return f0, f1


def pool_span(probs, t0: float, t1: float) -> list[np.ndarray]:
    """p : la moyenne de chaque plan musx sur l'empan (6 vecteurs)."""
    n = int(np.asarray(probs[0]).shape[0])
    f0, f1 = frame_span(t0, t1, n)
    return [np.asarray(P[f0:f1], dtype=np.float64).mean(0) for P in probs]


def delta_q(p, voisines, lam: float) -> list[np.ndarray]:
    """q = norm(max(0, p + λS) + ε) sur triade+basse, norm(p + ε) ailleurs.

    `voisines` : liste de 0, 1 ou 2 postérieures mises en commun (les voisines
    PRÉSENTES ; une absente ne contribue rien, voir le docstring du module).
    """
    out = []
    for k, x in enumerate(p):
        y = np.asarray(x, dtype=np.float64)
        if k in DELTA_PLANES and voisines:
            S = sum(y - np.asarray(v[k], dtype=np.float64) for v in voisines)
            y = y + lam * S
        y = np.maximum(0.0, y) + EPS
        out.append(y / y.sum())
    return out


def scores(q, chord_dict: str | None = None) -> tuple[list[str], np.ndarray]:
    """(noms, log-observations) du décodeur vendu pour UNE postérieure q."""
    hmm = _hmm(chord_dict or SETTINGS.chord_dict)
    names, lp = hmm.get_chord_tag_obs([np.asarray(x)[None, :] for x in q])
    return names, lp[0]


def best_labels(q, *, top_k: int = 3, allow_n: bool = False,
                chord_dict: str | None = None) -> list[tuple[str, float]]:
    """Les `top_k` étiquettes musx sur q, avec leur part (softmax des log-obs
    sur le vocabulaire, N exclu sauf `allow_n`)."""
    names, lp = scores(q, chord_dict)
    lp = np.array(lp, dtype=np.float64)
    if not allow_n:
        lp[[i for i, n in enumerate(names) if n == "N"]] = -np.inf
    w = np.exp(lp - lp.max())
    w /= w.sum()
    order = np.argsort(-w)[:top_k]
    return [(names[i], float(w[i])) for i in order]


def relabel_segments(segments, probs, *, lam: float | None = None,
                     chord_dict: str | None = None):
    """[(t0, t1, label)] → (même liste, noms λ-renommés ; rapport).

    Frontières et nombre de segments identiques à l'entrée, par construction.
    Les voisines sont les segments-accords adjacents DE LA LISTE (pas les
    temps) ; une voisine « N » est absente. Le rapport :
    `{"lam", "n", "changed", "pairs": [[avant, après], …]}`.
    """
    lam = SETTINGS.relabel_lambda if lam is None else float(lam)
    segs = list(segments)
    pooled = [pool_span(probs, t0, t1) for t0, t1, _ in segs]
    out, pairs = [], []
    for i, (t0, t1, lab) in enumerate(segs):
        if lab == "N":
            out.append((t0, t1, lab))
            continue
        vois = [pooled[j] for j in (i - 1, i + 1)
                if 0 <= j < len(segs) and segs[j][2] != "N"]
        q = delta_q(pooled[i], vois, lam)
        new = best_labels(q, top_k=1, chord_dict=chord_dict)[0][0]
        # Seul le CONTRASTE renomme : si q à λ = 0 (la moyenne seule) donne
        # déjà ce nom, l'écart au Viterbi vient de la mise en commun (moyenne
        # des trames contre somme des log), pas des voisines — on garde le
        # Viterbi. Mesuré 2026-10-12 sur 4 morceaux : 5 accords sur 333
        # changeaient déjà à λ = 0 (C/G → Gm7, A♭ → B♭/A♭, A7 → A7sus4…).
        if new != lab and new == best_labels(delta_q(pooled[i], [], 0.0), top_k=1,
                                             chord_dict=chord_dict)[0][0]:
            new = lab
        if new != lab:
            pairs.append([lab, new])
        out.append((t0, t1, new))
    logger.info("relabel λ=%.2f : %d/%d accords renommés", lam, len(pairs),
                sum(1 for s in segs if s[2] != "N"))
    return out, {"lam": lam, "n": sum(1 for s in segs if s[2] != "N"),
                 "changed": len(pairs), "pairs": pairs}


def label_for_span(probs, span, prev_span=None, next_span=None, *,
                   lam: float | None = None, top_k: int = 3,
                   chord_dict: str | None = None) -> dict:
    """Le nom λ1 t+b d'un créneau, ses voisins donnés en secondes.

    Pour l'accord qu'on AJOUTE à la main (route `/api/chord-candidates`) :
    `prev_span`/`next_span` sont les accords juste avant et juste après LUI
    dans la suite éditée (après la coupe), None = absent (bord, ou N.C.).
    Rend `{"label", "root", "q", "bass", "top": [{label, root, q, bass, c}],
    "voisines": n, "lam"}` — `root/q/bass` au format du chart
    (`labels.to_chord`).
    """
    from harmonia.labels import to_chord
    lam = SETTINGS.relabel_lambda if lam is None else float(lam)
    p = pool_span(probs, *span)
    vois = [pool_span(probs, *s) for s in (prev_span, next_span) if s is not None]
    q = delta_q(p, vois, lam)
    top = []
    for lab, w in best_labels(q, top_k=top_k, chord_dict=chord_dict):
        ch = to_chord(lab) or {"root": 0, "q": "", "bass": -1}
        top.append({"label": lab, "root": ch["root"], "q": ch["q"],
                    "bass": ch["bass"], "c": round(w, 3)})
    return {**top[0], "top": top, "voisines": len(vois), "lam": lam}
