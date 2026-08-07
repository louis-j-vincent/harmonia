"""Les sections trouvées par la VOIX. Le mode `HARMONIA_SECTIONS=voice`.

Louis, 2026-08-07 : « est-ce que tu peux me pousser toutes les nouvelles
contributions qu'on a, les nouvelles façons de sélectionner des sections, en prod
comme ça je peux tester avec quelques chansons et voir ce que ça donne ».

CE QUE CE MODE FAIT, et en quoi il diffère de `harmonic_sections` (le mode livré).
Là-bas, les motifs sont cherchés dans la matrice harmonique seule et les blocs
sont posés sur une grille rigide. Ici l'ordre est inversé : **c'est la voix qui
cherche**.

  1. L'intro finit au premier début de bloc à partir du chant, avec un ARBITRE
     quand le chant démarre sur une mesure impaire : on garde le candidat dont
     le bloc chanté **se reproduit** le mieux ailleurs, parce qu'une intro, par
     définition, ne se reproduit pas. Mesuré 7/10 contre la vérité de Louis, 6/10
     sans l'arbitre.
  2. Le premier bloc de 8 mesures est glissé sur toute la chanson ; chacun de ses
     vrais pics est une reprise, toutes verrouillées d'un coup. Puis le premier
     bloc de 8 libre, et ainsi de suite.
  3. Ce qui reste est repassé en blocs de 4 : mesuré contre sa vérité, les blocs
     de 8 placent mieux les frontières mais ne couvrent que 90 % du morceau, les
     blocs de 4 en couvrent 95 %. C'est son idée — « les blocs de 8 ANCRENT la
     chanson, après il ne reste qu'à combler les trous ».

LE SCORE, validé par lui le 2026-08-07 : moyenne des deux voies (harmonie et
chant), chacune valant `hauteur + ½ finesse` de sa courbe de glissement, le tout
divisé par le score de l'ancre. Trois raisons, toutes mesurées :

  * l'harmonie dedans, parce que la voix seule ne recouvrait pas This Love ;
  * relatif à l'ancre, parce que sans ça les morceaux ne vivent pas au même
    étage — de 0.54 à 1.68 selon la chanson, donc aucun seuil commun possible ;
  * le chant pèse **ce qu'il peut entendre** : sur un passage muet il s'abstient
    au lieu de voter zéro. Sans ça le A de la mesure 37 de Let It Be, vu à 0.994
    par l'harmonie, était rejeté à cause du solo de guitare.

Et la TRANSPOSITION : une modulation fait tourner le vecteur de douze hauteurs,
donc une reprise transposée devient invisible. Sur Sunny les reprises scorent
0.10–0.16 sans rotation et 0.93–0.97 avec. Un seul demi-ton par bloc, le même
pour les deux voies, l'harmonie pesant plus quand une transposition est en jeu.

CE QUE CE MODE NE PROUVE PAS, ET POURQUOI IL N'EST PAS LE DÉFAUT. Il a été mesuré
contre les annotations de sections de Louis, jamais contre ce que
`harmonic_sections` produit sur les mêmes morceaux. Le faire défaut serait donc
un changement non mesuré sur toutes ses grilles existantes. Il coûte aussi une
séparation de voix (demucs) et un suivi de hauteur (pyin) au premier passage —
mis en cache ensuite, mais une minute la première fois.

    HARMONIA_SECTIONS=voice   pour l'essayer
    HARMONIA_SECTIONS=harmonic (défaut, inchangé)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("harmonia_min.voice_sections")

BLOCK = 8          # « nos blocs de 8 »
FILL = 4           # …puis on comble en 4
UNIT = 2           # une section ne commence pas sur une mesure impaire
THR8 = 0.66        # les seuils qui maximisent l'accord avec la vérité de Louis
THR4 = 0.70
SHARP_W = 0.5
ARB_MARGIN = 0.08
OFF_GRID = 0.10

_REPO = Path(__file__).resolve().parent.parent


def _scripts():
    """Les chargeurs de voix vivent dans `scripts/`.

    Couture assumée : `vocal_anchor` (demucs) et `vocal_melody` (pyin) sont des
    outils de recherche, avec leur cache et leurs réglages, et les recopier ici
    créerait deux vérités qui divergeraient. On les importe donc là où ils sont.
    Si ce mode devient le défaut, c'est la première chose à remonter proprement.
    """
    p = str(_REPO / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import vocal_anchor as VA          # noqa: E402
    import vocal_melody as VM          # noqa: E402
    import melody_ssm as MS            # noqa: E402
    import blocks8 as B8               # noqa: E402
    return VA, VM, MS, B8


# ── le score ────────────────────────────────────────────────────────────────

def _slide(M, a0, L, n):
    out = np.zeros(n)
    for c in range(0, n - L + 1):
        v = [M[a0 + i, c + i] for i in range(L) if a0 + i < n and c + i < n]
        out[c] = float(np.mean(v)) if v else 0.0
    return out


def _sharp(cur, p, n, unit=UNIT):
    return 2 * cur[p] - cur[max(0, p - unit)] - cur[min(n - 1, p + unit)]


def _one(cur, n, unit=UNIT, w=SHARP_W):
    return np.array([float(cur[p]) + w * max(0.0, _sharp(cur, p, n, unit))
                     for p in range(n)])


def block_score(cur_m, cur_h, b0, n, mute=None, block=BLOCK):
    """Le score validé : les deux voies, relatif à l'ancre, le chant abstenu s'il
    n'entend rien."""
    sm, sh = _one(cur_m, n), _one(cur_h, n)
    if mute is None:
        mix = (sm + sh) / 2.0
    else:
        heard = np.ones(n)
        for p in range(n):
            k = [i for i in range(block) if b0 + i < n and p + i < n]
            if k:
                heard[p] = float(np.mean([(not mute[b0 + i]) and (not mute[p + i])
                                          for i in k]))
        mix = (sh + heard * sm) / (1.0 + heard)
    return mix / max(1e-6, float(mix[b0]))


def _peaks(sc, b0, n, thr, block, claimed):
    """Les vrais pics : maximum local, au-dessus du seuil, sur la grille — et
    LE PLUS FORT SE SERT LE PREMIER (sinon le premier tue le meilleur)."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(sc, height=thr, distance=UNIT)
    cand = [int(p) for p in idx
            if (p - b0) % UNIT == 0 and abs(p - b0) >= block
            and p + block <= n and not claimed[p:p + block].any()]
    out = []
    for p in sorted(cand, key=lambda p: -sc[p]):
        if any(abs(p - q) < block for q in out):
            continue
        out.append(p)
    return sorted(out)


# ── l'intro ─────────────────────────────────────────────────────────────────

def intro_end(b_sing, M, n, unit=UNIT, margin=ARB_MARGIN, off_grid=OFF_GRID):
    """Où finit l'intro. La règle de la voix, arbitrée par « ça se reproduit ? »."""
    if b_sing is None:
        return 0
    lo = max(0, min(n - 1, b_sing - b_sing % unit))
    hi = max(0, min(n - 1, b_sing + (-b_sing) % unit))

    def rec(s, L=BLOCK):
        if s < 0 or s + L > n:
            return -1.0
        c = _slide(M, s, L, n)
        v = [c[p] for p in range(0, n - L + 1, unit) if abs(p - s) >= L]
        return max(v) if v else 0.0

    base = lo if (lo != hi and rec(lo) > rec(hi) + margin) else hi
    if base != b_sing and rec(b_sing) > rec(base) + off_grid:
        return b_sing
    return base


# ── l'ancrage ───────────────────────────────────────────────────────────────

def _pass(S, M, mute, n, start, block, thr, claimed, max_blocks=20):
    runs, cursor = [], start
    while cursor + block <= n and len(runs) < max_blocks:
        if claimed[cursor:cursor + block].any():
            cursor += UNIT
            continue
        cm, ch = _slide(M, cursor, block, n), _slide(S, cursor, block, n)
        sc = block_score(cm, ch, cursor, n, mute=mute, block=block)
        occ = _peaks(sc, cursor, n, thr, block, claimed)
        if occ:
            runs.append({"b0": cursor, "occ": occ, "block": block})
            for c in [cursor] + occ:
                claimed[c:min(n, c + block)] = True
        cursor += block
    return runs


def detect_sections(grid, triad, bars=None, audio=None):
    """[{b0, b1, label}] sur les indices de mesure — contigu, couvrant.

    Même contrat que `harmonic_sections.detect_sections` : la sortie pave le
    morceau sans trou ni chevauchement, ce qui est vérifié avant de rendre.
    """
    from harmonia_min import harmonic_sections as HS
    n = len(grid) - 1
    if n < 4 or audio is None:
        log.warning("sections=voice : pas d'audio, on retombe sur harmonic")
        return HS.detect_sections(grid, triad, bars)

    VA, VM, MS, B8 = _scripts()
    V = HS.harmonic_vectors(triad, grid)
    S = V @ V.T
    voc = VA.separate_vocals(Path(audio))
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)
    onset, *_ = B8.sing_onset(voc)
    start = intro_end(VA.bar_of(grid, onset), M, n)

    claimed = np.zeros(n, bool)
    runs = _pass(S, M, mute, n, start, BLOCK, THR8, claimed)
    runs += _pass(S, M, mute, n, start, FILL, THR4, claimed)

    owner = np.full(n, -1)
    for i, r in enumerate(runs):
        for c in [r["b0"]] + r["occ"]:
            owner[c:min(n, c + r["block"])] = i

    out, b = [], 0
    if start > 0:
        out.append({"b0": 0, "b1": start - 1, "label": "intro"})
        b = start
    while b < n:
        z = b
        while z + 1 < n and owner[z + 1] == owner[b]:
            z += 1
        out.append({"b0": b, "b1": z, "label": owner[b]})
        b = z + 1

    # les lettres dans l'ordre d'apparition ; ce qui n'est réclamé par personne
    # garde une lettre à lui plutôt que de disparaître — un pont existe.
    ren, k = {}, 0
    for s in out:
        if isinstance(s["label"], (int, np.integer)):
            if s["label"] not in ren:
                ren[s["label"]] = chr(ord("A") + k); k += 1
            s["label"] = ren[s["label"]]
    for a, c in zip(out, out[1:]):
        assert c["b0"] == a["b1"] + 1, f"trou/chevauchement : {a} -> {c}"
    assert out[0]["b0"] == 0 and out[-1]["b1"] == n - 1, "le pavage ne couvre pas"
    log.info("sections=voice : intro %d mesures, %d bloc(s), %d section(s)",
             start, len(runs), len(out))
    return out
