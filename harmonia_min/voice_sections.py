"""Les sections trouvées par la VOIX. Le mode `HARMONIA_SECTIONS=voice`.

Louis, 2026-08-07 : « est-ce que tu peux me pousser toutes les nouvelles
contributions qu'on a, les nouvelles façons de sélectionner des sections, en prod
comme ça je peux tester avec quelques chansons et voir ce que ça donne ».

CE QUE CE MODE FAIT, et en quoi il diffère de `harmonic_sections` (le mode livré).
Là-bas, les motifs sont cherchés dans la matrice harmonique seule et les blocs
sont posés sur une grille rigide. Ici l'ordre est inversé : **c'est la voix qui
cherche**.

  1. L'intro finit À LA PREMIÈRE MESURE CHANTÉE. Louis, 2026-08-07 : « par
     défaut, je veux que la chanson commence et que l'intro finisse quand la
     personne commence à chanter ». C'est son choix explicite et non le maximum
     mesuré — 5/10 contre ses annotations, là où la version arbitrée fait 7/10 —
     parce qu'elle est prévisible et ne surprend jamais. L'écart tient à un seul
     phénomène, la LEVÉE : sur quatre morceaux le chanteur attaque une mesure
     avant la barre. Voir `first_sung_bar`, et `intro_end` pour l'arbitrée.
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

PICKUP = 0.40      # au-delà, la 1re note est une LEVÉE, pas un début de section


def sung_start(notes, grid, mute, pickup=PICKUP):
    """Où commence vraiment la chanson : la 1re mesure chantée, LEVÉE comprise.

    Louis, 2026-08-07 : « par défaut, l'intro finit quand la personne commence à
    chanter » — puis, la règle littérale se trompant d'une mesure sur quatre
    morceaux, il a demandé deux hypothèses (la mesure du dessous, celle du
    dessus) arbitrées par la cohérence du reste.

    Ses deux critères ont été mesurés et sont trop faibles : compter les reprises
    donne 5/10, et sur Bein Green comme sur Norah les deux hypothèses donnent
    EXACTEMENT le même compte — décaler d'une mesure décale tous les blocs sans
    changer le nombre de pics. Sommer leurs scores monte à 6/10, sans plus : les
    blocs des deux hypothèses se recouvrent à sept huitièmes, donc tout critère
    bâti sur eux ne varie que de quelques pour cent, ce qui est le bruit.

    Ce qui sépare une levée d'un début de section n'est pas la répétition, c'est
    **où tombe la première note DANS la mesure** — et la mesure le montre sans
    ambiguïté :

        écart nul avec sa vérité    0.01  0.08  0.11  0.19  0.26  0.31
        section une mesure plus loin      0.48  0.67  0.85  0.92

    Rien entre 0.31 et 0.48. Le seuil à 0.40 tombe au milieu d'une région vide :
    il ne départage aucun cas limite, donc il n'est pas réglé sur ces dix
    morceaux, il est posé dans leur trou. 9/10, contre 5/10 pour la règle
    littérale et 7/10 pour l'arbitrage par la reprise vocale.

    Reste The Walk, où la voix chante quatre mesures d'intro : aucune règle à
    une mesure près ne peut l'atteindre (règle #4 — ce reste n'est PAS résolu).
    """
    import numpy as np
    b = first_sung_bar(mute)
    if b + 1 >= len(grid) - 1:
        return b
    inbar = [t for t, d, m in notes if grid[b] <= t < grid[b + 1]]
    if not inbar:
        return b
    phase = (min(inbar) - grid[b]) / max(1e-6, grid[b + 1] - grid[b])
    return b + 1 if phase >= pickup else b


def first_sung_bar(mute):
    """La première mesure où quelqu'un chante. Rien de plus.

    Louis, 2026-08-07 : « par défaut, je veux que la chanson commence et que
    l'intro finisse quand la personne commence à chanter ».

    C'est plus littéral que tout ce qu'on avait : ni arrondi sur la grille de
    deux, ni arbitrage, ni détecteur d'attaque vocale — la première mesure qui
    contient une note chantée. Mesuré contre ses propres annotations : 5/10,
    contre 6/10 pour la règle arrondie et 7/10 pour la version arbitrée.

    L'écart tient à UN seul phénomène, et il est très régulier : quatre morceaux
    se trompent d'exactement une mesure — Bein Green, Let It Be, Norah, Aretha —
    parce que le chanteur attaque en LEVÉE, avant la barre, et que Louis a
    annoté l'intro comme finissant après. Le cinquième écart est The Walk, où la
    voix chante quatre mesures d'intro.

    Cette règle est donc son choix explicite, pas le maximum mesuré, et c'est
    volontaire : elle est prévisible, elle ne surprend jamais, et il a demandé la
    prévisibilité. `intro_end` reste là pour la version arbitrée.
    """
    import numpy as np
    m = np.asarray(mute, bool)
    return int(np.argmin(m)) if (~m).any() else 0


def intro_end(b_sing, M, n, unit=UNIT, margin=ARB_MARGIN, off_grid=OFF_GRID):
    """Où finit l'intro, version ARBITRÉE — plus juste sur le corpus (7/10) mais
    elle déplace parfois le départ loin du premier mot chanté, ce que Louis ne
    veut pas par défaut. Gardée pour la recherche et pour comparaison."""
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
    # LA RÈGLE PAR DÉFAUT, celle que Louis a demandée : l'intro finit quand on
    # commence à chanter. Pas d'arrondi, pas d'arbitrage.
    start = sung_start(notes, grid, mute)

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
