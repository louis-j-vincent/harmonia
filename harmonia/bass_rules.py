"""Quelle note écrire à la basse d'un accord — les règles arbitrées à l'oreille.

Les tables et les seuils de ce module ne sont PAS des réglages : chacun a été
tranché par Louis sur des cas réels le 2026-09-15, en deux tours d'écoute
(33 arbitrages conservés dans `state/human/bass_verdicts.json`, rejoués par
`tests/test_bass_rules.py`). Le raisonnement complet, avec ce que chaque
mesure a coûté et ce qu'elle a réfuté, est dans `docs/bass_slash_rules.md`.

    LES TROIS BRANCHES, dans l'ordre
    1. la FONDAMENTALE de l'accord est retrouvée quelque part dans le span,
       à `FLOOR` ou plus  ->  c'est elle la basse, la sonde du temps 1 s'était
       trompée. (16/16 arbitrages justes)
    2. sinon la QUARTE au-dessus de la fondamentale (famille sus : Bb-7/Eb)
       est retrouvée  ->  c'est elle.
    3. sinon on garde la basse lue au temps 1, MAIS seulement si l'intervalle
       qu'elle forme avec la fondamentale est jouable (`PLAUSIBLE`). Sinon on
       écrit l'accord nu.

CE QUE CE MODULE NE FAIT PAS : il ne lit pas l'audio. On lui donne la
séquence de basses déjà mesurée (`harmonia.nnls_features.bass_pc_onset` à
l'attaque de chaque temps) et il décide. Il n'est pas non plus branché dans
`pipeline.py` — mesuré sur 6 morceaux / 351 accords seulement (voir le doc).
"""
from __future__ import annotations

#: Part minimale de l'énergie grave pour qu'une lecture compte comme trouvée.
#: Plateau large (les arbitrages tiennent de 20 % à 32 %), 30 % est au milieu.
FLOOR = 30.0

#: Intervalles, en demi-tons AU-DESSUS DE LA FONDAMENTALE DE L'ACCORD, qu'une
#: vraie basse de slash peut former. Établi au 2e tour : une 9e à la basse
#: (F/G, Db/Eb, E/Gb) est réelle 3 fois sur 3 — je l'avais classée douteuse,
#: l'oreille de Louis a corrigé ; une b9 (F-7/Gb) est un frottement et n'est
#: jamais autre chose qu'une erreur de lecture.
PLAUSIBLE = frozenset({0, 2, 3, 4, 5, 7, 10, 11})
#: Le complément : b9, b5, b13, 6te. Une basse à l'un de ces intervalles se
#: jette, quelle que soit sa confiance — c'est l'intervalle qui décide, pas
#: la confiance (le seul rejet juste du 2e tour était le PLUS confiant des
#: quatre, 32,4 % ; les trois vraies basses étaient à 24-28 %).
IMPOSSIBLE = frozenset(range(12)) - PLAUSIBLE

#: La quinte est ASYMÉTRIQUE (Louis, 2026-09-15). Entre deux lectures de
#: basse, +7 signifie « la seconde est la quinte de la première » : la
#: première est la fondamentale, la seconde la décore. +5 dit l'inverse (une
#: quarte au-dessus, c'est une quinte en dessous) : c'est la seconde qui
#: porte. Utiliser l'intervalle SANS cette direction ne marche pas — mesuré,
#: 6/11 -> 4/11.
DECORATIVE_ABOVE = 7
DECORATIVE_OCTAVE = 0
FUNDAMENTAL_ABOVE = 5

#: Intervalle de la famille sus, au-dessus de la fondamentale de l'accord
#: (Bb-7/Eb : Bb est la quinte d'Eb, donc Eb porte).
SUS_ABOVE_ROOT = 5


def decide_bass(root_pc: int, beats: list[dict], floor: float = FLOOR) -> tuple[int, str]:
    """(classe de hauteur de la basse à écrire, la branche qui a décidé).

    `beats` : la basse lue à l'attaque de chaque temps de l'accord, dans
    l'ordre, chaque entrée `{"pc": int, "share": float}` où `share` est en
    pourcents — exactement ce que rend `bass_pc_onset` suivi d'un argmax.
    Le premier élément est le temps 1.

    Rendre la fondamentale veut dire « pas de slash » : c'est au chart
    d'écrire l'accord nu quand la basse rendue est la fondamentale.
    """
    if not beats:
        return root_pc % 12, "aucune lecture"
    root_pc %= 12

    best: dict[int, float] = {}
    for b in beats:
        pc = int(b["pc"]) % 12
        best[pc] = max(best.get(pc, 0.0), float(b["share"]))

    if best.get(root_pc, 0.0) >= floor:
        return root_pc, "fondamentale retrouvée"

    sus_pc = (root_pc + SUS_ABOVE_ROOT) % 12
    if best.get(sus_pc, 0.0) >= floor:
        return sus_pc, "quarte (sus)"

    onset_pc = int(beats[0]["pc"]) % 12
    if (onset_pc - root_pc) % 12 in PLAUSIBLE:
        return onset_pc, "temps 1 conservé"
    return root_pc, "temps 1 jeté (intervalle impossible)"


def is_decoration(onset_pc: int, cand_pc: int, *, returns: bool) -> bool:
    """La seconde lecture décore-t-elle la première, ou porte-t-elle ?

    Sert à départager DEUX LECTURES DE BASSE d'un même accord, pas à choisir
    ce qu'on écrit (c'est `decide_bass`). Le retour de la première note
    (A-B-A) est un indice de décoration — SAUF sur une quarte, où les
    arbitrages disent l'inverse 4 fois sur 4 : quand la basse alterne entre
    une note et sa quinte, la fondamentale porte, qu'elle tombe sur le temps
    fort ou non.
    """
    iv = (int(cand_pc) - int(onset_pc)) % 12
    if iv in (DECORATIVE_ABOVE, DECORATIVE_OCTAVE):
        return True
    if iv == FUNDAMENTAL_ABOVE:
        return False
    return bool(returns)
