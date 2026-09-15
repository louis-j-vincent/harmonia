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
#: vraie basse de slash peut former. UNIQUEMENT ceux que les arbitrages
#: soutiennent, et rien d'autre :
#:
#:   +2  9e      F/G, Db/Eb, E/Gb  -> réels 3 fois sur 3
#:   +3  m3      Eb-/Gb            -> réel
#:   +4  M3      D/Gb, G/B         -> réels 2 fois sur 2
#:   +7  5te     D/A, C/G x2       -> réels 3 fois sur 5
#:
#: CORRECTION DU 2026-09-15 AU SOIR. La première version de cette table
#: contenait +5, +10 et +11, mis là par mon a priori musical et jamais
#: vérifiés contre les arbitrages. Or ceux-ci les REFUSENT :
#:   +11 (7M)     E/Eb et Bb/A jugés faux -> c'est ce qui écrivait « D/Db »,
#:                que Louis a vu tout de suite en ouvrant la page ;
#:   +5  (4te)    Eb/Ab jugé faux — le seul cas qui testait la famille sus ;
#:   +10 (b7)     jamais arbitré, donc jamais écrit.
#: Un intervalle qui n'a pas été entendu ne s'écrit pas : le défaut est
#: l'accord nu, pas le bénéfice du doute.
PLAUSIBLE = frozenset({0, 2, 3, 4, 7})
#: Réfutés par l'oreille : b9 (F-7/Gb), 4te (Eb/Ab), 7M (E/Eb, Bb/A).
REFUTED = frozenset({1, 5, 11})
#: Jamais soumis à l'oreille : b5, b13, 6te, b7. Traités comme impossibles
#: par défaut — à rouvrir si un arbitrage les valide un jour.
UNTESTED = frozenset({6, 8, 9, 10})
#: Tout ce qui ne s'écrit pas. C'est l'intervalle qui décide, pas la
#: confiance (le seul rejet juste du 2e tour était le PLUS confiant des
#: quatre, 32,4 % ; les trois vraies basses étaient à 24-28 %).
IMPOSSIBLE = REFUTED | UNTESTED

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
#:
#: BRANCHE DÉSARMÉE le 2026-09-15 au soir. Elle reposait sur c11 — qui s'est
#: révélé être un changement d'accord manqué, pas un slash — et le seul
#: arbitrage qui l'ait testée directement la refuse (r10, `Eb/Ab` jugé faux).
#: `SUS_ENABLED = False` la laisse dans le code, lisible et réactivable, mais
#: hors du chemin : la rallumer demande de nouveaux arbitrages.
SUS_ABOVE_ROOT = 5
SUS_ENABLED = False


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

    if SUS_ENABLED:
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
    ce qu'on écrit (c'est `decide_bass`, qui n'appelle pas cette fonction).

    ATTENTION, son socle de preuve s'est effondré le 2026-09-15 au soir.
    L'exception « le retour ne démonte pas une quarte » reposait sur quatre
    arbitrages (c04, c05, c11, c12) ; trois d'entre eux ont le candidat ÉGAL
    à la fondamentale de l'accord et sont donc déjà expliqués par
    `decide_bass` branche 1, et le quatrième (c11) s'est révélé être un
    changement d'accord manqué, pas une basse de slash (Ready mesure 49 :
    Bb-7 puis Eb6b9 — voir `state/human/bass_verdicts.json`, resolutions.o2).
    Rien ne la contredit, mais plus rien ne la soutient seule : l'asymétrie
    qu'elle encode vient de l'autre sens (voir DECORATIVE_ABOVE ci-dessus),
    qui lui est solidement mesuré. À re-arbitrer sur des cas neufs avant de
    s'en servir pour décider quoi que ce soit.
    """
    iv = (int(cand_pc) - int(onset_pc)) % 12
    if iv in (DECORATIVE_ABOVE, DECORATIVE_OCTAVE):
        return True
    if iv == FUNDAMENTAL_ABOVE:
        return False
    return bool(returns)
