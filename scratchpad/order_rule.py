"""La règle : un bloc de 4 double la file d'attente s'il se REJOUE À L'IDENTIQUE.

Le parcours est unique (les deux longueurs proposées à chaque ancre). Le bloc de
8 garde la priorité. Quand il n'y en a pas — parce qu'un pic dur le barre, ou
parce qu'il n'a pas de reprise — le bloc de 4 n'est posé TOUT DE SUITE que si ses
reprises valent au moins `LITERAL` fois l'ancre elle-même. Sinon il est différé à
la passe de comblement, exactement comme aujourd'hui.

Pourquoi ce critère et pas un seuil de détection : `block_score` est déjà relatif
à l'ancre (mesuré : mêmes échelles à 4 et à 8 mesures), donc 0,95 veut dire
« cette reprise est presque aussi proche de l'ancre que l'ancre l'est d'elle-même ».
C'est la définition d'un motif rejoué littéralement — le B de Blue Lights, le
refrain-crochet de 4 mesures. Un bloc de 4 à 0,75 est simplement la boucle
harmonique du morceau, et c'est elle qu'il ne faut pas laisser passer devant.
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

import order_search as OS                 # noqa: E402
import order_mixed as OM                  # noqa: E402

LITERAL = 0.95        # ce qu'une reprise de 4 mesures doit valoir pour passer
                      # devant ; plateau mesuré 0,92–1,00 (voir le journal)
STAT = "moy"          # "moy" = moyenne des reprises, "min" = la plus faible


def decide_literal(literal=LITERAL, stat=STAT, beat8=False):
    def dec(c8, c4, ctx):
        lit = None
        if c4 is not None:
            v = [c4["sc"][p] for p in c4["occ"]]
            m = float(np.mean(v)) if stat == "moy" else float(np.min(v))
            lit = c4 if m >= literal else None
        if c8 is not None:
            return lit if (beat8 and lit is not None) else c8
        return lit
    return dec


def sections(b, literal=LITERAL, stat=STAT, beat8=False, hard=None, **kw):
    return OM.sections(b, decide_literal(literal, stat, beat8), hard=hard,
                       stride=4, **kw)
