"""Les pics « suivants » comme veto AU MILIEU d'un bloc de 8, et rien d'autre.

Le profil fusionné retient les pics à ≥ 0,25 de proéminence relative et garde les
six suivants en réserve (`peak_profile.SHOW_EXTRA`). Ces suivants sont trop
imprécis pour servir de frontières dures (49 % de justesse), mais ils sont
exactement la donnée qui manque là où un bloc de 8 recouvre deux sections de 4 :
sur Blue Lights la mesure 49 est un « suivant » à 0,304 de profil, et l'ajouter
seule fait passer le morceau de 0,696 à 0,813 (docs/known_issues.md, 2026-08-12).

La règle testée ici est donc volontairement étroite : un suivant ne peut RIEN
interdire, sauf une chose — qu'un bloc de 8 ait son MILIEU dessus. Les blocs de 4
ne le voient jamais.
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
import harmonia_min.voice_sections as VS  # noqa: E402


def soft_marks(b, source="rest"):
    if source == "rest":
        return list(b["rest"])
    if source == "both":
        return sorted(set(b["hard"]) | set(b["rest"]))
    return []


def midpoint_dirty(p, block, marks, tol=1):
    return block >= 8 and any(abs(m - (p + block // 2)) <= tol for m in marks)


# ── M2 : l'ordre actuel (8 puis 4), le veto de milieu en plus ────────────────

def search_soft(b, hard=None, marks=None, tol=1, on_occ=True,
                passes=((8, VS.THR8), (4, VS.THR4)), tail_unit=OS.TAIL_UNIT,
                **kw):
    hard = b["hard"] if hard is None else hard
    marks = soft_marks(b) if marks is None else marks

    def gate(run, ctx):
        if midpoint_dirty(run["b0"], run["block"], marks, tol):
            return False
        if on_occ:
            run["occ"] = [c for c in run["occ"]
                          if not midpoint_dirty(c, run["block"], marks, tol)]
            if not run["occ"]:
                return False
        return True

    runs, _ = OS._passes(b, hard, list(passes), gate=gate, **kw)
    return OS.assemble(b, runs, hard, tail_unit=tail_unit)


# ── M1 : la concurrence 4/8, le veto de milieu en plus ──────────────────────

def decide_soft(marks, tol=1, on_occ=True):
    def dec(c8, c4, ctx):
        if c8 is not None:
            if midpoint_dirty(c8["b0"], 8, marks, tol):
                c8 = None
            elif on_occ:
                c8["occ"] = [c for c in c8["occ"]
                             if not midpoint_dirty(c, 8, marks, tol)]
                if not c8["occ"]:
                    c8 = None
        return c8 if c8 is not None else c4
    return dec


def search_mixed_soft(b, hard=None, marks=None, tol=1, on_occ=True,
                      tail_unit=OS.TAIL_UNIT, **kw):
    hard = b["hard"] if hard is None else hard
    marks = soft_marks(b) if marks is None else marks
    runs, _ = OM.mixed_runs(b, hard, decide_soft(marks, tol, on_occ), **kw)
    return OS.assemble(b, runs, hard, tail_unit=tail_unit)
