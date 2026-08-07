"""harmonia_min.voice_sections — la mise en forme des sections trouvées.

Deux comportements sont épinglés ici, tous deux corrigés le 2026-08-08 contre
les dix-sept morceaux annotés par Louis :

  * deux occurrences ADJACENTES d'un même bloc restent deux sections ;
  * deux lettres qui désignent la même musique n'en font qu'une.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia_min.voice_sections import merge_letters


def _ssm(n, groups, hi=0.99, lo=0.30):
    """Une SSM jouet : `hi` entre mesures du même groupe, `lo` sinon."""
    g = np.zeros(n, int)
    for k, (a, b) in enumerate(groups):
        g[a:b + 1] = k
    S = np.full((n, n), lo)
    for i in range(n):
        for j in range(n):
            if g[i] == g[j]:
                S[i, j] = hi
    return S


def _secs(spec):
    out = []
    for tok in spec.split():
        lab, rng = tok.split(":")
        a, b = rng.split("-")
        out.append({"b0": int(a), "b1": int(b), "label": lab})
    return out


def test_deux_lettres_pour_la_meme_musique_fusionnent():
    """Le cas Chain of Fools : un seul vamp, cinq lettres. Toutes les mesures
    se ressemblent, donc tout doit finir sous une seule lettre."""
    S = _ssm(32, [(0, 31)])
    out = merge_letters(S, _secs("A:0-7 B:8-15 C:16-23 D:24-31"))
    assert len({s["label"] for s in out}) == 1


def test_deux_musiques_differentes_ne_fusionnent_pas():
    S = _ssm(32, [(0, 7), (8, 15), (16, 23), (24, 31)])
    out = merge_letters(S, _secs("A:0-7 B:8-15 C:16-23 D:24-31"))
    assert len({s["label"] for s in out}) == 4


def test_liaison_moyenne_et_non_simple():
    """UNE paire de sections qui se ressemble ne doit pas recoller deux lettres
    qui, en moyenne, n'ont rien à voir. C'est l'enchaînement du lien simple, et
    il avait cassé This Love (annotation exacte -> 0,947)."""
    # A joue en 0-7 et 8-15 ; B en 16-23 et 24-31. Seules A[8-15] et B[16-23]
    # se ressemblent : une paire sur quatre.
    S = _ssm(32, [(0, 7), (8, 23), (24, 31)])
    out = merge_letters(S, _secs("A:0-7 A:8-15 B:16-23 B:24-31"))
    assert len({s["label"] for s in out}) == 2


def test_intro_et_outro_ne_sont_jamais_fusionnees():
    """Elles ne sont pas des lettres : les recoller à une section les ferait
    disparaître de la grille."""
    S = _ssm(32, [(0, 31)])
    out = merge_letters(S, _secs("intro:0-7 A:8-15 A:16-23 outro:24-31"))
    assert [s["label"] for s in out] == ["intro", "A", "A", "outro"]


def test_les_sections_trop_courtes_ne_decident_de_rien():
    """Sur deux mesures, une diagonale de similarité ne veut rien dire ; une
    telle section ne doit ni fusionner ni empêcher de fusionner."""
    S = _ssm(32, [(0, 31)])
    out = merge_letters(S, _secs("A:0-1 B:2-3"))
    assert len({s["label"] for s in out}) == 2


def test_les_lettres_restent_dans_lordre_dapparition():
    S = _ssm(32, [(0, 7), (8, 15), (16, 23), (24, 31)])
    out = merge_letters(S, _secs("A:0-7 B:8-15 C:16-23 D:24-31"))
    assert [s["label"] for s in out] == ["A", "B", "C", "D"]
