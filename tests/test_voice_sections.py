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


# ── le départ de la chanson ─────────────────────────────────────────────────
# L'intro dure un nombre PAIR de mesures, ou exactement une (2026-08-08). Les
# deux morceaux ci-dessous sont les cas réels qu'aucun seuil de phase ne pouvait
# départager : 0,29 voulait décaler, 0,31 voulait rester.

from harmonia_min.voice_sections import sung_start                  # noqa: E402


def _song(first_sung, phase, n=40, bar=2.0):
    """Une grille régulière, du silence jusqu'à `first_sung`, une note à
    `phase` de cette mesure-là."""
    grid = [i * bar for i in range(n + 1)]
    mute = np.ones(n, bool)
    mute[first_sung:] = False
    notes = [(grid[first_sung] + phase * bar, 0.5, 60)]
    return notes, grid, mute


def test_intro_impaire_interdite_vers_le_haut():
    """Every Breath You Take : le chant entre mesure 7, tôt dans la mesure
    (0,29). La phase seule garderait 7 ; une intro de 7 mesures n'existe pas."""
    assert sung_start(*_song(7, 0.29)) == 8


def test_intro_impaire_interdite_vers_le_bas():
    """Happy : le chant entre mesure 2 à mi-mesure (0,50). La phase seule
    pousserait à 3 ; une intro de 3 mesures n'existe pas non plus."""
    assert sung_start(*_song(2, 0.50)) == 2


def test_la_phase_tranche_encore_quand_les_deux_sont_admissibles():
    """Seul cas où elle sert : le chant entre mesure 0 ou 1, et l'intro d'une
    seule mesure est admissible. Chain of Fools (0,48) contre Grenade (0,08)."""
    assert sung_start(*_song(0, 0.48)) == 1
    assert sung_start(*_song(0, 0.08)) == 0


def test_lintro_dune_seule_mesure_reste_possible():
    """Sunny : le chant entre mesure 1 dès le début (0,01). Il ne faut ni la
    pousser à 2 ni la ramener à 0."""
    assert sung_start(*_song(1, 0.01)) == 1


def test_pas_dintro_du_tout():
    assert sung_start(*_song(0, 0.05)) == 0
