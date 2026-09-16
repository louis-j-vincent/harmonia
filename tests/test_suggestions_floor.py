"""Les deux listes de candidats que l'éditeur d'annotation affiche, et le
plancher qui décide de leur longueur (2026-09-16).

Louis : « je veux qu'il apparaisse les autres suggestions que juste le top 2,
plus les suggestions sur la ligne de basse […] top 3 basses, top 5 accords si
relevant. » Le « si relevant » est un PLANCHER, pas un remplissage : ces tests
gèlent les deux lois, et surtout le fait que les deux planchers de basse qui
cohabitent dans le projet ne répondent pas à la même question.

    .venv/bin/python -m pytest tests/test_suggestions_floor.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.nnls_features import _ROLL_TO_C
from harmonia.span_rescore import (BASS_SUG_FLOOR, SUG_FLOOR, bass_suggestions,
                                   musx_suggestions)

T = 100  # trames (~2,3 s)


# ── accords ──────────────────────────────────────────────────────────────────

def _probs(tail_mass: float = 0.05):
    """C:maj, A:min, et `tail_mass` étalée sur les 70 autres colonnes.

    Rappel de disposition (span_rescore.py) : triad col 0 = N, col i>=1 porte
    la racine (i-1)%12 VITE et le type (i-1)//12 LENT sur ("maj","min","sus4",
    "sus2","dim","aug") ; s7 = (aucune, maj7, b7, bb7).
    """
    triad = np.full((T, 73), tail_mass / 70)
    triad[:, 0] = 0.05                # N
    triad[:, 1] = 0.60                # racine C, type maj
    triad[:, 1 + 12 + 9] = 0.30       # racine A, type min
    s7 = np.tile([0.90, 0.02, 0.08, 0.0], (T, 1))
    dummy = np.zeros((T, 3))
    return [triad, np.zeros((T, 13)), s7, np.zeros((T, 4)), dummy, dummy]


def _probs_ambigus():
    """Cinq lectures réellement en lice — un accord sur lequel musx hésite,
    le cas que Louis veut voir en entier. C maj .30, A min .22, F maj .16,
    G maj .12, D min .08, N .05, le reste étalé sous le plancher."""
    triad = np.full((T, 73), 0.07 / 68)
    triad[:, 0] = 0.05                       # N
    triad[:, 1] = 0.30                       # C maj  (racine 0, type maj)
    triad[:, 1 + 12 + 9] = 0.22              # A min
    triad[:, 1 + 5] = 0.16                   # F maj
    triad[:, 1 + 7] = 0.12                   # G maj
    triad[:, 1 + 12 + 2] = 0.08              # D min
    s7 = np.tile([0.90, 0.02, 0.08, 0.0], (T, 1))
    dummy = np.zeros((T, 3))
    return [triad, np.zeros((T, 13)), s7, np.zeros((T, 4)), dummy, dummy]


def _one_chord():
    return [{"root": 0, "q": "", "nc": False, "t0": 0.5, "t1": 1.5}]


def test_le_plancher_coupe_la_queue_plutot_que_de_remplir_jusqua_cinq():
    """Avec une queue négligeable (0,0007 par case), le 4e et le 5e candidat
    sont sous le plancher : la liste s'arrête à 3 au lieu d'être rembourrée."""
    chords = _one_chord()
    musx_suggestions(_probs(tail_mass=0.05), chords)
    sug = chords[0]["sug"]
    assert len(sug) == 3
    assert all(s["c"] >= SUG_FLOOR for s in sug[1:])


def test_cinq_candidats_quand_cinq_passent_le_plancher():
    """Cinq lectures au-dessus de 2 % : on les montre toutes les cinq, là où
    l'ancien top-3 figé en cachait deux."""
    chords = _one_chord()
    musx_suggestions(_probs_ambigus(), chords)
    sug = chords[0]["sug"]
    assert len(sug) == 5
    assert all(s["c"] >= SUG_FLOOR for s in sug[1:])
    assert all(a["c"] >= b["c"] for a, b in zip(sug, sug[1:]))


def test_le_top_1_survit_toujours_au_plancher():
    """Un accord dont MÊME le meilleur candidat serait sous le plancher doit
    quand même ouvrir l'éditeur sur quelque chose — sinon la régression est
    pire que l'ancien comportement (l'éditeur bascule sur « By hand »)."""
    chords = _one_chord()
    musx_suggestions(_probs(), chords, floor=0.99)
    assert len(chords[0]["sug"]) == 1


def test_un_accord_tres_sur_ne_garde_que_sa_propre_case():
    """La conséquence que le rendu doit savoir gérer. Avant le plancher, `sug`
    portait toujours trois entrées, donc le compas — qui RETIRE l'accord écrit
    avant de placer ses orbes — en gardait au moins deux. Depuis le plancher,
    un accord que musx tient à 95 % ne laisse que SA case, et la roue se
    vide : c'est « le modèle est sûr », jamais « ce chart n'a pas de
    classement ». Mesuré sur les charts réels : 2 accords sur 23 (Yesterday),
    2 sur 86 (Lost Without U). `buildCompass` distingue les deux silences ;
    ce test gèle la situation qui l'y oblige."""
    triad = np.full((T, 73), 0.001 / 70)
    triad[:, 0] = 0.001
    triad[:, 1] = 0.998                       # C maj, écrasant
    s7 = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    dummy = np.zeros((T, 3))
    probs = [triad, np.zeros((T, 13)), s7, np.zeros((T, 4)), dummy, dummy]
    chords = [{"root": 0, "q": "", "nc": False, "t0": 0.0, "t1": 2.0}]
    musx_suggestions(probs, chords)
    sug = chords[0]["sug"]
    assert len(sug) == 1
    assert (sug[0]["root"], sug[0]["q"]) == (0, "")     # sa propre case
    others = [s for s in sug if not (s["root"] == 0 and s["q"] == "")]
    assert others == []                                 # le compas n'a rien à orbiter


def test_le_defaut_est_cinq_pas_trois():
    """Le top_k par défaut a bougé le 2026-09-16 ; c'est le défaut qui compte,
    puisque `pipeline.py` appelle sans argument."""
    import inspect
    assert inspect.signature(musx_suggestions).parameters["top_k"].default == 5


# ── basse ────────────────────────────────────────────────────────────────────

def _bothchroma(shares: dict[int, float], n_frames: int = 200, dt: float = 0.05):
    """(arr, times) synthétique : la moitié grave porte `shares`, une part par
    classe de hauteur (index 0 = C APRÈS le roulement du module)."""
    times = np.arange(n_frames) * dt
    arr = np.zeros((n_frames, 24), dtype=np.float32)
    for pc, v in shares.items():
        arr[:, (pc - _ROLL_TO_C) % 12] = v
    return arr, times


def _span(**kw):
    c = {"root": 0, "q": "", "nc": False, "t0": 0.0, "t1": 2.0}
    c.update(kw)
    return c


def test_trois_basses_triees_par_part_denergie():
    arr, times = _bothchroma({0: 0.50, 7: 0.25, 4: 0.15, 2: 0.10})
    chords = [_span()]
    assert bass_suggestions(arr, times, chords) == 1
    sug = chords[0]["sugBass"]
    assert [s["pc"] for s in sug] == [0, 7, 4]          # top-3, la 4e tombe
    assert sug[0]["c"] == pytest.approx(0.50, abs=0.01)
    assert all(a["c"] >= b["c"] for a, b in zip(sug, sug[1:]))


def test_le_plancher_de_basse_coupe_les_lectures_faibles():
    """Une basse franche : la 2e et la 3e classe sont du bruit sous 12,5 %,
    on n'écrit pas trois cercles pour faire joli."""
    arr, times = _bothchroma({0: 0.90, 7: 0.06, 4: 0.04})
    chords = [_span()]
    bass_suggestions(arr, times, chords)
    sug = chords[0]["sugBass"]
    assert len(sug) == 1 and sug[0]["pc"] == 0
    arr2, times2 = _bothchroma({0: 0.50, 7: 0.30, 4: 0.20})
    chords2 = [_span()]
    bass_suggestions(arr2, times2, chords2)
    assert len(chords2[0]["sugBass"]) == 3
    assert all(s["c"] >= BASS_SUG_FLOOR for s in chords2[0]["sugBass"][1:])


def test_la_basse_nc_les_carry_et_les_spans_casses_sont_sautes():
    arr, times = _bothchroma({0: 0.6, 7: 0.4})
    chords = [_span(nc=True), _span(carry=True), {"root": 0, "q": ""}, _span()]
    assert bass_suggestions(arr, times, chords) == 1
    assert all("sugBass" not in c for c in chords[:3])
    assert "sugBass" in chords[3]


def test_montrer_une_basse_nest_pas_lecrire():
    """LE point de la fonction : aucun slash n'est posé. `bass` reste tel
    quel, donc le chart ne change pas et le banc corpus que known_issues
    exige avant de DÉCIDER une basse n'est pas court-circuité."""
    arr, times = _bothchroma({0: 0.45, 7: 0.35, 4: 0.20})
    chords = [_span(bass=-1)]
    bass_suggestions(arr, times, chords)
    assert chords[0]["bass"] == -1
    assert chords[0]["root"] == 0 and chords[0]["q"] == ""


def test_les_deux_planchers_de_basse_restent_distincts():
    """Garde-fou contre l'erreur de calibration silencieuse (CLAUDE.md #1) :
    `bass_rules.FLOOR` (30 %, en POURCENTS, décide d'écrire un slash) et
    `BASS_SUG_FLOOR` (0,125, en PART, décide d'afficher une lecture) ne sont
    ni la même échelle ni la même question. Si un jour quelqu'un les aligne,
    ce test doit rougir avant l'affichage."""
    from harmonia.bass_rules import FLOOR as DECISION_FLOOR
    assert DECISION_FLOOR == 30.0                     # pourcents
    assert 0.0 < BASS_SUG_FLOOR < 1.0                 # part
    assert BASS_SUG_FLOOR * 100 < DECISION_FLOOR      # montrer < écrire
    assert BASS_SUG_FLOOR > 1 / 12                    # et > l'uniforme
