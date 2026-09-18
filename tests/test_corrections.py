"""tests/test_corrections.py — le registre de ce que Louis corrige.

Louis, 2026-09-18, avant deux semaines d'import et de correction : « je veux
que tu fasses quelque chose qui récupère automatiquement toutes les différences
d'annotation et les persiste quelque part, afin que quand on a récolté assez de
corrections un agent dédié puisse apprendre de mes corrections ».

Le contrat épinglé ici : une correction est notée avec SON AVANT, un accord
confirmé à l'identique ne l'est pas, et le registre ne fait jamais échouer une
sauvegarde — son annotation est le travail, la ligne n'est qu'une trace.
"""
from __future__ import annotations

import json

import pytest

from harmonia import corrections


@pytest.fixture(autouse=True)
def registre_temporaire(tmp_path, monkeypatch):
    monkeypatch.setattr(corrections, "DOSSIER", tmp_path / "corrections")
    return tmp_path / "corrections"


def _chart():
    """Deux mesures, deux accords : C puis G, avec confiance et suggestions."""
    return {
        "key": {"tonic": 0, "mode": "major"},
        "meta": {"bpm": 120.0},
        "sections": [{"label": "A", "barRanges": [[0, 1]], "bars": [
            [{"bar": 0, "beat": 0, "root": 0, "q": "", "bass": -1, "nc": False,
              "c": 0.94, "t0": 0.0, "t1": 2.0,
              "sug": [{"root": 0, "q": "", "c": 0.94},
                      {"root": 9, "q": "min", "c": 0.31}]}],
            [{"bar": 1, "beat": 0, "root": 7, "q": "", "bass": -1, "nc": False,
              "c": 0.81, "t0": 2.0, "t1": 4.0, "sug": []}]]}],
    }


def test_une_correction_garde_ce_que_la_machine_disait():
    """Le cœur : « il a mis Fm » n'apprend rien, « la machine disait C à 0,94,
    ses suggestions étaient C puis Am, il a écrit Fm » est une leçon."""
    n = corrections.note_corrections_accords(
        "min_essai", _chart(),
        [{"bar": 0, "beat": 0, "root": 5, "q": "min", "bass": -1}])
    assert n == 1
    (ligne,) = corrections.lire("min_essai")
    assert ligne["quoi"] == "accord" and ligne["mesure"] == 0
    assert ligne["avant"]["texte"] == "C"
    assert ligne["avant"]["confiance"] == 0.94
    assert len(ligne["avant"]["suggestions"]) == 2
    assert ligne["apres"]["texte"] == "Fm"
    assert ligne["contexte"]["suivant"] == "G"
    assert ligne["contexte"]["section"] == "A"
    assert ligne["quand"]


def test_un_accord_confirme_a_l_identique_n_est_pas_une_correction():
    """Il confirme beaucoup d'accords justes. Les noter noierait les vraies
    corrections et n'apprendrait rien sur les erreurs de la machine."""
    n = corrections.note_corrections_accords(
        "min_essai", _chart(),
        [{"bar": 0, "beat": 0, "root": 0, "q": "", "bass": -1}])
    assert n == 0 and corrections.lire("min_essai") == []


def test_changer_la_SEULE_basse_compte_comme_une_correction():
    """C → C/G est une correction, et c'est même la famille la plus utile :
    le projet a redéfini sa cible sur la basse qui SONNE."""
    n = corrections.note_corrections_accords(
        "min_essai", _chart(),
        [{"bar": 0, "beat": 0, "root": 0, "q": "", "bass": 7}])
    assert n == 1
    assert corrections.lire("min_essai")[0]["apres"]["texte"] == "C/G"


def test_le_registre_est_en_AJOUT_SEUL():
    """Une correction refaite trois fois laisse trois lignes — il a hésité, et
    c'est une information. Rien n'est jamais réécrit ni compacté."""
    for root in (5, 2, 9):
        corrections.note_corrections_accords(
            "min_essai", _chart(),
            [{"bar": 0, "beat": 0, "root": root, "q": "min", "bass": -1}])
    lignes = corrections.lire("min_essai")
    assert [l["apres"]["root"] for l in lignes] == [5, 2, 9]


def test_une_ligne_illisible_ne_fait_pas_perdre_le_reste(registre_temporaire):
    """Un fichier en ajout seul peut porter une écriture interrompue."""
    registre_temporaire.mkdir(parents=True, exist_ok=True)
    f = registre_temporaire / "min_essai.jsonl"
    f.write_text('{"quoi":"accord","morceau":"min_essai"}\n{coupé\n'
                 '{"quoi":"accord","morceau":"min_essai","mesure":9}\n')
    lignes = corrections.lire("min_essai")
    assert len(lignes) == 2 and lignes[-1]["mesure"] == 9


def test_les_sections_sont_notees_par_plage_qui_change():
    """Une ligne par plage contiguë dont l'étiquette bouge — la forme qui se
    lit et se compte, alors qu'une ligne par section rendrait incomparables
    deux découpages aux frontières différentes."""
    avant = [{"label": "A", "barRanges": [[0, 7]]},
             {"label": "B", "barRanges": [[8, 15]]}]
    apres = [{"label": "intro", "barRanges": [[0, 3]]},
             {"label": "A", "barRanges": [[4, 15]]}]
    n = corrections.note_corrections_sections("min_essai", avant, apres, 16)
    lignes = corrections.lire("min_essai")
    assert n == 2
    assert [(l["mesure_debut"], l["mesure_fin"], l["avant"], l["apres"])
            for l in lignes] == [(1, 4, "A", "intro"), (9, 16, "B", "A")]


def test_un_decoupage_inchange_ne_note_rien():
    secs = [{"label": "A", "barRanges": [[0, 7]]}]
    assert corrections.note_corrections_sections("min_essai", secs, secs, 8) == 0


def test_le_registre_ne_fait_jamais_echouer_l_appelant(monkeypatch):
    """Son annotation est le travail ; la ligne n'est qu'une trace. Un disque
    plein ne doit pas lui faire perdre une correction faite à l'oreille."""
    def casse(*_a, **_k):
        raise OSError("disque plein")
    monkeypatch.setattr(corrections.Path, "open", casse, raising=False)
    monkeypatch.setattr(corrections, "_fichier", lambda s: corrections.Path("/x/y"))
    assert corrections.noter("min_essai", "accord", {"a": 1}) is False


def test_compter_repond_a_en_a_t_on_assez():
    corrections.note_corrections_accords(
        "min_essai", _chart(), [{"bar": 0, "beat": 0, "root": 5, "q": "min"}])
    corrections.note_corrections_sections(
        "min_autre", [{"label": "A", "barRanges": [[0, 3]]}],
        [{"label": "B", "barRanges": [[0, 3]]}], 4)
    c = corrections.compter()
    assert c["total"] == 2 and c["morceaux"] == 2
    assert c["par_type"] == {"accord": 1, "section": 1}
