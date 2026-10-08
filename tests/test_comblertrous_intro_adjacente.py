"""tests/test_comblertrous_intro_adjacente.py — un trou de tête ne mange pas
l'intro que Louis a tracée plus loin.

Louis, 2026-10-08, sur « La Jupe En Laine » : il a tracé son intro à partir du
Fm7 (mesure 2), PAS sur le N.C. de la mesure 1 — et pourtant ce N.C. s'est
retrouvé ajouté à son intro après « Valider les sections ».

LE BUG (`comblerTrous`, `harmonia/static/screens/sections_editor.js`) : le
trou de tête (mesure 1, sans section) est TOUJOURS nommé « intro » quand il
commence à la mesure 1 (`base=(d===1)?"intro":"queue"`), même si la vraie
section qui le suit immédiatement s'appelle déjà « intro ». Les deux rangent
alors sous le même label, adjacentes, et `sections_et_fusions` les restitue
comme UNE section à deux occurrences touchantes ([[1,16],[0,0]]) — qui se lit
ensuite comme un intro continu de la mesure 0 à 16.

Ce test n'exerce pas `comblerTrous` elle-même (JS, pas de runner dans ce
projet) mais verrouille le CONTRAT dont elle dépend côté serveur : deux
sections de labels différents, même adjacentes, ne doivent jamais se
retrouver fusionnées sous un même label par `sections_et_fusions`. Le vrai
correctif (`sections_editor.js`) utilise ce contrat en donnant au trou de tête
un nom DIFFÉRENT (« queue ») dès que la section qui le suit est déjà « intro ».
"""
from __future__ import annotations

from harmonia.soudure import scinder_multiples, sections_et_fusions
from harmonia.refold import refold


def _chart_minimal(n_mesures: int) -> dict:
    grid = [float(i) * 2.0 for i in range(n_mesures + 1)]
    chords = [{"t0": grid[i], "t1": grid[i + 1], "root": i % 4, "q": "",
               "bass": -1, "nc": i == 0} for i in range(n_mesures)]
    return {"title": "bidon", "barGrid": grid, "nBars": n_mesures,
            "prompter": {"chords": chords}, "audio_url": "/audio/absent.m4a",
            "file": "min_bidon", "sections": []}


def test_le_trou_de_tete_nomme_queue_ne_fusionne_pas_avec_lintro(tmp_path, monkeypatch):
    """Le comportement voulu APRÈS le correctif de `comblerTrous` : le trou
    de la mesure 1 sort nommé « queue » (pas « intro »), parce que la vraie
    intro commence juste après. Les deux doivent rester deux sections
    distinctes, jamais une « intro » étendue en douce à la mesure 0."""
    chart = _chart_minimal(20)
    (tmp_path / "absent.m4a").write_bytes(b"")
    secs = scinder_multiples([
        {"label": "queue", "mesure_debut": 1, "mesure_fin": 1},
        {"label": "intro", "mesure_debut": 2, "mesure_fin": 9},
        {"label": "A", "mesure_debut": 10, "mesure_fin": 17},
        {"label": "A", "mesure_debut": 18, "mesure_fin": 20},
    ])
    bars, rap = refold(chart, secs, tmp_path)
    neuves, _fusions = sections_et_fusions(chart, secs, bars=bars,
                                           fold_report=rap.get("rapport"))
    intros = [s for s in neuves if s["label"] == "intro"]
    assert len(intros) == 1, neuves
    assert intros[0]["barRanges"] == [[1, 8]], intros[0]
    queues = [s for s in neuves if s["label"] == "queue"]
    assert queues and queues[0]["barRanges"] == [[0, 0]], neuves


def test_reproduction_du_bug_si_le_trou_est_nomme_intro(tmp_path):
    """Documente le bug AVANT le correctif, pour qu'une régression du JS
    (redonner « intro » au trou de tête) se voie ici aussi : deux sections
    « intro » adjacentes se lisent comme une seule, mesure 0 à 8."""
    chart = _chart_minimal(20)
    (tmp_path / "absent.m4a").write_bytes(b"")
    secs = scinder_multiples([
        {"label": "intro", "mesure_debut": 1, "mesure_fin": 1},
        {"label": "intro", "mesure_debut": 2, "mesure_fin": 9},
        {"label": "A", "mesure_debut": 10, "mesure_fin": 17},
        {"label": "A", "mesure_debut": 18, "mesure_fin": 20},
    ])
    bars, rap = refold(chart, secs, tmp_path)
    neuves, _fusions = sections_et_fusions(chart, secs, bars=bars,
                                           fold_report=rap.get("rapport"))
    intros = [s for s in neuves if s["label"] == "intro"]
    assert len(intros) == 1
    ranges = sorted(tuple(r) for r in intros[0]["barRanges"])
    assert ranges == [(0, 0), (1, 8)], (
        "si ceci casse, c'est que sections_et_fusions ne fusionne plus les "
        "labels adjacents — comblerTrous pourrait alors redevenir plus "
        "simple, mais vérifier d'abord pourquoi ce test a changé")
