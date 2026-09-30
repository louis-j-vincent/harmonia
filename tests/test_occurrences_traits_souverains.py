"""Un trait de Louis n'est jamais mangé par les reprises d'une autre brique.

The Walk, 2026-09-30 (docs/plots/the_walk_decalage.html) : Louis trace intro
1–8, A 9–16, B 25–36. Les traits sont traités dans l'ordre des mesures ; les
reprises de A (boucle de 4, posée décalée à 27, 31, 35) sont placées AVANT
que B ne soit traité, recouvrent son trait, et TOUS les passages de B sont
alors refusés : B disparaissait de la réponse. Or « du moment qu'un humain
annote une section, c'est LA vérité terrain » (2026-09-17) — une reprise
proposée ne peut recouvrir aucun trait, qu'il soit avant ou après.
"""
from __future__ import annotations

from harmonia.sections import simulation as sim


def test_reprise_ne_recouvre_pas_un_trait_posterieur(monkeypatch):
    def faux(chart, b0, b1, audio_dir, label=""):
        if label == "A":      # ses passages + une reprise qui chevauche le trait B
            return [(8, 11, 1.0, 0), (12, 15, 1.0, 0), (26, 29, 0.91, 0), (40, 43, 0.99, 0)], {}
        return [(24, 27, 1.0, 0), (28, 31, 0.9, 0), (32, 35, 0.9, 0)], {}
    monkeypatch.setattr(sim, "cherche_brique", faux)
    chart = {"barGrid": list(range(61))}
    T = lambda a, b, lbl: {"mesure_debut": a + 1, "mesure_fin": b + 1, "label": lbl}  # noqa: E731
    occ, _, _ = sim.occurrences_de_tous(chart, [T(8, 15, "A"), T(24, 35, "B")], None)
    plages = {(o["m0"], o["m1"], o["label"]) for o in occ}
    assert (26, 29, "A") not in plages
    assert {(24, 27, "B"), (28, 31, "B"), (32, 35, "B")} <= plages
    assert (40, 43, "A") in plages
