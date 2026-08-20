"""Les deux chemins qui replient un morceau lisent la MÊME loi.

2026-08-20 : `pipeline.analyze_steps` et `refold.refold` lisaient chacun
`HARMONIA_MERGE` avec son propre défaut écrit en dur. Passer la pipeline aux
postérieures a laissé `refold` sur les CQT — This Love, qui a un découpage à la
main et repasse donc par `refold`, est ressorti du recuit avec l'ancienne loi
(`C- F-7 … % G`, sa cadence `A♭` perdue). Ce test interdit la divergence.
"""
import re
from pathlib import Path

import pytest

from harmonia_min.folding import MERGE_DEFAUT, loi_de_merge

REPO = Path(__file__).resolve().parents[1]


def test_le_defaut_est_les_posterieures(monkeypatch):
    monkeypatch.delenv("HARMONIA_MERGE", raising=False)
    assert MERGE_DEFAUT == "mean"
    assert loi_de_merge() == "mean"


@pytest.mark.parametrize("valeur,attendu", [
    ("cqt", "cqt"), ("CQT", "cqt"), (" cqt ", "cqt"),
    ("mean", "mean"), ("off", "mean"), ("", "mean"), ("nimportequoi", "mean"),
])
def test_ce_que_la_variable_demande(monkeypatch, valeur, attendu):
    monkeypatch.setenv("HARMONIA_MERGE", valeur)
    assert loi_de_merge() == attendu


@pytest.mark.parametrize("module", ["pipeline.py", "refold.py"])
def test_personne_ne_relit_la_variable_dans_son_coin(module):
    """Le vrai garde-fou : un défaut en dur ailleurs, et la divergence revient."""
    src = (REPO / "harmonia_min" / module).read_text(encoding="utf-8")
    fautes = re.findall(r'environ\.get\(\s*["\']HARMONIA_MERGE["\']\s*,', src)
    assert not fautes, (f"{module} relit HARMONIA_MERGE avec son propre défaut ; "
                        "passe par folding.loi_de_merge()")
