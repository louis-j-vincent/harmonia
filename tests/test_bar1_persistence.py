"""tests/test_bar1_persistence.py — « Set bar 1 » doit SURVIVRE à la
ré-inférence. Rouge d'abord.

Louis, 2026-08-13 : « lorsque je recale la première barre, j'ai l'impression
que le recalage se fait mal lorsque le chart est réinféré, comme s'il y avait
un décalage ».

Ce n'est pas la phase qui se calcule mal — mesuré sur Virtual Insanity, la
mesure marquée commence à 0 ms du temps le plus proche de la marque, avec ou
sans décalage volontaire. C'est que la marque **n'était écrite nulle part** :

* le ChartModel ne portait aucun champ `bar1` ;
* `pipeline.analyze()` — la version non-générateur — n'acceptait même pas
  l'argument `bar1_time` ;
* `scripts/rebake_library.py` appelle `analyze()` sans la marque.

Or le rebake est une étape prescrite de `/ship` : « une loi de merge, un
moteur de sections, un changement de chart réécrivent tous les charts servis ».
Chaque rebake rendait donc à la pipeline la phase du tracker et effaçait
silencieusement TOUS les « Set bar 1 » jamais posés. Le décalage que voit
Louis, c'est sa marque qui a disparu.

Le contrat épinglé ici :
* le modèle porte la marque, en secondes, telle qu'elle a été posée ;
* `analyze()` la prend et la transmet ;
* la cuisson (`tools.golden.cuire`, moteur `harmonia_min` — ce que
  `scripts/rebake_library.py` faisait avant d'être remplacé par
  `tools.golden --publish` au sprint 20) relit la marque du chart existant et
  la repasse.
"""
from __future__ import annotations

import inspect
import json

import harmonia.pipeline as _pipeline_impl
import tools.golden as golden
from harmonia_min import pipeline


def test_analyze_accepte_la_marque():
    """`analyze()` doit pouvoir porter la marque — sinon aucun appelant
    non-streaming (le rebake, les scripts) ne peut la respecter."""
    sig = inspect.signature(pipeline.analyze)
    assert "bar1_time" in sig.parameters, (
        "analyze() n'accepte pas bar1_time : tout ré-calcul non-streaming "
        "perd le recalage de Louis")


def test_le_modele_porte_la_marque(monkeypatch):
    """Le ChartModel doit écrire `bar1`, sinon rien ne peut la relire.

    `pipeline.analyze` (le pont `harmonia_min.pipeline`) est le MÊME objet
    fonction que `harmonia.pipeline.analyze` — mais son corps résout
    `analyze_steps` dans les globals du module où il est DÉFINI
    (`harmonia.pipeline`), pas dans ceux du pont. Patcher
    `harmonia_min.pipeline.analyze_steps` ne serait donc jamais vu par
    `analyze()` depuis le refactor (sprint 10, `pipeline.py` a déménagé sous
    `harmonia/`) — le patch vise le module réel.
    """
    vu = {}

    def faux_steps(audio_path, **kw):
        vu.update(kw)
        yield "final", {"file": "min_x", "bar1": kw.get("bar1_time")}

    monkeypatch.setattr(_pipeline_impl, "analyze_steps", faux_steps)
    m = pipeline.analyze("/tmp/x.m4a", bar1_time=12.5)
    assert vu["bar1_time"] == 12.5
    assert m["bar1"] == 12.5


def test_le_golden_cuire_repasse_la_marque(tmp_path, monkeypatch):
    """La cuisson (moteur `harmonia_min`) doit relire la marque du chart
    existant et la repasser.

    C'est le chemin qui efface réellement : `/ship` prescrit `tools.golden
    --publish` dès qu'une loi de repli change, et il recuit toute la
    bibliothèque."""
    recu = {}

    def faux_analyze(audio, **kw):
        recu.update(kw)
        return {"file": kw.get("file_key"), "bar1": kw.get("bar1_time"),
                "nBars": 1, "barGrid": [0.0, 2.0],
                "sections": [{"label": "A"}], "fold": {}}

    monkeypatch.setattr("harmonia_min.pipeline.analyze", faux_analyze)
    # Le rapport d'or refuse de lancer un modèle sur un cache froid — hors
    # sujet ici, on épingle seulement le passage de la marque.
    monkeypatch.setattr(golden, "caches_manquants", lambda eng, stem, audio: [])
    monkeypatch.setattr(golden, "AUDIO", tmp_path / "audio")

    (tmp_path / "charts").mkdir()
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "bar1_persistence_stem.m4a").write_bytes(b"\0")
    (tmp_path / "charts" / "min_bar1_persistence_stem.json").write_text(json.dumps(
        {"file": "min_bar1_persistence_stem", "title": "T",
         "audio_url": "/audio/bar1_persistence_stem.m4a", "bar1": 7.25}),
        encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    eng = golden.charger_moteur("harmonia_min")
    r = golden.cuire(eng, "min_bar1_persistence_stem", tmp_path / "charts", out)
    assert r["status"] == "ok", r
    assert recu.get("bar1_time") == 7.25, (
        "la cuisson a rejoué la pipeline sans la marque : le « Set bar 1 » "
        "de Louis serait effacé à chaque publication")
