"""tests/test_suppression_morceau.py — supprimer ne doit RIEN perdre de sa main.

Louis, 2026-09-17 : « je veux une option pour facilement supprimer une chanson
si j'en veux plus ».

Le contrat épinglé ici est celui qui compte : un morceau supprimé quitte la
bibliothèque, son audio et ses caches régénérables s'en vont, mais **tout ce
qu'il a fait à la main part dans `state/human/corbeille/`** — chart, accords
confirmés, découpage des sections (vérité et brouillon), marque de mesure 1 ou
de tempo, et sa ligne de `chart_meta`.

Pourquoi ce test existe : une annotation faite à la main a déjà été perdue une
fois (2026-08-12) pour avoir vécu dans un dossier ignoré par git, et la
séparation `state/human` / `state/cache` est le correctif. Une suppression qui
efface ces fichiers rouvrirait exactement ce trou, en silence et sans retour.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from harmonia.settings import SETTINGS


@pytest.fixture()
def faux_projet(tmp_path, monkeypatch):
    """Un morceau complet — chart, audio, et tout son travail à la main."""
    reglages = dataclasses.replace(
        SETTINGS,
        charts_dir=tmp_path / "charts",
        audio_dir=tmp_path / "audio",
        human_dir=tmp_path / "human",
        sections_dir=tmp_path / "human" / "sections",
        sections_draft_dir=tmp_path / "human" / "sections_draft",
        annotations_dir=tmp_path / "human" / "annotations",
        marks_dir=tmp_path / "human" / "marks",
        chart_meta_path=tmp_path / "human" / "chart_meta.json",
    )
    from harmonia import annotations as an
    from harmonia.server.routes import library as lib
    monkeypatch.setattr(lib, "SETTINGS", reglages)
    monkeypatch.setattr(lib, "CHARTS_DIR", reglages.charts_dir)
    # C'est `harmonia.annotations` qui possède l'emplacement du sidecar, pas
    # `SETTINGS` : la route le lui demande, donc le test doit le déplacer LÀ.
    monkeypatch.setattr(an, "ANNOT_DIR", reglages.annotations_dir)

    stem, cle = "morceau_essai", "min_morceau_essai"
    for d in (reglages.charts_dir, reglages.audio_dir, reglages.sections_dir,
              reglages.sections_draft_dir, reglages.annotations_dir,
              reglages.marks_dir):
        d.mkdir(parents=True, exist_ok=True)
    (reglages.charts_dir / f"{cle}.json").write_text(json.dumps({
        "title": "Essai", "audio_url": f"/audio/{stem}.m4a",
        "barGrid": [0.0, 1.0, 2.0], "nBars": 2, "video_id": "abc123"}))
    (reglages.audio_dir / f"{stem}.m4a").write_bytes(b"x" * 2048)
    (reglages.annotations_dir / f"{cle}.json").write_text(
        json.dumps({"chords": [{"bar": 1, "root": 0, "q": ""}]}))
    (reglages.sections_dir / f"{stem}.json").write_text(
        json.dumps({"validated": True, "sections": [{"label": "A",
                                                     "b0": 0, "b1": 1}]}))
    (reglages.sections_draft_dir / f"{stem}.json").write_text(
        json.dumps({"validated": False, "sections": []}))
    (reglages.marks_dir / f"{stem}.json").write_text(json.dumps({"bar1": 1.23}))
    reglages.chart_meta_path.write_text(
        json.dumps({cle: {"titre": "Essai"}, "min_autre": {"titre": "Autre"}}))

    from harmonia.server import app as app_mod
    return reglages, cle, stem, app_mod.create_app().test_client()


def test_le_travail_a_la_main_part_en_corbeille_pas_a_la_poubelle(faux_projet):
    reglages, cle, _stem, client = faux_projet
    d = client.delete(f"/api/chart/{cle}").get_json()
    assert d["ok"]

    corbeille = reglages.human_dir / "corbeille" / d["corbeille"]
    dedans = {f.name for f in corbeille.iterdir()}
    assert dedans == {"chart.json", "annotations.json", "sections.json",
                      "sections_brouillon.json", "marque.json",
                      "chart_meta.json"}
    # et le contenu est INTACT, pas un fichier vide qui porte le bon nom
    assert json.loads((corbeille / "annotations.json").read_text())["chords"]
    assert json.loads((corbeille / "sections.json").read_text())["validated"]
    assert json.loads((corbeille / "marque.json").read_text())["bar1"] == 1.23
    # le chart garde de quoi retrouver la vidéo si Louis change d'avis
    assert json.loads((corbeille / "chart.json").read_text())["video_id"]


def test_le_morceau_quitte_vraiment_la_bibliotheque(faux_projet):
    reglages, cle, stem, client = faux_projet
    client.delete(f"/api/chart/{cle}")
    assert not (reglages.charts_dir / f"{cle}.json").exists()
    assert not (reglages.audio_dir / f"{stem}.m4a").exists()
    assert not (reglages.annotations_dir / f"{cle}.json").exists()
    assert not (reglages.sections_dir / f"{stem}.json").exists()
    assert not (reglages.marks_dir / f"{stem}.json").exists()


def test_seule_SA_ligne_de_chart_meta_est_retiree(faux_projet):
    """Un autre morceau ne doit pas perdre ses métadonnées au passage."""
    reglages, cle, _stem, client = faux_projet
    client.delete(f"/api/chart/{cle}")
    reste = json.loads(reglages.chart_meta_path.read_text())
    assert cle not in reste
    assert reste["min_autre"] == {"titre": "Autre"}


def test_supprimer_un_morceau_absent_ne_casse_pas(faux_projet):
    """Le bouton part au bout de cinq secondes ; si le fichier a déjà disparu
    entre-temps, la route répond au lieu de lever."""
    _reglages, _cle, _stem, client = faux_projet
    assert client.delete("/api/chart/min_jamais_vu").get_json()["ok"]


def test_deux_suppressions_ne_se_marchent_pas_dessus(faux_projet):
    """Le dossier de corbeille porte l'horodatage ET le nom : deux morceaux
    supprimés la même seconde ne peuvent pas s'écraser."""
    reglages, cle, _stem, client = faux_projet
    d1 = client.delete(f"/api/chart/{cle}").get_json()
    (reglages.charts_dir / "min_autre.json").write_text(json.dumps(
        {"title": "Autre", "audio_url": "/audio/autre.m4a",
         "barGrid": [0.0, 1.0], "nBars": 1}))
    d2 = client.delete("/api/chart/min_autre").get_json()
    assert d1["corbeille"] != d2["corbeille"]
