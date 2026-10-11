"""tests/test_explication_sections.py — l'écran « Pourquoi ? » explique LE
calcul de Valider, pas un autre.

Louis, 2026-10-11 : l'explication doit marcher « automatiquement sur toutes
les chansons », calculée en direct. La garantie qui compte : pour les mêmes
traits, `/api/sections/expliquer` rend exactement les sections que
`/api/sections/inferer` (le bouton « Valider les sections »), et il tourne
sans exception. On le vérifie sur La Jupe En Laine (brique B 18–25, le fil
rouge du plan) et sur CHAQUE morceau annoté à la main
(`state/human/sections/*.json`, ses sections servant de traits).

Les charts et l'audio vivent dans `state/cache/` (non suivi) : un morceau
absent de cette machine est sauté, pas compté comme réussi.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from harmonia.settings import SETTINGS


def _client():
    from harmonia.server import app as app_mod
    return app_mod.create_app().test_client()


def _chart_et_audio(stem: str):
    p = SETTINGS.charts_dir / f"min_{stem}.json"
    if not p.exists():
        return None
    ch = json.loads(p.read_text(encoding="utf-8"))
    st = Path(ch.get("audio_url") or "").stem
    if not st or not (SETTINGS.audio_dir / f"{st}.m4a").exists():
        return None
    return p.stem


def _cas():
    out = [("ujgPfUZ7Whc", [{"label": "B", "mesure_debut": 18, "mesure_fin": 25}])]
    d = SETTINGS.sections_dir
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        humain = [{"label": s["label"], "mesure_debut": int(s["b0"]) + 1,
                   "mesure_fin": int(s["b1"]) + 1}
                  for s in doc.get("sections") or []
                  if isinstance(s, dict) and "b0" in s and "b1" in s and s.get("label")]
        if humain:
            out.append((f.stem, humain))
    return out


CAS = _cas()


@pytest.mark.parametrize("stem,humain", CAS, ids=[c[0] for c in CAS])
def test_expliquer_rend_exactement_valider(stem, humain):
    fichier = _chart_et_audio(stem)
    if fichier is None:
        pytest.skip(f"pas de chart/audio pour {stem} sur cette machine")
    c = _client()
    r1 = c.post(f"/api/sections/inferer/{fichier}", json={"humain": humain})
    assert r1.status_code == 200
    t = time.perf_counter()
    r2 = c.post(f"/api/sections/expliquer/{fichier}", json={"humain": humain})
    duree = time.perf_counter() - t
    assert r2.status_code == 200, r2.get_json()
    d = r2.get_json()
    assert d["valider"] == r1.get_json()
    # chaque chapitre est là
    for k in ("accords", "S", "briques", "arbitrage", "mot", "soudures",
              "nommage", "ecriture", "pales"):
        assert k in d
    assert len(d["accords"]) == d["n"] == len(d["S"])
    print(f"\n{stem}: {duree:.2f} s")


def test_la_jupe_les_chiffres_de_la_carte():
    """Les valeurs relevées à la main dans docs/sections_pipeline_carte_2026-10-11.md."""
    fichier = _chart_et_audio("ujgPfUZ7Whc")
    if fichier is None:
        pytest.skip("La Jupe absente de cette machine")
    d = _client().post(f"/api/sections/expliquer/{fichier}", json={"humain": [
        {"label": "B", "mesure_debut": 18, "mesure_fin": 25}]}).get_json()
    b = d["briques"][0]
    assert b["boucle"]["boucle"] == 8                       # pas 2 × 4
    assert [(x["l"], x["r"]) for x in b["periode"]] == [(4, 2)]
    pris = sorted(p["p"] + 1 for p in b["pics"] if p["pris"])
    assert pris == [35, 43, 60, 68, 76]
    # les lettres (2026-10-11) : accords identiques ⇒ même lettre, toujours ;
    # le seuil est la ressemblance la plus basse entre jetons identiques
    import json as _json
    from harmonia import soudure
    from harmonia.settings import SETTINGS
    chart = _json.loads((SETTINGS.charts_dir / f"{fichier}.json").read_text())
    M, bars = d["mot"], soudure.accords_par_mesure(chart)
    bo, le = M["bornes"], M["lettres"]
    for i in range(len(le)):
        for k in range(i + 1, len(le)):
            if soudure._meme_jeton(bars, bo[i], bo[i + 1], bo[k], bo[k + 1]):
                assert le[i] == le[k], (i, k, le)
    assert M["regle"] == "accords" and M["n_identiques"] > 0
    assert d["soudures"]["couts"] == {"4": 27, "6": 38}       # lettres du 2026-10-11 (avant : 19 / 34)
    assert [x["m0"] for x in d["nommage"]["intro_avale"]] == [0, 8]
    # suggestions pâles : 3 B sur 5, 35 et 43 écartées par la parité
    pales = d["pales"][0]
    # depuis le 2026-10-11 les pâles = le calcul de Valider : les 5 reprises
    # (avant : 60, 68, 76 seulement, 35 et 43 écartées par la parité)
    assert [it["b0"] + 1 for it in pales["proposees"]] == [35, 43, 60, 68, 76]


def test_trace_ne_change_rien_au_calcul():
    """Sans `trace`, la recherche rend la même chose qu'avec."""
    import numpy as np

    from harmonia.sections.simulation import cherche_sur_matrices
    rng = np.random.default_rng(0)
    V = rng.random((40, 12))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    V[20:28] = V[4:12]
    S = V @ V.T
    a = cherche_sur_matrices(S, V, 4, 11)
    tr: dict = {}
    b = cherche_sur_matrices(S, V, 4, 11, trace=tr)
    assert a[0] == b[0]
    assert tr["boucle"] and "transposees" in tr


def test_combler_trous_suit_le_client():
    """La copie Python de `comblerTrous` (sections_editor.js) : tête → intro,
    milieu → queue, suffixe quand (label, longueur) est déjà pris."""
    from harmonia.sections.explication import combler_trous
    secs = [{"label": "A", "mesure_debut": 3, "mesure_fin": 6},
            {"label": "B", "mesure_debut": 9, "mesure_fin": 12}]
    assert combler_trous(secs, 14) == [
        {"label": "intro", "mesure_debut": 1, "mesure_fin": 2},
        {"label": "A", "mesure_debut": 3, "mesure_fin": 6},
        {"label": "queue", "mesure_debut": 7, "mesure_fin": 8},
        {"label": "B", "mesure_debut": 9, "mesure_fin": 12},
        {"label": "queue 2", "mesure_debut": 13, "mesure_fin": 14}]
    # tête suivie d'une intro tracée plus loin : pas deux intros adjacentes
    assert combler_trous([{"label": "intro", "mesure_debut": 2, "mesure_fin": 5}], 5)[0]["label"] == "queue"
