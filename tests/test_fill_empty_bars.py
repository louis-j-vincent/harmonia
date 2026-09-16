"""harmonia.server.routes.annotate.fill_empty_bars — insérer un accord dans
une mesure entièrement tenue (`bars == []`), le seul cas qu'`overlay()`
abandonne en silence faute d'accord-hôte à cloner (2026-09-16, Louis :
« quand je clique sur un endroit vide de la barre je peux ajouter un accord
dessus »). Le reste de la fonctionnalité (l'outil « barre en grand ») est
vérifié par Playwright ; ceci est la seule brique serveur nouvelle, donc la
seule sans couverture jusqu'ici.
"""
from harmonia.server.routes.annotate import _locate_empty_prefix_bar, fill_empty_bars


def _model(bar_ranges, bars, tail=0):
    sec = {"barRanges": bar_ranges, "bars": bars}
    if tail:
        sec["endings"] = {"tail": tail}
    return {"sections": [sec]}


def test_insere_dans_la_mesure_vide_par_arithmetique_barranges():
    """min_Ju8Hr50Ckwk.json, section outro : une seule mesure, [[128,128]]."""
    m = _model([[128, 128]], [[]])
    ann = {"chords": [{"bar": 128, "beat": 0, "root": 4, "q": "-",
                       "bass": -1, "t0": 300.0, "t1": 302.0}]}
    out = fill_empty_bars(m, ann)
    bar = out["sections"][0]["bars"][0]
    assert len(bar) == 1
    c = bar[0]
    assert (c["root"], c["q"], c["confirmed"], c["nc"]) == (4, "-", True, False)
    assert c["bar"] == 128 and c["beat"] == 0


def test_idempotent_si_deja_presente():
    """overlay() a peut-être déjà écrit cette clé (bar, beat) — ne pas doubler."""
    m = _model([[10, 11]], [[{"root": 0, "q": "", "bar": 10, "beat": 0}], []])
    ann = {"chords": [{"bar": 10, "beat": 0, "root": 7, "q": "7", "t0": 0, "t1": 1}]}
    out = fill_empty_bars(m, ann)
    assert out["sections"][0]["bars"][0] == [{"root": 0, "q": "", "bar": 10, "beat": 0}]


def test_rien_a_faire_sans_correction_de_racine():
    """Un fix sans `root` (une autre sorte de correction) ne doit rien tenter."""
    m = _model([[5, 5]], [[]])
    assert fill_empty_bars(m, {"chords": [{"bar": 5, "beat": 0}]}) is m


def test_mesure_hors_prefixe_de_fin_alternative_non_couverte():
    """Une mesure vide DANS la queue 1./2. n'est pas résolue par cette
    arithmétique (voir la docstring de `_locate_empty_prefix_bar`) — elle
    ne casse rien, elle est juste ignorée, comme `overlay()` l'ignorait déjà."""
    m = _model([[20, 23]], [[], [], [], []], tail=2)
    assert _locate_empty_prefix_bar(m, 22) is None
    assert _locate_empty_prefix_bar(m, 23) is None
    out = fill_empty_bars(m, {"chords": [{"bar": 23, "beat": 0, "root": 0,
                                          "q": "", "t0": 0, "t1": 1}]})
    assert out["sections"][0]["bars"] == [[], [], [], []]


def test_mesure_deja_hote_non_touchee_par_fill_empty_bars():
    """Une mesure `!= []` est le travail d'`overlay()` — `_locate_empty_prefix_bar`
    ne la propose jamais comme cible, même si la clé demandée n'y est pas."""
    m = _model([[0, 0]], [[{"root": 0, "q": "", "bar": 0, "beat": 0}]])
    assert _locate_empty_prefix_bar(m, 0) is None
