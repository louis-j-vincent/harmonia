"""Le repli ne réécrit jamais une mesure que Louis a confirmée.

2026-08-20 : « les annotations utilisateur prennent toujours le dessus sur nos
inférences ». `confirmed` n'apparaissait alors nulle part dans `folding` ni
dans `refold` : une mesure corrigée à la main repassait sous le gabarit au
recuit suivant, sans un mot.
"""
from harmonia.folding import _write_position

BPB = 4
GRID = [0.0, 2.0, 4.0, 6.0]
TEMPLATE = [{"root": 7, "q": "", "bass": -1, "nc": False, "beat": 0, "c": 0.9}]


def _bar(root, q="", **kw):
    return [{"root": root, "q": q, "bass": -1, "nc": False, "carry": False,
             "beat": 0, "bar": 0, "c": 0.5, **kw}]


def test_une_mesure_confirmee_n_est_pas_reecrite():
    bars = [_bar(0, "-7", confirmed=True), _bar(2), _bar(4)]
    avant = [dict(c) for c in bars[0]]
    assert _write_position(bars, GRID, 0, TEMPLATE, BPB, n_obs=5) is False
    assert bars[0] == avant


def test_une_mesure_ordinaire_est_bien_reecrite():
    bars = [_bar(0, "-7"), _bar(2), _bar(4)]
    assert _write_position(bars, GRID, 0, TEMPLATE, BPB, n_obs=5) is True
    assert bars[0][0]["root"] == 7
    assert bars[0][0]["folded"] is True


def test_le_veto_est_par_MESURE_pas_par_lettre():
    """Une mesure confirmée ne gèle pas ses voisines."""
    bars = [_bar(0, confirmed=True), _bar(2), _bar(4)]
    _write_position(bars, GRID, 0, TEMPLATE, BPB, n_obs=5)
    assert _write_position(bars, GRID, 1, TEMPLATE, BPB, n_obs=5) is True
    assert bars[1][0]["root"] == 7


def test_un_seul_accord_confirme_protege_toute_sa_mesure():
    bars = [[{"root": 0, "q": "", "bass": -1, "nc": False, "carry": False,
              "beat": 0, "bar": 0, "c": 0.5},
             {"root": 5, "q": "7", "bass": -1, "nc": False, "carry": False,
              "beat": 2, "bar": 0, "c": 0.5, "confirmed": True}], _bar(2), _bar(4)]
    avant = [dict(c) for c in bars[0]]
    assert _write_position(bars, GRID, 0, TEMPLATE, BPB, n_obs=5) is False
    assert bars[0] == avant
