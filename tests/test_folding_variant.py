"""harmonia.folding — l'accord optionnel « en petit au-dessus », comme iReal.

Louis, 2026-09-15, Easy On Me section B : « c'est un F puis la basse
descend sur D (donc ça donne un D-7) avant d'atterrir sur le C, sur les
passes suivantes des fois la basse fait quelque chose de plus complexe, donc
difficile à dire, typiquement le genre de cas où j'aimerais avoir juste F
affiché et le D-7 en optionnel en petit au-dessus en suggestion, comme iReal
fait. »

Cas concret rejoué ici (mesuré le 2026-09-16 sur `min_X-yIEMduRXk`, section
B, position 0 du gabarit de période 4) : 6 passes empilées et gardées, 3
d'entre elles (mesures 14, 18, 35) décodent F PUIS D-(7) au 4e temps, à
0.749/0.698/0.722 de confiance ; les 3 autres (39, 52, 56) ne tiennent qu'un
F. Le gabarit moyenné écrit un seul F pour toute la mesure — c'est CE
qu'`harmonia.folding._bar_variant` doit retrouver.
"""
from __future__ import annotations

from harmonia.folding import VAR_MIN_CONF, _attach_variant, _bar_variant


def _chord(root, q, beat, c, bass=-1, carry=False, nc=False):
    return {"root": root, "q": q, "beat": beat, "c": c, "bass": bass,
            "carry": carry, "nc": nc}


# ── le cas Easy On Me, tel que mesuré ───────────────────────────────────────

def _bar_f_seul(c=0.80):
    return [_chord(5, "", 0, c)]                       # F, rien d'autre


def _bar_f_puis_dmin(c_f, c_d, q_d="-"):
    return [_chord(5, "", 0, c_f), _chord(2, q_d, 3, c_d)]


def test_easy_on_me_f_puis_d_mineur_devient_la_variante():
    membres = [
        _bar_f_puis_dmin(0.939, 0.749, "-"),    # mes. 14
        _bar_f_puis_dmin(0.944, 0.698, "-7"),   # mes. 18
        _bar_f_puis_dmin(0.957, 0.722, "-"),    # mes. 35
        _bar_f_seul(0.802),                     # mes. 39
        _bar_f_seul(0.804),                     # mes. 52
        _bar_f_seul(0.795),                     # mes. 56
    ]
    gabarit = [_chord(5, "", 0, 0.77)]          # ce que le gabarit écrit : F, toute la mesure
    v = _bar_variant(membres, gabarit)
    assert v is not None
    assert (v["root"], v["q"]) == (2, "-")      # la plus confiante des 3 (mes. 14)
    assert v["n"] == 3                          # 3 des 6 passes le jouent
    assert v["c"] == 0.749
    assert v["beat"] == 3


def test_confiance_insuffisante_nest_pas_une_variante():
    membres = [_bar_f_puis_dmin(0.9, VAR_MIN_CONF - 0.01)]
    gabarit = [_chord(5, "", 0, 0.8)]
    assert _bar_variant(membres, gabarit) is None


def test_le_gabarit_qui_attaque_deja_la_neutralise():
    """Si le gabarit lui-même attaque à ce temps, ce n'est pas un accord
    minoritaire effacé — c'est un désaccord sur l'attaque, une autre
    question que cette fonction ne traite pas."""
    membres = [_bar_f_puis_dmin(0.9, 0.75)]
    gabarit = [_chord(5, "", 0, 0.8), _chord(2, "-", 3, 0.6)]
    assert _bar_variant(membres, gabarit) is None


def test_meme_famille_que_ce_qui_est_tenu_nest_pas_une_variante():
    """F7 sous un F tenu : même famille (maj/dom se distinguent, mais
    `family` range les deux « accords », teste la vraie règle : min vs maj)."""
    membres = [[_chord(5, "", 0, 0.9), _chord(5, "^7", 3, 0.8)]]  # F puis F^7 : même famille
    gabarit = [_chord(5, "", 0, 0.85)]
    assert _bar_variant(membres, gabarit) is None


def test_variante_ignore_les_carry_et_les_nc():
    membres = [[_chord(5, "", 0, 0.9),
                _chord(2, "-", 3, 0.8, carry=True)],       # tenue : pas une attaque
               [_chord(5, "", 0, 0.9),
                _chord(0, "", 3, 0.9, nc=True)]]            # N.C. : jamais un accord
    gabarit = [_chord(5, "", 0, 0.85)]
    assert _bar_variant(membres, gabarit) is None


def test_pas_de_variante_sans_ecart():
    """Toutes les passes s'accordent avec le gabarit : rien à montrer."""
    membres = [_bar_f_seul(), _bar_f_seul(), _bar_f_seul()]
    gabarit = [_chord(5, "", 0, 0.9)]
    assert _bar_variant(membres, gabarit) is None


def test_gabarit_vide_ne_plante_pas():
    assert _bar_variant([_bar_f_puis_dmin(0.9, 0.8)], []) is None


# ── le groupage par famille (D- et D-7 sont la même lecture) ────────────────

def test_d_mineur_et_d_mineur_7_se_regroupent():
    """3 passes : deux disent D-, une dit D-7 — un seul groupe, n=3, et la
    plus confiante des trois donne l'orthographe affichée (2026-09-16 :
    décision ouverte, voir le rapport — pas un mélange inventé)."""
    membres = [
        _bar_f_puis_dmin(0.9, 0.70, "-"),
        _bar_f_puis_dmin(0.9, 0.75, "-7"),      # la plus confiante
        _bar_f_puis_dmin(0.9, 0.65, "-"),
    ]
    gabarit = [_chord(5, "", 0, 0.8)]
    v = _bar_variant(membres, gabarit)
    assert v["n"] == 3
    assert v["q"] == "-7"
    assert v["c"] == 0.75


def test_deux_lectures_minoritaires_distinctes_la_plus_soutenue_gagne():
    """2 passes entendent D-, 1 seule entend autre chose (Bb) : le groupe
    D- (n=2) l'emporte sur Bb (n=1), même si Bb est mieux noté isolément."""
    membres = [
        _bar_f_puis_dmin(0.9, 0.70, "-"),
        _bar_f_puis_dmin(0.9, 0.65, "-"),
        [_chord(5, "", 0, 0.9), _chord(10, "", 3, 0.95)],   # Bb, seul à le dire
    ]
    gabarit = [_chord(5, "", 0, 0.8)]
    v = _bar_variant(membres, gabarit)
    assert (v["root"], v["n"]) == (2, 2)


# ── l'attache sur le bon accord écrit (`_attach_variant`) ───────────────────

def test_attache_sur_laccord_qui_sonne_a_ce_temps():
    chords_k = [_chord(0, "", 0, 0.9), _chord(5, "", 2, 0.8)]
    variant = {"root": 2, "q": "-", "bass": -1, "c": 0.7, "n": 2, "beat": 3}
    _attach_variant(chords_k, variant)
    assert "var" not in chords_k[0]
    assert chords_k[1]["var"] == {"root": 2, "q": "-", "bass": -1, "c": 0.7, "n": 2}


def test_attache_sur_le_seul_accord_dune_mesure_a_un_seul_accord():
    chords_k = [_chord(5, "", 0, 0.9)]
    variant = {"root": 2, "q": "-", "bass": -1, "c": 0.75, "n": 3, "beat": 3}
    _attach_variant(chords_k, variant)
    assert chords_k[0]["var"] == {"root": 2, "q": "-", "bass": -1, "c": 0.75, "n": 3}
