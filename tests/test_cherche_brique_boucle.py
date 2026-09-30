"""tests/test_cherche_brique_boucle.py — les trois règles de Louis (2026-09-30).

Sur la page `docs/plots/occurrences_brique_2026-09-30.html`, Louis : « le
détecteur avec pics est bon », il lui faut trois changements :

1. « Interdire la recherche de pics dans la zone qui est donnée de base, ce
   ne sont pas des pics valides. » — une fenêtre qui chevauche la brique n'est
   ni un pic, ni un morceau du FOND contre lequel on juge les autres pics.
2. « Quand je donne une section, il faut regarder si elle-même est une
   répétition d'une boucle minimale [...] C'est la boucle minimale qui définit
   la section qui devra ensuite être cherchée (ça devrait fixer San Francisco
   Street). »
3. « Ce n'est pas la hauteur du pic qui dicte, mais la hauteur du pic par
   rapport au reste. »

Les tests 1 à 3 tournent sur une VRAIE matrice chord-tone (accords → vecteurs
de 12 classes → cosinus), seuls le modèle audio et la grille sont simulés.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.sections import simulation as sim

# des triades, en classes de hauteur
ACC = {"C": (0, 4, 7), "F#": (6, 10, 1), "Bb": (10, 2, 5), "D": (2, 6, 9),
       "Eb": (3, 7, 10), "Ab": (8, 0, 3), "E": (4, 8, 11), "A": (9, 1, 4),
       "B": (11, 3, 6), "G": (7, 11, 2)}


def _vecteurs(accords):
    V = np.zeros((len(accords), 12))
    for i, a in enumerate(accords):
        for pc in ACC[a]:
            V[i, pc] = 1.0
    return V / np.linalg.norm(V, axis=1, keepdims=True)


@pytest.fixture()
def chanson(tmp_path, monkeypatch):
    """Un chart dont chaque mesure joue un accord donné : la matrice est la
    vraie SSM chord-tone de cette suite d'accords."""
    (tmp_path / "essai.m4a").write_bytes(b"x")

    def pose(accords):
        from harmonia.sections import similarity as vs
        V = _vecteurs(accords)
        monkeypatch.setattr("harmonia.musx.frame_posteriors",
                            lambda *a, **k: [np.zeros((10, 73), np.float32)])
        monkeypatch.setattr(vs, "harmonic_vectors", lambda *a, **k: V)
        monkeypatch.setattr(vs, "ssm", lambda *a, **k: V @ V.T)
        n = len(accords)
        chart = {"audio_url": "/audio/essai.m4a",
                 "barGrid": [float(i) for i in range(n + 1)], "nBars": n}
        return chart, tmp_path

    return pose


def _mesures(occ):
    out = set()
    for a, b, *_r in occ:
        out |= set(range(a, b + 1))
    return out


# ── règle 2 : la boucle minimale ─────────────────────────────────────────────

BOUCLE = ["C", "C", "F#", "Bb"]
QUEUE = ["Eb", "Eb", "Ab", "Ab"]           # la dernière fois, autre fin
PONT = ["E", "A", "E", "A", "E", "A", "B", "B"]


def test_san_francisco_street_une_brique_de_24_est_5_boucles_plus_une_fin(chanson):
    """SF Street : Louis surligne 24 mesures (la boucle de 4 jouée 5 fois +
    une fin), et la reprise suivante n'en fait que 16 (3 fois + la même fin).
    L'ancienne recherche à longueur FIXE (24) ne la voyait pas. En cherchant la
    boucle, les 16 mesures sont retrouvées, fin comprise."""
    accords = (["D"] * 4 + BOUCLE * 5 + QUEUE + PONT
               + BOUCLE * 3 + QUEUE + PONT + ["D"] * 4)
    chart, audio = chanson(accords)
    b0, b1 = 4, 27                              # la brique de Louis, 24 mes.
    occ, info = sim.cherche_brique(chart, b0, b1, audio, label="A")
    assert info["boucle"] == 4
    # la brique elle-même devient ses 5 passages + sa fin
    dedans = sorted((a, b) for a, b, *_r in occ if b0 <= a and b <= b1)
    assert dedans == [(4, 7), (8, 11), (12, 15), (16, 19), (20, 23), (24, 27)]
    # la reprise de 16 mesures (36-51) est couverte en entier, rien d'autre
    dehors = _mesures(o for o in occ if o[0] > b1)
    assert dehors == set(range(36, 52))


def test_la_boucle_ne_descend_jamais_sous_quatre_mesures(chanson):
    """Deux accords qui alternent se répètent toutes les 2 mesures, mais une
    section, c'est au moins 4 mesures (Louis, 2026-08-16) : la boucle est 4,
    jamais 2."""
    alt = ["C", "F#"] * 4
    accords = ["D"] * 4 + alt + ["E", "B", "G", "Ab"] * 2 + alt + ["D"] * 4
    chart, audio = chanson(accords)
    occ, info = sim.cherche_brique(chart, 4, 11, audio, label="A")
    assert info["boucle"] == 4
    assert all(b - a + 1 == 4 for a, b, *_r in occ)
    assert _mesures(o for o in occ if o[0] > 11) == set(range(20, 28))


def test_une_brique_sans_boucle_interne_reste_entiere(chanson):
    """Stand By Me : 8 mesures qui ne se répètent pas en elles-mêmes — la
    brique reste une brique de 8, et ses reprises font 8."""
    grille = ["C", "C", "A", "A", "F#", "G", "C", "C"]
    accords = grille + ["Eb", "B", "Eb", "B"] + grille * 2
    chart, audio = chanson(accords)
    occ, info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert info["boucle"] == 8
    assert [(a, b) for a, b, *_r in occ] == [(0, 7), (12, 19), (20, 27)]


# ── règle 1 : la zone donnée n'est ni un pic, ni le fond ─────────────────────

@pytest.fixture()
def courbe(tmp_path, monkeypatch):
    """La courbe de reprise pilotée directement (comme `test_cherche_brique`) ;
    la matrice est nulle, donc aucune boucle interne."""
    (tmp_path / "essai.m4a").write_bytes(b"x")
    chart = {"audio_url": "/audio/essai.m4a",
             "barGrid": [float(i) for i in range(41)], "nBars": 40}

    def pose(scores):
        from harmonia.sections import similarity as vs
        monkeypatch.setattr("harmonia.musx.frame_posteriors",
                            lambda *a, **k: [np.zeros((10, 73), np.float32)])
        monkeypatch.setattr(vs, "ssm", lambda *a, **k: np.zeros((40, 40)))
        monkeypatch.setattr(vs, "harmonic_vectors",
                            lambda *a, **k: np.zeros((40, 12)))
        monkeypatch.setattr(vs, "_slide",
                            lambda M, a0, L, n: np.asarray(scores, float))
        return chart, tmp_path

    return pose


def test_la_zone_donnee_ne_gonfle_pas_le_seuil(courbe):
    """Les décalages de la brique sur elle-même scorent haut (0,99) : dans
    l'ancienne loi ils tiraient le seuil du morceau au plafond et deux vraies
    reprises à 0,90 étaient recalées. Exclues du fond, elles passent."""
    sc = [0.55] * 33
    for p in range(1, 16):                      # fenêtres qui chevauchent 8-15
        sc[p] = 0.99
    sc[8] = 1.0
    sc[16] = sc[28] = 0.90
    chart, audio = courbe(sc)
    occ, _info = sim.cherche_brique(chart, 8, 15, audio, label="A")
    assert [(a, b) for a, b, *_r in occ] == [(8, 15), (16, 23), (28, 35)]


# ── règle 3 : la hauteur du pic par rapport au reste ─────────────────────────

def test_un_pic_bas_mais_isole_bat_un_pic_haut_noye_dans_son_fond(courbe):
    """0,88 au milieu d'un fond à 0,55 est une reprise ; 0,97 sur un plateau à
    0,95 n'en est pas une (tout le passage ressemble à la brique, aucun départ
    ne se détache). L'ancienne loi, absolue, faisait l'inverse — ici elle ne
    trouvait rien du tout."""
    sc = [0.55] * 33
    sc[0] = 1.0
    for p in range(1, 8):
        sc[p] = 0.60
    for p in range(14, 33):
        sc[p] = 0.95
    sc[10] = 0.88
    sc[26] = 0.97
    chart, audio = courbe(sc)
    occ, info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, *_r in occ] == [(0, 7), (10, 17)]
    assert info["n_reprises"] == 1


# ── le trait de Louis reste le sien ──────────────────────────────────────────

def test_les_passages_de_la_brique_restent_ses_traits(monkeypatch):
    """Sa brique découpée en boucles : chaque passage DANS ce qu'il a tracé
    reste marqué `trace` (la route en fait des sections « humain »), une
    reprise ailleurs non."""
    monkeypatch.setattr(sim, "cherche_brique", lambda chart, b0, b1, ad, **k: (
        [(0, 3, 1.0, 0), (4, 7, 1.0, 0), (20, 23, 0.95, 0)],
        {"label": "A", "n_reprises": 1, "raison": None}))
    monkeypatch.setattr("harmonia.soudure.traits_propres",
                        lambda traits, n: ([(0, 7, "A")], []))
    occ, _e, _d = sim.occurrences_de_tous(
        {"barGrid": [float(i) for i in range(41)]}, [], "/nulle/part")
    assert [(o["m0"], o["trace"]) for o in occ] == [(0, True), (4, True),
                                                    (20, False)]


# ── règle 4 : la référence du morceau (2026-09-30 soir) ──────────────────────

def test_let_it_be_un_jumeau_loin_de_la_reference_est_refuse(courbe):
    """Let It Be : les vraies reprises à 0,99–1,0, le refrain (jumeau) à
    0,90–0,91 un peu partout, sur un fond à 0,6 — la règle 3 seule le prenait.
    La meilleure reprise hors zone (1,0) dit ce qu'est une reprise DANS CE
    MORCEAU ; les jumeaux en sont séparés par un trou bien plus grand que la
    dispersion des vraies reprises, et ils sont la MAJORITÉ des pics : c'est
    le niveau du reste du morceau, pas quelques variantes."""
    sc = [0.60] * 33
    sc[0] = 1.0
    sc[8] = 1.0
    sc[16] = 0.994
    sc[24], sc[26], sc[28] = 0.91, 0.905, 0.90   # les jumeaux
    chart, audio = courbe(sc)
    occ, info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, *_r in occ] == [(0, 7), (8, 15), (16, 23)]


def test_every_breath_quelques_reprises_a_fin_differente_restent(courbe):
    """Every Breath You Take : 3 reprises exactes et une à fin différente
    (0,919). Même trou que Let It Be, mais le groupe du bas n'est pas la
    majorité : c'est une variante, pas le niveau du morceau. Elle reste."""
    sc = [0.60] * 33
    sc[0] = 1.0
    sc[8], sc[16], sc[24] = 0.999, 0.997, 0.919
    chart, audio = courbe(sc)
    occ, _info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, *_r in occ] == [(0, 7), (8, 15), (16, 23), (24, 31)]


def test_stand_by_me_un_plateau_de_reprises_bruitees_est_garde(courbe):
    """Stand By Me : les reprises s'étalent de 0,90 à 0,94 sans trou net, et
    la meilleure est elle-même loin de 1,0 (le morceau est bruité). Toutes
    restent."""
    sc = [0.62] * 33
    sc[0] = 1.0
    for p, v in ((8, 0.943), (16, 0.914), (24, 0.896)):
        sc[p] = v
    chart, audio = courbe(sc)
    occ, _info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, *_r in occ] == [(0, 7), (8, 15), (16, 23), (24, 31)]


# ── règle 5 : un départ doit dépasser ses voisins (2026-09-30 nuit) ───────────
#
# Louis : « si on hésite entre un départ et celui de ses deux voisins, on
# prend celui qui est plus grand que ses voisins d'avant et d'après
# (généralise au cas où on aurait des boucles plus grandes que 2) ».

ACC5 = dict(ACC, Bm=(11, 2, 6), **{"C#m": (1, 4, 8), "F#m": (6, 9, 1)})


@pytest.fixture()
def chanson5(tmp_path, monkeypatch):
    (tmp_path / "essai.m4a").write_bytes(b"x")

    def pose(accords):
        from harmonia.sections import similarity as vs
        V = np.zeros((len(accords), 12))
        for i, a in enumerate(accords):
            for pc in ACC5[a]:
                V[i, pc] = 1.0
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        monkeypatch.setattr("harmonia.musx.frame_posteriors",
                            lambda *a, **k: [np.zeros((10, 73), np.float32)])
        monkeypatch.setattr(vs, "harmonic_vectors", lambda *a, **k: V)
        monkeypatch.setattr(vs, "ssm", lambda *a, **k: V @ V.T)
        n = len(accords)
        return {"audio_url": "/audio/essai.m4a",
                "barGrid": [float(i) for i in range(n + 1)], "nBars": n}, tmp_path

    return pose


def test_the_walk_la_boucle_reste_en_phase_meme_si_decalee_en_case_plus(chanson5):
    """The Walk : A = A–Bm–A–Bm (boucle de 4, sous-période 2). Le passage de
    A qui suit le B dure 6 boucles (20–43), précédé du C♯m–Bm de la fin du B
    et suivi de A–D. Décalées de 2 mesures, on case 7 boucles (18–45), dont
    les deux de bord ne valent que 0,917 (elles commencent sur le C♯m / finissent
    sur le D) ; en phase, 6 boucles à 1,0. L'ancien choix « le plus de
    boucles » prenait les 7 décalées. Règle 5 : 18 est plus bas que son
    voisin 20, 42 plus bas que 40 — ce ne sont pas des départs."""
    accords = (["D"] * 4 + ["A", "Bm"] * 4
               + ["E", "E", "F#m", "F#m", "D", "D", "C#m", "Bm"]
               + ["A", "Bm"] * 12 + ["A", "D", "E", "E"] + ["D"] * 4)
    chart, audio = chanson5(accords)
    occ, info = sim.cherche_brique(chart, 4, 11, audio, label="A")
    assert info["boucle"] == 4 and info["sous_periode"] == 2
    dehors = sorted((a, b) for a, b, *_r in occ if a > 11)
    assert dehors == [(20, 23), (24, 27), (28, 31), (32, 35), (36, 39), (40, 43)]


def test_boucle_de_8_un_depart_decale_d_une_mesure_ne_bat_pas_le_bon(courbe):
    """La même chose avec une boucle de 8 (la fenêtre de voisinage est ±7) :
    les vraies reprises en 10 et 18 (0,99 ; 0,985), une traîne décalée d'une
    mesure en 9, 17, 25 (0,968 ; 0,963 ; 0,958 — pas de trou pour la règle 4).
    Décalé, on case 3 reprises ; en phase, 2. Chaque départ décalé a un voisin
    plus haut à moins de 8 mesures : ce ne sont pas des départs.

    RÈGLE 5b (2026-09-30) : l'égalité est la tolérance du morceau (ici
    2 × 0,01 = 0,02). La traîne était à 0,975/0,97/0,965 — 0,015 sous les
    vraies, donc À ÉGALITÉ en 5b, et la couverture reprenait les 3 décalées.
    Remise à 0,022 sous les vraies : hors tolérance, la règle 5 s'applique."""
    sc = [0.55] * 33
    sc[0] = 1.0
    sc[10], sc[18] = 0.99, 0.985
    sc[9], sc[17], sc[25] = 0.968, 0.963, 0.958
    chart, audio = courbe(sc)
    occ, info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert info["boucle"] == 8
    assert [(a, b) for a, b, *_r in occ] == [(0, 7), (10, 17), (18, 25)]


def test_regle5_un_plateau_de_sous_periode_reste_entier(courbe):
    """Deux départs à moins de 0,01 l'un de l'autre sont À ÉGALITÉ (c'est le
    bruit de moyenne, `BRUIT_PIC`) : aucun n'efface l'autre, la couverture
    choisit. Sinon un plateau (sous-période, tout à 0,995–1,0) serait réduit à
    un seul départ au hasard du bruit."""
    sc = [0.55] * 33
    sc[0] = 1.0
    for p, v in ((8, 0.998), (10, 1.0), (12, 0.996), (14, 0.999), (16, 0.997)):
        sc[p] = v
    chart, audio = courbe(sc)
    occ, _info = sim.cherche_brique(chart, 0, 7, audio, label="A")
    assert [(a, b) for a, b, *_r in occ] == [(0, 7), (8, 15), (16, 23)]
