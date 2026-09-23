"""L'import iReal rend la grille QU'IREAL ÉCRIT — mêmes accords, mêmes sections.

Louis, 2026-09-18 : « teste de A à Z que tu puisses chercher un chart,
l'importer, et l'afficher dans notre app, avec les mêmes accords et les mêmes
sections que irealb ».

Les chaînes ci-dessous sont des chaînes d'accords iReal BRUTES, prises telles
quelles sur de vrais charts du corpus (voir le nom de chaque test). On les
ré-embrouille pour fabriquer une URL, exactement comme le fait
`irealb_export.py` — l'embrouillage d'iReal est une involution, donc
`_unscramble_chord_string` sert dans les deux sens.
"""
from __future__ import annotations

import urllib.parse

import pytest
from pyRealParser.pyRealParser import Tune

from harmonia.integrations.irealb_import import (
    _measures, _sections, parse_key, url_to_chart,
)

NN = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _url(chords: str, *, title="Test", composer="X", style="Medium Swing",
         key="C") -> str:
    scrambled = Tune._unscramble_chord_string(chords)
    raw = "=".join([title, composer, style, key,
                    Tune._chords_prefix + scrambled]) + "="
    return "irealb://" + urllib.parse.quote(raw, safe="=")


def _txt(c) -> str:
    if c.get("nc"):
        return "N.C."
    return (NN[c["root"]] + c["q"]
            + ("/" + NN[c["bass"]] if c.get("bass", -1) >= 0 else ""))


def _grid(bars) -> str:
    """Une liste de mesures → « C-7 | F7 | % » (une mesure tenue = %)."""
    return " | ".join(" ".join(_txt(c) for c in b) or "%" for b in bars)


# ── les accords ──────────────────────────────────────────────────────────────

def test_les_accords_sont_ceux_d_ireal_sans_traduction():
    """`q` est DÉJÀ la queue iReal dans tout Harmonia (harmonia/labels.py) :
    importer, c'est ne rien traduire. Un `^7` reste `^7`."""
    m = url_to_chart(_url("T44C^7XyQ|A-7XyQ|D-7XyQ|G7b13XyQZ"))
    assert _grid(m["sections"][0]["bars"]) == "C^7 | A-7 | D-7 | G7b13"


def test_la_basse_slash_survit():
    m = url_to_chart(_url("T44Bb^7/DXyQ|F7/CXyQZ"))
    assert _grid(m["sections"][0]["bars"]) == "Bb^7/D | F7/C"


def test_n_c_est_un_accord_nc_pas_un_do_majeur():
    """`n` d'iReal = pas d'accord. Le lire comme un C majeur silencieux
    écrirait un accord que le musicien n'a pas écrit."""
    m = url_to_chart(_url("T44nXyQ|C^7XyQZ"))
    bars = m["sections"][0]["bars"]
    assert bars[0][0]["nc"] is True
    assert bars[1][0]["nc"] is False


@pytest.mark.parametrize("tok", ["cl", "x"])
def test_la_mesure_repetee_devient_une_mesure_TENUE(tok):
    """`cl` et `x` disent tous deux « rejoue la mesure précédente ». Les deux
    encodages existent dans le corpus iReal ; n'en lire qu'un faisait
    DISPARAÎTRE la mesure et décalait toute la suite d'un cran."""
    m = url_to_chart(_url(f"T44C^7XyQ|{tok}XyQ|F7XyQZ"))
    assert _grid(m["sections"][0]["bars"]) == "C^7 | % | F7"
    assert m["nBars"] == 3


def test_deux_accords_dans_une_mesure_tombent_sur_1_et_3():
    m = url_to_chart(_url("T44G-7 C7LZF^7XyQZ"))
    bar = m["sections"][0]["bars"][0]
    assert [c["beat"] for c in bar] == [0, 2]


def test_les_barres_obliques_placent_l_accord():
    """`C7 p p F7` dit « C7 sur 1, F7 sur 4 » — la barre oblique d'iReal est
    une information de PLACEMENT, pas un accord de plus."""
    m = url_to_chart(_url("T44C7 p p F7LZG7XyQZ"))
    bar = m["sections"][0]["bars"][0]
    assert [_txt(c) for c in bar] == ["C7", "F7"]
    assert [c["beat"] for c in bar] == [0, 3]


def test_l_accord_optionnel_entre_parentheses_devient_var():
    """iReal écrit l'accord de rechange entre parenthèses ; le chart le rend
    en petit au-dessus (`ch.var`, chart.js, attendu depuis 2026-08-17)."""
    m = url_to_chart(_url("T44Ab7#9(Bo)XyQ|G7XyQZ"))
    ch = m["sections"][0]["bars"][0][0]
    assert _txt(ch) == "Ab7#9"
    assert ch["var"]["root"] == 11 and ch["var"]["q"] == "o"


# ── les sections ─────────────────────────────────────────────────────────────

def test_les_reperes_de_section_deviennent_les_sections():
    """`pyRealParser` les EFFACE (`_remove_annotations`) — c'est la raison
    d'être de ce module : on relit la chaîne brute pour les garder."""
    m = url_to_chart(_url("*AT44C^7XyQ|F^7XyQ]*BD-7XyQ|G7XyQZ"))
    assert [s["label"] for s in m["sections"]] == ["A", "B"]
    assert _grid(m["sections"][0]["bars"]) == "C^7 | F^7"
    assert _grid(m["sections"][1]["bars"]) == "D-7 | G7"


def test_une_reprise_sans_fin_se_joue_deux_fois():
    m = url_to_chart(_url("*A{T44C^7XyQ|F^7XyQ}Z"))
    sec = m["sections"][0]
    assert sec["reps"] == 2
    assert len(sec["barRanges"]) == 2
    assert m["nBars"] == 4          # 2 mesures écrites, 4 jouées


def test_premiere_et_deuxieme_fin():
    """Le contrat de `folding._ireal_endings`, à la lettre : `bars` = un
    passage ENTIER (tronc + queue du 1er variant), `endings.variants[i].bars`
    = la queue de ce variant."""
    m = url_to_chart(_url(
        "*A{T44C^7XyQ|F^7XyQ|A-7XyQ|N1D-7XyQ|G7XyQ}|N2D-7XyQ|C6XyQZ"))
    sec = m["sections"][0]
    assert sec["reps"] == 2
    e = sec["endings"]
    assert e["tail"] == 2
    assert len(sec["bars"]) == 5                       # 3 de tronc + 2 de queue
    assert _grid(sec["bars"][:3]) == "C^7 | F^7 | A-7"
    assert _grid(e["variants"][0]["bars"]) == "D-7 | G7"
    assert _grid(e["variants"][1]["bars"]) == "D-7 | C6"
    assert _grid(sec["bars"][3:]) == "D-7 | G7"        # la queue du 1er passage
    assert [v["passes"] for v in e["variants"]] == [[0], [1]]


def test_la_deuxieme_fin_ne_fait_PAS_une_section_a_elle_seule():
    """La 2e fin arrive APRÈS la barre de reprise fermante : sans exception
    explicite, la règle « après un `}`, nouvelle section » la détachait."""
    m = url_to_chart(_url(
        "{T44C^7XyQ|N1F^7XyQ}|N2G7XyQ][*BD-7XyQ|A-7XyQZ"))
    assert [s["label"] for s in m["sections"]] == ["A", "B"]
    assert m["sections"][0]["endings"]["tail"] == 1


def test_un_repere_i_est_un_intro():
    m = url_to_chart(_url("*iT44nXyQ|nXyQ]*AC^7XyQ|F^7XyQZ"))
    assert [s["label"] for s in m["sections"]] == ["intro", "A"]


def test_une_double_barre_coupe_une_section_sans_repere():
    """Beaucoup de grilles du forum n'ont AUCUN `*X` : la double barre est
    alors le seul indice de structure qu'iReal laisse."""
    m = url_to_chart(_url("T44C^7XyQ|F^7XyQ]D-7XyQ|G7XyQZ"))
    assert [s["label"] for s in m["sections"]] == ["A", "B"]


# ── le chiffrage ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tok,sig,bpb", [
    ("T44", "4/4", 4), ("T34", "3/4", 3), ("T54", "5/4", 5),
    ("T68", "6/8", 2),
    # `T12` est un CODE, pas « 1 sur 2 » : c'est 12/8, battu à 4.
    # pyRealParser lit (1, 2) ; 21 des 2207 standards en dépendent.
    ("T12", "12/8", 4),
])
def test_le_chiffrage_est_un_code_a_deux_chiffres(tok, sig, bpb):
    m = url_to_chart(_url(f"{tok}C^7XyQ|F^7XyQZ"))
    assert m["meta"]["timeSig"] == sig
    assert m["bpb"] == bpb


def test_la_tonalite_mineure_d_ireal():
    assert parse_key("G-") == {"tonic": 7, "mode": "minor"}
    assert parse_key("Bb") == {"tonic": 10, "mode": "major"}


# ── le ChartModel que l'app sait ouvrir ──────────────────────────────────────

def test_le_modele_porte_ce_que_le_shell_exige():
    """`chart.js:loadModel` déréférence `s.spans[0][0]` et `m.key.tonic` sans
    garde, et `qClass(q)` appelle `q.startsWith` : un champ manquant ne
    dégrade pas le rendu, il ferme le chart (« Could not open that chart »)."""
    m = url_to_chart(_url(
        "*A{T44C^7XyQ|F^7XyQ|N1D-7XyQ}|N2G7XyQ]*BA-7 D7LZG^7XyQZ"))
    assert m["key"]["tonic"] == 0 and m["key"]["mode"] == "major"
    assert m["audio_url"] == ""              # un chart écrit ne se joue pas
    assert len(m["barGrid"]) == m["nBars"] + 1
    for s in m["sections"]:
        assert s["id"] and s["label"]
        assert s["spans"] and len(s["spans"]) == s["reps"]
        assert len(s["barRanges"]) == s["reps"]
        tail = (s.get("endings") or {}).get("tail", 0)
        n_rows = len(s["bars"]) - tail + sum(
            len(v["bars"]) for v in (s.get("endings") or {}).get("variants", []))
        assert len(s["barSpans"]) == n_rows
        allb = list(s["bars"]) + [b for v in (s.get("endings") or {}).get("variants", [])
                                  for b in v["bars"]]
        for bar in allb:
            for c in bar:
                assert isinstance(c["root"], int) and isinstance(c["q"], str)
                assert c["t1"] > c["t0"]     # jamais d'empan nul : `chordAt`
                assert c["c"] == 1.0         # sinon `confColor` peint tout en rouge
                assert c["n"] == 0


def test_les_empans_de_section_ne_se_chevauchent_pas():
    m = url_to_chart(_url("*A{T44C^7XyQ|F^7XyQ}*BD-7XyQ|G7XyQZ"))
    plats = [sp for s in m["sections"] for sp in s["spans"]]
    plats.sort()
    for (a0, a1), (b0, _b1) in zip(plats, plats[1:]):
        assert a1 <= b0, f"{a0}-{a1} chevauche {b0}"


def test_le_stem_est_derive_du_titre():
    m = url_to_chart(_url("T44C^7XyQZ", title="Autumn Leaves"))
    assert m["file"] == "ireal_autumn_leaves"


# ── non-régression sur la vraie grille ───────────────────────────────────────

AUTUMN = ("*A{T44C-7XyQ|F7XyQ|Bb^7XyQ|Eb^7XyQ|Ah7XyQ|D7b13XyQ|G-6XyQKcl  }"
          "*B[Ah7XyQ|D7b13XyQ|G-6XyQKcl LZC-7XyQ|F7XyQ|Bb^7XyQ|Eb^7XyQ]"
          "*C[Ah7XyQ|D7b13XyQ|G-7 Gb7LZF-7 E7LZAh7XyQ|D7b13XyQ|G-6XyQKcl  Z")


def test_autumn_leaves_verbatim():
    """La grille des standards iReal, mot pour mot : A (×2), B, C, 8 mesures
    chacune, 32 mesures jouées."""
    m = url_to_chart(_url(AUTUMN, title="Autumn Leaves", key="G-"))
    assert m["keyName"] == "G minor"
    assert m["nBars"] == 32
    assert [(s["label"], s["reps"], len(s["bars"])) for s in m["sections"]] == [
        ("A", 2, 8), ("B", 1, 8), ("C", 1, 8)]
    assert _grid(m["sections"][0]["bars"]) == \
        "C-7 | F7 | Bb^7 | Eb^7 | Ah7 | D7b13 | G-6 | %"
    assert _grid(m["sections"][2]["bars"]) == \
        "Ah7 | D7b13 | G-7 Gb7 | F-7 E7 | Ah7 | D7b13 | G-6 | %"


def test_une_cellule_de_remplissage_n_est_pas_une_mesure():
    """`XyQ` est de la mise en page (une cellule vide d'iReal). La compter
    comme mesure gonflait la grille de mesures fantômes."""
    bars, _num, _den = _measures("T44C^7XyQ|XyQ|XyQXyQ|F^7XyQZ")
    assert len(bars) == 2


def test_une_grille_vide_echoue_fort():
    """Pas de chart à moitié importé : sans mesure, l'import refuse (le
    serveur répond 400) plutôt que d'écrire un fichier vide dans la
    bibliothèque."""
    with pytest.raises(ValueError):
        url_to_chart(_url("T44XyQZ"))


def test_le_decoupage_survit_a_un_repere_colle_a_la_double_barre():
    """« ...]*B[Ah7 » : le repère arrive ENTRE deux barres, donc sur une
    cellule vide. Sans report des marques, la section B perdait son nom."""
    bars, _n, _d = _measures("T44C^7XyQ]*B[Ah7XyQZ")
    blocks = _sections(bars)
    assert len(blocks) == 2
    assert blocks[1][0].marker == "B"


# ── le repliement (Louis, 2026-09-18, sur Virtual Insanity) ─────────────────
# « il n'y a pas un repliement comme je l'attendrais des sections »
# iReal écrit un chart LINÉAIRE : son refrain `*B` est réécrit en entier à
# chacune de ses reprises. Harmonia écrit la musique une fois et compte les
# passages. Trois défauts se cachaient là-dedans, un par test ci-dessous.

def test_un_refrain_reecrit_quatre_fois_est_ecrit_UNE_fois():
    """Le défaut principal : quatre blocs « B » identiques l'un sous l'autre,
    là où le chart doit montrer un B ×4 dont les passages sont dispersés."""
    refrain = "{*BC^7XyQ|A-7XyQ}"
    m = url_to_chart(_url("*AT44D-7XyQ|G7XyQ]" + refrain
                          + "*CE-7XyQ|A7XyQ]" + refrain + "Z"))
    labels = [s["label"] for s in m["sections"]]
    assert labels == ["A", "B", "C"]          # B une seule fois
    b = next(s for s in m["sections"] if s["label"] == "B")
    assert b["reps"] == 4                     # 2 écritures × 2 reprises
    assert len(b["bars"]) == 2                # …mais 2 mesures écrites
    # …et ses passages sont bien aux quatre endroits où il se joue
    assert [r[0] for r in b["barRanges"]] == [2, 4, 8, 10]
    assert m["nBars"] == 12


def test_les_lettres_suivent_l_ordre_de_JEU():
    """Louis, 2026-09-18 : « l'ordre des sections c'est TOUJOURS ABCDE… ».

    On n'hérite donc pas des repères d'iReal : ils ne sortent pas dans
    l'ordre. Ici iReal écrit un bloc anonyme, puis `*A`, puis `*B` — la
    grille se lit A, B, C de haut en bas.
    """
    m = url_to_chart(_url("T44C^7XyQ|F^7XyQ]*AD-7XyQ|G7XyQ]*BE-7XyQ|A7XyQZ"))
    assert [s["label"] for s in m["sections"]] == ["A", "B", "C"]


def test_une_section_qui_revient_reprend_SA_lettre():
    """L'alphabet n'avance que sur une musique NOUVELLE : c'est le repliement
    qui garantit qu'un retour reprend sa lettre, pas la numérotation."""
    m = url_to_chart(_url("*AT44C^7XyQ]*BD-7XyQ]*CC^7XyQ]*DE-7XyQZ"))
    occ = sorted((r[0], s["label"]) for s in m["sections"] for r in s["barRanges"])
    assert [l for _b, l in occ] == ["A", "B", "A", "C"]


def test_deux_blocs_identiques_fusionnent_meme_sous_deux_reperes_differents():
    """iReal nomme `*A` et `*C` deux passages rigoureusement identiques dans
    131 des 2207 standards. Puisque les lettres sont réattribuées dans
    l'ordre de jeu, les garder séparés afficherait deux fois la même musique
    sous deux lettres — ce qu'un chart replié ne fait jamais."""
    m = url_to_chart(_url("*AT44C^7XyQ|F^7XyQ]*CC^7XyQ|F^7XyQZ"))
    assert [s["label"] for s in m["sections"]] == ["A"]
    assert m["sections"][0]["reps"] == 2
    assert m["nBars"] == 4


def test_le_prompteur_suit_l_ordre_de_JEU_pas_l_ordre_d_ecriture():
    """Le prompteur défile dans le temps : un bloc replié y réapparaît à
    chacun de ses passages, dans l'ordre."""
    m = url_to_chart(_url("*AT44C^7XyQ]*BD-7XyQ]*AC^7XyQZ"))
    t = [c["t0"] for c in m["prompter"]["chords"]]
    assert t == sorted(t)
    assert [c["root"] for c in m["prompter"]["chords"]] == [0, 2, 0]


# ── la basse qui marche sous un accord tenu (`W/`) ──────────────────────────

def test_W_slash_est_le_meme_accord_avec_une_autre_basse():
    """`W/G♭` n'est PAS un accord de sol♭ : c'est l'accord précédent sur une
    basse de sol♭ (le refrain de *Virtual Insanity* : `E♭-7 · /G♭`). 877
    occurrences dans 154 des 2207 standards, *Round Midnight* compris."""
    m = url_to_chart(_url("T44Eb-7 W/Gb Ab-11 W/BbLZC^7XyQZ"))
    assert _grid(m["sections"][0]["bars"][:1]) == "Eb-7 Eb-7/Gb Ab-11 Ab-11/Bb"


def test_W_slash_traverse_la_barre_de_mesure():
    m = url_to_chart(_url("T44C^7XyQ|W/EXyQZ"))
    assert _grid(m["sections"][0]["bars"]) == "C^7 | C^7/E"


def test_W_slash_sans_accord_avant_ne_plante_pas():
    m = url_to_chart(_url("T44W/GXyQ|C^7XyQZ"))
    assert m["sections"][0]["bars"][0][0]["root"] == 7


# ── le chiffrage du chart est le PREMIER, pas le dernier ────────────────────

def test_une_mesure_de_2_4_au_milieu_ne_change_pas_le_chiffrage_du_chart():
    """*Virtual Insanity* est en 4/4 avec des mesures de 2/4 de passage. On
    rendait le DERNIER chiffrage vu, donc la grille entière pouvait basculer
    sur une mesure de transition."""
    m = url_to_chart(_url("T44C^7XyQ|F^7XyQ|T24G7XyQ|T44A-7XyQZ"))
    assert m["meta"]["timeSig"] == "4/4" and m["bpb"] == 4


def test_un_intro_qui_porte_des_accords_garde_ses_mesures():
    """`chart.js` réduit en lecture TOUTE section nommée « intro » à une
    bande d'une ligne (Louis, 2026-08-08, pour l'intro SANS accords). Le `*i`
    d'iReal est de la vraie musique écrite : sous ce nom, les 4 mesures
    d'intro de *Virtual Insanity* disparaissaient du chart."""
    m = url_to_chart(_url("*iT44C^7XyQ|F^7XyQ]*AD-7XyQ|G7XyQZ"))
    assert m["sections"][0]["label"] != "intro"
    assert _grid(m["sections"][0]["bars"]) == "C^7 | F^7"


def test_un_intro_VIDE_reste_un_intro():
    """L'autre moitié de la règle : sans accord à lire, la bande « intro … »
    est exactement ce qu'il faut."""
    m = url_to_chart(_url("*iT44nXyQ|nXyQ]*AD-7XyQ|G7XyQZ"))
    assert m["sections"][0]["label"] == "intro"


def test_la_pedale_porte_son_drapeau_pour_le_rendu():
    """`ch.pedal` dit au chart de n'écrire que « /G♭ », comme iReal — tout en
    gardant l'accord complet dans le modèle (pour l'analyse, l'export, la
    lecture). Le champ doit ausi traverser la liste blanche de `loadModel`."""
    m = url_to_chart(_url("T44Eb-7 W/GbLZC^7XyQZ"))
    bar = m["sections"][0]["bars"][0]
    assert bar[0].get("pedal") is not True     # l'accord écrit, lui, s'écrit
    assert bar[1]["pedal"] is True
    assert (bar[1]["root"], bar[1]["q"], bar[1]["bass"]) == (3, "-7", 6)


# ── une écriture unique par accord (Louis, 2026-09-18) ──────────────────────
# « choisis une écriture unique pour absorber les passages ». Le transcripteur
# d'iReal écrit le même accord de plusieurs façons dans le même morceau — sur
# Virtual Insanity la même mesure s'écrit `Cø7 B△7`, `Cø7 B△` et `Cø B△`, ce
# qui donnait trois sections quasi identiques pour une seule musique.

def test_deux_passages_qui_ne_different_que_par_la_7e_fusionnent():
    m = url_to_chart(_url("*AT44C^XyQ|A-XyQ]*BD-7XyQ|G7XyQ]*CC^7XyQ|A-7XyQZ"))
    assert [s["label"] for s in m["sections"]] == ["A", "B"]
    a = m["sections"][0]
    assert a["reps"] == 2
    assert [r[0] for r in a["barRanges"]] == [0, 4]


def test_la_graphie_la_plus_frequente_gagne():
    """Deux passages écrivent `C^7`, un seul écrit `C^` : c'est `C^7` qui
    s'écrit. On ne choisit pas à la place du transcripteur, on tranche son
    hésitation avec ses propres statistiques."""
    m = url_to_chart(_url(
        "*AT44C^XyQ|A-7XyQ]*BD-7XyQ|G7XyQ]"
        "*CC^7XyQ|A-7XyQ]*DD-7XyQ|G7XyQ]*EC^7XyQ|A-7XyQZ"))
    a = next(s for s in m["sections"] if s["label"] == "A")
    assert a["reps"] == 3
    assert _grid(a["bars"]) == "C^7 | A-7"


def test_a_egalite_la_premiere_graphie_gagne():
    m = url_to_chart(_url("*AT44C^XyQ|A-7XyQ]*BD-7XyQ|G7XyQ]*CC^7XyQ|A-7XyQZ"))
    a = next(s for s in m["sections"] if s["label"] == "A")
    assert _grid(a["bars"]) == "C^ | A-7"


def test_majeur_et_dominante_ne_fusionnent_PAS():
    """La famille efface la 7e et les extensions, jamais la TIERCE ni la
    fonction : `C` et `C7` restent deux accords différents."""
    m = url_to_chart(_url("*AT44CXyQ|A-7XyQ]*BD-7XyQ|G7XyQ]*CC7XyQ|A-7XyQZ"))
    assert len(m["sections"]) == 3


# ── deux passages qui ne diffèrent que sur leur queue → 1re/2e fin ──────────

def test_meme_longueur_meme_debut_queue_differente_devient_une_fin():
    m = url_to_chart(_url(
        "*AT44C^7XyQ|F^7XyQ|A-7XyQ|D-7XyQ]*BE-7XyQ|A7XyQ]"
        "*CC^7XyQ|F^7XyQ|A-7XyQ|G7XyQZ"))
    a = next(s for s in m["sections"] if s["label"] == "A")
    assert a["reps"] == 2
    e = a["endings"]
    assert e["tail"] == 1
    assert _grid(e["variants"][0]["bars"]) == "D-7"
    assert _grid(e["variants"][1]["bars"]) == "G7"
    assert [v["passes"] for v in e["variants"]] == [[0], [1]]


def test_une_queue_qui_ne_change_pas_de_fondamentale_n_est_pas_une_fin():
    """Même exigence que `folding._ireal_endings` : suspendre un crochet
    « 1./2. » pour une nuance de couleur, c'est sur-noter du bruit."""
    m = url_to_chart(_url(
        "*AT44C^7XyQ|F^7XyQ|A-7XyQ|G7XyQ]*BE-7XyQ|A7XyQ]"
        "*CC^7XyQ|F^7XyQ|A-7XyQ|G7b9XyQZ"))
    # même fondamentale (G) des deux côtés → la famille suffit à les fusionner
    a = next(s for s in m["sections"] if s["label"] == "A")
    assert a["reps"] == 2 and not a.get("endings")


def test_JAMAIS_de_fusion_entre_deux_longueurs_DIFFERENTES():
    """Louis, 2026-07-30 : « I'd rather under fold than over fold every
    time ». Un passage de 2 mesures écrit dans un bloc de 4 est un chart qui
    ment sur sa propre longueur — et le curseur de lecture y traverse la
    section à la mauvaise vitesse."""
    m = url_to_chart(_url(
        "*AT44C^7XyQ|F^7XyQ|A-7XyQ|D-7XyQ]*BC^7XyQ|F^7XyQZ"))
    assert len(m["sections"]) == 2
    assert [len(s["bars"]) for s in m["sections"]] == [4, 2]
    for s in m["sections"]:
        for b0, b1 in s["barRanges"]:
            assert b1 - b0 + 1 == len(s["bars"])


def test_des_fins_de_LONGUEURS_differentes_ne_sont_pas_des_fins():
    """iReal l'écrit (154 des 2207 standards : *A Ballad* a une 1re fin de 5
    mesures et une 2e de 6), mais `bars` d'Harmonia est UN passage entier, et
    chaque passage se joue à cette longueur-là. Les garder ensemble donnait un
    chart qui ment sur son compte de mesures, et `chart.js:endingBlocks`
    indexe `barSpans` à `prefixLen + vi*tail + j` — il se décalait aussi."""
    m = url_to_chart(_url(
        "{T44C^7XyQ|F^7XyQ|N1D-7XyQ}|N2G7XyQ|E-7XyQ|A7XyQZ"))
    assert all(not s.get("endings") for s in m["sections"])
    assert [len(s["bars"]) for s in m["sections"]] == [3, 5]
    assert m["nBars"] == 8


def test_le_compte_de_mesures_est_exactement_ce_qui_se_joue():
    """L'invariant qui aurait attrapé le bug des fins inégales : la somme des
    `reps × longueur` doit valoir `nBars`, et les passages doivent paver le
    morceau sans trou."""
    for raw in ("*A{T44C^7XyQ|F^7XyQ|N1D-7XyQ|G7XyQ}|N2D-7XyQ|C6XyQ]"
                "*BE-7XyQ|A7XyQ|D-7XyQ|G7XyQZ",
                "{T44C^7XyQ|F^7XyQ|N1D-7XyQ}|N2G7XyQ|E-7XyQZ",
                "*AT44C^XyQ|A-XyQ]*BD-7XyQ|G7XyQ]*CC^7XyQ|A-7XyQZ"):
        m = url_to_chart(_url(raw))
        assert sum(s["reps"] * len(s["bars"]) for s in m["sections"]) == m["nBars"]
        occ = sorted(r for s in m["sections"] for r in s["barRanges"])
        assert occ[0][0] == 0 and occ[-1][1] == m["nBars"] - 1
        for (_a0, a1), (b0, _b1) in zip(occ, occ[1:]):
            assert a1 + 1 == b0


def test_toutes_les_fins_d_un_bloc_ont_la_meme_longueur():
    """Le contrat que `chart.js:endingBlocks` suppose (`prefixLen + vi*tail + j`)."""
    m = url_to_chart(_url(
        "*A{T44C^7XyQ|F^7XyQ|A-7XyQ|N1D-7XyQ|G7XyQ}|N2D-7XyQ|C6XyQZ"))
    e = m["sections"][0]["endings"]
    assert len({len(v["bars"]) for v in e["variants"]}) == 1
    assert e["tail"] == len(e["variants"][0]["bars"])


# ── la demi-barre (Louis, 2026-09-18, sur Virtual Insanity) ────────────────
# « un passage à la fin qui ne fait que 2 temps, qui est noté comme du 2/4,
# en fait c'est une demi-barre et puis ça reprend, très jamiroquai — il faut
# le noter ». iReal change de chiffrage EN COURS de morceau : 698 mesures
# dans 101 des 2207 standards. On ne lisait que le premier chiffrage.

def test_une_mesure_de_2_4_dure_deux_fois_moins():
    m = url_to_chart(_url("T44C^7XyQ|F^7XyQ|T24G7XyQ|T44A-7XyQZ"))
    g = m["barGrid"]
    pleine = g[1] - g[0]
    assert abs((g[3] - g[2]) - pleine / 2) < 1e-6      # la demi-barre
    assert abs((g[4] - g[3]) - pleine) < 1e-6          # et ça reprend


def test_le_chiffrage_de_chaque_mesure_est_ecrit_dans_le_modele():
    """`sections[].meters` suit exactement l'ordre d'émission de `barSpans`
    (tronc, puis la queue de chaque variant) — c'est ce que `chart.js` lit
    avec le même curseur."""
    m = url_to_chart(_url("T44C^7XyQ|F^7XyQ|T24G7XyQ|T44A-7XyQZ"))
    s = m["sections"][0]
    assert s["meters"] == [4, 4, 2, 4]
    assert len(s["meters"]) == len(s["barSpans"])
    assert m["meta"]["meterChanges"] == 1


def test_une_demi_barre_ne_se_replie_PAS_avec_une_mesure_pleine():
    """La subtilité même : deux passages aux mêmes accords dont l'un finit
    sur 2 temps et l'autre sur 4 ne sont pas le même passage. Le repli les
    fusionnait, et la demi-barre disparaissait."""
    m = url_to_chart(_url(
        "*AT44C^7XyQ|F^7XyQ|A-7XyQ|T24G7XyQ]"
        "*BE-7XyQ|A7XyQ|D-7XyQ|G7XyQ]"
        "*CT44C^7XyQ|F^7XyQ|A-7XyQ|G7XyQZ"))
    a = next(s for s in m["sections"] if s["label"] == "A")
    # même accords, mais l'un est une demi-barre : deux fins, pas une fusion
    assert a["reps"] == 2
    assert a["endings"]["tail"] == 1
    assert a["meters"][-2:] == [2, 4]


def test_les_accords_d_une_demi_barre_tiennent_dans_ses_deux_temps():
    m = url_to_chart(_url("T44C^7XyQ|T24G7 A-7LZT44F^7XyQZ"))
    g = m["barGrid"]
    bar = m["sections"][0]["bars"][1]
    assert [c["beat"] for c in bar] == [0, 1]
    assert g[1] - 1e-6 <= bar[0]["t0"] and bar[-1]["t1"] <= g[2] + 1e-6


def test_les_temps_de_pulsation_suivent_les_chiffrages():
    """`beatTimes` ne peut plus être `nBars × bpb` : une demi-barre n'apporte
    que deux pulsations."""
    m = url_to_chart(_url("T44C^7XyQ|T24G7XyQ|T44F^7XyQZ"))
    assert len(m["beatTimes"]) == 4 + 2 + 4 + 1
    assert all(b >= a for a, b in zip(m["beatTimes"], m["beatTimes"][1:]))
