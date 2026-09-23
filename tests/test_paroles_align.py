"""tests/test_paroles_align.py — le placement des mots sur l'audio.

Ces tests ne chargent PAS le modèle : ils portent sur ce qui l'entoure — le
texte qu'on lui donne et les trous qu'on rend. La partie modèle est mesurée sur
de vrais morceaux, pas simulée ici.
"""
from __future__ import annotations

from harmonia.integrations.paroles_align import (SCORE_DOUTEUX, _normalise,
                                                 avec_trous,
                                                 lignes_de_paroles,
                                                 mots_a_aligner)

TAB = (
    "[Verse 1]\n"
    "[ch]C[/ch]      [ch]Am[/ch]\n"
    "premiere ligne ici\n"
    "\n"
    "[Chorus]\n"
    "[ch]F[/ch]   [ch]G[/ch]\n"
    "deuxieme ligne la\n"
)


def test_on_ecarte_les_accords_et_les_entetes():
    """Une ligne d'accords et un `[Verse]` ne sont pas des mots chantés."""
    lignes = lignes_de_paroles(TAB)
    assert [l["texte"] for l in lignes] == ["premiere ligne ici",
                                            "deuxieme ligne la"]
    assert [l["ligne"] for l in lignes] == [2, 6]


def test_le_mot_est_normalise_pour_le_vocabulaire_du_modele():
    """Le vocabulaire est A-Z plus l'apostrophe : tout le reste doit tomber
    AVANT l'alignement, sinon la trame n'a aucune façon d'être expliquée."""
    assert _normalise("Été") == "ETE"
    assert _normalise("don't") == "DON'T"
    assert _normalise("l'âme") == "L'AME"
    assert _normalise("123") == ""


def test_chaque_mot_garde_sa_ligne_et_son_rang():
    mots = mots_a_aligner(lignes_de_paroles(TAB))
    assert [m["mot"] for m in mots] == ["premiere", "ligne", "ici",
                                        "deuxieme", "ligne", "la"]
    assert [m["rang"] for m in mots] == [0, 1, 2, 3, 4, 5]
    assert [m["ligne"] for m in mots] == [2, 2, 2, 6, 6, 6]


# ── les trous (Louis, 2026-09-23) ────────────────────────────────────────────

def _m(rang, t0, t1, score):
    return {"rang": rang, "mot": f"m{rang}", "ligne": 0, "t0": t0, "t1": t1,
            "score": score}


def test_un_mot_au_score_effondre_est_marque_douteux():
    """L'alignement forcé ne sait pas dire non : il PLACE tout ce qu'on lui
    donne. Le score est la seule alarme, donc elle doit ressortir."""
    al = avec_trous([_m(0, 0.0, 1.0, 0.9), _m(1, 1.0, 2.0, 0.05)])
    assert [x["trou"] for x in al] == [False, True]


def test_un_long_silence_devient_un_trou_INSERE():
    """Entre deux couplets il n'y a pas de mot : le lecteur doit se taire, pas
    surligner le dernier mot pendant dix secondes."""
    al = avec_trous([_m(0, 0.0, 1.0, 0.9), _m(1, 11.0, 12.0, 0.9)],
                    silence_mini=3.0)
    assert len(al) == 3
    assert al[1]["silence"] is True and al[1]["mot"] is None
    assert (al[1]["t0"], al[1]["t1"]) == (1.0, 11.0)


def test_un_silence_court_ne_coupe_rien():
    al = avec_trous([_m(0, 0.0, 1.0, 0.9), _m(1, 2.0, 3.0, 0.9)],
                    silence_mini=3.0)
    assert len(al) == 2 and all(not x.get("silence") for x in al)


def test_le_seuil_par_defaut_est_celui_du_module():
    al = avec_trous([_m(0, 0.0, 1.0, SCORE_DOUTEUX - 0.01)])
    assert al[0]["trou"] is True
