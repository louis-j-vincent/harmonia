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


# ── la piste de temps de la page karaoké (2026-09-23) ───────────────────────
# Ce qui est testé ici est la LOGIQUE, pas le modèle : quel tic sonne à
# l'instant t, ce que devient un trou, et ce que le sous-titre annonce.

from harmonia.integrations.paroles_align import (accords_du_chart,  # noqa: E402
                                                 index_courant, page, piste,
                                                 resume)


def test_index_courant_trouve_celui_qui_sonne():
    items = [{"t0": 0.0, "t1": 1.0}, {"t0": 1.0, "t1": 2.0},
             {"t0": 5.0, "t1": 6.0}]
    assert index_courant(items, 0.0) == 0
    assert index_courant(items, 0.999) == 0
    assert index_courant(items, 1.0) == 1       # la borne appartient au suivant
    assert index_courant(items, 5.5) == 2


def test_index_courant_rend_moins_un_dans_un_trou():
    """Un trou est un trou. Rendre quand même le mot le plus proche mentirait
    sur le temps — c'est exactement ce que la page sert à détecter."""
    items = [{"t0": 0.0, "t1": 1.0}, {"t0": 5.0, "t1": 6.0}]
    assert index_courant(items, 3.0) == -1      # entre les deux
    assert index_courant(items, -1.0) == -1     # avant le premier
    assert index_courant(items, 9.0) == -1      # après le dernier
    assert index_courant([], 1.0) == -1


def test_piste_separe_les_tics_des_silences():
    al = avec_trous([_m(0, 0.0, 1.0, 0.9), _m(1, 11.0, 12.0, 0.05)],
                    silence_mini=3.0)
    p = piste(al)
    assert [t["rang"] for t in p["ticks"]] == [0, 1]
    assert [t["trou"] for t in p["ticks"]] == [False, True]
    assert p["silences"] == [{"t0": 1.0, "t1": 11.0}]


def test_resume_compte_les_mots_la_confiance_et_le_plus_grand_trou():
    p = piste(avec_trous([_m(0, 0.0, 1.0, 0.9), _m(1, 2.0, 3.0, 0.1),
                          _m(2, 12.0, 13.0, 0.5)], silence_mini=3.0))
    r = resume(p["ticks"], seuil=0.3)
    assert r["n"] == 3
    assert r["median"] == 0.5
    assert r["part_douteux"] == round(1 / 3, 3)
    assert r["plus_grand_trou"] == 9.0          # de 3,0 s à 12,0 s
    assert resume([])["n"] == 0                 # sans mot, pas de division


def test_accords_du_chart_nomme_avec_la_seule_definition_du_projet():
    ch = {"prompter": {"chords": [
        {"t0": 0, "t1": 2, "root": 0, "q": "", "bass": -1, "nc": True},
        {"t0": 2, "t1": 4, "root": 7, "q": "7", "bass": 11, "nc": False}]}}
    assert accords_du_chart(ch) == [
        {"t0": 0.0, "t1": 2.0, "nom": "N.C.", "nc": True},
        {"t0": 2.0, "t1": 4.0, "nom": "G7/B", "nc": False}]


def test_le_mot_voyage_avec_son_tic_jusqua_la_page():
    """Louis, 2026-09-24 : « le mode karaoké c'est juste pour debug, tu peux
    afficher les paroles ». Le mot doit donc traverser `piste` puis `page` —
    la version d'avant le retirait, et le tic seul ne permet pas de juger LA
    SYLLABE : il peut tomber pile sur une attaque et concerner le mot d'à côté.

    Le jeton utilisé ici est inventé, pour que le test n'ait pas besoin d'un
    vrai texte pour dire ce qu'il a à dire."""
    al = [{"rang": 0, "mot": "zzgloub", "ligne": 0, "t0": 1.0, "t1": 1.4,
           "score": 0.8}]
    p = piste(avec_trous(al))
    assert p["ticks"][0]["mot"] == "zzgloub", "piste doit garder le mot"
    h = page("Un titre", "un sous-titre",
             {"audio": "/audio/x.m4a", "duree": 10, "accords": [],
              "ticks": p["ticks"], "silences": p["silences"]})
    assert "zzgloub" in h
    assert '"t0": 1.0' in h or '"t0":1.0' in h   # et le tic est toujours là


def test_un_silence_na_pas_de_mot():
    """Un trou n'est pas un mot muet : il ne doit rien porter à afficher."""
    al = [{"rang": 0, "mot": "aa", "ligne": 0, "t0": 0.0, "t1": 0.5, "score": .9},
          {"rang": 1, "mot": "bb", "ligne": 0, "t0": 9.0, "t1": 9.5, "score": .9}]
    p = piste(avec_trous(al, silence_mini=3.0))
    assert len(p["silences"]) == 1
    assert [t["mot"] for t in p["ticks"]] == ["aa", "bb"]


def test_la_page_n_utilise_pas_requestAnimationFrame():
    """Piège iOS payé cher : une boucle rAF empêche le moteur audio de WebKit
    de démarrer, et le son ne part jamais. L'horloge est `timeupdate`."""
    h = page("T", "s", {"audio": "/audio/x.m4a", "duree": 10, "accords": [],
                        "ticks": [], "silences": []})
    assert "requestAnimationFrame" not in h
    assert "timeupdate" in h and "setInterval" in h
    assert "fetch(D.audio)" in h and "createObjectURL" in h
