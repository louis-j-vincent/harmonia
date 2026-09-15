"""harmonia.roles — l'équivalence de deux accords dans une cadence.

Les exemples de Louis (2026-09-15) d'abord, en C majeur : G7 ≡ F/G (= Gsus)
≡ G7b9 ≡ Db7. Puis chaque règle du module, et ce qu'elle ne doit PAS
confondre.
"""
from harmonia.roles import equivalent, role, role_name, parse_key_name, pcs

C = (0, "major")


def ch(root, q="", bass=-1):
    return {"root": root, "q": q, "bass": bass, "nc": False}


G7, FsurG, G7b9, Db7 = ch(7, "7"), ch(5, "", 7), ch(7, "7b9"), ch(1, "7")


def test_louis_deux_cinq_un_en_c():
    assert equivalent(G7, FsurG, C)
    assert equivalent(G7, G7b9, C)
    assert equivalent(G7, Db7, C)
    assert equivalent(FsurG, Db7, C)
    assert role_name(role(G7, C), C) == "V (Db7 / G7)"


def test_regle_1_la_couleur_ne_change_pas_le_role():
    assert equivalent(ch(0), ch(0, "^7"))
    assert equivalent(ch(0), ch(0, "6"))
    assert equivalent(ch(2, "-"), ch(2, "-7"))
    assert equivalent(ch(7, "7"), ch(7, "13"))
    assert equivalent(ch(7, "7"), ch(7, "sus4"))
    assert equivalent(ch(7, "7"), ch(7, "7sus4"))
    assert not equivalent(ch(0), ch(0, "-"))          # majeur ≠ mineur
    assert not equivalent(ch(0), ch(7, "7"))          # I ≠ V


def test_regle_2_la_basse_etrangere_fait_la_fondamentale():
    assert equivalent(ch(5, "", 7), ch(7, "7sus4"))   # F/G = G7sus4
    assert equivalent(ch(2, "-7", 7), ch(7, "7"))     # D-7/G = G11 ≡ G7
    assert equivalent(ch(0, "", 4), ch(0))            # C/E : renversement
    assert equivalent(ch(2, "-", 5), ch(2, "-"))      # D-/F : renversement
    assert not equivalent(ch(5, "", 7), ch(5))        # F/G n'est pas un F


def test_regle_3_le_triton_fait_la_dominante():
    assert equivalent(ch(7, "7"), ch(1, "7"))         # G7 ≡ Db7
    assert equivalent(ch(11, "o7"), ch(7, "7"))       # Bo7 ≡ G7(b9)
    assert equivalent(ch(11, "o"), ch(7, "7"))        # Bo ≡ G7
    assert not equivalent(ch(7, "7"), ch(0, "7"))     # G7 ≠ C7 (autre triton)


def test_regle_4a_le_v_majeur_est_une_dominante_avec_la_tonalite():
    assert equivalent(ch(7), ch(7, "7"), C)
    assert not equivalent(ch(7), ch(7, "7"))          # sans tonalité : on ne devine pas


def test_regle_4b_le_demi_diminue_sur_la_sensible_est_un_v9():
    Eb_min = (3, "minor")
    assert equivalent(ch(2, "h7"), ch(10, "7"), Eb_min)      # Dh7 ≡ Bb7 en Eb mineur
    assert not equivalent(ch(2, "h7"), ch(10, "7"))          # sans tonalité, non
    assert not equivalent(ch(2, "h7"), ch(7, "7"), (0, "minor"))   # en C mineur, Dh7 est le iiø


def test_regle_4c_la_septieme_sur_le_i_est_une_couleur():
    A = (9, "major")
    assert equivalent(ch(9, "7"), ch(9, "^7"), A)     # A7 ≡ A^7 sur le I de The Walk
    assert equivalent(ch(9, "7"), ch(9), A)
    assert not equivalent(ch(9, "7"), ch(9, "^7"))    # sans tonalité, non
    assert not equivalent(ch(0, "7"), ch(0, "^7"), (5, "major"))   # C7 en F = V, pas I


def test_memes_notes_meme_accord():
    assert equivalent(ch(2, "-7"), ch(5, "6"))        # D-7 = F6
    assert equivalent(ch(9, "-7"), ch(0, "6"))        # A-7 = C6
    assert equivalent(ch(2, "h7"), ch(5, "-6"))       # Dh7 = F-6
    assert pcs(ch(2, "-7")) == pcs(ch(5, "6"))


def test_nc_et_tonalite():
    nc = {"root": 0, "q": "", "bass": -1, "nc": True}
    assert equivalent(nc, dict(nc))
    assert not equivalent(nc, ch(0))
    assert parse_key_name("Eb major") == (3, "major")
    assert parse_key_name("G# minor") == (8, "minor")
    assert parse_key_name(None) is None
