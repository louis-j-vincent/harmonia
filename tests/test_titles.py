"""harmonia_min.titles — artiste + titre depuis un titre YouTube.

Les cas sont pris sur les VRAIS morceaux de la bibliothèque de Louis (les titres
réels, relevés par yt-dlp le 2026-08-09), pas inventés.
"""
from __future__ import annotations

import pytest

from harmonia_min.titles import (pretty_from_slug, slugify, split, strip_junk)


# ── le nettoyage ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("Let It Be (Remastered 2009)", "Let It Be"),
    ("Georgia On My Mind (Official Video)", "Georgia On My Mind"),
    ("Hot N Cold (Official Music Video)", "Hot N Cold"),
    ("Stand By Me [Audio]", "Stand By Me"),
    ("She Will Be Loved - Official Music Video", "She Will Be Loved"),
    ("Yesterday - Remastered 2009", "Yesterday"),
    ("Blue Lights | A COLORS SHOW", "Blue Lights"),
    ("Sunny (Official Audio) HD", "Sunny"),
    ("Come Away With Me", "Come Away With Me"),
])
def test_strip_junk(raw, want):
    assert strip_junk(raw) == want


def test_un_titre_qui_est_du_bruit_survit():
    """Un morceau peut vraiment s'appeler « Live » ou « Audio ». On préfère
    rendre le titre d'avant plutôt qu'une chaîne vide."""
    assert strip_junk("Live") == "Live"
    assert strip_junk("(Official Video)") != ""


def test_le_bruit_au_milieu_reste():
    """« Live » n'est du bruit qu'à la fin ou entre parenthèses."""
    assert strip_junk("Live and Let Die") == "Live and Let Die"
    assert strip_junk("Video Killed the Radio Star") == \
        "Video Killed the Radio Star"


# ── la coupe artiste / titre ────────────────────────────────────────────────

@pytest.mark.parametrize("raw,artist,title", [
    ("Frankie Valli - Can't Take My Eyes Off You (Official Audio)",
     "Frankie Valli", "Can't Take My Eyes Off You"),
    ("Maroon 5 - She Will Be Loved (Official Music Video)",
     "Maroon 5", "She Will Be Loved"),
    ("Ben E. King - Stand By Me (Audio)", "Ben E. King", "Stand By Me"),
    ("Jorja Smith - Blue Lights | A COLORS SHOW", "Jorja Smith", "Blue Lights"),
    ("Norah Jones – Don't Know Why", "Norah Jones", "Don't Know Why"),
])
def test_split_sur_le_tiret(raw, artist, title):
    assert split(raw) == (artist, title)


def test_un_tiret_dans_un_nom_ne_coupe_pas():
    """« Jay-Z » n'est pas un séparateur : le tiret doit avoir des espaces."""
    a, t = split("Jay-Z - 99 Problems")
    assert (a, t) == ("Jay-Z", "99 Problems")


def test_le_tiret_de_remaster_nest_pas_un_separateur():
    """« Yesterday - Remastered 2009 » : la droite du tiret est du bruit pur.
    Couper là donnerait l'artiste « Yesterday » et un titre vide."""
    a, t = split("Yesterday - Remastered 2009", uploader="TheBeatlesVEVO")
    assert t == "Yesterday"
    assert a != "Yesterday"


def test_artiste_prefixe_deux_fois():
    """Cas réel de la bibliothèque : « The Beatles - The Beatles - Let It Be ».
    On ne coupe qu'au premier séparateur, donc l'artiste reste collé en tête du
    titre s'il n'est pas enlevé."""
    assert split("The Beatles - The Beatles - Let It Be (Remastered 2009)") \
        == ("The Beatles", "Let It Be")


def test_separateur_slash_des_labels():
    assert split('Mac DeMarco // "My Kind Of Woman"') \
        == ("Mac DeMarco", "My Kind Of Woman")


def test_les_balises_music_gagnent():
    """Quand YouTube Music donne artist/track, c'est de la métadonnée
    d'éditeur : elle passe avant le texte du titre."""
    assert split("BILLIE JEAN (Official Video) 4K", artist="Michael Jackson",
                 track="Billie Jean") == ("Michael Jackson", "Billie Jean")


def test_na_de_ytdlp_compte_pour_vide():
    """yt-dlp écrit littéralement « NA » quand le champ manque."""
    a, t = split("Bill Withers - Ain't No Sunshine", artist="NA", track="NA")
    assert (a, t) == ("Bill Withers", "Ain't No Sunshine")


def test_sans_separateur_on_prend_la_chaine():
    a, t = split("Bein' Green", uploader="Sesame Street")
    assert (a, t) == ("Sesame Street", "Bein' Green")


@pytest.mark.parametrize("chan,want", [
    ("TheBeatlesVEVO", "TheBeatles"),
    ("Norah Jones - Topic", "Norah Jones"),
    ("Maroon5VEVO", "Maroon5"),
    ("RHINO", "RHINO"),
])
def test_nettoyage_de_chaine(chan, want):
    assert split("Un Titre", uploader=chan)[0] == want


def test_jamais_de_coupe_inventee():
    """LE garde-fou. Sans séparateur ET sans chaîne, l'artiste reste VIDE : un
    champ vide se corrige d'un tap, un mauvais artiste se propage."""
    assert split("Georgia On My Mind") == ("", "Georgia On My Mind")


# ── le repli sur le stem ────────────────────────────────────────────────────

def test_pretty_from_slug():
    assert pretty_from_slug("ray_charles_georgia_on_my_mind_official_video") \
        == "Ray Charles Georgia On My Mind"
    assert pretty_from_slug("let_it_be_remastered_2009") == "Let It Be"


def test_pretty_from_slug_na_devine_pas_lartiste():
    """Même règle : pas de séparateur dans un slug, donc pas d'artiste."""
    assert "/" not in pretty_from_slug("bein_green")


# ── l'aller-retour qui sert de vérification au backfill ─────────────────────

@pytest.mark.parametrize("title,stem", [
    ("Ray Charles - Georgia On My Mind (Official Video)",
     "ray_charles_georgia_on_my_mind_official_video"),
    ("Katy Perry - Hot N Cold (Official Music Video)",
     "katy_perry_hot_n_cold_official_music_video"),
    ("Norah Jones - Don't Know Why", "norah_jones_don_t_know_why"),
])
def test_slugify_reproduit_les_stems_de_la_bibliotheque(title, stem):
    """C'est CETTE égalité qui autorise `backfill_titles.py` à accepter une
    vidéo trouvée par recherche : si le titre trouvé se re-slugifie en le stem
    qu'on a, c'est la même vidéo. Sinon on refuse."""
    assert slugify(title)[:60] == stem[:60]
