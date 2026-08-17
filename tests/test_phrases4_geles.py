"""tests/test_phrases4_geles.py — une section tracée à la main est un MUR.

Trouvé le 2026-08-17 en branchant le bouton « Valider les sections » de l'outil
du chart sur Let It Be : Louis envoie ses **15** sections annotées à la main,
il en revient **une seule** sous son nom, les six autres portent des lettres de
l'algorithme.

LE MÉCANISME. `sections_inferer` passait ses traits à `phrases(depart=…)`, et
`merges4` reprend l'agglomération À PARTIR de ces jetons — donc il continue de
les souder entre eux dès que la paire se répète et que la longueur tient sous
`cible`. Son couplet de 4 mots et son refrain de 2 mots font 6 : à `cible=6` ils
fusionnent en un seul jeton, la frontière qu'il venait de tracer disparaît, le
contenu soudé ne correspond plus à aucune de ses sections et le bloc repart avec
une lettre neuve.

C'est la bonne règle pour l'outil Soudure, dont les soudures à la main sont des
morceaux de section à prolonger. C'est la mauvaise pour l'outil Sections, où un
trait est une section ENTIÈRE : elle ne se prolonge pas, elle se garde.
"""
from __future__ import annotations

from harmonia_min.phrases4 import merges4, phrases

# quatre mots, puis deux, répétés : la forme couplet/refrain la plus banale.
MOT = "abcdefabcdefabcdef"


def _spans(toks):
    return {(j0, j1) for j0, j1, _t in toks}


def test_sans_gel_lalgo_soude_par_dessus_le_trait():
    """Le comportement d'origine, celui de Soudure — inchangé."""
    depart = [(0, 4, MOT[0:4]), (4, 6, MOT[4:6]),
              (6, 10, MOT[6:10]), (10, 12, MOT[10:12]),
              (12, 16, MOT[12:16]), (16, 18, MOT[16:18])]
    fin = merges4(MOT, cible=6, depart=depart)[-1]["jetons"]
    assert (0, 6) in _spans(fin), (
        "sans gel, les deux traits voisins doivent bien fusionner "
        f"(c'est la règle de Soudure) — obtenu {_spans(fin)}")


def test_un_jeton_gele_ne_fusionne_jamais():
    depart = [(0, 4, MOT[0:4]), (4, 6, MOT[4:6]),
              (6, 10, MOT[6:10]), (10, 12, MOT[10:12]),
              (12, 16, MOT[12:16]), (16, 18, MOT[16:18])]
    geles = {(j0, j1) for j0, j1, _ in depart}
    fin = merges4(MOT, cible=6, depart=depart, geles=geles)[-1]["jetons"]
    assert _spans(fin) == geles, (
        f"les traits gelés ont bougé : {_spans(fin)} != {geles}")


def test_phrases_rend_a_chaque_trait_sa_section():
    depart = [(0, 4, MOT[0:4]), (4, 6, MOT[4:6]),
              (6, 10, MOT[6:10]), (10, 12, MOT[10:12]),
              (12, 16, MOT[12:16]), (16, 18, MOT[16:18])]
    geles = {(j0, j1) for j0, j1, _ in depart}
    secs, _info = phrases(MOT, depart=depart, geles=geles)
    assert [(s["j0"], s["j1"]) for s in secs] == sorted(geles), (
        "chaque trait doit rester une section à lui seul, "
        f"obtenu {[(s['j0'], s['j1']) for s in secs]}")
    # et un trait plus court que la cible n'est PAS une queue : c'est une
    # section qu'il a affirmée, elle ne part pas dans grouper_restes.
    assert not any(s["queue"] for s in secs), (
        f"un trait gelé est sorti en queue : {secs}")
