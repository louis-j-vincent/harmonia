"""tests/test_phrases4_lettres.py — deux sections différentes, deux lettres.

Trouvé le 2026-08-14 en branchant le bouton « Appliquer » de l'outil Soudure
sur Lost Without U : la première section (une bi-mesure isolée, type `a`) et
les huit sections `baba` du morceau sortaient TOUTES marquées **B**. Deux
choses différentes, une seule lettre — le découpage ne veut plus rien dire.

LE MÉCANISME. `nommer` attribue A à `a`, B à `baba`, C à `c`. Puis
`grouper_restes` retire leur lettre aux restes et la réattribue… par COMPTAGE :

    k = len({v for v in base.values()})      # ici {B} -> 1
    base[t] = LETTERS[k % len(LETTERS)]      # -> LETTERS[1] = "B"

Le comptage suppose que les lettres déjà posées sont A, B, C… sans trou. Elles
en ont un dès qu'un reste a emporté sa lettre avec lui : A est partie avec le
reste, il ne demeure que B, le compteur dit « une lettre posée » et repart de
la deuxième — celle qui est déjà prise.

Le bug est celui de `scripts/quatre_mots.py`, d'où ce code vient : les pages
de recherche du 2026-08-12 le portent aussi.
"""
from __future__ import annotations

from harmonia_min.phrases4 import grouper_restes, merges4, nommer, phrases

# le mot de Lost Without U, tel que l'app le calcule
MOT = "ababababababacbabababdbabababacbabababababababa"


def _par_lettre(secs):
    out: dict = {}
    for s in secs:
        out.setdefault(s["label"], set()).add(s["type"])
    return out


def test_deux_types_ne_partagent_pas_une_lettre():
    nom = nommer(merges4(MOT, 4)[-1]["jetons"], 4)
    secs = grouper_restes(nom, MOT, 4)
    collisions = {L: t for L, t in _par_lettre(secs).items() if len(t) > 1}
    assert not collisions, (
        "deux sections de contenu différent partagent une lettre : "
        f"{collisions}")


def test_phrases_ne_collisionne_pas_non_plus():
    secs, _info = phrases(MOT)
    collisions = {L: t for L, t in _par_lettre(secs).items() if len(t) > 1}
    assert not collisions, collisions


def test_les_lettres_restent_stables_pour_un_type_donne():
    """Un même contenu garde la même lettre partout dans le morceau."""
    secs, _ = phrases(MOT)
    par_type: dict = {}
    for s in secs:
        par_type.setdefault(s["type"], set()).add(s["label"])
    incoherents = {t: L for t, L in par_type.items() if len(L) > 1}
    assert not incoherents, incoherents
