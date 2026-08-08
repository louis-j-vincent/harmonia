"""La règle d'or de Louis (2026-08-08), rendue exécutable.

    « La représentation doit être la plus compacte possible, MAIS toutes les
      variations de sections doivent être représentées — donc si on affiche 2
      fois les 8 mêmes mesures, c'est un loupé, on peut compacter. »

Deux propriétés, et la seconde n'est pas négociable :

  (1) COMPACITÉ — aucune suite de mesures n'est dessinée deux fois sur la page.
  (2) RECONSTRUCTION — la forme minimale rejoue EXACTEMENT la suite d'accords
      d'origine, mesure par mesure.

(2) est le garde-fou contre la sur-compression. Sans lui, « compact » finit par
vouloir dire « faux » : le repli livré écrivait 4 mesures × 3 pour un A qui joue
22 / 8 / 20 mesures, 13 mesures ne parvenant jamais au chart. Un test de
reconstruction l'aurait attrapé le jour même.

Les tests marqués `corpus` lisent le cache de `scripts/minimal_form_capture.py`
et se sautent tout seuls s'il est absent (`python scripts/minimal_form_capture.py`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from harmonia_min import minimal_form as MF                          # noqa: E402

REFERENCE = ["bein_green", "maroon_5_this_love", "norah_jones_don_t_know_why",
             "let_it_be_remastered_2009", "ben_e_king_stand_by_me_audio"]
_ROOTS = {n: i for i, n in
          enumerate("C Db D Eb E F Gb G Ab A Bb B".split())}
_ROOTS.update({n: i for i, n in
               enumerate("C C# D D# E F F# G G# A A# B".split())})


# ── de quoi écrire une grille à la main ─────────────────────────────────────

def bars(spec: str):
    """« C | A- | F | G7 » -> des mesures au format ChartModel.

    Une mesure par `|`. Deux accords dans la même mesure : « F G ». Une tenue :
    « C~ ». `//` sépare les phrases, uniquement pour la lecture.
    """
    out = []
    for i, cell in enumerate(spec.replace("//", "|").split("|")):
        bar = []
        for beat, tok in enumerate(cell.split()):
            carry = tok.endswith("~")
            tok = tok.rstrip("~")
            if tok == "N.C.":
                bar.append({"root": 0, "q": "", "bass": -1, "nc": True,
                            "beat": beat, "carry": carry, "bar": i})
                continue
            k = 2 if tok[1:2] in ("b", "#") else 1
            bar.append({"root": _ROOTS[tok[:k]], "q": tok[k:], "bass": -1,
                        "nc": False, "beat": beat, "carry": carry, "bar": i})
        out.append(bar)
    return out


def sig(spec: str):
    return MF.signatures(bars(spec))


def one_letter(label: str, *specs: str):
    """Un ChartModel minimal : une lettre, une section par spec."""
    allbars, sections, b = [], [], 0
    for s in specs:
        bb = bars(s)
        sections.append({"label": label, "barRanges": [[b, b + len(bb) - 1]]})
        allbars += bb
        b += len(bb)
    return sections, allbars


# ── (2) RECONSTRUCTION — le garde-fou, non négociable ───────────────────────

def test_reconstruction_exacte_boucle_simple():
    """4 mesures qui bouclent : la forme les rejoue mesure pour mesure."""
    secs, bb = one_letter("A", "C | A- | F | G | C | A- | F | G")
    form = MF.compress_song(secs, bb)
    assert MF.check_reconstruction(form, secs, bb) == []


def test_reconstruction_exacte_avec_variation_de_queue():
    secs, bb = one_letter("A", "C | A- | F | G | C | A- | F | G7 C")
    form = MF.compress_song(secs, bb)
    assert MF.check_reconstruction(form, secs, bb) == []


def test_reconstruction_exacte_occurrences_de_longueurs_differentes():
    """Under-fold : 4 mesures puis 8 mesures, rien n'est étiré ni perdu."""
    secs, bb = one_letter("A", "C | A- | F | G", "C | A- | F | G | C | A- | F | G")
    form = MF.compress_song(secs, bb)
    assert MF.check_reconstruction(form, secs, bb) == []
    rec = MF.reconstruct(form.letters[0])
    assert [len(x) for x in rec] == [4, 8]


def test_reconstruction_exacte_section_sans_boucle():
    """Une section qui ne boucle pas est écrite au long — et se relit telle."""
    secs, bb = one_letter("A", "C | Eb | Gb | A | D- | B | F#7 | Bb-")
    form = MF.compress_song(secs, bb)
    assert MF.check_reconstruction(form, secs, bb) == []
    assert form.written == form.played          # rien à gagner, rien de perdu


def test_reconstruction_attrape_une_sur_compression():
    """Le garde-fou doit ÊTRE capable d'échouer : on casse la forme exprès.

    Sans ce test, `check_reconstruction` pourrait renvoyer [] par construction
    et les autres tests ne prouveraient rien.
    """
    secs, bb = one_letter("A", "C | A- | F | G | C | A- | F | G7 C")
    form = MF.compress_song(secs, bb)
    assert MF.check_reconstruction(form, secs, bb) == []
    blk = form.letters[0].blocks[0]
    blk.patches = []                                   # on jette la variation
    blk.renditions = [MF.Rendition(r.b0, r.occ, ()) for r in blk.renditions]
    assert MF.check_reconstruction(form, secs, bb) == ["A"]


# ── (1) COMPACITÉ — la règle d'or ───────────────────────────────────────────

def test_deux_passes_identiques_ne_sont_ecrites_qu_une_fois():
    """« Si on affiche 2 fois les 8 mêmes mesures, c'est un loupé. »"""
    huit = "C | A- | F | G | E- | A7 | D-7 | G7"
    secs, bb = one_letter("A", huit, huit)
    form = MF.compress_song(secs, bb)
    assert form.written == 8 and form.played == 16
    assert MF.duplicate_units(form) == []


def test_aucun_doublon_meme_avec_des_variations():
    secs, bb = one_letter("A", "C | A- | F | G | C | A- | F | G7 C",
                          "C | A- | F | G | C | A- | D-7 | G7")
    form = MF.compress_song(secs, bb)
    assert MF.duplicate_units(form) == []


def test_deux_lettres_qui_bouclent_pareil_ne_sont_dessinees_qu_une_fois():
    """A et B sur la même boucle, fins différentes : la boucle est écrite 1 fois."""
    quatre = "C | A- | F | G"
    secs = [{"label": "A", "barRanges": [[0, 3]]},
            {"label": "A", "barRanges": [[4, 7]]},
            {"label": "B", "barRanges": [[8, 11]]},
            {"label": "B", "barRanges": [[12, 15]]}]
    bb = bars(f"{quatre} | {quatre} | {quatre} | C | A- | F | G7 Db")
    form = MF.compress_song(secs, bb)
    assert MF.duplicate_units(form) == []
    assert [b.alias_of for lf in form.letters for b in lf.blocks] == [None, "A"]
    assert MF.check_reconstruction(form, secs, bb) == []


# ── ce que Louis a nommé : les trois formes attendues ───────────────────────

def test_quatre_mesures_qui_bouclent():
    secs, bb = one_letter("A", "C | A- | F | G | C | A- | F | G | C | A- | F | G")
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    assert blk.period == 4 and len(blk.renditions) == 3 and blk.patches == []


def test_quatre_mesures_qui_bouclent_les_deux_dernieres_varient():
    secs, bb = one_letter(
        "A", "C | A- | F | G | C | A- | F | G | C | A- | D-7 | G7")
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    assert blk.period == 4
    assert len(blk.patches) == 1
    p = blk.patches[0]
    assert p.pos == 2 and len(p.bars) == 2 and p.is_ending(blk.period)
    assert blk.renditions[2].patches == (0,)


def test_huit_mesures_qui_bouclent_les_deux_dernieres_varient():
    huit = "C | A- | F | G | E- | A7 | D-7 | G7"
    fin = "C | A- | F | G | E- | A7 | Db7 | C^7"
    secs, bb = one_letter("A", huit, huit, fin)
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    assert blk.period == 8
    assert len(blk.patches) == 1 and blk.patches[0].pos == 6
    assert blk.patches[0].is_ending(8)
    assert blk.written == 10 and blk.played == 24


def test_une_variation_au_milieu_est_representee_aussi():
    """Les variations réelles ne tombent pas toutes en fin de section."""
    secs, bb = one_letter("A", "C | A- | F | G | C | Eb7 | F | G")
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    assert blk.period == 4
    assert [(p.pos, len(p.bars)) for p in blk.patches] == [(1, 1)]
    assert not blk.patches[0].is_ending(4)


# ── under-fold, never over-fold ─────────────────────────────────────────────

def test_une_occurrence_etrangere_sort_dans_son_propre_bloc():
    """Elle est écrite à la longueur qu'elle joue, pas coulée dans le moule."""
    secs, bb = one_letter("A", "C | A- | F | G", "C | A- | F | G",
                          "Eb | Ab | Db | Gb")
    form = MF.compress_song(secs, bb)
    lf = form.letters[0]
    assert len(lf.blocks) == 2
    assert lf.blocks[0].period == 4 and len(lf.blocks[0].renditions) == 2
    assert lf.blocks[1].period == 0            # écrite au long
    assert MF.check_reconstruction(form, secs, bb) == []


def test_une_boucle_vue_une_seule_fois_n_en_est_pas_une():
    secs, bb = one_letter("A", "C | A- | F | G")
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    assert blk.period == 0 and blk.written == 4


# ── les dégénérescences mesurées le 2026-08-08 (tests rouges d'abord) ───────

def test_une_cellule_d_une_mesure_n_est_pas_une_forme():
    """Bein Green sortait en cellule `Bb^7` + quinze variantes d'une mesure.

    Moins de mesures écrites que la lecture en 8 mesures, et pourtant illisible :
    ni période, ni reprise. La cellule ne rendait que 5 des 32 mesures.
    """
    secs, bb = one_letter("A", "Bb^7 | A7 | Ab7 | G7 | C-7 | F7 | Bb^7 | F7",
                          "Bb^7 | A7 | Ab7 | G7 | C-7 | F7 | Bb^7 | Bb^7")
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    assert blk.period == 8
    assert blk.period != 1


def test_une_cellule_qui_boucle_sur_du_silence_est_refusee():
    """Une « boucle » d'une mesure n'est admise que si elle est vraiment tenue."""
    secs, bb = one_letter("A", "C~ | C~ | C~ | C~ | C~ | C~")
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    assert blk.period == 1 and blk.patches == [] and len(blk.renditions) == 6
    assert blk.written == 1


def test_un_litteral_repete_coute_a_chaque_fois():
    """La page redessine un littéral à chaque endroit : il doit le payer.

    Le compter une fois rendait GRATUITE une mauvaise période — Let It Be
    sortait en boucle de 2 mesures avec `G | A- F` écrit huit fois (son couplet
    est une boucle de 4), Stand By Me en boucle de 6 avec `A | A~` écrit neuf
    fois (la boucle fait 8). Le garde-fou vit ici et dans
    `test_corpus_pas_de_litteral_redessine`.
    """
    deux = MF.signatures(bars("Eb | Db"))
    blk = MF.Block(2, tuple(MF.signatures(bars("C | A-"))), [],
                   [MF.Rendition(0, 0)],
                   [MF.Literal(2, 0, tuple(deux)),
                    MF.Literal(6, 0, tuple(deux))])
    assert blk.written == 2 + 2 + 2


def test_la_boucle_ecrite_est_jouee_telle_quelle_au_moins_une_fois():
    secs, bb = one_letter("A", "C | A- | F | G | C | A- | F | G | C | A- | D- | G")
    blk = MF.compress_song(secs, bb).letters[0].blocks[0]
    plain = [r for r in blk.renditions if not r.patches]
    assert plain, "la cellule dessinée doit s'entendre au moins une fois"


# ── le corpus : les deux règles sur les vraies chansons ─────────────────────

def _capture(stem):
    pytest.importorskip("numpy")
    from minimal_form_capture import CACHE
    if not (CACHE / f"{stem}.pkl").exists():
        pytest.skip(f"cache absent — lancer scripts/minimal_form_capture.py")
    import pickle
    with open(CACHE / f"{stem}.pkl", "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("stem", REFERENCE)
@pytest.mark.parametrize("loose", [False, True])
def test_corpus_reconstruction_exacte(stem, loose):
    d = _capture(stem)
    form = MF.compress_song(d["sections"], d["bars"], loose=loose)
    assert MF.check_reconstruction(form, d["sections"], d["bars"],
                                   loose=loose) == []


@pytest.mark.parametrize("stem", REFERENCE)
def test_corpus_aucun_bloc_dessine_deux_fois(stem):
    """Aucune suite de 2 mesures ou plus n'apparaît deux fois sur la page."""
    d = _capture(stem)
    form = MF.compress_song(d["sections"], d["bars"])
    dups = [w for u, w in MF.duplicate_units(form) if len(u) >= 2]
    assert dups == []


@pytest.mark.parametrize("stem", REFERENCE)
def test_corpus_aucune_boucle_ecrite_deux_fois(stem):
    """Deux blocs ne peuvent pas dessiner la même boucle (alias obligatoire)."""
    d = _capture(stem)
    form = MF.compress_song(d["sections"], d["bars"])
    seen = set()
    for lf in form.letters:
        for blk in lf.blocks:
            if not blk.cell or blk.alias_of is not None:
                continue
            assert blk.cell not in seen, f"{lf.label} redessine une boucle"
            seen.add(blk.cell)


@pytest.mark.parametrize("stem", REFERENCE)
def test_corpus_pas_de_litteral_redessine(stem):
    """Un même littéral posé deux fois dans un bloc = la période est fausse.

    C'est le symptôme qui a désigné les deux bugs du 2026-08-08 : Let It Be
    écrivait `G | A- F` huit fois, Stand By Me `A | A~` neuf fois.
    """
    d = _capture(stem)
    form = MF.compress_song(d["sections"], d["bars"])
    for lf in form.letters:
        for blk in lf.blocks:
            if not blk.period:
                continue                        # écrit au long : un seul bloc
            got = [l.bars for l in blk.literals]
            assert len(got) == len(set(got)), f"{lf.label} redessine un littéral"


@pytest.mark.parametrize("stem", REFERENCE)
def test_corpus_la_forme_comprime_vraiment(stem):
    """Garde-fou de non-régression : ces cinq morceaux tiennent sous 0,60."""
    d = _capture(stem)
    form = MF.compress_song(d["sections"], d["bars"])
    assert form.ratio <= 0.60, f"{stem}: {form.written}/{form.played}"
