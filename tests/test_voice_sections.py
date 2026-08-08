"""harmonia_min.voice_sections — la mise en forme des sections trouvées.

Deux comportements sont épinglés ici, tous deux corrigés le 2026-08-08 contre
les dix-sept morceaux annotés par Louis :

  * deux occurrences ADJACENTES d'un même bloc restent deux sections ;
  * deux lettres qui désignent la même musique n'en font qu'une.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia_min.voice_sections import merge_letters


def _ssm(n, groups, hi=0.99, lo=0.30):
    """Une SSM jouet : `hi` entre mesures du même groupe, `lo` sinon."""
    g = np.zeros(n, int)
    for k, (a, b) in enumerate(groups):
        g[a:b + 1] = k
    S = np.full((n, n), lo)
    for i in range(n):
        for j in range(n):
            if g[i] == g[j]:
                S[i, j] = hi
    return S


def _secs(spec):
    out = []
    for tok in spec.split():
        lab, rng = tok.split(":")
        a, b = rng.split("-")
        out.append({"b0": int(a), "b1": int(b), "label": lab})
    return out


def test_deux_lettres_pour_la_meme_musique_fusionnent():
    """Le cas Chain of Fools : un seul vamp, cinq lettres. Toutes les mesures
    se ressemblent, donc tout doit finir sous une seule lettre."""
    S = _ssm(32, [(0, 31)])
    out = merge_letters(S, _secs("A:0-7 B:8-15 C:16-23 D:24-31"))
    assert len({s["label"] for s in out}) == 1


def test_deux_musiques_differentes_ne_fusionnent_pas():
    S = _ssm(32, [(0, 7), (8, 15), (16, 23), (24, 31)])
    out = merge_letters(S, _secs("A:0-7 B:8-15 C:16-23 D:24-31"))
    assert len({s["label"] for s in out}) == 4


def test_liaison_moyenne_et_non_simple():
    """UNE paire de sections qui se ressemble ne doit pas recoller deux lettres
    qui, en moyenne, n'ont rien à voir. C'est l'enchaînement du lien simple, et
    il avait cassé This Love (annotation exacte -> 0,947)."""
    # A joue en 0-7 et 8-15 ; B en 16-23 et 24-31. Seules A[8-15] et B[16-23]
    # se ressemblent : une paire sur quatre.
    S = _ssm(32, [(0, 7), (8, 23), (24, 31)])
    out = merge_letters(S, _secs("A:0-7 A:8-15 B:16-23 B:24-31"))
    assert len({s["label"] for s in out}) == 2


def test_intro_et_outro_ne_sont_jamais_fusionnees():
    """Elles ne sont pas des lettres : les recoller à une section les ferait
    disparaître de la grille."""
    S = _ssm(32, [(0, 31)])
    out = merge_letters(S, _secs("intro:0-7 A:8-15 A:16-23 outro:24-31"))
    assert [s["label"] for s in out] == ["intro", "A", "A", "outro"]


def test_les_sections_trop_courtes_ne_decident_de_rien():
    """Sur deux mesures, une diagonale de similarité ne veut rien dire ; une
    telle section ne doit ni fusionner ni empêcher de fusionner."""
    S = _ssm(32, [(0, 31)])
    out = merge_letters(S, _secs("A:0-1 B:2-3"))
    assert len({s["label"] for s in out}) == 2


def test_les_lettres_restent_dans_lordre_dapparition():
    S = _ssm(32, [(0, 7), (8, 15), (16, 23), (24, 31)])
    out = merge_letters(S, _secs("A:0-7 B:8-15 C:16-23 D:24-31"))
    assert [s["label"] for s in out] == ["A", "B", "C", "D"]


# ── le départ de la chanson ─────────────────────────────────────────────────
# L'intro dure un nombre PAIR de mesures, ou exactement une (2026-08-08). Les
# deux morceaux ci-dessous sont les cas réels qu'aucun seuil de phase ne pouvait
# départager : 0,29 voulait décaler, 0,31 voulait rester.

from harmonia_min.voice_sections import sung_start                  # noqa: E402


def _song(first_sung, phase, n=40, bar=2.0):
    """Une grille régulière, du silence jusqu'à `first_sung`, une note à
    `phase` de cette mesure-là."""
    grid = [i * bar for i in range(n + 1)]
    mute = np.ones(n, bool)
    mute[first_sung:] = False
    notes = [(grid[first_sung] + phase * bar, 0.5, 60)]
    return notes, grid, mute


def test_intro_impaire_interdite_vers_le_haut():
    """Every Breath You Take : le chant entre mesure 7, tôt dans la mesure
    (0,29). La phase seule garderait 7 ; une intro de 7 mesures n'existe pas."""
    assert sung_start(*_song(7, 0.29)) == 8


def test_intro_impaire_interdite_vers_le_bas():
    """Happy : le chant entre mesure 2 à mi-mesure (0,50). La phase seule
    pousserait à 3 ; une intro de 3 mesures n'existe pas non plus."""
    assert sung_start(*_song(2, 0.50)) == 2


def test_la_phase_tranche_encore_quand_les_deux_sont_admissibles():
    """Seul cas où elle sert : le chant entre mesure 0 ou 1, et l'intro d'une
    seule mesure est admissible. Chain of Fools (0,48) contre Grenade (0,08)."""
    assert sung_start(*_song(0, 0.48)) == 1
    assert sung_start(*_song(0, 0.08)) == 0


def test_lintro_dune_seule_mesure_reste_possible():
    """Sunny : le chant entre mesure 1 dès le début (0,01). Il ne faut ni la
    pousser à 2 ni la ramener à 0."""
    assert sung_start(*_song(1, 0.01)) == 1


def test_pas_dintro_du_tout():
    assert sung_start(*_song(0, 0.05)) == 0


# ── la transposition ────────────────────────────────────────────────────────

from harmonia_min.voice_sections import _rot_sim                    # noqa: E402


def _vecs(n, pattern, shift_from=None, shift=0):
    """Un vecteur de douze hauteurs par mesure, unitaire, éventuellement
    transposé à partir de `shift_from`."""
    V = np.zeros((n, 12))
    for i in range(n):
        pc = pattern[i % len(pattern)]
        if shift_from is not None and i >= shift_from:
            pc = (pc + shift) % 12
        V[i, pc] = 1.0
    return V


def test_une_reprise_transposee_est_reconnue():
    """Le cas Sunny : la reprise rejoue la même chose un demi-ton plus haut.
    Sans rotation le cosinus est nul, avec rotation il vaut 1."""
    V = _vecs(16, [0, 5, 7, 2], shift_from=8, shift=1)
    assert _rot_sim(V, 0, 8, 8) > 0.99
    assert _diag_plain(V, 0, 8, 8) < 0.1


def _diag_plain(V, a, b, L):
    return float(np.mean([float(V[a + i] @ V[b + i]) for i in range(L)]))


def test_deux_passages_differents_ne_sont_pas_sauves_par_la_rotation():
    """Avec douze rotations à essayer, le maximum monte tout seul : le seuil
    haut doit empêcher n'importe quoi de passer pour une modulation."""
    V = np.zeros((16, 12))
    for i in range(8):
        V[i, [0, 5, 7, 2][i % 4]] = 1.0
    for i in range(8, 16):
        V[i, [3, 3, 9, 1][i % 4]] = 1.0
    assert _rot_sim(V, 0, 8, 8) < 0.5


# ── le mode par défaut ──────────────────────────────────────────────────────

def test_voice_est_le_mode_par_defaut(monkeypatch):
    """Épinglé parce que c'est un choix mesuré, pas une préférence : 0,769
    contre 0,599 pour `harmonic` sur les dix-sept morceaux annotés. Si ce
    défaut repasse à `harmonic` sans nouvelle mesure, ce test doit tomber."""
    import numpy as np
    from harmonia_min import sections as HS
    monkeypatch.delenv("HARMONIA_SECTIONS", raising=False)
    seen = {}

    def fake(grid, triad, bars, audio):
        seen["voice"] = True
        return [{"b0": 0, "b1": len(grid) - 2, "label": "A"}]

    import harmonia_min.voice_sections as VS
    monkeypatch.setattr(VS, "detect_sections", fake)
    grid = [float(i) for i in range(41)]
    HS.detect_sections(grid, None, None, bars=None,
                       triad=np.zeros((10, 36)), audio="x.m4a")
    assert seen.get("voice"), "le défaut doit être `voice`"


def test_sans_audio_le_repli_est_bruyant(monkeypatch, caplog):
    """Le repli n'est plus le choix d'un appelant curieux mais l'échec de la
    voie normale : il doit s'entendre au niveau ERROR."""
    import logging
    import numpy as np
    from harmonia_min import sections as HS
    monkeypatch.delenv("HARMONIA_SECTIONS", raising=False)
    grid = [float(i) for i in range(41)]
    with caplog.at_level(logging.ERROR):
        HS.detect_sections(grid, None, None, bars=None,
                           triad=np.zeros((10, 36)), audio=None)
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ── la parité, et son repli ─────────────────────────────────────────────────
# 2026-08-08. La grille de 2 refuse toute reprise posée à un nombre IMPAIR de
# mesures de son ancre. C'est un bon prior — Louis commence ses sections sur des
# mesures paires — mais une chanson qui gagne une mesure en route bascule toute
# sa seconde moitié sur l'autre parité, et plus rien n'y est visible. She Will
# Be Loved en est une : deux de ses reprises scorent 1,00 et 0,97 et sont jetées
# sans être regardées. Le repli n'ouvre la parité impaire qu'aux ancres qui, en
# pair, ne trouvent RIEN — donc là où il n'y a rien à casser — et lui demande
# `ODD_BONUS` de plus que le seuil.

from harmonia_min.voice_sections import (_pass, _peaks, BLOCK, FILL,  # noqa: E402
                                        ODD_BONUS, THR4, THR8)


def _two_lane(n, occ, block=BLOCK, hi=0.995, lo=0.15):
    """Une SSM où le bloc `[0, block)` se rejoue exactement en chaque `occ`."""
    S = np.full((n, n), lo)
    np.fill_diagonal(S, 1.0)
    for c in occ:
        for i in range(block):
            for j in range(block):
                if i == j:
                    S[i, c + j] = S[c + j, i] = hi
    for a in occ:
        for b in occ:
            for i in range(block):
                S[a + i, b + i] = S[b + i, a + i] = hi
    return S


def test_une_reprise_a_distance_impaire_etait_invisible():
    """Le cas She Will Be Loved : la reprise est parfaite et posée en 17, soit
    une distance IMPAIRE. La grille de 2 la refusait sans lire son score."""
    n = 40
    S = _two_lane(n, [17])
    runs = _pass(S, S, np.zeros(n, bool), n, 0, BLOCK, THR8,
                 np.zeros(n, bool))
    assert runs and 17 in runs[0]["occ"], "la reprise impaire doit être trouvée"


def test_la_parite_paire_garde_la_priorite():
    """Le repli ne doit servir QU'À défaut : quand une reprise paire existe,
    c'est elle qu'on prend, même si une impaire traîne à côté."""
    n = 48
    S = _two_lane(n, [16])
    runs = _pass(S, S, np.zeros(n, bool), n, 0, BLOCK, THR8,
                 np.zeros(n, bool))
    assert runs and runs[0]["occ"] == [16]


def test_le_repli_impair_exige_davantage_que_le_seuil():
    """Sinon il rouvrirait la porte à tout ce que la grille protégeait. Testé
    là où la constante est consommée : un pic impair posé entre `THR8` et
    `THR8 + ODD_BONUS` passe le seuil ordinaire et pas le seuil relevé."""
    n = 40
    sc = np.full(n, 0.20)
    sc[17] = THR8 + ODD_BONUS / 2
    z = np.zeros(n, bool)
    assert _peaks(sc, 0, n, THR8, BLOCK, z, par=1) == [17]
    assert _peaks(sc, 0, n, THR8 + ODD_BONUS, BLOCK, z, par=1) == []


def test_le_repli_ne_vaut_que_pour_les_blocs_de_huit():
    """Étendu à la passe de comblement, il perd Yesterday (-0,100) et Let It Be
    (-0,073) : sur une boucle de 4 mesures un bloc de 4 colle partout, et le
    pic impair n'est qu'une position à moitié décalée."""
    n = 40
    S = _two_lane(n, [17], block=FILL)
    runs = _pass(S, S, np.zeros(n, bool), n, 0, FILL, THR4,
                 np.zeros(n, bool))
    assert not any(17 in r["occ"] for r in runs)
