"""harmonia.relabel — λ1 t+b : chaque accord re-nommé par sa différence aux
deux accords voisins (Louis, 2026-10-12).

Postérieures synthétiques, vocabulaire et lecture d'étiquette du décodeur
vendu (réels). Le cas de prod mesuré sur 4 morceaux est dans
docs/known_issues.md (2026-10-12).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from harmonia import musx
from harmonia.musx import FRAME_DT
from harmonia.relabel import frame_span, label_for_span, relabel_segments
from harmonia.settings import SETTINGS

N = 300


def _col(root, typ=0):
    """Colonne triade musx : 1 + racine + 12 × type (0 maj, 1 min)."""
    return 1 + root % 12 + 12 * typ


def _probs(plan):
    """plan : [(f0, f1, {col triade: masse}, {pc basse: masse})] → 6 plans."""
    tri = np.full((N, 73), 1e-3)
    bas = np.full((N, 13), 1e-3)
    for f0, f1, t, b in plan:
        for c, m in t.items():
            tri[f0:f1, c] = m
        for pc, m in b.items():
            bas[f0:f1, 1 + pc] = m
    tri /= tri.sum(1, keepdims=True)
    bas /= bas.sum(1, keepdims=True)

    def ext(k):
        e = np.full((N, k), 0.02)
        e[:, 0] = 1.0
        return e / e.sum(1, keepdims=True)
    return [tri, bas, ext(4), ext(4), ext(3), ext(3)]


def _t(f):
    return f * FRAME_DT


G, Bb, C, F = 7, 10, 0, 5


# ── règle 1 : l'hypothèse porteuse, contre la constante amont ──────────────

def test_frame_dt_est_celui_du_clone_vendu():
    """FRAME_DT = HOP/SR du clone (settings.py amont), pas un nombre recopié."""
    src = (SETTINGS.musx_dir / "settings.py").read_text()
    sr = int(re.search(r"^DEFAULT_SR\s*=\s*(\d+)", src, re.M).group(1))
    hop = int(re.search(r"^DEFAULT_HOP_LENGTH\s*=\s*(\d+)", src, re.M).group(1))
    assert (musx.MUSX_SR, musx.MUSX_HOP) == (sr, hop)
    assert FRAME_DT == hop / sr


def test_un_segment_du_viterbi_retombe_sur_ses_trames_exactes():
    """Les frontières de `_tags_to_lab` sont des multiples de FRAME_DT ;
    `frame_span` doit rendre exactement les trames qui portaient l'étiquette."""
    tags = ["A"] * 37 + ["B"] * 101 + ["C"] * 5
    lab = musx._tags_to_lab(tags)
    spans = [frame_span(t0, t1, len(tags)) for t0, t1, _ in lab]
    assert spans == [(0, 37), (37, 138), (138, 143)]


# ── la règle ────────────────────────────────────────────────────────────────

def _trois(milieu_tri, milieu_bas, vois_tri, vois_bas, lab_milieu="G:min"):
    probs = _probs([(0, 100, vois_tri, vois_bas), (100, 200, milieu_tri, milieu_bas),
                    (200, 300, vois_tri, vois_bas)])
    segs = [(_t(0), _t(100), "G:min"), (_t(100), _t(200), lab_milieu),
            (_t(200), _t(300), "G:min")]
    return probs, segs


def test_la_moyenne_dit_gm_le_contraste_revele_bb():
    """Gm résonne des deux côtés ; au milieu Gm et B♭ sont presque à égalité.
    La moyenne seule lit Gm, la différence aux voisines lit B♭."""
    probs, segs = _trois({_col(G, 1): 0.45, _col(Bb): 0.40}, {G: 0.45, Bb: 0.45},
                         {_col(G, 1): 0.85, _col(Bb): 0.05}, {G: 0.9})
    zero, _ = relabel_segments(segs, probs, lam=0.0)
    assert zero[1][2] == "G:min"
    out, rep = relabel_segments(segs, probs, lam=1.0)
    assert out[1][2] == "Bb:maj"
    assert rep["changed"] == 1 and rep["pairs"] == [["G:min", "Bb:maj"]]
    # les frontières ne bougent jamais
    assert [(a, b) for a, b, _ in out] == [(a, b) for a, b, _ in segs]


def test_un_accord_tenu_entre_deux_autres_garde_son_nom():
    probs = _probs([(0, 100, {_col(F): 0.8}, {F: 0.8}),
                    (100, 200, {_col(C): 0.7, _col(F): 0.1}, {C: 0.8}),
                    (200, 300, {_col(G): 0.8}, {G: 0.8})])
    segs = [(_t(0), _t(100), "F:maj"), (_t(100), _t(200), "C:maj"),
            (_t(200), _t(300), "G:maj")]
    out, rep = relabel_segments(segs, probs, lam=1.0)
    assert [s[2] for s in out] == ["F:maj", "C:maj", "G:maj"]
    assert rep["changed"] == 0


def test_bord_une_seule_voisine_comptee_une_fois():
    """1er accord : S = p − p_voisine (pas 2×). Ici compter la voisine deux
    fois ferait basculer X → Y ; une fois, X reste."""
    X, Y = _col(C), _col(Bb)
    probs = _probs([(0, 100, {X: 0.5, Y: 0.3}, {C: 0.8}),
                    (100, 200, {X: 0.55, Y: 0.2}, {C: 0.8})])
    segs = [(_t(0), _t(100), "C:maj"), (_t(100), _t(200), "C:maj")]
    out, _ = relabel_segments(segs, probs, lam=1.0)
    assert out[0][2] == "C:maj"
    # et le même calcul, voisine doublée à la main, aurait basculé :
    double = label_for_span(probs, (_t(0), _t(100)), (_t(100), _t(200)),
                            (_t(100), _t(200)), lam=1.0)
    assert double["root"] == Bb


def test_nc_jamais_renomme_et_jamais_voisine():
    probs, segs = _trois({_col(G, 1): 0.45, _col(Bb): 0.40}, {G: 0.45, Bb: 0.45},
                         {_col(G, 1): 0.85, _col(Bb): 0.05}, {G: 0.9})
    probs[0][0:100, :] = 0.0
    probs[0][0:100, 0] = 1.0                          # 1er segment : silence
    segs[0] = (segs[0][0], segs[0][1], "N")
    out, rep = relabel_segments(segs, probs, lam=1.0)
    assert out[0][2] == "N"
    # le N.C. compte comme absent : même nom que sans lui (une voisine, à droite)
    seul = label_for_span(probs, (_t(100), _t(200)), None, (_t(200), _t(300)), lam=1.0)
    assert out[1][2] == seul["label"]
    assert rep["n"] == 2


def test_un_accord_n_est_jamais_renomme_en_nc():
    """Plus de N au milieu que chez les voisines : on ne retire pas l'accord."""
    probs = _probs([(0, 100, {_col(F): 0.9}, {F: 0.9}),
                    (100, 200, {0: 0.5, _col(C): 0.3}, {C: 0.6}),
                    (200, 300, {_col(G): 0.9}, {G: 0.9})])
    segs = [(_t(0), _t(100), "F:maj"), (_t(100), _t(200), "C:maj"),
            (_t(200), _t(300), "G:maj")]
    out, _ = relabel_segments(segs, probs, lam=1.0)
    assert out[1][2] != "N"


def test_label_for_span_rend_un_top3_au_format_du_chart():
    probs, _ = _trois({_col(G, 1): 0.45, _col(Bb): 0.40}, {G: 0.45, Bb: 0.45},
                      {_col(G, 1): 0.85, _col(Bb): 0.05}, {G: 0.9})
    r = label_for_span(probs, (_t(100), _t(200)), (_t(0), _t(100)), (_t(200), _t(300)))
    assert (r["root"], r["q"], r["bass"]) == (Bb, "", -1)
    assert len(r["top"]) == 3 and r["top"][0]["label"] == r["label"]
    assert r["voisines"] == 2 and r["lam"] == SETTINGS.relabel_lambda
    assert all(t["label"] != "N" for t in r["top"])


# ── la route de l'écran Annotate ────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from harmonia.server import app as app_mod
    from harmonia.server.routes import annotate as an_route
    charts, audio = tmp_path / "charts", tmp_path / "audio"
    charts.mkdir(); audio.mkdir()
    (charts / "min_essai.json").write_text(json.dumps({"audio_url": "/audio/essai.m4a"}))
    (audio / "essai.m4a").write_bytes(b"x")
    monkeypatch.setattr(an_route, "CHARTS_DIR", charts)
    monkeypatch.setattr(an_route, "AUDIO_DIR", audio)
    probs, _ = _trois({_col(G, 1): 0.45, _col(Bb): 0.40}, {G: 0.45, Bb: 0.45},
                      {_col(G, 1): 0.85, _col(Bb): 0.05}, {G: 0.9})
    monkeypatch.setattr(musx, "frame_posteriors", lambda a: probs)
    return app_mod.create_app().test_client()


def test_route_nouvel_accord_nomme_par_ses_deux_voisins(client):
    """Case vide touchée → le nouvel accord reçoit le nom λ1 t+b calculé avec
    l'accord juste avant ET juste après lui (suite éditée)."""
    r = client.post("/api/chord-candidates/min_essai.json", json={
        "t0": _t(100), "t1": _t(200), "prev_t0": _t(0), "prev_t1": _t(100),
        "next_t0": _t(200), "next_t1": _t(300)}).get_json()
    assert r["relabel"]["label"] == "Bb:maj"
    assert r["relabel"]["voisines"] == 2
    assert len(r["relabel"]["top"]) == 3
    assert "sug" in r and "source" in r            # le compas garde ses candidats


def test_route_sans_suivant_une_seule_voisine(client):
    r = client.post("/api/chord-candidates/min_essai.json", json={
        "t0": _t(100), "t1": _t(200), "prev_t0": _t(0), "prev_t1": _t(100)}).get_json()
    assert r["relabel"]["voisines"] == 1
