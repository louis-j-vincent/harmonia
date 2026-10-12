"""Le garde-fou des changements de rythme (Louis, That's Life, 2026-10-12).

« quand il y a un changement de rythme, vérifier s'il fait sens » — un
changement de tempo du traqueur n'est gardé que si l'harmonie le soutient ;
un hoquet qui casse la phase est recousu sur l'ancienne grille.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from harmonia import beats as B
from harmonia.musx import FRAME_DT
from harmonia.rythme import changements_accords, verifier_changements

P = 0.8          # 75 BPM, comme That's Life
BPB = 4


def _triad(changes, dur):
    """Postérieures musx factices : un accord différent à chaque instant de
    `changes` (one-hot, 73 classes comme le vrai réseau)."""
    n = int(dur / FRAME_DT) + 1
    tri = np.zeros((n, 73))
    bornes = [0.0] + list(changes) + [dur + 1]
    for k in range(len(bornes) - 1):
        a, z = int(bornes[k] / FRAME_DT), int(bornes[k + 1] / FRAME_DT)
        tri[a:z, 1 + (k * 5) % 24] = 1.0
    return tri


def test_hypothese_porteuse_frame_dt_et_fps():
    """Règle #1 : l'axe des temps des deux modèles. musx : hop/sr = 512/22050 ;
    Beat This! : 50 images/s (ses temps sont des multiples de 0,02 s)."""
    assert FRAME_DT == pytest.approx(512 / 22050)
    from beat_this.model.postprocessor import Postprocessor
    assert Postprocessor().fps == 50


def test_changements_accords_retrouve_les_changements():
    ch = [3.2, 6.4, 9.6]
    got = changements_accords(_triad(ch, 12.0), FRAME_DT, P)
    assert np.allclose(got, ch, atol=2 * FRAME_DT)


def _morceau_avec_hoquet():
    """32 mesures régulières, avec au milieu (mesures 13-16) le traqueur qui
    pose une marque tous les TIERS de temps : la phase d'après glisse."""
    vrais = [0.5 + i * P for i in range(32 * BPB + 1)]
    z0, z1 = 12 * BPB, 16 * BPB                  # zone en indices de vrais temps
    trq = vrais[:z0 + 1]
    t = vrais[z0]
    # 13 vrais temps... rendus en paquets de tiers (l'arithmétique des mesures
    # en sort décalée d'un temps : 16 vrais temps -> 16*3 - 3 marques)
    for q in range(1, (z1 - z0) * 3 - 3):
        trq.append(vrais[z0] + q * (vrais[z1] - vrais[z0]) / ((z1 - z0) * 3 - 3))
    trq += vrais[z1:]
    dbs = [vrais[i] for i in range(0, z0 + 1, BPB)] + \
          [vrais[i] for i in range(z1, len(vrais), BPB)]
    # un accord par mesure, sur le temps 1 — de la vraie grille
    changes = [vrais[i] for i in range(BPB, len(vrais) - 1, BPB)]
    return vrais, trq, dbs, _triad(changes, vrais[-1] + 1)


def test_hoquet_ponctuel_est_recousu():
    vrais, trq, dbs, tri = _morceau_avec_hoquet()
    nb, ndb, rap = verifier_changements(trq, dbs, tri, FRAME_DT, BPB)
    assert [r["verdict"][:7] for r in rap] == ["recousu"]
    # la grille recousue a exactement les vrais temps, à 1 ms près
    assert len(nb) == len(vrais)
    assert np.allclose(nb, vrais, atol=1e-3)
    # et les downbeats de la zone sont reposés sur la phase d'avant
    attendus = [vrais[i] for i in range(0, len(vrais), BPB)]
    assert np.allclose(sorted(ndb), attendus, atol=1e-3)


def test_vrai_changement_soutenu_par_l_harmonie_est_garde():
    """Le traqueur voit un passage lent (deux temps de 1,6 s), et APRÈS, les
    accords suivent la phase qu'il impose — pas celle de l'ancienne grille
    prolongée. Le garde-fou ne doit pas « réparer » la musique."""
    avant = [0.5 + i * P for i in range(12 * BPB + 1)]
    t = avant[-1]
    zone = [t + 1.6, t + 3.2]
    apres = [zone[-1] + (i + 1) * P for i in range(20 * BPB)]
    trq = avant + zone + apres
    # phase du traqueur : indices multiples de BPB
    dbs = [trq[i] for i in range(0, len(trq), BPB)]
    changes = [trq[i] for i in range(BPB, len(trq) - 1, BPB)]
    tri = _triad(changes, trq[-1] + 1)
    nb, ndb, rap = verifier_changements(trq, dbs, tri, FRAME_DT, BPB)
    assert rap and all(r["verdict"].startswith("gardé") for r in rap), rap
    assert nb == trq


def test_rien_sans_zone():
    vrais = [0.5 + i * P for i in range(100)]
    tri = _triad([vrais[i] for i in range(4, 96, 4)], 81.0)
    nb, ndb, rap = verifier_changements(vrais, vrais[::4], tri, FRAME_DT, BPB)
    assert nb == vrais and rap == []


_CHART = Path("state/cache/beats/UCENTf_LWYA__3017421.json")
_PROBS = Path("data/cache/musx_probs/UCENTf_LWYA__3017421.npz")


@pytest.mark.skipif(not (_CHART.exists() and _PROBS.exists()),
                    reason="caches de That's Life absents")
def test_thats_life_mesure_40():
    """Le cas réel : 119-145 s, Beat This! suit les triolets du shuffle."""
    d = B._clean(json.loads(_CHART.read_text()))
    g = B.check_grid(d["beats"], d["downbeats"], "UCENTf_LWYA")
    tri = np.load(_PROBS)["triad"]
    nb, ndb, rap = verifier_changements(d["beats"], g["downbeats"], tri,
                                        FRAME_DT, 4)
    rec = [r for r in rap if r["verdict"].startswith("recousu")]
    assert len(rec) == 1 and 119 < rec[0]["t0"] < 120
    # les mesures recousues durent toutes ~3,2 s entre 112 et 160 s
    b = np.asarray(nb)
    i0 = int(np.abs(b - 112.66).argmin())
    bars = b[i0::4]
    durs = np.diff(bars[bars < 158])
    assert np.all(np.abs(durs - 3.2) < 0.25), durs


_AUDIO = Path("docs/audio/UCENTf_LWYA.m4a")


@pytest.mark.skipif(not (_AUDIO.exists() and _CHART.exists() and _PROBS.exists()),
                    reason="audio/caches de That's Life absents")
def test_pipeline_thats_life_mesures_regulieres():
    """Bout en bout (caches chauds) : le chart brut n'a plus de mesures de
    1,6 s entre 119 et 147 s — c'était le « on décale tout » de Louis."""
    from harmonia.pipeline import analyze_steps
    for kind, model in analyze_steps(_AUDIO, title="That's Life",
                                     file_key="test_UCENTf_LWYA", head_s=0):
        if kind == "raw":
            break
    g = np.asarray(model["barGrid"])
    mid = g[(g > 100) & (g < 158)]
    durs = np.diff(mid)
    assert np.all(np.abs(durs - 3.2) < 0.35), np.round(durs, 2)
    assert model["meta"]["rythme"], "le rapport du garde-fou doit être dans meta"
