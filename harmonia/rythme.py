"""Le garde-fou des changements de rythme (Louis, 2026-10-12).

« Sur That's Life, vers la mesure 40/41, on décale tout, il y a eu un bug sur
le rythme ?? Il devrait y avoir un failsafe pour vérifier si, quand il y a un
changement de rythme, il fait sens. »

CE QUI SE PASSAIT (mesuré, `docs/plots/thats_life_rythme_2026-10-12.html`).
Le morceau bat à 0,80 s le temps d'un bout à l'autre ; les accords changent
toutes les 3,2 s, pile sur la mesure. Entre 119 s et 147 s, Beat This!
décroche : il pose une marque tous les TIERS de temps (0,26 s, le triolet du
shuffle), puis des paquets de 0,26 / 0,54. Les réparations existantes
(`beats.drop_duplicate_beats`, `drop_inserted_beats`) en retirent une partie,
pas tout. Or les mesures sont posées par ARITHMÉTIQUE sur les indices de
temps (`bars.layout_bars` : une mesure = `bpb` temps, phase globale) : dans
la zone, une « mesure » de quatre marques dure 1,6 à 2,4 s au lieu de 3,2,
quatorze mesures du chart couvrent huit vraies mesures et demie, et à la
sortie tout le reste du morceau est décalé d'un temps — chaque accord tombe
sur le temps 2.

LA RÈGLE, un principe et pas un seuil réglé :

    Un changement de tempo du traqueur n'est GARDÉ que s'il laisse une
    trace. S'il retombe exactement sur l'ancienne grille (même tempo, nombre
    entier de temps à travers la zone) ET que les changements d'accords
    d'après ne suivent PAS mieux la phase qu'il impose, c'est un hoquet du
    traqueur : on recoud l'ancienne grille à travers la zone.

Concrètement, pour chaque zone où les intervalles entre temps s'écartent de
la période du morceau (hors de `SPAN_LO..SPAN_HI`, la même bande « une seule
période » que les jumeaux larges de `beats.py`) :

  1. ANCRES. Le dernier temps régulier avant, le premier après ; il faut au
     moins une mesure régulière de chaque côté (un retour qui ne dure pas une
     mesure n'est pas un retour, on fusionne les zones).
  2. RETOMBE-T-ON SUR L'ANCIENNE GRILLE ? n = round(écart / période locale).
     Si l'écart n'est pas un nombre entier de temps à `BPM_TOL` près (la
     tolérance « une battue tombe juste » de `bpm_rigide`), le tempo ou la
     phase a vraiment changé : on garde le traqueur.
  3. L'HARMONIE TRANCHE LA PHASE. Si le traqueur compte un nombre de temps
     différent de n dans la zone, la phase des mesures d'après n'est pas la
     même selon qu'on le croit ou non. On mesure, sur les `FENETRE_MESURES`
     mesures qui suivent, la distance moyenne des changements d'accords musx
     au début de mesure le plus proche, sous les deux phases. Le traqueur ne
     garde sa phase que s'il fait STRICTEMENT mieux ; à égalité, la
     continuité gagne.
  4. RECOUDRE : n−1 temps également espacés entre les ancres, et les
     downbeats de la zone posés sur la phase d'avant.

CE QUE ÇA NE RÉSOUT PAS (règle #4) :
  * un vrai changement de MESURE avec des temps réguliers (une mesure à 2/4
    insérée, un passage en 3/4) : aucun intervalle n'est anormal, la règle ne
    se déclenche pas, et `layout_bars` garde sa phase GLOBALE — la mesure
    ajoutée décale encore tout ce qui suit. C'est le reste du problème.
  * une zone qui touche le début ou la fin du morceau (pas d'ancre des deux
    côtés) — l'intro relève de `harmonia.debut`.
  * une zone où le traqueur se trompe de tempo SANS revenir à l'ancien
    (verrou d'octave jusqu'à la fin) : c'est le bouton ÷2/×2.
  * un rubato : l'écart n'est pas un nombre entier de temps, on garde le
    traqueur (c'est voulu — un point d'orgue n'est pas un hoquet).
"""
from __future__ import annotations

import logging

import numpy as np

from harmonia.beats import SPAN_HI, SPAN_LO

logger = logging.getLogger(__name__)

#: Deux phrases de quatre mesures : la fenêtre où l'on regarde si les accords
#: suivent la phase du traqueur ou celle d'avant la zone.
FENETRE_MESURES = 8


def changements_accords(triad: np.ndarray, frame_dt: float,
                        periode: float) -> np.ndarray:
    """Les instants où l'accord musx le plus probable change, en secondes.

    Lissage sur un demi-temps, et un accord ne compte que s'il tient au moins
    un temps — deux échelles prises sur le TEMPS du morceau, pas réglées.
    Indépendant des battues par construction : c'est ce qui en fait un
    témoin et non un écho du traqueur.
    """
    from scipy.ndimage import uniform_filter1d
    w = max(1, int(round(0.5 * periode / frame_dt)))
    lab = uniform_filter1d(np.asarray(triad, float), w, axis=0).argmax(1)
    cuts = np.flatnonzero(np.diff(lab)) + 1
    segs, s0 = [], 0
    for c in list(cuts) + [len(lab)]:
        segs.append((s0, c, int(lab[s0])))
        s0 = c
    min_len = periode / frame_dt
    kept = []
    for s in segs:
        if s[1] - s[0] < min_len:
            continue
        if kept and kept[-1][2] == s[2]:
            kept[-1] = (kept[-1][0], s[1], s[2])
        else:
            kept.append(s)
    return np.array([s[0] * frame_dt for s in kept[1:]], float)


def _zones(b: np.ndarray, bpb: int) -> list[tuple[int, int]]:
    """Zones (i0, i1) d'indices de temps : b[i0] et b[i1] sont les ancres."""
    ibi = np.diff(b)
    P = float(np.median(ibi))
    reg = (ibi / P > SPAN_LO) & (ibi / P < SPAN_HI)
    zones, i = [], 0
    while i < len(ibi):
        if reg[i]:
            i += 1
            continue
        j = i
        while j < len(ibi) and not reg[j]:
            j += 1
        zones.append([i, j])          # intervalles irréguliers i..j-1
        i = j
    # un retour plus court qu'une mesure n'est pas un retour : on fusionne
    fus = []
    for z in zones:
        if fus and z[0] - fus[-1][1] < bpb:
            fus[-1][1] = z[1]
        else:
            fus.append(z)
    # ancres : au moins une mesure régulière de chaque côté
    out = []
    for i0, j in fus:
        if i0 < bpb or j + bpb > len(ibi):
            continue
        if not (reg[i0 - bpb:i0].all() and reg[j:j + bpb].all()):
            continue
        out.append((i0, j))           # ancres b[i0], b[j]
    return out


def _poids_metrique(pos: int, bpb: int) -> int:
    """Le poids d'un temps dans la mesure (hiérarchie métrique classique) :
    0 le temps 1, 1 la demi-mesure (temps 3 en 4/4), 2 les autres temps."""
    if pos == 0:
        return 0
    if bpb % 2 == 0 and pos == bpb // 2:
        return 1
    return 2


def _positions(onsets, temps, indices, off, bpb):
    """Poids métrique de chaque changement d'accord, sous une hypothèse de
    grille (`temps` et l'indice de mesure de chacun)."""
    t = np.asarray(temps, float)
    out = []
    for o in onsets:
        q = int(np.abs(t - o).argmin())
        out.append(_poids_metrique((indices[q] - off) % bpb, bpb))
    return np.array(out, int)


def _signe_significatif(gagnes: int, perdus: int) -> bool:
    """Test du signe unilatéral au seuil conventionnel de 5 % : la grille
    d'avant place-t-elle les accords sur des temps plus forts que le traqueur
    PLUS SOUVENT que le hasard ne le ferait ? Les égalités ne votent pas."""
    from math import comb
    n = gagnes + perdus
    if n == 0:
        return False
    p = sum(comb(n, k) for k in range(gagnes, n + 1)) / 2.0 ** n
    return p < 0.05


def verifier_changements(beats, downbeats, triad, frame_dt: float,
                         bpb: int = 4) -> tuple[list, list, list[dict]]:
    """(beats, downbeats, rapport) — les hoquets du traqueur recousus.

    `rapport` liste chaque zone examinée, gardée ou recousue, et pourquoi :
    c'est ce que la page de diagnostic affiche.
    """
    b = np.asarray([float(t) for t in beats], float)
    if len(b) < 4 * bpb or triad is None:
        return list(beats), list(downbeats), []
    db = np.asarray([float(t) for t in downbeats], float)
    zones = _zones(b, bpb)
    if not zones:
        return list(beats), list(downbeats), []
    ibi = np.diff(b)
    Pg = float(np.median(ibi))
    onsets = changements_accords(triad, frame_dt, Pg)
    rapport, a_recoudre = [], []
    for k, (i0, j) in enumerate(zones):
        ta, tc = float(b[i0]), float(b[j])
        Pa = float(np.median(np.r_[ibi[i0 - bpb:i0], ibi[j:j + bpb]]))
        x = (tc - ta) / Pa
        n_trq = j - i0
        r = {"t0": round(ta, 3), "t1": round(tc, 3), "temps_traqueur": n_trq,
             "temps_grille": round(x, 2)}
        # phase d'avant la zone : résidu modal des downbeats qui la précèdent
        idx_db = [int(np.abs(b - t).argmin()) for t in db if t < ta]
        if not idx_db:
            r["verdict"] = "gardé : aucun downbeat avant la zone"
            rapport.append(r)
            continue
        off = int(np.bincount(np.array(idx_db[-8:]) % bpb,
                              minlength=bpb).argmax())
        # la fenêtre : la zone, puis FENETRE_MESURES mesures (ou jusqu'à la
        # zone suivante)
        fin_idx = min(len(b) - 1, j + FENETRE_MESURES * bpb)
        if k + 1 < len(zones):
            fin_idx = min(fin_idx, zones[k + 1][0])
        ons = onsets[(onsets > ta + 0.5 * Pa) & (onsets <= b[fin_idx])]
        apres = list(b[j:fin_idx + 1])

        # hypothèse TRAQUEUR : ses temps, ses indices
        w_trq = _positions(ons, list(b[i0:fin_idx + 1]),
                           list(range(i0, fin_idx + 1)), off, bpb)

        def hyp_grille(n):
            pont = [ta + (tc - ta) * q / n for q in range(n)]
            temps = pont + apres
            return temps, list(range(i0, i0 + n + len(apres)))

        # candidats « l'ancienne grille continue » : x arrondi des deux côtés
        # (un léger rubato dans la zone laisse x entre deux entiers)
        cands = sorted({c for c in (int(np.floor(x + 1e-6)), int(np.ceil(x - 1e-6)))
                        if c >= 1},
                       key=lambda c: abs(c - x))
        if not cands:
            r["verdict"] = "gardé : zone plus courte qu'un temps"
            rapport.append(r)
            continue
        best = None
        for c in cands:
            w = _positions(ons, *hyp_grille(c), off, bpb)
            g, p = int((w < w_trq).sum()), int((w > w_trq).sum())
            if best is None or g - p > best[1] - best[2]:
                best = (c, g, p)
        n, gagnes, perdus = best
        r.update({"temps_grille": n, "accords": int(len(ons)),
                  "plus_forts_grille": gagnes, "plus_forts_traqueur": perdus})
        if (n_trq - n) % bpb == 0 and gagnes == 0:
            # même phase à la sortie et rien dans la zone ne départage
            r["verdict"] = "gardé : la phase d'après est la même dans les deux cas"
            rapport.append(r)
            continue
        if not _signe_significatif(gagnes, perdus):
            r["verdict"] = ("gardé : l'harmonie ne montre pas que l'ancienne "
                            "grille continue")
            rapport.append(r)
            continue
        r["verdict"] = "recousu : hoquet du traqueur, l'ancienne grille continue"
        a_recoudre.append((i0, j, n))
        rapport.append(r)
    if not a_recoudre:
        return list(beats), list(downbeats), rapport

    # RECOUDRE — de la fin vers le début, les indices d'avant restent valides
    nb = list(b)
    db_out = [float(t) for t in db]
    for i0, j, n in sorted(a_recoudre, reverse=True):
        ta, tc = nb[i0], nb[j]
        pont = [ta + (tc - ta) * q / n for q in range(1, n)]
        idx_db = [int(np.abs(np.asarray(nb) - t).argmin())
                  for t in db_out if t < ta]
        off = int(np.bincount(np.array(idx_db[-8:]) % bpb,
                              minlength=bpb).argmax()) if idx_db else None
        nb = nb[:i0 + 1] + pont + nb[j:]
        db_out = [t for t in db_out if not (ta < t < tc)]
        if off is not None:
            db_out += [nb[q] for q in range(i0 + 1, i0 + n) if (q - off) % bpb == 0]
        db_out.sort()
    nb = [round(float(t), 4) for t in nb]
    keep = set(nb)
    db_out = [round(float(t), 4) for t in db_out]
    # invariant downbeats ⊆ beats (les downbeats du traqueur sont ses temps)
    db_out = [t for t in db_out if t in keep or
              np.min(np.abs(np.asarray(nb) - t)) < 1e-3]
    n_rec = sum(1 for r in rapport if r["verdict"].startswith("recousu"))
    if n_rec:
        logger.info("rythme: %d zone(s) recousue(s) — %s", n_rec,
                    [(r["t0"], r["t1"]) for r in rapport
                     if r["verdict"].startswith("recousu")])
    return nb, db_out, rapport
