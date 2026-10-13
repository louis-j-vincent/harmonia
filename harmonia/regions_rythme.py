"""Les régions de rythme : des portions du morceau qui ont chacune leur barre de mesure.

Louis, 2026-10-13, sur The APL Song : « l'intro a un tempo différent du reste
du morceau. Il faut une brique qui détecte les différentes portions avec des
rythmes différents. »

CE QUI SE PASSAIT (mesuré, `docs/plots/apl_song_tempo_2026-10-13.html`).
Le tempo, lui, ne change pas : Beat This! bat à 0,674 s du début de l'intro
(16,3 s) à la fin, 89,0 BPM dans l'intro comme dans le corps (pente des
battues, écart max 0,02 s). Ce qui change, c'est la BARRE DE MESURE : l'intro
(l'échantillon, sans batterie, 16,3 → 38,5 s) dure 33 temps — huit mesures
et UN temps. Les accords de l'intro changent sur une grille, ceux du corps
(la batterie entre à 38,5 s) sur la même grille décalée d'un temps, et le
traqueur le dit aussi (ses downbeats sautent d'un temps à 38,5 s). Or
`bars.layout_bars` pose UNE phase pour tout le morceau : celle de la marque
« mesure 1 » (ici `harmonia.debut`, calée sur l'intro), donc les 66 mesures
du corps sont écrites un temps trop tôt — chaque accord tombe sur le temps 2.
Sans la marque, c'est l'intro qui serait fausse. Aucune phase unique ne peut
écrire les deux.

LA RÈGLE (un principe, pas un seuil réglé) :

    Une portion du morceau a sa propre barre de mesure si ses changements
    d'accords tombent, plus souvent que le hasard, sur des temps plus forts
    sous SA phase que sous celle de la portion voisine — et réciproquement
    pour la voisine. Chaque portion dure au moins une phrase (4 mesures).

Les deux côtés doivent gagner chacun chez soi : c'est ce qui sépare une vraie
frontière d'une poignée d'accords anticipés. Le juge est le même que celui du
garde-fou des changements de rythme (`harmonia.rythme`) : poids métrique
(1 > 3 > 2/4) et test du signe unilatéral à 5 %. Il faut donc au moins cinq
changements d'accords qui votent du même côté (0,5⁵ = 3 %) — c'est le test qui
le dit, pas un réglage.

Les changements d'accords sont ceux de musx, lus image par image
(`rythme.changements_accords`) : ils ne connaissent pas les battues, ce qui en
fait un témoin et non un écho du traqueur.

LA FRONTIÈRE. Entre le dernier changement d'accord de la portion d'avant et le
premier de celle d'après, la nouvelle barre se pose sur le premier temps qui
est un temps 1 sous la nouvelle phase — en préférant un downbeat du traqueur
s'il y en a un (il entend où la mesure recommence : 38,5 s sur APL).

CE QUE ÇA DONNE AU CHART : `layout_bars(..., regions=…)` pose les barres de
chaque portion sur sa propre phase ; la dernière mesure de la portion d'avant
est COURTE (1 à 3 temps, ce qui reste avant la nouvelle barre) — c'est ce
qu'écrit une partition qui insère une mesure à 1/4 ou 2/4. Le chart porte la
liste dans `tempoRegions` = [{t0, t1, bpm, beatsPerBar, phase}].

COMPOSITION AVEC LE GARDE-FOU (007d51c). Le garde-fou passe AVANT, sur les
battues : il recoud les hoquets du traqueur (That's Life, les triolets) ; les
régions ne touchent pas aux battues, seulement à l'endroit où l'on coupe les
mesures. Un hoquet recousu n'a donc plus de trace à juger ici.

CE QUE ÇA NE RÉSOUT PAS (règle #4) :
  * un vrai changement de TEMPO n'a rien à faire ici : les mesures sont
    comptées en temps du traqueur, elles suivent déjà son tempo. La grille
    rigide (`beats._poser_grille_rigide`) n'est posée que si 85 % des battues
    tombent sur UN tempo — un morceau à deux tempos garde ses battues.
  * un changement de MÉTRIQUE (un passage en 3/4) : `bpb` reste global ; la
    portion serait vue, au mieux, comme une suite de décalages.
  * une portion sans pulsation (les 16 s de bruit d'ambiance au début d'APL) :
    la grille rigide y prolonge les temps du corps, et le chart y écrit des
    mesures vides. Elle n'a pas d'accords, donc aucun vote ici.
  * un rubato, un accelerando progressif : rien de régulier à phaser.
  * une portion plus courte qu'une phrase (une seule mesure de 2/4 suivie
    d'un retour à la phase d'avant) : invisible par construction — un
    décalage qui se défait tout de suite ne change pas la phase d'après.
  * la mesure courte s'affiche comme une case pleine (pas de « 1/4 » écrit).
"""
from __future__ import annotations

import logging

import numpy as np

from harmonia.rythme import _poids_metrique, _signe_significatif, changements_accords

logger = logging.getLogger(__name__)

#: Une phrase de quatre mesures : la plus petite portion qui puisse avoir sa
#: propre barre de mesure. En dessous, c'est une mesure isolée, pas une portion.
PHRASE_MESURES = 4

#: Profondeur de la découpe : au plus 2³ = 8 portions. Une borne de sûreté sur
#: le temps de calcul, pas un réglage — chaque frontière doit passer le test.
PROFONDEUR_MAX = 3


def _poids(r: np.ndarray, phi: int, bpb: int) -> np.ndarray:
    return np.array([_poids_metrique(int((x - phi) % bpb), bpb) for x in r], int)


def _meilleure_phase(r: np.ndarray, bpb: int, prefere: int | None) -> int:
    """La phase qui pose ces changements d'accords sur les temps les plus forts.
    À égalité, la phase héritée gagne (la continuité, comme au garde-fou)."""
    couts = [int(_poids(r, phi, bpb).sum()) for phi in range(bpb)]
    m = min(couts)
    if prefere is not None and couts[prefere] == m:
        return prefere
    return int(np.argmin(couts))


def _duel(r: np.ndarray, chez_soi: int, voisin: int, bpb: int) -> tuple[int, int]:
    """(gagnés, perdus) : combien de changements tombent sur un temps plus
    fort sous la phase de leur portion que sous celle de la voisine."""
    a, b = _poids(r, chez_soi, bpb), _poids(r, voisin, bpb)
    return int((a < b).sum()), int((a > b).sum())


def _frontiere(qa: int, qb: int, phi: int, bpb: int, db_idx: set[int]) -> int | None:
    """Le premier temps 1 de la nouvelle phase après le dernier accord de
    l'ancienne (qa) et au plus tard sur le premier accord de la nouvelle (qb) ;
    un downbeat du traqueur dans l'intervalle passe devant."""
    cands = [k for k in range(qa + 1, qb + 1) if (k - phi) % bpb == 0]
    if not cands:
        return None
    entendus = [k for k in cands if k in db_idx]
    return entendus[0] if entendus else cands[0]


def detecter_regions(beat_times, downbeats, triad, frame_dt: float,
                     bpb: int = 4) -> dict:
    """{"regions": [...], "rapport": [...]} — vide si le morceau n'a qu'une phase.

    Une région : {"i0", "i1" (indices de temps, i1 exclu), "phase" (résidu
    des temps 1 modulo bpb), "t0", "t1", "bpm", "beatsPerBar"}. Les régions
    se suivent sans trou et couvrent tous les temps.
    """
    b = np.asarray([float(t) for t in beat_times], float)
    vide = {"regions": [], "rapport": []}
    if triad is None or len(b) < 4 * PHRASE_MESURES * bpb:
        return vide
    P = float(np.median(np.diff(b)))
    ons = changements_accords(triad, frame_dt, P, sans_nc=True)
    ons = ons[(ons >= b[0] - 0.5 * P) & (ons <= b[-1] + 0.5 * P)]
    if len(ons) < 10:
        return vide
    q = np.array([int(np.abs(b - o).argmin()) for o in ons], int)
    q = np.unique(q)                         # un temps ne vote qu'une fois
    db_idx = {int(np.abs(b - float(t)).argmin()) for t in downbeats}
    phrase = PHRASE_MESURES * bpb
    rapport: list[dict] = []

    def couper(lo: int, hi: int, k0: int, k1: int, phi: int, prof: int) -> list:
        """Les frontières dans q[lo:hi], portion de temps [k0, k1) de phase phi."""
        if prof >= PROFONDEUR_MAX or hi - lo < 10:
            return []
        best = None
        for s in range(lo + 5, hi - 5 + 1):
            rL, rR = q[lo:s], q[s:hi]
            pL = _meilleure_phase(rL, bpb, phi)
            pR = _meilleure_phase(rR, bpb, phi)
            if pL == pR:
                continue
            kb = _frontiere(int(q[s - 1]), int(q[s]), pR, bpb, db_idx)
            if kb is None or kb - k0 < phrase or k1 - kb < phrase:
                continue
            gL, perdL = _duel(rL, pL, pR, bpb)
            gR, perdR = _duel(rR, pR, pL, bpb)
            if not (_signe_significatif(gL, perdL) and _signe_significatif(gR, perdR)):
                continue
            gain = (int(_poids(rL, phi, bpb).sum() - _poids(rL, pL, bpb).sum())
                    + int(_poids(rR, phi, bpb).sum() - _poids(rR, pR, bpb).sum()))
            if best is None or gain > best[0]:
                best = (gain, s, kb, pL, pR, (gL, perdL), (gR, perdR))
        if best is None:
            return []
        gain, s, kb, pL, pR, dL, dR = best
        rapport.append({"t": round(float(b[kb]), 3), "temps": int(kb),
                        "phase_avant": pL, "phase_apres": pR,
                        "avant_gagnes_perdus": list(dL),
                        "apres_gagnes_perdus": list(dR), "gain": gain,
                        "downbeat_traqueur": kb in db_idx})
        return (couper(lo, s, k0, kb, pL, prof + 1) + [(kb, pL, pR)]
                + couper(s, hi, kb, k1, pR, prof + 1))

    phi0 = _meilleure_phase(q, bpb, None)
    cuts = couper(0, len(q), 0, len(b), phi0, 0)
    if not cuts:
        return vide

    # CONSOLIDATION. La découpe récursive propose des frontières ; chaque
    # portion reprend ensuite la phase de SES PROPRES accords, et chaque
    # frontière est rejugée entre ses deux voisines réelles. Elle tombe si
    #   (1) les deux phases sont les mêmes, ou
    #   (2) l'un des deux côtés ne gagne pas chez soi (test du signe), ou
    #   (3) LE DEUXIÈME TÉMOIN se tait ou contredit : les downbeats du
    #       traqueur, dans chaque portion, doivent tomber le plus souvent sur
    #       la phase que l'harmonie lui donne. Deux témoins indépendants (ce
    #       que musx entend, ce que Beat This! entend) — une portion où un
    #       seul des deux voit une autre barre de mesure n'en est pas une.
    # On retire la frontière la plus faible et on recommence.
    bornes = sorted({k for k, _, _ in cuts})
    db_arr = np.array(sorted(db_idx), int)

    def juger(bornes):
        lims = [0] + bornes + [len(b)]
        regs = []
        prev = None
        for i0, i1 in zip(lims, lims[1:]):
            rq = q[(q >= i0) & (q < i1)]
            phi = _meilleure_phase(rq, bpb, prev) if len(rq) else prev
            regs.append((i0, i1, phi, rq))
            prev = phi
        verdicts = []
        for j in range(1, len(regs)):
            (a0, a1, pa, ra), (b0, b1, pb, rb) = regs[j - 1], regs[j]
            ok, pourquoi = True, ""
            if pa == pb:
                ok, pourquoi = False, "même phase des deux côtés"
            else:
                dA, dB = _duel(ra, pa, pb, bpb), _duel(rb, pb, pa, bpb)
                if not (_signe_significatif(*dA) and _signe_significatif(*dB)):
                    ok, pourquoi = False, "un côté ne gagne pas chez soi"
                else:
                    for (i0, i1, phi, _r) in (regs[j - 1], regs[j]):
                        d = db_arr[(db_arr >= i0) & (db_arr < i1)]
                        if not len(d) or int(np.bincount(d % bpb, minlength=bpb).argmax()) != phi:
                            ok, pourquoi = False, "le traqueur ne voit pas cette barre"
                            break
            verdicts.append((ok, pourquoi, (len(ra) + len(rb))))
        return regs, verdicts

    rejets = []
    while bornes:
        regs, verdicts = juger(bornes)
        faibles = [(v[2], j) for j, v in enumerate(verdicts) if not v[0]]
        if not faibles:
            break
        _n, j = min(faibles)
        rejets.append({"t": round(float(b[bornes[j]]), 3), "rejet": verdicts[j][1]})
        bornes.pop(j)
    if not bornes:
        return {"regions": [], "rapport": sorted(rapport, key=lambda r: r["t"]),
                "rejets": rejets}

    # la frontière se recale sur la phase définitive de la portion d'après
    regs, _v = juger(bornes)
    for j in range(1, len(regs)):
        prev_q = q[q < bornes[j - 1]]
        next_q = q[q >= bornes[j - 1]]
        qa = int(prev_q[-1]) if len(prev_q) else bornes[j - 1] - 1
        qb = int(next_q[0]) if len(next_q) else bornes[j - 1]
        kb = _frontiere(qa, max(qb, bornes[j - 1]), regs[j][2], bpb, db_idx)
        if kb is not None:
            bornes[j - 1] = kb
    regs, _v = juger(sorted(set(bornes)))

    regions = []
    for i0, i1, phi, _r in regs:
        seg = b[i0:min(i1 + 1, len(b))]
        per = float(np.median(np.diff(seg))) if len(seg) > 1 else P
        regions.append({"i0": int(i0), "i1": int(i1), "phase": int(phi),
                        "t0": round(float(b[i0]), 3),
                        "t1": round(float(b[i1]) if i1 < len(b) else float(b[-1] + P), 3),
                        "bpm": round(60.0 / per, 2), "beatsPerBar": int(bpb)})
    logger.info("regions_rythme: %d portions — frontières %s", len(regions),
                [r["t0"] for r in regions[1:]])
    return {"regions": regions, "rapport": sorted(rapport, key=lambda r: r["t"]),
            "rejets": rejets}


def lignes_de_mesure(regions: list[dict], n_beats: int, bpb: int,
                     phase_region: dict[int, int] | None = None) -> list[int]:
    """Indices de temps des barres de mesure, portion par portion.

    Portion 0 : toutes les barres de sa phase, y compris avant le premier
    temps (indice négatif : la levée tombe dans la mesure 0, comme avant).
    Portion r > 0 : sa première barre est sa frontière `i0`, puis tous les
    `bpb` temps. La dernière mesure d'une portion s'arrête sur la frontière
    suivante : elle est courte quand les phases diffèrent.

    `phase_region` remplace la phase d'une portion (la marque « mesure 1 » de
    Louis reste souveraine, mais seulement sur SA portion).
    """
    phase_region = phase_region or {}
    lignes: list[int] = []
    for j, r in enumerate(regions):
        phi = phase_region.get(j, r["phase"])
        i0, i1 = r["i0"], r["i1"]
        if j == 0:
            # comme `off` de layout_bars : la mesure 0 commence au 1er temps 1,
            # les temps d'avant (la levée) tombent dans la mesure 0
            k = phi % bpb
        else:
            k = i0 if (i0 - phi) % bpb == 0 else i0 + ((phi - i0) % bpb)
            if k != i0:
                lignes.append(i0)     # frontière posée hors phase : une barre quand même
        while k < i1:
            lignes.append(k)
            k += bpb
    return sorted(set(lignes))
