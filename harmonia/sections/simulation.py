"""Deux façons de découper un morceau à partir des briques de Louis.

CE MODULE EST EN PRODUCTION depuis le 2026-09-17. Il est né comme banc
d'essai — Louis : « n'essaye pas de fix, c'est moi qui dirai comment fix »,
puis « fais-moi une simulation des 2 stratégies pour que je puisse les tester
interactivement » — et il a été acté le soir même : « notre brique +
SongFormer comme méthode principale, c'est acté, tu me le mets en prod ».

`sections_inferer` (la route qu'appelle « Valider les sections ») lit
`occurrences_de_tous` + `remplir_par_songformer`. `remplir_par_algo` n'est
plus qu'un point de comparaison, gardé pour la route `/api/sections/simuler/`
et la page `tools/chart_simule.py`, où Louis peut confronter les deux.

LE POINT DE DÉPART COMMUN, et c'est la découverte de la journée. Aujourd'hui
`sections_inferer` jette le trait de Louis dans une agglomération générique
qui redécoupe tout le morceau. Lui veut l'inverse :

    « Une fois que j'ai annoté une section, ça devient une brique, et la
    première chose à faire c'est de trouver d'autres occurrences de cette
    section. »

`cherche_brique` fait exactement ça, et c'est mesuré : sur Don't Want My Love
avec sa brique A (mesures 7-14), elle rend 7-14, 15-22 et 34-41 — les trois
occurrences que SongFormer trouve, et exactement celles que la production
rate (elle propose 15-22, 29-36, 49-56). Le résultat ne bouge pas entre 0,85
et 0,95 de seuil.

LES DEUX STRATÉGIES ne diffèrent QUE par ce qu'elles mettent dans les trous
— et c'est SongFormer qui a été retenu :

  * `remplir_par_algo` — l'agglomération des quatre mots, mais confinée aux
    trous, toutes les occurrences gelées ;
  * `remplir_par_songformer` — ce que le modèle avait trouvé, au lieu de le
    jeter. C'est la seule des deux qui sache écrire « intro » et « outro » :
    les lettres de l'autre ne peuvent venir que de Louis ou de l'alphabet.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("harmonia.sections.simulation")

#: plancher absolu : sous ce score, ce n'est jamais la même brique
PLANCHER_REPRISE = 0.85
#: combien d'écarts-types au-dessus de la médiane DU MORCEAU une reprise doit
#: se détacher — le seuil est relatif, parce que la distribution ne l'est pas
Z_REPRISE = 1.5
#: on ne dépasse jamais ça : au-delà, même la brique elle-même serait recalée
PLAFOND_REPRISE = 0.999
#: ces rôles ne se rejouent pas — Louis, 2026-09-16 : « une intro ne se rejoue
#: pas plus tard »
JAMAIS_REJOUEES = {"intro", "outro", "silence"}
#: l'alphabet des lettres neuves, d'où l'on retire celles de Louis
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: CE QU'UNE REPRISE TRANSPOSÉE DOIT PAYER EN PLUS. Repris tel quel de
#: `section_tool.ROT_BONUS` (2026-09-19) : une brique tournée ressemble à plus
#: de choses qu'une brique droite, donc elle doit se détacher davantage pour
#: qu'on la croie.
BONUS_TRANSPOSE = 0.06
#: EN DESSOUS DE QUATRE MESURES, ON NE CHERCHE PAS DE TRANSPOSITION. Mesuré
#: dans `section_tool` sur Bein Green : un motif de 2 mesures tourné colle
#: PARTOUT — dix « reprises transposées », toutes fausses. Un motif court n'a
#: pas assez de matière pour que sa transposition veuille dire quelque chose.
MIN_MESURES_TRANSPOSE = 4


def cherche_brique(chart: dict, b0: int, b1: int, audio_dir,
                   label: str = "") -> tuple[list[tuple], dict]:
    """(reprises, diagnostic) pour la brique [b0, b1] — mesures 0-indexées,
    fin incluse.

    Chaque reprise est `(m0, m1, score, rot)`, `rot` étant le nombre de
    demi-tons dont la brique a été TRANSPOSÉE pour être reconnue (0 = telle
    quelle). Le diagnostic dit pourquoi on n'a rien trouvé quand c'est le cas —
    voir `RAISONS` en fin de docstring.

    On fait glisser la brique le long de la matrice de ressemblance chord-tone
    (`sections.similarity`, la même que l'app utilise déjà pour répondre « où
    ce bloc se rejoue-t-il ? ») et on retient les pics sans recouvrement. La
    brique elle-même est TOUJOURS rendue, en tête.

    LE SEUIL EST RELATIF AU MORCEAU, et c'est tout l'enjeu. Un seuil absolu de
    0,90 marchait sur Don't Want My Love et cassait Cry Me A River : Louis,
    2026-09-17, « j'ai annoté, l'outil n'a complètement pas respecté ce que
    j'avais écrit ». Mesuré, les deux distributions n'ont rien à voir :

        Don't Want My Love   médiane 0,692 · 1,000 0,996 0,992 0,990 puis
                             CHUTE de 0,135 à 0,855
        Cry Me A River       médiane 0,896 · 1,000 0,963 0,963 0,961 0,952…
                             chutes de 0,000 à 0,009, AUCUNE falaise

    Sur le second, 38 départs sur 80 passaient 0,90 : sa brique était donc
    recopiée sur la moitié du morceau. Le morceau est une ballade dont toutes
    les mesures se ressemblent harmoniquement — le canal ne porte pas
    l'information, et un seuil fixe ne peut pas le savoir.

    D'où : `médiane + 1,5 écart-type`, borné par `PLANCHER_REPRISE` et
    `PLAFOND_REPRISE`. Un morceau SANS CONTRASTE s'abstient tout seul, ce qui
    est la bonne réponse — la brique de Louis reste posée là où il l'a mise et
    le remplissage des trous garde le reste. C'est la même loi que le projet
    applique déjà ailleurs (`soudure._otsu`, un seuil par morceau ; mémoire
    « per-song threshold, and a channel with no contrast abstains »).

    LES REPRISES TRANSPOSÉES (2026-09-19). Le dernier refrain monté d'un ton
    était le trou écrit de cette fonction : la matrice compare des vecteurs de
    classes de hauteur ABSOLUES, donc une reprise modulée n'y ressemble à
    rien. On refait donc glisser la brique sur les onze matrices tournées
    (`V @ roll(V, k).T`, comme `section_tool._rotated_S`), et une reprise
    trouvée là porte son `rot`. Deux garde-fous, tous deux repris de
    `section_tool` où ils ont été payés :

      * `BONUS_TRANSPOSE` — une brique tournée ressemble à plus de choses
        qu'une brique droite, donc elle doit se détacher DAVANTAGE ;
      * `MIN_MESURES_TRANSPOSE` — sous quatre mesures on ne cherche pas du
        tout. Sur Bein Green, un motif de 2 mesures tourné donnait dix
        « reprises transposées », toutes fausses.

    Les reprises DROITES se servent les premières, à tous les coups : une
    reprise à l'identique vaut toujours mieux qu'une reprise modulée au même
    endroit.

    UNE INTRO NE SE REJOUE PAS (Louis, 2026-09-16). Un rôle de
    `JAMAIS_REJOUEES` ne cherche aucune reprise — sur Cry Me A River, « intro »
    était recopié SEPT fois. Cette règle existait dans l'ancienne route et
    avait été perdue en la réécrivant.

    CE QUE ÇA NE RÉSOUT PAS : sur un morceau sans contraste on ne trouve
    RIEN, même quand la section se rejoue vraiment — on préfère rater une
    reprise (le remplissage tient alors le trou) que d'écrire son nom sur la
    moitié du morceau. Le diagnostic le DIT au lieu de rendre une liste vide
    silencieuse, mais il ne le répare pas. Et une reprise qui change de mode
    (le refrain passé en mineur) n'est toujours pas retrouvée : une rotation
    déplace les douze classes ensemble, elle ne change pas les tierces.

    RAISONS possibles dans le diagnostic : `"role"` (une intro ne se rejoue
    pas), `"pas_d_audio"`, `"trop_court"`, `"sans_contraste"` (rien ne dépasse
    le seuil du morceau), ou `None` quand des reprises ont été trouvées.
    """
    import numpy as np

    from harmonia import musx as _musx
    from harmonia.sections import similarity as VS
    seule = [(b0, b1, 1.0, 0)]
    info = {"label": label, "m0": b0, "m1": b1, "n_reprises": 0,
            "seuil": None, "mediane": None, "sigma": None, "raison": None}

    if (label or "").strip().lower() in JAMAIS_REJOUEES:
        info["raison"] = "role"
        return seule, info

    grid = chart.get("barGrid") or []
    n = len(grid) - 1
    L = b1 - b0 + 1
    stem = Path(chart.get("audio_url") or "").stem
    audio = Path(audio_dir) / f"{stem}.m4a"
    if n < 2 or L < 1 or not stem or not audio.exists():
        # Sans audio il n'y a pas de matrice, donc aucune reprise à chercher.
        # Ce n'est pas un repli muet — on le DIT — et ça ne trahit pas son
        # geste : ça se contente de ne rien ajouter.
        logger.info("cherche_brique : pas d'audio pour %s, la brique reste "
                    "seule", stem or "?")
        info["raison"] = "pas_d_audio"
        return seule, info

    triad = _musx.frame_posteriors(audio)[0]
    V = VS.harmonic_vectors(triad, grid)
    S = VS.ssm(triad, grid)
    dom = np.asarray(VS._slide(S, b0, L, n))[:max(1, n - L + 1)]
    if len(dom) < 3:
        info["raison"] = "trop_court"
        return seule, info

    seuil = min(PLAFOND_REPRISE,
                max(PLANCHER_REPRISE,
                    float(np.median(dom) + Z_REPRISE * dom.std())))
    info.update(seuil=round(seuil, 3), mediane=round(float(np.median(dom)), 3),
                sigma=round(float(dom.std()), 3))

    pris, out = [(b0, b1)], list(seule)

    def _moissonne(scores, seuil_k, rot):
        """Les pics de `scores` qui ne recouvrent rien de déjà pris."""
        for b in sorted(range(len(scores)), key=lambda x: -scores[x]):
            if scores[b] < seuil_k:
                break
            if any(not (b + L <= c or d < b) for c, d in pris):
                continue
            pris.append((b, b + L - 1))
            out.append((b, b + L - 1, float(scores[b]), rot))

    # LES DROITES D'ABORD, toutes, avant de regarder une seule transposition.
    _moissonne(dom, seuil, 0)
    if L >= MIN_MESURES_TRANSPOSE:
        seuil_rot = min(PLAFOND_REPRISE, seuil + BONUS_TRANSPOSE)
        for k in range(1, 12):
            Sk = V @ np.roll(V, k, axis=1).T
            _moissonne(np.asarray(VS._slide(Sk, b0, L, n))[:len(dom)],
                       seuil_rot, k)

    info["n_reprises"] = len(out) - 1
    if info["n_reprises"] == 0:
        info["raison"] = "sans_contraste"
    tournees = sum(1 for o in out if o[3])
    logger.info("cherche_brique %s [%d-%d] « %s » : seuil %.3f (médiane %.3f, "
                "σ %.3f) → %d occurrence(s), dont %d transposée(s)", stem,
                b0 + 1, b1 + 1, label, seuil, info["mediane"], info["sigma"],
                len(out), tournees)
    return sorted(set(out)), info


def occurrences_de_tous(chart: dict, traits: list[dict],
                        audio_dir) -> tuple[list[dict], list[dict], list[dict]]:
    """(occurrences, traits écartés, diagnostics) — chaque trait de Louis, plus
    ses reprises, sans que deux briques se marchent dessus.

    Les traits sont traités dans l'ordre où il les a posés : les reprises
    d'une brique ne peuvent pas recouvrir un trait déjà placé, ni les reprises
    d'une brique précédente. Le premier arrivé garde sa place, ce qui est la
    seule règle qui ne trahisse pas son intention.

    Chaque occurrence porte `rot` : les demi-tons dont la brique a été
    transposée pour être reconnue là (0 = telle quelle).

    Le troisième retour est la liste des diagnostics de `cherche_brique`, un
    par trait gardé — DE QUOI DIRE À L'ÉCRAN pourquoi une brique n'a pas
    trouvé de reprise, au lieu de laisser croire à une panne. Louis,
    2026-09-19 : un morceau sans contraste doit se DIRE, pas se taire.
    """
    from harmonia.soudure import traits_propres
    n = len((chart.get("barGrid") or [])) - 1
    gardes, ecartes = traits_propres(traits, n)
    pris: list[tuple] = []
    out: list[dict] = []
    diagnostics: list[dict] = []
    for b0, b1, label in gardes:
        reprises, info = cherche_brique(chart, b0, b1, audio_dir, label=label)
        diagnostics.append(info)
        for x, y, sc, rot in reprises:
            if any(not (y < c or d < x) for c, d in pris):
                continue
            pris.append((x, y))
            out.append({"m0": x, "m1": y, "label": label,
                        "score": round(sc, 3), "rot": int(rot),
                        "trace": (x == b0 and y == b1)})
    return sorted(out, key=lambda o: o["m0"]), ecartes, diagnostics


def remplir_par_algo(chart: dict, occ: list[dict], audio_dir) -> list[dict]:
    """Les trous, remplis par l'algorithme des quatre mots.

    Toutes les occurrences entrent comme unités GELÉES — pas seulement celles
    que Louis a tracées. L'agglomération ne travaille donc que dans les trous
    et ne peut plus donner le nom d'une brique à un bloc qu'elle a fabriqué
    elle-même.

    CE QUE ÇA NE RÉSOUT PAS : cet algorithme ne sait produire que des lettres.
    Ni « intro » ni « outro » ne peuvent en sortir, quelle que soit la musique.
    """
    from harmonia.phrases4 import grouper_restes, merges4, nommer, phrases
    from harmonia.soudure import mot_sur_traits

    traits = [(o["m0"], o["m1"]) for o in occ]
    bornes, mot, _src = mot_sur_traits(chart, traits, audio_dir=audio_dir)
    idx = {b: j for j, b in enumerate(bornes)}
    fixes, par_jeton = [], {}
    for o in occ:
        j0, j1 = idx.get(o["m0"]), idx.get(o["m1"] + 1)
        if j0 is None or j1 is None:
            continue
        fixes.append((j0, j1 - 1))
        par_jeton[(j0, j1)] = o["label"]
    depart, j = [], 0
    for j0, j1 in fixes:
        while j < j0:
            depart.append((j, j + 1, mot[j])); j += 1
        depart.append((j0, j1 + 1, mot[j0:j1 + 1])); j = j1 + 1
    while j < len(mot):
        depart.append((j, j + 1, mot[j])); j += 1
    geles = {(j0, j1 + 1) for j0, j1 in fixes}

    _secs, info = phrases(mot, depart=depart, geles=geles)
    steps = merges4(mot, cible=info["cible"], depart=depart, geles=geles)
    blocs = grouper_restes(nommer(steps[-1]["jetons"], info["cible"],
                                  geles=geles), mot, info["cible"])
    siennes = {L.upper() for L in par_jeton.values()}
    libres = [c for c in ALPHABET if c not in siennes]
    renom, k, out = {}, 0, []
    for b in blocs:
        sien = par_jeton.get((b["j0"], b["j1"]))
        if sien is not None:
            nom, source = sien, "brique"
        else:
            if b["label"] not in renom:
                renom[b["label"]] = libres[k % len(libres)] if libres else b["label"]
                k += 1
            nom = renom[b["label"]] + ("′" if b["prime"] else "")
            source = "algo"
        out.append({"m0": bornes[b["j0"]], "m1": bornes[b["j1"]] - 1,
                    "label": nom, "source": source, "mot": b["type"]})
    return out


def remplir_par_songformer(chart: dict, occ: list[dict],
                           auto: list[dict]) -> list[dict]:
    """Les trous, remplis par ce que le modèle avait trouvé.

    Les mesures couvertes par une brique de Louis gardent SON nom, et deux
    occurrences voisines de la même brique restent deux blocs — le morceau les
    joue deux fois. Partout ailleurs on garde les SEGMENTS de SongFormer tels
    qu'il les a posés, rognés autour des briques.

    On ne repasse PAS par un tableau « une étiquette par mesure » : ça fondait
    deux segments voisins portant la même lettre en un seul bloc, et un
    morceau qui joue A puis A ressortait avec un A de seize mesures. Les
    frontières du modèle sont une information, pas un effet de bord.

    `auto` est la liste des segments du modèle, chacun avec ses `barRanges` —
    la réponse que la production jetait avant le 2026-09-17.
    """
    n = len((chart.get("barGrid") or [])) - 1
    pris = [False] * n
    out = []
    for o in occ:
        a, b = max(0, o["m0"]), min(n - 1, o["m1"])
        if b < a:
            continue
        out.append({"m0": a, "m1": b, "label": o["label"], "source": "brique",
                    "mot": ""})
        for i in range(a, b + 1):
            pris[i] = True
    for sec in auto or []:
        lab = sec.get("label") or "?"
        for a, b in (sec.get("barRanges") or []):
            i = max(0, a)
            fin = min(n - 1, b)
            while i <= fin:
                if pris[i]:
                    i += 1
                    continue
                j = i
                while j + 1 <= fin and not pris[j + 1]:
                    j += 1
                out.append({"m0": i, "m1": j, "label": lab,
                            "source": "songformer", "mot": ""})
                i = j + 1
    return sorted(out, key=lambda b: b["m0"])
