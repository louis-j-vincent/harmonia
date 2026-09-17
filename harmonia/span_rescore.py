"""span_rescore.py — le classement du modèle sur chaque accord.

Pour un accord du chart, calcule le classement des autres (racine, famille)
que musx aurait pu écrire à sa place, à partir de ses propres postérieures de
trame (``musx_suggestions``) : c'est ce que l'écran d'annotation de Louis
affiche comme « candidats » sous chaque accord — le champ ``sug`` de chaque
accord du chart.

Deux vocabulaires cohabitent ici parce que le modèle vendu et le chart
n'emploient pas la même grammaire de qualité : ``_TRIAD_TYPES`` ×
``_SEV_TYPES`` est la sortie du modèle (deux têtes catégorielles, triade et
septième) ; elle se replie sur QUAL5 (maj/min/dom/hdim/dim, la famille que le
chart affiche) via ``_FOLD``, un tenseur de sélection construit une seule
fois par ``_build_fold_tensor``. ``idx_of``/``token_of`` vont et viennent
entre cette grille (12 racines × 5 familles = 60 candidats) et un index plat.

Ce que ce module ne fait PLUS. Jusqu'au 2026-08-20, ce fichier portait aussi
la PROPAGATION : verrouiller un accord et faire migrer ce verrou sur ses
voisins via un second vocabulaire de repli, un scoreur de contexte appris, et
un Viterbi d'ordre 2 sur tout le lattice des accords du morceau. Louis a fait
retirer cette fonction ce jour-là (« ça se propage sur les accords suivants
mais cette fonction est deprecated, enlève-la ») : le bouton a disparu du
shell, et l'ancienne route répond maintenant un refus lisible plutôt qu'un
effet silencieux. L'historique complet — le design, les réglages essayés, les
mesures — reste sous le tag git ``pre-refactor-2026-09-14``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from harmonia import cache

# QUAL5 order is load-bearing: `idx_of`/`token_of` and every (root, family)
# candidate index below assume this exact order.
QUAL5 = ["maj", "min", "dom", "hdim", "dim"]

N_ROOTS = 12
N_QUAL5 = len(QUAL5)          # 5
N_CANDIDATES = N_ROOTS * N_QUAL5   # 60, index = root_pc * 5 + qual5_idx


def idx_of(root_pc: int, qual5_idx: int) -> int:
    return int(root_pc) % N_ROOTS * N_QUAL5 + int(qual5_idx)


def token_of(idx: int) -> tuple[int, int]:
    """60-index -> (root_pc, qual5_idx)."""
    return divmod(int(idx), N_QUAL5)


# ═══════════════════════════════════════════════════════════════════════════
# Vocabulaire : (type de triade, type de septième) du modèle vendu -> QUAL5
# ═══════════════════════════════════════════════════════════════════════════
# Triad-type order in the vendored model's own enum, EXCLUDING 'none'/
# 'power'/'one' which the trained 73-wide triad head never predicts —
# verified against the model's own legal-vocabulary file: maj/min/dim/aug/
# sus4/sus2 are all present as bare triads, 'power'/'one' never appear.
_TRIAD_TYPES = ("maj", "min", "sus4", "sus2", "dim", "aug")
# Seventh-type order in the vendored model's own enum: none, add_7 (natural 7
# / maj7 interval), add_b7 (flat 7 — dominant/min7), add_bb7 (diminished 7th).
_SEV_TYPES = ("none", "maj7", "b7", "bb7")

# (triad type, seventh type) -> QUAL5 family, by chord theory: maj7->maj,
# dom7->dom, m7b5->hdim, dim7->dim, sus2/sus4(bare)->maj, 7sus4->dom,
# aug->maj, aug7->dom. The fine-grained spellings differ (musx says "7", the
# chart says "dom7") but the family assignment is the same chord. Verified
# against the model's own legal-vocabulary list: maj, min, 7, maj7, min7,
# dim, dim7, hdim7, aug, sus2, sus4, sus4(b7) are the ONLY (triad, seventh)
# combinations the model was ever trained to predict. Combinations absent
# from that vocabulary (e.g. an augmented major 7th) get a defensive
# same-triad-family fallback below — their posterior mass is ~0 in practice,
# never trained-for.
_TRIAD_SEV_TO_QUAL5 = {
    ("maj", "none"): "maj", ("maj", "maj7"): "maj",
    ("maj", "b7"): "dom",   ("maj", "bb7"): "maj",
    ("min", "none"): "min", ("min", "maj7"): "min",
    ("min", "b7"): "min",   ("min", "bb7"): "min",
    ("sus4", "none"): "maj", ("sus4", "maj7"): "maj",
    ("sus4", "b7"): "dom",   ("sus4", "bb7"): "maj",
    ("sus2", "none"): "maj", ("sus2", "maj7"): "maj",
    ("sus2", "b7"): "dom",   ("sus2", "bb7"): "maj",
    ("dim", "none"): "dim", ("dim", "maj7"): "dim",
    ("dim", "b7"): "hdim",  ("dim", "bb7"): "dim",
    ("aug", "none"): "maj", ("aug", "maj7"): "maj",
    ("aug", "b7"): "dom",   ("aug", "bb7"): "maj",
}


def _build_fold_tensor() -> np.ndarray:
    """(6 triad types, 4 seventh types, 5 QUAL5) one-hot selection tensor."""
    fold = np.zeros((len(_TRIAD_TYPES), len(_SEV_TYPES), N_QUAL5), dtype=np.float64)
    q5_idx = {q: i for i, q in enumerate(QUAL5)}
    for ti, t in enumerate(_TRIAD_TYPES):
        for si, s in enumerate(_SEV_TYPES):
            fam = _TRIAD_SEV_TO_QUAL5[(t, s)]
            fold[ti, si, q5_idx[fam]] = 1.0
    return fold


_FOLD = _build_fold_tensor()   # (6,4,5)


def musx_cache_path(audio_path: Path | str) -> Path:
    """Le chemin où vivrait la moisson musx de ce fichier, SOUS LA CLÉ NEUVE
    (``<stem>__<taille>``, sprint 15) — SANS la charger.

    ATTENTION, piégé pour un test d'existence : cette fonction ne rend QUE le
    chemin neuf, jamais le chemin historique (le stem seul) où vit encore tout
    cache musx de la bibliothèque (`data/cache/`, non renommée avant la
    bascule prod, sprint 21). `musx_cache_path(audio).exists()` répond donc
    FAUX pour un cache chaud sous l'ancienne clé — c'est le bug trouvé au
    sprint 15 dans `refold.py` (le repli d'empilement se dégradait en
    silence sur toute la bibliothèque). Pour un VRAI test « ce cache existe-t-
    il, neuf ou historique ? », utiliser `harmonia.cache.exists("musx_probs",
    audio)`. Cette fonction ne sert plus qu'à des scripts hors-ligne qui
    veulent le chemin d'ÉCRITURE (`scripts/backfill_musx_sug.py`).
    """
    return cache.path("musx_probs", Path(audio_path))


def pool_span_musx(
    probs: list[np.ndarray], spans: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool musx's triad(73) + s7(4) frame posteriors per span.

    ``probs`` is exactly musx's own ``frame_posteriors`` return (``[triad,
    bass, s7, s9, s11, s13]``); only triad and s7 are read here (s9/s11/s13
    don't affect QUAL5 — a dominant 9th still folds to "dom"). Frame ``f``
    covers ``[f*FRAME_DT, (f+1)*FRAME_DT)``.

    Returns ``(pooled_triad (n_spans,73), pooled_s7 (n_spans,4))``. An empty
    span (shorter than one frame, or past the end of the posteriors) falls
    back to the single nearest frame, the same empty-interval rule used
    elsewhere in this project for beat pooling.
    """
    from harmonia.musx import FRAME_DT

    triad, s7 = probs[0], probs[2]
    n_frame = triad.shape[0]
    n = len(spans)
    out_triad = np.zeros((n, triad.shape[1]), dtype=np.float64)
    out_s7 = np.zeros((n, s7.shape[1]), dtype=np.float64)
    for i, (t0, t1) in enumerate(spans):
        f0 = max(0, int(round(t0 / FRAME_DT)))
        f1 = min(n_frame, int(round(t1 / FRAME_DT)))
        if f1 <= f0:
            j = int(np.clip(round(0.5 * (t0 + t1) / FRAME_DT), 0, max(n_frame - 1, 0)))
            out_triad[i] = triad[j]
            out_s7[i] = s7[j]
        else:
            out_triad[i] = triad[f0:f1].mean(0)
            out_s7[i] = s7[f0:f1].mean(0)
    return out_triad, out_s7


def acoustic_logp_musx(
    pooled_triad: np.ndarray, pooled_s7: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(pooled triad, pooled s7) -> (log-posterior (n,60), n_mass (n,)).

    ``n_mass`` is the pooled "no chord at all" mass (triad column 0) — kept
    OUT of the 60-candidate space, reported separately so a caller can flag a
    likely-silent span without it competing for a (root, quality) slot it
    doesn't have.

    Root and triad-TYPE are NOT independent in this model — the 73-wide triad
    head predicts them JOINTLY (one-of-73, incl. "no chord") — so this is an
    exact marginal-times-conditional combination for that pair, only the
    seventh factor is treated as combining independently of root (which is
    also the vendored model's own factorization: triad, seventh, ninth, ...
    are trained as separate categorical heads).
    """
    n = pooled_triad.shape[0]
    n_mass = np.clip(pooled_triad[:, 0].astype(np.float64), 0.0, 1.0)
    # Column i>=1 of the 73-wide triad posterior is root=(i-1)%12, type=
    # (i-1)//12 (root is the FAST-varying axis, type is SLOW) — a real bug
    # once got this backwards (root agreement 0.0 on This Love before the
    # fix): a naive ``.reshape(n, 12, 6)`` implicitly assumes the opposite
    # (root slow, type fast) and silently transposes every chord's root+type.
    # Reshape as (type, root) — the true flat order — then move the axes to
    # (root, type).
    chord = (pooled_triad[:, 1:].astype(np.float64)
            .reshape(n, len(_TRIAD_TYPES), N_ROOTS).transpose(0, 2, 1))
    sev = pooled_s7.astype(np.float64)
    # joint[n,r,q] = sum_t sum_s chord[n,r,t] * sev[n,s] * FOLD[t,s,q]
    joint = np.einsum("nrt,ns,tsq->nrq", chord, sev, _FOLD)
    denom = np.clip(1.0 - n_mass, 1e-9, None)          # == joint.sum((1,2)) up to fp error
    joint = joint / denom[:, None, None]
    logp = np.log(np.clip(joint, 1e-12, None)).reshape(n, N_CANDIDATES)
    return logp, n_mass


# QUAL5 index -> the iReal tail the shell renders (the inverse of
# ``ireal_q_to_q5`` below).
Q5_TAIL = {0: "", 1: "-", 2: "7", 3: "-7b5", 4: "o"}

#: Les quatre colonnes de la tête de SEPTIÈME du modèle vendu
#: (`complex_chord.SeventhTypes`) : aucune, la majeure, la mineure, la
#: diminuée.
_SEV_NONE, _SEV_MAJ7, _SEV_B7, _SEV_BB7 = 0, 1, 2, 3


def queue_du_candidat(q5_idx: int, pooled_s7=None) -> str:
    """La queue écrite d'un candidat — `^7` et `-7` compris quand la septième
    du modèle les soutient.

    Louis, 2026-09-17 : « typiquement l'endroit ou j'ai marqué un Emaj7, ca ne
    le proposait jamais ». Mesuré : `^7` apparaissait dans 177 suggestions de
    la bibliothèque, et les 177 fois l'accord ÉCRIT était déjà ce maj7. Comme
    alternative : zéro. Idem `-7`, pourtant écrit 348 fois.

    La cause est dans l'espace de candidats lui-même : 12 racines × CINQ
    familles, où `_TRIAD_SEV_TO_QUAL5` replie ("maj","maj7") sur "maj" et
    ("min","b7") sur "min". `Q5_TAIL` écrivait ensuite `""` et `"-"`.

    Ce qui change ici : RIEN au classement — même espace, mêmes probabilités,
    même ordre. Seul le NOM est raffiné, avec une information que musx produit
    déjà et que `acoustic_logp_musx` lit depuis toujours pour scorer. On ne
    devine rien, on cesse de jeter.

    Conservateur et SANS SEUIL NOUVEAU : une famille « maj » ne devient `^7`
    que si la septième la plus probable EST la maj7, une « min » ne devient
    `-7` que si c'est la b7. Une « maj » avec une b7 garde sa triade nue —
    cette combinaison a déjà sa famille (`dom`), et la nommer `^7`
    contredirait le repli qui a produit le classement.

    `pooled_s7` absent ou vide : on rend l'ancienne queue. Un chart d'avant ou
    un appelant sans la tête ne doit pas casser.
    """
    base = Q5_TAIL[q5_idx]
    if pooled_s7 is None:
        return base
    import numpy as np
    v = np.asarray(pooled_s7, dtype=float).ravel()
    if v.size < 4 or not float(v.sum()) > 0.0:
        return base
    sev = int(np.argmax(v))
    if q5_idx == 0 and sev == _SEV_MAJ7:          # maj -> maj7
        return "^7"
    if q5_idx == 1 and sev == _SEV_B7:            # min -> min7
        return "-7"
    return base


def ireal_q_to_q5(q: str | None) -> int:
    """Queue iReal (`-7`, `^7`, `h7`, `o`…) → l'index QUAL5 des candidats.

    Rapatriée de `context_rescore` le 2026-08-20, quand la propagation d'un
    accord sur ses voisins a été retirée : c'est ici qu'on en a besoin, pour
    savoir quelle case du plan des candidats est celle de l'accord ÉCRIT.
    Mapping inchangé, au caractère près.
    """
    if not q:
        return 0
    if q.startswith("-7b5") or q.startswith("h"):
        return 3
    if q.startswith("-") or q.startswith("m"):
        return 1
    if q.startswith("o") or q.startswith("dim"):
        return 4
    if q.startswith("^") or "maj7" in q or "M7" in q:
        return 0
    if any(t in q for t in ("7", "9", "13", "alt")):
        return 2
    return 0


#: Sous ce niveau, un candidat d'accord ne s'affiche pas (2026-09-16, demande
#: de Louis : « top 5 accords si relevant » — le « si relevant » est ce
#: plancher, pas un remplissage jusqu'à 5).
#:
#: LA LOI, en une phrase : un candidat ne s'affiche que s'il est plus probable
#: que le pur hasard sur son propre jeu de candidats. musx répartit sa masse
#: sur 60 cases (12 racines × 5 familles) ; l'uniforme vaut donc 1/60 = 1,67 %,
#: et 2 % est ce seuil arrondi vers le haut.
#:
#: Mesuré sur les deux morceaux ambigus du jour (Yesterday 19 accords,
#: Lost Without U 74 accords), échelle des postérieures au rang 5 :
#: médiane 0,017-0,025 — le plancher tombe donc pile là où le 5e candidat
#: cesse d'être informatif. Ce qu'il donne : 3,8 candidats par accord en
#: moyenne (médiane 4-5), 4-10 % des accords n'en gardent qu'UN (les très
#: sûrs, qui doivent continuer à ressembler à aujourd'hui). À 5 % on
#: retomberait à 2,4-2,9, c'est-à-dire l'ancien top-3 : rien n'aurait changé.
#:
#: CE QUE ÇA NE RÉSOUT PAS : le plancher est le même pour tous les morceaux et
#: n'a pas été arbitré à l'oreille — il est calibré sur la FORME de l'échelle
#: (où le 5e rang décroche), pas sur « ce candidat-là était juste ». Deux
#: morceaux seulement (règle #5, CLAUDE.md) ; à rouvrir si Louis trouve la
#: liste trop longue ou trop courte.
SUG_FLOOR = 0.02

#: Même loi, pour la basse : la lecture de `bass_pc_onset` répartit sa masse
#: sur 12 classes de hauteur, l'uniforme vaut 1/12 = 8,3 %, et on demande une
#: fois et demie l'uniforme.
#:
#: CE SEUIL NE DÉCIDE RIEN DU CHART : « est-ce que je MONTRE cette lecture ? »
#: et non « est-ce que j'ÉCRIS un slash ? ». Il gouverne un affichage de
#: l'écran d'annotation, pas le contenu du chart.
#: Il a longtemps cohabité avec `bass_rules.FLOOR` (30 %), qui décidait
#: l'écriture — les confondre aurait été l'erreur de calibration silencieuse
#: de CLAUDE.md (règle #1), l'un étant une part et l'autre des pourcents.
#: `bass_rules` a été retiré le 2026-09-16 : l'écriture d'un slash n'a plus
#: de plancher du tout, c'est l'argmax de la tête basse de musx sur la durée
#: de l'accord (`pipeline._write_sounding_bass`). Ce seuil-ci reste, seul, et
#: garde son échelle : une PART entre 0 et 1.
#: À 30 %, 84-90 % des accords n'afficheraient qu'une seule note : la demande
#: de Louis (voir la ligne de basse) serait vide de sens. À 12,5 %, mesuré :
#: 2,2 notes par accord en moyenne, un tiers des accords en montrent 3.
BASS_SUG_FLOOR = 0.125


def delta_candidates(probs: list[np.ndarray], span: tuple[float, float],
                     prev_span: tuple[float, float] | None,
                     *, top_k: int = 5, floor: float = SUG_FLOOR) -> dict:
    """Les candidats d'un créneau, classés par ce qu'ils GAGNENT sur le
    précédent — au lieu de leur probabilité brute.

    Louis, 2026-09-16 : « le delta dans l'espace des postérieures de musx […]
    pour chaque créneau, on prend les 60 candidats, et au lieu de les classer
    par leur probabilité sur ce créneau, on les classe par ce qu'ils ont gagné
    depuis le créneau précédent. L'idée étant que la résonance et la pédale
    font que l'accord d'avant continue de bien scorer — le delta l'annule,
    puisqu'il était déjà haut. »

    Le pooling est celui de `musx_suggestions`, au caractère près (mêmes
    `pool_span_musx` + `acoustic_logp_musx`, même espace de 60) : SEUL le
    classement change. Deux façons de mettre en commun les postérieures
    seraient deux lois pour une décision.

    Rend `{"sug": [{root, q, c, gain}], "source": "delta"|"posterior"}` —
    `c` reste la probabilité SUR CE CRÉNEAU (ce que vaut le candidat), `gain`
    est ce qu'il a pris depuis le précédent (ce qui l'a classé). `source`
    dit laquelle des deux lois a servi, pour que l'écran ne puisse pas
    laisser croire à un classement qu'il n'a pas appliqué.

    SANS créneau précédent (le tout premier accord d'un morceau), il n'y a
    pas de delta à calculer : on rend le classement brut, et `source` le dit.
    Pas de repli muet.

    UN ÉCART ASSUMÉ par rapport à la description de Louis, à confirmer : le
    plancher s'applique AVANT le classement, pas après. Un candidat qui passe
    de 0,1 % à 2 % a « gagné » plus qu'un vrai accord qui passe de 30 % à
    31 %, et sortirait devant lui sur le delta seul. On ne classe donc par
    gain que les candidats qui valent déjà quelque chose sur ce créneau
    (`floor`, le même que les suggestions acoustiques). Sur le cas mesuré par
    Louis — Ab^7 à 6,7 % — le plancher ne change rien, il est très au-dessus.

    CE QUE ÇA NE RÉSOUT PAS, et il faut le dire : c'est une HYPOTHÈSE, pas un
    résultat. Mesurée sur 2 accords d'UN morceau (Ready, PJ Morton, mes. 3 et
    4, corrigés à l'oreille par Louis) : le delta les remonte de #3 à #2 et de
    #6 à #4 — dans le bon sens les deux fois, mais **aucun des deux n'atteint
    #1**, la pédale n'est pas complètement annulée. Règle #5 de CLAUDE.md :
    une trouvaille sur un morceau est une hypothèse. Ce qui manque pour en
    faire un résultat, c'est de la vérité terrain sur l'IDENTITÉ des accords
    (13 accords annotés en tout ; les GT de brick0 sont condamnées à
    l'oreille) — le banc par les tablatures est la piste ouverte, et il
    appartient à l'autre session.
    """
    cur_pool = pool_span_musx(probs, [span])
    p_cur = np.exp(acoustic_logp_musx(*cur_pool)[0][0])
    plancher = [i for i in range(N_CANDIDATES) if float(p_cur[i]) >= floor]
    if not plancher:                      # créneau sans aucun candidat crédible
        plancher = [int(np.argmax(p_cur))]
    if prev_span is None:
        ordre = sorted(plancher, key=lambda i: -float(p_cur[i]))
        gains = {i: None for i in plancher}
        source = "posterior"
    else:
        prev_pool = pool_span_musx(probs, [prev_span])
        p_prev = np.exp(acoustic_logp_musx(*prev_pool)[0][0])
        gains = {i: float(p_cur[i]) - float(p_prev[i]) for i in plancher}
        ordre = sorted(plancher, key=lambda i: -gains[i])
        source = "delta"
    sug = []
    s7 = cur_pool[1][0]                   # la septième que le modèle entend ICI
    for i in ordre[:top_k]:
        r, q5 = token_of(int(i))
        e = {"root": int(r), "q": queue_du_candidat(q5, s7),
             "c": round(float(p_cur[i]), 3)}
        if gains[i] is not None:
            e["gain"] = round(gains[i], 3)
        sug.append(e)
    return {"sug": sug, "source": source}


def musx_suggestions(probs: list[np.ndarray], chords: list[dict],
                     *, top_k: int = 5, floor: float = SUG_FLOOR) -> int:
    """Attach musx's own top-``top_k`` candidates to each chord as ``c["sug"]``.

    Candidates below ``floor`` are dropped — but the top one is ALWAYS kept,
    so a chord musx is certain about still opens the editor on something
    rather than on nothing (see ``SUG_FLOOR``). ``top_k`` went 3 → 5 on
    2026-09-16 at Louis's request.

    This is what the annotation editor's Compass/Guide renders as "candidates
    the model considered" — and since 2026-08-07 (Louis's report) it must BE
    that: the ranking of the model whose decode the chart displays, not
    another scorer's. The previous source (``harmonic_key._challenges``, NNLS
    chroma mass × diatonic fit) was premise-checked against musx on
    2026-07-31: musx backed the WRITTEN chord over the proposed alternate
    126/148 times (85%), the alternate's median musx posterior was 0.037, and
    0% of alternates added a note. Here ``c`` is the real quantity: the
    pooled musx posterior of that (root, family) over the chord's own span,
    given a chord is sounding (N-mass excluded, so the 60 candidates sum to 1
    per span).

    ``chords`` are flat/wire chord dicts ({t0,t1,root,q,nc,...}). N.C. and
    malformed entries are skipped; carry entries pool their carried span.
    Mutates in place (sorted by descending posterior), returns the number of
    chords annotated.

    NOT solved here — the five-family bottleneck (module docstring): a
    candidate is QUAL5, so an alternate can't distinguish ``-7`` from ``-9``.
    One exception: a candidate that IS the written chord's own (root, family)
    cell keeps the written tail, so tapping it re-picks the same chord
    instead of silently stripping its seventh. Frame mapping ignores the
    decode latency — same convention as ``musx.label_confidence``.
    """
    spans, kept = [], []
    for c in chords:
        if c.get("nc"):
            continue
        try:
            t0, t1 = float(c["t0"]), float(c["t1"])
            root = int(c["root"]) % 12
        except (KeyError, TypeError, ValueError):
            continue
        spans.append((t0, t1))
        kept.append((c, root))
    if not kept:
        return 0
    pooled_triad, pooled_s7 = pool_span_musx(probs, spans)
    logp, _n_mass = acoustic_logp_musx(pooled_triad, pooled_s7)
    post = np.exp(logp)
    for k, ((c, root), p) in enumerate(zip(kept, post)):
        own = idx_of(root, ireal_q_to_q5(c.get("q")))
        s7 = pooled_s7[k]                  # la septième que le modèle entend ICI
        sug = []
        for rank, i in enumerate(np.argsort(p)[::-1][:top_k]):
            if rank and float(p[i]) < floor:
                break          # l'échelle décroît : le reste est sous le plancher
            r, q5 = token_of(int(i))
            sug.append({"root": int(r),
                        "q": c.get("q", "") if int(i) == own
                             else queue_du_candidat(q5, s7),
                        "c": round(float(p[i]), 3)})
        c["sug"] = sug
    return len(kept)


def bass_suggestions(arr: np.ndarray, times: np.ndarray, chords: list[dict],
                     *, top_k: int = 3, floor: float = BASS_SUG_FLOOR) -> int:
    """Attach the top-``top_k`` sounding-bass readings to each chord (``c["sugBass"]``).

    Demandé par Louis le 2026-09-16 : « les suggestions sur la ligne de basse
    en faisant des petits cercles autour des lettres du cercle pour montrer
    les basses qui sont détectées ». C'est le pendant de ``musx_suggestions``
    pour la basse, et il vit ici pour la même raison : les deux champs que
    l'éditeur d'annotation affiche comme « candidats » (``sug`` et
    ``sugBass``) se décident au même endroit.

    ``arr``/``times`` sont la sortie de ``nnls_features.extract_bothchroma``
    — déjà en cache et déjà lue par ``pipeline.py`` à cette étape, donc ceci
    ne coûte aucune extraction. Chaque accord est lu à son ATTAQUE
    (``bass_pc_onset``, 150 premières ms), pas en moyenne sur tout le span :
    c'est la mesure du 2026-09-15 (6/11 → 11/11 sur "Ready"), voir
    ``nnls_features.BASS_ONSET_S``.

    Chaque entrée est ``{"pc": 0..11, "c": part de l'énergie grave à
    l'attaque, 0..1}``, triée par ``c`` décroissant. La première est toujours
    gardée ; les suivantes doivent passer ``floor``. Mêmes exclusions que
    ``musx_suggestions`` (N.C. et entrées malformées sautées, les ``carry``
    sont laissés à l'appelant). Mute en place, rend le nombre d'accords
    annotés.

    CE QUE ÇA NE FAIT PAS — et c'est le point important. Ceci n'ÉCRIT aucune
    basse : le champ ``bass`` de l'accord n'est pas touché, aucun slash
    n'apparaît dans le chart, et ``pipeline._write_sounding_bass`` — qui, lui,
    écrit — n'est pas appelé d'ici.
    C'est un AFFICHAGE de ce que la mesure voit, pas une décision. La
    distinction est ce qui permet de le brancher aujourd'hui : `known_issues`
    (2026-09-15) exige un banc corpus avant tout branchement live de la basse
    — cette exigence porte sur la DÉCISION d'écrire un slash, qui reste
    intouchée, pas sur le fait de montrer la lecture à Louis.
    """
    from harmonia.nnls_features import bass_pc_onset

    n = 0
    for c in chords:
        if c.get("nc") or c.get("carry"):
            continue
        try:
            t0, t1 = float(c["t0"]), float(c["t1"])
        except (KeyError, TypeError, ValueError):
            continue
        share = bass_pc_onset(arr, times, t0, t1)
        sug = []
        for rank, pc in enumerate(np.argsort(share)[::-1][:top_k]):
            if rank and float(share[pc]) < floor:
                break
            sug.append({"pc": int(pc), "c": round(float(share[pc]), 3)})
        c["sugBass"] = sug
        n += 1
    return n


# ── LE COMPAS EN CASCADE ────────────────────────────────────────────────────
# Louis, 2026-09-17 : « fais en sorte qu'on puisse avoir des extensions, et
# ensuite j'ai envie que tu me fasses une démo d'un compas amélioré, où je
# sélectionne d'abord l'accord en maj/min, ensuite dès qu'on le sélectionne on
# select la 7ème, puis la 9ème, puis la 11ème, puis la 13ème si elle est
# suggérée. »
#
# POURQUOI UN CHAMP À PART, ET PAS `sug`. `sug` vit dans l'espace à 60 cases
# (12 racines × QUAL5) : c'est celui du DÉCODAGE, et il replie les septièmes
# (`maj7` -> `maj`, `min7` -> `min`). La cascade a besoin de l'inverse — la
# triade nue d'un côté, les degrés ajoutés de l'autre — donc elle lit les
# têtes de musx là où elles sont encore séparées. C'est la même inférence,
# déjà en mémoire : on met en commun (`mean`) sur l'empan de l'accord et on
# écrit, aucun modèle n'est relancé.
#
# CE QUE ÇA NE RÉSOUT PAS. Les têtes d'extension ne peuvent pas CONCOURIR au
# décodage : mesuré le 2026-09-17, le vocabulaire complet (382 étiquettes)
# donne un décodage identique, la 9e ne gagne que 3 fois sur 1255 et la
# 11e/13e jamais, parce que la classe « aucune » domine et rafle Viterbi. On
# ne les fait donc pas concourir, on les MONTRE — et c'est l'oreille de Louis
# qui tranche, ce qui est exactement ce que la cascade est.
CASCADE_TOP = 8
#: les six familles de la tête de triade de musx, dans son ordre de colonnes.
TRIAD_FAMILIES = ["maj", "min", "sus4", "sus2", "dim", "aug"]
#: les quatre têtes d'extension, dans l'ordre où la cascade les demande, avec
#: le nom de chaque classe. La colonne 0 est toujours « rien ajouté ».
EXT_HEADS = (("sev", 2, ["", "maj7", "b7", "bb7"]),
             ("neuf", 3, ["", "9", "#9", "b9"]),
             ("onze", 4, ["", "11", "#11"]),
             ("treize", 5, ["", "13", "b13"]))


def cascade_suggestions(probs: list[np.ndarray], chords: list[dict],
                        *, top_k: int = CASCADE_TOP) -> int:
    """Attache à chaque accord les têtes de musx telles quelles (``c["casc"]``).

    ``probs`` est le retour de ``musx.frame_posteriors`` : ``[triade(73),
    basse(13), s7(4), s9(4), s11(3), s13(3)]``. On met en commun par moyenne
    sur l'empan de l'accord, exactement comme ``pool_span_musx``, et on écrit :

        c["casc"] = {"base": [{"root": 0-11, "type": 0-5, "c": float}, ...],
                     "sev": [4 floats], "neuf": [4], "onze": [3], "treize": [3]}

    ``base`` est le classement de la tête de TRIADE (colonne 0 = « pas
    d'accord », exclue ; colonne j>=1 -> racine ``(j-1) % 12``, famille
    ``(j-1) // 12``), tronqué à ``top_k``. Aucun plancher : le compas en
    cascade dessine ce qu'il peut et dit lui-même ce qui passe sous le seuil
    de suggestion.

    Les accords ``nc`` et ceux dont l'empan est illisible sont sautés.
    Modifie sur place, rend le nombre d'accords annotés.
    """
    from harmonia.musx import FRAME_DT

    spans, kept = [], []
    for c in chords:
        if c.get("nc"):
            continue
        try:
            t0, t1 = float(c["t0"]), float(c["t1"])
        except (KeyError, TypeError, ValueError):
            continue
        spans.append((t0, t1))
        kept.append(c)
    if not kept:
        return 0

    def _pool(head: np.ndarray) -> np.ndarray:
        n_frame = head.shape[0]
        out = np.zeros((len(spans), head.shape[1]), dtype=np.float64)
        for i, (t0, t1) in enumerate(spans):
            f0 = max(0, int(round(t0 / FRAME_DT)))
            f1 = min(n_frame, int(round(t1 / FRAME_DT)))
            if f1 <= f0:                      # empan plus court qu'une trame
                j = int(np.clip(round(0.5 * (t0 + t1) / FRAME_DT),
                                0, max(n_frame - 1, 0)))
                out[i] = head[j]
            else:
                out[i] = head[f0:f1].mean(0)
        return out

    tri = _pool(np.asarray(probs[0], dtype=float))
    ext = {nom: _pool(np.asarray(probs[i], dtype=float))
           for nom, i, _lab in EXT_HEADS}
    for k, c in enumerate(kept):
        p = tri[k]
        ordre = np.argsort(p[1:])[::-1][:top_k] + 1
        c["casc"] = {
            "base": [{"root": int((j - 1) % 12), "type": int((j - 1) // 12),
                      "c": round(float(p[j]), 4)} for j in map(int, ordre)],
            **{nom: [round(float(x), 4) for x in ext[nom][k]]
               for nom, _i, _lab in EXT_HEADS},
        }
    return len(kept)
