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
    from harmonia_min.musx import FRAME_DT

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
#: CE N'EST PAS `bass_rules.FLOOR` (30 %), et la confusion des deux serait
#: exactement l'erreur de calibration silencieuse que CLAUDE.md décrit
#: (règle #1). Les deux seuils ne répondent pas à la même question :
#:   * `bass_rules.FLOOR` = 30 % : « est-ce que j'ÉCRIS un slash ? » — une
#:     décision qui change le chart, arbitrée à l'oreille sur 33 cas ;
#:   * `BASS_SUG_FLOOR` = 12,5 % : « est-ce que je MONTRE cette lecture ? » —
#:     un affichage qui ne change rien au chart et n'engage personne.
#: À 30 %, 84-90 % des accords n'afficheraient qu'une seule note : la demande
#: de Louis (voir la ligne de basse) serait vide de sens. À 12,5 %, mesuré :
#: 2,2 notes par accord en moyenne, un tiers des accords en montrent 3.
BASS_SUG_FLOOR = 0.125


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
    for (c, root), p in zip(kept, post):
        own = idx_of(root, ireal_q_to_q5(c.get("q")))
        sug = []
        for rank, i in enumerate(np.argsort(p)[::-1][:top_k]):
            if rank and float(p[i]) < floor:
                break          # l'échelle décroît : le reste est sous le plancher
            r, q5 = token_of(int(i))
            sug.append({"root": int(r),
                        "q": c.get("q", "") if int(i) == own else Q5_TAIL[q5],
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
    n'apparaît dans le chart, ``bass_rules.decide_bass`` n'est pas appelé.
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
