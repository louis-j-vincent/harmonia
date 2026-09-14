"""Sub-section folding, phase 1: OBSERVATION STACKING.

Louis's spec (2026-07-31): the fold's purpose is double — (1) stack the bars
sitting at the same position of a repeating sub-section so each chord gets
MANY observations instead of one (noise ↓ ~√n), (2) display synthesis
(cosmetic, NOT built yet). This module is (1) only:

  detect the internal period of each section (This Love's A = a 4-bar loop,
  B = a 2-bar cell) → stack same-position bars across every occurrence and
  every same-letter section → AVERAGE their musx frame posteriors → run the
  musx beat-grid decode a SECOND time on the averaged template → write the
  template's chords back onto every contributing bar.

Hard-coded exception (Louis): the LAST bar/cell of a section sometimes
changes to transition into the next section — any member too far from its
stack centroid is left UNFOLDED (variant), it keeps its own first-pass decode.

Thresholds MEASURED (2026-07-31/08-01 studies, This Love / Let It Be / Close):
  * PERIOD_MIN_SCORE = 0.80 — real loops score 0.86–0.94 (This Love A 0.944,
    B 0.859, Let It Be A 0.904); Close to You's through-composed sections
    score 0.65–0.77 and stay honestly UNFOLDED.
  * OUTLIER_Z = 3.0 — Louis's individual-vs-collective deviation rule (see
    the constant below); true variants measure z=5.7–38, normal members ≤1.7.
  * STACK_COHERENCE = 0.85 — median pairwise cos per position (bimodal guard).
Two-song-family calibration = hypothesis (CLAUDE.md rule #5).

Ce que ce module ne fait PLUS (refactor plan, décision 4, 2026-09-14 — « une
seule loi de production par choix survit, les alternatives disparaissent, git
en garde l'histoire », tag `pre-refactor-2026-09-14`) :

  * **loi de merge = moyenne des postérieures, toujours.** L'alternative CQT
    (`combine="cqt"`, moyenner les spectres avant une seconde inférence,
    validée à l'oreille le 2026-08-08 puis renversée le 2026-08-19) est
    supprimée avec `loi_de_merge()`, `MERGE_DEFAUT`, `HARMONIA_MERGE`,
    `_cqt_template`, `_adhesion`, le paramètre `cqt`/`check_thr`.
  * **repli = boucle interne PUIS occurrence, toujours** (2026-08-11) : le
    paramètre `loop` disparaît, son comportement `"occurrence"` (déjà le
    défaut de production) reste le seul chemin.
  * **veto de cohérence = par LETTRE entière, toujours** (`gate="letter"`).
    Les alternatives `"bar"` et `"bibar"` (2026-08-19, veto mesure par mesure
    ou bi-mesure) sont supprimées avec le paramètre `gate`.
  * **transposition : désactivée, toujours** (2026-08-19 sur Bora Bora — une
    section qui module n'est plus ramenée dans un ton commun avant l'empilage).
    Supprimés avec elle : le paramètre `transpose`, `decalage_semitons`,
    `_rot_probs`, `HARMONIA_FOLD_TRANSPOSE` ; `_rot12` et `_transpose_accords`
    aussi, devenus l'identité partout où ils n'étaient plus appelés qu'avec
    une rotation de 0.
  * **écriture = gabarit (re-décodé), jamais vote.** `ecriture="vote"` et
    `vote_des_passages` (2026-08-19, Bora Bora : « musx se trompe … comment
    avoir des accords + carrés ? ») n'ont jamais été branchés (aucun appelant
    de production, aucun test) ; supprimés avec `_signature`, leur seul
    dépendant.
  * **pondération/basse = défauts de production, toujours** (`weight=None`,
    `bass_mode="avg"`) : aucun appelant ne demandait `"entropy"` ou `"skip"`.
    `_entropy_weights` disparaît (son seul appelant) ; `_combine_stack` perd
    les branches `median`/`trim20`/`logpool` et son paramètre `how`.
  * **granularité quart-de-temps = lue dans `SETTINGS.quarter_bar`**, pas dans
    `HARMONIA_QUARTER_BAR` : un seul endroit lit l'environnement (`settings.py`).

Ce que ça n'a PAS changé : les seuils, l'ordre des opérations et les valeurs
numériques du chemin de production ne bougent pas — chaque loi supprimée
avait déjà son défaut réglé sur ce qui survit ici.
"""
from __future__ import annotations

import logging

import numpy as np

from harmonia import musx as _musx
from harmonia.labels import to_chord
from harmonia.settings import SETTINGS

logger = logging.getLogger(__name__)

PERIOD_MIN_SCORE = 0.80
OUTLIER_Z = 3.0          # Louis's refined rule (2026-08-01): compare each
                         # member's INDIVIDUAL deviation from the stack
                         # centroid to the COLLECTIVE norm (median + MAD of
                         # all members' deviations). Above median +
                         # OUTLIER_Z*MAD = VARIANT, keeps its own first-pass
                         # decode. Transition bars (section ends) are where
                         # variants are EXPECTED, but the test runs on every
                         # member — interior cadence cells (This Love's Ab G
                         # inside the final B, z≈38) are caught too.
                         # Measured: true variants z=5.7–38, normal members
                         # z≤1.7. Replaces the absolute cosine member gate
                         # AND the VAR_MAX position skip.
STACK_COHERENCE = 0.85   # per-position MEDIAN PAIRWISE cos of the gated
                         # stack must reach this, else the letter does NOT
                         # fold. Mean-to-centroid was tautological after the
                         # member gate; the median pairwise separates cleanly
                         # (measured: This Love positions 0.88-0.94, Let It
                         # Be's verse+chorus composite 0.74-0.81, Stand By
                         # Me 0.80 — the mixed stacks that rewrote real
                         # content, "Am F" -> "F C", both fall under 0.85).
PERIODS = (2, 4, 8)
#: Bornes du repli « occurrence contre occurrence » : sous 2 mesures il n'y a
#: pas de section, au-delà de 32 le template décodé (P mesures pavées ×3)
#: devient plus long que bien des morceaux.
MIN_OCC_PERIOD, MAX_OCC_PERIOD = 2, 32

#: Combien de mesures de FIN de section restent hors de la pile de sous-phrase
#: (voir le bloc « LA FIN D'UNE SECTION NE S'EMPILE PAS AVEC SON MILIEU »).
#: 1 suffit sur This Love, dont seule la 8e mesure sort du motif ; Louis avait
#: dit « souvent les 2 dernières » le 2026-08-19, donc 2 est le réglage à
#: essayer ensuite — il coûte une observation par passage, à mesurer avant.
FIN_SECTION_HORS_PILE = 1
CV_MAX = 0.51            # Louis's validated metric (2026-08-01): CV =
                         # std/mean of the RAW half-bar chroma across the
                         # gated stack members, per half-bar — dimensionless,
                         # volume-invariant (measured: var/mean scaled 2x
                         # with a 2x gain; CV didn't). Threshold = 5th
                         # percentile of 400 deliberately-MIXED stacks over
                         # 5 songs → false-merge rate 5.0% by construction
                         # (Louis's requirement: <5%). Corpus-wide 52% of
                         # same-written-chord stacks pass; ALL currently
                         # validated folds (This Love A/B, CV 0.31-0.48)
                         # pass. A position failing on either half-bar is
                         # NOT squashed.


#: Cosinus minimal entre les CENTROÏDES de deux lettres pour les fusionner
#: avant le repli (Louis, 2026-09-14, en écoutant les mesures de Sunny
#: Afternoon : « la section A et C sont les mêmes, il faudrait les merger »).
#: Vérifié : A (instrumental) et C (refrain) partagent la même progression —
#: cosinus 0.98–0.99 selon l'occurrence de C comparée, PLUS haut que la
#: cohérence interne de A elle-même (0.96) ou de B (0.98). songformer nomme
#: un passage par son RÔLE (couplet/refrain/instrumental…), pas par sa
#: musique — un instrumental qui rejoue le refrain reçoit sa propre lettre
#: même si les accords sont identiques.
#: Volontairement très haut : ça ne doit fusionner que ce qui est
#: authentiquement le même passage. HYPOTHÈSE, un seul morceau — à vérifier
#: sur le corpus avant de devenir le défaut (voir `SETTINGS.merge_letters`,
#: passé à `fold_letter_groups(merge_letters=...)` — plus de variable
#: d'environnement depuis le refactor, 2026-09-14).
MERGE_LETTERS_COS = 0.93


def merge_similar_letters(sections: list[dict], Vb: np.ndarray,
                          threshold: float = MERGE_LETTERS_COS) -> None:
    """Réétiquette IN PLACE les lettres dont les centroïdes se confondent.

    Compare la moyenne des vecteurs de mesure (même substrat que la
    détection et le repli) de chaque lettre à chaque autre ; au-dessus du
    seuil, la plus tardive des deux prend le nom de la plus ancienne
    (`_lettres()` nomme déjà par ordre de première apparition — on prolonge
    la même convention). `intro`/`outro` ne sont jamais comparées : ce sont
    des repères de structure, pas de la matière musicale à confondre avec
    autre chose.

    NE RÈGLE PAS : deux occurrences fusionnées peuvent avoir des longueurs
    différentes (ex. le refrain court de 6 mesures) — `fold_letter_groups`
    et `minimal_fold` gèrent déjà ce cas (empilement occurrence contre
    occurrence, groupage par (lettre, longueur)), donc rien de plus n'est
    nécessaire ici, mais ce n'est pas cette fonction qui le garantit.
    """
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        groups[s["label"]].append(s)
    letters = [l for l in groups if l not in ("intro", "outro")]
    if len(letters) < 2:
        return

    def centroid(label):
        idx = [b for s in groups[label]
               for b in range(s["barRanges"][0][0], s["barRanges"][0][1] + 1)]
        v = Vb[idx].mean(axis=0)
        return v / max(np.linalg.norm(v), 1e-9)

    cent = {l: centroid(l) for l in letters}
    first = {l: min(s["barRanges"][0][0] for s in groups[l]) for l in letters}
    order = sorted(letters, key=lambda l: first[l])
    remap = {l: l for l in letters}
    for i, li in enumerate(order):
        for lj in order[i + 1:]:
            if remap[li] != li or remap[lj] != lj:
                continue                      # déjà fusionnée ce tour-ci
            sim = float(cent[li] @ cent[lj])
            if sim >= threshold:
                remap[lj] = li
                logger.info("fold: lettres %s et %s fusionnées (centroïdes "
                            "à %.3f) — %s absorbe %s", li, lj, sim, li, lj)
    if all(v == k for k, v in remap.items()):
        return
    for s in sections:
        s["label"] = remap.get(s["label"], s["label"])


def _bar_vecs(F: np.ndarray, n_bars: int) -> np.ndarray:
    """(n_bars, 48) unit vectors from the half-bar features (2 rows per bar)."""
    V = np.array([F[2 * b:2 * b + 2].reshape(-1) for b in range(n_bars)])
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def section_period(Vb: np.ndarray, b0: int, b1: int) -> tuple[int | None, float]:
    """Smallest confident repeating period of a section, or (None, best)."""
    L = b1 - b0 + 1
    scores = {}
    for P in PERIODS:
        if L < 2 * P:
            continue
        scores[P] = float(np.mean([Vb[b] @ Vb[b + P]
                                   for b in range(b0, b1 + 1 - P)]))
    if not scores:
        return None, 0.0
    best = max(scores.values())
    if best < PERIOD_MIN_SCORE:
        return None, best
    pick = min(P for P, v in scores.items() if v >= max(PERIOD_MIN_SCORE,
                                                        0.95 * best))
    return pick, scores[pick]


def _resample(mat: np.ndarray, n_out: int) -> np.ndarray:
    """Linear time-resample of a (T, k) posterior block to (n_out, k)."""
    T = mat.shape[0]
    if T == 0:
        return np.zeros((n_out, mat.shape[1]), dtype=mat.dtype)
    if T == 1:
        return np.repeat(mat, n_out, axis=0)
    xs = np.linspace(0, T - 1, n_out)
    lo = np.floor(xs).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (xs - lo)[:, None]
    return (1 - w) * mat[lo] + w * mat[hi]


def fold_letter_groups(sections, bars, grid, probs, bpb: int,
                       arr=None, times=None, *,
                       merge_letters: bool = False) -> dict:
    """Stack + re-decode + redistribute, per letter group. Mutates `bars`
    IN PLACE (each bar list object is shared with the section slices).

    Returns the fold report: {letter: {period, positions: n_obs list,
    variants: [bar indices], changed: [bar indices]}} — the data a future
    validation UI ("interface ludique") will present.

    LA loi de production, sans alternative (refactor décision 4, voir « Ce que
    ce module ne fait PLUS » en tête de fichier) : moyenne des postérieures,
    boucle interne puis occurrence entière, veto de cohérence par LETTRE
    entière, pas de transposition.

      merge_letters False (déf.) | True — fusionne d'abord les lettres dont
                les centroïdes se confondent (voir `merge_similar_letters`),
                AVANT le groupage par lettre ci-dessous. Sans lui, deux
                occurrences identiques nommées différemment par le
                détecteur de sections ne se rencontrent jamais. Seul réglage
                qui reste un argument explicite : recherche de qualité de
                section en cours (2026-09-14), pas encore une loi tranchée.
    """
    from harmonia_min.sections import halfbar_features
    n_bars = len(grid) - 1
    F = halfbar_features(grid, arr, times)   # same substrate as detection
    Vb = _bar_vecs(F, n_bars)

    def _raw_half(b: int, half: int):
        mid = 0.5 * (grid[b] + grid[b + 1])
        t0, t1 = ((grid[b], mid), (mid, grid[b + 1]))[half]
        sel = (times >= t0) & (times < t1)
        return arr[sel].mean(0) if sel.any() else \
            arr[int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))]
    # frame slices per bar on the musx grid
    def bar_probs(b):
        a = max(0, int(round(grid[b] / _musx.FRAME_DT)))
        z = min(probs[0].shape[0], int(round(grid[b + 1] / _musx.FRAME_DT)))
        return [p[a:z] for p in probs]

    med_bar = float(np.median(np.diff(grid)))
    Lf = max(bpb, int(round(med_bar / _musx.FRAME_DT)))   # frames per bar

    if merge_letters:
        merge_similar_letters(sections, Vb)

    # ── group sections by letter, pick the group period ─────────────────────
    groups: dict[str, list[dict]] = {}
    for s in sections:
        groups.setdefault(s["label"], []).append(s)
    report = {}
    for letter, secs in groups.items():
        picks = []
        for s in secs:
            b0, b1 = s["barRanges"][0]
            P, sc = section_period(Vb, b0, b1)
            if P is not None:
                picks.append(P)
        if not picks:
            # LA SECTION ELLE-MÊME EST LA PÉRIODE (2026-08-11).
            # Le repli ne savait empiler que les positions d'une BOUCLE
            # INTERNE (2, 4 ou 8 mesures dans la section). Une section
            # écrite d'un trait — un couplet de 8 mesures qui ne se répète
            # pas à l'intérieur — n'a aucune boucle, donc la lettre était
            # refusée ENTIÈRE… alors même qu'elle est jouée 10 fois.
            # C'est le cas que Louis décrivait au départ (« quand une section
            # est jouée N fois, on a N observations du MÊME enchaînement »),
            # et c'était le refus le plus fréquent du corpus : 61 lettres sur
            # 162, dont 45 avec au moins deux occurrences de MÊME longueur.
            # On empile alors occurrence contre occurrence : position k = la
            # k-ième mesure de la section.
            lens = [b1 - b0 + 1 for b0, b1 in (s["barRanges"][0] for s in secs)]
            common = [L for L in set(lens) if lens.count(L) >= 2
                      and MIN_OCC_PERIOD <= L <= MAX_OCC_PERIOD]
            if common:
                P = max(common, key=lambda L: (lens.count(L), L))
                secs = [s for s in secs
                        if (s["barRanges"][0][1] - s["barRanges"][0][0] + 1) == P]
                logger.info("fold %s: pas de boucle interne — on empile les "
                            "%d occurrences de %d mesures entre elles",
                            letter, len(secs), P)
            else:
                # Longueurs toutes différentes : under-fold, never over-fold.
                report[letter] = {"period": None,
                                  "reason": "no confident loop, occurrences "
                                            "of unequal length"}
                continue
        else:
            P = int(np.bincount(picks).argmax())  # group consensus period

        # ── stacks: position k -> member bar indices ─────────────────────────
        #
        # LA FIN D'UNE SECTION NE S'EMPILE PAS AVEC SON MILIEU (Louis,
        # 2026-08-20 : « attention quand tu empiles toujours pareil à ne pas
        # empiler les fins de sections qui sont vraiment différentes — je pense
        # à This Love »).
        #
        # Le piège est propre à l'empilement par SOUS-PHRASE. This Love B fait
        # 8 mesures bâties sur une pompe de 2 : `C- F- | B♭ E♭` quatre fois…
        # sauf la dernière, qui est `A♭ G`. Avec P=2, la mesure 8 tombe dans la
        # même pile que les mesures 2, 4 et 6 — la cadence se fait écraser par
        # trois fois plus de mesures qui disent autre chose, et le chart perd
        # exactement la mesure qui fait la chanson.
        #
        # La dernière mesure de chaque passage sort donc de la pile et garde
        # son décodage de première passe (mécanisme `variants`, déjà en place).
        # UNIQUEMENT sous P < longueur de section : quand la période EST la
        # section (cas « occurrence contre occurrence »), les fins s'empilent
        # entre elles, ce qui est le cas sain — une cadence contre les
        # cadences des autres passages.
        pos_members: list[list[int]] = [[] for _ in range(P)]
        section_last = set()
        fins_hors_pile: list[int] = []
        for s in secs:
            b0, b1 = s["barRanges"][0]
            section_last.add(b1)
            sous_phrase = P < (b1 - b0 + 1)
            for b in range(b0, b1 + 1):
                if sous_phrase and b > b1 - FIN_SECTION_HORS_PILE:
                    fins_hors_pile.append(b)
                    continue
                pos_members[(b - b0) % P].append(b)
        if fins_hors_pile:
            logger.info("fold %s: %d fin(s) de section gardée(s) hors de la "
                        "pile de sous-phrase (mes. %s)", letter,
                        len(fins_hors_pile),
                        [b + 1 for b in sorted(fins_hors_pile)])

        variants, gated = list(fins_hors_pile), [[] for _ in range(P)]
        for k in range(P):
            mem = pos_members[k]
            if len(mem) < 3:
                gated[k] = mem
                continue
            cen = np.mean([Vb[b] for b in mem], axis=0)
            cen /= max(np.linalg.norm(cen), 1e-9)
            d = {b: 1.0 - float(Vb[b] @ cen) for b in mem}
            med = float(np.median(list(d.values())))
            mad = float(np.median(np.abs(np.array(list(d.values())) - med))) + 1e-6
            for b in mem:
                if (d[b] - med) / mad <= OUTLIER_Z:
                    gated[k].append(b)
                else:
                    variants.append(b)            # écart individuel anormal vs
                                                  # l'écart collectif → variante,
                                                  # garde son décodage 1ʳᵉ passe
        if sum(len(g) for g in gated) < 2 * P:
            report[letter] = {"period": P, "reason": "too few gated members"}
            continue
        coh_by_pos: list[float | None] = [None] * P
        for k in range(P):
            g = gated[k]
            if len(g) < 2:
                continue
            pw = [float(Vb[a] @ Vb[b])
                  for i, a in enumerate(g) for b in g[i + 1:]]
            coh_by_pos[k] = float(np.median(pw))
        coh = [c for c in coh_by_pos if c is not None]
        # Veto de cohérence PAR LETTRE ENTIÈRE : min(coh) < STACK_COHERENCE
        # refuse toute la lettre (pas de veto mesure-par-mesure ni bi-mesure —
        # voir « Ce que ce module ne fait PLUS » en tête de fichier).
        if not coh or min(coh) < STACK_COHERENCE:
            report[letter] = {"period": P,
                              "reason": f"stack incoherent (min median pairwise "
                                        f"{min(coh):.2f})" if coh else "stacks too thin"}
            continue

        # CV squash-verifier (Louis's metric, calibrated FP<5%): a position
        # whose gated members vary too much on either half-bar is NOT
        # squashed — every occurrence keeps its own decode.
        cv_skip = set()
        for k in range(P):
            g = gated[k]
            if len(g) < 2:
                continue
            for half in (0, 1):
                X = np.array([_raw_half(b, half) for b in g])
                c_v = float(np.sqrt(X.var(0).mean())
                            / max(X.mean(0).mean(), 1e-9))
                if c_v > CV_MAX:
                    cv_skip.add(k)
        if cv_skip:
            logger.info("fold %s: positions %s not squashed (CV > %.2f)",
                        letter, sorted(cv_skip), CV_MAX)

        # Audit note (2026-08-01): a position with ZERO gated members borrows
        # one rejected variant bar as decode CONTEXT only — the write loop
        # below iterates gated[k], so such a position is never rewritten.
        # Accepted risk: the variant's posteriors mildly colour the
        # neighbouring positions' transitions in the template decode.
        pos_chords = _template_chords(
            [g or [pos_members[k][0]] for k, g in enumerate(gated)],
            bar_probs, len(probs), Lf, bpb, P)
        if pos_chords is None:
            report[letter] = {"period": P, "reason": "template decoded empty"}
            continue

        n_obs = [len(g) for g in gated]
        changed = []
        for k in range(P):
            if not pos_chords[k] or k in cv_skip:
                continue                          # empty or CV-refused
            for b in gated[k]:
                if _write_position(bars, grid, b, pos_chords[k], bpb, n_obs[k]):
                    changed.append(b)
        # cv_skip DANS le report (handoff §2.3 : l'affichage repliait ×N ce
        # que le code avait refusé d'écraser, faute de cette clé).
        # `demiton`/`pos_skip` : toujours vides depuis le refactor (la
        # transposition et le veto mesure-par-mesure/bi-mesure ont disparu,
        # voir « Ce que ce module ne fait PLUS ») — gardés dans le rapport
        # pour ne pas casser sa forme (le rapport d'or le compare tel quel).
        report[letter] = {"period": P, "n_obs": n_obs,
                          "demiton": {},
                          "variants": sorted(set(variants)),
                          "changed": sorted(set(changed)),
                          "cv_skip": sorted(cv_skip),
                          "pos_skip": [],
                          "coh": [None if c is None else round(c, 3)
                                  for c in coh_by_pos]}
        logger.info("fold %s: P=%d, obs/pos %s, %d variants, %d bars changed",
                    letter, P, n_obs, len(set(variants)), len(set(changed)))
    return report




def _combine_stack(M: np.ndarray) -> np.ndarray:
    """Combine les membres (axe 0) d'une pile (n, Lf, k) → (Lf, k) : la
    moyenne — N tirages indépendants du même accord, indulgent. Seule loi de
    production ; `median`/`trim20`/`logpool` et la pondération par entropie
    étaient des alternatives jamais choisies, supprimées (voir « Ce que ce
    module ne fait PLUS »)."""
    return np.mean(M, axis=0)


def _template_chords(pos_members, bar_probs, n_probs, Lf, bpb, P):
    """Average each position's member-bar posteriors, decode the P-bar
    template once (tiled ×3 against Viterbi edge effects), return the
    per-position chord lists (sustains write a carry at beat 0), or None
    if the template decodes empty.
    """
    tmpl = []
    for k in range(P):
        mems = [bar_probs(b) for b in pos_members[k]]
        blocks = [[_resample(m[i], Lf) for m in mems] for i in range(n_probs)]
        avg = [_combine_stack(np.stack(blocks[i])) for i in range(n_probs)]
        tmpl.append(avg)
    cat = [np.concatenate([tmpl[k][i] for k in range(P)] * 3)
           for i in range(n_probs)]
    return _decode_template(cat, Lf, bpb, P)


def _decode_template(cat, Lf, bpb, P):
    """Le décodage du gabarit empilé."""
    step = Lf * _musx.FRAME_DT / bpb
    beats = [i * step for i in range(3 * P * bpb + 1)]
    # Half-bar transitions cost the same as bar transitions (15), quarter-
    # bar stays expensive (100): Louis 2026-08-01 — the aggregated posterior
    # showed Ddim lasting only HALF its window but the default mid-bar cost
    # (45) glued it to a full bar. Consistent with the half-bar snap rule.
    # NOTE (2026-08-07): since the half-bar-only zeroing, "expensive (100)"
    # had silently become FORBIDDEN — the grade-4 beats were dropped before
    # the penalty could apply. Restriction lifted the same day (Louis: « on ne
    # met plus de restrictions sur la granularité »): the documented cost is
    # the real one again, so the template decode cannot erase sub-half-bar
    # chords the first pass found. `SETTINGS.quarter_bar = False` restores the
    # half-bar-only template decode (settings.py is the only place this is
    # read; it used to be `HARMONIA_QUARTER_BAR`, see module docstring).
    _q = "all" if SETTINGS.quarter_bar else None
    lab, _ = _musx.redecode(beats, cat, downbeat_times=beats[::bpb],
                            beat_trans_penalty=(15.0, 15.0, 100.0),
                            quarter_beats=_q)
    T0, T1 = P * Lf * _musx.FRAME_DT, 2 * P * Lf * _musx.FRAME_DT
    # The window that keeps the MIDDLE copy of the ×3-tiled template needs a
    # tolerance of half a musx frame, not 1e-6 s (Louis, 2026-08-09: « pk sur
    # This Love on loupe le Cm du refrain ? »). `redecode` returns boundaries
    # snapped to musx's own 23.2 ms frame grid, so the chord that starts
    # exactly ON the window edge lands up to half a frame BEFORE it — the
    # chorus's Cm came out of the Viterbi at 5.052 s against T0 = 5.062 s and
    # was thrown away for 10 ms, leaving the bar showing only its second
    # chord. Cost of the 1e-6 tolerance, measured across the served library:
    # 18 chords heard with 0.85–0.94 confidence and silently dropped, ALL of
    # them on the first beat of fold position 0 — the only slot that sits on
    # the window edge. The decode itself was never wrong: on This Love the
    # Viterbi returns Cm Fm Bb Eb exactly, aggregation and all.
    edge_tol = _musx.FRAME_DT / 2
    events = []
    for t0, t1, l in lab:
        if t0 < T0 - edge_tol or t0 >= T1 - edge_tol:
            continue
        beat = int(round((max(t0, T0) - T0) / step))
        # ACOUSTIC confidence, measured on the averaged template this decode
        # actually read — the same quantity, on the same scale, as an unfolded
        # chord's. Averaging n members legitimately raises it, so a folded
        # chord scores higher for the right reason instead of being handed
        # `0.5 + 0.08*n_obs`, which is a repetition count wearing a posterior's
        # clothes (measured cost: 2 points of AUC, see musx.label_confidence).
        conf = round(_musx.label_confidence(cat[0], t0, min(t1, T1), l), 3)
        events.append((beat // bpb, beat % bpb, l, conf))
    if not events or all(l == "N" for _, _, l, _ in events):
        return None
    pos_chords: list[list[dict]] = [[] for _ in range(P)]
    cur = None
    for k in range(P):
        evk = [e for e in events if e[0] == k]
        if (not evk or evk[0][1] > 0) and cur is not None:
            pos_chords[k].append({**cur, "beat": 0, "carry": True})
        for _, beat, l, conf in evk:
            ch = to_chord(l)
            entry = ({"root": 0, "q": "", "bass": -1, "nc": True,
                      "beat": beat, "c": conf}
                     if ch is None else {**ch, "nc": False, "beat": beat,
                                         "c": conf})
            pos_chords[k].append(entry)
            cur = {kk: vv for kk, vv in entry.items()
                   if kk not in ("beat", "carry")}
    return pos_chords


def _write_position(bars, grid, b, chords_k, bpb, n_obs):
    """Rewrite bar b with a template position's chords (real bar times).

    `c` comes per-chord from the template decode (an acoustic posterior on the
    averaged evidence); `n_obs` is carried ALONGSIDE it, never folded into it.
    Two chords in one bar can be supported to different degrees and must be
    allowed to say so — the old signature took one confidence for the whole bar
    because that confidence was a property of the section, not of the chord.
    """
    # CE QUE LOUIS A ÉCRIT NE SE FAIT PAS ÉCRASER (2026-08-20 : « les
    # annotations utilisateur prennent toujours le dessus sur nos
    # inférences »). `confirmed` est posé par l'éditeur d'annotation ; le repli
    # ne le regardait pas, donc une mesure corrigée à la main repassait sous le
    # gabarit au recuit suivant — et rien ne le disait. Une seule mesure suffit
    # à bloquer SA mesure, pas la lettre : le reste de la pile s'écrit
    # normalement.
    if any(c.get("confirmed") for c in bars[b]):
        logger.info("repli: mesure %d gardée telle quelle — accord confirmé "
                    "à la main", b + 1)
        return False
    bw = grid[b + 1] - grid[b]
    new = []
    for j, e in enumerate(chords_k):
        t0 = grid[b] + e["beat"] / bpb * bw
        nxt = (chords_k[j + 1]["beat"] / bpb * bw
               if j + 1 < len(chords_k) else bw)
        new.append({"root": e["root"], "q": e["q"], "bass": e["bass"],
                    "nc": e["nc"], "carry": bool(e.get("carry")),
                    "beat": e["beat"], "bar": b,
                    "c": float(e.get("c", 0.5)), "n_obs": n_obs,
                    "folded": True,
                    "t0": round(t0, 3), "t1": round(grid[b] + nxt, 3)})
    changed = [(c["root"], c["q"], c.get("carry", False)) for c in bars[b]] \
        != [(c["root"], c["q"], c.get("carry", False)) for c in new]
    bars[b][:] = new
    return changed


# ── phase 2: DISPLAY fold (Louis, 2026-08-01) ────────────────────────────────

def display_fold(sections: list[dict], bars, grid, probs=None, bpb=4,
                 loop_folded: set | None = None) -> list[dict]:
    """Write a repeated section ONCE ×N — the ChartModel reps/spans/barSpans
    contract app_shell already renders — with the divergent tail (≤2 last
    bars) carried per variant as `endings` ("le débordement en dessous").

    UNDER-FOLD doctrine (Louis, 2026-07-30): only same-letter sections of the
    SAME length whose bars agree everywhere except the last ≤2 fold together;
    a pass of a different length stays written out (This Love: B8+B8 fold ×2,
    the 24-bar final B stays its own block).
    """
    def barsig(b):
        return tuple((c["root"], c["q"], c["nc"]) for c in bars[b])

    used = [False] * len(sections)
    out = []
    for i, sec in enumerate(sections):
        if used[i]:
            continue
        b0, b1 = sec["barRanges"][0]
        L = b1 - b0 + 1
        group = [i]
        for j in range(i + 1, len(sections)):
            if used[j] or sections[j]["label"] != sec["label"]:
                continue
            c0, c1 = sections[j]["barRanges"][0]
            if c1 - c0 + 1 != L:
                continue
            diff = [r for r in range(L) if barsig(b0 + r) != barsig(c0 + r)]
            if all(r >= L - 2 for r in diff):
                group.append(j)
        if len(group) == 1:
            out.append(sec)
            used[i] = True
            continue
        for j in group:
            used[j] = True
        ranges = [sections[j]["barRanges"][0] for j in group]
        # ── CROSS-PASS observation stacking (Louis, 2026-08-01): the folded
        # passes' musx posteriors are averaged position by position and
        # re-decoded, so the folded block shows a CONSENSUS of all passes —
        # not pass 0's chords. Skipped when the letter already loop-folded in
        # phase 1 (its bars are already a template consensus pooled across
        # every occurrence, including unfolded-length passes like This Love's
        # 24-bar final B — a broader pool than these passes alone). The
        # divergent tail (endings) keeps each pass's own decode.
        tail_probe = 0
        for c0, _ in ranges[1:]:
            diff = [r for r in range(L) if barsig(b0 + r) != barsig(c0 + r)]
            if diff:
                tail_probe = max(tail_probe, L - min(diff))
        if probs is not None and sec["label"] not in (loop_folded or set()):
            med_bar = float(np.median(np.diff(grid)))
            Lf = max(bpb, int(round(med_bar / _musx.FRAME_DT)))

            def bar_probs(b):
                a = max(0, int(round(grid[b] / _musx.FRAME_DT)))
                z = min(probs[0].shape[0],
                        int(round(grid[b + 1] / _musx.FRAME_DT)))
                return [pp[a:z] for pp in probs]

            Pp = L - tail_probe
            if Pp < 1:
                Pp = 0                            # nothing shared: no stack
            members = [[c0 + r for c0, _ in ranges] for r in range(Pp)]
            pos_chords = _template_chords(members, bar_probs, len(probs),
                                          Lf, bpb, Pp) if Pp >= 1 else None
            if pos_chords is not None:
                k_obs = len(ranges)
                nch = 0
                for r in range(Pp):
                    if not pos_chords[r]:
                        continue
                    for c0, _ in ranges:
                        nch += _write_position(bars, grid, c0 + r,
                                               pos_chords[r], bpb, k_obs)
                logger.info("display fold %s: cross-pass stack ×%d, %d bars "
                            "rewritten to consensus", sec["label"], k_obs, nch)
        # tail depth = deepest divergence vs pass 0, within the last 2 bars
        tail = 0
        for c0, _ in ranges[1:]:
            diff = [r for r in range(L) if barsig(b0 + r) != barsig(c0 + r)]
            if diff:
                tail = max(tail, L - min(diff))
        prefix_len = L - tail
        folded = {
            "id": sec["id"], "label": sec["label"], "tag": "",
            "reps": len(group),
            "spans": [[grid[c0], grid[c1 + 1]] for c0, c1 in ranges],
            "barRanges": [[c0, c1] for c0, c1 in ranges],
            "bars": bars[b0:b0 + prefix_len]
                    + bars[b0 + prefix_len:b1 + 1],   # prefix + rep tail
            "barSpans": [[[grid[c0 + r], grid[c0 + r + 1]] for c0, _ in ranges]
                         for r in range(prefix_len)],
        }
        if tail:
            # variants: passes grouped by their tail content; each variant's
            # bars come from ITS first pass (absolute times of that pass)
            bytail: dict[tuple, list[int]] = {}
            for pi, (c0, _) in enumerate(ranges):
                key = tuple(barsig(c0 + r) for r in range(prefix_len, L))
                bytail.setdefault(key, []).append(pi)
            variants = []
            for key, passes in sorted(bytail.items(), key=lambda kv: kv[1][0]):
                v0 = ranges[passes[0]][0]
                variants.append({"passes": passes,
                                 "bars": bars[v0 + prefix_len:v0 + L]})
                for r in range(prefix_len, L):
                    folded["barSpans"].append(
                        [[grid[ranges[pi][0] + r], grid[ranges[pi][0] + r + 1]]
                         for pi in passes])
            folded["endings"] = {"tail": tail, "variants": variants}
            # representative block keeps the FULL bar list (prefix + pass-0
            # tail) — an endings-aware renderer slices bars[:-tail] itself
        logger.info("display fold: %s ×%d (tail %d, %d variant(s))",
                    sec["label"], len(group), tail,
                    len(folded.get("endings", {}).get("variants", [])) or 1)
        out.append(folded)
    return out


# ── phase 2b: MINIMAL fold into the EXISTING app UI (Louis, 2026-08-02) ─────

def minimal_fold(sections, bars, grid, fold_report) -> list[dict]:
    """One ChartModel section per LETTER — the validated minimal folding,
    rendered by the UNCHANGED app_shell (Louis: the folding logic is right,
    the old UI stays). Golden rule: chords written once.

    Block per letter (validated on /min): the fold cell padded to >=4 bars;
    if pass tails diverge — equal-length passes: the whole representative
    pass (This Love B = 3 cells + cadence); unequal: cell + divergent tail
    cell (This Love A = 8 bars, « 8 suffisent largement »).
    reps = number of passes; spans/barRanges per pass (chronological);
    barSpans[r] maps rendered bar r onto each pass PROPORTIONALLY (the old
    app's own contract for length-mismatched passes). The FORM strip then
    shows the sequence — the timeline that fait foi.
    """
    def sig(b):
        return tuple((c["root"], c["q"], c["nc"]) for c in bars[b]) \
            if 0 <= b < len(bars) else ()

    def t_at(b0, x):                      # grid time at fractional bar b0+x
        i = int(x)
        f = x - i
        lo = grid[min(len(grid) - 1, b0 + i)]
        hi = grid[min(len(grid) - 1, b0 + i + 1)]
        return round(lo + f * (hi - lo), 4)

    # GROUPÉ PAR (LETTRE, LONGUEUR), pas par lettre seule — « under-fold,
    # never over-fold » (Louis, 2026-07-30 : « écris chaque section à la
    # longueur qu'elle joue vraiment »). Regrouper toutes les occurrences
    # d'une lettre quelles que soient leurs longueurs, c'est écrire UN bloc
    # pour des passages qui n'ont pas la même musique : sur Another Day, un D
    # dont la 1re occurrence fait 8 mesures était rendu en 4, et la lecture
    # se décalait d'une barre à chaque reprise (Louis, 2026-08-18 : « la
    # dernière section c'est la section A mais décalée d'une barre »).
    # `soudure.sections_pour_chart` groupait déjà comme ça depuis le chemin
    # Soudure ; les deux chemins rendaient donc DEUX charts différents pour le
    # même découpage. Les blocs gardent le même nom : deux longueurs d'un même
    # refrain restent deux B, chacun écrit à sa longueur.
    by_letter: dict[tuple, list] = {}
    order: list[tuple] = []
    for s in sections:
        for r in s["barRanges"]:
            k = (s["label"], int(r[1]) - int(r[0]))
            if k not in by_letter:
                order.append(k)
            by_letter.setdefault(k, []).append(tuple(r))
    out = []
    _vus: dict[str, int] = {}

    def _evidence(rng):
        """How many REAL chord onsets a pass carries (carries and N.C. don't
        count) — what makes a pass worth showing as the letter's block."""
        c0, c1 = rng
        return sum(1 for b in range(c0, c1 + 1) for c in (bars[b] if 0 <= b < len(bars) else [])
                   if not c["nc"] and not c.get("carry"))

    for _k in order:
        L = _k[0]
        ranges = sorted(by_letter[_k])
        # The block shown for a letter used to be `ranges[0]` — the
        # chronologically first pass, whatever it contained. On a song that
        # fades in, that is the WORST pass: Stand By Me's first A holds 2
        # onsets over 8 bars (the fade-in) while its eight later A's are
        # identical and complete (F#m D E A), so the chart rendered ~empty
        # while the song's chords were sitting in passes we never displayed
        # (Louis, 2026-08-10: « je vois plein d'accords, et sur le chart j'ai
        # juste des NC partout »). Pick the pass with the most real onsets
        # instead; ties keep the earliest, so every already-correct chart is
        # untouched (a letter whose passes agree has equal evidence).
        # `ranges` STAYS chronological — spans/barRanges/the playhead map all
        # depend on it. Only which pass is WRITTEN changes.
        b0, b1 = max(ranges, key=lambda r: (_evidence(r), -ranges.index(r)))
        # Only a fold that was ACCEPTED may drive the display. Every refusal
        # in fold_letter_groups still records the period it was testing
        # ({"period": P, "reason": ...} at the "too few gated members",
        # "stack incoherent" and "template decoded empty" exits) — reading
        # `period` without checking `reason` silently rebuilt the chart from a
        # loop the stacker had just rejected. Norah Jones' A was refused at
        # "stack incoherent (min median pairwise 0.61)" and displayed anyway:
        # P=2 wrote `Bb Bb | Eb^7 D` twice over a first pass that really goes
        # `Bb Bb | Eb^7 D | G-7 C7 | F7 F7`, so bars 2-7 never reached the
        # chart and 72.6% of A's playing time was misrepresented (2026-08-02).
        # Under-fold, never over-fold: a refused fold writes its bars out.
        _rep = fold_report.get(L) or {}
        P = None if _rep.get("reason") else _rep.get("period")
        cell = [bars[b] for b in range(b0, min(b1, b0 + P - 1) + 1)] if P \
            else [bars[b] for b in range(b0, b1 + 1)]
        ref_tail = [sig(ranges[0][1] - k) for k in (1, 0)]
        div = next(((c0, c1) for c0, c1 in ranges[1:]
                    if [sig(c1 - k) for k in (1, 0)] != ref_tail), None)
        lens = {c1 - c0 + 1 for c0, c1 in ranges}
        # Bars of the representative pass that the fold itself flagged as NOT
        # matching the cell. They are real music the cell does not contain, so
        # the written block has to reach them (Louis, 2026-08-05, on Norah
        # Jones: « tu me bouffes la répétition de la fin du A ; tu peux
        # techniquement l'intégrer au A mais il faut l'ÉCRIRE sur le chart, car
        # c'est pas clair là »). Her A was three passes of 22 / 8 / 20 bars
        # written as 4 bars × 3, with 13 bars — the 2-bar tag among them —
        # flagged as variants and never rendered. Under-fold, never over-fold.
        var_in_pass = [v for v in (_rep.get("variants") or []) if b0 <= v <= b1]
        if P and div and len(lens) == 1:
            block_rng = list(range(b0, b1 + 1))
        elif P:
            block_rng = list(range(b0, b0 + len(cell)))
            while len(block_rng) < min(4, b1 - b0 + 1):
                block_rng += list(range(b0, b0 + len(cell)))
            block_rng = block_rng[:max(len(cell), min(4, b1 - b0 + 1))]
            if var_in_pass:
                end = min(b1, max(var_in_pass))
                if end >= b0 + len(block_rng):
                    block_rng = list(range(b0, end + 1))
        else:
            block_rng = list(range(b0, b1 + 1))
        Lb = len(block_rng)
        is_pass_block = bool(P and div and len(lens) == 1)
        tail_len = 0                              # Louis 2026-08-02: le bloc
        tail_sigs = []                            # = la cellule, échelle de
                                                  # répétition minimale (×N
                                                  # vivent dans la strip)
        # ── playhead map BY CONTENT (fix 2026-08-02: the proportional map
        # stretched an 8-bar block over a 16-bar pass — by the cell's 2nd
        # repetition the highlight sat on the wrong rows). A cell row lights
        # at EVERY repetition of the cell inside every pass (multiple time
        # windows per row — the UI's tspans accept any number); tail rows
        # light only on the pass whose ending matches the divergent tail.
        rows = [[] for _ in range(Lb)]
        for c0, c1 in ranges:
            Lk = c1 - c0 + 1
            pass_tail = bool(tail_len) and \
                [sig(b) for b in range(c1 - tail_len + 1, c1 + 1)] == tail_sigs
            for b in range(c0, c1 + 1):
                if pass_tail and b > c1 - tail_len:
                    r = len(cell) + (b - (c1 - tail_len + 1))
                elif P and not is_pass_block:
                    r = (b - c0) % (len(cell) if tail_len else Lb)
                else:
                    r = min(Lb - 1, int((b - c0) * Lb / Lk))
                rows[r].append([grid[b], grid[min(len(grid) - 1, b + 1)]])
        _vus[L] = _vus.get(L, 0) + 1
        out.append({
            "id": f"L{L}" + ("" if _vus[L] == 1 else str(_vus[L])),
            "label": L, "tag": "", "reps": len(ranges),
            "spans": [[grid[c0], grid[min(len(grid) - 1, c1 + 1)]]
                      for c0, c1 in ranges],
            "barRanges": [[c0, c1] for c0, c1 in ranges],
            "bars": [bars[b] for b in block_rng],
            "barSpans": rows,
        })
    # ── LA FAÇON IREAL : une lettre = un seul bloc écrit ────────────────────
    # (Louis, 2026-09-14 : « tu ne peux pas afficher un A deux fois »).
    # Spec complète : docs/spec_affichage_sections.md.
    for blk in out:
        _ireal_endings(blk, bars, grid)
    return _ireal_cascade(out, bars, grid, fold_report)


# ── LA FAÇON IREAL ─────────────────────────────────────────────────────────
# Voir docs/spec_affichage_sections.md. Une lettre nomme UN bloc écrit ; la
# répétition est une marque (×N, fins 1./2., prime), jamais une deuxième copie
# de la lettre. iReal écrit `*A` une fois et met le reste dans `{ }`, `N1`/`N2`,
# `S`/`Q` — deux `*A` dans une grille n'existent pas.

#: Marques de variante, dans l'ordre (règle 5). A′ se lit « A prime », c'est la
#: notation Real Book pour « la même section, mais pas tout à fait ».
_PRIMES = ("\u2032", "\u2033", "\u2034", "\u2057")

#: Au-delà, ce n'est plus une fin alternative, c'est une autre musique.
_ENDING_TAIL_MAX = 2


def _barsig(bar) -> tuple:
    """Ce qu'une mesure JOUE, pour comparer deux passages entre eux."""
    return tuple((c["root"], c["q"], bool(c["nc"])) for c in (bar or []))


def _ireal_endings(blk: dict, bars, grid) -> None:
    """Règle 2 : des passages qui ne diffèrent QUE sur leur queue deviennent
    UN bloc avec une 1ʳᵉ et une 2ᵉ fin. Mute `blk` sur place.

    Le rendu existe depuis 2026-07-21 dans `app_shell.html` (le crochet
    « ⌐1. » suspendu) mais rien ne l'alimentait sur le chemin vivant : il
    n'était écrit que par `display_fold`, que la pipeline n'appelle pas. Le
    contrat qu'il attend, dans cet ordre : `bars` = le passage ENTIER,
    `endings.variants[i].bars` = la queue de ce variant, et `barSpans` listé
    « les mesures du tronc commun, puis la queue de chaque variant ».

    Ne s'applique QUE si le bloc écrit est un passage entier : quand le repli
    a déjà réduit le bloc à sa cellule (4 mesures pour un passage de 8), il
    n'y a pas de place pour suspendre une queue.
    """
    ranges = [tuple(r) for r in blk["barRanges"]]
    if len(ranges) < 2 or blk.get("endings"):
        return
    lens = {c1 - c0 + 1 for c0, c1 in ranges}
    if len(lens) != 1:
        return
    L = lens.pop()
    if len(blk["bars"]) != L or L < 3:
        return
    sigs = [[_barsig(bars[c0 + r]) for r in range(L)] for c0, _ in ranges]
    prefix_len = L
    for r in range(L):
        if len({s[r] for s in sigs}) > 1:
            prefix_len = r
            break
    tail = L - prefix_len
    if not (1 <= tail <= _ENDING_TAIL_MAX) or prefix_len < 2:
        return
    bytail: dict[tuple, list[int]] = {}
    for pi, s in enumerate(sigs):
        bytail.setdefault(tuple(s[prefix_len:]), []).append(pi)
    if len(bytail) < 2:
        return
    # UNE VRAIE 2e FIN CHANGE D'ACCORD, PAS DE COULEUR (2026-09-14). Sur Sunny
    # Afternoon la queue des trois refrains ne diffère que par `C7` contre
    # `Cm7` sur la dernière mesure — et cette mesure-là est justement celle que
    # le repli met hors de la pile (la cadence de fin de section), donc son
    # décodage est du premier jet, passage par passage. Suspendre un crochet
    # « 1./2. » pour ça, c'est sur-noter du bruit. On exige donc que la queue
    # change de FONDAMENTALE quelque part (This Love : `B♭ E♭` contre `A♭ G`).
    roots = {tuple(tuple(ch[0] for ch in bar) for bar in key)
             for key in bytail}
    if len(roots) < 2:
        return
    variants = []
    rows = [[[grid[c0 + r], grid[min(len(grid) - 1, c0 + r + 1)]]
             for c0, _ in ranges] for r in range(prefix_len)]
    for _key, passes in sorted(bytail.items(), key=lambda kv: kv[1][0]):
        v0 = ranges[passes[0]][0]
        variants.append({"passes": passes,
                         "bars": [bars[v0 + r] for r in range(prefix_len, L)]})
        for r in range(prefix_len, L):
            rows.append([[grid[ranges[pi][0] + r],
                          grid[min(len(grid) - 1, ranges[pi][0] + r + 1)]]
                         for pi in passes])
    blk["endings"] = {"tail": tail, "variants": variants}
    blk["barSpans"] = rows
    logger.info("ireal: %s ×%d → une 1re/2e fin (tronc %d mes., queue %d, "
                "%d variantes)", blk["label"], len(ranges), prefix_len, tail,
                len(variants))


def _ireal_cascade(out: list[dict], bars, grid, fold_report) -> list[dict]:
    """Règles 3 et 5 + l'invariant : jamais deux blocs sous la même étiquette.

    Règle 3 — un passage COUPÉ (il s'arrête en cours de boucle) rejoint le
    bloc long : même lettre, `reps` augmenté, et il n'allume que les mesures
    qu'il joue réellement. C'est le cas de Sunny Afternoon, dont le « A de 6
    mesures » est la cellule de 4 jouée une fois et demie.

    Règle 5 — tout ce que la cascade n'a pas réduit prend un prime (A′, A″).
    Un deuxième « A » nu ne sort jamais d'ici.
    """
    by_label: dict[str, list[dict]] = {}
    for blk in out:
        by_label.setdefault(blk["label"], []).append(blk)
    drop: set[int] = set()
    for label, blks in by_label.items():
        if len(blks) < 2:
            continue
        # l'hôte = le plus joué (à égalité, le premier dans le morceau)
        host = max(blks, key=lambda b: (b["reps"], -out.index(b)))
        rep = (fold_report or {}).get(label) or {}
        # un repli REFUSÉ ne sert pas de cellule (sa `period` est celle qu'il
        # testait, pas celle qu'il a validée) — on passe alors au test préfixe
        P = None if rep.get("reason") else rep.get("period")
        cell = [_barsig(b) for b in host["bars"][:P]] if P else None
        rank = 0
        for blk in blks:
            if blk is host:
                continue
            if _merge_coupe(host, blk, cell, P, grid):
                drop.add(id(blk))
                continue
            rank += 1
            blk["label"] = label + (_PRIMES[rank - 1] if rank <= len(_PRIMES)
                                    else str(rank))
            logger.info("ireal: %s devient %s (même lettre, autre musique)",
                        label, blk["label"])
    return [b for b in out if id(b) not in drop]


def _merge_coupe(host: dict, blk: dict, cell, P, grid) -> bool:
    """Règle 3. Rend True si `blk` a été absorbé par `host`.

    Le test se fait contre la CELLULE de la lettre, pas contre le bloc long :
    c'est la cellule qui est l'unité réelle (le bloc long, c'est la cellule
    jouée deux fois, avec parfois une mesure qui garde la lecture d'un
    passage). Sunny Afternoon en est l'exemple : le bloc de 8 a un `B♭m7` de
    plus en mesure 2, le bloc de 6 suit la cellule à la note près.
    """
    if host.get("endings") or len(blk["bars"]) >= len(host["bars"]):
        return False
    if not P or not cell:
        # Pas de cellule connue (repli refusé, ou découpage fait à la main :
        # le chemin Soudure n'a pas de rapport de repli) — on retombe sur le
        # test le plus simple et le plus sûr : le bloc court est-il le DÉBUT
        # du bloc long ? Un passage qui commence pareil et s'arrête plus tôt
        # est un passage coupé, que le repli des accords ait abouti ou non.
        cell, P = [_barsig(b) for b in host["bars"]], len(host["bars"])
    if any(_barsig(bar) != cell[i % P] for i, bar in enumerate(blk["bars"])):
        return False
    n_rows = len(host["barSpans"])
    for c0, c1 in blk["barRanges"]:
        for b in range(c0, c1 + 1):
            r = b - c0
            if r < n_rows:
                host["barSpans"][r].append(
                    [grid[b], grid[min(len(grid) - 1, b + 1)]])
    host["barRanges"] += [list(r) for r in blk["barRanges"]]
    host["spans"] += [list(s) for s in blk["spans"]]
    host["reps"] += blk["reps"]
    logger.info("ireal: passage coupé mes. %s rejoint %s (×%d désormais)",
                blk["barRanges"], host["label"], host["reps"])
    return True
