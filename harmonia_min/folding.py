"""harmonia_min/folding.py — sub-section folding, phase 1: OBSERVATION STACKING.

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
"""
from __future__ import annotations

import logging
import os

import numpy as np

from harmonia_min import musx as _musx
from harmonia_min.labels import to_chord

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
#: Bornes du repli « occurrence contre occurrence » (loop="occurrence") :
#: sous 2 mesures il n'y a pas de section, au-delà de 32 le template décodé
#: (P mesures pavées ×3) devient plus long que bien des morceaux.
MIN_OCC_PERIOD, MAX_OCC_PERIOD = 2, 32
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
                       gate: str = "letter", combine: str = "mean",
                       weight: str | None = None,
                       bass_mode: str = "avg", cqt=None,
                       check_thr: float | None = None,
                       loop: str = "internal") -> dict:
    """Stack + re-decode + redistribute, per letter group. Mutates `bars`
    IN PLACE (each bar list object is shared with the section slices).

    Returns the fold report: {letter: {period, positions: n_obs list,
    variants: [bar indices], changed: [bar indices]}} — the data a future
    validation UI ("interface ludique") will present.

    Les quatre mots-clés sont les leviers du chantier merge-d'occurrences
    (branche feat/occurrence-merge, handoff 2026-08-08) ; leurs défauts
    reproduisent la prod à l'identique :

      gate      "letter" (prod) : min(coh) < STACK_COHERENCE refuse TOUTE la
                lettre. "bibar" : le veto se décide par BI-MESURE (la
                granularité des blocs détectés en phase 1, directive Louis
                2026-08-08) — une bi-mesure incohérente est seulement exclue
                de l'écriture (pos_skip), les autres gardent leur merge.
                Mesuré (sonde levier 1) : 21/44 positions individuellement
                cohérentes (≥0.85) étaient jetées par le veto lettre.
      combine   "mean" (prod) | "median" | "trim20" | "logpool" — levier 2/3.
      weight    None (prod) | "entropy" — levier 4 (pondération par la
                confiance de chaque occurrence, via l'entropie de ses frames).
      bass_mode "avg" (prod) | "skip" — levier 5 (la basse hors de la
                moyenne : flux basse rendu uniforme dans le template, chaque
                occurrence garde son inversion).
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

    def bar_cqt(b):
        """Le CQT de la mesure b, sur la même grille de frames."""
        a = max(0, int(round(grid[b] / _musx.FRAME_DT)))
        z = min(cqt.shape[0], int(round(grid[b + 1] / _musx.FRAME_DT)))
        return cqt[a:z]

    med_bar = float(np.median(np.diff(grid)))
    Lf = max(bpb, int(round(med_bar / _musx.FRAME_DT)))   # frames per bar

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
        if not picks and loop == "occurrence":
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
        elif not picks:
            report[letter] = {"period": None, "reason": "no confident loop"}
            continue
        else:
            P = int(np.bincount(picks).argmax())  # group consensus period

        # ── stacks: position k -> member bar indices ─────────────────────────
        pos_members: list[list[int]] = [[] for _ in range(P)]
        section_last = set()
        for s in secs:
            b0, b1 = s["barRanges"][0]
            section_last.add(b1)
            for b in range(b0, b1 + 1):
                pos_members[(b - b0) % P].append(b)

        variants, gated = [], [[] for _ in range(P)]
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
            pw = [float(Vb[a] @ Vb[b]) for i, a in enumerate(g) for b in g[i + 1:]]
            coh_by_pos[k] = float(np.median(pw))
        coh = [c for c in coh_by_pos if c is not None]
        pos_skip: set[int] = set()
        if gate == "letter":
            if not coh or min(coh) < STACK_COHERENCE:
                report[letter] = {"period": P,
                                  "reason": f"stack incoherent (min median pairwise "
                                            f"{min(coh):.2f})" if coh else "stacks too thin"}
                continue
        else:
            # gate="bibar" : le veto de cohérence se décide par bi-mesure —
            # la granularité que la phase 1 des sections détecte (blocs de 2,
            # UNIT=2). Une bi-mesure dont une position mesurée passe sous le
            # seuil est exclue de l'écriture ; une position sans mesure (<2
            # membres) suit le verdict de sa partenaire, comme la prod qui ne
            # la comptait pas dans le min. Une lettre ne refuse en bloc que
            # si TOUTES ses bi-mesures sont exclues.
            for j in range((P + 1) // 2):
                ks = [k for k in (2 * j, 2 * j + 1) if k < P]
                vals = [c for k in ks if (c := coh_by_pos[k]) is not None]
                if not vals or min(vals) < STACK_COHERENCE:
                    pos_skip.update(ks)
            if len(pos_skip) == P:
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
            bar_probs, len(probs), Lf, bpb, P,
            combine=combine, weight=weight, bass_mode=bass_mode,
            bar_cqt=(bar_cqt if cqt is not None else None),
            check_thr=check_thr)
        if pos_chords is None:
            report[letter] = {"period": P, "reason": "template decoded empty"}
            continue

        n_obs = [len(g) for g in gated]
        changed = []
        for k in range(P):
            if not pos_chords[k] or k in cv_skip or k in pos_skip:
                continue                          # empty, CV- or bibar-refused
            for b in gated[k]:
                if _write_position(bars, grid, b, pos_chords[k], bpb, n_obs[k]):
                    changed.append(b)
        # cv_skip/pos_skip DANS le report (handoff §2.3 : l'affichage repliait
        # ×N ce que le code avait refusé d'écraser, faute de cette clé).
        report[letter] = {"period": P, "n_obs": n_obs,
                          "variants": sorted(set(variants)),
                          "changed": sorted(set(changed)),
                          "cv_skip": sorted(cv_skip),
                          "pos_skip": sorted(pos_skip),
                          "coh": [None if c is None else round(c, 3)
                                  for c in coh_by_pos]}
        logger.info("fold %s: P=%d, obs/pos %s, %d variants, %d bars changed",
                    letter, P, n_obs, len(set(variants)), len(set(changed)))
    return report




def _renorm(X: np.ndarray) -> np.ndarray:
    """Re-normalise chaque frame sur son simplexe (médiane/produit cassent
    la somme à 1 que le décodeur attend)."""
    return X / np.maximum(X.sum(axis=1, keepdims=True), 1e-9)


def _combine_stack(M: np.ndarray, how: str, w=None) -> np.ndarray:
    """Combine les membres (axe 0) d'une pile (n, Lf, k) → (Lf, k).

    mean     prod : N tirages indépendants, indulgent.
    median   robuste à UN membre contaminé (solo, modulation) — levier 2.
    trim20   moyenne tronquée (~20 % par bout, n≥4) — levier 2 bis.
    logpool  produit des postérieures (= somme des log-probs) : « le même
             accord vu N fois », un dissident confiant a un veto — levier 3.
    `w` (poids par membre, levier 4) ne s'applique qu'à mean/logpool.
    """
    if how == "mean":
        return np.average(M, axis=0, weights=w)
    if how == "median":
        return _renorm(np.median(M, axis=0))
    if how == "trim20":
        n = M.shape[0]
        if n < 4:
            return np.average(M, axis=0, weights=w)
        t = max(1, n // 5)
        S = np.sort(M, axis=0)[t:n - t]
        return _renorm(S.mean(axis=0))
    if how == "logpool":
        return _renorm(np.exp(np.average(np.log(np.clip(M, 1e-9, 1.0)),
                                         axis=0, weights=w)))
    raise ValueError(f"combine: unknown mode {how!r}")


def _entropy_weights(blocks0: list[np.ndarray]) -> np.ndarray | None:
    """Poids par membre depuis l'entropie moyenne de SES frames triad —
    une occurrence noyée (solo par-dessus) est plate donc entropique, elle
    pèse moins ; gratuit, aucune vérité terrain requise (levier 4)."""
    if len(blocks0) < 2:
        return None
    Hs = []
    for X in blocks0:
        p = np.clip(X, 1e-9, 1.0)
        Hs.append(float(-(p * np.log(p)).sum(axis=1).mean()))
    Hs = np.asarray(Hs)
    w = np.exp(-(Hs - Hs.min()))          # le moins entropique pèse 1
    s = w.sum()
    return w / s if s > 1e-9 else None


def _cqt_template(pos_members, bar_cqt, Lf, P, drop=()):
    """Le CQT MOYENNÉ des membres, position par position, pavé ×3.

    C'est l'agrégation que Louis a retenue à l'oreille le 2026-08-08 (« les
    CQT moyennés ça marche très bien ») : on empile les répétitions EN AMONT
    du modèle, dans le domaine du spectre. Pas le signal — il se déphase et
    la superposition devient une bouillie ; pas les postérieures — c'est
    l'aval, et le modèle a déjà tranché avant qu'on additionne.

    Le pavage ×3 se fait sur le CQT lui-même, avant l'inférence : le réseau
    voit ainsi un vrai contexte de part et d'autre des coutures.
    """
    blocks = []
    for k in range(P):
        mems = [b for b in pos_members[k] if b not in drop] or pos_members[k]
        blocks.append(np.mean([_resample(bar_cqt(b), Lf) for b in mems],
                              axis=0))
    return np.concatenate(blocks * 3)


def _adhesion(pos_members, bar_probs, events, Lf, bpb, P, thr):
    """« Regarder le score de prédiction final de musx et voir s'il est sous
    un certain seuil » (Louis, 2026-08-08), placé AVANT l'addition.

    L'adhésion d'un membre = le score que musx donne aux accords du consensus
    quand il ne regarde que CE membre. Elle n'est mesurée que sur les accords
    du consensus qui sont eux-mêmes confiants : punir un membre de ne pas
    coller à un accord que le consensus ne soutient pas exclurait à tort.
    """
    anchors = [e for e in events if e[3] >= thr]
    if not anchors:
        return {}, set()
    step = Lf * _musx.FRAME_DT / bpb
    out, drop = {}, set()
    for k in range(P):
        for b in pos_members[k]:
            tri = _resample(bar_probs(b)[0], Lf)
            scs = []
            for kk, beat, lab, _c in anchors:
                if kk != k:
                    continue
                a = max(0, int(round(beat * step / _musx.FRAME_DT)))
                z = min(tri.shape[0], a + max(1, int(round(step / _musx.FRAME_DT))))
                if z <= a:
                    continue
                col = _musx.TRIAD_FAMILY.get(lab.partition(":")[2])
                if col is None:
                    continue
                from harmonia_min.labels import parse_root
                j = 1 + (col - 1) * 12 + parse_root(lab.partition(":")[0])
                scs.append(float(tri[a:z, j].mean()))
            if scs:
                out[b] = round(float(np.mean(scs)), 3)
                if out[b] < thr:
                    drop.add(b)
    return out, drop


def _template_chords(pos_members, bar_probs, n_probs, Lf, bpb, P,
                     combine: str = "mean", weight: str | None = None,
                     bass_mode: str = "avg", bar_cqt=None,
                     check_thr: float | None = None):
    """Average each position's member-bar posteriors, decode the P-bar
    template once (tiled ×3 against Viterbi edge effects), return the
    per-position chord lists (sustains write a carry at beat 0), or None
    if the template decodes empty.

    `combine="cqt"` remplace la moyenne de postérieures par la moyenne de
    CQT suivie d'une nouvelle inférence — l'agrégation validée à l'oreille.
    `check_thr` arme le contrôle d'adhésion : les membres qui ne soutiennent
    pas le consensus sont écartés et le template est refait sans eux.
    """
    if combine == "cqt" and bar_cqt is not None:
        cat = [np.asarray(x, dtype=np.float64) for x in
               _musx.posteriors_from_cqt(_cqt_template(pos_members, bar_cqt,
                                                       Lf, P))]
        return _decode_template(cat, pos_members, bar_probs, Lf, bpb, P,
                                bar_cqt=bar_cqt, check_thr=check_thr)
    tmpl = []
    for k in range(P):
        mems = [bar_probs(b) for b in pos_members[k]]
        blocks = [[_resample(m[i], Lf) for m in mems] for i in range(n_probs)]
        w = _entropy_weights(blocks[0]) if weight == "entropy" else None
        avg = [_combine_stack(np.stack(blocks[i]), combine, w)
               for i in range(n_probs)]
        if bass_mode == "skip" and n_probs > 1:
            # levier 5 : la basse SORT de la moyenne — deux reprises changent
            # d'inversion exprès (la cible du projet est la basse SONNANTE).
            # Flux basse rendu uniforme : le template ne tranche pas
            # l'inversion, chaque occurrence garde la sienne en 1ʳᵉ passe.
            avg[1] = np.full_like(avg[1], 1.0 / avg[1].shape[1])
        tmpl.append(avg)
    cat = [np.concatenate([tmpl[k][i] for k in range(P)] * 3)
           for i in range(n_probs)]
    return _decode_template(cat, pos_members, bar_probs, Lf, bpb, P)


def _decode_template(cat, pos_members, bar_probs, Lf, bpb, P,
                     bar_cqt=None, check_thr=None):
    """Le décodage commun aux deux lois de combinaison."""
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
    # chords the first pass found. HARMONIA_QUARTER_BAR=off restores the
    # half-bar-only template decode.
    _q = None if os.environ.get("HARMONIA_QUARTER_BAR", "").strip().lower() \
        in ("off", "0", "false") else "all"
    lab, _ = _musx.redecode(beats, cat, downbeat_times=beats[::bpb],
                            beat_trans_penalty=(15.0, 15.0, 100.0),
                            quarter_beats=_q)
    T0, T1 = P * Lf * _musx.FRAME_DT, 2 * P * Lf * _musx.FRAME_DT
    events = []
    for t0, t1, l in lab:
        if t0 < T0 - 1e-6 or t0 >= T1 - 1e-6:
            continue
        beat = int(round((t0 - T0) / step))
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
    # ── contrôle d'adhésion, EN AMONT de l'addition (Louis, 2026-08-08) ────
    # Une répétition qui ne soutient pas le consensus est écartée et le
    # template refait sans elle. Uniquement sur la loi CQT : c'est la seule
    # où « refaire l'addition » veut dire quelque chose (on ré-infère sur un
    # spectre moyen différent). Jamais sur une pile de moins de trois membres
    # — écarter là ne laisserait plus de consensus.
    if check_thr is not None and bar_cqt is not None:
        adh, drop = _adhesion(pos_members, bar_probs, events, Lf, bpb, P,
                              check_thr)
        drop = {b for b in drop
                if any(b in pos_members[k] and len(pos_members[k]) >= 3
                       for k in range(P))}
        if drop:
            logger.info("adhésion < %.2f : %d membre(s) écarté(s) %s",
                        check_thr, len(drop), {b: adh[b] for b in sorted(drop)})
            cat2 = [np.asarray(x, dtype=np.float64) for x in
                    _musx.posteriors_from_cqt(
                        _cqt_template(pos_members, bar_cqt, Lf, P, drop=drop))]
            return _decode_template(cat2, pos_members, bar_probs, Lf, bpb, P)
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

    by_letter: dict[str, list] = {}
    order: list[str] = []
    for s in sections:
        L = s["label"]
        if L not in by_letter:
            order.append(L)
        by_letter.setdefault(L, []).extend(
            [tuple(r) for r in s["barRanges"]])
    out = []
    for L in order:
        ranges = sorted(by_letter[L])
        b0, b1 = ranges[0]
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
        out.append({
            "id": f"L{L}", "label": L, "tag": "", "reps": len(ranges),
            "spans": [[grid[c0], grid[min(len(grid) - 1, c1 + 1)]]
                      for c0, c1 in ranges],
            "barRanges": [[c0, c1] for c0, c1 in ranges],
            "bars": [bars[b] for b in block_rng],
            "barSpans": rows,
        })
    return out
