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

import json
import logging

import numpy as np

from harmonia import musx as _musx
from harmonia.labels import to_chord
# re-exportés : ils vivaient ici avant le 2026-10-01 (tests, outils)
from harmonia.roles import accord_sonnant, famille_de, lectures, meme_lecture  # noqa: F401
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
#: L'UNANIMITÉ DIRECTE BAT LA MOYENNE INDIRECTE (Louis, 2026-09-24 : « on ne
#: devrait pas souvent contredire musx »).
#:
#: Le gabarit est un outil de LISSAGE : on moyenne les postérieures des
#: occurrences d'une position et on re-décode. Quand les passages propres d'une
#: mesure sont UNANIMES contre ce gabarit sur la tierce, ce sont eux qui ont
#: raison — on ne lisse pas une observation unanime.
#:
#: LE CAS QUI L'A RÉVÉLÉ, This Love, fin de refrain. La mesure écrite n'a que
#: deux passages, et les DEUX entendent un fa majeur :
#:
#:     occurrence 1 (56,6 s)   F7   confiance 0,629
#:     occurrence 2 (107,1 s)  F    confiance 0,675
#:     écrit sur le chart      F-   ← contredit ses deux sources
#:
#: Le mineur venait du gabarit : le repliement avait détecté une boucle interne
#: de 2 mesures, donc cette mesure était empilée avec les mesures 17, 19, 21 de
#: chaque occurrence — 18 postérieures moyennées, dominées par les fa mineurs
#: de la boucle. Le tab de Louis écrit lui aussi un fa majeur là, et son oreille
#: a tranché dans le même sens.
#:
#: La règle ne demande aucun seuil : pas de distance, pas de cohérence. Elle
#: compare la tierce du gabarit à celle des passages de la MÊME POSITION DANS
#: LA SECTION (pas dans la boucle interne), et s'abstient d'écrire quand ils
#: sont unanimes contre lui.
UNANIMITE_BAT_GABARIT = True

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


#: FUSIONNER DEUX LETTRES : on compare ce qu'on a DÉCODÉ, pas ce qu'on a
#: ENTENDU (2026-09-18, six morceaux analysés puis vérifiés à l'oreille par
#: Louis — `docs/plots/sections_6morceaux.html`).
#:
#: songformer (supprimé 2026-09-30) nommait un passage par son RÔLE (couplet/refrain/instrumental),
#: pas par sa musique : sur Fallin', qui est UNE boucle E-…B- du début à la
#: fin, il rend huit lettres. Rien en aval ne lui répond « harmoniquement,
#: c'est le même passage » — d'où cette fonction.
#:
#: CE QUI A ÉTÉ MESURÉ ET ÉCARTÉ. La première version comparait les
#: CENTROÏDES de chroma (seuil 0.93). Mesurée sur les six : juste sur cinq,
#: elle CASSE Let It Be (couplet ↔ refrain à 0.970, ils fusionnaient), et
#: aucun seuil ne sauve les six — fusionner Fallin' exige ≤ 0.961, garder
#: Let It Be exige > 0.967. Deux défauts du substrat, pas du seuil : un
#: centroïde est un SAC de notes (`C G | A- F | C G | F C` et
#: `A- G | F C | C G | F C` ont le même sac), et la chroma NNLS porte la
#: PRODUCTION en plus de l'harmonie. Rendre la mesure sensible à l'ordre en
#: comparant les SÉQUENCES de chroma a été essayé aussi : échoue pareil
#: (Fallin' 0.839 contre Let It Be 0.864, l'ordre reste inversé).
#:
#: CE QUI SÉPARE : les accords décodés. Une mesure devient ses couples
#: (fondamentale, famille) — `B-` et `B-7` sont le même moment, `B-` et `B`
#: non ; une mesure muette est un JOKER, sinon un couplet nu de basse ne
#: ressemble plus à sa propre reprise (Stand By Me : 0.67 → 0.88). Sur les
#: six, la fenêtre valable est (0.50, 0.75] — on prend le milieu.
MERGE_LETTERS_ACCORDS = 0.65
#: Deux lettres qui ne se ressemblent pas ASSEZ mais partagent leur TÊTE sont
#: un même passage à deux queues — une 1re/2e fin, pas deux sections (Sunny
#: A/B et C/D, Every Breath A/A″). Les fusionner ici est ce qui permet à
#: `_ireal_endings`, qui ne regarde jamais à travers deux lettres, de voir
#: enfin les queues divergentes. Mesuré : les trois cas partagent 3 mesures
#: de tête, TOUTES les paires qui doivent rester séparées en partagent 0.
MERGE_LETTERS_PREFIXE = 3
#: Un bloc plus court que ça ne sert pas de preuve de préfixe : c'est un
#: fragment (Let It Be écrit trois « sections » de 2 mesures).
MERGE_LETTERS_MIN_BARS = 4
#: Gardé pour la page de mesure `tools/page_sections.py`, qui montre les deux
#: substrats côte à côte. N'est plus consulté par la production.
MERGE_LETTERS_COS = 0.93
#: Règle 2 du 2026-10-01 (`dernier_tour_partiel`) : quand la 1re mesure d'un
#: dernier tour ne colle pas au A, il en faut au moins autant qui collent
#: juste après pour croire que c'est le début du A.
DERNIER_TOUR_MIN_COLLE = 2
#: À partir de ce cran du curseur, la dernière section d'une AUTRE lettre
#: peut rejoindre une lettre comme son dernier tour (`_ireal_cascade`).
CRAN_DERNIER_TOUR_AUTRE_LETTRE = 3


def _sig_repli(bar) -> tuple | None:
    """Une mesure vue par la fusion : ses couples (fondamentale, famille).

    `None` quand la mesure ne porte aucun accord réel — c'est un JOKER, qui
    s'accorde avec n'importe quoi. Sans ça, le couplet nu de Stand By Me (que
    la basse seule tient, donc décodé en N.C.) ne ressemblait plus à sa propre
    reprise complète : 0.67 au lieu de 0.88.
    """
    reels = [c for c in (bar or []) if not c.get("nc")]
    if not reels:
        return None
    return _MesureRepli(reels)


class _MesureRepli(tuple):
    """Les accords réels d'une mesure, vus par la fusion des lettres : deux
    mesures sont égales (`==`, ce qu'utilisent `concordance`,
    `tete_commune` et `periode_interne`) quand elles ont autant d'accords et
    que chaque paire est le même accord au sens de `meme_lecture` (grain
    "famille"). Pas hachable : ce n'est pas une égalité transitive."""
    __hash__ = None

    def __eq__(self, other):
        if not isinstance(other, _MesureRepli):
            return NotImplemented
        return len(self) == len(other) and all(
            meme_lecture(a, b, grain="famille") for a, b in zip(self, other))

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r


def periode_interne(A: list) -> int:
    """La plus petite période qui décrit la suite — 2 pour `E-|B-|E-|B-`, la
    longueur entière pour un passage à travers-composé.

    Elle borne les ROTATIONS qu'on s'autorise. Sans cette borne, la
    concordance cyclique fait passer n'importe quelle rotation pour le même
    passage : sur l'Autumn Leaves d'iReal — une structure écrite par un
    humain, donc notre vérité terrain — la section B est la section A
    décalée de 4 mesures, et elles fusionnaient. Une boucle de 2 mesures a le
    droit de commencer sur l'une ou l'autre (c'est Fallin', dont les lettres
    sont coupées à des endroits différents de la même boucle) ; un couplet de
    8 mesures n'a pas le droit de commencer au milieu de lui-même.
    """
    n = len(A)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        if not any(x is not None for x in A[:p]) and any(x is not None for x in A):
            continue          # une cellule faite QUE de jokers ne décrit rien
        if all(A[i] is None or A[i % p] is None or A[i] == A[i % p]
               for i in range(n)):
            return p
    return n


def concordance(A: list, B: list) -> float:
    """Meilleure concordance de deux passages, rapportée au plus court.

    ON COMPARE LES CELLULES, PAS LES BLOCS TUILÉS. Chaque suite est d'abord
    réduite à sa période interne : sinon un accord commun est compté autant de
    fois que la boucle se répète, et deux progressions différentes se
    rapprochent artificiellement. Sur Hot N Cold, `G D A- C` et `E- C G D`
    sortaient à 0.75 en tuilé sur 8 mesures (le `G D` compté deux fois) — et
    à 0.50 cellule contre cellule, ce qui est la vérité : deux des quatre
    accords diffèrent.

    UNE ROTATION NE SE JUSTIFIE QUE SUR UNE BOUCLE : quand la plus petite
    période est strictement plus courte que le plus long des deux passages.
    Fallin' (période 2, blocs de 2 à 9) oui ; Autumn Leaves (période 8, blocs
    de 8) non — sans ça la section B, qui est la A décalée de 4 mesures, passe
    pour la même section.

    UN JOKER (mesure sans accord propre) s'accorde avec tout mais ne prouve
    rien : si moins de la moitié des positions comparées sont de vrais accords
    des deux côtés, on rend 0. Sans ça, sur le Syracuse d'iReal, une section
    de 2 mesures tombait en face de mesures tenues d'une section sans rapport
    et sortait à 1.00.
    """
    if not A or not B:
        return 0.0
    # La preuve qu'on a affaire à une BOUCLE se lit sur les longueurs
    # d'ORIGINE (une cellule de 2 vue sur 8 mesures), pas après réduction —
    # une fois réduites, deux cellules ont souvent la même taille et la
    # preuve aurait disparu.
    pa, pb = periode_interne(A), periode_interne(B)
    boucle = min(pa, pb) < max(len(A), len(B))
    A, B = A[:pa], B[:pb]
    k = min(len(A), len(B))
    rot = len(B) if boucle else 1
    best = 0.0
    for r in range(rot):
        ok = reels = 0
        for j in range(k):
            a, b = A[j % len(A)], B[(j + r) % len(B)]
            if a is not None and b is not None:
                reels += 1
                if a == b:
                    ok += 1
            else:
                ok += 1                        # joker : s'accorde, ne prouve rien
        if reels * 2 < k:
            continue                           # trop de jokers pour conclure
        best = max(best, ok / k)
    return best


def tete_commune(A: list, B: list) -> int:
    """Combien de mesures de TÊTE deux passages partagent VRAIMENT — les
    jokers ne comptent pas.

    C'est la mesure d'un passage à deux queues (1re/2e fin) que la
    concordance globale rate. Compter les jokers ici était un trou béant : la
    lettre A de Hot N Cold porte 5 mesures muettes sur 8, elle « partageait
    donc sa tête » avec toutes les autres et absorbait le morceau entier.
    """
    n = 0
    for a, b in zip(A, B):
        if a is None or b is None:
            continue                           # ne prouve rien, ne casse rien
        if a != b:
            break
        n += 1
    return n


def merge_similar_letters(sections: list[dict], bars: list,
                          seuil: float = MERGE_LETTERS_ACCORDS,
                          prefixe: int = MERGE_LETTERS_PREFIXE) -> None:
    """Réétiquette IN PLACE les lettres qui sont le même passage.

    Deux lettres se rencontrent si leurs ACCORDS DÉCODÉS concordent (voir
    `MERGE_LETTERS_ACCORDS` pour ce qui a été mesuré et écarté), ou si elles
    partagent assez de mesures de tête pour n'être qu'un passage à deux
    queues. La plus tardive prend le nom de la plus ancienne — `_lettres()`
    nomme déjà par ordre de première apparition, on prolonge la convention.
    `intro`/`outro` ne sont jamais comparées : ce sont des repères de
    structure, pas de la matière à confondre.

    NE RÈGLE PAS : deux occurrences fusionnées peuvent avoir des longueurs
    différentes — `fold_letter_groups` et `minimal_fold` gèrent déjà ce cas
    (empilement occurrence contre occurrence, groupage par (lettre,
    longueur)) ; ni la TRANSPOSITION (Sunny rejoue sa forme un demi-ton plus
    haut, et le repli ne compare jamais deux hauteurs).
    """
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        groups[s["label"]].append(s)
    letters = [l for l in groups if l not in ("intro", "outro")]
    if len(letters) < 2:
        return

    def suite(label):
        b0, b1 = groups[label][0]["barRanges"][0]
        return [_sig_repli(bars[b]) for b in range(b0, min(b1 + 1, len(bars)))]

    suites = {l: suite(l) for l in letters}
    first = {l: min(s["barRanges"][0][0] for s in groups[l]) for l in letters}
    order = sorted(letters, key=lambda l: first[l])
    remap = {l: l for l in letters}
    for i, li in enumerate(order):
        for lj in order[i + 1:]:
            if remap[li] != li or remap[lj] != lj:
                continue                      # déjà fusionnée ce tour-ci
            A, B = suites[li], suites[lj]
            sim = concordance(A, B)
            tete = tete_commune(A, B)
            assez_long = min(len(A), len(B)) >= MERGE_LETTERS_MIN_BARS
            if sim >= seuil:
                pourquoi = f"accords concordants à {sim:.2f}"
            elif assez_long and tete >= prefixe:
                pourquoi = f"{tete} mesures de tête communes (passage à deux queues)"
            else:
                continue
            remap[lj] = li
            logger.info("fold: lettres %s et %s fusionnées (%s) — %s absorbe %s",
                        li, lj, pourquoi, li, lj)
    if all(v == k for k, v in remap.items()):
        return
    for s in sections:
        if s["label"] in remap:
            s["label"] = remap[s["label"]]


def _tierce(accord) -> str | None:
    """« maj », « min » ou None pour UN accord — la note qui décide.

    On lit la qualité DÉCODÉE, qui est le verdict de musx, et non le chroma :
    fa majeur et fa mineur y partagent deux notes sur trois et se ressemblent
    donc à tort. None quand l'accord ne tranche pas (N.C., suspendu) — il ne
    vote alors ni dans un sens ni dans l'autre.
    """
    if not accord or accord.get("nc"):
        return None
    q = str(accord.get("q") or "")
    if q.startswith("sus"):
        return None
    if q.startswith(("-", "o", "h", "°", "ø")) or (
            q.startswith("m") and not q.startswith("maj")):
        return "min"
    return "maj"


def _tierces_par_position(barre) -> dict:
    """{fondamentale: tierce} pour les accords d'une mesure.

    On indexe par FONDAMENTALE et non par rang : deux passages d'une même
    mesure peuvent ne pas découper le temps pareil (un accord tenu ici, deux
    accords là), mais quand ils parlent du même degré ils parlent de la même
    chose. Comparer rang par rang ferait dire n'importe quoi au premier
    passage qui a un accord de plus.
    """
    out = {}
    for c in barre or []:
        t = _tierce(c)
        if t is None:
            continue
        out.setdefault(int(c.get("root", -1)) % 12, t)
    return out


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


#: Confiance minimale, sur la PREMIÈRE passe d'une occurrence empilée, pour
#: qu'un accord qu'elle seule joue mérite d'être montré comme variante « en
#: petit au-dessus », à la iReal (Louis, 2026-09-16, Easy On Me section B :
#: « c'est un F puis la basse descend sur D (donc ça donne un D-7) avant
#: d'atterrir sur le C, sur les passes suivantes des fois la basse fait
#: quelque chose de plus complexe, donc difficile à dire, typiquement le
#: genre de cas où j'aimerais avoir juste F affiché et le D-7 en optionnel en
#: petit au-dessus en suggestion, comme iReal fait »). Mesuré sur ce morceau :
#: à la position qui écrit F, 3 des 6 passes empilées décodent F PUIS D-(7)
#: sur le 4e temps, à 0,70/0,70/0,72 de confiance ; les 3 autres ne tiennent
#: qu'un F. Le seuil est mis sous ce plancher pour garder ces trois occurrences,
#: largement au-dessus du bruit des candidats `sug` jamais écrits (souvent
#: < 0,15 dans ces mêmes mesures). HYPOTHÈSE sur un seul morceau (règle #5,
#: CLAUDE.md) — à vérifier corpus entier avant de resserrer ou d'assouplir.
VAR_MIN_CONF = 0.5


def _bar_variant(members_raw: list[list[dict]], written: list[dict]) -> dict | None:
    """La meilleure lecture minoritaire qu'une pile de passes porte à une
    position, et que le gabarit moyenné (`written`, ce qui va s'écrire sur
    CHAQUE passe empilée à cette position) a fait disparaître.

    `members_raw` : le décodage de première passe de chaque passe empilée à
    cette position, PRIS AVANT que `_write_position` ne les réécrive toutes
    au gabarit — la seule fenêtre où cette information existe encore (elle
    n'est jamais recalculée après coup : une passe rejetée par la pile garde
    son propre décodage, `rejetees`, mais une passe GARDÉE le perd dès que le
    gabarit s'écrit).

    Une occurrence du groupe est une variante quand elle joue, à un temps où
    le gabarit N'ATTAQUE PAS (il tient l'accord précédent), un accord réel
    (confiance ≥ VAR_MIN_CONF) d'une famille différente de ce que le gabarit
    tient là. Les occurrences sont regroupées par temps et par
    `meme_lecture` (grain "fonction" : même fondamentale, même
    `harmonia.roles.family`) — qui absorbe « D- » et « D-7 » dans le même
    groupe, la même lecture à l'oreille dite avec plus ou moins de septième —
    et le groupe le plus soutenu (le plus de passes, puis la plus haute
    confiance) l'emporte ; c'est SA propre meilleure occurrence qui donne la
    couleur affichée, jamais un mélange qu'aucune passe n'a réellement joué.

    NE RÉSOUT PAS (règle #4, CLAUDE.md) : une seule variante par position, à
    l'image du champ `var` du chart (voir `_attach_variant`) — si deux temps
    portaient chacun leur propre variante minoritaire, seul le plus soutenu
    des deux survivrait. Non rencontré sur le corpus mesuré à ce jour.
    Ne regarde que les passes GARDÉES (`gated`, l'appelant) : une passe déjà
    rejetée par la pile (`rejetees`) affiche déjà son propre décodage, elle
    n'a besoin d'aucune annotation en plus.
    """
    written_onsets = sorted((e["beat"], e) for e in written if not e.get("carry"))
    if not written_onsets:
        return None
    written_beats = {wb for wb, _ in written_onsets}

    def _sounding_at(beat):
        cur = None
        for wb, e in written_onsets:
            if wb <= beat:
                cur = e
        return cur

    groups: list[dict] = []          # une lecture par (temps, meme_lecture)
    for bar in members_raw:
        for e in bar:
            if e.get("nc") or e.get("carry"):
                continue
            beat = e.get("beat", 0)
            if beat in written_beats:
                continue                      # le gabarit ATTAQUE déjà ici
            conf = float(e.get("c", 0.0))
            if conf < VAR_MIN_CONF:
                continue
            held = _sounding_at(beat)
            if held is not None and meme_lecture(held, e, grain="fonction"):
                continue                      # même lecture que ce qui est tenu
            slot = next((g for g in groups if g["beat"] == beat
                         and meme_lecture(g["_e"], e, grain="fonction")), None)
            if slot is None:
                slot = {"root": int(e["root"]) % 12, "q": e.get("q", ""),
                        "bass": e.get("bass", -1), "beat": beat, "c": 0.0,
                        "n": 0, "_e": e}
                groups.append(slot)
            slot["n"] += 1
            if conf > slot["c"]:
                slot["c"] = round(conf, 3)
                slot["q"] = e.get("q", "")
                slot["bass"] = e.get("bass", -1)
    if not groups:
        return None
    best = dict(max(groups, key=lambda g: (g["n"], g["c"])))
    best.pop("_e")
    return best


def _attach_variant(chords_k: list[dict], variant: dict) -> None:
    """Pose `variant` (voir `_bar_variant`) sur l'accord de `chords_k` qui
    sonne à son temps — le champ `var` du chart, déjà lu par
    `harmonia/static/screens/chart.js` (posé le 2026-08-17 pour « l'autre
    lecture de cette case, sur un chart replié », jamais alimenté côté
    serveur avant ce jour) : un objet {root,q,bass,c,n}, pas une chaîne déjà
    mise en forme — l'orthographe (dièse/bémol selon le ton) reste la
    responsabilité du rendu (`kit.js::noteEl`/`setSpelling`, 2026-09-16),
    comme pour l'accord principal.
    """
    beat = variant["beat"]
    target = chords_k[0]
    for e in chords_k:
        if e["beat"] <= beat:
            target = e
        else:
            break
    target["var"] = {k: v for k, v in variant.items() if k != "beat"}


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
    from harmonia.sections.similarity import halfbar_features
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
        merge_similar_letters(sections, bars)

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
        #: mesure -> son rang DANS sa section (pas dans la boucle interne) :
        #: deux mesures de même rang sont « la même mesure, jouée deux fois »
        offset_section: dict[int, int] = {}
        for s in secs:
            b0, b1 = s["barRanges"][0]
            for b in range(b0, b1 + 1):
                offset_section[b] = b - b0
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
        # LA QUEUE NE REFUSE PAS LA LETTRE (Louis, 2026-10-12, Evening Star :
        # « il faudrait pouvoir toujours bien gérer les queues »). Quand on
        # empile occurrence contre occurrence (la période EST la section) et
        # que les seules positions incohérentes sont dans la queue que les
        # passages jouent différemment (`tronc_et_queue`, sur leur décodage
        # de 1re passe) — c'est une 1re/2e fin, pas deux musiques. Ces
        # positions sortent de la pile comme les fins de sous-phrase
        # (`fins_hors_pile` : chaque passage garde SA fin, que
        # `_ireal_endings` écrira en 1./2.), et le veto juge le tronc seul.
        # Une incohérence AVANT la queue refuse toujours la lettre entière.
        queue_hors_pile = 0
        incoherentes = [k for k, c in enumerate(coh_by_pos)
                        if c is not None and c < STACK_COHERENCE]
        if incoherentes and all(
                s["barRanges"][0][1] - s["barRanges"][0][0] + 1 == P
                for s in secs):
            tronc = tronc_et_queue([bars[s["barRanges"][0][0]:
                                         s["barRanges"][0][1] + 1] for s in secs])
            if tronc is not None and min(incoherentes) >= tronc:
                queue_hors_pile = P - tronc
                for k in range(tronc, P):
                    fins_hors_pile.extend(pos_members[k])
                    variants.extend(b for b in pos_members[k] if b not in variants)
                    gated[k] = []
                    coh_by_pos[k] = None
                logger.info("fold %s: queue de %d mesure(s) jouée différemment "
                            "— hors pile, le tronc seul juge la cohérence",
                            letter, queue_hors_pile)
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
            # L'ACCORD OPTIONNEL « EN PETIT AU-DESSUS » (Easy On Me, voir
            # `_bar_variant`) : lu sur le décodage de première passe des
            # membres GARDÉS de CETTE position, avant que la ligne du dessous
            # ne les réécrive tous au même gabarit.
            variant = _bar_variant([bars[b] for b in gated[k]], pos_chords[k])
            if variant:
                _attach_variant(pos_chords[k], variant)
            # L'UNANIMITÉ DIRECTE BAT LA MOYENNE INDIRECTE (voir
            # `UNANIMITE_BAT_GABARIT`). On regroupe les mesures gardées par
            # POSITION DANS LA SECTION — c'est ce qu'un lecteur voit comme
            # « la même mesure, jouée N fois » — et si leurs décodages de
            # première passe s'accordent tous sur une tierce que le gabarit
            # contredit, on n'écrit pas : musx ne s'est pas contredit, nous
            # n'avons pas à le faire.
            sans_ecraser = set()
            if UNANIMITE_BAT_GABARIT:
                gab = _tierces_par_position(pos_chords[k])
                par_offset: dict[int, list[int]] = {}
                for b in gated[k]:
                    o = offset_section.get(b)
                    if o is not None:
                        par_offset.setdefault(o, []).append(b)
                for o, membres in par_offset.items():
                    vus = [_tierces_par_position(bars[b]) for b in membres]
                    # une fondamentale que TOUS les passages nomment, et tous
                    # de la même tierce, contre un gabarit qui dit l'inverse
                    for racine, t_gab in gab.items():
                        dits = [v.get(racine) for v in vus]
                        if any(x is None for x in dits) or not dits:
                            continue
                        if len(set(dits)) == 1 and dits[0] != t_gab:
                            sans_ecraser.update(membres)
                            logger.info(
                                "fold %s: position %d, les %d passage(s) sont "
                                "unanimes en %s sur la fondamentale %d contre "
                                "un gabarit en %s — on garde leur décodage "
                                "(mes. %s)", letter, k + 1, len(membres),
                                dits[0], racine, t_gab,
                                [b + 1 for b in sorted(membres)])
                            break
            for b in gated[k]:
                if b in sans_ecraser:
                    continue
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
                          # Les VRAIES rejetées (écart anormal à la pile), à
                          # part des fins de section gardées hors pile
                          # exprès : la façade (voir `facade`) remplace les
                          # premières par le consensus, jamais les secondes.
                          "rejetees": sorted(set(variants) - set(fins_hors_pile)),
                          "changed": sorted(set(changed)),
                          "cv_skip": sorted(cv_skip),
                          "pos_skip": [],
                          **({"queue": queue_hors_pile} if queue_hors_pile else {}),
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
    lab = _musx.redecode(beats, cat, downbeat_times=beats[::bpb],
                            beat_trans_penalty=(15.0, 15.0, 100.0),
                            quarter_beats=_q)
    # λ1 t+b sur le gabarit (2026-10-12) : un accord replié est un accord
    # inféré comme un autre et recevrait son nom par la même règle que le
    # brut (`harmonia.relabel`) ; le gabarit est tuilé ×3, donc la copie du
    # milieu — la seule gardée — a toujours ses deux voisines réelles.
    # ÉTEINT par défaut : voir `SETTINGS.relabel_gabarit` pour ce que ça
    # casse aujourd'hui (le dernier A d'Evening Star).
    if SETTINGS.relabel_delta_voisins and SETTINGS.relabel_gabarit:
        from harmonia.relabel import relabel_segments
        lab, _ = relabel_segments(lab, cat)
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
    # THE CHORD SUSTAINED INTO POSITION 0 (Louis, 2026-09-15, Let It Be:
    # « le premier accord (C) n'est jamais propagé sur la première barre »).
    # The template is tiled ×3 precisely so that the middle copy's first
    # beat has music before it; the chord still sounding when the middle
    # copy starts is the last chord of the loop (F C | → C). Its segment
    # begins in the FIRST copy, so the window test below dropped it, and
    # `cur` started empty: every other position wrote its sustain as a
    # carry at beat 0, position 0 alone got a hole (Let It Be's verse read
    # « · G | Am F | … » and its C was never shown). Keep that chord as the
    # sustain that opens the loop. Never an N.C. (bars.py rule, 2026-08-10).
    wrap = None
    for t0, t1, l in lab:
        if t0 < T0 - edge_tol:
            if t1 > T0 + edge_tol and l != "N" and to_chord(l) is not None:
                wrap = {**to_chord(l), "nc": False,
                        "c": round(_musx.label_confidence(
                            cat[0], T0, min(t1, T1), l), 3)}
            continue
        if t0 >= T1 - edge_tol:
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
    cur = wrap
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
                    # L'accord optionnel « en petit au-dessus » (`_bar_variant`,
                    # Easy On Me 2026-09-16) : posé par l'appelant sur `e` avant
                    # cet appel, jamais recalculé ici.
                    **({"var": e["var"]} if e.get("var") else {}),
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

def pass_evidence(bars, rng) -> int:
    """How many REAL chord onsets a pass carries (carries and N.C. don't
    count) — what makes a pass worth showing as the letter's block.

    ONE law for the two renderers (`minimal_fold` here, and
    `soudure.sections_pour_chart` for hand-validated sections). Audit
    2026-09-15: Stand By Me's hand-sectioned chart still showed its first A
    (the bass-only fade-in, 5 N.C. over 8 bars, ×6) because that second
    path had kept the old `occ[0]` rule — the 2026-08-10 fix below only
    covered the automatic path.
    """
    c0, c1 = rng
    return sum(1 for b in range(c0, c1 + 1)
               for c in (bars[b] if 0 <= b < len(bars) else [])
               if not c["nc"] and not c.get("carry"))


def rejected_in(rep: dict, rng) -> list[int]:
    """The bars of pass `rng` the stack rejected as outliers. Section endings
    kept out of the stack on purpose (`fins_hors_pile`) are NOT outliers and
    stay the pass's own music."""
    c0, c1 = rng
    rej = set(rep.get("rejetees") or [])
    return [b for b in range(c0, c1 + 1) if b in rej]


def pass_rank(bars, rng, rep: dict, ranges) -> tuple:
    """The pass written ×N for a letter: most real onsets, ties → earliest
    (Louis's 2026-08-10 law, Stand By Me). ONE key for both renderers.

    Tried and dropped the same day (2026-09-15): « fewest rejected bars
    first ». It moved Every Breath's A onto a pass whose UNSTACKED positions
    (cv_skip) were decoded worse (« Db Eb » → « Db~ Ab/Eb ») — the rejected
    bars are handled by `facade` anyway, so the ranking need not avoid them.
    `rep` is kept in the signature for a future criterion that would need
    the stack's verdicts."""
    rs = [tuple(r) for r in ranges]
    return (pass_evidence(bars, rng),
            -rs.index(tuple(rng)) if tuple(rng) in rs else 0)


def facade(bars, grid, ranges, rng, rep: dict) -> dict[int, list]:
    """{bar: chords} for the pass shown ×N: a bar the stack REJECTED shows the
    consensus its siblings received — the same loop position in the nearest
    accepted pass, as the template wrote it — retimed onto this bar.

    `bars` itself is untouched: playback, annotation and the other passes
    keep that bar's own decode. Only the WRITTEN block changes, because a
    block that stands for N passes must show what the N passes agree on, not
    what one outlier heard (Louis, 2026-09-15: « la variante ne devrait même
    pas être utilisée »). Positions the stack refused to squash (cv_skip) or
    never wrote have no consensus: the bar stays its own.
    What this does NOT do: show the rejected bar's chord as an alternative.
    That is the « accord optionnel en petit au-dessus » Louis asked for on
    Easy On Me — a display feature still to build.
    """
    out: dict[int, list] = {}
    if rep.get("reason"):
        return out
    rej = set(rep.get("rejetees") or [])
    var = set(rep.get("variants") or [])
    ranges = [tuple(r) for r in ranges]
    c0, c1 = rng
    if len(ranges) < 3:
        # Une passe seule (boucle interne) : ses mesures rejetées sont sa
        # vraie musique — la fin, le turnaround — et s'ÉCRIVENT (under-fold,
        # Norah Jones 2026-08-05). À deux passes, « l'autre » n'est pas un
        # consensus. La façade ne parle qu'à partir de trois.
        return out

    def _meme_mesure(a, b):
        return len(a) == len(b) and all(
            bool(x.get("carry")) == bool(y.get("carry"))
            and meme_lecture(x, y, grain="exact") for x, y in zip(a, b))

    for b in range(c0, c1 + 1):
        if b not in rej:
            continue
        # LA MÊME MESURE DE LA SECTION dans les autres passes (même décalage
        # depuis le début de la passe — pas « même position modulo P », qui
        # irait chercher la pompe deux mesures plus tôt dans la MÊME passe
        # et effacerait une cadence réelle, Sam Smith « F C » → « F A »).
        off = b - c0
        cands = [r0 + off for r0, r1 in ranges
                 if (r0, r1) != (c0, c1) and r0 + off <= r1
                 and (r0 + off) not in var and 0 <= r0 + off < len(bars)
                 and any(c.get("folded") for c in bars[r0 + off])]
        if len(cands) < 2:
            continue
        groupes_: list[tuple] = []        # (mesure-témoin, passes)
        for bb in cands:
            g = next((g for g in groupes_ if _meme_mesure(g[0], bars[bb])), None)
            if g is None:
                groupes_.append((bars[bb], [bb]))
            else:
                g[1].append(bb)
        best = max((g[1] for g in groupes_), key=len)
        if len(best) < 2:
            continue                      # les autres passes ne s'accordent pas
        s = min(best, key=lambda bb: abs(bb - b))
        sw = max(1e-6, grid[s + 1] - grid[s])
        bw = grid[b + 1] - grid[b]
        new = []
        for c in bars[s]:
            f0 = (c.get("t0", grid[s]) - grid[s]) / sw
            f1 = (c.get("t1", grid[s + 1]) - grid[s]) / sw
            new.append({**c, "bar": b, "facade": s,
                        "t0": round(grid[b] + f0 * bw, 3),
                        "t1": round(grid[b] + f1 * bw, 3)})
        out[b] = new
    return out


def facade_view(bars, grid, groups: dict, fold_report: dict | None) -> list:
    """A DISPLAY copy of `bars`: in every letter written ×N (three or more
    passes of the same length), each pass's rejected bars show the consensus
    (see `facade`). The caller's `bars` is untouched — playback, annotation
    and the raw chart keep every bar's own decode.

    One view for everything the renderers read — the written block, the
    iReal endings, the cascade — so a rejected bar cannot come back as a
    « 1. » ending after being corrected in the block (The Lazy Song, first
    render of the façade: block « B F# », ending « 1. B G#m »).
    `groups`: {(label, …): [(b0, b1), …]} — the renderer's own grouping by
    letter and length, whatever its key.
    """
    view = list(bars)
    for key, ranges in groups.items():
        label = key[0] if isinstance(key, tuple) else key
        rep = (fold_report or {}).get(label) or {}
        if rep.get("reason") or len(ranges) < 3:
            continue
        for rng in ranges:
            for b, chords in facade(bars, grid, ranges, tuple(rng), rep).items():
                view[b] = chords
    return view


# ── LA BOUCLE MINIMALE, ÉCRITE UNE FOIS ─────────────────────────────────────
# Louis, 2026-09-30 : (1) sur le D de Julia, « pas la peine d'écrire sur 3
# lignes si on peut l'écrire sur une ligne […] tant pis si ça finit pas
# pile » ; (2) « assure-toi qu'en conséquence les sections soient montrées par
# leur boucle minimale, donc This Love par exemple la section A ne devrait
# faire que 4 barres et non 8 ». Même critère que la règle 2 des briques
# (`sections.simulation.boucle_minimale`) : chaque mesure colle à sa
# partenaire une boucle plus loin, « coller » étant RELATIF au passage.

#: la plus petite boucle cherchée, en mesures
BOUCLE_MIN = 2
#: une boucle plus courte que ça s'écrit doublée : une cellule de 2 mesures
#: tient une rangée de 4 (arbitré sur Julia, « c'est parfait »)
BOUCLE_RANGEE = 4
#: la plus longue : au-delà ce n'est plus une mini-boucle mais la forme du
#: morceau rejouée (la pipeline, sans sections, lisait des « boucles » de
#: 26 à 34 mesures sur des morceaux entiers)
BOUCLE_MAX = 8
#: au-delà de cette part de lectures qui contredisent la cellule écrite, ce
#: n'est pas UNE boucle (Fjufjv4rH0s B et Sunny C : même grille un
#: demi-ton plus haut au 2e tour, 1/3 des cases) — on écrit le passage
CONFLATION_MAX = 0.25


def _muette(bar) -> bool:
    return not bar or all(c.get("nc") for c in bar)


def _suite(colle, L: int, l: int) -> int:
    """Combien de mesures, depuis le début, suivent la boucle de `l`."""
    r = 0
    while r < L - l and colle(r, r + l):
        r += 1
    return l + r


def _etiquette(bar) -> str:
    noms = "C Db D Eb E F Gb G Ab A Bb B".split()
    return " ".join("N.C." if c.get("nc") else noms[c["root"] % 12] + (c.get("q") or "")
                    for c in bar or []) or "%"


def _accords_par_temps(bar) -> tuple | None:
    """L'accord qui sonne à chaque temps d'une mesure. Une mesure vide
    (accord tenu) vaut None, un joker."""
    if not bar:
        return None
    fin = max(4, max(int(c.get("beat") or 0) for c in bar) + 1)
    out = []
    for k in range(fin):
        cur = bar[0]
        for c in bar:
            if (c.get("beat") or 0) <= k:
                cur = c
        out.append(cur)
    return tuple(out)


def meme_harmonie(a, b) -> bool:
    """Deux mesures jouent la même harmonie (couleurs mises à part) : à
    chaque temps, le même accord au sens de `meme_lecture` (grain
    "famille"). Une mesure vide (accord tenu) est un joker.

    C'est la mesure de différence de la factorisation par boucle (Louis,
    2026-09-30 : « ne la fais pas quand il y a vraiment un changement
    important »). Depuis le 2026-10-01, « le même accord » = la mesure de
    ressemblance (`chord_similarity.v2` au seuil du cran) : couleurs, sus
    proche des deux tierces, D-7 ≡ G7sus4 ; une tierce contraire ou une
    autre harmonie reste un vrai changement. Au cran ≥ 2 un décalage d'un
    temps (2 au cran 3) dans la mesure est toléré (`CRANS_V2`)."""
    fa, fb = _accords_par_temps(a), _accords_par_temps(b)
    if fa is None or fb is None:
        return True
    n = max(len(fa), len(fb))
    fa = fa + (fa[-1],) * (n - len(fa))
    fb = fb + (fb[-1],) * (n - len(fb))
    # LE RYTHME HARMONIQUE SUIT LE CURSEUR (Louis, 2026-10-01 : « c'est au
    # cas par cas […] l'utilisateur peut choisir ») : au cran du curseur,
    # chaque temps d'une mesure peut trouver son accord jusqu'à `dec` temps
    # plus tôt ou plus tard dans l'autre (0 au défaut), dans les deux sens.
    from harmonia.chord_similarity import CRANS_V2
    from harmonia.roles import cran_courant
    dec = CRANS_V2[cran_courant()][1]

    def trouve(x, i, autre):
        return any(meme_lecture(x, autre[j], grain="famille")
                   for j in range(max(0, i - dec), min(n, i + dec + 1)))
    return all(trouve(fa[i], i, fb) for i in range(n)) and \
        all(trouve(fb[i], i, fa) for i in range(n))


def boucle_minimale_ecrite(bars, vue, grid, occ, rep, S=None,
                           fusion: int | None = None) -> dict | None:
    """Le bloc écrit d'une lettre dont TOUS les passages ne sont qu'une
    boucle plus courte jouée ≥ 2 fois — ou None (on écrit alors comme avant).

    Rend {"bars", "barSpans", "boucle"} ; `boucle` = {cellule, rangee, tours,
    tours_pleins, conflations}.

    LE CRITÈRE (règle 2, `simulation.boucle_minimale`) : pour une boucle de
    `l` mesures (la plus petite ≥ `BOUCLE_MIN` qui marche pour chaque
    passage), chaque mesure colle à celle `l` mesures plus loin depuis le
    DÉBUT du passage, sur une boucle puis au moins la moitié de la suivante, et ce qui reste après ne
    dépasse pas une boucle — la dernière fois peut finir autrement ou
    s'arrêter en route. « Coller » : avec la matrice chord-tone `S` (quand
    l'audio est là), ≥ (1 + médiane des ressemblances internes du passage)/2,
    relatif au passage comme pour les briques ; sans `S`, même accord à la
    famille près (`meme_harmonie`). Un fondu muet en fin de passage est
    ignoré (Julia : « c'est une section qui se finit plus tôt »). Un passage
    qui ne tient qu'un accord n'est pas une boucle (phase indécidable, rien
    à gagner) : on s'abstient.

    L'ÉCRITURE : la cellule (doublée si elle fait moins de `BOUCLE_RANGEE`),
    prise dans la façade `vue` du passage représentatif `rep`. La carte de
    lecture allume la rangée r à chaque tour, dernier tour incomplet et fondu
    compris.

    L'HONNÊTETÉ : là où un tour joue autre chose que ce qui est écrit (une
    cadence de dernier tour, A7 contre A7sus4…), lu sur le décodage BRUT
    `bars` et pas sur la façade, la lecture minoritaire la plus jouée est
    posée en `var` (« l'autre lecture de cette case », déjà rendue par
    chart.js) et listée dans `boucle.conflations`. CE QUE ÇA NE RÉSOUT PAS :
    une seule variante par case (comme `var` partout) ; une vraie fin de
    dernier tour n'est pas un crochet 1./2. mais une petite lecture en
    exposant — les crochets indexent des passages, pas des tours ; l'heure
    de chaque ACCORD dans les tours suivants n'est pas recalculée (la case
    allumée, elle, suit la carte `barSpans`).

    LE CURSEUR « Section merging » (`fusion`, Louis, 2026-09-30 : « mets-moi
    le curseur qui permet de choisir à quel point tu factorises fort ou pas
    par la plus petite boucle ») — le même que celui des fins, une seule
    question : à quel point deux passages peuvent différer et s'écrire une
    fois. 0 : jamais de boucle. 1 (défaut) : boucle seulement si tous les
    tours jouent la même harmonie (`meme_harmonie` : couleurs permises) —
    sinon le passage s'écrit en entier (This Love : les deuxièmes moitiés
    sont vraiment autres). 2 / 3 : un tour qui diffère est toléré si ses
    notes ressemblent à ≥ 0,90 / 0,70 à la case écrite (`_FUSION_SEUIL`), et
    montré en `var`. 4 : toujours, les différences en `var`.
    """
    import numpy as np
    niveau = FUSION_DEFAUT if fusion is None else fusion
    if niveau <= 0:
        return None
    efficaces = []
    for a, b in occ:
        fin = b
        while fin >= a and _muette(vue[fin]):
            fin -= 1
        efficaces.append((a, fin - a + 1))
    Lmin = min(L for _a, L in efficaces)
    if Lmin < 2 * BOUCLE_MIN:
        return None
    ra = rep[0]
    reels = [vue[x] for x in range(ra, rep[1] + 1) if not _muette(vue[x])]
    if not reels or all(meme_harmonie(reels[0], x) for x in reels[1:]):
        return None                                   # un accord tenu

    def colle_de(a, L):
        if S is not None:
            vals = [S[a + i, a + j] for i in range(L) for j in range(i + 1, L)]
            tau = (1.0 + float(np.median(vals))) / 2.0 if vals else 1.0
            return lambda i, j: S[a + i, a + j] >= tau
        return lambda i, j: meme_harmonie(vue[a + i], vue[a + j])

    colles = [(L, colle_de(a, L)) for a, L in efficaces]
    cellule = None
    for l in range(BOUCLE_MIN, min(BOUCLE_MAX, Lmin // 2) + 1):
        if any(_muette(vue[ra + i]) for i in range(l)):
            continue
        # au moins une boucle entière PUIS la moitié de la suivante qui colle,
        # et au plus une demi-boucle qui finit autrement (This Love A : la 8e
        # mesure cadence). Un dernier tour ENTIER différent est une autre
        # phrase, pas une autre fin (San Francisco Street A : 4 mesures de
        # turnaround après 5 tours) — on sous-replie.
        if all(_suite(c, L, l) >= l + -(-l // 2)
               and L - _suite(c, L, l) <= max(1, l // 2)
               for L, c in colles):
            cellule = l
            break
    if cellule is None:
        return None
    W = cellule * -(-BOUCLE_RANGEE // cellule) if cellule < BOUCLE_RANGEE else cellule
    if any(L // W < 2 for _a, L in efficaces) or ra + W > len(vue):
        return None
    ecrit = [[dict(c) for c in vue[ra + r]] for r in range(W)]
    carte: list[list[list[float]]] = [[] for _ in range(W)]
    lectures_: list[list[int]] = [[] for _ in range(W)]
    for a, b in occ:
        for x in range(a, b + 1):
            carte[(x - a) % W].append([grid[x], grid[min(len(grid) - 1, x + 1)]])
            if not _muette(bars[x]):
                lectures_[(x - a) % W].append(x)
    conflations = []
    n_lus = n_contre = 0
    for r in range(W):
        # la lecture MAJORITAIRE des tours s'écrit (The Walk B : la passe
        # représentative disait Db-7 là où 6 tours sur 9 jouent A7)
        groupes: list[tuple] = []
        for x in lectures_[r]:
            g = next((g for g in groupes if meme_harmonie(g[0], bars[x])), None)
            if g is None:
                groupes.append((bars[x], [x]))
            else:
                g[1].append(x)
        if not groupes:
            continue
        n_lus += len(lectures_[r])
        maj = max(groupes, key=lambda g: (len(g[1]), -g[1][0]))
        if not ecrit[r] or not any(meme_harmonie(g[0], ecrit[r]) and len(g[1]) == len(maj[1])
                                   for g in groupes):
            ecrit[r] = [dict(c) for c in vue[maj[1][0]]]
        autres = [g for g in groupes if not meme_harmonie(g[0], ecrit[r])]
        if not autres:
            continue
        # LE CURSEUR : au défaut, un tour qui change VRAIMENT d'harmonie
        # interdit la boucle — le passage s'écrit en entier, jamais caché en
        # petite suggestion ; aux crans 2/3, seulement s'il ressemble assez
        seuil = _FUSION_SEUIL.get(niveau)
        if niveau <= FUSION_DEFAUT or (
                seuil is not None and any(_ressemblance_queues([ecrit[r]], [bars[g[1][0]]]) < seuil
                                          for g in autres)):
            return None
        n_contre += sum(len(g[1]) for g in autres)
        _k, xs = max(autres, key=lambda g: (len(g[1]), -g[1][0]))
        minor = bars[xs[0]]
        lus = [c for c in ecrit[r] if not c.get("nc")]
        cand = next((c for c in minor if not c.get("nc")
                     and not any(meme_lecture(c, w, grain="famille") for w in lus)),
                    next((c for c in minor if not c.get("nc")), None))
        if cand is not None and ecrit[r]:
            beat = cand.get("beat", 0)
            cible = ecrit[r][0]
            for c in ecrit[r]:
                if c.get("beat", 0) <= beat:
                    cible = c
            cible["var"] = {"root": int(cand["root"]) % 12, "q": cand.get("q", ""),
                            "bass": cand.get("bass", -1),
                            "c": cand.get("c", 0.0), "n": len(xs)}
        conflations.append({"rang": r, "ecrit": _etiquette(ecrit[r]),
                            "autre": _etiquette(minor), "mesures": xs,
                            "sur": len(lectures_[r])})
    if n_lus and n_contre / n_lus > CONFLATION_MAX:
        return None
    Lrep = rep[1] - rep[0] + 1
    return {"bars": ecrit, "barSpans": carte,
            "boucle": {"cellule": cellule, "rangee": W,
                       "tours": round(Lrep / W, 2),
                       "tours_pleins": min(L // W for _a, L in efficaces),
                       # un par passage (même ordre que barRanges) : la
                       # marque ×N du chart dit ce que CHAQUE passage joue
                       "tours_par_passe": [L // W for _a, L in efficaces],
                       "conflations": conflations}}


def minimal_fold(sections, bars, grid, fold_report, S=None) -> list[dict]:
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
    # LA FAÇADE (Louis, 2026-09-15) : tout ce qui suit lit une VUE de `bars`
    # où, dans une lettre écrite ×N, les mesures rejetées par la pile
    # montrent le consensus — le bloc, les fins iReal et la cascade, d'un
    # seul tenant. La liste de l'appelant n'est pas touchée.
    # `bars` (brut) sert au CLASSEMENT des passes (`_evidence`) ; `vue` à
    # tout ce qui s'écrit. Classer sur la vue faisait gagner la passe qui
    # venait d'être corrigée (elle gagne des attaques), et changeait de
    # bloc affiché sans raison musicale (Cry Me A River, Uw5OLnN7UvM).
    vue = facade_view(bars, grid, by_letter, fold_report)
    out = []
    _vus: dict[str, int] = {}

    def _evidence(rng):
        return pass_evidence(bars, rng)

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
        _rep = fold_report.get(L) or {}
        b0, b1 = max(ranges, key=lambda r: pass_rank(bars, r, _rep, ranges))
        # LA BOUCLE MINIMALE (Louis, 2026-09-30, « This Love A ne devrait
        # faire que 4 barres ») : même écriture que le chemin des sections
        # à la main (`soudure.sections_pour_chart`).
        _bcl = boucle_minimale_ecrite(bars, vue, grid, ranges, (b0, b1), S)
        if _bcl:
            _vus[L] = _vus.get(L, 0) + 1
            out.append({
                "id": f"L{L}" + ("" if _vus[L] == 1 else str(_vus[L])),
                "label": L, "tag": "", "reps": len(ranges),
                "spans": [[grid[c0], grid[min(len(grid) - 1, c1 + 1)]]
                          for c0, c1 in ranges],
                "barRanges": [[c0, c1] for c0, c1 in ranges],
                **_bcl,
            })
            continue
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
        cell = [vue[b] for b in range(b0, min(b1, b0 + P - 1) + 1)] if P \
            else [vue[b] for b in range(b0, b1 + 1)]
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
            "bars": [vue[b] for b in block_rng],
            "barSpans": rows,
        })
    # ── LA FAÇON IREAL : une lettre = un seul bloc écrit ────────────────────
    # (Louis, 2026-09-14 : « tu ne peux pas afficher un A deux fois »).
    # Spec complète : docs/spec_affichage_sections.md.
    # Les fins iReal et la cascade lisent le BRUT, comme avant la façade
    # (Louis, 2026-09-15, sur la page avant/après : « j'ai l'impression que
    # ça a empiré tous les morceaux » — lues sur la vue, les fins 1./2.
    # apparaissaient sur Cry Me A River (5 groupes / 8 passes), Hot N Cold,
    # Uw5OLnN7UvM : les passes s'accordant enfin sur leur tronc, le
    # mécanisme des fins exposait le bruit des dernières mesures gardées
    # hors pile). Ce que ça laisse : une mesure rejetée peut encore
    # remonter dans un crochet (Lazy Song « 1. B G#m ») — question à part,
    # avec la loi candidate « pas de crochets quand les queues se
    # dispersent », dans known_issues.
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

# ── LE CURSEUR « FUSION DES SECTIONS » (Louis, 2026-09-30) ──────────────────
# « Il faudrait un curseur sur lequel on peut jouer qui force à quel point tu
# merges les sections ou tu les laisses avoir des bouts alternatifs. »
#
# Il ne pilote QUE la décision « ces passes d'une même lettre ont-elles des
# fins différentes ? » (`_ireal_endings`) — c'est la seule qui soit un choix
# d'ÉCRITURE. Les portes du repli (cohérence de pile, z des mesures rejetées)
# décident quels ACCORDS on écrit, pas combien de blocs : les brancher sur un
# réglage d'affichage ferait changer des accords en bougeant un curseur, et
# demanderait de relancer l'empilement (musx) à chaque cran.
#
#   0  garder toutes les fins : une simple couleur (C7 / C-7) fait une fin
#   1  le défaut, le comportement d'avant le curseur (une vraie fin change
#      de fondamentale sonnée)
#   2  fusionner les fins PROCHES : une queue dont les notes ressemblent à
#      ≥ 0,90 à la queue la plus jouée la rejoint
#   3  fusionner les fins APPARENTÉES : même chose à ≥ 0,70
#   4  tout fusionner : un bloc ×N, écrit avec la fin la plus jouée
#
# « Ressembler » = le cosinus des notes sonnées (basse comprise, pondérées par
# la durée), mesure par mesure, moyenné sur la queue — la ressemblance
# chord-tone du projet (Bb plus près de G- que de F). Mesuré sur la
# bibliothèque le 2026-09-30 (11 blocs à fins) : Claustrophobic C9 F7 / Bb7
# F^7 contre C7sus4 F^7 à 0,94-0,96, Bb-7 à la place d'Eb7 à 0,71 ; This
# Love ≥ 0,81 ; My Cherie Amour Ab7 / A7 à 0,65 ; She Will Be Loved Ab / Bb
# à 0,00. Les seuils sont posés entre ces valeurs, pas au milieu d'un nuage.
FUSION_NIVEAUX = (0, 1, 2, 3, 4)
FUSION_DEFAUT = 1
_FUSION_SEUIL = {2: 0.90, 3: 0.70}


def _vec_notes(bar) -> list[float]:
    v = [0.0] * 12
    from harmonia.roles import pcs
    for c in bar or []:
        if c.get("nc"):
            continue
        try:
            d = float(c["t1"]) - float(c["t0"])
        except (KeyError, TypeError, ValueError):
            d = 1.0
        for p in pcs(c):
            v[p] += max(d, 1e-3)
    return v


def _ressemblance_queues(qa: list, qb: list) -> float:
    """Cosinus des notes sonnées, mesure par mesure, moyenné sur la queue."""
    def cos(a, b):
        na = sum(x * x for x in a) ** .5
        nb = sum(x * x for x in b) ** .5
        if not na or not nb:
            return 1.0 if na == nb else 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)
    if not qa:
        return 1.0
    return sum(cos(_vec_notes(a), _vec_notes(b))
               for a, b in zip(qa, qb)) / len(qa)


def cle_sections(sections: list) -> str:
    """L'empreinte d'un découpage (lettres + passes) : les variantes de fusion
    stockées dans un chart ne valent que pour le découpage qui les a vues
    naître. Une route qui réécrit `sections` sans les refaire les périme."""
    import hashlib
    brut = json.dumps([[s.get("label"), s.get("barRanges")]
                       for s in sections or []], sort_keys=True)
    return hashlib.sha1(brut.encode()).hexdigest()[:16]


def sections_par_fusion(construire) -> tuple[list, dict]:
    """(sections au défaut, variantes) — `construire(niveau)` rend les
    sections du chart à ce niveau de fusion.

    Les variantes vont dans `chart["sectionsFusion"]` : un niveau qui rend
    exactement le défaut n'y est pas ; deux niveaux qui rendent la même chose
    partagent une entrée (la seconde vaut le NOM de la première). `cle` =
    l'empreinte du découpage (`cle_sections`)."""
    defaut = construire(FUSION_DEFAUT)
    alt: dict = {}
    vus = [(FUSION_DEFAUT, defaut)]
    for n in FUSION_NIVEAUX:
        if n == FUSION_DEFAUT:
            continue
        s = construire(n)
        pareil = next((m for m, t in vus if t == s), None)
        if pareil == FUSION_DEFAUT:
            continue
        if pareil is not None:
            alt[str(n)] = str(pareil)
        else:
            alt[str(n)] = s
            vus.append((n, s))
    if alt:
        alt["cle"] = cle_sections(defaut)
    return defaut, alt


def appliquer_fusion(model: dict, niveau) -> dict:
    """Le chart tel que servi au client, au niveau de fusion choisi par Louis
    pour ce morceau (`chart_meta.json`, clé `fusion`). Retire les variantes
    (elles ne voyagent pas jusqu'au téléphone) ; pose `fusion` (le niveau
    effectif) et `fusionDispo` (le curseur change-t-il quelque chose ici ?).
    Variantes périmées (découpage réécrit depuis) → le défaut, sans bruit."""
    alt = model.pop("sectionsFusion", None) or {}
    try:
        niveau = int(niveau)
    except (TypeError, ValueError):
        niveau = FUSION_DEFAUT
    if niveau not in FUSION_NIVEAUX:
        niveau = FUSION_DEFAUT
    frais = bool(alt) and alt.get("cle") == cle_sections(model.get("sections"))
    model["fusionDispo"] = frais
    if frais and niveau != FUSION_DEFAUT:
        s = alt.get(str(niveau))
        if isinstance(s, str):
            s = alt.get(s)
        if isinstance(s, list):
            model["sections"] = s
    model["fusion"] = niveau
    return model


def _barsig(bar) -> tuple:
    """Ce qu'une mesure JOUE, accord par accord — la signature que comparent
    les fins 1./2. et la cascade. Se compare avec `_compat`, qui passe par
    `meme_lecture` à la granularité qu'on lui demande."""
    return tuple(bar or [])


def _compat(a, b, grain: str = "exact") -> bool:
    """Deux signatures (mesure, ou suite de mesures) se valent : même nombre
    d'accords, et chaque paire est le même accord (`meme_lecture`, au
    `grain` donné).

    Deux usages, deux granularités, et c'est voulu : "exact" cherche une
    DIFFÉRENCE (les queues divergentes d'une 1re/2e fin), "famille" cherche
    une RESSEMBLANCE (un passage coupé est-il le même moment musical ? — la
    septième entendue une fois sur deux est du bruit de décodage : sur
    Fallin', dont toutes les lettres sont la boucle `E-|B-`, un seul `B-7`
    au lieu d'un `B-` suffisait à déclarer deux passages autres). "racine" :
    une fin qui change de fondamentale quelque part."""
    if len(a) != len(b):
        return False
    if a and isinstance(a[0], dict):
        return all(meme_lecture(xa, xb, grain=grain) for xa, xb in zip(a, b))
    return all(_compat(xa, xb, grain) for xa, xb in zip(a, b))


def fenetre_de_queue(L: int) -> int:
    """Combien de mesures de fin une 1re/2e fin peut occuper dans une section
    de `L` mesures, quand le tronc n'est pas identique à la note près : le
    DERNIER QUART, et jamais plus de `_ENDING_TAIL_MAX` (2).

    Pourquoi un quart (Louis, 2026-10-12, Evening Star) : une section se
    bâtit en phrases de 4 mesures ; la fin alternative est la CADENCE de la
    dernière phrase — sa 2e moitié, soit les 2 dernières mesures d'un A de 8,
    la dernière d'une cellule de 4 (le « tour » qui relance ou qui sort). Au-delà
    d'un quart, c'est la phrase elle-même qui change : une autre section, pas
    une autre fin. Le plafond 2 est celui de la règle exacte d'avant
    (`_ENDING_TAIL_MAX`), pour qu'une section de 16 ne reçoive pas des fins de
    4 mesures par cette porte-ci."""
    return min(_ENDING_TAIL_MAX, max(0, L // 4))


def tronc_et_queue(passes: list) -> int | None:
    """La longueur du TRONC commun de passages de même longueur `L` qui ne
    diffèrent que par leur queue — ou None s'il n'y a pas de 1re/2e fin à
    écrire. La queue = les `L - tronc` dernières mesures.

    Deux portes, la 2e n'ajoutant que des cas que la 1re refusait :
      1. la règle exacte d'avant (2026-07-21) : tronc identique accord pour
         accord (`_compat` exact), queue de 1 à `_ENDING_TAIL_MAX` mesures,
         tronc ≥ 2 ;
      2. LE TRONC À LA COULEUR PRÈS (Louis, 2026-10-12 : « il faudrait
         pouvoir toujours bien gérer les queues ») : toutes les mesures avant
         le dernier quart (`fenetre_de_queue`) sont le même moment
         (`meme_moment` : famille, accord de passage en plus ou en moins) ;
         la queue commence à la PREMIÈRE mesure de ce dernier quart où les
         passages ne s'écrivent pas pareil, et il faut qu'une vraie
         différence (pas le même moment) s'y trouve. Evening Star : `C7`
         contre `C/G C7` (mes. 2) et `F C` contre `F` (mes. 7) sont des
         couleurs, `G-7 C7` contre `F B♭` (mes. 8) une autre harmonie →
         tronc 6, queue `F C | G-7 C7` contre `F | F B♭`.

    CE QUE ÇA NE FAIT PAS : regrouper les queues (`_ireal_endings` le fait, à
    l'égalité exacte) ; une différence au milieu (phase ≠ fin) refuse
    toujours ; dans le tronc, c'est le passage écrit qui donne la couleur.
    """
    r = _porte_de_queue(passes)
    return None if r is None else r[0]


def _porte_de_queue(passes: list) -> tuple[int, list | None] | None:
    """(tronc, groupes) — voir `tronc_et_queue`. `groupes` vaut None par la
    porte exacte (`_ireal_endings` groupe alors les queues à l'égalité
    exacte, comme avant) ; par la porte « à la couleur près », c'est la liste
    des groupes de passes qui jouent la MÊME fin au même moment près
    (`meme_moment` mesure par mesure) — deux couleurs d'une même fin ne font
    pas deux fins. Under-fold : si une passe se range dans deux groupes à la
    fois (ressemblance non transitive), ou s'il ne reste qu'un groupe, None."""
    passes = [list(p) for p in passes or []]
    if len(passes) < 2:
        return None
    L = len(passes[0])
    if any(len(p) != L for p in passes) or L < 3:
        return None

    def tous(r, egal):
        return all(egal(pa[r], pb[r]) for i, pa in enumerate(passes)
                   for pb in passes[i + 1:])

    def exact(a, b):
        return _compat(_barsig(a), _barsig(b))
    # 1. la règle exacte
    prefix = next((r for r in range(L) if not tous(r, exact)), L)
    if 1 <= L - prefix <= _ENDING_TAIL_MAX and prefix >= 2:
        return prefix, None
    # 2. le tronc à la couleur près, la queue dans le dernier quart
    W = fenetre_de_queue(L)
    if W < 1 or L - W < 2:
        return None
    if not all(tous(r, meme_moment) for r in range(L - W)):
        return None
    fen = range(L - W, L)
    if all(tous(r, meme_moment) for r in fen):
        return None                      # que des couleurs : pas de 2e fin
    prefix = next(r for r in fen if not tous(r, exact))

    def meme_fin(pa, pb):
        return all(meme_moment(pa[r], pb[r]) for r in range(prefix, L))
    groupes: list[list[int]] = []
    for pi, p in enumerate(passes):
        oui = [g for g in groupes if meme_fin(passes[g[0]], p)]
        if len(oui) > 1 or (oui and not all(meme_fin(passes[q], p) for q in oui[0])):
            return None                  # ambigu : on n'écrit pas de fins
        if oui:
            oui[0].append(pi)
        else:
            groupes.append([pi])
    if len(groupes) < 2:
        return None
    return prefix, groupes


def _ireal_endings(blk: dict, bars, grid, fusion: int = FUSION_DEFAUT) -> None:
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

    `fusion` : le cran du curseur « fusion des sections » (FUSION_NIVEAUX).
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
    # le tronc et la queue : voir `tronc_et_queue` (Evening Star, 2026-10-12 :
    # deux couleurs dans le tronc ne cachent plus la vraie queue)
    porte = _porte_de_queue([[bars[c0 + r] for r in range(L)]
                             for c0, _ in ranges])
    if porte is None:
        return
    prefix_len, par_moment = porte
    tail = L - prefix_len
    # les groupes de queues, dans l'ordre de leur première passe ; une passe
    # rejoint le premier groupe dont la passe-témoin se lit pareil — au même
    # moment près quand le tronc n'était pas identique (`_porte_de_queue`)
    groupes: list[dict] = []
    if par_moment is not None:
        groupes = [{"sig": tuple(sigs[g[0]][prefix_len:]), "passes": list(g),
                    "rep": g[0]} for g in par_moment]
    for pi, s in enumerate(sigs if par_moment is None else []):
        t = tuple(s[prefix_len:])
        g = next((g for g in groupes if _compat(g["sig"], t)), None)
        if g is None:
            groupes.append({"sig": t, "passes": [pi], "rep": pi})
        else:
            g["passes"].append(pi)
    if len(groupes) < 2:
        return

    # LE CURSEUR (voir FUSION_NIVEAUX). Au défaut, ce bloc ne fait rien.
    def queue(pi):
        return [bars[ranges[pi][0] + r] for r in range(prefix_len, L)]
    if fusion > FUSION_DEFAUT:
        # l'hôte = la queue la plus jouée (à égalité, la première entendue)
        hote = max(groupes, key=lambda g: (len(g["passes"]), -g["passes"][0]))
        q_hote = queue(hote["rep"])
        seuil = _FUSION_SEUIL.get(fusion)       # None au cran max : tout
        for g in [g for g in groupes if g is not hote]:
            if seuil is None or \
                    _ressemblance_queues(queue(g["rep"]), q_hote) >= seuil:
                hote["passes"] = sorted(hote["passes"] + g["passes"])
                groupes.remove(g)
        if len(groupes) < 2:
            # toutes les fins ont rejoint l'hôte : un bloc ×N, écrit avec
            # la fin la plus jouée ; la carte de lecture (une rangée par
            # mesure, toutes les passes) reste celle du bloc entier.
            blk["bars"] = list(blk["bars"][:prefix_len]) + q_hote
            logger.info("ireal: %s ×%d, fins fusionnées (curseur %d)",
                        blk["label"], len(ranges), fusion)
            return
    # UNE VRAIE 2e FIN CHANGE D'ACCORD, PAS DE COULEUR (2026-09-14). Sur Sunny
    # Afternoon la queue des trois refrains ne diffère que par `C7` contre
    # `Cm7` sur la dernière mesure — et cette mesure-là est justement celle que
    # le repli met hors de la pile (la cadence de fin de section), donc son
    # décodage est du premier jet, passage par passage. Suspendre un crochet
    # « 1./2. » pour ça, c'est sur-noter du bruit. On exige donc que la queue
    # change de FONDAMENTALE quelque part (This Love : `B♭ E♭` contre `A♭ G`).
    # Au cran 0 (« garder toutes les fins »), une couleur suffit.
    sg = [g["sig"] for g in groupes]
    meme_fond = all(_compat(ra, rb, "racine") for i, ra in enumerate(sg)
                    for rb in sg[i + 1:])
    if meme_fond and fusion != 0:
        return
    variants = []
    rows = [[[grid[c0 + r], grid[min(len(grid) - 1, c0 + r + 1)]]
             for c0, _ in ranges] for r in range(prefix_len)]
    for g in sorted(groupes, key=lambda g: g["passes"][0]):
        passes = g["passes"]
        v0 = ranges[g["rep"]][0]
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


def _ireal_cascade(out: list[dict], bars, grid, fold_report,
                   tonique: tuple[int, str] | None = None) -> list[dict]:
    """Règles 3 et 5 + l'invariant : jamais deux blocs sous la même étiquette.

    Règle 3 — un passage COUPÉ (il s'arrête en cours de boucle) rejoint le
    bloc long : même lettre, `reps` augmenté, et il n'allume que les mesures
    qu'il joue réellement. C'est le cas de Sunny Afternoon, dont le « A de 6
    mesures » est la cellule de 4 jouée une fois et demie.

    Règle 2 de Louis du 2026-10-01 — la dernière section du morceau qui joue
    le début de la lettre puis tient la tonique et s'éteint en est le dernier
    tour (`_merge_dernier_tour`) ; `tonique` = (hauteur, mode) du morceau.

    Règle 5 — tout ce que la cascade n'a pas réduit prend un prime (A′, A″).
    Un deuxième « A » nu ne sort jamais d'ici.
    """
    by_label: dict[str, list[dict]] = {}
    for blk in out:
        by_label.setdefault(blk["label"], []).append(blk)
    drop: set[int] = set()
    fin_morceau = max((r[1] for b in out for r in b.get("barRanges") or []),
                      default=-1)
    for label, blks in by_label.items():
        if len(blks) < 2:
            continue
        # l'hôte = le plus joué (à égalité, le premier dans le morceau) —
        # JAMAIS une plage muette (Louis, 2026-09-30, Lost Without U : la
        # mesure 24, un N.C. seul étiqueté B, était l'hôte de B à égalité de
        # passages ; le vrai B de 4 mesures ne pouvait pas s'y fondre et
        # sortait B′ — « y'a juste un A et un B »).
        sonores = [b for b in blks if not _bloc_muet(b)] or blks
        host = max(sonores, key=lambda b: (b["reps"], -out.index(b)))
        rep = (fold_report or {}).get(label) or {}
        # un repli REFUSÉ ne sert pas de cellule (sa `period` est celle qu'il
        # testait, pas celle qu'il a validée) — on passe alors au test préfixe
        P = None if rep.get("reason") else rep.get("period")
        # grain "famille", pas "exact" : `_merge_coupe` cherche une
        # RESSEMBLANCE (un passage coupé est-il le MÊME moment musical ?),
        # jamais une différence — voir la docstring de `_compat` (le cas
        # Fallin', un seul `B-7` au lieu de `B-` cassait le raccord).
        # `_ireal_endings`, juste au-dessus, cherche l'inverse (une VRAIE
        # fin change d'accord) et compare au grain "exact".
        cell = list(host["bars"][:P]) if P else None
        rank = 0
        for blk in blks:
            if blk is host:
                continue
            if _merge_muet(host, blk) or _merge_boucle(host, blk, grid) or \
                    _merge_coupe(host, blk, cell, P, grid) or \
                    _merge_dernier_tour(host, blk, bars, grid, fin_morceau,
                                        tonique):
                drop.add(id(blk))
                continue
            rank += 1
            blk["label"] = label + (_PRIMES[rank - 1] if rank <= len(_PRIMES)
                                    else str(rank))
            logger.info("ireal: %s devient %s (même lettre, autre musique)",
                        label, blk["label"])
    # AU CRAN HAUT, UNE FIN D'UNE AUTRE LETTRE (Louis, 2026-10-01, Lost
    # Without U : son « outro » est le dernier tour du A — « au cas par cas,
    # l'utilisateur peut choisir ») : la dernière section, seule de sa
    # lettre, qui est le dernier tour d'une autre lettre la rejoint. Jamais
    # au défaut : l'étiquette est celle que Louis a posée.
    from harmonia.roles import cran_courant
    if cran_courant() >= CRAN_DERNIER_TOUR_AUTRE_LETTRE:
        fin = next((b for b in out if id(b) not in drop
                    and any(r[1] == fin_morceau for r in b.get("barRanges") or [])), None)
        if fin is not None and len(by_label.get(fin["label"], [])) == 1:
            hotes = sorted((b for b in out if id(b) not in drop and b is not fin
                            and b["label"] != fin["label"] and not _bloc_muet(b)),
                           key=lambda b: (-b["reps"], out.index(b)))
            for host in hotes:
                if _merge_dernier_tour(host, fin, bars, grid, fin_morceau, tonique):
                    logger.info("ireal: %s rejoint %s (dernier tour, cran %d)",
                                fin["label"], host["label"], cran_courant())
                    drop.add(id(fin))
                    break
    return [b for b in out if id(b) not in drop]


def _bloc_muet(blk: dict) -> bool:
    """Un bloc dont toutes les mesures sont N.C. — une plage muette."""
    return all(_muette(b) for b in blk.get("bars") or [[]])


def _accord_de_passage(a, b) -> bool:
    """Deux mesures sont LE MÊME MOMENT à un accord de passage près : même
    premier accord (à la famille près), et chaque accord de la mesure qui en
    a le moins se retrouve dans l'autre. `F^7` contre `F^7 G-7` (Lost
    Without U, mesure 36 contre la boucle A : la relance G-7 que ce passage
    ne joue pas). Une autre fondamentale sur le temps fort n'en est jamais
    un."""
    fa = [c for c in a or [] if not c.get("nc")]
    fb = [c for c in b or [] if not c.get("nc")]
    if not fa or not fb or len(fa) == len(fb) \
            or not meme_lecture(fa[0], fb[0], grain="famille"):
        return False
    court, long_ = (fa, fb) if len(fa) < len(fb) else (fb, fa)
    return all(any(meme_lecture(x, y, grain="famille") for y in long_)
               for x in court)


def meme_moment(a, b) -> bool:
    """Le test de RACCORD d'un passage à la cellule de sa lettre (règle 3 et
    `_merge_boucle`) : même harmonie (`meme_harmonie`, `_compat` à la
    famille près) ou même mesure à un accord de passage près."""
    return (meme_harmonie(a, b)
            or _compat(_barsig(a), _barsig(b), "famille")
            or _accord_de_passage(a, b))


def dernier_tour_partiel(fin, ref, tonique: tuple[int, str] | None = None
                         ) -> dict | None:
    """La section finale `fin` est-elle le DERNIER TOUR, partiel, du passage
    `ref` ? Rend {ouverture, premiere_toleree, tenue, fondu, tonique_aux}
    ou None.

    Louis, 2026-10-01 (Lost Without U, mes. 83-89) : « quand une section est
    la dernière du morceau, qu'elle joue le début du A puis tient la tonique
    et s'éteint, la traiter comme le dernier tour du A ».

    LA RÈGLE, mesure par mesure (`meme_moment`, donc `roles.meme_lecture`) :
      1. le fondu muet final (N.C.) est mis de côté ;
      2. l'OUVERTURE : les premières mesures collent à celles de `ref`, en
         phase 0. La 1re peut ne pas coller (Lost Without U joue G7sus4 là
         où le A joue D-7 : la même cadence sur la pédale de dominante, que
         la mesure de similarité apprise saura lire ; d'ici là on la
         tolère) — mais il faut alors au moins `DERNIER_TOUR_MIN_COLLE`
         mesures qui collent APRÈS elle ;
      3. la TENUE : ce qui suit l'ouverture commence par l'accord sur lequel
         l'ouverture est arrivée, et cet accord est la tonique (`tonique` =
         (hauteur, "major"|"minor") ; sans tonalité connue, un accord majeur
         ou mineur, jamais une dominante). Elle ne dure pas plus longtemps
         que l'ouverture ;
      4. le tout est plus court que `ref` (sinon c'est un passage entier),
         et il y a une tenue ou un fondu (sinon c'est un passage coupé, que
         `_merge_coupe` sait déjà raccorder).

    CE QUE ÇA NE RÉSOUT PAS : une section finale qui joue le MILIEU du A
    (phase ≠ 0) ; une tenue sur un autre degré que la tonique (une fin sur
    le IV) ; une fin plus longue que son ouverture ; un tag répété trois fois.
    """
    fin, ref = list(fin or []), list(ref or [])
    T = list(fin)
    fondu = 0
    while T and _muette(T[-1]):
        T.pop()
        fondu += 1
    if not T or len(fin) >= len(ref) or len(T) > len(ref):
        return None
    colle = [meme_moment(T[i], ref[i]) for i in range(len(T))]
    toleree = not colle[0]
    k = 1
    while k < len(T) and colle[k]:
        k += 1
    if sum(colle[1:k]) < (DERNIER_TOUR_MIN_COLLE if toleree else 1):
        return None
    tenue = len(T) - k
    if tenue == 0:
        return ({"ouverture": k, "premiere_toleree": toleree, "tenue": 0,
                 "fondu": fondu, "tonique_aux": []} if fondu else None)
    if tenue > k:
        return None
    arrivee = next((c for c in reversed(T[k - 1]) if not c.get("nc")), None)
    tenu = next((c for c in T[k] if not c.get("nc")), None)
    if arrivee is None or tenu is None or \
            not meme_lecture(arrivee, tenu, grain="famille"):
        return None
    if tonique is not None:
        mode = "m" if str(tonique[1]).startswith("min") else "M"
        if not meme_lecture(tenu, {"root": int(tonique[0]) % 12,
                                   "q": "-" if mode == "m" else ""},
                            grain="famille"):
            return None
    elif _role_family_de(tenu) not in ("maj", "min"):
        return None
    aux = [i for i in range(k, len(T))
           if T[i] and not T[i][0].get("nc")
           and meme_lecture(T[i][0], tenu, grain="famille")]
    return {"ouverture": k, "premiere_toleree": toleree, "tenue": tenue,
            "fondu": fondu, "tonique_aux": aux}


def dernier_tour_des_passes(fin, refs, tonique=None) -> dict | None:
    """`dernier_tour_partiel` contre TOUS les passages entiers d'une lettre
    (décodage brut) : il faut que la section finale soit le dernier tour
    d'au moins la moitié d'entre eux. Rend le verdict du premier qui dit
    oui, avec `passes` = (oui, total).

    Pourquoi pas le seul bloc écrit : sur Lost Without U, la passe que la
    façade choisit d'écrire (mes. 17-24) décode `G7sus4 D-7 | D-7 G7 | C |
    C G-7` — l'harmonie décalée d'un demi-temps — et la fin ne colle pas à
    elle ; elle colle à 7 des 9 passages du A."""
    refs = [r for r in refs or [] if r]
    if not refs:
        return None
    oui = [r for r in (dernier_tour_partiel(fin, ref, tonique) for ref in refs)
           if r is not None]
    if not oui or 2 * len(oui) < len(refs):
        return None
    return dict(oui[0], passes=(len(oui), len(refs)))


def ouverture_finale(fin, refs, tonique=None) -> bool:
    """La fin du morceau `fin` est-elle un DERNIER TOUR, partiel, d'une
    lettre dont `refs` sont les passages entiers ? Deux formes, en phase 0 :
      * la règle 2 du 2026-10-01 (`dernier_tour_des_passes` : début du A,
        tonique tenue et/ou fondu) ;
      * le passage COUPÉ par la fin du morceau (Evening Star, mes. 37–43 :
        sept des huit mesures du A, chacune le même moment — `meme_moment`
        — que la mesure de même rang, puis plus rien), pour au moins la
        moitié des passages, comme `dernier_tour_des_passes`.
    Sert au NOMMAGE (`simulation.nommer_dernier_tour`) ; l'écriture le
    raccorde ensuite par `_merge_coupe` ou `_merge_dernier_tour`.

    Ce que ça ne couvre pas : une fin qui joue le MILIEU du A (phase ≠ 0)."""
    if dernier_tour_des_passes(fin, refs, tonique) is not None:
        return True
    refs = [r for r in refs or [] if r]
    fin = list(fin or [])
    T = list(fin)
    while T and _muette(T[-1]):
        T.pop()
    if not refs or not T:
        return False
    # au moins la MOITIÉ du passage doit SONNER (le fondu muet ne compte
    # pas) : une mesure de tonique suivie du silence est la fin du morceau,
    # pas un tour de la lettre (Ic5vxw3eijY : `E | N.C.` contre un A de 4)
    oui = [r for r in refs if len(fin) < len(r) and 2 * len(T) >= len(r)
           and all(meme_moment(T[i], r[i]) for i in range(len(T)))]
    return 2 * len(oui) >= len(refs)


def _role_family_de(c: dict) -> str:
    from harmonia.roles import family
    return family(c.get("q") or "")


def _merge_dernier_tour(host: dict, blk: dict, bars, grid, fin_morceau: int,
                        tonique) -> bool:
    """La règle 2 de Louis du 2026-10-01 (`dernier_tour_partiel`) : la
    DERNIÈRE section du morceau, de la même lettre, qui joue le début du
    bloc puis tient la tonique et s'éteint, rejoint le bloc comme son
    dernier tour partiel.

    Le test se fait sur le décodage BRUT, contre chaque passage entier de la
    lettre (`dernier_tour_des_passes` : au moins la moitié).

    Comme un passage coupé (`_merge_coupe`) : `reps` augmenté, et elle
    n'allume que les rangées qu'elle joue — ouverture et tenue, jamais le
    fondu muet. Là où elle TIENT la tonique au lieu de jouer la rangée
    écrite, la tonique se pose en petite lecture `var` de la case (si la
    case n'en a pas déjà une : une seule variante par case, comme partout),
    sur une copie de la mesure écrite.

    Sur un bloc écrit en 1re/2e fin (les rangées de `barSpans` y sont tronc
    puis variantes), seulement si le tour tient dans le tronc (2026-10-12) ;
    sinon il n'est pas raccordé ici.
    """
    if bars is None or len(blk.get("barRanges") or []) != 1:
        return False
    c0, c1 = blk["barRanges"][0]
    if c1 != fin_morceau or not host.get("bars"):
        return False
    W = len(host["bars"])
    L_host = max(b - a + 1 for a, b in host["barRanges"])
    refs = [bars[a:b + 1] for a, b in host["barRanges"] if b - a + 1 == L_host]
    r = dernier_tour_des_passes(bars[c0:c1 + 1], refs, tonique)
    if r is None:
        return False
    # Un hôte à 1re/2e fin (2026-10-12) : seulement si le dernier tour reste
    # dans le TRONC — ses rangées sont alors les mêmes qu'un bloc sans fins.
    if host.get("endings") and \
            r["ouverture"] + r["tenue"] > W - int(host["endings"]["tail"]):
        return False
    n_rows = len(host["barSpans"])
    for i in range(r["ouverture"] + r["tenue"]):
        if i % W < n_rows:
            host["barSpans"][i % W].append(
                [grid[c0 + i], grid[min(len(grid) - 1, c0 + i + 1)]])
    for i in r["tonique_aux"]:
        t = bars[c0 + i][0]
        row = i % W
        mesure = [dict(c) for c in host["bars"][row]]
        if not mesure or any(meme_lecture(c, t, grain="famille") for c in mesure
                             if not c.get("nc")):
            continue
        cible = mesure[0]
        if "var" in cible:
            continue
        cible["var"] = {"root": int(t["root"]) % 12, "q": t.get("q", ""),
                        "bass": t.get("bass", -1), "c": t.get("c", 0.0), "n": 1}
        host["bars"][row] = mesure
    host["barRanges"].append([c0, c1])
    host["spans"] += [list(sp) for sp in blk["spans"]]
    host["reps"] += blk["reps"]
    bcl = host.get("boucle")
    if bcl is not None:
        bcl["tours_par_passe"] = bcl.get("tours_par_passe", []) + [0]
    logger.info("ireal: dernier tour mes. %d-%d rejoint %s (ouverture %d%s, "
                "tenue %d, fondu %d)", c0, c1, host["label"], r["ouverture"],
                ", 1re tolérée" if r["premiere_toleree"] else "", r["tenue"],
                r["fondu"])
    return True


def _merge_muet(host: dict, blk: dict) -> bool:
    """Une plage MUETTE (que des N.C.) étiquetée d'une lettre rejoint le bloc
    de cette lettre au lieu de prendre un prime (Lost Without U : la mesure
    24, un B d'une mesure en N.C. ; la mesure 57, un A d'une mesure en N.C.
    — les deux « trous » du morceau). Elle reste dans la forme (`barRanges`,
    `spans` : la bande la montre à sa place) mais n'allume aucune case — il
    n'y a rien à lire pendant un silence. CE QUE ÇA NE RÉSOUT PAS : la
    bande de forme la dessine comme la lettre, pas comme un trou (le trou de
    Read, `S._muettes` dans chart.js, ne vise qu'une section entière muette).
    """
    if not _bloc_muet(blk) or _bloc_muet(host):
        return False
    host["barRanges"] += [list(r) for r in blk["barRanges"]]
    host["spans"] += [list(s) for s in blk["spans"]]
    host["reps"] += blk["reps"]
    bcl = host.get("boucle")
    if bcl is not None:
        bcl["tours_par_passe"] = bcl.get("tours_par_passe", []) + \
            [0] * len(blk["barRanges"])
    logger.info("ireal: plage muette mes. %s rejoint %s", blk["barRanges"],
                host["label"])
    return True


def _merge_boucle(host: dict, blk: dict, grid) -> bool:
    """Une lettre écrite comme BOUCLE (`boucle_minimale_ecrite`) absorbe un
    autre bloc de la même lettre qui ne fait que jouer sa cellule, quelle que
    soit sa longueur — plus court, plus long, ou lui-même une boucle (San
    Francisco Street : un A de 24 et un A de 16 mesures, la même cellule de
    4 jouée 6 et 4 fois ; sans ceci le second sortait A′). Chaque mesure du
    bloc allume la rangée de sa position modulo la cellule écrite.
    """
    bcl = host.get("boucle")
    if not bcl or not blk.get("bars"):
        return False
    W = len(host["bars"])
    for i, bar in enumerate(blk["bars"]):
        if _muette(bar):
            continue
        if not meme_moment(bar, host["bars"][i % W]):
            return False
    for c0, c1 in blk["barRanges"]:
        for x in range(c0, c1 + 1):
            host["barSpans"][(x - c0) % W].append(
                [grid[x], grid[min(len(grid) - 1, x + 1)]])
    host["barRanges"] += [list(r) for r in blk["barRanges"]]
    host["spans"] += [list(s) for s in blk["spans"]]
    host["reps"] += blk["reps"]
    autre = blk.get("boucle") or {}
    bcl["tours_par_passe"] = bcl.get("tours_par_passe", []) + (
        autre.get("tours_par_passe")
        or [(c1 - c0 + 1) // W for c0, c1 in blk["barRanges"]])
    bcl["conflations"] = bcl.get("conflations", []) + autre.get("conflations", [])
    if autre:
        bcl["tours_pleins"] = min(bcl["tours_pleins"], autre["tours_pleins"])
    logger.info("boucle: %s mes. %s rejoint la boucle %s (×%d)", blk["label"],
                blk["barRanges"], host["label"], host["reps"])
    return True


def _merge_coupe(host: dict, blk: dict, cell, P, grid) -> bool:
    """Règle 3. Rend True si `blk` a été absorbé par `host`.

    Le test se fait contre la CELLULE de la lettre, pas contre le bloc long :
    c'est la cellule qui est l'unité réelle (le bloc long, c'est la cellule
    jouée deux fois, avec parfois une mesure qui garde la lecture d'un
    passage). Sunny Afternoon en est l'exemple : le bloc de 8 a un `B♭m7` de
    plus en mesure 2, le bloc de 6 suit la cellule à la note près.

    LA PHASE DU RACCORD EST CHERCHÉE, PAS SUPPOSÉE À 0 (Louis, 2026-09-25,
    San Francisco Street : « c'est encore le B qui boucle mais qui ne se
    termine pas, tu considères ça comme un B encore »). Un passage coupé
    n'est pas toujours un DÉBUT interrompu — mesure 40 de ce morceau est un
    accord TENU, qui prolonge la DERNIÈRE position de la cellule (0,97 de
    ressemblance) et non la première (0,61). Le test balaie toutes les
    phases de la cellule au lieu de fixer `i % P` à la phase 0 ; la phase 0
    reste toujours essayée en premier, donc un raccord qui marchait déjà
    (le cas measuré, phase 0) continue à marcher identiquement — ce
    changement ne fait qu'AJOUTER des raccords, jamais en retirer.
    """
    # Un host à 1st/2nd-ending n'est plus écarté d'office (Louis, même
    # échange que ci-dessus). `barSpans` y a déjà les rangées du tronc PUIS
    # celles de chaque variante, dans l'ordre où `_ireal_endings` les a
    # écrites ; la recherche de phase ci-dessous retombe naturellement sur
    # la bonne rangée — SI le raccord colle au contenu ÉCRIT (`host["bars"]`,
    # la variante retenue), il rejoint sa rangée ; sinon (il colle à une
    # AUTRE variante, jamais testée ici) la comparaison échoue proprement et
    # il garde sa propre lettre, comme avant.
    if len(blk["bars"]) >= len(host["bars"]):
        return False
    # UN BLOC PLUS LONG RESTE REFUSÉ, et c'est un choix repris en arrière.
    # Essayé le 2026-09-18 (un bloc plus long qui n'est QUE de la répétition
    # n'ajoute aucune musique — sur Fallin' il restait trois blocs pour une
    # boucle de deux mesures) : ça marche dans le régime des charts en cache,
    # ça n'a PAS été validé sur une ré-analyse fraîche, où le découpage amont
    # est tout autre. Tant que ce n'est pas mesuré dans le bon régime, on ne
    # le livre pas. Voir known_issues, « fusion de lettres ».
    if not P or not cell:
        # Pas de cellule connue (repli refusé, ou découpage fait à la main :
        # le chemin Soudure n'a pas de rapport de repli) — on retombe sur le
        # test le plus simple et le plus sûr : le bloc court est-il le DÉBUT
        # du bloc long ? Un passage qui commence pareil et s'arrête plus tôt
        # est un passage coupé, que le repli des accords ait abouti ou non.
        cell, P = list(host["bars"]), len(host["bars"])
    # `meme_moment` : famille, ou un accord de passage en plus ou en moins
    # (Lost Without U, 2026-09-30). Un fondu muet en fin de passage ne
    # compte pas contre le raccord, comme dans `_merge_boucle` — mais il
    # faut au moins une mesure qui joue.
    L = len(blk["bars"])
    sonnent = [i for i in range(L) if not _muette(blk["bars"][i])]
    if not sonnent or sonnent[-1] != len(sonnent) - 1:
        sonnent = list(range(L))          # un N.C. AU MILIEU reste un N.C.
    # UN HÔTE À 1re/2e FIN, LU FIN PAR FIN (2026-10-12, Evening Star : le
    # dernier A, coupé par la fin du morceau, joue la mesure 7 de la 2e fin
    # — `F`, pas le `F C` de la 1re). Le bloc écrit n'en porte qu'une ; on
    # essaie le tronc suivi de CHAQUE fin, et la mesure qui tombe dans la
    # queue allume la rangée de la fin qu'elle joue (`barSpans` = tronc,
    # puis les rangées de chaque variante, dans l'ordre).
    end = host.get("endings")
    W = len(host["bars"])
    if end and P >= W and end.get("variants"):
        tail = int(end["tail"])
        pre = W - tail
        lectures_ = [(list(host["bars"][:pre]) + list(v["bars"]),
                      list(range(pre)) + [pre + j * tail + t for t in range(tail)])
                     for j, v in enumerate(end["variants"])]
    else:
        lectures_ = [(cell, None)]
    offset, lignes = None, None
    for cel, rangs in lectures_:
        Pc = len(cel) if rangs is not None else P
        offset = next((off for off in range(Pc)
                       if all(meme_moment(blk["bars"][i], cel[(off + i) % Pc])
                              for i in sonnent)),
                      None)
        if offset is not None:
            lignes = rangs
            break
    if offset is None:
        return False
    n_rows = len(host["barSpans"])
    for c0, c1 in blk["barRanges"]:
        for i, b in enumerate(range(c0, c1 + 1)):
            r = offset + i
            if lignes is not None:
                r = lignes[r] if r < len(lignes) else n_rows
            if r < n_rows:
                host["barSpans"][r].append(
                    [grid[b], grid[min(len(grid) - 1, b + 1)]])
    host["barRanges"] += [list(r) for r in blk["barRanges"]]
    host["spans"] += [list(s) for s in blk["spans"]]
    host["reps"] += blk["reps"]
    logger.info("ireal: passage coupé mes. %s rejoint %s en phase %d "
                "(×%d désormais)", blk["barRanges"], host["label"], offset,
                host["reps"])
    return True
