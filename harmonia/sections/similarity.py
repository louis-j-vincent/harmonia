"""harmonia/sections/similarity.py — les briques de ressemblance mesure×mesure.

RÈGLE DE LOUIS (2026-07-30, catégorique) : toute SSM de ce projet doit être
CHORD-TONE, jamais fondamentale seule — un Si♭ majeur est plus proche d'un Sol
mineur (ils partagent Ré et Si♭) que d'un Fa majeur (rien en commun), et une
matrice qui ne regarde que la fondamentale ne peut pas le voir. `harmonic_
vectors`/`ssm` ci-dessous SONT cette matrice — celle que sert déjà l'outil
sections, Soudure et `/ssm` ; il ne doit jamais en exister une seconde.

D'OÙ VIENT CE MODULE (refactor sprint 9, 2026-09-14). Le dispatcher à quatre
détecteurs (`harmonia_min/sections.py`), le détecteur harmonique
(`harmonic_sections.py`) et le détecteur voix (`voice_sections.py`) sont
supprimés — songformer est désormais le seul détecteur de sections
(`harmonia/sections/songformer.py`). Mais trois consommateurs GARDÉS
empruntaient leurs briques de CALCUL, jamais leur décision de découpage :
l'outil sections (finger-swipe + « find repeats »,
`harmonia_min/section_tool.py`), Soudure (`harmonia_min/soudure.py`) et la
page `/ssm` (`harmonia_min/ssm_page.py`). Ce module est leur seule source pour
ces briques désormais — c'est tout ce qui a survécu de trois fichiers :

  * de `harmonic_sections.py` : `TRIAD_TONES`, `chord_tone_matrix`, `_unit`,
    `harmonic_vectors`, `ssm` — le substrat chord-tone lui-même.
  * de `voice_sections.py` : `_diag`, `_slide`, `_sharp`, `_one`,
    `block_score`, `_peaks`, `ODD_BONUS` — la comparaison bloc-à-bloc (score
    = harmonie relative à l'ancre) que `section_tool.find_repeats` et
    `soudure._matrice_bimesures` réutilisent TELLE QUELLE, pour ne jamais
    faire dire à un second scorer autre chose que ce que dit déjà le chart.
  * de `harmonia_min/sections.py` (l'ancien détecteur chroma) :
    `halfbar_features` — le substrat NNLS demi-mesure que `harmonia.folding`
    utilise pour empiler les occurrences d'une lettre (même grain que
    l'ancienne détection, réutilisé pour une raison différente : là il
    servait à COUPER, ici à MOYENNER ce qui a déjà été coupé).

Tout le reste de ces trois fichiers — la recherche de motifs, l'ancrage sur la
voix, la fusion de lettres, les boucles de détection elles-mêmes — meurt avec
eux ; seule l'histoire de git le garde.

`FRAME_DT` vient de `harmonia.musx` (la constante précise, dérivée du hop et
du sample-rate du modèle) plutôt que d'être réécrite ici : `harmonic_sections`
en portait une copie arrondie (23.22e-3 contre 0.0232199546485…), une seconde
définition de la même quantité que rien ne forçait à rester synchrone avec la
première (règle « pas deux implémentations d'un même concept »). Cette brique
n'est PAS sur le chemin de l'analyse automatique — elle ne sert que l'outil,
Soudure et `/ssm` — donc ce changement ne bouge aucun chart du rapport d'or.

CE QUE CE MODULE NE FAIT PAS : il ne détecte aucune frontière et ne nomme
aucune section — ce sont des NOMBRES de ressemblance, à charge des
consommateurs de les lire.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from harmonia.musx import FRAME_DT

# ── le substrat chord-tone (ex harmonic_sections.py) ────────────────────────
# musx triad plane: col 0 = N; col i≥1 is root (i−1)%12, type (i−1)//12+1 in
# {maj, min, sus4, sus2, dim, aug}. Chord tones as semitone offsets:
TRIAD_TONES = {1: (0, 4, 7), 2: (0, 3, 7), 3: (0, 5, 7),
               4: (0, 2, 7), 5: (0, 3, 6), 6: (0, 4, 8)}


def chord_tone_matrix(n_cols: int) -> np.ndarray:
    """(n_cols, 12) — each chord class as its pitch-class indicator."""
    M = np.zeros((n_cols, 12))
    for i in range(1, n_cols):
        root, typ = (i - 1) % 12, (i - 1) // 12 + 1
        for s in TRIAD_TONES.get(typ, (0, 4, 7)):
            M[i, (root + s) % 12] = 1.0
    return M


def _unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def harmonic_vectors(triad: np.ndarray, grid) -> np.ndarray:
    """(n_bars, 12) — each bar's chord posterior, as pitch classes.

    `triad` is musx's triad-plane frame posterior array — the caller already
    holds it (`probs[0]`), so nothing is decoded or re-loaded here.
    """
    M = chord_tone_matrix(triad.shape[1])
    n = len(grid) - 1
    P = []
    for b in range(n):
        a = int(grid[b] / FRAME_DT)
        z = max(a + 1, int(grid[b + 1] / FRAME_DT))
        seg = triad[a:min(z, len(triad))]
        P.append(seg.mean(0) if len(seg) else np.zeros(triad.shape[1]))
    return _unit(np.array(P) @ M)


def ssm(triad: np.ndarray, grid) -> np.ndarray:
    """(n_bars, n_bars) cosine SSM on the chord-tone vectors above — la
    matrice de Louis, jamais fondamentale-seule."""
    V = harmonic_vectors(triad, grid)
    return V @ V.T


# ── des mesures en demi-mesure (ex harmonia_min/sections.py) ────────────────
def halfbar_features(grid: list[float], arr, times) -> np.ndarray:
    """(2*n_bars, 24) mean raw NNLS bothchroma per half-bar, L2-normalised
    per half (bass and treble each unit-norm, so neither half dominates)."""
    edges = []
    for b in range(len(grid) - 1):
        mid = 0.5 * (grid[b] + grid[b + 1])
        edges += [(grid[b], mid), (mid, grid[b + 1])]
    F = np.zeros((len(edges), 24))
    for i, (t0, t1) in enumerate(edges):
        sel = (times >= t0) & (times < t1)
        if not sel.any():
            j = int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))
            v = arr[j]
        else:
            v = arr[sel].mean(0)
        for h in (slice(0, 12), slice(12, 24)):
            n = np.linalg.norm(v[h])
            F[i, h] = v[h] / n if n > 1e-9 else 0.0
    return F / np.sqrt(2.0)          # whole row ~unit norm when both halves live


# ── comparaison bloc-à-bloc (ex voice_sections.py) ──────────────────────────
UNIT = 2           # une section ne commence pas sur une mesure impaire
BLOCK = 8          # la taille de bloc par défaut de `block_score`
SHARP_W = 0.5      # le poids du bonus de netteté (pic local) dans `_one`
ODD_BONUS = 0.05   # ce qu'une reprise à distance IMPAIRE doit payer en plus
                   # au-dessus du seuil normal (repli de parité, pour les
                   # appelants qui l'implémentent — ex. `section_tool.find_repeats`)


def _diag(X, a, b, L):
    """Similarité bloc-à-bloc, alignée : moyenne de X[a+i, b+i]."""
    v = [X[a + i, b + i] for i in range(L)
         if a + i < X.shape[0] and b + i < X.shape[0]]
    return float(np.mean(v)) if v else 0.0


def _slide(M, a0, L, n):
    """Fait glisser le bloc [a0, a0+L) le long de la diagonale : le score de
    reprise à chaque position de départ possible."""
    out = np.zeros(n)
    for c in range(0, n - L + 1):
        v = [M[a0 + i, c + i] for i in range(L) if a0 + i < n and c + i < n]
        out[c] = float(np.mean(v)) if v else 0.0
    return out


def _sharp(cur, p, n, unit=UNIT):
    return 2 * cur[p] - cur[max(0, p - unit)] - cur[min(n - 1, p + unit)]


def _one(cur, n, unit=UNIT, w=SHARP_W):
    return np.array([float(cur[p]) + w * max(0.0, _sharp(cur, p, n, unit))
                     for p in range(n)])


def block_score(cur_m, cur_h, b0, n, mute=None, block=BLOCK):
    """Le score validé : les deux voies (harmonie + chant), relatif à
    l'ancre, le chant abstenu s'il n'entend rien (`mute`)."""
    sm, sh = _one(cur_m, n), _one(cur_h, n)
    if mute is None:
        mix = (sm + sh) / 2.0
    else:
        heard = np.ones(n)
        for p in range(n):
            k = [i for i in range(block) if b0 + i < n and p + i < n]
            if k:
                heard[p] = float(np.mean([(not mute[b0 + i]) and (not mute[p + i])
                                          for i in k]))
        mix = (sh + heard * sm) / (1.0 + heard)
    return mix / max(1e-6, float(mix[b0]))


def _peaks(sc, b0, n, thr, block, claimed, par=0):
    """Les vrais pics : maximum local, au-dessus du seuil, sur la grille — et
    LE PLUS FORT SE SERT LE PREMIER (sinon le premier tue le meilleur).

    `par` est la parité exigée de la distance à l'ancre : 0 (le défaut) est la
    grille de 2, 1 est le repli de parité impaire des appelants qui le gèrent.
    """
    idx, _ = find_peaks(sc, height=thr, distance=UNIT)
    cand = [int(p) for p in idx
            if (p - b0) % UNIT == par and abs(p - b0) >= block
            and p + block <= n and not claimed[p:p + block].any()]
    out = []
    for p in sorted(cand, key=lambda p: -sc[p]):
        if any(abs(p - q) < block for q in out):
            continue
        out.append(p)
    return sorted(out)
