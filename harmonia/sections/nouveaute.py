"""harmonia/sections/nouveaute.py — le détecteur de Foote, repêché.

CE QUE C'EST, en une phrase : on fait glisser un damier le long de la
diagonale de la matrice de ressemblance, et là où le damier « colle » — les
mesures d'avant se ressemblent entre elles, celles d'après aussi, et les deux
moitiés ne se ressemblent pas — il y a une frontière de section.

D'OÙ ÇA VIENT. Le code est repris de `8d5137b:harmonia_min/sections.py`,
supprimé le 2026-09-14 avec le reste du dispatcher à quatre détecteurs
(`_blur` l.69, `_novelty` l.80, `tile_threshold` l.153, `tiling_runs` l.169).
On le repêche parce que depuis le 2026-09-19 il n'y a plus AUCUN détecteur
automatique (SongFormer, supprimé le 2026-09-30, coûtait 2,7 Go de
poids et plus de 16 Go de RAM pour nommer des sections) : le chart sort avec
une seule section « A » et Louis trace ses briques à la main. Ce module ne
prétend pas remplacer son trait — il propose des coupures POUR LES TROUS,
c'est-à-dire pour ce qu'il n'a pas tracé.

CE QUI A CHANGÉ PAR RAPPORT À L'ANCIEN CODE, et il faut le dire parce que ce
n'est pas un simple copier-coller :

  1. **Le substrat n'est plus le même.** L'ancien détecteur lisait la chroma
     NNLS brute en DEMI-MESURES (24 dimensions, basse + aigu). Ici l'entrée
     est la matrice chord-tone par MESURE du projet
     (`harmonia.sections.similarity.ssm`) — la seule matrice de ressemblance
     autorisée (règle de Louis, 2026-07-30, catégorique : il ne doit jamais
     en exister une seconde). Attention, l'histoire est ambiguë et il faut la
     connaître : le 2026-07-31 Louis avait justement REJETÉ une SSM par
     mesure pour ce détecteur — « le SSM a une structure, mais tu la lis
     mal » — parce que la v1 la construisait sur les accords DÉCODÉS, si bien
     que toutes les mesures de This Love s'effondraient sur le même vecteur
     « famille de do mineur ». Ce n'est pas ce qu'on fait ici :
     `similarity.ssm` moyenne la POSTÉRIEURE musx (soft, 12 classes de
     hauteur pondérées) et ne décode rien. Le défaut qu'il décrivait est donc
     atténué, pas supprimé — et c'est exactement ce que le tableau plus bas
     mesure.
  2. **Le grain est la mesure, donc la demi-largeur du noyau est divisée par
     deux.** `KERNEL_HB = 16` comptait des DEMI-mesures, soit 8 mesures de
     contexte de chaque côté ; à notre grain c'est `NOYAU_MESURES = 8`. Garder
     16 aurait doublé le contexte en douce — le genre d'erreur d'unité qui
     corrompt tout en aval en produisant des nombres plausibles.
  3. **La courbe est normalisée.** L'ancien noyau n'était pas normalisé : la
     valeur de nouveauté était en unités arbitraires, et aux bords elle
     n'avait carrément plus la même échelle (moins de masse sous le noyau).
     L'ancien code corrigeait ça en MASQUANT les bords. Ici on divise chaque
     moitié du damier par la masse réellement couverte, donc la courbe vaut
     « ressemblance moyenne dedans − ressemblance moyenne à travers », en
     points de cosinus, partout, bords compris. C'est ce qui rend lisible un
     seuil absolu (`CONTRASTE_MIN`) et donc l'abstention sur un morceau plat.
  4. **On ne porte PAS la couche des lettres** (`LABEL_COS = 0.96` dans
     l'ancien fichier) : nommer les sections est le travail de Louis et de
     `sections.simulation.cherche_brique`. Ce module ne rend que des
     positions.

CE QU'ON A MESURÉ (2026-09-20, trois morceaux que Louis a annotés à la main,
`state/human/sections/`). `frontieres(S)` a tourné sur le morceau ENTIER,
sans aucune brique, contre ses frontières à lui. Numérotation humaine
(mesure 1 = première mesure du chart). « exacte » = la même mesure ;
« à ±2 » = à deux mesures près, c'est-à-dire la tolérance à laquelle une
proposition reste réparable d'un geste dans l'outil.

    Maroon 5 — This Love (80 mesures, 11 sections)
      Louis     9 17 25 29 37 45 49 57 65 73
      proposé  14 17 23 27 37 44 47 50 54 57
      exactes 3/10 (17, 37, 57) · à ±2 : rappel 7/10, précision 8/10
      inventées (à plus de 2 mesures de toute frontière) : 14 et 54
      nouveauté : médiane du morceau 0,028 · aux frontières de Louis 0,043

    Ben E. King — Stand By Me (87 mesures, 11 sections)
      Louis     7 15 23 31 39 47 55 63 71 79
      proposé  22
      exactes 0/10 · à ±2 : rappel 1/10, précision 1/1
      nouveauté : médiane du morceau 0,020 · aux frontières de Louis 0,019

    Norah Jones — Don't Know Why (66 mesures, 11 sections)
      Louis     5 13 15 23 31 39 47 55 63 65
      proposé  24 31 40 47
      exactes 2/10 (31, 47) · à ±2 : rappel 4/10, précision 4/4
      nouveauté : médiane du morceau 0,021 · aux frontières de Louis 0,069

LECTURE HONNÊTE, et c'est la ligne la plus importante de ce fichier : **la
dernière colonne dit tout**. Là où la nouveauté harmonique aux frontières de
Louis dépasse nettement la médiane du morceau — Don't Know Why, 3,3× — le
détecteur ne se trompe pas : ses 4 propositions sont toutes bonnes à ±2, il
n'invente RIEN, il en rate seulement. Là où elle est au niveau du hasard —
Stand By Me, 0,019 contre une médiane de 0,020, c'est-à-dire STRICTEMENT
RIEN — il se tait presque complètement, ce qui est le bon comportement mais
ne rend aucun service. This Love est entre les deux (1,5×) et c'est là qu'il
invente ses deux seules fausses frontières.

POURQUOI Stand By Me est à zéro, et ce n'est pas un bug : le morceau entier
est une seule boucle de quatre accords. Ses sections ne se distinguent PAS
par l'harmonie — elles se distinguent par le chant et l'arrangement. Un
détecteur qui ne regarde que les accords ne peut structurellement pas les
voir. C'est la même chose que le projet avait déjà mesurée en 2026-08-08
quand le détecteur « voix » (0,769) avait battu le détecteur « harmonique »
(0,599) sur les morceaux annotés de Louis. Ce module hérite de cette limite,
entièrement.

PRÉMISSE VÉRIFIÉE AVANT D'ÉCRIRE (règle 2 du CLAUDE.md). On a comparé le
substrat imposé ici — la SSM chord-tone par mesure — au substrat de l'ancien
détecteur — la chroma NNLS brute en demi-mesures — sur ces trois morceaux :
la part des frontières de Louis qui passent au-dessus du 85e centile de leur
propre courbe vaut 3/5/3 pour le chord-tone contre 5/0/4 pour le NNLS. Ni
l'un ni l'autre ne gagne, et le NNLS s'effondre précisément sur Stand By Me.
Le substrat n'est donc PAS ce qui manque ; garder la matrice unique du projet
est le bon choix.

CE QUE ÇA NE RÉSOUT PAS (et qu'il ne faut pas lui attribuer) :

  * **Il ne voit QUE l'harmonie.** Une frontière que Louis entend au chant,
    à la batterie ou à l'arrangement, sans changement d'accords, lui est
    invisible — Stand By Me ci-dessus, 0 sur 10. C'est la limite principale,
    pas un détail.
  * **Aucune résolution sous 8 mesures.** Le noyau fait 8 mesures de
    demi-largeur ; deux frontières plus proches que ça fusionnent en un seul
    pic, et une section de 2 ou 4 mesures (queue, turnaround, pont court) est
    invisible. Mesuré ci-dessus sur Don't Know Why, dont les frontières 13/15
    et 63/65 sont à deux mesures l'une de l'autre : aucune des deux paires
    n'est rendue.
  * **Rien dans les 8 premières et les 8 dernières mesures** (`MASQUE_BORD`) :
    une intro courte ou une coda ne sera jamais proposée. Ce n'est pas un
    oubli, c'est le masque de bord — et il est mesuré (voir sa constante).
  * **Aucun nom.** Ce module rend des positions, jamais de lettre ni de rôle.
    Savoir que deux blocs sont LE MÊME est le travail de
    `simulation.cherche_brique`, et écrire « intro » celui de Louis.
  * **Aucune loi de parité.** L'ancien code snappait les coupures sur les
    mesures PAIRES (« les sections sont toujours des multiples de 2 mesures »,
    Louis 2026-07-31). Ce module ne le fait pas : il rend la mesure où le
    damier colle le mieux. Si un appelant veut la loi des multiples de 2, il
    l'applique lui-même — on ne veut pas cacher un arrondi dans un détecteur.
  * **Rien sur la phase.** Le tuilage (`tuilages`) trouve la PÉRIODE d'une
    boucle (2, 4 ou 8 mesures) et le début du segment tuilé, mais il ne dit
    pas où la boucle commence VRAIMENT si elle démarre décalée ; c'est le
    même trou que `score_periods()` avait avant lui.
  * **Rien si la matière revient TRANSPOSÉE.** `similarity.ssm` compare des
    vecteurs de classes de hauteur absolues : un dernier refrain monté d'un
    ton ne ressemble à rien de ce qui précède, donc le damier y voit une
    frontière (correct) mais le tuilage n'y voit plus sa boucle (faux).
  * **Rien sur un morceau sans contraste.** En dessous de `CONTRASTE_MIN` la
    fonction rend une liste VIDE — délibérément. Une ballade dont toutes les
    mesures se ressemblent (le cas Cry Me A River de
    `simulation.cherche_brique`) ne doit pas recevoir des coupures de bruit.
  * **Rien n'est branché.** Ce module n'est importé par aucun étage de
    `pipeline.analyze_steps`, aucune route, aucun réglage de `settings.py`.
    C'est une brique disponible, pas une décision.
"""
from __future__ import annotations

import numpy as np

# ── les constantes de l'ancien fichier, à leur valeur d'origine ─────────────
#: (`8d5137b:harmonia_min/sections.py` l.29) demi-largeur du damier, en
#: DEMI-MESURES — « = 8 bars context » dit le commentaire d'origine.
KERNEL_HB = 16
#: la même chose à NOTRE grain (la mesure). C'est la conversion d'unité du
#: point 2 du docstring : 16 demi-mesures = 8 mesures.
NOYAU_MESURES = KERNEL_HB // 2
#: (l.30) flou gaussien appliqué à la SSM, en mesures. 0 = pas de flou.
#: Mesuré le 2026-08-05 sur 285 morceaux Billboard / 2781 débuts annotés :
#: retirer le flou fait passer le repère de nouveauté de 27,6 % à 30,1 % de
#: mesures exactes, et la fusion pics+tuilages de 40,2 % à 41,8 %. Ce que le
#: flou coûtait, c'est la POSITION du pic — il étale la frontière qu'il est
#: censé localiser. Gardé à 0, et `_flou` gardé pour que ça reste un réglage
#: qu'on peut rouvrir plutôt qu'un morceau de code supprimé.
BLUR_SIGMA = 0.0
#: (l.32) on garde les pics qui atteignent cette fraction du plus fort.
PEAK_FRAC = 0.5
#: (l.33) on refuse les échardes. « NOT an 8-bar prior », disait l'original.
MIN_SEG_BARS = 2
#: (l.95) une mesure « tuile » à la période P quand sa ressemblance avec la
#: mesure ±P se classe dans les 5 % les plus hauts de la matrice DE CE
#: MORCEAU. Remplace un seuil fixe à 0,80, qui mesurait en réalité
#: l'homogénéité harmonique du morceau et pas sa répétition : il passait de
#: 12,5 % à 46,0 % de la matrice selon le morceau (p10–p90), là où les pics
#: NETS sont eux à un RANG stable (95,7 / 95,8 / 95,8 e centile sur les trois
#: grilles fiables). Sur 285 morceaux Billboard l'exactitude est un match nul
#: (37,0 % contre 37,6 %) ; ce que ça achète c'est la STABILITÉ — la part de
#: matrice acceptée passe de 26,0 % ± 17,2 à 9,9 % ± 0,4.
TILE_QUANTILE = 0.95
#: (l.145) en dessous de cet écart, deux mesures sont trivialement semblables
#: et tireraient le quantile vers le haut.
TILE_LAG_MIN = 2
#: (l.147) en dessous de cette fraction de mesures tuilées, le morceau n'est
#: pas construit en boucles et le tuilage n'a rien à dire. Gardé pour les
#: appelants qui veulent le savoir — `frontieres` ne l'utilise PAS pour
#: basculer d'algorithme (voir `tuilage=` plus bas).
RUN_COVERAGE_MIN = 0.5

# ── ce qui est NOUVEAU ici, et pourquoi ────────────────────────────────────
#: Le contraste minimum, en points de cosinus, pour qu'on accepte de couper.
#: La courbe vaut « ressemblance dedans − ressemblance à travers » : 0,02
#: veut dire « les deux côtés doivent différer d'au moins 2 points de cosinus
#: en moyenne ». Sans ce plancher, les deux seuils hérités sont RELATIFS
#: (une fraction du max, une moyenne + 0,5 σ) et une matrice plate se fait
#: découper sur son bruit numérique. Mesuré le 2026-09-20 : les trois
#: morceaux du tableau plafonnent à 0,131 / 0,106 / 0,204, soit cinq à dix
#: fois au-dessus, pendant qu'une SSM constante donne exactement 0,0. La
#: valeur n'est donc pas un réglage fin qu'on optimise, c'est un
#: interrupteur « y a-t-il du contraste, oui ou non ».
CONTRASTE_MIN = 0.02

#: Combien de mesures on refuse de couper à chaque BOUT du morceau.
#: L'ancien code masquait exactement la demi-largeur du noyau (`nov[kw:n-kw]`,
#: l.362 de `8d5137b`) parce qu'aux bords le damier dépasse et que « leurs
#: valeurs sont des artefacts ». On a d'abord cru que normaliser la courbe par
#: la masse réellement couverte (point 3 du docstring du module) suffirait à
#: les rendre comparables. **Mesuré le 2026-09-20 : non.** Sans masque, les 12
#: plus fortes valeurs de Stand By Me contiennent les mesures 2-5 et 84-87,
#: c'est-à-dire les deux bouts et rien d'autre, et This Love met ses mesures 2
#: et 3 en tête. La raison est simple : au bord, la moyenne « dedans » se
#: calcule sur une ou deux mesures, elle est donc bruitée, pas seulement
#: réduite. Le masque reste. `(n-1)//4` le rétrécit sur les morceaux trop
#: courts pour qu'un masque de 8 mesures laisse quoi que ce soit.
MASQUE_BORD = NOYAU_MESURES


def _flou(S: np.ndarray, sigma: float) -> np.ndarray:
    """Flou gaussien séparable sur les deux axes de la SSM.

    Repris tel quel de `8d5137b:harmonia_min/sections.py::_blur`.
    `BLUR_SIGMA = 0` veut dire « ne floute pas » et c'est le défaut mesuré.
    """
    if sigma <= 0:
        return S
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, S)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)


def _damier(kw: int) -> np.ndarray:
    """Le noyau en damier (2kw × 2kw), pondéré par une gaussienne 2D.

    +1 sur les deux blocs diagonaux (« avant avec avant », « après avec
    après ») et −1 sur les deux blocs croisés (« avant avec après »). La
    gaussienne donne plus de poids à ce qui touche la frontière qu'à ce qui
    en est loin. Repris de `_novelty`, où il était construit à chaque appel.
    """
    K = np.zeros((2 * kw, 2 * kw))
    K[:kw, :kw] = K[kw:, kw:] = 1.0
    K[:kw, kw:] = K[kw:, :kw] = -1.0
    g = np.exp(-0.5 * ((np.arange(2 * kw) - (kw - 0.5)) / (0.6 * kw)) ** 2)
    return K * np.outer(g, g)


def courbe_de_nouveaute(S: np.ndarray, *, kw: int = NOYAU_MESURES,
                        sigma: float = BLUR_SIGMA) -> np.ndarray:
    """La courbe de nouveauté de Foote : UNE valeur par mesure.

    `S` est la matrice de ressemblance mesure×mesure du projet —
    `harmonia.sections.similarity.ssm(triad, grid)`, avec
    `triad = harmonia.musx.frame_posteriors(audio)[0]` (en cache disque sous
    `data/cache/musx_probs`, la bibliothèque entière est chaude). Il n'existe
    qu'UNE matrice de ressemblance dans ce projet et c'est celle-là.

    La valeur en mesure *i* vaut ::

        ressemblance moyenne DEDANS − ressemblance moyenne À TRAVERS

    c'est-à-dire « les 8 mesures d'avant se ressemblent entre elles, les 8
    d'après aussi, et les deux paquets ne se ressemblent pas ». Les deux
    moyennes sont pondérées par la gaussienne du damier (ce qui touche la
    frontière pèse plus) et chacune est divisée par la masse RÉELLEMENT
    couverte — c'est ce qui rend la courbe comparable à elle-même y compris
    aux deux bords du morceau, là où le damier dépasse.

    L'unité est donc le point de cosinus, entre −1 et +1 en théorie. Mesuré
    le 2026-09-20 sur les trois morceaux annotés du docstring du module, elle
    vaut en médiane 0,020 à 0,028 et plafonne à 0,106 / 0,131 / 0,204 — un
    vrai changement de section, quand il s'entend dans les accords, se lit
    entre 0,05 et 0,20.

    MESURÉ, et c'est le chiffre à retenir : la valeur moyenne de la courbe
    AUX frontières de Louis, comparée à la médiane du morceau, vaut 3,3× sur
    Don't Know Why, 1,5× sur This Love et **0,9× sur Stand By Me** — sur ce
    dernier la courbe ne sait littéralement rien, parce que le morceau est
    une seule boucle d'accords et que ses sections se distinguent au chant.
    Ce rapport est le meilleur indicateur « est-ce que ce morceau est du
    ressort de ce détecteur ».

    CE QUE ÇA NE RÉSOUT PAS : la courbe ne voit QUE l'harmonie (voir Stand By
    Me ci-dessus), ne sait rien voir de plus fin que la demi-largeur du noyau
    (8 mesures) — deux frontières plus rapprochées fusionnent en un seul pic,
    une queue de 2 mesures est invisible — et ses deux extrémités sont des
    artefacts, le damier y dépassant du morceau (`frontieres` les masque,
    pas elle). Elle ne dit pas non plus qu'un pic EST une frontière : c'est
    `frontieres` qui tranche, et elle ne nomme rien du tout.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S doit être carrée, reçu {S.shape}")
    n = S.shape[0]
    if n == 0:
        return np.zeros(0)
    kw = max(1, min(int(kw), n))
    S = _flou(S, sigma)
    K = _damier(kw)
    Kp = np.clip(K, 0, None)
    Kn = np.clip(-K, 0, None)
    nov = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - kw), min(n, i + kw)
        sub = S[a:b, a:b]
        ka, kb = kw - (i - a), kw + (b - i)
        p, q = Kp[ka:kb, ka:kb], Kn[ka:kb, ka:kb]
        mp, mq = p.sum(), q.sum()
        if mp <= 0 or mq <= 0:
            continue
        nov[i] = float((sub * p).sum() / mp - (sub * q).sum() / mq)
    return nov


def seuil_de_tuilage(S: np.ndarray) -> float:
    """Le `TILE_QUANTILE`-ième quantile des ressemblances hors-diagonale DE CE
    MORCEAU — le seuil au-dessus duquel on dit « ça se rejoue ».

    Repris de `8d5137b:harmonia_min/sections.py::tile_threshold`, qui prenait
    les vecteurs de mesure `Vb` et recalculait `Vb @ Vb.T` : ici on reçoit
    directement la matrice, qui EST ce produit.

    Sans échelle par construction : la même part de matrice est acceptée sur
    tous les morceaux, quelle que soit leur homogénéité harmonique. Rend 1.0
    (« rien ne passe ») sur un morceau trop court pour qu'on puisse rien dire.

    CE QUE ÇA NE RÉSOUT PAS : un quantile passe TOUJOURS ses 5 % les plus
    hauts. Un morceau à travers-composé, qui ne se rejoue jamais, reçoit donc
    quand même des tuilages — c'est à l'appelant de regarder la couverture
    (`RUN_COVERAGE_MIN`) avant de les croire.
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    if n < 2 * TILE_LAG_MIN + 2:
        return 1.0
    i = np.arange(n)
    off = S[np.abs(i[:, None] - i[None, :]) >= TILE_LAG_MIN]
    if not off.size:
        return 1.0
    return float(np.quantile(off, TILE_QUANTILE))


def tuilages(S: np.ndarray) -> list[dict]:
    """Les zones qui BOUCLENT : `[{"b0", "b1", "period"}]`, mesures incluses.

    Une mesure « tuile » à la période P (2, 4 ou 8 mesures, la plus fine
    d'abord) quand elle ressemble à la mesure P plus loin ou P plus tôt
    au-dessus de `seuil_de_tuilage`. On garde les intervalles contigus de
    même période qui contiennent au moins DEUX tuiles.

    Repris de `8d5137b:harmonia_min/sections.py::tiling_runs` (option A,
    Louis 2026-08-01) : « une frontière de section, c'est là où un tuilage
    s'arrête et un autre commence ».

    MESURÉ LE 2026-09-20, ET C'EST UN RÉSULTAT NÉGATIF À CONNAÎTRE : sur la
    matrice chord-tone PAR MESURE, cette fonction ne rend presque rien — 1
    zone (5 % du morceau) sur This Love, 0 sur Stand By Me, 0 sur Don't Know
    Why. Ce n'est pas le seuil : 42/80, 36/87 et 28/66 mesures le passent.
    C'est la règle « au moins 2×P mesures d'affilée à la MÊME période ». Sur
    l'ancien substrat (chroma NNLS lissée en demi-mesures) les mesures
    tuilées formaient des blocs contigus ; ici un vecteur d'accord change
    complètement d'une mesure à l'autre, les mesures tuilées sont éparpillées
    et alternent entre P=2, 4 et 8, donc aucune suite n'atteint la longueur
    demandée. Quiconque veut rendre cette couche utile sur ce substrat doit
    attaquer la contiguïté (lisser `periode`, ou compter les tuiles sans
    exiger qu'elles se touchent), pas le seuil.

    CE QUE ÇA NE RÉSOUT PAS : la PHASE. On sait que ça boucle toutes les 4
    mesures et où la zone tuilée commence ; on ne sait pas si la boucle
    elle-même démarre sur cette mesure-là ou deux plus loin. Et une boucle
    TRANSPOSÉE n'est pas vue du tout.
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    thr = seuil_de_tuilage(S)
    periode = [0] * n
    for P in (2, 4, 8):
        for b in range(n):
            if periode[b]:
                continue
            avant = b - P >= 0 and float(S[b, b - P]) >= thr
            apres = b + P < n and float(S[b, b + P]) >= thr
            if avant or apres:
                periode[b] = P
    runs, b = [], 0
    while b < n:
        P = periode[b]
        e = b
        while e + 1 < n and periode[e + 1] == P:
            e += 1
        if P and e - b + 1 >= 2 * P:
            runs.append({"b0": b, "b1": e, "period": P})
        b = e + 1
    return runs


def frontieres(S: np.ndarray, *, dans: list[tuple[int, int]] | None = None,
               min_bars: int = MIN_SEG_BARS,
               tuilage: bool = True) -> list[int]:
    """Les mesures où une nouvelle section peut commencer — index 0-basés.

    Rend une liste triée d'index de mesure `c` : « une section commence à la
    mesure `c` », donc la précédente finit à `c-1`. La mesure 0 n'est jamais
    rendue (c'est déjà un début par construction).

    DEUX SOURCES, RÉUNIES. Les pics de `courbe_de_nouveaute` ET les bords des
    zones qui bouclent (`tuilages`). L'ancien code choisissait UNE des deux —
    les bords de tuilage quand ils couvraient au moins `RUN_COVERAGE_MIN` du
    morceau, les pics sinon — puis supprimait tout pic tombant dans une zone
    tuilée. Sur 285 morceaux Billboard : bords seuls 35,8 % de débuts exacts,
    pics seuls 30,1 %, **les deux ensemble 41,8 %** (F de frontière 0,232 /
    0,217 / 0,281). Louis, 2026-08-05 : « pas l'un ou l'autre mais les deux
    ensemble ». Passer `tuilage=False` ne garde que les pics.

    ATTENTION, CE CHIFFRE NE SE TRANSPORTE PAS, et il faut le dire avant que
    quelqu'un s'appuie dessus : ces 41,8 % ont été mesurés sur l'ANCIEN
    substrat (chroma NNLS en demi-mesures), pas sur la matrice chord-tone
    d'ici. Mesuré le 2026-09-20 sur nos trois morceaux, la seconde source est
    **muette** : 1 zone / 5 % de couverture sur This Love, 0 zone sur les
    deux autres, donc `tuilage=True` et `tuilage=False` rendent exactement la
    même chose. Le seuil de tuilage n'est pas en cause (28 à 42 mesures sur
    n le passent) : c'est la règle de contiguïté « au moins deux tuiles de
    MÊME période d'affilée » qui ne survit pas à un substrat par mesure, où
    un vecteur d'accord saute d'une mesure à l'autre et casse les suites.
    Voir `tuilages`. On garde l'union parce qu'elle ne coûte rien et que la
    loi est celle de Louis, pas parce qu'elle rapporte quelque chose ici.

    `dans` restreint les frontières à des TROUS `[(b0, b1), …]` (mesures
    incluses) — typiquement ce que Louis n'a PAS tracé, entre ses briques. La
    courbe, elle, est toujours calculée sur le morceau ENTIER : couper le
    signal aux bords du trou reviendrait à priver le damier de son contexte
    et à fabriquer un faux pic à chaque bord. Seule la SÉLECTION est
    restreinte, et elle l'est strictement à l'intérieur : `c` doit vérifier
    `b0 + min_bars <= c <= b1 + 1 - min_bars`. **Une brique de Louis n'est
    donc jamais touchée** : ni son début, ni sa fin, ni rien entre les deux.

    `min_bars` est le plancher de longueur d'un morceau de section — il sert
    à la fois de fenêtre de maximum local et d'écart minimal entre deux
    frontières rendues.

    ABSTENTION. Si le contraste maximal de la courbe reste sous
    `CONTRASTE_MIN`, on rend `[]`. Mieux vaut ne rien proposer que découper
    une ballade homogène sur son bruit.

    MESURÉ (2026-09-20, sans aucune brique, morceau entier, contre les
    sections faites main de Louis) : Don't Know Why 4 propositions, toutes
    bonnes à ±2 mesures, 4 frontières sur 10 retrouvées ; This Love 10
    propositions, 8 bonnes à ±2, 7 sur 10 retrouvées, 2 inventées ; Stand By
    Me 1 proposition et 1 frontière sur 10 — le morceau est une boucle
    d'accords unique et il n'y a rien à voir dans l'harmonie. Le détail
    mesure par mesure est dans le docstring du module.

    CE QUE ÇA NE RÉSOUT PAS : rien sous 8 mesures (la demi-largeur du noyau),
    aucun nom de section, aucune loi de parité (l'ancien code snappait sur
    les mesures paires, pas nous), et aucune garantie que le nombre de
    frontières rendues soit le bon — c'est une PROPOSITION, arbitrée par
    Louis ou par l'étage qui l'appelle.
    """
    nov = courbe_de_nouveaute(S)
    n = len(nov)
    min_bars = max(1, int(min_bars))
    if n < 2 * min_bars + 1:
        return []
    marge = min(MASQUE_BORD, max(min_bars, (n - 1) // 4))
    lo, hi = marge, n - marge + 1            # positions légales (c < hi)
    interieur = nov[lo:hi]
    if not interieur.size or float(interieur.max()) < CONTRASTE_MIN:
        return []
    seuil = max(PEAK_FRAC * float(interieur.max()),
                float(interieur.mean() + 0.5 * interieur.std()),
                CONTRASTE_MIN)
    pics = [i for i in range(lo, hi)
            if nov[i] >= seuil
            and nov[i] >= nov[max(lo, i - min_bars):i + min_bars + 1].max()]

    bords: list[int] = []
    if tuilage:
        for r in tuilages(S):
            bords += [r["b0"], r["b1"] + 1]

    # le tri : d'abord les bords de tuilage (ils portent la forme), puis les
    # pics du plus fort au plus faible — le premier servi garde sa place.
    ordre = ([(0, -float(nov[c]), c) for c in sorted(set(bords))
              if lo <= c < hi]
             + [(1, -float(nov[c]), c) for c in pics])
    ordre.sort()

    def _dans_un_trou(c: int) -> bool:
        if dans is None:
            return True
        return any(b0 + min_bars <= c <= b1 + 1 - min_bars for b0, b1 in dans)

    gardes: list[int] = []
    for _, _, c in ordre:
        if not _dans_un_trou(c):
            continue
        if any(abs(c - g) < min_bars for g in gardes):
            continue
        gardes.append(c)
    return sorted(gardes)
