# P0-T2b — vitesse du port numpy de `nnls-chroma`

## Résultat en 3 lignes

Le suspect était le bon : la boucle Python par trame appelant
`nnls_lh`/`numpy.linalg.lstsq` (145 713 appels pour 5 168 trames) faisait
27,4 s des 32,0 s ; remplacée par une résolution NNLS **batchée sur toutes
les trames à la fois** (FISTA — descente de gradient projeté accélérée,
même dictionnaire fixe pour toutes les trames, restriction par colonne
"signifIndex" devenue un masque plutôt qu'une matrice réduite), plus la FFT
par trame vectorisée (`sliding_window_view` + FFT batchée + un seul matmul
`mag @ K.T`) : **240 s d'audio en 2,99 s** (cible ≤ 4 s — atteinte, marge
25 %), parité inchangée (argmax basse 100 %, max abs diff 0,0001-0,0003,
même ordre de grandeur que le port P0-T2 d'origine).

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Profilage d'abord** (`cProfile`, extrait de 240 s d'`autumn_leaves.m4a`,
`.venv/bin/python` sur un script ad hoc, cf. méthode) : 31,97 s total,
27,4 s cumulés dans `nnls_lh` (dont 21,6 s rien que dans
`numpy.linalg.lstsq`, appelé en moyenne ~28 fois par trame car l'active-set
de Lawson-Hanson résout un `lstsq` à chaque itération interne), 3,4 s dans
la boucle Python de `extract_bothchroma` elle-même (construction de trame +
fenêtrage), 0,87 s dans `rfft`. Confirmé : NNLS était bien le suspect n°1
noté dans `p0_t2_nnls_numpy.md`.

**Remplacement NNLS (piste 2 du brief)** — `nnls_batch_fista` dans
`mobile/harmonia_mobile/nnls_chroma.py` : le dictionnaire `D` (256×84) est
**le même pour toutes les trames** (contrairement à ce que suggérait la
lecture rapide du code d'origine — la restriction `Ad = D[:, sig]` par
trame n'est qu'une contrainte "x_j=0 pour les colonnes hors support", donc
mathématiquement identique à un masque appliqué à chaque itération sur un
solve à dictionnaire fixe). Ça permet une seule paire de matmuls
`(T,84)@(84,84)` par itération FISTA pour TOUTES les trames d'un coup,
plutôt qu'un `lstsq` par trame. Pas de scipy/numba — `np.linalg.eigvalsh`
(une fois, pour le pas de Lipschitz) et des matmuls BLAS.

**FFT vectorisée (piste 4)** : `sliding_window_view` + fenêtrage + `rfft`
batchés sur toutes les trames, puis un seul `mag @ K.T` remplace la boucle
`K @ mag` par trame — supprime les 3,4 s + 0,87 s de la boucle Python. Cette
partie est utilisée par LES DEUX méthodes (`fista` et `lawson`) : c'est une
pure vectorisation de la même arithmétique, pas une approximation.

**Sweep d'itérations FISTA** (`sweep_iters.py` ad hoc, `autumn_leaves.m4a`
complet) : la parité plafonne à `max_abs_diff≈0,00029` à partir de
`max_iter=250` (250 et 350 itérations donnent le même chiffre — c'est le
plancher de convergence, pas un manque d'itérations) ; `max_iter=150`
laissait plus de bruit (jusqu'à 0,00087, encore sous la cible 1e-3 mais avec
moins de marge). Retenu : `max_iter=250` par défaut.

**Validation complète** (`.venv/bin/python mobile/tools/validate_nnls_chroma.py`,
méthode par défaut `fista`, `max_iter=250`) :

| morceau | durée | avant (Lawson, P0-T2) | après (FISTA batché) | max abs diff | argmax basse |
|---|---|---|---|---|---|
| maroon_5_this_love | 205,3 s | ~26,7 s (7,7x tr.) | 2,5 s (80,7x tr.) | 0,00011 | 100 % |
| ben_e_king_stand_by_me_audio | 177,5 s | ~23,0 s (7,7x tr.) | 1,9 s (94,4x tr.) | 0,00022 | 100 % |
| autumn_leaves (jazz) | 422,3 s | ~54,8 s (7,7x tr.) | 6,7 s (63,5x tr.) | 0,00029 | 100 % |
| 4 min (extrait autumn_leaves) | 240,0 s | 31,0-32,0 s | **2,99 s** | — | — |

(Colonne "avant" : le P0-T2 original n'a été chronométré qu'une fois sur
l'extrait 4 min à 31,0-32,0 s (~7,7x temps réel) ; les temps par morceau
"avant" ci-dessus sont extrapolés à ce ratio pour la comparaison — seule la
ligne "4 min" est une mesure directe avant/après sur le MÊME extrait.)

Vitesse gagnée : **~10-11x sur la brique NNLS+FFT**, cible de vitesse
(≤ 4 s/240 s) **atteinte avec marge** (2,99 s, 25 % de marge). Parité
inchangée par rapport au P0-T2 d'origine (même ordre de grandeur de max abs
diff, argmax basse toujours 100 % identique sur les 3 morceaux) — pas de
compromis à proposer, les deux objectifs sont tenus simultanément.

**Sanity check `method="lawson"`** : un premier test tronqué à 60 s a donné
un max abs diff de 1,52 (faux positif) — l'estimation de tuning est
**globale** (moyenne sur tout le fichier, cf. docstring du module), donc
tronquer l'audio change l'estimation et casse la comparaison contre une
référence pleine longueur ; ce n'est pas un bug du port. Réessayé sur le
morceau complet (`ben_e_king_stand_by_me_audio.m4a`, 177,5 s, 19,18 s,
`method="lawson"`) : max abs diff 0,000246, argmax basse 100 % — identique
aux chiffres P0-T2 d'origine. Confirme que la boucle FFT vectorisée
(partagée par les deux méthodes) n'a rien changé mathématiquement.

## Ce qui reste ouvert

- **Vitesse téléphone toujours inconnue** — ces chiffres sont Mac, un seul
  thread numpy. T5 tranchera. Le point positif : la nouvelle méthode
  remplace ~146k petits appels `lstsq` (mauvais pour BLAS mobile, overhead
  Python par appel) par ~250 gros matmuls BLAS `(T,84)@(84,84)` — un profil
  beaucoup plus favorable à un ARM sans JIT que la boucle Lawson-Hanson.
  Reste à vérifier que le BLAS embarqué (numpy pip wheel via Chaquopy) a une
  implémentation `dgemm` raisonnable, pas une référence naïve.
- **FISTA n'a pas de critère d'arrêt adaptatif** — `max_iter=250` fixe pour
  toutes les trames, y compris celles qui convergent en 20 itérations ;
  fonctionnellement inoffensif (les itérations supplémentaires ne changent
  plus rien une fois convergé) mais pas optimal en calcul. Un critère sur la
  norme du gradient projeté par trame apporterait un peu de marge de vitesse
  en plus mais complique le batch (des trames à des étapes différentes) ;
  pas fait ici, le budget de temps (≤ 4 s) est déjà tenu sans.
- **`rollon > 0` reste sur l'ancienne boucle Python par trame** (non
  vectorisé) — toujours non exercé par `nnls_features.py` (paramètre par
  défaut = 0), donc hors du chemin chaud ; signalé tel quel dans le
  P0-T2 d'origine, inchangé ici.
- **Pas torturé sur cas dégénérés** — comme le P0-T2 d'origine, testé
  seulement sur les 3 morceaux réels ; FISTA converge vers le même optimum
  que Lawson-Hanson pour un problème bien posé (dictionnaire de rang plein,
  ce qui est le cas ici), mais aucune garantie formelle sur un dictionnaire
  pathologique n'a été vérifiée.

## Fichiers produits

- `mobile/harmonia_mobile/nnls_chroma.py` — modifié en place : nouvelle
  fonction `nnls_batch_fista` (FISTA batché), FFT/fenêtrage vectorisés dans
  `extract_bothchroma` (utilisés par les deux méthodes), paramètre
  `method="fista"|"lawson"` (défaut `fista`, `max_iter=250`) ; l'ancienne
  boucle Lawson-Hanson par trame (`nnls_lh`, inchangée) reste accessible via
  `method="lawson"`.
- `mobile/tools/validate_nnls_chroma.py` — inchangé (script de validation
  déjà existant, réutilisé tel quel).
