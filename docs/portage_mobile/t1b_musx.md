# T1b — musx sur le téléphone : ExecuTorch bloqué, numpy pur marche

## Résultat en 3 lignes

ExecuTorch (avec ou sans XNNPACK) est écarté pour musx : sans partitionneur
la lowering finit (6 min, une fois) mais l'inférence est 8 à 10x trop lente
(≈960 s pour 4 min de musique, ensemble 5 folds) ; XNNPACK reste bloqué
(>27 min, déjà mesuré en amont, non refait ici). Le port numpy pur
(`musx_net.py`) avait un vrai bug (InstanceNorm2d imprécis sur tableau non
contigu, invisible sur un petit extrait, énorme sur un morceau entier) —
corrigé, il est maintenant identique à torch à 2e-6 près, 100 % des accords
argmax identiques sur 3 morceaux, et tourne l'ensemble à 5 folds en 29-42 s
pour 4 min de musique (Mac, 1 thread). **Recommandation : numpy, pas
ExecuTorch, pour musx.**

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Ce que « parité » doit vouloir dire (lu dans le code, pas supposé).**
`harmonia/musx.py::_run_nets` charge les 5 checkpoints (`MODEL_NAMES`, fold
s0..s4) et fait la **moyenne simple des 5 sorties** ; le script d'origine du
clone (`third_party/musx_ismir2019/chord_recognition.py:39`) fait exactement
la même chose (`np.mean([p[i] for p in probs], axis=0)`). Ce n'est PAS un
seul réseau : chaque inférence coûte 5x un forward CNN+biLSTM, et « parité »
veut dire comparer la moyenne des 5, pas un fold isolé.

### Piste 1 — ExecuTorch sans XNNPACK (`to_edge` → `to_executorch()`)

`mobile/tools/export_musx.py --mode portable` (fold 0, fenêtre fixe 500
frames = 11,61 s d'audio) :

| étape | temps |
|---|---:|
| `torch.export()` | 7,5 s |
| `to_edge` (aucun partitionneur) | 213,8 s |
| `to_executorch()` | 133,7 s |
| **total lowering (une fois)** | **≈355 s (5,9 min)** |
| taille du `.pte` | 4,79 Mo |

La lowering **finit** (contrairement à XNNPACK, qui ne finit jamais — voir
`p0_t1_executorch.md`, >27 min, tué). Mais l'inférence, mesurée avec le
runtime python ExecuTorch, 1 thread, sur le même extrait 500 frames
(`mobile/tools/speed_musx_executorch.py`) :

- **9130 ms ± 1567 ms par chunk de 500 frames (11,61 s d'audio), UN SEUL
  fold.**
- extrapolé à 4 min de musique (21 chunks) : **192 s pour 1 fold, 959 s
  (16 min) pour l'ensemble à 5 folds.**

Le budget total du téléphone pour TOUT le pipeline est 120 s (porte T5 du
PLAN). Musx seul en ExecuTorch portable en consommerait 8x plus. **Rejeté
sur la vitesse**, malgré une lowering qui aboutit.

Cause probable (déjà notée dans `p0_t1_executorch.md`, confirmée
indirectement ici) : `torch.export` déroule la `nn.LSTM` bidirectionnelle en
un graphe d'environ 500 pas de temps × plusieurs dizaines d'opérations
chacun — un graphe énorme que l'exécuteur portable interprète nœud par nœud
sans fusion, d'où le coût par chunk.

### Piste 2 — XNNPACK conv-only, LSTM portable : écartée sans re-mesurer

Piste explicitement autorisée à être écartée par le raisonnement si la
prémisse tombe (règle du projet : cribler avant d'implémenter). Vérifié dans
le partitionneur XNNPACK installé (`executorch.backends.xnnpack.partition.
xnnpack_partitioner.ALL_PARTITIONER_CONFIGS`, 52 configs) : **aucune config
ne cible les ops LSTM/RNN.** Autrement dit, la tentative XNNPACK d'origine
(`p0_t1_executorch.md`) ciblait déjà UNIQUEMENT les convolutions — le LSTM
n'a jamais été candidat à la délégation XNNPACK, avec ou sans restriction
explicite. Le blocage à >27 min n'est donc pas un problème de « quelles ops
déléguer », mais la passe de correspondance de motifs du partitionneur qui
explose sur l'énorme graphe déroulé (la piste 1, MÊME graphe, SANS XNNPACK,
a déjà mis 3,5 min rien que pour `to_edge`). Refaire la piste 2 en pratique
n'aurait rien changé à ce diagnostic — budget de 40 min non consommé,
réinvesti dans la piste 3 (qui en avait besoin : un vrai bug y a été trouvé).

### Piste 3 — réseau entier en numpy (`mobile/harmonia_mobile/musx_net.py`)

Architecture reproduite exactement depuis `chordnet_ismir_naive.py`
(vérifié sur `state_dict()` du fold 0) : CNN (8 conv2d + InstanceNorm2d +
SELU + maxpool, 252→84→28→7→5→3 bins) → bi-LSTM(240→96×2) → Linear(192→100)
→ split `[triad(73), bass(13), s7(4), s9(4), s11(3), s13(3)]` (bornes
lues dans `ChordNet.forward`, PAS dans le docstring de `ChordTypeLimit` —
les deux donnent le même total mais des découpes différentes sur papier).

**Bug trouvé et corrigé.** Premier essai : posteriors proches sur un
extrait de 200 frames (~2e-4) mais l'écart EXPLOSAIT sur un morceau entier
(0,0015 à 205 s → 0,056 à 422 s), avec l'accord argmax d'accord descendant à
99,04 % sur le morceau le plus long (cible 99,9 %). Diagnostic couche par
couche (torch vs numpy, mêmes poids) : le `conv2d` brut matchait à 1e-6 près,
mais la PREMIÈRE `InstanceNorm2d` divergeait déjà de 0,7 en absolu sur le
morceau entier. Cause isolée : `_conv2d` renvoie un tableau non contigu
(`.transpose(2,0,1)`), et réduire (`mean`/`var`) sur un tableau non contigu
avec beaucoup d'éléments (jusqu'à T×F ≈ 4,6M par canal sur le plus long
morceau) perd de la précision en float32 — vérifié isolément sur un tableau
synthétique de même forme : erreur moyenne 3,5e-5 (non contigu) contre 7,7e-8
(copie contiguë), soit 450x. Corrigé : forcer une copie contiguë en float64
pour la réduction seulement (`_instance_norm`, `musx_net.py`).

**Parité après correction** (`mobile/tools/parity_musx.py`, ensemble 5
folds, référence = torch réel, mêmes poids) :

| morceau | durée | écart max (5 têtes) | argmax accord | numpy (5 folds) | normalisé/4min |
|---|---:|---:|---:|---:|---:|
| maroon_5_this_love | 205,3 s | 2e-6 | **100,00 %** | 35,8 s | 41,9 s |
| ben_e_king_stand_by_me | 177,5 s | 2e-6 | **100,00 %** | 21,5 s | 29,1 s |
| autumn_leaves (jazz) | 422,3 s | 1e-6 | **100,00 %** | 72,6 s | 41,3 s |

Cibles : écart max 1e-3 (marge ~500-1000x), argmax ≥99,9 % (tenu à 100,00 %
partout). **Vitesse : 29-42 s pour 4 min de musique, ensemble 5 folds
complet, 1 thread, Mac** — sur les 120 s de budget total téléphone (porte
T5), ça laisse de la marge pour battues+CQT+NNLS+reste.

`_conv2d` optimisé en cours de route : la première version (9 `tensordot`
dans une boucle python) coûtait 2,5 s pour une seule couche à pleine
résolution (Cin=Cout=16, T=8840, F=252) ; remplacée par UN SEUL matmul (les
9 tirages du noyau empilés sur l'axe canal) — même résultat, ~2 à 7x plus
rapide selon la couche. C'est la CNN qui domine le coût (≈15 s avant
optimisation sur un morceau de 205 s, 1 fold), le bi-LSTM est rapide malgré
la boucle python par pas de temps (≈0,5 s sur le même morceau, 8840 pas ×
2 sens — le coût par pas est petit, hidden=96).

### Décodage HMM (`xhmm_ismir.py`)

**Non branché, faute de temps** (budget 2h30 consommé par le diagnostic du
bug numpy, qui en valait la peine). Le décodeur est déjà pur numpy
(confirmé en le lisant), donc portable tel quel. Comme les postérieures sont
maintenant identiques à torch à 2e-6 près, le décodage final héritera de
cette fidélité — mais ce n'est PAS mesuré ici, seulement une conséquence
attendue.

## Ce qui reste ouvert

- **Aucune mesure sur téléphone** — tout ce document est Mac, 1 thread. Le
  facteur Mac→téléphone est inconnu (T5 du PLAN, bloqué en attente du
  téléphone).
- **Décodage HMM final non vérifié bout-en-bout** (voir ci-dessus) — à faire
  avant de clore la parité complète du pipeline musx.
- **`_conv2d` encore perfectible** : une seule passe BLAS par couche, mais
  pas de fusion inter-couches ; si le budget téléphone est serré une fois
  T1b+T1c+T2 assemblés, c'est le premier endroit à reprofiler (la CNN domine
  largement le bi-LSTM ici).
- **Piste 2 (XNNPACK conv-only) écartée par raisonnement, pas mesurée en
  pratique** — si une version future d'ExecuTorch change son partitionneur
  (moins sensible à la taille du graphe), ça vaudrait la peine de refaire un
  essai court ; pour cette version installée (executorch 1.5.1), le
  diagnostic ci-dessus est jugé suffisant.
- **Disque à 5,9 Go libres** en fin de tâche (seuil d'arrêt 3 Go, jamais
  atteint) — d'autres agents (T1c) écrivent aussi de gros `.pte` (82 Mo x2)
  dans le même dossier `mobile/models/`, à surveiller si plusieurs tâches
  tournent encore en parallèle.
- **Le bug InstanceNorm2d est probablement générique** : tout code numpy qui
  réduit (mean/var/sum) sur un tableau issu d'un `.transpose(...)` sans le
  rendre contigu risque la même perte de précision à grande échelle. Si
  T1c ou un futur port numpy fait des réductions similaires, vérifier ce
  point précis en premier — c'est allé de "ça a l'air bon sur un petit
  test" à "faux de 0,7 en absolu sur un morceau entier" sans aucun signal
  d'erreur (pas de crash, pas de warning).

## Fichiers produits

- `mobile/harmonia_mobile/musx_net.py` — port numpy pur (CNN + bi-LSTM +
  tête linéaire + split + softmax), **le chemin recommandé**. Fonctions clé :
  `predict_fold`, `predict_ensemble`, `load_weights_npz`,
  `dump_weights_from_torch`.
- `mobile/tools/dump_musx_weights.py` — convertit les 5 checkpoints torch en
  `.npz` plats (une fois, avec le venv mobile) pour que l'inférence numpy
  n'ait plus besoin de torch du tout.
- `mobile/tools/parity_musx.py` — parité + vitesse, numpy vs torch, sur les
  3 morceaux de référence (table ci-dessus).
- `mobile/tools/export_musx.py` — mis à jour : `--mode portable` (celui qui
  aboutit, mesuré ici) et `--mode xnnpack` (celui qui bloque, gardé pour
  mémoire), docstring mise à jour avec le verdict et la recommandation.
- `mobile/tools/speed_musx_executorch.py` — vitesse d'inférence du `.pte`
  portable (piste 1), runtime python ExecuTorch, 1 thread.
- `mobile/models/musx_fold{0..4}_weights.npz` (gitignorés, 2 Mo chacun),
  `mobile/models/musx_fold0_portable.pte` (gitignoré, 4,79 Mo, piste 1 —
  produit mais non recommandé), `mobile/models/parity_musx_numpy_result.npz`
  (résultats bruts de la table ci-dessus).
