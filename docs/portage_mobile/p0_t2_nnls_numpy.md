# P0-T2 — port numpy de `nnls-chroma`

## Résultat en 3 lignes

OUI : le port numpy pur (`mobile/harmonia_mobile/nnls_chroma.py`, ~400 lignes,
sans scipy, NNLS Lawson-Hanson maison) reproduit `vamp.collect(...,
"nnls-chroma:nnls-chroma", output="bothchroma")` avec un écart max < 3e-4 sur
les 3 morceaux (cible 1e-3 — large marge) et un argmax de la basse (ce que
`_write_sounding_bass` consomme) identique à 100 % frame par frame sur les
3 morceaux. Vitesse desktop (Mac, pas le téléphone) : ~6-8x temps réel pour
un morceau de 4 min (31 s pour 240 s d'audio) — repli NDK pas nécessaire pour
ce brique-là.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Spécification exacte** — `harmonia/nnls_features.py:103` appelle
`vamp.collect` SANS dictionnaire de paramètres, donc ce sont les défauts du
plugin qui font foi. Lus dans `~/Library/Audio/Plug-Ins/Vamp/nnls-chroma.n3`
(RDF du plugin réellement installé sur la machine) et confirmés dans la
source C++ (`NNLSBase.cpp`, `NNLSBase::NNLSBase`, `getParameterDescriptors`) :

    useNNLS=1  rollon=0%  tuningmode=0 (global)  whitening=1.0  s=0.7
    chromanormalize=0 (aucune — nnls_features.py fait sa propre L2 par moitié)
    stepSize=2048  blockSize=16384  (SR=44100 → featureRate=21.5332 Hz)

Source ported (GPL, github.com/c4dm/nnls-chroma, fetché via `curl` — pas de
`git clone`) : `NNLSBase.cpp/.h`, `NNLSChroma.cpp`, `chromamethods.cpp/.h`,
`nnls.c`. Plus `vamp-plugin-sdk` (`PluginInputDomainAdapter.cpp`, `Window.h`)
et `vamp` python (`vamp/frames.py`) pour la géométrie exacte des trames :
trame `i` = échantillons bruts `[i*step, i*step+block)`, **pas de padding
centré au début** (frame 0 commence à l'échantillon 0) ; le mode `"matrix"`
de `vamp.collect` réindexe les temps de sortie sur `arange(T)*step/SR`, donc
seul l'ORDRE des trames compte, pas l'horodatage interne du plugin.

**Algorithme porté** (le pipeline complet, dans l'ordre) :
1. fenêtre de Hann **périodique** (dénominateur N, pas N-1 — `Window.h`
   `cosinewin` avec a0=a1=0.5) sur chaque trame de 16384 échantillons.
2. FFT réelle, magnitude des 8192 premiers bins (bin de Nyquist ignoré),
   clip à `blockSize`, mise à zéro si le pic de la trame est sous le seuil
   de silence (`2.0` pour blockSize=16384).
3. projection en spectre log-fréquence à 256 bins (3 bins/demi-ton, 7
   octaves) via un noyau creux construit une seule fois
   (`build_logfreq_kernel`, port de `chromamethods.cpp:logFreqMatrix` —
   noyau cosinus sur-échantillonné ×80, indépendant de l'audio, ~0.15 s à
   construire).
4. estimation de tuning **globale** (seul mode exercé par `nnls_features.py`
   — tuningmode local NON porté) : moyenne sur tout le fichier de l'énergie
   par position-dans-le-demi-ton, projetée en angle complexe → décalage
   fractionnaire de bin.
5. interpolation "tuned log-freq spectrum" avec ce décalage, puis
   blanchiment spectral (moyenne/écart-type glissants, noyau Hamming demi-
   octave de 19 bins, `SpecialConvolution` — convolution "valid" avec
   réplication des bords, pas de zero-padding).
6. NNLS (Lawson-Hanson maison, `nnls_lh`, float64, sans scipy) contre un
   dictionnaire de 84 profils harmoniques décroissants (`s=0.7`, 20
   harmoniques, port de `dictionaryMatrix`), restreint aux notes candidates
   à énergie non nulle (comme le C++ le fait avec `signifIndex`).
7. pliage en chroma basse/aiguë par les fenêtres `basswindow`/`treblewindow`
   (recopiées telles quelles depuis `chromamethods.h`), empilées
   `[bass(12), treble(12)]`, index 0 = A — même convention que
   `nnls_features.py` attend.

**Références** générées avec le vrai plugin sur le Mac (`.venv` du worktree,
rien installé) : `mobile/tools/gen_nnls_ref.py` →
`mobile/tools/nnls_ref/{maroon_5_this_love,ben_e_king_stand_by_me_audio,
autumn_leaves}.npz` (`arr`, `times`, `sr`, `step`, `duration` — un seul
`.npz` par morceau plutôt que des `.npy` séparés, mêmes données, un seul
fichier à suivre). Le 3ᵉ morceau (jazz) = `autumn_leaves.m4a`.

**Validation** (`mobile/tools/validate_nnls_chroma.py`, sortie complète du
run du 2026-09-23) :

| morceau | durée | max abs diff | mean abs diff | corr/trame moy. | argmax basse identique |
|---|---|---|---|---|---|
| maroon_5_this_love | 205.3 s | 0.00011 | 0.000005 | 1.00000 | 100.00 % |
| ben_e_king_stand_by_me_audio | 177.5 s | 0.00022 | 0.000010 | 1.00000 | 100.00 % |
| autumn_leaves (jazz) | 422.3 s | 0.00029 | 0.000012 | 1.00000 | 100.00 % |

Cible du plan (max abs diff < 1e-3) : **atteinte sur les 3 morceaux**, marge
×3-9. Le seul écart vient de float64 (port) vs float32 (plugin C++) partout
dans le calcul — pas d'écart structurel (fenêtrage/tuning/blanchiment tous
alignés). Comparaison faite sur la portion commune des deux sorties
(longueur = min des deux `T`, les deux utilisent le même pas de 2048/44100 s
donc l'alignement trame-à-trame est exact, pas d'interpolation nécessaire).

**Vitesse** (`validate_nnls_chroma.py`, même machine, un seul thread numpy,
construction noyau+dictionnaire = ~0.15 s, non recomptée par morceau car
indépendante de l'audio) : morceau de 240 s (4 min, extrait d'`autumn_leaves`)
traité en 31.0 s → ~7.7x temps réel **sur ce Mac** — pas une mesure
téléphone, juste "la brique n'est pas elle-même impossible à porter côté
vitesse" ; T5 (vitesse sur téléphone réel) tranchera pour de vrai, et le
NNLS par trame (boucle Python + `numpy.linalg.lstsq` par sous-ensemble actif)
est le suspect n°1 si ARM+Chaquopy est lent.

## Ce qui reste ouvert

- **Licence GPL** (comme demandé, sans bloquer) : le port copie l'ALGORITHME
  documenté dans la thèse/papier ISMIR 2010 de Mauch & Dixon et vérifié
  contre la source GPL de `c4dm/nnls-chroma` — c'est un travail dérivé de
  cette lecture, pas une traduction ligne à ligne du C++ (les boucles
  d'accumulation ont été reformulées en numpy vectorisé, mais suivent
  exactement la même récurrence mathématique). Pour une distribution future
  (Play Store, App Store), il faut trancher : soit le traiter comme "dérivé
  de GPL" et l'app entière devient GPL pour cette brique, soit refaire
  l'implémentation depuis la seule lecture du papier ISMIR (propre room) sans
  jamais avoir ouvert le `.cpp`. Usage actuel = perso/sideload, donc pas
  bloquant, mais NE PAS publier tel quel sans cette décision.
- **`tuningmode=1` (tuning local) non porté** — `nnls_features.py` ne
  l'exerce jamais (aucun `parameters={}` dans l'appel), donc pas nécessaire
  pour ce port, mais un futur appel avec des paramètres différents cesserait
  silencieusement de matcher ce port sans avertissement — il faudrait un
  garde-fou (assert sur les paramètres attendus) si un jour quelqu'un change
  `nnls_features.py`.
- **`rollon > 0` codé mais non testé** — le chemin par défaut est 0 (skip),
  jamais exercé par la suite de validation.
- **Vitesse téléphone inconnue** — la boucle NNLS par trame (~4400-9100
  trames selon le morceau) est en pur Python + `numpy.linalg.lstsq`, pas
  vectorisée across frames (chaque trame a un sous-ensemble actif de taille
  différente). Sur ARM sans BLAS optimisé ni JIT, ce point est le risque n°1
  pour la porte T5 (4 min < 2 min de calcul) — à profiler en premier une fois
  Chaquopy disponible (T1/T4).
- **NNLS maison non torturé sur cas dégénérés** — testé seulement sur audio
  musical réel (jamais de colonnes parfaitement colinéaires observées dans
  les 3 morceaux) ; pas de garantie formelle de convergence sur un dictionnaire
  pathologique, contrairement au `nnls.c` original qui a `itmax` et un mode
  d'erreur explicite. Le port a un `max_iter` de sécurité mais ne signale pas
  d'échec de convergence — silencieux si jamais ça arrive.

## Fichiers produits

- `mobile/harmonia_mobile/nnls_chroma.py` — le port numpy (module public :
  `extract_bothchroma_numpy(y, sr)`).
- `mobile/tools/gen_nnls_ref.py` — génère les références VAMP réelles
  (Mac only, `.venv` du worktree).
- `mobile/tools/validate_nnls_chroma.py` — compare port vs référence,
  chronomètre le port sur un extrait de 4 min.
- `mobile/tools/nnls_ref/maroon_5_this_love.npz`
- `mobile/tools/nnls_ref/ben_e_king_stand_by_me_audio.npz`
- `mobile/tools/nnls_ref/autumn_leaves.npz`
