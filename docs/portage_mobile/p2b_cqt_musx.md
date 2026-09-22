# P2b — port numpy pur du CQT front-end de musx (`extractors/cqt.py:CQTV2`)

## Résultat en 3 lignes

Oui, reproductible, et bien au-delà des cibles une fois le vrai bug trouvé :
argmax accord **99,99–100 %** identique (cible 99,5 %), écart max des
features **2,6e-5 à 3,5e-5 relatif** (cible 1e-3), **~1,2–2,9 s** pour 4 min
d'audio sur Mac (cible ≤5 s) — le suspect annoncé (le resampler `soxr_hq`
interne, non reproductible bit-exact) s'est avéré INNOCENT : un test de
contrôle (basis + STFT réels de librosa, moteur `soxr_hq` réel appelé
directement) donnait la MÊME erreur que mon port, ce qui a mené au vrai bug
— `__early_downsample` (un sous-échantillonnage 2x fait AVANT la boucle par
octave) était absent du port, et se déclenche bel et bien pour l'appel
réel (238 bins, pas 288) ; une fois ajouté, l'erreur a chuté d'un facteur
~200 (5,5e-3 → 2,6e-5). La branche pseudo-CQT (36+14 bins aigus, ~17 % du
spectre, sans aucun resampling) était déjà exacte à 1e-7 dès le premier
essai — c'est la branche récursive/basse (83 % du spectre, celle qui porte
la fondamentale) qui a tout le risque, et l'a résolu.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**La chaîne exacte, lue dans le code (pas supposée)** :
`third_party/musx_ismir2019/extractors/cqt.py:CQTV2.extract` appelle
`librosa.core.hybrid_cqt(y, bins_per_octave=36, fmin=note_to_hz('F#0')≈23.125 Hz,
n_bins=288, tuning=None, hop_length=512)` puis `abs(...).astype(float32)`,
sur `y=librosa.load(path, sr=22050, mono=True)` (`mir/io/implement/music_io.py`,
SANS normalisation — le `(y-mean)/std` est en commentaire). `harmonia/musx.py:
frame_posteriors` (`song_cqt`) chdir dans le clone (`_InMusxDir`) et appelle
CE MÊME chemin. Le réseau (`chordnet_ismir_naive.ChordNet.inference`,
lignes 179-181) NE consomme PAS les 288 colonnes : il tronque
`x[:, 18:270]` (252 colonnes — 18 = `SHIFT_HIGH(6)*SHIFT_STEP(3)`, la marge
de pitch-shift d'augmentation à l'entraînement, inutilisée à l'inférence) et
le passe TEL QUEL (magnitude brute, sans log, sans normalisation externe —
`InstanceNorm2d` est dans le réseau) au CNN+biLSTM. `librosa` 0.11.0
(`.venv`), `res_type` par défaut = `soxr_hq` (jamais surchargé par CQTV2).

**Références** (`.venv/bin/python mobile/tools/cqt_ref_extract.py`, appelle
le vrai clone via `harmonia.musx._InMusxDir`, sur les 3 morceaux de
`docs/audio/`) → `mobile/tools/cqt_ref/*.npz` (`y` float32 22050 Hz, `cqt`
(n_frames,288) float32, `net_input` (n_frames,252), `tuning` estimé
0,10 à -0,44 bin — **non nul**, donc bien exercé, CQTV2 passe `tuning=None`
et pas le défaut librosa `0.0`).

**Lecture de `librosa/core/constantq.py` (1500 lignes) et `librosa/filters.py`** :
`hybrid_cqt` sépare les bins en pseudo-CQT (STFT+banque de filtres, un seul
FFT, pas de resampling) pour ceux dont `2^ceil(log2(longueur_filtre)) <
2*hop_length`, et CQT récursive (Schörkhuber & Klapuri) pour le reste — split
mesuré **50 bins pseudo (indices 238-287, ~1,4 octave aiguë) / 238 bins
récursifs (indices 0-237, tout le reste)**, constant sur les 3 morceaux
malgré la variation de tuning. `gamma=0` partout (jamais le mode ERB).
Filtres construits en numpy pur (`phasor` = `exp(1j*angle)`, fenêtre Hann
PÉRIODIQUE, normalisation L1) — aucune dépendance scipy/numba dans CETTE
partie, portée telle quelle.

**Port** : `mobile/harmonia_mobile/cqt_musx.py` — `cqt_musx(y, sr=22050)` →
`(n_frames,252)` float32, `song_cqt_288()` pour les 288 brutes. STFT maison
(framing + `np.fft.rfft`, center=True, zero-pad, comme `librosa.stft`).
Resampling 2x (soit dans la boucle par octave, soit dans
`__early_downsample`) : filtre FIR demi-bande Kaiser-sinc (65 coefficients,
β=10, coupure 0,25 cycle/échantillon) — PAS soxr. `estimate_tuning` (piptrack
+ interpolation parabolique + histogramme) porté aussi, vérifié exact à
1e-8 près contre la référence sur les 3 morceaux.

**Diagnostic qui a trouvé le bug** (le vrai contenu de cette tâche) :
d'abord suspecté le resampler (comme annoncé dans le brief). Test de
contrôle décisif : remplacer mon filtre par `librosa.resample(...,
res_type='soxr_hq')` (le VRAI moteur soxr) dans mon pipeline → **écart
identique** (0,05643 avant, 0,05643 après, même position exacte
frame=7232, bin=185) → le resampler n'était pas la cause. Isolé ensuite en
comparant, bin par bin, MA banque de filtres et MON STFT contre les
fonctions privées réelles de librosa (`__vqt_filter_fft`, `stft`) sur
l'octave la plus aiguë de la branche récursive (i=0, AUCUN resampling) :
banque de filtres et STFT matchaient à 1e-7 — donc le bug n'était dans
NI L'UN NI L'AUTRE. Trouvé en recalculant `__early_downsample_count`
pour le VRAI appel interne (`cqt(fmin=fmin_global, n_bins=238,
bins_per_octave=36)`, pas le split à 288 bins fait en amont par
`hybrid_cqt` pour décider pseudo/récursif) : `cutoff≈2,2 kHz` (pas 5,9 kHz
comme pour 288 bins — le cutoff dépend de la fréquence MAX du sous-ensemble,
et 238 bins n'atteint pas aussi haut que 288) → `early_downsample_count=1`,
PAS 0. Un sous-échantillonnage 2x était donc appliqué par librosa AVANT
la boucle par octave, et mon port ne le faisait pas — ajouté
(`_early_downsample`, appelle le même filtre demi-bande), écart divisé par
~200.

**Parité et vitesse par morceau** (`.venv/bin/python
mobile/tools/parity_cqt_musx.py`, argmax = triade musx, 73 classes,
moyenne des 5 folds comme `harmonia.musx.posteriors_from_cqt`) :

| morceau | durée | écart max features (relatif) | argmax accord identique | écart max postérieures | temps CQT (réel) | temps normalisé 4 min |
|---|---:|---:|---:|---:|---:|---:|
| maroon_5_this_love | 205,3 s | 3,45e-05 | **100,00 %** | 0,00008 | 2,03 s | 2,37 s |
| ben_e_king_stand_by_me_audio | 177,5 s | 2,62e-05 | **100,00 %** | 0,00013 | 1,73 s | 2,34 s |
| autumn_leaves (jazz) | 422,3 s | 3,50e-05 | **99,99 %** | 0,00008 | 5,08 s | 2,89 s |
| 4 min réels (extrait autumn_leaves) | 240,0 s | — | — | — | **2,23 s** | — |

(« temps CQT » = pipeline complet avec estimation de tuning incluse, propre
tuning contre celui de référence donne les MÊMES scores d'argmax à chaque
fois — la tuning estimée matche la référence à 1e-8 près.)

**Cible vitesse ≤5 s pour 4 min sur Mac : tenue avec ~2x de marge.** Budget
téléphone total = 2 min pour tout le morceau (T2b: NNLS 2,99 s ; T1: réseaux
torch, hors scope ici) — le CQT n'est clairement pas le goulot.

## Ce qui reste ouvert

- **Pas du soxr bit-exact** — le filtre demi-bande Kaiser-sinc maison
  n'est PAS soxr, mais le test de contrôle prouve que ça n'a AUCUN effet
  mesurable une fois le vrai bug (early-downsample manquant) corrigé :
  substituer le vrai soxr donnait un résultat identique au mien, preuve
  que mon filtre demi-bande est déjà largement assez bon pour cette tâche.
  Pas de raison de le complexifier.
- **Pas testé sur téléphone** — ces temps sont Mac, un seul thread numpy,
  comme T2b. `np.convolve` (mode `valid`) pour le filtre demi-bande n'est
  pas vectorisé en FFT batchée comme le reste (T2b l'a fait pour NNLS) —
  si le budget de 2 min sur téléphone est serré une fois T1+T2+ce module
  assemblés, c'est le premier endroit à optimiser (mais actuellement ~1 %
  du budget total, pas prioritaire).
- **Sparsify_rows en dense, pas creux** — je reproduis la sparsification
  (perte volontaire de queue d'énergie <1 %) mais fais le produit matriciel
  en dense (`fft_basis.dot(D)`), pas via `scipy.sparse` — mathématiquement
  équivalent, juste sans le gain mémoire/vitesse de la sparsité (non
  nécessaire ici, la taille des matrices est petite).
- **Numba/scipy hors de cette brique mais pas hors du portage** —
  `librosa` elle-même importe `numba` (`from numba import jit`, en tête de
  `constantq.py`, pour `__num_two_factors`) — sans incidence sur CE port
  (je l'ai réécrit sans numba), mais un rappel que `librosa` entière reste
  interdite sur Android (déjà noté en T3).
- **Un seul type de tuning testé** (tuning global, pas de variation
  intra-morceau) — les 3 morceaux ont un tuning quasi-standard (0,10 à
  -0,44 bin sur 36/octave, donc <2 % de décalage fréquentiel) ; un morceau
  fortement désaccordé n'a pas été testé.

## Fichiers produits

- `mobile/harmonia_mobile/cqt_musx.py` — le port (STFT, banque de filtres
  CQ, pseudo-CQT, boucle récursive par octave + `_early_downsample`,
  resampler demi-bande Kaiser-sinc, `estimate_tuning`). Entrée `cqt_musx(y,
  sr=22050)` → `(n_frames,252)` float32 (le tenseur réseau) ;
  `song_cqt_288(y, sr=22050)` → `(n_frames,288)` float32 (le CQT complet,
  utile pour du debug/cache comme `harmonia.musx.song_cqt`).
- `mobile/tools/cqt_ref_extract.py` — dump des références (`y`, `cqt`
  288-bin, `net_input` 252-bin, `tuning`) via le vrai clone, chdir compris.
- `mobile/tools/cqt_ref/*.npz` — les 3 références (maroon_5_this_love,
  ben_e_king_stand_by_me_audio, autumn_leaves).
- `mobile/tools/parity_cqt_musx.py` — script de validation (écart features,
  argmax accord via les vrais réseaux torch musx, vitesse), table ci-dessus.
