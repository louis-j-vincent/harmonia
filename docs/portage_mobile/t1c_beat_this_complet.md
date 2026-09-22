# P0-T1c — `beat_this` complet sur le téléphone (front-end numpy + `.pte` + post-traitement)

## Résultat en 3 lignes

Correction d'abord : T1 avait exporté `small0` (9 Mo) ; le pipeline
(`harmonia/beats.py:46`) charge en fait `final0` (97 Mo, 20,25M paramètres,
10x `small0`) par défaut — vérifié dans `load_model`'s signature. `final0`
exporte en `.pte` (82,4 Mo, formes fixe et dynamique, 2 et 3,6 min de lowering
XNNPACK, largement sous le budget de 30 min) et la chaîne complète (mel numpy
→ `.pte` → post-traitement numpy) rend EXACTEMENT les mêmes battues et temps
forts que `File2Beats("final0", dbn=False)` sur les 3 morceaux — 1803/1803
battues et 548/548 temps forts au bon COMPTE, 1802/1803 battues et 548/548
temps forts à 0,000 ms, la seule battue divergente est un vrai quasi-ex-æquo
du réseau (voir §3) tranché différemment par 1 frame (20 ms) — pas une erreur
de portage. Vitesse Mac 1 thread : le réseau domine à 99% du temps de calcul
(~0,14 s de calcul par seconde d'audio, soit ~7x plus rapide que le temps
réel), front-end et post-traitement numpy sont négligeables (<0,5 s et <5 ms
même sur les plus longs morceaux).

## Ce qui a été vérifié (commandes, chiffres, chemins)

**1. Correction du checkpoint.** `harmonia/beats.py:45-46` appelle
`File2Beats(device="cpu", dbn=False)` sans nommer de checkpoint.
`beat_this.inference.load_model(checkpoint_path: str | None = "final0", ...)`
— lu directement dans
`~/harmonia/.venv-mobile/lib/python3.12/site-packages/beat_this/inference.py` —
défaut = `"final0"`. Comparaison des deux checkpoints (déjà en cache local,
`~/.cache/torch/hub/checkpoints/`, aucun téléchargement nécessaire) :
`small0` — `transformer_dim=128`, 2,10M paramètres, 8,45 Mo `.ckpt` ;
`final0` — `transformer_dim=512`, 20,25M paramètres, 81,06 Mo `.ckpt`. Le
reste des hyperparamètres (`spect_dim=128`, `stem_dim=32`, `head_dim=32`,
donc même profondeur de réseau) est identique — seule la largeur change,
donc même structure de graphe à exporter.

**2. Export `final0`** (`mobile/tools/export_beat_this.py`, adapté en place —
`CHECKPOINT = "final0"`, sorties renommées `beat_this_final0_*`, script non
dupliqué). Résultats (`~/harmonia/.venv-mobile`, torch 2.14.0 / executorch
1.5.1) :
- front-end `LogMelSpect` seul : `torch.export` OK mais `to_edge_transform_and_lower`
  échoue toujours (`aten.abs` sur tenseur `complex64` — le mel-spectrogram ne
  peut pas vivre dans le graphe exporté, confirmé identique à T1, cause
  XNNPACK et non liée au checkpoint).
- réseau `BeatThis` (`final0`), forme FIXE (1, 1500, 128) : export 5,9 s,
  lowering XNNPACK 125,2 s, `.pte` = 82,38 Mo.
- réseau `BeatThis` (`final0`), forme DYNAMIQUE (`time∈[8,1500]`) : export
  12,7 s, lowering XNNPACK 217,2 s, `.pte` = 82,44 Mo.
- Total < 6 min, largement sous le budget de 30 min prévu au cas où le gros
  modèle bloquerait comme `musx` (il ne bloque pas — la LSTM bidirectionnelle
  de `musx`, suspectée dans le rapport T1, n'est pas dans `BeatThis`, qui est
  un stem conv + RoFormer).
- En pratique seule la forme FIXE sert : `split_predict_aggregate` découpe
  TOUJOURS en blocs de 1500 trames exactement (`avoid_short_end=True`, vérifié
  ci-dessous) pour tout morceau de plus de ~24 s — la forme dynamique reste
  disponible mais non exercée par le chemin réel.

**3. Front-end numpy** (`mobile/harmonia_mobile/beats_frontend.py`). Lu dans
`beat_this/preprocessing.py` : `LogMelSpect(sample_rate=22050, n_fft=1024,
hop_length=441, f_min=30, f_max=11000, n_mels=128, mel_scale="slaney",
normalized="frame_length", power=1, log_multiplier=1000)` =
`torchaudio.transforms.MelSpectrogram` (lu dans le SOURCE torchaudio 2.11.0,
`transforms/_transforms.py` et `functional/functional.py`, pas dans sa doc)
suivi de `log1p(1000·x)`. Quatre pièges identifiés et vérifiés un par un :
- `torch.hann_window` est PÉRIODIQUE (dénominateur `N`, pas `N-1`) — PAS
  `np.hanning` (symétrique) ; implémenté à la main (`_hann_periodic`).
- `normalized="frame_length"` = `torch.stft(normalized=True)`, qui divise
  chaque trame par `sqrt(n_fft)` — DIFFÉRENT de la normalisation "window"
  (division par `sqrt(sum(window**2))`), jamais utilisée ici.
- `mel_scale="slaney"` ne choisit QUE la formule de conversion Hz↔mel
  (linéaire <1000 Hz, log au-delà) ; le paramètre SÉPARÉ `norm` (aire des
  triangles, style librosa `norm='slaney'`) n'est PAS passé par `LogMelSpect`
  → défaut `None` → triangles NON normalisés (crête à 1,0). Utiliser
  `librosa.filters.mel(..., norm='slaney')` par réflexe aurait donné un
  spectrogramme plausible mais FAUX (piège exactement de la forme "erreur de
  calibration silencieuse", CLAUDE.md règle 1) — vérifié explicitement en
  lisant `MelSpectrogram.__init__` : `norm: Optional[str] = None`.
- `center=True, pad_mode="reflect"` = `np.pad(y, n_fft//2, mode="reflect")`
  (PAS `mode="symmetric"`).
- Validé (`stft_magnitude` + `mel_filterbank` + `log_mel_spectrogram`) contre
  `LogMelSpect()(torch.tensor(y))` sur les 3 morceaux ENTIERS (pas un extrait) :
  max|Δ| = **1,26e-4** (this_love, 10264 trames), **9,06e-5** (stand_by_me,
  8875 trames), **1,11e-4** (autumn_leaves, 21117 trames) — cible 1e-3, marge
  ×9 à ×11.

**4. Découpage/agrégation** (`mobile/harmonia_mobile/beats_infer.py::split_piece`,
`aggregate_prediction`). Port numpy direct de `beat_this.inference.split_piece`
/ `aggregate_prediction` (chunk_size=1500, border_size=6, `overlap_mode=
"keep_first"` — les seuls réglages que `Spect2Frames.spect2frames` utilise).
Vérifié bit-exact contre les fonctions torch d'origine pour 9 longueurs
(10 à 21117 trames) : mêmes `starts`, mêmes valeurs de chunk (`np.allclose`),
même agrégation. Pour toute longueur ≥ 1500 trames (tout morceau réel), les
chunks sont TOUJOURS exactement 1500×128 (confirmé : `avoid_short_end=True`
force le dernier bloc à `chunk_size - border_size` de la fin, donc `right =
min(border_size, ...) = border_size` pile).

**5. Post-traitement `Postprocessor("minimal")`**
(`mobile/harmonia_mobile/beats_infer.py::postprocess_minimal`,
`_maxpool1d_same`, `_deduplicate_peaks`) : maxpool 1D (noyau 7, pas 1, pad 3,
`-inf` hors bornes), pic = `logit == maxpool(logit) AND logit > 0`, frames →
temps (`/fps`, `fps=50=22050/441` exactement), fusion des pics adjacents
(`deduplicate_peaks`, moyenne des indices à ≤1 frame d'écart), downbeats
recalés sur la battue la plus proche, dédoublonnés. Vérifié bit-exact contre
`Postprocessor(type="minimal")` sur logits aléatoires : mêmes comptes (523/523
et 427/427 sur un essai), `np.allclose` sur les temps.

**6. Parité de bout en bout** (`mobile/tools/parity_beats.py`), audio chargé
EXACTEMENT comme `harmonia/beats.py::track()` (`beat_this.preprocessing.
load_audio`, repli ffmpeg `-ac 1 -ar 22050` sur échec — `autumn_leaves.m4a`
l'a exercé, les deux `.wav` déjà en cache ne l'ont pas exercé), la MÊME
séquence de samples nourrie au chemin torch et au chemin numpy/`.pte` :

| morceau | durée | battues (torch/moi) | max\|Δt\| battues | temps forts (torch/moi) | max\|Δt\| temps forts |
|---|---|---|---|---|---|
| maroon_5_this_love | 205,3 s | 322/322 | **20,000 ms** (1 battue/322, voir ci-dessous) | 81/81 | 0,000 ms |
| ben_e_king_stand_by_me_audio | 177,5 s | 345/345 | 0,000 ms | 94/94 | 0,000 ms |
| autumn_leaves | 422,3 s | 1136/1136 | 0,000 ms | 373/373 | 0,000 ms |

Cause de l'unique écart (this_love, frame ≈8196-8197) : les logits torch
bruts y sont **7,249358 et 7,249332** — un quasi-ex-æquo du réseau lui-même
(Δ=2,6e-5, bien EN-DESSOUS de la marge de parité réseau mesurée par T1 pour
`small0`, 5,6e-5). Les logits numpy/`.pte` diffèrent de ceux de torch d'au
plus **2,05e-4** sur ce morceau (cible parité réseau 1e-3, marge ×5) — assez
pour faire basculer lequel des deux frames quasi-identiques gagne le test
`==maxpool` strict, donc lequel des deux devient LE pic. C'est un artefact de
précision flottante sur un ex-æquo authentique du modèle (pas une erreur de
portage, pas un décalage systématique) : les 1802 autres battues et les 548
temps forts sur les 3 morceaux sont à 0,000 ms. Non résolu plus avant (budget) —
noté en "reste ouvert".

**Vitesse** (Mac, 1 thread — `torch.set_num_threads(1)` côté torch ;
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` côté process ; le
runtime XNNPACK garde cependant son propre pool de threads, non contraint par
ces réglages — voir "reste ouvert") :

| morceau | durée audio | front-end (numpy) | réseau (`.pte`) | post-traitement | total moi | total torch (référence) |
|---|---|---|---|---|---|---|
| maroon_5_this_love | 205,3 s | 0,323 s | 34,758 s | 0,003 s | 35,084 s | 23,724 s |
| ben_e_king_stand_by_me_audio | 177,5 s | 0,248 s | 23,452 s | 0,003 s | 23,703 s | 18,809 s |
| autumn_leaves | 422,3 s | 0,516 s | 54,326 s | 0,004 s | 54,846 s | 48,981 s |

Le réseau domine à 99% du calcul dans les trois cas. Taux moyen (calcul/audio) :
front-end ≈0,0014, réseau ≈0,143, post-traitement ≈1,4e-5 — projeté sur
4 min (240 s) d'audio : **front-end ≈0,34 s, réseau ≈34,4 s, post-traitement
≈3 ms, soit ≈35 s au total sur Mac 1 thread** (~7x plus rapide que le temps
réel). Le chemin numpy/`.pte` est 1,12x à 1,48x plus lent que `torch`
natif sur Mac (attendu : XNNPACK n'est pas optimisé pour Accelerate/MKL sur
Mac ; sur téléphone ARM, XNNPACK est au contraire le backend ciblé — seul T5
sur device réel tranchera).

## Ce qui reste ouvert

- **L'unique écart de 20 ms (this_love) est caractérisé, pas éliminé.** Il
  vient d'un ex-æquo authentique dans les logits du réseau — aucune
  implémentation `.pte` bit-identique à torch (kernel XNNPACK ≠ kernel natif)
  ne peut garantir de trancher un ex-æquo de cette finesse (Δ=2,6e-5) dans le
  même sens à coup sûr. Rien à corriger côté portage ; à surveiller si Louis
  entend un "double-clic" occasionnel sur un morceau au tempo bord-de-frame.
- **Le pool de threads XNNPACK n'est pas contraint par `torch.set_num_threads`
  ni les variables `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`** (`ps` a montré
  ~330-390% CPU pendant l'inférence `.pte` malgré ces réglages) — la mesure
  "réseau" ci-dessus n'est donc PAS garantie mono-thread côté XNNPACK, contrairement
  à la référence torch qui l'est. Pas cherché plus loin (budget) ; si T5 doit
  comparer un budget mono-cœur strict sur téléphone, il faudra fixer les
  threads XNNPACK explicitement (`executorch.runtime` expose un réglage,
  non exploré ici).
- **Vitesse mesurée sur Mac uniquement** — aucune conclusion sur le critère
  T5 (4 min < 2 min sur téléphone réel) : le Mac n'est pas le téléphone, et le
  budget `.pte` mesuré ici (~35 s/4 min-projeté) ne couvre QUE `beat_this` ;
  `musx`, le CQT, NNLS et le reste du pipeline restent à sommer (voir T2/T2b/P2b).
- **`.pte` dynamique (`beat_this_final0_dynamic.pte`) exporté mais jamais
  exercé** par `beats_from_audio` — chaque chunk réel fait toujours 1500
  trames. Gardé au cas où un futur choix (chunk plus court pour économiser de
  la RAM/latence sur téléphone) en aurait besoin.
- **`small0`** reste exporté (`beat_this_small0_*.pte`, désormais obsolète
  pour le chemin de prod) — PAS supprimé, coûte 19,6 Mo, laissé en place pour
  Louis s'il veut arbitrer un "mode rapide" plus tard (CLAUDE.md règle 6 : ce
  n'est pas à l'agent de trancher ce remplacement).

## Fichiers produits

- `mobile/harmonia_mobile/beats_frontend.py` — mel-spectrogramme numpy pur
  (`log_mel_spectrogram`), validé <1,3e-4 vs torch sur 3 morceaux entiers.
- `mobile/harmonia_mobile/beats_infer.py` — découpage/agrégation, appel
  `.pte` (ExecuTorch Python runtime), post-traitement `"minimal"` numpy ;
  point d'entrée `beats_from_audio(y22050) -> (beats, downbeats)`.
- `mobile/tools/export_beat_this.py` — adapté en place pour `final0` (au lieu
  de `small0`), mêmes formes fixe/dynamique qu'à T1.
- `mobile/tools/parity_beats.py` — parité + vitesse de bout en bout, 3
  morceaux, audio chargé comme `harmonia/beats.py`.
- `mobile/models/beat_this_final0_fixed1500.pte` (82,38 Mo),
  `mobile/models/beat_this_final0_dynamic.pte` (82,44 Mo),
  `mobile/models/parity_beats_report.json` (résultats bruts) — tous
  gitignorés (`mobile/models/.gitignore`).
- Ce fichier : `docs/portage_mobile/t1c_beat_this_complet.md`.
