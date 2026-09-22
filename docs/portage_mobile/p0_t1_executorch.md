# P0-T1 — ExecuTorch : `beat_this` et `musx` s'exportent-ils, mêmes sorties que torch ?

## Résultat en 3 lignes

`beat_this` (small0, 9 Mo) exporte en `.pte`, tourne dans le runtime Python
ExecuTorch, et rend les MÊMES battues que torch (max|Δ logit|=5.6e-5, battues
décodées identiques à 0 ms sur 47 battues) — **feu vert**. `musx` (fold 0,
CNN+LSTM) bloque : `torch.export()` réussit mais le passage XNNPACK
(`to_edge_transform_and_lower`) tourne plus de 27 min à 98 % CPU sans sortir
ni planter — tué au-delà du budget de 45 min ; cause suspectée : la LSTM
bidirectionnelle (voir §2). Dans les deux cas, le prétraitement audio→tenseur
(mel pour `beat_this`, CQT pour `musx`) est HORS du graphe exportable — à
refaire en numpy pur pour Android (tableau ci-dessous).

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Environnement** : `~/harmonia/.venv-mobile` créé depuis
`/Users/vincente/miniforge3/bin/python3.12` (base interpreter, pas un venv
existant). `pip install executorch` → torch 2.14.0 + executorch 1.5.1 +
torchao 0.18.0. `.venv` du worktree non touché. Disque avant/après :
7,5 Go → 11 Go libres sur `/System/Volumes/Data` (jamais descendu sous 5,2 Go,
seuil d'arrêt = 3 Go, jamais atteint). `pip install torch torchaudio soxr
einops rotary_embedding_torch beat_this soundfile` (mobile venv uniquement).
Pour charger un fold `musx`, la chaîne d'imports du paquet vendu `mir/`
force `pretty_midi`, `librosa`, `pydub`, `h5py` — installés dans le venv
mobile SEULEMENT (outillage Mac, pas la cible Android ; la règle PLAN
"interdit sur le téléphone" porte sur l'APK, pas sur ce venv de build).

**beat_this — chaîne exacte** (lu dans `harmonia/beats.py` et
`.venv/lib/python3.12/site-packages/beat_this/{inference,preprocessing}.py`) :
`File2Beats(device="cpu", dbn=False)` → `load_audio` (torchaudio/soundfile/
madmom) → mono → resample à 22050 Hz via **soxr** (C, pas torch) →
`LogMelSpect` (`torchaudio.transforms.MelSpectrogram`, n_fft=1024,
hop=441, 128 mels, Slaney) → `log1p(1000·x)` → découpage en fenêtres FIXES de
1500 frames (30 s, bord 6 frames, exactement `split_predict_aggregate` déjà
dans le code — pas une chose à inventer pour mobile) → réseau `BeatThis`
(stem conv + 3 blocs frontend + 6 blocs RoFormer, `rotary_embedding_torch`)
→ logits beat/downbeat → `Postprocessor("minimal")` (maxpool 1D + seuil +
dédup, pur torch/numpy).

**musx — chaîne exacte** (lu dans `harmonia/musx.py` et
`third_party/musx_ismir2019/{chordnet_ismir_naive,chord_recognition}.py`,
`extractors/cqt.py`) : audio → `librosa.core.hybrid_cqt` (36 bins/octave,
288 bins, hop=512, sr=22050) → `ChordNet.inference(cqt)` : crop centré
288→252 bins (`x[:,18:270]`, fenêtre du pitch-shift training) →
`CNNFeatureExtractor` (8 conv2d + InstanceNorm2d + SELU + maxpool, réduit
252→3×80) → `nn.LSTM` bidirectionnelle (192) → `Linear` → split 6 têtes →
softmax par tête. Le décodeur beat-aware en aval (`extractors/xhmm_ismir.py`,
`make_beat_arr`) est pur numpy — portable tel quel, aucun souci.

**Export beat_this** (`mobile/tools/export_beat_this.py`) :
- Front-end `LogMelSpect` seul : `torch.export()` RÉUSSIT (capture le
  `torch.stft`), mais `to_edge_transform_and_lower` + XNNPACK ÉCHOUE :
  `SpecViolationError` sur `aten.view_copy`/`aten.abs` en `complex64`
  (XNNPACK ne supporte pas les tenseurs complexes que produit la STFT).
  **Le mel-spectrogram ne peut pas aller dans le graphe exporté.**
- Réseau `BeatThis` seul, entrée = mel-spect `(1, 1500, 128)` :
  forme fixe → export OK (5,4 s export + 71-94 s lowering, `.pte` 9,77 Mo).
  Forme dynamique `time∈[8,1500]` → export OK aussi (`.pte` 9,83 Mo,
  `Dim("time", min=8, max=1500)` — au-delà de 1500 le tracer se spécialise,
  donc la fenêtre de 30 s est une limite dure de CE réseau, pas un choix
  arbitraire : coller à `split_predict_aggregate` est obligatoire).

**Parité beat_this** (`mobile/tools/parity.py`, runtime Python
`executorch.runtime.Runtime`, sur un extrait 30 s de
`maroon_5_this_love.wav`, forme fixe ET dynamique) :
max|Δ logit beat|=5,6e-5, max|Δ logit downbeat|=5,4e-5 (cible 1e-3, marge
×18) ; décodage `Postprocessor` : 47/47 battues communes, **Δt max = 0,00 ms**
sur les deux formes.

**Vitesse beat_this** (Mac, 1 thread `torch.set_num_threads(1)`, 5 runs après
warmup, chunk 1500 frames = 30 s) : 3198 ms ± moyenne (2796–3989 ms) par
chunk → **≈22,4 s de calcul réseau pour un morceau de 205 s** (7 chunks),
Mac ≠ téléphone, aucune conclusion sur le critère "< 2 min pour 4 min" (T5).

**Export musx** (`mobile/tools/export_musx.py`, fold 0, CQT pré-calculée du
cache disque `data/cache/musx_cqt/maroon_5_this_love.npz`, fenêtre 500
frames) : `torch.export()` RÉUSSIT (avec un avertissement : les poids LSTM
`_flat_weights` sont assignés hors `register_buffer`, signalé par torch
lui-même comme fragile pour l'export). `to_edge_transform_and_lower`
(partitionneur XNNPACK) a tourné **27 min 30 s à 98 % CPU, 0 octet de
sortie, mémoire stable (~900 Mo, pas de fuite)**, sans erreur ni résultat —
tué au process 4893 après dépassement du budget de 45 min. Aucun `.pte`
musx produit.

## Ce qui reste ouvert

- **musx bloque précisément à la lowering XNNPACK, pas à `torch.export`.**
  Cause la plus probable (non confirmée, faute de temps) : la `nn.LSTM`
  bidirectionnelle — le partitionneur XNNPACK décompose les LSTM en
  primitives et l'explosion combinatoire ou une boucle de contraintes lente
  est un problème XNNPACK/ExecuTorch connu sur ce genre de couche. Prochaine
  étape : réessayer SANS partitionneur XNNPACK (portable CPU delegate) pour
  voir si `to_edge` seul passe vite — si oui, isoler la LSTM en la import-
  ant seule ; si le blocage est le CNN (InstanceNorm2d), le confirmer pareil.
- Le warning `_flat_weights` sur la LSTM peut aussi signaler un risque de
  CORRECTION silencieuse (poids non retrouvés au retracé) — à vérifier avant
  de refaire confiance à un export qui "marcherait".
- Aucune parité ni vitesse mesurée pour musx — bloqué en amont.
- Le CQT (`librosa.hybrid_cqt`) n'a AUCUNE implémentation torch ou numpy
  pure ici — c'est un chantier séparé, pas mesuré dans ce sprint (proche de
  T2 nnls-chroma en nature, mais pas le même composant).
- `torchaudio` publié (2.11.0) est en retard sur `torch` (2.14.0) et son
  I/O par défaut requiert maintenant `torchcodec` (absent) — seul
  `soundfile` a marché pour charger le wav de référence ; sans incidence sur
  l'export (le mel-spectrogram, lui, a marché), mais à surveiller si
  `torchaudio` est réutilisé ailleurs dans l'outillage mobile.
- Vitesse mesurée sur Mac seulement (single-thread) : 22,4 s/chanson pour le
  réseau beat_this seul ; le CQT+ChordNet musx, le NNLS et le reste du
  pipeline restent à chiffrer — aucune conclusion sur le critère T5.

## Fichiers produits

- `mobile/tools/export_beat_this.py`, `mobile/tools/export_musx.py`,
  `mobile/tools/parity.py`
- `mobile/models/beat_this_small0_fixed1500.pte` (9,77 Mo),
  `mobile/models/beat_this_small0_dynamic.pte` (9,83 Mo),
  `mobile/models/beat_this_reference.npz` (tenseurs de référence torch)
- `~/harmonia/.venv-mobile` (nouveau venv, torch+executorch+beat_this+
  librosa/pretty_midi/pydub/h5py pour charger les checkpoints musx — outil de
  build Mac, jamais embarqué)
- Ce fichier : `docs/portage_mobile/p0_t1_executorch.md`
