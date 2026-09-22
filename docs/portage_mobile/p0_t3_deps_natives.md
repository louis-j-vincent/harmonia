# P0-T3 — dépendances natives du chemin critique mobile, et leur remplaçant

## Résultat en 3 lignes

18 sites d'appel natif sur le chemin critique (routes `library`+`analyze` +
`server/jobs.py` + `analyze_steps`) ; TOUS les décodages audio (6 sites, 3
taux différents : 22050 pour battues+musx, 44100 pour nnls-chroma, 44100 pour
le mic/upload) passent par `ffmpeg` en sous-processus ou `librosa.load` — le
vrai risque n'est pas leur remplacement un par un (Java `MediaCodec`, mécanique)
mais que les **trois** redécodages retombent bit-pour-bit sur le même SR/mono/
premier-échantillon que `ffmpeg`, faute de quoi les accords glissent par
rapport aux temps (règle n°1 CLAUDE.md). SongFormer (torch, le module le plus
craint) est déjà ÉTEINT pour ce chemin — `SETTINGS.sections="aucune"`
(`settings.py:91`) — donc **hors scope T3 pour le v1**, comme le CQT front-end
de musx (`librosa.core.hybrid_cqt`), qui lui reste un vrai trou : ni T1 ni T2
ne le couvrent aujourd'hui. Chaquopy 17.0 sert encore le dépôt `pypi-13.1`
(confirmé, inchangé depuis #808) : `scipy` (1.8.1, cp3.10 max), `numpy`
(1.26.2, jusqu'à cp3.13), `soxr` (0.3.6, cp3.8 seulement) et `h5py` (2.10.0,
cp3.8 seulement) y sont, `librosa`/`vamp`/`torchaudio`/`pumpp`/`jams` n'y sont
pas — `torch` y est mais à 1.8.1/cp3.8, pas 1.13, et hors-sujet puisque le
plan remplace torch entier par ExecuTorch.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Index Chaquopy** — le dépôt de wheels natives n'a pas changé de version
depuis Chaquopy #808 (changelog : « All Android wheels are now downloaded
from https://chaquo.com/pypi-13.1/ — the old pypi-7.0 URL is no longer used »),
donc `pypi-13.1` EST le dépôt courant, pas un dépôt périmé — vérifié en
listant `https://chaquo.com/pypi-13.1/<pkg>/` pour chaque nom :

| Paquet | Dans l'index ? | Version(s) max | Python max |
|---|:-:|---|---|
| `librosa` | non | — | — |
| `vamp` | non | — | — |
| `torchaudio` | non | — | — |
| `pumpp` | non | — | — |
| `jams` | non | — | — |
| `resampy`, `audioread`, `yt-dlp`, `curl_cffi`, `mir_eval` | non | — | — |
| `scipy` | **oui** | 1.8.1 | cp3.10 (pas de wheel cp3.11/3.12/3.13) |
| `numpy` | **oui** | 1.26.2 | cp3.13 |
| `torch` | **oui** | **1.8.1**, pas 1.13 comme supposé dans la commande | cp3.8 seulement |
| `soundfile` | **oui** | 0.13.1 | cp3.10 |
| `h5py` | **oui** | 2.10.0 | cp3.8 seulement |
| `scikit-learn` | **oui** | 0.23.2 | cp3.8 seulement |
| `matplotlib` | **oui** | 3.6.0 | cp3.8 seulement |
| `numba`/`llvmlite` | **oui** | 0.48.0 / 0.31.0 | cp3.8 seulement |
| `soxr` | **oui** | 0.3.6 | cp3.8 seulement |

Conséquence directe : si on veut `soxr` (le resampler exact de Beat This!,
`beat_this/inference.py:275`) et/ou `h5py` tel quel, Android est plafonné à
Python 3.8 — sinon (comme le plan le prévoit, tout le calcul lourd part en
ExecuTorch) rien de tout ça n'a besoin d'être installé sur le téléphone.

**Grep exhaustif** (imports top-level ET en milieu de fonction,
`^\s*(import|from)\s+lib`) sur les 23 fichiers du minimum vital listés par
l'audit — table triée par risque décroissant :

| fichier:ligne | appel | ce que ça fait | remplaçant | lignes | risque |
|---|---|---|---|---:|:-:|
| `harmonia/beats.py:945-948` (fallback de `f2b()`, Beat This! lui-même via `torchaudio`/`soundfile`/`madmom`) | `subprocess.run(["ffmpeg",...,-ac 1,-ar 22050])` puis, en interne à Beat This!, `soxr.resample` si `sr≠22050` (`beat_this/inference.py:274-275`) | décode + mono + resample 22050 Hz, LE SIGNAL QUI DONNE LES TEMPS | Java `MediaCodec`→PCM float32 puis resample 22050 (même contrat que musx, §4) | ~80 (décodeur MediaCodec réutilisable) | **Élevé** — tout glissement d'échantillon ici désaligne CHAQUE accord du CHAQUE beat |
| `third_party/musx_ismir2019/mir/io/implement/music_io.py:6` (`MusicIO.read`, appelé par `musx.py:322-328`) | `librosa.load(filename, sr=22050, mono=True)` | décode + mono + resample 22050 Hz pour le front-end CQT des accords | même décodeur Java que ci-dessus, sortie partagée avec beats.py | 0 (réutilise le même) | **Élevé** — alimente la grille de frames `FRAME_DT=512/22050=23.22 ms` sur laquelle `make_beat_arr` (`musx.py:102`) fait un `np.round`, donc un décalage de phase silencieux |
| `third_party/musx_ismir2019/extractors/cqt.py:52-58` (`CQTV2.extract`) | `librosa.core.hybrid_cqt(y, bins_per_octave=36, fmin=librosa.note_to_hz('F#0'), n_bins=288, hop_length=512)` — CQT hybride (FFT pour l'aigu, vraie CQT récursive+resample pour le grave) | le FRONT-END du réseau d'accords musx | port numpy (FFT + banque de filtres CQT récursive) OU le faire entrer dans le graphe tracé par T1 | 150-300 (aucune des deux options n'est faite aujourd'hui) | **Élevé, ET ouvert** — ni T1 ni T2 ne couvrent ce module ; c'est un TROU du plan |
| `harmonia/nnls_features.py:100` (`extract_bothchroma`) | `librosa.load(str(audio_path), sr=44100, mono=True)` | décode + mono + resample **44100 Hz** (SR différent des deux lignes au-dessus) | même décodeur Java, cible 44100 | 0 (réutilise) | **Élevé** — 3e SR différent sur le même fichier ; si le resampler diffère même légèrement entre 22050 et 44100 côté Android vs Mac, le chroma et les battues ne pointent plus sur les mêmes instants |
| `harmonia/debut.py:410-413` (`_energie_double_fft` ou équivalent, doc : « rien ici ne mérite une bibliothèque ») | `subprocess.run(["ffmpeg",...,-ac 1,-ar str(sr),-f f32le,"-"])` | décode fenêtre mono à un SR paramétrable, pour détecter le VRAI DÉBUT du morceau (ancre la mesure 1) | Java `MediaCodec` + resample `sr` | ~20 | **Élevé** — un décalage ici bouge l'origine de TOUTE la grille de mesures |
| `harmonia/server/routes/analyze.py:154-163,189` (`_ffmpeg()`, appelé sur l'upload micro/fichier) | `subprocess.run(["ffmpeg",...,-ac 1,-ar 44100,-c:a aac])` | normalise l'enregistrement navigateur (webm/opus Chrome, mp4/aac Safari) avant analyse | Java `MediaCodec` (décodage du conteneur WebM/AAC produit par `MediaRecorder` Android) | ~40 | **Élevé** — 4e chemin de décodage, conteneurs variables, jamais mesuré sur Android |
| `harmonia/pipeline.py:406-409` (`_head_chart`) | `subprocess.run(["ffmpeg","-t",HEAD_SECONDS,...,-ac 1,-ar 22050])` | tronque + mono + resample les ~45 premières secondes pour l'aperçu affiché en premier | Java `MediaCodec`, même contrat que beats.py | 0 (réutilise) | Moyen — un aperçu qui diverge du chart final est trompeur mais pas silencieusement faux (il est explicitement marqué provisoire, `pipeline.py:378-395`) |
| `harmonia/nnls_features.py:101` (`vamp.collect`) | plugin natif VAMP `nnls-chroma:nnls-chroma`, sortie `bothchroma` (24-dim) | le chroma NNLS lui-même (pas le décodage) | port numpy déjà en cours (**T2**, brique séparée) | — (propriété de T2) | Moyen — risque porté par T2, pas T3 |
| `harmonia/musx.py:204,212,262` (`_device`, `_patch_init_hidden`, `_run_nets`) | `import torch` ; 5 checkpoints CNN+biLSTM + décodeur Viterbi (`XHMMDecoder`) | inférence des postérieures d'accords | export ExecuTorch (**T1**, brique séparée) | — (propriété de T1) | Moyen — risque porté par T1 |
| `harmonia/pipeline.py:361-372` (`duration_seconds`) et `harmonia/server/jobs.py:191-193` (doublon, cf. docstring `pipeline.py:352-357` qui documente déjà ce doublon comme non résolu) | `subprocess.run/check_output(["ffprobe",...])` | durée du fichier en secondes | `MediaExtractor` : `getTrackFormat(i).getLong(MediaFormat.KEY_DURATION)` | ~15 (×2 call sites, ou 1 si le TODO du sprint 12 est enfin fait) | Faible — cosmétique/affichage, ne touche aucun timing d'accord |
| `harmonia/sections/similarity.py:53` (import top-level) vs `:183` (`_peaks`, seule fonction qui l'utilise) | `from scipy.signal import find_peaks` | pics locaux au-dessus d'un seuil, pour l'outil de re-découpage manuel (Soudure/`section_tool`) — **PAS** pour `halfbar_features` (`:103`), la seule fonction que `folding.py:517` appelle | rien à faire pour le chemin CORE (le module n'a pas besoin d'être réellement appelé) ; si `scipy` gêne (cp3.10 max, cf. tableau), déplacer l'import dans `_peaks()` (1 ligne) ou porter en numpy (~15 lignes, max-local + distance mini) | 1–15 | Faible — l'audit le classait « indispensable » au niveau du FICHIER, mais la fonction scipy-dépendante n'est PAS sur le chemin `folding.py` |
| `harmonia/sections/songformer.py:106,177,202` (`import torch`, self-relaunch `subprocess.run([sys.executable,"-m","harmonia.sections.songformer",...])`, poids téléchargés depuis Hugging Face) | détection automatique de sections (MuQ+MusicFM+Transformer) | **désactivé pour le mobile v1** : `SETTINGS.sections="aucune"` (`settings.py:91`, défaut depuis 2026-09-19) → `pipeline.py:750` prend la branche qui pose UNE section « A », jamais la branche `songformer` (`pipeline.py:766,785`) ; déjà noté dans `JOURNAL.md` (2026-09-22) : « SongFormer est ÉTEINT » | aucun — hors chemin critique mobile tant que `sections="aucune"` tient | 0 | Aucun (pour le v1) |
| `harmonia/nnls_features.py:190-191,217,230` (`NNLS24Heads`) | `import torch` — MLP têtes root/qualité pour un front-end alternatif | mort : `pipeline.py` ne sélectionne jamais `feature_frontend="nnls24"` (confirmé par l'audit, §Mort pour mobile) | suppression pure | 0 | Aucun |
| `harmonia/server/youtube.py:57,116,182,215,259` | `import yt_dlp` ; 4 `subprocess.run/Popen` (métadonnées, serveur PO-Token Node, téléchargement) | import YouTube | déjà tranché par `PLAN.md` : `youtubedl-android` (sideload) | — (hors périmètre T3) | Faible, déjà décidé — mais parité métadonnées PAS vérifiée (voir « ouvert ») |

**`third_party/musx_ismir2019` — imports réellement exécutés** (méthode : trace
`sys.modules` avant/après avoir fait, depuis `third_party/musx_ismir2019`
comme cwd, exactement les imports que `musx.py:frame_posteriors` fait — `mir`,
`mir.io`, `mir.DataEntry`, `extractors.cqt.CQTV2`, `mir.nn.train.NetworkInterface`,
`chordnet_ismir_naive.ChordNet`, `extractors.xhmm_ismir.XHMMDecoder` — sans
appeler leur code, donc sans audio ni poids requis) :

- **`h5py` : présent, mais CONFIRMÉ mort à l'exécution pour l'inférence** —
  il n'entre que parce que `chordnet_ismir_naive.py:4` fait
  `from mir.nn.data_storage import FramedRAMDataStorage,FramedH5DataStorage`
  (import inconditionnel des DEUX classes) ; `frame_posteriors` ne charge que
  les poids `.sdict`, jamais un `.h5d`. Remplaçant : rien à porter, juste
  garder le module importable (ou stub `h5py` si on ne veut pas de la wheel
  cp3.8).
- **`matplotlib`, `jams`, `pumpp` : CONFIRMÉS absents du chemin runtime** —
  ils n'apparaissent que dans `results_ismir2017.py`, `test_for_all.py`,
  `results.py`, `io_new/jams_io.py`, `extractors/ismir2017_chord_extractor.py`
  — aucun de ces fichiers n'est importé par la trace. L'hypothèse de l'audit
  était juste.
- **Surprise non anticipée par l'audit** : `librosa` (confirmé, voir tableau
  ci-dessus), `mido`, `pretty_midi`, `pydub`, `joblib`, `six`, `lazy_loader`
  sont aussi tirés — via `mir/data_file.py` (`joblib`, `pydub.utils.mediainfo`)
  et `mir/io/implement/midi_io.py` (`mido`, `pretty_midi`). Pas vérifié si
  ces quatre sont eux aussi morts au runtime d'inférence comme `h5py`, ou
  réellement exercés (voir « ouvert »).
- **`scipy` : n'apparaît PAS à l'import seul** — `librosa` 0.11.0 le
  lazy-importe (vérifié : `'scipy' in sys.modules` est `False` juste après
  `import librosa`, `True` après un appel réel à `librosa.cqt`). Donc `scipy`
  EST bien tiré au runtime réel (dès que `CQTV2.extract` tourne), juste pas
  par un `import` visible dans le code source.

## Ce qui reste ouvert

- **Le CQT front-end de musx (`hybrid_cqt`) n'a de remplaçant assigné nulle
  part** — ni T1 (qui exporte les RÉSEAUX, pas forcément le front-end DSP),
  ni T2 (qui ne couvre que nnls-chroma). Si T1 ne trace pas le front-end DANS
  le graphe ExecuTorch, quelqu'un doit porter `hybrid_cqt` (CQT récursive +
  pseudo-CQT FFT, resample interne inclus) en numpy — je ne sais pas combien
  ça coûte sans l'avoir tenté, mon estimé (150-300 lignes) est une supposition.
- **`mido`/`pretty_midi`/`pydub`/`joblib`** : présents dans la trace d'import,
  pas vérifié un par un s'ils sont exercés à l'inférence (comme `h5py`,
  probablement pas) ou nécessaires (comme `librosa`, sûrement oui via
  `MusicIO`). Un audit à la `h5py` (chercher qui les instancie réellement dans
  `frame_posteriors`/`_run_nets`) reste à faire.
- **Le contrat de décodage exact (SR/mono/dtype/premier-échantillon) n'est
  écrit nulle part côté Java** — je documente ce qu'il DOIT être (ci-dessous)
  mais je n'ai testé aucun code Android ; le risque n°1 cité partout dans ce
  document (resampler Android ≠ soxr/librosa Mac) n'est vérifiable qu'une
  fois `MediaCodec` codé, hors périmètre lecture-seule de T3.
- **Parité métadonnées `youtubedl-android`** : pas vérifiée contre le
  `yt_dlp` Python actuel (titre/artiste, `titles.py`) — PLAN.md a déjà tranché
  l'outil, pas la parité de sortie.

### Le contrat de décodage audio (à respecter par la version Java)

Trois consommateurs, TROIS taux cibles différents, sur le MÊME fichier
source :

| Consommateur | SR cible | Mono | dtype | Contrainte de phase |
|---|---:|:-:|---|---|
| Beat This! (battues) | 22050 Hz | oui, **moyenne simple** des canaux (`beat_this/inference.py:268-269`, PAS pondérée) | float32 | aucune contrainte de fenêtrage connue (le modèle refait son propre STFT) mais le resampler doit reproduire `soxr` (`beat_this/inference.py:275`) d'assez près pour ne pas décaler un temps de plus de quelques ms |
| musx (postérieures d'accords) | 22050 Hz | oui, moyenne (`librosa.load(..., mono=True)` fait `np.mean` des canaux) | float32 | le premier échantillon PCM doit correspondre au frame 0 de la CQT avec la MÊME convention de centrage que `librosa.core.hybrid_cqt` (par défaut `center=True`, donc un padding réflexif de part et d'autre) — sinon `make_beat_arr` (`musx.py:102`, `np.round(bt/FRAME_DT)`) mappe les temps de battue sur le mauvais indice de frame |
| nnls-chroma (accords/basse) | 44100 Hz | oui, moyenne | float32 | le pas de `vamp.collect` (`out["matrix"][0]`) doit être reproduit à l'identique par le port numpy de T2 — le décodage lui-même n'a pas de contrainte de phase supplémentaire au-delà du SR exact |
| mic/upload (`analyze.py`) | 44100 Hz | oui | float32 (avant ré-encodage AAC actuel — sur Android, sans ffmpeg, il n'y a plus besoin de ré-encoder, juste de décoder puis d'alimenter les 3 chemins ci-dessus directement) | — |

Le risque le plus concret et le moins visible : **le délai d'amorçage
(« priming ») des encodeurs AAC/Opus**. `ffmpeg` respecte les métadonnées de
conteneur (edit lists, `iTunSMPB`) qui disent où le VRAI signal commence
après les échantillons d'amorçage du codec ; si `MediaCodec` ne fait pas la
même chose, TOUT le PCM décodé est en avance ou en retard de quelques
dizaines de ms par rapport à ce que produit `ffmpeg` — un décalage constant,
silencieux, exactement le type de bug de la règle n°1 de CLAUDE.md, et qui ne
casserait RIEN de visible dans les logs.

## Fichiers produits

- `docs/portage_mobile/p0_t3_deps_natives.md` (ce fichier)
