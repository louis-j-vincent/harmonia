# Audit — minimum vital pour un portage mobile autonome (2026-09-22)

Question posée : si on veut « audio en entrée → chart jouable en sortie » sur
un téléphone, SANS Mac, calcul embarqué — quelle fraction des 17 642 lignes
du paquet `harmonia/` est réellement sur le chemin critique, et dans quoi
faut-il investir pour la porter ?

**Méthode** : partir de `harmonia.pipeline.analyze_steps` (le point d'entrée
unique), suivre TOUS les imports réellement exécutés — y compris ceux faits
au milieu des fonctions (`from harmonia.x import y`), qui sont la norme dans
ce paquet — puis faire la même chose depuis les 7 familles de routes
serveur. Chaque ligne du paquet est comptée une fois, dans une seule des
quatre catégories : **moteur** (CORE, ce qu'`analyze_steps` exécute
toujours), **service minimal** (CORE_SERVE, le minimum pour qu'un chart
calculé arrive à l'écran), **accessoire** (une fonctionnalité réelle de
l'app Mac, mais pas nécessaire pour écouter un premier chart), **mort pour
mobile** (du code vivant dans le dépôt mais jamais exécuté par l'app, ou
utile seulement à un diagnostic humain).

## Résultat en un coup d'œil

| Bucket | Lignes | % du paquet |
|---|---:|---:|
| **Moteur (CORE)** — le calcul lui-même | **7 050** | **40,0 %** |
| **Service minimal (CORE_SERVE)** — pour voir/stocker le chart | 1 715 | 9,7 % |
| **= Minimum vital total (CORE + CORE_SERVE)** | **8 765** | **49,7 %** |
| Accessoire (vivant, pas nécessaire pour un premier chart) | 6 877 | 39,0 % |
| Mort pour mobile (recherche, diagnostic, code orphelin) | 2 000 | 11,3 % |
| **Total paquet `harmonia/`** | **17 642** | 100 % |

**Verdict sur l'hypothèse (< 40 %) : vraie de justesse pour le moteur pur
(39,96 %, littéralement à la limite), fausse dès qu'on compte ce qu'il faut
en plus pour qu'un chart calculé soit effectivement consultable
(job/cache/overlay/YouTube/titre) : c'est alors 49,7 %, quasiment la
moitié du paquet.** Le paquet n'est pas gonflé à 60 % de recherche comme on
pourrait le craindre en voyant 17 600 lignes — il est déjà taillé serré. Le
vrai gisement d'économie n'est pas dans `harmonia/` : c'est le reste du
dépôt (`tools/` 21 000 lignes, `archive/` 123 000, `scratchpad/` 18 800,
`tests/` 7 300) qui ne compte même pas dans ces 17 642 et n'a AUCUN rôle
dans le portage.

## (a) Tableau module → catégorie → LOC → indispensable

Catégories de portage : **A** logique pure (Python/numpy, port mécanique) ·
**B** DSP scipy/librosa/FFT (à réécrire avec une lib native) · **C** modèle
torch (à convertir ONNX/CoreML/TFLite) · **D** binaire natif externe (VAMP,
ffmpeg, yt-dlp).

### Moteur (CORE) — exécuté à CHAQUE analyse par `analyze_steps`

| Module | LOC | Cat. | Rôle | Indispensable |
|---|---:|:-:|---|:-:|
| `pipeline.py` | 903 | A | orchestration : la fonction elle-même | oui |
| `beats.py` | 966 | C(+D) | Beat This! (torch, pip `beat-this`) ; ffmpeg si m4a | oui |
| `musx.py` | 483 | C | posteriors d'accords, ensemble 5-fold ISMIR2019 (torch) | oui |
| `bars.py` | 312 | A | découpe des temps en mesures | oui |
| `key_profiles.py` | 162 | A | profils de tonalité (Krumhansl etc.), numpy | oui |
| `labels.py` | 96 | A | tables accord ↔ pitch-classes | oui |
| `settings.py` | 117 | A | config | oui |
| `cache.py` | 127 | A | npz/json cache par morceau | oui |
| `debut.py` | 513 | B(+D) | détection du vrai début (FFT numpy maison) ; ffmpeg pour décoder | oui (a un repli try/except, mais tourne à chaque 1ère analyse) |
| `span_rescore.py` | 602 | A | pooling des suggestions d'accords sur des probs déjà en mémoire | oui (appelé sans condition dans le yield "raw" et "final") |
| `nnls_features.py` (partie utilisée) | 174/295 | D | `extract_bothchroma` : plugin natif VAMP nnls-chroma (Mauch & Dixon) + librosa pour charger l'audio | oui |
| `folding.py` | 1 496 | A | le repli (occurrences → une section, affichage) | oui |
| `harmonic_key.py` | 371 | A | tonalité, couleurs, accords non-diatoniques | oui |
| `roles.py` | 172 | A | familles d'accords | oui |
| `sections/__init__.py` | 43 | A | dispatcher (1 seule route : songformer) | oui |
| `sections/songformer.py` | 321 | C | SEUL détecteur de sections : modèle pré-entraîné (MuQ + MusicFM + transformer), poids téléchargés depuis Hugging Face à l'exécution, PAS vendored dans ce dépôt | oui — et sans repli (voir docstring : un échec doit ÉCHOUER) |
| `sections/similarity.py` | 192 | B | `scipy.signal.find_peaks`, chroma demi-mesure | oui (utilisé par `folding.py`) |
| **Sous-total moteur** | **7 050** | | | **40,0 %** |

### Service minimal (CORE_SERVE) — pour qu'un chart calculé arrive à l'écran

| Module | LOC | Cat. | Rôle | Indispensable |
|---|---:|:-:|---|:-:|
| `server/app.py` | 195 | A | fabrique l'app (sera de toute façon jetée/réécrite native, mais encode la logique de câblage) | oui (équivalent nécessaire) |
| `server/jobs.py` | 317 | A | file de jobs, appelle `analyze_steps`, écrit le chart sur disque | oui |
| `server/routes/analyze.py` | 236 | A | déclenche un job (le point d'entrée utilisateur) | oui |
| `server/routes/library.py` | 243 | A | liste/charge/sert un chart, applique l'overlay d'annotations | oui |
| `server/youtube.py` | 299 | D | téléchargement YouTube (yt-dlp en sous-processus + anti-bot) | oui SI on garde l'import par lien (pas requis si mic/fichier seulement) |
| `annotations.py` | 192 | A | schéma des corrections, overlay sur le chart servi | oui (sans lui un chart déjà corrigé se réaffiche faux) |
| `titles.py` | 184 | A | artiste/titre depuis un titre YouTube | oui (cosmétique, mais servi par `library.py`) |
| `__init__.py` × 4 (paquet, server, `__main__`, routes) | 49 | A | plomberie | oui |
| **Sous-total service** | **1 715** | | | **9,7 %** |
| **TOTAL MINIMUM VITAL** | **8 765** | | | **49,7 %** |

### Répartition du minimum vital par type de coût de portage

| Catégorie | LOC | % du minimum vital | Coût de portage |
|---|---:|---:|---|
| A — logique pure | 5 817 | 66,4 % | port mécanique (Swift/Kotlin direct) |
| C — modèle torch | 1 770 | 20,2 % | conversion CoreML/TFLite/ONNX, 3 architectures distinctes |
| B — DSP scipy/FFT | 705 | 8,0 % | réécriture avec une lib DSP native |
| D — binaire natif externe | 473 | 5,4 % | VAMP (aucun équivalent mobile), yt-dlp, ffmpeg |

La ligne la plus trompeuse de ce tableau est la catégorie A à 66 % : elle
donne l'impression que l'essentiel est un « simple » portage mécanique.
Mais la catégorie A n'a de valeur que si elle reçoit les bonnes entrées, et
ces entrées viennent presque toutes de B/C/D (`folding.py`, 1 496 lignes de
pure logique A, ne calcule RIEN sans les postérieures d'accords de `musx.py`
[C] et le chroma NNLS de `nnls_features.py` [D]). Le vrai risque est
concentré dans les 20 % de C et les 5 % de D.

## (b) Chaque route serveur : indispensable à « écouter un chart » ?

| Route | LOC (route + dépendances exclusives) | Indispensable pour écouter un chart ? |
|---|---:|:-:|
| `library.py` | 243 | **oui** — lister/charger/servir un chart |
| `analyze.py` | 236 | **oui** — c'est le déclencheur de l'analyse |
| `annotate.py` | 227 + `corrections.py` 268 = 495 | non — corriger un accord à l'oreille est utile mais pas nécessaire pour écouter le premier chart |
| `sections.py` | 745 + `soudure.py` 1 042 + `section_tool.py` 396 + `sections/simulation.py` 347 + `phrases4.py` 236 + `refold.py` 149 = **2 915** | non — c'est l'outil de RE-DÉCOUPAGE manuel des sections ; SongFormer (dans le moteur) fournit déjà des sections automatiquement |
| `jam.py` | 120 + `jam.py` (module) 270 = 390 | non — mode live séparé, réutilise beats+musx sans dépendance neuve |
| `irealb.py` | 361 + `integrations/*` 2 716 = **3 077** | non — import/export vers l'écosystème iReal Pro / tabs, fonctionnalité annexe |
| `verdicts.py` | 102 | non, et mort pour mobile — page de diagnostic humain |

Seules **2 routes sur 7** (library, analyze — 479 lignes) sont strictement
nécessaires pour « écouter un chart » ; les 5 autres (6 877 lignes au
total, exactement le bucket ACCESSOIRE) sont des fonctionnalités réelles de
l'app Mac mais pas du chemin critique mobile.

## (c) Les 3 points les plus durs à porter

1. **`sections/songformer.py` (321 lignes, mais le module le plus lourd du
   paquet).** C'est LE seul détecteur de sections (pas de repli — un échec
   doit échouer, décision explicite de Louis) et il ne pèse que 321 lignes
   parce que le vrai modèle n'est même PAS dans ce dépôt : à l'exécution, le
   code télécharge un dépôt Hugging Face entier (`modeling_songformer.py`)
   et l'exécute dynamiquement (`importlib` + `exec_module`). Le modèle
   empile DEUX encodeurs audio pré-entraînés (MuQ, MusicFM) + une tête
   Transformer, 1052 tenseurs. Sur un Mac 16 Go, il faut déjà un
   sous-processus jetable parce que le processus se fait tuer (SIGKILL,
   code 137) sur un morceau de 7 minutes. Porter ça sur téléphone veut dire
   obtenir/exporter TROIS architectures empilées en CoreML/TFLite avec un
   budget mémoire bien plus serré qu'un Mac — c'est le seul module du
   paquet qui, à lui seul, peut faire échouer le portage.
2. **Le plugin VAMP `nnls-chroma` (Mauch & Dixon).** `extract_bothchroma`
   (dans `nnls_features.py`) l'appelle deux fois par analyse (repli +
   analyse harmonique/basse). C'est un binaire C++ natif compilé pour
   desktop (bibliothèque VAMP, pas un paquet pip) ; il n'existe aucune
   implémentation Python/numpy de référence à porter dans ce dépôt — il
   faut ré-écrire de zéro une déconvolution NNLS DSP en Swift/Accelerate à
   partir de la doc du plugin d'origine, avec le risque de reproduire une
   des « erreurs de calibration silencieuses » que CLAUDE.md liste comme le
   pire type de bug de ce projet (règle n°1).
3. **`musx.py` (483 lignes) + `third_party/musx_ismir2019/` (6 045 lignes,
   PAS comptées dans les 17 642 mais indispensables).** Un ensemble de 5
   checkpoints torch (CNN + bi-LSTM) entraînés séparément et moyennés, plus
   un décodeur de type Viterbi sur mesure (`XHMMDecoder`) dont les coûts de
   transition dépendent exactement de la grille de battues. Convertir 5
   checkpoints en CoreML est mécanique ; re-dériver fidèlement le front-end
   CQT (`mir`/`extractors`, du code de recherche vendored, pas maintenu) ET
   la grille temporelle exacte (`FRAME_DT = 512/22050`, 23,22 ms/image) sans
   introduire un décalage d'unité silencieux est le vrai risque — ce projet
   a déjà été piégé quatre fois par exactement ce type de bug (CLAUDE.md,
   règle n°1).

## (d) Ce qu'on peut jeter

### Mort pour mobile, à l'intérieur de `harmonia/` (2 000 lignes, 11,3 %)

| Module | LOC | Pourquoi |
|---|---:|---|
| `sections/nouveaute.py` | 500 | détecteur de Foote « repêché » — importé seulement par `tools/sections_nouveautes.py` et son test, JAMAIS par l'app |
| `ssm_page.py` | 634 | page `/ssm`, matrice de similarité cliquable — diagnostic humain |
| `models/harmonic_downbeat.py` | 490 | brique de recherche « kill-switched », jamais câblée dans `pipeline.py` |
| `models/local_key_context.py` | 153 | brique de recherche, jamais câblée dans `pipeline.py` |
| `server/routes/verdicts.py` | 102 | remontée d'arbitrages humains sur des pages de diagnostic |
| `nnls_features.py::NNLS24Heads` | 121 | tête MLP torch pour un front-end alternatif (`feature_frontend="nnls24"`) que `pipeline.py` ne sélectionne jamais |

### Accessoire, vivant mais pas nécessaire au MVP mobile (6 877 lignes, 39,0 %)

- **Outil d'édition de sections** (re-découpage manuel) — 2 915 lignes :
  `soudure.py`, `section_tool.py`, `sections/simulation.py`, `phrases4.py`,
  `refold.py`, `server/routes/sections.py`.
- **Intégrations iReal Pro / Ultimate Guitar** — 3 077 lignes :
  `server/routes/irealb.py` + tout `harmonia/integrations/`.
- **Corrections manuelles** — 495 lignes : `server/routes/annotate.py` +
  `corrections.py`.
- **Mode Jam (micro live)** — 390 lignes : `jam.py` + `server/routes/jam.py`.

### Hors paquet, sans aucun rôle dans le portage

`tools/` (21 000 lignes, scripts de recherche/rapport d'or), `tests/`
(7 347), `archive/` (123 360), `scratchpad/` (18 831, non suivi git) — soit
plus de dix fois le paquet `harmonia/` lui-même, entièrement composé
d'outillage, d'historique et de recherche qui n'a jamais eu vocation à
tourner en production, encore moins sur un téléphone.

## Dépendances lourdes qui ne comptent dans AUCUNE des lignes ci-dessus

Trois modèles pré-entraînés vivent hors du paquet et sont pourtant sur le
chemin critique — leur coût de portage n'apparaît nulle part dans les 17 642
lignes :

- **Beat This!** (pip `beat-this`, 2 834 lignes) — le traqueur de temps,
  torch, MIT.
- **`third_party/musx_ismir2019/`** (6 045 lignes vendored) — le modèle
  d'accords.
- **SongFormer** — pas vendored du tout, code + poids vivent sur Hugging
  Face et sont téléchargés à l'exécution.

