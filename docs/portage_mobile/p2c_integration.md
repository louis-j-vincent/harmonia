# P2c — le pipeline branché sur les briques mobiles, et la parité sur le Mac

## Résultat en 3 lignes

Le pipeline tourne de bout en bout avec les cinq briques mobiles (décodage,
battues `.pte`, CQT numpy, réseau musx numpy, chroma NNLS numpy) et **sans
importer torch, librosa, vamp ni scipy** — seul `torch` reste dans
`sys.modules`, tiré par le paquet pip `executorch` sur le Mac, pas par le
runtime natif d'Android. Le chart de sortie est **identique à 100 % des
mesures sur 2 morceaux sur 3, et à 98,84 % sur le troisième** (Stand By Me :
une seule mesure, la 8e, où le chemin mobile ré-écrit `A` là où le Mac tient
l'accord précédent) ; nombre de mesures, tonalité, sections et basse sonnante
identiques partout ; les battues collent à ≤ 0,9 ms sauf une battue en trop
sur ce même Stand By Me. Le côté Mac ne bouge PAS : les charts d'avant les
coutures et ceux d'après sont **octet pour octet identiques** sur les 3
morceaux, et la suite de tests passe pareil qu'avant (470/486 ; le seul rouge
est un test « sections » d'une autre session, rouge aussi sur un arbre propre).
Vitesse Mac du chemin mobile : **≈ 56 s pour 4 min d'audio** (0,235 s de calcul
par seconde de musique), dont 65 % dans le réseau musx en numpy.
Et la découverte qui a coûté le plus de temps, à garder pour
Android : **le pipeline du Mac décode le MÊME fichier de deux façons
différentes** — `ffmpeg` pour Beat This!, `librosa`/CoreAudio pour musx et le
chroma — et ces deux décodages diffèrent assez (max|Δ| = 0,387 sur Stand By
Me) pour changer le nombre de battues de 345 à 348 ; sur le téléphone il n'y
aura qu'un décodeur, donc cette divergence-là est **structurelle, pas un bug
à corriger**.

## Ce qui a été vérifié (commandes, chiffres, chemins)

### 1. La couture : un registre, huit fentes, aucune implémentation dedans

`harmonia/backends.py` (nouveau) ne contient **aucun code de calcul**. Il
nomme huit fentes et expose `BACKENDS.call("<fente>", <la_fonction_actuelle>,
…)`. Quand personne n'a rien installé, `call` appelle la fonction actuelle —
le comportement du Mac ne peut donc pas dériver par omission, parce qu'il n'y
a rien à ré-écrire pour l'obtenir. Le choix du moteur se fait par du code
(`mobile.harmonia_mobile.backends_mobile.install()`), jamais par une variable
d'environnement : `settings.py` reste le seul lecteur de `os.environ`.

### 2. Le Mac ne bouge pas — mesuré AVANT de toucher quoi que ce soit

Charts de référence produits d'abord avec le pipeline intact
(`mobile/tools/parity_out/before/`), puis re-produits après les coutures
(`…/reference/`), caches froids et isolés dans les deux cas :

| morceau | chart final | chart brut |
|---|---|---|
| maroon_5_this_love | **identique octet pour octet** | identique |
| ben_e_king_stand_by_me_audio | **identique octet pour octet** | identique |
| maroon_5_she_will_be_loved… | **identique octet pour octet** | identique |

(`cmp` sur les JSON ; seul `*.times.json`, qui ne contient que des durées,
diffère.) Aucun horodatage dans le chart : l'identité est stricte.

### 3. Le piège trouvé en route : le Mac a DEUX décodeurs pour le même fichier

Prémisse criblée avant de mesurer (CLAUDE.md règle 1). `p0_t3` supposait
ffmpeg partout, `p2a` avait corrigé en « c'est CoreAudio via `librosa.load` » ;
les deux sont vrais, à des endroits différents :

| site d'appel | décodeur réel sur le Mac | taux |
|---|---|---|
| `beats.track` → Beat This! | `ffmpeg -ac 1 -ar 22050` (aucun lecteur de `beat_this` n'ouvre un `.m4a` ici) | 22050 |
| `musx.frame_posteriors` → CQT | `librosa.load` → CoreAudio | 22050 |
| `nnls_features.extract_bothchroma` | `librosa.load` → CoreAudio | 44100 |
| `debut.enveloppe` | `ffmpeg -f f32le` | 8000 |

Les deux décodages à 22050 Hz **ne sont pas le même signal** :
`mobile/tools/diag_beats_decoder.py`, sur Stand By Me → 3 913 280 échantillons
(librosa) contre 3 913 728 (ffmpeg), **max|Δ| = 0,387** sur la partie commune
(0,134 sur la première seconde). Le MÊME réseau de battues porté rend
**345 battues** nourri par l'un et **348** nourri par l'autre. La déviation
est donc le DÉCODEUR, pas le portage — et elle se mesure sans device Android.

Conséquence : la parité ci-dessous rejoue, pour chaque site d'appel, le
décodage que ce site fait aujourd'hui (`mobile/tools/dump_reference_pcm.py`
fige le PCM une fois sous `.venv`, `backends_mobile` le relit). C'est ce qui
isole les cinq briques de calcul. Sur le téléphone il n'y aura qu'un décodeur
(MediaCodec) : la ligne « un seul décodeur » du tableau dit ce que ça coûte.

### 4. Parité, chart mobile contre chart Mac, mesure par mesure

Décodage tenu au niveau de chaque site d'appel (la parité de la PLOMBERIE) :

| morceau | mesures | battues | max\|Δt\| battues | accords/mesure identiques | basse sonnante | sections | tonalité |
|---|---|---|---|---|---|---|---|
| maroon_5_this_love | 80 = 80 | 322 = 322 | **0,9 ms** | **100,00 %** | identique | A (1) = A (1) | C minor = C minor |
| ben_e_king_stand_by_me_audio | 86 = 86 | 347 vs **348** | — (comptes ≠) | **98,84 %** (1 mesure) | 1 accord en plus | A (1) = A (1) | A major = A major |
| maroon_5_she_will_be_loved… | 107 = 107 | 430 = 430 | **0,2 ms** | **100,00 %** | identique | A (1) = A (1) | Bb major = Bb major |

La seule mesure qui diffère, sur les 273 comparées : **Stand By Me, mesure 8
— le chemin mobile écrit `A`, le Mac n'écrit rien** (il tient l'accord de la
mesure 7, qui est déjà `A`). Ce n'est pas un accord faux, c'est une
re-frappe : le décodeur beat-aware a une battue de plus à cet endroit et y
place un changement. Origine localisée : la battue en trop, elle-même due à
l'écart de logits entre torch et `.pte` (T1c l'avait déjà caractérisé : sur
this_love, une battue sur 322 basculait de 20 ms sur un vrai ex-æquo du
réseau, logits 7,249358 contre 7,249332).

Avec UN SEUL décodeur pour tout le morceau (ce que sera Android), même mesure,
`mobile/tools/parity_out/compare_un_seul_decodeur.json` :

| morceau | mesures | battues | accords/mesure identiques | basse | sections | tonalité |
|---|---|---|---|---|---|---|
| maroon_5_this_love | 80 = 80 | 322 = 322 (Δt ≤ 0,4 ms) | **100,00 %** | identique | identique | identique |
| ben_e_king_stand_by_me_audio | 86 = 86 | 347 vs 345 | **100,00 %** | identique | identique | identique |
| maroon_5_she_will_be_loved… | 107 = 107 | 430 = 430 (Δt ≤ 0,1 ms) | **100,00 %** | identique | identique | identique |

À lire ensemble : les deux configurations donnent ≥ 98,8 % de mesures
identiques, et laquelle des deux est « meilleure » ne se décide pas au
nombre — deux battues de moins au tout début de Stand By Me décalent les deux
premières mesures de 1,52 s sans changer un seul accord écrit. C'est un
arbitrage de Louis sur un avant/après, pas une régression.

Le réseau musx utilisé ici est bien le port numpy de T1b
(`mobile/harmonia_mobile/musx_net.py`, 5 folds moyennés, parité 2e-6 contre
torch) — PAS le repli `musx_torch_reference`, qui existe dans
`backends_mobile` mais n'a servi à aucune des mesures ci-dessus. C'est
vérifiable sans me croire : §5, `torch` n'entre que par ExecuTorch, donc
aucun réseau torch n'a tourné.

### 5. La preuve « ça tournera sous Chaquopy »

`sys.modules` à la fin de l'analyse des 3 morceaux par le chemin mobile :
**`torch` seul**, aucun `librosa`, aucun `vamp`, aucun `scipy`. Le `torch`
vient de `import executorch.runtime` (vérifié isolément), pas du pipeline :
sur Android le runtime ExecuTorch est une bibliothèque native chargée par JNI.
`mobile/tools/purity_check.py` fait tourner pour de vrai les quatre briques
purement numpy (CQT, ChordNet, NNLS, `find_peaks`) et confirme :
`cqt (862, 288) · musx [(200,73),(200,13),…] · nnls (431, 24) · pics 23` → **modules interdits importés : AUCUN** (220 modules chargés en tout, contre 3 245 pour le pipeline du Mac).

`find_peaks` n'est sur AUCUN chemin de chart (`folding.py` n'appelle que
`halfbar_features`), donc la parité du chart ne le teste pas : vérifié à part
contre scipy, `mobile/tools/check_find_peaks.py` → **300/300 vecteurs
identiques** (bruit, plateaux, courbes lissées). Premier essai : 216/300 —
le tri des ex-æquo. `scipy` fait `np.argsort(priority)` sans `kind=` ; j'avais
mis `kind="stable"`, ce qui casse les ex-æquo dans l'autre sens et change les
pics retenus dès qu'un signal a des plateaux. Corrigé, pas contourné.

### 6. Vitesse sur le Mac, chemin mobile, par étape

Deux mesures, parce qu'elles ne disent pas la même chose.

**(a) En série, une étape à la fois** (`mobile/tools/speed_mobile_steps.py`,
`OMP/OPENBLAS/MKL/VECLIB_NUM_THREADS=1`, décodage ffmpeg réel inclus) — c'est
le coût PROPRE de chaque brique :

| morceau | durée | décodage 22050 | décodage 44100 | battues (`.pte`) | CQT | réseau musx (numpy, 5 folds) | chroma NNLS | somme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| maroon_5_this_love | 205,3 s | 0,13 s | 0,15 s | 11,65 s | 0,72 s | **24,41 s** | 0,99 s | 38,1 s |
| ben_e_king_stand_by_me_audio | 177,5 s | 0,16 s | 0,14 s | 8,39 s | 0,73 s | **26,40 s** | 1,41 s | 37,2 s |
| maroon_5_she_will_be_loved… | 268,4 s | 0,33 s | 0,34 s | 21,57 s | 1,59 s | **50,86 s** | 2,96 s | 77,7 s |

Le réseau musx en numpy est le poste dominant (≈ 65 % du calcul), devant les
battues (≈ 25 %). Le décodage, le CQT et le chroma NNLS pèsent ensemble moins
de 6 % — les trois briques sur lesquelles P2b et T2b ont travaillé sont
largement sous le budget. Ramené à 4 min d'audio : **≈ 45 s sur le Mac**, soit
environ 5× le temps réel.

**(b) Dans le pipeline complet** (`parity_pipeline.py`, les mêmes 3 morceaux,
chart écrit) — durées MURALES, et `analyze_steps` lance les battues et musx
dans deux threads, donc ces colonnes SE RECOUVRENT (c'est pour ça que
« reste » sort négatif) :

| morceau | battues | CQT musx | réseaux musx | chroma NNLS | `enveloppe` | total mural |
|---|---:|---:|---:|---:|---:|---:|
| maroon_5_this_love | 17,13 s | 0,85 s | 51,48 s | 1,66 s | — | **54,4 s** |
| ben_e_king_stand_by_me_audio | 14,66 s | 15,46 s | 28,40 s | 1,37 s | 0,002 s | **45,6 s** |
| maroon_5_she_will_be_loved… | 22,40 s | 23,44 s | 43,16 s | 1,92 s | 0,002 s | **69,0 s** |

Pour comparer, le pipeline du Mac sur les mêmes morceaux, caches froids :
34,6 s / 24,7 s / 39,1 s. Le chemin mobile est donc **1,6 à 1,8× plus lent
que le Mac**, et l'écart est presque entièrement le réseau musx en numpy
contre les 5 réseaux torch (sur MPS). Aucune de ces valeurs ne dit quoi que ce
soit du téléphone : T5 reste la porte.

Somme sur les 3 morceaux : 651 s d'audio en 153 s de calcul, soit
**0,235 s de calcul par seconde d'audio → ≈ 56 s pour 4 min**. La règle des
deux minutes de T5 tient largement sur le Mac ; sur ARM, tout se joue sur le
réseau musx (numpy pur, sans XNNPACK).

### 7. Les tests, et un rouge qui n'est pas le mien

`.venv/bin/python -m pytest tests/ -q` → **470 passés, 15 ignorés, 1 échec**.
L'échec est `test_songformer_sections.py::test_minimal_fold_separe_les_
longueurs_dune_meme_lettre` (`minimal_fold` groupe par lettre seule au lieu de
(lettre, longueur)). Vérifié plutôt que supposé : rejoué sur un arbre
`git archive HEAD` PROPRE, sans aucune de mes modifications → **il échoue
pareil**. C'est du travail « sections » en cours d'une autre session, pas une
régression de P2c.

### 8. La liste EXACTE des lignes changées dans `harmonia/`

Aucune duplication de `analyze_steps`, aucun `if android:`. 171 lignes
ajoutées, 74 retirées, sur 6 fichiers + 1 nouveau. Les « + » sont à 80 % du
code DÉPLACÉ tel quel (le corps de l'appel natif devient une fonction nommée)
et des docstrings.

| fichier | ce qui change | pourquoi |
|---|---|---|
| `harmonia/backends.py` | **nouveau** (115 lignes, dont ~85 de documentation) | le registre. Zéro implémentation : les défauts restent dans les feuilles |
| `harmonia/beats.py` | +1 import ; `_beatthis_track()` (l. 913-939) = le corps de `track()` déplacé mot pour mot ; l. 969 : `bts, dbs = BACKENDS.call("beats", _beatthis_track, audio_path)` remplace 18 lignes | le tracker devient remplaçable sans toucher au cache, au `_clean()` ni au garde-fou de grille |
| `harmonia/musx.py` | +1 import ; `_cqt_clone()` (l. 295-314) = le bloc `_InMusxDir`+`CQTV2`, qui était **dupliqué** dans `frame_posteriors` et `song_cqt` ; l. 342 et 378 : `BACKENDS.call("musx_features", _cqt_clone, …)` ; l. 343 et 391 : `BACKENDS.call("musx_posteriors", _run_nets, …)` | deux fentes distinctes parce que les deux moitiés sont portées par deux tâches différentes (P2b le CQT, T1b le réseau) et doivent pouvoir bouger séparément |
| `harmonia/nnls_features.py` | +1 import ; `_bothchroma_vamp()` (l. 59-91) = le corps déplacé (librosa + `vamp.collect`) ; l. 123 : `arr, times = BACKENDS.call("nnls_bothchroma", _bothchroma_vamp, audio_path)` remplace 23 lignes | le plugin VAMP natif est le seul GPL du chemin ; la fente est aussi la porte de sortie de licence |
| `harmonia/pipeline.py` | +1 import ; `duration_seconds()` garde son nom et sa docstring, son corps ffprobe devient `_duration_ffprobe()` ; `_slice_to_wav_ffmpeg()` extrait de `_head_chart` | `duration_seconds` est publique (`analyze_steps` s'en sert pour décider s'il y a un chart de tête, et `server/jobs.py` a son propre doublon ffprobe) : elle garde son nom, sa signature et sa docstring, seul son corps se déplace. `slice_to_wav` ne sert qu'à l'aperçu de tête |
| `harmonia/debut.py` | +2 imports ; `_pcm_ffmpeg()` (l. 94-112) extrait ; `enveloppe()` passe par `BACKENDS.call("load_audio", _pcm_ffmpeg, …, 8000, duration=fenetre)` | 4e taux d'échantillonnage du chemin critique (8 kHz), et le seul consommateur d'audio décodé en dehors des 3 gros étages |
| `harmonia/sections/similarity.py` | **`from scipy.signal import find_peaks` retiré de l'en-tête** (seul retrait d'import du lot) ; `_find_peaks_scipy()` ajouté, qui importe scipy PARESSEUSEMENT ; `_peaks()` passe par la fente | `folding.py` importe ce module pour `halfbar_features` ; un import scipy en tête faisait planter tout le chemin critique sur téléphone pour une fonction qui n'y est pas |

Ce que ces 7 fichiers NE contiennent pas : aucune mention d'Android, aucun
test de plateforme, aucune variable d'environnement nouvelle.

## Ce qui reste ouvert

* **Le décodeur Android n'est PAS dans cette mesure**, et c'est délibéré : la
  parité ci-dessus rejoue le PCM du Mac pour isoler les cinq briques de
  calcul. Le résampleur de `p2a` a un défaut connu et non corrigé (max diff
  0,084 / moyenne 0,0022 contre `soxr_hq` sur fichier entier, 64 taps/phase
  insuffisants — JOURNAL du 2026-09-23), et le §3 ci-dessus montre qu'un écart
  de décodage de cet ordre suffit à bouger le compte de battues. **C'est le
  premier risque du portage, devant tout le reste.** À mesurer en rejouant ce
  même `parity_pipeline.py` avec `audio="mediacodec"` sur le device.
* **Une battue et une mesure d'écart sur Stand By Me** (mesure 8, `A`
  ré-écrit). Caractérisé, pas corrigé : c'est l'ex-æquo de logits torch↔`.pte`
  déjà documenté par T1c. Corriger en aval (forcer le compte de battues)
  serait exactement ce que la règle n°6 interdit.
* **La fente `slice_to_wav` (chart de tête) n'est pas exercée par la mesure** :
  la parité tourne avec `head_s=0`, parce que l'aperçu est rendu sur un yield
  séparé, n'entre pas dans le chart final, et coûterait trois analyses de 45 s
  de plus. Le code mobile existe (`wave` de la stdlib, PCM 16 bits comme le
  wav par défaut de ffmpeg) mais n'a été passé qu'en fumée.
* **`duration_seconds` mobile rend la longueur du PCM décodé**, pas
  `MediaExtractor.KEY_DURATION` — P2a a mesuré que la clé Android rend une
  valeur plus GRANDE que le PCM réellement décodé. Sans conséquence ici
  (`head_s=0`), à trancher quand l'aperçu sera branché sur le téléphone.
* **Le chemin mobile n'a jamais tourné sur un vrai téléphone.** Tous les
  chiffres de vitesse sont Mac. La porte dure reste T5.
* **3 morceaux, pas 44.** Le rapport d'or (`tools.golden`) n'a pas été passé :
  il compare des charts du Mac entre eux, et le chemin mobile ne s'y branche
  pas encore. Comme le Mac est octet-identique avant/après (§2), il n'y avait
  rien à arbitrer côté bibliothèque.

## Fichiers produits

* `harmonia/backends.py` — le registre (nouveau).
* Coutures dans `harmonia/{beats,musx,nnls_features,pipeline,debut}.py` et
  `harmonia/sections/similarity.py` (tableau du §8).
* `mobile/harmonia_mobile/backends_mobile.py` — installe les cinq briques ;
  `install(audio=…, musx=…)` choisit le décodage et, si besoin, le repli
  `musx_torch_reference` (explicite, jamais silencieux).
* `mobile/tools/parity_pipeline.py` — rejoue les 3 morceaux par chemin
  (caches froids et séparés) et compare mesure par mesure.
* `mobile/tools/dump_reference_pcm.py` — fige le PCM de référence du Mac,
  site d'appel par site d'appel.
* `mobile/tools/diag_beats_decoder.py` — la mesure du §3 (décodeur vs réseau).
* `mobile/tools/check_find_peaks.py` — port numpy vs scipy, 300 vecteurs.
* `mobile/tools/purity_check.py` — les 4 briques numpy + `sys.modules`.
* `mobile/tools/speed_mobile_steps.py` — vitesse par étape, en série.
* `mobile/tools/parity_out/` — charts, timings, comparaisons, PCM figé
  (227 Mo, gitignoré : tout est régénérable).
