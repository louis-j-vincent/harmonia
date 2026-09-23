# P2d — le pipeline complet SUR L'APPAREIL (2026-09-23)

## Résultat en 3 lignes

**Les trois morceaux de seed s'analysent entièrement sur l'émulateur et
rendent un chart : 273 mesures comparées au Mac, 271 identiques (99,27 %),
même nombre de mesures, même tonalité, mêmes sections et même tempo sur les
trois.** Les deux mesures qui diffèrent disent la même chose : l'émulateur
écrit un `Eb/Bb` là où le Mac écrit `Eb` — une basse, pas un accord. La porte de la
phase 2 est franchie, avec trois réserves qu'il faut dire tout de suite :
(1) **l'émulateur ne peut pas exécuter XNNPACK** — son CPU invité annonce
`sme2` (l'hyperviseur d'Apple laisse passer les bits de la puce hôte) mais
l'instruction SME `rdsvl`, que XNNPACK exécute une fois à l'initialisation,
y lève un SIGILL ; les battues tournent donc avec un `.pte` PORTABLE, correct
mais ~10× plus lent, et **aucun chiffre de vitesse des battues ici ne vaut pour
le téléphone de Louis** ; (2) il a fallu **borner la mémoire de deux étages**
(réseau musx et chroma NNLS) — sans ça le système tuait l'app avant le premier
accord ; les deux tuilages sont bit-exacts et, surprise, **plus rapides** ;
(3) **l'aperçu de tête est désactivé sur téléphone** : il nourrit Beat This!
avec un bloc de moins de 1500 trames, que le `.pte` en forme fixe refuse — un
vrai défaut du portage, trouvé ici, qui toucherait le Mac de la même façon.
APK : **193 Mio**.

## Ce qui a été vérifié (commandes, chiffres, chemins)

### 1. Le runtime ExecuTorch natif marche, et le `.pte` de 1.5.1 passe sur l'AAR 1.4.0

Le piège annoncé par le brief (« compatibilité de format `.pte` ») a été
criblé avant d'écrire une ligne de Kotlin, et il **n'a pas mordu** :

* le `.pte` a été exporté par le paquet pip `executorch 1.5.1`
  (`~/harmonia/.venv-mobile`) ;
* la dernière version **publiée** de l'AAR sur Maven Central est
  **1.4.0** (2026-08-07 ; `curl .../org/pytorch/executorch-android/maven-metadata.xml`).
  `download.pytorch.org/maven` répond 403, l'archive S3 de la 1.5.1 n'existe
  pas — il n'y a pas d'AAR 1.5.1 à prendre ;
* l'identifiant de schéma du fichier est `ET12` (`xxd -l 16` sur le `.pte`),
  et le runtime 1.4.0 **charge le programme sans broncher** :
  `harmonia.beats: pte chargé: 82375304 octets, 4 threads, 14 ms`.

`BeatsRuntime.kt` charge le module en `LOAD_MODE_MMAP` (82 Mo mappés depuis
le fichier plutôt que lus dans le tas — l'appareil n'a que 2 Go) et règle les
threads XNNPACK au nombre de coeurs par le **troisième paramètre de
`Module.load`**, vérifié dans la javadoc de l'AAR et non supposé
(`load(String modelPath, Integer loadMode, Integer numThreads)`). Le log du
runtime le confirme : `ExecuTorch: Resetting threadpool to 4 threads.`

La fente côté Python est `beats_infer.set_chunk_runner(fn)` : **une seule
ligne du chemin change entre le Mac et le téléphone**, l'appel au réseau. Le
mel, le découpage en blocs, l'agrégation et le post-traitement restent le
même code numpy des deux côtés — sinon la comparaison « chart émulateur vs
chart Mac » ne mesurerait plus le portage mais deux implémentations.
`backends_mobile.install(beats_runtime="python"|"java")` choisit, et
`"python"` reste le défaut, donc **aucune mesure Mac ne bouge**.

### 2. Ce qui bloque vraiment sur l'émulateur : une instruction SME

Le premier `forward` tuait le process. Désassemblé plutôt que deviné :

```
F libc : Fatal signal 4 (SIGILL), code 1 (ILL_ILLOPC), fault addr 0x...774
#00 pc 0x310774  libexecutorch.so
#03 pc          libc.so (pthread_once+148)     <- initialisation unique
```
```
$ llvm-objdump -d --start-address=0x310774 libexecutorch.so
  310774: 04bf5820   rdsvl x0, #0x1
```

`rdsvl` (*read streaming vector length*) est une instruction **SME**.
XNNPACK l'exécute une fois, à l'initialisation, pour choisir ses noyaux.
`adb shell cat /proc/cpuinfo` montre que le CPU invité **annonce** `sme2`,
`smef32f32`, `smei8i32`… : l'hyperviseur d'Apple laisse passer les bits
d'identification de la puce hôte, mais l'état SME n'est pas activé pour
l'invité, donc l'instruction piège. Vérifié aussi que ce n'est pas une
question de version : les AAR 1.0.1, 1.2.0, 1.3.1 et 1.4.0 contiennent
**toutes** des instructions SME (23 à 32 occurrences).

**Ce n'est pas un défaut du portage.** Un téléphone réel ou bien n'annonce pas
SME (et XNNPACK prend ses noyaux NEON), ou bien l'annonce et l'active
vraiment. Mais sur CET émulateur, aucun chart n'aurait pu être produit.

Contournement, explicite et hors de l'APK : un second `.pte` exporté **sans
délégué** (`export_beat_this.py --portable`, 82,3 Mo, 13 s de lowering). Il
n'est **pas embarqué** — il se pousse à la main sur l'appareil de test, et
`mobile_entry` le prend s'il le trouve en le **hurlant dans le log** (`BEAT
THIS! en mode PORTABLE … la VITESSE mesurée ici ne vaut RIEN pour le
téléphone`). Un chiffre de vitesse pris sur l'émulateur ne peut donc pas être
confondu avec celui du téléphone.

Au passage : l'export portable minimal échoue dans ce venv
(`AssertionError: expected compiled_fn to be GraphModule`) alors que le chemin
complet de `export_beat_this.py` passe — la décomposition de torch 2.14 dépend
du préambule (passe avant sous `no_grad`, exemple réel plutôt que des zéros).
Plutôt que de deviner lequel des deux compte, `main_portable` **refait le même
préambule**. Noté, pas élucidé.

### 3. La mémoire : 1,95 Go → 0,55 Go, et le calcul n'a pas bougé d'un bit

Premier essai d'analyse : `TOTAL PSS` passe de 292 Mo à 1183 Mo en trois
secondes et `lowmemorykiller` tue l'app (« low watermark is breached »). Deux
étages débordaient, et **dans les deux cas la prémisse « on découpe en temps
avec un recouvrement égal au champ réceptif » était fausse** — c'est le
résultat le plus utile de la tâche :

| étage | le tableau coupable | ce qui interdit le découpage naïf | ce qui a été fait |
|---|---|---|---|
| réseau musx | l'im2col de `conv1b`/`conv1c`, **1283 Mo chacun** (9·16·8840·252·4 o) | `InstanceNorm2d` normalise sur **tout** l'axe temps : couper le réseau change les statistiques à n'importe quel recouvrement | tuiler **chaque opération** (conv, norm+selu) au lieu de couper le réseau ; stats globales conservées, bi-LSTM entière |
| chroma NNLS | l'étage FFT : `windowed` (T, 16384) float64 et son `rfft` (T, 8193) complex128, **758 Mo chacun** sur 4 min 28 | l'estimation d'**accordage** est une moyenne sur toutes les trames du fichier | tuiler la FFT, garder `nm_all` (T, 256) entier — 12 Mo — pour l'accordage global |

Le champ réceptif a quand même été calculé explicitement (10 convs 3×3, et
les 3 maxpool sont `MaxPool2d(1, k)` : **noyau 1 en temps**, donc aucun
sous-échantillonnage temporel, RF = 21 trames) — il est dans la docstring de
`_cnn_forward`, parce que la question se reposera.

Résultat, mesuré sur les 3 morceaux de référence :

| | écart max vs non tuilé | argmax | pic mémoire propre | vitesse (s / 4 min) |
|---|---|---|---|---|
| réseau musx (`chunk_frames=256`) | **1,8e-7** (seuil 1e-5) | 100,0000 % sur les 6 têtes | 2458 → **489 Mo** | 30,4 → **23,6** (−24 %) |
| chroma NNLS (`chunk_frames=256`) | **0.0 — bit-identique** | basse 100 % | 2462 → **79 Mo** (÷31) | 1,69 → **1,08** (−36 %) |

Les deux sont **plus rapides**, pas plus lents : c'est la localité de cache.
`None` reste le défaut des deux fonctions, donc le Mac garde son chemin
d'origine octet pour octet.

Sur l'appareil, pic `TOTAL PSS` mesuré par `adb shell dumpsys meminfo
fr.harmonia.app` toutes les 2 s :

| configuration | pic PSS |
|---|---:|
| sans tuilage (musx seul, morceau entier) | tué par le système (1183 Mo au dernier relevé avant la mort) |
| tuiles musx 512, NNLS non tuilé | **1947 Mo** — l'app survit de justesse, puis meurt sur le morceau suivant |
| tuiles musx 256 **et** NNLS 256 | **546 Mo** (RSS 653 Mo) |

`resource.getrusage(RUSAGE_SELF).ru_maxrss` s'est révélé **non fiable** ici
(il redescend d'un relevé à l'autre, ce qu'un maximum ne peut pas faire) —
tous les chiffres de mémoire ci-dessus viennent de `dumpsys`, pas de lui.

### 4. Trois bugs trouvés en route, qui n'étaient pas au programme

1. **Le pipeline décodait le même fichier six fois.** `duree`, aperçu de tête,
   battues, CQT, chroma, enveloppe : chaque étage rappelait le décodeur.
   Mesuré sur l'émulateur : 17,9 s rien que pour obtenir la DURÉE.
   Corrigé par (a) `MediaExtractor.KEY_DURATION` au lieu d'un décodage complet
   (17,87 s → **0,02 s**) et (b) un cache d'**une seule** entrée dans
   `_load` — une seule, parce que le pipeline demande 22050 quatre fois de
   suite puis 44100 puis 8000, et que garder deux signaux à la fois n'a pas de
   sens à 2 Go. Le signal rendu par le cache est marqué **non inscriptible** :
   deux étages le partagent, et une modification en place de l'un corromprait
   l'autre en silence.
2. **Deux threads décodaient le même fichier en même temps** (les battues et
   musx tournent en parallèle dans `analyze_steps`) et le décodeur natif
   **avortait le process** (SIGABRT). Un verrou autour du décodage, et c'est
   aussi la seule façon que le cache serve à quelque chose.
3. **L'aperçu de tête casse Beat This!** — le vrai défaut du portage.
   `_head_chart` analyse les 45 premières secondes ; le mel y fait ~1138
   trames, donc **moins** qu'un bloc, et le `.pte` en forme fixe refuse net :
   `Attempted to resize a static tensor. Expected shape (1, 1500, 128), but
   received (1, 1138, 128)`. Ni T1c (qui ne mesurait que des morceaux entiers,
   « tout bloc réel fait 1500 trames ») ni P2c (qui tournait avec `head_s=0`)
   ne pouvaient le voir. **Le Mac ferait exactement la même chose.**
   L'aperçu est **désactivé sur mobile** plutôt que rendu faux : bourrer le
   bloc de zéros jusqu'à 1500 changerait l'attention du transformer, donc les
   battues, sans rien signaler. La vraie correction est de router les blocs
   courts vers le `.pte` **dynamique**, déjà exporté par T1c et jamais exercé.

Et un quatrième, de plomberie : `harmonia/musx.py::redecode` (le décodage HMM)
fait `chdir` dans le clone musx vendu. Chaquopy garde les modules Python dans
une archive, pas en vrais fichiers — le clone a donc dû passer par les
**assets** et être recopié dans `filesDir`, avec `HARMONIA_MUSX_DIR` (une des
trois seules variables que lit `settings.py`) pour le désigner.

### 5. Parité, chart de l'émulateur contre chart du Mac, mesure par mesure

Référence = `mobile/tools/parity_out/reference/` (le chart du Mac produit par
`parity_pipeline.py run --backend reference`, caches froids, `head_s=0`).
Chart du téléphone rapatrié tel quel (`parity_pipeline.py pull`, qui lit le
FICHIER écrit par `server/jobs.py`, pas une route qui le reconstruirait).

| morceau | mesures | battues | max abs Δt battues | accords/mesure identiques | basse sonnante | sections | tonalité | tempo |
|---|---|---|---:|---:|---|---|---|---|
| maroon_5_this_love | 80 = 80 | 322 = 322 | **0,5 ms** | **98,75 %** (1 mesure) | 1 basse en plus | A (1) = A (1) | C minor = C minor | 95,06 = 95,06 |
| ben_e_king_stand_by_me_audio | 86 = 86 | 347 vs **348** | — (comptes ≠) | **100,00 %** | identique | A (1) = A (1) | A major = A major | 120,0 = 120,0 |
| maroon_5_she_will_be_loved… | 107 = 107 | 430 = 430 | **0,3 ms** | **99,07 %** (1 mesure) | 1 basse en plus | A (1) = A (1) | Bb major = Bb major | 101,99 = 101,99 |

**273 mesures comparées, 2 diffèrent** (99,27 %). Et les deux sont la MÊME
chose : une basse, pas un accord.

| morceau | mesure | Mac | émulateur |
|---|---:|---|---|
| this_love | 68 | `Bb` `Eb` | `Bb` `Eb/Bb` |
| she_will_be_loved | 63 | `Bb` | `Eb/Bb` `Bb` |

Dans les deux cas l'émulateur entend un `Eb` sur un `Bb` à la basse là où le
Mac ne l'écrit pas. Ce n'est pas « un accord faux » : c'est un renversement
écrit d'un côté et pas de l'autre, exactement la classe d'écart que P2c avait
déjà observée entre deux décodages du même fichier sur le Mac. **C'est un
avant/après à arbitrer par Louis, pas une régression** — et il ne se décide
pas au nombre.

Le compte de battues de Stand By Me (347 contre 348) est **le même phénomène
que P2c avait mesuré sans aucun téléphone** : sur le Mac, le même fichier
décodé par `ffmpeg` donne 348 battues et décodé par CoreAudio 345. Le
téléphone n'a qu'un décodeur ; il tombe sur 348. Zéro mesure ne change.

Et le chart s'ouvre pour de vrai dans l'app, pas seulement en JSON :
`mobile/android/smoke/p2d_chart_device.png` — « Maroon 5 This Love, C minor,
3:25 », la grille complète (`G/B · Cm · Fm7 · Dø7 …`), la barre de lecture et
les quatre onglets. Tout ce qui est sur cette capture a été calculé par le
téléphone.

### 6. À qui revient l'écart : décodeur ou arithmétique ARM

Les battues seules, comparées hors pipeline (`/api/debug/beats` sur
l'appareil, `compare_beats_device.py` sur le Mac avec le PCM de référence
figé) :

| morceau | battues Mac / émulateur | Δt médian | Δt max | temps forts Mac / émulateur |
|---|---|---:|---:|---|
| maroon_5_this_love | 322 / **322** | **0,000 ms** | 20,0 ms | 81 / **81** (max 20,0 ms) |
| ben_e_king_stand_by_me_audio | 348 / **348** | **0,000 ms** | 20,0 ms | 96 / 91 |
| maroon_5_she_will_be_loved… | 430 / **430** | **0,000 ms** | 20,0 ms | 108 / **108** (max 20,0 ms) |

(Mac = même PCM de référence figé, réseau `.pte` XNNPACK, runtime python.
Émulateur = PCM MediaCodec, `.pte` portable, runtime natif JNI. Les échantillons
décodés diffèrent de 12 sur les 3 morceaux : 4 526 189 contre 4 526 177,
3 913 728 contre 3 913 716, 5 917 632 contre 5 917 620.)

Lecture : **la médiane des écarts est 0,000 ms** et le maximum vaut une
trame du réseau (20 ms = 1/50 s). Autrement dit, le décodeur MediaCodec et le
`.pte` portable sur ARM rendent la MÊME grille que le Mac, à une battue près
qui bascule d'une trame — le même quasi-ex-æquo de logits que T1c avait déjà
caractérisé sur ce morceau. P2c avait montré qu'un simple changement de
décodeur, sur le Mac, suffisait à faire passer Stand By Me de 345 à 348
battues : **l'écart mesuré ici est nettement plus petit que celui-là.**

### 6 bis. Le rééchantillonneur de P2a-bis est DÉJÀ dans toutes ces mesures

Le brief demandait de rejouer la mesure si P2a-bis terminait pendant la
tâche. Vérifié plutôt que supposé : `AudioDecoder.kt` a été modifié à
**01:43** (β du Kaiser porté de 8,6 à 12,26), et mon premier `assembleDebug`
date de **01:55**. Les cinq APK que j'ai construits contiennent donc tous
cette version — **rien à rejouer**. C'est cohérent avec ce qu'on lit dans les
chiffres : P2a-bis annonce « décalage 0 échantillon partout », et les trois
comparaisons de battues ci-dessus donnent bien une médiane de 0,000 ms.

### 7. Temps par étape sur l'émulateur — et pourquoi ils ne valent pas pour le téléphone

L'émulateur tourne en arm64 natif sous l'hyperviseur d'Apple, mais avec
4 coeurs, 2 Go, un disque virtuel, et **sans XNNPACK** pour les battues. Le
téléphone de Louis n'est pas cette machine. Les chiffres sont donnés quand
même, parce qu'ils disent où va le temps :

Mesuré par les chronomètres de `mobile_timing.py` (une ligne dans logcat par
brique), caches froids, tuiles 256 pour musx et pour le chroma :

| morceau | durée audio | durée (`KEY_DURATION`) | CQT musx (décodage 22050 compris) | réseau musx (5 folds) | battues (mel + `.pte` PORTABLE + post) | enveloppe (décodage 8000) | chroma NNLS (décodage 44100 compris) | total mural |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ben_e_king_stand_by_me_audio | 177,5 s | 0,01 s | 13,15 s | 24,52 s | **210,84 s** | 5,40 s | 8,47 s | **226 s** |
| maroon_5_this_love | 205,3 s | 0,02 s | 8,74 s | 26,71 s | **244,45 s** | 6,24 s | 9,71 s | — |
| maroon_5_she_will_be_loved… | 268,4 s | 0,05 s | 11,86 s | 36,63 s | **354,59 s** | 7,99 s | 12,53 s | **406 s** |

(La ligne « battues » de This Love vient d'un passage froid séparé : le
passage final a relu le cache de battues, comme le ferait une ré-analyse.)

Décomposition du décodage, mesurée à part (`GET /api/debug/stages`) :
**11,2 s** pour décoder 205 s à 22050 Hz, **2,4 s** de CQT. Le décodage
MediaCodec est donc le deuxième poste après les battues — et c'est aussi
pour ça que les six décodages du même fichier (§4) coûtaient si cher.

**Ce que ça dit, et ce que ça ne dit pas.** Hors battues, tout le pipeline
tient en **69 s pour 4 min 28 d'audio**, soit ≈ **62 s pour 4 min** — la
moitié du budget de la porte T5, sur une machine qui n'est pas le téléphone.
Les battues, elles, prennent 1,3 s par seconde de musique **parce qu'elles
tournent sans XNNPACK** ; sur le Mac le même réseau avec XNNPACK fait 0,14 s
par seconde de musique, soit **presque 10× moins**. Le chiffre qui compte
n'existe pas encore : il faut le téléphone.

Mémoire, `TOTAL PSS` relevé toutes les 2 s par `dumpsys meminfo` :

| configuration | pic PSS | verdict |
|---|---:|---|
| aucun tuilage | — | **tué** par le système (1183 Mo au dernier relevé avant la mort) |
| tuiles musx 512, chroma non tuilé | 1947 Mo | survit un morceau, **tué** au suivant |
| tuiles 256 / 256, battues relues du cache | **546 Mo** | confortable |
| tuiles 256 / 256, battues calculées (`.pte` portable) | 1531–1630 Mo | passe, mais de justesse |

La dernière ligne est encore élevée : l'exécuteur **portable** d'ExecuTorch
matérialise tous les tenseurs intermédiaires du graphe, là où XNNPACK
planifie une arène. C'est une raison de plus de ne pas lire ces chiffres
comme ceux du téléphone.

### 8. Taille de l'APK et commande d'installation

| | octets | |
|---|---:|---|
| APK P1 (squelette, sans modèles) | 46 122 399 | 44 Mio |
| + AAR ExecuTorch + `.pte` + 5 `.npz` musx | 142 037 709 | 135 Mio |
| + clone musx vendu (assets) | **202 479 523** | **193 Mio** |

Dont 27 Mo de `cache_data` (`.sdict`) que le téléphone **n'ouvre jamais** :
`_musx_dir()` les exige pour se déclarer valide, et poser un fichier vide pour
faire passer cette vérification serait exactement le genre de mensonge qui
coûte cher ici. La vraie économie est une fente `musx_redecode` dans
`harmonia/backends.py` (voir « reste ouvert »).

```bash
cd mobile/android
export JAVA_HOME=/opt/homebrew/opt/openjdk@17 ANDROID_HOME=~/Library/Android/sdk
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# ÉMULATEUR SEULEMENT — le .pte sans XNNPACK, volontairement hors de l'APK.
# Sur un vrai téléphone, NE PAS le pousser : l'app prend alors le .pte
# XNNPACK embarqué, qui est celui qu'on veut mesurer.
adb push mobile/models/beat_this_final0_fixed1500_portable.pte /data/local/tmp/bt.pte
adb shell "run-as fr.harmonia.app sh -c 'cat > files/harmonia_models/beat_this_final0_fixed1500_portable.pte' < /data/local/tmp/bt.pte"
adb shell rm /data/local/tmp/bt.pte
adb shell am force-stop fr.harmonia.app && adb shell am start -n fr.harmonia.app/.MainActivity

# lancer une analyse depuis le Mac et récupérer le chart
adb forward tcp:7790 tcp:7772
curl -s -X POST http://127.0.0.1:7790/api/analyze -H 'Content-Type: application/json' \
     -d '{"url":"local:maroon_5_this_love"}'
.venv/bin/python mobile/tools/parity_pipeline.py pull --out device
.venv/bin/python mobile/tools/parity_pipeline.py compare --a reference --b device
```

## Ce qui reste ouvert

* **La vitesse des battues sur l'émulateur ne vaut rien** (`.pte` portable,
  sans XNNPACK). C'est la porte T5 qui tranchera, sur le téléphone. Tant
  qu'on n'a pas ce chiffre, **le budget « 4 min de musique en moins de 2 min »
  n'est ni tenu ni infirmé** : tout le reste du pipeline tient en ~55 s sur
  l'émulateur, les battues sont l'inconnue.
* **L'aperçu de tête est éteint sur téléphone.** Correction propre : router
  les blocs de moins de 1500 trames vers `beat_this_final0_dynamic.pte`
  (déjà exporté, jamais exercé). Il faut que `BeatsRuntime` sache tenir
  **deux** modules, et qu'on exporte aussi une variante portable du
  dynamique pour l'émulateur.
* **27 Mo de poids torch inutiles dans l'APK.** Une fente `musx_redecode`
  dans `harmonia/backends.py` supprimerait la dépendance au clone entier et
  ferait retomber l'APK autour de 165 Mio. Pas fait : ça touche `harmonia/`
  et sortait du périmètre du temps imparti.
* **`RuntimeWarning: overflow encountered in expm1`** dans `musx_net.py:83`
  pendant l'inférence sur ARM. Les accords sortent justes, donc le calcul
  masqué derrière un `np.where` est probablement correct — mais c'est
  exactement la forme d'un bug de calibration silencieux, et ça n'a PAS été
  vérifié. À regarder avant de croire une mesure fine sur téléphone.
* **`pull` passe par `run-as`**, qui n'existe que sur un APK de debug. Pour
  un APK signé, il faudra rapatrier le chart par la route HTTP.
* **La comparaison « PCM du device rejoué sur le Mac »** (route
  `/api/debug/dump_pcm`, écrite) n'a pas été jouée : les écarts de battues se
  sont révélés trop petits (médiane 0,000 ms) pour qu'il y ait quoi que ce
  soit à attribuer. À ressortir si un morceau diverge vraiment.
* **3 morceaux, pas 44**, et sur un émulateur. Le rapport d'or n'a pas été
  passé : il compare des charts du Mac entre eux.
* **Disque : 4,5 Go libres** en fin de tâche (seuil 3 Go, frôlé à 2,9 Go au
  moment du build le plus gros). `mobile/tools/parity_out/pcm/` a été réduit
  aux 3 fichiers `__22050beats.npy` encore utilisés ; le reste est
  régénérable par `dump_reference_pcm.py`.

## Fichiers produits

* `mobile/android/app/src/main/java/fr/harmonia/app/BeatsRuntime.kt` — le
  runtime ExecuTorch natif (charge le `.pte` en mmap, règle les threads
  XNNPACK, exécute un bloc). **Nouveau.**
* `mobile/android/app/src/main/python/mobile_beats_runtime.py` — le pont
  Chaquopy, même signature que `beats_infer._run_pte`. **Nouveau.**
* `mobile/android/app/src/main/python/mobile_timing.py` — chronomètres par
  brique + `GET /api/debug/timings`. **Nouveau.**
* `mobile/harmonia_mobile/beats_infer.py` — la fente `set_chunk_runner`.
* `mobile/harmonia_mobile/backends_mobile.py` — `beats_runtime=`,
  `models_dir=`, `pte_path=`, `musx_chunk_frames=`, `nnls_chunk_frames=`,
  le verrou + cache de décodage, `duration_seconds` par `KEY_DURATION`.
* `mobile/harmonia_mobile/musx_net.py` — CNN tuilé en temps
  (`chunk_frames`), bit-exact, −24 % de temps.
* `mobile/harmonia_mobile/nnls_chroma.py` — étage FFT tuilé
  (`chunk_frames`), bit-identique, −36 % de temps.
* `mobile/android/app/src/main/python/mobile_entry.py` — installe les
  backends mobiles AVANT le pipeline, choisit le `.pte`, pose
  `HARMONIA_MUSX_DIR`, éteint l'aperçu de tête.
* `mobile/android/app/src/main/python/mobile_debug.py` — routes
  `/api/debug/beats`, `/api/debug/stages`, `/api/debug/dump_pcm`.
* `mobile/android/app/src/main/java/fr/harmonia/app/MainActivity.kt` —
  `seedModelsOnce()`, `seedMusxOnce()` + copie d'assets récursive.
* `mobile/android/app/build.gradle.kts` — AAR ExecuTorch 1.4.0, assets
  modèles, assets clone musx, staging de `mobile/harmonia_mobile/`.
* `mobile/tools/export_beat_this.py` — mode `--portable`.
* `mobile/tools/parity_pipeline.py` — sous-commande `pull`.
* `mobile/tools/compare_beats_device.py` — battues appareil vs Mac. **Nouveau.**
* `mobile/android/smoke/p2d_chart_device.png` — le chart de This Love rendu
  dans l'app sur l'émulateur, calculé de bout en bout par l'appareil.
* `mobile/android/.gitignore` — les deux dossiers d'assets générés
  (`harmonia_models/`, `harmonia_musx/`, 122 Mo) exclus du suivi.
* `mobile/tools/mem_profile_mobile.py`, `mem_chunk_parity.py`,
  `mem_speed_chunk.py`, `nnlsmem_*.py` — les mesures de mémoire et de parité
  des deux tuilages. **Nouveaux.**
* `harmonia/server/jobs.py` — la durée passe par `pipeline.duration_seconds`
  (donc par `harmonia.backends`) au lieu d'un `ffprobe` en dur : c'était le
  dernier sous-processus du chemin d'analyse.
* `harmonia/pipeline.py` — docstring de `duration_seconds` mise à jour.
