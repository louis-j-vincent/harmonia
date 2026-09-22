# P1 — squelette Android (2026-09-23)

## Résultat en 3 lignes

`./gradlew assembleDebug` est vert : APK debug de 44 Mo (`fr.harmonia.app`,
Chaquopy 17 + Python 3.12 + numpy/flask, arm64-v8a seul, minSdk 26). Preuve
la plus forte obtenue : la copie EXACTE de `harmonia/` embarquée dans l'APK
(filtrée, 3 fichiers exclus) s'importe et sert `/api/library` + `/` avec
SEULEMENT numpy+flask installés — torch/scipy/vamp/yt-dlp/pyRealParser sont
tous importés à l'intérieur de fonctions dans ce paquet, jamais au niveau
module, donc rien ne bloque l'import du serveur en phase 1. Pas de capture
d'écran émulateur : l'AVD `harmonia_35` refuse de démarrer sur cette machine
(besoin de 12 Go pour la partition userdata, 3,7-5,2 Go dispo pendant la
session) — constaté après coup, ce n'est pas un problème du projet Android.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Environnement**, vérifié avant de builder :
- `JAVA_HOME=/opt/homebrew/opt/openjdk@17` → `openjdk version "17.0.20.1"`.
- `~/Library/Android/sdk/platforms/android-35` et `build-tools/35.0.0`
  présents dès le début de la tâche.
- `df -h /System/Volumes/Data` : 9,8 Go libres au départ → 3,7-5,2 Go après
  l'installation de l'AVD `harmonia_35` par un autre agent en cours de
  route (~4,8 Go) + le premier `assembleDebug` (~600 Mo dans `~/.gradle`,
  conforme à l'estimation du brief). Jamais passé sous le seuil de 3 Go
  pendant que j'avais la main ; surveillé à chaque étape lourde.

**Build** :
```
cd mobile/android
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export ANDROID_HOME=~/Library/Android/sdk
./gradlew assembleDebug
```
→ `BUILD SUCCESSFUL`, `app/build/outputs/apk/debug/app-debug.apk`, 44 Mo
(46 122 399 octets). `local.properties` (non committé, `.gitignore` ajouté)
pointe `sdk.dir` vers le SDK ci-dessus.

**Installation (téléphone/émulateur, non exécutée faute d'appareil — voir
"ce qui reste ouvert")** :
```
adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n fr.harmonia.app/.MainActivity
adb logcat -s HarmoniaMain
```

**Deux problèmes de tooling résolus en cours de route** (à consigner pour
la suite du portage, personne d'autre ne les a rencontrés) :
1. Pas de `gradle` système sur la machine → wrapper bootstrappé à la main :
   `gradle-wrapper.jar`/`gradlew`/`gradlew.bat` téléchargés directement
   depuis `raw.githubusercontent.com/gradle/gradle/v8.10.2/...` (le jar du
   wrapper est un fichier fixe par version, pas besoin d'installer Gradle
   pour l'obtenir). `./gradlew --version` confirme Gradle 8.10.2.
2. Chaquopy a migré de `chaquo.com/maven` vers **Maven Central** à partir de
   la 13.x — la doc historique pointe encore l'ancien repo. Vérifié via
   `curl https://repo1.maven.org/maven2/com/chaquo/python/gradle/maven-metadata.xml`
   → dernière version `17.0.0` (2025-11-30), Python 3.10-3.14 supportés,
   AGP compatible 7.3-9.2, minSdk 24. Utilisé AGP 8.7.3 (dans la plage) et
   Kotlin 2.0.21 (pour compiler `MainActivity.kt` — plugin Kotlin absent de
   ma première ébauche, `compileDebugJavaWithJavac NO-SOURCE` sans lui,
   donc le `.kt` n'était tout simplement pas compilé : `classes.dex` de
   34 Ko au lieu de 2,36 Mo après correction).
3. **Bug de syntaxe Kotlin instructif** : un commentaire KDoc contenant
   littéralement la sous-chaîne `/**` dans son texte (`"harmonia_seed/**
   vers filesDir"`) a ouvert un commentaire bloc IMBRIQUÉ — contrairement à
   Java/C, **Kotlin imbrique `/* */`** — laissant un niveau ouvert jusqu'à
   la fin du fichier (`Syntax error: Unclosed comment` à la dernière
   ligne). Corrigé en reformulant la phrase.

**Preuve d'auto-suffisance des imports** (la vérification demandée dans le
brief, `create_app().test_client().get("/api/library")`), faite deux fois :
d'abord sur `harmonia/` complet du worktree, puis — plus fort — sur la copie
EXACTE filtrée que Gradle embarque (`app/build/harmoniaStage`, exclusions
appliquées), dans un venv Python 3.12 propre avec SEULEMENT `numpy<2.5` et
`flask` installés (rien d'autre — pas de torch, scipy, vamp, librosa,
yt-dlp, pyRealParser) :
```
$ HARMONIA_STATE_DIR=<fake_phone_state> HARMONIA_AUDIO_DIR=<fake_phone_audio> \
  python -c "from harmonia.server.app import create_app; ..."
SELF-SUFFICIENCY TEST: /api/library status 200
charts: ['min_ben_e_king_stand_by_me_audio',
         'min_maroon_5_she_will_be_loved_official_music_video',
         'min_maroon_5_this_love']
index status 200
```
Aussi testé : `/api/chart-model/min_maroon_5_this_love` → 200 (53 140
octets), `/audio/maroon_5_this_love.m4a` avec `Range: bytes=0-100` → 206
avec `Content-Range` correct. **Pourquoi ça marche sans torch/scipy/vamp/
yt-dlp/pyRealParser installés** : grep exhaustif des imports de
`harmonia.server.app` et de tout ce qu'il enregistre (les 7 blueprints) —
`beats.py`, `musx.py`, `nnls_features.py`, `youtube.py`,
`sections/similarity.py` (scipy), `integrations/*` (pyRealParser) importent
TOUS leurs dépendances lourdes À L'INTÉRIEUR de fonctions, jamais au niveau
module (norme du paquet, confirmée par CLAUDE.md). Seule
`harmonia/sections/simulation.py`, `soudure.py`, `folding.py` etc. (logique
pure) sont sur le chemin d'import forcé de `library.py` — et ils n'ont pas
de dépendance lourde.

**Ce que `library.py` force à embarquer, contredisant partiellement
l'audit** : `library.py` importe `sections.py` (route) au niveau module
(`from harmonia.server.routes.sections import _safe_stem`), qui importe à
son tour `harmonia.sections.simulation`. La table (b) de l'audit classait
la route `sections.py` (+ `soudure.py`, `section_tool.py`, `phrases4.py`,
`refold.py` — 2 915 lignes) comme "accessoire, pas nécessaire pour écouter
un chart" — vrai fonctionnellement, FAUX pour l'import Python : sans ces
fichiers, `import harmonia.server.app` échoue tout court. Embarqués donc,
malgré l'étiquette "accessoire".

**Modules exclus de la copie embarquée** (Gradle task `stageHarmoniaPython`,
`app/build.gradle.kts`) — les 3 seuls confirmés morts par grep exhaustif
(aucune référence, nulle part dans `harmonia/`, en dehors de leur propre
fichier) :
- `harmonia/sections/nouveaute.py` (500 lignes)
- `harmonia/models/harmonic_downbeat.py` (490 lignes)
- `harmonia/models/local_key_context.py` (153 lignes)

Tout le reste de `harmonia/` (1,8 Mo source → 1,4 Mo une fois filtré) est
embarqué tel quel, y compris `ssm_page.py` (import paresseux depuis
`sections.py`, route `/ssm` jamais appelée en phase 1 mais présente) et les
intégrations iReal/UG (`harmonia/integrations/*`, pyRealParser non
installé — la route `/api/irealb-*` échouerait avec un `ImportError` propre
si appelée, cohérent avec "hors périmètre P1", jamais rewrite des routes).
`docs/pwa/` (4 fichiers, icônes + manifest) copié à côté de `harmonia/` dans
le même dossier généré, pour que `SETTINGS.repo / "docs" / "pwa"`
résolve correctement une fois extrait sur le téléphone.

**Modification à `harmonia/settings.py`** (SEUL fichier de `harmonia/`
touché, comme demandé) : deux nouvelles variables d'environnement,
`HARMONIA_STATE_DIR` et `HARMONIA_AUDIO_DIR`, lues UNE FOIS au niveau module
(`_STATE_ROOT`, `_AUDIO_ROOT`), avec pour défaut les chemins Mac inchangés
(`REPO/state`, `REPO/docs/audio`) — donc **zéro changement de comportement
sur le Mac**. Tous les champs `state_dir`/`human_dir`/`cache_dir`/
`charts_dir`/`beats_dir`/`songformer_dir`/`reports_dir`/`sections_dir`/
`sections_draft_dir`/`annotations_dir`/`marks_dir`/`chart_meta_path`/
`folders_path`/`audio_dir` dérivent maintenant de ces deux racines au lieu
d'être chacun câblé en dur sur `REPO / "state" / ...`. Note ajoutée dans la
docstring du module expliquant pourquoi (Android : `REPO` est en lecture
seule dans l'APK). `data_cache` (`REPO/data/cache`) n'a pas été touché —
hors chemin serveur, pas concerné par l'écriture sur téléphone.

**Point d'entrée Python mobile** (`mobile/android/app/src/main/python/
mobile_entry.py`, PAS dans `harmonia/`) : `start_server(state_dir, audio_dir,
port)` pose les 3 variables d'env (+ `HARMONIA_PORT`), crée les
sous-dossiers d'état que `settings.py` documente explicitement ne pas
créer lui-même, importe `harmonia.server.app.create_app()` APRÈS avoir posé
l'environnement, et lance `app.run(host="127.0.0.1", port=port,
threaded=True, use_reloader=False)` dans un thread daemon.

**Copie des assets au premier lancement** (`MainActivity.kt`,
`seedStateOnce()`) : les assets Android (`assets/harmonia_seed/`, dans
l'APK) ne sont PAS des chemins fichier — copiés via `AssetManager` vers
`context.getFilesDir()/harmonia_state` et `.../harmonia_audio` une seule
fois (marqueur `.seeded`). 3 morceaux de seed choisis :
`maroon_5_this_love`, `ben_e_king_stand_by_me_audio`,
`maroon_5_she_will_be_loved_official_music_video` (chart JSON `min_*.json`
+ `.m4a` + extrait de `chart_meta.json`, copiés depuis le dépôt principal —
PAS depuis `~/harmonia/`, le clone à ne jamais toucher). 10 Mo au total dans
`app/src/main/assets/harmonia_seed/`.

**WebView / réseau** : `javaScriptEnabled=true`, `domStorageEnabled=true`,
`mediaPlaybackRequiresUserGesture=false`, `network_security_config.xml`
limite le cleartext à `127.0.0.1`/`localhost` uniquement (j'avais d'abord
mis `android:usesCleartextTraffic="true"` en plus au niveau `<application>`
— contradictoire avec "127.0.0.1 uniquement", retiré). `MainActivity`
sonde `GET /api/library` toutes les 200 ms (timeout 20 s) avant de charger
`http://127.0.0.1:7772/` dans la WebView — 7772 est SUR LE TÉLÉPHONE ici,
jamais le Mac (PLAN.md).

## Ce qui reste ouvert

- **Pas de vérification sur device réel.** `./gradlew assembleDebug` est la
  porte de sortie explicitement acceptée par le brief quand "pas
  d'émulateur pour l'instant (disque)" — situation confirmée : l'AVD
  `harmonia_35` (installé par un autre agent pendant cette tâche) refuse de
  démarrer ici, `FATAL: Not enough space to create userdata partition.
  Available: 5364.80 MB ... need 12288.00 MB` (log emulator). Le disque a
  ensuite continué de descendre (3,7-3,9 Go en fin de session) avec le
  build — je n'ai RIEN supprimé pour en libérer (hors périmètre, pas
  d'autorisation). Donc : ni capture d'écran, ni `adb logcat` réel, ni
  confirmation que la WebView affiche effectivement la bibliothèque — tout
  ce qui précède est vérifié par un test Python équivalent (self-sufficiency
  test ci-dessus), pas par l'app qui tourne.
- `adb install` et le lancement sur le téléphone de Louis restent à faire
  quand l'appareil sera branché (semaine du 2026-09-28 selon PLAN.md) — ou
  sur un émulateur une fois > 12 Go libres.
- Rien n'a été testé sur ce qui casse visuellement en WebView Android par
  rapport à Safari iOS (100dvh, safe-area, audio) faute de pouvoir afficher
  quoi que ce soit — reporté à quand un device sera disponible.
- Les routes accessoires embarquées mais non fonctionnelles en l'état
  (`/api/irealb-*` sans pyRealParser installé, `/ssm` dont le module est
  présent mais jamais testé ici) n'ont pas été vérifiées individuellement —
  seules `/`, `/static/*`, `/api/library`, `/api/chart-model/<f>`,
  `/audio/<f>` (celles du périmètre v1, PLAN.md) l'ont été.
- `chaquopy { defaultConfig { version = "3.12" ... } }` a réussi à
  télécharger un numpy prébuilt (`numpy-1.26.2-0-cp312-...`, pas la version
  la plus récente) depuis l'index Chaquopy dédié
  (`chaquo.com/pypi-13.1`) — normal, Chaquopy maintient son propre miroir
  de wheels Android ; pas un problème, juste à savoir si une version numpy
  précise est un jour requise.

## Fichiers produits

- `mobile/android/` — projet Gradle complet : `settings.gradle.kts`,
  `build.gradle.kts`, `gradle.properties`, `gradlew`(`.bat`),
  `gradle/wrapper/`, `.gitignore`, `app/build.gradle.kts` (Copy task
  `stageHarmoniaPython` + config Chaquopy/Android),
  `app/src/main/AndroidManifest.xml`,
  `app/src/main/java/fr/harmonia/app/MainActivity.kt`,
  `app/src/main/python/mobile_entry.py`,
  `app/src/main/res/{values,xml}/*`,
  `app/src/main/assets/harmonia_seed/{charts,audio,chart_meta.json,folders.json}`
  (10 Mo, 3 morceaux).
- `harmonia/settings.py` — modifié (2 nouvelles racines d'environnement,
  voir ci-dessus), seul fichier de `harmonia/` touché.
- `app/build/outputs/apk/debug/app-debug.apk` — 44 Mo, généré (pas
  committé, dans `.gitignore`).
- Ce fichier + une ligne dans `docs/portage_mobile/JOURNAL.md`.
