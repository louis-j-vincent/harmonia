# P1 — squelette Android (2026-09-23)

## Résultat en 3 lignes

**Marche de bout en bout sur l'émulateur** : `./gradlew assembleDebug` vert
(APK 44 Mo), `adb install` + lancement → Flask démarre dans l'app, la
WebView affiche la bibliothèque avec les 3 morceaux de seed (captures :
`mobile/android/smoke/screen_ok.png`, `screen_library.png`) et
`adb forward tcp:7790 tcp:7772` + `curl /api/library` répond 200 depuis le
Mac. Premier smoke test blanc (écran crème vide) diagnostiqué et corrigé :
CE N'ÉTAIT PAS un bug Android/Chaquopy, mais `chart.js` important
`pedalGlyph` d'un `kit.js` qui ne l'exportait pas — la fonction existait
mais sur une AUTRE branche jamais mergée (`feat/import-ireal`,
2026-09-18) ; portée ici verbatim (13 lignes, aucune autre modification de
kit.js). Ajouté au passage : un relais `WebChromeClient.onConsoleMessage`
(tag `harmonia.js`) — sans lui, l'erreur JS était invisible dans logcat.

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

**Installation + run sur l'émulateur** (AVD `harmonia_35` — a fini par
démarrer après coup, voir "ce qui reste ouvert" pour le premier échec
disque) :
```
adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n fr.harmonia.app/.MainActivity
adb logcat -s HarmoniaMain:V python.stdout:V python.stderr:V harmonia.js:V
```

**Premier passage — écran blanc.** Logcat (00:37:49) montrait Flask levé
(`Serving Flask app 'harmonia.server.app'`, `HarmoniaMain: Flask ready=true`,
`GET /api/library 200`) et tous les `.js` récupérés (304/200), donc le
pipeline Chaquopy → Flask → WebView marchait — mais rien à l'écran
(`screenshot_p1.png`, `smoke/screen_40s.png`, `smoke/screen_70s.png`, fond
crème uni). La ligne qui manquait : `chromium: [INFO:CONSOLE(15)] "Uncaught
SyntaxError: The requested module '../ui/kit.js' does not provide an export
named 'pedalGlyph'", source: .../screens/chart.js`. `router.js` importe
`chart.js` de façon STATIQUE (`router.js:9`) : une erreur de résolution
d'un module ES à l'import bloque tout le graphe avant le premier rendu,
bibliothèque comprise — quel que soit l'écran demandé.

**Root cause, pas un résidu du portage.** `git show HEAD:harmonia/static/ui/
kit.js` (commit de base `71f66fd`) ne contient `pedalGlyph` nulle part ;
`git log -p -S pedalGlyph --all` la retrouve implémentée dans
`e8a486665233` (2026-09-18) sur la branche `feat/import-ireal` — **jamais
mergée** dans la lignée de cette branche (confirmé : `git merge-base
--is-ancestor e8a4866… HEAD` → non-ancestor). Pendant ce temps, `chart.js`
sur CETTE lignée (commit `b0b7053d`, même jour) a commencé à IMPORTER
`pedalGlyph` sans que l'implémentation suive — deux sessions parallèles le
18/09, jamais réconciliées. **Un navigateur Mac sur ce commit afficherait le
même écran blanc** — pas spécifique à Android.

**Corrigé** (hors du périmètre strict "ne touche qu'à MainActivity/WebView/
mobile_entry.py/manifest", mais nécessaire pour la porte de sortie
demandée, et vérifié sans risque de collision — `git status` montrait
`kit.js`/`chart.js` propres, aucune édition en cours d'une session
concurrente) : `pedalGlyph` copiée verbatim depuis `feat/import-ireal`
(13 lignes, une fonction, aucun autre changement) dans
`harmonia/static/ui/kit.js`, juste avant `export function glyph`. Validé
`node --check` sur les deux fichiers avant et après. Rebuild + reinstall +
relaunch : plus aucune ligne `harmonia.js`/`chromium CONSOLE` d'erreur,
`GET /static/ui/kit.js` repasse en 200 (plus de cache 304), l'app affiche
l'accueil puis « Mes charts » avec les 3 morceaux exacts et leurs bonnes
métadonnées (`Ben E. King / Stand By Me — A major · 87 bars`, `Maroon 5 /
She Will Be Loved — B♭ major · 107 bars`, `Maroon 5 / This Love — C minor ·
80 bars`) — `smoke/screen_ok.png` et `smoke/screen_library.png`.

**Vérification `/api/library` depuis le Mac** (comme demandé, jamais par
127.0.0.1:7772 côté Mac directement) :
```
adb forward tcp:7790 tcp:7772
curl http://127.0.0.1:7790/api/library   # → 200, les 3 charts, JSON complet
adb forward --remove tcp:7790
```

**Ajout diagnostique permanent** : `MainActivity.kt` installe désormais un
`WebChromeClient` avec `onConsoleMessage` → `Log.i("harmonia.js", …)` —
sans lui, cette erreur JS était invisible dans `adb logcat` (Chromium ne
relaie pas la console par défaut). Indispensable pour la suite du portage.

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

- **L'AVD a d'abord refusé de démarrer, disque.** Premier essai :
  `FATAL: Not enough space to create userdata partition. Available:
  5364.80 MB ... need 12288.00 MB` (l'installation de l'AVD par un autre
  agent avait fait tomber le disque à 3,7-5,2 Go). Un second essai, plus
  tard dans la session, a démarré (la partition userdata s'est avérée
  passée à 4 Go — le disque avait entre-temps remonté à ~10 Go, purge de
  purgeable APFS probablement). Depuis, le smoke test complet est passé.
- `adb install` et le lancement sur le téléphone RÉEL de Louis restent à
  faire quand l'appareil sera branché (semaine du 2026-09-28 selon
  PLAN.md) — tout ce qui précède est vérifié sur l'émulateur `harmonia_35`
  (Pixel 6, android-35 google_apis, arm64-v8a), pas sur un device physique.
- **Le fix `pedalGlyph` corrige l'écran blanc, pas le fond du problème** :
  `feat/import-ireal` (2026-09-18, "construit + EN PROD sur :7772")
  contient probablement d'AUTRES changements à `kit.js`/`chart.js`/ailleurs
  jamais mergés dans cette lignée — je n'ai porté QUE la fonction manquante
  qui bloquait le chargement, pas fait un diff complet des deux branches.
  Un futur écran (pas juste la bibliothèque) pourrait révéler d'autres
  trous du même genre. À signaler à Louis : sa branche `feat/import-ireal`
  attend toujours un merge.
- Rien n'a été testé sur ce qui casse visuellement en WebView Android par
  rapport à Safari iOS au-delà de la page d'accueil et de la bibliothèque
  (pas encore ouvert un chart individuel, pas testé l'audio, le clavier,
  les gestes tactiles, `100dvh`/safe-area sur un vrai clavier ou une
  rotation) — le screenshot montre une UI qui rend correctement en 1080×2400
  portrait, rien de plus vérifié.
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
  voir ci-dessus), seul fichier de `harmonia/` touché dans le plan initial.
- `harmonia/static/ui/kit.js` — modifié HORS plan initial, pour débloquer
  le smoke test : `pedalGlyph` (13 lignes) portée verbatim depuis
  `feat/import-ireal` (commit `e8a486665233`), rien d'autre changé.
- `mobile/android/app/src/main/java/fr/harmonia/app/MainActivity.kt` —
  ajout du relais `WebChromeClient.onConsoleMessage` (tag `harmonia.js`).
- `mobile/android/smoke/screen_ok.png`, `screen_library.png` — captures du
  smoke test réussi (accueil, puis « Mes charts » avec les 3 morceaux).
  `screen_40s.png`/`screen_70s.png`/`logcat.txt`/`emulator.log` dans le même
  dossier sont les artefacts du premier passage (écran blanc, avant fix).
- `app/build/outputs/apk/debug/app-debug.apk` — 44 Mo, généré (pas
  committé, dans `.gitignore`).
- Ce fichier + une ligne dans `docs/portage_mobile/JOURNAL.md`.
