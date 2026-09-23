# P4 — l'APK, la notice, la mesure T5 dans l'app (2026-09-23)

## Résultat en 3 lignes

**L'APK signé est livré : `~/harmonia/apk/harmonia-beta-2026-09-23.apk`, 160 Mio
(168 049 944 octets), sha256 `78a057fe…9f930`, `apksigner verify` vert avec
le keystore beta.** Retrait des 27 Mo de poids torch inutiles (168 vs 188 Mio à
sdict près, même build) ; détection d'émulateur qui refuse le `.pte` XNNPACK
AVANT le SIGILL, avec message clair — vérifiée en isolation (Mac, 4 scénarios)
PUIS en vrai sur l'émulateur, sans rien simuler : un fresh install sans override
portable a refusé de démarrer avec exactement ce message dans logcat, au lieu
du SIGILL muet (voir §5) ; mesure T5 posée dans le JSON du job et affichée une
fois dans l'écran d'analyse, avec un correctif de 2e passe (ci-dessous) qui
marque chaque étape lue d'un cache et préfixe la ligne par « ⚠ cache chaud —
pas une mesure » quand les trois briques de calcul viennent toutes d'un cache
— vérifié sur trois états réels (tout en cache, partiellement en cache, tout
froid après avoir vidé les caches). Deux réserves : `assembleRelease` marche
du premier coup (pas de repli debug nécessaire) mais **livrer un `.pte`
XNNPACK non testé au-delà de l'export** reste le vrai inconnu — l'émulateur ne
peut pas le charger (P2d) ; la porte T5 elle-même n'est PAS tranchée ici
(attend le téléphone).

## Ce qui a été vérifié (commandes, chiffres, chemins)

### 1. Détection d'émulateur — le bon `.pte` selon l'appareil

`mobile/android/app/src/main/python/mobile_entry.py::_looks_like_emulator()` :
`Build.FINGERPRINT`/`HARDWARE`/`MODEL` via Chaquopy (`from java import
jclass`), repli `/proc/cpuinfo` (`goldfish`/`ranchu`) si `java` n'est pas
joignable. `_install_mobile_backends` refuse le `.pte` XNNPACK et lève AVANT
`BeatsRuntime.load()`/`forward()` si aucun `.pte` PORTABLE n'est poussé ET
que l'appareil ressemble à un émulateur — remplace un SIGILL muet (P2d §2,
`rdsvl`, désassemblé) par un message qui dit la cause et le geste de
contournement.

Vérifié en isolation, SANS l'émulateur (import direct du module dans
`.venv`, `java` et `/proc/cpuinfo` simulés) :

| scénario | `_looks_like_emulator()` |
|---|---|
| `Build.FINGERPRINT` = AVD (`sdk_gphone64_arm64`, `HARDWARE=ranchu`) | **True** |
| `Build.FINGERPRINT` = téléphone réel (`samsung/a55x…`, `HARDWARE=qcom`) | **False** |
| `/proc/cpuinfo` contient `Hardware : ranchu` (repli, `java` absent) | **True** |
| ni l'un ni l'autre (le Mac) | **False** |

Et le refus complet, mêmes mocks, `models_dir` sans `.pte` portable :
```
ERROR harmonia.mobile.entry: Beat This!: .pte XNNPACK REFUSÉ — cet appareil
ressemble à un émulateur … Pousse le .pte portable …
RAISED: Beat This!: refus de charger le .pte XNNPACK sur un émulateur détecté …
```
Le refus arrive AVANT tout appel natif — pas de crash à masquer, l'exception
est la première chose qui se passe. L'outil de contournement (`.pte`
portable poussé à la main, jamais dans l'APK) reste documenté et inchangé
(P2d) ; ce commit ajoute seulement le refus explicite quand il est absent.

### 2. Alléger l'APK — les 27 Mo de poids torch inutiles

Vérifié avant de couper (pas supposé) : `_musx_dir()` n'était appelée que par
`redecode()` -> `_decoder()` (le décodage HMM, pur numpy/logique, lit
`data/{chord_dict}_chord_list.txt`) sur le chemin mobile — jamais par
`_nets()` (le passage torch, qui LUI ouvre `cache_data/*.sdict`, et que
mobile ne calcule jamais, voir `mobile/harmonia_mobile/musx_net.py`).
`harmonia/musx.py::_musx_dir()` corrigée pour exiger
`data/submission_chord_list.txt` au lieu de `cache_data/*.sdict` (sa
docstring documente le raisonnement complet). `build.gradle.kts::
copyMusxClone` exclut `cache_data/*.sdict` de la copie vers les assets — et a
dû passer de `Copy` à **`Sync`** : une build antérieure avait déjà recopié
les 5 fichiers, et `Copy` ne supprime jamais un fichier de destination
devenu hors-filtre (bug trouvé en vérifiant `unzip -l`, pas supposé résolu).

Taille de l'APK **release**, même build sinon (signature, `isMinifyEnabled
= false`) — la seule variable est la présence des `.sdict` :

| | octets | Mio |
|---|---:|---:|
| AVANT (avec les 5 `.sdict`, 27 Mo) | 196 781 287 | 187,7 |
| APRÈS (sans, `_musx_dir()` corrigée) | 168 048 992 | 160,3 |
| APRÈS + correctif cache-chaud (livré) | **168 049 944** | **160,3** |

`unzip -l app-release.apk \| sort -k1 -n \| tail -20` après coupe : plus
aucun fichier `cache_data/*` (`grep -i sdict` : 0 résultat) ; les 12 fichiers
les plus gros restants sont tous nécessaires — le `.pte` de Beat This!
(82,4 Mo, XNNPACK), les 5 `.npz` musx (2,0 Mo chacun), les 3 `.imy` Chaquopy
(stdlib + requirements + le code app, 18,9 + 4,3 + 28,2 Mo), `libexecutorch.so`
(8,9 Mo), `libpython3.12.so` (6,7 Mo), les 3 `.m4a` de seed (10 Mo au total,
intentionnels). Rien d'inattendu.

### 3. Signature de release

Keystore généré HORS dépôt (`keytool -genkeypair -keystore
~/harmonia/keys/harmonia-beta.jks -alias harmonia -keyalg RSA -keysize 2048
-validity 10000`), mot de passe aléatoire (`openssl rand -base64 24`), écrit
dans `~/harmonia/keys/README.txt`, permissions **600** sur le `.jks` et le
`.txt`, dossier `~/harmonia/keys` en **700**.

`mobile/android/app/build.gradle.kts` : bloc `signingConfigs.release` lu
depuis `mobile/android/keystore.properties` (les 4 clés :
`storeFile`/`storePassword`/`keyAlias`/`keyPassword`) — absent, il n'est
simplement pas déclaré (pas d'échec de configuration Gradle, seul
`assembleRelease` échouerait alors, proprement). `keystore.properties` ajouté
à `mobile/android/.gitignore` ; `keystore.properties.example` (committable,
valeurs `CHANGE_ME`) à côté.

`buildTypes.release { isMinifyEnabled = false }` — Chaquopy + réflexion
(Flask, `yt-dlp`, les modules `mobile.harmonia_mobile.*` importés par nom) :
R8 casserait des symboles nécessaires au bytecode Python. `assembleRelease`
a réussi du **premier coup**, aucune raison Chaquopy de replier sur un APK
debug.

Vérifié :
```
$ apksigner verify --print-certs app-release.apk
Signer #1 certificate DN: CN=Louis, OU=Harmonia, O=Harmonia, L=, ST=, C=FR
Signer #1 certificate SHA-256 digest: cbaa5f28323d2a700d64b1311637f4b93b1c36852f1b80353cf375d0ecd6c3b4
```
et `aapt2 dump badging` : pas de ligne `application-debuggable` sur l'APK
final (confirmé aussi de façon plus dure : `adb shell run-as fr.harmonia.app`
refuse dessus — "package not debuggable" — alors qu'il marchait sur la build
temporaire `isDebuggable=true` du smoke test).

**Commande de build reproductible** (depuis la racine du worktree) :
```bash
cd mobile/android
export JAVA_HOME=/opt/homebrew/opt/openjdk@17 ANDROID_HOME=~/Library/Android/sdk
./gradlew assembleRelease --console=plain
# -> app/build/outputs/apk/release/app-release.apk (signé, non minifié)
```
Préalable non répété ici (inchangé depuis P2d) : `mobile/models/*.pte`/`*.npz`
et les 3 `.m4a` de seed doivent exister sur disque (régénérables,
`mobile/tools/export_beat_this.py`, `dump_musx_weights.py`, ou copiés depuis
`docs/audio/`) — sinon les tâches `copyModels`/`copySeedAudio` échouent avec
un message qui dit lequel.

**Ce qui est signé avec quoi** : `app-release.apk` est signé avec
`~/harmonia/keys/harmonia-beta.jks` (alias `harmonia`, RSA 2048, auto-signé,
validité ~27 ans) via `signingConfigs.release`, lui-même lu depuis
`mobile/android/keystore.properties` (hors git). Aucun autre artefact livré
n'est signé (les screenshots et rapports ne le sont pas).

### 4. La mesure T5 dans l'app

`harmonia/backends.py::Backends.call` chronomètre DÉJÀ les 8 fentes des deux
côtés (Mac ET mobile) — rien à instrumenter de neuf. `harmonia/server/
jobs.py::_job_timings(t0)` (nouveau) : regroupe les 8 fentes en 5 postes du
vocabulaire de Louis (décodage/battues/accords/chroma/reste), ne rend
`dict` que si `all(getattr(BACKENDS, s) is not None for s in BACKENDS.SLOTS)`
— vrai UNIQUEMENT quand `backends_mobile.install()` a tourné (P2c, toujours
tous les 8), donc invisible sur Mac SANS code de plateforme séparé. Posé sur
`job["timings"]` aux deux points où `status="done"` (fin normale + repli
après échec du raffinement). `BACKENDS.reset_timings()` ajouté juste avant
`analyze_steps` (sinon les cumuls d'un job précédent s'additionneraient —
`Backends.timings` est un dict de process).

`harmonia/static/screens/analyse.js::paintLoading` (+8 lignes de JS, sous la
limite de 15) : si `done && j.timings`, une ligne sous "N mesures · M
sections" — `Analyse : 22 s (décodage 22 s, reste 0 s)` (accords/battues/
chroma omis si < 0,05 s, c'est-à-dire relus du cache). `L.meta` gagne
`flex-wrap:wrap` (CSS, 1 ligne) pour que cette ligne passe sous la première
au lieu de se tasser dedans.

**Vu en vrai, deux fois, sur l'émulateur** (build temporaire
`isDebuggable=true`, seul moyen de piloter la WebView réelle par Chrome
DevTools Protocol — `adb forward tcp:9222 localabstract:webview_devtools_
remote_<pid>`, `Runtime.evaluate` appelant `startAnalysis()` exactement comme
le ferait un tap) :

* Analyse **froide** (curl direct, hors WebView, premières mesures/battues/
  musx calculées) : `{"total_s": 264.58, "battues_s": 248.85, "accords_s":
  36.6, "chroma_s": 9.47, "decodage_s": 6.02, "reste_s": 0.0}` — sauvegardé
  dans `mobile/android/smoke/p4_cold_timings_this_love.json`.
* Ré-analyse **depuis l'écran réel** (cache chaud, battues/musx/chroma
  relus) : `{"total_s": 22.11, "decodage_s": 21.7, "reste_s": 0.41,
  "battues_s": 0, "accords_s": 0, "chroma_s": 0}` — capture d'écran
  `mobile/android/smoke/p4_release_chart.png`, la ligne « Analyse : 22 s
  (décodage 22 s, reste 0 s) » visible sous "80 bars · 1 section".

Les deux mesures sont cohérentes avec le mécanisme (rien fabriqué pour les
postes non recalculés) — c'est la preuve que le raccordement à `BACKENDS`
mesure vraiment ce qui tourne, pas un chiffre plaqué. **Aucune de ces deux
valeurs n'est la mesure T5** (portable, sans XNNPACK, pas le téléphone) — le
premier nombre (froid) montre juste que le mécanisme capte le vrai travail
quand il a lieu.

**Le deuxième exemple ci-dessus (22 s, décodage 22 s) était en réalité le
bug corrigé en §4 bis** : rien ne disait que battues/accords/chroma étaient
à 0 parce que RELUS d'un cache, pas parce que le calcul était rapide — Louis
aurait recopié cette ligne comme une mesure T5 valable. Laissé tel quel
ci-dessus (au lieu de corrigé après coup) parce que c'est exactement la trace
qui a fait trouver le problème.

### 4 bis. Correction cache-chaud (2e passe, après revue)

**Le signal exact, trouvé dans le code plutôt que deviné par un seuil** :
`beats.track()`, `musx.frame_posteriors()` et `nnls_features.
extract_bothchroma()` lisent tous leur cache disque (`harmonia/cache.py`,
kinds `"beats"`/`"musx_probs"`/`"nnls"`) et rendent AVANT d'appeler
`BACKENDS.call(...)` sur un coup de cache — la fente n'est alors jamais
invoquée, donc sa clé est ABSENTE de `BACKENDS.timings` (pas "proche de
zéro" : absente, testé par `"beats" not in t`, exact). `_job_timings` calcule
`battues_cache`/`accords_cache`/`chroma_cache` sur cette absence — décodage
n'a pas de cache par morceau (pas de drapeau pour ce poste).
`analyse.js::paintLoading` : chaque poste porte son tag `(cache)` s'il
s'applique, et si `battues_cache && accords_cache && chroma_cache`, la ligne
s'ouvre par `⚠ cache chaud — pas une mesure : `.

**Vérifié sur les trois états réels**, avec la même technique CDP que §4
(build temporaire `isDebuggable=true`, réanalyses lancées depuis la vraie UI,
pas curl) :

| état des 3 caches | résultat observé | capture |
|---|---|---|
| tous chauds (2e réanalyse, rien vidé) | `⚠ cache chaud — pas une mesure : Analyse : 31 s (battues 0 s (cache), accords 0 s (cache), chroma 0 s (cache), décodage 31 s, reste 0 s)` | `mobile/android/smoke/p4_hotcache_warning.png` |
| partiel (battues cache, musx/chroma froids — la mémoire `data/cache` de Chaquopy avait été réinitialisée par une réinstallation d'APK entre-temps, trouvaille annexe notée en "reste ouvert") | pas de ⚠ (pas tous cachés), chaque poste correct : `Analyse : 2 min 19 s (battues 0 s (cache), accords 1 min 8 s, chroma 43 s, décodage 28 s, reste 1 s)` | `mobile/android/smoke/p4_partial_cache.png` |
| tous froids (les 3 caches vidés à la main par `run-as … rm -rf` sur `files/chaquopy/AssetFinder/app/data/cache/{musx_probs,nnls_infer}` et `files/harmonia_state/cache/beats`) | pas de ⚠, temps réels (inflation par contention hôte assumée, portable+émulateur ne mesure jamais T5) : `Analyse : 14 min 13 s (battues 12 min 42 s, accords 1 min 47 s, chroma 50 s, décodage 41 s)` — JSON complet `mobile/android/smoke/p4_cold_verify_final.json` | `mobile/android/smoke/p4_cold_no_warning.png` |

Les trois captures viennent du MÊME morceau (This Love), dans l'ordre, sur la
même installation d'app — la seule variable est l'état des caches. Le
mécanisme réagit exactement comme prévu aux trois états, pas seulement à
l'état "tout ou rien" demandé initialement.

`docs/portage_mobile/INSTALLER.md` (section e) réécrite en conséquence : dit
explicitement qu'une ligne ⚠ ne compte pas, que la bibliothèque de départ
contient des charts mais PAS les caches de calcul (vérifié : `assets/
harmonia_seed/` ne contient que `charts/*.json` + `audio/*.m4a` +
`chart_meta.json`/`folders.json`, aucun cache de battues/accords) — donc la
toute première analyse de This Love sur un téléphone neuf EST froide.

### 5. Smoke test final

Procédure : release APK (temporairement `isDebuggable=true`), `.pte` PORTABLE
poussé par `run-as` (outil de dev documenté, P2d) → boot propre (logcat :
`Flask ready=true`, `pte chargé … PORTABLE … la VITESSE mesurée ici ne vaut
RIEN pour le téléphone`) → bibliothèque à 3 morceaux → analyse de This Love
déclenchée depuis la vraie UI (CDP, pas curl) → chart rendu, ligne de temps
visible → capture `mobile/android/smoke/p4_release_chart.png`. Puis
`isDebuggable` retiré, `assembleRelease` REFAIT (pas juste renommé), APK
final re-signé et re-vérifié (boot, bibliothèque à 3 morceaux, `run-as`
refuse bien dessus) avant copie vers `~/harmonia/apk/`.

Répété une 2e fois pour livrer le correctif §4 bis (même cycle : temporaire
`isDebuggable=true` → vérifications → retiré → `assembleRelease` final →
re-signé, re-vérifié, copié). **Trouvaille non cherchée, la meilleure preuve
du §1** : entre les deux cycles, un `adb uninstall` + réinstallation à neuf
de l'APK final (non-debuggable) a fait tourner le VRAI chemin de refus
d'émulateur en conditions réelles, pas simulées — sans `.pte` portable pour
la première fois depuis P2d, le démarrage a échoué avec exactement le message
prévu, dans le vrai logcat de l'app :
```
W python.stderr: Beat This!: .pte XNNPACK REFUSÉ — cet appareil ressemble à
  un émulateur (Build.FINGERPRINT/HARDWARE ou /proc/cpuinfo) …
E HarmoniaMain: startup failed
E HarmoniaMain: com.chaquo.python.PyException: RuntimeError: Beat This!:
  refus de charger le .pte XNNPACK sur un émulateur détecté …
    at <python>.mobile_entry._install_mobile_backends(mobile_entry.py:157)
```
Le §1 initial n'avait vérifié la détection qu'en isolation (Mac, `java`
simulé) — c'est ici, en marge de la 2e vérification, que le mécanisme complet
(Kotlin → Chaquopy → Python → exception → thread de démarrage qui l'attrape
→ WebView jamais chargée) s'est confirmé de bout en bout sur l'appareil.
`/api/library` répondait vide (connexion refusée) exactement parce que Flask
n'avait jamais démarré — pas un bug de cette vérification, la conséquence
directe et voulue du refus.

### 6. Disque

1re livraison : seuil 3 Go, frôlé à **3,1 Gio** au pire moment (juste après
avoir copié l'APK dans `~/harmonia/apk/`, avant nettoyage). 2e livraison
(correctif §4 bis, cycle de build séparé) : jamais descendu sous **5,2 Gio**
— le nettoyage systématique après chaque `assembleRelease` (§5, un seul APK
intermédiaire vivant à la fois sous `app/build/outputs/apk/` et sous
`app/build/intermediates/**apk**/`, supprimé dès sa copie vers
`~/harmonia/apk/` ou dès qu'un build suivant le remplace) a suffi les deux
fois ; `./gradlew clean` n'a jamais été utilisé (aurait tout reconstruit).
`~/.gradle` (3,1 Go de cache de dépendances) laissé intact les deux fois —
jamais lui qui a menacé le seuil.

## Ce qui reste ouvert

* **La porte T5 elle-même n'est pas tranchée.** Ni l'émulateur (pas de
  XNNPACK) ni ce rapport ne peuvent le faire — ça attend le téléphone de
  Louis et le protocole en section (e) de `INSTALLER.md`.
* **Le `.pte` XNNPACK embarqué n'a jamais tourné sur du vrai matériel ARM
  avec SME réellement désactivé/activé.** T1c l'a validé bit-exact sur le
  Mac (runtime python) ; P2d a validé le runtime NATIF mais avec le `.pte`
  PORTABLE (l'émulateur ne peut pas faire mieux). Le chemin XNNPACK natif +
  `.pte` réel + téléphone réel n'est vérifié nulle part avant l'installation
  de Louis — c'est exactement ce que T5 doit trancher.
* **`isDebuggable=true` a dû revenir pour le smoke test**, seul moyen connu
  de pousser le `.pte` portable sur un APK signé non-debuggable via
  `run-as`. Retiré avant l'APK livré (vérifié : `run-as` y échoue bien), mais
  ça veut dire que le smoke test final tel que reproduit ici NE PEUT PAS
  courir tel quel sur l'APK réellement livré — un futur test devra soit
  rajouter `isDebuggable=true` temporairement (et le retirer après, comme
  ici), soit construire une route HTTP de dev pour pousser le `.pte`
  portable sans `run-as` (pas fait, hors budget).
* **`RuntimeWarning: overflow encountered in expm1`** (musx_net.py, notée
  par P2d) toujours pas regardée.
* ~~Détection d'émulateur non exercée EN VRAI sur l'AVD~~ — **résolue en marge
  de la 2e vérification** (§5 bis) : un fresh install sans `.pte` portable a
  déclenché le vrai refus, message identique à celui prédit en isolation.
* **`data/cache` (Chaquopy `AssetFinder/app/data/cache`, où vivent les caches
  musx_probs/nnls) ne survit pas à une réinstallation d'APK** — trouvaille
  annexe du §4 bis, pas cherchée : `beats_dir` (sous `HARMONIA_STATE_DIR`,
  stable) a gardé son cache entre deux `adb install -r`, mais `data_cache`
  (`REPO/data/cache`, où `REPO` dépend de `__file__` dans l'archive Chaquopy)
  a été réinitialisé. Effet pratique pour Louis : après une future mise à
  jour de l'app, un morceau déjà analysé aura un ⚠ pour les battues mais PAS
  pour les accords/chroma — surprenant si on ne sait pas pourquoi, sans
  gravité (juste un peu plus lent que prévu). Pas creusé plus loin, hors
  budget de cette tâche.
* **27 Mo retirés, pas 92 Mo.** L'APK reste dominé par les poids réellement
  utilisés (82,4 Mo `.pte` + 10 Mo `.npz` musx) — aucune autre coupe
  identifiée sans perdre une capacité (la stdlib/requirements Chaquopy et le
  code applicatif ne se compressent pas plus).
* **`docs/audio/` et `mobile/models/` restent gitignorés**, donc un clone
  frais du dépôt ne peut PAS reproduire cet APK sans d'abord régénérer ces
  fichiers (chemin documenté dans le commentaire Gradle, pas répété ici).

## Fichiers produits

* `~/harmonia/apk/harmonia-beta-2026-09-23.apk` + `.sha256` — **l'APK livré**
  (hors dépôt).
* `~/harmonia/keys/harmonia-beta.jks` + `README.txt` — **le keystore** (hors
  dépôt, 600).
* `mobile/android/keystore.properties` (gitignoré, nouveau) — pointe vers le
  keystore ci-dessus.
* `mobile/android/keystore.properties.example` (nouveau, committable).
* `mobile/android/.gitignore` — `+keystore.properties`.
* `mobile/android/app/build.gradle.kts` — `signingConfigs.release`,
  `buildTypes.release` (signature + `isMinifyEnabled=false`), `copyMusxClone`
  passé de `Copy` à `Sync` + exclusion `cache_data/*.sdict`, `versionName`
  mis à jour.
* `harmonia/musx.py` — `_musx_dir()` n'exige plus `cache_data/*.sdict`
  (docstring étendue avec le raisonnement).
* `mobile/android/app/src/main/python/mobile_entry.py` —
  `_looks_like_emulator()` (nouveau) + refus explicite du `.pte` XNNPACK sur
  émulateur détecté.
* `mobile/android/app/src/main/java/fr/harmonia/app/MainActivity.kt` —
  commentaire de `seedMusxOnce` mis à jour (les `.sdict` ne sont plus
  embarqués).
* `harmonia/server/jobs.py` — `_job_timings(t0)` (nouveau), `BACKENDS`
  importé, `reset_timings()` avant `analyze_steps`, `job["timings"]` posé aux
  deux points `status="done"` ; **correctif 2e passe** :
  `battues_cache`/`accords_cache`/`chroma_cache` (absence de la fente dans
  `BACKENDS.timings`, signal exact).
* `harmonia/static/screens/analyse.js` — la ligne "Analyse : …" (8 lignes
  JS) + `flex-wrap` sur `L.meta` ; **correctif 2e passe** : tag `(cache)` par
  poste + préfixe `⚠ cache chaud — pas une mesure` si les 3 briques de calcul
  sont toutes en cache (9 lignes JS, toujours sous la limite de 15).
* `mobile/android/smoke/p4_release_chart.png` — le chart + la ligne de temps,
  rendus par l'APK release sur l'émulateur.
* `mobile/android/smoke/p4_cold_timings_this_love.json` — le job JSON complet
  d'une analyse froide (battues/accords/chroma non nuls), 1re passe.
* `mobile/android/smoke/p4_hotcache_warning.png` — les 3 briques en cache,
  le ⚠ visible. **Nouveau.**
* `mobile/android/smoke/p4_partial_cache.png` — battues en cache, musx/chroma
  non : pas de ⚠, chaque poste tagué correctement. **Nouveau.**
* `mobile/android/smoke/p4_cold_no_warning.png` + `p4_cold_verify_final.json`
  — les 3 caches vidés à la main, analyse réellement froide, pas de ⚠,
  temps réels. **Nouveaux.**
* `docs/portage_mobile/INSTALLER.md` (nouveau, réécrit en 2e passe — section
  (e) explique le ⚠ et pourquoi la 1re analyse d'un téléphone neuf est
  froide) — la notice pour Louis.
* `docs/portage_mobile/p4_apk_release.md` — ce fichier.
