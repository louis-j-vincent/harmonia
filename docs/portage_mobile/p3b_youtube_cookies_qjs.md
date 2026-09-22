# P3b — import YouTube sur téléphone : cookies + QuickJS, mesuré de bout en bout (2026-09-23)

## Résultat en 3 lignes

**Les deux dépendances marchent, il y en avait une TROISIÈME (script EJS à
mettre en cache), et une fragilité mesurée plus sévère que prévu (cookies
rotés en ~20 min).** Sur l'émulateur : cookies statiques + QuickJS-ng compilé
NDK (1,20 Mo) + `remote_components=["ejs:github"]` → `191P7nIeECo.m4a`
téléchargé (2 507 233 octets, identique au Mac), titre + artiste résolus
("READY (Acoustic Version)", PJ Morton), ~30 s. Le job échoue ensuite sur
`ffprobe` absent — attendu, hors périmètre P3b.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Étape 0 (Mac, `.venv-mobile`, yt-dlp 2026.08.19)** — `brew install
quickjs-ng` (1 Mo, `qjs` = 72 Ko, lié à `libqjs.0.dylib`) ; cookies export
`yt-dlp --cookies-from-browser firefox --cookies /tmp/yt_cookies.txt
--skip-download <url>` → 26 lignes `.youtube.com` / 2563 lignes totales.
`--js-runtimes` accepte `deno, node, quickjs, bun` (aide CLI, confirmé aussi
par recherche web) — **quickjs et quickjs-ng sont officiellement supportés**,
pas de repli à chercher.

| morceau | quickjs seul (PATH sans deno/node), 1ère passe |
|---|---|
| Chasing Pavements | passe, 21 s, 3,58 Mo |
| Lost Without U | passe, 15 s, 4,07 Mo |
| READY Acoustic | passe, 14 s, 2,51 Mo |

3/3 — **mais ce résultat était trompeur** : `~/.cache/yt-dlp/challenge-solver/
lib.json` existait déjà (téléchargé par un test `deno` fait juste avant dans
la même session). En le vidant, la commande identique échoue : `[jsc]
Remote component challenge solver script (quickjs) was skipped` puis `n
challenge solving failed`. **Troisième dépendance, pas dans le brief** : le
runtime JS exécute un script solveur ("EJS") que yt-dlp télécharge et met en
cache séparément (`--remote-components ejs:github`, déjà nommé dans
`harmonia/server/youtube.py` côté Mac — je ne l'avais pas relié à `quickjs`
avant de lire le code). Avec le flag, ça repasse.

**Rotation des cookies mesurée en direct** : le fichier exporté à 00:51 a
échoué à 01:03 (~12–20 min plus tard, Mac ET émulateur, message identique
« The provided YouTube account cookies are no longer valid »), pas une panne
— un ré-export immédiat a de nouveau fonctionné. Donc pas "expirent au bout
d'un moment" comme prévu dans le brief mais un budget concret : **moins de
20 minutes entre l'export et l'usage** dans ce test, probablement lié à la
rotation de sécurité Google, pas à une durée de session fixe.

**Étape 1 (NDK)** : `ndk;27.3.13750724` installé (r27, LTS — 2,1 Go, disque
10,9 → 8,7 Go, jamais sous 3 Go). Source `quickjs-ng` v0.17.0 (clone shallow,
amalgamation seule insuffisante — pas de `qjs.c`/`quickjs-libc.c`). Build
CMake+Ninja (`brew install cmake ninja`, 68 Mo) avec
`android.toolchain.cmake` du NDK, `ANDROID_ABI=arm64-v8a`,
`ANDROID_PLATFORM=android-26` : **`qjs` = ELF aarch64 PIE, NEEDED = libc.so,
libdl.so, libm.so uniquement** (rien à embarquer en plus, ce sont les libs
Bionic systeme). 6,9 Mo avec debug_info, **1,23 Mo `llvm-strip`é** →
`mobile/android/app/src/main/jniLibs/arm64-v8a/libqjs.so`. Testé nu (avant
tout code Android) : `adb push` + `chmod 755` + `adb shell
/data/local/tmp/libqjs.so -e "print(1+1)"` → `2`, sur l'émulateur réel
(`emulator-5554`, Android 15, arm64-v8a).

**Étape 2 (app)** — nouveaux fichiers :
`mobile/android/app/src/main/python/mobile_youtube.py` : `yt_dlp.YoutubeDL`
EN PROCESSUS (pas de sous-process, pas de binaire `yt-dlp` sur PATH Android),
`cookiefile=<filesDir>/youtube_cookies.txt`, `js_runtimes={"quickjs":
{"path": <nativeLibraryDir>/libqjs.so}}`, `remote_components=["ejs:github"]`,
`format="bestaudio[ext=m4a]/bestaudio"`. `filesDir` et `nativeLibraryDir`
lus via `com.chaquo.python.Python.getPlatform().getApplication()` — pas
besoin de toucher `MainActivity.kt`. Monkeypatch dans `mobile_entry.py`
(`mobile_youtube.install()`, 6 lignes ajoutées) sur
`harmonia.server.youtube._ytdlp_bin/_video_meta/_download_audio` —
`harmonia/server/youtube.py` intact. `build.gradle.kts` : `install("yt-dlp")`
ajouté au bloc `chaquopy.pip` (roue pure Python, résolue par l'index PyPI
normal, pas l'index Chaquopy dédié — comme flask) ; ajout
`packaging.jniLibs.useLegacyPackaging = true` (AGP 8 sert les `.so` direct
depuis le zip par défaut ; un exécutable a besoin d'un vrai fichier +x sur
disque). `gradlew assembleDebug` (JDK : aucun sur PATH, trouvé
`openjdk@17` déjà installé par brew mais non lié — `JAVA_HOME=$(brew --prefix
openjdk@17)`) → **BUILD SUCCESSFUL, 43,4 Mo (44 218 560 octets) debug APK**.

**Étape 3 (mesure)** — `adb devices` (émulateur déjà debout) → install → 
cookies `adb push` + `run-as fr.harmonia.app cp ... files/youtube_cookies.txt`
→ `adb forward tcp:7790 tcp:7772` → `POST /api/analyze
{"url":"https://www.youtube.com/watch?v=191P7nIeECo"}` → poll `/api/job/<id>`
(note : la route est `/api/job/<id>` singulier, pas `/api/jobs/`).
**Résultat : `files/harmonia_audio/191P7nIeECo.m4a`, 2 507 233 octets
(identique au Mac), titre "READY (Acoustic Version)", artiste "PJ Morton",
~30 s bout en bout, puis `error: [Errno 2] No such file or directory:
'ffprobe'`** — attendu, `harmonia.pipeline` n'est pas censé tourner sur
device en P3b.

## Ce qui reste ouvert

- **Cookies statiques rotent vite** (mesuré <20 min ici) : inutilisable
  tel quel pour Louis — il faudra soit une UI de connexion (mentionnée dans
  le brief comme "viendra plus tard"), soit une automatisation du renvoi
  périodique depuis le Mac. Procédure manuelle actuelle (3 lignes) :
  1. Mac : `.venv-mobile/bin/yt-dlp --cookies-from-browser firefox --cookies
     /tmp/yt_cookies.txt --skip-download <une_url_youtube>`
  2. `adb push /tmp/yt_cookies.txt /data/local/tmp/`
  3. `adb shell run-as fr.harmonia.app cp /data/local/tmp/yt_cookies.txt
     files/youtube_cookies.txt`
- **Le challenge JS/EJS est un point mouvant côté YouTube** (déjà noté dans
  `p3_youtube.md`) — susceptible de recasser indépendamment des cookies, à
  la prochaine rotation de leur player. Le cache `ejs:github` téléchargé par
  l'app persiste dans `files/` (pas mesuré ici où exactement, ni sa taille)
  — donc un premier import sur le téléphone de Louis fera un aller réseau
  GitHub de plus que les suivants.
- **`ffprobe` manquant** n'est pas résolu ici (hors périmètre : "le job
  échouera ensuite, normal"), mais signale que `harmonia/server/youtube.py`
  et/ou le pipeline appellent `ffprobe` quelque part sur le chemin qui suit
  `_resolve_audio` — à vérifier si c'est bloquant pour P4/P5 (device réel).
- **`_video_meta`/`_download_audio` patchés perdent la boucle multi-client
  (`YTDLP_CLIENTS`) et le retry PO-Token** du Mac — une seule tentative,
  client par défaut. Pas mesuré si ça suffit sur d'autres vidéos que les 3
  testées.
- Taille du binaire QuickJS mesurée (1,23 Mo stripped) ; taille de l'APK
  mesurée (43,4 Mo) ; PAS mesuré : l'incrément dû SPÉCIFIQUEMENT à yt-dlp+qjs
  (pas de build "avant" comparable dans cette session).

## Fichiers produits

- `mobile/android/app/src/main/python/mobile_youtube.py` (nouveau)
- `mobile/android/app/src/main/python/mobile_entry.py` (+6 lignes, monkeypatch)
- `mobile/android/app/build.gradle.kts` (+`install("yt-dlp")`, +packaging.jniLibs)
- `mobile/android/app/src/main/jniLibs/arm64-v8a/libqjs.so` (nouveau, 1,23 Mo)
- `docs/portage_mobile/p3b_youtube_cookies_qjs.md` — ce fichier.
- Scratchpad (non committé) : source clonée `quickjs-ng` v0.17.0 + build
  Android sous le dossier scratchpad de session.
