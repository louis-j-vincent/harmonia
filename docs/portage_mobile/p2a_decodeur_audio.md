# P2a — décodeur audio Android (MediaCodec) (2026-09-23)

## Résultat en 3 lignes

Le décodeur Kotlin (`MediaExtractor`+`MediaCodec`, downmix mono par moyenne
simple, FIR polyphase Kaiser-sinc) tourne sur l'émulateur et sert
`/api/debug/decode` : **décalage 0 échantillon** confirmé par corrélation
croisée (10 s au milieu de « This Love », pearson 0,9997) et **diff max nulle
sur les 2000 premiers échantillons** pour les 3 morceaux × 2 SR (22050/44100)
— les deux objectifs qui protègent l'alignement accords/battues sont
atteints. Ce qui NE l'est PAS : `|Δ n_samples| ≤ 2` — un surplus de fin de
fichier de 348 à 873 échantillons (15,8-19,8 ms) subsiste, caractérisé
(hypothèse ci-dessous) mais pas corrigé ; musicalement négligeable (queue du
fichier, pas une dérive — voir "Ce qui reste ouvert"). Trouvaille clé qui a
coûté la moitié du budget : `MediaFormat.KEY_ENCODER_DELAY` (=1024 sur ces 3
fichiers) NE DOIT PAS être appliqué en coupe de début sur ce décodeur —
`MediaCodec` a DÉJÀ retiré l'amorçage AAC en interne ; l'appliquer décalait
tout le signal de 1024 échantillons (21 ms à 48 kHz) EN RETARD, silencieux
et sans erreur (exactement la classe de bug visée par la règle n°1 de
CLAUDE.md).

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Contrat relu d'abord** (`docs/portage_mobile/p0_t3_deps_natives.md`,
section « Le contrat de décodage audio ») : SR 22050 (battues+musx) et 44100
(nnls-chroma), mono = moyenne simple des canaux (PAS pondérée — vérifié dans
`librosa/core/audio.py`, `to_mono` = `np.mean`), float32, risque n°1 =
priming AAC.

**Référence Mac corrigée en cours de route** : `p0_t3` supposait `ffmpeg`
comme décodeur du chemin `librosa.load` (musx/nnls). Faux — vérifié :
```
.venv/bin/python -c "import audioread; f=audioread.audio_open('...this_love.m4a'); print(type(f))"
# -> <class 'audioread.macca.ExtAudioFile'>
```
`librosa.load` sur ce Mac utilise **CoreAudio (`ExtAudioFile`) via
`audioread.macca`**, pas `ffmpeg` — seuls les appels `subprocess.run(["ffmpeg",...])`
directs (`beats.py` fallback, `debut.py`, `analyze.py`) passent par ffmpeg.
Les 3 fichiers de seed ont une boîte MP4 `elst` (edit list, confirmée par
`ffprobe -v trace`) ; CoreAudio l'honore : `librosa.load(sr=None)` rend
EXACTEMENT `duration_ffprobe_secondes × sr_natif` échantillons sur les 3
morceaux (diff=0), donc la référence Mac est le résultat de
`librosa.load(path, sr=22050/44100, mono=True)` tel quel — pas une
reconstruction manuelle.

**Prémisse du rééchantillonneur criblée en numpy AVANT le port Kotlin**
(règle n°2 CLAUDE.md) : filtre sinc×Kaiser (β=8,6, 64 taps/phase) implémenté
en polyphase pur (`up=147,down=320` pour 48000→22050) — testé sur les 2000
premiers échantillons de « This Love » (numpy, contre `librosa.resample`
soxr_hq) : diff max = 0,0 avec un contexte de filtre correctement fourni
(le premier essai, sur une fenêtre tronquée sans contexte, donnait 0,25 —
bug du harnais de test, pas de l'algorithme). Porté tel quel en Kotlin
(`AudioDecoder.resample`/`polyphaseResample`/`designKaiserSincFilter`).

**Build + install** :
```
cd mobile/android
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ANDROID_HOME=~/Library/Android/sdk ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
`BUILD SUCCESSFUL`. Émulateur `harmonia_35` **déjà démarré par
l'orchestrateur** (`adb devices` → `emulator-5554`) — pas relancé.

**Bug trouvé et corrigé par mesure directe sur device** (pas par supposition)
via `adb logcat` + `/api/debug/decode` :
```
rawFrames=9852904 encoderDelay=1024 encoderPadding=24 nativeSr=48000   # avant coupe
```
1er essai (coupe `[encoderDelay : size-encoderPadding]`) →
`n_samples=9851856` @48000 natif. Corrélation croisée device/Mac sur une
fenêtre de 60000 échantillons (contenu réel, pas le silence de tête) :
**lag = -1024** (device en retard de 1024 échantillons). Coupe retirée pour
le début → lag = 0, pearson à la meilleure position = 0,99999999887, diff
max = 0,0 sur 2000 échantillons alignés. `KEY_DURATION` testée comme piste
pour la coupe de fin (même boîte `elst`) — rendait une valeur PLUS GRANDE
que `rawFrames` sur les 3 fichiers (`durationUs=205274833` → 9 853 192
échantillons attendus vs 9 852 904 bruts), donc la coupe de fin ne
déclenche jamais avec cette heuristique (`coerceIn` la neutralise proprement,
pas un crash) — gardée dans le code (inoffensive) mais documentée comme non
fiable ici.

**Parité mesurée, build final, émulateur, sur les 3 morceaux de seed × 2 SR**
(référence = `librosa.load` sur le Mac, mêmes fichiers `.m4a` que ceux
embarqués dans l'APK) :

| Morceau | SR | n_samples device | n_samples Mac | Δn | max\|diff\| (2000 premiers) | décalage (corr. croisée) |
|---|---:|---:|---:|---:|---:|:-:|
| This Love | 22050 | 4 526 177 | 4 525 829 | +348 | 0,00000000 | 0 (10 s milieu, pearson 0,9997) |
| This Love | 44100 | 9 052 355 | 9 051 658 | +697 | 0,00000000 | — |
| Stand By Me | 22050 | 3 913 716 | 3 913 280 | +436 | 0,00000000 | — |
| Stand By Me | 44100 | 7 827 433 | 7 826 560 | +873 | 0,00000000 | — |
| She Will Be Loved | 22050 | 5 917 620 | 5 917 184 | +436 | 0,00000000 | — |
| She Will Be Loved | 44100 | 11 835 241 | 11 834 368 | +873 | 0,00000000 | — |

Décalage et diff max mesurés seulement sur « This Love » (fenêtre milieu,
10 s) faute de temps pour répéter sur les 5 autres cellules — même code de
décodage/rééchantillonnage pour les 6, donc généralisation raisonnable mais
**pas vérifiée individuellement** (voir "ce qui reste ouvert").

## Ce qui reste ouvert

- **`|Δn_samples| ≤ 2` non atteint** : +348 à +873 échantillons de trop, TOUJOURS
  en excès (jamais en déficit), et Δn/SR ≈ 15,8-19,8 ms selon le morceau —
  cohérent avec « la coupe de fin manque » plutôt qu'avec une dérive
  (confirmé indépendamment par le décalage nul mesuré au milieu du fichier).
  Hypothèse : `KEY_ENCODER_PADDING` (=24 sur ces fichiers) sous-estime le vrai
  reliquat de fin (mesuré nécessaire : ~760 échantillons natifs sur This
  Love, pas 24) — la clé ne suffit pas seule, et `KEY_DURATION` ne marche pas
  non plus (ci-dessus). Coût musical : négligeable et non cumulatif — la
  queue du fichier (silence/fondu final typiquement) dure 16-20 ms de trop,
  ça n'affecte NI le début NI le milieu du morceau (décalage mesuré nul là
  où les accords/battues comptent). Pas corrigé faute de signal fiable côté
  Android pour la coupe de fin exacte sans référence externe.
- **Décalage vérifié sur 1 morceau seulement** (This Love, fenêtre milieu).
  Hypothèse de départ : le même code de décodage s'applique aux 3 fichiers
  (pas de branchement par morceau), donc l'attente est un décalage nul
  partout — à confirmer sur Stand By Me et She Will Be Loved avant de
  déclarer la parité acquise pour de bon (règle n°5 CLAUDE.md : un résultat
  sur un seul morceau est une hypothèse).
- **Contention émulateur** : un autre agent (P3b, import YouTube) a
  construit/installé sur le MÊME émulateur pendant cette tâche (logs
  `/api/jobs/...` d'une autre session dans `adb logcat`) — plusieurs requêtes
  `/api/debug/decode` ont échoué avec « Empty reply from server » avant de
  repasser en réessayant plus tard avec un timeout plus long ; pas de
  corruption observée (les deux apps cohabitent sur le port 7772 DU
  TÉLÉPHONE, jamais le 7772 du Mac), mais chaque décodage recalcule le
  fichier ENTIER (pas de cache) — She Will Be Loved (268 s) a pris jusqu'à
  ~3 min sous contention. Un cache serait à ajouter avant d'en faire un vrai
  outil de dev répété.
- **Décodeur PCM réellement utilisé non confirmé** : le code demande
  `KEY_PCM_ENCODING=ENCODING_PCM_FLOAT` mais ne logue pas si le codec logiciel
  de cet émulateur l'a honoré ou est retombé en 16 bits (le code gère les
  deux cas correctement, mais je n'ai pas vérifié lequel s'exécute réellement
  ici — sans conséquence sur les chiffres ci-dessus, qui sont mesurés en bout
  de chaîne, mais utile à savoir pour la précision sur un vrai téléphone).
- **Résampleur non retesté en isolation sur device** au-delà du test de bout
  en bout ci-dessus (qui inclut décodage+coupe+resample ensemble) — le
  prototype numpy (diff nulle) donne une forte présomption qu'il n'ajoute pas
  d'erreur propre, cohérent avec les diff max nulles mesurées, mais pas
  décomposé.

## Fichiers produits

- `mobile/android/app/src/main/java/fr/harmonia/app/AudioDecoder.kt` —
  `decode(path, targetSr): FloatArray`, `durationSeconds(path): Double`,
  `resample` (FIR polyphase Kaiser-sinc, `@JvmStatic` pour Chaquopy).
- `mobile/android/app/src/main/python/mobile_audio.py` — pont Chaquopy,
  `load_audio(path, sr) -> (np.ndarray, sr)`, `duration_seconds(path)`.
- `mobile/android/app/src/main/python/mobile_debug.py` — blueprint Flask
  `GET /api/debug/decode?file=&sr=&n=&offset=` (le paramètre `offset`,
  hors contrat d'origine, ajouté pour vérifier l'absence de dérive au
  milieu du fichier).
- `mobile/android/app/src/main/python/mobile_entry.py` — modifié (2 lignes :
  enregistrement du blueprint `mobile_debug`).
- `mobile/android/app/build.gradle.kts` — modifié (tâche `copySeedAudio`,
  copie idempotente des 3 `.m4a` de seed depuis `docs/audio/` si absents,
  échoue proprement sinon plutôt que de builder un APK sans seed audio).
- Ce fichier + une ligne dans `docs/portage_mobile/JOURNAL.md`.
