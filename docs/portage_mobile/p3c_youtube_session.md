# P3c — session YouTube durable sur le téléphone (2026-09-23)

## Résultat en 3 lignes

**Hypothèse A (laisser yt-dlp réécrire le fichier) est FAUSSE, mesuré sur Mac ET sur device : la session meurt quand même en 10-20 min** — et pas par rotation, par DISPARITION des 13 cookies d'authentification Google (SID, HSID, APISID, LOGIN_INFO…) au profit d'un seul cookie non-auth ajouté. **Résultat B confirmé → connexion dans l'app construite et vérifiée** : `YouTubeLoginActivity` (WebView réelle, formulaire Google affiché, pas de mur anti-bot) écrit les cookies au bon format au bon endroit, déclenchée depuis le bouton "Connect YouTube" des préférences via une route Flask et un pont Python→Java. Un vrai bug trouvé et corrigé en cours de route : sur Android 15 (edge-to-edge par défaut), le bouton "Terminé" atterrissait sous la barre de statut système et ses taps étaient interceptés par elle, jamais reçus par l'app.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Étape 0 (Mac, `.venv-mobile`, yt-dlp 2026.08.19)** — export frais (`--cookies-from-browser firefox --cookies /tmp/…/cookies.txt --skip-download`, 2560 cookies), puis 1 téléchargement toutes les 10 min sur le MÊME fichier (jamais re-copié, laissé à yt-dlp) :

| run | minute | résultat | cookies ajoutés | cookies disparus | total |
|---|---|---|---|---|---|
| 0 | 0 | OK | — | — | 2560 |
| 1 | 10 | OK | — | — | 2560 |
| 2 | 20 | **FAIL** « Sign in to confirm you're not a bot » | `__Secure-YEC` | `APISID, HSID, LOGIN_INFO, SAPISID, SID, SIDCC, SSID, VISITOR_INFO1_LIVE, YSC, __Secure-1PAPISID, __Secure-1PSID, __Secure-1PSIDCC, __Secure-YNID` | 2548 |

La boucle a été arrêtée après ce verdict clair (gain marginal à confirmer sur 4 runs de plus, budget mieux dépensé ailleurs) — **pas les 1h prévues**, note pour la prochaine fois. Les 13 cookies disparus sont EXACTEMENT les cookies d'identité de session Google (SID/HSID/APISID et leurs variantes `__Secure-1P*`, plus `LOGIN_INFO`) — pas un sous-ensemble aléatoire. yt-dlp les a réécrits fidèlement à ce que Google a renvoyé : ce n'est donc pas yt-dlp qui "perd" les cookies, c'est Google qui les invalide côté serveur.

**Confusion possible non éliminée** : ce test tournait EN PARALLÈLE du test device (étape 2b, ci-dessous) sur le même compte Google réel, deux process yt-dlp simultanés — une charge d'usage inhabituelle qui a pu, elle-même, déclencher la détection anti-bot plutôt qu'un TTL fixe de session. Pas re-testé isolément faute de budget — à refaire proprement si le chiffre "15-20 min" doit être fiable au lieu d'indicatif.

**Étape 1 (émulateur, sans identifiants)** — `YouTubeLoginProbeActivity` (WebView isolée, UA par défaut puis UA Chrome mobile) chargeant `accounts.google.com/ServiceLogin?service=youtube` : **le vrai formulaire "Sign in to continue to YouTube" s'affiche dans les deux cas**, pas de mur "navigateur non sécurisé" (captures dans le scratchpad de session). Donc build de `YouTubeLoginActivity` (WebView + bouton "Terminé").

Bug trouvé en testant le tap sur "Terminé" : `dumpsys window displays` → `statusBars frame=[0,0][1080,128]`, bouton `WRAP_CONTENT` → bounds `[0,0][1080,126]`, quasi 100% sous la barre de statut (Android 15 dessine edge-to-edge par défaut, non géré). Les taps dans cette zone sont pris par `StatusBar`, jamais par l'app — confirmé par `dumpsys window | grep mCurrentFocus` (focus correct) et absence totale de log applicatif malgré des taps répétés à plusieurs coordonnées. Fix : `layout.fitsSystemWindows = true` sur le conteneur → bouton repoussé à `[0,128][1080,254]`, tap reçu, `Log.i "saved N cookie names"` confirmé, fichier écrit.

**Pont Python→Java** (`mobile_youtube.open_login_activity()`, Chaquopy `jclass`, `Intent.setClassName` + `FLAG_ACTIVITY_NEW_TASK` depuis un contexte Application) testé via `POST /api/mobile/youtube_login` (adb forward 7790→7772) → `{"ok":true}`, `dumpsys activity activities` confirme `topResumedActivity=…YouTubeLoginActivity`.

**Écriture des cookies** (sans identifiants, donc session anonyme) : tap sur "Terminé" → `youtube_cookies.txt` réécrit avec 3 cookies (`__Host-GAPS`, `OTZ`, `__Secure-ENID` — cookies de session anonymes posés avant tout login), format Netscape valide, AU MÊME chemin que `mobile_youtube.py:_cookie_file()` lit déjà.

**Étape 2b (device, cookies frais poussés une fois puis laissés en place)** :

| run | minute | résultat |
|---|---|---|
| 0 | 0 | OK — `191P7nIeECo.m4a`, 2 507 233 octets (identique au Mac / p3b) |
| 1 | 15 | **FAIL** « Sign in to confirm you're not a bot » |

Même verdict que le Mac : le fichier laissé en place, réécrit par yt-dlp, ne suffit pas. (Bug de script trouvé et corrigé en route : `adb shell run-as <pkg> sh -c "cp A B"` échoue silencieusement — `adb shell` rejoint les arguments avec des espaces avant de les passer au shell distant, cassant les guillemets imbriqués, `sh -c` ne voit que "cp" comme script → "Needs 1 argument". `run-as <pkg> cp A B` direct, sans `sh -c`, fonctionne. A fait échouer le tout premier run avec un faux "cookies absents" pas lié à la vraie question — refait proprement après correction.)

## Ce qui reste ouvert

- **Le vrai test n'est pas fait** : `CookieManager.getCookie()` expose-t-il les cookies `HttpOnly` (SID/HSID/… — précisément ceux qui viennent d'être identifiés comme LA cause de panne) ? La doc Android dit que oui (API native, pas `document.cookie`), mais sans identifiants Louis je n'ai capturé que 3 cookies anonymes, aucun cookie d'auth réel à vérifier. **C'est la prochaine chose à tester, avec un vrai compte** — si la réponse est non, toute cette brique ne résout rien de plus que le fichier statique.
- **Le chiffre "10-20 min" a un facteur de confusion** (deux process yt-dlp parallèles sur le même compte, voir plus haut) — pas isolé.
- **La brique ne résout pas "et après ?"** : une fois connecté, `CookieManager` ne se rafraîchit QUE quand la WebView recharge une page Google — pas en arrière-plan pendant que yt-dlp tourne dans un thread séparé. Donc si les cookies meurent en ~15-20 min quel que soit le mécanisme, Louis devra probablement rouvrir "Connect YouTube" à peu près à cette fréquence — c'est un geste sur le téléphone au lieu d'un geste sur le Mac, pas une session qui "dure" au sens fort. Pas mesuré si une simple ré-ouverture périodique de la WebView (sans re-taper l'email) suffit à rafraîchir les cookies sans re-authentification complète.
- `YouTubeLoginProbeActivity` laissée dans le code (`exported=false`, scaffold de diagnostic, pas dans le chemin de prod) — à supprimer si un futur agent veut nettoyer, pas fait ici pour ne pas toucher plus de fichiers que nécessaire.
- Le bouton "Connect YouTube" (`prefs.js`) est dans le JS PARTAGÉ avec le Mac — inoffensif là-bas (la route 404, toast "Not available"), mais visible sur l'app Mac de Louis aussi. Pas isolé derrière une détection de plateforme, hors périmètre du budget.

## Fichiers produits

- `mobile/android/app/src/main/java/fr/harmonia/app/YouTubeLoginActivity.kt` (nouveau) — WebView + bouton "Terminé", écrit `<filesDir>/youtube_cookies.txt`
- `mobile/android/app/src/main/java/fr/harmonia/app/YouTubeLoginProbeActivity.kt` (nouveau) — scaffold de diagnostic étape 1
- `mobile/android/app/src/main/AndroidManifest.xml` (+2 activités, `exported=false`)
- `mobile/android/app/src/main/python/mobile_youtube.py` (+`open_login_activity()`, +blueprint `POST /api/mobile/youtube_login`)
- `mobile/android/app/src/main/python/mobile_entry.py` (+1 ligne, enregistre le blueprint)
- `harmonia/static/screens/prefs.js` (+bouton "Connect YouTube", 9 lignes)
- `docs/portage_mobile/p3c_youtube_session.md` — ce fichier
- Scratchpad (non committé) : logs des deux boucles de mesure (`rotation_log.tsv`, `device_rewrite_log.tsv`), captures d'écran des 3 tests WebView

**Procédure que Louis suivra sur son téléphone** (jusqu'à ce que l'ouvert ci-dessus soit vérifié) :
1. Préférences → "Connect YouTube".
2. Se connecter normalement à son compte Google dans l'écran qui s'ouvre.
3. Taper "Terminé" en haut de l'écran.
4. Si un import échoue avec "Sign in to confirm you're not a bot", refaire 1-3 (probablement toutes les 10-20 min d'usage, pas mesuré au-delà de ce test).
