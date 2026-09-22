# P1b — WebView : chart + lecture audio (2026-09-23)

## Résultat en 3 lignes

**Porte de sortie atteinte** : le chart de « This Love » se rend intégralement
(grille, tonalité C minor, sections, dock de lecture) et la tête de lecture
avance réellement — bar en cours, compteur temporel (`0:00` → `1:54`) et
barre de progression tous confirmés par deux captures écartées de 2 s après
`play`. Practise, retour bibliothèque, et l'écran d'analyse (YouTube/iReal
Pro/Tablatures) marchent aussi, clavier compris. `/audio/<fichier>` répond
bien en 206 avec `Content-Range`, et `adb logcat -s harmonia.js` ne montre
aucune erreur JS sur toute la session.
**Une fausse alerte notable** : juste après `play`, le dock et la barre du
haut disparaissent — ce n'est PAS un bug WebView, c'est le mode immersif
existant (`S.chartChrome`, `harmonia/static/screens/chart.js:786-824`,
armé par `immersiveArm()` dans `audio.js:409`) qui se déclenche après 2,6 s
sans interaction, comme sur Safari. Un tap réveille le chrome normalement.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Build + install** (rien de neuf à rebuilder : `./gradlew assembleDebug` →
`BUILD SUCCESSFUL`, 45/46 tâches `UP-TO-DATE`, APK inchangé depuis P1,
44 218 560 octets) :
```
adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n fr.harmonia.app/.MainActivity
```
Bibliothèque toujours affichée (`smoke/screen_library.png`) : accueil avec
« Mes charts · 3 chansons » puis, après tap, la liste des 3 morceaux exacts
(Ben E. King/Stand By Me A major·87 bars, Maroon 5/She Will Be Loved
B♭ major·107 bars, Maroon 5/This Love C minor·80 bars) —
`smoke/screen_library_list.png`.

**Chart** (tap sur « This Love », `smoke/screen_chart.png`) : grille rendue
en entier (intro/A/B/B alt/queue/bridge), tonalité « C minor », mini-nav de
sections (i A B q A B q b B) en haut, dock de transport en bas (play,
piste 0:00–3:25, bouton A–B loop) et onglets Read/Analyse/Annotate/Practise —
rien de coupé.

**Lecture** : tap sur play (110, 2071 en coordonnées 1080×2400) puis deux
captures à 5 s et 7 s (`smoke/screen_playing.png`, `screen_playing2.png`) :
la case surlignée dans la grille avance de la 2e mesure de l'intro (Cm) à la
4e mesure de la section A (Dø7/A♭m7), et le mini-nav du haut passe de « B »
à « A » en surbrillance — le tick est donc bien piloté par la lecture réelle,
pas figé. Logcat serveur (`adb logcat -s python.stdout python.stderr`) :
```
GET /audio/maroon_5_this_love.m4a HTTP/1.1" 200   (range=None -> None, probe initiale)
GET /audio/maroon_5_this_love.m4a HTTP/1.1" 206   (range='bytes=0-3337053' -> Content-Range correct)
```
`adb logcat -s harmonia.js` : aucune ligne (zéro erreur console JS) sur
toute la session, chart compris.

**Mode immersif, vérifié pas juste supposé** : un tap sur la grille
(`smoke/screen_wake.png`) restaure instantanément toolbar + dock — bouton
pause visible, compteur `1:54 / 3:25`, barre de progression à ~48 %, cellule
« G » (section queue) surlignée. Confirme que la disparition n'était pas un
blocage : c'est `chart.js:952` (`if(S.chartChrome) w.appendChild(chartDock());
else w.appendChild(immersiveFade());`), un flex-child (`flex:0 0 auto`), pas
un `position:absolute` mal calé — donc pas un candidat 100dvh/safe-area.

**Practise** (`smoke/screen_practise.png`) : accord courant en grand (E♭,
annotation « ♭III — modal-interchange borrow »), filmstrip de mesures,
contrôles «‹‹4 / pause / 4›› / loop bars 49–52 » — lecture continue depuis le
chart (pause affichée = lecture en cours), rien de coupé.

**Retour bibliothèque** (`smoke/screen_back_home.png`) : accueil, la carte
« REPRENDRE » a basculé sur « Maroon 5 / This Love » — l'état de session
persiste bien.

**Écran d'analyse** (`smoke/screen_search.png`, tap sur le champ de
recherche) : onglets YouTube/iReal Pro/Tablatures, hint « Harmonia écoute
l'audio et déduit la grille », champ de saisie pinné en haut même avec le
clavier Gboard ouvert (pas de recouvrement, pas de scroll cassé) — affichage
seul, aucune analyse lancée.

## Ce qui reste ouvert

- Coordonnées de tap fixées à la main sur les captures (1080×2400) — pas de
  script de test automatisé ; à refaire si le layout bouge.
- Pas testé : rotation d'écran, clavier + saisie réelle sur le champ de
  recherche, geste de swipe pour révéler le chrome immersif (seul le tap a
  été essayé), onglets Analyse/Annotate du chart (seul Read + Practise
  vérifiés — hors du périmètre strict de la porte de sortie demandée).
- `-no-audio` sur l'émulateur confirmé cohérent avec le plan : aucun son
  entendu (normal), mais l'élément `<audio>` avance quand même — pas vérifié
  au-delà de ~10 s de lecture cumulée.
- Le premier `<audio>` fetch initial répond 200 sans Range (`range=None ->
  None`) avant que le second, avec Range complet, réponde 206 — comportement
  normal de Chromium (probe puis vraie requête), non creusé plus loin.
- Rien de cassé trouvé qui nécessite une correction MainActivity/WebSettings
  ou JS dans ce passage — aucun changement de code produit par cette tâche.

## Fichiers produits

- `mobile/android/smoke/screen_library.png`, `screen_library_list.png`,
  `screen_chart.png`, `screen_playing.png`, `screen_playing2.png`,
  `screen_wake.png`, `screen_practise.png`, `screen_back_home.png`,
  `screen_search.png` — captures du smoke test complet.
- Ce fichier + une ligne dans `docs/portage_mobile/JOURNAL.md`.
