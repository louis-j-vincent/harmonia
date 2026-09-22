# P3 — import YouTube sur téléphone (2026-09-23)

## Résultat en 3 lignes

**Étape 0 falsifie la prémisse : STOP avant tout code Android.** 15/15
combinaisons (3 morceaux × 5 clients : défaut, android, ios, web, tv), sans
cookies ni PO-token, échouent identiquement (« Sign in to confirm you're not
a bot »). Diagnostic poussé un cran plus loin (hors mandat strict de l'étape
0, mais nécessaire pour comprendre le mécanisme) : avec cookies Firefox le
téléchargement PASSE, mais seulement parce que **yt-dlp 2026.08.19 exige
maintenant un moteur JS externe** (deno/node, priorité deno>node>phantomjs)
pour résoudre le challenge de signature `n` — retiré `deno` du PATH → échec
net (« n challenge solving failed »), même avec cookies valides. Le blocage
n'est donc plus seulement le PO-token connu du Mac : c'est cookies + moteur
JS, et le client `android` (le seul qui aurait pu éviter les cookies) **ne
supporte pas les cookies du tout** dans cette version.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Setup** : `~/harmonia/.venv-mobile` (Python 3.12, séparé du `.venv` de
prod), `pip install yt-dlp` → `2026.08.19`. Disque 11 Go libres avant/après
(> seuil 3 Go, jamais approché).

**3 URL réelles** (IDs pris dans `state/human/chart_meta.json`) :
`08DjMT-qR9g` (Chasing Pavements), `0DdCoNbbRvQ` (Lost Without U),
`191P7nIeECo` (READY Acoustic).

**Étape 0 — matrice complète, sans cookies, sans serveur PO-token** :
```
yt-dlp -f "bestaudio[ext=m4a]/bestaudio" --no-playlist \
  --extractor-args "youtube:player_client=<client>" -o out.%(ext)s <url>
```
| client | Chasing Pavements | Lost Without U | READY Acoustic |
|---|---|---|---|
| défaut | bloque | bloque | bloque |
| android | bloque | bloque | bloque |
| ios | bloque | bloque | bloque |
| web | bloque | bloque | bloque |
| tv | bloque | bloque | bloque |

15/15, message identique : `ERROR: [youtube] <id>: Sign in to confirm you're
not a bot. Use --cookies-from-browser or --cookies for the authentication.`
Script + JSON bruts dans `/tmp/yt_test_out/results.json` (scratchpad, pas
committé).

**Diagnostic complémentaire** (pour comprendre le mécanisme, pas dans le
mandat strict mais requis pour proposer une vraie alternative) :
- `--cookies-from-browser firefox`, client défaut, `deno` sur le PATH
  (`/opt/homebrew/bin/deno` 2.9.2, présent sur ce Mac) → **succès**,
  `out.m4a` 3.41 Mo en ~3 s, log montre `[jsc:deno] Solving JS challenges
  using deno`.
- Même commande, PATH filtré pour retirer `deno` → **échec** :
  `WARNING: n challenge solving failed: ... Ensure you have a supported
  JavaScript runtime ... EJS`, puis `ERROR: Requested format is not
  available` (seules des images restent).
- `--cookies-from-browser firefox --extractor-args
  "youtube:player_client=android"`, sans deno → **échec différent** :
  `WARNING: Skipping client "android" since it does not support cookies`.

Recherche web (yt-dlp EJS, 2026-09) confirme que ce n'est pas spécifique à
cette machine : yt-dlp 2026.08.19 a rendu un runtime JS externe (deno, node,
phantomjs, priorité dans cet ordre) **obligatoire** pour le challenge `n`
sur la plupart des clients ; un plugin tiers, `ytdlp-jsc` (PyPI, QuickJS
embarqué en WASM, sans runtime externe) prétend lever cette dépendance —
non testé ici (hors budget), piste pour la suite.

## Ce qui reste ouvert

**Verdict pour l'étape Android (ÉTAPE 1/2 du brief) : ARRÊTÉ, comme prévu
par le protocole** — aucun client ne passe sans cookies+JS-solver, donc pas
de code Chaquopy/mobile_youtube.py écrit. Trois alternatives, par ordre de
simplicité :
1. **Partage depuis l'app YouTube** (le repli le plus léger cité dans le
   brief) — Android sait recevoir un `Intent.ACTION_SEND` d'URL depuis
   l'app YouTube ; ne contourne pas le blocage serveur, juste évite de le
   déclencher soi-même. Ne donne jamais le fichier audio (YouTube ne le
   partage pas), donc ne résout pas le besoin réel : il faudrait encore
   extraire l'audio après coup, même problème.
2. **`ytdlp-jsc`** (piste la plus prometteuse) : plugin pip, moteur JS
   QuickJS embarqué (pas de binaire externe deno/node à porter sur
   Android), donc potentiellement installable via Chaquopy si ses roues
   couvrent arm64 Android — À VÉRIFIER, pas fait ici (budget).
3. **`youtubedl-android` (yausername)** : toujours actif (issues récentes
   sur GitHub), mais aucune confirmation trouvée qu'il gère le nouveau
   challenge `n`/EJS de 2026 — probablement touché par le même problème
   que yt-dlp nu, à re-tester avant d'investir dedans.

Fragilité attendue si l'un de ces chemins aboutit : cookies Firefox
expirent (session YouTube vivante requise, comme sur le Mac aujourd'hui) ;
le challenge JS/EJS est un point mouvant côté YouTube — susceptible de
recasser à chaque changement de leur player, indépendamment du PO-token.
Aucun test fait sur émulateur/téléphone (bloqué avant l'étape 2 par le
résultat de l'étape 0).

## Fichiers produits

- `docs/portage_mobile/p3_youtube.md` — ce fichier.
- `/tmp/yt_test_out/` (scratchpad, non committé) — script de test et
  résultats bruts JSON.
