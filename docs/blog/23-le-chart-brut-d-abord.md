# 23 — Le chart brut d'abord : 1,3 s au lieu de 45

*2026-08-07*

Louis : « il faut arriver au chart brut le plus rapidement possible. Liste les
étapes minimales qu'il faut pour arriver à ce chart brut, tu me changes l'écran
de chargement pour juste m'afficher ces étapes-là, dès que le chart brut est
dispo tu l'affiches direct, et le reste des tâches qui sont à faire après
(détection gammes, harmonies, sections) tu le fais en background. »

## D'abord mesurer, ensuite couper

La question « quel est le chemin le plus rapide » n'a pas de réponse d'opinion.
J'ai chronométré chaque étape d'`analyze()` sans la modifier (monkeypatch des
fonctions qu'elle appelle), sur quatre morceaux de 68 à 224 secondes, avec
`HARMONIA_SECTIONS=voice` — la configuration réellement en production.

**Cache chaud** (secondes) :

| étape | autumn 1:08 | yesterday 2:06 | this love 3:25 | close to you 3:44 |
|---|---|---|---|---|
| 1 battues (Beat This!) | 0,00 | 0,00 | 0,00 | 0,00 |
| 2 posteriors musx | 0,01 | 0,01 | 0,02 | 0,02 |
| 3 re-décodage sur la grille | 0,32 | 0,71 | 1,66 | 1,60 |
| 6 chroma NNLS (×2 appels) | 0,00 | 0,00 | 0,00 | 0,01 |
| **7 détection de sections** | **18,15** | **27,19** | **46,95** | **54,21** |
| 7b/7c repli | 0,00 | 0,00 | 0,49 | 0,30 |
| 8 clé harmonique | 0,00 | 0,01 | 0,01 | 0,07 |
| **total** | 18,49 | 27,93 | 48,68 | 55,93 |

La détection de sections pèse **96 à 98 %** du temps. Tout le reste — battues,
posteriors, re-décodage, mise en mesures — tient sous 2 secondes.

**Cache froid** (musx, battues et NNLS redirigés vers un dossier jetable ; les
stems demucs, eux, restaient chauds) :

| étape | autumn | yesterday | this love |
|---|---|---|---|
| 1 battues | 8,42 | 3,58 | 6,23 |
| 2 posteriors musx | 17,95 | 5,71 | 10,14 |
| 3 re-décodage | 1,02 | 0,70 | 1,64 |
| 6 chroma NNLS | 2,65 | 2,25 | 4,08 |
| **7 sections** | **30,43** | **30,84** | **52,50** |
| 8 clé | 0,00 | 0,01 | 0,02 |
| **total** | 60,47 | 43,10 | 74,62 |

Même à froid, ce qu'on peut différer représente 55 à 77 % du travail. (Le premier
morceau paie en plus le chargement des modèles ONNX et Beat This! : c'est
pourquoi le plus court a les plus gros chiffres en 1 et 2.)

Il n'y avait donc rien à arbitrer.

## Les étapes minimales retenues

Ce qu'il faut pour qu'un chart s'affiche **et se joue** :

1. les battues et les temps forts (Beat This!) ;
2. les posteriors d'accords image par image (musx) ;
3. le re-décodage calé sur nos temps ;
4. la disposition en mesures (arithmétique d'indices de temps) ;
5. une clé provisoire — parce que l'app orthographie ses accords à partir de
   `model.key` et ne sait pas s'en passer.

Le reste — sections, repli des répétitions, analyse harmonique (couleurs,
alternatives, clés locales) — part en tâche de fond.

La clé provisoire se lit sur les accords déjà décodés : un histogramme de
hauteurs pondéré par les durées, passé à `key_profiles.infer_key`. Quelques
millisecondes, aucune lecture d'audio. `chord_pcs` et `infer_key` étaient
importés dans `pipeline.py` et morts depuis la suppression de l'ancienne
étape 4 ; ils reprennent du service exactement là.

## Une seule pipeline, coupée en deux temps

`analyze_steps()` est un **générateur** : il rend `("raw", modèle)` après la mise
en mesures, puis reprend et rend `("final", modèle)`. Un seul corps de fonction,
interrompu puis relancé — pas une seconde pipeline allégée qui divergerait (ce
dépôt a déjà payé ça). `analyze()` reste le raccourci qui vide le générateur et
renvoie le second modèle : `chord_lm_prod_eval`, `pattern_dict_core`,
`chord_lm_bass_fusion_eval` et `scratchpad/raw_app_charts.py` n'ont pas bougé
d'une ligne, et `HARMONIA_RAW_CHART=1` fait toujours exactement ce qu'il faisait.

Une contrainte à ne pas oublier : `bars` est muté sur place par le repli et par
l'analyse harmonique. L'appelant doit **sérialiser le modèle brut avant de
redemander le suivant**. `server._run_job` le fait, de façon synchrone, dans le
thread du job.

Côté serveur, le job expose deux jalons au lieu d'un :

* `chart_url` dès que le brut est écrit, `status` restant `running` ;
* `status=done` quand le raffinement a **écrasé le même fichier** — même
  `file_key`, donc rien ne se dédouble dans la bibliothèque.

Effet de bord utile : si le raffinement plante alors que le brut est déjà publié,
le job ne passe plus en `error` (ce qui effacerait un chart qui marche) mais en
`done` avec un champ `refine_error`. La trace complète reste dans le log.

## L'écran de chargement

Cinq lignes, en français, exactement les cinq étapes ci-dessus. Ce qui a disparu
et pourquoi :

* « Hearing the harmony » attendait la clé, qui n'arrivait qu'à la toute fin —
  la ligne restait donc allumée pendant 95 % de l'attente ;
* « Sketching the chords » puis « Naming the chords — music-x-lab » décrivaient
  une passe brouillon suivie d'une passe fine qui n'existent plus dans
  `harmonia_min` (il n'y a qu'un re-décodage) ;
* le cercle des quintes « fold k/N » se déclenchait sur `job.musx_fold`, que ce
  serveur ne peuple jamais : du code mort depuis le début.

Une fois le chart ouvert, un liseré discret au-dessus de la grille dit
« Sections et gamme en cours — joue, le chart se mettra à jour tout seul », et
disparaît quand le raffinement arrive.

## Ce qui a été vérifié, et comment

Bout en bout par l'API (`POST /api/analyze` avec `{"url":"local:<stem>"}`),
caches chauds, processus serveur chaud :

| morceau | avant | après : chart brut | après : raffiné |
|---|---|---|---|
| Yesterday (2:06) | 28,90 s | **0,80 s** | 27,6 s |
| Bein Green (2:59) | 44,63 s | **1,30 s** | 41,1 s |
| This Love (3:25) | 51,29 s | **1,54 s** | 47,1 s |
| Let It Be (4:03) | 62,36 s | **1,54 s** | 58,3 s |

De 33 à 40 fois plus rapide jusqu'au premier chart jouable. Le temps total ne
bouge pas : on n'a rien accéléré, on a arrêté d'attendre.

Le « avant » a été mesuré sur le serveur qui tournait encore avec l'ancien code
en mémoire (il n'a pas de rechargeur) — donc sur du vrai code d'avant, pas sur
une reconstruction.

Et surtout, deux morceaux vérifiés dans un vrai navigateur : le chart s'ouvre
1,5 s / 2,5 s après le clic, on lance la lecture, et quand les sections arrivent
la page bascule sur le chart replié **sans couper le son** — les captures
montrent 0:29 et 0:41 toujours en lecture, tête de lecture sur les mesures 9 et
13. Aucune erreur JS.

## Ce qui reste fragile

* **Le cache demucs du mode `voice` vit dans le dossier temporaire d'une session
  morte** (`/private/tmp/claude-501/…/997f81c7-…/scratchpad/stems`, 2,9 Go, 30
  morceaux). S'il est balayé, chaque détection de sections repaie une séparation
  de sources complète. Les 18–54 s mesurées ci-dessus supposent ce cache chaud ;
  sans lui, c'est une minute de plus par morceau. À remonter dans
  `data/cache/` — c'est une ligne à changer dans `scratchpad/rhythm_ssm.py`.
* Le raffinement écrase le fichier même si Louis a corrigé un accord entre-temps.
  L'app le protège (elle propose « Recharger » au lieu de recharger d'office
  quand une annotation est en cours), mais la protection est **côté client
  seulement**.
* `abba_chiquitita` échoue toujours sur le garde-fou de grille (81 % de mesures
  à 4 temps, seuil 85 %) — antérieur à ce travail, chemin d'erreur inchangé.
