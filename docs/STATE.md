# État du projet — 2026-09-14

Ce fichier dit ce que l'app fait aujourd'hui, où vivent ses fichiers, et
comment vérifier qu'un changement n'a rien cassé. Pour le POURQUOI de chaque
choix, voir `docs/refactor_2026-09/` (le plan et le journal du refactor qui a
produit ce paquet).

## L'app en une phrase

On donne un lien YouTube, un fichier, ou le micro. L'app entend les temps
(beats), les accords, et les sections de la chanson (intro, couplet,
refrain, pont…), et écrit un chart — un tableau d'accords qu'on peut lire,
transposer, corriger, et utiliser pour s'entraîner (Practise) ou jouer en
groupe (Jam). Un seul utilisateur (Louis), pas de compte, servi en Flask sur
le port **7772**, ouvert depuis un Mac et un iPhone via Tailscale.

## La carte du paquet `harmonia/`

Chaque ligne est un fichier, en une phrase.

| Fichier | Rôle |
|---|---|
| `settings.py` | Le SEUL endroit qui lit les variables d'environnement (port, dossier et appareil du modèle musx). Tout le reste (quel détecteur de sections, quelle loi de fusion…) est une CONSTANTE ici, pas un réglage. |
| `cache.py` | Une seule façon de nommer un fichier de cache par morceau (beats, accords musx, chroma, sections). |
| `pipeline.py` | Le chef d'orchestre : audio → temps → accords → mesures → sections → repli → tonalité → chart. |
| `beats.py` | Les temps de la chanson (modèle Beat This!). |
| `musx.py` | Les probabilités d'accord, image par image (modèle musx, un ensemble ISMIR 2019). |
| `nnls_features.py` | Le chroma (quelle note sonne quand) qui nourrit `musx.py`. |
| `bars.py` | Découpe les temps en mesures. |
| `sections/` | Où sont les frontières de section (SongFormer, un modèle qui écoute le SON, pas les accords) et la ressemblance mesure-à-mesure que Soudure et l'outil sections réutilisent. |
| `folding.py`, `refold.py` | Le repli : une section jouée plusieurs fois s'écrit UNE fois, en empilant les probabilités des répétitions. |
| `harmonic_key.py` | La tonalité et les couleurs de la grille. |
| `span_rescore.py` | Les accords alternatifs proposés dans l'écran Annotate. |
| `soudure.py` | Le « mot » d'un morceau (une lettre par mesure) et le jeu de Soudure : c'est lui qui réécrit les sections d'un chart quand Louis a découpé à la main (`sections_pour_chart`). |
| `section_tool.py` | « Je passe le doigt sur ces mesures : où ça se rejoue ? » — le moteur de l'éditeur de sections. |
| `phrases4.py` | L'algo des phrases à quatre mots : il nomme les sections (A, B, queue…) à partir du mot. |
| `ssm_page.py` | La page `/ssm` : la matrice de ressemblance d'un morceau, cliquable, avec tête de lecture. |
| `jam.py` | Le mode Jam en direct (Beat This! + musx sur le micro). |
| `titles.py` | Artiste + titre depuis un titre YouTube. Sans état ni chemin. |
| `annotations.py` | Le schéma des annotations d'accords de Louis (le sidecar), et comment on les repose sur un chart. |
| `labels.py`, `key_profiles.py` | Tables de conversion, sans état ni chemin. |
| `integrations/` | iReal Pro (recherche + export) et Ultimate Guitar (recherche de tablatures) — le seul code de l'ancien paquet legacy à avoir survécu. |
| `server/app.py` | Construit l'app Flask. |
| `server/jobs.py` | La liste des analyses en cours. |
| `server/youtube.py` | Téléchargement YouTube (yt-dlp). |
| `server/routes/*.py` | Une famille de pages web par fichier : bibliothèque, lancer une analyse, corrections, outils de section, Jam, iReal/UG. |
| `static/` | Le frontend (JS/HTML), en modules par écran. |
| `assets/` | Petits fichiers embarqués (table d'accords, poids légers) suivis par git. |

`third_party/musx_ismir2019/` est un clone d'un dépôt de recherche externe
(licence MIT) : le modèle `musx.py` l'exécute directement (pas juste une
lecture), donc il reste dans le dépôt tel quel.

**`harmonia_min/` n'est plus l'app** (sprint 22, 2026-09-16). Le paquet a été
supprimé ; il n'en reste que `harmonia_min/chord_lm/`, une recherche en cours
qui ne fait pas partie de l'application, plus le `__init__.py` qui la garde
importable. Rien dans `harmonia/` n'importe plus `harmonia_min`. Pour
retrouver l'ancien paquet entier : tag `pre-refactor-2026-09-14`.

## Où vivent les fichiers (`state/`)

Deux dossiers, une seule règle : **ce que Louis écrit à la main est suivi par
git ; ce qu'un modèle peut refabriquer ne l'est pas.**

| Dossier | Suivi par git ? | Contenu |
|---|---|---|
| `state/human/` | Oui | sections découpées à la main, annotations d'accords, la marque « Set bar 1 » (le point où Louis dit « la mesure 1 commence ici »), titres/dossiers de la bibliothèque |
| `state/cache/` | Non | charts calculés, temps, sections détectées par le modèle, rapports |

Une annotation faite à la main s'est perdue une fois (2026-08-12) parce
qu'elle vivait dans un dossier ignoré par git — cette séparation est la
correction.

## Comment lancer l'app

```
.venv/bin/python -m harmonia.server
```

Sert sur `http://127.0.0.1:7772/` (port réglable par `HARMONIA_PORT`). Pour
le téléphone, deux adresses Tailscale existent (voir
`docs/archive/known_issues_2026-07_2026-09.md`, entrée du 2026-08-20) :

- `http://100.89.209.63:7772/` — marche pour lire/écrire un chart ;
- `https://louiss-macbook-air.tail87ced3.ts.net:8443/` — **obligatoire** pour
  le micro (Record/Jam) : Safari cache le micro sur une adresse http.

**Redémarrer : toujours par PID du port, jamais par nom de process.**
`kill $(lsof -ti tcp:7772)` puis relancer. Le serveur n'a pas de rechargement
automatique ; un process de plusieurs jours a déjà servi du vieux code sans
rien dire.

## Le rapport d'or : vérifier qu'un changement n'a rien déplacé

« Cuire » veut dire faire tourner la pipeline sur un morceau et écrire son
chart.

```
.venv/bin/python -m tools.golden --engine harmonia --out state/cache/golden/run \
    --baseline state/cache/golden/baseline
```

Ça recuit les 46 morceaux de la bibliothèque qui ont déjà tous leurs caches
chauds (jamais un modèle relancé pour le rapport — un morceau sans cache est
marqué « froid », pas mesuré), et compare mesure par mesure à une version de
référence gelée (`baseline`). **« Identiques 46/46 » veut dire : rien n'a
changé pour l'auditeur.** Un chart différent n'est PAS automatiquement une
régression — c'est un fait à montrer à Louis, qui tranche.

Quand des mesures changent :

```
.venv/bin/python -m tools.avant_apres --before state/cache/golden/baseline --after state/cache/golden/run
```

écrit une page où chaque mesure changée s'écoute en un clic (AVANT / APRÈS),
et republie les deux versions dans la bibliothèque pour les ouvrir dans la
vraie app. Louis écoute, puis dit : on garde, ou on corrige avant de commiter.

Pour republier une bibliothèque après un vrai changement de comportement :
`tools.golden --publish --backup <dossier de sauvegarde>` (voir
`.claude/skills/ship/SKILL.md`).

## Règle de synchronisation entre sessions

Louis fait tourner plusieurs sessions Claude sur le même dépôt en même temps.
Avant de modifier un fichier que vous n'avez pas créé vous-même dans cette
session, `git log`/`git status` dessus : un changement récent inattendu est
probablement le travail d'une autre session, pas une erreur à corriger.
`git add <chemins> && git commit` en une seule commande — un ajout puis un
commit séparés a déjà vu ses fichiers ajoutés récupérés par le commit d'une
autre session entre les deux.

## Ce qui reste ouvert

Voir `docs/known_issues.md` (court, à jour) pour la liste complète et les
détails. En résumé : un test hérité qui documente un changement de
comportement pas encore mis à jour, la recherche UG bloquée par un anti-bot,
deux morceaux de la bibliothèque sans cache SongFormer, et la suite de la
recherche sur la qualité des sections (repoussée après la fin du refactor,
décision de Louis).

## Où est l'historique

- `docs/refactor_2026-09/plan.md` — les décisions verrouillées du refactor.
- `docs/refactor_2026-09/sprints.md` — le journal, sprint par sprint, avec le
  raisonnement (pas juste le résultat).
- Tag git `pre-refactor-2026-09-14` — l'état du dépôt juste avant que le
  refactor commence.
- `archive/` — scripts et migrations d'avant le refactor, gardés pour
  mémoire mais plus exécutés en prod.
- `docs/archive/known_issues_2026-07_2026-09.md` — l'ancien journal
  d'incidents (27 984 lignes), pour retrouver un détail ancien.
