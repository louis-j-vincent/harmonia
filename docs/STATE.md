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
| `settings.py` | Le SEUL endroit qui lit les variables d'environnement (port, dossier et appareil du modèle musx). Tout le reste (quelle loi de fusion, quel repli…) est une CONSTANTE ici, pas un réglage. |
| `cache.py` | Une seule façon de nommer un fichier de cache par morceau (beats, accords musx, chroma, sections). |
| `pipeline.py` | Le chef d'orchestre : audio → temps → accords → mesures → sections → repli → tonalité → chart. |
| `beats.py` | Les temps de la chanson (modèle Beat This!). |
| `musx.py` | Les probabilités d'accord, image par image (modèle musx, un ensemble ISMIR 2019). |
| `nnls_features.py` | Le chroma (quelle note sonne quand) qui nourrit `musx.py`. |
| `bars.py` | Découpe les temps en mesures. |
| `sections/` | Pas de détecteur automatique (le chart sort en une section « A » + son intro, `pipeline._one_section` ; SongFormer supprimé le 2026-09-30) : la ressemblance mesure-à-mesure chord-tone que Soudure et l'outil sections réutilisent, et la recherche des reprises d'une brique tracée à la main. |
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
deux morceaux de la bibliothèque sans audio, et la suite de la
recherche sur la qualité des sections (repoussée après la fin du refactor,
décision de Louis).

## Le dessin de la grille (journal des décisions d'affichage)

L'écran chart (`harmonia/static/screens/chart.js`, `buildIReal()`) a déjà
régressé sans trace ; toute décision de LOOK s'écrit ici AVANT d'être codée.

**Deux styles de grille, au choix dans la feuille « Aa » → « Grid »**
(préférence `PREF.grille`, `localStorage["harmGrille"]`, comme le thème) :

- **Épurée (`epure`, par défaut depuis le 2026-09-30).** Demandée par Louis
  sur capture (« pas de cellules visibles qui séparent chaque barre, juste
  des lignes verticales sans lignes horizontales »). La capture est l'ancienne
  page statique de juillet (`docs/plots/inferred_bye_bye_blackbird.html`,
  classe `.measure`). Ce qu'on reprend :
  - aucun trait horizontal, aucun cadre arrondi : seules des barres de mesure
    verticales fines (`T.rule`, 1px), continues d'une rangée à l'autre, y
    compris à travers l'écart entre deux sections (l'écart est un
    `padding-top` de la rangée, pas une marge, sinon le trait se coupe) ;
  - un trait à droite de la 4e colonne, et la double barre fine (║, 4px
    `double`) à gauche de la première rangée de chaque section — elle
    remplace la barre noire continue du treillis ;
  - l'étiquette de section (A, B…) posée DANS le haut de sa rangée, en haut à
    gauche, pas dans un gap au-dessus ;
  - des rangées un peu plus hautes (+8px) : sans trait horizontal, c'est l'air
    qui sépare les rangées ;
  - un accord seul sur le temps 1 est centré dans sa mesure (comme la
    capture) ; dès deux accords, la grille des temps reste la règle ;
  - la fin du morceau porte une barre finale d'encre (fine + épaisse), pas
    rouge — le rouge est réservé à la tête de lecture ;
  - la mesure jouée : une pastille teintée arrondie derrière les accords, qui
    « pop » à chaque changement de mesure, et la barre de mesure de droite qui
    s'allume en rouge (3px) — c'est la barre qui avance ;
  - les crochets de fins (1./2., « A alt ») restent des dessins par-dessus ;
    une rangée de fins nue garde ses seuls traits verticaux ;
  - la place au-dessus des accords reste libre pour les petites suggestions
    d'accords à la iReal (rien de neuf n'y est dessiné).
  - **la typographie des accords de la capture (Louis, 2026-09-30 : « même
    dans la police d'écriture des accords et les symboles des extensions »).**
    Recopiée de la page de juillet (CSS `.chord .root/.qual/sup/.acc` et
    `typesetQuality()`), dans `glyphEpure()` (`ui/kit.js`) :
    Georgia italique **graisse normale** (400, pas 600) ; fondamentale 27px,
    qualité 17px, exposant = 0,62 × la qualité ; ♭/♯ de la fondamentale à
    0,6em, remontés de 0,12em, collés de −0,1em ; deux accords dans la mesure
    → 19/12px. Symboles : `m` petit en ligne puis l'exposant (Gm⁷), △ pour
    maj7 (F△⁷), ø et ° en ligne, `6`, `7`, `9`, `13`, `alt` en exposant
    (D⁷ᵃˡᵗ), `sus4` en ligne, basse slash `/A` à la taille de la qualité.
    Écart assumé à la page de juillet : `7sus4`/`9sus` montent en entier
    en exposant (⁷ˢᵘˢ⁴, comme iReal), pas sus4⁷ — en ligne, `D9sus4` à deux
    accords débordait sur la mesure voisine (vu au rendu 390px) ; un `sus4`
    seul reste en ligne. Sur ordinateur les tailles montent de ×1,18 (même
    rapport). Ne vaut que pour **Clean + notation Normal** ; l'écriture
    Compact garde `glyphTight`, et « Cells » garde l'ancienne police (600,
    exposants à 0,44). Les couleurs de certitude, N.C., la pédale et les
    petites suggestions au-dessus de l'accord ne changent pas.
- **Cellules (`treillis`).** Le treillis d'avant, inchangé (§1–§8 du
  2026-09-17, commit 817d5ab) : cellules bordées, barre noire continue à
  gauche, badge dans le gap.

**Les reprises `||: … :|| ×N` (2026-09-30, Louis : « parfait » à l'option
(c)).** Une section écrite par sa boucle minimale (champ `boucle` de la
section, `folding.boucle_minimale_ecrite`) porte, dans les DEUX styles : à
gauche de sa première mesure deux points de reprise (après la barre
d'ouverture, qui devient la reprise « ||: ») ; à droite de sa dernière mesure
écrite, deux points + une double barre fine-épaisse d'encre (« :|| ») ; et
« ×N » en petit, encre discrète, en haut à droite de cette mesure — N = les
tours ENTIERS (un dernier tour incomplet reste audible et suivi par la tête
de lecture, il ne compte pas). Des passages qui ne jouent pas le même nombre
de tours écrivent « ×3/2 » (dans l'ordre des passages). Couleur d'encre
(`T.ink`), jamais le rouge (réservé à la tête de lecture) ; rien n'est posé
au-dessus des accords (place gardée aux petites suggestions). Les autres
lectures d'un tour restent la petite lecture `var` de la case.

**Le curseur « Section merging » pilote aussi la boucle (2026-10-01, Louis :
« mets-moi le curseur qui permet de choisir à quel point tu factorises »).**
Un seul curseur, une seule question — à quel point deux passages peuvent
différer et s'écrire une fois. Crans : 0 « tout écrit » (aucune boucle,
toutes les fins) ; 1 défaut (boucle ×N seulement si tous les tours ont la
même harmonie, couleurs permises ; une fin seulement si l'accord change) ;
2 / 3 (différences proches / apparentées tolérées, montrées en petite
lecture) ; 4 « un bloc » (toujours la boucle, différences en petite
lecture). Bornes affichées : « write it out » ↔ « one block ».

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
