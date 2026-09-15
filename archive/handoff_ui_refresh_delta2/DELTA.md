# Harmonia UI refresh — lot 2 (delta)

Suite du premier handoff, **déjà en prod**. Rien ici ne le remplace : ce sont les
écrans suivants, plus un correctif sur un fichier déjà livré.

Maquette de référence : **`Harmonia UI Refresh.dc.html`**, turns 3 à 6
(badges `3a`, `4a`, `5a`, `6a`). `STYLE.md` est joint à l'identique du lot 1 —
si la session est neuve, le coller en premier.

Ordre de collage, à la suite de ce qui est déjà en place :

```
tight_glyph.js  →  immersive_play.js  →  library_folders.js  →  search_sources.js
```

Tous sont des blocs à coller **dans l'IIFE de `window.APP`**, comme au lot 1.

---

## ⚠ Correctif sur un fichier déjà livré

`chart_chrome.js` est joint **corrigé**. Le bouton clé/forme du sous-titre
(`const meta = el("button", …)` dans `chartToolbar()`) faisait 24 px de haut : un
vrai contrôle interactif sous la barre des 44, c'est-à-dire exactement ce que
`STYLE.md` interdit et ce que le contrôle d'acceptation n°2 attrape. Il garde son
allure légère — pas de fond, pas de bordure — mais prend `height:44px` avec des
marges négatives, donc la barre d'outils ne grandit pas.

Si le lot 1 est en prod tel quel, c'est le seul fichier à reprendre.

---

## §6 — Lecture immersive (`immersive_play.js`) → design `3a`

Jouer, c'est lire. Pendant la lecture en mode Read, les deux bandeaux s'effacent
après 2,6 s sans interaction et la grille prend tout l'écran — les cellules passent
de 74 à 88 px, c'est le gain réel de l'opération, pas du vide en plus.

**Faire revenir les bandeaux**, trois gestes, tous déjà présents dans le shell :
la poignée en haut de la grille (un tap), un glissé vers le bas depuis le haut,
ou un tap n'importe où entre deux accords. La poignée doit être **visible** —
elle ne l'était pas, et sans elle le mode est une impasse.

En **Annotate on ne cache jamais** : l'utilisateur y travaille, ses outils
restent sous la main.

**Toucher un accord fait monter une carte de voicing par-dessus la grille** —
deux claviers, une main chacune — sans couper l'audio et sans ouvrir la grande
feuille modale. Elle se referme au tap suivant. À l'arrêt, un tap ouvre la
feuille comme avant : c'est là qu'on édite.

Nouveau réglage dans la feuille `Aa`, `S.piano` :

| valeur | comportement |
| --- | --- |
| `never` | rien ne monte, la grille reste seule |
| `tap` | la carte apparaît sur l'accord touché — **défaut** |
| `follow` | le clavier reste en bas et suit le playhead |

`follow` **remplace** l'ancien `S.coach` au lieu de le doubler ; la migration
depuis `harmCoach` est dans le fichier. Deux commandes qui disent la même chose,
c'est une de trop.

## §7 — Barres à plusieurs accords (`tight_glyph.js`) → design `4a`

Relevé sur la capture iReal de « 9.20 Special » (barre `C7 B7 B♭7 A7` dans ~90 px).
Trois règles, dans cet ordre d'importance :

1. **Empiler, pas répartir.** Dès 2 accords, la barre les pose à gauche avec un
   écart fixe. C'est un changement de *layout* dans `buildIReal()`, pas de glyphe,
   et c'est de loin le plus gros gain — c'est la seule raison pour laquelle quatre
   accords tiennent. Ce qu'on perd : l'abscisse ne dit plus le temps. Personne ne
   lit le temps à la position horizontale sur un chart ; on le lit au nombre
   d'accords dans la barre. iReal fonctionne comme ça depuis toujours.
2. **La qualité en indice, tuckée.** Colonne étroite collée à la lettre
   (`margin-left:-1px`), altération en haut, qualité en bas sur la ligne de base.
   L'italique de Georgia dépasse à droite : ce dépassement est de la place
   gratuite, on s'en sert au lieu de le subir.
3. **La taille suit le nombre d'accords** : 30 / 25 / 22 / 19 px pour 1 / 2 / 3 / 4.
   19 est un plancher dur — en dessous on ne lit plus à un mètre avec les mains
   occupées, et c'est la barre qu'il faut couper.

L'exception à garder : un accord qui ne tombe pas sur un temps régulier (`beat`
non entier) garde sa position proportionnelle, sinon le chart ment. Le mode
« normal » (non compact) garde la grille de temps : le serrage est le contrat de
l'écriture compacte, pas un changement global.

## §8 — Accueil & dossiers (`library_folders.js`) → design `5a`

Training mode et Section cleanup **sortent de l'accueil**. Il ne liste plus rien :
deux portes (*Chercher une nouvelle chanson*, *Mes charts*) et *Reprendre*. Le
champ de recherche est **dans** la première carte — on ne fait pas taper
l'utilisateur sur un écran pour l'emmener sur un autre où il retape.

`renderCharts()` est le nouvel écran : filtre, Dossiers avec compteur et
*Nouveau*, puis *Hors dossier*. En mode Modifier, chaque ligne gagne *Ranger*.

**Un chart appartient à au plus un dossier** — c'est un classeur, pas des tags :
« ranger » doit vider « Hors dossier ». Si des tags deviennent nécessaires un
jour, ce sera une autre fonctionnalité, pas une extension de celle-ci.

Les chansons s'écrivent **`Artiste / Chanson`**, artiste en gris, titre en encre,
sur une seule ligne avec ellipse — jamais deux lignes, la liste doit se scanner.

Deux dépendances backend, petites, détaillées en bas du fichier :

- `/api/library` gagne un champ `artist` quand il est connu. En attendant,
  `chartName()` le devine sur le tiret du titre YouTube, ce qui marche pour
  « Artiste - Titre » et échoue sur le reste. Il doit être **éditable** côté
  client — le titre YouTube ment souvent : `POST /api/chart-meta/<file>`.
- `POST /api/folders` avec `{"order": [...], "of": {"<file>": "<dossier>"}}`.
  L'appel est déjà là et échoue en silence ; jusque-là tout vit en localStorage.

## §9 — Recherche à trois sources (`search_sources.js`) → design `6a`

Nouvel écran `search`. Une source à la fois, choisie en haut, **YouTube par
défaut**. La ligne sous le sélecteur dit ce que la source *implique*, parce que
les trois ne font pas la même chose : YouTube écoute et déduit (attente, écran de
chargement) ; iReal et les tablatures importent une grille déjà écrite et
**ouvrent le chart directement, sans écran d'analyse**.

La troisième source est nouvelle côté UI et demande deux routes :

```
POST /api/tab-search  {q}    → {results:[{id,title,artist,kind,rating,url}]}
POST /api/tab-import  {url}  → {url:"/chart/<file>"} | {error}
```

`harmonia/tab_fetcher.py` + `tab_parser.py` + `tab_aligner.py` font déjà le
travail : ces routes ne sont qu'une façade HTTP dessus. Tant qu'elles n'existent
pas, laisser l'onglet **visible** et renvoyer une liste vide — l'état « rien
trouvé » est déjà écrit, et masquer l'onglet coûterait un deuxième chemin de
rendu à retirer plus tard.

Retirer de l'ancien `renderLibrary()` : la barre de recherche, le toggle
YouTube/iReal, les cartes Training mode et Section cleanup.

---

## Acceptance (lot 2)

1. `[...document.querySelectorAll('button')].filter(b=>{const r=b.getBoundingClientRect();return r.width<44||r.height<44}).map(b=>b.textContent)`
   revient vide sur chart, prompter, accueil, charts et search.
   **Une seule exception**, documentée dans `search_sources.js` : le `×` du champ
   de recherche, qui vit dans un champ de 48 px déjà tapable sur toute sa surface.
2. Aucun emoji : `grep -P '[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]' harmonia_min/app_shell.html` vide.
3. En lecture Read, les bandeaux partent après 2,6 s et **les trois gestes** les
   ramènent. En Annotate ils ne bougent jamais.
4. `S.piano="never"` → aucune carte ne monte. `"follow"` → le clavier du bas suit,
   et un tap sur un accord ouvre la grande feuille.
5. Une barre de 4 accords : zéro chevauchement, contenu dans la cellule, aucun
   glyphe sous 19 px.
6. La pastille des résultats (`Analyser`, `Importer`, `Déjà là`) n'a **pas** de
   `onclick` — c'est la carte qui porte le tap.
7. Un chart rangé dans un dossier disparaît de « Hors dossier ». Un chart supprimé
   disparaît des dossiers sans nettoyage explicite.
