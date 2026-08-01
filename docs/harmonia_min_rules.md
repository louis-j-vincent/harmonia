# HARMONIA-MIN — Les règles de base (référence)

Dictées par Louis le 2026-08-01. Ce document est **la recette de référence** :
tout écart du pipeline par rapport à ces règles doit être soit corrigé, soit
listé explicitement dans l'audit en bas de ce fichier. À maintenir à jour.

## Règle 0 — la source

**La source de base est l'audio.** Tout part de là.

## Règle d'or — LA GRANULARITÉ EST LA BARRE

> « La granularité, c'est la barre. Ce n'est pas le temps, ce n'est pas les
> accords, c'est la barre. »

Une fois la grille établie, on ne travaille plus qu'en logique de **barre,
demi-barre, quart de barre**. Chaque barre correspond à une fenêtre
temporelle. La rigidité de la barre est **la seule source de vérité pour
l'alignement** — c'est elle qui garantit qu'on ne se trompe pas quand on
représente les notes, quand la tête de lecture suit, quand on détecte les
sections.

## L'ordre des processus pour étudier une chanson

1. **BPM + grille de barres.** Trouver le tempo et établir la grille : savoir
   dire « cette barre est ici, cette barre est là ». Outil actuel :
   **Beat This!** (beats + downbeats natifs). À partir d'ici, tout est en
   logique de barre (règle d'or).

2. **Accords par musx.** musx détecte les accords ; chaque changement est
   **snappé à la demi-barre la plus proche**.

3. **Propagation.** Pas d'accord détecté sur une demi-barre → **le dernier
   accord se propage**. Résultat garanti : chaque demi-barre porte un accord
   (aucune barre vide ; le « % » n'est qu'une surcouche d'affichage,
   ajoutée plus tard).

4. **Répétitions par la SSM.** La matrice de similarité détecte les motifs
   répétés — des patterns de 2 ou 4 barres qui se répètent, visibles
   directement dans la matrice.

5. **Sections.** Les répétitions détectées sont mergées pour définir les
   sections (A, B, C…).

6. **Détection des variantes — écart individuel vs écart collectif**
   (raffiné par Louis le 2026-08-01). Hypothèse classique : une section se
   répète 3-4×, mais la DERNIÈRE occurrence (ou sa/ses dernière(s) barre(s))
   peut différer pour transitionner vers la section suivante. Le test n'est
   donc PAS la variance globale de la pile : on compare **l'écart individuel
   de chaque barre au centroïde de sa pile** contre **la norme collective**
   (médiane + MAD des écarts de tous les membres). Une barre anormalement
   écartée (z robuste > OUTLIER_Z=3) est une **variante** : elle n'est pas
   squashée et garde son propre décodage. Les barres de transition (fins de
   section) sont l'endroit ATTENDU des variantes, mais le test s'applique à
   tous les membres (les cellules cadence intérieures sont attrapées aussi).
   Mesuré sur This Love : vraies variantes z=5.7–38, membres normaux z≤1.7
   — dont la barre D° pré-bridge, détectée automatiquement.

7. **Squash + empilement + deuxième passe.** Si les occurrences sont
   similaires : on les **squashe** (représentation unique ×2/×3/×4 —
   l'affichage), ET on les **empile** : la version moyennée est redonnée au
   prédicteur (musx) qui réinfère une deuxième fois avec l'information
   moyennée — la variance du bruit est réduite (~1/√n). Les accords du
   consensus se redistribuent sur chaque occurrence contributrice.

---

## Audit d'harmonia_min contre ces règles (2026-08-01)

| # | règle | état | détail |
|---|---|---|---|
| 0 | source = audio | ✅ | docs/audio/*.m4a → ffmpeg→wav → tout en découle |
| or | granularité = barre | ✅ (voir écart 1) | grille = downbeats réels Beat This! ; layout accords, sections, repli, playhead : tout est indexé en barres ; les secondes ne servent qu'à l'affichage/audio |
| 1 | BPM + grille (Beat This!) | ✅ | beats.py, librosa banni, échec bruyant |
| 2 | accords musx snappés à la **demi-barre** | ❌ ÉCART 1 | le re-décodage autorise un changement à chaque **temps** (coûts 15/45/100). L'audit du 2026-08-01 a réfuté ma note « de facto 100 % demi-barre » : Let It Be livre des onsets au temps 1 (barres 19/33/35/69, dont un G F C à 3 accords/barre). Question Q1 du doc d'audit — à trancher par Louis |
| 3 | propagation du dernier accord | ✅ | carries écrits partout, zéro barre vide (« % » supprimé, redeviendra une surcouche) |
| 4 | répétitions via SSM | ✅ | SSM chroma NNLS demi-barre, damier flouté, autocorrélation de périodes {2,4,8} |
| 5 | merge → sections | ✅ | lettres par blocs hors-diagonale + failsafe « se recoupent » + fusion singletons |
| 6 | écart individuel vs collectif (variantes) | ✅ (raffiné + branché 2026-08-01) | folding.py OUTLIER_Z=3 : z robuste (médiane+MAD) de l'écart au centroïde, appliqué à chaque membre ; variantes = dépliées, gardent leur 1ʳᵉ passe. Vérifié : This Love attrape le D° pré-bridge (z=14) et les cellules Ab G intérieures (z≈38) ; She Will Be Loved refuse de plier (cohérence 0.59) tant que l'alignement n'est pas réparé |
| 7 | squash + empilement + 2ᵉ passe musx | ✅ | folding.py : boucles internes (phase 1) + passages pliés (phase 2), _template_chords, tuilage ×3, redistribution |

**Écart 1 (à trancher)** : rendre le snap demi-barre structurel dans le
re-décodage (make_beat_arr ne poserait des transitions légales que sur les
demi-barres) ou garder les quarts autorisés (les turnarounds à 2 accords/
demi-barre existent — « Ab G » = 2 accords dans la même demi-barre serait
impossible en demi-barre stricte… en fait « Ab G » = beats 0 et 2 = deux
demi-barres ✓). Note : la grille d'affichage en quarts reste inchangée quoi
qu'il arrive.

**Écart 2 — RÉSOLU (2026-08-01)** : rôles clarifiés par Louis — cosinus =
décision de merge, variance normalisée = vérification du squash par
demi-barre (VAR_MAX=0.15 provisoire, un seul endroit à changer). Le décodage
des templates autorise désormais le changement à la demi-barre au même coût
qu'à la barre (quarts chers) — cas mesuré : la 2e moitié du Dø7 de This Love
est réellement partagée entre passages (Ddim 0.32 / Bb 0.16 / Ab 0.08), le
consensus garde Dø7 ; descendre VAR_MAX vers 0.10 rendrait cette demi-barre
à ses passages.
