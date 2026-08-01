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

6. **Test de merge des sous-sections — moyenne/variance du chroma.**
   Pour chaque sous-section répétée (ex. : les 4 premières barres répétées
   3×), on concatène les occurrences et on calcule **par demi-barre** :
   * la **moyenne** du chroma,
   * la **variance** du chroma,
   * la variance **normalisée par la moyenne** (pour neutraliser
     l'intensité sonore propre à chaque demi-barre).
   Si la variance normalisée de chaque (demi-)barre passe **sous un seuil**,
   les occurrences sont déclarées identiques. Le seuil est **choisi par
   Louis sur graphiques** (métrique tracée sur cas similaires ET cas
   différents, pour vérifier qu'elle différencie).

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
| 2 | accords musx snappés à la **demi-barre** | ⚠️ ÉCART 1 | le re-décodage autorise un changement à chaque **temps** (quart de barre), coûts gradués downbeat/mi-barre/autre. Sur This Love le résultat tombe de fait 100 % sur barre/demi-barre (76 beat-0, 41 beat-2, 0 ailleurs), mais la règle n'est pas STRUCTURELLE. À confirmer par Louis : interdire structurellement les quarts ? |
| 3 | propagation du dernier accord | ✅ | carries écrits partout, zéro barre vide (« % » supprimé, redeviendra une surcouche) |
| 4 | répétitions via SSM | ✅ | SSM chroma NNLS demi-barre, damier flouté, autocorrélation de périodes {2,4,8} |
| 5 | merge → sections | ✅ | lettres par blocs hors-diagonale + failsafe « se recoupent » + fusion singletons |
| 6 | test de merge = **variance/moyenne du chroma** | ⚠️ ÉCART 2 | les gates actuels sont des cosinus (membre→centroïde 0.85, médiane pairwise 0.85) — PAS la variance normalisée dictée. Graphiques produits (scratchpad/fold_variance_study.png) pour que Louis choisisse le seuil ; à brancher ensuite à la place des cosinus |
| 7 | squash + empilement + 2ᵉ passe musx | ✅ | folding.py : boucles internes (phase 1) + passages pliés (phase 2), _template_chords, tuilage ×3, redistribution |

**Écart 1 (à trancher)** : rendre le snap demi-barre structurel dans le
re-décodage (make_beat_arr ne poserait des transitions légales que sur les
demi-barres) ou garder les quarts autorisés (les turnarounds à 2 accords/
demi-barre existent — « Ab G » = 2 accords dans la même demi-barre serait
impossible en demi-barre stricte… en fait « Ab G » = beats 0 et 2 = deux
demi-barres ✓). Note : la grille d'affichage en quarts reste inchangée quoi
qu'il arrive.

**Écart 2 (en cours)** : la métrique variance-normalisée est calculée et
tracée ; dès que Louis fixe le seuil sur les graphiques, elle remplace les
gates cosinus dans folding.py (mêmes points d'accrochage).
