# X ≈ U·V·W : trois étages — mesuré, et négatif

Louis, 2026-08-17 : « est-ce qu'on peut faire un X = UVW, avec la granularité
2 mesures, et la matrice V qui fait la liaison en bi-mesures pour en faire des
blocs de 4 ou les laisser en queues ? »

**Oui, on peut. Je l'ai fait. Les deux étages perdent contre ce qui est déjà
livré.** Ce document dit pourquoi, parce que le mécanisme est plus utile que le
résultat.

## La forme exacte, telle qu'il la décrit

    X  (n_bi × 24)    une bi-mesure = 2 mesures d'accords mises bout à bout
    W  (m × 24)       l'alphabet des bi-mesures
    H  (n_bi × m)     quelle bi-mesure est quel motif
    P  (n_bi−1 × 2m)  TOUTES les paires adjacentes, encodées par fente
    V  (k × 2m)       la grammaire : quelles bi-mesures se soudent
    U  (n_bi−1 × k)   où chaque soudure se joue

L'encodage **par fente** (les m premières colonnes = la 1ʳᵉ bi-mesure, les m
suivantes = la 2ᵈᵉ) est indispensable : sans lui une section est un *ensemble*
de bi-mesures et l'ordre est perdu.

## Ce que ça recoupe dans le projet

Cette pile existe déjà, en **dur** :

| étage | version livrée | comment |
|---|---|---|
| W, l'alphabet | `soudure.grille_et_mot` | égalité chord-tone au seuil 0,93 |
| V, la liaison | `bpe_lab` / `phrases4` | on soude la paire la plus **fréquente** |

L'idée de Louis en est la version **molle** : une bi-mesure devient un mélange
de motifs au lieu d'une lettre, et l'incertitude du bas se propage au lieu
d'être jetée. C'était une bonne raison de l'essayer.

## Résultat 1 — la liaison ne trouve pas la couture

Easy On Me, grille de 2 mesures **rigide depuis la mesure 0, aucune couture
donnée**. m = 8, k = 6. Force de liaison = pureté de U × (1 − résidu), puis
sélection gloutonne sans chevauchement.

    LIAISON TROUVÉE : 13 blocs, 5 queues
    queues : mes. 17-18, 27-28, 33-34, 55-56, 61-62

La vraie liaison du morceau est **mes. 25-26**, le vrai résidu **mes. 63**.
Aucune des deux n'est posée au bon endroit : la plus proche tombe à deux
mesures dans les deux cas (27-28 et 61-62), c'est-à-dire **la bi-mesure d'à
côté** — pour un objet qui fait deux mesures, c'est le rater. Pire : le bloc
« mes. 23-26 » **enjambe** la couture avec une force élevée (0,83), collant la
fin du tag au début du couplet 2.

(Piège de mesure que j'ai failli publier : avec une tolérance de ±2 mesures ces
deux ratés comptaient comme des succès. À ce grain, la tolérance vaut la moitié
de l'objet mesuré.)

## Résultat 2 — l'alphabet mou perd contre l'alphabet dur

18 morceaux annotés. On ne mesure QUE l'alphabet : deux bi-mesures que Louis
met dans la même section reçoivent-elles la même lettre ? À taille d'alphabet
égale (6,8 lettres en moyenne) :

| alphabet | précision | rappel | F1 |
|---|---|---|---|
| **dur** (égalité, livré) | 0,568 | **0,491** | **0,469** |
| mou (NMF, argmax) | 0,572 | 0,336 | 0,402 |

Le rappel s'effondre : la version molle **sépare** des bi-mesures qui sont la
même musique.

## Le mécanisme, et il est le même pour les deux étages

**La NMF optimise la RECONSTRUCTION ; les deux étages posent une question de
RÉPÉTITION. Les deux objectifs tirent en sens inverse.**

Reconstruire récompense le fait de **découper** une matière fréquente en
plusieurs composantes — c'est ce qui fait baisser le résidu le plus vite.
Répéter récompense le fait de la **regrouper**. D'où :

* l'alphabet : la NMF éclate la cellule de couplet, très fréquente, en deux ou
  trois composantes ; le rappel tombe ;
* la liaison : « ces deux bi-mesures vont ensemble » est un fait de
  **comptage** (cette paire revient-elle ailleurs ?), et l'objectif de
  reconstruction ne le note nulle part.

Et ça explique rétrospectivement pourquoi la démo `X ≈ UV` du 2026-08-17
marchait si bien : **je lui avais donné les bons blocs**. Toute l'information
de répétition était déjà dans l'entrée ; la factorisation ne faisait que la
dernière étape, la facile.

## Ce qu'il faut en garder

1. **Là où la question est la répétition, garder les algorithmes qui comptent** :
   l'alphabet par égalité, la liaison par BPE. Tous deux déjà livrés, tous deux
   meilleurs ici.
2. **Utiliser la factorisation là où le comptage ne peut rien** : rapprocher
   deux occurrences d'une même section **de longueurs différentes**. Le
   comptage exige une correspondance exacte ; `merge_letters` exige la même
   longueur et le même alignement (comparaison en diagonale sur min(L₁,L₂)) —
   c'est exactement ce qui casse Yesterday, deux passages annotés B de 6 et
   8 mesures. C'est le seul endroit mesuré où la factorisation apporte ce que
   rien d'autre n'apporte.
3. Donc : **pas un remplacement à trois étages, un greffon d'un étage.** Garder
   le découpage actuel, remplacer seulement `merge_letters` par un
   rapprochement sans alignement.

## Ce que ça ne dit pas

* Un seul jeu d'hyperparamètres (m = 8, k = 6) et une seule graine. Un balayage
  pourrait remonter la version molle — mais il devrait remonter de 0,402 à plus
  de 0,469 rien que pour égaler, et le mécanisme ci-dessus dit que non.
* La sélection de la liaison est **gloutonne**. Une programmation dynamique
  ferait mieux ; elle ne changerait pas le fait que le signal noté ignore la
  répétition.
* La liaison n'a été essayée que sur Easy On Me (règle #5 : c'est une
  hypothèse, pas un résultat). L'alphabet, lui, est mesuré sur les 18.

**À écouter plutôt qu'à lire** : `/plots/uvw_demo.html` — les deux alphabets en
bandes de couleurs alignées sur les mesures, les soudures de V, la vraie
couture par-dessus, chaque plaque cliquable.

Scripts : `scripts/uvw_demo.py` (la page), `scripts/uvw_screen.py` (les trois
étages, la liaison), `scripts/uvw_alphabet.py` (dur contre mou sur les 18).


---

# Suite (même jour) : le trou placé par le fit — ça, ça marche

Louis : « on reste sur un X = UV, mais on lui donne l'opportunité de pouvoir
faire des trous de 1 ou 2 mesures pour qu'il puisse fitter au mieux ? » puis
« et oui il faut un petit k ».

**Résultat positif.** C'est la réponse à la limite laissée ouverte plus haut —
« la factorisation résout le nommage, pas le découpage ». Avec le droit de
sauter 1 ou 2 mesures, elle résout aussi le découpage.

## Pourquoi ça marche là où le UVW échouait

Mon objection au UVW était : *la reconstruction ne mesure pas la répétition*.
Elle tombe ici, et pour une raison précise — **le k serré**. Avec 5 composantes
pour 15 blocs, la seule façon de tout reconstruire est que les blocs tombent
vraiment dans 5 groupes qui se répètent. Une grille décalée fabrique 15 blocs
tous différents, et rien de rang 5 ne peut les rendre. Le petit k transforme
« ça fitte » en « ça se répète ».

## Ce qui est mesuré

Easy On Me, balayage de tous les trous de 1 ou 2 mesures, **à nombre de blocs
égal** (sans cette contrainte, une grille qui perd un bloc gagne sans rien dire) :

    k =  2  3  4  5  6  8
    rang  3  5  7  1  6  1     (sur 73 grilles candidates)

La vraie grille — sauter les mesures 25-26 — est **première à k=5 et k=8**, et
dans les 10 % de tête partout. Sur ce morceau elle fait passer les frontières
tombant sur un bord de bloc de **4/10 à 10/10**.

Sur les 18 annotés (k=5, un seul trou) : **0,449 → 0,678** de frontières sur un
bord de bloc, **7 morceaux montent, aucun ne recule**.

## Le résultat inattendu

Sur Stand By Me et Happy, le meilleur trou est à la **mesure 2**. Ce n'est pas
une couture interne : c'est la **levée** du morceau. Le fit retrouve tout seul
ce que « Set bar 1 » fait à la main — et sur ces deux morceaux il fait passer
l'alignement de 0,00 à 1,00.

## Ce que ça ne dit pas, et c'est important

* **Le choix de k n'est pas résolu.** Le rang ne décroît pas avec k (3, 5, 7,
  1, 6, 1) : « il faut un petit k » est vrai comme mécanisme, faux comme règle
  monotone. C'est la première chose à régler avant de brancher.
* **Un seul trou.** Plusieurs trous demandent une recherche conjointe — même
  leçon que `soudure._meilleure_grille`, où la meilleure paire ne contient pas
  le meilleur élément seul.
* L'alignement des frontières est un **proxy** de la qualité des sections, pas
  le score de sections lui-même. Le vrai test est de brancher cette grille dans
  le détecteur et de repasser `section_bench`.

**À écouter** : `/plots/trou_demo.html` — le paysage des trous (chaque barre
s'écoute), la grille avant/après, et les 18 morceaux.
Script : `scripts/trou_demo.py`.
