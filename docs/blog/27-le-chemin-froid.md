# 27 — Le chemin froid : 42 s → 23 s, et pourquoi « 8 mesures à la fois » n'est qu'un aperçu

*2026-08-18*

Louis : « j'aimerais speedup l'arrivée des premiers accords de la chanson, est-ce
qu'on peut trouver des astuces ? demander à musx de faire 8 barres par 8 barres
par exemple ? »

L'entrée 23 avait coupé le chart brut du raffinement, ce qui règle le cas
**cache chaud** (1,5 s). Ce qui reste long, c'est un morceau **jamais analysé**.
Personne n'avait chronométré ce chemin-là.

## Où passent les secondes (cache vide, `h_D3VFfhvs4`, 9 min 26)

| étape | avant | après | comment |
|---|---|---|---|
| battues (Beat This!, CPU) | 11,8 s | 11,8 s | inchangé — voir plus bas |
| CQT | 3,9 s | 3,9 s | inchangé |
| 5 réseaux musx | 14,4 s | **3,1 s** | MPS (GPU Apple) |
| re-décodage (8 latences) | 3,1 s | 2,4 s | — |
| **chart brut** | **42,5 s** | **≈23 s** | |

Le téléchargement yt-dlp s'ajoute par-dessus : 18 s mesurées sur 8,7 Mo dans le
log du 9 août, sans `-N` (fragments concurrents).

## Le gain gratuit : MPS

Les 5 réseaux tournaient sur CPU. Ils tournent sur le GPU intégré **sans changer
d'un bit ce qu'ils répondent** :

| morceau | durée | CPU | MPS | accords redécodés |
|---|---|---|---|---|
| 0DdCoNbbRvQ | 252 s | 6,20 s | 1,75 s (×3,5) | 100/100 identiques |
| autumn_leaves | 422 s | 10,96 s | 2,52 s (×4,3) | 160/160 identiques |
| 4JkIs37a2JE | 235 s | 5,96 s | 1,68 s (×3,6) | 163/163 identiques |
| DksSPZTZES0 | 290 s | 8,06 s | 2,00 s (×4,0) | 118/118 identiques |

« Identiques » au sens fort : argmax des postérieures égal à 100 %, |Δ| max
0,0000, segments du re-décodage égaux en label **et** en frontière, et sur
`h_D3VFfhvs4` les mesures du chart brut sont octet pour octet les mêmes
(`HARMONIA_MUSX_DEVICE=cpu` vs `mps`).

Un seul obstacle : `ChordNet.init_hidden` fabriquait l'état caché du LSTM sur
CPU en dur. Corrigé dans `harmonia_min/musx.py`, pas dans le clone.

**Résultat négatif à ne pas refaire : Beat This! est 9× PLUS LENT sur MPS**
(7,3 s CPU contre 67,5 s MPS, mêmes battues à 0 ms près). Le tracker reste sur
CPU. Ce qui tombe bien : les deux étapes occupent alors deux unités différentes
et pourraient tourner en parallèle (non fait).

## « 8 mesures à la fois » : ça marche, mais ça change les accords

Mesuré sur `0DdCoNbbRvQ` (4 min 12, mesure de 2,8 s), tranches recollées avec du
contexte de chaque côté, même grille de battues, même re-décodage :

| tranche | 1re tranche prête | total | accords identiques au morceau entier |
|---|---|---|---|
| 8 mesures (+4 s de contexte) | 0,62 s | 7,9 s | 80,0 % (1re tranche : 93,7 %) |
| 8 mesures (+10 s) | 0,75 s | 11,0 s | 84,5 % (1re tranche : 93,7 %) |
| 16 mesures (+10 s) | 1,28 s | 8,2 s | 88,1 % |
| 32 mesures (+10 s) | 2,31 s | 6,8 s | 91,8 % |
| morceau entier | — | 5,9 s | référence |

Deux choses expliquent l'écart, et aucune n'est un bug de recollage :

1. le CNN normalise en **InstanceNorm**, donc ses statistiques sont calculées
   sur la fenêtre qu'on lui donne — changer la longueur de la fenêtre change les
   postérieures **partout dans la tranche**, pas seulement aux bords ;
2. le LSTM est **bidirectionnel** : sa passe arrière voit tout le futur du
   morceau, qu'une tranche lui retire.

Le coût total monte aussi (le contexte est recalculé à chaque tranche) : le
temps par frame est quasi linéaire, 0,52 ms à 16 s contre 0,65 ms à 566 s.
Découper n'accélère donc pas le calcul — ça accélère seulement le **premier
affichage**.

Conclusion : découper est un **aperçu**, pas une réponse finale. Le bon usage
serait un troisième palier avant le chart brut, corrigé quand la passe complète
arrive — exactement le geste de l'entrée 23, un cran plus tôt.

## La tête d'un morceau suffit pour les battues

Pour un aperçu il faut aussi les battues, et elles coûtent maintenant plus cher
que les réseaux. Bonne nouvelle mesurée sur deux morceaux : les battues des 30 /
45 / 60 premières secondes sont **les mêmes que celles du morceau entier**
(écart max 0 à 20 ms, même phase de mesure), pour ~0,9 s au lieu de 4 à 9 s.

Un « chart de tête » (≈45 s d'audio : battues 0,9 s + CQT 0,4 s + réseaux 0,3 s
+ re-décodage 0,2 s) coûterait donc **≈2 s**. Non implémenté : c'est un palier
d'affichage de plus, et il réécrit des accords sous les yeux de Louis.

## Ce qui reste sur la table, par rendement

1. `yt-dlp -N 8` (fragments concurrents) — le téléchargement est le plus gros
   poste sur un morceau neuf, ~18 s sans rien de concurrent.
2. battues **en parallèle** des posteriors : `max(11,8 ; 7)` au lieu de la somme,
   soit ≈−7 s. Attention, `_InMusxDir` fait un `os.chdir` visible par tout le
   processus.
3. grille de latences : 8 décodages Viterbi pour un seul retenu (0 ms dans 3 cas
   sur 5 des logs, mais 240 et 280 ms existent — on ne peut pas la supprimer,
   seulement la paralléliser).
4. chart de tête (ci-dessus), si Louis accepte qu'un accord change après coup.
