# 29 — Un écran, une chose à faire

*2026-08-19*

Cinq tâches d'un handoff de nettoyage (`handoff_cleanup/CLEANUP_SPEC.md`), une
par commit, chacune vérifiée à l'écran avant d'être livrée. La règle qui les
gouverne toutes : **Read lit, Annotate corrige, Practise joue** — rien d'un mode
n'apparaît dans un autre, et tout outil qui n'est pas l'action principale de
l'écran vit derrière un bouton unique.

## Ce qui a changé

| | Avant | Après |
| --- | --- | --- |
| La forme | rangée de chips 44 px, deux lignes dès ~10 passages | une bande de 22 px, une ligne quel que soit le nombre |
| Annotate | légende dégradée + deux blocs titrés + 4 boutons | un « ··· » et une ligne d'italique |
| Le doute | un « ? » qui sortait de la case | un point de 5 px sous le glyphe |
| Practise | 7 blocs empilés dans un stage qui défilait | 3 zones, plus aucun défilement, et un clavier |
| Les sections | (rien de plein écran) | un écran à soi, frontières sous le doigt |

## Les deux idées qui portent tout

**Grouper, puis proportionner.** La bande tient les 17 passages du pire cas de
Louis (`A×3 B A×3 B A×3 B bridge B A×3`) sur une ligne parce qu'elle ne dessine
pas 17 boîtes : les passages consécutifs d'une même section n'en font qu'une,
avec N−1 traits fins à l'intérieur. Il n'y a plus de `×N` à écrire — la
répétition se voit. Et la largeur d'un groupe est sa part de la **durée**, donc
un A de 8 mesures et un tag de 2 ne sont plus dessinés pareil.

**Le plancher tactile ne grossit jamais le dessin.** 22 px de bande, 44 px de
doigt : padding + marges négatives sur une couche de tap invisible, posée *hors*
du dessin clippé — `overflow:hidden` mange les zones de tap qui dépassent.

## Trois pièges payés en chemin

*Un plancher en `min-width` avec une base flex à 0 fait déborder la rangée.*
Quand la répartition proportionnelle viole les `min-width`, le navigateur ne
peut plus rien rendre : la bande de T3 sortait de l'écran au neuvième groupe.
Le plancher doit passer par le **flex-basis** — on paie les planchers d'abord,
on répartit le reste.

*Une zone de tap invisible fait défiler un écran qui ne défile pas.* La bande de
16 px en bas de Practise porte 44 px de tap : les 14 px qui dépassent sous elle
comptaient dans le `scrollHeight`. Un écran « sans défilement » se mettait à
défiler de 4 px, sans rien montrer de plus.

*Deux fonctions qui écrivent la même propriété se battent.* `paintFormChip`
écrivait `background`/`borderColor` sur le même élément que `paintFormRail`.
Elle DÉCIDE maintenant (elle rend un index), elle ne dessine plus.

## Ce que le mock ne pouvait pas voir

Le harnais de test (`handoff_cleanup/mock_live.html`, l'app + un prélude qui
mocke `/api/*`) a validé les vingt-et-un critères d'acceptation. Deux défauts ne
sont apparus qu'en ouvrant l'app pour de vrai sur Bora Bora :

* la bande écrivait `a` en italique bordeaux à côté d'un `A` droit — `A'` est
  une **lettre**, pas une section nommée ;
* le morceau commence sur un N.C., et la carte AU PIANO s'ouvrait sur un
  rectangle blanc : le clavier n'était bâti qu'au premier accord.

C'est la règle #5 du projet en miniature : un harnais valide une contrainte, il
ne remplace pas l'ouverture du vrai rendu.

## Ce qui reste ouvert

L'éditeur de sections plein écran (T3) est écrit et vérifié mais **n'a aucune
entrée dans l'app**, sur arbitrage de Louis : l'outil au doigt du chart occupe
encore cette porte, et deux éditeurs pour une même question, c'est exactement ce
que ce nettoyage combat. On y arrive par `?open=<file>&edit=sections`, le temps
qu'il tranche à l'œil.
