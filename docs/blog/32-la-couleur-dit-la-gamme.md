# 32 — La couleur dit la gamme, la taille dit la proba

*2026-09-17*

Trois demandes de Louis en une soirée, et chacune a retiré une information de
l'écran plutôt que d'en ajouter une. C'est le fil de la séance.

## « Je peux pas avoir 30 % plus grand que 20 % »

Le compas encode la probabilité par l'AIRE de l'orbe depuis le 2026-08-08.
Sauf que la formule portait un plancher : `rayon = 22 + √p × (rMax - 22)`, où
22 px est le seuil TACTILE — la moitié des 44 px qu'un doigt réclame.

Mesuré sur *Lost Without U*, où le modèle donne 83 / 5 / 1 / 0,7 % :

| proba | rayon avant | rayon après |
|---|---|---|
| 83 % | 40 px | 38,1 px |
| 5 %  | 33 px | 9,4 px |
| 1 %  | 26 px | 5,1 px |
| 0,7 %| 25 px | 4,0 px |

Un rapport de **118** en probabilité devenait **1,6** en rayon. Le plancher
avait mangé toute l'échelle : l'encodage était honnête dans son intention et
faux dans ses valeurs.

La sortie n'était pas de retoucher la formule mais de **séparer les deux
rôles**. Le disque dit la probabilité, exactement : `rMax × √p`, sans
plancher. Le plancher tactile devient une zone de contact **invisible** de
44 px posée par-dessus — ce que les anneaux de basse faisaient déjà. Un orbe à
0,7 % est alors un point de 4 px, et il reste touchable. Sous 19 px, son
étiquette sort et se pose dans le prolongement du rayon.

Vérification : √(83/5) = 4,07 et 38,1/9,4 = 4,05. La proportion est mesurée,
pas annoncée.

## « Une couleur par gamme »

La teinte venait de la fondamentale du candidat, au cercle des quintes. Cinq
candidats faisaient cinq couleurs — et ces cinq couleurs ne disaient rien que
les cinq glyphes ne disaient déjà.

Louis : « par défaut de la couleur de la gamme dans laquelle cette partie de
la chanson est, sauf si l'accord lui même est en dehors de cette gamme auquel
cas il est de la couleur de la gamme dans laquelle il projette via les notes
qui sortent de la gamme ».

La règle travaille sur *les notes qui sortent*, donc l'objet coloré doit être
un **jeu de sept notes** — pas une tonique, pas un mode. Conséquence assumée :
si♭ majeur et sol mineur sont la même gamme et portent la même couleur. C'est
exact, ce sont les mêmes sept notes, et c'est ce qui empêche un morceau en
mineur de clignoter à chaque emprunt à sa relative.

Une première version donnait à chaque accord SA gamme (la table des
chord-scales : ionien sur le I, dorien sur le ii, altéré sur la dominante du
mineur…). Résultat : sept couleurs sur huit accords. Joli, et muet. La version
de Louis est meilleure parce qu'elle est **presque constante** : la roue
repose sur la couleur de la région, les candidats dans la tonalité s'y
fondent, et seul celui qui sort apparaît — dans la couleur de sa destination.

Sur *Autumn Leaves* (22 accords sur 26 dans la gamme) le `D7` sort par son
fa♯ et projette à 3 quintes ; descendre d'un cran montre alors que `D7` et
`D6` sortent vers sol, et `Dmaj7` une quinte plus loin encore. **Le compas dit
maintenant à quelle distance chaque extension vous emmène hors du ton.**
C'est la seule chose que la couleur-fondamentale ne pouvait pas dire.

La teinte reste la place sur le cercle des quintes (`kit.js::rootHue`), donc
l'écart de teinte entre un orbe et son fond EST sa distance harmonique. Et
c'est déjà la convention de la lentille « key » du chart en mode analyse : le
compas et le chart disent la même chose de la même couleur.

## L'arbitrage qui reste

Les douze gammes sont les douze jeux **diatoniques**, et la mineure harmonique
n'en est pas un. Le `D7` de sol mineur — le V le plus ordinaire du monde — est
donc lu comme une sortie de 3 quintes vers sol, et pas comme la sensible de sa
propre tonalité. Exact quant aux notes, discutable quant à la fonction.
Ajouter les mineures harmoniques rendrait la cadence muette : on ne peut pas
avoir les deux. La démo qui a servi à trancher est `docs/plots/gammes.html`.

## Deux défauts trouvés en mettant en prod

Le thème sombre a montré ce que le clair cachait. Le fond de la roue prenait
le `keyTint` du compas (clarté 90 %) : composité sur la carte sombre, il
rendait un gris boueux qui avait perdu sa teinte — donc « le fond, c'est la
gamme » ne se lisait plus du tout. Il prend maintenant `fill` (clarté 72 %) à
faible opacité, qui garde sa teinte des deux côtés.

Et le pointillé qui dit « sous le seuil de suggestion » se rendait en ÉTOILE
sur un disque de 4 px : l'information secondaire détruisait la principale, la
taille. Sous 12 px de rayon, trait plein.

Même famille que le défaut trouvé la veille dans le compas de l'app :
`petalFill` et `keyTint` n'ont pas de branche sombre — ils sont clairs par
construction — mais le texte posé dessus prenait `T.ink`, qui bascule en
crème. Crème sur pâle, contraste 1,1:1. `kit.js` exporte désormais
`INK_ON_PETAL` / `FAINT_ON_PETAL`, figés sur l'encre du thème clair.

## Ce qui n'a pas été republié, et pourquoi

Le rapport d'or dit **0 mesure changée sur 46 morceaux** pour cette séance :
le champ `casc` s'ajoute, la musique ne bouge pas. Mais un morceau (*Bora
Bora*) diffère de 33 mesures à cause du travail d'une AUTRE session sur le
même arbre. Republier la bibliothèque aurait donc fait entrer dans les charts
un découpage de sections que Louis n'a pas arbitré.

Le champ a donc été posé par un geste minimal et vérifiable : ouvrir chaque
chart, AJOUTER le champ, réécrire — avec une empreinte de tout ce qui fait la
musique (accords, mesures, sections, grille) comparée avant et après, chart
par chart. 72 charts, 5196 accords, aucune empreinte modifiée.
