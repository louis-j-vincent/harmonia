# 31 — Cinq lectures, et la basse qu'on entend

*2026-09-16*

Louis, en une phrase : « je veux qu'il apparaisse les autres suggestions que
juste le top 2, plus les suggestions sur la ligne de basse en faisant des
petits cercles autour des lettres du cercle pour montrer les basses qui sont
détectées, avec pareil toujours la taille du cercle proportionnelle à la
proba d'être détectée, top 3 basses, top 5 accords si relevant. »

Deux demandes, et un mot qui porte tout le travail : **si relevant**.

## Le mot qui décide

« Top 5 si relevant » n'est pas « top 5 ». C'est un plancher. Sans plancher,
un accord que musx tient à 91 % afficherait quatre candidats à 1 % autour de
lui — quatre façons de se tromper, présentées comme des options. La question
n'était donc pas *combien en montrer*, mais *à partir de quand une lecture
mérite d'être montrée*.

La réponse tient en une ligne, et elle vaut pour les accords comme pour la
basse : **un candidat s'affiche s'il bat le pur hasard sur son propre jeu de
candidats.** musx étale sa masse sur 60 cases (12 racines × 5 familles) ;
l'uniforme y vaut 1,67 %, le plancher est 2 %. La basse étale la sienne sur
12 classes de hauteur ; l'uniforme vaut 8,3 %, le plancher est 12,5 %.

C'est une loi, pas un réglage : elle se dérive du nombre de candidats, elle ne
se trouve pas par tâtonnement.

## Vérifier le seuil avant d'écrire le rendu

Avant la première ligne d'affichage, on a mesuré l'échelle des postérieures
sur les deux morceaux les plus ambigus du banc — Yesterday (19 accords) et
Lost Without U (74).

| rang | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| médiane (Yesterday) | 0,729 | 0,085 | 0,043 | 0,031 | 0,025 |
| médiane (Lost Without U) | 0,657 | 0,143 | 0,051 | 0,029 | 0,017 |

Le 5e rang vit entre 1,7 % et 2,5 %. Le plancher tombe donc exactement là où
le 5e candidat cesse de dire quelque chose — ce n'est pas une coïncidence
heureuse, c'est ce que « battre l'uniforme » veut dire quand la masse est
déjà largement prise par les deux premiers.

Effet : **3,7 candidats par accord** en moyenne, contre 3 figés avant ; 12
accords sur 19 et 47 sur 74 en montrent au moins 4. Et le contrôle qui
compte : à 5 % de plancher on retomberait à 2,4–2,9 par accord, c'est-à-dire
l'ancien top-3. Un seuil trop haut n'aurait rien changé du tout, en donnant
l'impression d'avoir livré.

## La basse ne coûtait rien

`bass_pc_onset` existait depuis la veille : la classe de hauteur de la basse,
lue sur les **150 premières millisecondes** de l'accord, parce que la
fondamentale domine à l'attaque et que sa quinte harmonique s'accumule
ensuite (6/11 → 11/11 sur « Ready », le 2026-09-15). Elle rend déjà une
distribution sur 12 notes qui somme à 1 : « top 3 basses avec leur proba »
n'était pas un modèle à construire, c'était trois lignes de tri.

Mieux : `pipeline.py` lit déjà le `bothchroma` à cette étape, pour la
couleur harmonique. La basse est donc une lecture de 150 ms dans une matrice
déjà en mémoire. Aucune extraction, aucun modèle, aucun entraînement.

## Où poser les cercles

Louis a dit « autour des lettres du cercle », et c'est le bon endroit pour une
raison qui n'est pas graphique : **une lettre de la roue est une classe de
hauteur**, pas un accord. C'est exactement ce qu'une basse nomme. L'anneau ne
force donc aucune analogie — il annote la roue dans son propre vocabulaire, et
il ne prend pas la place des orbes d'accord, qui vivent à l'intérieur du
cercle.

Deux détails qui auraient été des bugs silencieux :

**La couleur.** Les orbes se colorent par position sur le cycle des quintes.
Si les anneaux de basse partageaient cette teinte, un anneau se lirait comme
une suggestion d'accord. Ils sont bleus — la basse est une autre *voix*, pas
un autre accord.

**L'ordre de peinture.** En SVG, le dernier peint gagne. Un disque posé
par-dessus la lettre rendrait illisible la note qu'on vient précisément lire.
L'anneau se pose donc *avant* le texte.

Et la taille suit la convention déjà posée le 2026-08-08 pour les orbes :
rayon ∝ √probabilité, donc **aire** ∝ probabilité. C'est l'encodage honnête,
et surtout c'est le même partout dans l'écran : « plus gros = plus sûr » veut
dire la même chose à deux endroits.

## Ce qu'on n'a pas fait, exprès

Les anneaux ne sont **pas tapables**, et le chart ne gagne aucun slash.

C'est délibéré, et c'est ce qui permettait de livrer aujourd'hui. Le tracker
exige un banc corpus avant de brancher la basse en direct — mais cette
exigence porte sur la **décision** d'écrire `C/G`, pas sur le fait de montrer
à Louis ce que la mesure entend. Montrer n'engage personne ; écrire change le
chart. Les deux planchers de basse qui cohabitent maintenant disent la même
distinction : `bass_rules.FLOOR` vaut 30 % et décide, `BASS_SUG_FLOOR` vaut
12,5 % et affiche. Un test les tient séparés — les confondre serait
exactement l'erreur de calibration silencieuse qui a déjà coûté quatre fois à
ce projet.

Reste ouvert : le plancher est calibré sur la *forme* de l'échelle, pas
arbitré à l'oreille, et sur deux morceaux. La basse mobile — une note par
temps — reste résumée par une seule lecture à l'attaque. Et rendre un anneau
tapable, si Louis le veut, veut dire rouvrir la question de l'écriture des
slashs, avec le banc qui va avec.
