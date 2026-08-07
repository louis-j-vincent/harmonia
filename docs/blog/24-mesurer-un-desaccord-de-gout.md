# 24 — Mesurer un désaccord de goût

*2026-08-08, nuit.*

Louis part se coucher en laissant deux consignes. La première : construire et
**vérifier** la distance entre deux annotations de sections. La seconde : une
fois qu'elle tient, itérer seul, et quand ça ne marche pas, chercher pourquoi
jusqu'à prouver l'impasse.

## Le problème que la métrique doit résoudre

Il l'avait posé la veille : « des fois je vais différencier un A d'un B alors
que d'autres considéreront que c'est la même chose ; des fois on va merger un B
et un C ensemble et appeler ça B. Ça, c'est à l'appréciation. » Une métrique qui
compte les mesures d'accord punirait ce désaccord de goût exactement comme une
erreur, et on optimiserait alors contre son jugement plutôt que vers lui.

Sa spécification est précise : apparier les sections, vérifier que chacune
couvre le même espace que son équivalent, perdre des points sur le
désalignement, et **tenir compte des queues** — ces deux ou trois mesures de fin
qu'on peut légitimement rattacher à la section précédente ou ériger en section.

Le score final est un produit, `découpage × noms`. Le découpage apparie les
segments et compte les mesures que le partenaire ne couvre pas. Les noms
vérifient qu'une correspondance constante existe entre les lettres.

Le seul point vraiment délicat est de séparer une **queue** d'un **décalage**.
Les deux ont la même apparence : quelques mesures attribuées ailleurs, contre
une frontière. Ce qui les sépare est ce qu'il y a en face. Une queue tombe dans
un segment que personne n'a apparié — une section qu'un seul des deux
annotateurs a inventée. Un décalage tombe dans le voisin, qui a déjà son propre
partenaire : les deux grilles ont les mêmes sections, la frontière n'est
simplement pas au même endroit. Sans ce test, tout décalage d'une mesure
passerait pour une queue, et la métrique cesserait de mesurer l'alignement —
c'est-à-dire la chose pour laquelle elle existe.

Et le produit plutôt que la moyenne, pour une raison qu'un test épingle : mettre
tout le morceau sous une seule lettre rend les noms trivialement « constants »,
faute de contradicteur. Une moyenne remonterait ce néant à 0,50.

## Ce qu'elle a trouvé, dans l'ordre

Le premier résultat n'était pas un réglage mais un **bug d'écriture**. Nos
sections s'écrivaient `owner[mesure] = numéro du motif`, donc deux occurrences
adjacentes du même bloc étaient indistinguables d'une seule section deux fois
plus longue. Le A joué deux fois de suite ressortait comme un A de seize
mesures. La reprise était détectée, puis perdue au moment de la mettre en forme.
0,672 → 0,711, et Bein Green comme This Love tombent pile sur son annotation.

Le deuxième : nos lettres ne venaient pas du contenu. Elles venaient de **qui
avait réclamé la mesure**. Les ancres se posent de gauche à droite et ne prennent
que ce qui est libre, donc un passage déjà à moitié réclamé repartait sous une
lettre neuve même en rejouant exactement le motif d'avant. Chain of Fools, un
seul vamp du début à la fin, sortait avec cinq lettres pour la section unique
qu'il entend. Une passe de fusion par similarité harmonique : 0,743.

Le troisième vient d'une **preuve plutôt que d'une mesure**. La règle d'intro
reposait sur la phase de la première note dans sa mesure. Every Breath You Take
est à 0,29 et il place la section à la mesure suivante ; She Will Be Loved est à
0,31 et il garde celle-là. Deux centièmes, deux réponses opposées : aucun seuil
sur ce seul nombre ne peut les séparer, et le balayage le confirme — 13/15
partout, jamais mieux. La deuxième dimension était sous nos yeux dans ses
annotations : ses dix-sept intros font 0, 1, 1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 6, 8,
8, 8 mesures. Toujours un nombre pair, sauf une mesure isolée — la levée qu'il
nous demandait justement de savoir trouver. Nos deux erreurs proposaient 3 et 7.
15/17 départs justes, 0,764.

Le quatrième a commencé par une gêne. La docstring du module décrivait une
invariance à la transposition — un demi-ton par bloc, les deux voies, l'harmonie
pondérée plus fort quand une modulation est en jeu. `grep` ne trouve aucune
rotation dans le fichier. Cette version-là avait été décrite, jamais écrite.
Branchée là où elle a du sens, dans la fusion : 0,768, et l'intégralité du gain
est Sunny, le seul morceau du lot qui module.

## Les impasses, qui sont la moitié du travail

Cinq idées sont mortes cette nuit, et chacune laisse une contrainte pour la
suite.

**Les seuils ne sont pas le levier** : un oracle autorisé à choisir le meilleur
seuil par morceau plafonne à 0,744. Ce sur quoi nos annotations divergent est le
regroupement, pas la survie des blocs.

**La voix ne peut pas arbitrer une fusion.** Sur vingt-cinq fusions candidates,
dix-huit sont justes et sept fausses ; la similarité vocale vaut 0,12 à 0,69 sur
les justes et 0,19 à 0,50 sur les fausses. Les intervalles se recouvrent
entièrement — le balayage a débranché la voie tout seul. Et les sept fausses sont
toutes des morceaux où couplet et refrain partagent la grille d'accords, donc
exactement là où l'harmonie ne peut structurellement rien dire.

**Aucun critère interne ne choisit le départ.** Le mécanisme vaut mieux que le
constat : décaler le départ d'une mesure laisse L−1 des L termes de chaque
diagonale inchangés. Tout critère bâti sur les scores de blocs bouge donc dans
le bruit, alors que la couverture bouge de façon monotone — et c'est elle qui
emporte l'argmax. Le départ doit se décider hors de la machinerie des blocs.

**Le « rapport de reprise » est retiré.** Cette idée attendait depuis la veille
un deuxième morceau à intro chantée pour être jugeable. ABC en est un : la voix
entre mesure 0, sa section commence mesure 2. Verdict — The Walk 0,451, ABC
1,001, huit négatifs entre les deux. Le mécanisme condamne toute la famille
« l'intro est ce qui ne se répète pas » : l'intro d'ABC **est** le refrain, donc
elle se répète. L'énergie, essayée ensuite, échoue autrement : les intros ne sont
pas systématiquement plus légères, et The Walk est à 0,99 — aucun contraste.

## Où on en est

0,672 → 0,768 sur ses dix-sept morceaux validés, vérifié sur le chemin livré et
pas seulement sur le banc. Le découpage est à 0,90, les noms à 0,84.

Ce qui reste coûte cher et se sait : The Walk et ABC valent ensemble +0,027, et
aucune règle qui choisit entre `b` et `b+1` ne les atteindra. Il écrit huit
sections de douze mesures là où nous n'en écrivons aucune — les coller après coup
ne rend rien, il faudra les trouver pendant la recherche.
