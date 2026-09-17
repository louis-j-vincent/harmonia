# Pourquoi la règle du début se trompe — les six explications de Louis

Le 2026-09-17, après avoir tranché le début de 42 morceaux, Louis a écouté les
six que la règle (« la première note de basse, calée sur la grille ») rate, et
a dit pourquoi. Ce document est le cahier des charges de la règle suivante :
ses mots d'abord, ce qu'ils imposent ensuite.

## Ce qu'il a dit, morceau par morceau

**Urdlvw0SSEc** — « Parce qu'il y a une intro du clip avec une vraie musique,
c'est donc bien détecté. Une règle en plus : si on a un début de chanson puis
plus rien derrière (on le voit aux endroits où le bruit ambiant est plus grand
que le reste, ou alors il n'y a pas de musique), **plus** que la musique, le
bpm, toute l'identité musicale change après la pause (sinon ça peut juste être
une pause dans la musique), alors c'est une intro musicale. »

**Chain of Fools** — « Une intro à la basse un peu ambiguë, genuinely dur à
choper. Ici il faut choper le rythme de la batterie et la ligne de basse,
c'est ce qui nous permet de déduire le vrai début de la chanson, qui est
légèrement avant mon marquage je crois. »

**fd02pGJx0s0** — « Ici pareil, une intro de clip avec beaucoup de bruit,
basse batterie et accords sont faussement détectés. La vraie chanson commence
avec la basse, les accords, et la batterie qui prend un rythme **CONSTANT**
(il faut voir l'accord rythmique de la batterie avec le reste de la chanson,
sinon c'est pas la bonne batterie et on détecte un faux positif). »

**h_D3VFfhvs4** — « Même histoire que sur la chanson précédente, une vraie
musique au début du clip qui est bien détectée mais qui ne correspond pas au
vrai début de la chanson. »

**Billie Jean** — « La règle a raison aussi, le traqueur aussi en soi : il
commence au début de la batterie mais avant que la basse arrive, les deux
peuvent être considérés comme des premiers temps légitimes. **Mon marquage à
moi était faux.** »

**Be My Baby** — « Ici le souci c'est qu'on détecte un début de basse et
d'accords alors qu'il n'y en a pas. La vraie envolée est au temps après la
règle, quand il y a le vrai pattern accords et basse qui commence.
**L'intensité n'est parfois pas assez, c'est le début du pattern qui guide le
début de la chanson.** »

## Ce que ça impose

Les six explications disent la même chose sous six formes. La règle
d'aujourd'hui cherche un SEUIL D'INTENSITÉ — la première fois que la basse
sonne assez fort. Louis cherche autre chose :

> **Le morceau commence là où commence le pattern qui tient pour le reste de
> la chanson.**

D'où trois mécanismes, par ordre de portée :

1. **La récurrence.** Une intro de clip est de la vraie musique, détectée à
   juste titre — mais c'est une AUTRE musique : elle ne se rejoue jamais
   ensuite. Le premier instant dont la matière se retrouve plus loin dans le
   morceau est le début. Couvre Urdlvw0SSEc, fd02pGJx0s0, h_D3VFfhvs4, et
   Be My Baby (dont l'intro n'est pas le pattern).
2. **La régularité rythmique.** « La batterie qui prend un rythme CONSTANT »,
   « l'accord rythmique de la batterie avec le reste de la chanson ». Une
   batterie qui ne tient pas le tempo du reste est un faux positif. Couvre
   Chain of Fools et renforce fd02pGJx0s0.
3. **La coupure d'identité.** Un trou (bruit ambiant au-dessus du reste, ou
   pas de musique) SUIVI d'un changement de bpm et de matière : ce qui
   précède était une intro, pas le morceau. C'est la règle explicite de
   Louis sur Urdlvw0SSEc, et elle exige les deux conditions à la fois — le
   trou seul serait une simple pause dans la musique.

Billie Jean sort du lot : il a DEUX réponses justes (le début de la batterie
et l'entrée de la basse), et sa vérité terrain a été corrigée en conséquence
(`state/human/debuts.json`). Chain of Fools est marqué « à revoir », Louis
pensant que le vrai début est un peu avant sa marque.
