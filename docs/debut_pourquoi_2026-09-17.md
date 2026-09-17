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

## Quatre tentatives, quatre échecs — et ce qu'ils prouvent

Les trois mécanismes ci-dessus ont été implémentés et mesurés le jour même,
sur les 40 morceaux dont la grille est exploitable. Référence : la règle de la
basse seule, **35/40**.

| tentative | résultat | mécanisme de l'échec |
|---|---|---|
| la récurrence harmonique SEULE (première mesure dont la matière se rejoue) | 27/40 | — |
| la récurrence harmonique en VETO sur la basse | 35/40, **0 gagné 0 perdu** à tous les réglages | la ressemblance chord-tone d'une intro de clip avec le corps du morceau est déjà au-dessus de 0,90 : ils partagent la tonalité et le vocabulaire d'accords. La matrice n'a AUCUN rythme dedans, or les six explications de Louis parlent toutes de rythme. |
| la récurrence RYTHMIQUE en veto (profil d'attaques aiguës, 16 cases par mesure) | au mieux **+1 gagné, −10 perdus** | un profil de 16 cases normalisé ne distingue pas deux batteries au même tempo, et distingue à tort le même motif joué avec un autre mixage. Le seuil qui attrape le seul vrai cas en casse dix autres. |
| le changement de TEMPO après la pause, sur `beatTimes` | 0/2 des intros mesurables | **Beat This! impose un tempo unique sur tout le fichier.** La rupture que Louis décrit est déjà effacée par le traqueur avant que je puisse la lire. Urdlvw0SSEc et fd02pGJx0s0 affichent 0,0 % d'écart. |
| le changement de tempo LOCAL, mesuré sur l'audio (autocorrélation glissante) | aucun seuil ne sépare | 17–22 % d'écart sur les trois intros de clip, mais 33 % sur The Walk et 21 % sur She Will Be Loved, qui sont justes. Dans une intro clairsemée (quelques notes, pas de batterie) l'autocorrélation n'a pas de période franche et rend un tempo quasi arbitraire. |

### Ce que ça prouve vraiment

Le problème n'est PAS le choix de l'indice. Chaque règle que Louis décrit est
une **conjonction** de deux ou trois conditions seuillées (un trou, ET un
changement de bpm, ET un changement de matière). Or la bibliothèque ne compte
que **trois à six exemples** du mode d'échec. Un seuil ajusté sur trois
exemples positifs sépare par chance, et les tableaux ci-dessus le montrent :
dès qu'un réglage attrape le vrai cas, il en casse dix autres.

**La contrainte qui bloque est le nombre d'exemples, pas la finesse du
détecteur.** Deux suites possibles, dans cet ordre de coût :

1. **Brancher la règle de la basse telle quelle** (35/40), les marques à la
   main de Louis restant souveraines sur les ratés. Le gain est réel et
   disponible tout de suite ; il passe par un rapport d'or qu'il arbitre.
2. **Récolter plus d'intros de clip** avant de retenter. Louis sait lesquels
   de ses fichiers viennent d'un « official music video » ; une dizaine
   d'exemples rendrait un seuil de conjonction mesurable au lieu d'être
   ajusté au hasard.

Une piste non testée, faute de temps et parce qu'elle mérite ses propres
exemples : une intro de clip contient souvent de la **parole**, et la courbe
de bruit ambiant la montre franchement sur Sam Smith (platitude spectrale
haute pendant 40 s, effondrée dès l'entrée du groupe). Détecter la parole est
un problème mieux posé que « détecter une autre musique ».


## Ce qui a fini par marcher : le trou, à pleine résolution

Les quatre échecs ci-dessus avaient un point commun que je n'avais pas vu :
ils cherchaient tous à comparer l'intro au CORPS du morceau (par la
ressemblance, par le tempo). La règle de Louis ne compare rien — elle regarde
**ce qu'il y a ENTRE les deux** :

> « Si on a un début de chanson puis plus rien derrière [...] alors c'est une
> intro musicale. »

`harmonia.debut.dernier_trou` : la tête de basse de musx reste sous 0,15
pendant au moins 3 secondes. Ce qui joue avant le DERNIER trou n'est pas le
morceau. La première note de basse est alors cherchée à partir de là.

| règle | bonne mesure |
|---|---|
| le traqueur seul | 28/42 |
| la 1re note de basse | 35/40 |
| **la 1re note de basse APRÈS le dernier trou** | **37/40, zéro cas cassé** |

Les deux gagnés sont exactement ceux que Louis avait expliqués : Urdlvw0SSEc
(22,77 s, le trou finit à 21,9) et fd02pGJx0s0 (9,98 s, trou fini à 9,9). Le
résultat tient sur tout un plateau de réglages — seuil 0,15 à 0,20, durée 2 à
5 s donnent tous 37/40 avec zéro perdu — ce qui est le signe d'un vrai effet
et non d'un seuil ajusté à trois exemples.

Les trois ratés restants : Be My Baby ouvre sur un break de batterie sans
harmonie, donc sans trou à trouver ; Smooth Criminal trouve bien son trou
(fini à 66,6 s) mais tombe une mesure trop loin, sur une grille elle-même
trouée ; Chain of Fools rend 3,66 s là où Louis a marqué 5,87 — mais il dit
lui-même que le vrai début est « légèrement avant » sa marque, donc c'est
peut-être la marque qui a tort.

### L'agent « détecter l'intro » : testé, et il a trouvé mieux qu'une réponse

Louis, 2026-09-17 : « il faudrait un agent spécial "détecter l'intro" qui soit
lancé et dise s'il y a une intro ou pas ». La prémisse a été testée avant de
construire quoi que ce soit (règle 2 du CLAUDE.md) : un agent a reçu les
quatre courbes de huit morceaux, résumées en 30 tranches de 0 à 9, sans les
réponses.

Il a bien répondu — et il a surtout **invalidé mon protocole** : le résumé en
tranches de 0 à 9 avait écrasé à zéro la basse faible-mais-non-nulle des
intros de clip. Un agent qui réussit là-dessus réussit sur un indice qui
n'existe pas dans l'audio. Il a aussi noté que ma courbe d'énergie n'avait pas
le même nombre de tranches que les autres, donc n'était pas alignée, et que la
polarité du bruit ambiant s'inverse d'un morceau à l'autre (haut pendant le
préambule de The Walk, nul pendant les 30 s de dialogue de Hot N Cold) — donc
« bruit ambiant haut = intro » casserait sur deux morceaux sur huit.

Enfin, il a dit franchement qu'un calcul déterministe ferait presque aussi
bien que son raisonnement sur ce dossier. C'est ce qui a mené au test du trou
à pleine résolution ci-dessus, qui fait 37/40. **L'agent n'a donc pas été
construit : il a servi à trouver la règle, ce qui est mieux.**

## La règle du préfixe : mesurée, rejetée — et le vrai coupable trouvé

Louis, 2026-09-17, sur Don't Want My Love : « pourquoi le C c'est `bbbaa` et
après le D c'est `bbbaaaa` ? Ce D c'est clairement un C avec un `aa` derrière
qui est une outro ». D'où la règle candidate : **un bloc qui commence par un
autre bloc est ce bloc, plus une queue.**

Mesurée sur ses 18 découpages annotés (806 mesures), protocole du 2026-09-16
— on simule son geste, il marque la première occurrence de chaque lettre puis
valide, et on mesure l'accord lettre-par-mesure sur ce qu'il n'a PAS marqué.
Rapport complet : `docs/regle_prefixe_2026-09-17.md`.

| règle | accord lettre/mesure | paires groupées | gagne | perd |
|---|---|---|---|---|
| témoin (exact + prime) | 56,5 % | 75,1 % | — | — |
| A — préfixe entier | 54,0 % | 73,7 % | **0** | **3** |
| B — plus long préfixe commun | 53,7 % | 74,6 % | **0** | **4** |
| G — correctif de `grouper_restes` | 55,3 % | **76,2 %** | 2 | 7 |

**Zéro gagnant pour A et B, à tous les réglages.** Perdants : Yesterday
55 → 36 %, Let It Be 91 → 74 %, Be My Baby 29 → 17 %. Avec un garde-fou
honnête (partie partagée ≥ 4 bi-mesures), A et B deviennent strictement
identiques au témoin : le témoin, en plus compliqué.

### Pourquoi, en trois faits

1. **Sur son propre morceau la règle ne peut pas se déclencher.** Son C est
   `abbb` et son D `bbbaaaa` : le plus long préfixe commun vaut **zéro**. La
   matière partagée — les trois `b` — est bien dans les deux, mais **décalée
   d'une bi-mesure**. Sur tout le corpus : partie partagée EN TÊTE
   1,61 bi-mesure, partie partagée N'IMPORTE OÙ 2,38. La bonne primitive
   n'est pas « commence par », c'est « contient le même motif, décalé ».
2. **Le mot pauvre piège la règle.** Les deux morceaux les plus cassés sont
   les deux plus pauvres parmi ceux où elle tire : Let It Be (71 % d'une
   seule lettre, −17 points) et Yesterday (70 %, −19 points). Les fusions
   plausibles n'apparaissent que sur les mots riches (Sunny, 10 lettres).
3. **`nommer` avait déjà raison ; c'est `grouper_restes` qui casse.** Tracé
   sur Don't Want My Love avec son trait A :

        après `nommer`            après `grouper_restes`
        23-26  C  « bb »          23-28  D  « bbb »
        27-28  D  « b »
        37-42  A  « aaa »         37-48  C  « aaaccc »
        43-46  E  « cc »
        47-48  F  « c »

   Les fragments sont déclarés « queue » parce qu'ils sont plus courts que la
   cible, puis recollés — et le recollage **invente une lettre** sans rapport
   avec celles des fragments. Le bloc dont Louis se plaint n'est pas un
   défaut de nommage : c'est un recollage de morceaux déjà nommés.

### La piste qui reste

La variante G — « un reste dont le type se répète ailleurs n'est pas du
reste, il garde sa lettre » — est la seule qui bouge la STRUCTURE (76,2 %
contre 75,1 %) et elle débloque deux des trois morceaux que ce document note
comme coincés : **The Walk 0 → 43 %**, **Easy On Me 28 → 55 %**. Trop brutale
telle quelle : 256 blocs contre 198, elle fragmente. Il lui faut un
adoucissement — recoller quand même, mais **garder la lettre du plus long
fragment** au lieu d'en inventer une neuve.
