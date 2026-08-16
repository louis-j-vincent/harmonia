# 26 — L'algo du retour

*2026-08-16. Louis, en fin de séance : « alors c'est vraiment top, log celui-là
comme très satisfaisant !!!! »*

Un état à ne pas perdre. Ce billet fige **pourquoi** l'algo est dans cette
forme-là, parce que trois des huit règles ont été posées, retirées, puis
reposées différemment dans la même séance, et que rien dans le code seul ne dit
lequel des trois essais était le bon.

## L'idée

On ne cherche pas une frontière — où ça change — on cherche un **retour** — où
ça recommence. En partant de la première mesure, on avance jusqu'à trouver une
mesure qui lui ressemble fortement. Le morceau de musique compris entre les
deux est un « mot ». Si ce mot fait plus de 6 mesures et que le mot suivant est
le même, c'est une section, et on va chercher toutes ses répétitions. Puis on
retire ces répétitions, on **recoud** ce qui reste, et on recommence.

Le code : `harmonia_min/retour.py`. La démo pas à pas sur dix morceaux annotés
à la main : `docs/plots/retour_10morceaux.html`.

## Les huit décisions, et ce que chacune a coûté

| # | La règle | Ce qu'elle répare |
|---|---|---|
| 1 | Seuil de « ressemblance forte » par **Otsu**, pas par quantile | Le quantile 0,90 vaut 0,9996 sur Let It Be : plus aucun retour ne passe, zéro section. Les cosinus de vecteurs positifs s'écrasent contre 1. |
| 2 | Si le premier retour échoue, essayer le suivant | La règle littérale s'arrête au premier et abandonne le départ. |
| 3 | Si aucun retour ne marche, avancer d'une mesure | Les mesures enjambées restent sans section — intros, ponts, outros. |
| 4 | **Exemption : une mesure, la dernière, sur une suite ≥ 8** | Voir plus bas : trois versions. |
| 5 | Le seuil d'**occurrence** n'est pas le seuil de retour | Reconnaître est plus exigeant que repérer. This Love 50–53 entrait dans le A à 0,726 pour un seuil de 0,672, en plein milieu du pont. |
| 6 | Entre plusieurs retours valides, prendre le **multiple de 4** | Stand By Me sortait un mot de 7 mesures : une longueur qui n'existe pas dans ce morceau. |
| 7 | La **boucle interne** devient le modèle, pas le mot | Rattrape les morceaux dont l'intro fait déjà tourner un bout du A. |
| 8 | **Une section est une boucle jouée N fois** — boucle ≥ 4 mesures, N ≥ 2 | Dans une musique à quatre accords, quatre mesures se ressemblent partout : un seul tour est une coïncidence, pas un retour. |

## L'aller-retour qui a coûté le plus : l'exemption

Quatre versions en une séance. C'est celle qui mérite d'être écrite.

- **v1** — queue = 0 sur une boucle, 2 sur un mot. Sur une boucle de quatre,
  exempter deux mesures ne comparerait que la moitié du motif.
- **v2** — Louis : « exempte une seule mesure sur une boucle de 4, ok go ». Le
  gain visé était réel : les secondes moitiés du B de This Love passaient de
  0,86 à 0,98.
- **v3** — retour à zéro, sur son constat suivant : « sur This Love tu m'as
  découpé les 8 mesures 50 à 57 alors que le A ne prend que 50 à 53 ». Ces
  mesures étaient exactement ce que la v2 avait fait entrer. Jeter une mesure
  sur quatre, c'est jeter 25 % de la preuve, et le A se met à mordre sur le pont.
- **v4, celle qui tient** — « on ne va faire qu'une règle : lors d'une suite
  consécutive d'au moins 8 barres, exemption sur la dernière barre, et c'est
  tout ».

**Ce qui manquait aux trois premières n'était pas la valeur, c'était le
NIVEAU.** L'exemption ne porte pas sur le bloc, elle porte sur la **suite**. Sur
This Love, le B est une suite de 8 mesures faite de deux boucles de 4, et la
mesure qui change est la dernière de la *suite* (m.23, 43, 63, 71, 79) — jamais
celle du premier bloc. Exempter bloc par bloc pardonnait deux mesures sur huit
au lieu d'une. Un réglage juste appliqué au mauvais grain se lit exactement
comme un réglage faux.

Corollaire gardé : une mesure exemptée n'est pas une mesure oubliée. Chaque
suite porte `exemptee` et `variante` — les mesures qui divergent du modèle,
exemptées ou non — parce qu'au **repliement du chart il faudra les écrire** au
lieu de recopier le modèle par-dessus (demande explicite de Louis).

## La définition qui a tout mis en place

Louis, à la fin : « il faut réussir à définir une section comme une mini-boucle
interne jouée un certain nombre de fois. Une fois qu'on a ça, pour vérifier les
matchs de notre section, on a juste à voir combien de fois la boucle est
rejouée quelque part. »

C'est le modèle de données du module, et il change la question posée. Une
section n'est pas un gabarit de longueur fixe qu'on va chercher ailleurs ; c'est
**une boucle et un nombre de tours**. Chercher ses occurrences, c'est compter
les tours. Deux occurrences de la même section peuvent donc être de longueurs
différentes — 2 tours ici, 6 tours là — et c'est voulu : on écrit chaque section
à la longueur qu'elle joue vraiment.

Le détail qui débloque tout : **la boucle se reconnaît modulo sa dernière
mesure**. Sans ça la boucle `abac` n'est jamais trouvée — sur Stand By Me le mot
de 8 boucle à 4 avec 0,72 · 0,79 · 0,86 · **0,55**, moyenne 0,732 contre un
seuil de 0,732. Raté d'un cheveu, à cause de la seule mesure qui a le *droit* de
changer, celle qui cadence.

This Love se lit alors dans ses termes à lui : `A` = boucle de 4 mesures jouée
4 puis 3 fois ; `B` = boucle de 4 mesures jouée 2, 2 puis 6 fois.

## Où ça en est, sur This Love

L'algo et l'annotation à la main de Louis, mesure pour mesure :

| | l'algo | Louis |
|---|---|---|
| B | 16–23, 36–43, 56–63, 64–71, 72–79 | identiques |
| reste | 44–55 | sa queue 44–47 + son pont 48–55 |

Le A absorbe encore l'intro (il commence mesure 0 au lieu de 8) : rien n'ancre
la phase, et `chart["bar1"]` n'est pas lu.

## Le substrat : accords, ou accords × basse ?

Louis : « sur Let It Be on chope mal les différences harmoniques, il y a un A et
un B et on n'a que le A. Je me demande si utiliser la matrice SSM de la basse ne
pourrait pas aider. » Puis, aussitôt après une réponse trop agrégée : **« montre
moi les distances et laisse-MOI juger »**.

`docs/plots/retour_basse_vs_accords.html` met les trois matrices côte à côte
avec ses frontières dessus, et la distance entre chaque paire de ses sections,
terme par terme. La démo tourne sur `accords*basse` avec l'ancien substrat en
regard. **Le défaut du module reste `accords`** tant qu'il n'a pas tranché — une
SSM de basse seule serait ce qu'il a interdit le 2026-07-30.

Ce que ça donne, morceau par morceau : Stand By Me passe de 1 à 3 sections (la
basse y sépare enfin ce que les accords voyaient uniforme) ; Sunny, Don't Know
Why et Goodbye Yellow Brick Road se dégradent nettement ; **Let It Be, la
question de départ, ne bouge pas**.

## Ce que l'algo ne résout toujours pas

1. **Rien n'ancre la phase.** Le A commence mesure 0 là où Louis démarre
   mesure 8. `chart["bar1"]` existe pour ça et n'est pas lu.
2. **Deux sections qui partagent la grille ne se séparent pas.** Let It Be :
   couplet et refrain sur les mêmes accords. Ni la basse ni les accords n'y
   suffisent — il faudra le chant ou l'énergie.
3. **Rien ne nomme.** A, B, C dans l'ordre de découverte, pas
   couplet/refrain.
4. **Un morceau à un seul accord est hors d'atteinte.** Chain of Fools.
