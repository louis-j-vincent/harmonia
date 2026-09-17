# La règle du préfixe — mesure seule (2026-09-17)

**Verdict : non, ne la branche pas — mais la mesure a trouvé le vrai coupable,
et ce n'est pas `nommer`, c'est `grouper_restes` (§3 et §6).** Ni telle que briefée, ni dans sa
variante « plus long préfixe commun ». Et la raison n'est pas le chiffre :
c'est que **sur Don't Want My Love — le morceau qui a fait naître l'idée — la
règle ne peut pas se déclencher**, pour une raison de phase de grille.

Rien n'a été modifié dans `harmonia/`. Tout est dans
`scratchpad/regle_prefixe.py` (la copie de `nommer` + les variantes) et
`scratchpad/mesure.py` (le protocole), résultats bruts dans
`scratchpad/resultats.json`.

---

## 1. Le tableau

Protocole : celui du 2026-09-16, reproduit fidèlement (`sections_completion_page`
+ le corps de `sections_inferer`). On simule ton geste — tu marques la
**première occurrence de chaque lettre**, puis tu valides — et on mesure
l'accord **lettre-par-mesure sur ce que tu n'as PAS marqué**. 18 morceaux
(annotation validée + chart + audio en cache), 806 mesures évaluées.
Le témoin rend 56,5 %, ce qui recale bien sur les 55 % de `known_issues.md`.

| règle | accord lettre/mesure | paires groupées comme toi | blocs produits | gagne | perd |
|---|---|---|---|---|---|
| **témoin** (exact + prime) | **56,5 %** | **75,1 %** | 198 | — | — |
| **A** — préfixe entier, dans `nommer` | 54,0 % | 73,7 % | 193 | **0** | **3** |
| **B** — plus long préfixe commun, dans `nommer` | 53,7 % | 74,6 % | 190 | **0** | **4** |
| **C** — préfixe entier, *au bon endroit* (voir §3) | 57,3 % | 75,1 % | 203 | 2 | 2 |
| **D** — plus long préfixe commun, au bon endroit | 57,3 % | 75,1 % | 203 | 2 | 2 |
| **E** — C + garde anti-mot-pauvre (§5) | 56,5 % | 75,0 % | 200 | 1 | 1 |

« paires groupées comme toi » = pour deux mesures quelconques, la machine
dit-elle « même section » quand tu le dis ? C'est le mètre insensible au nom
et au décalage d'une mesure (celui de `known_issues.md`). Il ne bouge pas :
75,1 % partout. **Aucune variante ne change la STRUCTURE du découpage**, elles
ne font que redistribuer des lettres.

Balayage des garde-fous (longueur minimale de la partie partagée 2/3/4,
référence obligée d'être une section pleine) : à 4 bi-mesures ou avec
« référence pleine », A et B deviennent **strictement identiques au témoin** —
0 gagné, 0 perdu, 198 blocs, au bit près. Voir §4, c'est le cœur du problème.

---

## 2. Qui gagne, qui perd — nommément

### Variante A (et B, mêmes morceaux à un près)

| morceau | témoin | A | ce qui se passe |
|---|---|---|---|
| Yesterday | 55 % | **36 %** | perd |
| Let It Be | 91 % | **74 %** | perd |
| Be My Baby | 29 % | **17 %** | perd |
| Every Breath You Take | 75 % | 73 % (variante B seule) | perd |
| les 14 autres | — | inchangés | la règle ne se déclenche jamais |

**Zéro morceau gagné, à tous les réglages testés.**

### Variante C/D (la règle posée au bon endroit)

| morceau | témoin | C | |
|---|---|---|---|
| Sunny | 17 % | **33 %** | gagne, et musicalement juste (ci-dessous) |
| Bein Green | 75 % | **81 %** | gagne |
| Be My Baby | 29 % | **25 %** | perd |
| Yesterday | 55 % | **52 %** | perd |

### Trois découpages en toutes lettres

**Sunny** — le seul vrai gain, et il est beau.

```
TOI      : intro 1 · A 2-9 · B 10-17 · A 18-25 · B 26-33 · C 34-41 · D 42-49 · …
témoin   : intro 1 · A 2-9 · B 10-17 · A 18-33            · C 34-41 · D 42-49 · …
C        : intro 1 · A 2-9 · B 10-17 · A 18-25 · B 26-33 · C 34-41 · D 42-49 · …
```
Le témoin avait avalé l'alternance couplet/refrain dans un seul bloc de 16
mesures (`A 18-33`). La coupe (`aaaa` ⊂ `aaaaabca`) la rend exactement.
+8 mesures justes. **À écouter : 18-25 puis 26-33.**

**Yesterday** — la perte, et elle est laide.

```
TOI      : intro 1-2 · A 3-9 · A 10-16 · B 17-20 · B 21-24 · A 25-31 · B 32-35 · B 36-39 · A 40-46 · outro 47-48
témoin   : intro 1-2 · A 3-9 · A 10-16 · B 17-20 · B 21-24 · A 25-32 · C 33-40 · B 41-46 · outro 47-48
C        : intro 1-2 · A 3-9 · A 10-16 · B 17-20 · B 21-24 · A 25-30 · outro 31-32 · A 33-38 · C 39-46 · outro 47-48
A        : intro 1-2 · A 3-9 · A 10-16 · B 17-20 · B 21-32              · C 33-46              · outro 47-48
```
C fabrique un **« outro » au milieu du morceau** (mesures 31-32) : la queue
détachée est courte, elle ressemble à ton outro par la SSM, elle en prend le
nom. A, lui, fusionne tout : `B 21-32` et `C 33-46` sont deux blocs de
12 et 14 mesures qui ne correspondent à rien.

**Let It Be** — la plus grosse perte de A, par un effet indirect.

```
témoin   : … A 47-54 · C 55-56 · B 57-64 · B 65-68 · outro 69-70     (42/46 mesures justes)
A        : … A 47-54 · C 55-64          · B 65-68 · outro 69-70     (34/46)
```
La règle a coupé `aaaa` hors de `aaaab` ; le reste de 1 bi-mesure part en
queue, `grouper_restes` le recolle à son voisin, et **un bloc de 10 mesures
naît là où il y en avait deux**. La règle du préfixe fabrique des restes, et
les restes se recollent : le dégât se produit une étape plus loin.

---

## 3. Le point qui compte : la règle ne peut pas voir ton exemple

Tu regardais Don't Want My Love (`min_B6AHb9W_LkM`). Deux choses.

**(a) Le bloc qui t'agace n'est pas fabriqué par `nommer` — et `nommer` avait
déjà raison.** Voici ce que rend chaque étape, sous cible = 6, avec tes deux
traits :

```
nommer           A 'aaaa'   1-8  RESTE
                 B 'aaaaaa' 9-20
                 C 'a'     21-22 RESTE
                 D 'bbb'   23-28 RESTE     ←┐  MÊME lettre : nommer a vu
                 B 'aaaaaa' 29-41           │  que c'est deux fois la même
                 D 'bbb'   42-47 RESTE     ←┘  chose (ton alternance F#-/E-)
                 A 'aaaa'  48-56 RESTE

grouper_restes   A 'aaaa'      1-8
                 B 'aaaaaa'    9-20
                 C 'abbb'     21-28    ← a + bbb, recollés, REBAPTISÉS C
                 B 'aaaaaa'   29-41
                 D 'bbbaaaa'  42-56    ← bbb + aaaa, recollés, REBAPTISÉS D
```

**`nommer` avait donné la même lettre aux deux `bbb`.** C'est
`grouper_restes` qui les a recollés à des voisins différents — un `a` devant
d'un côté, un `aaaa` derrière de l'autre — et qui a jeté les deux lettres pour
en inventer deux neuves. Ton intuition est juste sur le fond (les deux blocs
contiennent la même chose), mais **le fautif est `grouper_restes`, pas
`nommer`**, et une règle ajoutée dans `nommer` passe avant que le problème
existe. (C'est pour ça que j'ai aussi mesuré C et D : la même règle posée
après `grouper_restes`. Sur ton exemple littéral `bbbaa` / `bbbaaaa` elle rend
bien `C` + `C` + une queue `aa` — elle fait exactement ce que tu demandes.)

**(b) Sur le chart d'aujourd'hui, les deux blocs ne partagent aucun préfixe.**
Avec tes deux traits, le mot est `aaaaaaaaaaabbbaaaaaabbbaaaa` et la cible 6
donne :

```
A 'aaaa'    mes. 1-8
B 'aaaaaa'  mes. 9-20
C 'abbb'    mes. 21-28     ← ton C
B 'aaaaaa'  mes. 29-41
D 'bbbaaaa' mes. 42-56     ← ton D
```

`abbb` et `bbbaaaa` : **plus long préfixe commun = 0**. La matière partagée
(les trois `b`, ton alternance F#-/E-) est là dans les deux, mais **à une
bi-mesure de décalage** — en tête dans D, en deuxième position dans C. Ton
oreille dit « D = C + une queue » ; le mot, lui, dit « D commence là où C en
est à son deuxième mot ». Un préfixe ne peut pas voir ça.

Ce n'est pas un accident de ce morceau. Sur les 18 morceaux, pour toutes les
paires (bloc court, bloc long) qui partagent au moins 2 bi-mesures :

| | longueur moyenne |
|---|---|
| partie partagée **en tête** (préfixe) | 1,61 bi-mesures |
| partie partagée **n'importe où** | 2,38 bi-mesures |

et **21 paires sur 76** partagent au moins 2 bi-mesures de plus ailleurs qu'en
tête. La bonne primitive n'est pas « commence par », c'est « contient le même
motif, éventuellement décalé » — autrement dit **une question de phase de
grille, pas de nommage**.

---

## 4. Pourquoi A et B ne gagnent jamais : elles ne se déclenchent que sur du vide

Journal de tous les déclenchements de A sur le corpus — 4 morceaux, 16 coupes :

```
Chain Of Fools   aaaa ⊂ aaaaaa   aaaaa ⊂ aaaaaa
Yesterday        aaa  ⊂ aaaa     (×4)
Let It Be        ba   ⊂ baba     aaaa ⊂ aaaab   (×2 chacune)
Be My Baby       ba   ⊂ baae     ba   ⊂ baa     (×6)
```

**Pas une seule fois la référence n'est une vraie section.** Ce sont des
références de 2 à 5 bi-mesures, presque toutes des suites d'une seule lettre.
`merges4` plafonne toute soudure à `cible` (4 ou 6) bi-mesures : pour qu'un
bloc « commence par » un autre, il faut que le plus long dépasse le plus court
**sans dépasser 6** — la fenêtre est d'une ou deux bi-mesures, et dans cette
fenêtre la seule chose qui ressemble à un préfixe, c'est une répétition du
même jeton. D'où le résultat d'un balayage qui ne ment pas :

| garde-fou | accord | gagne / perd |
|---|---|---|
| partie partagée ≥ 2 bi-mesures | 54,0 % | 0 / 3 |
| ≥ 3 | 55,7 % | 0 / 1 |
| **≥ 4** | **56,5 %** | **0 / 0** — identique au témoin |
| référence = section pleine | 56,5 % | 0 / 0 — identique au témoin |

Dès qu'on interdit les références dérisoires, **la règle ne fait plus rien du
tout**. Il n'existe pas de réglage où elle gagne quoi que ce soit.

---

## 5. Le piège du mot pauvre : vérifié, il se produit, et il coupe dans les deux sens

Ton chiffre est exact : sur Don't Want My Love, **22 jetons sur 28 portent la
lettre `a`, soit 79 %**. Le corpus entier, classé par pauvreté du mot :

| morceau | lettre majoritaire | lettres distinctes | la règle fusionne… |
|---|---|---|---|
| Chain Of Fools | **97 %** | 2 | `aaaa` ⊂ `aaaaaa`, `aaaaa` ⊂ `aaaaaaaa` — vide de sens |
| The Walk | **92 %** | 3 | `aaaaaa` ⊂ `aaaaaaaa` — vide de sens |
| The Lazy Song | 83 % | 3 | `aaaa` ⊂ `aaaabaaaa` — vide de sens |
| Let It Be | 71 % | 3 | `aaaa` ⊂ `aaaab` — **casse le morceau (−17 pts)** |
| Yesterday | 70 % | 3 | `aaa` ⊂ `aaaa` — **casse le morceau (−19 pts)** |
| … | | | |
| Every Breath You Take | 33 % | 8 | `abcb` ⊂ `abcbabca`, `daef` ⊂ `daefabcbc` — **musicalement plausible** |
| Sunny | 30 % | 10 | `aaaa` ⊂ `aaaaabca` — **le seul vrai gain** |

Oui, ça arrive, et exactement où tu le craignais : **les deux morceaux que la
règle casse le plus (Let It Be, Yesterday) sont les deux plus pauvres parmi
ceux où elle se déclenche.** Les fusions plausibles n'apparaissent que sur les
mots riches (8 à 10 lettres distinctes).

Le garde-fou évident — « la partie partagée doit contenir au moins deux
lettres différentes » — a été mesuré (variante E) : il **supprime bien les
dégâts** (Yesterday revient au témoin) mais il **supprime aussi le gain de
Sunny**, dont la coupe utile est précisément `aaaa` ⊂ `aaaaabca`, une référence
monotone. Résultat : 56,5 %, 1 gagné, 1 perdu — exactement le témoin, en plus
compliqué. Le mot ne porte pas l'information qui séparerait les deux cas.

---

## 6. Avis franc

**La règle ne vaut pas la peine, dans aucune des deux variantes.**

* **A (préfixe entier) et B (plus long préfixe commun) sont à rejeter sans
  réserve** : 0 morceau gagné à tous les réglages, 3 à 4 cassés, et quand on
  les rend prudentes elles deviennent l'identité. B n'apporte rien sur A : la
  coupe rétroactive ne se déclenche différemment que sur un seul morceau, et
  elle y perd.
* **C/D (la même règle, posée après `grouper_restes`, là où vit ton « D »)
  sont défendables mais ne sont pas un progrès** : +0,8 point d'accord, 2
  gagnés contre 2 perdus, structure inchangée (75,1 % de paires, au dixième
  près). C'est un échange, pas un gain. Et on sait déjà que le morceau qui a
  motivé l'idée n'en profiterait pas.

**Mécanisme de l'échec, en une phrase** : la règle suppose que deux sections
parentes sont alignées sur la même case de départ, or la grille de bi-mesures
n'a pas de phase garantie — la matière partagée se retrouve en moyenne
**décalée** (2,38 bi-mesures partagées n'importe où contre 1,61 en tête), et
ton propre exemple en est le cas extrême (0 en préfixe, 3 décalées).

### Ce que je ferais à la place, par ordre de coût

1. **Chercher le motif commun, pas le préfixe.** Remplacer « commence par »
   par « contient, à un décalage près » dans la variante C. C'est trois lignes
   de plus, ça se mesure avec le même harnais, et le tableau du §3 dit que
   c'est là qu'est la matière. **Je n'ai pas eu le temps de le mesurer** — mais
   c'est la seule suite que la mesure désigne.
2. **Regarder `grouper_restes` — c'est LÀ qu'est le levier, et je l'ai mesuré.**
   Variante G : *un reste dont le type se répète ailleurs dans le morceau n'est
   pas du reste — il garde sa lettre et n'est jamais recollé.*

   | | accord lettre/mesure | **paires groupées comme toi** | blocs |
   |---|---|---|---|
   | témoin | 56,5 % | 75,1 % | 198 |
   | A / B (ta règle) | 54,0 / 53,7 % | 73,7 / 74,6 % | 193 / 190 |
   | C / D (ta règle, bien placée) | 57,3 % | 75,1 % | 203 |
   | **G (grouper_restes)** | 55,3 % | **76,2 %** | 256 |

   **C'est la seule variante qui bouge la STRUCTURE** (le mètre insensible au
   nom et au décalage, celui qui dit la vraie qualité). Et elle débloque deux
   des trois morceaux que `known_issues.md` note comme coincés :
   **The Walk 0 % → 43 %** et **Easy On Me 28 % → 55 %**.
   Elle est trop brutale telle quelle — 2 gagnés / 7 perdus sur les lettres, et
   256 blocs au lieu de 198, donc elle fragmente : en refusant tout recollage
   elle laisse traîner des blocs d'une bi-mesure. Il lui faut un adoucissement
   (ne protéger que les restes d'au moins 2 bi-mesures, ou recoller quand même
   mais **garder la lettre du morceau le plus long** au lieu d'en inventer une
   neuve). **C'est la piste que je poursuivrais.**
3. **Ne rien changer sur `nommer`.** 14 morceaux sur 18 ne bougent d'aucune
   façon avec aucune variante de la règle du préfixe. Le levier n'est pas là.

### Ce que cette mesure ne dit PAS

* La vérité terrain est ton propre découpage, que tu dis imparfait. Ces
  pourcentages mesurent l'accord avec toi, pas la justesse musicale. Le seul
  découpage que j'ai regardé à l'œil et qui gagne **musicalement** est Sunny
  (§2) ; les autres mouvements sont des fragmentations.
* Don't Want My Love n'a pas de section annotée : il n'est PAS dans les 18
  mesurés. Son cas est traité qualitativement (§3).
* Je n'ai pas réussi à reproduire à l'identique le `C = bbbaa` que tu cites
  (le chart a été recuit plusieurs fois aujourd'hui : `charts.bak_20260917_1857`,
  `_1922`). J'obtiens `C = abbb` et **`D = bbbaaaa`, qui est bien ton D**. Si
  ton `bbbaa` est exact, alors la règle se déclencherait sur ton morceau — je
  l'ai vérifié sur l'exemple littéral, elle rend `C` + `C` + queue `aa` — mais
  le corpus dit quand même 2 gagnés / 2 perdus.
