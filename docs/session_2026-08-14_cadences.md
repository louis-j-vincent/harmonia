# 2026-08-14 — LA CADENCE : mesurer « la phrase finit sur quelque chose de cohérent »

*Journal de session, tenu au fil de l'eau. Budget 90 min (12:10 → 13:40).*

Louis : « Il y a une vraie logique harmonique : en général une section a un sens
mélodique, la phrase finit sur quelque chose de cohérent, je ne sais pas comment
l'expliquer. » Ce qu'il décrit s'appelle une **cadence** : une section ne
s'arrête pas au milieu d'une progression, elle se résout.

---

## 0. Un bug de calibration trouvé en chemin (12:30)

Le chroma NNLS (`capture()["arr"]`, d'où sortent « basse » et « harmonie » des 39
critères) a **l'index 0 = LA, pas DO** — c'est écrit dans
`harmonia_min/nnls_features.py:11` et personne ne l'avait payé jusqu'ici parce
que tous les usages précédents étaient invariants par transposition (SSM,
produits scalaires, noyau harmonique).

Dès qu'on compare la basse à un nom d'accord musx, ça compte : mesuré,
`racine_musx − argmax_basse ≡ 9` demi-tons dans **75 %** des mesures du corpus.
Sans la correction, « la racine est la tonique locale » tombait à **2 %** de taux
de base (au lieu de 43 %) et toutes les cadences étaient invisibles.

**Conversion : `pc = (index_nnls + 9) % 12`.** Rien d'autre dans le projet n'est
touché, mais tout futur croisement basse ↔ accord doit la faire.

---

## 1. Test de prémisse (le moins cher d'abord)

Aucun détecteur : juste « cet événement est-il sur-représenté autour des
frontières annotées ? ». Ratio = taux sur ces mesures / taux sur toutes les
autres. Colonne **0** = la mesure de frontière (= première mesure de la section
suivante), colonne **−1** = la dernière mesure de la section qui se termine.

| événement | −3 | −2 | **−1** | **0** | +1 | +2 | base |
|---|---|---|---|---|---|---|---|
| quinte desc. mesure→mesure | 1,02 | 1,09 | 0,99 | **0,98** | 0,65 | 0,77 | 21 % |
| quinte desc. dans la mesure (demi-mes.) | 1,08 | 1,49 | 0,80 | 1,11 | 0,76 | 1,10 | 24 % |
| tenue (même racine qu'avant) | 0,98 | 0,79 | **1,57** | 1,02 | 1,40 | 0,84 | 29 % |
| racine = tonique locale (musx) | 0,96 | 0,98 | 1,11 | **1,48** | 1,10 | 0,81 | 43 % |
| racine = tonique locale (basse) | 1,01 | 0,97 | 1,26 | **1,50** | 1,23 | 0,72 | 37 % |
| **V → I (clé locale)** | 0,41 | 1,15 | 0,97 | **2,85** | 0,13 | 0,20 | 8 % |
| V en b, I en b+1 (posé en b) | 1,14 | 1,22 | **2,69** | 0,19 | 0,27 | 0,78 | 7 % |
| IV → I (plagale) | 0,52 | 1,47 | 0,38 | **2,43** | 0,24 | 2,20 | 3 % |

**Trois résultats, dont un négatif franc.**

1. **La piste 1 telle qu'écrite dans le brief est morte.** La quinte descendante
   *nue* — sans savoir où est la tonique — ne dit rien du tout (0,98x). Elle est
   trop banale : 21 % des mesures du corpus en contiennent une (les boucles
   pop tournent en quintes en permanence). Ce n'est pas le mouvement qui fait la
   cadence, c'est **sur quoi il tombe**.
2. **La même quinte devient le meilleur indice du lot dès qu'on lui demande
   d'arriver sur la tonique locale : 2,85x.** V→I n'est pas rare pour rien —
   8 % des mesures — et il est presque trois fois plus fréquent sur une
   frontière.
3. **Le décalage est 0, pas −1.** La résolution tombe sur le **premier temps de
   la section suivante**, pas sur la dernière mesure de la section qui finit.
   Autrement dit le V est la dernière mesure, le I est la frontière. C'est la
   turnaround pop, et c'est une bonne nouvelle : le décalage est nul et il n'y a
   rien à corriger.

Deux détails qui serviront ensuite : la dernière mesure d'une section **tient
son accord** (1,57x — la progression s'immobilise avant de repartir), et la
tonique locale reste utile toute seule (1,5x) mais elle est bien trop fréquente
(37–43 % des mesures) pour servir de détecteur.

*(scripts : `scratchpad/premisse_cadence.py`, `scratchpad/premisse2.py`)*

---

## 2. Le tableau des dix-huit formulations (13:00)

`scripts/cadences.py --table`. Budget commun : **un pic toutes les 6 mesures**
pour toutes les lignes — sans ça, une courbe bruitée « trouve » tout. Rappel =
part des 197 frontières annotées touchées par un pic.

**La ligne la plus importante du tableau est la dernière.** Avec un pic toutes
les 6 mesures et une frontière toutes les 8, des pics tirés au SORT en touchent
déjà 43 % à ±1 mesure. Un rappel de 60 % ne veut donc pas dire « trois sur
cinq », il veut dire « +17 sur le hasard ».

| formulation | rappel ±0 | rappel ±1 | **gain** | pics/morceau | écart |
|---|---|---|---|---|---|
| la basse est sur la tonique locale | 26 % | 60 % | **+17** | 14,2 | +0,0 ± 1,0 |
| V→I (ou IV→I) vers la tonique locale | 29 % | 59 % | **+16** | 14,2 | +0,0 ± 1,0 |
| CADENCE harmonique (V→I + arrivée) | 29 % | 56 % | +13 | 14,1 | +0,0 ± 2,0 |
| V→I vers la tonique GLOBALE | 27 % | 56 % | +13 | 14,2 | +0,0 ± 2,0 |
| CADENCE = basse résout + chant se ferme | 23 % | 55 % | +12 | 14,2 | +0,0 ± 2,0 |
| (d) un SILENCE avant la phrase suivante | 20 % | 53 % | +10 | 14,2 | +1,0 ± 2,0 |
| (b) la dernière note est plus LONGUE | 18 % | 53 % | +10 | 14,2 | +0,0 ± 2,0 |
| surprise du modèle de progression | 22 % | 52 % | +9 | 14,2 | +1,0 ± 1,0 |
| ARRIVÉE sur la tonique | 26 % | 51 % | +8 | 14,2 | +0,0 ± 2,0 |
| les quatre mesures de chant réunies | 14 % | 51 % | +8 | 14,2 | +1,0 ± 2,0 |
| saut de surprise (phrase de 4 mesures) | 23 % | 48 % | +5 | 13,6 | +1,0 ± 1,0 |
| quinte descendante nue | 14 % | 47 % | +4 | 14,2 | +1,0 ± 2,0 |
| quinte descendante, demi-mesure | 17 % | 46 % | +3 | 14,2 | +0,0 ± 2,0 |
| saut de surprise (phrase de 8 mesures) | 20 % | 46 % | +3 | 13,7 | +1,0 ± 1,0 |
| (c) le contour DESCEND | 13 % | 45 % | +2 | 14,2 | +1,0 ± 2,0 |
| (a) la phrase finit sur la tonique/tierce | 12 % | 42 % | −1 | 13,6 | +1,0 ± 2,0 |
| **LE HASARD, même budget** | **17 %** | **43 %** | — | 14,2 | — |

Le classement tient à quatre densités de pics différentes (1/4, 1/6, 1/10,
1/16 — `scratchpad/diag_cadence.py`), donc il n'est pas un artefact du budget.

### Ce que chaque piste du brief a donné

**Piste 1 — quinte descendante : morte telle quelle, vivante recadrée.** Nue,
elle fait +4 (dans le bruit). Dès qu'on exige qu'elle ARRIVE sur la tonique
locale, +16. Le mouvement ne fait pas la cadence ; sa destination, oui.

**Piste 2 — retour à la tonique : la gagnante, et la version la plus simple
gagne.** « La basse est sur la tonique locale » (+17) bat « V→I » (+16) et bat
« ARRIVÉE sur la tonique » (+8). C'est contre-intuitif et c'est net : demander
un *mouvement* vers la tonique fait perdre du rappel par rapport à simplement
*être* sur la tonique. La tonique GLOBALE (une seule tonalité pour tout le
morceau) fait +13 contre +16 pour la locale : la clé locale apporte 3 points,
pas plus — ces 18 morceaux modulent peu.

**Piste 3 — la fermeture mélodique : la plus décevante.** Les quatre mesures ont
été testées séparément comme demandé. Deux marchent un peu — le **silence** avant
la phrase suivante (+10) et la **note finale longue** (+10) — et ce sont les deux
qui ne parlent PAS de hauteur. Les deux qui parlent de hauteur échouent : le
contour descendant fait +2, et « la phrase finit sur la tonique ou la tierce »
fait **−1, soit exactement le hasard**. Les réunir donne +8, moins que la
meilleure toute seule : elles ne sont pas indépendantes, elles se gênent.
Interprétation : ce que la voix apporte ici, c'est **qu'elle s'arrête**, pas
**sur quoi** elle s'arrête. (Réserve honnête : la hauteur de la dernière note
vient de `licks.notes_de`, du suivi de f0 sur une piste séparée — le bruit de
transcription est plus fort sur la dernière note d'une phrase, celle qui
decrescendo.)

**Piste 4 — la surprise du modèle de progression : +9, honorable, pas la
gagnante.** Bigramme de degrés relatifs à la clé locale, appris **en
leave-one-song-out** sur les 17 autres morceaux. La formulation exacte de Louis
(« surprise basse dans la phrase, haute au passage ») — le *saut* de surprise
sur une fenêtre de 4 ou 8 mesures — fait **moins bien** (+5 et +3) que la
surprise brute (+9). Le modèle appris n'a donc pas trouvé de « phrase cohérente »
à l'intérieur de laquelle la surprise serait basse ; il a juste appris que
certains enchaînements sont rares.

**Les combinaisons n'aident pas.** CADENCE (basse + chant) fait +12, moins que
la basse seule (+17). Ajouter la surprise : +10. Chaque ajout dilue. C'est le
signe que les cues ne sont pas indépendants là où ça compte.

## 3. Le décalage : nul, et stable

**Le pic tombe sur la frontière, pas sur la mesure d'avant.** Médiane de l'écart
= **+0,0 mesure** pour toutes les formulations harmoniques, MAD 1,0 à 2,0. Les
formulations de chant et de surprise sont à +1,0 (elles piquent une mesure trop
tard). Rien à corriger sur la famille gagnante — c'est le meilleur résultat
secondaire de la session, parce qu'un décalage variable aurait tué l'idée.

Musicalement : **le V est la dernière mesure de la section qui finit, le I est le
premier temps de la suivante.** La cadence enjambe la frontière.

## 4. La cadence en ÉVÉNEMENT (peu de pics, plus précis)

Sans budget de pics : l'événement se produit ou pas. « gain » = précision divisée
par la probabilité qu'une mesure au hasard tombe à ±1 d'une frontière (38 %).

| événement | /morceau | précision | gain | rappel |
|---|---|---|---|---|
| V→I basse continu, z ≥ 3,0 | 9,1 | 55 % | **1,44x** | 35 % |
| V→I basse continu, z ≥ 2,5 | 11,9 | 50 % | 1,33x | 43 % |
| V→I basse continu, z ≥ 2,0 | 14,3 | 49 % | 1,30x | 51 % |
| V→I strict (noms musx) | 5,8 | 44 % | 1,16x | 23 % |
| V/IV/bVII→I (noms musx) | 9,7 | 38 % | 1,00x | 23 % |
| silence du chant ≥ 1 mesure | 16,2 | 41 % | 1,07x | 39 % |
| **V/IV/bVII→I ET silence ET tenue** | **0,9** | **62 %** | **1,65x** | 5 % |

La conjonction des trois donne le meilleur gain du corpus (1,65x) mais **0,9
événement par morceau** : c'est un objet de type « ancre », pas un détecteur.
À comparer aux ancres actuelles : 1,4/morceau à 96 %. La cadence-conjonction est
plus fréquente mais bien moins sûre — **elle n'a pas sa place dans le vote des
ancres en l'état.**

## 5. Ce que la cadence attrape que les 39 critères ne voient pas : RIEN

C'était la question. La réponse est franche et négative.

Sur les 197 frontières, en donnant à chacun des 39 critères de
`criteres_sections.py` le même budget (un pic toutes les 6 mesures, ±1 mesure) :

* la frontière **médiane est vue par 23 des 39 critères** ;
* la **moins visible de tout le corpus est vue par 6 critères** ;
* **zéro frontière n'est invisible aux 39.**

Il n'y a donc pas de « frontière que seule la cadence voit ». Sur les 5
frontières les plus discrètes (≤8/39), la cadence en attrape 3 — soit son taux
ordinaire (59 %), pas mieux. **La cadence n'ouvre pas un canal nouveau ; elle
regarde le même endroit, moins bien.**

## 6. Par morceau : de +40 à −22 (et pourquoi)

Gain sur le hasard, formulation V→I basse, budget 1/6, ±1 mesure :

| gain | morceaux |
|---|---|
| **+27 à +40** | Blue Lights, Be My Baby, The Lazy Song, Sunny, Every Breath, She Will Be Loved, ABC |
| +11 à +25 | Let It Be, The Walk, Don't Know Why, Chain of Fools, Yesterday |
| **−1 à −22** | Bein' Green, Grenade, Happy, Goodbye Yellow Brick Road, **This Love (−12)**, **Stand By Me (−22)** |

Ce n'est PAS une affaire de mobilité harmonique — Blue Lights (3 racines) et
Sunny (12 racines) gagnent tous les deux ; This Love (8 racines, 96 % de mesures
qui changent d'accord) perd. La corrélation gain / régularité des cadences est
−0,13, donc l'hypothèse « la cadence est noyée dans la boucle » ne suffit pas
non plus.

**Le vrai mécanisme est une question de PHASE, pas de période.** Stand By Me est
le cas d'école : ses cadences tombent tous les 8 mesures — exactement la longueur
de ses sections — et c'est le pire morceau du corpus (−22). Son cycle I–vi–IV–V
résout au début de chaque boucle de 4 mesures, pas au début de chaque section.
Le détecteur a la bonne période et la mauvaise phase.

Vérifié à l'échelle du corpus : la phase de la grille de 4 mesures que
désignent les pics de cadence coïncide avec celle des frontières annotées dans
**6 morceaux sur 18 (33 %, hasard 25 %)**. Et la répartition de l'écart de phase
n'est pas uniforme : **9 morceaux sur 18 ont un écart de 2 mesures**, c'est-à-dire
le milieu de la phrase de 4. La cadence ne se trompe pas au hasard, elle marque
**la moitié de la phrase** aussi souvent que sa fin.

**Conclusion sur le mécanisme : la cadence détecte des fins de PHRASE, pas des
fins de SECTION.** Toute fin de section est une fin de phrase, l'inverse est
faux, et il y a 2 à 4 fins de phrase par section. Le gain de +17 points mesuré,
c'est exactement le taux « cette fin de phrase est aussi une fin de section ».

## 7. Verdict et recommandation

**Le critère d'arrêt n'est pas atteint.** La meilleure formulation fait 60 % de
rappel à un pic toutes les 6 mesures avec un écart stable de +0,0 ± 1,0 mesure —
mais le hasard fait 43 % dans les mêmes conditions. Le gain réel est **+17
points**, pas 60 %.

**Ce qui marche, à garder :**
1. La destination compte, pas le mouvement : « la basse est sur la tonique
   locale » (+17) est la formulation gagnante, et c'est la plus simple des dix-huit.
2. Le décalage est **nul et stable** (+0,0 ± 1,0). Le V est la dernière mesure,
   le I est la frontière.
3. Le bug NNLS index-0-=-LA est corrigé et documenté ; tout croisement futur
   basse ↔ accord en dépend.

**Ce qui ne marche pas, et pourquoi — pour ne pas le refaire :**
1. La quinte descendante nue (+4) : trop banale, 21 % des mesures du corpus en
   contiennent une.
2. La fermeture mélodique par la HAUTEUR (finir sur la tonique ou la tierce,
   contour descendant) : au niveau du hasard. Seul « la voix s'arrête » (silence,
   note longue) porte un peu d'information.
3. Le saut de surprise d'un modèle de progression (+3 à +5) : moins bon que la
   surprise brute. Il n'y a pas de « plateau de cohérence » intra-phrase mesurable
   avec un bigramme de degrés.
4. Toutes les combinaisons testées : elles diluent au lieu d'additionner.
5. La cadence comme canal complémentaire aux 39 critères : **aucune frontière ne
   lui est réservée.**

**La prochaine hypothèse à tester, et c'est un changement de rôle plutôt qu'un
réglage :** ne pas se servir de la cadence comme détecteur de frontière, mais
comme **détecteur de la grille de phrases**. Elle est très périodique (jusqu'à
88 % de ses pics sur la même phase mod 4 : The Walk, Sunny 87 %, Stand By Me
79 %, Let It Be 75 %) et elle a une phase propre, stable, mais décalée de 0 ou 2
mesures par rapport aux frontières. Le test à faire : **la cadence donne-t-elle
la période et la phase des PHRASES de 4 ou 8 mesures ?** Si oui, elle règle le
problème que `ancres.py` documente comme sa faiblesse connue — « sur un morceau
à une ou deux ancres, la phase élue est un tirage au sort » — et le filtre de
phase des ancres vaut +23 points de précision. Une cadence qui fournit la phase
sans jamais nommer une frontière vaudrait plus qu'une cadence qui nomme mal des
frontières.

## Artefacts

* `scripts/cadences.py` — les 18 formulations, `--table` et `--songs <clé>`.
* `docs/plots/cadences.html` — 4 morceaux, une ligne par formulation, chaque pic
  cliquable joue 2 mesures avant → 2 après.
  <http://100.89.209.63:7772/plots/cadences.html>
* `scratchpad/premisse_cadence.py`, `premisse2.py` — les tests de prémisse.
* `scratchpad/diag_cadence.py` — clé locale + balayage de densité vs hasard.
* `scratchpad/diag2.py` — événements discrets, conjonctions, complémentarité.
* `scratchpad/diag3.py` — les frontières invisibles aux 39, gain par morceau.
* `scratchpad/cadence_cache/*.pkl` — basse/harmonie/accords/chant par case.

## 8. Le mécanisme, visible en trois clics sur Sunny

Sunny module toutes les 16 mesures (Em → Fm → F#m → Gm). Les cadences détectées,
avec leurs noms d'accords, disent tout :

```
mes. 22  hors annotation   C -> B -> Em -> G      <- vraie cadence, milieu de section
mes. 26  PILE              C -> B -> Em -> G      <- la MÊME, fin de section
mes. 54  hors annotation   D -> C# -> F#m -> A
mes. 58  PILE              D -> C# -> F#m -> A
mes. 70  hors annotation   Eb -> D -> Gm -> Gm
mes. 74  PILE              Eb -> D -> Gm -> Bb
```

Le détecteur ne se trompe pas : **il trouve deux V→i identiques par section, l'un
au milieu, l'autre à la fin, et rien dans l'harmonie ne les distingue.** C'est
littéralement la même cadence jouée deux fois. Aucun réglage de seuil ne
séparera ces deux-là ; il faut une information de niveau supérieur (la longueur
de phrase, la répétition, la voix).

Les trois à écouter sur la page :
1. **Sunny, mes. 26** (vert) — `C → B7 → Em`, la cadence qui ferme la section.
2. **Sunny, mes. 22** (gris) — la même quatre mesures plus tôt, au milieu. Le
   détecteur a raison musicalement et tort structurellement.
3. **Every Breath You Take, mes. 33 et 75** (verts) — `Eb → Ab`, V→I en lab,
   deux fins de section propres.

*Note de précision : le tableau de la page affiche « +18 » là où ce document dit
« +17 » pour la formulation gagnante — deux tirages différents du hasard
(200 tirages chacun, écart-type ≈ 1 point). L'ordre du classement est le même.*
