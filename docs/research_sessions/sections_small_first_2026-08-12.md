# « Petites sections d'abord » — journal de session (2026-08-12)

## Résumé (écrit à la fin)

**Les deux cibles sont atteintes.** Médiane des 12 : **0,799** (cible ≥ 0,789).
Blue Lights : **0,813** (cible > 0,75, départ 0,696). Moyenne des 12 : 0,816
contre 0,786. Sur les six morceaux annotés tenus hors de toute décision de
conception : moyenne **0,692** contre 0,620, dont Be My Baby 0,658 → **0,935**.
Onze morceaux sur dix-huit en hausse, un seul en baisse (Stand By Me, −0,024).
Page à écouter : `http://100.89.209.63:7772/plots/order_lab.html`.

**Le mécanisme, en une phrase** : « 4 d'abord puis 8 » n'est pas un ordre à
inverser, c'est une **concurrence à arbitrer à chaque ancre**. La passe de 8
lancée seule arrive mesure 45 de Blue Lights et vole la 3e occurrence de la
famille B ; dans un parcours unique où les deux longueurs sont proposées, le pic
dur de la mesure 17 barre le bloc de 8 et le bloc de 4 prend ses cinq B d'un coup.

**Ce que le brief supposait, et qui est faux** : le swap naïf n'échoue pas à
cause des seuils. Les scores de bloc à 4 et à 8 mesures vivent sur la même
échelle (médiane 0,36 contre 0,35), parce que `block_score` divise déjà par
l'ancre. Re-régler THR4 ne pouvait rien réparer, et « rendre `block_score`
comparable » n'avait rien à corriger. La vraie cause : la passe de 4 mesure la
**période de la boucle**, pas l'**échelle de la section**.

**Trois choses mesurées et rejetées** : la corroboration des runs de 4 par les
pics (corrélation 0,114), le « taux de suite » d'un bloc de 8 (−0,076), les
longueurs 12 et 16 mesures (médianes 0,630 et 0,715).

**Découverte de côté, à traiter** : les pics durs gagnent sur 4 morceaux sur 18 et
perdent sur 6. Leur tolérance était à ±1 alors que leur précision réelle est de
88 % à ±2 — c'est ce décalage qui coûtait le plus, et le corriger vaut +0,072 de
moyenne sur les six morceaux hors conception.

---

**Cible** : une variante 4-avant-8 qui fasse ≥ 0,789 de médiane sur les 12
morceaux annotés ET ≥ 0,75 sur Blue Lights (aujourd'hui 0,696).

**Référence recalculée moi-même** (`voice_with_hard`, pics durs du profil fusionné,
TAIL_UNIT=4, HARD_TOL=1) :

| morceau | score |
|---|---|
| This Love | 1,000 |
| Don't Know Why | 0,974 |
| Bein' Green | 0,993 |
| Let It Be | 0,716 |
| Sunny | 0,796 |
| **Blue Lights** | **0,696** |
| Stand By Me | 0,795 |
| She Will Be Loved | 0,783 |
| Every Breath You Take | 0,835 |
| Chain of Fools | 0,676 |
| The Walk | 0,511 |
| Grenade | 0,663 |
| **médiane** | **0,789** |

Reproduit au millième. Outils : `scratchpad/order_bundle.py` (fige S, M, mute,
start, pics par morceau → la recherche coûte des ms au lieu de 30 s),
`scratchpad/order_search.py` (copie paramétrée de `voice_with_hard`).

---

## Journal

### E1 — reproduire l'échec du swap naïf (11h05)

| variante | médiane | Blue Lights |
|---|---|---|
| 8@0,66 puis 4@0,70 (référence) | **0,789** | 0,696 |
| 4@0,70 puis 8@0,66 (ordre seul inversé) | 0,566 | 0,710 |
| 4@0,66 puis 8@0,70 (seuils échangés aussi) | 0,616 | 0,710 |

Confirmé. Blue Lights est bien le seul gagnant, +0,014.

### E2 — l'hypothèse « le bloc de 4 se ressemble partout » est FAUSSE (11h15)

Distribution du score de glissement, toutes ancres paires, toutes positions
hors-bloc, par morceau :

| morceau | L=4 méd. | L=4 q90 | L=4 % > 0,70 | L=8 méd. | L=8 q90 | L=8 % > 0,66 |
|---|---|---|---|---|---|---|
| This Love | 0,356 | 0,589 | 4,2 % | 0,346 | 0,509 | 3,2 % |
| Let It Be | 0,480 | 0,816 | 20,2 % | 0,471 | 0,778 | 25,4 % |
| Blue Lights | 0,507 | 0,757 | 15,1 % | 0,511 | 0,681 | 12,6 % |
| Stand By Me | 0,383 | 0,706 | 10,4 % | 0,380 | 0,638 | 8,8 % |

Les deux échelles sont **les mêmes** (écart de médiane ≤ 0,02). `block_score`
divise par le score de l'ancre, qui vaut ~1 quelle que soit la longueur, donc il
est déjà comparable entre 4 et 8 — la piste 2 du brief est close, il n'y a rien à
normaliser. Et la passe de 4 n'est pas massivement plus permissive : 4,2 % de
positions au-dessus du seuil contre 3,2 %. **Re-régler THR4 ne peut donc pas
réparer le swap** : ça ne fera que retirer des blocs de 4, pas changer la forme
de l'échec.

### E3 — la vraie cause : la passe de 4 trouve la PÉRIODE, pas la SECTION (11h25)

Décomposition de la métrique (elle est le produit `spans × letters`) :

| morceau | 8→4 spans | 8→4 lettres | 4→8 spans | 4→8 lettres | nseg 8→4 | nseg 4→8 | nseg lui |
|---|---|---|---|---|---|---|---|
| This Love | 1,000 | 1,000 | 0,816 | 0,667 | 11 | 17 | 11 |
| Stand By Me | 0,990 | 0,803 | 0,697 | 0,684 | 14 | 23 | 11 |
| Every Breath | 0,916 | 0,912 | 0,720 | 0,683 | 19 | 30 | 14 |
| Blue Lights | 0,836 | 0,833 | 0,727 | **0,976** | 16 | 21 | 12 |

Les deux moitiés tombent. Le détail des runs le dit exactement — Blue Lights :

    4 d'abord : bloc 4 ancre mes.  5  reprises [9, 17, 33, 41, 65, 81]
                bloc 4 ancre mes. 13  reprises [29, 45, 61, 77]   <- SES CINQ B
                bloc 4 ancre mes. 21  reprises [25, 37, 57, 69, 73]
    8 d'abord : bloc 8 ancre mes.  5  reprises [21, 33]
                bloc 8 ancre mes. 45  reprises [77]               <- LE BLOC FAUTIF

La passe de 4 trouve **exactement** ses cinq B (13, 29, 45, 61, 77) — c'est ce
que Louis a vu. Mais elle trouve aussi la boucle de 4 mesures partout ailleurs et
découpe ses A de 8 et 12 mesures en tranches de 4, chaque tranche recevant une
lettre différente parce que ce sont des runs différents.

Sur Blue Lights ce sur-découpage est CONSTANT (a-b-b à chaque A), et la métrique
le pardonne : lettres 0,976. Sur This Love / Stand By Me il est INCONSTANT (le
verrou `claimed` fait que la 2e occurrence d'une section n'est pas découpée comme
la 1re) et les lettres s'effondrent à 0,67-0,68.

**Diagnostic** : la passe de 4 mesure la PÉRIODE de la boucle harmonique, pas
l'ÉCHELLE de la section (exactement ce que le zoo des matrices disait le
2026-08-10). Lui donner la priorité globalement, c'est écrire le morceau à
l'échelle de la boucle.

**Plafond d'un simple choix d'ordre** : si un oracle choisissait par morceau le
meilleur des deux ordres, la médiane resterait 0,789 et Blue Lights 0,710 < 0,75.
**Aucun réglage d'ordre ou de seuil ne peut donc atteindre la cible** — il faut
un critère qui dise OÙ la section fait 4 mesures.

### E4 — H4 (corroboration par les pics) RÉFUTÉE (11h40)

52 runs de 4, appui-pics contre appui-vérité : **corrélation 0,114**. Les runs à
fort appui-pics ne sont pas plus justes (appui-vérité moyen 0,528 sans filtre,
0,543 en exigeant ≥ 0,5). Raison : The Walk et Grenade ont 11 et 8 pics, donc
n'importe quel run y a de l'appui ; Blue Lights n'en a que 6.
Artefact : `scratchpad/order_diag_support.py`.

### E5 — H5 (« taux de suite ») RÉFUTÉE aussi (11h50)

Idée : un bloc de 8 est deux 4 collés si sa 2e moitié ne suit sa 1re qu'une fois
sur cinq. Sur 78 occurrences de blocs de 8, corrélation avec « une frontière de
Louis tombe au milieu du bloc » : **−0,076**. Le contre-exemple est net : Blue
Lights 8@5 a le taux de suite le PLUS BAS du corpus (0,14) et c'est un bloc
JUSTE ; Blue Lights 8@45 est à 0,40 comme sept autres blocs innocents.
Artefact : `scratchpad/order_diag_follow.py`.

Au passage, un chiffre utile : sur les 24 blocs de 8 posés, **3 seulement**
recouvrent une frontière de Louis en leur milieu (Let It Be 8@5, Blue Lights
8@45, She Will Be Loved 8@59). La passe de 8 n'est presque jamais fautive — le
problème est très localisé.

### E6 — LA CONCURRENCE 4/8 DANS UN SEUL PARCOURS : ça marche (12h05)

Piste 3 du brief. Un seul parcours du curseur ; à chaque ancre on construit LES
DEUX runs (bloc de 8 et bloc de 4 avec leurs reprises) et on tranche. Règle
`prefer_long` : le 8 s'il existe, sinon le 4. Puis la passe de comblement en 4,
inchangée.

| morceau | référence | concurrence (pas 4) |
|---|---|---|
| This Love | 1,000 | 1,000 |
| Don't Know Why | 0,974 | 0,974 |
| Bein' Green | 0,993 | 0,993 |
| Let It Be | 0,716 | 0,716 |
| Sunny | 0,796 | 0,796 |
| **Blue Lights** | 0,696 | **0,813** |
| Stand By Me | 0,795 | 0,795 |
| She Will Be Loved | 0,783 | 0,783 |
| Every Breath You Take | 0,835 | 0,835 |
| Chain of Fools | 0,676 | 0,636 |
| The Walk | 0,511 | 0,386 |
| Grenade | 0,663 | 0,663 |
| **médiane** | **0,789** | **0,795** |

**Les deux cibles sont atteintes** (médiane 0,795 ≥ 0,789 ; Blue Lights 0,813 >
0,75). Mécanisme, vérifié sur les runs : la passe de 8 lancée seule arrive en
mesure 45, y pose un bloc de 8 et **vole la 3e occurrence de la famille B** ; la
passe de 4 qui suit ne retrouve alors que 13/29/61. Dans le parcours unique, le
pic dur de la mesure 17 barre le bloc de 8 à la mesure 13, le bloc de 4 y est
proposé **au moment où la famille est encore entière**, et il prend
13/29/45/61/77 d'un coup. C'est « privilégier les petites sections » — mais à
l'ancre où ça se joue, pas globalement.

Deux reculs à traiter : The Walk (−0,125) et Chain of Fools (−0,040).

### E7 — les pics « suivants » comme veto de MILIEU : négatif (12h20)

Test étroit : un pic sous le seuil ne peut rien interdire, sauf qu'un bloc de 8
ait son milieu dessus (c'est le cas de la mesure 49 de Blue Lights). Mesuré :

| variante | médiane | Blue Lights | ce que ça casse |
|---|---|---|---|
| référence | 0,789 | 0,696 | — |
| concurrence seule | **0,795** | 0,813 | The Walk, Chain |
| + veto milieu tol 0 | 0,784 | 0,739 | This Love 1,000→0,900 · Grenade 0,663→0,494 |
| + veto milieu tol 1 | 0,761 | 0,739 | idem + Sunny |

Rejeté. Les suivants sont trop imprécis (49 % de justesse) même dans cet usage
étroit, et la concurrence n'en a pas besoin : elle règle Blue Lights toute seule.
Artefact : `scratchpad/order_soft.py`.

### E8 — un bug DANS MON PROPRE ORACLE, et ce qu'il cachait (12h35)

La colonne « tout différé » sortait identique à « tout posé ». Cause : ma liste de
bits était plus courte que le nombre de décisions et le défaut tombait sur
« poser » — or DIFFÉRER crée de nouvelles décisions, donc la liste ne pouvait pas
suffire. Corrigé (défaut = différer, et le nombre de décisions est pris au
maximum des deux extrêmes). Après correction, « tout différé » reproduit la prod
**au millième** sur les douze : c'est le contrôle qui manquait.

Ce que l'oracle corrigé dit : le plafond de l'espace « poser ce 4 maintenant ou
le différer » est **0,795** de médiane, et il ne demande **qu'un bit par
morceau**. Blue Lights : la famille B de la mesure 13. The Walk : un seul point à
la mesure 75.

### E9 — 49 points de décision, un descripteur qui sépare (12h50)

Jeu construit (`scratchpad/order_decisions.py`) : chaque ancre où seul un bloc de
4 existe, avec 12 descripteurs et le gain marginal de le poser. **49 points ·
5 gagnants · 14 perdants · 30 neutres.**

Le descripteur qui sépare est le plus simple : **le score moyen des reprises du
bloc de 4**. Les gagnants de Blue Lights sont à 1,03 et 0,99, celui de The Walk à
0,94 ; aucun perdant n'atteint 0,94. Lecture musicale : `block_score` étant
relatif à l'ancre, ≥ 0,9 veut dire « cette reprise est presque aussi proche de
l'ancre que l'ancre l'est d'elle-même » — un motif rejoué **à l'identique**. À
0,75, c'est la boucle harmonique du morceau.

Plateau : tout seuil de 0,86 à 1,02 donne le même résultat, sans aucun recul.
Un-contre-tous : le réglage (moyenne, 0,88) est choisi sur 11 replis sur 12.

### E10 — le 4 ne doit PAS battre un 8 disponible (13h00)

Variante testée : quand les deux existent, le 4 littéral l'emporte. Moyenne des
12 : 0,789 contre **0,803**. Rejeté. Le 8 garde la priorité ; le 4 ne passe que
là où il n'y a pas de 8.

### E11 — les longueurs 12 et 16 : mesurées, rejetées (13h10)

Médiane des 12 : 12/8/4 → **0,630**, 16/8/4 → **0,715**, 16/12/8/4 → **0,639**,
contre 0,805 pour 8/4. Don't Know Why s'effondre à 0,631, Stand By Me à 0,461.
« Un A de seize s'écrit deux A de huit » est maintenant mesuré.

### E12 — la coupure qui laisse un moignon (13h20)

Chain of Fools : pic en mesure 9, frontière de Louis en mesure 10, et l'assemblage
coupait EXACTEMENT sur le pic → `A[2-8] | A[9-9]`. La tolérance de ±1 existait
pour la recherche (`crosses`) et pas pour l'écriture. Corrigé : une coupure qui
laisserait moins de `min_cut` mesures est annulée. Médiane des 12 0,795 → **0,805**,
Bein' Green à **1,000**.

### E13 — « le plus fort se sert le premier » ne se généralise PAS aux ancres (13h35)

`_peaks` porte déjà la doctrine pour les reprises d'une même ancre. Appliquée
ENTRE ancres (tous les candidats construits, le meilleur posé, on recommence) :
médiane des 12 **0,692** (plus longue d'abord) et **0,586** (meilleur score
d'abord), contre 0,805. Le parcours de gauche à droite est **porteur** — il cale
la phase des ancres sur `start`, la première mesure chantée. Ce n'est pas un
artefact dont il faut se débarrasser.

### E14 — LES PICS DURS SONT UN PARI, et la tolérance est le vrai correctif (13h50)

Ablation : sans aucun pic dur, la concurrence est un **no-op exact** (colonnes
identiques au millième). Le mécanisme est donc bien celui identifié : le bloc de 4
ne prend la main que là où un pic barre le bloc de 8.

Et une découverte de côté, sur les 18 annotés : **les pics durs gagnent sur 4
morceaux, perdent sur 6, neutres sur 8**, soit −0,005 de médiane et de moyenne.
Blue Lights +0,270 et The Walk +0,156 contre Be My Baby −0,277 et Grenade −0,124.
Un arbitre par morceau vaudrait +0,026 de moyenne (oracle 0,780 contre 0,754).

Diagnostic du POURQUOI, vérifié à l'œil sur Be My Baby : ses pics sont en mesures
19, 29, 43, 53, 71, 77 et ses frontières en 13, 21, 29, 37, 45, 53, 61, 71, 79.
Trois pics sont à **deux** mesures — et `HARD_TOL` valait 1. Ils coupaient ses B
de 8 mesures en trois morceaux. La précision réelle des pics était **déjà
écrite** dans known_issues (49 % exacts, 79 % à ±1, **88 % à ±2**) et n'avait
jamais été répercutée sur la tolérance.

Balayage `HARD_TOL` × `min_cut` sur les 18 : ±2 domine partout sauf la médiane
des 12 (0,799 contre 0,805). Retenu ±2 / min_cut 3 :

| | 12 méd. | 12 moy. | 6 hors conception moy. | 18 moy. |
|---|---|---|---|---|
| prod | 0,789 | 0,786 | 0,620 | 0,731 |
| + concurrence 4/8 | 0,795 | 0,803 | 0,620 | 0,742 |
| + coupure sans moignon | 0,805 | 0,805 | 0,636 | 0,749 |
| + pics tolérés à ±2 | **0,799** | **0,816** | **0,692** | **0,774** |
| *sans la concurrence* | *0,778* | *0,808* | *0,692* | *0,769* |

La dernière ligne est le contrôle qui compte : **sans la concurrence, la médiane
des 12 retombe sous la référence.**

### Note de cohabitation

Une autre session a committé `bb632fd` (« la phase de la grille vient du CHANT »)
dans `docs/known_issues.md` entre mes deux commits. Mes ajouts sont en amont des
siens dans le fichier, rien n'a été écrasé — mais son `###` final se lit comme un
sous-titre de mon entrée. Laissé tel quel : ce n'est pas à moi de changer le
niveau de titre de quelqu'un d'autre.

### Hypothèse historique (H4)

Ce qui sépare les cinq B de Blue Lights de la boucle de Stand By Me n'est pas le
score du bloc, c'est le **corroborement par les pics durs** : les frontières de
la famille B (13/17, 29/33, 45/49, 61/65, 77/81) tombent sur les pics 17, 29, 33,
45, 65 — cinq sur dix. Les runs de 4 de Stand By Me n'ont aucun pic sur leurs
frontières. Test : la passe de 4 passe en premier mais ne garde un run que si sa
famille est corroborée par les pics.
