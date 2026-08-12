# « Petites sections d'abord » — journal de session (2026-08-12)

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

### Hypothèse historique (H4)

Ce qui sépare les cinq B de Blue Lights de la boucle de Stand By Me n'est pas le
score du bloc, c'est le **corroborement par les pics durs** : les frontières de
la famille B (13/17, 29/33, 45/49, 61/65, 77/81) tombent sur les pics 17, 29, 33,
45, 65 — cinq sur dix. Les runs de 4 de Stand By Me n'ont aucun pic sur leurs
frontières. Test : la passe de 4 passe en premier mais ne garde un run que si sa
famille est corroborée par les pics.
