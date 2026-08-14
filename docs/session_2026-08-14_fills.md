# Les fills de batterie comme marqueurs de section — session 2026-08-14

`scripts/fills.py` · page écoutable `/plots/fills.html` · 90 min, 12:10 → 13:40.

## En une phrase

**Le fill de batterie est un marqueur réel mais faible, et l'hypothèse de départ
est fausse dans son détail** : la meilleure variante marque **50 % des
frontières annotées** (hasard au même budget : **36 %**), et **multiplier
l'anomalie par le « retour au groove » en b+1 fait PERDRE 4 points** au lieu
d'en gagner. Le critère d'arrêt (≥ 60 % à moins d'un pic toutes les 8 mesures,
écart-type ≤ 1 mesure) **n'est pas atteint**. Rien n'est à brancher en l'état.

## Le dispositif

- **Substrat** : batterie séparée (demucs, déjà en cache pour les 18 morceaux)
  → enveloppe d'attaques en **4 bandes** (grosse caisse < 130 Hz · caisse
  claire/toms 130–1200 · médium-aigu 1200–5500 · cymbales > 5500), hop 256 à
  22 050 Hz = 86 trames/s, soit ~11 trames par double-croche à 120 bpm.
  Cache `scratchpad/fills_cache/*.pkl`.
- **Corpus** : les 18 morceaux validés, **1 540 mesures, 197 frontières**.
- **Budget commun** : chaque variante propose exactement `n/8` pics, espacés
  d'≥ 2 mesures. Sans ce budget, une courbe bruitée « trouve » tout.
- **Métrique** : rappel (part des frontières marquées à ±1 mesure). Jamais
  « faux positif » — les annotations ne sont pas exhaustives ; le coût est la
  densité de pics, fixée d'avance.
- **Position** : fractionnaire, en mesures, du début à la fin. Jamais « la case
  qui compte le pic » (le piège à 41 points de `ancres.py`).
- **Convention de décalage** : la courbe est indexée par la case où le
  phénomène se produit, jamais par la frontière qu'on en déduit. Un fill en b
  donne un pic en b, donc un écart de −1 avec la frontière b+1. Rien n'est
  décalé en douce.

**Le plancher, à lire avant tout chiffre.** Un tirage uniforme au même budget
marque déjà **36 %** des frontières (une tolérance de ±1 mesure couvre 3 mesures
sur 8). Toute variante en dessous de 40 % ne dit rien du tout.

## Le tableau des variantes (18 morceaux, granularité mesure, fenêtre 2)

| variante | rappel | densité | σ des écarts | ±½ mes. |
|---|---|---|---|---|
| damier batterie (**témoin**, un des 39) | **51 %** | 8,1 | 1,06 | 22 % |
| **anomalie centrée** | **50 %** | 8,1 | 0,99 | 24 % |
| densité × retour | 49 % | 8,1 | 1,11 | 18 % |
| anomalie seule | 48 % | 8,1 | 1,00 | 19 % |
| **anomalie × retour** (la formule du brief) | 46 % | 8,1 | 1,06 | 20 % |
| sortie de grille | 45 % | 8,1 | 1,22 | 15 % |
| retour seul | 44 % | 8,1 | 1,11 | 16 % |
| déplacement de bande | 43 % (47 % à W=4) | 8,1 | 1,12 | 18 % |
| densité d'attaques | 41 % | 8,1 | 1,31 | 16 % |
| médium / (kick+cymbales) | 41 % | 8,1 | 1,15 | 17 % |
| fill puis crash | 37 % | 9,4 | 1,17 | 13 % |
| break (trou × retour) | 34 % | 8,1 | 1,31 | 11 % |
| **crash sur le temps 1** | 34 % | 9,0 | 1,29 | 17 % |
| *hasard au même budget* | *36 %* | 8,1 | — | — |

Granularité et fenêtre, toutes variantes confondues :

| granularité | meilleur rappel | commentaire |
|---|---|---|
| **mesure** | **51 %** | le bon grain |
| demi-mesure | 44 % | −7 points, mais ±½ mesure passe de 24 % à 28 % |
| temps | 36 % | au niveau du hasard : le pic-picking se disperse |
| fenêtre W=2 | 50 % | légèrement meilleure que 4, nettement que 8 |
| fenêtre W=8 | 45 % | la « norme locale » sur 8 mesures avale les sections courtes |

## Les cinq résultats, dans l'ordre d'importance

**1. « Anomalie × retour » est moins bon qu'« anomalie seule » (46 % contre
48 %, 44 % contre 50 % en centré).** L'hypothèse du brief est contredite. Le
mécanisme, lu sur le profil de rang (`fills.py --profil`, rang moyen de la
courbe à frontière + décalage, 0,50 = rien) :

| décalage (mesures) | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| anomalie seule | 0,44 | 0,40 | 0,55 | **0,59** | 0,55 | 0,46 | 0,54 |
| anomalie × retour | 0,53 | 0,40 | 0,49 | 0,53 | **0,59** | 0,46 | 0,54 |
| damier batterie | 0,48 | 0,49 | 0,57 | **0,60** | 0,55 | 0,48 | 0,51 |

Le facteur « retour » **déplace le pic de 0 à +1** au lieu de le déplacer de −1
à 0. Pourquoi : `retour[b] = prox(b+1) − prox(b)` est grand quand la mesure b
est anormale ET la suivante normale — or **la mesure anormale, ce n'est pas
celle du fill, c'est la PREMIÈRE mesure de la nouvelle section**. Elle est
anormale par rapport aux 2–4 mesures d'avant (nouveau groove), et la deuxième
mesure de la section la « confirme » donc paraît un retour. Le facteur mesure
donc l'installation du nouveau groove, pas la résolution d'un fill.

**2. Le décalage n'est pas −1 : il est 0, et il est centré.** L'histogramme de
tous les couples (pic, frontière) à ≤ 3 mesures, pour la variante retenue :

| écart | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| paires | 16 | 14 | 34 | **47** | 21 | 15 | 26 |

Excès net sur un plancher plat (~25/case) : **+22 en 0, +9 en −1, −4 en +1**.
Donc oui, il y a un biais en avance, mais il est **secondaire** : la masse est
sur la frontière elle-même. Un décalage constant −1 appliqué en dur ferait
perdre, pas gagner. C'est mesuré : la correction du décalage en
*leave-one-song-out* (offset élu sur 17 morceaux, appliqué au 18e) donne
**exactement le même rappel** (50 % → 50 %) — il n'y a pas de décalage global à
corriger.

**Attention à l'estimateur.** La « médiane de l'écart au pic le plus proche »
vaut +0,00 pour absolument toutes les variantes : à densité 1 pic / 8 mesures
il y a presque toujours exactement un pic dans la fenêtre ±2,5, tiré presque
uniformément, donc son écart médian vaut 0 quoi qu'il arrive. **C'est un
artefact d'estimateur, pas une mesure de décalage.** Seul l'histogramme complet
(toutes les paires, plancher plat) dit quelque chose. Piège à noter pour la
prochaine session.

**3. Le centrage du motif vaut +2 à +6 points, et c'est le seul vrai gain de la
session.** Sur une enveloppe d'attaques toujours positive, le cosinus est saturé
par le socle commun à toutes les mesures (charleston sur chaque croche, kick sur
1 et 3) : deux mesures très différentes restent à cos ≈ 0,95. Retirer le motif
MOYEN DU MORCEAU avant de comparer fait passer l'anomalie de 48 % à 50 %, et le
damier de batterie **de 44 % à 51 %** (86 → 100 frontières marquées sur 197,
même budget, mesuré à part). Même diagnostic que
`rhythm_ssm._mean_center_renormalise` et que le plancher DC du cosinus de
chroma : **c'est la troisième fois que ce défaut coûte quelque chose dans ce
projet.**

**4. Aucun détecteur de fill ne bat le damier de Foote sur la batterie.** Le
témoin (51 %) est devant les neuf variantes de fill. Autrement dit : ce que la
batterie apporte à la segmentation, **c'est le changement de groove, pas le
fill**. Le fill est un cas particulier de changement de groove, et le mesurer
spécifiquement n'ajoute rien — au mieux ça égalise.

**5. Le crash sur le temps 1 est le plus mauvais de tous (34 %, au niveau du
hasard).** C'était la variante la plus spécifique et la seule qui donnait la
position exacte. Deux causes vues dans les données : (a) beaucoup de sections
commencent sans crash (les couplets, les fins de refrain) ; (b) les crashes
existants sont noyés — la bande > 5 500 Hz d'un stem demucs contient surtout le
charleston, présent en permanence, et le z-score par phase ne suffit pas à
l'isoler. Un vrai détecteur de crash demanderait une détection de transitoire
large bande avec décroissance longue, pas un maximum d'enveloppe.

## Ce que ça donne morceau par morceau (variante retenue)

L'hypothèse de Louis — chaque chanson a son propre jeu de signaux — tient, et
plus fort qu'ailleurs : **de 30 % à 73 % selon le morceau**.

| morceau | ftr | marquées | rappel |
|---|---|---|---|
| Bruno Mars — The Lazy Song | 11 | 8 | **73 %** |
| Bruno Mars — Grenade | 10 | 7 | **70 %** |
| Maroon 5 — This Love | 10 | 7 | **70 %** |
| Let It Be | 13 | 9 | **69 %** (et **0 pic hors annotation**) |
| Jorja Smith — Blue Lights | 11 | 7 | 64 % |
| The Ronettes — Be My Baby | 10 | 6 | 60 % |
| Pharrell — Happy | 18 | 9 | 50 % |
| Mayer Hawthorne — The Walk | 10 | 5 | 50 % |
| Sunny · Chain of Fools · ABC · Yesterday | | | 44–45 % |
| Every Breath You Take · She Will Be Loved | | | 35–38 % |
| Goodbye Yellow Brick Road · Bein' Green | | | 33 % |
| Stand By Me · Don't Know Why | 10 · 10 | 3 · 3 | **30 %** |

Lecture : **ça marche là où la batterie est franche et jouée par un batteur**
(Lazy Song, Grenade, This Love, Let It Be : 69–73 %), et ça tombe au hasard sur
les morceaux à batterie discrète ou à boîte à rythmes rigide (Stand By Me, Don't
Know Why : 30 %, sous le hasard). **La variante devrait donc être conditionnée
au morceau, pas appliquée partout** — exactement comme les 39 critères.

## Les pics hors annotation à écouter (ce ne sont pas des erreurs)

Boutons directs sur `/plots/fills.html` :

| morceau | mesure | timecode | pourquoi c'est intéressant |
|---|---|---|---|
| Happy | 95 | **2:25** | le pic le plus fort du morceau qui ne tombe sur aucune de tes frontières |
| Happy | 126 | **3:11** | même hauteur, dans la seconde moitié — deux changements non notés ? |
| Be My Baby | 63 | **1:56** | Hal Blaine ; si c'est un fill, il est massif et non annoté |
| This Love | 31 | 1:16 | au milieu du morceau, hors de tes 10 frontières |

Et le résultat inverse, qui vaut autant : **Let It Be n'a aucun pic hors
annotation** — ses 9 propositions tombent toutes sur (ou à côté de) tes 13
frontières.

## Ce qui ne marche pas, et pourquoi (pour ne pas le refaire)

| tentative | résultat | cause diagnostiquée |
|---|---|---|
| anomalie **× retour** en b+1 | −2 à −6 pts | le facteur capte l'installation du NOUVEAU groove (pic déplacé en +1), pas la résolution d'un fill |
| crash sur le temps 1, z-scoré | 34 % (= hasard) | la bande aiguë du stem est du charleston permanent ; beaucoup de sections n'ont pas de crash |
| break (trou × retour) | 34 % | trop rare — sur 1 540 mesures, presque aucune n'est un vrai break ; les pics sont des baisses de volume |
| densité d'attaques seule | 41 % | le charleston domine le compte : un fill de caisse claire ne change pas le nombre d'attaques, il change leur RÉPARTITION |
| sortie de grille | 45 % | correct, mais le grain de l'enveloppe (11 trames par double-croche) ne sépare pas triolets et doubles |
| granularité au temps | 36 % (= hasard) | le signal EXISTE au temps −1 (rang 0,60, le meilleur de tout le tableau) mais le pic-picking à budget fixe le disperse : le pic va au mauvais temps de la bonne mesure |
| fenêtre de référence à 8 mesures | −5 pts | une section dure 8 mesures : la norme locale contient déjà la section suivante |

**Le point le plus prometteur non exploité** : au grain du **temps**, l'anomalie
a son rang maximal à **−1 temps** (0,60), c'est-à-dire **sur le dernier temps de
la mesure d'avant** — la signature exacte d'un fill. Le signal est là ; c'est
l'extraction de pic qui le perd. La bonne construction est probablement :
détecter au temps, **agréger à la mesure** (max des 4 temps), et garder la
position fine du temps pour le décalage. Non testé faute de temps.

## Recommandation

1. **Ne rien brancher en prod.** 50 % contre 36 % de hasard, avec un écart-type
   d'écart de 1,0 mesure, ce n'est pas un marqueur exploitable seul.
2. **Ce qui est branchable tout de suite, en revanche** : le **centrage du
   motif de batterie** (retirer le motif moyen du morceau avant tout cosinus).
   Il fait passer le damier de batterie — qui, lui, est déjà l'un des 39
   critères et l'un des 7 signaux du vote de `ancres.py` — de 44 % à 51 %.
   C'est un changement d'une ligne dans `criteres_sections.signaux`, à mesurer
   sur la précision des ancres avant de le poser.
3. **La suite à tester** (par ordre de rendement attendu) : (a) anomalie au
   temps agrégée à la mesure par le maximum, qui est le seul endroit où le
   signal atteint 0,60 ; (b) restreindre le détecteur aux morceaux à batterie
   franche, mesuré par la stabilité du motif (les morceaux à 70 % sont ceux dont
   le motif moyen explique le plus de variance) ; (c) le fill comme **vote**
   avec les 6 autres signaux plutôt que comme détecteur autonome — c'est déjà
   ce que fait `ancres.py`, et le signal « rythme » y est le plus faible des 7.

## Journal

- **12:10** départ. Contexte lu : les deux entrées du 2026-08-14
  (`ancres.py`, `criteres_sections.py`), `rhythm_ssm.py`, `resume_texte.py`.
- **12:35** substrat en cache pour les 18 (4 bandes, hop 256).
- **12:50** premier passage : toutes les variantes entre 34 % et 47 %, hasard à
  36 %. Deux défauts trouvés à l'inspection : le damier de référence avait le
  **signe inversé** (il pointait à l'opposé de ce qu'il détecte) et le cosinus
  était **saturé par le socle du groove**.
- **13:05** correction des deux : le damier passe de 44 % à 51 %, l'anomalie
  centrée à 50 %. Le facteur « retour » reste négatif dans tous les réglages.
- **13:20** page écoutable (4 morceaux, 9 lignes, boutons 2 mesures avant → 2
  après). Extraction des pics hors annotation.
- **13:35** rédaction. Critère d'arrêt non atteint, résultat négatif caractérisé.
