# Audit — « une fois les sections trouvées, qu'est-ce qui est vraiment fusionné ? »

Question de Louis, 2026-08-08 : « Quand on a détecté des sections, qu'est-ce
qui se passe en termes de merging / robustesse ? On a bien musx qui lit les
chroma mergés pour chaque accord ? »

Audit de code + traces réelles. Tout ce qui suit est vérifié en exécutant le
chemin livré, pas lu dans une docstring.

---

## 1. La réponse

**Oui, mais pas des chroma, et pas partout.** Ce qui est moyenné entre
occurrences, ce sont les **postérieures de frames de musx** (les 6 flux bruts),
pas les chroma NNLS ; et **seulement 29 % des lettres** franchissent les
garde-fous — pour les autres, aucune évidence n'est fusionnée et le « ×N »
affiché est purement cosmétique.

Sur les 30 charts de la bibliothèque, les accords affichés se répartissent ainsi :

| ce que l'accord affiché est vraiment | part |
|---|---|
| décodé sur des postérieures musx **moyennées** entre occurrences | **16 %** |
| affiché « joué ×N » mais décodé sur **une seule** occurrence | **68 %** |
| section jouée une seule fois (rien à fusionner) | 16 % |

Les 68 % du milieu sont le vrai sujet : le chart écrit un bloc une fois et dit
qu'il se répète N fois, alors que ses accords viennent uniquement du **premier**
passage.

---

## 2. La chaîne réelle, maillon par maillon

| # | quoi | où |
|---|---|---|
| 1 | postérieures musx par frame (23 ms), 6 flux : `triad(73) bass(13) s7 s9 s11 s13` | `harmonia_min/pipeline.py:264` → `musx.py:168` |
| 2 | 1er décodage Viterbi sur la grille de temps → accords bruts | `pipeline.py:287` → `musx.py:328` |
| 3 | mise en mesures, un accord par temps | `pipeline.py:299-433` |
| 4 | **chart brut rendu ici** (`yield "raw"`) — aucune section, aucun repli | `pipeline.py:468` |
| 5 | détection des sections (mode `voice` par défaut) | `pipeline.py:498` → `sections.py:203` |
| 6 | **repli n°1 — la SEULE fusion d'évidence** | `pipeline.py:517` → `folding.py:114` |
| 7 | repli n°2 = **affichage seul**, un bloc par lettre | `pipeline.py:533` → `folding.py:468` |
| 8 | analyse harmonique (couleurs, tonalité) : chroma bruts, **par accord, sans fusion** | `pipeline.py:545` ; `harmonic_key.py:21` dit explicitement « NO structure fold » |
| 9 | suggestions musx (éditeur d'annotation) : postérieures brutes, **par accord, sans fusion** | `span_rescore.py:206` |

### Le maillon 6 en détail (`folding.fold_letter_groups`)

1. **Période interne** de chaque section, cherchée dans {2, 4, 8} mesures sur
   les chroma NNLS demi-mesure (`folding.py:81`, seuil 0.80). Échec ⇒ la lettre
   ne replie pas.
2. **Empilement** : la mesure `b` va à la position `(b - b0) % P`, `b0` étant le
   début de **sa propre** occurrence (`folding.py:167`). Une position rassemble
   donc les mesures homologues de **toutes** les occurrences de la lettre.
3. **Trois garde-fous**, tous sur les **chroma**, pas sur musx :
   - membre aberrant (z > 3 sur médiane+MAD des écarts au centroïde) ⇒ écarté
     comme « variante », garde son 1er décodage (`folding.py:181`) ;
   - cohérence du paquet (cos médian par paires ≥ 0.85) ⇒ sinon la lettre ne
     replie pas (`folding.py:197`) ;
   - vérificateur CV (std/mean des chroma bruts demi-mesure ≤ 0.51) ⇒ la
     position concernée n'est pas écrasée (`folding.py:206-219`).
4. **Moyenne** : les blocs de postérieures musx de chaque mesure membre sont
   ré-échantillonnés à une longueur commune puis **moyennés arithmétiquement**
   (`folding.py:259`). Les **six** flux, flux de basse compris.
5. **2e décodage Viterbi** sur le gabarit de P mesures, pavé ×3 contre les
   effets de bord (`folding.py:279`).
6. **Réécriture** : les accords du gabarit sont écrits sur **chaque** mesure
   contributrice (`folding.py:239` → `_write_position`), avec `folded: true`,
   `n_obs` (profondeur de la pile) et une confiance `c` mesurée sur les
   postérieures moyennées.

**Granularité** : la moyenne se fait **par mesure** (les frames d'une mesure
sont ré-échantillonnées vers une longueur fixe puis moyennées frame à frame).
Le décodage du gabarit, lui, autorise un changement d'accord **à chaque temps**
(coût gradué : mesure 15 / demi-mesure 15 / autre temps 100).

### Profondeur réelle des piles (traces)

```
This Love     B : P=2, obs/pos [18, 15]   -> 18 mesures moyennées par position
Hot n Cold    A : P=4, obs/pos [14,12,12,14]
Lazy Song     A : P=2, obs/pos [19, 27]
Sam Smith     A : P=2, obs/pos [20, 22]
```

Quand ça marche, ça marche fort : jusqu'à 27 observations pour une position.

---

## 3. Ce qui est fusionné, ce qui ne l'est pas

| | fusionné entre occurrences ? |
|---|---|
| postérieures musx (triad, s7, s9, s11, s13) | **oui**, moyenne arithmétique, si la lettre replie |
| postérieure musx de **basse** (flux 1) | **oui** — voir §4, point c |
| chroma NNLS | **non** — servent uniquement à décider *si* on a le droit de replier |
| couleurs / tonalité (`harmonic_key`) | **non**, par accord |
| suggestions de l'éditeur (`musx_suggestions`) | **non**, par accord |
| affichage « ×N » | **oui**, toujours — même quand rien n'a été fusionné |

### Combien de lettres replient vraiment

Sur les 30 charts en cache (`harmonia_min/state/charts/`) :

```
136 lettres, 39 replient (29 %)
motifs de refus : « no confident loop » 76 · « stack incoherent » 18 · « template decoded empty » 3
2 629 mesures au total, 169 (6,4 %) reçoivent un accord DIFFÉRENT du 1er passage
```

Attention à ne pas confondre deux chiffres : le repli **réécrit** beaucoup plus
de mesures qu'il n'en **change**. Sur 6 morceaux tracés, 288 mesures sont
réécrites par le gabarit ; seules ~60 changent d'étiquette. Les autres
confirment le 1er passage (et gagnent une confiance mieux fondée).

---

## 4. Fragilités, mesurées

### a) Une erreur de section d'UNE mesure contamine les occurrences saines

Protocole : décaler **une seule** occurrence de ±1 mesure (ou changer sa
longueur de ±1), tout le reste identique, et compter les mesures du chart final
qui changent d'accord.

**264 perturbations sur 7 morceaux : en moyenne 2,41 mesures cassées, dont 1,72
(71 %) tombent HORS de la section déplacée** — c'est-à-dire dans des
occurrences correctement segmentées.

| morceau | mesures cassées (moy.) | dont hors section déplacée | pire cas |
|---|---|---|---|
| Sam Smith — I'm Not the Only One | 6,19 | 5,72 | 21 mesures |
| Let It Be | 3,81 | 3,03 | 10 |
| Lazy Song | 3,45 | 2,05 | 10 |
| This Love | 2,69 | 0,94 | 6 |
| Hot n Cold | 0,88 | 0,54 | 4 |
| The Walk | 0,14 | 0,00 | 1 |
| Grenade | 0,00 | 0,00 | 0 |

Le cas concret, This Love, occurrence n°2 de B décalée de +1 mesure :

```
mesure   dans l'occurrence déplacée ?   1er passage   repli (juste)   repli (1 mes. de trop)
  36     oui                            C- F-         F-              C- F-
  42     oui                            C- F          F-              C- F
  62     NON — un autre passage         C- F7         C- F7           F-
  70     NON — un autre passage         C-7 F         C-7 F           F-
```

Les mesures 62 et 70 appartiennent à des occurrences de B qui n'ont **pas
bougé**. Elles perdent quand même leur `C-` parce que la mauvaise occurrence a
pollué la moyenne de la position. **Le repli transforme donc bien une erreur de
section en erreurs d'accord, y compris là où la section était juste.**

Le mécanisme est structurel : décaler une occurrence de 1 mesure fait tourner
**toutes** ses mesures d'une position modulo P. La garde de cohérence
(cos médian par paires) attrape ça quand la pile est mince (2-3 membres, où la
médiane *est* la paire fautive), mais pas quand elle est épaisse : sur 18
membres, une occurrence tournée reste minoritaire, la médiane tient, la pile
passe — et la moyenne est empoisonnée. Ce sont exactement les morceaux à piles
profondes (Sam Smith, Let It Be) qui souffrent le plus.

### b) Longueurs inégales : traitées silencieusement, jamais signalées

Les positions sont calculées `(b - b0) % P` **par occurrence**. Deux occurrences
de longueurs différentes s'empilent donc en phase par leur début, et les mesures
excédentaires de la plus longue retombent sur les positions 0, 1, … — la queue
se mélange à la tête. Rien ne le signale.

Bonne nouvelle mesurée : allonger une occurrence de +2 mesures n'a changé **0**
mesure sur This Love et Hot n Cold. Une pile de 12-18 membres absorbe un ou deux
intrus. Le risque est donc réel mais second par rapport au (a).

### c) Le flux de BASSE est moyenné ici, alors que l'ancien pipeline l'excluait

`_template_chords` moyenne `for i in range(n_probs)` — les six flux, basse
comprise (`folding.py:259`). L'implémentation historique
`harmonia/models/musx_posterior_fold.py` l'excluait explicitement
(`FOLDED_STREAMS == (0, 2, 3, 4, 5)`), épinglé par
`tests/test_musx_posterior_fold.py::test_bass_stream_is_never_folded`, au motif
qu'une occurrence peut se jouer sur un autre renversement. Vu que la cible du
projet est la **basse sonnante** (`sounding_bass_pc`), ça mérite d'être su.

Exposition mesurée : sur 288 mesures repliées (6 morceaux), **5 (2 %)** voient
leur basse/renversement réécrit par le gabarit. Petit, mais non nul, et
aujourd'hui non intentionnel — personne n'a décidé de replier la basse.

### d) Le vérificateur CV refuse de fusionner… et l'affichage replie quand même

Quand le CV d'une position dépasse 0.51, `fold_letter_groups` s'abstient de
l'écraser : « every occurrence keeps its own decode » (`folding.py:204`). Mais
`minimal_fold` ne lit pas cette information — elle n'est même pas dans le
rapport — et écrit le bloc **une fois ×N** de toute façon, en prenant les
mesures de la **première** occurrence (`folding.py:504`, `:576`).

Mesuré sur 10 morceaux, 5 sont concernés :

```
Sam Smith   A : P=2, ×8, positions refusées [1]     -> affichées quand même ×8
Grenade     A : P=4, ×5, positions refusées [3]     -> ×5
Norah Jones A : P=4, ×5, positions refusées [0, 1]  -> ×5
Norah Jones B : P=2, ×2, positions refusées [1]     -> ×2
Sunny       A : P=4, ×2, positions refusées [3]     -> ×2
```

Le module dit « under-fold, never over-fold » ; ici il fait l'inverse : il
constate que les occurrences divergent trop pour être moyennées, puis les
affiche comme identiques. C'est le défaut le plus facile à corriger des quatre.

---

## 5. Ce que la fusion rapporterait si on l'étendait

Elle existe déjà — elle est juste refusée dans 71 % des cas, et le refus est
prononcé par les **chroma**, pas par musx.

Sur les 68 % d'accords affichés « ×N » sans aucune fusion, l'ordre de grandeur
disponible est le `n_obs` de leur lettre : 2 à 27 observations. Moyenner N
observations divise l'écart-type du bruit **indépendant** par √N — mais le bruit
de musx sur deux passages du même refrain n'est pas indépendant (même modèle,
même timbre, même mixage), donc √N est un plafond, pas une prévision. Sur les
lettres qui replient déjà, l'effet observé est modeste : 6,4 % des mesures
changent d'étiquette.

Le gain le plus sûr n'est donc pas « replier plus », c'est **replier plus
sûrement** — voir les quatre correctifs ci-dessous. **Rien n'a été implémenté.**

---

## 6. Écarts entre le code et sa documentation

1. **`folding.display_fold` est du code mort.** Sa docstring décrit en détail un
   « CROSS-PASS observation stacking » (moyenne des passages puis re-décodage,
   `folding.py:383-422`). Aucun appelant : `pipeline.py:533` appelle
   `minimal_fold`. Vérifié par recherche sur tout le dépôt.
2. **L'en-tête de `folding.py` dit « This module is (1) only »** (empilement
   seulement, « display synthesis … NOT built yet »). Le module contient
   aujourd'hui deux phases d'affichage (`display_fold`, `minimal_fold`), et
   c'est la seconde qui est livrée.
3. **Le rapport de repli n'enregistre pas `cv_skip`.** Les positions refusées
   par le vérificateur CV sont journalisées puis perdues ; aucun consommateur —
   `minimal_fold` en particulier — ne peut en tenir compte. Voir §4(d).
4. **Attention aux tests trompeurs** : `tests/test_musx_posterior_fold.py` et
   `tests/test_section_fold_unequal.py` testent l'**ancien** dépôt
   (`harmonia.models.musx_posterior_fold`, `harmonia.output.chart_display`),
   pas `harmonia_min`. Ils décrivent un repli qui n'est pas celui de l'app.
   Le repli livré n'a qu'un test : `tests/test_folding_confidence.py`.

---

## 7. Correctifs proposés, par rapport bénéfice/risque

1. **§4(d)** — mettre `cv_skip` dans le rapport et faire écrire à `minimal_fold`
   les mesures en clair quand une position a été refusée. Petit, local, aligné
   sur la doctrine « under-fold ».
2. **§4(a)** — garde de **rotation** : avant d'empiler, tester chaque occurrence
   contre le centroïde pour les P décalages possibles ; si un décalage non nul
   gagne nettement, c'est la section qui est fausse — exclure l'occurrence (ou
   la recaler) plutôt que polluer la moyenne. C'est le levier qui vaut le plus :
   71 % des dégâts mesurés viennent de là.
3. **§4(c)** — décider explicitement du flux de basse (le replier ou non), et
   épingler la décision par un test, comme l'ancien module le faisait.
4. **§6(1)** — supprimer `display_fold` ou dire en une ligne qu'elle est morte.

## Reproduire

Scripts de l'audit (scratchpad de session, non versionnés) :
`trace_fold.py` (rejoue les étapes 1-6 du pipeline puis le repli),
`trace_fold2.py` (nomme les accords contaminés), `fold_sweep.py` (le balayage
des 264 perturbations), `cv_probe.py`, `bass_probe.py`. Ils lisent les caches
`data/cache/musx_probs/`, `data/cache/nnls_infer/` et les portées de sections
des charts en cache, donc ils ne relancent ni demucs ni la détection.
