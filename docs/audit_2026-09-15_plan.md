# Audit des bugs musicaux — plan et premières trouvailles (2026-09-15)

## Pourquoi cet audit

Louis, après le refactor : « I didn't want you to have 44/44 identical, you
could have changed some charts I would've been fine with it. » Le refactor a
préservé le comportement octet pour octet — donc il a préservé AUSSI toutes
les erreurs musicales. Cet audit cherche ces erreurs-là, par le test, pas par
la lecture.

**Ce que le rapport d'or ne pouvait PAS voir.** Il compare le nouveau code au
gelé, mais les DEUX lisent les mêmes caches chauds (battues, postérieures
musx, chroma NNLS). Tout ce qui est faux DANS ces caches est invisible pour
lui. C'est la première chose à tester.

## Trouvailles déjà mesurées (scan des 46 charts, coût nul)

### 1. La compensation de latence ne fait rien sur 44 morceaux sur 46 — SUSPECT

`musx.redecode` essaie 0→280 ms par pas de 40 et garde l'offset qui maximise
la log-vraisemblance du chemin (aucune vérité terrain requise).

| latence choisie | morceaux |
|---|---|
| 0 ms | 44 |
| 120 ms | 1 |
| 280 ms (borne haute, la recherche sature) | 1 (`min_dOQXg6rK86I`) |

Le principe même de l'étape, c'est que « le modèle entend les changements en
retard » — un effet mesuré, documenté. Si 44 morceaux sur 46 choisissent zéro
compensation, deux hypothèses :

- **H1** : l'objectif est plat ou dégénéré — la recherche ne distingue rien et
  rend le PREMIER candidat, qui est 0. Forme classique du bug de départage
  (`>` au lieu de `>=`, ou l'inverse).
- **H2** : la latence est réellement nulle sur ce corpus, et l'étape est
  inutile (mais alors pourquoi un morceau sature-t-il à 280 ?).

**Test** : instrumenter la recherche sur 3 morceaux, imprimer la
log-vraisemblance des 8 offsets candidats. Si les 8 valeurs sont identiques ou
quasi, c'est H1. Si elles forment une vraie courbe avec un maximum à 0, c'est
H2. Vérifier aussi le sens de la comparaison dans l'argmax.

### 2. Stand By Me : 50 % de N.C. — SUSPECT

`min_ben_e_king_stand_by_me_audio` : 11 entrées « pas d'accord » sur 22.
C'est une boucle de quatre accords du début à la fin. Historique connu : ce
morceau affichait 93 % de N.C. en août (un N.C. propagé sur onze mesures) ;
corrigé en ne propageant plus les N.C. 50 %, c'est mieux, mais ça reste
énorme. La mémoire du projet dit aussi « Stand By Me has a real drift ».
C'est un des cinq morceaux de l'oreille de Louis.

**Test** : rendre la page avant/après, écouter les mesures N.C., et regarder
si les postérieures musx brutes voyaient un accord là où le chart écrit N.C.
(c'est-à-dire : est-ce le modèle qui n'entend rien, ou notre seuil qui jette ?)

### 3. Ce qui va bien (mesuré, pas supposé)

- **Densité d'accords** : médiane 0,48 accord par mesure, maximum 2,5 sur un
  extrait de piano jazz. L'ancienne catastrophe « 20 accords par mesure » n'est
  plus là.
- **Invariant de la cascade iReal** : 0 chart sur 46 porte deux blocs sous la
  même étiquette. Le travail de la session concurrente tient.

## Le plan de test, par ordre de valeur

### A. Les caches sont-ils vrais ? (le plus important)

**A1. Recalcul à froid contre cache chaud.** Prendre 3 morceaux, recalculer
battues + postérieures + chroma SANS cache, comparer aux tableaux en cache.
S'ils diffèrent, le « 44/44 identiques » ne validait que le code en aval des
caches, et les charts reposent sur des features périmées. *C'est le test qui
dit si tout le reste de l'audit a un sol.*

**A2. La clé de cache ne couvre pas le code.** `<nom>__<taille>` identifie le
fichier audio, pas la version du code ni les constantes qui ont produit la
valeur. Tester : quelles étapes sont réellement mises en cache (features) vs
recalculées (décodage, repli) — donc quel changement de constante serait
masqué et lequel ne le serait pas.

### B. La grille (tout en dépend)

**B1. Octave de tempo.** Scanner les 46 : BPM détecté, battues par mesure,
qualité de grille. Repérer les morceaux où `check_grid` est marginal.
Croiser avec un BPM connu (iReal pour les standards) là où c'est possible.

**B2. Phase de mesure contre l'oreille de Louis.** Le vote de phase peut
décaler toute la grille. Les morceaux où Louis a posé une marque « Set bar 1 »
à la main sont exactement ceux où il n'était PAS d'accord avec la machine.
Croiser : sur quels morceaux le re-calage harmonique s'est-il déclenché, et
est-ce que ce sont les mêmes ?

**B3. Levées jetées.** Le code « déverse » les levées coincées dans la mesure 0
et le journalise. Scanner les logs : quels morceaux ont perdu des onsets réels ?

### C. Les accords

**C1. Latence** (voir trouvaille 1 ci-dessus).

**C2. N.C.** (voir trouvaille 2 ci-dessus) — et en général : pour chaque
mesure N.C., que disait la postérieure musx brute ?

**C3. Qualités d'accord.** Comparer les qualités écrites (maj7/m7/7…) aux
grilles iReal pour les standards de la bibliothèque. Trust order : iReal >
tablatures > modèle.

### D. Sections et repli

**D1. Frontières contre les 21 découpages validés à la main.** C'est la seule
vraie vérité terrain du projet. `tools/sections_bench/` a été construit pour
ça mais ne tourne plus (signature de `detect_sections` changée au sprint 9) —
le réparer est un préalable, et c'est déjà noté dans `known_issues.md`.

**D2. Le repli moyenne-t-il un passage qui ne devrait pas être dans la pile ?**
La question ouverte laissée par l'audit du 15/09 : un accord de substitution
dans un passage sur quatre est-il écarté comme exception, ou fausse-t-il la
moyenne en silence ? Test : chercher les piles où un membre diverge à une
position, comparer l'accord consensus à ce que la majorité des passages jouent.

### E. Tonalité

**E1. Tonalité contre iReal** pour les standards. Test bon marché, vérité
terrain réelle.

**E2. La dominante non résolue** — la trouvaille récupérée du legacy : le
nouveau traceur (chroma + CUSUM) se fait-il piéger À SA manière par une
dominante ambiguë jamais résolue ? Fabriquer le cas et écouter.

### F. Plomberie

**F1. Contrat brut/final.** Le chart brut doit être sérialisé AVANT que le
repli ne mute `bars`. Vérifier sur une analyse réelle que le fichier brut
écrit sur disque n'a jamais de sections repliées.

**F2. Les annotations survivent-elles à une ré-analyse ?** Chemin vivant, pas
seulement le test unitaire.

**F3. La marque « Set bar 1 » est-elle bien relue** depuis `state/human/marks/`
par `jobs.bar1_for` au ré-analyse ? (Neuf depuis le sprint 15.)

## Règle de travail pour cet audit

Chaque bug confirmé = une page avant/après écoutable, pas un chiffre. Louis
arbitre. Un chart qui change est le BUT, pas un risque — c'est la correction
explicite du 2026-09-15.

---

# Résultats (2026-09-15, après-midi) — ce que les tests ont dit

## A1 — Les caches sont VRAIS. L'audit a un sol.

Recalcul à froid (traceur Beat This!, 5 réseaux musx, chroma NNLS) sur
Stand By Me, This Love, Let It Be, comparé aux caches :

| étage | Stand By Me | This Love | Let It Be |
|---|---|---|---|
| battues (347 / 322 / 279) | identiques, médiane 0,0 ms | identiques | identiques |
| postérieures musx (max \|Δ\|) | 1e-6 | 0,07 sur 0,6 % des trames, **une trame de plus en queue** dans le cache, décalage temporel 0 | 6e-5 |
| chroma NNLS (max \|Δ\|) | 0,02 sur 4 trames / 3822 | 2e-4 | 2e-4 |

Verdict : aucune erreur de calibration cachée. Le « 44/44 identiques »
validait du code posé sur des features fidèles.

**A2** (par lecture) : sont en cache la sortie BRUTE du traceur (le
nettoyage `_clean` s'applique à la lecture — un changement de loi de
nettoyage n'est PAS masqué), les postérieures musx, le chroma, les sections
songformer. Tout ce que `settings.py` décide (re-décodage, mesures, repli,
tonalité) est recalculé à chaque cuisson : aucune constante n'est masquée.
Seul un changement de code DANS songformer ou dans les réseaux le serait.

## C1 — Latence : le modèle entend EN AVANCE, la recherche cherche en retard. BUG CONFIRMÉ.

Mesure directe, sans décodeur : énergie de changement des postérieures
(0,5·L1 entre trames voisines) corrélée à la grille de battues exacte du
décodeur, retards −300…+300 ms, 46 morceaux.

- **45 morceaux sur 46 : le pic est à −23 ou −46 ms** (une à deux trames
  AVANT la battue), jamais après. Courbe moyenne du corpus : pic à −23 ms
  (gain ×2,5), déjà retombée à ×1,1 à +70 ms.
- La recherche de `musx.redecode` ne teste que 0…+280 ms. « 0 gagne
  partout » n'est pas un bug de départage (H1 rejetée : la courbe est une
  vraie courbe, monotone) — c'est **une recherche bornée du mauvais côté**,
  bloquée au bord de sa grille. Le postulat de l'étape (« le modèle entend
  en retard ») venait d'un autre traceur de battues ; avec Beat This!, il
  est faux.
- **Another Day (`dOQXg6rK86I`) : 280 ms choisi = un alias.** Sa courbe de
  vraisemblance sur une grille symétrique est bimodale : vrai optimum à
  **−80 ms**, second pic à +280 ms, qui est −163 ms décalé d'une période
  de battue (443 ms). La recherche, interdite de négatif, a pris l'alias.
  Conséquence dans le chart : **99 battues sur 404 portent un autre accord
  qu'à L=0, dont 78 portent l'accord de la battue SUIVANTE** — les accords
  sont écrits un temps trop tôt sur un quart du morceau. C'est le morceau
  où Louis « a vu la grille glisser ».
- `rec_1787243982182` (enregistrement micro, 28 battues) : 120 ms choisi,
  courbe plate (gain max ×1,19) — du bruit, pas une mesure.
- Sur This Love et Stand By Me, L=−40 ms et L=0 donnent le MÊME chart
  (0 battue sur 321 / 346 ne change).

**Correction candidate** : latence fixée à 0, recherche supprimée (7
Viterbi de moins par morceau, plus d'alias possible). Page avant/après :
Another Day change, le reste est identique. Louis arbitre.

## C2 — Stand By Me : le N.C. vient du RENDU, pas du modèle. BUG CONFIRMÉ.

Le décodeur (L=0) dit A / A / F#m / F#m / D / E / A / A sur les couplets
chantés (mesures 30-69, P(accord) 0,8-0,97, P(N.C.) 0,00). Le chart écrit
N.C. sur 36 mesures sur 86.

Mécanisme : la section A a 6 occurrences (mes. 6-13, 14-21, 30-37, 38-45,
54-61, 62-69). Les deux premières sont l'intro basse + voix, où musx
n'entend aucun accord (P(N) 0,5-0,9 — limite du modèle, pas un bug). Le
repli empile les 4 couplets et réécrit bien le consensus dans LEURS
mesures. Mais **le bloc affiché pour la lettre est `occ[0]`, la première
occurrence chronologique** (`harmonia_min/soudure.py::sections_pour_chart`,
`"bars": bars[b0:b1+1]` avec `b0, b1 = occ[0]`) — celle que la pile a
elle-même rejetée (mesure 6 = variante). Résultat : 5 N.C. sur 8 mesures
affichées, ×6.

**Le correctif existe déjà — sur l'autre chemin.** `folding.minimal_fold`
choisit depuis le 2026-08-10 « la passe qui porte le plus d'attaques
réelles » (`_evidence`), motivé par CE morceau (« je vois plein d'accords,
et sur le chart j'ai juste des NC partout »). Mais les morceaux à
découpage validé à la main (20 sur 44) passent par `sections_pour_chart`,
qui a gardé l'ancienne loi. Deux chemins, deux lois — l'erreur n° 4 de la
liste du refactor, encore là. Et `sections_pour_chart` n'a jamais été
porté : `tools/golden.py` l'importe depuis `harmonia_min` même pour le
moteur `harmonia` (à régler au sprint 22).

Corpus (46 charts) : **40 sections repliées sur 112 affichent une première
passe que la pile a rejetée** (au moins une mesure « variante » dedans).
Stand By Me est le cas extrême (5 N.C.), pas le seul.

**Correction candidate** : une seule loi — `sections_pour_chart` choisit la
passe par `_evidence` comme `minimal_fold`. Page avant/après sur les 20
morceaux à découpage manuel. Louis arbitre.

## B1 — La grille : 3 charts à ré-écouter, pas de bug de code trouvé

| morceau | tempo | signe | verdict |
|---|---|---|---|
| `autumn_leaves` | 187,5 | 282 mesures, **18 lettres** dont 9 sections de 2 mesures, cohérence des downbeats 52 % | chart cassé ; pas de cache musx (morceau « froid ») — à ré-analyser |
| `autumn_leaves_easy_jazz_piano…` | 32,3 en 3/4 | 31 battues, 10 mesures, 2,5 accords/mesure | piano rubato : la grille n'a pas de sens ici |
| `fd02pGJx0s0` | 157,9 | cohérence 65 %, pas de tempo rigide trouvé | octave suspecte (79 ?) — à l'oreille |
| `Ju8Hr50Ckwk` | 120 en **3/4** | cohérence 76 % | métrique à confirmer à l'oreille |
| Come Away With Me | 80 en **3/4** | cohérence 100 % | probablement juste, à confirmer |
| `h_D3VFfhvs4` | 117,7 | refusé par `check_grid` (76 % < 85 %) | connu, exclu du rapport d'or |

Rien de systématique : 39 morceaux sur 46 ont une grille rigide posée avec
≥ 85 % de battues dedans, couverture ≥ 95 %.

## C1-bis — Le décodage du GABARIT cherche aussi une latence, et là c'est bien pire. BUG CONFIRMÉ, 18 morceaux.

Le premier rapport d'or avec `DEFAULT_LATENCY_GRID = (0.0,)` a changé
**18 morceaux sur 44 (239 mesures)** — dont 16 où la latence principale
était déjà 0. Isolé en trois runs : grille d'origine → 44/44 identiques ;
`(0.0,)` → 18 diffèrent ; le décodeur lui-même est sans état et rend les
mêmes segments quelle que soit la grille. La différence est en aval : le
repli.

`folding._decode_template` re-décode le gabarit empilé d'une lettre (une
boucle synthétique de 3×P mesures, battues à `i·step`) **avec la même
recherche de latence**. Sur ce gabarit il n'y a par construction aucune
latence à compenser — les postérieures y sont empilées mesure par mesure.
La recherche y est donc pure dégénérescence, et elle sature :

| latence choisie sur le gabarit | décodages (sur 96, 44 morceaux) |
|---|---|
| 0 ms | 7 |
| 40 ms | 15 |
| 80 ms | 52 |
| 200–240 ms | 14 |
| **280 ms (borne)** | **8** |

Résultat : les accords du consensus sont réécrits **un temps trop tôt**
dans TOUTES les mesures de la lettre. `Urdlvw0SSEc` (6/8) : les quatre
lettres à 280 ms, chaque accord de Em | Bm posé sur le temps 6 de la mesure
précédente au lieu du temps 1 — 80 mesures sur 108. Hot N Cold : 69
mesures. `meta.musx_latency_ms` ne montre que le décodage principal — ce
bug était invisible dans le chart.

Ces 18 morceaux sont exactement ceux où le gabarit a choisi 200–280 ms, ou
80 ms sur un tempo assez rapide pour que 80 ms retombe sur le temps
précédent.

**Correction** : la même — plus de recherche de latence, nulle part. La
page avant/après couvre les 18 morceaux.

## B2 — Le re-calage harmonique de la phase ne se déclenche JAMAIS. DEAD CODE CONFIRMÉ.

Les quatre morceaux où Louis a posé « Set bar 1 » à la main sont ceux où
il n'était pas d'accord avec la machine. Sans sa marque, la machine
(traceur + re-calage) se trompe sur 3 des 4 : Chasing Pavements (d'une
demi-mesure), Virtual Insanity (d'une demi-mesure), Goodbye Yellow Brick
Road (d'un temps). Oextk : d'accord.

Instrumenté sur les 44 : `bars._phase_correction` rend 0 partout — **0
déclenchement sur 44**. Ses deux seuils (consensus ≥ 0,55 ET part du temps
0 ≤ 0,15) ne sont jamais satisfaits ensemble sur des données réelles. Et
sur 2 des 3 morceaux corrigés par Louis, **le vote des accords pointait sur
SA phase** — à 45 % (Chasing Pavements) et 50 % (Virtual Insanity), sous
la barre des 55 %. Le seul autre morceau avec un vote non nul est Let It Be
(résidu 2 à 54 %, découpage validé à la main sur la grille actuelle).

Pas de correctif proposé aujourd'hui : baisser le seuil à 0,45 corrigerait
les deux marques de Louis mais re-calerait Let It Be d'une demi-mesure —
c'est son oreille qui doit dire si Let It Be est bien calé aujourd'hui.
Question ouverte, quatre morceaux, page à faire si Louis veut la trancher.

---

# La page avant/après — à arbitrer

**http://100.89.209.63:7772/reports/avant_apres.html** — 22 morceaux, les
deux versions de chacun sont dans la bibliothèque (« … — AVANT » /
« … — APRÈS », `python -m tools.avant_apres --clean` pour les retirer).

Deux corrections candidates, en attente de l'arbitrage de Louis, **non
commitées** (`harmonia/musx.py` : `DEFAULT_LATENCY_GRID = (0.0,)` ;
`harmonia/folding.py` + `harmonia_min/soudure.py` : `pass_evidence`, une
loi pour la passe écrite) :

| morceau | mesures | C1 (latence) | C2 (passe) | latence principale | latences des gabarits (prod) | tonalité | sections |
|---|---|---|---|---|---|---|---|
| Urdlvw0Ssec | 80 | 80 | 0 | 0 | 280 280 280 280 | B minor → E minor | 7 |
| Hot N Cold | 69 | 69 | 0 | 0 | 280 280 0 | G major | 8 |
| Another Day | 25 | 25 | 0 | 280 | 200 200 | Eb major | 10 → 9 |
| R3B2Xr8Kwq | 19 | 19 | 0 | 0 | 200 | A major → D major | 7 |
| Sam Smith — I'm Not The Only One | 15 | 15 | 0 | 0 | 0 240 240 40 | F major | 7 |
| Bobby Hebb — Sunny | 9 | 3 | 6 | 0 | 80 80 | G minor | 6 |
| Bein Green | 7 | 0 | 7 | 0 | 40 | Bb major | 4 |
| Goodbye Yellow Brick Road | 7 | 0 | 7 | 0 | — | F major | 9 |
| Sunny Afternoon | 7 | 7 | 0 | 0 | 80 40 | Eb major → C minor | 6 → 7 |
| Cry Me A River | 6 | 6 | 0 | 0 | 80 280 80 0 0 | G# minor | 8 |
| Uw5Olnn7Uvm | 6 | 6 | 0 | 0 | 80 | F minor | 6 |
| Every Breath You Take | 6 | 0 | 6 | 0 | 80 80 40 80 | Ab major | 7 |
| Stand By Me | 5 | 0 | 5 | 0 | 80 80 80 80 | A major | 3 |
| Lost Without U | 4 | 4 | 0 | 0 | 80 | C major | 8 |
| Let It Be | 3 | 0 | 3 | 0 | 40 40 40 | C major | 6 → 7 |
| Virtual Insanity | 2 | 2 | 0 | 0 | 80 80 | Eb minor | 7 |
| Easy On Me | 2 | 2 | 0 | 0 | 80 80 | F major → Bb major | 6 |
| The Lazy Song | 2 | 0 | 2 | 0 | 80 80 80 80 40 80 80 | E major | 5 |
| The Walk | 2 | 0 | 2 | 0 | 80 80 80 280 80 80 80 80 | A major | 5 |
| Grenade | 1 | 0 | 1 | 0 | 80 40 80 40 | D minor | 6 |
| She Will Be Loved | 1 | 0 | 1 | 0 | 80 80 80 80 80 | Bb major | 7 |
| Enregistrement du 20/08 (micro) | 1 | 1 | 0 | 120 | — | G# minor → C# minor | 2 |

Vérifié au rendu (Playwright, 390 px) : Stand By Me APRÈS lit
A A F#m F#m D E A A ; Urdlvw0SSEc AVANT affiche des « BBm » superposés
(l'accord posé au temps 6 déborde sur la case suivante), APRÈS lit
Em Bm Em Bm en E minor.

**Ma recommandation** : accepter les deux. C1 corrige un mécanisme (une
recherche du mauvais côté, qui aliase d'un temps) ; C2 applique une loi
que Louis a déjà validée sur l'autre chemin. Les trois changements de
tonalité (Urdlvw0SSEc, R3B2Xr8Kwq, Easy On Me, Sunny Afternoon) suivent
les accords remis à leur place — à écouter. Les changements de nombre de
sections (Let It Be B×7 → B×6 + B′, Sunny Afternoon C×4 → C×3 + C′, Another
Day 10 → 9) sont la cascade iReal qui re-décide sur les nouvelles mesures.

Après acceptation : supprimer la recherche de latence (pas seulement la
grille), re-geler la baseline, republier la bibliothèque, redémarrer :7772.

## Aussi trouvé en passant

- `tests/test_songformer_sections.py::test_minimal_fold_separe_les_longueurs_dune_meme_lettre`
  échoue déjà à HEAD (vérifié dans un worktree jetable) : la cascade iReal
  replie le B de 4 mesures dans le B de 8 comme un préfixe, ce que le test
  (2026-08-18) interdisait. Un des deux a raison, pas les deux.
- `sections_pour_chart` n'a jamais été porté : `tools/golden.py` l'importe
  de `harmonia_min.soudure` même pour le moteur `harmonia`. À faire au
  sprint 22 avec le reste de `soudure.py`.

## Reste à tester (plan initial)

B3 levées jetées · C3 qualités contre iReal · D1 (réparer
`tools/sections_bench`) · D2 pile contaminée · E1/E2 tonalité · F1–F3
plomberie.

## B3 · D2 · F1 · F3 — testés, rien de grave

- **B3, levées jetées** : 0 morceau sur 44 ne perd d'accord de levée
  (`bars: shed … clamped pickup chords` n'apparaît jamais).
- **D2, consensus contre accord sûr** : sur les 44, **5 mesures** où un
  accord de première passe à confiance ≥ 0,70 voit sa fondamentale réécrite
  par le consensus du repli. À écouter, pas de correctif proposé (c'est la
  loi du repli que Louis a validée, et le cas est rare) :
  `0DdCoNbbRvQ` B mes. 26 D-7(0,93) → G7 ; Easy On Me B mes. 15 et 36
  F D-(0,75) → F ; `oIv_Y2RPQ_A` C mes. 44 et 80 Gb(0,87) → (rien, porté).
- **F1, contrat brut/final** : le générateur mute bien l'objet brut après
  son `yield` (44/44 — c'est le contrat), et `server/jobs.py:227` sérialise
  sur disque À CHAQUE yield avant de reprendre : le fichier brut n'a jamais
  de sections repliées. Seule réserve : `jobs.py:242` garde une référence
  `raw_model=model` à l'objet qui sera muté — à ne pas lire après la reprise.
- **F3, marque « Set bar 1 »** : `jobs.bar1_for` ne lit que
  `state/human/marks/<stem>.json` (jamais le champ `bar1` du chart), et
  `tools/golden.py` passe par le même dossier. Les 4 marques existantes sont
  relues (vérifié B2).
- En passant : `min_autumn_leaves` reste froid parce que SongFormer se fait
  tuer par l'OS sur ce fichier de 6 min (déjà dans known_issues).

Non testés : C3 (qualités contre iReal — un seul standard dans la
bibliothèque), D1 (`tools/sections_bench` à réparer d'abord), E1/E2
(tonalité), F2 (annotations à travers une ré-analyse, chemin vivant).

---

# Accepté et livré (2026-09-15, 15 h 52)

Louis : « Ca m'a l'air mieux sur quasi toutes les chansons, j'accepte. »

Livré dans la foulée :
- `harmonia/musx.py` : la recherche de latence est SUPPRIMÉE, pas seulement
  réduite — plus de grille, plus de `path_loglik`, plus de garde ; `redecode`
  fait un décodage, sur les battues, et rend les segments. 528 → 468 lignes.
  `meta.musx_latency_ms` disparaît du chart ; `tools/golden.VOLATILE` est vide.
- `harmonia/folding.py` (`pass_evidence`, une loi) + `harmonia_min/soudure.py`.
- Vérifié : le code simplifié reproduit les 44 charts acceptés à l'octet près
  (modulo le champ disparu) ; 37 tests verts, la seule rouge est celle qui
  l'était déjà à HEAD.
- Baseline re-gelée (`state/cache/golden/baseline`, l'ancienne gardée sous
  `baseline_2026-09-14`) ; bibliothèque republiée chart par chart (44,
  sauvegarde `state/cache/golden/charts.bak_20260915`) ; paires AVANT/APRÈS
  retirées ; :7772 redémarré par PID de port ; rendu vérifié (Stand By Me lit
  A A F#m F#m D E A A dans la bibliothèque vivante).

Question de Louis pendant la livraison : « es-tu en train de réparer le bug
qui fait que le modèle entend 25-45 ms en avance ? » — Non : ce n'est pas un
bug, c'est une à deux trames (23 ms) de fenêtre d'analyse qui chevauche
l'attaque, et ça ne change aucun temps du chart (mesuré : −40 ms ⇒ 0 temps
sur 321 / 346). Le bug, c'était de chercher une compensation du mauvais côté.
