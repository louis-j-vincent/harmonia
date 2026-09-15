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
