# Sections — où on en est (2026-09-30)

Reconstitué depuis git log (août–sept.), `docs/known_issues.md`, `docs/blog/25–28`,
le code et la mémoire de projet. But : repartir sur « Louis donne UNE section →
trouver BIEN toutes ses occurrences ».

## 1. Ce qui tourne en production aujourd'hui

**Aucune détection automatique.** `SETTINGS.sections = "aucune"` depuis le
2026-09-19 : le pipeline rend une seule section « A » sur tout le morceau. Les
sections n'existent que si Louis annote.

**Le chemin quand Louis annote** (`harmonia/static/screens/sections_editor.js`) :

1. il passe le doigt sur des mesures → une marque, sauvée en brouillon
   (`state/human/sections_draft/`). **Le geste ne parle pas au serveur** : rien
   n'est cherché tant qu'il ne clique pas.
2. « Valider les sections » → `POST /api/sections/inferer/<file>`
   (`harmonia/server/routes/sections.py:409`) :
   - `simulation.occurrences_de_tous` → pour chaque trait,
     **`simulation.cherche_brique`** = LE chercheur d'occurrences en prod ;
   - `simulation.remplir_par_algo` → les trous sont nommés par l'agglomération
     des « quatre mots » (`phrases4`), avec lettres neuves.
3. le découpage est écrit dans le chart via `/api/soudure/valider`, puis
   `refold` réempile les accords.

**`cherche_brique`** (`harmonia/sections/simulation.py:68`) : la brique de L
mesures glisse le long de la SSM chord-tone (`sections/similarity.py`, un
vecteur de 12 classes par MESURE = postérieure musx moyennée) ; score à la
position c = moyenne de `S[b0+i, c+i]` ; seuil **relatif au morceau**
`médiane + 1,5·σ`, borné [0,85 ; 0,999] ; pics gloutons sans recouvrement ;
puis 11 transpositions (bonus +0,06, seulement si L ≥ 4). Longueur FIXE = L :
une occurrence ne peut être ni plus courte, ni plus longue, ni avoir une autre fin.
Intro/outro/silence ne cherchent jamais de reprise.

**Il existe un deuxième chercheur, mort côté UI** : `section_tool.find_repeats`
(`/api/section-repeats`, plus appelé par l'UI depuis 2026-08-17). Il devine une
cellule interne (L, L/2, L/4…), score normalisé par l'ancre (`block_score`),
grille de 2 mesures + repli de parité impaire, seuil 0,78. Mesuré sur 18
morceaux : ~81–86 % des occurrences couvertes, ~12 mesures en trop par lettre.
Réutilisé à l'aveugle il fusionne tout (San Francisco Street, 27 mesures).

**Le prototype `tools/repli_propose.py`** (2026-09-24, San Francisco Street) :
découverte AVEUGLE de cellules de 4 puis 2 mesures, vecteurs chord-tone
CONCATÉNÉS (l'ordre des mesures compte), cosinus ≥ 0,90, glouton. Reproduit
l'analyse à la main de ce morceau ; jamais validé sur un 2e morceau ; ne
revérifie pas une cellule de 4 qui serait une cellule de 2 (This Love B).

## 2. Chronologie courte

| date | quoi | issue |
|---|---|---|
| 08-09 | outil sections (doigt + « où ça se rejoue ? », `find_repeats`) | remplacé le 08-17 par Valider → inferer |
| 08-13/16 | Soudure puis **l'algo du retour** (`retour.py`) | « c'est vraiment top !!!! » — mort avec `harmonia_min` (git seulement) |
| 08-18 | SongFormer (modèle audio) en prod | reconnaît et NOMME intro/refrain |
| 09-17 | « une fois que j'ai annoté une section, c'est une brique, trouve ses occurrences » → `cherche_brique` + SongFormer dans les trous, EN PROD | Don't Want My Love : 3/3 occurrences |
| 09-17 | seuil fixe 0,90 → relatif (Cry Me A River recopiait la brique sur la moitié du morceau) | un morceau sans contraste s'abstient |
| 09-19 | SongFormer retiré (mémoire ∝ durée², un morceau de 326 s ne finit jamais, swap 9,5 Go) | trous remplis par les « quatre mots » ; plus d'intro/outro auto |
| 09-19 | transpositions ajoutées à `cherche_brique` | tirent sur 2 morceaux / 18 (Sunny, T64BgKEL-Sw) |
| 09-20 | grille de 4 mesures (76 % des 238 frontières de Louis) + nouveauté de Foote | échange 128/51 → 120/24, NON branché, attend arbitrage |
| 09-24/25 | San Francisco Street : brique gelée « trou noir », `repli_propose`, 2 correctifs `folding._merge_coupe` | appliqué en direct sur ce morceau |

## 3. Essayé et écarté (et pourquoi)

- **Seuil absolu 0,90** : casse les ballades sans contraste (Cry Me A River,
  médiane 0,896). → seuil relatif.
- **Règle du préfixe** (« un bloc qui commence par un autre bloc ») : 0 gagnant
  sur 18 morceaux ; la matière partagée est là mais DÉCALÉE d'une bi-mesure
  (commit 62b0fde). La bonne primitive est « contient le même motif, décalé ».
- **Centroïdes de chroma NNLS pour fusionner des lettres** : un centroïde est
  un sac de notes, la chroma porte la production ; aucun seuil ne sauve
  Fallin' + Let It Be. Les ACCORDS DÉCODÉS séparent (f08c6cd).
- **Nouveauté de Foote seule** : 10 frontières justes / 27 ; impasse prouvée
  sur Stand By Me (boucle de 4 accords : l'harmonie ne peut pas la découper).
- **SongFormer** : bon, mais coût mémoire rédhibitoire (fenêtre 150 s possible :
  112 s / 7,6 Go, noms du dernier tiers dégradés).
- **`find_repeats` en découverte aveugle** : transpositions en chaîne → tout le
  morceau devient une famille.
- **Moyenner une fenêtre** (au lieu de concaténer les mesures) : confond
  « D D Em Em » et « D G Em A ».

## 4. Problèmes ouverts qui touchent les occurrences

1. **Brique gelée = trou noir** (known_issues 2026-09-24) : une fois une brique
   posée, `remplir_par_algo` ne compare jamais les blocs libres à SON contenu ;
   le même motif ressort sous 3–4 lettres.
2. **Longueur fixe** : `cherche_brique` ne sait trouver qu'une occurrence de
   même longueur ; une occurrence avec mesure en plus/en moins ou une autre fin
   est soit ratée, soit coupée.
3. **Morceau sans contraste → abstention** (voulu, mais une vraie reprise est
   alors ratée).
4. **Reprise qui change de mode** (refrain passé en mineur) : invisible.
5. **Section unique à boucle interne 2× ne replie jamais**
   (`FIN_SECTION_HORS_PILE`, garantie arithmétique).
6. **Grille de 4 / phase** mesurée mais non branchée.

## 5. Préférences verrouillées de Louis

- **SSM chord-tone, toujours** (Si♭ plus proche de Sol- que de Fa) ; jamais
  fondamentale seule. `accords*basse` (produit) pour l'algo du retour.
- **Sous-replier plutôt que sur-replier** : deux passages de longueurs ou
  d'accords différents → deux blocs. Un chart qui ment sur sa longueur est pire.
- **Ce que Louis trace est la vérité** : sa brique sort aux mesures exactes,
  sous son nom ; une ressemblance de la machine reste une PROPOSITION, jamais
  une contrainte figée.
- **Une section = une mini-boucle (≥ 4 mesures, modulo la dernière) jouée
  ≥ 2 fois** ; l'exemption de cadence porte sur la SUITE, jamais sur le bloc.
- **Une intro ne se rejoue pas.**
- **Le décodage brut indépendant de chaque mesure bat le gabarit** d'une lettre
  (Am7 fantôme de San Francisco Street).
- **Seuil par morceau ; un canal sans contraste s'abstient** et le DIT.
- **Illustrer, pas quantifier** en phase exploratoire ; Louis arbitre tout
  changement de chart, et « n'essaye pas de fix, c'est moi qui dirai comment ».

## 6. À noter pour la suite

- Les charts `min_*.json` ne sont plus dans `state/cache/charts/` (38 fichiers,
  aucun `min_*`, dossier modifié le 2026-09-30 00:03) — la bibliothèque live n'en
  liste plus aucun. Les grilles de mesures utilisées pour l'analyse viennent des
  sauvegardes (`state/cache/_sans_fusion/`, `charts.bak_soudure/`) dont le
  nombre de mesures égale le `n` de la vérité terrain.
- Un autre agent corrige un bug de mesure 0 dans `bars.py` : une grille décalée
  d'une mesure déplacerait toutes les comparaisons ci-dessus d'autant.

## 7. Diagnostic du jour (étape 2-3)

`docs/plots/occurrences_brique_2026-09-30.html` et l'entrée du même jour dans
`docs/known_issues.md` : le seuil qui bute au plafond 0,999, puis la longueur
figée, puis les jumeaux harmoniques. Candidat 1 (plafond 0,97) prêt à
arbitrer ; candidat 2 (longueur libre) : prémisse vraie, version à refaire.

## 8. Verdicts de Louis sur la page du jour, et ses trois règles (branchées)

**Verdicts (vérité terrain désormais)** :
- **Stand By Me** : tout le morceau peut être considéré comme UNE section — les
  « jumeaux » A/C ne sont pas de fausses trouvailles.
- **The Walk** : B peut être considéré plus court — les 3 « en trop » de A
  (l'intro, la 2e moitié des B) sont acceptables.
- Sa conclusion : « le détecteur avec pics est bon », il lui faut trois changements.

**Ses trois règles, mot pour mot** :
1. « Interdire la recherche de pics dans la zone qui est donnée de base, ce ne
   sont pas des pics valides. »
2. « Quand je donne une section, il faut regarder si elle-même est une
   répétition d'une boucle minimale, car je peux parfois surligner une section
   alors que c'est plusieurs fois la section. C'est la boucle minimale qui
   définit la section qui devra ensuite être cherchée (ça devrait fixer San
   Francisco Street). »
3. « Ce n'est pas la hauteur du pic qui dicte, mais la hauteur du pic par
   rapport au reste, c'est clairement visible dans Stand By Me — trouve une
   façon de normaliser les pics par rapport aux autres pics voisins, sachant
   que ce sera plus simple une fois que tu auras identifié partout la boucle
   minimale. »

**Implémentation** (`harmonia/sections/simulation.py`, `cherche_sur_matrices`,
appelée par `cherche_brique` — donc « Valider les sections » après
redémarrage du serveur) :
1. un départ dont la fenêtre CHEVAUCHE la brique (même d'une mesure) n'est ni
   un pic, ni une valeur du fond/seuil qui juge les autres ;
2. `boucle_minimale` : brique ≥ 8 mesures = k × boucle de ≥ 4 mesures si chaque
   mesure colle à celle d'une boucle plus loin depuis le début (« coller » =
   ≥ (1 + médiane des ressemblances internes de la brique)/2, relatif à la
   brique), au moins 2 passages, le dernier pouvant finir autrement. La brique
   devient ses passages (tous marqués « tracés »), on cherche la boucle, et la
   fin de la brique est recherchée juste après chaque suite de boucles retrouvée ;
3. fond local = médiane de la courbe aux départs décalés de 1 à L−1 mesures
   (sans les multiples de la sous-période harmonique ni la zone donnée) ;
   reprise si (score − fond)/(1 − fond) ≥ 0,6 et score ≥ 0,85 ; parmi les
   départs qui passent, le plus de reprises sans recouvrement. Plus de seuil
   global ni de plafond 0,999. Brique < 8 mesures : ancienne loi, calculée hors
   zone donnée.

**Résultat** (page `docs/plots/occurrences_brique_v2_2026-09-30.html`, 20
découpages, en mesures) : justes 1270 → 1402, ratées 260 → 128, en trop
278 → 408 (jumeaux harmoniques : Let It Be A, She Will Be Loved A/D, This Love A,
X-yIEMduRXk B). San Francisco Street A 24 → 40/41, Stand By Me A 32 → 80/80,
Let It Be A 8 → 32/32 (+24 en trop, le refrain), The Walk A 12 → 40/40.
Chain of Fools : vamp, abstention (inchangé). Rapport d'or : 0 mesure changée.

## 9. Verdict de Louis sur la page v2, et sa règle 4 (branchée)

**Verdict** : « beaucoup beaucoup beaucoup mieux », sauf **Let It Be** — le
refrain B, vraiment différent, sort comme des A (24 mesures en trop).

**Sa règle 4, mot pour mot** : « Prends le score du pic le + grand trouvé en
dehors de la section déjà identifiée — ça nous donne une idée de la
variabilité inter-chanson. Ensuite utilise cet indice de variabilité + la
taille de tous les autres pics pour décider d'un score qui permet de
différencier le A du B dans Let It Be, tout en gardant Stand By Me. »

**Implémentation** (`simulation.reference_du_morceau`, commit eb72b62) : parmi
les départs qui passent la règle 3, R = le plus haut ; on trie les hauteurs et
on prend le plus grand trou entre deux consécutives ; tolérance =
`TROU_K (2) × max(1 − R, écart dans le groupe du haut, BRUIT_PIC 0,01)` ; on
coupe tout ce qui est sous le trou si trou > tolérance ET groupe du bas plus
nombreux que celui du haut (« la taille des autres pics »).

**Mesuré** (page `docs/plots/occurrences_brique_v3_2026-09-30.html`, rejouable
`python -m tools.occurrences_brique_v3 --page`) — la règle ne coupe que 3 lettres
sur les 20 découpages validés :

| lettre | R | trou | tolérance | haut / bas | justes / ratées / en trop v2 → v3 |
|---|---|---|---|---|---|
| Let It Be A | 1,000 | 0,066 (0,994 → 0,928) | 0,020 | 6 / 15 | 32/0/24 → 32/0/0 |
| Stand By Me A | 0,943 | 0,020 | 0,114 | 1 / 8 | 80/0/0 → 80/0/0 (pas de coupe) |
| The Walk B | 1,000 | 0,075 | 0,020 | 2 / 4 | 24/12/4 → 20/16/0 |
| X-yIEMduRXk A | 0,994 | 0,092 | 0,020 | 2 / 3 | 24/1/8 → 20/5/8 |

Total : justes 1402 → 1394, ratées 128 → 136, en trop 408 → 380. Tout le reste
identique à v2 (Every Breath You Take A : trou 0,072 mais 3 / 3, gardé — c'est
la seconde condition qui le sauve). Rapport d'or : 0 mesure changée (60 cuits,
2 froids, 1 erreur préexistante h_D3VFfhvs4).

**À arbitrer par Louis** (rien n'a été retouché) :
- The Walk B : la coupe enlève les boucles décalées de 2 mesures (51, 55, 79,
  83 à 0,92) ; le décalage disparaît (0 en trop) mais 57–58 et 85–86 sont
  perdues. Le 2e et le 3e passage de chaque B restent ratés, comme en v2.
- X-yIEMduRXk A : un jumeau dans le C (départ 44, 0,991) est aussi haut que la
  vraie reprise (23, 0,994) et gonfle le groupe du haut ; la vraie reprise
  31–34 (0,873) est coupée. La règle suppose que la meilleure trouvaille hors
  zone est une vraie reprise.
- Pas touché par la règle : She Will Be Loved A/D (jumeaux au niveau des vrais
  A, 14 / 13 et 17 / 13, aucun trou).
