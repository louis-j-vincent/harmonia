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

## 10. Le décalage de The Walk, et la règle 5 de Louis (branchée)

**Le défaut** (`docs/plots/the_walk_decalage.html`) : A donné en 9–16, boucle
de 4 (A–Bm–A–Bm, accords qui tournent toutes les 2 mesures). Le choix « le plus
de boucles sans recouvrement » prenait 7 boucles décalées de 2 dans 25–56
(départs 27…51, les deux du bord à 0,91 sur le C♯m du B) plutôt que 6 en phase
à 0,996 (29…49) ; de nouveau juste à 57, décalé encore à 83.

**Sa règle 5, mot pour mot** : « si on hésite entre un départ et celui de ses
deux voisins, on prend celui qui est plus grand que ses voisins d'avant et
d'après (généralise au cas où on aurait des boucles plus grandes que 2) ».

**Implémentation** (`simulation.maxima_locaux`, commit 63c3df7) : parmi les
départs qui passent la règle 3, un départ reste candidat seulement s'il est le
plus haut de ceux qui l'ont aussi passée à ±(Lb − 1) mesures (Lb = la boucle
minimale de la règle 2). Choix, dits dans le docstring :
- **égalité à 0,01 près (`BRUIT_PIC`) : les deux restent**, la couverture
  choisit. « Le plus tôt gagne » réduirait un plateau de sous-période (tout à
  0,995–1,0) à un seul départ, au hasard du bruit ;
- **les voisins sont les départs qui passent la règle 3, pas toute la courbe** :
  un plateau à 0,95 qui ne se détache pas de son fond n'efface pas un pic isolé
  à 0,88 (sinon le test de la règle 3 casse) ;
- la zone donnée (règle 1) n'est jamais voisine : elle n'efface pas une reprise
  posée juste après la brique ;
- **après la règle 4, avant le choix** : la règle 4 compte toujours les calages
  décalés parmi « tous les autres pics » (inchangée, Let It Be ne bouge pas) ;
- pas touchés : les passages de la brique (toujours rendus), la recherche de sa
  fin (posée derrière les boucles retrouvées, elle suit leur phase), les
  transposées et l'ancienne loi < 8 mesures (gloutons par hauteur).

**Mesuré** (page `docs/plots/occurrences_brique_v4_2026-09-30.html`, rejouable
`python -m tools.occurrences_brique_v4 --page`) — 4 lettres changent :

| lettre | départs v3 → v4 | justes / ratées / en trop v3 → v4 | 5b |
|---|---|---|---|
| The Walk A | 27…51, 83…95 → 29…49, 85, 89, 93 | 40/0/6 → 40/0/4 | = v4 |
| Lazy Song A | 24, 58, 62 → 60 | 48/0/8 → **40/8/8** | = v3 |
| She Will Be Loved A | 34 perdu | 36/0/28 → **32/4/28** | = v3 |
| She Will Be Loved D | 13, 17 → 15 | 10/0/60 → 10/0/52 | = v3 |

Total : justes 1394 → 1382, ratées 136 → 148, en trop 380 → 370. Rapport d'or :
0 mesure changée (60 cuits, 2 froids, 1 erreur préexistante h_D3VFfhvs4).

**Ce que ça ne résout pas** : sur deux accords qui alternent, le bon départ peut
être un peu PLUS BAS que son voisin décalé de 2 (Lazy Song : 0,963 contre
0,977–0,991 ; She Will Be Loved A : 0,977 contre 0,994) ; la règle tranche alors
sur du bruit et perd des mesures justes. L'harmonie seule ne connaît pas la
phase d'A–Bm–A–Bm.

**À arbitrer par Louis** : variante **5b**, non branchée — deux départs à
égalité tant que leur écart est sous la tolérance du morceau de la règle 4
(2 × max(1 − R, écart du groupe du haut, 0,01) ; 0,04–0,07 sur ces morceaux).
The Walk réglé pareil, aucune perte : justes 1394, en trop 378.

## 11. La 5b, et les candidats en suspens (branchés)

**Décision de Louis, mot pour mot** : « Oui pour la 5b très bien, mais alors il
faut trancher par rapport à la cohérence globale du morceau, en cherchant
d'autres sections et en gardant ces 2 candidats potentiels sans en acter un.
Puis on voit ce qui reste quand les autres sections ont mangé ce qu'elles ont à
manger. »

**Comment les autres sections arrivent** (lu dans le code, rien d'inventé) :
Louis trace plusieurs briques sur le chart, puis UN clic « Valider les
sections » envoie tous ses traits (`humain` + les reprises `propage` du tour
d'avant) à `/api/sections/inferer` → `occurrences_de_tous` (une recherche par
trait, dans l'ordre) → `remplir_par_algo` (quatre mots dans les trous). Le seul
endroit où TOUTES les sections connues sont là est donc la fin de
`occurrences_de_tous`, avant les quatre mots : c'est là que les suspens sont
tranchés. Ajouter une brique = re-cliquer Valider = tout est retranché.

**Implémentation** (`harmonia/sections/simulation.py`, commit 14cb9ad) :
- `maxima_locaux(c, Lb, candidats, tol)` : `tol` = la tolérance du morceau de
  la règle 4 (`EGALITE_5B`) ;
- `cherche_sur_matrices` : les départs candidats sont groupés (à moins d'une
  boucle de proche en proche) ; un groupe avec UNE façon de caser le plus de
  boucles est posé ; avec plusieurs, c'est un groupe EN SUSPENS
  (`detail["suspens"]`, alternatives exclusives, chacune avec sa fin de brique).
  Les zones en suspens sont réservées (pas de transposée dedans). Pour qui
  appelle une brique seule, le groupe est tranché avec ce que la brique sait ;
- `occurrences_de_tous` retire ce choix local, pose toutes les reprises sûres
  de toutes les briques, puis `resoudre_suspens` : (1) les boucles déjà prises
  par une autre section tombent, la plus complète reste ; (2) le plus de
  contacts avec les occurrences de la même brique (phase) ; (3) le moins de
  bouts libres de 1–3 mesures ; (4) le score. Les groupes que (1) tranche
  seul passent d'abord ;
- la route rend le choix en source `ressemble` : pointillés, compté « par
  ressemblance », PAS renvoyé comme contrainte au Valider suivant. Aucun
  changement d'UI (l'écran connaît déjà cette source).

**Mesuré** (page `docs/plots/occurrences_brique_v5_2026-09-30.html`,
`python -m tools.occurrences_brique_v5 --page`), sur le VRAI flux : chaque
découpage validé, 1re occurrence de chaque lettre tracée, toutes d'un coup.

| morceau, lettre | v4 j/r/t | 5b seule | v5 |
|---|---|---|---|
| Lazy Song A | 40/8/4 | 48/0/4 | 48/0/4 |
| She Will Be Loved A | 32/4/16 | 36/0/16 | 36/0/0 |
| She Will Be Loved B | 8/16/8 | 8/16/8 | 24/0/24 |
| She Will Be Loved C | 12/16/8 | 12/16/8 | 4/24/0 |
| Sunny A | 16/0/16 | 16/0/16 | **8/8/16** |
| Sunny C | 8/16/0 | 8/16/0 | 8/16/8 |
| **total** | 1232/218/81 | 1244/206/81 | 1244/206/81 |

The Walk inchangé (27/51 à 0,91 contre 0,996 : hors tolérance 0,049, la phase
reste 29…49). 14 groupes en suspens sur le corpus (She Will Be Loved 10,
X-yIEMduRXk 2, Sunny 1, The Walk 1), tous tranchés par « mangée » dans ce
flux. Rapport d'or : 60/60 identiques, 0 mesure changée.

**Ce que ça ne résout pas** :
- **Sunny A** (à arbitrer) : un suspens attend pendant que les reprises SÛRES
  des autres lettres se servent — y compris leurs jumeaux. La boucle de C (mêmes
  accords, 0,984) prend 18–25, qu'A (0,950) gardait avant seulement parce qu'il
  était tracé avant C. Options : (a) garder (la règle telle que dite) ; (b) seuls
  les TRAITS de Louis et les reprises des briques tracées AVANT mangent un
  suspens ; (c) un suspens n'est mangé que par une reprise plus haute que sa
  meilleure alternative de plus que la tolérance du morceau.
- Les « autres sections » sont les traits de Louis et leurs reprises sûres, pas
  les blocs que les quatre mots nommeront ensuite (ils ont besoin des
  occurrences pour tourner) ; le critère « bouts orphelins » n'en est qu'une
  approximation.
- Seuls des calages qui casent AUTANT de boucles sont suspendus ; 2 boucles
  contre 1 est acté par la couverture (Lazy Song), même si une autre section
  mange ensuite l'une des 2.
- Transposées, ancienne loi (< 8 mesures) et fin de brique des reprises sûres :
  inchangés.

## 12. Louis a choisi (c), 2026-09-30 (branché)

**Sa décision** (sur le coût Sunny A du § 11) : option (c) — une autre section
ne prend la zone d'un calage en suspens que si son score y est NETTEMENT plus
haut que celui du calage ; « nettement » = de plus que la tolérance du morceau
de la règle 4, la même qui définit l'égalité en 5b. Ses traits gagnent
toujours. Le résultat ne doit pas dépendre de l'ordre des briques.

**Implémentation** (`harmonia/sections/simulation.py`, `MANGE_NETTEMENT`) :
- chaque groupe en suspens porte sa `tolerance` (= `detail["egalite"]` de la
  brique, la 5b) ; aucune constante nouvelle ;
- `occurrences_de_tous` cherche TOUTES les briques d'abord, puis pose les
  reprises sûres. Une reprise sûre d'une brique qui recouvre un calage en
  suspens d'une AUTRE brique n'est posée que si son score dépasse celui de
  chacun des calages recouverts de plus que leur tolérance ; sinon elle est
  retenue. Un calage qui recouvre un trait de Louis ne protège rien (le trait
  le mangera). Puis `resoudre_suspens` (inchangé) ; puis les reprises retenues
  reviennent si leur zone est restée libre (le suspens a choisi un autre
  calage), dans l'ordre des traits dans les mesures — comme au premier passage.
  (Au score, Sunny donnait 26–33 à un jumeau de D, 0,959, au lieu du B juste,
  0,935 : essayé, écarté.)
- indépendance à l'ordre : toutes les zones en suspens sont connues avant la
  première reprise sûre ; testé en échangeant les places des traits A et C
  (`test_c_ne_depend_pas_de_l_ordre_des_briques`) et en inversant la liste.

**Mesuré** (colonne et bande « (c) » ajoutées à la page v5,
`docs/plots/occurrences_brique_v5_2026-09-30.html`, `python -m
tools.occurrences_brique_v5 --page`) :

| morceau, lettre | v5 j/r/t | (c) |
|---|---|---|
| Sunny A | 8/8/16 | **16/0/16** |
| Sunny C | 8/16/8 | 8/16/0 |
| X-yIEMduRXk A | 20/5/0 | **16/9/0** |
| X-yIEMduRXk C | 9/0/0 | 9/0/9 |
| X-yIEMduRXk D | 3/0/3 | 3/0/0 |
| **total** | 1244/206/81 | 1248/202/79 |

Tout le reste identique à v5 (Lazy Song, She Will Be Loved, The Walk).

**Ce que ça ne résout pas** :
- **X-yIEMduRXk** (à arbitrer) : la brique C (9 mesures) a deux calages en
  suspens dans le 2e A de Louis, 23 (0,938) et 27 (0,947) ; les reprises de A
  là (27–30 à 0,926, et la fin 31–35) sont plus BASSES, donc ne mangent plus
  le suspens. C prend 27–35 (jumeau), A perd 4 mesures. C'est la règle telle
  que dite : un jumeau en suspens plus haut qu'une vraie reprise la bat.
- Entre deux reprises SÛRES de briques différentes, le premier trait dans les
  mesures garde la priorité (inchangé ; `traits_propres` trie les traits, donc
  ce n'est pas l'ordre où Louis les a tracés, mais c'est un ordre).
- Les scores comparés ne sont pas à la même échelle quand les briques n'ont pas
  la même longueur (boucle de 4 contre brique de 9).

## 13. « Il manque une section avant un A : c'est une intro, pas un C » (branché)

Louis, 2026-09-30. Dans « Valider les sections », la zone libre de TÊTE
(mesure 1 … début de la première section posée − 1) sort désormais en UN bloc
« intro » au lieu d'une lettre neuve (`simulation.remplir_par_algo`). Périmètre
convenu : la tête seulement — les trous du milieu gardent leurs lettres, pas
de règle d'outro.

- UN bloc, même si l'algo y voyait deux blocs : deux « intro » de même
  longueur seraient repliées ensemble par `sections_pour_chart` (groupe sur
  label + longueur).
- L'intro ne prend pas de lettre : le premier bloc neuf après elle prend la
  première lettre libre (B si Louis n'a que A).
- Source « algo » : proposition recalculée à chaque Valider, jamais renvoyée
  comme contrainte.
- Tête muette (N.C.) : même règle ; chart.js la cache en lecture. Pas de double
  intro possible avec celle de la pipeline (`_one_section`) : la sortie de
  Valider remplace toutes les sections du chart.
- Pas de règle si rien n'est posé, ni si Louis a déjà posé sa propre « intro ».

**Mesuré** (lecture seule, 21 découpages validés lisibles sur 26 ; Louis est
supposé tracer la 1re occurrence de chaque lettre SAUF l'intro, puis le vrai
flux tourne ; script `scratchpad/intro_tete.py` de la session) :

| cas | morceaux |
|---|---|
| intro rendue = la sienne, aux mêmes mesures (9) | Bora Bora 1-4, Urdlvw0Ssec 1-11, Chain Of Fools 1, Stand By Me 1-6, Sunny 1, Let It Be 1-4, Every Breath 1-8, Be My Baby 1-4, Yesterday 1-2 |
| intro rendue sur des mesures qu'il a laissées sans section (2) | Lazy Song 1 (N.C.), Goodbye Yellow Brick Road 1-2 — l'app les nomme déjà « intro » (`comblerTrous`) |
| intro trop courte (1) | Don't Know Why : 1-2 au lieu de 1-4 (une reprise de sa « queue » prend 3-4) |
| intro RATÉE : une reprise d'une brique prend la mesure 1, il n'y a pas de zone libre (4) | San Francisco Street (B 1-4), She Will Be Loved (D 1-4), This Love (A 1-4), The Walk (A 1-4) |
| pas d'intro chez lui, aucune rendue (5) | Lost Without U, My Cherie Amour, American Boy, Easy On Me, Claustrophobic |

Aucune intro fausse. **Ce que ça ne résout pas** : quand l'intro joue la
matière d'une section (les 4 ratées), la reprise la mange avant que la règle
puisse s'appliquer — c'est « une intro ne se rejoue pas » vu de l'autre côté,
et il faudrait qu'une reprise en mesure 1 cède la tête. Non fait (hors
périmètre). Rapport d'or : 0 mesure changée (Valider n'est pas rejoué par lui).
