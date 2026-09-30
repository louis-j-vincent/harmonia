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
