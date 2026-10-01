# Une mesure de ressemblance entre deux accords (2026-10-01)

## v2, BRANCHÉE (2026-10-01, après les verdicts de Louis) — commit 6271b31

Louis sur la page : « G7 différent de Gm7, ce sont des familles différentes,
il faut faire peser la tierce. Un sus4 est ambigu entre mineur et majeur par
essence, donc proche des deux » ; « This Love, les mesures 4 et 8 sont la même
chose » ; « Dm7 tenu » contre « Dm7 | G7 » : « c'est au cas par cas […]
l'utilisateur peut choisir » (le curseur « Section merging »).

**La formule** (`chord_similarity.v2`) : on part de notes7, le cosinus des
notes qui sonnent sans les couleurs 9/11/13. La fondamentale est celle qui
sonne (`roles.accord_sonnant`). Ensuite la tierce compte :
- même fondamentale et tierces contraires → ×0,4 ;
- même fondamentale et une seule tierce (une sus) → on compare sans la case
  de la tierce (2 à 5 demi-tons) ;
- deux fondamentales qui ont chacune une tierce, sans que l'un des accords
  soit contenu dans l'autre → ×0,8.

| paire | notes7 | **v2** | plus petit cran qui fusionne |
|---|---|---|---|
| G7 / Gm7 | 0,75 | **0,30** | 4 (jamais avant « tout ») |
| G7sus4 / G7 | 0,75 | **1,00** | 1 |
| G7sus4 / Gm7 | 0,75 | **1,00** | 1 |
| D-7 / G7sus4 | 0,75 | **0,75** | 1 |
| B♭/C / C7sus4 | 0,75 | 0,75 | 1 |
| C / F | 0,33 | **0,27** | 4 |
| Fm/A♭ / Dø7 (This Love mes. 4 ≡ 8) | 0,87 | 0,87 | 1 (Fm/A♭ ⊂ Dø7) |
| C^7 / Am7, F7 / D-7, D^7 / F♯-7 | 0,75 | 0,60 | 3 |
| Dm7 / G7, G7 / D♭7 | 0,50 | 0,40 | 3 |
| C / Fmaj7 | 0,58 | 0,46 | 3 |
| Fm / F7, Am / A | 0,58 / 0,67 | 0,23 / 0,27 | 4 |
| A♭ / B♭, G / E♭ (This Love B) | 0 / 0,33 | 0 / 0,27 | 4 |

Le ×0,8 « deux tierces sur deux fondamentales » n'était pas demandé. Il est
venu du rapport d'or. Sans lui, avec seulement la tierce de même
fondamentale, deux décisions par défaut cassaient :
- **San Francisco Street** : le turnaround Gb-7 Gb-7 G^7 G^7 passait pour la
  boucle D^7 D^7 E-7 E-7 (I^7 ≈ iii-7 à 0,75), et la carte de lecture d'un A
  coupé se décalait de 8 mesures ;
- **Don't Know Why** : la 2e fin de B (D-7 au lieu de F7) disparaissait.

L'inclusion garde ce que Louis a validé : Fm/A♭ ⊂ Dø7 (This Love), B° ⊂ G7,
F ⊂ D-7.

**Branchement.** `roles.meme_lecture(a, b, contexte, grain="famille")` rend
`v2 ≥ seuil du cran`. Le cran vient de `contexte["cran"]`, sinon de
`roles.cran_de_fusion(n)` (posé par `soudure.sections_pour_chart` pour chaque
cran du curseur), sinon du défaut. `folding.meme_harmonie` tolère un décalage
de rythme du cran : chaque temps peut trouver son accord ±d temps plus loin
dans l'autre mesure. Au cran ≥ 3, la dernière section d'une AUTRE lettre
peut rejoindre une lettre comme son dernier tour
(`CRAN_DERNIER_TOUR_AUTRE_LETTRE`).

| cran | seuil v2 | décalage de rythme toléré | en plus, dans `folding` |
|---|---|---|---|
| 0 | 0,72 | 0 | pas de boucle, les fins de couleur se gardent |
| 1 (défaut) | 0,72 | 0 | — |
| 2 | 0,63 | ±1 temps | fins proches fusionnées (inchangé) |
| 3 | 0,38 | ±2 temps | fins apparentées ; une fin d'une autre lettre peut devenir le dernier tour d'une lettre |
| 4 | 0,38 | ±2 temps | tout en un bloc ×N (inchangé) |

**Rapport d'or** (contre une cuisson de HEAD avant branchement,
`state/cache/golden/avant_simil`) : 52/63 identiques, 3 mesures changées.
Seuls deux morceaux changent au défaut (page
`docs/plots/similarite_avant_apres/avant_apres.html`) :

| morceau | au défaut, avant | au défaut, après | pourquoi |
|---|---|---|---|
| This Love | intro écrite en 8 | intro en boucle de 4 ×2 | mes. 8 Dø7 ≡ mes. 4 D° Fm/A♭ (confirmé par Louis) |
| She Will Be Loved | A′ d'une mesure (mes. 33, A♭^7) | rejoint A | Cm ⊂ A♭^7 : la basse descend sous Cm. **À écouter** |

Inchangés au défaut : Claustrophobic (fusionne toujours aux crans de
fusion), Julia, The Walk, Let It Be, et les deuxièmes moitiés de This Love.
Changements aux crans 3-4 seulement :
- **Lost Without U** : l'outro rejoint A (le test `test_lost_without_u_*`
  le vérifie).
- **Be My Baby** : l'outro rejoint C.
- **The Walk** : l'outro rejoint B ; B et bridge en boucle.
- **San Francisco Street** : A″ rejoint A, B en boucle.
- **Bein Green** : le A′ rejoint A.
- **Easy On Me** : A en boucle.
- **Claustrophobic** : B en boucle.

**Ce que v2 ne résout pas, en plus de la liste plus bas :**
- les grains « exact », « fonction », « racine » gardent leurs règles. Une
  2e fin se garde toujours si elle change de FONDAMENTALE : C7 contre Cm7
  seul n'en fait pas une, alors que v2 les dit loin. Les crans 2-3 des fins
  gardent aussi leur cosinus par mesure (≥ 0,90 / 0,70) ;
- l'inclusion est stricte : Fm6 ⊂ Dø7 oui, mais un accord avec une note de
  plus d'un côté et une de moins de l'autre prend le ×0,8 ;
- le décalage de rythme ne joue qu'à l'intérieur d'une mesure, jamais d'une
  mesure à l'autre ;
- les entrées `old_/new_` d'un premier passage d'avant/après (San Francisco
  Street, Don't Know Why, avant le ×0,8) sont restées dans la bibliothèque,
  sans `--clean`.

---

## Étape 1-2 (avant les verdicts) — gardé pour l'historique

Louis : « il faut une mesure de similarité / de distance reproductible et
généralisable […] Pense aussi à prendre en compte l'harmonie du contexte. »

Page à écouter : `docs/plots/similarite_accords.html`
(http://100.89.209.63:7772/plots/similarite_accords.html). Code :
`harmonia/chord_similarity.py`. **Rien n'est branché dans le repli** : Louis
arbitre d'abord sur la page. Le point de branchement prévu est
`harmonia.roles.meme_lecture(a, b, contexte)` (commit d6b471e).

## En une ligne

**Recommandation : `notes7`.** C'est le cosinus des notes que l'accord fait
sonner (basse comprise), après avoir retiré les couleurs 9, 11 et 13. Pas de
table, pas de réglage. Seuil par défaut proposé : 0,71.

## Les candidates, et ce que chacune a montré

Les valeurs ci-dessous : D-7/G7sus4 (à rapprocher), C/F (à éloigner), Fm/F7
de This Love (à éloigner). Le seuil de chaque candidate est sa plus petite
valeur sur les 5 paires que Louis a déclarées équivalentes (D-7/G7sus4,
B♭/C/C7sus4, C7/C9, C/C^7, G7/G). Il est posé sans regarder les morceaux.

| candidate | D-7/G7sus4 | C/F | Fm/F7 | ce qu'elle montre |
|---|---|---|---|---|
| notes (cosinus chord-tone, la ressemblance d'aujourd'hui) | 0,75 | 0,33 | 0,58 | sépare bien ; mais C9 contre C7sus4 tombe à 0,67 : les couleurs diluent |
| **notes7** (la même, sans 9/11/13) | **0,75** | **0,33** | **0,58** | sépare avec 0,17 de marge ; C7 = C9 = C13 |
| notes_basse (basse ×2) | 0,57 | 0,33 | 0,77 | **casse le cas de départ** : D-7/G7sus4 change de basse |
| tiv (DFT des notes, espace tonal) | 0,86 | 0,61 | 0,79 | sépare de 0,07 seulement, et rapproche C de F (voisins par quintes) : il mesure une tonalité, pas un accord |
| musx (appris, 310 morceaux) | 0,026 | 0,007 | 0,04 | musx n'hésite presque jamais entre deux fondamentales ; ses hésitations sont les couleurs (C7/C9 0,54, Dø/D° 0,68) et les accords de même basse |
| fonction (T/PD/D dans la tonalité locale) | 0,50 | 0,00 | — | grossière : ii et V7sus4 n'ont pas la même fonction de manuel ; C/Em = 1 mais C/Am = 0,5 |
| notes+fonction, notes+musx | 0,62 / 0,39 | 0,17 / 0,17 | 0,58 / 0,31 | les mélanges n'apportent rien que notes7 ne fasse déjà |

### La table musx, en détail

`tools/chord_similarity_musx.py` reconstruit
`harmonia/assets/chord_similarity_musx.json` (131 Ko, environ 10 s). Méthode :
les six têtes de musx sont combinées comme le fait son propre décodeur. On
obtient ainsi une probabilité par étiquette (301 étiquettes), image par
image. On mesure ensuite combien de probabilité musx met sur deux étiquettes
**au même instant**, mis en commun par intervalle. Le résultat est
reproductible : r = 0,99 entre deux moitiés du corpus, r = 0,94 entre
GuitarSet (guitare seule) et le reste.

- **Ce n'est pas du bruit, c'est l'architecture de musx.** musx décide
  d'abord une basse et une triade, puis les couleurs une par une. Ses
  hésitations entre deux fondamentales sont donc toutes des hésitations
  **à basse égale** : A-/C ≈ C (0,19), C/G ≈ G (0,17). D-7 et G7sus4 n'ont
  pas la même basse, donc il ne les confond pas.
- Température : à 1, l'échelle du décodeur (c'est la table gardée).
  Aplatir à 2 ou 4 ne change pas le classement : D-7/G7sus4 reste la paire
  à deux fondamentales la plus proche (0,08 à T=2, 0,29 à T=4), mais tout
  remonte avec elle (C/F 0,03 puis 0,19).
- Version « propre au morceau » (le contexte = les hésitations de musx
  dans CE morceau), vérifiée vite et rejetée. Sur Lost Without U, D-7/G7
  vaut 0,06 seulement. Et des accords absents du morceau sortent hauts :
  A♭/B♭ vaut 0,38, parce qu'un cosinus sur des probabilités minuscules ne
  veut rien dire.

### Le contexte

Louis a demandé de penser au contexte. Trois tentatives :

1. **La fonction tonale.** Elle ne rapproche pas D-7 de G7sus4 (0,5) : au
   sens du manuel, un ii est prédominant et un V7sus4 est dominant (ou à
   cheval sur les deux).
2. **Les voisins** (`notes7+voisins` sur la page : chaque temps est lissé
   avec ses voisins, poids ¼ ½ ¼). Ça ne change aucune décision sur les
   trois morceaux.
3. **Le contexte que la décision fournit déjà.** Pour savoir si deux
   passes sont pareilles, on compare des passes d'une **même lettre**,
   temps par temps. La tonalité et les voisins sont donc déjà les mêmes des
   deux côtés. La mesure n'a qu'une question à trouver : « ces deux accords
   peuvent-ils tenir la même case ? ».

## Ce que ça change sur les trois morceaux (notes7, seuil 0,71)

Tableau de la page, partie 3. « Actuel » = `meme_lecture` au grain de
chaque décision, appliqué temps par temps comme les candidates.

| morceau | décision | aujourd'hui | notes7 | à partir de quel cran ça fusionne |
|---|---|---|---|---|
| Lost Without U | outro (G7sus4 \| G7 \| C \| C Gm7) contre le 2e tour du A | différent | différent | 3 |
| Lost Without U | boucle du A (mes. 5–8 contre 1–4) | différent | différent | 3 |
| Claustrophobic | fin C9 F7 contre C7sus4 F^7 | même | même | 1 |
| Claustrophobic | fins B♭m7… et B♭7 F^7 | différent | différent | 3 |
| This Love | 2e moitié de B (Cm F7 \| A♭ G contre Cm Fm \| B♭ E♭) | différent | **différent** (A♭/B♭ = 0) | 4 |
| This Love | 2e moitié de A, du bridge | différent | différent | 4 |
| This Love | **intro** (mes. 8 Dø7 contre mes. 4 D° puis Fm/A♭) | différent | **même** (0,87) | 1 |

Ce qui bloque Lost Without U, ce n'est pas D-7/G7sus4. Sur la page, ce
couple vaut bien « même accord » (outro mes. 1, A mes. 8). Ce qui bloque :
(1) mes. 2 du A, « Dm7 tenu » contre « Dm7 | G7 » : c'est un décalage de
rythme harmonique, et la mesure le lit comme Dm7 contre G7 (0,50) ; (2) mes.
4 de l'outro : le C tenu de la cadence finale contre Fmaj7 (0,58).

Dans le décodage brut, des mesures basculent sans changer de décision :
G7sus4 ≈ Dm7, G7sus4 ≈ Gm7, G7 ≈ Gm7, E♭7sus4 ≈ B♭m7 (le même cas que
D-7/G7sus4, en E♭).

## Définition reproductible

- `notes_jouees(c, couleurs=False)` : les notes de `roles.pcs(c)`, basse
  comprise. Lues depuis la fondamentale écrite, on retire la 9 et la 11 si
  l'accord a une tierce, la 9 et la 13 s'il a une septième mineure. La
  basse n'est jamais retirée.
- `notes7(a, b)` = cosinus des deux vecteurs 0/1 sur 12 notes. Deux N.C. :
  1. Un N.C. contre un accord : 0.
- Constantes versionnées dans le module : `VERSION = 1`, `CRANS`,
  `W_BASSE`, `TIV_POIDS`, `SEUIL_DIATONIQUE`. La table musx porte sa propre
  `version`, et `corpus_sha1` (la liste des fichiers). Aucun hasard : liste
  de fichiers triée, pas d'image fixe.
- Tests : `tests/test_chord_similarity.py`. Ils vérifient la symétrie,
  l'invariance par transposition (pour toutes les candidates), que les
  paires « proche » restent au-dessus des paires « loin » avec 0,1 de
  marge, B♭ plus près de Gm que de F, et que les seuils de cran sont à au
  moins 0,02 de toute valeur réelle.

## Proposition pour le curseur « Section merging » (5b521c8, a9500d3)

Les crans actuels mélangent deux règles. Le cran 1 compare la famille ;
les crans 2 et 3 prennent le cosinus des notes par mesure (≥ 0,90 / 0,70)
sur la queue. La proposition met **une seule règle** : notes7, temps par
temps. Une mesure vaut son pire temps. Chaque seuil tombe au milieu d'un
trou des valeurs que notes7 peut prendre :

| cran | seuil | ce qu'il ajoute (exemples) |
|---|---|---|
| 0 | mêmes notes (`notes` = 1) | rien : C7 ≠ C9 ; D-7 = F6 (mêmes notes) |
| 1 (défaut) | ≥ 0,71 | couleurs, D-7 ≈ G7sus4, B♭/C ≈ C7sus4, C7 ≈ C7sus4, Gm7 ≈ G7sus4, G7 ≈ Gm7, C^7 ≈ A-7, G7 ≈ B° |
| 2 | ≥ 0,63 | relatif mineur (C ≈ Am, C ≈ Em), tierce changée sur une triade (Am ≈ A), C ≈ Csus4 |
| 3 | ≥ 0,47 | triton (G7 ≈ D♭7), ii ≈ V (Dm7 ≈ G7), Fm ≈ F7 |
| 4 | tout | un bloc ×N |

`chord_similarity.cran(a, b)` donne le plus petit cran où deux accords se
valent.

## Ce que ça NE résout PAS

1. **Le rythme harmonique.** « Dm7 | G7 » contre « Dm7 | Dm7 » est un
   accord décalé, pas deux accords voisins. C'est ce qui bloque vraiment
   Lost Without U. Il faut une autre brique, qui aligne la suite d'accords,
   pas une meilleure ressemblance entre deux accords.
2. **La tierce ne pèse pas plus qu'une autre note.** Tout couple de
   quatre notes qui en partagent trois vaut 0,75 : G7 ≈ Gm7 autant que
   D-7 ≈ G7sus4. On a testé une variante « tierce × 2 » : G7/Gm7 tombe à
   0,43, mais Gm7/G7sus4, C7/C7sus4 et B♭/C/C7sus4 tombent avec (0,57).
   Elle n'est donc pas retenue. C'est à Louis de trancher.
3. **Un accord de passage en plus** (« A♭ » contre « A♭ G »). Ça reste une
   différence. C'est la tolérance de structure de `folding.meme_moment`,
   qui doit rester à côté de la mesure, pas dedans.
4. **Le contexte tonal n'entre pas.** C'est voulu pour comparer des passes
   d'une même lettre. Pour chercher une reprise entre deux lettres, ou une
   section transposée, il faudrait une version qui tient compte de la
   tonalité (ou transposer d'abord).
5. **Les accords du décodage ne sont pas tous justes.** La mesure compare
   des étiquettes, pas l'audio. Si musx a écrit un G7 là où l'on entend un
   Dm7/G, la mesure ne peut pas le corriger.

## Questions pour Louis

1. L'intro de This Love : la mes. 4 (D° puis Fm/A♭) et la mes. 8 (Dø7),
   même chose ? Si oui, notes7 la factorise en 4 mesures ×2.
2. G7 ≈ Gm7 (0,75, cran 1) : acceptable au défaut, ou la tierce doit-elle
   peser plus, au prix de Gm7/G7sus4 et C7/C7sus4 ?
3. « Dm7 tenu » contre « Dm7 | G7 » (Lost Without U, mes. 2) : est-ce la
   même mesure pour toi ? Si oui, le prochain chantier est l'alignement du
   rythme harmonique, pas la ressemblance.
