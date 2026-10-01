# Une mesure de ressemblance entre deux accords (2026-10-01)

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
