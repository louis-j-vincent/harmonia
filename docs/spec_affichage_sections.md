# Comment on affiche les sections — la façon iReal

Spec arrêtée le 2026-09-14 avec Louis, après qu'un chart ait affiché **deux
blocs « A »** (Sunny Afternoon : un A de 8 mesures et un A de 6). Elle
complète `docs/this_love_target_spec.md` (sa feuille écrite à la main du
2026-07-29), qui posait déjà 5 principes mais ne disait pas quoi faire quand
deux passages d'une même lettre n'ont pas la même longueur.

## Le principe

**Une lettre nomme UN SEUL bloc écrit.** La répétition est une *marque*
(`×N`, reprise, 1ʳᵉ/2ᵉ fin, coda), jamais une deuxième copie de la lettre.

C'est la convention iReal Pro / Real Book. Vocabulaire iReal, lu dans le
parseur qu'on utilise déjà (`pyRealParser`, et notre `harmonia/irealb_export.py`) :

| Signe | Sens |
|---|---|
| `*A` | marqueur de section (la lettre dans un cadre) |
| `{ }` | reprise : écrit une fois, joué deux fois |
| `N1` `N2` | 1ʳᵉ / 2ᵉ fin (les crochets « 1. » « 2. ») |
| `S` `Q` | segno / coda — les retours à grande échelle |
| `x` `r` | mesure (ou double-mesure) répétée |
| `( )` | **accord alternatif** — la « lecture optionnelle » d'une case |
| `< >` | commentaire texte au-dessus de la portée |
| `n` | pas d'accord |

## La cascade

Plusieurs passages portent la même lettre. On applique dans l'ordre, le
premier cas qui colle gagne :

| # | Les passages… | On écrit | État |
|---|---|---|---|
| 1 | sont identiques | un bloc + `×N` | ✅ en place (`reps`) |
| 2 | même longueur, seules les 1–2 dernières mesures diffèrent, **et la queue change de fondamentale** | un bloc + **fins 1./2.** | ✅ rendu en place, émis depuis 2026-09-14 |
| 3 | l'un est l'autre **coupé** (il s'arrête en cours de boucle) | un bloc + `×N`, le passage court n'allume que ses mesures | ✅ 2026-09-14 |
| 4 | l'un **ajoute** des mesures à la fin | un bloc + les mesures en plus (coda / tag) | ❌ à faire |
| 5 | diffèrent ailleurs qu'à la queue | **A′** (puis A″) — jamais un deuxième « A » nu | ✅ 2026-09-14 |

**Invariant dur, vérifié par `tests/test_sections_ireal.py` :** dans un chart
rendu, deux blocs ne portent jamais la même étiquette. Si la cascade n'a pas
réduit, la règle 5 force la marque distinctive.

## Ce que ça donne sur les morceaux de référence

- **Sunny Afternoon** — les deux « A » n'étaient pas deux sections : l'unité
  réelle est une cellule de 4 mesures (`B♭m7 E♭7 | E♭7 | A♭maj7 D7 | D♭maj7 C7`),
  le bloc de 8 la joue deux fois, celui de 6 une fois et demie. → règle 3, un
  seul A, le dernier passage s'arrête au milieu.
- **This Love** — règle 2 : B et C ont chacun une 1ʳᵉ et une 2ᵉ fin. C'est
  exactement la feuille de Louis du 2026-07-29.
- **She Will Be Loved** — le C de 4 mesures et le B de 8 tombent en règle 5 :
  lettres distinctes obligatoires. C'est la garde qui manquait à
  `merge_similar_letters` (2026-09-14), qui les avait confondus.

### Deux garde-fous appris en codant (2026-09-14)

- **Une vraie 2ᵉ fin change d'accord, pas de couleur.** Le premier jet
  suspendait un crochet « 1./2. » sur le refrain de Sunny Afternoon parce que
  sa dernière mesure lisait `C7` deux fois et `Cm7` une fois — or cette
  mesure-là est justement celle que le repli met hors de la pile (la cadence
  de fin de section), donc son accord vient du premier jet, passage par
  passage. On exige maintenant que la queue change de **fondamentale**
  quelque part (This Love : `B♭ E♭` contre `A♭ G`).
- **Un bloc qui porte déjà des fins n'absorbe pas un passage coupé** (règle 3
  cède à la règle 5) : les deux réécrivent `barSpans`, et les combiner
  demanderait un troisième contrat de mise en page. Le passage coupé prend
  donc un prime dans ce cas-là.

## Ce que ça ne règle PAS

- La règle 4 (passage rallongé) n'est pas codée : un passage plus long que le
  bloc tombe en règle 5 et prend un prime, alors qu'une coda serait plus juste.
- Les accords alternatifs `( )` d'iReal ne sont pas écrits par la pipeline.
  Le champ existe pourtant côté chart (`var` sur un accord) et côté app, et
  c'est la forme qui conviendrait à la proposition de Louis du 2026-09-14
  (« écrire D♭, et C7 en optionnel au-dessus ») quand une position reste à
  égalité stricte.
- La cascade ne juge pas le **découpage** ; elle ne fait que l'écrire. Une
  section mal détectée reste mal détectée, elle sera juste nommée proprement.
