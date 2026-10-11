# Carte du pipeline « une brique tracée → les autres sections » (2026-10-11)

Vérifiée dans le code et exécutée sur La Jupe En Laine (`min_ujgPfUZ7Whc`),
brique unique **B 18–25** (1-indexé). `sim` = `harmonia/sections/simulation.py`,
`VS` = `harmonia/sections/similarity.py`. Numéros de ligne au 2026-10-11 (à
revérifier avant de s'y fier). Page de plan : `docs/plots/plan_explication_sections_2026-10-11.html`.

La Jupe : 1 N.C. | 2–9 A | 10–17 A | 18–25 B | 26 C- | 27–34 A | 35–42 B | 43–50 B |
51 C- | 52–59 A | 60–67 B | 68–75 B | 76–83 B | 84 C- | 85–91 A.

| # | étape | code | ce qui est rendu / à instrumenter | La Jupe |
|---|---|---|---|---|
| 0 | suggestions pâles (AUTRE moteur, avant Valider) | `sections_editor.js` `proposerReprises` → `/api/section-repeats` → `section_tool.find_repeats` | rend `curve`, `tried`, `occurrences` ; courbes tournées non rendues | 3 B sur 5 (parité 2 mesures écarte 35, 43) ; scores tournés > 1 |
| 1 | envoi des traits | `validateSectionSelection` (js) | `humain` = sources humain+propage | `[B 18–25]` |
| 2 | nettoyage | `routes/sections.py sections_inferer` → `sim.occurrences_de_tous` → `soudure.traits_propres` | `gardes`, `perdus` | `(17,24,B)` |
| 3 | vecteurs chord-tone + SSM | `VS.harmonic_vectors`, `VS.ssm` | V n×12, S n×n | 91×12, 91×91 |
| 4 | boucle minimale de la brique | `sim.boucle_minimale`/`_periode`/`_niveau_brique` | rend `{boucle,suite,sous_periode,tau}` ; **r par décalage l non rendu** (boucle `for l` de `_periode`) | τ 0,686 ; l=4 : S=[1,0;0,703;0,052;0,425], r=2 → boucle 8 |
| 5 | courbe glissante (+ ancienne loi si L<8) | `VS._slide` dans `sim.cherche_sur_matrices` | `d["courbe"]`, `d["mode"]` ; **`cherche_brique` ne recopie ni courbe ni pics** | pics 35 .996, 43 .996, 60 .997, 68 .988, 76 .998 ; fond 0,15–0,69 |
| 6 | proéminence sur fond local | `sim._fond` | `d["pics"]=[{p,c,fond,rho,ok,pris}]` ; ok si c≥0,85 et rho≥0,6 | rho ≈ 0,99, fond ≈ 0,37 |
| 7 | référence du morceau | `sim.reference_du_morceau` | `d["reference"]` | R .998, trou .008, tol .02, pas de coupe |
| 8 | maxima locaux (5b) | `sim.maxima_locaux` | `pas_maximum`, `d["egalite"]` | 5/5 maxima |
| 9 | couverture + suspens | `sim._selections` | `d["suspens"]` ; **composantes `comps` non rendues** | aucun suspens |
| 10 | fin de brique | closure `fins()` | **st/sl non rendus** | sans objet |
| 11 | transposées | bloc rot k=1..11 | **courbes ck non rendues** | meilleur 0,627, aucune |
| 12 | paquet de la brique | `cherche_brique` | `out`, `info` | 5 reprises |
| 13 | arbitrage entre briques + suspens global | `sim.occurrences_de_tous`, `sim.resoudre_suspens` | `occ`, `diagnostics[i]["suspens_global"]` ; **`tenues`/`retenues` non rendues** | 6 B, rien à arbitrer |
| 14 | mot de bi-mesures | `soudure.mot_sur_traits`, `_matrice_bimesures`, `_otsu`, `_mot_depuis` | `bornes`, `mot` ; **seuil et germes locaux** | 44 jetons, Otsu 0,568, `a` = 22/44, jetons de 3 mesures 15–17/32–34/57–59 (décalés d'une mesure) |
| 15 | jetons gelés | `sim.remplir_par_algo` | **`depart`/`geles` locaux** | 6 B gelés `adac` |
| 16 | soudures « quatre mots » | `phrases4.phrases`, `merges4`, `cout` | `steps`, `couts` ; **locaux dans remplir_par_algo** | (a,b)×5,(ab,a)×5,(aba,c)×5 ; coût 4 = 19 vs 6 = 34 |
| 17 | nommer, primes, restes | `phrases4.nommer`, `grouper_restes` | locaux | A,A,B,A,B,B,A,B,B,B,A |
| 18 | B + queue par préfixe | `sim._prefixe_et_queue` | local | non déclenché |
| 19 | lettres neuves + intro de tête | `remplir_par_algo` | **liste avant queue_optionnelle locale** | 1–17 → intro (l'algo y voyait 2 A) |
| 20 | queue optionnelle (de97480) | `sim.queue_optionnelle` | `propositions` seul ; **hotes/groupes/retenus locaux** | avec 26/51/84, sans 43/68 → queues 26, 51, 84 |
| 21 | dernier tour | `sim.nommer_dernier_tour` → `folding.dernier_tour_des_passes` | rend `{ouverture,…}` | sans effet |
| 22 | réponse de la route | `routes/sections.py` | sources humain/propage/ressemble/algo ; `queues_proposees` jamais lu côté client ; `reste` toujours False | intro·B·q·A·B·B·q·A·B·B·B·q·A |
| 23 | client + écriture du chart | `comblerTrous`, `/api/soudure/valider`, `scinder_multiples`, `refold`, `sections_et_fusions`/`sections_pour_chart` | `rapport[label]={period,changed,reason}` | B empilé (22 mes.), A refusé 0,83<0,85, intro refusée 0,61 |

## Les 5 décisions où l'algo tranche à la place de Louis
1. Intro de tête (19) — avale ce qui précède la 1re brique.
2. Mot de bi-mesures (14–17) — seuil Otsu, calage des jetons, nommage sur mot exact.
3. Proéminence ≥ 0,6 et référence du morceau (6–7) — vraie reprise vs jumeau.
4. Suspens et option (c) (9, 13) — arbitrage le plus opaque.
5. Queue optionnelle (20) — ≥ 2 « avec », ≥ 1 « sans ».

## Docstrings contredits par le code (voir known_issues 2026-10-11)
simulation.py (remplir_par_songformer), routes/sections.py (transposées), sections_editor.js
(« propage = exactement identique »), `RESSEMBLE_MIN`/`_plus_proche` morts,
`sections_etat_2026-09-30.md` §1 (ancienne loi), docstring de `queue_optionnelle`
(seuls les `trace` sont protégés, pas les `propage`).
