# Les décisions qu'on prend sur les accords — inventaire du code réel

Audit read-only, 2026-09-16. **Tout ce qui suit vient de la lecture du code, pas
des docstrings.** Quand une docstring contredit le code, le code gagne et la
contradiction est signalée (§5).

État du dépôt lu : branche `feat/section-criteres`, HEAD `58a1f30`, plus **une
modification non commitée de `harmonia/pipeline.py`** (l'étape basse sonnante,
marquée ⚠️NON-COMMITÉ partout où elle apparaît).

Chemin suivi : `harmonia/pipeline.py::analyze_steps` (ligne 384), de bout en bout.

---

## 1. Le résumé en un écran

**64 points de décision** entre le fichier audio et un symbole d'accord imprimé
dans une mesure.

| Provenance | Nombre | Part | Ce que ça veut dire |
|---|---:|---:|---|
| **arbitré** | 25 | 39 % | remonte à un jugement daté de Louis |
| **posé** | 22 | 34 % | un nombre choisi sans trace de mesure ni d'arbitrage dans le dépôt |
| **mesuré** | 12 | 19 % | réglé sur une mesure de corpus |
| **hérité** | 5 | 8 % | vient de musx / Beat This! / SongFormer / Krumhansl — pas notre choix |

**Ces chiffres flattent la réalité, et voici pourquoi.** Trois précisions, dans
l'ordre d'importance :

1. **« Arbitré » veut presque toujours dire : la FORME de la règle a été
   arbitrée, son NOMBRE a été posé.** Louis a dit « ne pas empiler les fins de
   sections » — c'est un arbitrage. Que ce soit `FIN_SECTION_HORS_PILE = 1` et
   pas 2, c'est posé (et le commentaire dit lui-même que Louis avait dit « souvent
   les 2 dernières »). Même chose pour la façade (`≥ 3` passes), pour le bloc
   (`min(4, …)` mesures), pour les crochets de fin (`≤ 2` mesures).
2. **Un seul bloc est arbitré au sens FORT** — verdicts écrits sur disque et
   rejoués par un test qui rougit si la règle bouge : `harmonia/bass_rules.py`
   (`state/human/bass_verdicts.json` : 40 cas dont 33 tranchés, rejoués par
   `tests/test_bass_rules.py`, 33 cas paramétrés). Les 24 autres décisions
   « arbitrées » reposent sur une citation datée dans un commentaire. C'est une
   vraie preuve, mais **rien ne devient rouge si on change la règle**.
3. **Une seule mesure sur douze est reproductible depuis le dépôt.**
   `GRID_MIN_COVERAGE` / `GRID_MIN_DIRECT` le sont (`archive/scripts/beat_grid_report.py`
   importe les fonctions vivantes). Pour toutes les autres, le chiffre est dans
   un commentaire et le script a disparu : `scripts/` ne contient plus aucun
   source (seulement un `__pycache__` périmé), et la reproduction des règles de
   basse pointe vers **deux URL `claude.ai/artifact/…`**, pas vers du code
   (`docs/bass_slash_rules.md:211-217`). Un seuil « mesuré » qu'on ne peut pas
   re-mesurer se comporte comme un seuil posé dès qu'un composant change.

Et la règle #5 de CLAUDE.md (« un résultat sur un morceau est une hypothèse »)
s'applique telle quelle à `PERIOD_MIN_SCORE`, `STACK_COHERENCE`, `OUTLIER_Z`,
`MERGE_LETTERS_COS`, `BASS_ONSET_S` (1 morceau), `DEFAULT_PENALTY` (7), et à
tout le bloc de `harmonic_key.py`.

---

## 2. LE TITRE — les constantes `posé` qui peuvent changer un accord imprimé

Classées par le nombre de mesures qu'elles touchent. Ce sont les nombres à
arbitrer en priorité, parce qu'ils décident du contenu du chart sans que
personne n'ait jamais dit pourquoi. **★ = touche chaque accord de chaque
morceau.**

| # | Constante | Valeur | Où | Ce qu'elle décide | Si elle est fausse |
|---|---|---|---|---|---|
| ★ | `DEFAULT_PENALTY` | `40.0` | `harmonia/musx.py:53` | **à quelle fréquence les accords changent, sur tout le morceau** — c'est la pénalité Viterbi de changement d'accord | Le commentaire dit « pooled optimum 40 ; LOSO-stable (40 sur 6/7 morceaux, 55 sur le 7ᵉ) ». **Aucune trace de cette étude nulle part** : zéro résultat pour `pooled optimum`, `LOSO`, `penalty=40` dans `docs/*.md` et `archive/scripts/*.py`, et le corpus des 7 morceaux n'est pas nommé. Introduit le 2026-07-27 (`f36c327`). Le défaut du décodeur vendu est `30.0` — on l'a donc **surchargé** sur la foi d'une mesure invérifiable. Trop haut : accords collés, on rate les changements ; trop bas : hachis. |
| 1 | `STACK_COHERENCE` | `0.85` | `harmonia/folding.py:89` | si une lettre entière se replie ou non — donc si TOUTES ses mesures sont réécrites par le gabarit ou gardent leur premier jet | le levier le plus violent du chart : au-dessus, une section entière est réécrite ; en dessous, aucune. Mesuré sur 3 morceaux. |
| 2 | `PERIOD_MIN_SCORE` | `0.80` | `harmonia/folding.py:76` | si une section a une boucle interne, et donc la taille de la pile | une boucle ratée = zéro empilement (accords bruités gardés) ; une boucle inventée = des mesures différentes écrasées ensemble |
| 3 | `OUTLIER_Z` | `3.0` | `harmonia/folding.py:77` | quelle mesure sort de la pile (« variante ») et garde son propre décodage | trop bas : les cadences réelles sortent et le chart perd sa consistance ; trop haut : une cadence unique se fait écraser par les mesures de pompe |
| 4 | `CV_MAX` | `0.51` | `harmonia/folding.py:109` | si une position de la boucle est écrasée par le consensus ou laissée telle quelle | c'est le seul à avoir une calibration nommée (5ᵉ percentile de 400 piles mélangées, 5 morceaux) — mais ni le script ni les 400 piles ne sont dans le dépôt |
| 5 | `beat_trans_penalty` mi-mesure du **gabarit** | `15.0` (au lieu de `45.0`) | `harmonia/folding.py:505` | à quel prix un accord peut changer au milieu d'une mesure, **au deuxième décodage** | l'asymétrie avec la 1ʳᵉ passe (45) n'est justifiée que par une observation sur un accord (Ddim, 2026-08-01) ; elle fait que le repli peut couper une mesure en deux là où la 1ʳᵉ passe ne le ferait pas |
| 6 | `FIN_SECTION_HORS_PILE` | `1` | `harmonia/folding.py:108` | combien de mesures de fin de section échappent à l'empilement | le commentaire dit lui-même que Louis avait dit « souvent les 2 dernières » et que 2 « est le réglage à essayer ensuite » — c'est donc un 1 posé contre un arbitrage qui dit 2 |
| 7 | `MIN_OCC_PERIOD` / `MAX_OCC_PERIOD` | `2` / `32` | `harmonia/folding.py:101` | quelles longueurs de section peuvent s'empiler occurrence contre occurrence | borne haute posée « parce que ça devient plus long que bien des morceaux » — aucune mesure |
| 8 | `_ENDING_TAIL_MAX` | `2` | `harmonia/folding.py:1049` | combien de mesures peuvent devenir une 1ʳᵉ/2ᵉ fin | au-delà, deux fins différentes deviennent deux blocs — change ce qui est écrit, pas seulement la mise en page |
| 9 | seuil « une passe seule ne fait pas consensus » | `len(ranges) < 3` | `harmonia/folding.py:798` | à partir de combien de passages la façade remplace une mesure rejetée par le consensus | à 2, « l'autre » n'est pas un consensus (raison écrite) ; le 3 lui-même n'est appuyé par rien |
| 10 | seuil d'accord de la façade | `len(best) < 2` | `harmonia/folding.py:826` | combien de passes doivent dire la même chose pour écraser une mesure rejetée | 2 passes d'accord suffisent à réécrire une mesure que le modèle avait entendue autrement |
| 11 | `2 * P` (membres gardés) | — | `harmonia/folding.py:376` | plancher sous lequel une lettre ne se replie pas | pas de trace ; décide l'empilement ou non |
| 12 | `>= 3` membres avant de tester les variantes | — | `harmonia/folding.py:361` | sous 3, aucune mesure n'est écartée de la pile | une pile de 2 ne peut jamais rejeter une mesure aberrante |
| 13 | rembourrage du bloc à `min(4, …)` mesures | `4` | `harmonia/folding.py:976-978` | la longueur écrite d'un bloc replié | change ce qui est imprimé (une cellule de 2 mesures est écrite 2× pour faire 4) |
| 14 | chute d'un `N` de tête | `2.0 × temps médian` | `harmonia/bars.py:147` | si le silence initial devient une mesure N.C. ou disparaît | déplace TOUT le morceau d'une mesure si mal calibré |
| 15 | `eps` de découpe des temps | `0.05 s` | `harmonia/pipeline.py:177` ⚠️NON-COMMITÉ | quels temps comptent comme attaques pour la lecture de basse | change quels slashes sont écrits |
| 16 | `PLAUSIBLE` ↔ `UNTESTED` | `{6,8,9,10}` traités comme impossibles | `harmonia/bass_rules.py:53` | quels intervalles de basse ne s'écrivent JAMAIS | honnêtement documenté comme jamais soumis à l'oreille — donc `posé`, mais posé dans le bon sens (défaut = accord nu) |
| 17 | `1.25` décisivité de durée | `1.25` | `harmonia/harmonic_key.py:155` | si la durée seule nomme la tonique, ou si Krumhansl tranche | le commentaire dit « calibré sur deux morceaux = hypothèse (règle #5) » |
| 18 | `0.6` plancher de candidat | `0.6` | `harmonia/harmonic_key.py:159` | quelles fondamentales sont candidates à être la tonique | — |
| 19 | `MODE_MASS_MIN` | `1.5` | `harmonia/harmonic_key.py:64` | quand on bascule du test « accords sur la tonique » au test large | décide majeur/mineur, donc les couleurs ET l'orthographe ♭/♯ du chart |
| 20 | `Q`, `GAIN`, `LAMBDA` | `0.85`, `25.0`, `0.25` | `harmonia/harmonic_key.py:53-55` | la couleur de chaque accord (naturel / harmonique / dorien / mélodique) | « byte-identical to v4.1, do not retune here » — c'est l'inverse d'une justification |
| 21 | `P_N_STAY` … `P_R_TO_R` | `0.94, 0.02, 0.85, 0.12, 0.015` | `harmonia/harmonic_key.py:56-57` | la viscosité du HMM de couleur | idem |
| 22 | `CHALLENGE_MARGIN` + poids `0.7/0.3` | `1.25`, `0.7/0.3` | `harmonia/harmonic_key.py:60`, `:316` | quels accords reçoivent un drapeau « suspect » | drapeau seulement (n'imprime rien), mais oriente l'annotation |
| 23 | `_QUAL["11"] → "7sus4"` | — | `harmonia/labels.py:36` | **un accord `X:11` de musx s'imprime `X7sus4`** | mapping de théorie, commenté « nearest iReal tail », zéro arbitrage |
| 24 | `_QUAL["maj/b7"] → ("7", +10)` | — | `harmonia/labels.py:40` | **une triade majeure sur sa b7 s'imprime comme un dominante renversé** | idem, zéro arbitrage |
| 25 | `MERGE_LETTERS_COS` | `0.93` | `harmonia/folding.py:137` | quelles lettres fusionnent avant le repli | **désarmé** (`SETTINGS.merge_letters = False`) — n'imprime rien aujourd'hui |
| 26 | `PHASE_CONSENSUS_MIN` / `PHASE_BEAT0_MAX` | `0.55` / `0.15` | `harmonia/bars.py:84-85` | si les accords peuvent renverser la phase de mesure du traceur | **MORT** — ne se déclenche sur aucun des 44 (voir §3) |

### Ce qui ne change QUE le temps ou la mise en page

À séparer nettement du tableau ci-dessus — ces nombres ne peuvent pas changer
une lettre d'accord :

| Constante | Valeur | Où | Effet |
|---|---|---|---|
| `HEAD_SECONDS` | `45.0` | `harmonia/pipeline.py:245` | durée de l'aperçu de chargement |
| `HEAD_MIN_SONG` | `90.0` | `harmonia/pipeline.py:249` | à partir de quelle durée on fait un aperçu |
| `DELAI` (songformer) | `1800 s` | `harmonia/sections/songformer.py:93` | délai avant abandon du processus enfant |
| `GRID_MIN_CONSISTENCY` | `0.80` | `harmonia/beats.py:77` | **diagnostic seul depuis 2026-08-07** — ne décide plus rien |
| `top_k` des suggestions | `3` | `harmonia/span_rescore.py:224` | combien de candidats l'écran Annotate propose |
| `_PRIMES` | `′ ″ ‴ ⁗` | `harmonia/folding.py:1046` | comment on nomme un deuxième bloc sous la même lettre |
| `UNIT`, `BLOCK`, `SHARP_W`, `ODD_BONUS` | `2, 8, 0.5, 0.05` | `harmonia/sections/similarity.py:125-128` | **hors du chemin automatique** — n'alimentent que l'outil sections, Soudure et `/ssm` |

Cas limite à connaître : **les seuils de battues (`GRID_MIN_COVERAGE`,
`GRID_MIN_DIRECT`, `GRID_METRES`) ne changent pas un symbole d'accord — ils
décident si le morceau produit un chart du tout** (`check_grid` lève). Et
`DUP_TOL`/`LARGE_TOL`/`HALF_TOL`/`GRILLE_MIN_INLIERS` déplacent la grille, ce
qui change **dans quelle mesure** un accord s'écrit, pas lequel.

---

## 3. Code mort — ce qui ne peut plus se déclencher

| Quoi | Où | Vérification |
|---|---|---|
| **`bars._phase_correction`** — le ré-ancrage harmonique de la phase | `harmonia/bars.py:88-98`, appelé `:167` | **Mort AUJOURD'HUI, mais pas par conception — et vivant il y a 44 heures.** Voir l'encadré ci-dessous : c'est la trouvaille la plus importante de cette section. |
| **`folding.display_fold`** (~120 lignes) + son étage d'empilement inter-passes | `harmonia/folding.py:611-729` | **Aucun appelant.** La pipeline appelle `minimal_fold` (`pipeline.py:667`). Confirmé par grep sur tout le dépôt hors `.venv` : seules références = sa propre définition et `folding.py:1063` qui dit explicitement « `display_fold`, que la pipeline n'appelle pas ». |
| **`SETTINGS.sections`, `.merge`, `.fold_loop`, `.fold_gate`, `.fold_transpose`** | `harmonia/settings.py:67-72` | **Jamais lus nulle part.** Grep sur tout le dépôt : zéro lecture. Ce sont cinq constantes qui ressemblent à des réglages et documentent une loi appliquée en dur ailleurs. Changer `SETTINGS.merge = "cqt"` ne ferait strictement rien. Seuls `quarter_bar` (`pipeline.py:498`, `folding.py:503`) et `merge_letters` (`pipeline.py:652`) sont vraiment lus. |
| **`musx.make_beat_arr(quarter_beats=None)`** — la branche demi-mesure seule | `harmonia/musx.py:121-122` | Inatteignable en production : les deux appelants vivants passent `"all"` parce que `SETTINGS.quarter_bar` est `True` en dur (`settings.py:69`). |
| **`bass_rules` branche sus** | `harmonia/bass_rules.py:104-107` | Désarmée explicitement, `SUS_ENABLED = False` (`:78`). Honnêtement documenté. |
| **`bass_rules.is_decoration`** | `harmonia/bass_rules.py:115-138` | Sa propre docstring dit que `decide_bass` ne l'appelle pas et que son socle de preuve s'est effondré. Aucun appelant en production. |
| **`folding.merge_similar_letters`** | `harmonia/folding.py:140` | Atteignable seulement si `merge_letters=True` ; `SETTINGS.merge_letters = False` (`settings.py:76`). Gelé, pas mort. |
| **`harmonia.refold`** | `harmonia/refold.py` | Le serveur vivant importe `harmonia_min.refold` (`server/routes/sections.py:213`), qui est un shim vers celui-ci — donc vivant par ricochet, mais jamais importé directement par `harmonia/`. |
| **`harmonia/roles.py`** (172 lignes) | — | Sa propre docstring : « Rien ici n'est branché dans le repli ». Seul consommateur : `tools/roles_page.py`. Brique en attente d'arbitrage, pas du code mort à supprimer. |
| **`beats.grille_rigide` + `RIGIDE_ECART` / `RIGIDE_ZONE` / `RIGIDE_COLLE`** | `harmonia/beats.py:341-421`, `:224-228` | *(vérification déléguée — voir la note en fin de §3)* `_clean` (`:766`) appelle `_poser_grille_rigide` (`:780`), pas `grille_rigide`. Lecture du code : orpheline sur le chemin vivant. |

Trois précisions sur ce tableau, vérifiées en exécutant du code, pas en lisant :

- **`RIGIDE_COLLE` emporte avec lui une protection qui n'a PAS de remplaçante.**
  `grille_rigide` vérifiait que la grille rigide **colle vraiment** aux temps du
  traceur (95ᵉ percentile des écarts `≤ 0.25`, `beats.py:409-411`) — sans ça,
  `blue_bossa_150bpm` et `XpqqjU7u5Yc` passaient avec 5 % de leurs temps à un
  demi-temps de la case, « c'est-à-dire une grille rigide qui déplace la musique
  au lieu de la décrire » (`beats.py:402-408`). Le chemin vivant
  (`_poser_grille_rigide`) n'a **aucun** équivalent : il ne teste que
  `GRILLE_MIN_INLIERS = 0.85`. À regarder.
- `grille_rigide` et ses trois constantes sont exercées par
  `tests/test_beats_jumeaux.py:76-119` — donc testées, mais pas en production.
- `GRID_MIN_CONSISTENCY` n'est lu nulle part dans `harmonia/` (seulement
  `tools/sections_bench/` et `archive/scripts/`).

### L'encadré : pourquoi `_phase_correction` est mort, et ce que ça dit

Vérifié en **rejouant la vraie chaîne** sur les charts en cache
(`beats._clean` → `check_grid` → re-phase bar1 → arithmétique de `bars`, avec la
vraie fonction importée, pas une réimplémentation), et en ne gardant que les
morceaux dont la grille de temps reconstruite est identique à `chart.beatTimes`
à 1e-6 près : **82 charts sur 87, 44 morceaux distincts, 0 déclenchement.**

Trois choses en sortent :

1. **Ce n'est pas le seuil de consensus qui bloque, c'est `PHASE_BEAT0_MAX`.**
   Les résidus d'accord sont structurellement **bimodaux** — ils s'agglutinent
   sur le downbeat ET sur la mi-mesure — donc la part sur le temps 0 est
   toujours grosse. Minimum sur tout le corpus : **0.24** (Goodbye Yellow Brick
   Road), puis 0.38. Rien n'approche le plafond de 0.15 à moins de 9 points.
   Baisser `PHASE_CONSENSUS_MIN` à 0.45 comme le suggère l'audit ne servirait à
   rien : c'est l'autre garde qui ferme la porte.
2. **Il a été VIVANT jusqu'au 2026-09-15 à 15h24.** Les journaux du rapport d'or
   portent « harmonic re-anchor fired », `corr = -1`, sur `dOQXg6rK86I`, dans 5
   passages (le plus ancien `state/cache/golden/baseline_2026-09-14/_golden.log:296`,
   le plus récent `state/cache/golden/audit_revert/_golden.log:378`). Il est mort
   entre 15h24 et 15h54, c'est-à-dire **avec la suppression de la recherche de
   latence musx** (`ea539de`) : la latence déplaçait les frontières du
   re-décodage, donc les résidus. **Si la compensation de latence revient un
   jour, cette branche revient avec elle** — et personne ne le saura.
3. **Le commentaire qui justifie les seuils est faux sur les données
   d'aujourd'hui.** `bars.py:79-83` dit « This Love : 75/120 accords sur le temps
   3, 1/120 sur le temps 0 ». La reconstruction donne `{0: 76, 2: 41}` — soit
   **76 accords SUR le temps 0**. Les chiffres sont inversés par rapport à la
   réalité actuelle.

---

## 4. Décisions prises DEUX FOIS

| Décision | Endroit 1 | Endroit 2 | Est-ce cohérent ? |
|---|---|---|---|
| **Combien de temps par mesure (`bpb`)** | `harmonia/pipeline.py:487-489` (`_bpb_early`, décodage) et `harmonia/bars.py:150-152` (disposition) | `harmonia_min/jam.py:152-156` (le mode Jam, vivant via `server/routes/jam.py:35`) | **Écrite TROIS fois**, à l'identique (`round(median(diff(downbeats)) / median(diff(beats)))`, clampée 2..7 sinon 4). Rien ne force la synchro. |
| **Le prix d'un changement d'accord** | 1ʳᵉ passe : `(15, 45, 100)` — `harmonia/musx.py:443`, hérité du décodeur vendu | gabarit du repli : `(15, **15**, 100)` — `harmonia/folding.py:505` | **Volontairement différent**, mais la justification tient à un accord observé en 2026-08-01. Le repli peut donc couper une mesure en deux là où la première passe ne l'aurait pas fait. |
| **La tonalité du morceau** | `_draft_key` (`harmonia/pipeline.py:191-213`) : histogramme de hauteurs des accords → `key_profiles.infer_key` (Krumhansl) | `analyze_harmony` (`harmonia/harmonic_key.py:339`) : CUSUM causal sur la « masse interdite », puis `infer_key` en départage | La seconde **écrase** la première sans condition (`pipeline.py:704-710`). La première ne sert que le chart brut (l'app planterait sans `key` pour orthographier). Documenté, pas un bug — mais deux algorithmes de tonalité coexistent. |
| **L'orthographe ♭ vs ♯** | `harmonia/pipeline.py:708` : `maj in (7,2,9,4,11)` → dièses | `harmonia/static/ui/kit.js:53,56` : `FLAT_MAJ = {0,1,3,5,6,8,10}` → bémols | Les deux ensembles sont exactement complémentaires, donc **d'accord aujourd'hui** — mais c'est la même règle écrite en Python et en JS, sans test qui les lie. |
| **La période du temps (`median(diff(…))`)** | `beats.py:485,584,772,850` · `bars.py:147,149,150` · `folding.py:270,663` · `pipeline.py:487,488` · `jam.py:154,155` | — | **13 calculs indépendants** de la même quantité, sur des listes de temps à des stades de nettoyage différents. `bars.py:147` et `bars.py:149` recalculent la valeur **identique à deux lignes d'écart**. |
| **`infer_key`** | `harmonia/key_profiles.py:106` | importé par `pipeline.py:50` (`from harmonia.key_profiles`) ET par `harmonic_key.py:40` (`from harmonia_min.key_profiles`, un shim qui ré-exporte le même module) | Une fonction, **deux chemins d'import**, dont un passe par le paquet à supprimer. |
| **La confiance d'un accord (`c`)** | `bars._segment_confidence` (`harmonia/bars.py:68`) | `folding._decode_template` (`harmonia/folding.py:537,549`) | **Délibérément unifié** : les deux délèguent à `musx.label_confidence` (`musx.py:385`). C'est le bon modèle — la docstring explique que les séparer avait coûté 2 points d'AUC. |
| **`musx_suggestions`** | `harmonia/pipeline.py:579` (sur le chart brut) | `harmonia/pipeline.py:703` (sur les accords repliés) | Appelé deux fois **exprès** (2026-08-20, Bora Bora : un chart brut sans classement arrivait vide dans l'éditeur). |
| **Le repli des occurrences** | chemin automatique : `pipeline.py:650` | chemin Soudure (sections faites à la main) : `harmonia_min/refold.py` → `harmonia/refold.py:108` | Même fonction (`fold_letter_groups`), donc même loi — c'était le but du sprint. Mais `sections_pour_chart` vit encore dans `harmonia_min/soudure.py` (`known_issues.md:40`). |

---

## 5. Docstrings qui contredisent le code

Classées de la plus dangereuse à la plus anodine.

| # | Où | Ce que la docstring dit | Ce que le code fait |
|---|---|---|---|
| 1 | `harmonia/bass_rules.py:22` | « Il n'est pas non plus branché dans `pipeline.py` — mesuré sur 6 morceaux / 351 accords seulement » | ⚠️ **FAUX depuis la modification non commitée.** `pipeline.py:687` appelle `_write_sounding_bass`, qui appelle `decide_bass` sur **chaque accord de chaque chart**. Une règle calibrée sur 6 morceaux est maintenant en production. `docs/known_issues.md:59` porte la même affirmation périmée (« rien dans `pipeline.py` ne l'appelle »). |
| 2 | `harmonia/sections/songformer.py:6-7` | « C'est donc le détecteur de sections par défaut (`HARMONIA_SECTIONS=songformer`), et `voice` / `harmonic` / `chroma` restent joignables derrière la même variable » | Faux. `settings.py` ne lit que trois variables d'environnement, et aucune ne s'appelle `HARMONIA_SECTIONS`. Les trois autres détecteurs ont été supprimés (`sections/__init__.py:8-10`). |
| 3 | `harmonia/sections/songformer.py:54-55` | « le serveur … retombe sur le détecteur `voice` » | Faux et inversé : `sections/__init__.py:17-25` dit explicitement « **pas de repli** », l'exception remonte et l'analyse échoue. C'est une décision de Louis du 2026-09-14. |
| 4 | `harmonia/sections/songformer.py:265` | « `folding.fold_display` fait pareil depuis le 2026-08-18 » | `fold_display` n'existe pas. La fonction s'appelle `display_fold` — et elle n'est appelée par personne. |
| 5 | `harmonia/sections/songformer.py:268` | « `folding.fold_letter_groups(loop="occurrence")` » | Le paramètre `loop` a été supprimé au refactor (`folding.py:36-38`). Cet appel lèverait `TypeError`. |
| 6 | `harmonia/musx.py:106` | « la restriction demi-mesure … pour les appelants qui la veulent (`HARMONIA_QUARTER_BAR=off`, expériences) » | `HARMONIA_QUARTER_BAR` n'existe plus. `settings.py:69` a `quarter_bar: bool = True`, en dur, et aucun appelant vivant ne passe `None`. |
| 7 | `harmonia/folding.py:500-502` | « `SETTINGS.quarter_bar = False` restaure le décodage demi-mesure du gabarit » | Techniquement vrai, mais ce serait une édition de `settings.py` — pas un réglage. Formulé comme si c'était une option. |
| 8 | `harmonia/nnls_features.py:23-26` | « WHY this is opt-in and not a silent default … `infer_chords_v1(feature_frontend="nnls24")` selects it ; "bp48" (default) is bit-identical to before » | `infer_chords_v1` n'existe plus dans `harmonia/`. Le chroma NNLS n'est plus opt-in du tout : `pipeline.py:598` et `:678` l'appellent inconditionnellement. Docstring d'un monde disparu. |
| 9 | `harmonia/nnls_features.py:143-149` | « n=1 song — NOT corpus-validated, and **not wired into the live pipeline yet** » (à propos de `BASS_ONSET_S`) | ⚠️ **Faux depuis la modification non commitée** : `pipeline.py:181` appelle `bass_pc_onset` sur chaque temps de chaque accord. |
| 10 | `harmonia/pipeline.py:3-18` (en-tête) | Liste les étapes : sections → repli → repli d'affichage → analyse harmonique | Il manque l'étape **7d, la basse sonnante** (`:687`) insérée entre le repli et `analyze_harmony`. L'en-tête n'a pas été mis à jour par le changement non commité. |
| 11 | `harmonia/pipeline.py:560` | `"engine": "harmonia_min"` avec le commentaire « bascule sur "harmonia" au sprint 22 » | Cohérent, mais le champ ment sur le moteur qui a produit le chart — délibérément, pour que le rapport d'or compare octet par octet. |
| 12 | `harmonia/labels.py:73` | Le message d'erreur dit `harmonia_min.labels: unknown musx quality` | Le fichier est `harmonia/labels.py`. Cosmétique, mais envoie au mauvais fichier en cas de panne. |
| 13 | `harmonia/bars.py:79-83` | « This Love : 75/120 accords sur le temps 3, **1/120 sur le temps 0** » — c'est toute la justification des deux seuils de phase | Reconstruit sur les données d'aujourd'hui : `{0: 76, 2: 41}`, soit **76 accords SUR le temps 0**. Les chiffres sont inversés. La justification des seuils ne tient plus. |
| 14 | `harmonia/bars.py:43` (et `folding.py`, `pipeline.py`) | renvoie à `docs/postmusx_segment_loss.md` pour la doctrine « un filtre ne jette que les artefacts de sa propre coupe » | **Le fichier n'existe pas dans l'arbre principal.** Il ne survit que dans quatre worktrees `.claude/worktrees/*/docs/`. La règle citée quatre fois dans le code n'a plus de document. |
| 15 | `harmonia/musx.py:53` | « Pooled optimum 40 ; LOSO-stable (per-fold picks 40 sur 6/7 morceaux, 55 sur le 7ᵉ) » | Aucune trace de cette étude : ni doc, ni script, ni nom de corpus. Voir §2 ★. |

### ⚠️ Un bug vivant trouvé en chemin : le mode Jam est cassé

`harmonia_min/jam.py:157` (le seul chemin d'accords du mode Jam, atteint par
`harmonia/server/routes/jam.py:35`) :

```python
segments, _lat = _musx.redecode(beat_times, probs, ...)
```

Mais `harmonia/musx.py:468` rend **une liste nue** (`return lab`) depuis le
2026-09-15 15h54 (`ea539de`, « musx : plus de recherche de latence »). Le
second terme de retour n'existe plus.

`pipeline.py:514` et `folding.py:504` ont été mis à jour ce jour-là ; `jam.py`
n'a pas été touché depuis le 2026-08-20 (`5602e87`). **Troisième copie du même
appel, oubliée à la mise à jour.**

Deux issues possibles, toutes les deux mauvaises :
- si le re-décodage rend un nombre de segments ≠ 2 → `ValueError: too many
  values to unpack` à chaque passe de Jam ;
- s'il en rend exactement 2 → **ça passe en silence**, et `segments` devient un
  seul tuple `(t0, t1, label)` au lieu de la liste. `if not segments` est alors
  faux, et la suite itère sur trois scalaires.

Non vérifié en exécution (le Jam demande un micro) — établi par comparaison
signature / site d'appel. C'est exactement le motif de la règle #6 de CLAUDE.md
(un échange de composant change plus que sa cible).

### Dépendances cachées sur `harmonia_min` (le paquet à supprimer au sprint 22)

Le nouveau paquet en importe encore **quatre fois**, toutes en import différé,
donc invisibles au démarrage :

| Où | Import |
|---|---|
| `harmonia/musx.py:417` (dans `label_confidence`) | `from harmonia_min.labels import parse_root` |
| `harmonia/harmonic_key.py:40-41` (au module) | `from harmonia_min.key_profiles import infer_key` ; `from harmonia_min.labels import _TAIL_PCS, _TRIAD_PCS` |
| `harmonia/span_rescore.py:138` (dans `pool_span_musx`) | `from harmonia_min.musx import FRAME_DT` |
| `harmonia/refold.py:76` | `from harmonia_min.soudure import accords_par_mesure` |

Les trois premiers passent par des shims (`harmonia_min/labels.py`,
`key_profiles.py`, `musx.py` ré-exportent `harmonia.*`), donc ça marche
aujourd'hui. **Le jour où `harmonia_min/` disparaît, `musx.label_confidence`
lève `ImportError` pour tout label non-`N`** — c'est-à-dire pour chaque accord
de chaque chart, au moment du calcul de `c`, pas à l'import.

---

## 6. Le parcours, étape par étape

Notation des colonnes : **Prov.** = provenance (`arbitré` / `mesuré` / `hérité` /
`posé`).

### Étape 0 — L'aperçu des 45 premières secondes

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D0 | Faut-il calculer un chart d'aperçu avant la vraie passe | `pipeline.py:458-463`, `_head_chart:288` | si `durée > HEAD_MIN_SONG`, analyser les `HEAD_SECONDS` premières secondes avec le MÊME générateur et rendre son chart brut | `HEAD_SECONDS = 45.0` (`:245`), `HEAD_MIN_SONG = 90.0` (`:249`) | `HEAD_SECONDS` **mesuré** (2026-08-18, 2 morceaux : plus courte fenêtre donnant les mêmes battues) ; `HEAD_MIN_SONG` **posé** | L'aperçu affiche 94 % des bons accords (mesuré, 8 mesures) — 6 % changeront. Jamais fatal (`:330`). |

### Étape 1 — Les temps (`harmonia/beats.py`)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D1 | Quel traceur de temps | `beats.py:45-47` | `File2Beats(device="cpu", dbn=False)`. Pas de repli — librosa est **banni** | — | **arbitré** (2026-07-21, verrou d'octave 2× mesuré : 65 % vs 78 % sur POP909) + **hérité** (le modèle) | Une erreur d'octave de tempo casse tout en aval en silence |
| D2 | Retirer les temps « jumeaux » (deux marques pour un temps) | `beats.py:425` `drop_duplicate_beats` | intervalle `< 0.25 × médian` → jumeau ; ou `0.25..0.45 ×` **si** la paire ne couvre qu'une période locale | `DUP_TOL = 0.25` (`:216`), `LARGE_TOL = 0.45` (`:221`), `SPAN_LO/HI = 0.88/1.12` (`:222`) | **mesuré** (145 puis 199 morceaux en cache ; portée 9 morceaux / 15 paires) | Une mesure trop courte décale toutes les barres suivantes — le bug de Ready |
| D3 | Retirer les temps insérés (verrou double-tempo local) | `beats.py:546` `drop_inserted_beats` | deux intervalles consécutifs valant chacun `0.5 ± 0.18 ×` la médiane se recollent | `HALF_TOL = 0.18` (`:543`) | **mesuré** (66 morceaux : 59 intouchés, 7 modifiés, aucun changement de verdict) | Mesures trop courtes par endroits, invisibles aux autres gardes |
| D4 | Remplacer la grille du traceur par une grille rigide | `beats.py:766` `_poser_grille_rigide` → `bpm_rigide:245` | médiane des périodes de fenêtres glissantes → verrouillage de phase par `\|moyenne exp(2iπt/P)\|` sur 4001 périodes à ±3 % → raffinement sur les inliers | `BPM_FEN=16`, `BPM_TOL=0.15`, `BPM_MIN_INLIERS=0.6` (`:240-242`), `GRILLE_MIN_INLIERS=0.85` (`:777`), grille ±3 %/4001 (`:308`) | **arbitré** (Louis 2026-08-18 : « trouve-moi un algo rigide pour inférer le bpm ») + **mesuré** (84 morceaux, 25 passent) ; `GRILLE_MIN_INLIERS`, `BPM_TOL`, ±3 % et 4001 sont **posés** | Une grille rigide sur un morceau qui dérive **déplace la musique au lieu de la décrire** — c'est ce que garde `GRILLE_MIN_INLIERS` |
| D5 | Quelle métrique, et quels downbeats garder | `beats.py:602` `repair_grid` | pour chaque métrique candidate, paver les indices de temps en enjambant jusqu'à `MAX_BRIDGE` mesures ; ne retenir que les métriques que le traceur marque lui-même (`direct ≥ 0.15`) ; parmi elles, prendre **la mieux ancrée**, pas la mieux couverte | `GRID_METRES = (4,3,6)` (`:121`), `GRID_MIN_DIRECT = 0.15` (`:137`), `MAX_BRIDGE = 4` (`:162`) | `GRID_METRES` **arbitré** (2026-08-07, Alicia Keys : « je sais que ça va ») ; `GRID_MIN_DIRECT` **mesuré** mais **la marge est d'un seul morceau** (0.05 → 0.19, le code le dit) ; `MAX_BRIDGE` **posé** | Une valse lue en 6/8, ou un verrou demi-tempo accepté. Les trois pièges sont documentés ligne par ligne |
| D6 | **Refuser le morceau** | `beats.py:681` `check_grid` | si `< 30` mesures, laisser passer sans juger ; sinon `direct < 0.15` **ou** `coverage < 0.85` → `BeatTrackingError` | `GRID_MIN_BARS = 30` (`:80`), `GRID_MIN_COVERAGE = 0.85` (`:118`), `GRID_MIN_DIRECT = 0.15` | `GRID_MIN_BARS` **posé** ; `GRID_MIN_COVERAGE` **mesuré et — seul cas du dépôt — REPRODUCTIBLE** : `archive/scripts/beat_grid_report.py:69` importe les fonctions vivantes et refait la page morceau par morceau. Mais le fossé est mince : « 0.85 sits in that gap » entre 0.87 (Kermit) et 0.81 (Chiquitita), **deux morceaux** | Pas de chart du tout. C'est le bon échec (« un chart faux est pire que pas de chart »), mais le seuil est mince |

> `GRID_MIN_CONSISTENCY = 0.80` (`:77`) est explicitement **diagnostic seul**
> depuis 2026-08-07 — plus aucun test d'acceptation ne le lit.

### Étape 2 — Les probabilités d'accord (`harmonia/musx.py`)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D7 | Quel modèle d'accords | `musx.py:48` `MODEL_NAMES` | les 5 folds livrés avec le clone ISMIR 2019, moyennés (`_run_nets:235`) | 5 folds | **hérité** | — |
| D8 | La grille de trames | `musx.py:43-45` | `22050 / 512 = 23.22 ms` | `MUSX_SR=22050`, `MUSX_HOP=512` | **hérité** (constantes du clone) | La règle #1 de CLAUDE.md : une erreur d'unité ici corrompt tout en silence |
| D9 | CPU ou GPU | `musx.py:181` `_device()` | `mps` si disponible | `SETTINGS.musx_device = "auto"` | **mesuré** (2026-08-18, 4 morceaux : argmax 100 % identiques, \|Δ\| max 0.0000) | Aucun — vérifié bit-à-bit. Beat This! reste sur CPU (9× plus lent sur MPS) |

### Étape 3 — QUEL ACCORD SUR QUEL TEMPS (le re-décodage) — **le cœur**

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D10 | Combien de temps par mesure, pour le décodage | `pipeline.py:487-489` | `round(median(diff(downbeats)) / median(diff(beats)))`, gardé si `2 ≤ x ≤ 7`, sinon 4 | bornes `2..7`, défaut `4` | **posé** (les bornes) | Les positions « mi-mesure » tombent au mauvais endroit → grille de coûts fausse. **Dupliqué en `bars.py:150-152`** |
| D11 | Re-caler les downbeats du décodage sur la marque de Louis | `pipeline.py:499-513` | le temps le plus proche de `bar1_time` devient la phase ; les downbeats sont régénérés tous les `bpb` temps | — | **arbitré** (Louis 2026-08-09 : « l'accord devrait commencer au début de la barre ») | Sans ça, la phase d'affichage était corrigée mais pas celle des accords |
| D12 | Le vocabulaire d'accords possibles | `musx.py:425` | `data/submission_chord_list.txt` du clone : 25 gabarits × 12 fondamentales + `N` | `chord_dict = "submission"` | **hérité** (vérifié contre le fichier réel, `labels.py:3-8`) | Un `q` inconnu **lève** (`labels.py:72`) — jamais de repli silencieux vers majeur |
| D13 ★ | Le prix de changer d'accord (hors temps) | `musx.py:53` `DEFAULT_PENALTY = 40.0` | pénalité Viterbi `diff_trans_penalty` | `40.0` — **le défaut du décodeur vendu est `30.0`** (`third_party/musx_ismir2019/extractors/xhmm_ismir.py:7`) | **posé.** Le commentaire revendique une mesure (« pooled optimum 40 ; LOSO-stable, 40 sur 6/7 morceaux, 55 sur le 7ᵉ ») mais **rien ne la corrobore dans le dépôt** : zéro hit pour `pooled optimum`/`LOSO`/`penalty=40` dans `docs/` et `archive/scripts/`, corpus non nommé. Introduit le 2026-07-27 (`f36c327`) | Trop haut = accords collés (on rate les changements) ; trop bas = hachis. **C'est la constante qui touche le plus d'accords de tout le système** |
| D14 | **Où un accord a le droit de changer, et à quel prix** | `musx.py:62` `make_beat_arr`, appelé `:462` | `0` = interdit (hors des temps), `2` = downbeat (coût 15), `3` = mi-mesure (45), `4` = autre temps (100). `quarter_beats="all"` ouvre tous les temps | `beat_trans_penalty = (15.0, 45.0, 100.0)` (`musx.py:443`) — **exactement le défaut du décodeur vendu** ; `SETTINGS.quarter_bar = True` (`settings.py:69`) | triplet **hérité** ; l'ouverture quart-de-mesure **arbitrée** (Louis 2026-08-07 : « on ne met plus de restrictions sur la granularité ») ; le câblage des downbeats **arbitré** (Louis 2026-07-31, rapport This Love : deux accords un temps trop tôt à 1:56 et 2:52) | C'est ce qui décide si un accord tombe sur la barre ou un temps avant. L'étude de précision disait « graded ≈ flat » (−0.17 pp) — c'est le **placement** qui a tranché, pas la métrique |
| D15 | Étiquette musx → symbole imprimé | `labels.py:20` `_QUAL`, `:60` `to_chord` | table de 25 entrées ; les renversements gardent la qualité parente et rendent une basse | 25 lignes de table | mélange : la plupart **hérité** (correspondance 1:1) ; `"11" → "7sus4"` (`:36`) et `"maj/b7" → ("7", +10)` (`:40`) sont **posés** (commentaires de théorie, aucun arbitrage) | Chaque ligne fausse imprime un mauvais symbole sur tous les morceaux à la fois |

**Ce que D14 ne résout pas** (écrit dans `musx.py:453-456`) : un accord joué en
anticipation (mesuré à −220 ms sur `gbO7qQliXT8`) tombe quand même sur le temps
légal le plus proche — la grille n'a pas de demi-temps. La recherche de latence
par morceau a été **supprimée** le 2026-09-15 (audit accepté par Louis) : les
changements de postérieure sont 23–46 ms AVANT le temps, jamais après.

### Étape 4 — La disposition en mesures (`harmonia/bars.py`)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D16 | Jeter un `N` de tête court | `bars.py:146-148` | si le premier segment est `N` et dure `< 2 × temps médian`, il disparaît | `2.0` | **posé** | Un `N` long (l'intro de basse de Stand By Me) reste et prend sa mesure N.C. — c'est voulu |
| D17 | `bpb` pour la disposition | `bars.py:150-152` | même formule que D10 | bornes `2..7`, défaut `4` | **posé** — **doublon de D10** | — |
| D18 | La phase de mesure (où tombe la barre) | `bars.py:159-161` | résidu modal des indices de downbeat modulo `bpb` | — | **hérité** (vote du traceur) | Tout le chart décalé |
| D19 | Les accords peuvent-ils renverser cette phase | `bars.py:88` `_phase_correction`, appelé `:167` | `modal ≠ 0` ET `part(modal) ≥ 0.55` ET `part(0) ≤ 0.15` | `PHASE_CONSENSUS_MIN = 0.55` (`:84`), `PHASE_BEAT0_MAX = 0.15` (`:85`) | **posé** — « copié de l'app vivante » (`archive/scripts/render_youtube_chart.py:350`), justifié sur **un morceau**, et les chiffres du commentaire sont **inversés** par rapport aux données d'aujourd'hui | **MORT sur 44/44** — mais il a tiré sur `dOQXg6rK86I` jusqu'au 2026-09-15 15h24, et il est mort comme **effet de bord** de la suppression de la latence musx. Voir l'encadré §3 |
| D20 | La marque « Set bar 1 » écrase tout | `bars.py:172-185` | le temps le plus proche de la marque donne la phase ; rien n'est coupé avant | — | **arbitré** (Louis 2026-08-08, rapport Sam Smith) | Le repère de Louis ignoré — ça s'est produit, c'est ce que corrige `bar1` dans le modèle (`pipeline.py:551`) |
| D21 | Les levées rentrent de force en mesure 0 | `bars.py:200`, `:212` | `b = max(0, eff // bpb)`, et elles s'affichent au temps 0 | — | **posé** | Le résidu modulo n'a plus de sens une fois la mesure serrée — assumé |
| D22 | Un `N.C.` à cheval sur la barre devient l'accord au temps 0 | `bars.py:223-227` | si l'accord n'est pas `nc`, tombe après le temps 0, n'est pas une levée, ET que le segment précédent est un `N` **commencé dans une mesure antérieure** → on le ramène au temps 0 | — | **arbitré** (Louis 2026-08-09 : « l'accord devrait commencer au début de la barre ») | Un arrêt en milieu de mesure reste où il se produit — la distinction est explicite |
| D23 | `N.C.` et accord sur le même créneau → l'`N.C.` saute | `bars.py:229-239` | dans un créneau (mesure, temps) contenant les deux, tous les `nc` sont retirés | — | **arbitré** (Louis 2026-08-01, « Q4 ») | — |
| D24 | Trop d'accords dans une mesure | `bars.py:247-259` | la mesure 0 se débarrasse de ses levées serrées (la plus courte d'abord) ; si ça déborde encore → `RuntimeError` | `bpb` créneaux | **arbitré** (doctrine « un filtre ne jette que les artefacts de sa propre coupe », `docs/postmusx_segment_loss.md`) | Échec bruyant plutôt que perte silencieuse d'accords |
| D25 | **La règle de la mesure de Louis** : une mesure liste TOUS les accords qui y sonnent | `bars.py:279-292` | une mesure sans attaque réécrit l'accord qui sonne, marqué `carry` ; une mesure dont la 1ʳᵉ attaque est en cours écrit d'abord l'accord porté ; **jamais un `N.C.`** | — | **arbitré** (trois fois : 2026-07-31 la règle, 2026-07-31 soir « plus de % », 2026-08-10 « on propage les accords, pas les NC ! ») | Mesuré avant le correctif : Stand By Me affichait **93 % de N.C.** contre 8 % dans la détection brute |
| D26 | La confiance affichée sur chaque accord | `bars.py:68` → `musx.py:385` `label_confidence` | postérieure moyenne du plan triade de l'étiquette sur ses propres trames ; `0.5` (neutre) si fenêtre inutilisable ou qualité inconnue | `0.5` neutre | **mesuré** (2026-08-01, GuitarSet : mélanger deux échelles coûtait 2 points d'AUC) | Une confiance fausse est pire qu'une confiance non informative — c'est écrit |

### Étape 5 — Le chart brut (premier rendu)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D27 | La tonalité provisoire | `pipeline.py:191-213` `_draft_key` | histogramme des hauteurs sonnées, pondéré par la durée de chaque accord → `key_profiles.infer_key` (Krumhansl-Schmuckler, 24 clés, prior uniforme) | profils KS (`key_profiles.py:31-40`), durée plancher `0.05 s` | profils **hérités** (Krumhansl 1990) ; le plancher `0.05` **posé** | **Écrasée sans condition** par l'étape 10. Elle n'existe que pour que l'app puisse orthographier le chart brut |
| D28 | Les candidats d'accord du chart brut | `pipeline.py:578-580` → `span_rescore.musx_suggestions` | voir D61 | — | **arbitré** (Louis 2026-08-20, Bora Bora) | Un chart brut arrivait dans l'éditeur sans rien à proposer |

### Étape 6 — Les sections (`harmonia/sections/`)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D29 | Où sont les frontières, et quel rôle a chaque passage | `sections/songformer.py:295` | SongFormer (ASLP-lab) : MuQ + MusicFM, 10ᵉ couche, fenêtres 30 s et 420 s → Transformer → frontière + étiquette à 8,33 img/s. **Il écoute le SON, pas les accords** | `DEPOT = "ASLP-lab/SongFormer"`, `DELAI = 1800` | **arbitré** (Louis 2026-08-18, après `/plots/songformer.html` sur 19 morceaux : « je suis d'accord avec lui partout, on le prend en prod ») + **hérité** (le modèle) | Pas de repli : si le processus enfant meurt, l'analyse **échoue** (décision de Louis, 2026-09-14) |
| D30 | Chaque frontière tombe sur une barre de mesure | `songformer.py:214` `_sur_la_grille` | barre la plus proche ; deux frontières dans la même mesure → la seconde est jetée ; la 1ʳᵉ section part toujours de la mesure 0 | — | **posé** | Une section qui commence au milieu d'une mesure |
| D31 | Le silence n'est pas une section | `songformer.py:240` `_fondre_muets` | les segments `silence` rejoignent leur voisin | — | **posé** | — |
| D32 | Rôle → lettre | `songformer.py:251` `_lettres` | `intro`/`outro` gardent leur nom (seulement en 1ʳᵉ/dernière position) ; le reste devient A, B, C… **par rôle**, dans l'ordre de première apparition | listes `DEBUTS`/`FINS` (`:89-91`) | **arbitré** (« under-fold, never over-fold » tenu en aval, pas ici — le raisonnement est écrit `:258-275`) | Deux refrains de longueurs différentes reçoivent la MÊME lettre — voulu, et c'est le repli qui les sépare par longueur |
| D33 | La marque « Set bar 1 » est une frontière DURE | `pipeline.py:98-145` `_force_bar1_sections` + `:611-618` | la détection est **relancée** sur la région après la marque ; tout ce qui précède est l'intro ; une queue nommée « intro » fusionne EN AVANT ; les lettres sont renumérotées pour que la mesure marquée lise `A` | — | **arbitré** ×2 (2026-08-08 Sam Smith : « la section A commence quand même en décalé » ; 2026-08-09 chart en ré majeur : la version « renommer » créait un A d'une mesure) | — |

### Étape 7 — LE REPLI, phase 1 (`harmonia/folding.py`) — **le deuxième cœur**

C'est ici que les accords sont **réécrits**. Mesuré (`refold.py:18`) : le repli
change **42 accords sur 797 (5,3 %)** sur 20 charts. Ce n'est pas de
l'affichage.

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D34 | Fusionner deux lettres qui sont la même musique | `folding.py:140` `merge_similar_letters` | cosinus des centroïdes de vecteurs de mesure `≥ 0.93` → la plus tardive prend le nom de la plus ancienne ; `intro`/`outro` jamais comparées | `MERGE_LETTERS_COS = 0.93` (`:137`) | **arbitré sur UN morceau** (Louis 2026-09-14, Sunny Afternoon : « la section A et C sont les mêmes ») — le code le dit : « HYPOTHÈSE, un seul morceau ». **Désarmé** : `SETTINGS.merge_letters = False` | N'imprime rien aujourd'hui |
| D35 | La boucle interne d'une section | `folding.py:197` `section_period` | pour `P ∈ {2,4,8}` avec `L ≥ 2P` : moyenne de `Vb[b]·Vb[b+P]` ; si le meilleur `< 0.80`, pas de boucle ; sinon **le plus petit `P` qui atteint `max(0.80, 0.95 × meilleur)`** | `PERIODS = (2,4,8)` (`:97`), `PERIOD_MIN_SCORE = 0.80` (`:76`), tie-break `0.95` (`:211`) | **mesuré sur 3 morceaux** (This Love A 0.944 / B 0.859, Let It Be A 0.904 ; Close to You 0.65–0.77 reste non replié). Le code écrit lui-même « Two-song-family calibration = hypothesis (CLAUDE.md rule #5) ». `PERIODS` et `0.95` sont **posés** | Boucle ratée → aucun bénéfice de répétition ; boucle inventée → des mesures différentes écrasées ensemble |
| D36 | Pas de boucle interne → empiler occurrence contre occurrence | `folding.py:288-316` | si ≥ 2 occurrences partagent la même longueur `L ∈ [2, 32]`, `P = L` et on ne garde que celles-là ; sinon **refus** (under-fold) | `MIN_OCC_PERIOD=2`, `MAX_OCC_PERIOD=32` (`:101`) | **mesuré** pour la motivation (61 lettres refusées sur 162, dont 45 avec ≥2 occurrences de même longueur) ; les deux bornes sont **posées** | C'était le refus le plus fréquent du corpus |
| D37 | Les fins de section ne s'empilent pas avec le milieu | `folding.py:340-356` | si `P < longueur de section`, les `FIN_SECTION_HORS_PILE` dernières mesures de chaque passage sortent de la pile et gardent leur 1ʳᵉ passe | `FIN_SECTION_HORS_PILE = 1` (`:108`) | **arbitré** sur le principe (Louis 2026-08-20 : « ne pas empiler les fins de sections qui sont vraiment différentes — je pense à This Love ») mais la **valeur 1 est posée** : le commentaire dit que Louis avait dit « souvent les 2 dernières » et que 2 « est le réglage à essayer ensuite » | This Love B : la cadence `A♭ G` de la mesure 8 se faisait écraser par trois `B♭ E♭` |
| D38 | Quelle mesure est une « variante » (sort de la pile) | `folding.py:359-375` | pour ≥ 3 membres : écart de chacun au centroïde ; si `(d - médiane)/MAD > 3.0` → variante, garde son 1ᵉʳ décodage | `OUTLIER_Z = 3.0` (`:77`), plancher `3` membres (`:361`) | **arbitré** sur la forme (Louis 2026-08-01 : comparer l'écart INDIVIDUEL à la norme COLLECTIVE) ; la valeur `3.0` est **mesurée** sur les mêmes 2-3 morceaux (vraies variantes z=5.7–38, membres normaux z≤1.7) | Une cadence unique écrasée, ou au contraire une section qui ne se replie jamais |
| D39 | Trop peu de membres gardés → pas de repli | `folding.py:376` | `sum(len(g)) < 2*P` → refus | `2 × P` | **posé** | — |
| D40 | **Le veto de cohérence par LETTRE ENTIÈRE** | `folding.py:391` | cohérence d'une position = **médiane des cosinus par paire** des membres gardés ; si `min` sur les positions `< 0.85`, **toute la lettre** est refusée | `STACK_COHERENCE = 0.85` (`:89`) | **mesuré sur 3 morceaux** (This Love 0.88–0.94 ; Let It Be verse+chorus 0.74–0.81 ; Stand By Me 0.80) | **La constante la plus lourde du chart.** Elle décide en un nombre si une section entière est réécrite par le gabarit ou garde son premier jet. Le refus est le bon défaut (under-fold) |
| D41 | Une position est-elle écrasée par le consensus | `folding.py:400-413` | pour chaque demi-mesure : `CV = sqrt(mean(var)) / mean(mean)` du chroma brut des membres ; `CV > 0.51` sur l'une des deux → position **non écrasée** | `CV_MAX = 0.51` (`:109`) | **mesuré et nommé** : 5ᵉ percentile de 400 piles délibérément mélangées sur 5 morceaux → taux de fausse fusion 5,0 % par construction (l'exigence de Louis : < 5 %). **Ni le script ni les 400 piles ne sont dans le dépôt** | C'est le seul seuil du repli avec un taux d'erreur nommé. Mais non reproductible |
| D42 | **LE DEUXIÈME DÉCODAGE D'ACCORDS** (le gabarit) | `folding.py:470` `_template_chords` → `:487` `_decode_template` | moyenne des postérieures des membres par position → concaténation des `P` positions **pavée ×3** → `musx.redecode` → on garde la copie du MILIEU | `beat_trans_penalty = (15.0, **15.0**, 100.0)` (`:505`), `edge_tol = FRAME_DT/2` (`:520`), pavage `×3` (`:482`), `penalty` = `DEFAULT_PENALTY` par défaut | la loi « moyenne des postérieures » est **arbitrée** (Louis 2026-08-19, après `/plots/cqt_vs_post.html` : « je préfère les postérieures empilées c'est + propre » — renverse son arbitrage du 2026-08-12) ; le `15.0` mi-mesure est **posé** (une observation, un accord, 2026-08-01) ; `edge_tol` est **mesuré** (18 accords à 0.85–0.94 de confiance silencieusement jetés pour 10 ms) | C'est le décodage qui écrit vraiment le chart final. Le `×3` existe pour les effets de bord du Viterbi |
| D43 | L'accord tenu qui ouvre la boucle | `folding.py:532-539` | l'accord qui sonne encore quand la copie du milieu commence est gardé comme `carry` au temps 0 de la position 0 ; **jamais un `N.C.`** | — | **arbitré** (Louis 2026-09-15, Let It Be : « le premier accord (C) n'est jamais propagé sur la première barre ») | Le couplet de Let It Be lisait « · G \| Am F » et son C n'apparaissait jamais |
| D44 | Réécrire une mesure avec le gabarit | `folding.py:571` `_write_position` | réécrit `bars[b]` avec les accords de la position, retimés sur la vraie largeur de mesure — **SAUF si un accord de cette mesure porte `confirmed`** | — | **arbitré** (Louis 2026-08-20 : « les annotations utilisateur prennent toujours le dessus sur nos inférences ») | Avant le correctif, une mesure corrigée à la main repassait sous le gabarit au recuit suivant, sans rien dire |
| D45 | Les comptes de répétition | `pipeline.py:654-662` | recomptés sur les accords **repliés**, par `(root, première lettre de la qualité)` | — | **posé** | N'alimente que le libellé « joué N fois » |

### Étape 8 — Le repli d'affichage (`minimal_fold`)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D46 | Grouper par **(lettre, longueur)**, pas par lettre seule | `folding.py:904-911` | deux longueurs d'un même refrain restent deux B, chacun écrit à sa longueur | — | **arbitré** (« under-fold, never over-fold », Louis 2026-07-30 ; et 2026-08-18 Another Day : « la dernière section c'est la section A mais décalée d'une barre ») | Un passage de 8 mesures rendu en 4 → la lecture décale d'une barre à chaque reprise |
| D47 | Quelle passe est ÉCRITE pour une lettre | `folding.py:760` `pass_rank` → `:734` `pass_evidence` | la passe avec le plus d'**attaques réelles** (ni `carry` ni `nc`) ; à égalité, la plus ancienne. Classée sur `bars` BRUT, pas sur la vue | — | **arbitré** (Louis 2026-08-10, Stand By Me : « je vois plein d'accords, et sur le chart j'ai juste des NC partout »). Une variante « le moins de mesures rejetées » a été **essayée et jetée le même jour** (2026-09-15) | Le chart affiche le fade-in au lieu du refrain |
| D48 | Une mesure rejetée affiche le consensus de ses sœurs (« la façade ») | `folding.py:775` `facade`, `:842` `facade_view` | si ≥ 3 passages : pour chaque mesure rejetée, on prend la **même mesure de la section** (même décalage depuis le début, pas « même position modulo P ») dans les autres passes ; il faut ≥ 2 candidats et ≥ 2 d'accord ; la plus proche gagne. **`bars` n'est pas touché** — seul le bloc écrit change | `len(ranges) < 3` (`:798`), `len(cands) < 2` (`:821`), `len(best) < 2` (`:826`) | **arbitré** (Louis 2026-09-15 : « la variante ne devrait même pas être utilisée ») ; les trois seuils sont **posés** | « même position modulo P » allait chercher la pompe deux mesures plus tôt et effaçait une vraie cadence (Sam Smith, `F C` → `F A`) |
| D49 | Quelles mesures composent le bloc écrit | `folding.py:956-984` | un repli **refusé** (avec `reason`) n'a PAS le droit de piloter l'affichage ; sinon le bloc = la cellule, rembourrée par répétition jusqu'à `min(4, longueur)`, étendue si des variantes de la passe tombent plus loin | `4` (`:976-978`) | **arbitré** sur le principe (Louis 2026-08-05, Norah Jones : « tu me bouffes la répétition de la fin du A … il faut l'ÉCRIRE sur le chart ») ; le `4` est **posé** | Norah Jones : `A` refusé à « stack incoherent 0.61 » et affiché quand même → 72,6 % du temps de jeu de A mal représenté |
| D50 | Les crochets de 1ʳᵉ / 2ᵉ fin | `folding.py:1057` `_ireal_endings` | passages de même longueur ne différant que sur leur queue ; queue `1..2` mesures, tronc `≥ 2`, `L ≥ 3`, **et la queue doit changer de FONDAMENTALE quelque part** | `_ENDING_TAIL_MAX = 2` (`:1049`), tronc `≥ 2`, `L ≥ 3` | la règle « fondamentale, pas couleur » est **arbitrée** (2026-09-14, Sunny Afternoon : `C7` contre `Cm7` ne mérite pas un crochet) ; les trois bornes sont **posées** | Ouvert dans `known_issues.md:29-35` : les crochets notent du bruit dès que le tronc s'accorde (Cry Me A River, 5 fins / 8 passes) |
| D51 | Jamais deux blocs sous la même lettre | `folding.py:1124` `_ireal_cascade` + `:1164` `_merge_coupe` | l'hôte = le plus joué ; un passage COUPÉ (qui suit la cellule et s'arrête) le rejoint ; le reste prend un prime (A′, A″) | `_PRIMES` (`:1046`) | **arbitré** (Louis 2026-09-14 : « tu ne peux pas afficher un A deux fois » ; spec `docs/spec_affichage_sections.md`) | Un test échoue déjà à HEAD sur cette cascade (`known_issues.md:38`) |

### Étape 7d — LA BASSE SONNANTE ⚠️ **NON COMMITÉ**

C'est le seul étage **arbitré au sens fort** : 33 jugements de Louis conservés
sur disque et rejoués par un test.

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D52 | Lire la basse à l'attaque de chaque temps de l'accord | `pipeline.py:148-188` ⚠️ → `nnls_features.py:153` `bass_pc_onset` | l'accord est découpé aux temps qu'il traverse (marge `eps = 0.05 s`) ; pour chaque morceau on lit la demi-basse NNLS moyennée sur les **150 premières ms**, C-roulée, normalisée en somme ; `argmax` + part en % | `BASS_ONSET_S = 0.15` (`nnls_features.py:150`), `eps = 0.05` (`pipeline.py:177`) | `BASS_ONSET_S` **mesuré sur UN morceau** (Ready, 11 spans : 6/11 corrects en poolant tout le span, 11/11 en lisant 150 ms) ; `eps` **posé** | Pooler tout le span perd contre l'harmonique de quinte de la basse elle-même et contre les notes voisines |
| D53 | Quelle note écrire à la basse | `bass_rules.py:81` `decide_bass` | 3 branches dans l'ordre : (1) la **fondamentale** retrouvée à `≥ 30 %` quelque part dans le span → pas de slash ; (2) la quarte sus — **désarmée** ; (3) sinon la lecture du temps 1, **si** son intervalle au-dessus de la fondamentale est dans `{0,2,3,4,7}` ; sinon accord nu | `FLOOR = 30.0` (`:28`), `PLAUSIBLE = {0,2,3,4,7}` (`:48`), `REFUTED = {1,5,11}` (`:50`), `UNTESTED = {6,8,9,10}` (`:53`), `SUS_ENABLED = False` (`:78`) | **arbitré, le seul cas modèle du dépôt** : `state/human/bass_verdicts.json` porte **40 cas** (20 « basse ou note de passage ? » + 20 « la ligne écrite est-elle juste ? ») dont **33 tranchés** (7 « dunno »), plus 2 résolutions ; `tests/test_bass_rules.py` en rejoue 33 en paramétré + 13 assertions de table. Sa docstring dit : « Si une règle doit bouger, il faut de NOUVEAUX arbitrages, pas un ajustement du test. » `FLOOR = 30` est **arbitré** avec plateau explicite (`docs/bass_slash_rules.md:102-109`, table `10/17 → 14/17 → 16/17`, plateau 20–32 %). `UNTESTED` est **posé**, et le dit | Effet mesuré : 70 slashes → 28 (20 % → 8 % des accords), zéro intervalle impossible, zéro slash ajouté. **Mais sur 6 morceaux / 351 accords.** La docstring du module affirme encore qu'il n'est pas branché (§5 #1) |

**Ce que D52/D53 ne règlent pas** (écrit en toutes lettres) : découper un accord
dont la basse bouge (Ready mesure 49 = deux accords écrits comme un seul) ; et
16 accords sur 20 ont une basse mobile écrite avec un seul symbole.

### Étape 8bis — Tonalité, couleurs, drapeaux (`harmonia/harmonic_key.py`)

Tout ce bloc porte l'en-tête « Calibration constants are byte-identical to v4.1
(do not retune here) » (`:32`) — c'est-à-dire : **aucun de ces nombres n'a de
justification dans ce dépôt**, ils viennent d'un script de scratchpad
(`colour_hmm_song.py`) qui n'y est plus.

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D54 | Où la tonique change | `harmonic_key.py:106` `tonic_track` | CUSUM causal sur la « masse interdite » (le ♭2 et le ♯4 de chaque tonique rivale) ; tenir jusqu'à ce qu'on soit forcé ; les 12 premiers accords s'auto-ancrent | `TT_EPS = 0.02`, `TT_PEN = 0.8`, `TT_INIT_N = 12` (`:61-63`) | **posé** | Une modulation ratée ou inventée change les couleurs de tout un passage |
| D55 | Comment se nomme un segment de tonique | `harmonic_key.py:147-162` | la durée décide si elle est **décisive** (`≥ 1.25 ×` le second) ; sinon Krumhansl départage parmi les fondamentales tenant `≥ 0.6 ×` la plus longue | `1.25` (`:155`), `0.6` (`:159`) | **posé** — le code le dit : « Threshold calibrated on two songs = hypothesis (rule #5) » | Krumhansl seul nommait le voisin F♯-vs-F ; la durée seule laissait le IV s'asseoir à la place du I |
| D56 | Majeur ou mineur | `harmonic_key.py:176` `mode_audit` | part de tierce haussée sur les accords enracinés sur la tonique ; si la masse `< 1.5`, on retombe sur tous les accords qui voicent une tierce ; `> 0.5` → majeur | `MODE_MASS_MIN = 1.5` (`:64`), seuil `0.5` (`:201`) | **posé** | Décide majeur/mineur → décide les couleurs ET l'orthographe ♭/♯ du chart entier |
| D57 | La couleur de chaque accord | `harmonic_key.py:228` `_viterbi` + `:215` `_emission` | HMM collant à 4 états (naturel/harmonique/dorien/mélodique) sur des preuves de 6ᵉ et 7ᵉ degrés « gatées » par ce que l'accord voice | `Q=0.85`, `GAIN=25.0`, `LAMBDA=0.25` (`:53-55`), `P_N_STAY=0.94`, `P_N_TO_R=0.02`, `P_R_STAY=0.85`, `P_R_TO_N=0.12`, `P_R_TO_R=0.015` (`:56-57`) | **posé** (les 8) | N'imprime aucun symbole — colore la grille |
| D58 | Les inflexions (emprunt momentané) | `harmonic_key.py:269` | un accord dont la preuve propre dit une autre couleur que le chemin, au-dessus d'un plancher de preuve | `INFL_MIN_W = 1.0`, `INFL_MIN_MARGIN = 0.15` (`:58-59`) | **posé** | Affichage seul |
| D59 | Les accords « suspects » | `harmonic_key.py:296` `_challenges` | accord non diatonique à TOUTES les couleurs → on score 12×7 alternatives par `support × (0.7 + 0.3 × ajustement)` ; `≥ 1.25 ×` le score écrit → « challenge », sinon « suspect » | `CHALLENGE_MARGIN = 1.25` (`:60`), poids `0.7/0.3` (`:316`), vocabulaire de 7 qualités (`:67-69`) | **posé** | **Drapeau seulement** : les alternatives NNLS ont été réfutées au contrôle de prémisse (postérieure musx médiane 0.037, 0 % ajoutent une note) — `pipeline.py:696-701` ne garde que `kind`. 28 % des drapeaux tombent sur une vraie erreur |
| D60 | La tonalité affichée et son orthographe | `pipeline.py:704-710` | le segment le **plus long** donne la tonalité ; si le majeur relatif ∈ {G,D,A,E,B} → noms en dièses, sinon en bémols | ensemble `(7,2,9,4,11)` (`:708`) | **posé**, mais **d'accord** avec `kit.js:53` (`FLAT_MAJ = {0,1,3,5,6,8,10}`, exactement le complément) | Change l'orthographe de **chaque symbole imprimé** (B♭ vs A♯). Deux copies de la règle, aucun test qui les lie |

### Étape 9 — Les candidats de l'écran Annotate (`harmonia/span_rescore.py`)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D61 | Les 3 accords que le modèle aurait pu écrire à la place | `span_rescore.py:223` `musx_suggestions` | postérieures musx triade(73) + s7(4) moyennées sur le span → repliées sur 5 familles (`QUAL5`) via `_FOLD` → 60 candidats (12 × 5), masse `N` exclue → top 3 | `top_k = 3` (`:224`), `QUAL5` (`:37`), `_TRIAD_SEV_TO_QUAL5` (`:75-88`), `Q5_TAIL` (`:197`) | **arbitré** sur la SOURCE (Louis 2026-08-07 : l'éditeur doit montrer les accords que musx a prédits, pas ceux d'un autre scoreur) ; la table `_TRIAD_SEV_TO_QUAL5` est **hérité + théorie**, vérifiée contre le vocabulaire légal du modèle | N'imprime rien tant que Louis ne tape pas dessus. **Limite écrite** : un candidat est QUAL5, donc il ne distingue pas `-7` de `-9` |

### Étape 10 — Au service du chart (le dernier mot)

| # | Ce qui est décidé | Où | La règle | Constantes | Prov. | Si c'est faux |
|---|---|---|---|---|---|---|
| D62 | **Les corrections de Louis écrasent tout** | `harmonia_min/annotations.py:107` `_apply`, `:136` `overlay`, appelé par `server/routes/library.py` | un accord confirmé revient avec `confirmed = True` et `c = 1.0` | — | **arbitré** (Louis 2026-08-20) | Doublé par le garde-fou dans le repli (D44) |
| D63 | Bémols ou dièses à l'écran | `harmonia/static/ui/kit.js:53,56` | `FLAT_MAJ = {0,1,3,5,6,8,10}` : si le majeur relatif y est → bémols, sinon dièses. Réinitialisé à bémols hors contexte (`:743`) | — | **arbitré** (2026-07-19, rapport utilisateur : « G♭m7 » affiché en mi majeur) | Change tous les symboles imprimés. Règle dupliquée en Python (D60) |

---

## 7. Ce qu'il faut retenir en cinq lignes

1. **Deux décodages d'accords, pas un.** Le premier (`musx.redecode`, étape 3)
   écrit les accords ; le second (le gabarit du repli, `folding._decode_template`,
   étape 7) les **réécrit** sur 5,3 % des mesures, avec un coût de changement
   mi-mesure différent (15 au lieu de 45) que rien ne justifie sérieusement.
2. **Les deux constantes les plus lourdes sont `DEFAULT_PENALTY = 40`** (à quelle
   fréquence les accords changent, sur chaque morceau — mesure revendiquée,
   introuvable) **et `STACK_COHERENCE = 0.85`** (si une section entière est
   réécrite par le consensus ou garde son premier jet — calibré sur trois
   morceaux).
3. **Une seule famille de décisions est arbitrée au sens fort** (verdicts sur
   disque + test qui rougit) : les règles de basse. C'est le modèle à
   généraliser — et ironiquement, c'est aussi celle qui vient d'être branchée
   sans que sa docstring ni `known_issues.md` ne le sachent.
4. **Cinq « réglages » de `settings.py` ne sont lus nulle part.** `sections`,
   `merge`, `fold_loop`, `fold_gate`, `fold_transpose` documentent des décisions
   appliquées en dur ailleurs. Les changer ne ferait rien.
5. **Deux morts silencieuses.** Le ré-ancrage harmonique de la phase est mort
   comme effet de bord d'un autre changement, sans que personne ne l'ait décidé ;
   et le mode Jam est cassé depuis le même commit (§5, encadré). Les deux
   viennent de la même cause : trois copies du même appel, deux mises à jour.

---

## Annexe — les pages qui recueillent les arbitrages

La bonne boucle existe déjà et n'est utilisée que pour la basse. Les
générateurs de pages d'arbitrage vivants : `tools/page_basses.py` (règles de
basse — la seule qui a produit un ledger), `tools/arbitrage_repli.py` (repli),
`tools/roles_page.py` (équivalences de cadence), `tools/avant_apres.py`
(avant/après du rapport d'or). Depuis le commit `58a1f30` (aujourd'hui) une
route `/api/verdicts/<nom>` existe et écrit dans `state/human/verdicts/` —
**le dossier est encore vide.** C'est le canal par lequel les 22 constantes
`posé` pourraient devenir `arbitré`.
