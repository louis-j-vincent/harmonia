# Harmonia — Known Issues

Journal court, à jour. Historique complet (2026-07 à 2026-09, avant le
refactor `harmonia_min` → `harmonia`) : `docs/archive/known_issues_2026-07_2026-09.md`.
Mécanique du projet (comment vérifier un changement, où sont les fichiers) :
`docs/STATE.md`.

## Ouvert

### Audit 2026-09-15 — ce qui reste ouvert

Détail et mesures : `docs/audit_2026-09-15_plan.md`.

- **Le re-calage harmonique de la phase (`bars._phase_correction`) ne se
  déclenche sur aucun des 44** (seuils 0,55 / 0,15 jamais satisfaits
  ensemble). Sur 2 des 3 marques « Set bar 1 » de Louis, le vote des accords
  pointait sur sa phase à 45–50 %. Pas de correctif proposé : Let It Be
  (54 %) serait re-calé d'une demi-mesure — son oreille tranche.
- **Lost Without U (B, mes. 26) : la boucle repart avec une position de
  retard après une mesure de pause** (mes. 25, N.C. à 0,20). La grille est
  bonne ; la pile compte les positions depuis le début de la section, donc
  empile la 26 (D-7, vraie) avec les G7 des autres passes. Loi candidate :
  « une mesure de silence dans une boucle n'est pas une position » —
  contre-cas : un break qui fait partie de la boucle. Pas posée.
- **Les crochets de fin 1./2. notent du bruit dès que le tronc s'accorde** :
  les dernières mesures de chaque passe sont gardées hors pile exprès et
  décodées passe par passe ; `_ireal_endings` en fait autant de fins que
  de groupes (vu en lisant les fins sur la vue de façade : Cry Me A River
  5 fins / 8 passes ; et Lazy Song « 1. B G#m » remonte une mesure
  rejetée). Loi candidate « pas de crochets quand les queues se
  dispersent » — page dédiée à faire, pas posée.
- **Yesterday A : E- écrit, Eo entendu par Louis.** Consensus E- à 0,36
  contre Eo 0,14 — qualité d'accord du modèle, pas du repli. Annotation.
- `test_minimal_fold_separe_les_longueurs_dune_meme_lettre` échoue à HEAD
  depuis la cascade iReal (le B de 4 replié comme préfixe du B de 8).


- **RÈGLES D'OR de la basse — arbitrées à l'oreille, écrites, testées, PAS
  branchées** (2026-09-15). 33 arbitrages de Louis en deux tours d'écoute sur
  351 accords de 6 morceaux. Vérité terrain : `state/human/bass_verdicts.json`
  (ne pas régénérer). Code : `harmonia/bass_rules.py`. Garde-fou :
  `tests/test_bass_rules.py` rejoue les arbitrages — **une règle qui bouge doit
  faire rougir ce test**. Raisonnement complet, avec ce que chaque piste a
  coûté : `docs/bass_slash_rules.md`. En résumé : (1) la quinte est
  asymétrique — +7 décore, +5 porte et a la priorité ; (2) le retour de la
  basse de départ ne démonte pas une quarte (4 arbitrages sur 4) ; (3) le
  discriminant n'est aucune de nos mesures mais « le candidat est-il la
  fondamentale de l'accord » (16/17, contre 10/17 pour cinq curseurs) ; (4)
  ce qui jette une basse, c'est l'intervalle, jamais la confiance (un plancher
  de confiance se trompe 3 fois sur 4 ; une 9e à la basse est vraie, une b9
  jamais). Effet mesuré : 70 slashes → 28 (20 % → 8 % des accords), zéro
  intervalle impossible, zéro slash ajouté. **Ne résout pas** : 6 morceaux
  seulement ; rien dans `pipeline.py` ne l'appelle ; la basse mobile
  (16 accords sur 20 en ont une) reste écrite avec un seul symbole.
  **Révisions du soir même**, après arbitrage des deux cas restés ouverts :
  la règle (2) a perdu sa preuve propre (3 de ses 4 appuis sont déjà
  expliqués par la règle (3), le 4e était mal classé) — conservée mais
  marquée, et non utilisée par la procédure livrée. Et surtout, les deux cas
  ouverts n'étaient PAS des questions de basse : Ready mesure 3 est une
  **étiquette d'accord fausse** (Louis : `Ab^7`, pas `Eb`) — la règle de
  basse ne peut être juste que si l'accord l'est ; Ready mesure 49 est un
  **changement d'accord manqué** (Louis : `Bb-7` puis `Eb6b9`, un seul accord
  écrit sur 4,5 s à cheval sur deux mesures). Sur ce dernier, musx tient
  `Bb-` de 73 % à 95 % et ne voit jamais `Eb` (2,3 % max) : **c'est la
  lecture de basse qui pose la frontière que le détecteur d'accords rate**,
  exactement l'inverse de la règle (3). Piste suivante identifiée : se servir
  de la basse comme détecteur de frontière d'accord.
- **Basse détectée à l'attaque, pas en moyenne sur tout l'accord — 6/11 → 11/11
  sur un morceau, PAS ENCORE branché au pipeline live** (2026-09-15, piste
  ouverte par Louis : « dans le doute entre deux notes basses, si l'une est
  la quinte de l'autre, on prend la fondamentale »). Cette règle-là, testée
  telle quelle sur les 11 accords des mesures 1-8 de "Ready" (PJ Morton) :
  6/11 → 4/11, elle casse plus qu'elle ne répare — l'intervalle de quinte
  seul ne dit pas quel côté fait autorité (Ab^7 confirme l'intuition : Eb est
  bien un harmonique parasite d'Ab ; mais Bb-7, Eb7, F7 cassent pour la
  raison inverse, leur « quinte du dessus » est une vraie note jouée, une
  quarte ascendante). Le vrai signal : le **timing**, pas l'intervalle — ne
  lire que les 150 premières ms de chaque accord (l'attaque) au lieu de
  moyenner tout le span : 10/11, et le 11e (un "Db" lu "Eb") était en fait un
  Db/Eb — Eb était donc la bonne basse SONNANTE depuis le début, 11/11 réel.
  Mécanisme : la fondamentale domine à l'attaque ; sa propre quinte
  harmonique et les notes de passage voisines s'accumulent ensuite avec la
  résonance/pédale.
  Persisté : `harmonia.nnls_features.bass_pc_onset()` (+ `BASS_ONSET_S=0.15`),
  testé unitairement (`tests/test_bass_pc_onset.py`, données synthétiques).
  **Ce que ça NE résout PAS** : mesuré sur UN SEUL morceau (règle #5, CLAUDE.md)
  — pas de banc corpus encore. Pas câblé dans `infer` ni dans `c["bass"]` du
  chart live : `pool_beats()` (la version tout-le-span) reste inchangée et
  continue seule à nourrir les têtes entraînées (root/quality), qui ont été
  entraînées sur CE pooling précis — leur substituer l'attaque sans
  réentraînement dégraderait ces têtes en silence (règle #6). Prochaine étape
  avant tout branchement live : bench corpus (POP909 ou RWC, bass GT connue).
- **Corrections « Set bar 1 » / Annotate invisibles d'un checkout à l'autre**
  (2026-09-15, investigation demandée par Louis). Pas un bug du code de
  propagation lui-même — vérifié ligne par ligne (`api_bar1` → `jobs.start_job`
  → `_run_job` → `analyze_steps(bar1_time=...)`, et `tools/golden.py` pour le
  moteur `harmonia`) : tout est correctement câblé, et le chart servi
  (`state/cache/charts/min_191P7nIeECo.json`) porte bien la marque corrigée
  (1.355s) et la bonne grille. Le vrai problème : `state/human/marks/` et
  `state/human/annotations/` sont sous git mais PAS partagés entre checkouts —
  une correction faite sur le serveur d'un worktree (ici `:7773`,
  `.claude/worktrees/refactor`) reste invisible du serveur d'un AUTRE checkout
  (`:7772`, `feat/section-criteres`) tant qu'elle n'est pas commitée+mergée :
  vérifié, `state/human/marks/191P7nIeECo.json` et
  `state/human/annotations/min_191P7nIeECo.json` n'existent QUE dans le
  worktree refactor (`git status` les montre `??`, jamais ajoutés). Deux
  serveurs tournent en ce moment (`ps`: PID 44180 sur :7772, cwd = checkout
  principal ; PID 50047 sur :7773, cwd = worktree refactor, up depuis la
  veille ~21h — process de longue durée, code potentiellement pas rechargé
  depuis). Si Louis corrige un morceau sur un serveur puis regarde le résultat
  sur l'autre, ça ressemble exactement à « le recalage ne se transmet pas ».
  Pas de fix appliqué (délégué à l'agent dédié de Louis) — actions
  candidates : commit+push les fichiers `state/human/` non trackés avant de
  changer de checkout ; vérifier laquelle des deux instances (:7772/:7773)
  tourne du code à jour.
- **Test hérité rouge.** `tests/test_songformer_sections.py::test_minimal_fold_separe_les_longueurs_dune_meme_lettre`
  échoue depuis la fusion de la cascade iReal (commit `8146efd`, 2026-09-14) :
  la cascade a changé ce que ce test affirme sur le repli d'une même lettre à
  deux longueurs. Appartient à la piste qualité-des-sections (voir plus bas) ;
  pas un bug du refactor.
- **Recherche Ultimate Guitar : HTTP 500.** Le site renvoie un blocage
  anti-bot (même famille que le blocage YouTube ci-dessous) sur :7772 comme
  sur :7773 ; la route logge un WARNING et rend une liste vide. Fichier :
  `harmonia/integrations/tab_fetcher.py` (route : `harmonia/server/routes/irealb.py`).
- **Deux morceaux « froids » dans le rapport d'or.** `min_autumn_leaves` et
  `min_h_D3VFfhvs4` n'ont pas de cache SongFormer chaud — le rapport les
  saute plutôt que de relancer un modèle. Cache : `state/cache/songformer/`.
- **Deux clés toujours vides dans le rapport de repli** (`fold.demiton`,
  `fold.pos_skip`, `harmonia/folding.py`), gardées vides pour ne pas changer
  la forme du JSON du chart (sprint 7). À retirer avec une page avant/après
  quand le schéma du chart sera formalisé.
- **Trois scripts archivés lisent l'ancienne clé de cache musx.**
  `archive/scripts/page_concordance_barre.py`, `backfill_musx_sug.py`,
  `chart_retour.py` testent `musx_cache_path(audio).exists()` au lieu de
  `harmonia.cache.exists()` (seul à connaître le repli d'ancienne clé) — 20
  morceaux perdraient leurs sections en silence s'ils tournaient (sprint 15).
  Archivés, pas exécutés en prod ; à corriger seulement s'ils reviennent.
- **`data/cache/` garde ses anciennes clés.** Les caches beats/musx/nnls
  utilisent encore `<stem>` au lieu de `<stem>__<taille>` (`harmonia.cache`
  lit les deux, avec repli journalisé). Renommage en place prévu à la bascule
  prod (sprint 21 du refactor) — disque à 96 %, pas de copie possible avant.
- **SIGKILL SongFormer sur les longs morceaux, mitigé.** SongFormer peut
  saturer la mémoire sur un morceau long (mesuré : code retour 137 sur un
  morceau de 422 s, machine 16 Go). Depuis le sprint 9 du refactor, il tourne
  dans un sous-processus jetable (`harmonia/sections/songformer.py`) : sa
  mort fait échouer le job PROPREMENT (erreur remontée), au lieu de tuer le
  serveur. Vérifié dans le code (pas de `try/except` autour de l'appel
  enfant) ; pas re-testé sur un vrai morceau long depuis le portage.
- **`docs/harmonia_min_explainer.md` est désormais historique.** Il décrit
  `harmonia_min`, la référence du refactor, pas le paquet `harmonia/` actuel
  — utile pour le RAISONNEMENT (pourquoi SongFormer tourne en sous-processus,
  etc.), pas pour un chemin de fichier. Marqué en tête du fichier.
- **La recherche qualité-des-sections reprend après la bascule prod**
  (décision de Louis, `docs/refactor_2026-09/plan.md`). Le banc de recherche
  (`tools/sections_bench/`) n'importe PAS proprement — corrigé le 2026-09-16,
  cette ligne disait le contraire : une douzaine de ses scripts importent
  `harmonia_min.harmonic_sections`, `harmonia_min.voice_sections` et
  `harmonia_min.sections`, trois modules supprimés au sprint 9, bien avant la
  disparition de `harmonia_min` au sprint 22. Panne antérieure à ce sprint,
  jamais causée par lui. Et `bench.py --quick` ne
  tourne pas encore de bout en bout : `ssm_zoo.capture()` espionne
  `harmonia.sections.detect_sections`, dont la signature a changé au
  sprint 9 (elle ne reçoit plus les tableaux NNLS/musx pré-calculés que
  l'ancien dispatcher passait) — à réécrire pour lire ces features
  directement via `harmonia.nnls_features`/`harmonia.musx`.
- **`confirmed` protège l'AFFICHAGE, pas le calcul du repli** (trouvé
  2026-09-16 en vérifiant qu'un accord ajouté à la main survit à Souder).
  `library.chart_model` réapplique `annotations.overlay` à CHAQUE lecture, donc
  un accord verrouillé se réaffiche correctement même après un Souder qui a
  tout recalculé — vérifié : verrouillé un accord d'une mesure vide (l'outro
  de `min_Ju8Hr50Ckwk`), relancé Souder sur les mêmes sections (mesures
  réécrites), l'accord verrouillé était toujours là à la relecture. Mais rien
  n'alimente jamais `confirmed` dans les mesures BRUTES avant que
  `folding.fold_letter_groups`/`harmonia_min.folding` ne les empile — son
  garde-fou « `if any(c.get("confirmed") ...)` » (`harmonia/folding.py:587`,
  testé par `tests/test_fold_respecte_annotations.py`) ne se déclenche donc
  jamais en pratique : rien dans `pipeline.py`, `refold.py` ni
  `harmonia_min/soudure.py` ne lit le sidecar (`state/human/annotations/`)
  avant le repli. Conséquence concrète : une mesure verrouillée continue de
  fournir sa valeur D'AVANT verrouillage quand le repli MOYENNE les
  occurrences d'une même lettre — l'affichage de CETTE mesure reste juste
  (l'overlay la corrige après coup), mais elle peut fausser en silence la
  moyenne des mesures VOISINES de la même lettre. Pas de correctif proposé :
  il faudrait faire lire le sidecar à `accords_par_mesure`/aux étapes du
  pipeline avant le repli, une piste distincte. `harmonia/server/routes/
  annotate.py::fill_empty_bars` (2026-09-16) est un correctif plus étroit,
  sans rapport : il comble seulement le cas où `overlay` n'a même pas
  d'accord-hôte à modifier (une mesure entièrement tenue, `bars == []`).

## Trois trouvailles récupérées d'un arbre non commité, auditées le 2026-09-15

Trois tests non commités traînaient dans l'arbre principal (fichiers datés du
2026-07-21 au 2026-08-02, jamais commités ni jetés). Le code qu'ils testaient
a disparu avec le legacy (sprint 1) : les fichiers sont jetés, mais chaque
trouvaille est vérifiée contre le code vivant et notée ici.

- **Accord « add9 » mal lu comme dominant.** Un accord iReal comme « Aadd9 »
  (un accord majeur avec une neuvième ajoutée, sans septième) était compris
  comme un accord dominant, parce que le chiffre « 9 » de « add9 » suffisait
  à déclencher la règle. Vivait dans l'IMPORT iReal (`harmonia/irealb_fetcher.py`,
  supprimé) — l'import iReal n'existe plus dans l'app (seuls la recherche et
  l'export ont survécu, décision du refactor). **Rien à corriger aujourd'hui** ;
  si l'import revient un jour, ne pas chercher un chiffre dans le texte de
  l'accord, comparer le suffixe entier.
- **Un accord dominant non résolu devrait pencher vers le relatif mineur.**
  Sur une grille « Do maj7, Mi min7, Fa maj7, La7 », le La7 (qui contient un
  Do dièse) n'a de sens que comme le 5e degré de Ré mineur — mais sans accord
  de Ré qui suit pour le confirmer, l'ancien traceur de tonalité restait sur
  Do majeur quand même. Vivait dans `harmonia/theory/local_key.py` (supprimé,
  lisait le CHART, pas l'audio). **Pas transposable tel quel** : le traceur de
  tonalité vivant (`harmonia/harmonic_key.py`) lit le son réel (le chroma),
  pas les symboles d'accords, et accumule la preuve accord par accord
  (CUSUM) plutôt que de comparer des gammes. L'ancien bug ne peut plus
  arriver puisque son code est parti, mais personne n'a écouté si la même
  situation musicale (une dominante ambiguë, jamais résolue) trompe le
  nouveau traceur À SA manière. À vérifier à l'oreille quand la piste
  qualité-des-sections reprendra.
- **Un accord de substitution casse le comptage des répétitions.** Sur This
  Love, trois refrains identiques suivis d'un quatrième avec UN accord
  différent (une substitution dominante, très courante en vrai) étaient vus
  comme un seul bloc de 79 mesures au lieu de 3 répétitions + 1 variante,
  parce que la règle de groupage exigeait que CHAQUE accord corresponde
  exactement. Vivait dans `harmonia/output/chart_model.py` (supprimé, le
  rendu de l'ancienne app :7771). **Le groupage a changé de nature** : le
  repli vivant (`harmonia/folding.py`) groupe par l'ÉTIQUETTE que le
  détecteur de sections (SongFormer, qui écoute le son) a déjà posée, pas en
  comparant les accords — ce bug précis ne peut plus arriver là. Reste une
  question ouverte, différente : quand plusieurs passages de la même lettre
  sont empilés et moyennés, un passage avec une substitution est-il bien
  écarté comme exception (le contrôle de cohérence existant), ou fausse-t-il
  la moyenne en silence ? Pas vérifié ; même piste de recherche.

## Sprint 22 — fusionné (2026-09-16)

Fusionné dans `feat/section-criteres` (`3715466`). Vérifié indépendamment
après fusion, à part du merge lui-même : `harmonia_min/pipeline.py` avait
disparu, donc le rapport d'or a été relancé avec `--engine harmonia` contre
l'ancienne baseline, comparaison champ par champ (pas seulement « mesures
changées ») — **0 différence en dehors de `meta.engine`** sur les 44
morceaux communs. La baseline est re-gelée depuis le run PROPRE de l'agent
(`state/cache/golden/baseline_sprint22_2026-09-16/`, fait dans son worktree
isolé, donc jamais mélangé au travail de basse en cours dans l'arbre
partagé) — jamais depuis un run local, qui aurait mélangé les deux
chantiers. `pytest` : un test sentinelle ajouté le jour même par une autre
session (`test_jam_redecode.py::test_les_trois_appelants_existent_toujours`)
listait encore `harmonia_min/jam.py` en dur ; mis à jour vers
`harmonia/jam.py`, où le fichier vit maintenant — c'est exactement ce que ce
test est censé attraper. Suite complète : 303 verts, la seule rouge connue
inchangée.

## Résolu récemment

- **`harmonia_min` supprimé (sprint 22, 2026-09-16).** Le paquet n'était plus
  qu'un décor : sur ses 20 modules, 12 étaient des PONTS de 2 à 7 lignes
  (`from harmonia.X import *`) qui réexportaient `harmonia` — la dépendance
  était inversée depuis les sprints 2-10. Seuls 7 fichiers portaient du code
  que `harmonia/` n'avait pas ; ils sont déplacés tels quels (`git mv`, 97 à
  100 % de similarité détectée par git) à plat dans `harmonia/` : `soudure.py`,
  `ssm_page.py`, `section_tool.py`, `jam.py`, `phrases4.py`, `titles.py`,
  `annotations.py`. Les 22 sites d'import de `harmonia/` visent maintenant le
  module réel ; aucun `import *` n'a été ajouté (c'est ce que faisaient les
  ponts, et ils sautaient les noms à underscore — un ajout concurrent,
  `_ireal_cascade`, avait déjà cassé là-dessus le 2026-09-14).
  Vérifié champ par champ, pas seulement « 0 mesure changée » : le run frais
  contre la baseline gelée donne **un seul chemin JSON différent,
  `.meta.engine`, sur 44 morceaux sur 44**. Écrans Soudure, /ssm et éditeur de
  sections ouverts à 390×667 sur un serveur jetable (:7775) — zéro erreur
  console, contenu réel affiché. Tests : 267 passés, 1 rouge (le rouge hérité
  ci-dessus, seul et inchangé).
  **Ce que ça ne résout PAS** : (a) `harmonia_min/` existe toujours, réduit à
  son `__init__.py` + `chord_lm/` — recherche en cours d'une autre session,
  intouchée ; (b) `HARMONIA_MIN_PORT` reste lu par `settings.py`,
  volontairement : c'est le nom qu'emploient les serveurs de worktree pour ne
  PAS se poser sur :7772, le retirer les ferait tomber sur l'app vivante ;
  (c) la baseline n'est pas re-gelée (voir juste au-dessus).

- **Compass et Guide montrent jusqu'à cinq accords et la basse entendue
  (2026-09-16, demande de Louis).** « Je veux qu'il apparaisse les autres
  suggestions que juste le top 2, plus les suggestions sur la ligne de basse
  en faisant des petits cercles autour des lettres du cercle […] toujours la
  taille du cercle proportionnelle à la proba d'être détectée, top 3 basses,
  top 5 accords si relevant. »
  Les deux vues partagent déjà leur source (`candList`, un seul endroit), donc
  la liste d'accords s'allonge une fois pour les deux. Le « si relevant » est
  un PLANCHER, jamais un remplissage, et c'est la même loi pour les deux
  listes : **un candidat ne s'affiche que s'il bat le pur hasard sur son
  propre jeu de candidats**. musx répartit sa masse sur 60 cases (12 racines ×
  5 familles), l'uniforme vaut 1,67 %, plancher **2 %** (`SUG_FLOOR`) ; la
  basse répartit la sienne sur 12 classes de hauteur, l'uniforme vaut 8,3 %,
  plancher **12,5 %** (`BASS_SUG_FLOOR`, une fois et demie l'uniforme). Le
  premier candidat passe toujours : un accord très sûr ouvre l'éditeur sur lui
  seul, comme avant, jamais sur du vide.
  Le plancher a été choisi AVANT d'écrire le rendu, en mesurant l'échelle des
  postérieures sur les deux morceaux les plus ambigus du banc (Yesterday,
  19 accords ; Lost Without U, 74) : au rang 5 la médiane vaut 0,017–0,025,
  donc 2 % tombe pile où le 5e candidat cesse de dire quelque chose. Effet
  mesuré : **3,7 candidats par accord** en moyenne contre 3 figés avant, 12/19
  et 47/74 accords en montrent ≥ 4. À 5 % on retomberait à 2,4–2,9, c'est-à-dire
  l'ancien top-3 : rien n'aurait bougé.
  La basse est lue à l'ATTAQUE de chaque accord (`bass_pc_onset`, 150 ms — la
  mesure du 2026-09-15, 6/11 → 11/11 sur « Ready »), sur le `bothchroma` que
  `pipeline.py` a déjà en mémoire à cette étape : aucune extraction en plus.
  Rendu : un anneau bleu autour de la LETTRE de la roue (une lettre de la roue
  est une classe de hauteur — exactement ce qu'une basse nomme), aire ∝
  probabilité comme les orbes d'accord depuis le 2026-08-08. Bleu et pas la
  teinte de quintes de l'orbe : la basse est une autre VOIX, pas un autre
  accord.
  Code : `harmonia/span_rescore.py` (`SUG_FLOOR`, `BASS_SUG_FLOOR`,
  `bass_suggestions`), `harmonia/pipeline.py`, `harmonia/static/screens/
  annotate.js` (`candList`, `bassList`, `byArea`, `bassStrip`),
  `chart.js` + `prefs.js` (la liste blanche des champs, et la transposition).
  Tests : `tests/test_suggestions_floor.py`.
  **Régression trouvée et corrigée en route — le plancher vide la roue sur les
  accords les PLUS sûrs.** Le compas retire l'accord écrit avant de placer ses
  orbes (2026-08-08 : le redessiner sur la jante ne disait rien et volait un
  rayon). Avec un top-3 figé, en retirer un en laissait toujours deux. Avec le
  plancher, un accord que musx tient à 95 % ne garde que SA propre case : la
  roue se vidait, et `noCandBox` annonçait « written without the model's
  ranking » — l'exact contraire de la vérité, précisément là où le modèle est
  le plus sûr (2 accords sur 23 dans Yesterday, 2 sur 86 dans Lost Without U).
  `buildCompass` distingue maintenant deux silences : aucun classement du tout
  (→ `noCandBox`, inchangé) versus le modèle est sûr (→ la roue reste, avec
  son moyeu, ses anneaux de basse, et l'en-tête « the model is sure — 95 % on
  this chord »). Gelé côté Python par
  `test_un_accord_tres_sur_ne_garde_que_sa_propre_case`.
  **CE QUE ÇA N'ÉCRIT PAS, et c'est ce qui permet de le brancher aujourd'hui** :
  aucun slash n'apparaît dans le chart. Le champ `bass` de l'accord n'est pas
  touché, `bass_rules.decide_bass` n'est pas appelé. Le banc corpus qu'exige
  l'entrée « Basse détectée à l'attaque » ci-dessus porte sur la DÉCISION
  d'écrire une basse, pas sur le fait de MONTRER la lecture. Les deux
  planchers de basse qui cohabitent désormais (`bass_rules.FLOOR` = 30 %, en
  pourcents, décide ; `BASS_SUG_FLOOR` = 0,125, en part, affiche) sont gardés
  distincts par un test exprès — les confondre serait l'erreur de calibration
  silencieuse de CLAUDE.md #1.
  **Ce que ça ne résout pas** : (a) le plancher est calibré sur la FORME de
  l'échelle (où le 5e rang décroche), il n'a pas été arbitré à l'oreille, et
  sur deux morceaux seulement (règle #5) — à rouvrir si la liste paraît trop
  longue ou trop courte ; (b) les anneaux de basse ne sont pas tapables, donc
  on voit une basse sans pouvoir la choisir — décision à confirmer par Louis,
  la rendre tapable voudrait dire écrire un slash, ce que (ci-dessus) on
  refuse tant qu'il n'y a pas de banc ; (c) la basse MOBILE (une note par
  temps) reste résumée par une seule lecture à l'attaque de l'accord ; (d) les
  charts d'avant le 2026-09-16 n'ont ni 5 candidats ni `sugBass` tant qu'ils
  ne sont pas recuits — l'affichage marche sans, il montre juste ce qu'il a.

- **L'accord optionnel « en petit au-dessus », comme iReal (2026-09-16).**
  Louis, sur Easy On Me section B : « c'est un F puis la basse descend sur D
  (donc ça donne un D-7) avant d'atterrir sur le C … typiquement le genre de
  cas où j'aimerais avoir juste F affiché et le D-7 en optionnel en petit
  au-dessus en suggestion, comme iReal fait. » `harmonia.folding._bar_variant`
  lit, à chaque position d'un gabarit replié, le décodage de première passe
  de chaque passe GARDÉE avant que le gabarit ne les réécrive toutes — la
  seule fenêtre où cette information existe encore — et retient l'accord
  minoritaire (confiance ≥ `VAR_MIN_CONF = 0.5`) le plus soutenu, groupé par
  fondamentale + famille (`harmonia.roles.family`, pour que « D- » et « D-7 »
  comptent comme la même lecture). Posé comme champ `var` sur l'accord écrit
  — un champ que le rendu (`harmonia/static/screens/chart.js`) attendait déjà
  depuis le 2026-08-17 (« l'autre lecture de cette case, sur un chart replié
  ») sans qu'aucun code serveur ne l'alimente ; distinct de `sug` (le
  classement du modèle sur le span de l'accord ÉCRIT, une question d'
  incertitude intra-mesure — `var` est un accord RÉELLEMENT joué par une
  minorité d'AUTRES passes, une question de désaccord inter-passes).
  Mesuré sur Easy On Me (`min_X-yIEMduRXk`, section B position 0) : F écrit,
  3 des 6 passes empilées jouent F puis D- au 4e temps (0,749/0,698/0,722 de
  confiance) — `var: {root: D, q: "-", c: 0.749, n: 3}`. Rapport d'or sur 45
  morceaux cuits : 6 changent (`identique: false`, dont Easy On Me), tous à
  **0 mesure changée** — vérifié en retirant `var` de chaque chart, byte-
  identique à la baseline dans les 6 cas : aucune régression racine/qualité/
  basse, uniquement l'apparition du nouveau champ.
  **Ce que ça ne résout pas** (à confirmer par Louis) : (1) l'orthographe
  affichée est celle de la passe la PLUS confiante du groupe (ici « D- »,
  pas « D-7 » bien que 1 des 3 passes ait décodé la septième) — pas un
  mélange inventé, mais peut-être pas ce que Louis préfère voir ; (2) une
  seule variante par position (si deux temps différents portaient chacun
  leur propre lecture minoritaire, seule la plus soutenue survivrait) ;
  (3) seuil `VAR_MIN_CONF = 0.5` mesuré sur UN SEUL morceau (règle #5,
  CLAUDE.md) — à vérifier corpus entier ; (4) ne couvre que le chemin
  `fold_letter_groups` (sections auto-détectées) — le chemin
  `harmonia.soudure.sections_pour_chart` (découpage à la main, déplacé de
  `harmonia_min` au sprint 22, fusionné le même jour que cette entrée) n'a
  pas son propre calcul de variante.
  `harmonia/folding.py` (`_bar_variant`, `_attach_variant`, `VAR_MIN_CONF`),
  `harmonia/pipeline.py` (docstring ChartModel), `harmonia/static/screens/
  chart.js`, `tests/test_folding_variant.py`.

- **Un temps en trop à 0,28 fois le temps décale toutes les barres après lui
  (2026-09-15, Louis sur Ready de PJ Morton).** Son diagnostic : « entre la
  mesure 8 et la mesure 9 il y a un petit temps de pause qui devrait être
  détecté […] on décale tous les accords d'un quart de barre ». La pause est
  réelle et mesurée (le temps à 20,90 s est à −15 dB sous le niveau courant,
  musx y met 12–18 % sur « aucun accord »), mais ce n'est pas elle qui décale :
  Beat This! pose **deux marques à 180 ms** à 13,9 s, soit 0,281 fois le temps
  médian, et `drop_duplicate_beats` coupait à 0,25. La paire passait, la liste
  gardait un temps de trop, et toute la grille de mesures construite par
  arithmétique d'indices tombait un temps trop tôt à partir de là — mesure 9 à
  20,900 au lieu de 21,520.
  Le seuil ne pouvait pas monter : remesuré sur les 199 morceaux en cache, les
  intervalles courts ne font pas les trois paquets nets qu'annonçait la
  docstring — la distribution est continue de 0,10 à 0,40 et le paquet des
  triolets (0,34–0,36, 130 cas) est trop proche. Ce qui sépare proprement,
  c'est ce que la paire COUVRE du temps d'avant au temps d'après : **1,0
  période** (142 cas) ou **2,0 périodes** (459 cas), creux franc entre les
  deux. Règle ajoutée, additive, sans toucher au seuil existant.
  Vérification par un juge extérieur : les barres retombent exactement sur les
  temps forts de Beat This! (18,960 / 21,520 / 24,100 / 34,380) et les trois
  accords que Louis avait déplacés à la main retombent à 10 ms près tout
  seuls. Portée : 9 morceaux / 15 paires sur 197 en cache ; rapport d'or 16
  morceaux, 82 mesures (page d'effets de bord à part).
  `harmonia/beats.py::drop_duplicate_beats`, `tests/test_beats_jumeaux_larges.py`,
  pages `/plots/ready_jumeau.html` et `/reports/jumeau_large/avant_apres.html`.
  **Ce que ça ne règle pas** : (a) la phase des barres reste un seul nombre
  pour tout le morceau (`bars.py`, `Counter(...).most_common(1)`) — un morceau
  qui gagne vraiment un temps au milieu reste inécrivable ; (b) l'accord posé
  sur le temps où il « prend l'avantage » plutôt que sur celui où il est joué
  — hypothèse mesurée et **refusée comme loi** : sur 2132 changements / 46
  morceaux, l'accord écrit est déjà l'argmax dans 91,0 % des cas et le décaler
  d'un temps ne monte qu'à 91,4 % ; 4 % de cas stricts, dont 29 % sur le seul
  *Man I Need* (le même Gb, 25 fois, en section repliée). Voir
  `scratchpad/early_switch_screen.md`.

- **La façade d'un bloc ×N montre le consensus, jamais une mesure rejetée
  (2026-09-15, verdicts de Louis sur la page d'arbitrage).** Lazy Song A
  affichait « B Ab- » ×6, Every Breath « Ab » ×7, Yesterday « E- D-/A » ×4 :
  le consensus était juste, la passe affichée portait une mesure rejetée
  par la pile qui gardait son propre décodage. `folding.facade_view`, une
  vue d'affichage lue par les deux chemins de rendu pour le BLOC écrit ;
  les fins iReal et la cascade restent sur le brut (lues sur la vue,
  elles exposaient le bruit des dernières mesures — Louis : « ça a
  empiré tous les morceaux », retiré le soir même) ; `bars` intact ;
  jamais sur une passe seule ni à deux. 4 mesures changent, les trois
  verdicts.

- **La tenue qui ouvre une boucle repliée n'était jamais écrite
  (2026-09-15, Louis sur Let It Be : « le premier accord (C) n'est jamais
  propagé sur la première barre »).** `folding._decode_template` écrivait
  la tenue au temps 1 pour toutes les positions sauf la 0 ; le segment qui
  chevauche le début de la copie du milieu du gabarit ×3 devient cette
  tenue (jamais un N.C.). Let It Be A : C~ G | Am F | C G | F C.

- **Recherche de latence musx supprimée (2026-09-15, audit, accepté par
  Louis).** Mesuré sur 46 morceaux : les changements du modèle tombent
  23–46 ms AVANT la battue Beat This!, jamais après ; la recherche ne testait
  que 0…+280 ms. Relancée par le décodage du gabarit du repli, elle aliasait
  d'une période de battue : 18 morceaux (239 mesures) avec des accords un
  temps trop tôt, invisibles dans `meta.musx_latency_ms` (champ disparu).
  Une loi : pas de latence. Baseline re-gelée, bibliothèque republiée.
- **Passe affichée = la plus riche, sur les deux chemins (2026-09-15).**
  `sections_pour_chart` (découpage à la main) écrivait `occ[0]` ; Stand By
  Me affichait son intro basse-voix (5 N.C. sur 8, ×6). Même loi que
  `minimal_fold` (`folding.pass_evidence`). 9 morceaux changent de passe.

- Le paquet legacy `harmonia/` (pré-refactor), `scripts/harmonia_server.py`,
  et 137+ fichiers associés sont supprimés (tag `pre-refactor-2026-09-14`
  pour les retrouver). Voir `docs/refactor_2026-09/sprints.md`.
