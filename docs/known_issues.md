# Harmonia — Known Issues

Journal court, à jour. Historique complet (2026-07 à 2026-09, avant le
refactor `harmonia_min` → `harmonia`) : `docs/archive/known_issues_2026-07_2026-09.md`.
Mécanique du projet (comment vérifier un changement, où sont les fichiers) :
`docs/STATE.md`.

## Ouvert

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
  (`tools/sections_bench/`) importe proprement mais `bench.py --quick` ne
  tourne pas encore de bout en bout : `ssm_zoo.capture()` espionne
  `harmonia.sections.detect_sections`, dont la signature a changé au
  sprint 9 (elle ne reçoit plus les tableaux NNLS/musx pré-calculés que
  l'ancien dispatcher passait) — à réécrire pour lire ces features
  directement via `harmonia.nnls_features`/`harmonia.musx`.

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

## Résolu récemment

- Le paquet legacy `harmonia/` (pré-refactor), `scripts/harmonia_server.py`,
  et 137+ fichiers associés sont supprimés (tag `pre-refactor-2026-09-14`
  pour les retrouver). Voir `docs/refactor_2026-09/sprints.md`.
