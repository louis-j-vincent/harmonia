# Refactor « clean app » — journal des sprints

Plan complet et décisions verrouillées : `docs/refactor_2026-09/plan.md`
(copie du plan approuvé le 2026-09-14). Une entrée par sprint, avec le
raisonnement, pas seulement le résultat — c'est ce qu'une session fraîche lit
en premier.

Branche `refactor/clean-app`, worktree `.claude/worktrees/refactor` (durable,
exclu de git par `.git/info/exclude` — PAS sous `/private/tmp`, où neuf
worktrees d'autres sessions sont déjà morts). Point de retour : tag
`pre-refactor-2026-09-14` = `689a21c` sur `feat/section-criteres`.

## Sprint 0 — 2026-09-14 · outillage du rapport d'or

**Fait.** `tools/golden.py` (rejoue la bibliothèque avec un moteur, compare à
une baseline), `tools/avant_apres.py` (page AVANT/APRÈS entre deux dossiers,
généralisée depuis `scripts/avant_apres.py` du 2026-08-12), `tools/render_check.py`
(capture Playwright du VRAI serveur, depuis `handoff_cleanup/check.py`).
Serveur du worktree sur :7773 (`HARMONIA_MIN_PORT=7773`), jamais :7772.

**Mesuré.**
- 46 charts cuits, 2 froids (`min_autumn_leaves`, `min_h_D3VFfhvs4` : pas de
  cache songformer — on ne relance jamais un modèle pour le rapport), 0 erreur.
  ~1 s par morceau à caches chauds.
- Déterminisme : run B contre run A → **46/46 identiques, 0 mesure changée**.
  Baseline gelée dans `harmonia_min/state/golden/baseline/`.
- Disque vs code d'aujourd'hui : **4 morceaux diffèrent, 42 identiques**. Les
  quatre (`maroon_5_this_love`, `this_love_mergetest`,
  `she_will_be_loved_mergetest`, `nDUVEUjOKMw` = Sunny Afternoon) ont été
  écrits le 2026-09-14 entre 18:10 et 18:14 par la session concurrente qui a
  commité la fusion de lettres opt-in (`9d25f45`) — pas par un recuit. AVANT
  = son expérience allumée, APRÈS = le défaut. Rien à arbitrer pour le
  refactor : la baseline est le comportement par défaut.

**Trouvé en chemin (à traiter aux sprints indiqués).**
- Fichiers lus à l'exécution mais NON suivis par git : `harmonia/models/nnls24_heads.npz`
  et tout `harmonia/third_party/…/data/` (le décodeur musx vendu ouvre
  `data/submission_chord_list.txt` relativement à SON dossier). Le worktree
  les a reçus par lien symbolique ; sprint 1 en fait des assets suivis.
- Recette d'amorçage d'un worktree : liens `.venv`, `data`, `docs/audio`,
  `harmonia/models/nnls24_heads.npz`, `harmonia/third_party/…/data`,
  `…/cache_data` est suivi ; copie (instantané) de
  `harmonia_min/state/{charts,beats,sections,songformer,annotations,chart_meta.json}`.
- La bibliothèque VIVANTE contient encore 7 charts « … — AVANT » publiés par
  la comparaison du 2026-08-12 et jamais nettoyés (`old_4JkIs37a2JE`, …) —
  retirés de l'instantané du worktree seulement. Sprint 15 (état) les traite.
- `min_T64BgKEL-Sw_pile` partage l'audio de `min_T64BgKEL-Sw` (via
  `audio_url`) : il n'est pas froid, contrairement à ce que le plan supposait.
- Le seul champ volatil du chart est `meta.musx_latency_ms` (confirmé).
- Disque : 6,7 Go libres après le checkout du worktree (les 488 fichiers
  suivis de `scratchpad/` pèsent) ; 9,5 Go au moment du commit.

**Suivant.** Sprint 1 : libérer le nom `harmonia/`.

## Sprint 1 — 2026-09-14 · libérer le nom `harmonia/`

**Fait.** Le paquet legacy `harmonia/` (137 fichiers), `scripts/harmonia_server.py`
(l'ancienne app :7771), `golden/` (brick 0, condamné) et 89 tests (84 qui
importaient le legacy, 5 du chord-LM abandonné) sont supprimés de l'arbre.
Point de retour : tag `pre-refactor-2026-09-14`. Le nouveau `harmonia/` naît
avec `settings.py` (le seul endroit qui lit l'environnement ; les choix
d'algorithme y sont des constantes), `cache.py` (une convention de chemin,
clés inchangées jusqu'au sprint 15), `assets/nnls24_heads.npz` (suivi par
git, il était orphelin non suivi) et `integrations/` : `irealb_fetcher` (la
moitié « recherche » seulement — la conversion irealb://→ChordChart et le
rendu HTML servaient l'ancienne app), `irealb_export`, `tab_fetcher`.

**Correction au plan.** Le clone music-x-lab n'est pas « du code jamais
exécuté » : `musx.py` fait `chdir` dedans et importe `mir`, `extractors`,
`chordnet_ismir_naive`, et le décodeur ouvre `data/submission_chord_list.txt`.
Il reste donc entier, déplacé en `third_party/musx_ismir2019/` (dépendance
tierce, MIT), et son dossier `data/` (2,4 Mo, 29 fichiers) devient suivi —
il ne l'était pas, comme `nnls24_heads.npz` : deux fichiers dont le chemin
vivant dépendait et qu'un clone frais n'aurait pas eus.

**Rebranché dans harmonia_min.** `server.py` (3 imports →
`harmonia.integrations.*`), `musx._musx_dir` (repli → `third_party/musx_ismir2019`),
`nnls_features.HEADS_NPZ` (→ `harmonia/assets/`). `chord_lm/corpus.py` garde
un import legacy cassé : le chord-LM est abandonné et rien ne l'importe à
l'exécution (drapeau OFF) ; le dossier part avec harmonia_min au sprint 22
(il contient `acoustic.py`, non suivi, d'une autre session — ne pas y toucher).

**Mesuré.** Rapport d'or : 46/46 identiques, 0 mesure changée. `pytest
--collect-only` : 326 tests, propre. Routes : export iReal OK (URL produite),
recherche iReal OK par la route (résultats standards + forum).

**Trouvé en chemin.**
- **INCIDENT** : pour relancer le serveur du worktree, `pkill -f
  "harmonia_min.server"` a aussi tué l'app VIVANTE sur :7772 (même module,
  port dans l'environnement) et le script lui-même. Relancée depuis l'arbre
  principal, même commande, sans drapeau, HTTP 200. Règle : on tue par PID
  du PORT (`lsof -ti tcp:7773`), jamais par motif de ligne de commande.
- La recherche de tablatures UG est **cassée en prod aussi** : `curl_cffi`
  n'est pas dans le venv (ni dans requirements-lock, ni dans pyproject) ; la
  route renvoie `{"results": []}` avec un 200 et un WARNING dans le log. Pas
  une régression du port ; décision de Louis pour l'installer dans le venv
  partagé.
- `search_community` avale ses exceptions (`except Exception: results = []`,
  porté tel quel) — un repli silencieux à traiter au sprint 14 avec la route.
- Le test `test_irealb_fetcher.py` supprimé testait la moitié abandonnée et
  le réseau vivant ; la recherche n'a pas de test hors-ligne.

**Suivant.** Sprint 2 : `labels.py`, `key_profiles.py`.

## Sprint 2 — 2026-09-14 · `labels.py`, `key_profiles.py`

**Fait.** Les deux modules purs (aucun chemin, aucune variable d'environnement)
déménagent par `git mv` dans `harmonia/` ; `harmonia_min/labels.py` et
`harmonia_min/key_profiles.py` deviennent des ponts (`from harmonia.X import *`,
plus les deux tables privées `_TAIL_PCS`/`_TRIAD_PCS` que `harmonic_key`
importe). Les ponts disparaissent avec harmonia_min (sprint 22). Fait à la
main plutôt que délégué : deux déplacements et deux ponts de trois lignes.

**Docstrings.** `labels.py` commençait par son propre chemin de fichier
(liste des erreurs à ne pas reporter, point 3) ; `key_profiles.py` promettait
une détection de modulation que cette copie n'a jamais eue (tronquée au
2026-07-30) — remplacée par la phrase « ce que ce module ne fait PAS ».

**Mesuré.** Rapport d'or 46/46 identiques, 0 mesure changée. pytest : 326
passés en 5 s (aucun test n'importe ces deux modules directement).

**Suivant.** Sprint 3 : `beats.py` sur `harmonia.cache`.

**Décisions de Louis (2026-09-14, après le sprint 2).** (1) `curl_cffi` installé
dans le venv partagé et ajouté à `requirements-lock.txt` — la recherche UG
remarche sur :7773 et sur :7772. (2) Les sept charts « — AVANT » d'août
retirés de la bibliothèque vivante ; les deux copies « test fusion » écrites
aujourd'hui par sa session concurrente sont laissées. (3) `merge_letters`
(fusion de lettres identiques, opt-in du jour) reste un argument explicite de
`fold_letter_groups`, défaut off ; seul son drapeau d'environnement disparaît.
*Suite (même soir).* Huit charts « — AVANT » du 2026-08-12 en fait, tous retirés.
La recherche UG, `curl_cffi` installé, reçoit maintenant un **HTTP 500 du site**
(anti-bot, même famille que le blocage YouTube du 2026-09-13) — sur :7772 comme
sur :7773 ; le module le logge en WARNING et renvoie une liste vide. Rien à
faire côté refactor ; à traiter avec la route au sprint 14 si Louis y tient.

## Sprints 3, 4, 5 — 2026-09-14 · `beats.py`, `musx.py`, `nnls_features.py` sur `settings` + `cache`

**Fait (délégué à un agent Sonnet, un module à la fois, porte après chacun ;
`git mv` + ponts posés par la session principale avant, commits après).**
- `beats.py` : lecture/écriture du cache par `harmonia.cache` (kind `beats`) ;
  un cache corrompu se logge en WARNING puis se recalcule, au lieu du
  `except ValueError: pass`. `CACHE_DIR` reste comme pont (des scripts le lisent).
- `musx.py` : `_musx_dir()` et `_device()` lisent `SETTINGS` — plus aucun
  `os.environ` dans le module ; `_PROB_CACHE`/`_CQT_CACHE` sont des alias des
  dossiers de `cache.py` (span_rescore importe `_PROB_CACHE`) ;
  `frame_posteriors(cache_dir=None)` passe par le cache commun, un
  `cache_dir` explicite garde l'ancien comportement. Décodeur, patch MPS,
  `redecode` : intouchés.
- `nnls_features.py` : `HEADS_NPZ` = l'asset suivi ; cache par `cache.py`
  (kind `nnls`, `savez_compressed` pour les nouveaux fichiers, identiques à la
  relecture) ; **`get_heads()` lève `FileNotFoundError`** au lieu de renvoyer
  `None` avec un WARNING — l'asset est suivi par git, son absence est une
  erreur, et un `None` renvoyé obligeait chaque appelant à vérifier (deux
  scripts ne le faisaient qu'en silence). Docstrings : plus de chemins vers
  des fichiers supprimés ni de « voir X.py ».
- `cache.py` gagne `folder(kind)` pour les ponts, à la place d'un
  `path(kind, Path("_")).parent` bricolé.

**Mesuré (après chaque module, puis re-vérifié par la session principale).**
Rapport d'or 46/46 identiques, 0 mesure changée ; pytest 326 passés ;
`grep os.environ` vide sur les trois modules ; chaque dossier de cache
résolu par le nouveau code est exactement l'ancien emplacement.

**Suivant.** Sprint 6 : extraire `bars.py` de `pipeline.py:493-656`.

## Sprint 6 — 2026-09-14 · `bars.py` extrait de `pipeline.py`

**Fait (agent Sonnet).** La disposition en mesures — snap des onsets sur
l'index de temps, vote de phase, marque « Set bar 1 », report des accords
tenus, règle des pickups et du débordement — sort de `analyze_steps` pour
devenir `harmonia/bars.py` : une dataclass `BarLayout` (bpb, off, n_bars,
bar1_bar, grid, bars, step, segments) et un point d'entrée `layout_bars(...)`.
Les commentaires datés (règle de la barre 2026-07-31, jamais de « % », on
ne reporte jamais un N.C. 2026-08-10, la queue de N.C. qui traverse la barre
2026-08-09, le débordement de la mesure 0) voyagent avec le code. `pipeline.py`
passe de 923 à 710 lignes.

**Ce que l'agent a trouvé que le brief n'avait pas vu.** Le bloc RÉASSIGNE
`segments` (un N.C. de tête court est jeté) et ce `segments` rebondi est relu
par le prompteur et par le rapport final : il fait donc partie de la sortie
typée. Sans ça, le prompteur aurait reçu la liste non rognée — en silence.

**Mesuré.** Rapport d'or 46/46 identiques ; pytest 326 passés. Aucun pont
nécessaire : personne d'autre n'importait ces helpers (les scripts en ont
leurs propres copies). Docstring de `musx.py` mise à jour (elle citait
`pipeline._segment_confidence`).

**Suivant.** Sprint 7 : `folding.py` et `refold.py`, une seule loi.

## Sprint 7 — 2026-09-14 · `folding.py`, `refold.py` : une seule loi

**Fait (agent Sonnet, en parallèle du sprint 8 sur des fichiers disjoints).**
`folding.py` passe de 1 287 à 859 lignes : disparaissent la loi CQT
(`combine="cqt"`, `cqt`, `check_thr`, `_cqt_template`, `_adhesion`,
`_cqt_transpose`), `loi_de_merge()`/`MERGE_DEFAUT`, la transposition
(`transpose`, `decalage_semitons`, `_rot_probs`, `_rot12`,
`_transpose_accords` — devenus identité), le veto « par mesure »
(`gate="bar"`), le paramètre `loop` (le comportement `occurrence` — boucle
interne d'abord, puis l'occurrence entière — est désormais le seul), les
variantes jamais appelées `weight`, `bass_mode`, `ecriture="vote"`
(`vote_des_passages`, `_signature`, `_entropy_weights`, `_renorm`). La lecture
de `HARMONIA_QUARTER_BAR` devient `SETTINGS.quarter_bar`. Signature
survivante : `fold_letter_groups(sections, bars, grid, probs, bpb, arr=None,
times=None, *, merge_letters=False)` — `merge_letters` reste un argument
explicite (décision de Louis), alimenté par `SETTINGS.merge_letters`.
`refold.py` et le site d'appel de `pipeline.py` perdent leurs six lectures
d'environnement. Tests retirés : `test_loi_de_merge.py`, `test_fold_rot_probs.py`,
six tests de transposition dans `test_songformer_sections.py`.

**Compromis à retirer plus tard.** Le rapport de repli garde deux clés
toujours vides (`demiton: {}`, `pos_skip: []`) pour que le JSON des charts
reste identique ; à supprimer quand le schéma du chart sera formalisé
(avec une page avant/après, puisque le JSON changera de forme).

**Mesuré.** Rapport d'or 46/46 identiques, 0 mesure changée ; pytest 289
passés (326 − 24 tests supprimés par les sprints 7 et 8 − 13 du fichier
`test_chord_vocab_q8.py`).

## Sprint 8 — 2026-09-14 · `harmonic_key.py`, `span_rescore.py` allégé, le « reinfer » enterré

**Fait (agent Sonnet).** `span_rescore.py` passe de 749 à 268 lignes : reste
le classement des alternatives par span (`musx_suggestions`, le `sug` de
chaque accord que l'écran Annotate affiche) et ses tables ; partent le
rescoring en treillis, le rescoring différentiel, le scorer de contexte et
le repli NNLS-24 — la propagation est retirée depuis le 2026-08-20.
`harmonia_min/chord_context_prior.py` (884 lignes, dont un `except Exception:
pass` sur la lecture de son cache) est supprimé avec son test
`test_chord_vocab_q8.py`. `server._capabilities()` renvoie `["annotations"]`
et ne charge plus un scorer à chaque `GET /api/library` pour annoncer une
capacité dont la route répond 410. `harmonic_key.py` : docstring seulement.

**Ce que l'agent a corrigé dans le brief.** `refold.py` importe
`musx_cache_path` de span_rescore (le brief le croyait mort) : gardé,
réécrit sur `harmonia.cache`. La date de retrait de la propagation est le
2026-08-20, pas le 19.

**Mesuré.** Rapport d'or 46/46 identiques ; pytest 289 passés ; plus aucune
référence exécutable à `chord_context_prior`/`load_context_scorer`/
`lattice_rescore` dans `harmonia`, `harmonia_min`, `tests`, `tools`.

**Suivant.** Synchroniser le commit `befd8ee` de `feat/section-criteres`
(cascade d'affichage des sections, 160 lignes dans `minimal_fold`), puis
sprint 9 : sections/songformer seul.

## Synchronisation — 2026-09-14 · `feat/section-criteres` @ cfceadf entre dans le refactor

**Pourquoi.** La session concurrente a commité trois fois pendant les sprints
7–8 : la cascade iReal d'affichage (« une lettre = un seul bloc écrit » :
fins 1./2., passage coupé, A′/A″), puis la même cascade sur le chemin des
découpages à la main. Ça touche `folding.py` (devenu un pont) et `soudure.py`.

**Comment.** `git merge` ; conflit sur le pont `harmonia_min/folding.py` :
pont gardé, leur delta complet (689a21c..cfceadf) rejoué sur
`harmonia/folding.py` avec `git apply --3way` ; le pont régénéré réexporte
les nouveaux noms privés (`_ireal_cascade`, `_ireal_endings`) — piège :
`import *` saute les noms soulignés, un consommateur nouveau les fait
manquer. Première tentative avec le seul delta de befd8ee : 8 morceaux en
erreur (`fold_report` None) — la garde `(fold_report or {})` est dans le
commit suivant. Leçon : rejouer la PLAGE, pas un commit.

**Vérifié.** Bibliothèque cuite par le merge == bibliothèque cuite par
l'arbre vivant à cfceadf, 44/44 identiques. Contre l'ancienne baseline :
10 mesures sur 4 morceaux (Grenade, Let It Be, She Will Be Loved ×2 — tous
à sections manuelles), 39 charts changent de forme sans changer d'accord.
C'est l'effet que Louis a décidé dans l'autre session ; page AVANT/APRÈS
publiée pour mémoire ; baseline re-gelée (l'ancienne reste en
`baseline_689a21c`). pytest : 296 passés, **1 échec hérité**
(`test_minimal_fold_separe_les_longueurs_dune_meme_lettre`, échoue aussi sur
l'arbre vivant — la cascade a changé ce que ce test affirmait ; à leur main).

**Règle adoptée.** Après chaque commit de la session concurrente sur
`harmonia_min/`, synchroniser AVANT le sprint suivant, prouver l'identité
avec l'arbre vivant, re-geler la baseline.
