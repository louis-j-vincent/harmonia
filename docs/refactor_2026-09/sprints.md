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
