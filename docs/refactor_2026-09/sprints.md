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
