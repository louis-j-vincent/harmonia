# Harmonia — Known Issues

Journal court, à jour. Historique complet (2026-07 à 2026-09, avant le
refactor `harmonia_min` → `harmonia`) : `docs/archive/known_issues_2026-07_2026-09.md`.
Mécanique du projet (comment vérifier un changement, où sont les fichiers) :
`docs/STATE.md`.

## Ouvert

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

## Résolu récemment

- Le paquet legacy `harmonia/` (pré-refactor), `scripts/harmonia_server.py`,
  et 137+ fichiers associés sont supprimés (tag `pre-refactor-2026-09-14`
  pour les retrouver). Voir `docs/refactor_2026-09/sprints.md`.
