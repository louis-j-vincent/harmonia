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
