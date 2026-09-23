# Sauvegarde git 2026-09-24 — tous les checkouts

Chaque checkout est resté sur sa propre branche (jamais de `checkout` dans
l'arbre partagé) ; les 3 HEAD détachés ont reçu une nouvelle branche
`sauvegarde/<nom>` créée uniquement dans leur worktree.

| Checkout | Branche (créée ?) | Hash commit | Fichiers | Exclus | Suppressions state/human |
|---|---|---|---|---|---|
| `harmonia` (arbre partagé) | `feat/treillis-chart` (existante) | `4579f2c` | 290 | `docs/plots/exact_tree_diagnostic.html` (7,8 Mo), `docs/chord_lm_audition.html` (6,1 Mo — sauvegardé depuis par la session concurrente sur cette branche), `harmonia_min/state/` (138 Mo, cache régénérable de l'app legacy — idem, sauvegardé depuis par la session concurrente), `harmonia/models/*.npz\|*.pt` (gitignorés), `__pycache__`, `.DS_Store` | 2 : `state/human/annotations/min_191P7nIeECo.json`, `state/human/marks/191P7nIeECo.json` (détectée comme renommage vers `corbeille/2026-09-17_233342_191P7nIeECo/marque.json`, contenu identique) |
| `.claude/worktrees/mobile` | `feat/mobile-android` (existante) | `8ef0626` | 1 | — (les 2 fichiers vus en début de tâche étaient déjà traités par la session en cours sur ce worktree) | aucune |
| `.claude/worktrees/agent-a196db0f4016efca8` | `worktree-agent-a196db0f4016efca8` (existante) | `add30b0` | 1 | `docs/audio`, `state/cache` (symlinks locaux vers l'arbre partagé) | aucune |
| `.claude/worktrees/harmonic-challenge` | `feat/minimal-pipeline-harmonic-challenge` (existante) | `e9b9d04` | 1 | `docs/audio` (symlink) | aucune |
| `.claude/worktrees/harmonic-key` | `feat/harmonic-key` (existante) | `3a87f42` | 5 | — | aucune |
| `.claude/worktrees/bass-discrim` | `sauvegarde/bass-discrim` (**créée**, depuis HEAD détaché `6c20bf4`) | `734f85c` | 54 | `scratchpad/fresh_charts/_wav/*.wav` (7 fichiers, 14–24 Mo chacun), `docs/audio` (symlink) | aucune |
| `.claude/worktrees/postmusx-trace` | HEAD détaché `d13242c` (**pas de branche créée** — rien à sauvegarder) | — | 0 | `docs/audio` (symlink, seul fichier vu) | aucune |
| `.claude/worktrees/ug-aligner` | `sauvegarde/ug-aligner` (**créée**, depuis HEAD détaché `112cd4d`) | `1ae23e3` | 15 | `scratchpad/fresh_charts/_wav/*.wav` (7 fichiers, 14–24 Mo chacun), `docs/audio` (symlink), `harmonia/third_party/.../data` (dossier vide) | aucune |

**À noter** : une session concurrente travaillait EN CE MOMENT sur l'arbre
partagé (`feat/treillis-chart`) — après mon commit `4579f2c`, elle a
committé les deux gros fichiers que j'avais exclus par taille
(`docs/chord_lm_audition.html`, `harmonia_min/state/`), donc l'arbre partagé
est maintenant entièrement propre aussi. Aucun `git checkout` n'a été fait
dans l'arbre partagé.
