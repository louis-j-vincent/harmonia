# Basculement en prod (sprint 21) — la marche à suivre

Écrit le 2026-09-14 au soir, une fois les sprints 0–20 commités sur
`refactor/clean-app` (52a37b8). À exécuter dans l'ARBRE PRINCIPAL
(`/Users/vincente/Documents/Projets Perso/Code/harmonia`, branche
`feat/section-criteres`), après le feu vert de Louis, et pas pendant qu'il
annote. Chaque étape est réversible ; le point de retour est le tag
`pre-refactor-2026-09-14` et l'ancien dossier `harmonia_min/state/`, jamais
supprimé par la migration.

## 0. Ce que Louis doit faire avant (l'arbre vivant n'est pas propre)

`git status` dans l'arbre principal montre des modifications NON commitées sur
des fichiers que le merge supprime ou dé-suit : trois tests legacy
(`tests/test_local_key.py`, `tests/test_irealb_fetcher.py`,
`tests/test_chart_model.py`) et cinq images `scratchpad/colour_hmm_*.png`.
Git refusera le merge tant qu'elles sont là. Louis choisit : les commiter
(elles partiront avec le merge) ou les jeter (`git checkout -- <fichier>`,
dans SON arbre, par lui).

## 1. Vérifier avant de toucher

    cd <arbre principal>
    git branch --show-current            # feat/section-criteres
    git status --short | grep -vE "^\?\?" # rien d'autre que le point 0
    git merge-tree --write-tree feat/section-criteres refactor/clean-app | grep -c CONFLICT   # 0
    curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7772/api/library   # 200 (l'ancienne app)

## 2. Merger

    git merge --no-ff refactor/clean-app -m "merge refactor/clean-app : la nouvelle app (sprints 0-20)"

Attendu : pas de conflit. Le legacy disparaît, `harmonia_min/` reste (ponts),
`scripts/` devient `archive/scripts/` + `tools/`, `scratchpad/` n'est plus
suivi (les fichiers restent sur le disque).

## 3. Migrer l'état vivant (copie, jamais de suppression, idempotent)

    .venv/bin/python -m tools.migrate_state
    .venv/bin/python -m tools.migrate_state     # 2e passage : 0 copié

Ce que ça fait : `harmonia_min/state/{sections,annotations,chart_meta.json,
folders.json}` → `state/human/` ; `marks/` créé depuis le `bar1` de chaque
chart ; charts, battues, songformer → `state/cache/` (battues et songformer
renommés `<nom>__<taille>`). `data/cache/` (musx, nnls, 12 Go) n'est PAS
touché : le code lit l'ancienne clé en repli, avec une ligne de log.
Le dossier `harmonia_min/state/reports/` (98 Mo de pages de diagnostic)
n'est pas migré : copier `avant_apres.html` si on y tient.

    git add state/human && git commit -m "état humain migré sous state/human (sprint 21)"

## 4. Prouver que la nouvelle app cuit la même bibliothèque

    .venv/bin/python -m tools.golden --engine harmonia --out state/cache/golden/swap --baseline state/cache/golden/baseline

Attendu : `identiques 46/46 · mesures changées 0` (il faut d'abord copier la
baseline du worktree : `.claude/worktrees/refactor/state/cache/golden/baseline`
→ `state/cache/golden/baseline`). Si des mesures changent : page
`tools.avant_apres`, et Louis arbitre AVANT le redémarrage.

## 5. Redémarrer :7772 sur la nouvelle app — par PID du PORT, jamais par nom

    kill $(lsof -ti tcp:7772)
    nohup .venv/bin/python -m harmonia.server >> /private/tmp/harmonia_7772.log 2>&1 &
    sleep 10; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7772/api/library   # 200

`tailscale serve` (:8443 → 127.0.0.1:7772) ne change pas. Louis ouvre
http://100.89.209.63:7772/ sur le téléphone : bibliothèque, un chart, Practise.

## 6. L'oreille de Louis, puis 24 h de vie normale

Les cinq morceaux de la décision 12 : Stand By Me
(`?open=min_ben_e_king_stand_by_me_audio`), This Love (`min_maroon_5_this_love`),
Let It Be (`min_let_it_be_remastered_2009`), Bora Bora (`min_T64BgKEL-Sw`),
Sunny Afternoon (`min_nDUVEUjOKMw`). Puis une analyse YouTube neuve, un
enregistrement micro (https://…:8443), une marque « Set bar 1 », une section
à la main : chacun doit écrire dans `state/human/` (à commiter ensuite).

## 7. Retour arrière si quelque chose cloche

    kill $(lsof -ti tcp:7772)
    git reset --hard pre-refactor-2026-09-14      # OU git revert -m 1 <merge>, si des commits ont suivi
    nohup .venv/bin/python -m harmonia_min.server >> /private/tmp/harmonia_min_7772.log 2>&1 &

`harmonia_min/state/` n'a pas bougé : l'ancienne app retrouve tout.

## 8. Après la soak (sprint 22)

Supprimer `harmonia_min/` (ponts compris), rebrancher les tests et les outils
qui importent encore `harmonia_min.*`, passer `meta.engine` à « harmonia »
(le rapport d'or verra tous les charts « différer » sans mesure changée :
regeler la baseline), renommer les clés de `data/cache/` avec un script de
renommage EN PLACE (le disque est à 96 %), retirer le plist launchd mort
`com.harmonia.nightly-agent` (hors dépôt, sur décision de Louis), nettoyer les
cinq worktrees `.claude/worktrees/*` (peut-être à d'autres sessions) et le
fatras non suivi de la racine (`Harmonia.zip`, `handoff 3`, `handoff_ui_refresh*`,
`handoff_cleanup 2`, `harmonia.egg-info`, `.coverage`).
