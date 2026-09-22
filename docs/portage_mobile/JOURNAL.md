# Portage mobile — JOURNAL (append-only, une ligne par fait)

- 2026-09-22 Décision : Android d'abord (3-4 sem), iOS +1-2 sem ; import YouTube ne survit qu'en sideload Android → `portage_mobile_2026-09-22_synthese.md`
- 2026-09-22 Corrections aux audits, vérifiées dans le code : SongFormer est ÉTEINT (`settings.py:91`, branche `else` de `pipeline.py:750`) ; `nnls-chroma` EST sur le chemin critique (`pipeline.py:744`, `:845`)
- 2026-09-23 Machine : 0 Java, 0 Android SDK, 0 Xcode (CLT seulement), disque 4,1 Go libres/228 → SongFormer HF cache (2,7 Go, jamais exécuté) + caches pip/brew supprimés → 8,1 Go ; JDK 17 installé (`brew openjdk@17`)
- 2026-09-23 Worktree `.claude/worktrees/mobile`, branche `feat/mobile-android` sur `71f66fd` ; `main` est 378 commits en retard (pré-refactor) — signalé à Louis
- 2026-09-23 Anomalie disque : `~/.cache` mesuré à 458 Go (> disque) — mesure détaillée en cours
