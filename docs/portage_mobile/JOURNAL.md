# Portage mobile — JOURNAL (append-only, une ligne par fait)

- 2026-09-22 Décision : Android d'abord (3-4 sem), iOS +1-2 sem ; import YouTube ne survit qu'en sideload Android → `portage_mobile_2026-09-22_synthese.md`
- 2026-09-22 Corrections aux audits, vérifiées dans le code : SongFormer est ÉTEINT (`settings.py:91`, branche `else` de `pipeline.py:750`) ; `nnls-chroma` EST sur le chemin critique (`pipeline.py:744`, `:845`)
- 2026-09-23 Machine : 0 Java, 0 Android SDK, 0 Xcode (CLT seulement), disque 4,1 Go libres/228 → SongFormer HF cache (2,7 Go, jamais exécuté) + caches pip/brew supprimés → 8,1 Go ; JDK 17 installé (`brew openjdk@17`)
- 2026-09-23 Worktree `.claude/worktrees/mobile`, branche `feat/mobile-android` sur `71f66fd` ; `main` est 378 commits en retard (pré-refactor) — signalé à Louis
- 2026-09-23 Anomalie disque : `~/.cache` mesuré à 458 Go (> disque) — mesure détaillée en cours
- 2026-09-23 T3 : 18 sites natifs sur le chemin critique, tous des décodages ffmpeg/librosa à remplacer par un seul `MediaCodec` Java (3 SR cibles : 22050/22050/44100) ; SongFormer confirmé éteint (0 risque v1) ; trou trouvé : le CQT `hybrid_cqt` de musx (`extractors/cqt.py:52-58`) n'a de remplaçant assigné ni dans T1 ni T2 → `docs/portage_mobile/p0_t3_deps_natives.md`
- 2026-09-23 T2 : port numpy pur de nnls-chroma valide vs vamp.collect réel sur 3 morceaux — max abs diff 0.00011-0.00029 (< cible 1e-3), argmax basse 100% identique, ~7x temps réel sur Mac → docs/portage_mobile/p0_t2_nnls_numpy.md
