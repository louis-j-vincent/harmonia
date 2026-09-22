# Portage mobile — PLAN (le fichier que tout agent lit en premier)

Mandat de Louis (2026-09-22) : « je te laisse tourner jusqu'à ce qu'on ait une
app fonctionnelle testable sur Android ou iPhone ». Analyse d'un morceau **sans
Mac**, calcul embarqué. Décision et raisons : `portage_mobile_2026-09-22_synthese.md`.
Inventaire du chemin critique : `audit_portage_mobile_2026-09-22.md`.

## Où l'on travaille — RÈGLES NON NÉGOCIABLES

- **Worktree** : `/Users/vincente/Documents/Projets Perso/Code/harmonia/.claude/worktrees/mobile`
  Branche `feat/mobile-android`, basée sur `71f66fd` (tête de
  `feat/treillis-chart` au 2026-09-22 ; `main` a 378 commits de retard, ne PAS
  s'y baser). Toujours travailler ICI. Jamais dans l'arbre partagé
  (`…/Code/harmonia` sans `.claude/worktrees/mobile`) — une autre session y vit.
- **Git** : les agents NE COMMITTENT PAS. Ils écrivent des fichiers ; l'orchestrateur
  commit aux jalons. Jamais `checkout`, `switch`, `reset`, `stash`, `rebase`.
- **Venv** : `.venv` du worktree = le venv de prod, partagé et fragile. **N'y
  installer AUCUN paquet.** Tout ce qui est nouveau (executorch, coremltools…)
  va dans `~/harmonia/.venv-mobile` (Python 3.12, à créer si absent :
  `python3.12 -m venv ~/harmonia/.venv-mobile`).
- **Disque** : la machine est à ~8 Go libres. `df -h /System/Volumes/Data`
  avant toute installation > 500 Mo ; si < 3 Go libres, s'arrêter et le dire.
- **Serveur :7772** : l'app vivante de Louis. Ne jamais la tuer, ne jamais
  lancer un serveur sur ce port. Un serveur de test = `HARMONIA_PORT=7790+`.
- **Modèles** : `beat_this` (pip, checkpoints `small0` 9 Mo / `final0` 97 Mo),
  `musx` = `third_party/musx_ismir2019/` (5 × `cache_data/*.sdict`, 5,7 Mo chacun).
  Référence Python = ce qui tourne dans `harmonia/pipeline.py:analyze_steps`.

## Cible

Android d'abord (téléphone de Louis dès la semaine du 2026-09-28), iOS ensuite.

| Couche | Choix | Pourquoi |
|---|---|---|
| Glue Python (pipeline, bars, folding, clés) | **Chaquopy** | CPython + numpy sur Android, gratuit ; le code part tel quel |
| UI | `harmonia/static/` (JS) dans une **WebView**, servi par **Flask local** (127.0.0.1) | portage UI ≈ 0 |
| Réseaux `beat_this`, `musx` | **ExecuTorch** (`torch.export` → `.pte`, backend XNNPACK) | cross-plateforme, réutilisé sur iOS |
| `nnls-chroma` (plugin VAMP C++, GPL) | **port numpy** validé contre `vamp.collect` sur le Mac ; repli = compilation NDK | sans GPL, sans .so |
| Décodage audio | Java `MediaCodec` → PCM float → Python | pas de ffmpeg sur le téléphone |
| Import YouTube | `youtubedl-android` (sideload uniquement) | seule plateforme où ça survit |
| Interdit sur le téléphone | librosa (numba), scipy, soundfile, ffmpeg, vamp, torch complet | pas de roue Android / pas de JIT |

## Périmètre v1

Dedans : analyser (micro, fichier, YouTube), écouter le chart, bibliothèque
locale. Routes `library` + `analyze` seulement.
Dehors : annotation des sections, Jam, iReal, corrections, verdicts (v2).
Conséquence connue : un morceau neuf sort avec une seule section « A ».

## Phases et portes

| Phase | Tâche | Statut | Porte de sortie |
|---|---|---|---|
| **0** | T1 export ExecuTorch de `beat_this` + `musx` | en cours | sorties = torch à 1e-3 sur 1 morceau ; chaîne de prétraitement documentée |
| 0 | T2 port numpy de `nnls-chroma` | en cours | = `vamp.collect` à 1e-3 sur 3 morceaux |
| 0 | T3 inventaire des dépendances natives du chemin critique + plan de remplacement | en cours | tableau exhaustif, chaque ligne a un remplaçant |
| 0 | T4 toolchain Android (cmdline-tools, NDK, émulateur arm64) | bloqué disque | `gradle assembleDebug` d'un hello-world passe |
| 0 | **T5 vitesse sur téléphone réel** | attend le téléphone | **4 min de musique en < 2 min de calcul, sinon STOP** |
| 1 | Squelette Android : Chaquopy + Flask local + WebView, pipeline stubbé sur un chart en cache | — | l'émulateur affiche un chart de la bibliothèque |
| 2 | Pipeline réel sur device : décodage MediaCodec, réseaux `.pte`, NNLS numpy, remplacement librosa/scipy | — | un morceau analysé sur l'émulateur = chart Mac mesure par mesure |
| 3 | Import YouTube, micro, fichier | — | les 3 entrées marchent sur l'émulateur |
| 4 | APK signé + notice d'installation ; test sur le téléphone | — | Louis installe et analyse un morceau |
| 5 | iOS (ExecuTorch CoreML, BeeWare ou port Swift de la glue) | après 4 | — |

## Protocole de compte rendu (obligatoire, pour ne pas saturer les contextes)

Chaque tâche produit `docs/portage_mobile/<phase>_<tâche>_<nom>.md` avec, dans
cet ordre et sous ces titres exacts :

    ## Résultat en 3 lignes
    ## Ce qui a été vérifié (commandes, chiffres, chemins)
    ## Ce qui reste ouvert
    ## Fichiers produits

Puis UNE ligne datée en fin de `JOURNAL.md` : `- 2026-09-2x T1 : <verdict en
une phrase> → <fichier>`. La réponse finale de l'agent fait ≤ 250 mots et
recopie le « Résultat en 3 lignes ». L'orchestrateur ne lit que ça.

Le code va sous `mobile/` (app Android) et `mobile/tools/` (scripts d'export,
de validation). Rien dans `harmonia/` ne change en phase 0.

## Statut au 2026-09-23 (l'orchestrateur tient cette section à jour ; la table des phases ci-dessus est le plan d'origine)

| Brique | État | Rapport |
|---|---|---|
| NNLS-chroma numpy | **FAIT** — parité 1e-4, argmax basse 100 %, 3 s / 4 min | `p0_t2_nnls_numpy.md`, `p0_t2b_nnls_vitesse.md` |
| CQT musx numpy | **FAIT** — 2,6e-5, accords 99,99 %, 1,2-2,9 s / 4 min | `p2b_cqt_musx.md` |
| Inventaire natif + contrat décodage | **FAIT** | `p0_t3_deps_natives.md` |
| beat_this `.pte` | small0 OK (47/47 battues) ; **`final0` = le vrai checkpoint du pipeline, en cours (T1c)** avec front-end mel numpy + découpage | `p0_t1_executorch.md` |
| musx réseau | export OK, lowering XNNPACK bloque (BiLSTM) — **en cours (T1b)** : portable / partiel / numpy | `p0_t1_executorch.md` |
| Squelette Android | **FAIT** — APK 44 Mo, bibliothèque affichée sur l'émulateur (`mobile/android/smoke/screen_library.png`) | `p1_squelette_android.md` |
| Décodeur MediaCodec | en cours (P2a) | — |
| YouTube | sans cookies : 15/15 bloqués ; **en cours (P3b)** : cookies en fichier + QuickJS arm64 | `p3_youtube.md` |
| Intégration pipeline sur device (P2c) | attend T1b + T1c + P2a | — |
| Vitesse téléphone réel (T5, porte dure) | attend le téléphone (semaine du 28) | — |

Environnement : JDK 17 (`/opt/homebrew/opt/openjdk@17`), SDK `~/Library/Android/sdk` (platform 35, build-tools 35, emulator, image arm64), AVD `harmonia_35` (userdata 4 G), venv export `~/harmonia/.venv-mobile`. Disque ≈ 10 Go libres après nettoyage.
