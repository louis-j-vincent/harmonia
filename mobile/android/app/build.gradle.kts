// Portage mobile P1 (2026-09-23). Détail complet de ce qui est embarqué / exclu :
// docs/portage_mobile/p1_squelette_android.md
import org.gradle.api.tasks.Copy

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "fr.harmonia.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "fr.harmonia.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1-p1"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // Le contenu embarqué (harmonia/ filtré + docs/pwa/) est copié par
    // stageHarmoniaPython ci-dessous, dans build/ — jamais committé, jamais
    // dupliqué à la main. Voir la doc de la tâche.
    sourceSets {
        getByName("main") {
            assets.srcDirs("src/main/assets")
        }
    }

    // P3b (2026-09-23) : `jniLibs/arm64-v8a/libqjs.so` (voir
    // docs/portage_mobile/p3b_youtube_cookies_qjs.md) n'est pas une vraie
    // lib partagée — c'est le binaire QuickJS `qjs` renommé, exécuté par
    // subprocess/os.exec depuis mobile_youtube.py, pas dlopen()é. AGP 8 par
    // défaut mappe les .so directement depuis le zip de l'APK sans les
    // extraire sur disque (`useLegacyPackaging = false`) : un exécutable a
    // besoin d'un vrai chemin fichier avec le bit +x, donc legacy packaging.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

chaquopy {
    defaultConfig {
        version = "3.12"
        pip {
            // Phase 1 (ce squelette) : AUCUNE analyse. Seuls numpy (import
            // transitif par plusieurs modules harmonia/ même hors calcul,
            // ex. nnls_features.py, span_rescore.py) et flask (le serveur
            // local) sont installés. torch, scipy, vamp, librosa,
            // pyRealParser NE SONT PAS installés — voir le rapport pour la
            // preuve que `harmonia.server.app` s'importe et sert quand même
            // (tous ces imports lourds sont faits à l'intérieur de
            // fonctions, jamais au niveau module, dans ce paquet).
            install("numpy")
            install("flask")
            // P3b (2026-09-23) : import YouTube. yt-dlp est pur Python (pas
            // d'extension C) — s'installe comme flask, sans roue Chaquopy
            // dédiée. `mobile_youtube.py` l'utilise EN PROCESSUS
            // (yt_dlp.YoutubeDL), jamais en sous-process : il n'y a pas de
            // binaire `yt-dlp` sur PATH Android. Voir
            // docs/portage_mobile/p3b_youtube_cookies_qjs.md pour le moteur
            // JS (QuickJS natif, jniLibs/) et les cookies (fichier statique).
            install("yt-dlp")
        }
    }
    sourceSets {
        getByName("main") {
            // 1) notre point d'entrée (mobile_entry.py), committé ici.
            // 2) la copie générée de harmonia/ + docs/pwa/ (voir la tâche
            //    Copy ci-dessous) — build/harmoniaStage, jamais committée.
            srcDirs("src/main/python", layout.buildDirectory.dir("harmoniaStage").get().asFile.path)
        }
    }
}

// ── Embarquer harmonia/ sans le dupliquer à la main ─────────────────────────
// Copie, à chaque build, harmonia/ (moins les 3 fichiers confirmés MORTS pour
// mobile par l'audit — jamais importés par harmonia.server.app, voir le
// rapport §"Ce qui a été vérifié") + docs/pwa/ (icônes/manifest PWA) dans
// build/harmoniaStage/, avec le MÊME agencement relatif (harmonia/ et docs/
// côte à côte) qu'attend `harmonia.settings.REPO` à l'exécution. La source
// de vérité reste le worktree ; rien n'est édité à la main dans la copie.
val repoRoot = rootProject.projectDir.parentFile.parentFile  // .../worktrees/mobile
val stageDir = layout.buildDirectory.dir("harmoniaStage")

val stageHarmoniaPython = tasks.register<Copy>("stageHarmoniaPython") {
    group = "harmonia"
    description = "Copie harmonia/ (filtré) + docs/pwa/ dans build/harmoniaStage pour Chaquopy"

    into(stageDir)

    from(File(repoRoot, "harmonia")) {
        into("harmonia")
        // Confirmé jamais importé par aucun chemin atteignable depuis
        // harmonia.server.app (grep exhaustif, aucune référence ailleurs
        // dans harmonia/) : détecteur de Foote repêché (recherche),
        // brique kill-switchée, jamais câblée dans pipeline.py.
        exclude("sections/nouveaute.py")
        exclude("models/harmonic_downbeat.py")
        exclude("models/local_key_context.py")
        exclude("**/__pycache__/**")
        exclude("**/*.pyc")
    }
    from(File(repoRoot, "docs/pwa")) {
        into("docs/pwa")
    }

    doLast {
        logger.lifecycle("stageHarmoniaPython: ${stageDir.get().asFile} (" +
            "${fileTree(stageDir).files.size} fichiers)")
    }
}

// ── Seed audio (P2a, 2026-09-23) : app/src/main/assets/harmonia_seed/audio/
// est gitignoré (10 Mo de .m4a, voir .gitignore de ce module) — un clone
// frais du dépôt ne les a pas. `docs/audio/` (à la racine du dépôt) N'EST
// PAS non plus dans git (gitignoré repo-wide, bibliothèque locale de
// Louis) : cette tâche copie donc les 3 morceaux de seed DEPUIS LÀ quand ils
// y sont (le cas normal sur la machine de Louis), et échoue avec un message
// clair sinon plutôt que de laisser Chaquopy embarquer un dossier vide en
// silence. Idempotente : ne touche pas des fichiers déjà présents (utile
// dans ce worktree, où ils ont été copiés à la main pour P1/P2a).
val seedAudioNames = listOf(
    "maroon_5_this_love",
    "ben_e_king_stand_by_me_audio",
    "maroon_5_she_will_be_loved_official_music_video",
)
val seedAudioSrcDir = File(repoRoot, "docs/audio")
val seedAudioDestDir = File(projectDir, "src/main/assets/harmonia_seed/audio")

val copySeedAudio = tasks.register("copySeedAudio") {
    group = "harmonia"
    description = "Copie les 3 .m4a de seed depuis docs/audio (gitignoré) si absents"

    doLast {
        seedAudioDestDir.mkdirs()
        val missing = mutableListOf<String>()
        for (name in seedAudioNames) {
            val dest = File(seedAudioDestDir, "$name.m4a")
            if (dest.exists() && dest.length() > 0) continue
            val src = File(seedAudioSrcDir, "$name.m4a")
            if (src.exists()) {
                src.copyTo(dest, overwrite = true)
                logger.lifecycle("copySeedAudio: $name.m4a copié depuis $seedAudioSrcDir")
            } else {
                missing += name
            }
        }
        if (missing.isNotEmpty()) {
            throw org.gradle.api.GradleException(
                "copySeedAudio: ${missing.size} morceau(x) de seed introuvable(s) " +
                "ni dans $seedAudioDestDir ni dans $seedAudioSrcDir : $missing. " +
                "docs/audio/ n'est PAS suivi par git (bibliothèque locale) — " +
                "récupère ces .m4a depuis la bibliothèque Mac de Louis (docs/audio/) " +
                "ou depuis une autre copie déjà construite, et relance."
            )
        }
    }
}

tasks.named("preBuild") {
    dependsOn(stageHarmoniaPython)
    dependsOn(copySeedAudio)
}
// `preBuild` seul ne suffit pas : Gradle 8's task-output validation veut une
// dépendance EXPLICITE de la tâche Chaquopy qui LIT build/harmoniaStage
// (mergeDebugPythonSources / mergeReleasePythonSources) sur celle qui
// l'ÉCRIT (voir "implicit_dependency" dans le message d'erreur observé).
tasks.matching { it.name.contains("PythonSources") || it.name.contains("PythonBuildPackages") }
    .configureEach { dependsOn(stageHarmoniaPython) }
