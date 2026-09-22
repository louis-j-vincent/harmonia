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
}

chaquopy {
    defaultConfig {
        version = "3.12"
        pip {
            // Phase 1 (ce squelette) : AUCUNE analyse. Seuls numpy (import
            // transitif par plusieurs modules harmonia/ même hors calcul,
            // ex. nnls_features.py, span_rescore.py) et flask (le serveur
            // local) sont installés. torch, scipy, vamp, librosa, yt-dlp,
            // pyRealParser NE SONT PAS installés — voir le rapport pour la
            // preuve que `harmonia.server.app` s'importe et sert quand même
            // (tous ces imports lourds sont faits à l'intérieur de
            // fonctions, jamais au niveau module, dans ce paquet).
            install("numpy")
            install("flask")
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

tasks.named("preBuild") {
    dependsOn(stageHarmoniaPython)
}
// `preBuild` seul ne suffit pas : Gradle 8's task-output validation veut une
// dépendance EXPLICITE de la tâche Chaquopy qui LIT build/harmoniaStage
// (mergeDebugPythonSources / mergeReleasePythonSources) sur celle qui
// l'ÉCRIT (voir "implicit_dependency" dans le message d'erreur observé).
tasks.matching { it.name.contains("PythonSources") || it.name.contains("PythonBuildPackages") }
    .configureEach { dependsOn(stageHarmoniaPython) }
