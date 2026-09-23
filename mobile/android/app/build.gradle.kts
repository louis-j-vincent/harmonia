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

    // P2d (2026-09-23) : les poids sont des flottants, ils ne se compressent
    // pas (82 Mo de .pte gagnent quelques pourcents pour des minutes de
    // build et une décompression au premier lancement). Stockés tels quels,
    // la copie assets -> filesDir est une copie d'octets.
    androidResources {
        noCompress += listOf("pte", "npz", "sdict")
    }
}

dependencies {
    // P2d (2026-09-23) : le runtime ExecuTorch NATIF (libexecutorch.so +
    // l'API Java org.pytorch.executorch.Module). Le `.pte` de Beat This! a
    // été exporté avec le paquet pip executorch 1.5.1 ; 1.4.0 est la
    // dernière version PUBLIÉE de l'AAR sur Maven Central (2026-08-07) —
    // l'écart de version est donc un vrai risque de format, VÉRIFIÉ sur
    // l'émulateur et non supposé (voir docs/portage_mobile/p2d_pipeline_
    // device.md). L'identifiant de schéma du fichier est `ET12`
    // (`xxd -l 16` sur le .pte), inchangé entre ces deux versions.
    implementation("org.pytorch:executorch-android:1.4.0")
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
    // P2d (2026-09-23) : les briques de calcul mobiles (mel numpy, CQT numpy,
    // réseau musx numpy, chroma NNLS numpy, découpage/post-traitement des
    // battues) vivent dans `mobile/harmonia_mobile/`. C'est le MÊME code que
    // celui que la parité du Mac a mesuré (P2c) — embarqué tel quel, jamais
    // recopié à la main, sinon les deux plateformes divergeraient en silence.
    // `mobile/tools/` n'est PAS embarqué : ce sont les scripts de mesure, et
    // ils importent librosa/torch/scipy, interdits sur le téléphone.
    from(File(repoRoot, "mobile")) {
        into("mobile")
        include("__init__.py")
        include("harmonia_mobile/**")
        exclude("**/__pycache__/**")
        exclude("**/*.pyc")
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

// ── Poids des réseaux (P2d, 2026-09-23) ─────────────────────────────────────
// `mobile/models/` est gitignoré (les .pte/.npz sont régénérables par
// `mobile/tools/export_beat_this.py` et `dump_musx_weights.py`, et pèsent
// 92 Mo). Même logique que copySeedAudio : on copie dans les assets quand
// les fichiers sont là, et on échoue avec un message clair sinon plutôt que
// d'embarquer un dossier vide et de ne le découvrir qu'à l'exécution.
val modelFileNames = listOf(
    "beat_this_final0_fixed1500.pte",
    "musx_fold0_weights.npz",
    "musx_fold1_weights.npz",
    "musx_fold2_weights.npz",
    "musx_fold3_weights.npz",
    "musx_fold4_weights.npz",
)
val modelsSrcDir = File(repoRoot, "mobile/models")
val modelsDestDir = File(projectDir, "src/main/assets/harmonia_models")

val copyModels = tasks.register("copyModels") {
    group = "harmonia"
    description = "Copie les poids (.pte beat_this + 5 .npz musx) dans les assets"
    inputs.files(modelFileNames.map { File(modelsSrcDir, it) })
    outputs.dir(modelsDestDir)
    doLast {
        modelsDestDir.mkdirs()
        val missing = mutableListOf<String>()
        for (name in modelFileNames) {
            val src = File(modelsSrcDir, name)
            val dest = File(modelsDestDir, name)
            if (!src.exists()) { missing += name; continue }
            if (dest.exists() && dest.length() == src.length()) continue
            src.copyTo(dest, overwrite = true)
            logger.lifecycle("copyModels: $name (${src.length()} octets)")
        }
        if (missing.isNotEmpty()) {
            throw org.gradle.api.GradleException(
                "copyModels: poids introuvables dans $modelsSrcDir : $missing. " +
                "Régénère-les avec mobile/tools/export_beat_this.py (le .pte) " +
                "et mobile/tools/dump_musx_weights.py (les .npz), venv " +
                "~/harmonia/.venv-mobile."
            )
        }
    }
}

// ── Le clone musx vendu (P2d, 2026-09-23) ───────────────────────────────────
// `harmonia/musx.py::redecode` — le décodage HMM qui transforme les
// postérieures en accords écrits — fait `chdir` dans ce dossier et y importe
// `extractors.xhmm_ismir` (du numpy pur). Il lui faut donc un VRAI dossier
// sur le disque du téléphone.
//
// Pourquoi les ASSETS et pas le source set Chaquopy : Chaquopy garde les
// modules Python dans une archive et n'en extrait que des `.pyc` à la
// demande. Un `os.path.exists()` sur `third_party/musx_ismir2019/
// chord_recognition.py` y échoue, et un `chdir` dedans n'a aucun sens.
// Mesuré sur l'émulateur : l'analyse allait jusqu'aux postérieures puis
// mourait sur « music-x-lab clone not found ». Les assets, eux, se recopient
// en vrais fichiers (`MainActivity.seedMusxOnce`), et
// `HARMONIA_MUSX_DIR` — une des trois seules variables que lit
// `harmonia/settings.py` — dit au pipeline où ils sont.
//
// Ce que ça coûte, assumé : `_musx_dir()` exige aussi `cache_data/*.sdict`,
// soit 27 Mo de poids TORCH que le téléphone n'ouvre JAMAIS (il utilise les
// `.npz` numpy de T1b). On les embarque plutôt que de truquer la
// vérification — poser un fichier vide pour faire passer un test est
// exactement le genre de mensonge qui coûte cher ici. La vraie correction
// est une fente `musx_redecode` dans `harmonia/backends.py` : pas faite,
// voir le rapport P2d.
val musxSrcDir = File(repoRoot, "third_party/musx_ismir2019")
val musxDestDir = File(projectDir, "src/main/assets/harmonia_musx")

val copyMusxClone = tasks.register<Copy>("copyMusxClone") {
    group = "harmonia"
    description = "Copie third_party/musx_ismir2019 dans les assets (vrai dossier sur le téléphone)"
    from(musxSrcDir) {
        exclude("**/__pycache__/**")
        exclude("**/*.pyc")
        exclude(".git/**")
    }
    into(musxDestDir)
}

tasks.named("preBuild") {
    dependsOn(copyMusxClone)
    dependsOn(stageHarmoniaPython)
    dependsOn(copySeedAudio)
    dependsOn(copyModels)
}
// `preBuild` seul ne suffit pas : Gradle 8's task-output validation veut une
// dépendance EXPLICITE de la tâche Chaquopy qui LIT build/harmoniaStage
// (mergeDebugPythonSources / mergeReleasePythonSources) sur celle qui
// l'ÉCRIT (voir "implicit_dependency" dans le message d'erreur observé).
tasks.matching { it.name.contains("PythonSources") || it.name.contains("PythonBuildPackages") }
    .configureEach { dependsOn(stageHarmoniaPython) }

// Les poids et l'audio de seed sont écrits dans `src/main/assets/`, qui est
// un dossier de SOURCE déclaré statiquement : AGP 8 exige une dépendance
// explicite de la tâche qui LIT (merge*Assets) sur celles qui ÉCRIVENT,
// sinon le build échoue en validation (« implicit dependency »), ou pire,
// empaquette l'état d'avant la copie.
tasks.matching { it.name.contains("Assets") }.configureEach {
    dependsOn(copyModels)
    dependsOn(copySeedAudio)
    dependsOn(copyMusxClone)
}
